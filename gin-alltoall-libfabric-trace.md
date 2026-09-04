# Libfabric Call Trace: GIN Hybrid LSA All-to-All (16 Nodes, 128 Ranks)

## Overview

This document traces the libfabric calls for a **hybrid LSA all-to-all** operation using
NCCL's GPU-Initiated Networking (GIN) with the `putSignal` API. The scenario involves:

- **16 nodes** with **8 GPUs per node** = **128 total ranks**
- **Hybrid communication**: LSA (Load Store Access) for intra-node, GIN for inter-node
- **putSignal API**: Combines RDMA write with remote signal increment
- **Test**: `nccl-tests` all-to-all via GIN through NCCL Device API

> **Which GIN path does this trace cover?** The aws-ofi-nccl plugin exposes two GIN data
> paths (see [nccl-ep-vs-deepep-comparison.md](nccl-ep-vs-deepep-comparison.md), section 10):
>
> - **Host / CPU-proxy path** — a CPU progress thread posts network operations on the GPU's
>   behalf. Exported as `ncclRmaPlugin_v14` / `ncclRmaPlugin_v15` (and legacy
>   `ncclGinPlugin_v11` / `ncclGinPlugin_v13`) in
>   [src/rdma/gin/nccl_ofi_gin_api.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_api.cpp).
> - **EFA-GDA / GDAKI path** — the GPU kernel writes work-queue entries and polls
>   completion-queue entries directly, with **no CPU in the critical path**. Exported as
>   `ncclGinPlugin_v14` in
>   [src/rdma/gin/nccl_ofi_gin_gdaki.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_gdaki.cpp).
>
> **The detailed step-by-step trace below is the host / CPU-proxy path** (the `fi_writedata`
> + `fi_send` metadata + gdrcopy-signal sequence). GDAKI is auto-enabled by the plugin where
> supported (removed knob `OFI_NCCL_GIN_TYPE`, commit `80f2c78`); its very different call
> shape is described in the dedicated **[GDAKI (EFA-GDA) trace](#gdaki-efa-gda-trace)**
> section. Where NCCL binds the GDAKI op-table (EFA on P5en/P6, `NCCL_GIN_TYPE=5`), the proxy
> steps below do **not** occur on the data path.

> **libfabric data-path-direct (affects both paths):** The EFA provider now defaults
> `FI_EFA_USE_DATA_PATH_DIRECT=true`
> ([prov/efa/src/efa_env.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_env.c), `.use_data_path_direct = true`).
> On supported devices the provider **writes WQEs and reads CQEs directly** from userspace
> rather than calling into `ibv_post_send()` / `ibv_poll_cq()` for every operation. So even
> in the proxy trace below, a call such as `fi_writedata()` / `fi_cq_read()` is serviced by
> the provider's direct data path (`efa_data_path_direct_*`) instead of the classic verbs
> post/poll calls. This shortens the software path but does not change the SRD ordering
> semantics or the sequence of libfabric verbs shown here.

## Architecture Context

```
Application (nccl-tests)
    ↓
NCCL Device API Kernel (HybridAlltoAllKernel)
    ↓ (for remote peers)
GIN putSignal API
    ↓
aws-ofi-nccl Plugin — host/proxy path: src/rdma/gin/nccl_ofi_gin.cpp
                       GDAKI path:      src/rdma/gin/nccl_ofi_gin_gdaki.cpp
    ↓
Libfabric (EFA provider; FI_EFA_USE_DATA_PATH_DIRECT=true by default)
    ↓
rdma-core (libibverbs)   ← bypassed on the data path when data-path-direct is active
    ↓
EFA Kernel Driver
    ↓
EFA Hardware
```

## Hybrid All-to-All Communication Pattern

For each rank in a 16-node, 8-GPU-per-node setup:
- **Local peers (LSA)**: 7 peers on same node → direct memory access (no libfabric)
- **Remote peers (GIN)**: 120 peers on other 15 nodes → libfabric calls

### Per-Rank Communication Breakdown

Each rank performs:
- **120 remote puts** (one to each remote rank)
- **120 remote signal increments** (one per remote rank)
- **7 local LSA operations** (direct memory, no network)

**Total network operations per rank**: 120 putSignal operations

## Initialization Phase

### 1. Fabric and Domain Setup

**Location**: `aws-ofi-nccl/src/rdma/gin/nccl_ofi_gin_resources.cpp`

```c
// Query available providers
fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0, &hints, &info_list);
```

**Purpose**: Discover EFA provider capabilities
**Returns**: List of available fabric providers (EFA on p5 instances)

```c
// Open fabric
fi_fabric(info->fabric_attr, &fabric, NULL);
```

**Purpose**: Create fabric object for EFA provider
**Returns**: `fid_fabric` handle

```c
// Open domain
fi_domain(fabric, info, &domain, NULL);
```

**Purpose**: Create resource domain for memory registration and endpoint creation
**Returns**: `fid_domain` handle

### 2. Address Vector Creation

```c
// Create address vector for peer addressing
struct fi_av_attr av_attr = {
    .type = FI_AV_TABLE,
    .count = 128  // Total ranks
};
fi_av_open(domain, &av_attr, &av, NULL);
```

**Purpose**: Create address vector to map rank IDs to libfabric addresses
**Returns**: `fid_av` handle

```c
// Insert peer addresses (done for all 127 remote peers)
for (int peer = 0; peer < 128; peer++) {
    if (peer != my_rank) {
        fi_av_insert(av, peer_addr, 1, &fi_addr[peer], 0, NULL);
    }
}
```

**Purpose**: Map each peer rank to its libfabric address
**Count**: 127 calls per rank (all peers except self)

### 3. Endpoint and Completion Queue Setup

For each rail (EFA adapter):

```c
// Create completion queue
struct fi_cq_attr cq_attr = {
    .format = FI_CQ_FORMAT_DATA,  // Includes immediate data
    .size = 8192
};
fi_cq_open(domain, &cq_attr, &cq, NULL);
```

**Purpose**: Create CQ to receive completion notifications
**Count**: 1 per rail (typically 4 rails on p5 instances)
**Returns**: `fid_cq` handle

```c
// Create endpoint
fi_endpoint(domain, info, &ep, NULL);
```

**Purpose**: Create communication endpoint
**Count**: 1 per rail
**Returns**: `fid_ep` handle

```c
// Bind CQ to endpoint
fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV);
```

**Purpose**: Associate CQ with endpoint for send/recv completions

```c
// Bind address vector
fi_ep_bind(ep, &av->fid, 0);
```

**Purpose**: Associate address vector with endpoint

```c
// Enable endpoint
fi_enable(ep);
```

**Purpose**: Activate endpoint for data transfer

### 4. Memory Registration

**Location**: `aws-ofi-nccl/src/rdma/gin/nccl_ofi_gin_resources.cpp:reg_mr()`

For send buffer:
```c
struct fi_mr_attr mr_attr = {
    .mr_iov = &iov,
    .iov_count = 1,
    .access = FI_WRITE | FI_REMOTE_WRITE,
    .iface = FI_HMEM_CUDA,  // GPU memory
    .device.cuda = gpu_device_id,
    .requested_key = mr_key
};
fi_mr_regattr(domain, &mr_attr, 0, &send_mr);
```

**Purpose**: Register GPU send buffer for RDMA access
**Count**: 1 per symmetric window (send + recv buffers)
**Returns**: `fid_mr` handle with memory key

For receive buffer:
```c
fi_mr_regattr(domain, &mr_attr, 0, &recv_mr);
```

**Purpose**: Register GPU receive buffer
**Count**: 1 per symmetric window

For signal buffer:
```c
mr_attr.access = FI_WRITE | FI_REMOTE_WRITE;
fi_mr_regattr(domain, &mr_attr, 0, &signal_mr);
```

**Purpose**: Register signal counter buffer (for remote increments)
**Count**: 1 per rank

### 5. Receive Buffer Posting

**Location**: `aws-ofi-nccl/src/rdma/gin/nccl_ofi_gin_reqs.cpp:post()`

```c
// Post receive buffers for metadata messages
for (int i = 0; i < NUM_RECV_BUFFS; i++) {
    fi_recv(ep, recv_buff[i], sizeof(metadata_msg), 
            desc, FI_ADDR_UNSPEC, &ctx);
}
```

**Purpose**: Pre-post receive buffers for incoming metadata messages
**Count**: ~64 per rail (configurable)
**Total**: ~256 fi_recv calls (4 rails × 64 buffers)

## Timing and Concurrency Model

### Initialization Phase Timing

**Sequential Operations** (per rank):
- `fi_getinfo()`: ~100-500 μs (one-time, cached)
- `fi_fabric()`: ~50 μs
- `fi_domain()`: ~100 μs
- `fi_av_open()`: ~50 μs
- `fi_av_insert()` × 127: ~10-20 μs total (batched)
- `fi_cq_open()` × 4: ~200 μs total
- `fi_endpoint()` × 4: ~400 μs total
- `fi_ep_bind()` × 8: ~80 μs total
- `fi_enable()` × 4: ~200 μs total
- `fi_mr_regattr()` × 3: ~300-500 μs total (GPU memory registration)
- `fi_recv()` × 256: ~2-5 ms total (pre-posting)

**Total Initialization**: ~5-10 ms per rank

**Concurrency**: All 128 ranks initialize in parallel
**Synchronization**: MPI barrier after initialization before starting all-to-all

### Kernel Launch and Barrier Synchronization

**Location**: `nccl/examples/06_device_api/03_gin_alltoall_hybrid/main.cu`

```cuda
__global__ void HybridAlltoAllKernel(...) {
    // BARRIER 1: Entry barrier (cross-node synchronization)
    ncclBarrierSession<ncclCoopCta> bar { ... };
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, 
             ncclGinFenceLevel::Relaxed);
    
    // ... GIN puts and LSA operations ...
    
    // BARRIER 2: Exit barrier (ensure all operations complete)
    bar.sync(ncclCoopCta(), cuda::memory_order_release, 
             ncclGinFenceLevel::Relaxed);
}
```

#### Barrier 1: Entry Synchronization

**Purpose**: Ensure all ranks are ready before starting data transfer
**Mechanism**: GIN barrier using network round-trip
**Implementation**: Each CTA (block) participates in barrier

**Libfabric Operations**:
```c
// Each rank sends barrier message to all peers
for (peer = 0; peer < 128; peer++) {
    fi_send(ep, barrier_msg, size, desc, peer_addr, &ctx);
}

// Poll for barrier completions from all peers
while (barrier_count < 128) {
    fi_cq_read(cq, cqe_buffers, count);
    // Process barrier messages
}
```

**Timing**: ~100-200 μs (network RTT + processing)
**Concurrency**: All 128 ranks synchronize simultaneously
**Blocking**: Kernel blocks until all ranks reach barrier


## Data Transfer Phase: putSignal Operations with Timing

### Concurrency Model

**Massive Parallelism**: All operations happen concurrently across:
- **128 ranks** (all initiating simultaneously)
- **16 CTAs per rank** (NCCL_DEVICE_CTA_COUNT)
- **512 threads per CTA** = 8,192 threads per rank
- **4 rails per rank** (round-robin distribution)

### Thread-Level Parallelism

**Location**: `nccl/examples/06_device_api/03_gin_alltoall_hybrid/main.cu`

```cuda
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int nthreads = blockDim.x * gridDim.x;  // 8192 threads

// Each thread handles subset of remote peers
for (int r = tid; r < startLsa; r += nthreads) {
    gin.put(world, r, ...);  // Thread 0 → peer 0, thread 1 → peer 1, etc.
}
for (int r = startLsa + lsaSize + tid; r < world.nRanks; r += nthreads) {
    gin.put(world, r, ...);
}
```

**Work Distribution**:
- 120 remote peers / 8192 threads = ~0.015 peers per thread
- Most threads handle 0 peers, first 120 threads handle 1 peer each
- **Effectively**: 120 threads active, each posting 1 putSignal

### Single putSignal Timing Breakdown

**Location**: `aws-ofi-nccl/src/rdma/gin/nccl_ofi_gin.cpp:iputSignal()`

#### Step 1: fi_writedata() - RDMA Write-with-Immediate

```c
fi_writedata(ep, src, size, desc, imm_data, remote_addr, dest, key, &ctx);
```

**Timing**:
- **Post time**: ~100-200 ns (software overhead)
- **Hardware latency**: ~2-5 μs (EFA write latency)
- **Completion**: Asynchronous, completes when data reaches remote NIC

**Concurrency**:
- **Non-blocking**: Returns immediately (or -FI_EAGAIN if queue full)
- **Pipelined**: Multiple writes in-flight simultaneously
- **Per-rail queue depth**: ~1024 outstanding operations

**Dependencies**: None (can post immediately)

#### Step 2: fi_send() - Signal Metadata

```c
fi_send(ep, &metadata, sizeof(metadata), desc, remote_addr, &ctx);
```

**Timing**:
- **Post time**: ~100-200 ns
- **Hardware latency**: ~2-5 μs (EFA send latency)
- **Message size**: 48 bytes (fits in single packet)

**Concurrency**:
- **Non-blocking**: Returns immediately
- **Independent**: Can post before fi_writedata() completes
- **Ordering**: EFA guarantees ordering per QP (endpoint)

**Dependencies**: None (independent of fi_writedata)

### Aggregate Timing for 120 putSignals per Rank

#### Posting Phase (Initiator)

**Sequential posting time** (if single-threaded):
- 120 × (fi_writedata + fi_send) = 120 × 400 ns = **48 μs**

**Parallel posting time** (120 threads):
- All 120 operations posted in **~400 ns** (2 calls per thread)

**Actual time**: ~1-2 μs (accounting for thread scheduling)

**Queue Depth**: 
- 120 fi_writedata() operations in-flight
- 120 fi_send() operations in-flight
- Distributed across 4 rails = ~30 operations per rail

#### Network Transfer Phase

**Overlapped execution**:
- All 120 writes execute in parallel (limited by network bandwidth)
- All 120 sends execute in parallel

**Network bottleneck**:
- **Per-rank bandwidth**: 400 Gbps (p5.48xlarge: 8 GPUs × 4 EFAs = 32 total, 400 Gbps per GPU)
  → 400 Gbps ÷ 8 bits/byte = **50 GB/s**
- **Data per rank**: 120 peers × 4 KB = 480 KB
- **Transfer time**: 480 KiB ÷ 50 GB/s = 491,520 B ÷ 50×10⁹ B/s = **~9.8 μs**

**Latency-bound** (for small messages):
- **RTT**: ~35 μs (estimate for p5en with EFAv3; not measured in this trace)
- **Pipelined**: First completion at ~35 μs, last at ~45 μs

**Total data transfer time**: **~35-50 μs** (latency-dominated for small messages)

### Completion Processing Phase

#### Receiver Side: fi_cq_readfrom() Polling

**Location**: `aws-ofi-nccl/src/rdma/gin/nccl_ofi_gin_resources.cpp`

```c
// Continuous polling loop (progress thread)
while (true) {
    rc = fi_cq_readfrom(cq, cqe_buffers, 64, src_addrs);
    if (rc > 0) {
        // Process completions
        for (int i = 0; i < rc; i++) {
            handle_cq_entry(&cqe_buffers[i], src_addrs[i], rail_id);
        }
    }
}
```

**Timing**:
- **Poll frequency**: Continuous (busy-wait)
- **Poll latency**: ~50-100 ns per call (when no completions)
- **Batch processing**: Up to 64 completions per call
- **Processing time**: ~50-100 ns per completion

**Concurrency**:
- **Per-rail thread**: 4 progress threads per rank (one per rail)
- **Parallel processing**: All 4 rails polled simultaneously
- **Lock-free**: No contention between rails

#### Processing 120 Incoming Operations

**Receive pattern**:
- 120 fi_writedata() completions (RDMA writes)
- 120 fi_recv() completions (metadata messages)
- Total: 240 completions per rank

**Batched processing**:
- 240 completions / 64 per batch = ~4 fi_cq_readfrom() calls
- 4 calls × 100 ns = **~400 ns** (polling overhead)
- 240 × 100 ns = **~24 μs** (processing time)

**Total receive processing**: **~25 μs**

#### Signal Execution Timing (host / proxy path)

**Location**: `aws-ofi-nccl/src/rdma/gin/nccl_ofi_gin.cpp:do_gin_signal_and_trace()` and the
shared gdrcopy worker loop (see `gin_signal_work_entry` / `work_queue`).

```c
// Signal increment via GDRCopy (read-modify-write on the CPU-mapped counter)
copy_from_device(*work.gdr_handle, signal_offset, &old_value, 8);  // Read
uint64_t new_value = old_value + signal_value;
copy_to_device(&new_value, *work.gdr_handle, signal_offset, 8);    // Write
```

**Current architecture (updated):**
- The plugin now runs **one gdrcopy worker per process** with **signal coalescing** (commit
  `ad6dcac`), rather than doing the read-modify-write inline on each progress thread.
- Progress threads hand signal work to the worker through a new lock-free **MPSC ring**
  ([include/nccl_ofi_mpsc_ring.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_mpsc_ring.h), commit `3e5687e`); the worker drains the ring and coalesces
  updates to the same counter.
- Signal memory is **registered one segment at a time** in the host proxy (commit `80d995c`).
- The **recv-req pool return is deferred until the gdrcopy signal completes** (commit
  `5a7bcba`), so a request is not recycled before its counter update is durable.
- **Signals are always strong** now: `OFI_NCCL_GIN_STRONG_SIGNAL` and the weak-signal mode
  were removed in commit `aa80b54`. The `isStrongSignal` argument carried by the
  `ncclRma_v14/v15` `iputSignal` prototypes is accepted and ignored.

**Timing per signal** (approximate, unchanged order of magnitude):
- **GDRCopy read**: ~200-300 ns
- **Increment**: ~1 ns
- **GDRCopy write**: ~200-300 ns
- **Total**: ~500-600 ns per signal (before coalescing; coalescing reduces the effective
  count when many updates target the same counter)

**120 signals**: up to 120 × 600 ns ≈ **~72 μs** worst case; less with coalescing.

**Concurrency**: signal RMW is funneled through the single per-process gdrcopy worker.

#### Acknowledgment Phase

**Location**: `aws-ofi-nccl/src/rdma/gin/nccl_ofi_gin.cpp:send_writedata_ack()`

```c
// Send ACK back to initiator (zero-length write with immediate)
fi_writedata(ep, write_ack_buff, 0, desc, ack_imm_data, 
             remote_addr, dest, key, &ctx);
```

**Timing**:
- **Post time**: ~100 ns
- **Network latency**: ~35 μs (one-way)
- **120 ACKs**: Posted in parallel, ~1 μs total posting time

**Dependencies**: Must complete after signal execution

### Initiator Side: ACK Processing

```c
// Poll for ACK completions
fi_cq_readfrom(cq, cqe_buffers, 64, src_addrs);

// Check for ACK and mark complete
if (total_segms == WRITEDATA_ACK_NSEG) {
    rank_comm.active_put_signal[msg_seq_num] = false;
}
```

**Timing**:
- **ACK arrival**: ~35 μs after ACK sent (network RTT)
- **Processing**: ~100 ns per ACK
- **120 ACKs**: ~12 μs processing time

### Barrier 2: Exit Synchronization

**Purpose**: Ensure all putSignal operations complete before kernel exits
**Mechanism**: GIN barrier + flush

```cuda
// Wait for all remote operations to complete
int numRemotePeers = 120;
gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + numRemotePeers);
gin.flush(ncclCoopCta());

// Final barrier
bar.sync(ncclCoopCta(), cuda::memory_order_release, 
         ncclGinFenceLevel::Relaxed);
```

#### waitSignal: Local Polling

```cuda
// Poll local signal counter until all 120 increments received
while (signal_counter < expected_value) {
    // Busy-wait on GPU
}
```

**Timing**:
- **Poll frequency**: Every ~10-50 ns (GPU memory read)
- **Wait time**: Until all 120 remote peers complete their putSignals
- **Typical**: ~100-200 μs (depends on slowest peer)

**Dependencies**: 
- All 120 remote putSignals must complete
- All 120 signals must be executed
- All 120 ACKs must be sent

#### gin.flush: Ensure Ordering

**Purpose**: Flush any pending operations and ensure memory visibility

**Libfabric operations**: None (local fence)

**Timing**: ~10-50 ns (memory fence)

#### Final Barrier

**Purpose**: Ensure all ranks have completed all operations

**Libfabric operations**: Same as Barrier 1
- 128 × fi_send() (barrier messages)
- 128 × fi_cq_read() (barrier completions)

**Timing**: ~100-200 μs

**Dependencies**: All local operations must complete first

## Complete End-to-End Timing

### Per-Rank Timeline

```
T=0:        Barrier 1 starts
T=100 μs:   Barrier 1 completes, all ranks synchronized

T=100 μs:   Start posting putSignals (120 operations)
T=102 μs:   All 120 putSignals posted (parallel threads)

T=102 μs:   Network transfer begins (overlapped)
T=137 μs:   First completions arrive at receivers (~35 μs RTT)
T=152 μs:   Last completions arrive (~50 μs for all)

T=137 μs:   Receivers start processing completions
T=162 μs:   Signal execution completes (~72 μs for 120 signals)
T=163 μs:   ACKs posted back to initiators (~1 μs)

T=198 μs:   ACKs arrive at initiators (~35 μs RTT)
T=210 μs:   All ACKs processed

T=210 μs:   waitSignal completes (all signals received)
T=210 μs:   gin.flush() completes
T=310 μs:   Barrier 2 completes (~100 μs)

Total: ~310 μs per rank
```

### Critical Path Dependencies

**Dependency Chain**:
1. **Barrier 1** → All ranks ready
2. **Post putSignals** → Network transfer starts
3. **Network transfer** → Receiver gets data
4. **Receiver processing** → Signal execution
5. **Signal execution** → ACK sent
6. **ACK transfer** → Initiator marks complete
7. **waitSignal** → All signals received
8. **Barrier 2** → All ranks complete

**Blocking Points**:
- **Barrier 1**: All ranks must reach before any proceed
- **waitSignal**: Each rank waits for its 120 signals
- **Barrier 2**: All ranks must complete before any exit

### Concurrency Summary

**Parallel Operations**:
- ✅ All 128 ranks operate simultaneously
- ✅ All 120 putSignals per rank posted in parallel
- ✅ Network transfers overlapped across all peers
- ✅ 4 rails per rank handle operations concurrently
- ✅ Receiver processes completions in parallel across rails

**Sequential Operations**:
- ❌ Barrier 1 blocks until all ranks ready
- ❌ Signal execution per rank is sequential (120 × 600 ns)
- ❌ waitSignal blocks until all 120 signals received
- ❌ Barrier 2 blocks until all ranks complete

**Bottlenecks**:
1. **Network latency**: ~35 μs RTT (dominant factor)
2. **Signal execution**: ~72 μs per rank (sequential)
3. **Barrier synchronization**: ~100-200 μs (twice)

**Total Time**: ~310 μs for 128 ranks, 120 remote peers each

## NCCL-Tests Timing Methodology

### What is Measured

**Source**: `nccl-tests/src/common.cu:BenchTime()`

The nccl-tests timing **excludes initialization** and only measures **data transfer operations**.

### Timing Sequence

```c
// 1. INITIALIZATION (NOT TIMED)
TESTCHECK(args->collTest->initData(args, type, op, root, 99, in_place));

// 2. WARMUP (NOT TIMED)
TESTCHECK(startColl(args, type, op, root, in_place, 0));
TESTCHECK(completeColl(args));
Barrier(args);  // Synchronize all ranks

// 3. TIMED SECTION STARTS HERE
timer tim;  // CPU wall-clock timer starts

for (int iter = 0; iter < iters; iter++) {  // Default: 20 iterations
    if (agg_iters>1) NCCLCHECK(ncclGroupStart());
    for (int aiter = 0; aiter < agg_iters; aiter++) {
        TESTCHECK(startColl(args, type, op, root, in_place, iter*agg_iters+aiter));
    }
    if (agg_iters>1) NCCLCHECK(ncclGroupEnd());
}

// 4. WAIT FOR COMPLETION (STILL TIMED)
TESTCHECK(completeColl(args));  // cudaStreamSynchronize()

double deltaSec = tim.elapsed();  // Timer stops
deltaSec = deltaSec / (iters * agg_iters);  // Average per iteration

// 5. COMPUTE BANDWIDTH
args->collTest->getBw(count, wordSize(type), deltaSec, &algBw, &busBw, nranks);
```

### What is Included in Timing

**Included** ✅:
1. **Kernel launch overhead**: ~5-10 μs per launch
2. **Barrier synchronization**: Entry and exit barriers (~200 μs total)
3. **Data transfer**: All putSignal operations (~35-50 μs)
4. **Signal execution**: Remote signal increments (~72 μs)
5. **Acknowledgments**: ACK round-trip (~35 μs)
6. **waitSignal polling**: Waiting for all signals (~100-200 μs)
7. **Stream synchronization**: Final cudaStreamSynchronize()

**Excluded** ❌:
1. **NCCL initialization**: `ncclCommInitRank()` (~100-500 ms)
2. **Memory allocation**: `ncclMemAlloc()` (~10-50 ms)
3. **Memory registration**: `ncclCommWindowRegister()` (~1-10 ms)
4. **Device communicator creation**: `ncclDevCommCreate()` (~10-50 ms)
5. **Warmup iteration**: First call to ensure everything is initialized
6. **Data initialization**: Setting up send/receive buffers
7. **Validation**: Data correctness checking (done separately)

### Timing Breakdown for Hybrid All-to-All

For a single iteration with 128 ranks, 120 remote peers each:

```
Component                          Time        Included in nccl-tests?
─────────────────────────────────────────────────────────────────────
Initialization Phase:
  - ncclCommInitRank()             100-500 ms   ❌ NO
  - ncclMemAlloc()                 10-50 ms     ❌ NO
  - ncclCommWindowRegister()       1-10 ms      ❌ NO
  - ncclDevCommCreate()            10-50 ms     ❌ NO
  - Warmup iteration               ~500 μs      ❌ NO

Timed Section (per iteration):
  - Kernel launch                  ~10 μs       ✅ YES
  - Barrier 1 (entry)              ~100 μs      ✅ YES
  - Post putSignals                ~2 μs        ✅ YES
  - Network transfer               ~35-50 μs    ✅ YES
  - Signal execution               ~72 μs       ✅ YES
  - ACK round-trip                 ~35 μs       ✅ YES
  - waitSignal polling             ~100 μs      ✅ YES
  - Barrier 2 (exit)               ~100 μs      ✅ YES
  - Stream synchronization         ~1 μs        ✅ YES
─────────────────────────────────────────────────────────────────────
Total per iteration:               ~455 μs      ✅ YES
Total initialization:              ~150 ms      ❌ NO
```

### Reported Metrics

**Algorithm Bandwidth (algBw)**:
```c
// For all-to-all: (N-1) * count * typesize / time
double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;
*algBw = baseBw;
```

**Bus Bandwidth (busBw)**:
```c
// For all-to-all: same as algBw (no reduction factor)
*busBw = baseBw;
```

**Example Output**:
```
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
      524288        131072     float    none      -1    455.2   1150.0  1150.0    N/A    455.2  1150.0  1150.0    N/A
```

### Key Observations

1. **Initialization is expensive**: ~150 ms one-time cost
2. **Amortized over iterations**: Default 20 iterations means initialization is ~7.5 ms per iteration if amortized
3. **Steady-state performance**: nccl-tests reports steady-state bandwidth, not including setup
4. **Warmup is critical**: First iteration may be slower due to:
   - P2P connection establishment
   - Memory registration cache misses
   - EFA queue pair initialization
5. **Reported time is average**: Divided by number of iterations (default 20)

### For GIN Hybrid All-to-All Specifically

**Estimated time per iteration**: ~455 μs — this is the arithmetic sum of the component
estimates below, not a benchmark result:
- Dominated by network latency (~35 μs RTT estimate)
- Barriers add ~200 μs overhead
- Signal execution adds ~72 μs

**Not measured**:
- GIN communicator setup: ~50 ms
- Symmetric window registration: ~10 ms per window
- Receive buffer pre-posting: ~5 ms

**Total application time** (including initialization):
- First iteration: ~150 ms + 455 μs ≈ **150.5 ms**
- Subsequent iterations: **455 μs each**
- 20 iterations: 150 ms + (20 × 455 μs) ≈ **159 ms total**
- **Estimated steady-state average**: 455 μs (excludes initialization). Treat as an
  order-of-magnitude figure derived from the component estimates above; measure on your
  own instance before drawing conclusions.



---

## GDAKI (EFA-GDA) Trace

The trace above is the **host / CPU-proxy** path. On EFA-GDA-capable instances (P5en,
P6-B200, P6-B300) with a GDAKI-capable stack, NCCL binds the **`ncclGinPlugin_v14`** GDAKI
op-table ([src/rdma/gin/nccl_ofi_gin_gdaki.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_gdaki.cpp), `name = "Libfabric_GDAKI"`) instead. The plugin
auto-detects GDAKI via `nccl_ofi_gin_gdaki_capable()`
([include/rdma/gin/nccl_ofi_gin_gdaki.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/rdma/gin/nccl_ofi_gin_gdaki.h)): GDAKI compiled in (`HAVE_GDAKI`) **and** runtime
libfabric ≥ 2.5 **and** DMA-BUF viable **and** an EFA provider. `OFI_NCCL_GIN_TYPE` no longer
exists (removed in commit `80f2c78`); on the NCCL side `NCCL_GIN_TYPE=5` selects this table
because NCCL 2.31 otherwise defaults to the proxy table for EFA.

### How the data path differs from the proxy trace

| Stage | Host / proxy path (above) | GDAKI / EFA-GDA path |
|-------|---------------------------|----------------------|
| Who posts the op | CPU progress thread calls `fi_writedata()` / `fi_send()` | **GPU kernel** writes the WQE into the QP's send queue in GPU-visible memory and rings the doorbell — no plugin/host call per op |
| Completion | CPU progress thread polls `fi_cq_read()` / `fi_cq_readfrom()` | **GPU kernel** reads the CQE directly from the CQ buffer |
| Signal delivery | gdrcopy read-modify-write via the per-process gdrcopy worker + MPSC ring | **indexed hardware signals / counters** on the EFA device (target-indexed, asymmetric counts — commit `f3dd9cd`); GDRCopy registration is skipped entirely in GDAKI mode (commit `5b163ed`) |
| Metadata `fi_send` | separate 48-byte metadata send per putSignal | not used the same way; signaling is via device counters/signals |
| CPU involvement | GPU→CPU→NIC on the critical path | **none** on the critical path |

### GDAKI context / resource setup (once per context)

**Location**: `createContext_v14` in
[src/rdma/gin/nccl_ofi_gin_gdaki.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_gdaki.cpp) and
[src/rdma/gin/nccl_ofi_gin_gdaki_resources.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_gdaki_resources.cpp).

1. `createContext_v14` **validates `config->backendVersion`** against
   `NCCL_OFI_GDAKI_MAX_BACKEND_VERSION` (commit `f8e945e`) — it refuses a version it cannot
   build a matching device struct for, rather than risk silent memory corruption.
2. Per logical context `c`, bind to **rail `c % num_rails`**
   (`effective_rails = min(nContexts, num_rails)`, capped at `NCCL_OFI_GDAKI_MAX_RAILS = 2`,
   commit `cb00654`). Each rail's endpoints open on that rail's libfabric domain.
3. Open the GDAKI endpoint(s) with **`FI_HMEM` requested on the hints** (commit `06f08e5`),
   plus a **dedicated data endpoint used only for sending PutValues** (commit `6a3ba5f`)
   whose counter is bound for **reads as well as writes** (commit `d840aa1`).
4. Map the EFA MMIO doorbell region into the GPU address space (requires the NVIDIA driver
   with `PeerMappingOverride=1`); the GPU rings the doorbell directly.
5. Set up **indexed signal shadowing** to avoid a signal read-modify-write (commit
   `dea6e05`); the never-reset signal shadow is sized from the **full MR** (commit `5b46cd5`).
   Counter device handles get `local_cntr_value` populated (commit `44d0b55`).
6. The **EFA hardware completion counter** is gated per platform via
   `OFI_NCCL_GDAKI_EFA_HW_COUNTER` (`AUTO`/`ON`/`OFF`, default `AUTO`) and
   `PlatformAWS::config_gdaki_domain()` (commit `5b1f6dd`).
7. Doorbell coalescing uses **`FI_MORE`** on the aggregate hint (commit `a3d2680`).

### Per-op data path (per putSignal, GDAKI)

Because there is no plugin/host call per op, there is **no libfabric `fi_*` call per
putSignal** on the GDAKI data path. The GPU kernel:

1. Builds the SRD send WQE for the RDMA write in the QP's GPU-visible send queue.
2. Rings the EFA doorbell (MMIO write) — the WQE, and coalesced doorbells with `FI_MORE`
   where applicable, go straight to the NIC.
3. Signals the target via an **indexed device signal/counter** (target-indexed addressing),
   not a gdrcopy RMW.
4. Polls the CQ buffer directly (and/or the hardware completion counter) to observe
   completion.

The EFA provider's **data-path-direct** code (default on) is what makes the direct
WQE-write / CQE-read possible; `ibv_post_send()` / `ibv_poll_cq()` are not on this path.

### Timing intuition (GDAKI vs proxy)

For the same 16-node / 128-rank / 120-remote-peer alltoall:

- **Proxy path**: dominated by the GPU→CPU→NIC hop and the gdrcopy signal RMW; per-iteration
  time is in the ~455 μs range estimated above.
- **GDAKI path**: removes the CPU hop and the gdrcopy signal RMW from the critical path. The
  network RTT (~35 μs) and barrier overhead still dominate, but the per-op software cost
  drops sharply because the GPU posts and reaps completions directly. Expect lower dispatch/
  combine latency than the proxy path on the same hardware (exact figures depend on instance
  type and message size; treat proxy-vs-GDAKI deltas as workload-specific).

> **Note:** `NCCL_SYM_GIN_KERNELS_ENABLE=0` is currently required with EFA-GDA — the
> symmetric GIN kernels for NCCL collectives are not yet supported on that path (see the
> plugin's GIN getting-started guide).
