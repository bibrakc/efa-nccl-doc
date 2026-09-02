# Passthrough Modes and Optimizations

## Overview

This document covers advanced optimizations throughout the NCCL + OFI + libfabric + EFA stack. These techniques reduce overhead and improve both latency and bandwidth.

## Software Stack Overhead

### Typical Overhead Per Layer

```
Layer                      Overhead (per operation)
─────────────────────────────────────────────────────
NCCL collective breakdown  ~1-2 μs
NCCL → OFI plugin call     ~0.5-1 μs
OFI plugin processing      ~1-2 μs
  ├─ MR cache lookup       ~0.1-1 μs (hit)
  └─ State management      ~0.5 μs
Libfabric API              ~0.5-1 μs
  ├─ Endpoint lookup       ~0.2 μs
  └─ Validation            ~0.3 μs
EFA provider processing    ~1-2 μs
  ├─ WQE building          ~0.5 μs
  └─ Doorbell              ~0.5 μs
─────────────────────────────────────────────────────
Total software overhead:   ~5-10 μs
Network/hardware:          ~10-20 μs
─────────────────────────────────────────────────────
Total end-to-end:          ~15-30 μs
```

**Goal**: Minimize software overhead to get closer to hardware limits.

## FI_EFA_USE_DATA_PATH_DIRECT — the biggest current data-path win (default ON)

The single largest recent optimization in the stack is libfabric's EFA
**direct data path**, which now **defaults to TRUE**
([prov/efa/src/efa_env.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_env.c):
`.use_data_path_direct = true`; knob `FI_EFA_USE_DATA_PATH_DIRECT`).

### What it does

Normally libfabric issues sends/receives and polls completions through
rdma-core / libibverbs (`ibv_post_send`, `ibv_poll_cq`, …). The direct data path
**bypasses rdma-core on the data path entirely**: the EFA provider maps the
hardware SQ/RQ/CQ buffers and doorbells and writes work-queue entries and reads
completions itself
([prov/efa/src/efa_data_path_direct.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_data_path_direct.c),
[efa_data_path_ops.h](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_data_path_ops.h)).
It still uses rdma-core's `efadv` query interfaces to *discover* the hardware
queue attributes at setup; only the per-operation fast path is bypassed.

```c
// efa_data_path_ops.h: every data-path op checks the flag and routes to the
// direct implementation when enabled.
#if HAVE_EFA_DATA_PATH_DIRECT
    if (qp->data_path_direct_enabled)
        return efa_data_path_direct_post_send(qp, ...);
#endif
```

Direct path is enabled per-QP/CQ at creation
(`efa_data_path_direct_qp_initialize`, `efa_data_path_direct_cq_initialize`),
gated by `efa_env.use_data_path_direct` and the device supporting it
([efa_data_path_direct.c:187-214](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_data_path_direct.c)).

### 128-byte wide WQE format

The direct path can build **128-byte "wide" work-queue entries**
(`struct efa_io_tx_wqe_128`), versus the standard narrower WQE
([efa_data_path_direct_internal.h:712,752](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_data_path_direct_internal.h)).
Wide WQEs carry more inline data per entry (the provider gates this on
`efa_device_support_wide_wqe()` and it trades TX queue depth for inline
capacity — see the "wide WQE" unit tests in
`prov/efa/test/efa_unit_test_info.c`). This lets small messages be posted
inline in a single wide WQE instead of referencing a separate buffer.

### 64-bit request IDs

The direct SQ can use **64-bit request IDs** for completion matching
(`FI_EFA_USE_SQ_REQ_ID_64_BIT`, default ON: `efa_env.c` `.use_sq_req_id_64_bit
= 1`). The WQE metadata carries the full 64-bit `wr_id`
(`efa_set_req_id_64` / `efa_get_req_id_64`,
[efa_data_path_direct_internal.h:143-209](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_data_path_direct_internal.h)),
so the provider matches completions against a 64-bit ID rather than a narrower
one — avoiding an indirection to map a small hardware ID back to a request.

### Practical guidance

- It is on by default; you generally want it on. Disable only for debugging:
  `FI_EFA_USE_DATA_PATH_DIRECT=0` (falls back to the rdma-core data path).
- It requires a build with `HAVE_EFA_DATA_PATH_DIRECT` and a device that
  supports it; otherwise the flag is ignored and the standard path is used.

## GDAKI Auto-Enabled; Removed GIN Knobs

**GDAKI** (GPU-Direct Async Kernel-Initiated) is now **auto-enabled per
platform**; there is no user-selectable GIN "type" anymore.

- **`OFI_NCCL_GIN_TYPE` was removed** (commit **80f2c78**). Do not set it. GDAKI
  is selected automatically on platforms that support it. (A stale
  `NCCL_GIN_TYPE=5` example survives only in the plugin's
  `doc/gin-getting-started.md`.)
- **`OFI_NCCL_GIN_STRONG_SIGNAL` was removed** (commit **aa80b54**). Strong
  signal ordering is now handled internally via producer-side coalescing in the
  shared gdrcopy worker (see threading-model.md).
- **New knob `NCCL_GIN_PROXY_NTHREADS`** (commit **e6c4eb1**, default 1) selects
  the number of GIN progress threads, each bucketed onto its own endpoint. Note
  it is read via `getenv("NCCL_GIN_PROXY_NTHREADS")`, not via the `OFI_NCCL_`
  param machinery. See threading-model.md for the bucketing scheme.

## Doorbell Coalescing with FI_MORE (GIN)

The GIN data path coalesces doorbells using libfabric's `FI_MORE` flag
(commit **a3d2680**). When NCCL passes its
`ncclRmaOptFlagsAggregateRequests` hint (meaning another op for the same peer is
already queued), the GIN layer withholds the current op's doorbell by posting it
with `FI_MORE` and lets a later op on the same "pinned" rail ring the doorbell
for the batch
([src/rdma/gin/nccl_ofi_gin.cpp:652-793](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin.cpp)):

```c
// A single-stripe aggregating op withholds its doorbell...
wr_flags |= FI_MORE;   // defer; a later op on the pinned rail rings it
```

Progress/hang safety is built in: `defer` implies `aggregate`, and NCCL always
ends a batch with a non-aggregate op, which does not defer and is forced onto
the pinned rail — so it flushes the deferred doorbell. A rotation boundary
(every `GIN_REQS_PER_DOORBELL`) is a second flush point. On retry, `FI_MORE` is
always dropped
([nccl_ofi_gin_reqs.cpp:260-303](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_reqs.cpp)),
since a queued-for-retry request must not stay deferred.

**Benefit:** batching several small ops behind one MMIO doorbell write amortizes
the doorbell cost, which is significant on the latency-sensitive GIN path.

## EFA Hardware Completion Counters (GDAKI, per-platform)

The EFA kernel driver r3.3.0 **added Completion Counters support**
(`amzn-drivers/kernel/linux/efa/RELEASENOTES.md`: "Add Completion Counters
support"). In the plugin, EFA hardware completion counters used by GDAKI are
**gated per platform** (commit **5b1f6dd**):

- The platform table field `efa_hw_comp_cntr`
  ([platform-aws.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/platform-aws.cpp))
  is `true` **only** for `p5en/p6-b200` and `p6-b300`; `false` everywhere else.
- There is an override env var `OFI_NCCL_GDAKI_EFA_HW_COUNTER` (used by the
  platform-mapper unit test to force-authorize) but the safe default is the
  per-platform opt-in above.

Do not assume hardware completion counters are available; they are only wired up
on the GDAKI-capable platforms listed above and require the r3.3.0 driver.

## EFA / Plugin Environment Variables (verified defaults)

The following defaults are read directly from source as of this refresh
(libfabric `prov/efa/src/efa_env.c`, aws-ofi-nccl
`include/nccl_ofi_param.h`). Where a libfabric queue size shows `0`, that means
"provider chooses".

### libfabric EFA provider (`FI_EFA_*`)

| Variable | Default | Notes / source |
|----------|---------|----------------|
| `FI_EFA_USE_DATA_PATH_DIRECT` | **true** | Bypass rdma-core on data path (`.use_data_path_direct = true`) |
| `FI_EFA_USE_SQ_REQ_ID_64_BIT` | **1** | 64-bit SQ request IDs (unregistered param) |
| `FI_EFA_USE_HW_CNTR` | **1** | HW counter usage (unregistered param) |
| `FI_EFA_ENABLE_SHM_TRANSFER` | **1** | Intra-node via SHM provider (NOT 0) |
| `FI_EFA_USE_ZCPY_RX` | **1** | Zero-copy receive when feasible |
| `FI_EFA_USE_UNSOLICITED_WRITE_RECV` | **1** | Device unsolicited write-recv |
| `FI_EFA_USE_SM2` | **false** | Experimental SM2 intra-node provider |
| `FI_EFA_TX_SIZE` | **0** | 0 => provider default (NOT 256) |
| `FI_EFA_RX_SIZE` | **0** | 0 => provider default (NOT 256) |
| `FI_EFA_TX_MIN_CREDITS` | **32** | Aborts if set ≤ 0 |
| `FI_EFA_CQ_SIZE` | **8192** | Completion queue size |
| `FI_EFA_MAX_MEMCPY_SIZE` | **4096** | memcpy-vs-registration threshold |
| `FI_EFA_RECVWIN_SIZE` | `EFA_RDM_PEER_DEFAULT_REORDER_BUFFER_SIZE` | sliding recv window |
| `FI_EFA_READCOPY_POOL_SIZE` | **256** | readcopy packet pool |
| `FI_EFA_EFA_CQ_READ_SIZE` | **50** | CQEs read per progress iteration |
| `FI_EFA_INTER_MAX_GDRCOPY_MESSAGE_SIZE` | **32768** | gdrcopy message cutoff |
| `FI_EFA_INTER_READ_SEGMENT_SIZE` | **1073741824** | RDMA read segmentation |
| `FI_EFA_RUNT_SIZE` | **307200** | runting read-protocol eager bytes |
| `FI_EFA_INTERNAL_RX_REFILL_THRESHOLD` | **8** | internal rx pkt pool refill |
| `FI_EFA_IMPLICIT_AV_SIZE` | **0** | 0 => unbounded implicit AV |
| `FI_EFA_TRACK_MR` | **0 / false** | debug MR-leak tracking |
| `FI_EFA_FORK_SAFE` | **false** | fork support (disables huge pages) |
| `FI_EFA_USE_DEVICE_RDMA` | (device-dependent) | one/two-sided device RDMA if supported |
| `FI_EFA_MTU_SIZE` | **REMOVED — fatal** | provider aborts if set |
| `FI_EFA_TX_IOV_LIMIT` | **REMOVED — fatal** | provider aborts if set |
| `FI_EFA_RX_IOV_LIMIT` | **REMOVED — fatal** | provider aborts if set |
| `FI_EFA_SET_CUDA_SYNC_MEMOPS` | deprecated (info) | logs a deprecation notice |
| `FI_EFA_SHM_MAX_MEDIUM_SIZE` | deprecated (info) | logs a deprecation notice |
| `FI_EFA_ZCPY_RX_SEED` | deprecated (info) | logs a deprecation notice |

### aws-ofi-nccl plugin (`OFI_NCCL_*` unless noted)

| Variable | Default | Notes / source |
|----------|---------|----------------|
| `OFI_NCCL_EAGER_MAX_SIZE` | **-1 (eager disabled)** | `nccl_ofi_param.h:229`; positive value re-enables eager |
| `OFI_NCCL_RDMA_MIN_POSTED_EAGER_BUFFERS` | **64** | `nccl_ofi_param.h:187` |
| `OFI_NCCL_RDMA_MAX_POSTED_EAGER_BUFFERS` | **128** | `nccl_ofi_param.h:194` |
| `OFI_NCCL_GIN_CQ_PROCESS_MAX_ITER` | **4** | `nccl_ofi_param.h:109` |
| `NCCL_GIN_PROXY_NTHREADS` | **1** | plain `getenv`, GIN per-thread endpoints |
| `OFI_NCCL_GDAKI_EFA_HW_COUNTER` | per-platform opt-in | override for GDAKI HW counter authorization |
| `OFI_NCCL_GIN_TYPE` | **REMOVED** | GDAKI auto-enabled (commit 80f2c78) |
| `OFI_NCCL_GIN_STRONG_SIGNAL` | **REMOVED** | handled internally (commit aa80b54) |

> Deprecated aggregate knobs `OFI_NCCL_RDMA_MIN_POSTED_BOUNCE_BUFFERS` /
> `..._MAX_...` were split into the separate CONTROL and EAGER variants above
> (`nccl_ofi_param.h:169-180`).

Always cross-check names/defaults against the source for your build; the plugin
and provider evolve quickly. See also
`aws-ofi-nccl.wiki/Environment-Variables.md` for the full plugin list and
`man/fi_efa.7.md` for the provider list.

## Kernel Bypass (User-Space I/O)

### Traditional Network I/O

```
Application
    ↓ write(socket, ...)
Syscall boundary
    ↓
Kernel Network Stack
    ↓ TCP/IP processing
    ↓ Copy to kernel buffers
Driver
    ↓
Hardware

Overhead: ~50-100 μs per operation
```

### EFA Kernel Bypass

```
Application
    ↓ ibv_post_send() [user space]
    ↓ Write WQE to mapped memory
    ↓ MMIO doorbell write
Hardware
    ↓ DMA directly from user memory

Overhead: ~0.1-0.5 μs (significant speedup vs syscalls)
```

**Mechanisms:**
- **Memory mapping**: Queues mapped to user space via `mmap()`
- **MMIO doorbells**: Direct hardware register access
- **Completion polling**: User reads CQ from mapped memory
- **No syscalls**: Entire data path in user space

**Already in Use:**
- EFA driver inherently supports this
- Libfabric leverages it
- NCCL benefits automatically

## Memory Registration Optimizations

### Multi-Level Caching

```
Request for buffer 0x1000, size 4MB
    │
    ▼
┌─────────────────────────────┐
│  OFI Plugin MR Cache        │ ← Exact match cache
│  (addr, size) → mr          │    ~0.1 μs lookup
└──────────┬──────────────────┘
           │ Miss
           ▼
┌─────────────────────────────┐
│  Libfabric Provider Cache   │ ← Range-based cache
│  Overlap detection          │    ~0.5 μs lookup
└──────────┬──────────────────┘
           │ Miss
           ▼
┌─────────────────────────────┐
│  Register (ibv_reg_mr)      │ ← Actual registration
│  100-500 μs                 │
└─────────────────────────────┘
```

### On-Demand Registration (ODR)

```c
// NCCL uses buffers on-demand
// Register when first used, cache for reuse

ncclResult_t optimized_send(void* buf, size_t size) {
  // Try cache
  void* mr = mr_cache_find(buf, size);

  if (!mr) {
    // Register on-demand
    mr = fi_mr_reg(domain, buf, size, ...);
    mr_cache_insert(buf, size, mr);
  }

  // Use for transfer
  fi_send(ep, buf, size, fi_mr_desc(mr), ...);
}
```

### Pre-Registration

For known buffers:

```c
// At initialization, pre-register all GPU buffers
void init_nccl() {
  for (int i = 0; i < NUM_GPUS; i++) {
    cudaMalloc(&gpu_bufs[i], BUF_SIZE);

    // Pre-register
    fi_mr_reg(domain, gpu_bufs[i], BUF_SIZE,
              access, 0, 0, FI_HMEM_CUDA, &mrs[i], NULL);
  }
}

// At runtime: zero registration overhead!
void collective() {
  // MR already registered
  fi_send(ep, gpu_bufs[0], size, fi_mr_desc(mrs[0]), ...);
}
```

### Registration Cache Monitoring

Efficient cache invalidation:

```c
// Using memhooks (LD_PRELOAD)
void* malloc(size_t size) {
  void* ptr = real_malloc(size);
  // Track allocation
  return ptr;
}

void free(void* ptr) {
  // Invalidate any cached MRs
  mr_cache_remove(ptr);
  real_free(ptr);
}

// Using userfaultfd (kernel support)
void invalidation_thread() {
  while (1) {
    struct uffd_msg msg;
    read(uffd, &msg, sizeof(msg));

    if (msg.event == UFFD_EVENT_UNMAP) {
      mr_cache_invalidate(msg.arg.remove.start,
                         msg.arg.remove.end);
    }
  }
}
```

**Choose based on:**
- **memhooks**: Simple, works everywhere, some overhead
- **userfaultfd**: Kernel 4.11+, lower overhead, more complex

## Completion Handling Optimizations

### Batching

```c
// Bad: Poll after each operation
for (int i = 0; i < N; i++) {
  fi_send(...);
  fi_cq_read(...);  // N polls
}

// Good: Batch operations
for (int i = 0; i < N; i++) {
  fi_send(...);
}
fi_cq_read(..., N);  // 1 poll for all
```

**Benefits:**
- Amortize polling overhead
- Better CPU cache utilization
- Higher throughput

### Adaptive Polling

```c
int idle_count = 0;
int backoff_threshold = 1000;

while (active) {
  ret = fi_cq_read(cq, entries, batch_size);

  if (ret > 0) {
    // Got completions
    idle_count = 0;
    process_completions(entries, ret);
  } else {
    // No completions
    idle_count++;

    if (idle_count < backoff_threshold) {
      // Busy poll (low latency)
      _mm_pause();  // CPU hint
    } else {
      // Back off (save CPU)
      usleep(1);  // or nanosleep
      backoff_threshold *= 2;  // Exponential backoff
    }
  }
}
```

**Benefits:**
- Low latency when active
- Reduced CPU usage when idle
- Configurable trade-off

### Selective Completion

```c
// Not all operations need completion events

// Normal send (generates completion)
fi_send(ep, buf, size, desc, addr, context);

// Send without completion (if supported)
fi_send(ep, buf, size, desc | FI_INJECT_COMPLETE,
        addr, NULL);  // NULL context = no completion

// Poll less frequently
```

**Use for:**
- Fire-and-forget sends
- Reduce CQ load
- Higher throughput (fewer CQEs)

## Protocol Optimizations

### Eager Send (fi_inject)

For very small messages:

```c
// Standard send (async, requires MR, generates completion)
fi_send(ep, buf, size, desc, addr, context);

// Inject send (sync, no MR, no completion)
if (size <= inject_size) {
  fi_inject(ep, buf, size, addr);
  // Data copied immediately, buf can be reused
}
```

**Characteristics:**
- **Copy overhead**: Data copied to internal buffer
- **No MR needed**: No registration required
- **No completion**: Synchronous, returns when done
- **Size limit**: Typically 32-4096 bytes (EFA: ~4KB)

**Trade-off:**
- Small messages (< 1 KB): Faster due to no MR/completion overhead
- Large messages: Slower due to copy overhead

> This `fi_inject` discussion is about the **libfabric** inject path. It is
> distinct from the aws-ofi-nccl RDMA transport's own **eager-message
> protocol**, which is covered next and behaves differently in current builds.

### aws-ofi-nccl Eager Messages: DISABLED BY DEFAULT (correction)

> **Correction.** Earlier revisions of this document recommended relying on the
> plugin's eager protocol for small messages. **Eager messages in the
> aws-ofi-nccl RDMA transport are now disabled by default.**

The knob is `OFI_NCCL_EAGER_MAX_SIZE`
([include/nccl_ofi_param.h:224-229](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_param.h)):

```c
/* Eager message size limit when using RDMA protocol. Message sizes greater than
 * this limit will always be sent using RDMA write instead of eagerly. Setting
 * this to -1 disables eager messages entirely. */
OFI_NCCL_PARAM(int, eager_max_size, "EAGER_MAX_SIZE", -1);
```

The **default is `-1`, which disables eager messages entirely** (commits
**7df78cd**, **708572b**). With the default, *all* RDMA-transport messages —
including small ones — are sent via RDMA write rather than eagerly copied into a
pre-posted receive buffer.

**How to re-enable it and the tradeoffs.** Set `OFI_NCCL_EAGER_MAX_SIZE` to a
positive byte threshold; messages at or below it are sent eagerly:

```bash
# Send messages up to 8 KB eagerly (example)
OFI_NCCL_EAGER_MAX_SIZE=8192
```

- **Pro:** for very small messages, eager avoids the control-message round trip
  of the rendezvous/RDMA-write path, lowering latency.
- **Con:** eager consumes pre-posted receive (bounce) buffers and adds a copy on
  the receive side; it also interacts with the eager-message bookkeeping
  (per-batch `eager_seq`, multi-recv matching). The upstream decision to default
  it off reflects that RDMA write is the better default across the common size
  distribution on EFA.

Do **not** design tuning around eager being on by default; verify the value of
`OFI_NCCL_EAGER_MAX_SIZE` for your build before relying on eager behavior.

### Rendezvous Optimization

For large messages, use RDMA write:

```c
// Traditional: fi_send/fi_recv (two-sided)
Sender: fi_send(buf, 1GB)
Receiver: fi_recv(buf, 1GB)
// Both CPUs involved, messaging overhead

// Optimized: fi_write (one-sided)
// 1. Exchange buffer info (once)
// 2. Direct RDMA write
fi_write(ep, local_buf, 1GB,
         desc, addr, remote_buf_addr, rkey, ctx);
// Only sender CPU involved
// Receiver CPU free for other work
```

**Benefits:**
- Zero-copy on receiver
- No receiver CPU overhead
- Better for large transfers (> 1 MB)

### Protocol Selection

NCCL protocol auto-selection:

```
Message Size    Protocol    Rationale
─────────────────────────────────────────
< 32 KB         LL          Low latency, flag-based
32 KB - 2 MB    LL128       Balance latency/bandwidth
> 2 MB          Simple      High bandwidth, fewer protocol overheads
```

**Tuning:**
```bash
# Override auto-selection
NCCL_PROTO=LL,LL128,Simple  # Allow all
NCCL_PROTO=Simple           # Force Simple

# Adjust thresholds (internal, not user-configurable)
# NCCL chooses based on benchmarking
```

## Channel and Parallelism Optimizations

### Channel Tuning

```bash
# Default: 4-16 channels depending on topology
NCCL_NCHANNELS=8

# More channels = more parallelism
# But diminishing returns and overhead

Channel Count   Bandwidth    Overhead
─────────────────────────────────────
1               12 GB/s      Low
2               22 GB/s      Low
4               40 GB/s      Medium
8               48 GB/s      Medium-High
16              50 GB/s      High
32              50 GB/s      Very High (not recommended)
```

**Optimal:**
- 4-8 channels for most workloads
- More channels if many NICs available

### Multi-Rail Utilization

Explicitly use all EFA devices:

```c
// NCCL auto-detects multiple EFAs
// Each EFA appears as separate device

int num_devs;
ncclNet->devices(&num_devs);  // Returns 4 for p4d.24xlarge

// NCCL distributes channels across devices
// Channel 0 → EFA 0
// Channel 1 → EFA 1
// Channel 2 → EFA 2
// Channel 3 → EFA 3
// Channel 4 → EFA 0 (wrap around)
```

**Automatic scaling:**
- 4 EFAs × 12 GB/s = ~48 GB/s aggregate
- Limited by PCIe, not network

### Chunk Size Tuning

```bash
# NCCL chunk size (data per channel per step)
NCCL_CHUNK_SIZE=131072  # 128 KB (default)

# Larger chunks:
NCCL_CHUNK_SIZE=262144  # 256 KB
# + Better bandwidth (fewer protocol rounds)
# - Higher latency (larger units)

# Smaller chunks:
NCCL_CHUNK_SIZE=65536   # 64 KB
# + Lower latency (smaller units)
# - Lower bandwidth (more overhead)
```

## CPU Optimizations

### Affinity and NUMA

NCCL has no `NCCL_PROXY_CPU_AFFINITY` variable (verified against NCCL `src/`: the only
`PROXY`-prefixed params are `NCCL_PROXY_APPEND_BATCH_SIZE` and `NCCL_PROXY_DUMP_SIGNAL`).
Placement is a launcher/process concern; NCCL threads inherit the process mask.

```bash
# aws-ofi-nccl guidance: give every process a mask spanning ALL CPUs
mpirun --bind-to none ./nccl_app      # or: srun --cpu-bind=none
NCCL_IGNORE_CPU_AFFINITY=0            # default 0 = honor the inherited mask

# NUMA-aware allocation
numactl --cpunodebind=0 --membind=0 ./nccl_app

# Avoid cross-NUMA traffic:
# - Pin app to NUMA node with GPU
# - Pin proxy to same NUMA node
# - Bind memory to same NUMA node
# - Ensure NIC on same NUMA node if possible
```

**Check NUMA topology:**
```bash
lscpu | grep NUMA
numactl --hardware
lstopo  # requires hwloc
```

### CPU Frequency Scaling

```bash
# Disable CPU frequency scaling for consistent latency
sudo cpupower frequency-set -g performance

# Or per-CPU:
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance | sudo tee $cpu
done
```

### Process Priority

```bash
# Elevate priority (requires root)
nice -n -20 ./nccl_app

# Real-time scheduling (use carefully!)
chrt -f 99 ./nccl_app

# NOTE: there is no NCCL_PROXY_SCHED_PRIORITY variable. Priority is set on the
# process (chrt/nice above); NCCL's proxy threads inherit it.
```

## Libfabric-Specific Optimizations

### Progress Model

EFA requires manual progress:

```bash
FI_PROGRESS_MANUAL=1  # Default for EFA
```

**Implication**: Must poll frequently

```c
// Ensure frequent polling in proxy threads
while (1) {
  fi_cq_read(cq, ...);  // Drive progress
  // Do this often (microsecond frequency)
}
```

### Endpoint Scalability

Reuse endpoints when possible:

```c
// Bad: Create new endpoint for each transfer
for (int i = 0; i < 1000; i++) {
  fi_endpoint(domain, info, &ep, NULL);
  fi_send(ep, ...);
  fi_close(&ep->fid);  // Expensive!
}

// Good: Reuse endpoints
fi_endpoint(domain, info, &ep, NULL);
for (int i = 0; i < 1000; i++) {
  fi_send(ep, ...);
}
fi_close(&ep->fid);
```

### Resource Limits

```bash
# Increase libfabric resource limits
FI_EFA_TX_SIZE=512      # Larger TX queue (default: 0 => provider chooses)
FI_EFA_RX_SIZE=512      # Larger RX queue (default: 0 => provider chooses)
```

> **Correction.** `FI_EFA_TX_SIZE` / `FI_EFA_RX_SIZE` default to **0**, meaning
> "let the provider pick" (`efa_env.c`: `.tx_size = 0`, `.rx_size = 0`), not
> 256. Also, **`FI_EFA_TX_IOV_LIMIT` is deprecated and now fatal**: the EFA
> provider aborts at startup if it is set (`efa_env.c`,
> `abort_deprecated_env_vars`), along with `FI_EFA_MTU_SIZE` and
> `FI_EFA_RX_IOV_LIMIT`. Do not set these.

## EFA-Specific Optimizations

### GPUDirect Configuration

```bash
# Ensure GPUDirect is enabled
FI_EFA_USE_DEVICE_RDMA=1  # Default: enabled

# Verify dmabuf support (kernel 5.12+)
uname -r  # Check kernel version

# Check IOMMU
dmesg | grep -i iommu
# Should see IOMMU enabled
```

### SRD Tuning

EFA's Scalable Reliable Datagram protocol:

```bash
# Queue depths affect outstanding operations.
# Default is 0 for both, which lets the provider choose the depth.
FI_EFA_TX_SIZE=0
FI_EFA_RX_SIZE=0

# Larger queues:
# + More pipelining
# + Higher bandwidth
# - More memory
# - Higher latency variance
```

### Disable Intra-Node SHM

```bash
# Let NCCL handle intra-node (NVLink/SHM)
FI_EFA_ENABLE_SHM_TRANSFER=0

# Why:
# - Avoid double-buffering
# - NCCL's intra-node is better optimized
# - Simpler data path
```

> **Default note.** `FI_EFA_ENABLE_SHM_TRANSFER` **defaults to 1 (enabled)** in
> the EFA provider (`efa_env.c`: `.enable_shm_transfer = 1`). Setting it to `0`
> above is an explicit opt-out so NCCL owns intra-node transport; it is not the
> default.

## NCCL-Specific Optimizations

### Algorithm Selection

```bash
# Force specific algorithm
NCCL_ALGO=Ring   # or Tree

# For large messages: Ring
# For small messages: Tree
# For latency: Tree
# For bandwidth: Ring
```

### Buffsize

```bash
# Channel buffer size
NCCL_BUFFSIZE=4194304   # 4 MB (default)

# Larger:
NCCL_BUFFSIZE=8388608   # 8 MB
# + More pipelining
# - More memory

# Smaller:
NCCL_BUFFSIZE=2097152   # 2 MB
# - Less pipelining
# + Less memory
```

### Tuning for Workload

```bash
# Deep learning training (large models)
NCCL_ALGO=Ring
NCCL_PROTO=Simple
NCCL_NCHANNELS=8
NCCL_CHUNK_SIZE=262144  # 256 KB

# Deep learning inference (small batches)
NCCL_ALGO=Tree
NCCL_PROTO=LL128
NCCL_NCHANNELS=4
NCCL_CHUNK_SIZE=65536   # 64 KB

# Large-scale training (many nodes)
NCCL_ALGO=Ring
NCCL_PROTO=Simple
NCCL_NCHANNELS=16
NCCL_CROSS_NIC=2        # Multi-rail
```

## Measuring Performance

### Latency Measurement

```c
#include <time.h>

struct timespec start, end;

clock_gettime(CLOCK_MONOTONIC, &start);

// Operation
ncclAllReduce(sendbuf, recvbuf, count, ...);
cudaStreamSynchronize(stream);

clock_gettime(CLOCK_MONOTONIC, &end);

double latency = (end.tv_sec - start.tv_sec) * 1e6 +
                 (end.tv_nsec - start.tv_nsec) / 1e3;
printf("Latency: %.2f us\n", latency);
```

### Bandwidth Measurement

```c
size_t bytes = count * sizeof(float);
double bandwidth = bytes / (latency / 1e6) / 1e9;
printf("Bandwidth: %.2f GB/s\n", bandwidth);

// Bus bandwidth (accounts for algorithm overhead)
// For AllReduce Ring: 2 * (N-1) / N
double bus_bw = bandwidth * 2 * (nranks - 1) / nranks;
printf("Bus Bandwidth: %.2f GB/s\n", bus_bw);
```

### NCCL Tests

```bash
# Clone NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make

# Run all-reduce benchmark
./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8

# Output shows:
# - Size (bytes)
# - Latency (us)
# - Algorithm bandwidth (GB/s)
# - Bus bandwidth (GB/s)
```

### Profiling

```bash
# NCCL profiling
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=COLL,INIT,ENV

# NCCL graph dump
NCCL_GRAPH_DUMP_FILE=nccl_graph.txt

# Libfabric profiling
FI_LOG_LEVEL=debug
FI_LOG_PROV=efa

# System profiling
perf record -g ./nccl_app
perf report

# GPU profiling
nsys profile --trace=cuda,nvtx ./nccl_app
```

## Optimization Checklist

### Baseline Configuration

```bash
# Memory registration
FI_MR_CACHE_MONITOR=memhooks
FI_MR_CACHE_MAX_SIZE=unlimited

# EFA settings
FI_EFA_ENABLE_SHM_TRANSFER=0
FI_EFA_USE_DEVICE_RDMA=1
FI_EFA_TX_SIZE=256
FI_EFA_RX_SIZE=256

# NCCL settings
NCCL_DEBUG=WARN
NCCL_IB_DISABLE=0
NCCL_SOCKET_IFNAME=^docker,lo
```

### Advanced Tuning

```bash
# Multi-rail
NCCL_CROSS_NIC=2

# CPU affinity: no NCCL variable; use the launcher (--bind-to none /
# --cpu-bind=none) or numactl/taskset on the process
NCCL_IGNORE_CPU_AFFINITY=0

# Algorithm override (if needed)
NCCL_ALGO=Ring,Tree

# Protocol override (if needed)
NCCL_PROTO=LL128,Simple

# Larger queues for high bandwidth
FI_EFA_TX_SIZE=512
FI_EFA_RX_SIZE=512
```

## Common Performance Issues

### Low Bandwidth

**Symptoms**: < 10 GB/s per EFA

**Possible Causes:**
1. Memory registration cache misses
   - Solution: Enable cache, use memhooks
2. Too few channels
   - Solution: Increase NCCL_NCHANNELS
3. Wrong algorithm
   - Solution: Use Ring for large messages
4. CPU bottleneck
   - Solution: Pin threads, increase priority

### High Latency

**Symptoms**: > 50 μs for small messages

**Possible Causes:**
1. Wrong protocol
   - Solution: Use LL for small messages
2. CPU scheduling delays
   - Solution: Real-time priority, CPU affinity
3. NUMA cross-traffic
   - Solution: Pin to local NUMA node
4. Infrequent polling
   - Solution: Ensure proxy threads poll continuously

### Instability

**Symptoms**: Crashes, hangs, data corruption

**Possible Causes:**
1. Stale MR cache entries
   - Solution: Enable memhooks/userfaultfd
2. Resource exhaustion
   - Solution: Increase queue sizes, limits
3. IOMMU issues
   - Solution: Check IOMMU config, dmesg errors

## Summary

**Key Optimization Areas:**

1. **Memory Registration**
   - Enable multi-level caching
   - Pre-register hot buffers
   - Use memhooks for cache coherence

2. **Completion Handling**
   - Batch operations
   - Adaptive polling
   - Selective completion

3. **Protocol Selection**
   - Eager is OFF by default (`OFI_NCCL_EAGER_MAX_SIZE=-1`); RDMA write is used
     even for tiny messages unless you set a positive threshold
   - LL for small (< 32 KB)
   - LL128 for medium (32 KB - 2 MB)
   - Simple for large (> 2 MB)

4. **Parallelism**
   - Tune channel count (4-8 typical)
   - Utilize multi-rail (all EFAs)
   - Optimize chunk size

5. **CPU Optimization**
   - NUMA-aware placement
   - CPU affinity for threads
   - Disable frequency scaling
   - Elevate priority

6. **EFA-Specific**
   - GPUDirect enabled
   - Disable SHM (let NCCL handle it)
   - Tune queue sizes
   - Monitor device stats

**Performance Targets:**
- **Latency**: < 15 μs (small messages)
- **Bandwidth**: > 11 GB/s per 100G EFA
- **Multi-rail**: Linear scaling (4 EFAs → 40+ GB/s)
- **GPU-to-GPU**: Zero-copy, near line-rate

**Monitoring:**
- MR cache hit rate: > 99%
- CQ poll rate: Match operation rate
- CPU usage: High when active, low when idle
- Network utilization: > 90% for large transfers

All optimizations in this document work together to minimize software overhead and maximize the performance of NCCL over AWS EFA.
