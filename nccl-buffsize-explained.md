# NCCL_BUFFSIZE Explained: Channel Buffers, Pipelining, and Where Reduction Happens

## Overview

This document explains the `NCCL_BUFFSIZE` environment variable, channel buffers, pipelining, and clarifies where the actual reduction computation takes place in NCCL operations.

**TL;DR:**
- `NCCL_BUFFSIZE` controls the **channel buffer size** (default: 4MB)
- Larger buffers → larger chunks → better bandwidth, more memory
- Reduction happens **on the GPU** in GPU memory, **not** in channel buffers
- Channel buffers are **staging areas** for network transfers

---

## NCCL_BUFFSIZE: The Channel Buffer

### Definition

**File**: [src/init.cc](https://github.com/NVIDIA/nccl/blob/master/src/init.cc) (NCCL 2.28.9 lines 699)

```c
#define DEFAULT_BUFFSIZE (1 << 22) /* 4MiB */
```

**Environment Variable:**
```bash
export NCCL_BUFFSIZE=4194304  # 4MB (default)
```

### What It Controls

**File**: [src/init.cc](https://github.com/NVIDIA/nccl/blob/master/src/init.cc) - `computeBuffSizes()` (NCCL 2.28.9 lines 708-713)

```c
static ncclResult_t computeBuffSizes(struct ncclComm* comm) {
  int64_t envs[NCCL_NUM_PROTOCOLS] = { 
      ncclParamLlBuffSize(), 
      ncclParamLl128BuffSize(), 
      ncclParamBuffSize()  // NCCL_BUFFSIZE
  };
  int defaults[NCCL_NUM_PROTOCOLS] = { 
      DEFAULT_LL_BUFFSIZE, 
      DEFAULT_LL128_BUFFSIZE, 
      DEFAULT_BUFFSIZE  // 4MB
  };

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
  }
  // ...
}
```

**Storage**: `comm->buffSizes[NCCL_PROTO_SIMPLE]` - stored in communicator

### Channel Buffer Architecture

Each NCCL channel has dedicated buffers for each protocol:

```
Channel 0:
  ├─ Simple Protocol Buffer: 4MB (NCCL_BUFFSIZE)
  ├─ LL128 Protocol Buffer: ~2MB
  └─ LL Protocol Buffer: ~512KB

Channel 1:
  ├─ Simple Protocol Buffer: 4MB
  ├─ LL128 Protocol Buffer: ~2MB
  └─ LL Protocol Buffer: ~512KB

... (for all channels)
```

**Total Memory per GPU:**
```
Memory = num_channels × (buffSize_Simple + buffSize_LL128 + buffSize_LL)
       = 8 × (4MB + 2MB + 0.5MB)
       = 8 × 6.5MB
       = 52MB per GPU
```

---

## How BUFFSIZE Affects Performance

### The Calculation Chain

**File**: `calcCollChunking()`, in current NCCL at [src/enqueue/enqueue.cc](https://github.com/NVIDIA/nccl/blob/master/src/enqueue/enqueue.cc) (historically `src/enqueue.cc` at NCCL 2.28.9, line ~2023)

```c
int stepSize   = comm->buffSizes[info->protocol]/NCCL_STEPS;
int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && 
                  info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && 
                  info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
int chunkSize = stepSize*chunkSteps;
```

**For Simple Protocol + Ring Algorithm:**
```
NCCL_BUFFSIZE = 4MB (default)
NCCL_STEPS = 8
chunkSteps = 4 (ALLREDUCE_CHUNKSTEPS)
sliceSteps = 2 (ALLREDUCE_SLICESTEPS)

stepSize = 4MB / 8 = 512KB
chunkSize = 512KB × 4 = 2MB
sliceSize = 512KB × 2 = 1MB
```

### Impact of Increasing BUFFSIZE

#### Example: NCCL_BUFFSIZE=8MB

```
stepSize = 8MB / 8 = 1MB
chunkSize = 1MB × 4 = 4MB
sliceSize = 1MB × 2 = 2MB
```

**Result:**
- Each network send is now **2MB** instead of 1MB
- Fewer, larger sends
- Better bandwidth efficiency
- More memory used

#### Example: NCCL_BUFFSIZE=16MB

```
stepSize = 16MB / 8 = 2MB
chunkSize = 2MB × 4 = 8MB
sliceSize = 2MB × 2 = 4MB
```

**Result:**
- Each network send is now **4MB**
- Even fewer, larger sends
- Maximum bandwidth efficiency
- Significantly more memory

### Performance Trade-offs

**Larger BUFFSIZE (e.g., 8MB or 16MB):**

✅ **Benefits:**
- **Larger slices** → fewer network operations
- **Better bandwidth** → less per-operation overhead
- **Better pipelining** → more data in flight
- **Amortized latency** → setup cost spread over more data

❌ **Costs:**
- **More GPU memory** → less available for model/data
- **Higher latency for small messages** → overkill for small transfers
- **Longer pipeline fill time** → initial latency penalty

**Smaller BUFFSIZE (e.g., 2MB):**

✅ **Benefits:**
- **Less GPU memory** → more for application
- **Lower latency for small messages** → faster startup
- **Faster pipeline fill** → quicker to first byte

❌ **Costs:**
- **Smaller slices** → more network operations
- **Lower bandwidth** → more overhead per operation
- **More CPU overhead** → more isend() calls to process

---

## Where Does Reduction Actually Happen?

### Short Answer

**Reduction happens on the GPU in GPU memory, NOT in channel buffers.**

Channel buffers are **staging areas** for network transfers, not computation areas.

### The Data Flow

#### For AllReduce (Ring Algorithm)

**File**: [src/device/all_reduce.h](https://github.com/NVIDIA/nccl/blob/master/src/device/all_reduce.h) (NCCL 2.28.9 lines 55-64)

```c
// Reduce-Scatter phase
if (tid < nthreads) {
  if (flags & RoleInput) {
    // Receive from peer, reduce with local input, send result
    prims.directRecvReduceDirectSend(offset, offset, nelem);
  } else {
    // Receive from peer, reduce with local data, copy to output
    prims.directRecvReduceCopy(offset, offset, nelem, /*postOp=*/true);
  }
}
```

**What happens in `directRecvReduceDirectSend()`:**

**File**: [src/device/prims_simple.h](https://github.com/NVIDIA/nccl/blob/master/src/device/prims_simple.h) (NCCL 2.28.9 lines 925-927)

```c
__device__ __forceinline__ void directRecvReduceDirectSend(
    intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false) {
  genericOp<1, 1, 1, 1, Input, -1>(inpIx, outIx, eltN, postOp);
}
```

This calls `genericOp()` which performs:

**File**: [src/device/prims_simple.h](https://github.com/NVIDIA/nccl/blob/master/src/device/prims_simple.h) (NCCL 2.28.9 lines 270-285)

```c
// Inside genericOp() - the actual reduction
reduceCopy<Unroll, RedOp, T,
  MultimemSrcs, Recv + Src, Recv * MaxRecv + Src,
  MultimemDsts, Send + Dst, Send * MaxSend + Dst, PreOpSrcs>
  (tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, postOp,
    Recv * fan.nrecv() + Src, ncclShmem.groups[group].srcs,
    Send * fan.nsend() + Dst, ncclShmem.groups[group].dsts,
    workSize);
```

### Where Reduction Happens: GPU Memory

**The reduction operates on:**
1. **Input**: GPU memory (user's sendbuff)
2. **Received data**: From peer GPU (via channel buffer or direct)
3. **Output**: GPU memory (user's recvbuff or intermediate)

**Channel buffers are used for:**
- **Staging received data** from network
- **Staging data to send** to network
- **NOT for reduction computation**

### Memory Layout

```
GPU 0 Memory:
┌─────────────────────────────────────────────┐
│  User Input Buffer (sendbuff)               │  ← Original data
│  [A0, B0, C0, D0, ...]                      │
└─────────────────────────────────────────────┘
         │
         │ (GPU reads directly)
         ↓
┌─────────────────────────────────────────────┐
│  GPU Registers / L1 Cache                   │  ← Reduction happens here
│  temp = A0 + A_received                     │     (GPU ALU operations)
└─────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────┐
│  Channel Buffer (for network send)          │  ← Staging area
│  [temp data ready to send]                  │     (4MB default)
└─────────────────────────────────────────────┘
         │
         ↓ (DMA to NIC)
    Network Send

Incoming from Network:
         ↓ (DMA from NIC)
┌─────────────────────────────────────────────┐
│  Channel Buffer (received data)             │  ← Staging area
│  [A_from_peer]                              │
└─────────────────────────────────────────────┘
         │
         ↓ (GPU reads for reduction)
┌─────────────────────────────────────────────┐
│  GPU Registers / L1 Cache                   │  ← Reduction happens here
│  result = local_data + A_from_peer          │
└─────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────┐
│  User Output Buffer (recvbuff)              │  ← Final result
│  [A_reduced, B_reduced, ...]                │
└─────────────────────────────────────────────┘
```

### Key Insight

**Channel buffers are I/O buffers, not compute buffers.**

The GPU kernel:
1. **Reads** from user input buffer (GPU memory)
2. **Reads** received data from channel buffer (or directly from peer GPU)
3. **Performs reduction** in GPU registers/cache
4. **Writes** result to channel buffer (for sending) or output buffer (final result)

---

## Pipelining and NCCL_STEPS

### Pipeline Depth

**File**: [src/include/device.h](https://github.com/NVIDIA/nccl/blob/master/src/include/device.h) (NCCL 2.28.9 lines 24)

```c
#define NCCL_STEPS 8
```

**The channel buffer is divided into 8 slots:**

```
Channel Buffer (4MB total):
┌──────────┬──────────┬──────────┬──────────┐
│ Slot 0   │ Slot 1   │ Slot 2   │ Slot 3   │
│ 512KB    │ 512KB    │ 512KB    │ 512KB    │
└──────────┴──────────┴──────────┴──────────┘
┌──────────┬──────────┬──────────┬──────────┐
│ Slot 4   │ Slot 5   │ Slot 6   │ Slot 7   │
│ 512KB    │ 512KB    │ 512KB    │ 512KB    │
└──────────┴──────────┴──────────┴──────────┘

stepSize = BUFFSIZE / NCCL_STEPS = 4MB / 8 = 512KB
```

### Why 8 Steps?

**Pipelining allows overlap:**

```
Time ──────────────────────────────────────────────>

Step 0: [GPU→Buf][Send    ][Complete]
Step 1:     [GPU→Buf][Send    ][Complete]
Step 2:         [GPU→Buf][Send    ][Complete]
Step 3:             [GPU→Buf][Send    ][Complete]
Step 4:                 [GPU→Buf][Send    ][Complete]
Step 5:                     [GPU→Buf][Send    ][Complete]
Step 6:                         [GPU→Buf][Send    ][Complete]
Step 7:                             [GPU→Buf][Send    ][Complete]

Up to 8 operations can be in-flight simultaneously
```

**File**: [src/transport/net.cc](https://github.com/NVIDIA/nccl/blob/master/src/transport/net.cc) (NCCL 2.28.9 lines 1269-1271)

```c
// Check if we can post more sends (pipeline depth limit)
if (sub->posted < sub->nsteps && sub->posted < sub->done + maxDepth) {
  // maxDepth = min(NCCL_STEPS, NCCL_SHARED_STEPS/args->nsubs)
  // Typically maxDepth = 8
```

---

## Chunk Size and Slice Size Relationship

### The Hierarchy

**File**: [src/include/collectives.h](https://github.com/NVIDIA/nccl/blob/master/src/include/collectives.h) (NCCL 2.28.9 lines 16-18)

```c
// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)  // = 2
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)  // = 4
```

**Calculation:**

```
BUFFSIZE = 4MB (default)
NCCL_STEPS = 8

stepSize = BUFFSIZE / NCCL_STEPS = 4MB / 8 = 512KB

For AllReduce (Ring + Simple):
  sliceSteps = 2
  chunkSteps = 4
  
  sliceSize = stepSize × sliceSteps = 512KB × 2 = 1MB
  chunkSize = stepSize × chunkSteps = 512KB × 4 = 2MB
  
  SlicesPerChunk = chunkSteps / sliceSteps = 4 / 2 = 2
```

**Relationship:**
```
1 Chunk = 2 Slices
1 Slice = 2 Steps
1 Step = 512KB (stepSize)
```

### Why This Matters

**Slice is the unit of network transfer:**

**File**: [src/transport/net.cc](https://github.com/NVIDIA/nccl/blob/master/src/transport/net.cc) (NCCL 2.28.9 lines 1322)

```c
// One isend() call per slice
NCCLCHECK(proxyState->ncclNet->isend(
    resources->netSendComm, 
    buff, 
    size,  // This is sliceSize (e.g., 1MB)
    resources->tpRank, 
    sub->sendMhandle, 
    phandle, 
    sub->requests+buffSlot
));
```

**Advancement:**

**File**: [src/transport/net.cc](https://github.com/NVIDIA/nccl/blob/master/src/transport/net.cc) (NCCL 2.28.9 lines 1327)

```c
sub->transmitted += args->sliceSteps;  // Advance by 2 steps
```

---

## Performance Impact: Concrete Examples

### Example 1: Default (BUFFSIZE=4MB)

```
64MB AllReduce, 8 GPUs, 8 Channels

stepSize = 4MB / 8 = 512KB
sliceSize = 512KB × 2 = 1MB

Per Channel: 8MB
Per Rank: 8MB / 8 = 1MB
Slices per Rank: 1MB / 1MB = 1 slice

Total isend() calls: 8 channels × 14 steps × 1 slice = 112
Bytes per isend(): 1MB
```

### Example 2: Increased (BUFFSIZE=8MB)

```
64MB AllReduce, 8 GPUs, 8 Channels

stepSize = 8MB / 8 = 1MB
sliceSize = 1MB × 2 = 2MB

Per Channel: 8MB
Per Rank: 8MB / 8 = 1MB
Slices per Rank: 1MB / 2MB = 0.5 slice (rounds to 1)

Total isend() calls: 8 channels × 14 steps × 1 slice = 112
Bytes per isend(): 1MB (limited by data per rank)
```

**Wait - no benefit here!** The data per rank (1MB) is already ≤ sliceSize.

### Example 3: Larger Message (512MB, BUFFSIZE=8MB)

```
512MB AllReduce, 8 GPUs, 8 Channels

stepSize = 8MB / 8 = 1MB
sliceSize = 1MB × 2 = 2MB

Per Channel: 64MB
Per Rank: 64MB / 8 = 8MB
Slices per Rank: 8MB / 2MB = 4 slices

Total isend() calls: 8 channels × 14 steps × 4 slices = 448
Bytes per isend(): 2MB
```

**Compare to BUFFSIZE=4MB:**
```
sliceSize = 512KB × 2 = 1MB
Slices per Rank: 8MB / 1MB = 8 slices
Total isend() calls: 8 channels × 14 steps × 8 slices = 896
Bytes per isend(): 1MB
```

**Benefit: 50% fewer isend() calls, 2× larger transfers**

---

## When to Increase BUFFSIZE

### Good Candidates

1. **Large messages** (>100MB per GPU)
   - More data to amortize over
   - Slice size becomes limiting factor

2. **High-bandwidth networks** (400 Gbps+)
   - Larger transfers saturate bandwidth better
   - Overhead of small transfers more significant

3. **Many ranks** (>16 GPUs)
   - More steps in Ring algorithm
   - More opportunities for pipelining

4. **Bandwidth-bound workloads**
   - Training large models
   - Gradient synchronization dominates

### Poor Candidates

1. **Small messages** (<10MB per GPU)
   - Slice size already larger than data
   - No benefit, just wastes memory

2. **Memory-constrained** (large models)
   - Every MB of GPU memory matters
   - Channel buffers compete with model

3. **Latency-sensitive** (small frequent collectives)
   - Larger buffers increase initial latency
   - LL/LL128 protocols don't benefit as much

---

## Recommended Settings

### For Large-Scale Training (GPT-3 style)

```bash
# Large messages, many GPUs, bandwidth-critical
export NCCL_BUFFSIZE=8388608   # 8MB
export NCCL_NCHANNELS=8
export NCCL_PROTO=Simple
```

**Memory cost**: 8 channels × 8MB = 64MB per GPU
**Benefit**: 2× larger slices, fewer network calls

### For Medium-Scale Training

```bash
# Default is usually optimal
export NCCL_BUFFSIZE=4194304   # 4MB (default)
export NCCL_NCHANNELS=8
```

**Memory cost**: 8 channels × 4MB = 32MB per GPU
**Benefit**: Balanced performance and memory

### For Memory-Constrained

```bash
# Minimize memory usage
export NCCL_BUFFSIZE=2097152   # 2MB
export NCCL_NCHANNELS=4
```

**Memory cost**: 4 channels × 2MB = 8MB per GPU
**Benefit**: More memory for model, acceptable performance

---

## Where Reduction Actually Happens: Detailed

### GPU Kernel Execution

**File**: [src/device/prims_simple.h](https://github.com/NVIDIA/nccl/blob/master/src/device/prims_simple.h) - `genericOp()` (NCCL 2.28.9 lines 184-350)

The `genericOp()` function orchestrates:

1. **Wait for data** from peer (in channel buffer)
2. **Read data** from channel buffer to GPU registers
3. **Read local data** from user input buffer
4. **Perform reduction** in GPU registers (e.g., `temp = local + received`)
5. **Write result** to channel buffer (for sending) or output buffer

**Pseudocode:**
```c
// GPU kernel thread
T local_value = sendbuff[index];           // Read from GPU memory
T received_value = channel_buffer[slot];   // Read from channel buffer
T reduced = local_value + received_value;  // Reduction in GPU registers
channel_buffer[next_slot] = reduced;       // Write to channel buffer for send
```

### Reduction Operations

**Supported operations** (performed in GPU ALU):
- `ncclSum`: `a + b`
- `ncclProd`: `a * b`
- `ncclMax`: `max(a, b)`
- `ncclMin`: `min(a, b)`
- `ncclAvg`: `(a + b) / 2`

**These are GPU arithmetic operations**, not memory operations.

---

## Memory Breakdown

### Per-GPU Memory Usage

**For 8 channels, default settings:**

```
Channel Buffers:
  - Simple protocol: 8 × 4MB = 32MB
  - LL128 protocol: 8 × 2MB = 16MB
  - LL protocol: 8 × 512KB = 4MB
  Total: 52MB

User Buffers:
  - sendbuff: Application-defined (e.g., 1GB for model gradients)
  - recvbuff: Application-defined (e.g., 1GB)
  Total: 2GB (example)

Other NCCL Structures:
  - Communicator metadata: ~10MB
  - Topology info: ~1MB
  - Connection state: ~5MB
  Total: ~16MB

Grand Total: ~2.08GB per GPU
```

### Impact of BUFFSIZE=16MB

```
Channel Buffers:
  - Simple protocol: 8 × 16MB = 128MB (was 32MB)
  - LL128 protocol: 8 × 8MB = 64MB (was 16MB)
  - LL protocol: 8 × 2MB = 16MB (was 4MB)
  Total: 208MB (was 52MB)

Additional memory: 208MB - 52MB = 156MB per GPU
```

**For 8 GPUs**: 156MB × 8 = 1.25GB total additional memory

---

## Summary

### NCCL_BUFFSIZE Controls

1. **Channel buffer size** per protocol
2. **stepSize** = BUFFSIZE / 8
3. **sliceSize** = stepSize × sliceSteps
4. **chunkSize** = stepSize × chunkSteps

### Performance Statement Explained

**"Increasing BUFFSIZE can lead to better performance at the cost of memory"**

**Why better performance:**
- Larger slices → fewer network operations
- Fewer isend() calls → less CPU overhead
- Larger transfers → better bandwidth efficiency
- More data in pipeline → better overlap

**Why more memory:**
- Each channel needs larger buffer
- Multiple channels multiply the cost
- All protocols allocate buffers
- Memory not available for application

### Where Reduction Happens

**Reduction happens on the GPU in GPU registers/cache**, operating on:
- Data from user input buffer (GPU memory)
- Data received from peers (staged in channel buffer)
- Result written to output buffer or channel buffer

**Channel buffers are staging areas for network I/O, not computation areas.**

### Tuning Guideline

```
Message Size per GPU    Recommended BUFFSIZE
────────────────────────────────────────────
< 10MB                  2MB (save memory)
10MB - 100MB            4MB (default)
100MB - 1GB             8MB (better bandwidth)
> 1GB                   16MB (maximum bandwidth)
```

**Always measure**: The optimal setting depends on your specific workload, model size, and network characteristics.

---

## Related Documentation

- [nccl-message-breakdown-complete.md](nccl-message-breakdown-complete.md) - Complete message breakdown
- [nccl-core.md](nccl-core.md) - NCCL core concepts
- [nccl-datapath.md](nccl-datapath.md) - Data flow details
- [optimizations.md](optimizations.md) - Performance tuning

---

## Code References

### Key Files
- `nccl-2.28.9-1/src/init.cc` - Buffer size initialization
- `src/enqueue/enqueue.cc` - Chunk/slice calculation (was `src/enqueue.cc` before the 2.31 reorganization)
- `nccl-2.28.9-1/src/device/prims_simple.h` - GPU primitives
- `nccl-2.28.9-1/src/transport/net.cc` - Network sends

### Key Functions
- `computeBuffSizes()` - Initialize buffer sizes from env vars
- `calcCollChunking()` - Calculate chunk and slice sizes
- `genericOp()` - GPU kernel primitive operations
- `sendProxyProgress()` - Network send loop

### Environment Variables
- `NCCL_BUFFSIZE` - Simple protocol buffer size (default: 4MB)
- `NCCL_LL128_BUFFSIZE` - LL128 protocol buffer size
- `NCCL_LL_BUFFSIZE` - LL protocol buffer size

---

**Last Updated**: 2026-01-30

**Source**: NCCL 2.28.9 source code analysis, re-verified against NCCL v2.31.2-1 (`DEFAULT_BUFFSIZE = 1<<22` = 4 MiB in `src/init.cc`; `NCCL_STEPS = 8` in `src/include/device.h`; `computeBuffSizes()` in `src/init.cc`; `calcCollChunking()` in `src/enqueue/enqueue.cc`. Buffer/chunk sizing logic unchanged in 2.31.)
