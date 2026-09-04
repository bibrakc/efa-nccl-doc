# NCCL Message & Chunk Breakdown: Complete Analysis

## Overview

This document provides ground-truth analysis of how NCCL breaks a collective down
into wire messages — from application-level message size, through channels, algorithm
steps, chunks and slices, down to individual libfabric `fi_send()` calls and EFA
packets. It covers all three protocols (**Simple**, **LL128**, **LL**), the chunk-size
calculation chain, and the pipeline-depth reasoning behind `NCCL_STEPS`.

Information was originally verified from **NCCL 2.28.9** source and re-checked against
**NCCL v2.31.2-1**; the message-breakdown primitives cited here (`NCCL_STEPS = 8`, the
4 MiB default buffer, `calcCollChunking()`, per-collective chunk/slice steps) are
unchanged in 2.31. In current NCCL the enqueue/tuning code was reorganized: the file is
`src/enqueue/enqueue.cc` (was `src/enqueue.cc` pre-2.31), the per-collective chunk/slice
step assignment moved to `src/enqueue/task_prep/task_posttuning.cc`, and the algorithm
cost models moved to `src/tuning/`. The `nccl-2.28.9-1/...` paths and line numbers
retained below are historical.

> **Source lookups:** this document records mechanism, defaults, corrections and history. For current function bodies, call graphs and blast radius, query the code graph (`codegraph explore <symbol>`) rather than trusting a pasted copy here.

## Source Code References

**Key Files:**
- `src/include/collectives.h` — SLICESTEPS and CHUNKSTEPS definitions
- `src/include/device.h` — `NCCL_STEPS`, `MAXCHANNELS`, `NCCL_MAX_TREE_ARITY`
- `src/init.cc` — `DEFAULT_BUFFSIZE`, per-protocol buffer defaults
- `src/enqueue/enqueue.cc` — `calcCollChunking()`, message planning (was `src/enqueue.cc` pre-2.31)
- `src/enqueue/task_prep/task_posttuning.cc` — per-collective `chunkSteps`/`sliceSteps` assignment
- `src/transport/net.cc` — network transport, `ncclNet->isend()` calls (~line 1437 in 2.31)
- `src/device/*.h` — GPU kernel implementations (`prims_simple.h`, `prims_ll128.h`)
- `src/proxy.cc` — proxy thread that calls the network layer

> Source: `src/enqueue/enqueue.cc` — `calcCollChunking()`;
> `src/transport/net.cc` — `sendProxyProgress()`. Use `codegraph explore calcCollChunking`
> for the current body — it was materially reorganized after 2.28.9.

## Core Constants (VALUES — the numbers are the fact)

These are the fixed primitives NCCL builds every breakdown on. The code graph indexes
the symbols but not their values, so they are recorded here.

| Constant | Value | Where |
|----------|-------|-------|
| `NCCL_STEPS` | `8` (pipeline depth) | `src/include/device.h` line 26 |
| `MAXCHANNELS` | `64` | `src/include/device.h` line 91 |
| `NCCL_MAX_TREE_ARITY` | `3` | `src/include/device.h` line 193 |
| `DEFAULT_BUFFSIZE` | `1 << 22` (4 MiB) | `src/init.cc` line 863 |

Per-collective chunk/slice steps (`src/include/collectives.h`), each expressed relative
to `NCCL_STEPS`:

```c
// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS     (NCCL_STEPS / 4)   // = 2
#define ALLREDUCE_CHUNKSTEPS     (NCCL_STEPS / 2)   // = 4
#define ALLGATHER_SLICESTEPS     (NCCL_STEPS / 4)   // = 2
#define ALLGATHER_CHUNKSTEPS     (NCCL_STEPS / 2)   // = 4
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS / 4)   // = 2
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS / 2)   // = 4
#define BROADCAST_SLICESTEPS     1
#define BROADCAST_CHUNKSTEPS     1
#define ALLTOALL_SLICESTEPS      1
#define ALLTOALL_CHUNKSTEPS      1
#define REDUCE_SLICESTEPS        1
#define REDUCE_CHUNKSTEPS        1
```

The `SlicesPerChunk = CHUNKSTEPS / SLICESTEPS` derivation gives 2 for the ring
collectives (AllReduce/AllGather/ReduceScatter) and 1 for the tree/P2P ones.

## Message Breakdown Hierarchy

```
Application Message (e.g., 64MB)
         ↓
    [NCCL Planning]
         ↓
    Channels (e.g., 8 channels)
         ↓
    Algorithm Steps (Ring: 2×(N-1), Tree: 2×log₂(N))
         ↓
    Chunks (chunkSize = stepSize × chunkSteps)
         ↓
    Slices (sliceSize = stepSize × sliceSteps)
         ↓
    Protocol Subdivision (Simple: none, LL128: 128B, LL: 8B)
         ↓
    ncclNet->isend() calls
         ↓
    fi_send() calls (libfabric)
         ↓
    Network Packets (EFA MTU: 8KB)
```

## Calculation Formulas

### Step 1: Channel Division

```
bytes_per_channel = total_message_size / num_channels
```

### Step 2: Algorithm Steps

**Ring** (AllReduce, AllGather, ReduceScatter):
```
reduce_scatter_steps = num_ranks - 1
allgather_steps      = num_ranks - 1
total_steps          = 2 × (num_ranks - 1)
```

**Tree** (Broadcast, Reduce, some AllReduce):
```
up_steps    = log₂(num_ranks)
down_steps  = log₂(num_ranks)
total_steps = 2 × log₂(num_ranks)
```

### Step 3: Chunk and Slice Sizes

The core arithmetic `calcCollChunking()` performs (FORMULA — derived sizing, not a
verbatim paste; the current function body differs and moved fields around):

```c
stepSize   = comm->buffSizes[protocol] / NCCL_STEPS;
chunkSteps = (protocol == NCCL_PROTO_SIMPLE && algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
sliceSteps = (protocol == NCCL_PROTO_SIMPLE && algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
chunkSize  = stepSize * chunkSteps;
if (protocol == NCCL_PROTO_LL)    chunkSize /= 2;
if (protocol == NCCL_PROTO_LL128) chunkSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
```

Worked out for Ring + Simple:
```
chunkSteps = ALLREDUCE_CHUNKSTEPS   // 4 for AllReduce
sliceSteps = ALLREDUCE_SLICESTEPS   // 2 for AllReduce
chunkSize  = stepSize × chunkSteps
sliceSize  = stepSize × sliceSteps

// LL:    chunkSize = (stepSize × chunkSteps) / 2
// LL128: chunkSize = ((stepSize × chunkSteps) / NCCL_LL128_LINEELEMS) × NCCL_LL128_DATAELEMS
```

### Step 4: Network Sends

One `isend()` call per slice. `sendProxyProgress()` computes the slot and slice size
from the connection FIFO (IDIOM — how the buffer slot is selected):

```c
int stepSize = resources->buffSizes[p] / NCCL_STEPS;
int buffSlot = (sub->base + sub->transmitted) % NCCL_STEPS;
int size     = connFifo[buffSlot].size;               // slice size
char* buff   = shared ? localBuff + connFifo[buffSlot].offset
                       : localBuff + buffSlot * stepSize;
// → proxyState->ncclNet->isend(netSendComm, buff, size, tpRank, mhandle, phandle, requests+buffSlot)
```

> Source: `src/transport/net.cc` — `sendProxyProgress()`; the `isend()` call is around
> line 1437 in 2.31 (was line 1322 in 2.28.9). Use `codegraph explore sendProxyProgress`.

---

## AllReduce Breakdown

### Configuration

```
ALLREDUCE_SLICESTEPS = 2
ALLREDUCE_CHUNKSTEPS = 4
SlicesPerChunk = CHUNKSTEPS / SLICESTEPS = 2
```

ProtoSimple kernel-side chunk decomposition (FORMULA, from `src/device/all_reduce.h`):
each channel walks its element range in strides of `loopCount`, where

```c
loopCount = nranks * chunkCount;                       // src/device/all_reduce.h line 23
// tail handling: if (remCount < loopCount) chunkCount = alignUp(divUp(remCount, nranks), 16/sizeof(T));
```

so one loop iteration moves `nranks` chunks (one per rank) around the ring; `chunkCount`
is the per-rank chunk element count derived from the chunk-size chain above.

> Source: `src/device/all_reduce.h` — `runRing()` / NVLS variants. Use
> `codegraph explore all_reduce` for the current kernel body.

### Example: 64MB AllReduce, 8 GPUs, 8 Channels, Ring Algorithm

Channel division:
```
Total Message: 64MB   Channels: 8   →  8MB per channel
```

Ring algorithm steps:
```
Ranks: 8   Reduce-Scatter: 8-1 = 7   AllGather: 8-1 = 7   Total: 14
```

Chunk and slice calculation (buffSize = 4MB default):
```
stepSize  = 4MB / 8 = 512KB
chunkSteps = 4        sliceSteps = 2
chunkSize = 512KB × 4 = 2MB
sliceSize = 512KB × 2 = 1MB
```

Data per step = 8MB / 8 ranks = 1MB — which equals the **sliceSize**. In the Ring
algorithm the per-step transfer is one slice, not one chunk.

Protocol subdivision of the 1MB slice:
- **Simple** — no subdivision → 1 `isend(1MB)`
- **LL128** — 128-byte units → 8,192 `isend(128B)`
- **LL** — 8-byte units → 131,072 `isend(8B)`

### Complete AllReduce Breakdown (Simple Protocol)

```
64MB AllReduce, 8 GPUs, 8 Channels, Ring Algorithm

├─ Channel 0: 8MB
│  ├─ Reduce-Scatter Phase (7 steps)
│  │  └─ each step: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│  └─ AllGather Phase (7 steps)
│     └─ each step: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│
├─ Channel 1-7: same pattern, 14 steps each
│
Total:
  - isend() calls:   8 channels × 14 steps = 112
  - fi_tsend() calls: 112
  - Bytes per call:  1MB
  - Network packets: 112 × 128 = 14,336
```

---

## AllGather Breakdown

```
ALLGATHER_SLICESTEPS = 2   ALLGATHER_CHUNKSTEPS = 4   SlicesPerChunk = 2
```

AllGather has **one phase** (no reduce-scatter): each GPU forwards its chunk around the
ring. Same channel/slice sizing as AllReduce, but 7 steps (N-1) instead of 14.

```
64MB AllGather, 8 GPUs, 8 Channels

├─ Channel 0: 8MB total (each GPU contributes 1MB)
│  └─ Steps 0-6: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB)
├─ Channel 1-7: same pattern, 7 steps each
│
Total:
  - isend() calls:   8 × 7 = 56
  - Bytes per call:  1MB
  - Network packets: 56 × 128 = 7,168
```

---

## ReduceScatter Breakdown

```
REDUCESCATTER_SLICESTEPS = 2   REDUCESCATTER_CHUNKSTEPS = 4   SlicesPerChunk = 2
```

ReduceScatter is only the reduce-scatter phase of AllReduce; each GPU ends with 1/N of
the reduced data. 7 steps, identical send counts to AllGather:

```
Total:
  - isend() calls:   8 × 7 = 56
  - Bytes per call:  1MB
  - Network packets: 56 × 128 = 7,168
```

---

## Broadcast Breakdown

```
BROADCAST_SLICESTEPS = 1   BROADCAST_CHUNKSTEPS = 1   SlicesPerChunk = 1
```

Root GPU sends to all others, typically via a binary Tree (O(log N) latency). No
chunk/slice subdivision, but NCCL still pipelines in `stepSize` units:

```
Channels: 8    Bytes per Channel: 8MB    Tree steps: log₂(8) = 3
stepSize = 4MB / 8 = 512KB
8MB / 512KB = 16 pipeline chunks per tree step
```

```
64MB Broadcast, 8 GPUs, 8 Channels, Tree Algorithm

├─ Channel 0: 8MB
│  ├─ Tree Step 0 (Root → child):  16 pipeline chunks of 512KB
│  ├─ Tree Step 1 (2 nodes send):  16 pipeline chunks each
│  └─ Tree Step 2 (4 nodes send):  16 pipeline chunks each
├─ Channel 1-7: same pattern
│
Total (approx, depends on tree structure):
  - isend() calls:   8 × 3 tree steps × 16 = 384
  - Bytes per call:  512KB
  - Network packets: 384 × 64 = 24,576
```

---

## AllToAll Breakdown

```
ALLTOALL_SLICESTEPS = 1   ALLTOALL_CHUNKSTEPS = 1   SlicesPerChunk = 1
```

Each GPU sends unique data to every other GPU (peer-to-peer, not ring/tree). Each GPU
sends and receives (N-1) messages.

```
Channels: 8   Bytes per Channel: 8MB   Peers: 7 (N-1)   ~1.14MB per peer per channel

├─ Channel 0: 8MB → 7 peers × ~1.14MB each → isend(~1.14MB)
├─ Channel 1-7: same pattern
│
Total:
  - isend() calls:   8 × 7 = 56
  - Bytes per call:  ~1.14MB
  - Network packets: 56 × 143 = 8,008
```

---

## Reduce Breakdown

```
REDUCE_SLICESTEPS = 1   REDUCE_CHUNKSTEPS = 1   SlicesPerChunk = 1
```

All GPUs send toward the root via a Tree; only the root receives the final result.
Mirror of Broadcast:

```
Total:
  - isend() calls:   8 × 3 tree steps × 16 pipeline chunks = 384
  - Bytes per call:  512KB
```

---

## Worked Protocol Examples (LL128 and LL)

The AllReduce examples above use Simple. The following two examples trace smaller
messages through LL128 and LL to show how protocol subdivision multiplies the send
count while the chunk-size calculation stays identical.

### Example: 512KB Message, LL128 Protocol

Channel and algorithm division:
```
Message: 512KB   Channels: 8   →  64KB per channel
Ranks: 8   Chunk per step: 64KB / 8 = 8KB   Steps: 14 (Ring)
```

LL128 subdivides each chunk into 128-byte transfers (**120 bytes data + 8 bytes flag**;
flag-based synchronization, receiver polls the flag):
```
8KB chunk = 8,192 bytes    transfer = 128 bytes    → 64 transfers per step
```

Idiom — the OFI send loop for one 8KB chunk:
```c
for (int i = 0; i < 8192; i += 128) {          // 120B data + 8B flag per unit
  fi_tsend(comm->ep, data + i, 128, desc, comm->remote_addr, tag, req);
}
```

```
Total (512KB, LL128):
  - fi_tsend() calls: 8 channels × 14 steps × 64 transfers = 7,168
  - Bytes per call:   128 bytes
  - Network packets:  ~7,168 (each fits one packet)
```

The LL128 wire-word geometry (FORMULA, from `src/device/prims_ll128.h`):
```c
WireWordPerSlice = WARP_SIZE * NCCL_LL128_SHMEM_ELEMS_PER_THREAD;
DataEltPerSlice  = (WireWordPerSlice - WireWordPerSlice/NCCL_LL128_LINEELEMS)
                   * (sizeof(uint64_t)/sizeof(T));
```

### Example: 16KB Message, LL Protocol

```
Message: 16KB   Channels: 4 (fewer for small messages)   →  4KB per channel
Algorithm: Tree   Steps: 2 × log₂(8) = 6   Chunk per step: 4KB (tree sends full channel data)
```

LL subdivides into 8-byte transfers (**4 bytes flag + 4 bytes data**; lowest latency,
no ACK, GPU polls the flag directly):
```
4KB chunk = 4,096 bytes    transfer = 8 bytes    → 512 transfers per step
```

```c
for (int i = 0; i < 4096; i += 8) {            // 4B flag + 4B data per unit
  fi_tsend(comm->ep, data + i, 8, desc, comm->remote_addr, tag, req);
}
```

```
Total (16KB, LL):
  - fi_tsend() calls: 4 channels × 6 steps × 512 transfers = 12,288
  - Bytes per call:   8 bytes
  - Network packets:  ~12,288
```

---

## EFA MTU Packetization

EFA MTU is **8KB**. A single `fi_tsend()` of size S becomes `⌈S / 8KB⌉` packets on the
wire:

```
1MB fi_tsend()
  → EFA driver creates a WQE
  → hardware DMAs 1MB from memory
  → splits into ~128 network packets (8KB MTU)
  → packets sprayed across SRD paths, reassembled at receiver
  → completion event generated
```

Representative packet counts: 1MB → ~128 packets; 512KB → ~64; ~1.14MB → ~143; a 128B
or 8B transfer → 1 packet each.

---

## Pipelining and Pipeline Depth

Slicing exists to overlap GPU compute with network transfer. `NCCL_STEPS = 8` bounds how
many operations can be in flight per channel at once.

```
One 2MB chunk = 2 slices of 1MB (AllReduce, SlicesPerChunk = 2):

Time ─────────────────────────────────────────────>
Slice 0 (1MB):  [GPU Prepare][Network Send]
Slice 1 (1MB):        [GPU Prepare][Network Send]

Overlapped execution reduces wall-clock time; depth is capped at NCCL_STEPS = 8.
```

---

## Protocol Comparison

| Protocol | Payload | Best For | Latency | Bandwidth | Subdivision |
|----------|---------|----------|---------|-----------|-------------|
| **Simple** | full chunk | >2MB | Moderate | Highest | None |
| **LL128** | 128 bytes (120B data + 8B flag) | 32KB–2MB | Low | Good | 128B units |
| **LL** | 8 bytes (4B flag + 4B data) | <32KB | Lowest | Poor | 8B units |

### Subdivision & Overhead of a 1MB slice

| Protocol | Transfer Size | Transfers per 1MB | isend()/fi_tsend() | Packets | Efficiency |
|----------|---------------|-------------------|--------------------|---------|------------|
| Simple | 1MB | 1 | 1 | ~128 | ~99% (1MB / 1.001MB) |
| LL128 | 128B | 8,192 | 8,192 | ~8,192 | ~94% (1MB / 1.064MB) |
| LL | 8B | 131,072 | 131,072 | ~131,072 | ~67% (1MB / 1.5MB) |

### Protocol selection (message-size thresholds)

```
Message Size          Protocol Selected
─────────────────────────────────────────
< 8 KB                LL (Low Latency)
8 KB – 32 KB          LL or LL128
32 KB – 2 MB          LL128
> 2 MB                Simple
```

### Latency / bandwidth / CPU cost (illustrative)

```
8-byte message           64MB message                 64MB message
Protocol  Latency(μs)    Protocol  BW        Eff       Protocol  fi_tsend() calls
LL        ~25-30         Simple    ~95 Gbps  ~99%      Simple    112
LL128     ~30-40         LL128     ~85 Gbps  ~94%      LL128     917,504
Simple    ~40-60         LL        ~30 Gbps  ~67%      LL        14,680,064
```

---

## Summary Tables

### Collective Operation Characteristics

| Collective | SliceSteps | ChunkSteps | Algorithm | Phases | Steps (N=8) |
|------------|------------|------------|-----------|--------|-------------|
| AllReduce | 2 | 4 | Ring | 2 | 14 |
| AllGather | 2 | 4 | Ring | 1 | 7 |
| ReduceScatter | 2 | 4 | Ring | 1 | 7 |
| Broadcast | 1 | 1 | Tree | 1 | 3 |
| Reduce | 1 | 1 | Tree | 1 | 3 |
| AllToAll | 1 | 1 | P2P | 1 | 7 |

### Network Calls for 64MB Message (8 GPUs, 8 Channels, Simple Protocol)

| Collective | isend() Calls | Bytes/Call | fi_tsend() Calls | Packets (8KB MTU) |
|------------|---------------|------------|------------------|-------------------|
| AllReduce | 112 | 1MB | 112 | 14,336 |
| AllGather | 56 | 1MB | 56 | 7,168 |
| ReduceScatter | 56 | 1MB | 56 | 7,168 |
| Broadcast | ~384 | 512KB | ~384 | ~24,576 |
| Reduce | ~384 | 512KB | ~384 | ~24,576 |
| AllToAll | 56 | ~1.14MB | 56 | ~8,008 |

### Same 64MB AllReduce across protocols (chunk/step = 1MB slice)

| Protocol | fi_tsend() Calls | Bytes/Send | Total Packets |
|----------|------------------|------------|---------------|
| Simple | 112 | 1MB | ~14,336 |
| LL128 | 917,504 | 128B | ~917,504 |
| LL | 14,680,064 | 8B | ~14,680,064 |

The protocol choice dramatically changes the number of send operations while the
underlying chunk-size calculation stays the same.

---

## Formula Reference (all protocols)

```
bytes_per_channel   = total_message_size / nChannels
chunk_per_step      = bytes_per_channel / num_ranks        (Ring)
num_steps           = 2 × (num_ranks - 1)                  (Ring)
                    = 2 × log₂(num_ranks)                  (Tree)

Simple:  sends_per_step = 1                     bytes_per_send = chunk_per_step
LL128:   sends_per_step = chunk_per_step / 128  bytes_per_send = 128  (120B data + 8B flag)
LL:      sends_per_step = chunk_per_step / 8    bytes_per_send = 8    (4B flag + 4B data)

total_sends = nChannels × num_steps × sends_per_step
```

---

## Code Flow Summary

```
1. Application: ncclAllReduce(64MB)
   ↓
2. NCCL Planning (src/enqueue/enqueue.cc): calcCollChunking() → channels, algorithm,
   protocol, chunkSize, sliceSize
   ↓
3. GPU Kernel (src/device/*.h): copy data to channel buffers, signal proxy
   ↓
4. Proxy Thread (src/proxy.cc): poll for GPU work, call ncclNet->isend() per slice
   ↓
5. Network Transport (src/transport/net.cc, sendProxyProgress ~line 1437):
   ncclNet->isend(netSendComm, buff, size, ...)
   ↓
6. OFI Plugin (aws-ofi-nccl): fi_tsend(ep, data, size, ...)
   ↓
7. Libfabric EFA Provider: post to send queue, ring doorbell
   ↓
8. EFA Hardware: DMA, packetize (8KB MTU), send
```

---

## Key Insights

1. **Slice is the unit of network transfer** — each `isend()` sends one slice.
2. **Chunk contains multiple slices** for pipelining (AllReduce: 2 slices/chunk).
3. **Protocol determines subdivision** — Simple (none), LL128 (128B), LL (8B).
4. **Algorithm determines steps** — Ring 2×(N-1), Tree 2×log₂(N).
5. **Channels enable parallelism** — independent data paths, up to `MAXCHANNELS = 64`.
6. **Pipeline depth `NCCL_STEPS = 8`** — bounds GPU/network overlap per channel.
7. **Chunk size = message ÷ channels ÷ ranks**; protocol multiplies the send count only.

---

## Environment Variables

```bash
# Channels
NCCL_MIN_NCHANNELS=8        # Pin the channel count: set both clamps
NCCL_MAX_NCHANNELS=8        # to the same value
NCCL_MIN_NCHANNELS=4        # Minimum channels
NCCL_MAX_NCHANNELS=16       # Maximum channels

# Buffer size (affects stepSize)
NCCL_BUFFSIZE=4194304       # 4MB default
NCCL_BUFFSIZE=8388608       # 8MB (larger chunks)

# Protocol / algorithm
NCCL_PROTO=Simple           # Simple, LL128, or LL
NCCL_ALGO=Ring              # Ring, Tree, etc.

# Debug
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,NET
```

---

## Related Documentation

- [nccl-core.md](nccl-core.md) - NCCL core concepts
- [nccl-collectives.md](nccl-collectives.md) - Collective operations overview
- [nccl-buffsize-explained.md](nccl-buffsize-explained.md) - Buffer sizing deep dive
- [algorithms/ring-algorithm.md](algorithms/ring-algorithm.md) - Ring algorithm details
- [algorithms/tree-algorithm.md](algorithms/tree-algorithm.md) - Tree algorithm details
- [ofi-plugin.md](ofi-plugin.md) - OFI plugin implementation
- [ofi-plugin-protocols.md](ofi-plugin-protocols.md) - Send/recv protocols

---

**Last Updated**: 2026-01-29

**Source**: NCCL 2.28.9 source-code analysis, re-verified against NCCL v2.31.2-1
(chunk/slice/buffer primitives unchanged; enqueue/tuning code paths reorganized).

**License**: Same as efa-nccl-doc repository
