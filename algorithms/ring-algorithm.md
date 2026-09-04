# Ring AllReduce Algorithm

> ## ⚠️ DRAFT - NEEDS VERIFICATION ⚠️
>
> **This document contains theoretical derivations and performance analysis that have NOT been fully verified.**
>
> All latency formulas, bandwidth calculations, and performance claims should be independently validated before being used for system design or optimization decisions. The mathematical derivations may contain errors or oversimplifications.
>
> Please treat all content as preliminary and subject to revision.

## Overview

The **Ring algorithm** is NCCL's bandwidth-optimal algorithm for collective operations across multiple GPUs. It arranges GPUs in a logical ring topology and performs operations in a pipelined fashion, achieving near-perfect bandwidth utilization with linear latency scaling.

**Best For**: Large messages (> 1 MB) where bandwidth is more important than latency

**Key Properties**:
- Bandwidth utilization: ~100% of available link bandwidth
- Latency: O(N) - linear in number of GPUs
- Data movement: Each GPU sends and receives (N-1)/N of the total data
- Scalability: Bandwidth-optimal but latency limits scaling to hundreds of GPUs

**Source**: Ring ordering is built per channel by `ncclBuildRings()` in
[nccl/src/graph/rings.cc](https://github.com/NVIDIA/nccl/blob/master/src/graph/rings.cc)
(the ring search itself is in [src/graph/search.cc](https://github.com/NVIDIA/nccl/blob/master/src/graph/search.cc)
and [src/graph/connect.cc](https://github.com/NVIDIA/nccl/blob/master/src/graph/connect.cc));
the device-side reduce-scatter/all-gather steps run in
[src/device/all_reduce.h](https://github.com/NVIDIA/nccl/blob/master/src/device/all_reduce.h).

## Algorithm Description

### Logical Ring Topology

GPUs are arranged in a circular topology:

```
GPU 0 → GPU 1 → GPU 2 → ... → GPU (N-1) → GPU 0
```

Each GPU has:
- **Predecessor**: GPU from which it receives data
- **Successor**: GPU to which it sends data

For GPU `i`:
- Predecessor: GPU `(i - 1 + N) % N`
- Successor: GPU `(i + 1) % N`

### AllReduce in Two Phases

The Ring AllReduce consists of two phases:

1. **Reduce-Scatter**: Each GPU ends up with 1/N of the reduced data
2. **AllGather**: Each GPU collects all segments to reconstruct the full result

### Phase 1: Reduce-Scatter

**Goal**: Distribute reduction work across all GPUs

**Data Chunking**: Divide the input buffer into N chunks (one per GPU):
```
Chunk 0 | Chunk 1 | Chunk 2 | ... | Chunk (N-1)
```

**Process**:
Each GPU starts with its own chunk index and:
1. In each step, reduces the received chunk with its local chunk
2. Sends the reduced chunk to its successor
3. Receives a new chunk from its predecessor

After `(N-1)` steps, each GPU holds the fully reduced version of one chunk.

**Step-by-Step for N=4 GPUs**:

```
Initial state (each GPU has chunks 0-3):
GPU 0: [A0, B0, C0, D0]
GPU 1: [A1, B1, C1, D1]
GPU 2: [A2, B2, C2, D2]
GPU 3: [A3, B3, C3, D3]

Step 1 (send chunk i, reduce chunk (i-1+N)%N):
GPU 0: Sends D0, receives and reduces C3 → [A0, B0, C0+C3, D0]
GPU 1: Sends A1, receives and reduces D0 → [A1, B1, C1, D0+D1]
GPU 2: Sends B2, receives and reduces A1 → [A1+A2, B2, C2, D2]
GPU 3: Sends C3, receives and reduces B2 → [A3, B2+B3, C3, D3]

Step 2:
GPU 0: Sends C0+C3, receives and reduces B2+B3 → [A0, B0+B2+B3, C0+C3, D0]
GPU 1: Sends D0+D1, receives and reduces C0+C3 → [A1, B1, C0+C1+C3, D0+D1]
GPU 2: Sends A1+A2, receives and reduces D0+D1 → [A1+A2, B2, C2, D0+D1+D2]
GPU 3: Sends B2+B3, receives and reduces A1+A2 → [A1+A2+A3, B2+B3, C3, D3]

Step 3 (final reduce-scatter step):
GPU 0: [A0, B0+B1+B2+B3, C0+C3, D0]
GPU 1: [A1, B1, C0+C1+C2+C3, D0+D1]
GPU 2: [A1+A2, B2, C2, D0+D1+D2+D3]
GPU 3: [A0+A1+A2+A3, B2+B3, C3, D3]

Result after reduce-scatter:
GPU 0 holds: Chunk B (fully reduced)
GPU 1 holds: Chunk C (fully reduced)
GPU 2 holds: Chunk D (fully reduced)
GPU 3 holds: Chunk A (fully reduced)
```

### Phase 2: AllGather

**Goal**: Each GPU collects all reduced chunks

**Process**:
Each GPU forwards its reduced chunk around the ring. After `(N-1)` steps, all GPUs have all reduced chunks.

**Step-by-Step for N=4 GPUs** (continuing from reduce-scatter result):

```
Initial (after reduce-scatter):
GPU 0: [_, B_reduced, _, _]
GPU 1: [_, _, C_reduced, _]
GPU 2: [_, _, _, D_reduced]
GPU 3: [A_reduced, _, _, _]

Step 1:
GPU 0: Sends B_reduced, receives A_reduced → [A_reduced, B_reduced, _, _]
GPU 1: Sends C_reduced, receives B_reduced → [_, B_reduced, C_reduced, _]
GPU 2: Sends D_reduced, receives C_reduced → [_, _, C_reduced, D_reduced]
GPU 3: Sends A_reduced, receives D_reduced → [A_reduced, _, _, D_reduced]

Step 2:
GPU 0: Sends A_reduced, receives D_reduced → [A_reduced, B_reduced, _, D_reduced]
GPU 1: Sends B_reduced, receives A_reduced → [A_reduced, B_reduced, C_reduced, _]
GPU 2: Sends C_reduced, receives B_reduced → [_, B_reduced, C_reduced, D_reduced]
GPU 3: Sends D_reduced, receives C_reduced → [A_reduced, _, C_reduced, D_reduced]

Step 3 (final allgather step):
GPU 0: Receives C_reduced → [A_reduced, B_reduced, C_reduced, D_reduced]
GPU 1: Receives D_reduced → [A_reduced, B_reduced, C_reduced, D_reduced]
GPU 2: Receives A_reduced → [A_reduced, B_reduced, C_reduced, D_reduced]
GPU 3: Receives B_reduced → [A_reduced, B_reduced, C_reduced, D_reduced]

Final result: All GPUs have complete reduced buffer
```

## Mathematical Derivation of Latency

### Latency Model Parameters

**α-β Model** (standard communication model):
- **α (alpha)**: Latency per message (μs) - time to initiate communication
- **β (beta)**: Per-byte transmission time (μs/byte) = 1/bandwidth
- **S**: Total message size (bytes)
- **N**: Number of GPUs
- **γ (gamma)**: Reduction computation time per byte (μs/byte)

### Chunk Size

Total data is divided into N chunks:
```
Chunk size = S/N bytes
```

### Reduce-Scatter Phase Latency

**Number of Steps**: `N - 1` steps

In each step:
1. **Send chunk**: Time = `α + β * (S/N)`
2. **Receive chunk**: Overlapped with send
3. **Reduce chunk**: Time = `γ * (S/N)`

Assuming send/receive overlaps with reduction on modern GPUs:

```
Time per step ≈ max(α + β*S/N, γ*S/N)
```

For bandwidth-limited (typical case): `β*S/N >> γ*S/N`

**Total Reduce-Scatter Latency**:
```
T_reduce_scatter = (N-1) * (α + β*S/N)
                 = (N-1)*α + β*S*(N-1)/N
```

### AllGather Phase Latency

**Number of Steps**: `N - 1` steps

In each step, send and receive a chunk of size S/N:

```
Time per step = α + β*S/N
```

**Total AllGather Latency**:
```
T_allgather = (N-1) * (α + β*S/N)
            = (N-1)*α + β*S*(N-1)/N
```

### Total AllReduce Latency

**Ring AllReduce Total Latency**:

```
T_ring = T_reduce_scatter + T_allgather
       = 2(N-1)*α + 2*β*S*(N-1)/N
```

**Simplified**:
```
T_ring = 2(N-1)*α + 2*β*S*(1 - 1/N)
```

**Asymptotic for large N**:
```
T_ring ≈ 2*N*α + 2*β*S    (as N → ∞)
```

**Key Insight**: For large messages (β*S >> α), the latency is dominated by:
```
T_ring ≈ 2*β*S    (bandwidth-limited)
```

This means the algorithm achieves approximately 100% bandwidth utilization because it transfers `2*S*(N-1)/N ≈ 2*S` total bytes.

## Protocol-Specific Latency

### Simple Protocol (Full Bandwidth)

Parameters:
- α_simple ≈ 10-20 μs (typical network latency)
- β_simple = 1 / B_link (where B_link is link bandwidth)

**Latency**:
```
T_ring_simple = 2(N-1)*α_simple + 2*S/B_link * (N-1)/N
```

**Example** (N=8 GPUs, S=1 GB, B_link=400 Gbps = 50 GB/s):
```
T_ring_simple = 2*7*15μs + 2*1GB/50GB/s * 7/8
              = 210μs + 2*20ms*0.875
              = 210μs + 35ms
              ≈ 35.2 ms
```

Bandwidth achieved:
```
BW = S / (T/2) = 1GB / 17.6ms ≈ 56.8 GB/s    (> 100% - pipelining effect)
```

### LL128 Protocol (Medium Overhead)

Parameters:
- α_LL128 ≈ 7-10 μs
- β_LL128 = 1 / (B_link * efficiency), efficiency ≈ 0.6-0.7

**Latency**:
```
T_ring_LL128 = 2(N-1)*α_LL128 + 2*S/(B_link*0.65) * (N-1)/N
```

**Example** (same setup):
```
T_ring_LL128 = 2*7*8μs + 2*1GB/(50GB/s*0.65) * 7/8
             = 112μs + 2*30.77ms*0.875
             ≈ 53.8 ms
```

Bandwidth achieved:
```
BW = 1GB / 26.9ms ≈ 37.2 GB/s    (~74% of link bandwidth)
```

### LL Protocol (Low Latency, Low Bandwidth)

Parameters:
- α_LL ≈ 5-7 μs
- β_LL = 1 / (B_link * efficiency), efficiency ≈ 0.2-0.3

**Latency**:
```
T_ring_LL = 2(N-1)*α_LL + 2*S/(B_link*0.25) * (N-1)/N
```

**Example** (same setup):
```
T_ring_LL = 2*7*6μs + 2*1GB/(50GB/s*0.25) * 7/8
          = 84μs + 2*80ms*0.875
          ≈ 140 ms
```

Bandwidth achieved:
```
BW = 1GB / 70ms ≈ 14.3 GB/s    (~29% of link bandwidth)
```

**Conclusion**: LL is only used for small messages where latency dominates.

## Bandwidth Analysis

### Data Movement Per GPU

In the complete Ring AllReduce:
- Each GPU **sends**: `2 * S * (N-1)/N` bytes
- Each GPU **receives**: `2 * S * (N-1)/N` bytes

**For large N**:
```
Data sent/received per GPU ≈ 2*S bytes
```

### Bandwidth Utilization

**Theoretical Bandwidth**:
If each GPU has link bandwidth B:

```
Bandwidth utilization = (Data transferred) / (Time * Bandwidth)
                      = (2*S*(N-1)/N) / (T_ring * B)
                      ≈ (2*S) / (2*β*S * B)
                      = 1/β*B
                      = 1    (since β = 1/B)
```

**Result**: The Ring algorithm achieves 100% bandwidth utilization.

### Bus Bandwidth

**Bus Bandwidth** (metric used by NCCL tests):
```
BusBW = (Data sent + Data received) / Time
      = (2 * S * (N-1)/N + 2 * S * (N-1)/N) / T_ring
      = 4 * S * (N-1)/N / T_ring
```

For large messages:
```
BusBW ≈ 4*S / (2*β*S) = 2/β = 2*B
```

This means the bus bandwidth can exceed the link bandwidth by 2x due to simultaneous send/receive.

## Scalability Analysis

### Latency vs Number of GPUs

For fixed message size S:

```
T_ring = 2(N-1)*α + 2*β*S*(N-1)/N
```

**Latency-Dominated** (small S):
```
T_ring ≈ 2*N*α    (linear in N)
```

**Bandwidth-Dominated** (large S):
```
T_ring ≈ 2*β*S    (constant in N!)
```

**Key Insight**: For large messages, Ring latency is independent of N. This is the bandwidth-optimal property.

### Weak Scaling

If message size scales with N (S = N*S₀):

```
T_ring = 2(N-1)*α + 2*β*N*S₀*(N-1)/N
       ≈ 2*N*α + 2*β*N*S₀
```

**Latency scales linearly with N** - this limits scaling to hundreds of GPUs.

## Pipelining and Chunk Size

### Chunk Granularity

NCCL divides each channel's data into chunks and pipelines them; each chunk is further
split into slices and steps. The relevant compile-time constants (from
[nccl/src/include/collectives.h, lines 19-22](https://github.com/NVIDIA/nccl/blob/master/src/include/collectives.h),
unchanged in 2.31) are:

```
NCCL_STEPS            = 8            (FIFO depth per channel, src/include/device.h line 26)
ALLREDUCE_SLICESTEPS  = NCCL_STEPS/4 = 2
ALLREDUCE_CHUNKSTEPS  = NCCL_STEPS/2 = 4
```

The ring AllReduce kernel (`runRing` in
[src/device/all_reduce.h](https://github.com/NVIDIA/nccl/blob/master/src/device/all_reduce.h))
instantiates the Simple protocol as
`ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>`
(i.e. `SlicePerChunk = 2`, `StepPerSlice = 2`). It walks the buffer in units of
`loopCount = nranks * chunkCount`, where `chunkCount` is computed per channel by
`ncclCollCbdPart()`; the tail iteration re-aligns `chunkCount` to a 16-byte multiple.
So the effective decomposition is:

```
Per-channel data → chunks (chunkCount elements each)
Each chunk       → SlicePerChunk (=2) slices
Each slice       → StepPerSlice (=2) FIFO steps, bounded by NCCL_STEPS = 8 in flight
```

**Effect**: Smaller chunks reduce the pipeline bubble but increase per-step overhead;
the `NCCL_STEPS = 8` in-flight budget is the hard cap on how deeply a single channel can
pipeline. Thread count per block is chosen by the tuner
([src/tuning/tuning_general.cc](https://github.com/NVIDIA/nccl/blob/master/src/tuning/tuning_general.cc)),
not fixed at 8-16.

### Pipeline Efficiency

With fine-grained chunking, GPUs can overlap:
- Send chunk i
- Reduce chunk i-1
- Receive chunk i+1

This increases effective bandwidth but doesn't change the fundamental latency equation.

## Multi-Node Considerations

### Inter-Node Communication

For multi-node AllReduce with M nodes, each with P GPUs per node:
- Intra-node: Fast NVLink/PCIe (200-600 GB/s)
- Inter-node: Slower network (EFA: 400 Gbps = 50 GB/s)

**Hierarchical Ring**:
1. Reduce-scatter within each node (fast)
2. Ring across nodes (slow)
3. AllGather within each node (fast)

**Effective Latency**:
```
T_multi_node ≈ 2(P-1)*α_nvlink + 2*β_nvlink*S_local
             + 2(M-1)*α_network + 2*β_network*S_inter
             + 2(P-1)*α_nvlink + 2*β_nvlink*S_local
```

Where S_inter = S/(M*P) is the inter-node data per step.

## Comparison with Other Algorithms

### Ring vs Tree

| Aspect | Ring | Tree (Double Binary) |
|--------|------|---------------------|
| Latency | O(N) | O(log N) |
| Bandwidth | 100% | 100% |
| Best for | Large messages | Small-medium messages |
| Scalability | Hundreds of GPUs | Thousands of GPUs |

**Crossover Point** (estimated):
```
T_ring < T_tree when:
2*N*α + 2*β*S < 4*log₂(N)*α + 4*β*S
2*N*α < 4*log₂(N)*α + 2*β*S
N*α - 2*log₂(N)*α < β*S
α*(N - 2*log₂(N)) < β*S

S > α*(N - 2*log₂(N))/β
```

For N=8, α=10μs, β = 1/(50 GB/s) = **0.02 ns/byte** (400 Gbps ÷ 8 bits/byte = 50 GB/s):
```
S > 10 μs * (8 - 2*3) / 0.02 ns/byte
  > 20 μs / 0.02 ns/byte
  > 20e-6 s / 0.02e-9 s/byte
  > 1,000,000 bytes = 1 MB
```

Note β is **0.02** ns/byte, not 20. A link that took 20 ns to move one byte would run at
50 MB/s. The same β appears in [pat-algorithm.md](pat-algorithm.md) and
[tree-algorithm.md](tree-algorithm.md).

**Conclusion**: Ring is better for messages > 1 MB.

## Optimization Opportunities

### Chunk Size Tuning

**Too large**: Pipeline underutilization, higher latency
**Too small**: Excessive overhead, reduced bandwidth

**Optimal chunk size** (rule of thumb):
```
Chunk size ≈ 128 KB - 1 MB
```

### Protocol Selection

- **S < 32 KB**: Use Tree + LL (avoid Ring overhead)
- **32 KB < S < 1 MB**: Use Ring + LL128
- **S > 1 MB**: Use Ring + Simple

### Thread Count

More threads → finer pipelining → better overlap

Typical: 8-16 threads per GPU for Ring operations

## Summary

| Aspect | Value |
|--------|-------|
| **Latency** | `T = 2(N-1)*α + 2*β*S*(N-1)/N` |
| **Bandwidth** | ~100% of link bandwidth |
| **Scalability** | O(N) latency, best for N < 1000 |
| **Data per GPU** | `2*S*(N-1)/N` bytes sent/received |
| **Steps** | `2(N-1)` total steps |
| **Best for** | Large messages (> 1 MB) |
| **Protocols** | Simple (100% BW), LL128 (60-70% BW), LL (20-30% BW) |

**Key Takeaways**:
1. **Ring achieves 100% bandwidth** by pipelining data transfers
2. **Latency scales linearly with N** due to O(N) steps
3. **For large messages, latency is dominated by β*S** (bandwidth-limited)
4. **Each GPU transfers approximately 2*S bytes** for large N
5. **Optimal for messages > 1 MB** where bandwidth is critical
6. **Pipeline depth affects overlap** but not fundamental latency
7. **Multi-node requires hierarchical approach** to handle slow inter-node links

**Related Documentation**:
- [tree-algorithm.md](tree-algorithm.md) - Binary tree algorithm for low latency
- [nvls-tree-algorithm.md](nvls-tree-algorithm.md) - NVLS-accelerated tree algorithm
- [nccl-collectives.md](../nccl-collectives.md) - Overview of all collective algorithms

**References**:
1. [NCCL Performance Model](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
2. [Demystifying NCCL: An In-depth Analysis](https://arxiv.org/html/2507.04786v1)
3. Kurth et al., "Exascale Deep Learning for Climate Analytics"
