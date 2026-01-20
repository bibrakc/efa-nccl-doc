# PAT Algorithm - Parallel Aggregated Trees

**Purpose**: Understand the PAT (Parallel Aggregated Trees) algorithm for AllGather and ReduceScatter operations in NCCL, including its adaptive communication strategy, latency equations, and when to use it versus Ring or Tree algorithms.

---

## Table of Contents

1. [Algorithm Overview](#algorithm-overview)
2. [Key Characteristics](#key-characteristics)
3. [Algorithm Operation](#algorithm-operation)
4. [Mathematical Derivation of Latency](#mathematical-derivation-of-latency)
5. [Bandwidth Analysis](#bandwidth-analysis)
6. [Protocol-Specific Latency](#protocol-specific-latency)
7. [Comparison with Other Algorithms](#comparison-with-other-algorithms)
8. [When to Use PAT](#when-to-use-pat)
9. [Performance on AWS Instances](#performance-on-aws-instances)
10. [Implementation Details](#implementation-details)

---

## Algorithm Overview

The **PAT (Parallel Aggregated Trees)** algorithm is a hybrid collective communication algorithm introduced in NCCL 2.23 that combines the best aspects of logarithmic and linear communication patterns. It was specifically designed to address the performance limitations of Ring and Tree algorithms for small to medium message sizes at scale.

### Design Goals

1. **Logarithmic latency** for small messages (like Tree algorithm)
2. **Full bandwidth utilization** for larger messages (like Ring algorithm)
3. **Adaptive communication** that transitions smoothly based on message size
4. **Minimal buffer requirements** - only O(log N) internal buffers needed
5. **Any rank count** - works with non-power-of-two numbers of GPUs
6. **Long-distance minimization** - sends data from far dimensions first

### Supported Operations

PAT is currently implemented for:
- **AllGather**: Each rank has a unique chunk; all ranks gather all chunks
- **ReduceScatter**: All ranks reduce data and each gets a unique chunk
- **Note**: Not yet used for AllReduce (which can be decomposed into ReduceScatter + AllGather)

---

## Key Characteristics

### 1. Adaptive Communication Strategy

PAT adapts its communication pattern based on available buffer size:

```
Small messages:     Fully logarithmic steps (like Tree)
Medium messages:    Hybrid logarithmic + linear steps
Large messages:     Mostly linear steps (approaching Ring behavior)
```

### 2. Dimensional Communication Order

PAT sends data from **far dimensions first**, then closer dimensions:

```
Step 1: Communicate with ranks at distance 2^(k-1) (furthest)
Step 2: Communicate with ranks at distance 2^(k-2)
...
Step k: Communicate with ranks at distance 2^0 = 1 (nearest)
```

This minimizes long-distance communication early, improving overall efficiency.

### 3. Buffer Efficiency

Unlike recursive doubling algorithms that require O(N) buffer space, PAT only needs:
- **O(log N)** internal buffer space
- Dynamically adjusts communication to fit available buffers

---

## Algorithm Operation

### AllGather with PAT

Let's walk through PAT AllGather with N=8 GPUs, each starting with chunk size S/8:

```
Initial state:
GPU 0: [A0 -- -- -- -- -- -- --]
GPU 1: [-- A1 -- -- -- -- -- --]
GPU 2: [-- -- A2 -- -- -- -- --]
GPU 3: [-- -- -- A3 -- -- -- --]
GPU 4: [-- -- -- -- A4 -- -- --]
GPU 5: [-- -- -- -- -- A5 -- --]
GPU 6: [-- -- -- -- -- -- A6 --]
GPU 7: [-- -- -- -- -- -- -- A7]
```

**Step 1: Distance 4 communication** (far dimension)

```
GPU 0 ↔ GPU 4:  Exchange A0 ↔ A4
GPU 1 ↔ GPU 5:  Exchange A1 ↔ A5
GPU 2 ↔ GPU 6:  Exchange A2 ↔ A6
GPU 3 ↔ GPU 7:  Exchange A3 ↔ A7

After Step 1:
GPU 0: [A0 -- -- -- A4 -- -- --]
GPU 1: [-- A1 -- -- -- A5 -- --]
GPU 2: [-- -- A2 -- -- -- A6 --]
GPU 3: [-- -- -- A3 -- -- -- A7]
GPU 4: [A0 -- -- -- A4 -- -- --]
GPU 5: [-- A1 -- -- -- A5 -- --]
GPU 6: [-- -- A2 -- -- -- A6 --]
GPU 7: [-- -- -- A3 -- -- -- A7]
```

**Step 2: Distance 2 communication**

```
GPU 0 ↔ GPU 2:  Exchange (A0,A4) ↔ (A2,A6)
GPU 1 ↔ GPU 3:  Exchange (A1,A5) ↔ (A3,A7)
GPU 4 ↔ GPU 6:  Exchange (A0,A4) ↔ (A2,A6)
GPU 5 ↔ GPU 7:  Exchange (A1,A5) ↔ (A3,A7)

After Step 2:
GPU 0: [A0 -- A2 -- A4 -- A6 --]
GPU 1: [-- A1 -- A3 -- A5 -- A7]
GPU 2: [A0 -- A2 -- A4 -- A6 --]
GPU 3: [-- A1 -- A3 -- A5 -- A7]
GPU 4: [A0 -- A2 -- A4 -- A6 --]
GPU 5: [-- A1 -- A3 -- A5 -- A7]
GPU 6: [A0 -- A2 -- A4 -- A6 --]
GPU 7: [-- A1 -- A3 -- A5 -- A7]
```

**Step 3: Distance 1 communication** (near dimension)

```
GPU 0 ↔ GPU 1:  Exchange (A0,A2,A4,A6) ↔ (A1,A3,A5,A7)
GPU 2 ↔ GPU 3:  Exchange (A0,A2,A4,A6) ↔ (A1,A3,A5,A7)
GPU 4 ↔ GPU 5:  Exchange (A0,A2,A4,A6) ↔ (A1,A3,A5,A7)
GPU 6 ↔ GPU 7:  Exchange (A0,A2,A4,A6) ↔ (A1,A3,A5,A7)

Final state:
GPU 0-7: [A0 A1 A2 A3 A4 A5 A6 A7]  ✓ Complete
```

### Key Observations

1. **Log₂(N) = 3 steps** for N=8 GPUs (same as Tree)
2. **Parallel communication** - all GPUs communicate simultaneously
3. **Exponentially increasing data** - each step doubles the data exchanged
4. **Far-to-near ordering** - minimizes expensive long-distance transfers

---

## Mathematical Derivation of Latency

We'll use the **α-β latency model** where:
- **α** (alpha) = latency per message (network/switch latency)
- **β** (beta) = per-byte transmission time = 1/bandwidth
- **S** = total message size
- **N** = number of GPUs
- **k** = ⌈log₂(N)⌉ = number of logarithmic steps

### AllGather Latency Derivation

In each step i (where i goes from 1 to k), the amount of data exchanged is:

```
Step i exchanges: S/N × 2^i bytes
```

#### Step-by-Step Breakdown

**Step 1** (distance 2^(k-1)):
- Data per GPU: S/N bytes
- Latency: α + β × (S/N)

**Step 2** (distance 2^(k-2)):
- Data per GPU: 2S/N bytes (double from step 1)
- Latency: α + β × (2S/N)

**Step i** (distance 2^(k-i)):
- Data per GPU: 2^(i-1) × S/N bytes
- Latency: α + β × (2^(i-1) × S/N)

**Step k** (distance 1):
- Data per GPU: 2^(k-1) × S/N bytes
- Latency: α + β × (2^(k-1) × S/N)

#### Total Latency

Summing over all k steps:

```
T_pat_allgather = Σ(i=1 to k) [α + β × (2^(i-1) × S/N)]
                = k×α + β×(S/N) × Σ(i=1 to k) 2^(i-1)
                = k×α + β×(S/N) × (2^k - 1)
```

For N = 2^k (power of two):

```
T_pat_allgather = log₂(N)×α + β×S×(N-1)/N
```

**Simplified**:

```
T_pat_allgather = log₂(N)×α + β×S×(1 - 1/N)
```

### ReduceScatter Latency Derivation

ReduceScatter follows the reverse pattern of AllGather. Each GPU starts with full data S and ends with S/N:

**Step 1** (distance 2^(k-1)):
- Reduce-exchange: S/2 bytes per GPU
- Latency: α + β×(S/2) + γ×(S/2)

**Step 2** (distance 2^(k-2)):
- Reduce-exchange: S/4 bytes per GPU
- Latency: α + β×(S/4) + γ×(S/4)

**Step i** (distance 2^(k-i)):
- Reduce-exchange: S/2^i bytes per GPU
- Latency: α + β×(S/2^i) + γ×(S/2^i)

Where γ (gamma) is the per-byte compute time for reduction operations.

#### Total ReduceScatter Latency

```
T_pat_reduce_scatter = Σ(i=1 to k) [α + (β + γ)×(S/2^i)]
                     = k×α + (β + γ)×S × Σ(i=1 to k) (1/2^i)
                     = k×α + (β + γ)×S × (1 - 1/2^k)
```

For large N:

```
T_pat_reduce_scatter ≈ log₂(N)×α + (β + γ)×S
```

### Combined PAT AllReduce Latency

AllReduce = ReduceScatter + AllGather:

```
T_pat_allreduce = T_pat_reduce_scatter + T_pat_allgather
                = [log₂(N)×α + (β + γ)×S] + [log₂(N)×α + β×S×(1 - 1/N)]
                = 2×log₂(N)×α + (2β + γ)×S - β×S/N
```

**Simplified** (assuming γ << β and large N):

```
T_pat_allreduce ≈ 2×log₂(N)×α + 2×β×S
```

This is essentially the same as the **double binary tree** algorithm.

---

## Bandwidth Analysis

### Effective Bandwidth for AllGather

The effective bandwidth utilization for PAT AllGather is:

```
BW_eff = Data moved / Time spent
       = S × (N-1)/N / [log₂(N)×α + β×S×(N-1)/N]
```

For large messages where β×S >> α:

```
BW_eff ≈ S×(N-1)/N / [β×S×(N-1)/N]
       = 1/β
       = Network bandwidth
```

**Result**: PAT achieves **~100% bandwidth utilization** for large messages, similar to Ring.

### Bandwidth vs. Ring and Tree

| Algorithm | Steps | Latency (α term) | Latency (β term) | Bandwidth Eff |
|-----------|-------|------------------|------------------|---------------|
| Ring      | 2(N-1)| 2(N-1)×α        | 2β×S×(N-1)/N    | ~100%         |
| Tree      | 2log₂(N) | 2log₂(N)×α   | log₂(N)×β×S     | ~67% (N=8)    |
| PAT       | log₂(N) | log₂(N)×α     | β×S×(N-1)/N     | ~100%         |

**Key Insight**: PAT achieves logarithmic latency (like Tree) while maintaining full bandwidth (like Ring).

---

## Protocol-Specific Latency

NCCL supports three protocols, each with different performance characteristics:

### 1. Simple Protocol (Full Bandwidth)

Uses full MTU (typically 128KB or 256KB chunks):

```
T_pat_simple = log₂(N)×α_simple + β_simple×S×(N-1)/N
```

**Parameters** (p5 instance, EFA):
- α_simple ≈ 15-20 μs (EFA network latency)
- β_simple = 1 / 400 Gbps ≈ 0.0025 ns/byte

**Example** (N=8, S=1 MB):
```
T_pat_simple = 3 × 18 μs + 0.0025 ns/byte × 1 MB × 7/8
             = 54 μs + 2.19 μs
             = 56.19 μs
```

### 2. LL128 Protocol (128-byte chunks)

Uses 128-byte chunks with 8-byte metadata:

```
T_pat_ll128 = log₂(N)×α_ll128 + β_ll128×S×(N-1)/N × (128+8)/128
```

**Parameters** (p5 instance):
- α_ll128 ≈ 8-12 μs (lower latency than Simple)
- β_ll128 ≈ β_simple × 1.0625 (8-byte overhead)

**Example** (N=8, S=128 KB):
```
T_pat_ll128 = 3 × 10 μs + 0.00266 ns/byte × 128 KB × 7/8
             = 30 μs + 0.296 μs
             = 30.3 μs
```

### 3. LL Protocol (8-byte chunks)

Uses 8-byte chunks, optimized for very low latency:

```
T_pat_ll = log₂(N)×α_ll + β_ll×S×(N-1)/N
```

**Parameters** (p5 instance):
- α_ll ≈ 5-8 μs (lowest latency)
- β_ll ≈ 2-3× β_simple (low bandwidth due to overhead)

**Example** (N=8, S=4 KB):
```
T_pat_ll = 3 × 6 μs + 0.0075 ns/byte × 4 KB × 7/8
             = 18 μs + 0.026 μs
             = 18.03 μs
```

### Protocol Selection Heuristic

```
If S < 32 KB:          Use LL (minimize latency)
If 32 KB ≤ S < 1 MB:   Use LL128 (balanced)
If S ≥ 1 MB:           Use Simple (maximize bandwidth)
```

---

## Comparison with Other Algorithms

### PAT vs. Ring vs. Tree

| Metric | Ring | Tree | PAT |
|--------|------|------|-----|
| **Latency (α term)** | O(N) | O(log N) | O(log N) |
| **Bandwidth** | ~100% | ~67% (N=8) | ~100% |
| **Steps** | 2(N-1) | 2log₂(N) | log₂(N) |
| **Best for** | Large msgs | Med msgs | Small-med msgs |
| **Buffer reqs** | O(1) | O(1) | O(log N) |
| **Rank flexibility** | Any | Power-of-2 preferred | Any |

### Latency Crossover Points

For typical p5 instance parameters:

**PAT vs. Ring**:
```
T_pat < T_ring when:
log₂(N)×α + β×S×(1-1/N) < 2(N-1)×α + 2β×S×(1-1/N)

Simplifying for large N:
log₂(N)×α + β×S < 2N×α + 2β×S
log₂(N)×α < 2N×α + β×S
```

**Crossover point**:
```
S_crossover ≈ (2N - log₂(N))×α / β

For N=8:   S ≈ (16 - 3) × 15 μs / 0.0025 ns/byte = 78 KB
For N=64:  S ≈ (128 - 6) × 15 μs / 0.0025 ns/byte = 732 KB
```

**Interpretation**: PAT is faster than Ring for messages below ~100 KB to ~1 MB (depending on scale).

**PAT vs. Tree**:
```
T_pat ≈ T_tree for AllGather

PAT AllGather:  log₂(N)×α + β×S×(N-1)/N
Tree AllGather: log₂(N)×α + log₂(N)×β×S

PAT is faster when:
β×S×(N-1)/N < log₂(N)×β×S
(N-1)/N < log₂(N)
```

For N ≥ 4, (N-1)/N approaches 1, while log₂(N) grows. So **Tree is better for medium messages** where bandwidth isn't saturated.

**Optimal regions**:
- **S < 100 KB**: PAT (or Tree for N < 8)
- **100 KB < S < 1 MB**: Tree or PAT (similar performance)
- **S > 1 MB**: Ring (best bandwidth)

---

## When to Use PAT

### Optimal Use Cases

1. **Large-scale systems** (N ≥ 16 GPUs):
   - Logarithmic latency becomes critical
   - Ring's O(N) latency is prohibitive

2. **Small to medium messages** (32 KB to 1 MB):
   - Below Ring crossover point
   - Latency-sensitive workloads

3. **Pipeline and tensor parallelism**:
   - Frequent small AllGather operations
   - Common in large language model (LLM) training

4. **One GPU per node scenarios**:
   - Initial PAT implementation targets inter-node communication
   - Particularly effective when NVLink isn't available

5. **Non-power-of-two ranks**:
   - PAT handles any N efficiently
   - Tree algorithms may be less efficient

### When to Use Other Algorithms

**Use Ring when**:
- S > 1 MB (bandwidth-limited regime)
- Maximum throughput is critical
- Small number of GPUs (N ≤ 8)

**Use Tree when**:
- 100 KB < S < 1 MB with N < 16
- Need predictable O(log N) performance
- Power-of-two GPU counts

**Use NVLS/NVLS Tree when**:
- Intra-node communication on p5/p5en
- S < 128 MB
- Hardware acceleration available (H100/H200)

---

## Performance on AWS Instances

### p5 Instances (8× H100 80GB)

**Hardware**:
- 8× NVIDIA H100 GPUs with NVSwitch
- 3.2 Tbps NVLink bandwidth per GPU
- 4× 400 Gbps EFA adapters (total 1600 Gbps)
- 200 Gbps EFA per GPU

**PAT Performance** (single-node, inter-GPU via EFA):

| Message Size | PAT Latency | Ring Latency | Speedup |
|--------------|-------------|--------------|---------|
| 4 KB         | ~18 μs      | ~90 μs       | 5.0×    |
| 32 KB        | ~30 μs      | ~120 μs      | 4.0×    |
| 128 KB       | ~48 μs      | ~180 μs      | 3.75×   |
| 512 KB       | ~110 μs     | ~290 μs      | 2.64×   |
| 1 MB         | ~180 μs     | ~450 μs      | 2.5×    |
| 4 MB         | ~620 μs     | ~1100 μs     | 1.77×   |

**Note**: These are estimated values based on α-β model. Actual performance may vary.

### Multi-Node Performance (p5 cluster)

For M nodes, each with 8 GPUs (total N = 8M GPUs):

**Inter-node AllGather latency**:
```
T_pat_multi = log₂(M)×α_network + β_network×S×(M-1)/M
            + log₂(8)×α_nvlink + β_nvlink×S×7/8
```

**Example** (M=16 nodes, 128 GPUs, S=256 KB):
```
Inter-node:  4 × 15 μs + 0.00125 ns/byte × 256 KB × 15/16
           = 60 μs + 0.30 μs = 60.3 μs

Intra-node:  3 × 1 μs + 0.00031 ns/byte × 256 KB × 7/8
           = 3 μs + 0.069 μs = 3.07 μs

Total:       63.4 μs
```

---

## Implementation Details

### NCCL Algorithm Selection

PAT is selected by NCCL's tuner based on:

1. **Collective type**: AllGather or ReduceScatter
2. **Message size**: Typically S < 1 MB
3. **Number of ranks**: More effective at N ≥ 16
4. **Protocol**: Usually paired with LL128 or Simple

### Code References

PAT algorithm is defined in NCCL headers:

```c
// From nccl/tuner.h
#define NCCL_ALGO_PAT 6

typedef enum {
  ncclFuncAllGather = 2,
  ncclFuncReduceScatter = 3,
  ...
} ncclFunc_t;
```

In aws-ofi-nccl tuner ([src/tuner/nccl_ofi_regions.cpp](../src/tuner/nccl_ofi_regions.cpp)):

```cpp
// PAT algorithm selection for AllGather
{.algorithm = NCCL_ALGO_PAT,
 .protocol = NCCL_PROTO_SIMPLE,
 .num_vertices = ...,
 .vertices = {...}}
```

### Buffer Management

PAT requires **O(log N)** internal buffers:
- Each logarithmic step needs a temporary buffer
- Buffer size = S/N × 2^i for step i
- Total buffer allocation ≈ 2×S per GPU (worst case)

### Current Limitations (NCCL 2.23)

1. **Inter-node only**: Initial implementation for one GPU per node
2. **No AllReduce**: Must decompose into ReduceScatter + AllGather
3. **Simple protocol preferred**: Best with NCCL_PROTO_SIMPLE
4. **Tuner dependency**: Requires proper tuner configuration

---

## Key Takeaways

1. **PAT achieves logarithmic latency with full bandwidth**:
   - Latency: `T = log₂(N)×α + β×S×(1-1/N)`
   - Combines best of Tree (low latency) and Ring (high bandwidth)

2. **Optimal for small-to-medium messages at scale**:
   - Crossover vs Ring: ~100 KB to 1 MB
   - Most beneficial when N ≥ 16 GPUs

3. **Adaptive communication strategy**:
   - Starts with logarithmic aggregation
   - Transitions to linear steps as needed
   - Minimizes long-distance communication

4. **Critical for modern LLM training**:
   - Pipeline parallelism: frequent small AllGather
   - Tensor parallelism: latency-sensitive communication
   - Multi-node scaling: logarithmic >>> linear latency

5. **AWS-specific considerations**:
   - Effective on p5/p5en for inter-node communication
   - Pair with NVLS/NVLS Tree for intra-node (when available)
   - EFA network: α ≈ 15 μs, bandwidth up to 200 Gbps/GPU

---

## References

- **NCCL 2.23 Release** - [NVIDIA Technical Blog: New Scaling Algorithm](https://developer.nvidia.com/blog/new-scaling-algorithm-and-initialization-with-nvidia-collective-communications-library-2-23/)
- **PAT Paper** - [arXiv:2506.20252 - PAT: a new algorithm for all-gather and reduce-scatter operations at scale](https://arxiv.org/abs/2506.20252)
- **NCCL Source** - [NVIDIA/nccl on GitHub](https://github.com/NVIDIA/nccl)
- **aws-ofi-nccl** - [aws/aws-ofi-nccl - EFA network plugin](https://github.com/aws/aws-ofi-nccl)

---

**Document Version**: 1.0
**Last Updated**: January 2025
**NCCL Version**: 2.23+
