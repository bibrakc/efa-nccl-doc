# PAT Algorithm - Parallel Aggregated Trees

> ## ⚠️ DRAFT - NEEDS VERIFICATION ⚠️
>
> **This document contains theoretical derivations and performance analysis that have NOT been fully verified.**
>
> All latency formulas, bandwidth calculations, and performance claims should be independently validated before being used for system design or optimization decisions. The mathematical derivations may contain errors or oversimplifications.
>
> Please treat all content as preliminary and subject to revision.

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

**PAT runs with the Simple protocol only.** The algorithm registry contains exactly one PAT
entry and no LL or LL128 variant:

```c
// nccl/src/config/algorithm_registry.cc, line 52
{"PAT_SIMPLE", ALGBIT(20), F_AG | F_RS, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, -1},
```

Contrast Ring, which has `RING_LL`, `RING_LL128` and `RING_SIMPLE` (lines 35-37). Two
consequences follow, and both are easy to get wrong:

- `NCCL_PROTO=LL` or `LL128` **cannot** be combined with PAT. Requesting PAT with a
  non-Simple protocol does not yield a faster PAT; it takes PAT out of the running.
- `F_AG | F_RS` means PAT applies to **AllGather and ReduceScatter only** — not AllReduce.

The LL and LL128 figures below are therefore kept only as a **contrast against Ring at the
same message size**, showing why PAT's Simple-only restriction costs little for the sizes
where PAT is chosen. They are not selectable PAT configurations.

### 1. Simple Protocol (the only PAT protocol)

Uses full MTU (typically 128KB or 256KB chunks):

```
T_pat_simple = log₂(N)×α_simple + β_simple×S×(N-1)/N
```

**Parameters** (p5 instance, EFA):
- α_simple ≈ 15-20 μs (EFA network latency)
- β_simple = 1 / 400 Gbps = 1 / (50 GB/s) = **0.02 ns/byte**

**Example** (N=8, S=1 MB):
```
T_pat_simple = 3 × 18 μs + 0.02 ns/byte × 1 MiB × 7/8
             = 54 μs + 18.35 μs
             = 72.4 μs
```

### 2. LL128 Protocol — for comparison only (Ring/Tree can use it, PAT cannot)

Uses 128-byte chunks with 8-byte metadata:

```
T_ll128 = log₂(N)×α_ll128 + β_ll128×S×(N-1)/N × (128+8)/128
```

**Parameters** (p5 instance):
- α_ll128 ≈ 8-12 μs (lower latency than Simple)
- β_ll128 ≈ β_simple × 1.0625 = 0.02125 ns/byte (8-byte metadata per 128-byte chunk)

**Example** (N=8, S=128 KB):
```
T_ll128 = 3 × 10 μs + 0.02125 ns/byte × 128 KiB × 7/8
        = 30 μs + 2.44 μs
        = 32.4 μs
```

### 3. LL Protocol — for comparison only (Ring/Tree can use it, PAT cannot)

Uses 8-byte chunks, optimized for very low latency:

```
T_ll = log₂(N)×α_ll + β_ll×S×(N-1)/N
```

**Parameters** (p5 instance):
- α_ll ≈ 5-8 μs (lowest latency)
- β_ll ≈ 3× β_simple = 0.06 ns/byte (LL carries 8 bytes of payload per 16-byte line)

**Example** (N=8, S=4 KB):
```
T_ll = 3 × 6 μs + 0.06 ns/byte × 4 KiB × 7/8
     = 18 μs + 0.22 μs
     = 18.2 μs
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

For N=8:   S ≈ (16 - 3) × 15 μs / 0.02 ns/byte = 9.8 MB
For N=64:  S ≈ (128 - 6) × 15 μs / 0.02 ns/byte = 92 MB
```

**Interpretation, with a caveat.** Taken literally the closed form says PAT wins below
roughly 10 MB at N=8 and 90 MB at N=64. Do not use those figures as operating thresholds:
they are far larger than where the tuner actually switches. The discrepancy comes from the
model's premise that Ring pays twice PAT's bandwidth term, which overstates Ring's cost —
Ring AllGather is bandwidth-optimal. Treat the formula as showing the *shape* (PAT's
`log₂(N)` latency term beats Ring's `2(N-1)` term, and that advantage shrinks as the message
grows), and take real thresholds from the tuner: `chunkSizeTuningAllGatherPatSimpleP5en()`
steps down from a 1 MiB ceiling (512 KiB at `nNodes >= 16`) to a 32 KiB floor. See
[nccl-tuner.md](../nccl-tuner.md).

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

**PAT performance: not tabulated here, deliberately.**

An earlier revision of this document carried a table of PAT-versus-Ring latencies for p5. Those
figures were **model output presented as measurements**, and they did not even follow from the
model's own β. They have been removed rather than corrected, because a plausible-looking
latency table is precisely the kind of thing that gets quoted in a design review.

What the model does support, qualitatively:

- PAT's advantage is largest for **small messages**, where the `log₂(N)` versus `2(N-1)` latency
  term dominates, and shrinks as the message grows and the bandwidth term takes over.
- The advantage grows with **N**, since `2(N-1)` grows linearly while `log₂(N)` does not.

For numbers you can act on, measure on the target instance:

```bash
# PAT applies to AllGather and ReduceScatter only
./build/all_gather_perf -b 8 -e 128M -f 2 -g 8      # let the tuner choose
NCCL_ALGO=PAT ./build/all_gather_perf -b 8 -e 128M -f 2 -g 8    # force PAT
NCCL_ALGO=Ring ./build/all_gather_perf -b 8 -e 128M -f 2 -g 8   # force Ring
```

Then compare against the tuner's own thresholds in [nccl-tuner.md](../nccl-tuner.md), which
records what AWS actually ships per instance type.

### Multi-Node Performance (p5 cluster)

For M nodes, each with 8 GPUs (total N = 8M GPUs):

**Inter-node AllGather latency**:
```
T_pat_multi = log₂(M)×α_network + β_network×S×(M-1)/M
            + log₂(8)×α_nvlink + β_nvlink×S×7/8
```

**Example** (M=16 nodes, 128 GPUs, S=256 KB):
```
Inter-node:  4 × 15 μs + 0.005 ns/byte × 256 KiB × 15/16   (1600 Gbps = 200 GB/s per node)
           = 60 μs + 1.23 μs = 61.2 μs

Intra-node:  3 × 1 μs + 0.0025 ns/byte × 256 KiB × 7/8      (3.2 Tbps = 400 GB/s per GPU)
           = 3 μs + 0.57 μs = 3.6 μs

Total:       64.8 μs
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

### Current Limitations (NCCL 2.23; still applies through NCCL 2.31)

1. **Inter-node only**: Initial implementation for one GPU per node
2. **No AllReduce**: Must decompose into ReduceScatter + AllGather
3. **Simple protocol preferred**: Best with NCCL_PROTO_SIMPLE
4. **Tuner dependency**: Requires proper tuner configuration

> **AWS OFI tuner PAT optimizations (current).** The AWS region tuner now
> actively tunes PAT in the one-rank-per-node (0x7) topology:
> - **AllGather PAT/Simple chunk sizing on P5en** (commit cb33a4c) via the v6
>   `getChunkSize` callback — steps a per-cluster chunk ceiling (1 MiB for
>   `nNodes<16`, 512 KiB otherwise) down to 32 KiB to keep mid/large messages in
>   a single pipelined phase (PAT pipelines within a phase but not across
>   phases; inflight budget is `NCCL_STEPS = 8` per channel).
> - **PAT channel selection on P6 and P6-B300** (commit 4a8ed08) — for
>   PAT/Simple AllGather/ReduceScatter with `nBytes ≤ 32 MiB` and
>   `num_nodes == num_ranks`, `nChannels` is overridden (1–2 channels) via
>   `calculateBestNChannelPat()`. See [nccl-tuner.md](../nccl-tuner.md).

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
   - Effective on p5/p5en/p6/p6-B300 for inter-node communication
   - Pair with NVLS/NVLS Tree for intra-node (when available)
   - EFA network: α ≈ 15 μs, bandwidth up to 200 Gbps/GPU

---

## References

- **NCCL 2.23 Release** - [NVIDIA Technical Blog: New Scaling Algorithm](https://developer.nvidia.com/blog/new-scaling-algorithm-and-initialization-with-nvidia-collective-communications-library-2-23/)
- **PAT Paper** - [arXiv:2506.20252 - PAT: a new algorithm for all-gather and reduce-scatter operations at scale](https://arxiv.org/abs/2506.20252)
- **NCCL Source** - [NVIDIA/nccl on GitHub](https://github.com/NVIDIA/nccl)
- **aws-ofi-nccl** - [aws/aws-ofi-nccl - EFA network plugin](https://github.com/aws/aws-ofi-nccl)

---

**Document Version**: 1.1
**Last Updated**: September 2026
**NCCL Version**: PAT introduced in 2.23; cross-checked against NCCL v2.31.2-1 and aws-ofi-nccl v1.21.1
