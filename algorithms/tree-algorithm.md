# Tree AllReduce Algorithm

> ## ⚠️ DRAFT - NEEDS VERIFICATION ⚠️
>
> **This document contains theoretical derivations and performance analysis that have NOT been fully verified.**
>
> All latency formulas, bandwidth calculations, and performance claims should be independently validated before being used for system design or optimization decisions. The mathematical derivations may contain errors or oversimplifications.
>
> Please treat all content as preliminary and subject to revision.

## Overview

The **Tree algorithm** (specifically NCCL's **Double Binary Tree**) is a latency-optimized algorithm for collective operations that achieves logarithmic latency while maintaining full bandwidth utilization. It uses two independent binary trees to avoid bottlenecks at the root.

**Best For**: Small to medium messages (32 KB - 1 MB) where latency is critical

**Key Properties**:
- Latency: O(log N) - logarithmic in number of GPUs
- Bandwidth utilization: ~100% (with double tree)
- Data movement: Each GPU sends and receives approximately S bytes
- Scalability: Excellent for thousands of GPUs

**Source**: The double binary tree is constructed by `ncclGetDtree()` / `ncclGetBtree()`
in [nccl/src/graph/trees.cc](https://github.com/NVIDIA/nccl/blob/master/src/graph/trees.cc);
these per-node trees are wired across nodes in
[src/graph/connect.cc](https://github.com/NVIDIA/nccl/blob/master/src/graph/connect.cc)
(calls to `ncclGetDtree`), and the device-side reduce/broadcast steps run in
[src/device/all_reduce.h](https://github.com/NVIDIA/nccl/blob/master/src/device/all_reduce.h).

## Binary Tree Topology

### Single Binary Tree Structure

A binary tree with N GPUs has:
- **Depth**: ⌈log₂(N)⌉ levels
- **Root**: One GPU (potential bottleneck)
- **Internal nodes**: GPUs with children
- **Leaves**: GPUs with no children

Each GPU has:
- Up to 2 children (left and right)
- 1 parent (except root)

> Implementation note: the per-tree *btree* built by `ncclGetBtree()` is genuinely binary
> (each node has `d0`/`d1`), so the mathematical derivation below assumes 2 children. NCCL's
> device tree structure, however, allows `NCCL_MAX_TREE_ARITY = 3` children
> ([nccl/src/include/device.h, line 193](https://github.com/NVIDIA/nccl/blob/master/src/include/device.h))
> so that an inter-node tree parent can also fan out to intra-node local ranks. This does
> not change the O(log N) latency scaling.

**Example for N=8 GPUs**:
```
         GPU 0 (root)
        /            \
     GPU 1          GPU 2
    /     \        /     \
  GPU 3  GPU 4  GPU 5  GPU 6
          |
        GPU 7
```

**Problem with Single Tree**: The root becomes a bottleneck, sending/receiving all data.

## Double Binary Tree Innovation

**NCCL's Solution**: Use **two independent binary trees** where:
1. No GPU is a non-leaf in both trees
2. At most one GPU is a leaf in both trees
3. Each tree handles half the data

**Key Insight**: If each tree processes S/2 bytes:
- Each GPU receives ≤ S/2 bytes twice = S bytes total
- Each GPU sends ≤ S/2 bytes twice = S bytes total
- **Same data movement as Ring** but in O(log N) steps instead of O(N)!

### Example Double Binary Tree (N=8)

The following is the **actual** output of NCCL's `ncclGetDtree(8, rank)` (mirror case,
since `nranks = 8` is even) from
[src/graph/trees.cc](https://github.com/NVIDIA/nccl/blob/master/src/graph/trees.cc). Note
that NCCL's btree roots the first tree at rank 0 with a single child, so the trees are
"caterpillar" binary trees rather than the perfectly balanced tree drawn in textbooks.

**Tree 0** (root 0 — parent/children per rank: 0→[4]; 4→[2,6]; 2→[1,3]; 6→[5,7]):
```
        0
        |
        4
       / \
      2   6
     / \ / \
    1  3 5  7
```

**Tree 1** (mirror; root 7 — 7→[3]; 3→[1,5]; 1→[2,0]; 5→[6,4]):
```
        7
        |
        3
       / \
      1   5
     / \ / \
    2  0 6  4
```

**Property Check** (from the computed parent/child relations above):
- GPU 0: Root in Tree 0, Leaf in Tree 1 ✓
- GPU 1: Leaf in Tree 0, Internal in Tree 1 ✓
- GPU 2: Internal in Tree 0, Leaf in Tree 1 ✓
- GPU 3: Leaf in Tree 0, Internal in Tree 1 ✓
- GPU 4: Internal in Tree 0, Leaf in Tree 1 ✓
- GPU 5: Leaf in Tree 0, Internal in Tree 1 ✓
- GPU 6: Internal in Tree 0, Leaf in Tree 1 ✓
- GPU 7: Leaf in Tree 0, Root in Tree 1 ✓

Every rank that is internal (non-leaf) in one tree is a leaf in the other, so no rank is a
bottleneck in both trees — this is the double-tree invariant. (The construction algorithm
itself is described in detail in the *Tree Construction in NCCL* section below.)

## Algorithm Phases

Tree AllReduce consists of two phases:

1. **Reduce Phase**: Data flows from leaves to root (bottom-up)
2. **Broadcast Phase**: Reduced data flows from root to leaves (top-down)

Each phase operates independently on each tree with half the data.

### Phase 1: Reduce (Bottom-Up)

**Goal**: Accumulate data from all GPUs at the root

**Process**:
Starting from leaves, moving up to root:
1. Leaf GPUs send their data to parents
2. Internal GPUs:
   - Wait for children's data
   - Reduce (sum) with local data
   - Send result to parent
3. Root receives and reduces all data

**Steps** = ⌈log₂(N)⌉ (tree depth)

**Example for Tree 1 (N=8, depth=3)**:

```
Step 1 (leaves to parents):
  GPU 1 → GPU 2: Send chunk A₁
  GPU 3 → GPU 2: Send chunk A₃
  GPU 5 → GPU 4: Send chunk A₅
  GPU 7 → GPU 4: Send chunk A₇

Step 2 (internal to parents):
  GPU 2: Reduce A₂ + A₁ + A₃ → GPU 0
  GPU 4: Reduce A₄ + A₅ + A₇ → GPU 0

Step 3 (children to root):
  GPU 0: Reduce A₀ + (A₂+A₁+A₃) + (A₄+A₅+A₇)
  Result: A_reduced = A₀+A₁+A₂+A₃+A₄+A₅+A₆+A₇
```

### Phase 2: Broadcast (Top-Down)

**Goal**: Distribute reduced result from root to all GPUs

**Process**:
Starting from root, moving down to leaves:
1. Root sends result to children
2. Internal GPUs:
   - Receive data from parent
   - Forward to children
3. Leaves receive final result

**Steps** = ⌈log₂(N)⌉ (tree depth)

**Example for Tree 1 (N=8, depth=3)**:

```
Step 1 (root to children):
  GPU 0 → GPU 2: Send A_reduced
  GPU 0 → GPU 4: Send A_reduced

Step 2 (parents to children):
  GPU 2 → GPU 1: Send A_reduced
  GPU 2 → GPU 3: Send A_reduced
  GPU 4 → GPU 5: Send A_reduced
  GPU 4 → GPU 7: Send A_reduced

Step 3 (complete):
  All GPUs have A_reduced
```

### Double Tree Operation

Both trees operate **in parallel** on different halves of the data:

**Tree 1**: Processes chunks 0 to (S/2 - 1)
**Tree 2**: Processes chunks S/2 to (S - 1)

**Timeline**:
```
Time 0: Both trees start reduce phase (log₂(N) steps)
Time T_reduce: Both trees start broadcast phase (log₂(N) steps)
Time T_reduce + T_broadcast: Complete
```

## Mathematical Derivation of Latency

### Latency Model Parameters

**α-β Model**:
- **α (alpha)**: Latency per message (μs)
- **β (beta)**: Per-byte transmission time (μs/byte) = 1/bandwidth
- **S**: Total message size (bytes)
- **N**: Number of GPUs
- **D**: Tree depth = ⌈log₂(N)⌉
- **γ (gamma)**: Reduction computation time per byte (μs/byte)

### Single Tree Latency

**Reduce Phase**:
Each step up the tree involves sending S bytes:

```
Time per step = α + β*S + γ*S
```

For D steps:
```
T_reduce_single = D * (α + β*S + γ*S)
                = D*α + D*(β + γ)*S
```

**Broadcast Phase**:
Each step down the tree involves sending S bytes (no reduction):

```
Time per step = α + β*S
```

For D steps:
```
T_broadcast_single = D * (α + β*S)
                   = D*α + D*β*S
```

**Total Single Tree**:
```
T_single_tree = T_reduce_single + T_broadcast_single
              = 2*D*α + D*(2*β + γ)*S
```

**Problem**: Each step involves full message size S → bottleneck at root

### Double Tree Latency

With two trees, each handling S/2 bytes:

**Reduce Phase per Tree**:
```
T_reduce_per_tree = D*α + D*(β + γ)*S/2
```

**Broadcast Phase per Tree**:
```
T_broadcast_per_tree = D*α + D*β*S/2
```

**Total per Tree**:
```
T_per_tree = 2*D*α + D*(2*β + γ)*S/2
```

**Both Trees in Parallel**:
```
T_double_tree = max(T_tree1, T_tree2)
              = 2*D*α + D*(2*β + γ)*S/2
              = 2*D*α + D*β*S + D*γ*S/2
```

**Simplified** (assuming γ << β for GPU reductions):
```
T_tree = 2*D*α + D*β*S
       = 2*log₂(N)*α + log₂(N)*β*S
```

**Key Result**: Latency is logarithmic in N!

### Comparison with Ring

**Ring Latency**:
```
T_ring = 2*(N-1)*α + 2*β*S*(N-1)/N
       ≈ 2*N*α + 2*β*S    (for large N)
```

**Tree Latency**:
```
T_tree = 2*log₂(N)*α + log₂(N)*β*S
```

**Latency Ratio**:
```
T_tree / T_ring = [2*log₂(N)*α + log₂(N)*β*S] / [2*N*α + 2*β*S]
```

For large messages (β*S >> α):
```
T_tree / T_ring ≈ log₂(N)*β*S / 2*β*S
                = log₂(N) / 2
```

**For N=1024**:
```
T_tree / T_ring ≈ 10 / 2 = 5x faster
```

## Protocol-Specific Latency

### Simple Protocol

Parameters:
- α_simple ≈ 10-20 μs
- β_simple = 1 / B_link

**Latency**:
```
T_tree_simple = 2*log₂(N)*α_simple + log₂(N)*S/B_link
```

**Example** (N=8 GPUs, S=128 KB, B_link=400 Gbps = 50 GB/s):
```
D = log₂(8) = 3
T_tree_simple = 2*3*15μs + 3*128KB/50GB/s
              = 90μs + 3*2.56μs
              = 97.7μs
```

Compare to Ring:
```
T_ring_simple = 2*7*15μs + 2*128KB/50GB/s * 7/8
              = 210μs + 2*2.56μs*0.875
              = 214.5μs
```

**Tree is 2.2x faster** for this message size.

### LL128 Protocol

Parameters:
- α_LL128 ≈ 7-10 μs
- β_LL128 = 1 / (B_link * 0.65)

**Latency**:
```
T_tree_LL128 = 2*log₂(N)*α_LL128 + log₂(N)*S/(B_link*0.65)
```

**Example** (same setup):
```
T_tree_LL128 = 2*3*8μs + 3*128KB/(50GB/s*0.65)
             = 48μs + 3*3.94μs
             = 59.8μs
```

### LL Protocol

Parameters:
- α_LL ≈ 5-7 μs
- β_LL = 1 / (B_link * 0.25)

**Latency**:
```
T_tree_LL = 2*log₂(N)*α_LL + log₂(N)*S/(B_link*0.25)
```

**Example** (same setup):
```
T_tree_LL = 2*3*6μs + 3*128KB/(50GB/s*0.25)
          = 36μs + 3*10.24μs
          = 66.7μs
```

**Observation**: For small messages, LL protocol with Tree is optimal.

## Bandwidth Analysis

### Data Movement Per GPU

In double binary tree:
- **Worst case** (root in one tree): Send/receive up to S bytes per tree
- **Best case** (leaf in both trees): Send S/2 bytes per tree
- **Average**: Approximately S bytes total

**For large N**:
```
Data per GPU ≈ S bytes (vs 2*S for Ring)
```

This is why Tree can be bandwidth-competitive with Ring.

### Bandwidth Utilization

**Effective Bandwidth**:
```
BW_tree = S / (T_tree / 2)
        = S / [(2*D*α + D*β*S) / 2]
        = 2*S / (2*D*α + D*β*S)
```

For large messages (β*S >> α):
```
BW_tree ≈ 2*S / (D*β*S)
        = 2 / (D*β)
        = 2*B_link / D
        = 2*B_link / log₂(N)
```

**For N=8** (D=3):
```
BW_tree = 2*B_link / 3 ≈ 67% of link bandwidth
```

**For N=1024** (D=10):
```
BW_tree = 2*B_link / 10 = 20% of link bandwidth
```

**Conclusion**: Tree bandwidth decreases with N, but this is offset by logarithmic latency.

## Scalability Analysis

### Latency vs Number of GPUs

**Tree Latency**:
```
T_tree = 2*log₂(N)*α + log₂(N)*β*S
```

**Latency Growth**:
- N=8: D=3 → T ∝ 3
- N=64: D=6 → T ∝ 6 (2x)
- N=512: D=9 → T ∝ 9 (3x)
- N=4096: D=12 → T ∝ 12 (4x)

**Each doubling of N adds one step**, not doubling latency (unlike Ring).

### Tree vs Ring Crossover

**Tree is better when**:
```
T_tree < T_ring
2*log₂(N)*α + log₂(N)*β*S < 2*N*α + 2*β*S
2*log₂(N)*α - 2*N*α < 2*β*S - log₂(N)*β*S
2*α*(log₂(N) - N) < β*S*(2 - log₂(N))
```

For N > 2, log₂(N) < N, so left side is negative.
For N > 4, log₂(N) < 2, so right side is positive.

Rearranging:
```
S < 2*α*(N - log₂(N)) / [β*(2 - log₂(N))]
```

**For N=8**, α=10μs, β = 1/(50 GB/s) = **0.02 ns/byte** (not 20 — see
[ring-algorithm.md](ring-algorithm.md)):
```
S < 2*10 μs*(8 - 3) / [0.02 ns/byte * (2 - 3)]
  < 100 μs / (-0.02 ns/byte)

The denominator is negative, so this closed form yields no positive crossover:
the algebra has been pushed past where the approximation holds.

What is true: Tree wins for small S, Ring for large S. Take the boundary from
measurement or the tuner, not from this expression.
```

**Empirical Crossover** (from NCCL tuning):
- N=8: Tree better for S < 256 KB - 1 MB
- N=64: Tree better for S < 128 KB - 512 KB
- N=512: Tree better for S < 64 KB - 256 KB

**Rule of Thumb**: Tree for S < 1 MB, Ring for S > 1 MB.

## Tree Construction in NCCL

### Tree Building Algorithm

In NCCL the double binary tree is applied **across nodes** (the inter-node dimension):
intra-node the GPUs are connected by the NVLink/PCIe rings/topology, and the two binary
trees connect one representative rank per node. The trees themselves are built by
`ncclGetDtree()` in
[nccl/src/graph/trees.cc](https://github.com/NVIDIA/nccl/blob/master/src/graph/trees.cc)
and wired up by `ncclGetDtree` callers in
[src/graph/connect.cc](https://github.com/NVIDIA/nccl/blob/master/src/graph/connect.cc)
(lines ~148 and ~275).

The construction is **not** an ad-hoc "invert to avoid common nodes"; it is a precise
bit-manipulation scheme (verified in `trees.cc`):

1. **First tree** — `ncclGetBtree()` builds a binary tree rooted at rank 0 that
   *alternates leaves and internal nodes*. Because there are `pow2 − 1`-style rank
   patterns, parent/child relationships are found by locating the first non-zero bit of
   the rank and manipulating bits:
   - parent: `xx01[0] → xx10[0]` (or `xx00[0]` if out of bounds); `xx11[0] → xx10[0]`
   - children: `xx10[0] → xx01[0]` and `xx10[0] → xx11[0]` (or `-1` for leaves)

2. **Second tree** — derived from the first by `ncclGetDtree()`:
   - **even `nranks`**: a **mirror** tree — take the btree of rank `nranks−1−rank` and map
     each node `u` back via `nranks−1−u`.
   - **odd `nranks`**: a **shift by one** — take the btree of `(rank−1+nranks) % nranks`
     and map each node `u` back via `(u+1) % nranks`.

   This construction guarantees the double-tree property: no rank is a non-leaf in both
   trees, so no single rank is a bandwidth bottleneck, and each tree carries half the
   data.

**Illustration from the source** (comment in `trees.cc`, `nranks = 12`, mirror case):

```
First tree (root 0)                    Second tree (mirror)
 0---------------8                      3----------------11
          ______/ \                    / \______
         4         \                  /         7
       /   \        \                /        /   \
     2       6       10            1        5      9
    / \     / \     /  \          / \      / \    / \
   1   3   5   7   9   11        0   2    4   6  8   10
```

**Goal**: Minimize bandwidth bottlenecks — since no rank is internal in both trees, the
root's traffic in one tree is offset by it being a leaf in the other.

### Multi-Node Trees

For M nodes with P GPUs per node:
- **Intra-node subtrees**: Use NVLink (fast)
- **Inter-node connections**: Use network links

**Latency**:
```
T_tree_multi = T_intra_reduce + T_inter_reduce + T_inter_broadcast + T_intra_broadcast
```

Where:
```
T_intra = 2*log₂(P)*α_nvlink + log₂(P)*β_nvlink*S
T_inter = 2*log₂(M)*α_network + log₂(M)*β_network*S
```

**Total**:
```
T_tree_multi ≈ 2*log₂(P)*α_nvlink + 2*log₂(M)*α_network
             + log₂(P)*β_nvlink*S + log₂(M)*β_network*S
```

## Optimization Opportunities

### Tree Depth Balancing

**Unbalanced tree**: Some paths longer than others → higher latency
**Balanced tree**: All paths ≈ log₂(N) → optimal latency

NCCL attempts to balance tree depth automatically.

### Pipeline Depth

Similar to Ring, Tree can pipeline chunks:
- Divide S into multiple chunks
- Start broadcasting chunk 1 while reducing chunk 2

**Effect**: Hides latency for large messages

### Protocol Selection

- **S < 32 KB**: Tree + LL (lowest latency: ~50-100 μs)
- **32 KB < S < 256 KB**: Tree + LL128 (balanced)
- **256 KB < S < 1 MB**: Tree + Simple (transition to Ring)
- **S > 1 MB**: Ring + Simple (better bandwidth)

## Comparison with Ring

| Aspect | Tree (Double Binary) | Ring |
|--------|----------------------|------|
| **Latency** | `2*log₂(N)*α + log₂(N)*β*S` | `2*N*α + 2*β*S` |
| **Complexity** | O(log N) | O(N) |
| **Bandwidth** | `2*B/log₂(N)` (~67% for N=8) | `B` (~100%) |
| **Data per GPU** | ~S bytes | ~2*S bytes |
| **Best for** | Small-medium messages | Large messages |
| **Scalability** | Excellent (thousands) | Good (hundreds) |
| **Steps** | `2*log₂(N)` | `2*(N-1)` |

**Example** (N=64):
- Ring: 126 steps
- Tree: 12 steps (10.5x fewer)

## Summary

| Aspect | Value |
|--------|-------|
| **Latency** | `T = 2*log₂(N)*α + log₂(N)*β*S` |
| **Bandwidth** | `2*B_link / log₂(N)` (decreases with N) |
| **Scalability** | O(log N) - excellent for large clusters |
| **Data per GPU** | ~S bytes sent/received |
| **Steps** | `2*log₂(N)` total steps |
| **Best for** | Small-medium messages (< 1 MB) |
| **Protocols** | LL for < 32 KB, LL128 for 32-256 KB, Simple for 256 KB-1 MB |

**Key Takeaways**:
1. **Tree achieves O(log N) latency** vs Ring's O(N)
2. **Double binary tree maintains full bandwidth** by splitting data
3. **Each GPU sends/receives ~S bytes** (vs 2*S for Ring)
4. **Bandwidth decreases with N** as `2*B/log₂(N)`
5. **Optimal for messages < 1 MB** where latency dominates
6. **Scalability is excellent**: 64 GPUs use 12 steps vs 126 for Ring
7. **Multi-node trees** combine fast intra-node with network inter-node
8. **NCCL automatically selects** Tree vs Ring based on message size

**Related Documentation**:
- [ring-algorithm.md](ring-algorithm.md) - Ring algorithm for bandwidth optimization
- [nvls-tree-algorithm.md](nvls-tree-algorithm.md) - NVLS-accelerated tree
- [nccl-collectives.md](../nccl-collectives.md) - Overview of all algorithms

**References**:
1. [Massively Scale Your Deep Learning Training with NCCL 2.4](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/)
2. [Demystifying NCCL: An In-depth Analysis](https://arxiv.org/html/2507.04786v1)
3. NCCL Source Code: [src/graph/trees.cc](https://github.com/NVIDIA/nccl/blob/master/src/graph/trees.cc) (`ncclGetDtree`/`ncclGetBtree`), [src/graph/connect.cc](https://github.com/NVIDIA/nccl/blob/master/src/graph/connect.cc) (inter-node tree wiring), [src/device/all_reduce.h](https://github.com/NVIDIA/nccl/blob/master/src/device/all_reduce.h) (device reduce/broadcast)
