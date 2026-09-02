# NVLS Tree AllReduce Algorithm

> ## ⚠️ DRAFT - NEEDS VERIFICATION ⚠️
>
> **This document contains theoretical derivations and performance analysis that have NOT been fully verified.**
>
> All latency formulas, bandwidth calculations, and performance claims should be independently validated before being used for system design or optimization decisions. The mathematical derivations may contain errors or oversimplifications.
>
> Please treat all content as preliminary and subject to revision.

## Overview

The **NVLS Tree algorithm** combines NCCL's tree-based approach with **NVLink SHARP** (NVLS) hardware acceleration available in NVSwitch systems. It leverages in-network multicast and reduction capabilities to achieve dramatically lower latency while maintaining high bandwidth.

**Best For**: Medium to large messages on NVSwitch-enabled systems (Hopper H100/H200, Blackwell GPUs)

**Key Properties**:
- Latency: Significantly reduced vs standard Tree through hardware acceleration
- Bandwidth utilization: ~90-100% with NVSwitch multicast
- Hardware requirement: NVSwitch with NVLink4+ (Hopper/Blackwell)
- Intra-node: Hardware-accelerated via NVLS
- Inter-node: Tree-based fan-out (not CollNet)

**Source**: [NCCL NVLS transport: src/transport/nvls.cc](https://github.com/NVIDIA/nccl)

## NVLink SHARP (NVLS) Technology

### Hardware Architecture

**NVSwitch** (3rd generation with NVLink4):
- Integrated SHARP engines for in-network reduction
- Multicast capability (one-to-many communication)
- Direct GPU-to-GPU paths without CPU involvement
- Available in DGX H100/H200, GB200 NVL systems

**NVIDIA SHARP** (Scalable Hierarchical Aggregation and Reduction Protocol):
- In-network reduction operations (SUM, MIN, MAX, etc.)
- Multicast distribution (single send → multiple receives)
- Hardware-offloaded collective operations

### NVSwitch Multicast Mechanism

**Traditional Communication** (without multicast):
```
GPU 0 → GPU 1 (1 send)
GPU 0 → GPU 2 (1 send)
GPU 0 → GPU 3 (1 send)
...
GPU 0 → GPU 7 (1 send)

Total: N-1 sequential sends = O(N) latency
```

**NVSwitch Multicast**:
```
GPU 0 → [GPU 1, GPU 2, GPU 3, ..., GPU 7] (1 multicast)

Total: 1 multicast operation = O(1) latency!
```

**Latency Reduction**: From O(N) individual sends to O(1) multicast → estimated 3-9x faster for small messages

### In-Network Reduction

**Traditional Reduction**:
```
1. GPU 1 sends A₁ to GPU 0
2. GPU 0 computes: A₀ + A₁
3. GPU 2 sends A₂ to GPU 0
4. GPU 0 computes: (A₀ + A₁) + A₂
...
Total: N-1 steps, sequential reduction
```

**NVLS In-Network Reduction**:
```
1. All GPUs send to NVSwitch simultaneously
2. NVSwitch reduction engine computes: A₀ + A₁ + ... + A₇
3. Result available in ~2 inter-GPU synchronizations

Total: ~O(1) reduction time
```

## NVLS vs NVLS Tree

NCCL provides two NVLS-accelerated algorithms:

### NVLS (Plain)

**Intra-node**: Uses NVLS multicast/reduction
**Inter-node**: Uses CollNet (switch-based aggregation)

**Best for**: Systems with CollNet-enabled network switches

### NVLS Tree

**Intra-node**: Uses NVLS multicast/reduction
**Inter-node**: Uses standard Tree algorithm

**Best for**: AWS EC2 (p5/p5en), cloud environments without CollNet switches

**Focus of this document**: NVLS Tree (relevant for AWS EFA)

## Algorithm Description

### Single-Node NVLS Tree (Intra-Node Only)

For N GPUs in a single NVSwitch domain:

**Phase 1: NVLS Reduce**:
1. All GPUs write data to shared NVSwitch memory space
2. NVLS reduction engine computes aggregate (hardware-accelerated)
3. Result available to all GPUs

**Phase 2: NVLS Multicast**:
1. Root GPU (or NVLS engine) multicasts result
2. All GPUs receive simultaneously via multicast

**Steps**: ~2-3 inter-GPU synchronizations (vs 2*log₂(N) for standard Tree)

**For N=8 GPUs**:
- Standard Tree: 2*3 = 6 steps
- NVLS Tree: ~2-3 steps
- **Speedup**: ~2-3x

### Multi-Node NVLS Tree

For M nodes, each with P GPUs:

**Phase 1: Intra-Node Reduce** (NVLS):
- Each node reduces its P GPUs' data using NVLS
- Result: M partial sums (one per node)
- Time: ~O(1) via NVLS hardware

**Phase 2: Inter-Node Reduce** (Tree):
- M nodes arranged in binary tree
- Standard tree reduction across nodes
- Time: O(log M) steps

**Phase 3: Inter-Node Broadcast** (Tree):
- Tree broadcast of final result to all nodes
- Time: O(log M) steps

**Phase 4: Intra-Node Broadcast** (NVLS):
- Within each node, multicast result to all P GPUs
- Time: ~O(1) via NVLS multicast

**Total Steps**: 2 (intra-NVLS) + 2*log₂(M) (inter-Tree)

## Mathematical Derivation of Latency

### Latency Model Parameters

**α-β Model with NVLS**:
- **α_nvls**: NVLS operation latency (μs) - multicast/reduce overhead
- **β_nvlink**: NVLink per-byte time = 1/B_nvlink
- **α_network**: Network latency for inter-node (μs)
- **β_network**: Network per-byte time = 1/B_network
- **S**: Total message size (bytes)
- **P**: GPUs per node
- **M**: Number of nodes
- **γ**: Computation time (negligible with NVLS hardware)

### Single-Node NVLS Tree Latency

**Intra-Node Reduce via NVLS**:
```
T_nvls_reduce = α_nvls + β_nvlink*S
```

Where:
- α_nvls ≈ 1-3 μs (NVLS operation overhead)
- β_nvlink*S = time to transfer S bytes over NVLink

**Intra-Node Broadcast via NVLS Multicast**:
```
T_nvls_broadcast = α_nvls + β_nvlink*S
```

**Total Single-Node**:
```
T_nvls_tree_single = 2*α_nvls + 2*β_nvlink*S
```

**Comparison with Standard Tree** (single node, P=8 GPUs):
```
T_tree_standard = 2*log₂(8)*α_tree + log₂(8)*β_nvlink*S
                = 6*α_tree + 3*β_nvlink*S

T_nvls_tree = 2*α_nvls + 2*β_nvlink*S
```

**Ratio**:
```
T_nvls / T_standard = (2*α_nvls + 2*β_nvlink*S) / (6*α_tree + 3*β_nvlink*S)
```

For α_nvls ≈ 2μs, α_tree ≈ 10μs:
```
T_nvls / T_standard ≈ (4μs + 2*β*S) / (60μs + 3*β*S)
```

For small S (S=128 KB, β*S ≈ 1μs on NVLink):
```
T_nvls / T_standard ≈ 6μs / 63μs ≈ 0.1 (10x faster!)
```

For large S (S=128 MB, β*S ≈ 1000μs):
```
T_nvls / T_standard ≈ 2004μs / 3060μs ≈ 0.65 (1.5x faster)
```

**Key Insight**: NVLS Tree provides massive speedup for small-medium messages.

### Multi-Node NVLS Tree Latency

**Phase 1: Intra-Node Reduce** (all nodes in parallel):
```
T_intra_reduce = α_nvls + β_nvlink*S
```

**Phase 2: Inter-Node Reduce** (tree across M nodes):
```
T_inter_reduce = log₂(M)*α_network + log₂(M)*β_network*S
```

**Phase 3: Inter-Node Broadcast**:
```
T_inter_broadcast = log₂(M)*α_network + log₂(M)*β_network*S
```

**Phase 4: Intra-Node Broadcast** (all nodes in parallel):
```
T_intra_broadcast = α_nvls + β_nvlink*S
```

**Total Multi-Node NVLS Tree**:
```
T_nvls_tree_multi = 2*α_nvls + 2*β_nvlink*S
                  + 2*log₂(M)*α_network + 2*log₂(M)*β_network*S
```

**Simplified**:
```
T_nvls_tree = 2*α_nvls + 2*log₂(M)*α_network
            + 2*β_nvlink*S + 2*log₂(M)*β_network*S
```

### Comparison with Standard Tree (Multi-Node)

**Standard Tree Multi-Node** (P GPUs/node, M nodes):
```
T_tree_multi = 2*log₂(P)*α_nvlink + 2*log₂(M)*α_network
             + log₂(P)*β_nvlink*S + 2*log₂(M)*β_network*S
```

**NVLS Tree Multi-Node**:
```
T_nvls_tree = 2*α_nvls + 2*log₂(M)*α_network
            + 2*β_nvlink*S + 2*log₂(M)*β_network*S
```

**Difference (NVLS saves)**:
```
ΔT = [2*log₂(P)*α_nvlink + log₂(P)*β_nvlink*S] - [2*α_nvls + 2*β_nvlink*S]
   ≈ 2*log₂(P)*α_nvlink - 2*α_nvls + (log₂(P) - 2)*β_nvlink*S
```

For P=8, α_nvlink=5μs, α_nvls=2μs:
```
ΔT ≈ 2*3*5μs - 2*2μs + (3-2)*β_nvlink*S
   = 30μs - 4μs + β_nvlink*S
   = 26μs + β_nvlink*S
```

**NVLS Tree saves 26μs + (intra-node transfer time)** → significant for small messages.

## Protocol-Specific Latency

### Simple Protocol (NVLS Tree)

Parameters:
- α_nvls ≈ 2-3 μs
- α_network ≈ 15-20 μs
- B_nvlink ≈ 600 GB/s (NVLink4)
- B_network ≈ 400 Gbps = 50 GB/s (EFA on p5)

**Latency** (M=4 nodes, P=8 GPUs/node, S=128 MB):
```
T_nvls_tree = 2*2μs + 2*log₂(4)*15μs + 2*128MB/600GB/s + 2*log₂(4)*128MB/50GB/s
            = 4μs + 2*2*15μs + 2*213μs + 4*2.56ms
            = 4μs + 60μs + 426μs + 10.24ms
            ≈ 10.73 ms
```

Compare to Standard Tree:
```
T_tree_multi = 2*3*5μs + 2*2*15μs + 3*213μs + 4*2.56ms
             = 30μs + 60μs + 639μs + 10.24ms
             ≈ 10.97 ms
```

**NVLS Tree saves ~240μs** (~2.2% faster) for this configuration.

**Observation**: Savings more significant for smaller messages where intra-node dominates.

### LL128 Protocol (NVLS Tree)

Parameters:
- α_nvls ≈ 1-2 μs (lower overhead for LL128)
- Efficiency ≈ 0.65

**Latency** (S=4 MB):
```
T_nvls_tree = 2*1.5μs + 2*2*15μs + 2*4MB/(600GB/s) + 4*4MB/(50GB/s*0.65)
            = 3μs + 60μs + 2*6.67μs + 4*123μs
            = 3μs + 60μs + 13.3μs + 492μs
            ≈ 568μs
```

### LL Protocol (NVLS Tree)

For very small messages, NVLS provides maximum benefit:

**Latency** (S=64 KB):
```
T_nvls_tree = 2*2μs + 4*15μs + 2*64KB/600GB/s + 4*64KB/(50GB/s*0.25)
            = 4μs + 60μs + 0.21μs + 20.5μs
            ≈ 85μs
```

Compare to Standard Tree LL:
```
T_tree_ll = 6*6μs + 4*15μs + 3*0.21μs + 4*20.5μs
          = 36μs + 60μs + 0.63μs + 82μs
          ≈ 179μs
```

**NVLS Tree is 2.1x faster** for small messages with LL protocol.

## Bandwidth Analysis

### Effective Bandwidth

**NVLS Tree Bandwidth**:
```
BW_nvls = S / T_nvls_tree
```

For large messages dominated by network:
```
BW_nvls ≈ S / (2*log₂(M)*β_network*S)
        = 1 / (2*log₂(M)*β_network)
        = B_network / (2*log₂(M))
```

**For M=4 nodes**:
```
BW_nvls = 50 GB/s / (2*2) = 12.5 GB/s effective
```

**Observation**: Inter-node network becomes bottleneck for large messages.

### Intra-Node Bandwidth

Within a node, NVLS achieves near-peak NVLink bandwidth:
```
BW_intra = S / (2*β_nvlink*S) = 1 / (2*β_nvlink) = B_nvlink/2
```

For B_nvlink = 600 GB/s:
```
BW_intra = 300 GB/s (50% of NVLink bandwidth)
```

**Note**: 50% factor from reduce + broadcast phases.

## Scalability Analysis

### Intra-Node Scalability

**NVLS Latency** (single node):
```
T_nvls = 2*α_nvls + 2*β_nvlink*S
```

**Independent of P** (number of GPUs in node)!

**Standard Tree Latency**:
```
T_tree = 2*log₂(P)*α + log₂(P)*β*S
```

**For P=8**: NVLS saves 2*3*α + (3-2)*β*S ≈ 6*α + β*S
**For P=16**: NVLS saves 2*4*α + (4-2)*β*S ≈ 8*α + 2*β*S

**Benefit increases with P**.

### Multi-Node Scalability

**NVLS Tree Latency**:
```
T_nvls_tree = 2*α_nvls + 2*log₂(M)*α_network + ...
```

**Scales as O(log M)** for inter-node communication, same as standard Tree.

**Advantage**: Eliminates intra-node log₂(P) term → better for large nodes.

## Multi-GPU System Performance (p5/p5en)

### p5.48xlarge (8× H100, NVSwitch)

**Configuration**:
- P = 8 GPUs per node
- NVLink4: 900 GB/s bidirectional per GPU
- NVSwitch: Full connectivity with NVLS support

**Single-Node AllReduce** (S=128 MB):
```
T_nvls = 2*2μs + 2*128MB/600GB/s
       = 4μs + 426μs
       ≈ 430μs
```

**Standard Tree** (for comparison):
```
T_tree = 6*10μs + 3*426μs
       = 60μs + 1278μs
       ≈ 1338μs
```

**NVLS is 3.1x faster** for this configuration.

### p5en.48xlarge (8× H200, NVSwitch, EFAv3)

**Configuration**:
- P = 8 GPUs per node
- NVLink4: 900 GB/s
- EFAv3: 3200 Gbps = 400 GB/s total (50 GB/s per GPU)
- 35% lower network latency vs p5

**Multi-Node AllReduce** (M=16 nodes, S=1 GB):
```
T_nvls_tree = 2*2μs + 2*log₂(16)*12μs + 2*1GB/600GB/s + 2*4*1GB/50GB/s
            = 4μs + 2*4*12μs + 3.3ms + 8*20ms
            = 4μs + 96μs + 3.3ms + 160ms
            ≈ 163.4 ms
```

**Bandwidth**:
```
BW = 1GB / 163.4ms ≈ 6.1 GB/s
```

## Hardware Requirements and Limitations

### NVSwitch Requirements

**Minimum**: NVSwitch generation 3 with NVLink4
**GPUs**: Hopper (H100/H200), Blackwell (B100/B200/B300)

**Not available on**:
- Ampere GPUs (A100) - lacks NVLS support
- Volta GPUs (V100)
- Systems without NVSwitch (PCIe-only)

### AWS Instance Availability

**Supported**:
- p5.48xlarge (H100)
- p5e.48xlarge (H100)
- p5en.48xlarge (H200)
- p6 (Blackwell B200) and p6-B300 — both are now recognized platforms in the AWS OFI tuner (`NCCL_OFI_TUNER_P6`, `NCCL_OFI_TUNER_P6_B300`)

**Not supported**:
- p4d/p4de (A100 - no NVSwitch)
- p3 (V100)

### Enable NVLS in NCCL

```bash
# Enable NVLS algorithms (default: auto-detected)
export NCCL_ALGO=NVLS_TREE

# Or let tuner decide
export NCCL_TUNER_PLUGIN=/path/to/libnccl-ofi-tuner.so
```

NCCL automatically detects NVLS capability and uses it when beneficial.

## Optimization Opportunities

### Message Size Tuning

**Optimal NVLS Tree Usage**:
- S < 32 KB: Tree + LL (NVLS minimal benefit)
- 32 KB < S < 128 MB: NVLS Tree + LL128/Simple (sweet spot)
- S > 128 MB: Ring + Simple (bandwidth dominates)

**AWS OFI Tuner** automatically selects NVLS Tree for appropriate sizes on p5/p5en.

### Channel Count

NVLS operations can use multiple channels for parallelism:
- Typical: 8-16 channels
- More channels → better overlap of intra-node NVLS with inter-node Tree

### Hierarchical Tuning

For very large clusters:
- Use NVLS Tree for intra-rack (fast NVSwitch)
- Use Ring for inter-rack (slower network)

## Comparison: NVLS Tree vs Alternatives

| Algorithm | Latency (Intra) | Latency (Inter) | Bandwidth | Hardware Req |
|-----------|----------------|-----------------|-----------|--------------|
| **NVLS Tree** | ~O(1) | O(log M) | High | NVSwitch+NVLS |
| **Standard Tree** | O(log P) | O(log M) | Medium | Any |
| **Ring** | O(P) | O(M) | Highest | Any |
| **NVLS** (CollNet) | ~O(1) | ~O(1) | Highest | NVSwitch+CollNet |

**When to Use NVLS Tree**:
- Medium messages (1-128 MB) on p5/p5en
- Multi-node clusters without CollNet switches
- Latency-sensitive workloads with H100/H200

**When to Use Ring**:
- Very large messages (> 128 MB)
- Maximum bandwidth needed
- Older hardware (A100, V100)

## Summary

| Aspect | Value |
|--------|-------|
| **Latency (Single-Node)** | `T = 2*α_nvls + 2*β_nvlink*S` |
| **Latency (Multi-Node)** | `T = 2*α_nvls + 2*log₂(M)*α_net + 2*β_nvlink*S + 2*log₂(M)*β_net*S` |
| **Intra-Node Speedup** | Estimated 2-10x vs standard Tree (message size dependent) |
| **Hardware** | NVSwitch with NVLink4+ (Hopper/Blackwell) |
| **Scalability** | O(1) intra-node, O(log M) inter-node |
| **Best for** | Medium messages (32 KB - 128 MB) on p5/p5en |
| **AWS Instances** | p5, p5e, p5en, p6 (B200), p6-B300 |

**Key Takeaways**:
1. **NVLS provides hardware-accelerated multicast/reduction** via NVSwitch
2. **Intra-node latency is ~O(1)** vs O(log P) for standard Tree
3. **Estimated 2-10x faster for small-medium messages** on NVSwitch systems
4. **Multi-node uses Tree for inter-node** (not CollNet) → suitable for AWS
5. **Requires Hopper+ GPUs** (H100/H200/B100/B200)
6. **AWS OFI tuner automatically selects** NVLS Tree for appropriate message sizes
7. **Bandwidth competitive with Ring** for medium messages on p5/p5en
8. **Latency improvement most significant** for 1-128 MB range

**Related Documentation**:
- [tree-algorithm.md](tree-algorithm.md) - Standard binary tree algorithm
- [ring-algorithm.md](ring-algorithm.md) - Ring algorithm for comparison
- [nccl-tuner.md](../nccl-tuner.md) - NVLS Tree selection in AWS OFI tuner

**References**:
1. [NVIDIA NCCL Multi-Node Tuning Guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/nccl.html)
2. [3x Faster AllReduce with NVSwitch and TensorRT-LLM MultiShot](https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/)
3. [Question about NCCL_ALGO_NVLS algorithm](https://github.com/NVIDIA/nccl/issues/807)
