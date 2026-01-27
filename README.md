# NCCL on AWS EFA Documentation

Comprehensive technical documentation for NCCL (NVIDIA Collective Communications Library) using the OFI plugin with libfabric and AWS EFA (Elastic Fabric Adapter).

## Purpose

This documentation provides in-depth coverage of the complete stack used for GPU-to-GPU communication on AWS EC2 instances with EFA. It's designed for engineers working on optimization, debugging, and understanding the performance characteristics of distributed deep learning workloads.

## Documentation Structure

### Core Concepts

- **[overview.md](overview.md)** - Architecture overview and component interactions
- **[nccl-core.md](nccl-core.md)** - NCCL fundamentals, topology, communicators
- **[nccl-usage.md](nccl-usage.md)** - Communicators, operations, threading, and practical usage patterns
- **[nccl-channels.md](nccl-channels.md)** - NCCL channels: parallelism, establishment, multi-rail support, tuning
- **[nccl-collectives.md](nccl-collectives.md)** - Collective operations, algorithms (Ring/Tree), protocols (Simple/LL/LL128)
- **[nccl-tuner.md](nccl-tuner.md)** - Cost-based algorithm/protocol selection, AWS region-based tuner, channel count tuning
- **[nccl-datapath.md](nccl-datapath.md)** - Complete data path from GPU through network

### Collective Algorithms (Detailed Analysis)

- **[algorithms/ring-algorithm.md](algorithms/ring-algorithm.md)** - Ring algorithm arranges GPUs in a circular topology for bandwidth-optimal collective operations
- **[algorithms/tree-algorithm.md](algorithms/tree-algorithm.md)** - Double binary tree algorithm provides latency-optimized communication for small to medium messages
- **[algorithms/nvls-tree-algorithm.md](algorithms/nvls-tree-algorithm.md)** - NVLS Tree combines tree-based communication with NVSwitch hardware acceleration for reduced latency
- **[algorithms/pat-algorithm.md](algorithms/pat-algorithm.md)** - Parallel Aggregated Trees uses adaptive communication patterns for AllGather and ReduceScatter operations

### EFA Hardware and Protocol

- **[efa-hardware-architecture.md](efa-hardware-architecture.md)** - EFA queue pairs, completion queues, work queue entries, memory layout, hardware capabilities
- **[srd-protocol.md](srd-protocol.md)** - Scalable Reliable Datagram protocol: multipath routing, congestion control, reliability mechanisms
- **[kernel-efa-driver.md](kernel-efa-driver.md)** - Linux kernel EFA driver internals

### Transport Layer

- **[ofi-plugin.md](ofi-plugin.md)** - OFI NCCL plugin architecture and implementation
- **[libfabric-overview.md](libfabric-overview.md)** - Libfabric API, capabilities, data transfer operations
- **[efa-provider.md](efa-provider.md)** - EFA provider specifics, eager/rendezvous protocols
- **[rdma-core-and-verbs.md](rdma-core-and-verbs.md)** - rdma-core userspace libraries, libibverbs API, kernel uverbs interface, EFA provider implementation

### System Level

- **[threading-model.md](threading-model.md)** - Threading across the stack, proxy threads, concurrency
- **[rdma-memreg.md](rdma-memreg.md)** - RDMA operations, memory registration, caching strategies

### Performance

- **[optimizations.md](optimizations.md)** - Kernel bypass, caching, tuning parameters, best practices
- **[optimization-opportunities.md](optimization-opportunities.md)** - Detailed analysis of optimization opportunities with priorities

### Deep Dives

- **[lkey-rkey-explained.md](lkey-rkey-explained.md)** - Understanding local and remote keys in RDMA
- **[ofi-plugin-protocols.md](ofi-plugin-protocols.md)** - Connection establishment and send/recv protocols
- **[topology-and-binding.md](topology-and-binding.md)** - Topology detection and interface binding mechanisms
- **[freelist-allocator.md](freelist-allocator.md)** - Freelist allocator for high-performance object pooling (estimated 20-100x faster than malloc)
- **[mr-cache-implementation.md](mr-cache-implementation.md)** - Memory registration cache implementation details (approximately 25x speedup)

### GPU and Accelerator Memory

- **[cuda-memory.md](cuda-memory.md)** - NVIDIA CUDA memory: dynamic loading, cross-version compatibility, GPUDirect, GDRCopy
- **[dmabuf-gpu-memory.md](dmabuf-gpu-memory.md)** - DMA-BUF framework for vendor-neutral GPU memory registration
- **[neuron-memory.md](neuron-memory.md)** - AWS Trainium/Inferentia with P2P registration (not dmabuf)
- **[rocm-memory.md](rocm-memory.md)** - AMD ROCm with HIP API (CUDA-compatible)

## Quick Start

### Understanding the Stack

Start with these documents in order:

1. [overview.md](overview.md) - Get the big picture
2. [nccl-core.md](nccl-core.md) - Understand NCCL basics
3. [nccl-datapath.md](nccl-datapath.md) - Follow a message through the system
4. [optimizations.md](optimizations.md) - Apply optimizations

### For Specific Topics

- **Performance tuning** → [optimizations.md](optimizations.md), [optimization-opportunities.md](optimization-opportunities.md)
- **Debugging network issues** → [ofi-plugin.md](ofi-plugin.md), [efa-provider.md](efa-provider.md)
- **Communicator usage patterns** → [nccl-usage.md](nccl-usage.md)
- **Channel configuration and tuning** → [nccl-channels.md](nccl-channels.md)
- **Algorithm selection** → [nccl-collectives.md](nccl-collectives.md), [algorithms/](algorithms/) for detailed analysis
- **Collective algorithms deep dive** → [algorithms/ring-algorithm.md](algorithms/ring-algorithm.md), [algorithms/tree-algorithm.md](algorithms/tree-algorithm.md), [algorithms/nvls-tree-algorithm.md](algorithms/nvls-tree-algorithm.md), [algorithms/pat-algorithm.md](algorithms/pat-algorithm.md)
- **Memory registration problems** → [rdma-memreg.md](rdma-memreg.md), [mr-cache-implementation.md](mr-cache-implementation.md)
- **Threading issues** → [threading-model.md](threading-model.md)
- **Connection protocols** → [ofi-plugin-protocols.md](ofi-plugin-protocols.md)
- **Topology and multi-NIC** → [topology-and-binding.md](topology-and-binding.md), [nccl-channels.md](nccl-channels.md)
- **Understanding RDMA keys** → [lkey-rkey-explained.md](lkey-rkey-explained.md)
- **Memory allocators & caching** → [freelist-allocator.md](freelist-allocator.md), [mr-cache-implementation.md](mr-cache-implementation.md)
- **GPU memory registration** → [cuda-memory.md](cuda-memory.md), [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md), [neuron-memory.md](neuron-memory.md), [rocm-memory.md](rocm-memory.md)
- **Kernel driver details** → [kernel-efa-driver.md](kernel-efa-driver.md)
- **Userspace RDMA API** → [rdma-core-and-verbs.md](rdma-core-and-verbs.md)

## Key Concepts

### Architecture Stack

```
Application (PyTorch, TensorFlow)
    ↓
NCCL Library (Collectives, Algorithms)
    ↓
OFI NCCL Plugin (Network Transport)
    ↓
Libfabric (Fabric Abstraction)
    ↓
EFA Provider (Hardware-Specific)
    ↓
rdma-core (libibverbs)
    ↓
EFA Kernel Driver (efa.ko)
    ↓
EFA Hardware (p4d: 4×100G, p5+: 32×100G EFAv3)
```

### Data Flow

1. **Application** calls NCCL collective (e.g., AllReduce)
2. **NCCL** breaks into algorithm steps (Ring/Tree)
3. **Proxy threads** handle network I/O
4. **OFI plugin** translates to libfabric operations
5. **Libfabric** posts to EFA provider
6. **EFA provider** builds WQEs, rings doorbells
7. **EFA driver** DMAs from GPU memory
8. **EFA hardware** transmits over network with SRD reliability
9. **Remote side** receives and completes in reverse

### Critical Performance Factors

1. **Memory Registration Caching** - Estimated 100-500x speedup for repeated transfers
2. **Algorithm Selection** - Ring for large messages (>1 MB), Tree for medium (100 KB-1 MB), PAT for small-medium at scale (32 KB-1 MB), NVLS Tree for intra-node with NVSwitch. See [algorithms/](algorithms/) for complete mathematical analysis
3. **Protocol Selection** - LL for latency (<32 KB), LL128 for balanced (32 KB-1 MB), Simple for bandwidth (>1 MB)
4. **Multi-rail** - Utilize all EFA adapters (4-8 per instance)
5. **CPU Affinity** - Pin proxy threads to avoid NUMA remote access
6. **Kernel Bypass** - Direct user-space access to hardware

## Additional Resources

### Official Documentation

- NCCL: https://docs.nvidia.com/deeplearning/nccl/
- OFI Plugin: https://github.com/aws/aws-ofi-nccl
- Libfabric: https://ofiwg.github.io/libfabric/
- AWS EFA: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html

### Repositories

- NCCL: https://github.com/NVIDIA/nccl
- aws-ofi-nccl: https://github.com/aws/aws-ofi-nccl
- libfabric: https://github.com/ofiwg/libfabric
- nccl-tests: https://github.com/NVIDIA/nccl-tests

## Contributing

This documentation is focused on understanding the existing implementation for optimization purposes. For improvements to the actual software components:

- NCCL issues/PRs: https://github.com/NVIDIA/nccl
- OFI plugin issues/PRs: https://github.com/aws/aws-ofi-nccl
- Libfabric issues/PRs: https://github.com/ofiwg/libfabric

## Changelog

### 2026-01-20

#### Documentation Organization and Improvements
- **Reorganized rdma-core documentation** - Moved [rdma-core-and-verbs.md](rdma-core-and-verbs.md) from Deep Dives to Transport Layer section in README, better reflecting its role as the critical layer between libfabric and kernel driver
- **Added rdma-core to stack diagrams** - Updated [overview.md](overview.md) Architecture Stack to include rdma-core (libibverbs) layer with userspace verbs library, EFA provider plugin, and zero-copy queue operations
- **Enhanced data path documentation** - Updated [nccl-datapath.md](nccl-datapath.md) to include rdma-core layer showing ibv_post_send/recv operations (~100-200ns), memory-mapped queue writes, and doorbell rings with zero syscalls
- **Component roles expansion** - Added rdma-core component description in overview.md explaining its purpose as userspace RDMA verbs library with hardware-agnostic API and provider plugins
- **Data flow clarification** - Expanded data flow steps to explicitly show rdma-core posting work requests via verbs API and EFA kernel driver managing memory registration via uverbs interface
- **Added DRAFT warnings** - Added prominent verification warnings to all algorithm documentation files (Ring, Tree, NVLS Tree, PAT) indicating theoretical derivations need independent validation
- **Simplified algorithm descriptions** - Changed algorithm descriptions in README to simple one-sentence summaries without formulas or performance claims

### 2026-01-19

#### Algorithm Documentation
- Added **[algorithms/ring-algorithm.md](algorithms/ring-algorithm.md)** with complete mathematical latency derivation (`T = 2(N-1)×α + 2β×S×(N-1)/N`), bandwidth analysis proving ~100% utilization, O(N) scaling analysis, protocol-specific equations (Simple/LL128/LL), and concrete p5 instance examples
- Added **[algorithms/tree-algorithm.md](algorithms/tree-algorithm.md)** with double binary tree algorithm, O(log N) latency derivation (`T = 2×log₂(N)×α + log₂(N)×β×S`), ~67% bandwidth utilization analysis for N=8, tree topology construction, and comparison with Ring showing log vs linear scaling
- Added **[algorithms/nvls-tree-algorithm.md](algorithms/nvls-tree-algorithm.md)** documenting NVLink SHARP hardware acceleration, NVSwitch multicast mechanisms, in-network reduction, intra-node latency (`T = 2×α_nvls + 2×β_nvlink×S`), multi-node extensions, 2-10× speedup for small messages, and p5/p5en specific performance on Hopper/Blackwell GPUs
- Added **[algorithms/pat-algorithm.md](algorithms/pat-algorithm.md)** for Parallel Aggregated Trees (NCCL 2.23+), adaptive logarithmic-to-linear communication strategy, AllGather derivation (`T = log₂(N)×α + β×S×(1-1/N)`), achieving O(log N) latency with ~100% bandwidth, optimal for 32 KB-1 MB messages at scale, and LLM training use cases
- All algorithm files include complete α-β model derivations, step-by-step walkthroughs, protocol analysis, crossover point calculations, and AWS instance benchmarks

#### Hardware and Protocol Documentation
- Added comprehensive **[efa-hardware-architecture.md](efa-hardware-architecture.md)** documenting EFA queue pairs, completion queues, work queue entries, memory layout, and programming model
- Added **[srd-protocol.md](srd-protocol.md)** documenting AWS's Scalable Reliable Datagram protocol with multipath routing (64 paths), hardware-based congestion control, and reliability mechanisms
- Added **[nccl-tuner.md](nccl-tuner.md)** documenting NCCL's cost-based tuner, algorithm/protocol selection, channel count tuning, and AWS OFI region-based tuner with geometric selection

#### Improvements
- Updated all numerical performance claims to be properly qualified as estimates or approximations
- Removed file numbering from all documentation files to ease maintenance
- Removed Troubleshooting section from README
- Added algorithm documentation section to Quick Start guide

### 2026-01-17
- Initial comprehensive documentation
- Covered full stack: NCCL → EFA hardware
- Included optimization guide

---

**Audience**: Engineers optimizing distributed deep learning workloads on AWS EC2 with EFA
**Focus**: Technical depth, performance optimization, debugging
**Scope**: NCCL + OFI plugin + libfabric + EFA driver + EFA hardware
