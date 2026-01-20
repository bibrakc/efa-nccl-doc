# NCCL, OFI Plugin, and Libfabric on AWS EFA - Overview

## Introduction

This documentation covers the integration of NVIDIA NCCL (NVIDIA Collective Communications Library) with AWS EFA (Elastic Fabric Adapter) through the OFI (OpenFabrics Interfaces) plugin and libfabric.

## Architecture Stack

```
┌─────────────────────────────────────────────┐
│         Application (PyTorch, etc.)         │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│              NCCL Library                   │
│  - Collectives (AllReduce, AllGather, etc.) │
│  - Algorithms (Ring, Tree, etc.)            │
│  - Transport abstraction                    │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│          OFI NCCL Plugin                    │
│  - NCCL Net Plugin Interface                │
│  - Libfabric integration                    │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│            Libfabric                        │
│  - Provider-agnostic API                    │
│  - EFA provider implementation              │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│         rdma-core (libibverbs)              │
│  - Userspace verbs library                  │
│  - EFA provider plugin                      │
│  - Zero-copy queue operations               │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│          EFA Kernel Driver                  │
│  - Device control (uverbs interface)        │
│  - Memory registration                      │
│  - Hardware resource management             │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│     AWS EFA Hardware (NIC)                  │
└─────────────────────────────────────────────┘
```

## Component Roles

### NCCL
- **Purpose**: High-performance multi-GPU/multi-node collective communication
- **Key Features**: Optimized collective algorithms, topology awareness, transport abstraction
- **Focus Areas**: Collective operations, data path, algorithms

### OFI NCCL Plugin
- **Purpose**: Bridge between NCCL and libfabric
- **Key Features**: Implements NCCL's network plugin interface using libfabric APIs
- **Focus Areas**: Plugin architecture, transport implementation, buffer management

### Libfabric
- **Purpose**: Vendor-neutral fabric communication interface
- **Key Features**: Provider abstraction, RDMA support, various transport modes
- **Focus Areas**: EFA provider, threading model, memory registration, passthrough modes

### rdma-core (libibverbs)
- **Purpose**: Userspace RDMA verbs library
- **Key Features**: Hardware-agnostic verbs API, provider plugins, zero-copy operations
- **Focus Areas**: ibv_post_send/recv, ibv_poll_cq, memory-mapped queues, EFA provider

### AWS EFA
- **Purpose**: High-performance network interface for EC2 instances
- **Key Features**: RDMA-like performance, OS bypass, scalable
- **Focus Areas**: Driver architecture, capabilities, performance characteristics

## Data Flow

1. **Application** issues collective operation (e.g., AllReduce)
2. **NCCL** breaks down collective into point-to-point operations based on algorithm
3. **OFI Plugin** translates NCCL network calls to libfabric operations
4. **Libfabric EFA Provider** converts to libfabric-agnostic operations
5. **rdma-core (libibverbs)** posts work requests via verbs API (ibv_post_send/recv)
6. **EFA Kernel Driver** manages memory registration and device control via uverbs
7. **EFA Hardware** performs zero-copy DMA and transmits over network fabric

## Key Concepts

### Collectives
Operations that involve multiple processes/GPUs working together:
- AllReduce, AllGather, Broadcast, Reduce, ReduceScatter, etc.

### Algorithms
NCCL implements multiple algorithms per collective:
- Ring, Tree, CollNet, NVLS

### Protocols
Communication patterns within NCCL:
- Simple, LL (Low Latency), LL128

### RDMA (Remote Direct Memory Access)
- Direct memory access from one node's memory to another
- Bypasses CPU and OS kernel for performance
- Critical for low-latency, high-bandwidth operations

### Memory Registration
- Process of making memory accessible to RDMA hardware
- Pins physical pages, creates mapping for NIC
- Performance vs. flexibility tradeoff

## Document Organization

- `nccl-core.md` - NCCL core concepts and operations
- `nccl-collectives.md` - NCCL collective operations and algorithms
- `nccl-datapath.md` - NCCL network data path and protocols
- `ofi-plugin.md` - OFI NCCL plugin architecture
- `libfabric-overview.md` - Libfabric architecture and APIs
- `efa-provider.md` - EFA provider specifics in libfabric
- `rdma-core-and-verbs.md` - rdma-core userspace library and libibverbs API
- `threading-model.md` - Threading and concurrency
- `rdma-memreg.md` - RDMA operations and memory registration
- `efa-driver.md` - EFA kernel driver architecture
- `optimizations.md` - Passthrough modes and optimizations

## Performance Considerations

Key areas affecting performance:
1. **Algorithm selection** - Ring vs Tree vs others
2. **Protocol choice** - Simple vs LL vs LL128
3. **Memory registration strategy** - Cache vs on-demand
4. **Threading model** - Progress threads, locking
5. **EFA-specific features** - SRD (Scalable Reliable Datagram), write operations
6. **Passthrough modes** - Reducing software overhead

## References

- NCCL: https://github.com/NVIDIA/nccl
- OFI Plugin: https://github.com/aws/aws-ofi-nccl
- Libfabric: https://github.com/ofiwg/libfabric
- EFA: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html
