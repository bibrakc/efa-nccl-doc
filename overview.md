# NCCL, OFI Plugin, and Libfabric on AWS EFA - Overview

## Introduction

This documentation covers the integration of NVIDIA NCCL (NVIDIA Collective Communications Library) with AWS EFA (Elastic Fabric Adapter) through the OFI (OpenFabrics Interfaces) plugin and libfabric.

**Current Versions (as of June 2026):**
- NCCL: 2.30.x (v2.30.4-1)
- OFI Plugin (aws-ofi-nccl): C++ codebase, latest release v1.20.0, master at [f3dd9cd](https://github.com/aws/aws-ofi-nccl/commit/f3dd9cd)
- Libfabric: 2.6.x series with EFA provider
- rdma-core: v63+
- EFA Hardware: v2 (P5) and v3 (P5en, P6)

## Architecture Stack

The stack has two data paths: the standard **NCCL Net path** for collective
operations (AllReduce, AllGather, etc.) and the **GIN path** for one-sided
put operations used by workloads like DeepEP (Mixture-of-Experts dispatch).

```
┌─────────────────────────────────────────────────────────┐
│         Application (PyTorch, DeepSpeed, etc.)          │
└─────────────────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────────────────┐
│              NCCL Library (2.30.x)                      │
│  - Collectives (AllReduce, AllGather, etc.)             │
│  - Algorithms (Ring, Tree, PAT, NVLS)                   │
│  - GIN Device API (put, putSignal for GPU-initiated     │
│    one-sided operations used by DeepEP/MoE workloads)   │
│  - Transport abstraction (Net, CollNet, GIN plugins)    │
└──────────────┬──────────────────────┬───────────────────┘
               │ ncclNet_v12_t API    │ ncclGin_v13_t API
┌──────────────▼──────────────────────▼───────────────────┐
│          OFI NCCL Plugin (aws-ofi-nccl)                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │ RDMA         │ │ SendRecv     │ │ GIN          │   │
│  │ Transport    │ │ Transport    │ │ Transport    │   │
│  │ (P5/P5en/P6) │ │ (P4d)       │ │ (wraps RDMA) │   │
│  └──────────────┘ └──────────────┘ └──────────────┘   │
│  C++ class hierarchy: plugin→device→domain→ep→comm     │
│  Ownership model: shared_ptr/weak_ptr                   │
│  Modules: Connection Manager, MR Cache, Tuner           │
└──────────────────────────┬──────────────────────────────┘
                           │ libfabric API (fi_tsend,
                           │ fi_write, fi_mr_reg, etc.)
┌──────────────────────────▼──────────────────────────────┐
│            Libfabric (2.x)                              │
│  - Provider-agnostic API                                │
│  - EFA provider implementation                          │
└──────────────────────────┬──────────────────────────────┘
                           │ libibverbs API (ibv_post_send,
                           │ ibv_reg_mr, etc.)
┌──────────────────────────▼──────────────────────────────┐
│         rdma-core (libibverbs, v61+)                    │
│  - Userspace verbs library                              │
│  - EFA provider plugin                                  │
│  - Zero-copy queue operations via memory-mapped rings   │
└──────────────────────────┬──────────────────────────────┘
                           │ ioctl / mmap
┌──────────────────────────▼──────────────────────────────┐
│          EFA Kernel Driver  (efa.ko)                    │  ◄── lowest open-source layer
│  - Device control (uverbs interface)                    │
│  - Memory registration (page pinning, IOMMU mapping)    │
│  - GPU/accelerator peer memory (P2P) + DMA-BUF          │
│  - Admin-queue command interface; hardware resource mgmt │
└──────────────────────────┬──────────────────────────────┘
            admin queue cmds │ + data-path doorbells/CQs
┌──────────────────────────▼──────────────────────────────┐
│     EFA NIC Firmware  (closed)                          │
│  - SRD (Scalable Reliable Datagram) protocol engine     │
│  - Multi-path routing (up to 64 paths), congestion ctrl │
│  - ACK/retransmit, reordering                           │
└──────────────────────────┬──────────────────────────────┘
                           │ PCIe / hardware
┌──────────────────────────▼──────────────────────────────┐
│     AWS EFA Hardware (NIC ASIC, closed)                 │
│  - DMA engines, queue pairs, completion queues          │
│  - OS bypass for data path                              │
└─────────────────────────────────────────────────────────┘
```

The EFA kernel driver is the lowest open-source layer; the SRD protocol engine and
congestion control run in closed NIC firmware, reached only through admin-queue commands
and data-path doorbells/CQs. See [kernel-efa-driver.md](kernel-efa-driver.md) for the
firmware boundary, admin-queue phase-bit protocol, and GPU peer-memory internals.

## Component Roles

### NCCL
- **Purpose**: High-performance multi-GPU/multi-node collective communication
- **Key Features**: Optimized collective algorithms, topology awareness, transport abstraction
- **Net Plugin API** (`ncclNet_v12_t`, newest; older versions down to v4 also exported for back-compat): Standard send/recv/write interface for collective operations
- **GIN Plugin API** (`ncclGin_v13_t`, newest; `ncclGin_v11_t` also exported): One-sided put/signal interface for GPU-initiated networking (DeepEP/MoE)
- **Focus Areas**: Collective operations, data path, algorithms

### OFI NCCL Plugin
- **Purpose**: Bridge between NCCL and libfabric
- **Key Features**: Implements NCCL's network plugin interface using libfabric APIs
- **Architecture**: C++ class hierarchy with smart pointer ownership
  - `nccl_net_ofi_plugin_t` → `nccl_net_ofi_device_t` → `nccl_net_ofi_domain_t` → `nccl_net_ofi_ep_t` → communicators
  - Comms hold `shared_ptr<ep>`, endpoints hold `shared_ptr<domain>`, tables hold `weak_ptr` (non-owning caches)
  - Destruction cascades automatically when last comm closes, no locks needed
- **Transports**: RDMA (P5+, default), SendRecv (P4d), GIN (wraps RDMA for DeepEP)
- **Modules**: Connection Manager (`src/cm/`), MR Cache (`src/nccl_ofi_mr.cpp`), Tuner (`src/tuner/`), Scheduler (`src/nccl_ofi_scheduler.cpp`)
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
- **Key Features**: RDMA-like performance, OS bypass, scalable, SRD protocol
- **Hardware Generations**:
  - EFAv2: P5/P5e instances, 100 Gbps per adapter, 75 us latency
  - EFAv3: P5en/P6 instances, 100 Gbps per adapter, 35 us latency
- **Focus Areas**: Driver architecture, capabilities, performance characteristics

## Data Flow

### Standard NCCL Net Path (Collectives)

1. **Application** issues collective operation (e.g., AllReduce)
2. **NCCL** breaks down collective into point-to-point operations based on algorithm
3. **NCCL proxy thread** calls plugin API (isend/irecv/iwrite/iflush)
4. **OFI Plugin** translates NCCL network calls to libfabric operations
5. **Libfabric EFA Provider** converts to rdma-core verbs calls
6. **rdma-core (libibverbs)** posts work requests via verbs API (ibv_post_send/recv)
7. **EFA Kernel Driver** manages memory registration and device control via uverbs
8. **EFA Hardware** performs zero-copy DMA and transmits over SRD protocol

### GIN Path (One-Sided Put Operations for DeepEP/MoE)

1. **Application** calls DeepEP dispatch (MoE token routing)
2. **NCCL GIN Device API** kernel calls put/putSignal from GPU
3. **GIN proxy** calls plugin GIN API (iput, iputSignal)
4. **GIN Transport** translates to RDMA write operations (fi_write)
5. Same **libfabric → rdma-core → EFA** path as standard
6. Remote side polls signal to detect completion (ginProgress)

## Key Concepts

### Collectives
Operations that involve multiple processes/GPUs working together:
- AllReduce, AllGather, Broadcast, Reduce, ReduceScatter, etc.

### Algorithms
NCCL implements multiple algorithms per collective:
- **Ring**: Bandwidth-optimal for large messages, GPUs arranged in a circle
- **Tree**: Latency-optimal for small messages, double binary tree
- **PAT**: Parallel Aggregated Trees for AllGather/ReduceScatter
- **NVLS**: NVSwitch-accelerated for intra-node communication
- **CollNet**: Offloaded collectives (not used on EFA)

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
- Cached per-domain in the plugin to avoid expensive re-registration
- Supports host memory and GPU memory (via GPUDirect or DMA-BUF)

### Plugin Object Ownership

The OFI plugin uses `shared_ptr`/`weak_ptr` for automatic lifetime management:

```
  plugin (singleton)
    └── device (one per NIC / multi-rail group)
          └── domain_table: weak_ptr<domain> (non-owning cache)
                └── domain (one per thread scope, holds fi_domain + MR cache)
                      └── ep_table: weak_ptr<ep> (non-owning cache)
                            └── ep (one per thread, holds fi_endpoint)
                                  └── comm (listen/send/recv)
                                        holds: shared_ptr<ep>

  Ownership chain:
    comm ──shared_ptr──> ep ──shared_ptr──> domain ──raw ptr──> device
```

Destruction cascades automatically when the last comm is closed. No manual
reference counting, no locks during destruction. See [ofi-plugin.md](ofi-plugin.md)
for details.

### GIN (Group Interconnect Network)
- NCCL extension API for one-sided put operations from GPU kernels
- Used by workloads like DeepEP (Mixture-of-Experts dispatch)
- Provides `iput` (data transfer) and `iputSignal` (data + atomic notification)
- Wraps RDMA transport with additional resource management
- Two modes selected by `OFI_NCCL_GIN_TYPE`: **PROXY** (default; CPU proxy issues
  the libfabric writes) and **GDAKI** (`--enable-gdaki` builds; the GPU kernel
  drives the EFA queues directly — kernel-initiated networking, no CPU on the data
  path, requires CUDA + DMA-BUF + libfabric 2.5+)
- See [nccl-ep-vs-deepep-comparison.md](nccl-ep-vs-deepep-comparison.md) for NCCL EP vs DeepEP analysis
- See [gin-alltoall-libfabric-trace.md](gin-alltoall-libfabric-trace.md) for detailed libfabric call trace

## Document Organization

### Core Concepts
- `overview.md` - This file: architecture overview and component interactions
- `nccl-core.md` - NCCL fundamentals, topology, communicators
- `nccl-usage.md` - Communicators, operations, threading, practical usage
- `nccl-channels.md` - NCCL channels: parallelism, multi-rail support, tuning
- `nccl-collectives.md` - Collective operations, algorithms, protocols
- `nccl-tuner.md` - Cost-based algorithm/protocol selection, AWS tuner
- `nccl-datapath.md` - Complete data path from GPU through network

### Collective Algorithms
- `algorithms/ring-algorithm.md` - Ring algorithm (bandwidth-optimal)
- `algorithms/tree-algorithm.md` - Double binary tree (latency-optimal)
- `algorithms/nvls-tree-algorithm.md` - NVLS Tree with NVSwitch acceleration
- `algorithms/pat-algorithm.md` - Parallel Aggregated Trees

### OFI Plugin
- `ofi-plugin.md` - Plugin architecture, class hierarchy, ownership model
- `ofi-plugin-protocols.md` - Connection establishment, RDMA/SendRecv/GIN protocols
- `mr-cache-implementation.md` - Memory registration cache implementation

### NCCL Message Processing
- `nccl-message-breakdown-complete.md` - How NCCL breaks down messages for all collectives
- `nccl-chunk-breakdown.md` - Chunk size calculation and pipeline depth
- `nccl-buffsize-explained.md` - NCCL_BUFFSIZE channel buffers and pipelining

### GIN and Expert Parallelism
- `nccl-ep-vs-deepep-comparison.md` - NCCL EP vs DeepEP-NCCL comparison
- `gin-alltoall-libfabric-trace.md` - GIN hybrid all-to-all libfabric call trace

### EFA Hardware and Protocol
- `efa-hardware-architecture.md` - Queue pairs, completion queues, hardware capabilities
- `srd-protocol.md` - Scalable Reliable Datagram protocol
- `kernel-efa-driver.md` - Linux kernel EFA driver internals
- `efa-driver.md` - EFA driver architecture

### Transport Layer
- `libfabric-overview.md` - Libfabric API, capabilities, data transfer operations
- `efa-provider.md` - EFA provider specifics, eager/rendezvous protocols
- `rdma-core-and-verbs.md` - rdma-core userspace libraries, libibverbs API

### System Level
- `threading-model.md` - Threading across the stack, proxy threads, concurrency
- `rdma-memreg.md` - RDMA operations, memory registration, caching strategies
- `topology-and-binding.md` - Topology detection and interface binding

### Performance
- `optimizations.md` - Kernel bypass, caching, tuning parameters, best practices
- `optimization-opportunities.md` - Detailed analysis of optimization opportunities

### Deep Dives
- `lkey-rkey-explained.md` - Understanding local and remote keys in RDMA
- `freelist-allocator.md` - Freelist allocator for high-performance object pooling
- `dmabuf-gpu-memory.md` - DMA-BUF framework for vendor-neutral GPU memory registration

### GPU and Accelerator Memory
- `cuda-memory.md` - NVIDIA CUDA memory: dynamic loading, GPUDirect, GDRCopy
- `neuron-memory.md` - AWS Trainium/Inferentia with P2P registration
- `rocm-memory.md` - AMD ROCm with HIP API

## Performance Considerations

Key areas affecting performance:
1. **Algorithm selection** - Ring vs Tree vs PAT vs NVLS (message size dependent)
2. **Protocol choice** - Simple vs LL vs LL128 (latency vs bandwidth tradeoff)
3. **Memory registration strategy** - MR cache hit rate, page alignment, DMA-BUF vs peermem
4. **Buffer sizing** - NCCL_BUFFSIZE, chunk sizes, pipeline depth (see [nccl-buffsize-explained.md](nccl-buffsize-explained.md))
5. **Threading model** - Progress threads, locking, proxy thread efficiency
6. **EFA-specific features** - SRD multi-path routing, RDMA write, EFAv3 latency improvements
7. **Multi-rail** - Using all available NICs, rail reordering for optimal GPU-NIC pairing
8. **Topology awareness** - GPU-NIC affinity, NVLink vs PCIe vs C2C paths

## References

- NCCL: https://github.com/NVIDIA/nccl
- OFI Plugin: https://github.com/aws/aws-ofi-nccl
- Libfabric: https://github.com/ofiwg/libfabric
- rdma-core: https://github.com/linux-rdma/rdma-core
- EFA: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html
- DeepEP: https://github.com/deepseek-ai/DeepEP
