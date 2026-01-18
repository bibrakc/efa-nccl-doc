# NCCL on AWS EFA Documentation

Comprehensive technical documentation for NCCL (NVIDIA Collective Communications Library) using the OFI plugin with libfabric and AWS EFA (Elastic Fabric Adapter).

## Purpose

This documentation provides in-depth coverage of the complete stack used for GPU-to-GPU communication on AWS EC2 instances with EFA. It's designed for engineers working on optimization, debugging, and understanding the performance characteristics of distributed deep learning workloads.

## Documentation Structure

### Core Concepts

- **[00-overview.md](00-overview.md)** - Architecture overview and component interactions
- **[01-nccl-core.md](01-nccl-core.md)** - NCCL fundamentals, topology, communicators, channels
- **[02-nccl-collectives.md](02-nccl-collectives.md)** - Collective operations, algorithms (Ring/Tree), protocols (Simple/LL/LL128)
- **[03-nccl-datapath.md](03-nccl-datapath.md)** - Complete data path from GPU through network

### Transport Layer

- **[04-ofi-plugin.md](04-ofi-plugin.md)** - OFI NCCL plugin architecture and implementation
- **[05-libfabric-overview.md](05-libfabric-overview.md)** - Libfabric API, capabilities, data transfer operations
- **[06-efa-provider.md](06-efa-provider.md)** - EFA provider specifics, SRD protocol, eager/rendezvous protocols

### System Level

- **[07-threading-model.md](07-threading-model.md)** - Threading across the stack, proxy threads, concurrency
- **[08-rdma-memreg.md](08-rdma-memreg.md)** - RDMA operations, memory registration, caching strategies
- **[09-efa-driver.md](09-efa-driver.md)** - EFA kernel driver architecture, queue management, GPUDirect

### Performance

- **[10-optimizations.md](10-optimizations.md)** - Kernel bypass, caching, tuning parameters, best practices
- **[11-optimization-opportunities.md](11-optimization-opportunities.md)** - Detailed analysis of optimization opportunities with priorities

### Deep Dives

- **[12-lkey-rkey-explained.md](12-lkey-rkey-explained.md)** - Understanding local and remote keys in RDMA
- **[13-ofi-plugin-protocols.md](13-ofi-plugin-protocols.md)** - Connection establishment and send/recv protocols
- **[14-topology-and-binding.md](14-topology-and-binding.md)** - Topology detection and interface binding mechanisms

## Quick Start

### Understanding the Stack

Start with these documents in order:

1. [00-overview.md](00-overview.md) - Get the big picture
2. [01-nccl-core.md](01-nccl-core.md) - Understand NCCL basics
3. [03-nccl-datapath.md](03-nccl-datapath.md) - Follow a message through the system
4. [10-optimizations.md](10-optimizations.md) - Apply optimizations

### For Specific Topics

- **Performance tuning** → [10-optimizations.md](10-optimizations.md), [11-optimization-opportunities.md](11-optimization-opportunities.md)
- **Debugging network issues** → [04-ofi-plugin.md](04-ofi-plugin.md), [06-efa-provider.md](06-efa-provider.md)
- **Algorithm selection** → [02-nccl-collectives.md](02-nccl-collectives.md)
- **Memory registration problems** → [08-rdma-memreg.md](08-rdma-memreg.md)
- **Threading issues** → [07-threading-model.md](07-threading-model.md)
- **Connection protocols** → [13-ofi-plugin-protocols.md](13-ofi-plugin-protocols.md)
- **Topology and multi-NIC** → [14-topology-and-binding.md](14-topology-and-binding.md)
- **Understanding RDMA keys** → [12-lkey-rkey-explained.md](12-lkey-rkey-explained.md)

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
EFA Kernel Driver
    ↓
EFA Hardware (100G+ NIC)
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

1. **Memory Registration Caching** - 100-500x speedup for repeated transfers
2. **Algorithm Selection** - Ring for large messages, Tree for small
3. **Protocol Selection** - LL for latency, Simple for bandwidth
4. **Multi-rail** - Utilize all EFA adapters (4-8 per instance)
5. **CPU Affinity** - Pin proxy threads to avoid NUMA remote access
6. **Kernel Bypass** - Direct user-space access to hardware

## Common Configurations

### Production (High Bandwidth)

```bash
# Memory registration
FI_MR_CACHE_MONITOR=memhooks
FI_MR_CACHE_MAX_SIZE=unlimited

# EFA
FI_EFA_ENABLE_SHM_TRANSFER=0
FI_EFA_USE_DEVICE_RDMA=1
FI_EFA_TX_SIZE=256
FI_EFA_RX_SIZE=256

# NCCL
NCCL_ALGO=Ring
NCCL_PROTO=Simple
NCCL_NCHANNELS=8
NCCL_DEBUG=WARN
```

### Low Latency

```bash
# For small messages, frequent collectives
NCCL_ALGO=Tree
NCCL_PROTO=LL,LL128
NCCL_NCHANNELS=4
NCCL_CHUNK_SIZE=65536
```

### Debugging

```bash
# Verbose logging
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,COLL,NET,PROXY

# Libfabric logging
FI_LOG_LEVEL=info
FI_LOG_PROV=efa

# Graph dump
NCCL_GRAPH_DUMP_FILE=nccl_graph.txt
```

## Performance Targets

### Latency (Single Message)

| Message Size | Target Latency | Notes |
|--------------|----------------|-------|
| 8 bytes      | 10-15 μs       | Protocol overhead dominates |
| 1 KB         | 12-20 μs       | LL protocol optimal |
| 64 KB        | 20-30 μs       | LL128 protocol |
| 1 MB         | 100-200 μs     | Simple protocol |

### Bandwidth (Sustained)

| Configuration | Target Bandwidth | Notes |
|---------------|------------------|-------|
| 1 EFA (100G)  | 11-12 GB/s       | ~96% line rate |
| 2 EFAs        | 22-23 GB/s       | Linear scaling |
| 4 EFAs        | 40-45 GB/s       | PCIe limit approaching |
| 8 EFAs (p5)   | 80-100 GB/s      | PCIe Gen5 systems |

### AllReduce (8 GPUs, Ring Algorithm)

| Data Size | Latency | Bus Bandwidth | Notes |
|-----------|---------|---------------|-------|
| 128 MB    | 5-10 ms | 100+ GB/s     | Typical gradient size |
| 1 GB      | 40-80 ms| 100+ GB/s     | Large model |

## Troubleshooting

### Low Bandwidth

Check:
- MR cache hit rate (enable logging)
- Channel count (NCCL_NCHANNELS)
- Algorithm selection (use Ring for large)
- CPU affinity (avoid NUMA remote)

### High Latency

Check:
- Protocol (use LL/LL128 for small)
- CPU frequency scaling (disable)
- Proxy thread priority
- Polling frequency

### Hangs/Crashes

Check:
- MR cache invalidation (memhooks enabled?)
- Resource limits (queue sizes)
- IOMMU configuration (dmesg)
- Version compatibility

## Benchmarking

### NCCL Tests

```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make

# All-reduce
./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8

# All-gather
./build/all_gather_perf -b 8 -e 1G -f 2 -g 8

# Broadcast
./build/broadcast_perf -b 8 -e 1G -f 2 -g 8
```

### Interpreting Results

- **AlgBW**: Algorithm bandwidth (useful for comparison)
- **BusBW**: Actual network bandwidth utilized
- **Time**: Average latency per operation

For AllReduce with Ring algorithm:
```
BusBW = AlgBW × 2 × (N-1) / N
```

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

### 2026-01-17
- Initial comprehensive documentation
- Covered full stack: NCCL → EFA hardware
- Included optimization guide
- Added troubleshooting section

---

**Audience**: Engineers optimizing distributed deep learning workloads on AWS EC2 with EFA
**Focus**: Technical depth, performance optimization, debugging
**Scope**: NCCL + OFI plugin + libfabric + EFA driver + EFA hardware
