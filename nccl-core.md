# NCCL Core Concepts and Operations

## Overview

NCCL (NVIDIA Collective Communications Library) is a library providing optimized primitives for collective multi-GPU communication. It's designed for high-bandwidth, low-latency communication between GPUs across nodes.

## Core Architecture

### Communication Domains

#### Intra-node (Within a Node)
- **PCIe/NVLink**: Direct GPU-to-GPU communication
- **NVLink**: NVIDIA's high-speed interconnect (900 GB/s on recent systems)
- **PCIe**: Fallback when NVLink not available
- **Shared Memory**: Used for CPU-GPU and GPU-GPU data transfer

#### Inter-node (Between Nodes)
- **Network Adapters**: NICs like EFA, InfiniBand, RoCE
- **NCCL Net Plugin**: Abstraction layer for network communication
- **Multiple NICs**: NCCL can utilize multiple NICs per node

### Topology Awareness

NCCL builds a topology model to optimize communication patterns:

```
Node 1                          Node 2
┌──────────────────────┐       ┌──────────────────────┐
│  GPU0 ←─→ GPU1       │       │  GPU0 ←─→ GPU1       │
│   │  NVLink  │       │       │   │  NVLink  │       │
│   └────┬─────┘       │       │   └────┬─────┘       │
│        │             │       │        │             │
│    ┌───▼────┐        │       │    ┌───▼────┐        │
│    │  PCIe  │        │       │    │  PCIe  │        │
│    └───┬────┘        │       │    └───┬────┘        │
│        │             │       │        │             │
│    ┌───▼────┐        │       │    ┌───▼────┐        │
│    │  NIC   │◄───────┼───────┼────►│  NIC   │        │
│    └────────┘ Network│       │    └────────┘        │
└──────────────────────┘       └──────────────────────┘
```

**Topology Detection:**
- PCI bus information
- NVLink connectivity matrix
- Network adapter assignments
- NUMA domains
- CPU affinity

**Impact on Performance:**
- Path selection for data movement
- Algorithm choice (Ring vs Tree)
- Chunk size determination
- Channel assignment

### Communicator

**`ncclComm_t`** - Communicator handle ([NCCL src/nccl.h.in](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in) - NCCL core, opaque structure):

```c
ncclComm_t comm;
ncclCommInitRank(&comm, nranks, commId, rank);
```

**Communicator Properties:**
- **Rank**: Unique identifier for each process (0 to N-1)
- **Size**: Total number of participating processes
- **Unique ID**: Shared identifier for all ranks in communicator
- **Topology**: Cached topology information
- **Channels**: Communication channels for parallelism

**Communicator Initialization:**
1. Exchange unique ID among all ranks
2. Establish bootstrap network for setup
3. Build topology graph
4. Allocate channels and resources
5. Setup intra-node connections (NVLink/PCIe)
6. Setup inter-node connections (network)

### Channels

NCCL uses multiple channels for parallel data transfer:

```
GPU Memory
┌─────────────────────────────┐
│  Data Buffer                │
│                             │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐  │
│  │Ch0│ │Ch1│ │Ch2│ │Ch3│  │
│  └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘  │
└────┼─────┼─────┼─────┼─────┘
     │     │     │     │
     ▼     ▼     ▼     ▼
   Independent parallel paths
```

**Channel Characteristics:**
- Typically 4-16 channels per communicator
- Each channel handles a portion of data
- Allows pipelining and overlap
- Maps to different physical paths when possible

**Channel Selection:**
- Based on topology (avoid conflicts)
- Balanced across available NICs
- Considers NVLink connectivity
- Configurable via `NCCL_NCHANNELS`

### Operation Lifecycle

```
Application Call
     │
     ▼
┌─────────────────────┐
│  Enqueue Operation  │  ← ncclAllReduce(), etc.
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Build Task Graph   │  ← Break into steps
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Schedule Channels  │  ← Assign to channels
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Execute on GPU     │  ← Launch CUDA kernels
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Network Transfer   │  ← Via OFI plugin
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Completion         │  ← Synchronize
└─────────────────────┘
```

## Memory Management

### Buffer Types

1. **Device Buffers**: GPU memory (main data)
2. **Host Buffers**: CPU memory (bounce buffers)
3. **Registration Cache**: Pre-registered memory regions
4. **Proxy Buffers**: Intermediate buffers for network

### CUDA Integration

NCCL operations work with CUDA streams:

```c
ncclAllReduce(sendbuff, recvbuff, count, datatype,
              op, comm, stream);
```

**Stream Semantics:**
- Operations are asynchronous
- Enqueued to CUDA stream
- Kernel launch for data movement
- Network operations overlapped with compute

## Group Calls

Multiple operations can be launched as a group:

```c
ncclGroupStart();
ncclSend(sendbuff1, count, datatype, peer, comm, stream);
ncclRecv(recvbuff1, count, datatype, peer, comm, stream);
ncclGroupEnd();
```

**Benefits:**
- Single synchronization point
- Better scheduling opportunities
- Reduced overhead
- Fused algorithm selection

## Transport Abstraction

NCCL has a pluggable transport layer:

### Built-in Transports
1. **NET**: Network transport (plugin interface)
2. **P2P**: PCIe/NVLink direct GPU access
3. **SHM**: Shared memory for intra-node
4. **Collnet**: Collective offload (for switches)

### Net Plugin Interface

**`ncclNet_t`** - Network plugin interface ([NCCL net.h](https://github.com/NVIDIA/nccl/blob/master/src/include/net.h)):

```c
typedef struct {
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_t* props);
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  ncclResult_t (*regMr)(void* comm, void* data, int size,
                        int type, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  ncclResult_t (*isend)(void* sendComm, void* data, int size,
                        int tag, void* mhandle, void** request);
  ncclResult_t (*irecv)(void* recvComm, int n, void** data,
                        int* sizes, int* tags, void** mhandles,
                        void** request);
  ncclResult_t (*iflush)(void* recvComm, int n, void** data,
                         int* sizes, void** mhandles, void** request);
  ncclResult_t (*test)(void* request, int* done, int* size);
  ncclResult_t (*close)(void* comm);
} ncclNet_t;
```

**OFI Plugin implements this interface using libfabric** - See [ofi-plugin.md](ofi-plugin.md) for implementation details.

> **ABI note (NCCL v2.31.2-1).** The struct above is a *simplified, illustrative*
> view of the classic net interface. The current net-plugin ABI NCCL core looks up
> is **`ncclNetPlugin_v12`** (`typedef ncclNet_v12_t ncclNet_t;` in
> [nccl/src/include/plugin/nccl_net.h](https://github.com/NVIDIA/nccl/blob/master/src/include/plugin/nccl_net.h)),
> with backward-compatible shims for v6–v11. Real signatures differ from the sketch
> above (e.g. `isend`/`irecv` take extra size/tag arrays and per-request phints, and
> there are device-side / GPU-initiated-networking (GIN) and RMA entry points). The
> sketch is kept for conceptual clarity; consult the versioned `net_v*.h` headers for
> the exact current ABI.

## Configuration and Tuning

### Key Environment Variables

```bash
# Number of channels (affects parallelism)
NCCL_NCHANNELS=8

# Algorithm selection
NCCL_ALGO=Ring,Tree

# Protocol selection
NCCL_PROTO=Simple,LL,LL128

# Debugging
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,TUNING

# Network interface selection
NCCL_SOCKET_IFNAME=eth0
NCCL_IB_HCA=mlx5_0

# Topology
NCCL_TOPO_FILE=/path/to/topology.xml
NCCL_GRAPH_FILE=/path/to/graph.xml

# Performance
NCCL_BUFFSIZE=4194304  # Channel buffer size
NCCL_P2P_LEVEL=NVL     # NVLink level
```

### Runtime Behavior

**Lazy Initialization:**
- Communicator setup on first collective
- Topology detection happens once
- Channel setup deferred until needed

**Kernel Fusion:**
- Multiple operations can share kernel launches
- Reduces CUDA overhead
- Enabled via group calls

## Performance Characteristics

### Latency Components

1. **Software Overhead**:
   - NCCL library processing: ~1-2 μs
   - CUDA kernel launch: ~5-10 μs

2. **Data Movement**:
   - Intra-node (NVLink): ~1 GB = 1-2 μs
   - Inter-node (EFA): ~1 GB = 10-50 μs (depends on size)

3. **Synchronization**:
   - Barrier overhead: ~5-20 μs
   - Increases with rank count

### Bandwidth Optimization

**Factors Affecting Bandwidth:**
- Message size (larger = better efficiency)
- Number of channels (more = higher bandwidth)
- Algorithm choice (Ring better for large, Tree for small)
- Network topology (affects path efficiency)

**Achieved Bandwidth:**
- Intra-node: Up to 300-600 GB/s (NVLink)
- Inter-node: Up to 90-95% of wire speed (EFA: ~95 Gbps on 100G)

## Error Handling

NCCL provides error checking:

```c
ncclResult_t result = ncclAllReduce(...);
if (result != ncclSuccess) {
  printf("Error: %s\n", ncclGetErrorString(result));
}

// Async error checking
ncclResult_t asyncErr;
ncclCommGetAsyncError(comm, &asyncErr);
```

**Error Types:**
- `ncclSuccess`: Operation succeeded
- `ncclInvalidArgument`: Bad parameters
- `ncclInvalidUsage`: API misuse
- `ncclSystemError`: System-level error
- `ncclInternalError`: NCCL bug
- `ncclRemoteError`: Remote rank failed

## Summary

NCCL provides:
- High-performance collective operations
- Topology-aware communication
- Pluggable transport layer (OFI plugin)
- Multi-channel parallelism
- Asynchronous operation with CUDA streams
- Extensive tuning knobs for optimization

The OFI plugin integrates into this architecture through the `ncclNet_t` interface, translating NCCL's network requirements into libfabric operations.

**Related Documentation**:
- [nccl-datapath.md](nccl-datapath.md) - Detailed data flow from GPU to GPU
- [nccl-collectives.md](nccl-collectives.md) - Collective algorithms (AllReduce, etc.)
- [ofi-plugin.md](ofi-plugin.md) - OFI plugin implementation of ncclNet_t
- [topology-and-binding.md](topology-and-binding.md) - Topology detection details

---

## Code References

### Structures

**NCCL Core (External - NVIDIA)** - Referenced but not linked (NCCL internals):
- `ncclComm_t` - Communicator handle (opaque structure)
- `ncclNet_t` - Network plugin interface ([net.h](https://github.com/NVIDIA/nccl/blob/master/src/include/net.h))
- `ncclNetProperties_t` - Network device properties
- `ncclResult_t` - Return code enumeration

### Functions

**NCCL Core API (External - NVIDIA)** - Referenced but not linked:
- `ncclCommInitRank()` - Initialize communicator with rank
- `ncclAllReduce()` - AllReduce collective operation
- `ncclSend()` / `ncclRecv()` - Point-to-point operations
- `ncclGroupStart()` / `ncclGroupEnd()` - Group call batching
- `ncclGetErrorString()` - Get error description
- `ncclCommGetAsyncError()` - Check for async errors

**ncclNet_t Interface Functions** - Plugin implementation required:
- `init()` - Initialize plugin
- `devices()` - Get device count
- `getProperties()` - Query device properties
- `listen()` - Create listening endpoint
- `connect()` - Connect to remote endpoint
- `accept()` - Accept incoming connection
- `regMr()` - Register memory region
- `deregMr()` - Deregister memory region
- `isend()` - Non-blocking send
- `irecv()` - Non-blocking receive
- `iflush()` - Flush operation (GPUDirect)
- `test()` - Test for completion
- `close()` - Close connection

### Environment Variables

**NCCL Configuration**:
- `NCCL_NCHANNELS` - Number of channels (default: auto)
- `NCCL_ALGO` - Algorithm selection (Ring, Tree, CollNet)
- `NCCL_PROTO` - Protocol selection (Simple, LL, LL128)
- `NCCL_DEBUG` - Debug level (WARN, INFO, TRACE)
- `NCCL_DEBUG_SUBSYS` - Debug subsystems
- `NCCL_BUFFSIZE` - Channel buffer size (default: 4 MB)
- `NCCL_P2P_LEVEL` - P2P connection level (NVL, PIX, PHB, etc.)
- `NCCL_NET_GDR_LEVEL` - GPUDirect level (0-5)
- `NCCL_SOCKET_IFNAME` - Network interface name
- `NCCL_IB_HCA` - InfiniBand HCA name
- `NCCL_TOPO_FILE` - Custom topology XML file
- `NCCL_GRAPH_FILE` - Custom graph XML file

### Error Codes

**ncclResult_t values**:
- `ncclSuccess` - Operation succeeded
- `ncclInvalidArgument` - Invalid argument
- `ncclInvalidUsage` - API misuse
- `ncclSystemError` - System error
- `ncclInternalError` - Internal NCCL error
- `ncclRemoteError` - Remote rank error

### Total Code References
- **4 NCCL structures** (external NVIDIA)
- **6 NCCL API functions** (external NVIDIA)
- **13 ncclNet_t interface functions** (plugin must implement)
- **12 environment variables**
- **6 error codes**

**Note**: NCCL core is proprietary NVIDIA software. The ncclNet_t interface is the public API that network plugins (like aws-ofi-nccl) must implement. For OFI plugin implementation details, see [ofi-plugin.md](ofi-plugin.md).
