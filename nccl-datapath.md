# NCCL Network Data Path and Protocols

## Data Path Overview

The NCCL data path describes how data flows from source GPU through the network to destination GPU.

```
┌──────────────────────────────────────────────────────────────┐
│                        NCCL Data Path                         │
└──────────────────────────────────────────────────────────────┘

Source GPU                                    Destination GPU
┌──────────┐                                      ┌──────────┐
│  GPU     │                                      │  GPU     │
│  Memory  │                                      │  Memory  │
└────┬─────┘                                      └────▲─────┘
     │                                                 │
     ▼                                                 │
┌──────────┐                                      ┌──────────┐
│  NCCL    │                                      │  NCCL    │
│  Kernel  │                                      │  Kernel  │
└────┬─────┘                                      └────▲─────┘
     │                                                 │
     ▼                                                 │
┌──────────┐                                      ┌──────────┐
│  Proxy   │                                      │  Proxy   │
│  Thread  │                                      │  Thread  │
└────┬─────┘                                      └────▲─────┘
     │                                                 │
     ▼                                                 │
┌──────────────────────────────────────┐   ┌──────────────────┐
│      OFI Plugin (ncclNet_t)         │   │   OFI Plugin     │
└────┬─────────────────────────────────┘   └────▲─────────────┘
     │                                           │
     ▼                                           │
┌──────────────────────────────────────┐   ┌────────────────┐
│        Libfabric (EFA Provider)     │   │   Libfabric    │
└────┬─────────────────────────────────┘   └────▲───────────┘
     │                                           │
     ▼                                           │
┌──────────────────────────────────────┐   ┌────────────────┐
│     rdma-core (libibverbs)          │   │  rdma-core     │
│     - ibv_post_send/recv            │   │  - ibv_poll_cq │
└────┬─────────────────────────────────┘   └────▲───────────┘
     │                                           │
     ▼                                           │
┌──────────────────────────────────────┐   ┌────────────────┐
│      EFA Driver (uverbs)            │   │   EFA Driver   │
└────┬─────────────────────────────────┘   └────▲───────────┘
     │                                           │
     └────────────────► Network ────────────────┘
```

## Detailed Data Flow

### Step-by-Step for Inter-Node Communication

#### 1. Application Issues Collective

```c
// User code
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat,
              ncclSum, comm, stream);
```

- **Action**: Enqueue to CUDA stream
- **Timing**: ~1-2 μs (software overhead)

#### 2. NCCL Planning

```
Tasks:
- Determine algorithm (Ring/Tree)
- Select protocol (Simple/LL/LL128)
- Break data into chunks
- Assign chunks to channels
- Build operation graph
```

- **Action**: Create work items for each channel
- **Timing**: < 1 μs (usually cached)

#### 3. CUDA Kernel Launch

```
For each channel:
  Launch NCCL kernel on GPU
```

**Kernel Responsibilities:**
- Read data from source GPU memory
- Perform local reduction if needed
- Write to proxy buffers
- Signal proxy thread

- **Timing**: ~5-10 μs (kernel launch latency)

#### 4. Proxy Thread Activation

NCCL uses proxy threads for network communication:

```
Proxy Thread (CPU):
  while (active) {
    poll for work from GPU
    if (work available) {
      issue network send/recv
      poll for completion
      signal GPU
    }
  }
```

**Why Proxy Threads:**
- Network operations are CPU-driven
- GPU focuses on compute
- Enables async progress
- CPU can manage complex protocols

- **Timing**: Wakeup ~1-2 μs

#### 5. OFI Plugin Send

**`ncclNet_t`** interface ([net.h](https://github.com/NVIDIA/nccl/blob/master/src/include/net.h) - NCCL core, external):

```c
// Proxy calls into OFI plugin
ncclNet->isend(sendComm, data, size, tag, mhandle, &request);
```

**Plugin Actions:**
- Translate NCCL request to libfabric
- Get memory handle (if not cached)
- Post `fi_send()` or `fi_write()` ([libfabric fi_msg.h](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_msg.h))
- Return request handle

- **Timing**: ~1-3 μs (if memory registered)

#### 6. Libfabric Processing

```c
// Inside OFI plugin
fi_send(ep, buf, len, desc, dest_addr, context);
// or
fi_write(ep, buf, len, desc, dest_addr, remote_addr,
         remote_key, context);
```

**Libfabric Actions:**
- Validate endpoint
- Prepare EFA-specific headers
- Queue work request
- Call rdma-core verbs

- **Timing**: ~500-1000 ns

#### 7. rdma-core (libibverbs)

**`ibv_post_send()`** / **`ibv_post_recv()`** ([rdma-core verbs.h:2554, 2562](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L2554)):

```c
// Libfabric EFA provider calls verbs API
ibv_post_send(qp, &wr, &bad_wr);
// or
ibv_post_recv(qp, &wr, &bad_wr);
```

**rdma-core Actions:**
- Write work queue entry (WQE) to memory-mapped send queue
- Ring doorbell (MMIO write to notify hardware)
- **Zero syscalls** - entirely userspace operation

- **Timing**: ~100-200 ns (memory writes + doorbell)

#### 8. EFA Driver

**Driver Actions:**
- Hardware reads WQE from memory-mapped queue (triggered by doorbell)
- DMA setup (uses pre-registered memory mapping)
- Build EFA protocol header (SRD)
- Hardware fetches data directly from source memory

- **Timing**: Hardware operation (no driver software in data path)
- **Memory Access**: Direct DMA via IOMMU

#### 9. Network Transfer

**Hardware Actions:**
- EFA NIC reads memory via PCIe
- Packetizes data
- Sends over network fabric
- Remote NIC receives
- DMA to remote memory

- **Timing**: Depends on size and distance
  - Latency: ~10-20 μs (base)
  - Bandwidth: ~100 Gbps (EFA)

#### 10. Receiver Side

**Reverse Flow:**
```
EFA Hardware
  ↓ DMA to memory, write completion
EFA Driver
  ↓ Completion in memory-mapped CQ
rdma-core
  ↓ ibv_poll_cq() reads completion
Libfabric
  ↓ fi_cq_read()
OFI Plugin
  ↓ irecv completion
Proxy Thread
  ↓ Signal GPU
NCCL Kernel
  ↓ Copy to destination
GPU Memory
```

#### 11. Completion

- Proxy polls completion queue (via ibv_poll_cq in rdma-core)
- Updates NCCL state
- GPU kernel completes
- CUDA stream progresses

## Memory Flow Details

### Buffer Types

#### GPU Device Memory

```
GPU Memory (HBM)
┌─────────────────────────┐
│  User Buffer            │  ← Application data
├─────────────────────────┤
│  NCCL Work Buffers      │  ← Channel buffers
├─────────────────────────┤
│  Temporary Buffers      │  ← Reduction scratch
└─────────────────────────┘
```

**Characteristics:**
- High bandwidth (1-2 TB/s)
- Directly accessible by CUDA kernels
- Must be registered for RDMA

#### Proxy Buffers

Some protocols use CPU memory:

```
Host Memory (CPU)
┌─────────────────────────┐
│  Proxy Bounce Buffers   │  ← Optional staging
├─────────────────────────┤
│  Control Structures     │  ← NCCL metadata
└─────────────────────────┘
```

**When Used:**
- Certain protocol variants
- Small message optimization
- Control messages

### Memory Registration

For RDMA (EFA), memory must be registered:

```
Memory Registration Process:

1. User provides GPU memory buffer
2. NCCL/OFI plugin calls:
   ncclNet->regMr(comm, data, size, type, &mhandle)
3. OFI plugin calls libfabric:
   fi_mr_reg(domain, buf, len, access, 0, 0, 0, &mr, NULL)
4. Libfabric/EFA:
   - Pins physical pages
   - Creates IOMMU mapping
   - Returns memory key
5. mhandle stored in cache
```

**Performance Impact:**
- Registration: ~100-500 μs (expensive!)
- Deregistration: ~50-100 μs
- Cache hit: < 1 μs

**NCCL Memory Registration Cache:**
```
Cache Structure:
  Key: (buffer address, size)
  Value: memory handle

On regMr():
  if (in_cache) return cached_handle;
  else register_and_cache();
```

**Tuning:**
```bash
# Cache size (default: unlimited)
NCCL_REGISTRATION_CACHE_SIZE=unlimited
```

## Protocol Deep Dive

### Simple Protocol Data Flow

```
Sender                          Receiver
──────                          ────────

1. Allocate channel buffer (4 MB default)
2. CUDA kernel copies data to buffer
3. Proxy thread posts fi_send()
                                4. Pre-posted fi_recv() matches
                                5. DMA transfer to recv buffer
                                6. Completion event
                                7. CUDA kernel processes data
8. Poll for send completion
9. Reuse buffer
```

**Buffer Management:**
- Each channel has dedicated buffers
- Size: `NCCL_BUFFSIZE` (default 4 MB)
- Ring buffer for pipelining
- Multiple slots for outstanding ops

**Pipelining:**
```
Time ──────────────────────────────────────>

Chunk 0: [Send─────][Complete]
Chunk 1:      [Send─────][Complete]
Chunk 2:           [Send─────][Complete]
```

### LL Protocol Data Flow

```
Sender                          Receiver
──────                          ────────

1. Prepare 8-byte data + flag
2. CUDA kernel writes data+flag
3. GPU polls flag location
                                4. Receiver GPU polls for flag
                                5. Flag detected → data ready
                                6. GPU reads data
7. No explicit completion needed
```

**Flag Mechanism:**
```c
struct llData {
  uint32_t flag;    // Sequence number
  uint32_t data[2]; // 8 bytes of payload
};
```

**Flag Sequence:**
```
Transfer 0: flag = 0
Transfer 1: flag = 1
Transfer 2: flag = 2
...
Wraps at MAX_INT
```

**Why Fast:**
- No network round-trip for ACK
- Direct GPU polling
- Minimal CPU involvement

**Why Slow (bandwidth):**
- Only 8 bytes per transfer
- Overhead of flag per transfer

### LL128 Protocol Data Flow

```
Sender                          Receiver
──────                          ────────

1. Prepare 128-byte data + flag
2. CUDA kernel writes data+flag
3. GPU polls flag location
                                4. Receiver GPU polls for flag
                                5. Flag detected → data ready
                                6. GPU reads 128 bytes
7. Continue...
```

**Payload:**
- 120 bytes data + 8 bytes flag = 128 bytes
- Better bandwidth than LL
- Still low latency

## Network Layer Details

### OFI Plugin Interface

```c
// Send operation
ncclResult_t ncclNetIsend(void* sendComm, void* data,
                          int size, int tag,
                          void* mhandle, void** request)
{
  struct ncclOfiSendComm* comm = sendComm;

  // Get memory descriptor
  struct fid_mr* mr = mhandle;
  void* desc = fi_mr_desc(mr);

  // Post send
  ret = fi_send(comm->ep, data, size, desc,
                comm->remote_addr, request);

  return ncclSuccess;
}
```

### Libfabric Operations

**Send (for Simple/LL protocols):**
```c
fi_send(ep, buf, len, desc, dest_addr, context);
```
- Two-sided operation
- Requires matching recv on remote
- Used for initial setup and control

**Write (for RDMA-style transfers):**
```c
fi_write(ep, buf, len, desc, dest_addr,
         remote_addr, remote_key, context);
```
- One-sided operation
- Direct write to remote memory
- Higher performance (no CPU involvement)
- Requires pre-exchanged addresses

**Recv (pre-posted):**
```c
fi_recv(ep, buf, len, desc, src_addr, context);
```
- Posted before data arrives
- Matched with incoming fi_send()

### Completion Handling

**Completion Queue:**
```c
// Poll for completions
struct fi_cq_data_entry entry;
ret = fi_cq_read(cq, &entry, 1);

if (ret > 0) {
  // entry.op_context → original request
  // Mark as complete
}
```

**Completion Models:**
1. **Polling**: Proxy thread continuously polls
2. **Event-driven**: Use fi_wait (less common in NCCL)

## Message Ordering and Dependencies

### Channel Independence

```
Channel 0: [Chunk 0] → [Chunk 4] → [Chunk 8]  ...
Channel 1: [Chunk 1] → [Chunk 5] → [Chunk 9]  ...
Channel 2: [Chunk 2] → [Chunk 6] → [Chunk 10] ...
Channel 3: [Chunk 3] → [Chunk 7] → [Chunk 11] ...
```

**Properties:**
- Channels are independent
- No ordering between channels
- Ordering within each channel
- Allows parallel progress

### Ring Algorithm Dependencies

```
Step 0: GPU 0 → GPU 1 (chunk A)
        GPU 1 → GPU 2 (chunk B)

Step 1: GPU 0 → GPU 1 (chunk D)
        GPU 1 → GPU 2 (chunk A+B)  ← Depends on Step 0
```

**Synchronization:**
- NCCL kernels handle dependencies
- Flags/counters track progress
- Proxy threads ensure ordering

## Performance Characteristics

### Latency Breakdown (Typical 8-byte message)

```
Component                    Time (μs)
────────────────────────────────────
NCCL enqueue                 1-2
Kernel launch                5-10
Proxy wakeup                 1-2
OFI plugin                   1-2
Libfabric                    1-2
EFA driver                   1-2
Network (one-way)            10-20
Receiver processing          5-10
────────────────────────────────────
Total (one-way)              25-50 μs
```

### Bandwidth Breakdown (1 GB message)

```
Component                    Bandwidth Impact
─────────────────────────────────────────────
PCIe (GPU → NIC)             ~25 GB/s (PCIe Gen4 x16)
EFA Network                  ~12.5 GB/s (100 Gbps)
Memory Registration          Cached (no impact)
Libfabric overhead           Minimal (~1-2%)
Protocol overhead (Simple)   ~1-2%
─────────────────────────────────────────────
Effective                    ~11-12 GB/s (88-96 Gbps)
```

### Multi-Channel Scaling

```
Channels    Bandwidth    Notes
────────────────────────────────────────
1           ~12 GB/s     Single path
2           ~22 GB/s     Not quite 2x
4           ~40 GB/s     Diminishing returns
8           ~48 GB/s     Approaching limit
16          ~50 GB/s     Overhead increases
```

**Why Not Linear:**
- Shared PCIe bandwidth
- Shared EFA resources
- CPU overhead per channel
- Memory bandwidth limits

## Optimization Techniques

### Pipelining

```
Without Pipelining:
[Send Chunk 0][Wait][Send Chunk 1][Wait][Send Chunk 2][Wait]

With Pipelining:
[Send Chunk 0                ]
      [Send Chunk 1                ]
            [Send Chunk 2                ]
```

**Benefit:**
- Hides latency
- Keeps network busy
- Better throughput

### Chunk Sizing

```
Too Small:               Too Large:
High overhead            Poor pipelining
Low efficiency           Higher latency

Optimal:
Balance latency/bandwidth
Typically 128 KB - 2 MB per step
```

**NCCL Tuning:**
```bash
# Chunk size (bytes)
NCCL_CHUNK_SIZE=131072  # 128 KB default
```

### Memory Registration Caching

```
First Send (cold):
  Register: 100 μs
  Transfer: 50 μs
  Total:    150 μs

Subsequent Sends (cached):
  Cache hit: 1 μs
  Transfer: 50 μs
  Total:    51 μs
```

**Significant speedup for repeated transfers**

## Summary

**Key Data Path Components:**
1. **NCCL Kernels**: Orchestrate data movement on GPU
2. **Proxy Threads**: Handle network I/O on CPU
3. **OFI Plugin**: Translate NCCL to libfabric
4. **Libfabric**: Vendor-neutral fabric API
5. **EFA Driver**: Hardware-specific operations
6. **EFA NIC**: Physical data transfer

**Performance Factors:**
- **Protocol Selection**: Simple vs LL vs LL128
- **Memory Registration**: Caching is critical
- **Channel Count**: More channels = more bandwidth
- **Pipelining**: Hides latency
- **Message Size**: Affects protocol and algorithm

**Bottlenecks:**
- Small messages: Latency-bound (use LL, Tree)
- Large messages: Bandwidth-bound (use Simple, Ring)
- Many small ops: Registration overhead (use caching)
- CPU overhead: Proxy thread scheduling

**Next Steps:**
Understanding libfabric EFA provider specifics to optimize this data path further.

**Related Documentation**:
- [ofi-plugin.md](ofi-plugin.md) - OFI plugin implementation details
- [libfabric-overview.md](libfabric-overview.md) - Libfabric API reference
- [rdma-core-and-verbs.md](rdma-core-and-verbs.md) - rdma-core libibverbs API
- [efa-hardware-architecture.md](efa-hardware-architecture.md) - EFA hardware queues and descriptors
- [mr-cache-implementation.md](mr-cache-implementation.md) - Memory registration caching
- [nccl-collectives.md](nccl-collectives.md) - NCCL collective algorithms

---

## Code References

### Functions

**NCCL Core (External - NVIDIA)** - Referenced but not linked:
- `ncclAllReduce()` - AllReduce collective operation
- `ncclNet->isend()` - Non-blocking send via network plugin
- `ncclNet->irecv()` - Non-blocking receive via network plugin
- `ncclNet->regMr()` - Register memory region for RDMA
- `ncclNet->deregMr()` - Deregister memory region
- `ncclNet->iflush()` - Flush operation for GPUDirect

**libfabric (OFI)** - Referenced but not linked (standard libfabric API):
- `fi_send()` - Send message ([fi_msg.h](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_msg.h))
- `fi_recv()` - Receive message
- `fi_write()` - RDMA write
- `fi_mr_reg()` - Register memory region ([fi_domain.h:413](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L413))
- `fi_cq_read()` - Read completion queue ([fi_eq.h](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h))

**rdma-core (libibverbs)** - Referenced but not linked (standard RDMA API):
- `ibv_post_send()` - Post send work request ([verbs.h:2554](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L2554))
- `ibv_post_recv()` - Post receive work request ([verbs.h:2562](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L2562))
- `ibv_poll_cq()` - Poll completion queue ([verbs.h:2576](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L2576))

### Structures

**NCCL Core (External - NVIDIA)** - Referenced but not linked:
- `ncclNet_t` - Network plugin interface ([net.h](https://github.com/NVIDIA/nccl/blob/master/src/include/net.h))
- `ncclDataType_t` - Data type enumeration
- `ncclRedOp_t` - Reduction operation enumeration

**libfabric (OFI)** - See libfabric-overview.md for detailed struct permalinks:
- `struct fid_ep` - Endpoint handle
- `struct fid_cq` - Completion queue handle
- `struct fid_mr` - Memory region handle

**rdma-core (libibverbs)** - See rdma-core-and-verbs.md for detailed struct permalinks:
- `struct ibv_qp` - Queue pair
- `struct ibv_cq` - Completion queue
- `struct ibv_send_wr` - Send work request
- `struct ibv_recv_wr` - Receive work request
- `struct ibv_wc` - Work completion

### Configuration Parameters

**NCCL Environment Variables**:
- `NCCL_ALGO` - Force specific algorithm (Ring, Tree, etc.)
- `NCCL_PROTO` - Force specific protocol (Simple, LL, LL128)
- `NCCL_NCHANNELS` - Number of channels (default: auto-detected)
- `NCCL_BUFFSIZE` - Channel buffer size (default: 4 MB)
- `NCCL_NET_GDR_LEVEL` - GPUDirect level (0-5)
- `NCCL_IB_GID_INDEX` - GID index for RDMA
- `NCCL_DEBUG` - Debug verbosity (WARN, INFO, TRACE)

**OFI Plugin Environment Variables**:
- `OFI_NCCL_CUDA_FLUSH_ENABLE` - Enable CUDA GPUDirect flush
- `FI_EFA_USE_DEVICE_RDMA` - Enable native RDMA on EFA Gen3+

### Total Code References
- **6 NCCL functions** (external NVIDIA)
- **5 libfabric functions** (standard OFI API)
- **3 rdma-core functions** (standard verbs API)
- **3 NCCL structures** (external NVIDIA)
- **3 libfabric structures** (see libfabric-overview.md)
- **4 rdma-core structures** (see rdma-core-and-verbs.md)
- **7 NCCL environment variables**
- **2 OFI plugin environment variables**

**Cross-References**:
- For libfabric struct details: [libfabric-overview.md](libfabric-overview.md)
- For rdma-core struct details: [rdma-core-and-verbs.md](rdma-core-and-verbs.md)
- For EFA hardware descriptors: [efa-hardware-architecture.md](efa-hardware-architecture.md)
- For OFI plugin implementation: [ofi-plugin.md](ofi-plugin.md)
