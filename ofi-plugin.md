# OFI NCCL Plugin Architecture

## Overview

The OFI (OpenFabrics Interfaces) NCCL plugin (`aws-ofi-nccl`) provides a network transport for NCCL using libfabric. It implements NCCL's network plugin interface (`ncclNet_t`) and is the primary way NCCL communicates over AWS EFA.

**Repository**: https://github.com/aws/aws-ofi-nccl

## AWS Instance Types and Protocol Support

The OFI plugin supports different communication protocols based on instance capabilities:

### P4 Series (A100 GPUs)

| Instance | GPUs | EFA Config | Total BW | Protocol | Notes |
|----------|------|------------|----------|----------|-------|
| p4d.24xlarge | 8×A100 | 4×100G | 400 Gbps | **Send/Recv only** | No native RDMA write |
| p4de.24xlarge | 8×A100 | 4×100G | 400 Gbps | **Send/Recv only** | No native RDMA write |

### P5 Series (H100/H200 GPUs)

| Instance | GPUs | EFA Config | Total BW | Per-GPU BW | Protocol | Latency | Notes |
|----------|------|------------|----------|------------|----------|---------|-------|
| p5.48xlarge | 8×H100 | 32 EFAs | 3200 Gbps | 400 Gbps | **RDMA + Send/Recv** | 75 μs | EFAv2 |
| p5e.48xlarge | 8×H100 | 32 EFAs | 3200 Gbps | 400 Gbps | **RDMA + Send/Recv** | 75 μs | EFAv2 |
| p5en.48xlarge | 8×H200 | 32 EFAs | 3200 Gbps | 400 Gbps | **RDMA + Send/Recv** | 35 μs | **EFAv3**, 35% lower latency |

**Key Differences:**
- **p4d/p4de**: Use send/recv protocol only (tagged messaging). RDMA write is emulated, not native.
- **p5/p5e/p5en**: Support native RDMA write for improved performance. Plugin defaults to RDMA protocol.
- **Multi-rail**: Each GPU can use 4 EFA adapters (p4d: 4×100G, p5+: sharing 32 total EFAs across 8 GPUs)

**Protocol Selection** ([src/platform-aws.cpp](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/platform-aws.cpp#L81-L259)):
```cpp
// P4d/P4de: default_protocol = PROTOCOL::SENDRECV
// P5/P5e/P5en: default_protocol = PROTOCOL::RDMA
```

## Architecture

```
┌───────────────────────────────────────────────┐
│              NCCL Core                        │
└───────────────────┬───────────────────────────┘
                    │ ncclNet_t API
┌───────────────────▼───────────────────────────┐
│          OFI NCCL Plugin                      │
│  ┌─────────────────────────────────────┐     │
│  │  Connection Management              │     │
│  ├─────────────────────────────────────┤     │
│  │  Memory Registration Cache          │     │
│  ├─────────────────────────────────────┤     │
│  │  Send/Recv Queueing                 │     │
│  ├─────────────────────────────────────┤     │
│  │  Completion Handling                │     │
│  ├─────────────────────────────────────┤     │
│  │  Multi-rail Support                 │     │
│  └─────────────────────────────────────┘     │
└───────────────────┬───────────────────────────┘
                    │ libfabric API
┌───────────────────▼───────────────────────────┐
│            Libfabric                          │
│              EFA Provider                     │
└───────────────────────────────────────────────┘
```

## ncclNet_t Interface Implementation

The plugin implements all functions in the `ncclNet_t` interface:

### Initialization

**`nccl_net_ofi_init()`** ([src/nccl_ofi_net.c:1849](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_net.c#L1849)):

```c
ncclResult_t nccl_net_ofi_init(ncclDebugLogger_t logFunction)
{
  // Initialize libfabric
  fi_getinfo(..., &ofi_info);  // See fi_getinfo() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fabric.h#L315)

  // Discover available devices (EFA adapters)
  // Setup domain, fabric, etc.

  // Initialize plugin state
  return ncclSuccess;
}
```

**Actions:**
- Query libfabric for EFA providers
- Enumerate EFA devices (NICs)
- Create plugin-global state
- Setup logging

### Device Discovery

**`nccl_net_ofi_devices()` / `nccl_net_ofi_getProperties()`** ([src/nccl_ofi_net.c:1903](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_net.c#L1903), [src/nccl_ofi_net.c:1911](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_net.c#L1911)):

```c
ncclResult_t nccl_net_ofi_devices(int* ndev)
{
  *ndev = num_efa_devices;
  return ncclSuccess;
}

ncclResult_t nccl_net_ofi_getProperties(int dev,
                                        ncclNetProperties_t* props)
{
  props->name = "AWS EFA";
  props->pciPath = pci_path[dev];
  props->guid = device_guid[dev];
  props->ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
  props->speed = 100000; // 100 Gbps
  props->port = 0;
  props->maxComms = MAX_COMMS;
  return ncclSuccess;
}
```

**Properties:**
- **pciPath**: For topology detection
- **ptrSupport**: GPU direct support
- **speed**: Link speed (100 Gbps for EFA)
- **maxComms**: Concurrent connections limit

### Connection Establishment

#### Listener (Passive Side)

**`nccl_net_ofi_listen()`** ([src/nccl_ofi_net.c:1262](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_net.c#L1262)):

```c
ncclResult_t nccl_net_ofi_listen(int dev, void* handle,
                                 void** listenComm)
{
  // Create endpoint
  fi_endpoint(domain, info, &ep, NULL);  // See fi_endpoint() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_endpoint.h#L182)

  // Bind to completion queue
  fi_ep_bind(ep, cq, 0);  // See fi_ep_bind() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_endpoint.h#L203)

  // Enable endpoint
  fi_enable(ep);  // See fi_enable() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_endpoint.h#L211)

  // Get local address
  fi_getname(ep, &local_addr, &addrlen);  // See fi_getname() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_cm.h#L154)

  // Return address in handle (for sharing)
  memcpy(handle, &local_addr, addrlen);

  *listenComm = comm_state;
  return ncclSuccess;
}
```

#### Connector (Active Side)

**`nccl_net_ofi_connect()`** ([src/nccl_ofi_net.c:1365](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_net.c#L1365)):

```c
ncclResult_t nccl_net_ofi_connect(int dev, void* handle,
                                  void** sendComm)
{
  // Create endpoint
  fi_endpoint(domain, info, &ep, NULL);  // See fi_endpoint() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_endpoint.h#L182)

  // Bind to CQ
  fi_ep_bind(ep, cq, 0);
  fi_enable(ep);

  // Extract remote address from handle
  memcpy(&remote_addr, handle, addrlen);

  // Store for future sends
  comm->remote_addr = fi_av_insert(av, &remote_addr, 1, ...);  // See fi_av_insert() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_cm.h#L206)

  *sendComm = comm_state;
  return ncclSuccess;
}
```

#### Accept (Complete Passive Connection)

**`nccl_net_ofi_accept()`** ([src/nccl_ofi_net.c:1537](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_net.c#L1537)):

```c
ncclResult_t nccl_net_ofi_accept(void* listenComm,
                                 void** recvComm)
{
  // Connection already established (libfabric is connectionless for EFA)
  // Just return state
  *recvComm = comm_state;
  return ncclSuccess;
}
```

**Note**: EFA uses connectionless (UD-like) transport, so "connection" is mostly state tracking.

### Memory Registration

**`nccl_net_ofi_regMr()`** ([src/nccl_ofi_net.c:724](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_net.c#L724)):

```c
ncclResult_t nccl_net_ofi_regMr(void* comm, void* data,
                                int size, int type,
                                void** mhandle)
{
  // Check cache first
  if (find_in_cache(data, size, &mr)) {
    *mhandle = mr;
    return ncclSuccess;
  }

  // Register with libfabric
  uint64_t access = FI_SEND | FI_RECV | FI_REMOTE_READ | FI_REMOTE_WRITE;

  if (type == NCCL_PTR_CUDA) {
    // GPU memory
    ret = fi_mr_reg(domain, data, size, access,  // See fi_mr_reg() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L413)
                    0, 0, FI_HMEM_CUDA, &mr, NULL);
  } else {
    // Host memory
    ret = fi_mr_reg(domain, data, size, access,
                    0, 0, 0, &mr, NULL);
  }

  // Add to cache
  add_to_cache(data, size, mr);

  *mhandle = mr;
  return ncclSuccess;
}
```

**Registration Cache:**
```c
struct mr_cache {
  void* addr;
  size_t size;
  struct fid_mr* mr;
  int refcount;
  struct mr_cache* next;
};
```

(See `struct fid_mr` ([include/rdma/fi_domain.h:131-138](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L131-L138)))

**Cache Benefits:**
- Avoid expensive re-registration
- Critical for performance (100+ μs saved)
- Reference counting for shared regions

### Send Operations

**`nccl_net_ofi_isend()`** ([src/nccl_ofi_net.c:1054](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_net.c#L1054)):

```c
ncclResult_t nccl_net_ofi_isend(void* sendComm, void* data,
                                int size, int tag,
                                void* mhandle, void** request)
{
  struct nccl_ofi_send_comm* comm = sendComm;  // See struct nccl_net_ofi_send_comm (https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/include/nccl_ofi.h#L869-L901)
  struct fid_mr* mr = mhandle;

  // Allocate request tracking
  struct nccl_ofi_req* req = get_request();  // See struct nccl_net_ofi_req (https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/include/nccl_ofi.h#L132-L134)
  req->comm = comm;
  req->size = size;
  req->state = REQ_PENDING;

  // Get memory descriptor
  void* desc = fi_mr_desc(mr);  // See fi_mr_desc() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L147)

  // Post send operation
  // Use fi_tsend for tagged messaging (tag identifies channel/operation)
  ret = fi_tsend(comm->ep, data, size, desc,  // See fi_tsend() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_tagged.h#L121)
                 comm->remote_addr, tag, req);

  if (ret == -FI_EAGAIN) {
    // Queue is full, retry later
    queue_request(req);
  }

  *request = req;
  return ncclSuccess;
}
```

**Tagged Send:**
- Tag identifies message type/channel
- Allows multiple outstanding operations
- Receiver matches by tag

### Receive Operations

**`nccl_net_ofi_irecv()`** ([src/nccl_ofi_net.c:1132](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_net.c#L1132)):

```c
ncclResult_t nccl_net_ofi_irecv(void* recvComm, int n,
                                void** data, int* sizes,
                                int* tags, void** mhandles,
                                void** request)
{
  struct nccl_ofi_recv_comm* comm = recvComm;  // See struct nccl_net_ofi_recv_comm (https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/include/nccl_ofi.h#L903-L935)

  // NCCL can post multiple receives at once (up to n)
  struct nccl_ofi_req* req = get_request();
  req->nrecvs = n;

  for (int i = 0; i < n; i++) {
    struct fid_mr* mr = mhandles[i];
    void* desc = fi_mr_desc(mr);

    // Pre-post receives
    ret = fi_trecv(comm->ep, data[i], sizes[i], desc,  // See fi_trecv() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_tagged.h#L98)
                   FI_ADDR_UNSPEC, tags[i], 0, req);

    if (ret < 0) {
      // Handle error
    }
  }

  *request = req;
  return ncclSuccess;
}
```

**Pre-posting:**
- Receives posted before data arrives
- Matched when sender transmits
- Reduces latency

### Flush Operations (RDMA Write Completion)

**`nccl_net_ofi_iflush()`** ([src/nccl_ofi_net.c:1176](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_net.c#L1176)):

```c
ncclResult_t nccl_net_ofi_iflush(void* recvComm, int n,
                                 void** data, int* sizes,
                                 void** mhandles, void** request)
{
  // For RDMA write semantics
  // Ensures remote writes are visible

  struct nccl_ofi_req* req = get_request();

  // EFA: Issue read operation to ensure ordering
  // (Write followed by read ensures write completion)
  for (int i = 0; i < n; i++) {
    fi_read(comm->ep, flush_buf, 0, NULL,
            comm->remote_addr, 0, 0, req);
  }

  *request = req;
  return ncclSuccess;
}
```

**Why Flush:**
- RDMA writes are one-sided (no remote notification)
- Flush ensures write is visible to receiver
- EFA: Use dummy read for ordering

### Completion Testing

**`nccl_net_ofi_test()`** ([src/nccl_ofi_net.c:991](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_net.c#L991)):

```c
ncclResult_t nccl_net_ofi_test(void* request, int* done,
                               int* size)
{
  struct nccl_ofi_req* req = request;

  // Poll completion queue
  struct fi_cq_data_entry cq_entry;  // See struct fi_cq_data_entry (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L215-L220)
  ret = fi_cq_read(comm->cq, &cq_entry, 1);  // See fi_cq_read() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L311)

  if (ret > 0) {
    // Match completion to request
    struct nccl_ofi_req* comp_req = cq_entry.op_context;

    if (comp_req == req) {
      *done = 1;
      *size = cq_entry.len;
      free_request(req);
      return ncclSuccess;
    }
  } else if (ret == -FI_EAGAIN) {
    // No completions yet
    *done = 0;
    return ncclSuccess;
  }

  // Check for errors
  struct fi_cq_err_entry err_entry;  // See struct fi_cq_err_entry (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L233-L246)
  if (fi_cq_readerr(comm->cq, &err_entry, 0) > 0) {  // See fi_cq_readerr() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L348)
    // Handle error
    return ncclSystemError;
  }

  *done = 0;
  return ncclSuccess;
}
```

**Polling Model:**
- NCCL proxy threads continuously poll
- Non-blocking (returns immediately)
- Batches completions for efficiency

### Cleanup

**`nccl_net_ofi_deregMr()` / `nccl_net_ofi_close()`** ([src/nccl_ofi_net.c:796](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_net.c#L796), [src/nccl_ofi_net.c:1809](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_net.c#L1809)):

```c
ncclResult_t nccl_net_ofi_deregMr(void* comm, void* mhandle)
{
  struct fid_mr* mr = mhandle;

  // Decrement refcount in cache
  if (decref_cache(mr) == 0) {
    // Last reference, actually deregister
    fi_close(&mr->fid);  // See fi_close() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fabric.h#L150)
    remove_from_cache(mr);
  }

  return ncclSuccess;
}

ncclResult_t nccl_net_ofi_close(void* comm)
{
  // Close endpoint
  fi_close(&comm->ep->fid);

  // Close CQ
  fi_close(&comm->cq->fid);

  // Free state
  free(comm);

  return ncclSuccess;
}
```

## Advanced Features

### Multi-Rail Support

EFA instances can have multiple NICs. The plugin supports using all available NICs:

#### p4d/p4de Configuration (4×100G EFAs per instance)

Each GPU uses all 4 EFAs, distributed across the instance:

```
Instance: 8 GPUs, 4 EFAs (100 Gbps each)
GPU 0 ──┬─→ EFA 0 (100G) ─→ Network
        ├─→ EFA 1 (100G) ─→ Network
        ├─→ EFA 2 (100G) ─→ Network
        └─→ EFA 3 (100G) ─→ Network
                          = 400 Gbps total / 8 GPUs = 50 GB/s per GPU

GPU 1-7: Same 4 EFAs shared across all GPUs
```

#### p5/p5e/p5en Configuration (32 EFAs per instance)

32 total EFAs provide 3200 Gbps, with 4 EFAs effectively allocated per GPU:

```
Instance: 8 GPUs, 32 EFAs (100 Gbps each)
GPU 0 ──┬─→ EFA 0-3  (4×100G) ─→ Network
GPU 1 ──┬─→ EFA 4-7  (4×100G) ─→ Network
GPU 2 ──┬─→ EFA 8-11 (4×100G) ─→ Network
...
GPU 7 ──┬─→ EFA 28-31 (4×100G) ─→ Network
                          = 400 Gbps per GPU = 50 GB/s per GPU
```

#### Alternative Multi-Rail Configurations

Modern instances can use different EFA aggregations:

| Config | EFAs/GPU | Per-EFA Speed | Total per GPU | Use Case |
|--------|----------|---------------|---------------|----------|
| 4×100G | 4 | 100 Gbps | 400 Gbps | p4d, p5 standard |
| 2×200G | 2 | 200 Gbps | 400 Gbps | Future instances |
| 1×400G | 1 | 400 Gbps | 400 Gbps | High-radix topologies |
| 2×200G | 2 | 200 Gbps | 400 Gbps per GPU | Reduced NIC count, simpler topology |

**Implementation:**
- Expose each EFA as separate device to NCCL
- NCCL distributes channels across devices
- Aggregate bandwidth increases linearly with NICs
- Plugin handles rail reordering for optimal pairing ([src/platform-aws.cpp:983](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/platform-aws.cpp#L983))

**Configuration:**
```bash
# Automatic detection (default)
# Or explicitly select
NCCL_NET_GDR_LEVEL=LOC  # GPU-local NICs preferred
NCCL_NCHANNELS=8       # Channels distributed across rails
```

### GPU Direct Support

The plugin supports direct GPU memory registration:

```c
// CUDA memory
fi_mr_reg(domain, gpu_buf, size, access,
          0, 0, FI_HMEM_CUDA, &mr, NULL);
```

**Benefits:**
- Zero-copy: NIC reads directly from GPU memory
- No staging through CPU memory
- Lower latency, higher bandwidth

**Requirements:**
- EFA with GPUDirect support
- Libfabric compiled with CUDA support
- Proper IOMMU configuration

### Request Pipelining

The plugin maintains multiple outstanding requests:

```c
#define MAX_REQUESTS 256

struct req_pool {
  struct nccl_ofi_req requests[MAX_REQUESTS];
  int head, tail;
};
```

**Benefits:**
- Hide latency via pipelining
- Keep network saturated
- Better throughput

### Eager Protocol Optimization

For small messages, the plugin can use eager send:

```c
if (size < EAGER_THRESHOLD) {
  // Send immediately, no pre-posted recv needed
  fi_inject(ep, data, size, remote_addr);  // See fi_inject() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_msg.h#L133)
} else {
  // Standard fi_send
  fi_send(ep, data, size, desc, remote_addr, context);  // See fi_send() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_msg.h#L104)
}
```

**fi_inject:**
- Copies data to internal buffer
- Returns immediately
- No completion event
- Limited to small sizes (< 4 KB on EFA)

## Libfabric API Usage

### Core APIs Used

```c
// Initialization
fi_getinfo()      // Query providers (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fabric.h#L315)
fi_fabric()       // Create fabric (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fabric.h#L334)
fi_domain()       // Create domain (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L409)
fi_endpoint()     // Create endpoint (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_endpoint.h#L182)
fi_cq_open()      // Create completion queue (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L363)
fi_av_open()      // Create address vector (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_cm.h#L218)

// Memory
fi_mr_reg()       // Register memory (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L413)
fi_mr_desc()      // Get descriptor (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L147)

// Data transfer
fi_send()         // Send message (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_msg.h#L104)
fi_tsend()        // Tagged send (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_tagged.h#L121)
fi_recv()         // Receive message (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_msg.h#L67)
fi_trecv()        // Tagged receive (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_tagged.h#L98)
fi_read()         // RDMA read (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_rma.h#L82)
fi_write()        // RDMA write (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_rma.h#L111)
fi_inject()       // Eager send (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_msg.h#L133)

// Completion
fi_cq_read()      // Poll completions (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L311)
fi_cq_readerr()   // Read error (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L348)

// Cleanup
fi_close()        // Close resources (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fabric.h#L150)
```

### Endpoint Configuration

(See `struct fi_info` ([include/rdma/fabric.h:198-232](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fabric.h#L198-L232)))

```c
struct fi_info hints = {
  .caps = FI_MSG | FI_RMA | FI_TAGGED,
  .mode = FI_CONTEXT,
  .domain_attr = {
    .mr_mode = FI_MR_LOCAL | FI_MR_ALLOCATED,
    .threading = FI_THREAD_SAFE,
  },
  .ep_attr = {
    .type = FI_EP_RDM,  // Reliable datagram
  },
  .tx_attr = {
    .size = 256,  // TX queue depth
  },
  .rx_attr = {
    .size = 256,  // RX queue depth
  },
};
```

**EFA-Specific:**
- **FI_EP_RDM**: Reliable datagram endpoint (EFA's native mode)
- **FI_TAGGED**: For message matching
- **FI_RMA**: For RDMA operations

## Error Handling

### Completion Errors

```c
struct fi_cq_err_entry err;
if (fi_cq_readerr(cq, &err, 0) > 0) {
  switch (err.err) {
    case FI_ETIMEDOUT:
      // Network timeout
      break;
    case FI_ECANCELED:
      // Operation canceled
      break;
    case FI_EMSGSIZE:
      // Message too large
      break;
    // ... handle other errors
  }
}
```

### Retry Logic

```c
int retry_count = 0;
while (retry_count < MAX_RETRIES) {
  ret = fi_send(...);

  if (ret == 0) break;  // Success

  if (ret == -FI_EAGAIN) {
    // Resource temporarily unavailable
    // Progress CQ and retry
    fi_cq_read(cq, &entry, 1);
    retry_count++;
    continue;
  }

  // Permanent error
  return ncclSystemError;
}
```

## Performance Tuning

### Environment Variables

```bash
# Plugin-specific
NCCL_OFI_USE_GDRCOPY=1           # Enable GDR copy
NCCL_OFI_NCCL_VERSION_CHECK=0    # Disable version check

# General NCCL + OFI
FI_EFA_ENABLE_SHM_TRANSFER=0     # Disable SHM (use NCCL's SHM)
FI_EFA_TX_SIZE=256               # TX queue depth
FI_EFA_RX_SIZE=256               # RX queue depth
```

### Memory Registration Optimization

```bash
# Increase cache size
FI_MR_CACHE_MAX_SIZE=unlimited
FI_MR_CACHE_MAX_COUNT=unlimited

# Monitor cache hits
FI_LOG_LEVEL=info
```

### Completion Queue Tuning

```c
// Larger CQ for more pipelining
struct fi_cq_attr cq_attr = {
  .size = 4096,  // Default is smaller
  .format = FI_CQ_FORMAT_DATA,
};
```

## Summary

**OFI Plugin Responsibilities:**
1. **Interface Translation**: NCCL ↔ Libfabric
2. **Connection Management**: Setup, teardown
3. **Memory Registration**: With caching
4. **Message Passing**: Tagged send/recv
5. **Completion Handling**: Polling, error handling
6. **Multi-rail**: Utilize all NICs
7. **GPU Direct**: Zero-copy transfers

**Key Design Patterns:**
- **Stateful Connections**: Track per-connection state
- **Request Pooling**: Reuse request objects
- **MR Caching**: Avoid expensive re-registration
- **Async Operations**: All ops are non-blocking
- **Polling Model**: Continuous progress via proxy threads

**Performance Critical Paths:**
- Memory registration cache hit rate
- Completion queue polling efficiency
- Request allocation/free overhead
- Tag matching performance

**Next**: Understanding libfabric's EFA provider implementation for deeper optimization.
