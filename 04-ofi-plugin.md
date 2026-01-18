# OFI NCCL Plugin Architecture

## Overview

The OFI (OpenFabrics Interfaces) NCCL plugin (`aws-ofi-nccl`) provides a network transport for NCCL using libfabric. It implements NCCL's network plugin interface (`ncclNet_t`) and is the primary way NCCL communicates over AWS EFA.

**Repository**: https://github.com/aws/aws-ofi-nccl

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

```c
ncclResult_t nccl_net_ofi_init(ncclDebugLogger_t logFunction)
{
  // Initialize libfabric
  fi_getinfo(..., &ofi_info);

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

```c
ncclResult_t nccl_net_ofi_listen(int dev, void* handle,
                                 void** listenComm)
{
  // Create endpoint
  fi_endpoint(domain, info, &ep, NULL);

  // Bind to completion queue
  fi_ep_bind(ep, cq, 0);

  // Enable endpoint
  fi_enable(ep);

  // Get local address
  fi_getname(ep, &local_addr, &addrlen);

  // Return address in handle (for sharing)
  memcpy(handle, &local_addr, addrlen);

  *listenComm = comm_state;
  return ncclSuccess;
}
```

#### Connector (Active Side)

```c
ncclResult_t nccl_net_ofi_connect(int dev, void* handle,
                                  void** sendComm)
{
  // Create endpoint
  fi_endpoint(domain, info, &ep, NULL);

  // Bind to CQ
  fi_ep_bind(ep, cq, 0);
  fi_enable(ep);

  // Extract remote address from handle
  memcpy(&remote_addr, handle, addrlen);

  // Store for future sends
  comm->remote_addr = fi_av_insert(av, &remote_addr, 1, ...);

  *sendComm = comm_state;
  return ncclSuccess;
}
```

#### Accept (Complete Passive Connection)

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
    ret = fi_mr_reg(domain, data, size, access,
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

**Cache Benefits:**
- Avoid expensive re-registration
- Critical for performance (100+ μs saved)
- Reference counting for shared regions

### Send Operations

```c
ncclResult_t nccl_net_ofi_isend(void* sendComm, void* data,
                                int size, int tag,
                                void* mhandle, void** request)
{
  struct nccl_ofi_send_comm* comm = sendComm;
  struct fid_mr* mr = mhandle;

  // Allocate request tracking
  struct nccl_ofi_req* req = get_request();
  req->comm = comm;
  req->size = size;
  req->state = REQ_PENDING;

  // Get memory descriptor
  void* desc = fi_mr_desc(mr);

  // Post send operation
  // Use fi_tsend for tagged messaging (tag identifies channel/operation)
  ret = fi_tsend(comm->ep, data, size, desc,
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

```c
ncclResult_t nccl_net_ofi_irecv(void* recvComm, int n,
                                void** data, int* sizes,
                                int* tags, void** mhandles,
                                void** request)
{
  struct nccl_ofi_recv_comm* comm = recvComm;

  // NCCL can post multiple receives at once (up to n)
  struct nccl_ofi_req* req = get_request();
  req->nrecvs = n;

  for (int i = 0; i < n; i++) {
    struct fid_mr* mr = mhandles[i];
    void* desc = fi_mr_desc(mr);

    // Pre-post receives
    ret = fi_trecv(comm->ep, data[i], sizes[i], desc,
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

```c
ncclResult_t nccl_net_ofi_test(void* request, int* done,
                               int* size)
{
  struct nccl_ofi_req* req = request;

  // Poll completion queue
  struct fi_cq_data_entry cq_entry;
  ret = fi_cq_read(comm->cq, &cq_entry, 1);

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
  struct fi_cq_err_entry err_entry;
  if (fi_cq_readerr(comm->cq, &err_entry, 0) > 0) {
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

```c
ncclResult_t nccl_net_ofi_deregMr(void* comm, void* mhandle)
{
  struct fid_mr* mr = mhandle;

  // Decrement refcount in cache
  if (decref_cache(mr) == 0) {
    // Last reference, actually deregister
    fi_close(&mr->fid);
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

```
GPU 0 ──┬─→ EFA 0 ─→ Network
        └─→ EFA 1 ─→ Network

GPU 1 ──┬─→ EFA 0 ─→ Network
        └─→ EFA 1 ─→ Network
```

**Implementation:**
- Expose each EFA as separate device to NCCL
- NCCL distributes channels across devices
- Aggregate bandwidth increases

**Configuration:**
```bash
# Automatic detection (default)
# Or explicitly select
NCCL_NET_GDR_LEVEL=LOC  # GPU-local NICs preferred
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
  fi_inject(ep, data, size, remote_addr);
} else {
  // Standard fi_send
  fi_send(ep, data, size, desc, remote_addr, context);
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
fi_getinfo()      // Query providers
fi_fabric()       // Create fabric
fi_domain()       // Create domain
fi_endpoint()     // Create endpoint
fi_cq_open()      // Create completion queue
fi_av_open()      // Create address vector

// Memory
fi_mr_reg()       // Register memory
fi_mr_desc()      // Get descriptor

// Data transfer
fi_send()         // Send message
fi_tsend()        // Tagged send
fi_recv()         // Receive message
fi_trecv()        // Tagged receive
fi_read()         // RDMA read
fi_write()        // RDMA write
fi_inject()       // Eager send

// Completion
fi_cq_read()      // Poll completions
fi_cq_readerr()   // Read error

// Cleanup
fi_close()        // Close resources
```

### Endpoint Configuration

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
