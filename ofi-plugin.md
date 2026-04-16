# OFI NCCL Plugin Architecture

## Overview

The OFI (OpenFabrics Interfaces) NCCL plugin (`aws-ofi-nccl`) provides a network transport for NCCL using libfabric. It implements both NCCL's network plugin interface (`ncclNet_v11_t`) and the GIN plugin interface (`ncclGin_v11_t`), and is the primary way NCCL communicates over AWS EFA.

**Repository**: https://github.com/aws/aws-ofi-nccl

The plugin is written in C++ and organized as a class hierarchy
(`plugin_t` → `device_t` → `domain_t` → `ep_t` → communicators) with
`shared_ptr`/`weak_ptr` ownership for automatic resource management.
It supports three transports: RDMA (P5+), SendRecv (P4d), and GIN
(wraps RDMA for one-sided put operations used by DeepEP/MoE workloads).

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

### P6 Series and Beyond (B200/GB200 GPUs)

| Instance | GPUs | Protocol | Latency | Notes |
|----------|------|----------|---------|-------|
| p6-b200 | 8×B200 | **RDMA** | 35 μs | EFAv3, NETDEVS_POLICY=max:1 |
| p6e-gb200+ | GB200+ | **RDMA** | 35 μs | Catch-all for future P-series (regex `^p([5-9]\|[0-9]{2,}).*`) |

### Other EFA-Capable Instances

| Instance | GPUs | Protocol | Latency | Notes |
|----------|------|----------|---------|-------|
| g7e (12xl+) | 1-4×L40S | **RDMA** | 35 μs | Inference workloads |
| trn1/trn2 | Trainium | **RDMA** | 75 μs | AWS Neuron |

**Key Differences:**
- **p4d/p4de**: Use send/recv protocol only (tagged messaging). RDMA write is emulated, not native.
- **p5/p5e**: Support native RDMA write. Plugin defaults to RDMA protocol. EFAv2, 75 μs latency.
- **p5en/p6**: EFAv3 with 35 μs latency. Uses `NCCL_NETDEVS_POLICY=max:1` for optimal GPU-NIC pairing.
- **p6e-gb200+**: Catch-all regex matches future P-series instances with same RDMA defaults.
- **Multi-rail**: Each GPU can use 4 EFA adapters (p4d: 4×100G, p5+: sharing 32 total EFAs across 8 GPUs)

**Protocol Selection** ([src/platform-aws.cpp:81-250](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/platform-aws.cpp#L81-L250)):
```cpp
// P4d/P4de: default_protocol = PROTOCOL::SENDRECV
// P5/P5e:   default_protocol = PROTOCOL::RDMA     (latency = 75.0)
// P5en/P6:  default_protocol = PROTOCOL::RDMA     (latency = 35.0)
// P6e-GB200+: default_protocol = PROTOCOL::RDMA   (catch-all for future P-series)
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

The plugin implements all functions in the `ncclNet_v11_t` interface (previously `ncclNet_t`).
The API layer is in `src/nccl_ofi_api.cpp`, which dispatches to transport-specific
implementations in `src/nccl_ofi_rdma.cpp` (RDMA) or `src/nccl_ofi_sendrecv.cpp` (SendRecv).

### Initialization

**`nccl_net_ofi_init()`** ([src/nccl_ofi_api.cpp:58](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_api.cpp#L58)):

```cpp
ncclResult_t nccl_net_ofi_init(ncclDebugLogger_t logFunction)
{
  // Initialize environment variable system
  ofi_nccl_parameters_init();

  // Create plugin: queries libfabric (fi_getinfo), discovers EFA adapters,
  // builds topology, selects transport (RDMA or SendRecv based on platform)
  nccl_net_ofi_create_plugin(&plugin);  // See src/nccl_ofi_net.cpp

  // Plugin probes each device by creating a test endpoint
  // (get_ep → creates domain + ep, then releases)
  plugin->complete_init();

  return ncclSuccess;
}
```

**Actions:**
- Initialize environment variable system (`ofi_nccl_parameters_init`)
- Query libfabric for EFA providers (`fi_getinfo`)
- Build topology from discovered NICs
- Select transport (RDMA or SendRecv) based on platform type
- Create plugin object with device array
- Probe each device by creating a test endpoint
- Setup logging

### Device Discovery

**`nccl_net_ofi_devices()` / `nccl_net_ofi_get_properties()`** ([src/nccl_ofi_api.cpp:111](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_api.cpp#L111), [src/nccl_ofi_api.cpp:129](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_api.cpp#L129)):

```cpp
ncclResult_t nccl_net_ofi_devices(int *num_devices)
{
  *num_devices = plugin->get_num_devices();
  return ncclSuccess;
}

ncclResult_t nccl_net_ofi_get_properties(int dev_id,
                                         nccl_ofi_properties_t *ofi_properties)
{
  // Get device object from plugin (C++ class hierarchy)
  nccl_net_ofi_device_t *device = plugin->get_device(dev_id);

  // Virtual dispatch to transport-specific implementation
  // (RDMA or SendRecv device overrides get_properties)
  int ret = device->get_properties(ofi_properties);
  return nccl_net_ofi_retval_translate_impl(ret);
}
```

**Properties:**
- **pciPath**: For topology detection
- **ptrSupport**: GPU direct support
- **speed**: Link speed (100 Gbps for EFA)
- **maxComms**: Concurrent connections limit

### Connection Establishment

Connection establishment uses the Connection Manager (CM) module (`src/cm/`).
The process is a multi-call state machine — NCCL calls connect/accept
repeatedly until the connection is established.

#### Listener (Passive Side)

**`nccl_net_ofi_listen()`** ([src/nccl_ofi_api.cpp:150](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_api.cpp#L150)):

```cpp
ncclResult_t nccl_net_ofi_listen(int dev_id, void *handle,
                                 void **lComm,
                                 unsigned int domain_key,
                                 unsigned int resource_key)
{
  nccl_net_ofi_device_t *device = plugin->get_device(dev_id);

  // Get or create endpoint via shared_ptr (lazy creation)
  // domain_key selects the domain, tid selects the endpoint within it
  std::shared_ptr<nccl_net_ofi_ep_t> ep =
      device->get_ep(domain_key, nccl_net_ofi_gettid());

  // Virtual dispatch to transport-specific listen
  // (creates listen_comm, gets local address via fi_getname)
  ret = ep->listen(handle, listen_comm);

  // Store shared_ptr<ep> on listen_comm to keep endpoint alive
  (*listen_comm)->ep = ep;
}
```

Inside the transport-specific `ep->listen()`, the following libfabric calls happen:

```cpp
  // Create endpoint (if not already created for this thread)
  fi_endpoint(domain, info, &ep, NULL);  // See fi_endpoint() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_endpoint.h#L182)

  // Bind to completion queue
  fi_ep_bind(ep, cq, 0);  // See fi_ep_bind() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_endpoint.h#L203)

  // Enable endpoint
  fi_enable(ep);  // See fi_enable() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_endpoint.h#L211)

  // Get local address
  fi_getname(ep, &local_addr, &addrlen);  // See fi_getname() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_cm.h#L154)

  // Return address in handle (for sharing via NCCL bootstrap)
  memcpy(handle, &local_addr, addrlen);

  *listenComm = comm_state;
  return ncclSuccess;
}
```

#### Connector (Active Side)

**`nccl_net_ofi_connect()`** ([src/nccl_ofi_api.cpp:220](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_api.cpp#L220)):

This is a **multi-call state machine**. NCCL calls connect() repeatedly until
`sendComm != NULL`. On the first call, the endpoint and send_comm are created.
On subsequent calls, the existing comm is retrieved from the handle's state.

```cpp
ncclResult_t nccl_net_ofi_connect(int dev_id, void *handle,
                                  void **sComm, int trafficClass,
                                  unsigned int domain_key,
                                  unsigned int resource_key)
{
  nccl_net_ofi_conn_handle_t *ofi_handle = (nccl_net_ofi_conn_handle_t *)handle;

  std::shared_ptr<nccl_net_ofi_ep_t> ep;

  if (ofi_handle->state.comm == nullptr) {
    // First call: get or create endpoint
    nccl_net_ofi_device_t *device = plugin->get_device(dev_id);
    ep = device->get_ep(domain_key, nccl_net_ofi_gettid());
  } else {
    // Subsequent calls: reuse ep from the comm created on first call
    ep = ofi_handle->state.comm->ep;
  }

  // Virtual dispatch to transport-specific connect
  // (CM sends connect request, waits for response)
  ret = ep->connect(handle, send_comm, trafficClass);

  // ep shared_ptr drops on scope exit. No manual release needed —
  // the comm holds its own shared_ptr<ep> to keep the endpoint alive.
}
```

Inside the transport-specific `ep->connect()`:
```cpp
  // Extract remote address from handle
  memcpy(&remote_addr, handle, addrlen);

  // Insert remote address into address vector
  fi_av_insert(av, &remote_addr, 1, ...);  // See fi_av_insert() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_cm.h#L206)

  // CM: send connect request message to peer
  // CM: wait for connect response (may return incomplete on first call)
```

#### Accept (Complete Passive Connection)

**`nccl_net_ofi_accept()`** ([src/nccl_ofi_api.cpp:294](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_api.cpp#L294)):

Also a multi-call state machine. The CM module handles the handshake:

```cpp
ncclResult_t nccl_net_ofi_accept(void *lComm, void **rComm)
{
  nccl_net_ofi_listen_comm *l_comm = (nccl_net_ofi_listen_comm *)lComm;

  // Virtual dispatch to transport-specific accept
  // CM: receive connect request, create recv_comm, send response
  ret = l_comm->accept(&r_comm);

  // r_comm->ep is set from l_comm->ep (shared_ptr copy)
  // This keeps the endpoint alive for the recv_comm
}
```

**Note**: EFA uses connectionless (UD-like) transport, so "connection" is mostly state tracking.

### Memory Registration

**`nccl_net_ofi_regMrDmaBuf()`** ([src/nccl_ofi_api.cpp:329](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_api.cpp#L329)):

The plugin now uses a unified registration function that handles both standard
memory (iovec) and GPU memory via DMA-BUF. Registration is dispatched to the
transport-specific comm (send or recv), which uses the per-domain MR cache.

```cpp
ncclResult_t nccl_net_ofi_regMrDmaBuf(void* comm, void* data, size_t size,
                                       int type, uint64_t offset,
                                       int fd, void** mhandle)
{
  nccl_net_ofi_comm *base_comm = (nccl_net_ofi_comm *)comm;

  // Build cache key — supports two memory types:
  //   fd == -1: standard memory (iovec with address + length)
  //   fd >= 0:  GPU memory via DMA-BUF (fd + offset + length)
  const nccl_ofi_mr_ckey_t cache_key = (fd == -1)
      ? nccl_ofi_mr_ckey_mk_vec(data, size, base_comm->ep.get())
      : nccl_ofi_mr_ckey_mk_dmabuf(fd, offset, size, data, base_comm->ep.get());

  // Virtual dispatch to transport-specific regMr
  // (checks MR cache first, calls fi_mr_reg on miss)
  switch (base_comm->type) {
    case NCCL_NET_OFI_SEND_COMM:
      ret = ((nccl_net_ofi_send_comm *)base_comm)->regMr(&cache_key, type, mhandle);
      break;
    case NCCL_NET_OFI_RECV_COMM:
      ret = ((nccl_net_ofi_recv_comm *)base_comm)->regMr(&cache_key, type, mhandle);
      break;
  }
}
```

Inside the transport-specific `regMr()`, the MR cache is consulted:

```cpp
  // 1. Check cache (per-domain, sorted array by page-aligned address)
  handle = nccl_ofi_mr_cache_lookup_entry(domain->mr_cache, &cache_key, endpoint_mr);
  if (handle) return handle;  // Cache hit — refcount incremented

  // 2. Cache miss — register with libfabric
  fi_mr_regattr(domain, &mr_attr, flags, &mr);  // See fi_mr_reg() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L413)

  // 3. Insert into cache
  nccl_ofi_mr_cache_insert_entry(domain->mr_cache, &cache_key, endpoint_mr, handle);
```

**Registration Cache** ([include/nccl_ofi_mr.h](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/include/nccl_ofi_mr.h)):
```cpp
// Cache entry — one per registered memory region
typedef struct nccl_ofi_reg_entry {
    uintptr_t addr;         // Page-aligned base address
    size_t pages;           // Number of pages covered
    int refcnt;             // Reference count (shared registrations)
    void *handle;           // fi_mr handle from libfabric
    nccl_net_ofi_ep_t *ep;  // Endpoint that owns this MR (for endpoint-specific MRs)
} nccl_ofi_reg_entry_t;

// Cache structure — sorted array for containment lookup
typedef struct nccl_ofi_mr_cache {
    nccl_ofi_reg_entry_t **slots;  // Sorted array of entries (by address)
    size_t system_page_size;       // Page size (typically 4096)
    size_t size;                   // Allocated capacity (grows 2x)
    size_t used;                   // Current number of entries
    uint32_t hit_count;            // Statistics: cache hits
    uint32_t miss_count;           // Statistics: cache misses
    pthread_mutex_t lock;          // Thread-safe access
} nccl_ofi_mr_cache_t;

// Cache key — supports iovec and DMA-BUF memory types
struct nccl_ofi_mr_ckey {
    union {
        struct iovec iovec;                // Standard memory (virtual address + length)
        struct fi_mr_dmabuf fi_mr_dmabuf;  // GPU memory via DMA-BUF (fd + offset + length)
    };
    nccl_net_ofi_ep_t *ep;                 // Endpoint (part of key for endpoint-specific MRs)
    enum nccl_ofi_mr_ckey_type type;       // NCCL_OFI_MR_CKEY_IOVEC or NCCL_OFI_MR_CKEY_DMABUF
};
```

(See `struct fid_mr` ([include/rdma/fi_domain.h:131-138](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L131-L138)))

**Cache Benefits:**
- Avoid expensive re-registration
- Critical for performance (100+ μs saved)
- Reference counting for shared regions

### Send Operations

**`nccl_net_ofi_isend()`** ([src/nccl_ofi_api.cpp:441](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_api.cpp#L441)):

The API layer casts the comm and dispatches to the transport-specific `send()` virtual method:

```cpp
ncclResult_t nccl_net_ofi_isend(void* sendComm, void* data,
                                size_t size, int tag,
                                void* mhandle, void** request)
{
  nccl_net_ofi_send_comm *send_comm = (nccl_net_ofi_send_comm *)sendComm;
  nccl_net_ofi_mr_handle_t *handle = (nccl_net_ofi_mr_handle_t *)mhandle;
  nccl_net_ofi_req **base_req = (nccl_net_ofi_req **)request;

  // Virtual dispatch to transport-specific send
  int ret = send_comm->send(data, size, tag, handle, base_req);
  return nccl_net_ofi_retval_translate_impl(ret);
}
```

Inside the transport-specific `send()` (e.g., RDMA or SendRecv):

```cpp
  // Get memory descriptor for the registered region
  void* desc = fi_mr_desc(mr);  // See fi_mr_desc() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L147)

  // Post tagged send operation (tag identifies channel/operation)
  ret = fi_tsend(ep, data, size, desc,  // See fi_tsend() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_tagged.h#L121)
                 remote_addr, tag, &context);

  if (ret == -FI_EAGAIN) {
    // Provider queue full — progress CQ and retry
  }
```

**Tagged Send:**
- Tag identifies message type/channel
- Allows multiple outstanding operations
- Receiver matches by tag

### Receive Operations

**`nccl_net_ofi_irecv()`** ([src/nccl_ofi_api.cpp:471](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_api.cpp#L471)):

```cpp
ncclResult_t nccl_net_ofi_irecv(void* recvComm, int n,
                                void** data, size_t* sizes,
                                int* tags, void** mhandles,
                                void** request)
{
  nccl_net_ofi_recv_comm *recv_comm = (nccl_net_ofi_recv_comm *)recvComm;

  // Virtual dispatch — NCCL can post multiple receives at once (up to n)
  int ret = recv_comm->recv(n, data, sizes, tags, mr_handles, base_req);
}
```

Inside the transport-specific `recv()`:

```cpp
  for (int i = 0; i < n; i++) {
    void* desc = fi_mr_desc(mr_handles[i]);

    // Pre-post tagged receives
    ret = fi_trecv(ep, data[i], sizes[i], desc,  // See fi_trecv() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_tagged.h#L98)
                   FI_ADDR_UNSPEC, tags[i], 0, &context);
  }
```

**Pre-posting:**
- Receives posted before data arrives
- Matched when sender transmits with matching tag
- Reduces latency by having buffers ready

### Flush Operations (RDMA Write Completion)

**`nccl_net_ofi_iflush()`** ([src/nccl_ofi_api.cpp:537](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_api.cpp#L537)):

```cpp
ncclResult_t nccl_net_ofi_iflush(void* rComm, int n,
                                 void** buffers, int* sizes,
                                 void** mhandles, void** request)
{
  nccl_net_ofi_recv_comm *recv_comm = (nccl_net_ofi_recv_comm *)rComm;

  // Virtual dispatch to transport-specific flush
  int ret = recv_comm->flush(n, buffers, sizes, mr_handles, base_req);
}
```

Inside the RDMA transport flush:

```cpp
  // Issue a small RDMA read as an ordering fence
  // The read completion guarantees all prior writes are visible
  fi_read(ep, flush_buf, flush_size, desc,
          remote_addr, 0, 0, &context);
```

**Why Flush:**
- RDMA writes are one-sided (no remote CPU notification)
- Flush ensures write is visible to receiver's GPU
- EFA: Uses a small `fi_read()` as an ordering fence
- On P5+, `NCCL_NET_FORCE_FLUSH=0` means NCCL only flushes when necessary

### Completion Testing

**`nccl_net_ofi_test()`** ([src/nccl_ofi_api.cpp:524](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_api.cpp#L524)):

```cpp
ncclResult_t nccl_net_ofi_test(void* request, int* done, int* size)
{
  nccl_net_ofi_req *req = (nccl_net_ofi_req *)request;

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

**`nccl_net_ofi_deregMr()`** ([src/nccl_ofi_api.cpp:399](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_api.cpp#L399)):

```cpp
ncclResult_t nccl_net_ofi_deregMr(void *comm, void *mhandle)
{
  nccl_net_ofi_comm *base_comm = (nccl_net_ofi_comm *)comm;

  // Virtual dispatch to transport-specific deregMr
  // Decrements refcount in MR cache; if refcount reaches 0:
  //   fi_close(&mr->fid)  // See fi_close() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fabric.h#L150)
  //   remove entry from cache
  switch (base_comm->type) {
    case NCCL_NET_OFI_SEND_COMM:
      ret = send_comm->deregMr(mr_handle);
      break;
    case NCCL_NET_OFI_RECV_COMM:
      ret = recv_comm->deregMr(mr_handle);
      break;
  }
}
```

**`nccl_net_ofi_closeSend()` / `nccl_net_ofi_closeRecv()` / `nccl_net_ofi_closeListen()`**:

With the shared_ptr ownership model, closing a comm drops the `shared_ptr<ep>`.
If this was the last comm using the endpoint, the ep destructor runs automatically,
closing all libfabric resources (fi_endpoint, fi_cq, fi_av). No manual
`release_ep()` or `release_domain()` calls are needed.

```cpp
ncclResult_t nccl_net_ofi_closeSend(void *sComm)
{
  nccl_net_ofi_send_comm *send_comm = (nccl_net_ofi_send_comm *)sComm;

  // Virtual dispatch to transport-specific close
  // Frees comm resources (freelists, request pools)
  int ret = send_comm->close();

  // send_comm destructor drops shared_ptr<ep>
  // → if last comm: ep destructor runs → domain destructor runs
  delete send_comm;
}
```

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
- Plugin handles rail reordering for optimal pairing ([src/platform-aws.cpp:983](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/platform-aws.cpp#L973))

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


## C++ Class Hierarchy and Object Ownership

The plugin uses a C++ class hierarchy with smart pointer ownership for
automatic lifetime management. This was introduced to replace manual
reference counting (`release_ep()`, `release_domain()`) which had complex
conditional logic, lock ordering issues, and `delete this` patterns.

### Class Hierarchy

```
nccl_net_ofi_plugin_t                    (singleton, one per process)
  │  Owns devices via raw pointers (plugin always outlives devices)
  │  Created during init(), destroyed during finalize()
  │  Source: include/nccl_ofi.h, src/nccl_ofi_net.cpp
  │
  └── nccl_net_ofi_device_t              (one per NIC / multi-rail group)
        │  Holds: fi_info, device properties, comm ID pool
        │  Caches domains: domain_table (map<key, weak_ptr<domain>>)
        │  Source: include/nccl_ofi.h
        │
        │  Transport subclasses:
        │  ├── nccl_net_ofi_rdma_device_t     (include/nccl_ofi_rdma.h)
        │  └── nccl_net_ofi_sendrecv_device_t (include/nccl_ofi_sendrecv.h)
        │
        └── nccl_net_ofi_domain_t            (one per thread scope)
              │  Holds: fi_domain, fi_av, fi_cq, MR cache, MR key pool
              │  Caches endpoints: ep_table (map<key, weak_ptr<ep>>)
              │  Inherits: enable_shared_from_this<domain>
              │  Source: include/nccl_ofi.h
              │
              └── nccl_net_ofi_ep_t          (one per thread)
                    │  Holds: fi_endpoint, rx_buffers, ep_lock
                    │  Inherits: enable_shared_from_this<ep>
                    │  Source: include/nccl_ofi.h
                    │
                    └── nccl_net_ofi_comm     (listen/send/recv)
                          Holds: shared_ptr<ep>, dev_id, type
                          Source: include/nccl_ofi.h
```

### Ownership Model

```
  Ownership chain (shared_ptr):
    comm ──shared_ptr──> ep ──shared_ptr──> domain ──raw ptr──> device

  Cache tables (weak_ptr, non-owning):
    device.domain_table: map<key, weak_ptr<domain>>
    domain.ep_table:     map<key, weak_ptr<ep>>
```

**Rules:**
- Comms are the real owners of endpoints (via `shared_ptr<ep>`)
- Endpoints are the real owners of domains (via `shared_ptr<domain>`)
- Tables are non-owning caches — they hold `weak_ptr` for lookup only
- When the last comm drops its `shared_ptr<ep>`, the ep destructor runs
- The ep destructor drops `shared_ptr<domain>`, potentially triggering domain cleanup
- No locks are needed during this destruction cascade

### Lazy Cleanup Pattern

The `get_ep()` and `get_domain()` functions use a lazy cleanup pattern
([src/nccl_ofi_net.cpp](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_net.cpp)):

```cpp
// domain_t::get_ep(endpoint_key)
shared_ptr<ep> get_ep(long key) {
    lock_guard(domain_lock);

    auto iter = ep_table.find(key);
    if (iter != ep_table.end()) {
        auto ep_ptr = iter->second.lock();  // promote weak_ptr
        if (ep_ptr) return ep_ptr;          // cache hit: still alive
        ep_table.erase(iter);               // expired: clean up stale entry
    }

    // Cache miss: purge other expired entries to prevent unbounded growth
    // (important for GIN which keys by comm_id, not fixed thread ID)
    for (auto it = ep_table.begin(); it != ep_table.end();)
        it->second.expired() ? it = ep_table.erase(it) : ++it;

    // Create new endpoint, insert weak_ptr, return shared_ptr
    auto *raw_ep = create_endpoint();
    shared_ptr<ep> ep_ptr(raw_ep);
    ep_table.insert({key, ep_ptr});
    return ep_ptr;
}
```

### Destruction Cascade

```
  Last comm destroyed
    └── shared_ptr<ep> drops → ep refcount hits 0
          └── ep destructor runs
                ├── Closes fi_endpoint, frees rx_buffers, destroys mutex
                └── shared_ptr<domain> drops → domain refcount hits 0
                      └── domain destructor runs
                            ├── Checks for leaked eps (live weak_ptrs = bug)
                            ├── Clears ep_table
                            ├── Closes fi_domain, fi_av, fi_cq
                            └── Frees MR cache
```

## GIN Transport

GIN (Group Interconnect Network) is an NCCL extension for one-sided put
operations, used by workloads like DeepEP for Mixture-of-Experts dispatch.

**Source files:**
- `src/rdma/gin/nccl_ofi_gin_api.cpp` — GIN plugin API implementation
- `src/rdma/gin/nccl_ofi_gin.cpp` — GIN connection establishment (bootstrap ring)
- `src/rdma/gin/nccl_ofi_gin_resources.cpp` — GIN resource management
- `src/rdma/gin/nccl_ofi_gin_reqs.cpp` — GIN request handling
- `src/rdma/gin/nccl_ofi_gin_allgather.cpp` — GIN AllGather for connection setup
- `include/rdma/gin/nccl_ofi_gin.h` — GIN listen/put comm classes
- `include/rdma/gin/nccl_ofi_gin_resources.h` — GIN resources and ep_holder

### GIN Plugin API (`ncclGin_v11_t`)

The plugin implements these GIN functions
([3rd-party/nccl/cuda/include/nccl/net_v11.h](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/3rd-party/nccl/cuda/include/nccl/net_v11.h)):

| Function | Purpose | Source |
|----------|---------|--------|
| `init` | Initialize GIN support | `nccl_ofi_gin_api.cpp` |
| `listen` | Create listening endpoint for bootstrap | `nccl_ofi_gin_api.cpp` |
| `connect` | Bootstrap ring + create GIN resources | `nccl_ofi_gin.cpp` |
| `iput` | One-sided RDMA write to remote rank | `nccl_ofi_gin_reqs.cpp` |
| `iputSignal` | RDMA write + atomic signal increment | `nccl_ofi_gin_reqs.cpp` |
| `ginProgress` | Progress outstanding operations | `nccl_ofi_gin_reqs.cpp` |
| `regMrSym` | Register memory for symmetric access | `nccl_ofi_gin_resources.cpp` |
| `test` | Test request completion | `nccl_ofi_gin_reqs.cpp` |
| `closeColl` / `closeListen` | Cleanup | `nccl_ofi_gin_api.cpp` |

### GIN Connection Flow

```
  1. listen(ctx, dev, handle, &listenComm)
     ├── Creates RDMA endpoint via device->get_ep()
     ├── Calls ep->listen() to create RDMA listen_comm
     ├── Sets l_comm->ep = shared_ptr<ep>
     └── Wraps in nccl_ofi_rdma_gin_listen_comm (holds shared_ptr<ep>)

  2. connect(ctx, handles[], nranks, rank, listenComm, &collComm)
     ├── Bootstrap ring: connect to rank (rank+1)%nranks, accept from prev rank
     │    └── Uses standard RDMA connect/accept via CM module
     ├── Create nccl_ofi_gin_resources on the endpoint
     │    └── gin_ep_holder keeps endpoint alive via shared_ptr<ep>
     ├── AllGather connection handles via bootstrap send/recv ring
     └── Returns nccl_ofi_rdma_gin_put_comm with per-peer state

  3. iput(collComm, srcOff, srcMh, size, dstOff, dstMh, rank, &req)
     └── fi_write() to remote rank's registered memory

  4. iputSignal(collComm, srcOff, srcMh, size, dstOff, dstMh,
               rank, signalOff, signalMh, value, op, &req)
     └── fi_write() for data + atomic signal increment on remote rank

  5. ginProgress(collComm)
     └── fi_cq_read() on all rails, process completions
```

### GIN Resource Ownership

```
  nccl_ofi_rdma_gin_listen_comm
    └── shared_ptr<ep>           (keeps endpoint alive during bootstrap)

  nccl_ofi_gin_resources
    └── nccl_ofi_gin_ep_holder
          └── shared_ptr<ep>     (keeps endpoint alive for GIN operations)

  nccl_ofi_rdma_gin_put_comm
    └── nccl_ofi_gin_resources&  (reference to resources on the endpoint)
```

## Connection Manager (CM)

The CM module handles connection establishment as a state machine,
separating connection logic from transport logic.

**Source files:**
- `src/cm/nccl_ofi_cm.cpp` — Core CM logic
- `src/cm/nccl_ofi_cm_reqs.cpp` — CM request handling
- `src/cm/nccl_ofi_cm_resources.cpp` — CM resource management
- `include/cm/nccl_ofi_cm.h` — CM interface

### CM State Machine

```
  Connect side (sender):             Accept side (receiver):
  ─────────────────────              ──────────────────────

  COMM_CREATE_START                  COMM_CREATE_START
       │                                  │
       ▼                                  ▼
  Create send_comm                   (wait for connect request)
  CM: fi_tsend(connect_request)           │
       │                                  ▼
       ▼                             CM: fi_trecv() → got connect request
  COMM_CONN_REQ_PENDING              Create recv_comm
  (waiting for response)             CM: fi_tsend(connect_response)
       │                                  │
       ▼                                  ▼
  CM: fi_trecv() → got response      COMM_CONN_RESP_REQ_PENDING
  COMM_CONNECTED                     (waiting for delivery confirmation)
                                          │
                                          ▼
                                     COMM_CONNECTED
```

### CM Key Classes

- `nccl_ofi_cm` — Main CM class, creates connectors/receivers/listeners
- `nccl_ofi_cm_send_connector` — Active side: sends connect request, waits for response
- `nccl_ofi_cm_receiver` — Passive side: receives connect request, sends response
- `nccl_ofi_cm_listener` — Listens for incoming connection requests via fi_trecv

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
