# OFI NCCL Plugin Architecture

## Overview

The OFI (OpenFabrics Interfaces) NCCL plugin (`aws-ofi-nccl`) provides a network transport for NCCL using libfabric. It implements NCCL's network plugin interface (up to `ncclNet_v12_t`) and the GPU-initiated networking interfaces — GIN (up to `ncclGin_v14_t`, GDAKI-only) and RMA (up to `ncclRma_v15_t`) — and is the primary way NCCL communicates over AWS EFA.

**Repository**: https://github.com/aws/aws-ofi-nccl

The plugin is written in **C++20** (mandatory since master `a615420`; `configure.ac`
uses `AX_CXX_COMPILE_STDCXX([20], [noext], [mandatory])` with `-fno-rtti`) and organized
as a class hierarchy
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

**Protocol Selection** ([src/platform-aws.cpp:81-250](https://github.com/aws/aws-ofi-nccl/blob/master/src/platform-aws.cpp)):
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

The plugin implements the `ncclNet_v12_t` interface (newest), and also exports older interface versions down to `ncclNet_v4` so a given NCCL build links the highest version it supports.
The API layer is in `src/nccl_ofi_api.cpp`, which dispatches to transport-specific
implementations in `src/nccl_ofi_rdma.cpp` (RDMA) or `src/nccl_ofi_sendrecv.cpp` (SendRecv).

### Exported plugin symbols

NCCL discovers a plugin by dlopen-ing the shared object and looking up versioned
symbols, binding the **highest version it supports**. Every export is tagged with the
`NCCL_OFI_EXPORT_SYMBOL` macro. As of master `d840aa1` the plugin exports the following
(verify with `grep -rn 'NCCL_OFI_EXPORT_SYMBOL' src`):

| Symbol | Interface | Source file | Notes |
|--------|-----------|-------------|-------|
| `ncclNetPlugin_v6`..`ncclNetPlugin_v12` | `ncclNet_v*_t` | [src/nccl_ofi_interface_nvidia.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_interface_nvidia.cpp) | CUDA/ROCm build. `v12` is newest |
| `ncclNetPlugin_v4`, `ncclNetPlugin_v5`, `ncclNetPlugin_v6` | `ncclNet_v*_t` | [src/nccl_ofi_interface_neuron.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_interface_neuron.cpp) | AWS Neuron build exports v4/v5/v6 |
| `ncclGinPlugin_v11`, `ncclGinPlugin_v13` | `ncclGin_v*_t` | [src/rdma/gin/nccl_ofi_gin_api.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_api.cpp) | **proxy** GIN op tables |
| `ncclGinPlugin_v14` | `ncclGin_v14_t` | [src/rdma/gin/nccl_ofi_gin_gdaki.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_gdaki.cpp) | **GDAKI-only** (kernel-initiated), `name = "Libfabric_GDAKI"` |
| `ncclRmaPlugin_v14`, `ncclRmaPlugin_v15` | `ncclRma_v*_t` | [src/rdma/gin/nccl_ofi_gin_api.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_api.cpp) | **host / CPU-proxy** one-sided path |
| `ncclTunerPlugin_v2`, `ncclTunerPlugin_v3`, `ncclTunerPlugin_v6` | `ncclTuner_v*_t` | [src/tuner/nccl_ofi_tuner.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/tuner/nccl_ofi_tuner.cpp) | cost-model tuner |

> Note: earlier docs listed only `ncclNet_v12_t` and `ncclGin_v13_t`. The GPU-initiated
> path is now split across three separate op-table families exported by the same shared
> object.

#### The `ncclGin_*` vs `ncclRma_*` split

NCCL's GPU-initiated networking evolved into two distinct op-table shapes, and the
plugin exports both so a given NCCL build binds the one it knows:

- **`ncclGin_*` tables** carry the *device/kernel-facing* GIN interface. The proxy
  tables (`ncclGinPlugin_v11`, `ncclGinPlugin_v13`) advertise
  `netDeviceType = NCCL_NET_DEVICE_GIN_PROXY` and set `createContext = nullptr` for v11
  (proxy needs no device handle); v13 adds `createContext`/`destroyContext`, `iget`,
  and `iflush`. The **GDAKI** table `ncclGinPlugin_v14` (in `nccl_ofi_gin_gdaki.cpp`,
  `name = "Libfabric_GDAKI"`) is the **GPU-initiated (kernel-initiated) path**: its
  `createContext` builds a device-visible handle in GPU memory describing the SQ/CQ/RQ
  so the kernel posts WQEs and polls CQEs itself. It is only compiled and exported when
  `HAVE_GDAKI` is set.
- **`ncclRma_*` tables** (`ncclRmaPlugin_v14`, `ncclRmaPlugin_v15`) carry the
  **host / CPU-proxy** one-sided path. NCCL's GIN proxy binds these directly (via
  `rma_v14.cc` on the NCCL side). They reuse the proxy GIN implementation under the
  hood: e.g. `nccl_ofi_rma_createContext_v14` translates the RMA config into a
  `ncclGinConfig_v13_t` and calls `nccl_ofi_gin_createContext_v13`; the RMA
  `iputSignal` takes a trailing `isStrongSignal` flag that the plugin ignores (the
  plugin's signals are always strong). See the "v14 moves the host data path to the RMA
  op-table" comment in [nccl_ofi_gin_api.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_api.cpp).

In short: **`ncclGin_v14_t` carries the GPU-initiated (GDAKI) path; the `ncclRma_v14/v15`
tables carry the host/CPU-proxy path.** NCCL selects the highest version of each family
that it supports; the choice between proxy and GDAKI at runtime is made NCCL-side (see
the GIN Transport section and `doc/gin-getting-started.md`).

### Initialization

**`nccl_net_ofi_init()`** ([src/nccl_ofi_api.cpp:58](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp)):

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

**`nccl_net_ofi_devices()` / `nccl_net_ofi_get_properties()`** ([src/nccl_ofi_api.cpp:111](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp), [src/nccl_ofi_api.cpp:129](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp)):

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

**`nccl_net_ofi_listen()`** ([src/nccl_ofi_api.cpp:150](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp)):

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
  fi_endpoint(domain, info, &ep, NULL);  // See fi_endpoint() (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h)

  // Bind to completion queue
  fi_ep_bind(ep, cq, 0);  // See fi_ep_bind() (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h)

  // Enable endpoint
  fi_enable(ep);  // See fi_enable() (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h)

  // Get local address
  fi_getname(ep, &local_addr, &addrlen);  // See fi_getname() (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_cm.h)

  // Return address in handle (for sharing via NCCL bootstrap)
  memcpy(handle, &local_addr, addrlen);

  *listenComm = comm_state;
  return ncclSuccess;
}
```

#### Connector (Active Side)

**`nccl_net_ofi_connect()`** ([src/nccl_ofi_api.cpp:220](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp)):

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
  fi_av_insert(av, &remote_addr, 1, ...);  // See fi_av_insert() (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_cm.h)

  // CM: send connect request message to peer
  // CM: wait for connect response (may return incomplete on first call)
```

#### Accept (Complete Passive Connection)

**`nccl_net_ofi_accept()`** ([src/nccl_ofi_api.cpp:294](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp)):

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

**`nccl_net_ofi_regMrDmaBuf()`** ([src/nccl_ofi_api.cpp:329](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp)):

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
  // nccl_ofi_mr_cache is a C++ class; domain->mr_cache is a std::optional<nccl_ofi_mr_cache>
  // guarded by the domain's mr_cache_lock. See mr-cache-implementation.md.

  // 1. Check cache (per-domain, sorted array by page-aligned address)
  handle = domain->mr_cache->lookup_entry(cache_key, endpoint_mr);
  if (handle) return handle;  // Cache hit — refcount incremented

  // 2. Cache miss — register with libfabric
  fi_mr_regattr(domain, &mr_attr, flags, &mr);  // See fi_mr_reg() (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h)

  // 3. Insert into cache
  domain->mr_cache->insert_entry(cache_key, endpoint_mr, handle);
```

**Registration Cache** ([include/nccl_ofi_mr.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_mr.h)):
```cpp
// Cache entry — one per registered memory region.
// Still a struct, but it now has a constructor that starts refcnt at 1.
struct nccl_ofi_reg_entry {              // include/nccl_ofi_mr.h:186-195
    uintptr_t addr;         // Page-aligned base address
    size_t pages;           // Number of pages covered
    int refcnt;             // Reference count (shared registrations)
    void *handle;           // fi_mr handle from libfabric
    nccl_net_ofi_ep_t *ep;  // Endpoint that owns this MR (endpoint-specific MRs only)

    nccl_ofi_reg_entry(uintptr_t addr_arg, size_t pages_arg,
                       void *handle_arg, nccl_net_ofi_ep_t *ep_arg)
        : addr(addr_arg), pages(pages_arg), refcnt(1),
          handle(handle_arg), ep(ep_arg) {}
};
typedef struct nccl_ofi_reg_entry nccl_ofi_reg_entry_t;

// The cache is a C++ CLASS, not a struct — and it holds NO lock of its own.
class nccl_ofi_mr_cache {                // include/nccl_ofi_mr.h:201
public:
    nccl_ofi_mr_cache(size_t init_num_entries, size_t page_size_arg);
    ~nccl_ofi_mr_cache();                // logs hit/miss stats
    nccl_ofi_mr_cache(const nccl_ofi_mr_cache &) = delete;   // not copyable
    nccl_ofi_mr_cache &operator=(const nccl_ofi_mr_cache &) = delete;

    void *lookup_entry(nccl_ofi_mr_ckey_ref ckey, bool is_endpoint_mr);
    int   insert_entry(nccl_ofi_mr_ckey_ref ckey, bool is_endpoint_mr, void *handle);
    int   del_entry(void *handle);

private:
    std::vector<nccl_ofi_reg_entry_t *> slots;   // NOT a manually grown C array
    size_t page_size;
    uint32_t hit_count = 0;
    uint32_t miss_count = 0;

    void compute_page_address(uintptr_t addr, size_t size,
                              uintptr_t &page_addr, size_t &pages) const;
};
```

Two things to note, because older documentation and blog posts get both wrong:

- **There is no `nccl_ofi_mr_cache_t` typedef and no embedded `pthread_mutex_t`.** The
  former C API (`nccl_ofi_mr_cache_init()`, `_finalize()`, and a struct carrying its own
  lock) has been removed. See [mr-cache-implementation.md](mr-cache-implementation.md) for
  the removed-API mapping table.
- **Serialization is external.** The cache is guarded by the *domain's* `mr_cache_lock`
  (`std::mutex`, alongside `std::optional<nccl_ofi_mr_cache> mr_cache` in
  [include/nccl_ofi.h, lines 473-476](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi.h)),
  so every caller must hold that lock. A reader who assumes the cache locks itself will
  write a race.

```cpp
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

(See `struct fid_mr` ([include/rdma/fi_domain.h:131-138](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h)))

**Cache Benefits:**
- Avoid expensive re-registration
- Critical for performance (100+ μs saved)
- Reference counting for shared regions

### Send Operations

**`nccl_net_ofi_isend()`** ([src/nccl_ofi_api.cpp:441](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp)):

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
  void* desc = fi_mr_desc(mr);  // See fi_mr_desc() (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h)

  // Post tagged send operation (tag identifies channel/operation)
  ret = fi_tsend(ep, data, size, desc,  // See fi_tsend() (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_tagged.h)
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

**`nccl_net_ofi_irecv()`** ([src/nccl_ofi_api.cpp:471](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp)):

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
    ret = fi_trecv(ep, data[i], sizes[i], desc,  // See fi_trecv() (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_tagged.h)
                   FI_ADDR_UNSPEC, tags[i], 0, &context);
  }
```

**Pre-posting:**
- Receives posted before data arrives
- Matched when sender transmits with matching tag
- Reduces latency by having buffers ready

### Flush Operations (RDMA Write Completion)

**`nccl_net_ofi_iflush()`** ([src/nccl_ofi_api.cpp:537](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp)):

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

**`nccl_net_ofi_test()`** ([src/nccl_ofi_api.cpp:524](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp)):

```cpp
ncclResult_t nccl_net_ofi_test(void* request, int* done, int* size)
{
  nccl_net_ofi_req *req = (nccl_net_ofi_req *)request;

  // Poll completion queue
  struct fi_cq_data_entry cq_entry;  // See struct fi_cq_data_entry (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h)
  ret = fi_cq_read(comm->cq, &cq_entry, 1);  // See fi_cq_read() (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h)

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
  struct fi_cq_err_entry err_entry;  // See struct fi_cq_err_entry (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h)
  if (fi_cq_readerr(comm->cq, &err_entry, 0) > 0) {  // See fi_cq_readerr() (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h)
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

**`nccl_net_ofi_deregMr()`** ([src/nccl_ofi_api.cpp:399](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp)):

```cpp
ncclResult_t nccl_net_ofi_deregMr(void *comm, void *mhandle)
{
  nccl_net_ofi_comm *base_comm = (nccl_net_ofi_comm *)comm;

  // Virtual dispatch to transport-specific deregMr
  // Decrements refcount in MR cache; if refcount reaches 0:
  //   fi_close(&mr->fid)  // See fi_close() (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h)
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
- Plugin handles rail reordering for optimal pairing ([src/platform-aws.cpp:983](https://github.com/aws/aws-ofi-nccl/blob/master/src/platform-aws.cpp))

**Configuration:**
```bash
# Automatic detection (default)
# Or explicitly select
NCCL_NET_GDR_LEVEL=LOC  # GPU-local NICs preferred
NCCL_MIN_NCHANNELS=8   # Channels distributed across rails; set both
NCCL_MAX_NCHANNELS=8   # clamps to pin the count
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

### Request Pipelining and the RDMA request class hierarchy

The plugin maintains many outstanding requests to hide latency and keep the network
saturated. Request objects are pooled in a **freelist** rather than malloc'd per
operation.

**Refactor (master `f6f6223`, `960cb56`, `5b6d45f`, `1b5b830`, `8d2b9f0`, `cc7afb2`,
`4942f12`, `aebf544`, `1b513c4`, `4b9a63c`, `d7b50ca`):** the RDMA request object was
converted from a single base struct carrying a *union* of per-operation data structs
into a **C++ class hierarchy**. `nccl_net_ofi_rdma_req` is now an abstract base
(`class nccl_net_ofi_rdma_req : public nccl_net_ofi_req`, [include/nccl_ofi_rdma.h:396](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_rdma.h)) with
pure-virtual `free()`, `post()`, `handle_completion()` and an overriding `test()`,
plus concrete subclasses:

| Subclass | Purpose |
|----------|---------|
| `rdma_send_req` | application send (eager or rendezvous); carries eager fields |
| `rdma_recv_req` | application receive |
| `rdma_flush_req` | GDR flush (small `fi_read` ordering fence) |
| `rdma_rma_op_req` | one-sided `fi_write`/`fi_read` (RMA/GIN) |
| `rdma_rx_buff_req` | posted rx buffer, with `kind::EAGER` or `kind::CTRL` |
| `rdma_eager_copy_req` | copy of an eager payload from the rx buffer to the app buffer |

Per-type behaviour that used to switch on a `req_type` enum is now **virtual dispatch**:

- `post()` — each subclass performs its own libfabric post (`fi_tsend`, `fi_write`,
  `fi_read`, rx-buffer repost, …). `post_with_pending_retry()` wraps it and, on
  `-FI_EAGAIN`, pushes the request onto the endpoint's pending queue for later retry.
- `handle_completion(comp_flags, rail_id)` — the CQ dispatcher in `handle_cq_entry`
  peels off the `FI_RECV` path (rx-buffer handling) and forwards every other completion
  here; subclasses branch on `FI_SEND`/`FI_WRITE`/`FI_READ`. The base implementation
  logs and returns `-EINVAL` (reached only on a programming error).
- `free(dec_inflight_reqs)` — returns the object to its freelist.
- `get_type()` is a **non-virtual** field read of a private `const req_type`, kept only
  for logging/tracing (dispatch no longer uses it).

**Placement-new allocation out of the freelist.** Allocation no longer memsets a union;
it takes a freelist slot and constructs the concrete subclass in place, e.g.
([src/nccl_ofi_rdma.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_rdma.cpp)):

```cpp
nccl_ofi_freelist::fl_entry *elem = r_comm->nccl_ofi_reqs_fl->entry_alloc();
nccl_net_ofi_rdma_req *req = new (elem->ptr) rdma_eager_copy_req();   // placement new
// ...
req->~nccl_net_ofi_rdma_req();          // explicit virtual destructor call on free
nccl_ofi_reqs_fl->entry_free(elem);     // slot returned to the freelist
```

Because construction is per-allocation placement new, **destruction must be a matching
explicit destructor call** before the block returns to the freelist (see the
"Construction happens per-allocation via placement new" comment near
[src/nccl_ofi_rdma.cpp:1604](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_rdma.cpp)).

**Freelist sizing implication.** Every subclass shares one freelist per comm
(`nccl_ofi_reqs_fl`), so each slot must be large enough for the **largest** subclass —
the freelist element size is the max `sizeof` across the hierarchy, not a per-type size.
Adding a large field to any one subclass grows every slot. rx-buffer requests use a
separate freelist on the endpoint (`ep->rx_buff_reqs_fl`), and eager header buffers use
their own `eager_hdr_fl`, so control/eager buffer pools are sized independently of the
main request pool.

### Eager Protocol Optimization

For small messages, the RDMA transport *can* send the payload inline (eager) instead of
doing the rendezvous control-message handshake. **Eager is now DISABLED BY DEFAULT.**
`OFI_NCCL_EAGER_MAX_SIZE` (env `EAGER_MAX_SIZE`) defaults to **-1**, which disables the
eager path entirely ([include/nccl_ofi_param.h:229](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_param.h),
`OFI_NCCL_PARAM(int, eager_max_size, "EAGER_MAX_SIZE", -1)`). Set it to a positive byte
count (e.g. 8192) to re-enable eager for messages at or below that size. See
[ofi-plugin-protocols.md](ofi-plugin-protocols.md) for the full eager protocol
(8-byte header, grouped-recv batch draining, wrap-safe `eager_seq`) and the history of
the default flip.

Conceptually, when enabled:

```c
if (size <= eager_max_size) {
  // Eager: payload (prefixed by an 8-byte eager header) sent inline via
  // fi_sendmsg(..., FI_REMOTE_CQ_DATA); no prior control message
} else {
  // Rendezvous: control message, then fi_write
}
```

## Libfabric API Usage

### Core APIs Used

```c
// Initialization
fi_getinfo()      // Query providers (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h)
fi_fabric()       // Create fabric (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h)
fi_domain()       // Create domain (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h)
fi_endpoint()     // Create endpoint (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h)
fi_cq_open()      // Create completion queue (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h)
fi_av_open()      // Create address vector (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_cm.h)

// Memory
fi_mr_reg()       // Register memory (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h)
fi_mr_desc()      // Get descriptor (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h)

// Data transfer
fi_send()         // Send message (fi_endpoint.h:326)
fi_tsend()        // Tagged send (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_tagged.h)
fi_recv()         // Receive message (fi_endpoint.h:306)
fi_trecv()        // Tagged receive (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_tagged.h)
fi_read()         // RDMA read (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_rma.h)
fi_write()        // RDMA write (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_rma.h)
fi_inject()       // Eager send (fi_endpoint.h:346)

// Completion
fi_cq_read()      // Poll completions (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h)
fi_cq_readerr()   // Read error (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h)

// Cleanup
fi_close()        // Close resources (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h)
```

### Endpoint Configuration

(See `struct fi_info` ([include/rdma/fabric.h:198-232](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h)))

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
# Plugin-specific (all plugin knobs use the OFI_NCCL_ prefix; see
# include/nccl_ofi_param.h for the authoritative list and defaults)
OFI_NCCL_PROTOCOL=RDMA           # per-platform default (RDMA on P5+ via
                                 # platform-aws.cpp default_protocol);
                                 # param.h fallback is SENDRECV
OFI_NCCL_EAGER_MAX_SIZE=-1       # -1 (default) disables eager entirely

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
([src/nccl_ofi_net.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_net.cpp)):

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

### GIN Plugin API (`ncclGin_v13_t` proxy, `ncclGin_v14_t` GDAKI, `ncclRma_v14/v15_t` host proxy)

The plugin implements these GIN functions
([3rd-party/nccl/cuda/include/nccl/gin_v13.h](https://github.com/aws/aws-ofi-nccl/blob/master/3rd-party/nccl/cuda/include/nccl/gin_v13.h)):

| Function | Purpose | Source |
|----------|---------|--------|
| `init` | Initialize GIN support; reads `NCCL_GIN_PROXY_NTHREADS` | `nccl_ofi_gin_api.cpp` |
| `listen` | Create listening endpoint for bootstrap; per-thread EP bucketing | `nccl_ofi_gin_api.cpp` |
| `connect` | Bootstrap ring + create GIN resources | `nccl_ofi_gin_api.cpp` / `nccl_ofi_gin.cpp` |
| `createContext` / `destroyContext` | Device/RMA context (proxy: v13+; GDAKI builds device handle) | `nccl_ofi_gin_api.cpp` / `nccl_ofi_gin_gdaki.cpp` |
| `iput` | One-sided RDMA write to remote rank | `nccl_ofi_gin_reqs.cpp` |
| `iputSignal` | RDMA write + atomic signal increment | `nccl_ofi_gin_reqs.cpp` |
| `iget` / `iflush` | one-sided read / flush (v13+) | `nccl_ofi_gin_reqs.cpp` |
| `ginProgress` | Progress outstanding operations | `nccl_ofi_gin_reqs.cpp` |
| `regMrSym` / `regMrSymDmaBuf` | Register memory for symmetric access | `nccl_ofi_gin_resources.cpp` |
| `test` | Test request completion | `nccl_ofi_gin_reqs.cpp` |
| `closeColl` / `closeListen` | Cleanup | `nccl_ofi_gin_api.cpp` |

**`NCCL_GIN_PROXY_NTHREADS` — per-thread endpoint bucketing (master `e6c4eb1`).**
This is an **NCCL-side** environment variable read directly via `getenv()` in
`nccl_ofi_gin_init` (it is *not* an `OFI_NCCL_*` plugin param). Default **1**, capped at
`NCCL_GIN_MAX_CONNECTIONS`. NCCL calls `listen()` once per collComm; the plugin maps
each connection to a progress thread with `listen_seq % nthreads` and folds the thread
index into the endpoint key (`seq ^ (thread_idx << 32)`), so each GIN progress thread
drives **its own endpoint/CQ** instead of contending on a shared completion queue. With
`NCCL_GIN_PROXY_NTHREADS=1` all connections share one endpoint (legacy behaviour). See
[src/rdma/gin/nccl_ofi_gin_api.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_api.cpp) (`listen()`, lines ~79 and ~150).

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

### GDAKI (EFA-GDA) build and runtime requirements

GDAKI is the **GPU-initiated** GIN data path: the CUDA kernel drives the EFA send/completion/receive
queues directly, with no CPU proxy on the data path.

**Build-time (`HAVE_GDAKI`).** `configure.ac` enables GDAKI only when *all* of the
following hold ([configure.ac](https://github.com/aws/aws-ofi-nccl/blob/master/configure.ac), ~line 354):

- libfabric exposes `FI_EFA_GDA_OPS` (`ac_cv_have_decl_FI_EFA_GDA_OPS = yes`) — i.e. the
  libfabric ≥ 2.5 GDA ops ABI is present;
- the EFA hardware-completion-counter ABI is present (`have_fi_efa_comp_cntr = 1`);
- the device interface is **CUDA** (`have_device_interface = cuda`).

Otherwise the build is proxy-only GIN and `ncclGinPlugin_v14` is not exported. There is
no `--enable-gdaki` flag; support is auto-detected. The GDAKI GPU functional test needs
`nvcc`, but the plugin itself does not (it uses the **CUDA driver API**, moved from the
CUDA Runtime API in master `6eb9a96`/`62b63cc`).

**Vendored device sources.** The GDAKI device-side dependency is vendored under
`3rd-party/efa-gda/` (EFA-GDA CUDA sources from `efa-dp-direct`, pinned at commit
`81c4dc7`; commits `27920ab`, `4735ddd`, `1383e93`, `38febcc`, `68b016a`).
`3rd-party/efa-gda` is committed directly into the repo (not a git submodule); the
on-GPU queue layout is compatible with the `efa_cuda_qp`/`efa_cuda_cq` types the device
code consumes. `configure.ac` fails early if `3rd-party/efa-gda/CUDA/README.md` is
missing.

**Runtime (`nccl_ofi_gin_gdaki_capable()`, [include/rdma/gin/nccl_ofi_gin_gdaki.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/rdma/gin/nccl_ofi_gin_gdaki.h)).**
GDAKI runs only if compiled in **and** the runtime satisfies:

- runtime libfabric ≥ 2.5 (`FI_VERSION_GE(fi_version(), FI_VERSION(2,5))`),
- DMA-BUF viable (`nccl_ofi_dmabuf_viable()`),
- the provider name starts with `"efa"` (efa / efa-direct — the only family exposing GDA ops),
- GDRCopy 2.5+ (forced PCIe copy) — checked in `nccl_ofi_gin_init`.

On AWS, per `doc/gin-getting-started.md`, EFA-GDA additionally needs a P5en / P6-B200 /
P6-B300 instance, libfabric 2.6.0+, rdma-core `64.0amzn0`+, EFA driver 3.3.0+, and the
NVIDIA driver loaded with `PeerMappingOverride=1`. The hardware completion counter is
gated by `OFI_NCCL_GDAKI_EFA_HW_COUNTER` (AUTO/ON/OFF; AUTO follows
`PlatformAWS::config_gdaki_domain`). If a requested counter cannot be opened, GDAKI
context creation **fails rather than silently falling back**.

### Source tree layout

The plugin's tree is organised into subdirectories (fix any older docs that referenced
flat `src/nccl_ofi_tuner.cpp` / `src/nccl_ofi_gin*.cpp` paths):

```
src/cm/          Connection Manager
src/rdma/gin/    GIN + GDAKI (moved out of flat src/)
src/stats/       statistics, incl. histogram.cpp
src/tuner/       cost-model tuner (nccl_ofi_tuner.cpp)
include/{cm, internal, internal/tuner, ofi, rdma, rdma/gin, stats, tracing_impl, tuner}
```

Notable newer headers:

- `include/nccl_ofi_device_copy.h` — GDRCopy/device-copy abstraction (`get_device_copy()`, forced-PCIe-copy check used by GIN)
- `include/rdma/gin/nccl_ofi_gin_base.h` — shared GIN base types
- `include/nccl_ofi_mpsc_ring.h`, `include/nccl_ofi_spsc_ring.h` — lock-free MPSC/SPSC ring buffers
- `include/nccl_ofi_spinlock.h` — spinlock primitive
- `include/nccl_ofi_tsa.h` — Clang thread-safety-analysis annotations (capability/guarded-by attributes)

### Histogram statistics (`src/stats/histogram.cpp`)

A histogram collection/printing API was added under `src/stats/` (master `6d46493`),
guarded by a build-time toggle (`CHECK_ENABLE_HISTOGRAM` in `configure.ac`, controlling
`histograms_enabled_cfg`). Key entry points in
[src/stats/histogram.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/stats/histogram.cpp):
`histograms_enabled()` / `enable_histograms()`, a process-wide
`global_histogram_aggregator` singleton (`get_instance()`, `load_env_override()` for
print-format overrides), and a `print_table()` formatter that renders bucketed counts.
It is used to profile size/latency distributions on the data path when enabled.

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
