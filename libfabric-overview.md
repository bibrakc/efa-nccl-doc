# Libfabric Overview and APIs

## Introduction

Libfabric (also called OFI - OpenFabrics Interfaces) is a framework for exporting fabric communication services to applications. It provides a vendor-agnostic API for high-performance networking.

**Repository**: https://github.com/ofiwg/libfabric

**Version baseline**: this document reflects libfabric **2.7.0rc1** on `main`
(release notes head `v2.7.0`, [NEWS.md](https://github.com/ofiwg/libfabric/blob/main/NEWS.md);
`AC_INIT([libfabric], [2.7.0rc1], …)` in
[configure.ac](https://github.com/ofiwg/libfabric/blob/main/configure.ac)). It was
previously written against 2.6.x. Header line references below are cited against the
current `include/rdma/*.h`.

**What changed in the API surface since 2.6.x (relevant to this file):**
- **New FI_XPU capability + XPU API** — a standardized *device-initiated*
  (GPU-side) communication API: `include/rdma/fi_xpu.h` (host side) and
  `include/rdma/fi_xpu_device.h` (device/kernel side). Directly relevant to
  GPU-initiated networking / GDAKI. See [XPU API (device-initiated communication)](#xpu-api-device-initiated-communication).
- **`fi_av_lookup2` now has a default implementation in all providers**
  (core commit `e0f1e0285`, `ofi_av_lookup2` in
  [prov/util/src/util_av.c:941](https://github.com/ofiwg/libfabric/blob/main/prov/util/src/util_av.c)).
- **ABI bump for `fi_domain_attr`** — the struct grew a `max_xpu_ctx_cnt` field
  (and other XPU-related fields on `fi_ep_attr`/`fi_cq_attr`/`fi_cntr_attr`); the
  previously-missing ABI bump was corrected in 2.7
  (["Fix the missing ABI bump for fi_domain_attr"](https://github.com/ofiwg/libfabric/blob/main/NEWS.md)).
- **EFA data path no longer calls rdma-core per operation by default** —
  `FI_EFA_USE_DATA_PATH_DIRECT` now defaults to `true`. rdma-core still owns the control
  path and publishes the queue/doorbell pointers the provider writes to. See
  [Libfabric with EFA → Data Path Direct](#efa-data-path-direct-default) and the deeper
  treatment in [efa-provider.md](efa-provider.md) and
  [rdma-core-and-verbs.md](rdma-core-and-verbs.md).
- **`fi_tostr` now prints XPU attributes** (`FI_XPU` cap, `xpu_ctx`,
  `FI_MR_XPU_DESC`, `max_xpu_ctx_cnt` in
  [src/fi_tostr.c](https://github.com/ofiwg/libfabric/blob/main/src/fi_tostr.c)).

Companion documents: [efa-provider.md](efa-provider.md) (EFA provider internals,
data-path-direct mechanism, protocols, locking) and
[rdma-core-and-verbs.md](rdma-core-and-verbs.md) (libibverbs / rdma-core control &
fallback data path). This file is the transport-agnostic libfabric API overview and
links to those two rather than duplicating them.

## Architecture

```
┌──────────────────────────────────────────────┐
│          Application (OFI Plugin)            │
└────────────────┬─────────────────────────────┘
                 │ Libfabric API
┌────────────────▼─────────────────────────────┐
│           Libfabric Core                     │
│  ┌────────────────────────────────────┐     │
│  │   Provider-Agnostic Layer          │     │
│  │   - Capability negotiation         │     │
│  │   - Resource management            │     │
│  │   - Common utilities               │     │
│  └────────────────────────────────────┘     │
└────────────────┬─────────────────────────────┘
                 │ Provider Interface
┌────────────────▼─────────────────────────────┐
│             Provider Layer                   │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐    │
│  │ EFA  │  │ verbs│  │ sockets│ │ tcp │    │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘    │
└─────┼─────────┼─────────┼─────────┼─────────┘
      │         │         │         │
┌─────▼─────────▼─────────▼─────────▼─────────┐
│        Operating System / Drivers            │
└──────────────────────────────────────────────┘
```

### EFA Provider ↔ rdma-core layering (default in 2.7)

By default the EFA provider **does not** call libibverbs on the data path.
`FI_EFA_USE_DATA_PATH_DIRECT` defaults to `true`
([prov/efa/src/efa_env.c:41](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_env.c):
`.use_data_path_direct = true`), so the provider writes work-queue entries into the
mmap'd SQ/RQ and reads CQEs out of the mmap'd CQ buffer itself, ringing the doorbell
via MMIO. libibverbs is used only for the **control path** and as the **fallback
data path** when direct mode is disabled/unavailable:

```
┌──────────────────────────────────────────────┐
│          Application (OFI Plugin)            │
└────────────────┬─────────────────────────────┘
                 │ Libfabric API (fi_*)
┌────────────────▼─────────────────────────────┐
│        Libfabric EFA Provider                │
│                                              │
│   data_path_direct_enabled ?  (default YES)  │
│      │                     │                 │
│    YES│                  NO│  (fallback)     │
│      ▼                     ▼                 │
│  ┌────────────┐    ┌───────────────────┐    │
│  │ DATA PATH  │    │ libibverbs        │    │
│  │ DIRECT     │    │ ibv_wr_* /        │    │
│  │ mmap'd SQ/ │    │ ibv_start_poll    │    │
│  │ RQ/CQ +    │    │ (classic verbs    │    │
│  │ MMIO doorbl│    │  data path)       │    │
│  └─────┬──────┘    └─────────┬─────────┘    │
├────────┼─────────────────────┼──────────────┤
│        │   rdma-core / libibverbs           │ ← CONTROL PATH (always):
│        │   device/QP/CQ/MR/AH setup +        │   open, QP/CQ create, MR reg,
│        │   mmap() of queue buffers           │   AH create, queue mmap
├────────┼─────────────────────┼──────────────┤
│              EFA Kernel Driver               │
├──────────────────────────────────────────────┤
│              EFA Hardware                    │
└──────────────────────────────────────────────┘
```

Both branches write the same hardware WQE/CQE format into the same mmap'd buffers;
the only difference is whether libibverbs or the provider's own inline code does the
writing. Full mechanism (WQE build, doorbell, request-ID scheme, QP generation,
CQE decode) is in [efa-provider.md](efa-provider.md) and
[rdma-core-and-verbs.md](rdma-core-and-verbs.md); see also
[Data Path Direct](#efa-data-path-direct-default) below.

## Core Concepts

### Fabric Objects Hierarchy

```
fi_fabric (Fabric Domain)
    │
    ├─ fi_domain (Resource Domain)
    │     │
    │     ├─ fi_endpoint (Communication Endpoint)
    │     │     │
    │     │     ├─ fi_cq (Completion Queue)
    │     │     ├─ fi_cntr (Counter)
    │     │     └─ fi_eq (Event Queue)
    │     │
    │     ├─ fi_mr (Memory Region)
    │     ├─ fi_av (Address Vector)
    │     └─ fi_wait (Wait Set)
    │
    └─ fi_passive_ep (Passive Endpoint)
```

### Fabric

Top-level object representing a fabric provider:

```c
struct fi_info *info;
struct fid_fabric *fabric;

// Get provider info
fi_getinfo(FI_VERSION(2, 7), NULL, NULL, 0, &hints, &info);

// Open fabric
fi_fabric(info->fabric_attr, &fabric, NULL);
```

(`struct fi_info` [include/rdma/fabric.h:482-498](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h);
`fi_getinfo()` [include/rdma/fabric.h:603](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h);
`struct fid_fabric` [include/rdma/fabric.h:630-634](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h);
`fi_fabric()` [include/rdma/fabric.h:638](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h))

**Responsibilities:**
- Provider selection
- Fabric-wide resources
- Domain creation

### Domain

Resource domain within a fabric:

```c
struct fid_domain *domain;

fi_domain(fabric, info, &domain, NULL);
```

(`struct fid_domain` [include/rdma/fi_domain.h:353-357](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h);
`fi_domain()` [include/rdma/fi_domain.h:367](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h).
A second variant `fi_domain2()` [include/rdma/fi_domain.h:374](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h)
takes an extra `flags` argument.)

**Responsibilities:**
- Memory registration scope
- Endpoint creation
- Completion queue creation
- Address vector creation
- **XPU context creation** (new in 2.7 — `fi_xpu_ctx()`, see
  [XPU API](#xpu-api-device-initiated-communication))

**Domain Attributes:**

**`struct fi_domain_attr`** - Domain attributes
([include/rdma/fabric.h:436-472](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h)):

```c
struct fi_domain_attr {
  struct fid_domain	*domain;
  char			*name;
  enum fi_threading	threading;      // Thread safety
  enum fi_progress	control_progress;
  union {
    enum fi_progress	data_progress;  // aliased with progress
    enum fi_progress	progress;
  };
  enum fi_resource_mgmt	resource_mgmt;
  enum fi_av_type	av_type;
  int			mr_mode;        // Memory registration mode (bitmask)
  size_t		mr_key_size;
  size_t		cq_data_size;
  size_t		cq_cnt;
  size_t		ep_cnt;
  size_t		tx_ctx_cnt;
  size_t		rx_ctx_cnt;
  size_t		max_ep_tx_ctx;
  size_t		max_ep_rx_ctx;
  size_t		max_ep_stx_ctx;
  size_t		max_ep_srx_ctx;
  size_t		cntr_cnt;
  size_t		mr_iov_limit;
  uint64_t		caps;
  uint64_t		mode;
  uint8_t		*auth_key;
  size_t		auth_key_size;
  size_t		max_err_data;
  size_t		mr_cnt;
  uint32_t		tclass;
  size_t		max_ep_auth_key;
  uint32_t		max_group_id;
  uint64_t		max_cntr_value;
  uint64_t		max_err_cntr_value;
  size_t		max_xpu_ctx_cnt;  // NEW in 2.7: # of XPU contexts the
                                          // domain supports (0 = no XPU support)
};
```

> **ABI note (2.7):** `data_progress` is now a **union** with a `progress` alias,
> and the struct gained `max_xpu_ctx_cnt` at the end. The previously-missing ABI
> bump for `fi_domain_attr` was corrected in 2.7
> (["Fix the missing ABI bump for fi_domain_attr"](https://github.com/ofiwg/libfabric/blob/main/NEWS.md)).
> `max_xpu_ctx_cnt > 0` is how an application detects XPU support on a domain;
> `1` means a 1:1 domain↔XPU mapping (see [fi_xpu(3)](https://github.com/ofiwg/libfabric/blob/main/man/fi_xpu.3.md)).

### Endpoint

Communication endpoint for data transfer (`struct fid_ep`
[include/rdma/fi_endpoint.h:151-160](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h)):

```c
struct fid_ep *ep;

fi_endpoint(domain, info, &ep, NULL);

// Bind resources
fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV);
fi_ep_bind(ep, &av->fid, 0);

// Enable
fi_enable(ep);
```

(`fi_endpoint()` [include/rdma/fi_endpoint.h:187](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h),
`fi_ep_bind()` [include/rdma/fi_endpoint.h:212](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h),
`fi_enable()` [include/rdma/fi_endpoint.h:227](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h).
For XPU endpoints use `fi_endpoint2()`
[include/rdma/fi_endpoint.h:194](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h)
with the `FI_XPU` flag — see [XPU API](#xpu-api-device-initiated-communication).)

**Endpoint Types:**
1. **FI_EP_MSG**: Connected, message-oriented (like TCP)
2. **FI_EP_RDM**: Reliable datagram (like UD with reliability)
3. **FI_EP_DGRAM**: Unreliable datagram (like UDP)

**EFA uses FI_EP_RDM** - reliable datagram with provider-level reliability

### Completion Queue (CQ)

Reports completion of asynchronous operations (`struct fid_cq`
[include/rdma/fi_eq.h:283-286](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h)):

```c
struct fid_cq *cq;
struct fi_cq_attr cq_attr = {
  .size = 1024,
  .format = FI_CQ_FORMAT_DATA,
  .wait_obj = FI_WAIT_NONE,  // Polling mode
};

fi_cq_open(domain, &cq_attr, &cq, NULL);
```

(`fi_cq_open()` [include/rdma/fi_domain.h:392](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h).
Note `struct fi_cq_attr`
[include/rdma/fi_eq.h:254-263](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h)
gained an `xpu_ctx` field in 2.7 for XPU CQs.)

**CQ Formats:**
- **FI_CQ_FORMAT_CONTEXT**: Just context pointer
- **FI_CQ_FORMAT_MSG**: Context + flags
- **FI_CQ_FORMAT_DATA**: Context + flags + data
- **FI_CQ_FORMAT_TAGGED**: For tagged messaging

**Polling:**
```c
struct fi_cq_data_entry entry[16];
int ret = fi_cq_read(cq, entry, 16);

for (int i = 0; i < ret; i++) {
  void *context = entry[i].op_context;
  size_t len = entry[i].len;
  uint64_t flags = entry[i].flags;
  // Process completion
}
```

(`struct fi_cq_data_entry` [include/rdma/fi_eq.h:215-222](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h),
`fi_cq_read()` [include/rdma/fi_eq.h:404](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h))

### Address Vector (AV)

Maps addresses for connection-less endpoints (`struct fid_av`
[include/rdma/fi_domain.h:117-120](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h)):

```c
struct fid_av *av;
struct fi_av_attr av_attr = {
  .type = FI_AV_MAP,  // or FI_AV_TABLE
  .count = 256,
};

fi_av_open(domain, &av_attr, &av, NULL);

// Insert remote address
fi_addr_t remote_addr;
fi_av_insert(av, &peer_addr, 1, &remote_addr, 0, NULL);
```

(`fi_av_open()` [include/rdma/fi_domain.h:521](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h),
`fi_av_insert()` [include/rdma/fi_domain.h:534](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h).
The AV API moved into `fi_domain.h`; it is no longer in `fi_cm.h`.)

**Address lookup:**

```c
// Classic lookup: get the provider address back for a fi_addr_t
fi_av_lookup(av, remote_addr, addr_buf, &addrlen);

// Extended lookup with flags + optional XPU context (new default in 2.7)
fi_av_lookup2(av, remote_addr, raw_buf, &len, FI_XPU, xpu_ctx);
```

(`fi_av_lookup()` [include/rdma/fi_domain.h:563](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h),
`fi_av_lookup2()` [include/rdma/fi_domain.h:601](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h))

> **New in 2.7 — `fi_av_lookup2` default support in all providers.** `fi_av_lookup2`
> is an extended `fi_av_lookup` that takes `flags` and an `struct fid_xpu_ctx *`. It
> now has a **default core implementation** (`ofi_av_lookup2` in
> [prov/util/src/util_av.c:941](https://github.com/ofiwg/libfabric/blob/main/prov/util/src/util_av.c),
> wired into the util AV ops), so every provider supports it even without a
> custom override — core commit `e0f1e0285`
> (["Add default fi_av_lookup2 support to all providers"](https://github.com/ofiwg/libfabric/blob/main/NEWS.md)).
> Passing `FI_XPU` + an XPU context returns a **raw, device-usable address** (size
> reported by `fi_xpu_ctx_query()`), letting the same AV serve multiple XPUs — see
> [XPU API](#xpu-api-device-initiated-communication).

**AV Types:**
- **FI_AV_MAP**: Hash table lookup
- **FI_AV_TABLE**: Direct indexing (faster)

**Usage:**
- Required for RDM endpoints (like EFA)
- Maps peer addresses to fi_addr_t handles
- Used in send/recv/RMA operations

### Memory Region (MR)

Registered memory for RDMA (`struct fid_mr`
[include/rdma/fi_domain.h:134-138](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h)):

```c
struct fid_mr *mr;
void *buf = malloc(size);

uint64_t access = FI_SEND | FI_RECV | FI_REMOTE_READ | FI_REMOTE_WRITE;
fi_mr_reg(domain, buf, size, access, 0, 0, 0, &mr, NULL);

// Get descriptor for data operations
void *desc = fi_mr_desc(mr);

// Get key for remote access
uint64_t rkey = fi_mr_key(mr);
```

(`fi_mr_reg()` [include/rdma/fi_domain.h:420](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h),
`fi_mr_regattr()` [include/rdma/fi_domain.h:439](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h),
`fi_mr_desc()` [include/rdma/fi_domain.h:445](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h).
For XPU-initiated transfers, `fi_mr_get_xpu_desc()`
[include/rdma/fi_domain.h:509](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h)
returns a raw, device-usable descriptor — see
[XPU API](#xpu-api-device-initiated-communication). MRs can now carry the
`FI_MR_XPU_DESC` mr_mode bit
([include/rdma/fabric.h:253](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h)).)

**Access Flags:**
- **FI_SEND**: Can be source for send
- **FI_RECV**: Can be target for recv
- **FI_READ**: Can be source for RDMA read
- **FI_WRITE**: Can be target for RDMA write
- **FI_REMOTE_READ**: Remote can read
- **FI_REMOTE_WRITE**: Remote can write

**Memory Types:**
```c
// Host memory
fi_mr_reg(domain, buf, size, access, 0, 0, 0, &mr, NULL);

// GPU memory (CUDA)
fi_mr_reg(domain, gpu_buf, size, access, 0, 0, FI_HMEM_CUDA, &mr, NULL);
```

## Data Transfer Operations

### Message Operations

#### Send

**`fi_send()` / `fi_senddata()` / `fi_inject()`**
([include/rdma/fi_endpoint.h:326](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h),
[include/rdma/fi_endpoint.h:352](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h),
[include/rdma/fi_endpoint.h:346](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h)):

```c
// Standard send
ssize_t fi_send(struct fid_ep *ep,
                const void *buf,
                size_t len,
                void *desc,
                fi_addr_t dest_addr,
                void *context);

// Send with immediate data
ssize_t fi_senddata(struct fid_ep *ep,
                    const void *buf,
                    size_t len,
                    void *desc,
                    uint64_t data,
                    fi_addr_t dest_addr,
                    void *context);

// Inject (small, eager)
ssize_t fi_inject(struct fid_ep *ep,
                  const void *buf,
                  size_t len,
                  fi_addr_t dest_addr);
```

**Characteristics:**
- **fi_send**: Asynchronous, requires completion
- **fi_inject**: Synchronous, no completion, limited size (< 4 KB)
- **fi_senddata**: Carries immediate data to receiver

> The basic two-sided message ops (`fi_send`/`fi_recv`/`fi_inject`/`fi_senddata`)
> are `static inline` wrappers in `include/rdma/fi_endpoint.h` (they dispatch through
> `ep->msg->…`); there is **no** separate `fi_msg.h` header in this tree.

#### Receive

**`fi_recv()`** ([include/rdma/fi_endpoint.h:306](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h)):

```c
ssize_t fi_recv(struct fid_ep *ep,
                void *buf,
                size_t len,
                void *desc,
                fi_addr_t src_addr,
                void *context);
```

**Pre-posting:**
- Must be posted before data arrives (for RDM)
- Matched with incoming sends
- Can use FI_ADDR_UNSPEC for any source

### Tagged Messages

For message matching by tag:

**`fi_tsend()` / `fi_trecv()`**
([include/rdma/fi_tagged.h:121](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_tagged.h),
[include/rdma/fi_tagged.h:98](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_tagged.h)):

```c
// Tagged send
ssize_t fi_tsend(struct fid_ep *ep,
                 const void *buf,
                 size_t len,
                 void *desc,
                 fi_addr_t dest_addr,
                 uint64_t tag,
                 void *context);

// Tagged receive
ssize_t fi_trecv(struct fid_ep *ep,
                 void *buf,
                 size_t len,
                 void *desc,
                 fi_addr_t src_addr,
                 uint64_t tag,
                 uint64_t ignore,
                 void *context);
```

**Tag Matching:**
```c
// Receive matches if: (recv_tag & ~ignore) == (send_tag & ~ignore)

// Example: receive any tag in range 0x100-0x1FF
fi_trecv(ep, buf, len, desc, addr,
         0x100,      // tag
         0x0FF,      // ignore (lower 8 bits)
         context);
```

**NCCL uses tagged messages for channel identification**

### RMA Operations

Direct memory access without remote CPU involvement:

#### Read

**`fi_read()`** ([include/rdma/fi_rma.h:98](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_rma.h)):

```c
ssize_t fi_read(struct fid_ep *ep,
                void *buf,
                size_t len,
                void *desc,
                fi_addr_t src_addr,
                uint64_t remote_addr,
                uint64_t remote_key,
                void *context);
```

**Flow:**
```
Local                           Remote
  │                               │
  ├─ fi_read() ──────────────────→│ (no CPU)
  │                               │
  │←──────────── Data ────────────│ (DMA)
  │                               │
  CQ completion                   (silent)
```

#### Write

**`fi_write()`** ([include/rdma/fi_rma.h:119](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_rma.h)):

```c
ssize_t fi_write(struct fid_ep *ep,
                 const void *buf,
                 size_t len,
                 void *desc,
                 fi_addr_t dest_addr,
                 uint64_t remote_addr,
                 uint64_t remote_key,
                 void *context);
```

**Flow:**
```
Local                           Remote
  │                               │
  ├─ fi_write() ─────────────────→│ (no CPU)
  │                               │
  │────────────► Data ───────────→│ (DMA)
  │                               │
  CQ completion                   (silent)
```

**Key Exchange:**
- Remote memory must be registered
- Key (rkey) must be communicated out-of-band
- Address must be communicated out-of-band

### Atomic Operations

Atomic read-modify-write:

**`fi_atomic()`** ([include/rdma/fi_atomic.h:150](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_atomic.h)):

```c
ssize_t fi_atomic(struct fid_ep *ep,
                  const void *buf,
                  size_t count,
                  void *desc,
                  fi_addr_t dest_addr,
                  uint64_t remote_addr,
                  uint64_t remote_key,
                  enum fi_datatype datatype,
                  enum fi_op op,
                  void *context);
```

**Supported Ops:**
- FI_MIN, FI_MAX
- FI_SUM, FI_PROD
- FI_ATOMIC_WRITE, FI_ATOMIC_READ
- FI_CSWAP (compare-and-swap)

**EFA has limited atomic support**

## Capabilities

Libfabric uses capability bits to describe what operations are supported:

```c
struct fi_info {
  uint64_t caps;  // Capabilities
  uint64_t mode;  // Required modes
  // ...
};
```

### Capability Flags

Verified against [include/rdma/fabric.h:129-173](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h):

```c
// Primary capabilities
FI_MSG         // (1ULL << 1)  Message queue operations (send/recv)
FI_RMA         // (1ULL << 2)  RDMA operations (read/write)
FI_TAGGED      // (1ULL << 3)  Tagged messaging
FI_ATOMIC      // (1ULL << 4)  Atomic operations (FI_ATOMICS is an alias)
FI_MULTICAST   // (1ULL << 5)  Multicast
FI_COLLECTIVE  // (1ULL << 6)  Collective operations (rare)

// Secondary capabilities
FI_READ         // (1ULL << 8)   Can initiate RDMA read
FI_WRITE        // (1ULL << 9)   Can initiate RDMA write
FI_RECV         // (1ULL << 10)  Can receive messages
FI_SEND         // (1ULL << 11)  Can send messages (FI_TRANSMIT alias)
FI_REMOTE_READ  // (1ULL << 12)  Can be target of remote read
FI_REMOTE_WRITE // (1ULL << 13)  Can be target of remote write

// Features
FI_MULTI_RECV     // (1ULL << 16)  Multi-buffer receives
FI_REMOTE_CQ_DATA // (1ULL << 17)  Carry immediate/remote CQ data
FI_MORE           // (1ULL << 18)  Batch/coalesce hint
FI_TRIGGER        // (1ULL << 20)  Triggered operations
FI_FENCE          // (1ULL << 21)  Ordering via fences
FI_INJECT         // (1ULL << 25)  Eager/inline sends

// Newer capability bits relevant here
FI_PEER        // (1ULL << 43)  Peer/utility provider composition
FI_XPU         // (1ULL << 44)  Device-initiated (XPU/GPU) communication — NEW in 2.7
FI_HMEM        // (1ULL << 47)  Heterogeneous (device) memory support
```

> **`FI_XPU` (bit 44)** — new in 2.7. When set in `fi_getinfo` hints it requests a
> provider that can export EP/CQ/CNTR objects for device-initiated data-path
> operations. See [XPU API](#xpu-api-device-initiated-communication). `FI_INJECT`
> is a data-transfer op-flag/mode rather than a queried primary cap on EFA.

### Hints and Selection

```c
struct fi_info hints = {
  .caps = FI_MSG | FI_TAGGED | FI_RMA,
  .mode = FI_CONTEXT,
  .ep_attr = {
    .type = FI_EP_RDM,
  },
  .domain_attr = {
    .threading = FI_THREAD_SAFE,
    .mr_mode = FI_MR_LOCAL,
  },
};

struct fi_info *info;
fi_getinfo(FI_VERSION(2, 7), NULL, NULL, 0, &hints, &info);
```

**fi_getinfo() selects provider matching hints**

## Threading and Progress

### Threading Models

`enum fi_threading` ([include/rdma/fabric.h:262-269](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h)):

```c
enum fi_threading {
  FI_THREAD_UNSPEC,      // Provider default
  FI_THREAD_SAFE,        // Fully thread-safe (no app serialization needed)
  FI_THREAD_FID,         // App serializes per fabric object (fid)
  FI_THREAD_DOMAIN,      // App serializes per domain / per set of resources
  FI_THREAD_COMPLETION,  // App serializes per completion domain (CQ-partitioned)
  FI_THREAD_ENDPOINT,    // App serializes per endpoint
};
```

> **Correction vs older text:** the enum has **six** values; there is no
> `FI_THREAD_MAP`, and `FI_THREAD_ENDPOINT` was missing before. The descriptions
> follow the OFI serialization model (the value states *what the application must
> serialize*, not what the provider protects internally).

**EFA provider:** advertises **`FI_THREAD_DOMAIN`** as the default in
`prov/efa/src/efa_prov_info.c:45` and promotes it to **`FI_THREAD_SAFE`** for the
RDM info (`efa_prov_info.c:104`, `:596`). Per
[man/fi_efa.7.md](https://github.com/ofiwg/libfabric/blob/main/man/fi_efa.7.md),
both RDM and DGRAM endpoints now support **`FI_THREAD_SAFE`**,
**`FI_THREAD_COMPLETION`** and **`FI_THREAD_DOMAIN`**. The provider-internal locking
rework behind this is detailed in [efa-provider.md](efa-provider.md).

### Progress Models

`enum fi_progress` ([include/rdma/fabric.h:255-260](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h)):

```c
enum fi_progress {
  FI_PROGRESS_UNSPEC,
  FI_PROGRESS_AUTO,             // Provider drives progress (e.g. background thread)
  FI_PROGRESS_MANUAL,           // App must call fi_cq_read()/fi_cntr_read() to progress
  FI_PROGRESS_CONTROL_UNIFIED,  // Control progress unified with data progress
};
```

> **New value: `FI_PROGRESS_CONTROL_UNIFIED`.** It indicates that control-path
> progress does not need to be driven separately from data-path progress — the
> application only progresses the data path. The **EFA provider** uses this: when a
> domain is opened with `control_progress == FI_PROGRESS_CONTROL_UNIFIED` (typically
> together with `FI_THREAD_DOMAIN`), the provider makes the `util_domain` lock a
> **no-op** (`OFI_LOCK_NOOP`) instead of a mutex
> ([prov/efa/src/efa_domain_util.c:71-74](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_domain_util.c)),
> eliminating that lock from the fast path. See
> [efa-provider.md](efa-provider.md) for the locking discussion.

**Control Progress**: For connection management / control operations
**Data Progress**: For data transfer completions
(`data_progress` is unioned with `progress` in `fi_domain_attr`.)

**EFA defaults** (from `prov/efa/src/efa_prov_info.c`):
- Base info: control `FI_PROGRESS_AUTO`, data `FI_PROGRESS_AUTO` (lines 46-47)
- RDM info: control `FI_PROGRESS_MANUAL`, data `FI_PROGRESS_MANUAL` (lines 611-612)

**Manual progress requires the application to poll the CQ/counter.** An application
may request `FI_PROGRESS_CONTROL_UNIFIED` in hints to get the reduced-locking domain
path described above.

## XPU API (device-initiated communication)

**New in libfabric 2.7.** The XPU API is the standardized, provider-agnostic
*device-initiated* communication interface: it lets an accelerator ("XPU" — a GPU
or similar device) **post data transfers and read completions from device kernels**,
while the host CPU still performs all control operations. This is the OFI-level
substrate for GPU-initiated networking / GDAKI. The first provider to expose
device-side XPU dispatch is EFA (`FI_XPU_PROV_EFA = 1`).

Two headers:
- **Host side:** [include/rdma/fi_xpu.h](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_xpu.h)
- **Device/kernel side:** [include/rdma/fi_xpu_device.h](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_xpu_device.h)

Documented semantics: [man/fi_xpu.3.md](https://github.com/ofiwg/libfabric/blob/main/man/fi_xpu.3.md)
(`fi_xpu(3)`). Both headers were added to the build specs, `fi_tostr` now prints XPU
attrs, and `fi_domain_attr` was ABI-bumped for `max_xpu_ctx_cnt` — all in the 2.7
[NEWS.md](https://github.com/ofiwg/libfabric/blob/main/NEWS.md).

### Model

> As a general rule (from `fi_xpu(3)`): **control-path operations may only be invoked
> by the host CPU; data-path functions may only be invoked by the specified XPU.**
> EP, CQ and counter objects may be *exported* to an XPU. AV and MR lookups produce
> raw data usable by the XPU, but the AV/MR objects themselves stay host-only.

### The FI_XPU capability

`FI_XPU` is capability bit **`(1ULL << 44)`**
([include/rdma/fabric.h:169](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h)).
An application sets it in `fi_getinfo` hints to request an XPU-capable provider.
Support on the resulting domain is reported by
`fi_domain_attr::max_xpu_ctx_cnt` (0 = none, 1 = 1:1 domain↔XPU).

### XPU context (`fid_xpu_ctx`)

An XPU context groups the EP/CQ/CNTR resources that target one device and binds them
to it. Created from a domain via the domain op `xpu_ctx`
([include/rdma/fi_domain.h:329-331](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h),
wrapped by `fi_xpu_ctx()` in `fi_xpu.h`):

```c
// fi_xpu.h — input attribute
struct fi_xpu_attr {
  enum fi_hmem_iface  iface;    // e.g. FI_HMEM_CUDA, FI_HMEM_ZE
  uint64_t            device;   // XPU device ordinal
  struct fi_xpu_ops   *ops;     // optional memory callbacks (may be NULL)
};

int fi_xpu_ctx(struct fid_domain *domain, struct fi_xpu_attr *attr,
               struct fid_xpu_ctx **ctx, void *context);

// query capabilities/sizes for this context
int fi_xpu_ctx_query(struct fid_xpu_ctx *ctx, struct fi_xpu_ctx_attr *attr);

struct fi_xpu_ctx_attr {
  uint64_t  caps;          // FI_XPU_CAP_EP | _CQ | _CNTR
  size_t    av_addr_size;  // size of a raw AV addr from fi_av_lookup2(FI_XPU)
  size_t    mr_desc_size;  // size of a raw MR desc from fi_mr_get_xpu_desc(FI_XPU)
};

#define FI_XPU_CAP_EP    (1ULL << 0)
#define FI_XPU_CAP_CQ    (1ULL << 1)
#define FI_XPU_CAP_CNTR  (1ULL << 2)
```

The XPU context is threaded into resource creation via **new attribute fields**:
- `fi_ep_attr::xpu_ctx`   ([include/rdma/fabric.h:419-434](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fabric.h))
- `fi_cq_attr::xpu_ctx`   ([include/rdma/fi_eq.h:262](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h))
- `fi_cntr_attr::xpu_ctx` ([include/rdma/fi_eq.h:304](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h))

An XPU EP is created with `fi_endpoint2()` + the `FI_XPU` flag; XPU CQ/CNTR are
created by setting `FI_XPU` in `fi_cq_attr->flags` / `fi_cntr_attr->flags` plus
`xpu_ctx` before `fi_cq_open`/`fi_cntr_open`. All resources bound to one XPU EP must
share the same context (a host CQ without `xpu_ctx` may additionally be bound for
error handling).

### Exporting resources to the device

Once bound and enabled, host-side "export" calls hand the resource to the device.
Each returns a handle whose first member is `struct fid_xpu` (carrying `fclass`,
`prov_id`, and a provider-internal `prov_ctx`):

```c
int fi_ep_export_xpu  (struct fid_ep   *ep,   uint64_t flags, struct fid_xpu_ep   *xpu_ep);
int fi_cq_export_xpu  (struct fid_cq   *cq,   uint64_t flags, struct fid_xpu_cq   *xpu_cq);
int fi_cntr_export_xpu(struct fid_cntr *cntr, uint64_t flags, struct fid_xpu_cntr *xpu_cntr);
```

- `fi_ep_export_xpu()`   — inline at [include/rdma/fi_endpoint.h:366](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h);
  backing op `export_xpu` at [fi_endpoint.h:110](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h).
- `fi_cq_export_xpu()`   — inline at [include/rdma/fi_eq.h:489](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h).
- `fi_cntr_export_xpu()` — in `fi_eq.h` (counter ops).

The exported handle structs (`fid_xpu_ep`, `fid_xpu_cq`, `fid_xpu_cntr`) each embed
`struct fid_xpu` as their first member (`fi_xpu.h`).

### AV / MR data for the device

AV and MR objects stay host-owned, but the device needs *raw* forms:

```c
// raw device-usable peer address (size = ctx_attr.av_addr_size)
int fi_av_lookup2(struct fid_av *av, fi_addr_t fi_addr, void *buf, size_t *len,
                  uint64_t flags /* FI_XPU */, struct fid_xpu_ctx *ctx);

// raw device-usable MR descriptor / key (size = ctx_attr.mr_desc_size)
int fi_mr_get_xpu_desc(struct fid_mr *mr, void *buf, size_t *len,
                       uint64_t flags /* FI_XPU */, struct fid_xpu_ctx *ctx);
```

- `fi_av_lookup2()`      — [include/rdma/fi_domain.h:601](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h)
  (now with a default all-provider implementation, see the [AV section](#address-vector-av)).
- `fi_mr_get_xpu_desc()` — [include/rdma/fi_domain.h:509](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h);
  MRs may set the `FI_MR_XPU_DESC` mr_mode bit.

Because both take an XPU context, the *same* AV or MR can be queried per-device to
get device-specific representations. The host copies the raw AV addr, raw MR desc,
and the exported EP/CQ/CNTR handles into device-accessible memory before launching
the kernel.

### Memory callbacks (`fi_xpu_ops`) and flags

`fi_xpu_attr.ops` optionally provides device-memory callbacks the provider invokes
when it needs to allocate/import/free device memory (e.g. for HW queue buffers):

```c
struct fi_xpu_ops {
  size_t size;   // = sizeof(struct fi_xpu_ops) — versioning
  int  (*alloc) (uint64_t device, uint64_t size, uint64_t alignment,
                 uint64_t flags, void **addr, int *fd, uint64_t *offset);
  int  (*import)(uint64_t device, void *host_addr, uint64_t size,
                 uint64_t flags, void **dev_addr);
  void (*free)  (uint64_t device, void *addr);
};
```

Callback flags (`fi_xpu.h`):

| Flag | Value | Passed to | Meaning |
|------|-------|-----------|---------|
| `FI_XPU_ALLOC_DMABUF`    | `1ULL << 0` | alloc  | allocation must be exportable as a DMA-BUF fd |
| `FI_XPU_IMPORT_IOMEMORY` | `1ULL << 0` | import | host address is PCIe BAR MMIO (device I/O memory) |
| `FI_XPU_IMPORT_DEVICEMAP`| `1ULL << 1` | import | resulting pointer must be accessible from XPU kernels |

### Device-side API (`fi_xpu_device.h`)

`fi_xpu_device.h` is compiled by a single XPU kernel compiler at a time (CUDA `nvcc`,
ROCm `hipcc`, or oneAPI/SYCL `icpx -fsycl`); the `FI_XPU_FUNC` macro picks the right
qualifier (`__device__ static inline`, etc.). It uses a **provider-ID dispatch**
model: every generic device function switches on `handle->fid.prov_id`
(`enum fi_xpu_provider`, values start at 1 for a tight jump table) to route to a
provider implementation. In this baseline the switches contain only the `default:
return -FI_ENOSYS` arm plus a documented pattern for adding
`#include <rdma/fi_xpu_device_efa.h>` and a `case FI_XPU_PROV_EFA:` — i.e. the generic
dispatch scaffolding is in place and providers plug in their per-op implementations.

**Scope argument.** Every device-side call takes a `scope` selecting the set of
cooperating threads issuing the operation collectively (enables the implementation to
coalesce/optimize):

| Scope | CUDA | SYCL |
|-------|------|------|
| `FI_XPU_WORK_ITEM`  | thread       | work item  |
| `FI_XPU_SUBGROUP`   | warp         | subgroup   |
| `FI_XPU_WORK_GROUP` | thread block | work group |
| `FI_XPU_DEVICE`     | device       | device     |

**Device-side data transfer** (mirror the host APIs; see `fi_msg`/`fi_tagged`/
`fi_rma`/`fi_atomic` for semantics):
`fi_xpu_send`, `fi_xpu_recv`, `fi_xpu_tsend`, `fi_xpu_trecv`, `fi_xpu_write`,
`fi_xpu_read`, `fi_xpu_atomic`, `fi_xpu_fetch_atomic`, `fi_xpu_compare_atomic` —
all take `struct fid_xpu_ep *` and a `scope`.

**Device-side completion:**
- CQ: `fi_xpu_cq_read`, `fi_xpu_cq_readfrom`, `fi_xpu_cq_readerr`, `fi_xpu_cq_sread`,
  `fi_xpu_cq_sreadfrom` (on `struct fid_xpu_cq *`).
- Counter: `fi_xpu_cntr_read`, `fi_xpu_cntr_readerr`, `fi_xpu_cntr_wait`,
  `fi_xpu_cntr_add`, `fi_xpu_cntr_set`, `fi_xpu_cntr_adderr`, `fi_xpu_cntr_seterr`
  (on `struct fid_xpu_cntr *`).

Host-side reads (`fi_cq_read`, `fi_cq_sread`, `fi_cntr_read`, `fi_cntr_wait`) are
**not** available on an exported CQ/counter — the device owns the data path for those.

### Typical host setup flow (from `fi_xpu(3)`)

```
1.  hints->caps |= FI_XPU; fi_getinfo(FI_VERSION(2,7), …)
2.  fi_domain();  check info->domain_attr->max_xpu_ctx_cnt > 0
3.  fi_xpu_ctx(domain, &{iface=FI_HMEM_CUDA, device=0, ops=&my_ops}, &xpu_ctx)
4.  fi_xpu_ctx_query(xpu_ctx, &ctx_attr)  → caps, av_addr_size, mr_desc_size
5.  fi_av_open + fi_av_insert          (domain-level, no xpu_ctx)
6.  fi_cq_open / fi_cntr_open  with flags|=FI_XPU and .xpu_ctx = xpu_ctx
7.  info->ep_attr->xpu_ctx = xpu_ctx; fi_endpoint2(domain, info, &ep, FI_XPU, …)
8.  fi_mr_regattr()                    (domain-level)
9.  fi_ep_bind(av/cq/cntr) + fi_enable(ep)
10. fi_ep_export_xpu / fi_cq_export_xpu / fi_cntr_export_xpu
11. fi_av_lookup2(FI_XPU, xpu_ctx);  fi_mr_get_xpu_desc(FI_XPU, xpu_ctx)
12. copy raw addr/desc + exported handles to device memory
13. launch kernel → fi_xpu_send/…, fi_xpu_cntr_wait/…  (device-initiated)
14. fi_close() everything incl. xpu_ctx
```

**Relevance to EFA / GDAKI:** EFA is the first provider registered for device-side
dispatch (`FI_XPU_PROV_EFA`), so this API is the path by which a GPU kernel can post
EFA transfers and poll EFA completions without host round-trips. Provider-internal
EFA specifics are in [efa-provider.md](efa-provider.md); the data-path-direct queue
mechanics the device dispatch builds on are in
[rdma-core-and-verbs.md](rdma-core-and-verbs.md).

## Error Handling

### Completion Errors

```c
struct fi_cq_err_entry err_entry;
ret = fi_cq_readerr(cq, &err_entry, 0);

if (ret > 0) {
  printf("Error: %s\n", fi_strerror(err_entry.err));
  printf("Provider error: %s\n",
         fi_cq_strerror(cq, err_entry.prov_errno,
                        err_entry.err_data, NULL, 0));
}
```

(`struct fi_cq_err_entry` [include/rdma/fi_eq.h:233-247](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h),
`fi_cq_readerr()` [include/rdma/fi_eq.h:416](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h))

**Common Errors:**
- **FI_EAGAIN**: Resource busy, retry
- **FI_ETIMEDOUT**: Operation timed out
- **FI_ECANCELED**: Canceled
- **FI_EMSGSIZE**: Message too large
- **FI_EADDRNOTAVAIL**: Address not reachable

## Performance Features

### Batching

Multiple operations can be issued before polling:

```c
// Issue multiple sends
for (int i = 0; i < N; i++) {
  fi_send(ep, buf[i], len, desc, addr, ctx[i]);
}

// Single poll for all completions
fi_cq_read(cq, entries, N);
```

### Inject (Zero-Copy Send)

For small messages:

```c
// Data copied immediately, no completion
fi_inject(ep, small_buf, 128, dest_addr);

// Can reuse small_buf immediately
```

**Limits:**
- EFA: ~4 KB inject size
- No completion event
- Higher CPU cost

### Selective Completion

```c
// Send without completion
fi_send(ep, buf, len, desc | FI_INJECT_COMPLETE,
        addr, NULL);  // No context = no completion
```

### Memory Registration Cache

Some providers (including EFA) cache MRs:

```c
// First registration: slow (~100-500 μs)
fi_mr_reg(domain, buf1, size, access, 0, 0, 0, &mr1, NULL);

// Deregister
fi_close(&mr1->fid);

// Re-register same buffer: fast (cached)
fi_mr_reg(domain, buf1, size, access, 0, 0, 0, &mr1, NULL);
```

**Cache Tuning:**
```bash
FI_MR_CACHE_MAX_SIZE=unlimited
FI_MR_CACHE_MAX_COUNT=unlimited
FI_MR_CACHE_MONITOR=memhooks  # or userfaultfd
```

## Libfabric with EFA

### EFA-Specific Info

```c
struct fi_info *info;
fi_getinfo(FI_VERSION(2, 7), NULL, NULL, 0, &hints, &info);

// Provider name
info->fabric_attr->prov_name == "efa"

// Endpoint type
info->ep_attr->type == FI_EP_RDM

// Capabilities
info->caps & FI_MSG
info->caps & FI_TAGGED
info->caps & FI_RMA

// Max message size
info->ep_attr->max_msg_size  // ~2 GB on EFA

// Max RMA size
info->ep_attr->max_order_raw_size
```

### EFA Environment Variables

```bash
# Logging
FI_LOG_LEVEL=info      # or warn, debug, trace
FI_LOG_PROV=efa

# Queue sizes
FI_EFA_TX_SIZE=256     # Send queue depth
FI_EFA_RX_SIZE=256     # Recv queue depth

# Performance
# NOTE: FI_EFA_TX_IOV_LIMIT and FI_EFA_RX_IOV_LIMIT are DEPRECATED and FATAL --
# the provider aborts at init if either is set (prov/efa/src/efa_env.c:79,
# abort_deprecated_env_vars[], alongside FI_EFA_MTU_SIZE). Do not set them.
FI_EFA_CQ_SIZE=8192            # Completion queue depth

# Features
FI_EFA_ENABLE_SHM_TRANSFER=0  # Disable intra-node SHM
FI_EFA_USE_DEVICE_RDMA=1      # Enable RDMA
FI_EFA_USE_DATA_PATH_DIRECT=1 # Bypass libibverbs on data path (DEFAULT true)

# Memory registration
FI_MR_CACHE_MONITOR=memhooks
FI_MR_CACHE_MAX_SIZE=unlimited
```

### EFA Data Path Direct (default)

`FI_EFA_USE_DATA_PATH_DIRECT` **defaults to `true`**
([prov/efa/src/efa_env.c:41](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_env.c):
`.use_data_path_direct = true`; parsed at
[efa_env.c:162](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_env.c)).
With it enabled, the EFA provider **does not call into rdma-core / libibverbs on the
data path**. Instead it:

- writes send/recv Work-Queue Entries (WQEs) directly into the memory-mapped SQ/RQ,
- rings the hardware doorbell via its own MMIO helpers, and
- reads Completion-Queue Entries (CQEs) straight out of the memory-mapped CQ buffer.

rdma-core is still used for the **control path** (device open, QP/CQ creation, MR
registration, AH creation, and the `mmap()` of the queue buffers) and as the
**fallback data path** when direct mode is disabled (`FI_EFA_USE_DATA_PATH_DIRECT=0`),
the build lacks `HAVE_EFA_DATA_PATH_DIRECT`, or the device/rdma-core cannot hand back
the mapped queues. The runtime gate is in
[prov/efa/src/efa_data_path_direct.c:187](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_data_path_direct.c)
(`if (!efa_env.use_data_path_direct || efa_device_use_sub_cq()) …`).

This is why the layering diagram above shows libibverbs as **control-path / fallback**
rather than sitting under every data-path operation. The full mechanism — WQE build,
64/128-byte WQE format, MMIO doorbell, request-ID scheme, QP generation, CQE decode —
is documented in [efa-provider.md](efa-provider.md) and
[rdma-core-and-verbs.md](rdma-core-and-verbs.md); it is not repeated here.

## Summary

**Libfabric Abstractions:**
1. **Fabric/Domain**: Resource containers
2. **Endpoint**: Communication channel
3. **CQ**: Completion notification
4. **AV**: Address mapping (for RDM)
5. **MR**: Memory registration

**Operation Types:**
1. **Message**: Send/recv (two-sided)
2. **Tagged**: Send/recv with matching (used by NCCL)
3. **RMA**: Read/write (one-sided)
4. **Atomic**: Atomic operations

**Key Features:**
- **Provider abstraction**: Same API for EFA, verbs, etc.
- **Capability negotiation**: Query what's supported
- **Async operations**: All I/O is non-blocking
- **Polling model**: Manual progress for EFA
- **MR caching**: Critical for performance

**Performance Best Practices:**
- Cache memory registrations
- Batch operations before polling
- Use tagged messaging for matching
- Pre-post receives for RDM
- Use inject for tiny messages
- Poll frequently for manual progress

**Next**: Deep dive into EFA provider specifics and optimizations.
