# Libfabric Overview and APIs

## Introduction

Libfabric (also called OFI - OpenFabrics Interfaces) is a framework for exporting fabric communication services to applications. It provides a vendor-agnostic API for high-performance networking.

**Repository**: https://github.com/ofiwg/libfabric

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
fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0, &hints, &info);

// Open fabric
fi_fabric(info->fabric_attr, &fabric, NULL);
```

(`struct fi_info` ([fabric.h:198-232](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fabric.h#L198-L232)))

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

(`struct fid_domain` ([fi_domain.h:94-104](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L94-L104)))

**Responsibilities:**
- Memory registration scope
- Endpoint creation
- Completion queue creation
- Address vector creation

**Domain Attributes:**

**`struct fi_domain_attr`** - Domain attributes ([fabric.h:226-244](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fabric.h#L226-L244)):

```c
struct fi_domain_attr {
  struct fid_domain *domain;
  char *name;
  enum fi_threading threading;     // Thread safety
  enum fi_progress control_progress;
  enum fi_progress data_progress;
  enum fi_resource_mgmt resource_mgmt;
  enum fi_mr_mode mr_mode;        // Memory registration mode
  size_t mr_key_size;
  size_t cq_data_size;
  // ...
};
```

### Endpoint

Communication endpoint for data transfer (`struct fid_ep` ([fabric.h:168-178](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fabric.h#L168-L178))):

```c
struct fid_ep *ep;

fi_endpoint(domain, info, &ep, NULL);

// Bind resources
fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV);
fi_ep_bind(ep, &av->fid, 0);

// Enable
fi_enable(ep);
```

(`fi_endpoint()` ([fi_endpoint.h:182](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_endpoint.h#L182)), `fi_ep_bind()` ([fi_endpoint.h:203](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_endpoint.h#L203)), `fi_enable()` ([fi_endpoint.h:211](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_endpoint.h#L211)))

**Endpoint Types:**
1. **FI_EP_MSG**: Connected, message-oriented (like TCP)
2. **FI_EP_RDM**: Reliable datagram (like UD with reliability)
3. **FI_EP_DGRAM**: Unreliable datagram (like UDP)

**EFA uses FI_EP_RDM** - reliable datagram with provider-level reliability

### Completion Queue (CQ)

Reports completion of asynchronous operations (`struct fid_cq` ([fi_eq.h:138-148](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L138-L148))):

```c
struct fid_cq *cq;
struct fi_cq_attr cq_attr = {
  .size = 1024,
  .format = FI_CQ_FORMAT_DATA,
  .wait_obj = FI_WAIT_NONE,  // Polling mode
};

fi_cq_open(domain, &cq_attr, &cq, NULL);
```

(`fi_cq_open()` ([fi_eq.h:363](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L363)))

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

(`struct fi_cq_data_entry` ([fi_eq.h:215-220](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L215-L220)), `fi_cq_read()` ([fi_eq.h:311](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L311)))

### Address Vector (AV)

Maps addresses for connection-less endpoints (`struct fid_av` ([fi_cm.h:167-177](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_cm.h#L167-L177))):

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

(`fi_av_insert()` ([fi_cm.h:206](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_cm.h#L206)))

**AV Types:**
- **FI_AV_MAP**: Hash table lookup
- **FI_AV_TABLE**: Direct indexing (faster)

**Usage:**
- Required for RDM endpoints (like EFA)
- Maps peer addresses to fi_addr_t handles
- Used in send/recv/RMA operations

### Memory Region (MR)

Registered memory for RDMA (`struct fid_mr` ([fi_domain.h:131-138](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L131-L138))):

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

(`fi_mr_reg()` ([fi_domain.h:413](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L413)), `fi_mr_desc()` ([fi_domain.h:147](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L147)))

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

(`fi_send()` ([fi_msg.h:104](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_msg.h#L104)), `fi_inject()` ([fi_msg.h:133](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_msg.h#L133)))

**Characteristics:**
- **fi_send**: Asynchronous, requires completion
- **fi_inject**: Synchronous, no completion, limited size (< 4 KB)
- **fi_senddata**: Carries immediate data to receiver

#### Receive

```c
ssize_t fi_recv(struct fid_ep *ep,
                void *buf,
                size_t len,
                void *desc,
                fi_addr_t src_addr,
                void *context);
```

(`fi_recv()` ([fi_msg.h:67](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_msg.h#L67)))

**Pre-posting:**
- Must be posted before data arrives (for RDM)
- Matched with incoming sends
- Can use FI_ADDR_UNSPEC for any source

### Tagged Messages

For message matching by tag:

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

(`fi_tsend()` ([fi_tagged.h:121](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_tagged.h#L121)), `fi_trecv()` ([fi_tagged.h:98](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_tagged.h#L98)))

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

(`fi_read()` ([fi_rma.h:82](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_rma.h#L82)))

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

(`fi_write()` ([fi_rma.h:111](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_rma.h#L111)))

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

```c
// Primary capabilities
FI_MSG         // Message queue operations (send/recv)
FI_RMA         // RDMA operations (read/write)
FI_TAGGED      // Tagged messaging
FI_ATOMIC      // Atomic operations
FI_COLLECTIVE  // Collective operations (rare)

// Secondary capabilities
FI_SEND        // Can send messages
FI_RECV        // Can receive messages
FI_READ        // Can initiate RDMA read
FI_WRITE       // Can initiate RDMA write
FI_REMOTE_READ // Can be target of remote read
FI_REMOTE_WRITE// Can be target of remote write

// Features
FI_MULTI_RECV  // Multi-buffer receives
FI_FENCE       // Ordering via fences
FI_INJECT      // Eager sends
```

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
fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0, &hints, &info);
```

**fi_getinfo() selects provider matching hints**

## Threading and Progress

### Threading Models

```c
enum fi_threading {
  FI_THREAD_UNSPEC,      // No thread safety
  FI_THREAD_SAFE,        // Fully thread-safe
  FI_THREAD_FID,         // Safe per-object
  FI_THREAD_DOMAIN,      // Safe per-domain
  FI_THREAD_COMPLETION,  // Safe per-completion
};
```

**EFA provider: FI_THREAD_SAFE**

### Progress Models

```c
enum fi_progress {
  FI_PROGRESS_UNSPEC,
  FI_PROGRESS_AUTO,     // Provider drives progress
  FI_PROGRESS_MANUAL,   // App must call fi_cq_read()
};
```

**Control Progress**: For connection management
**Data Progress**: For data transfer completions

**EFA:**
- Control: FI_PROGRESS_MANUAL
- Data: FI_PROGRESS_MANUAL

**Manual progress requires polling CQ**

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

(`struct fi_cq_err_entry` ([fi_eq.h:233-246](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L233-L246)), `fi_cq_readerr()` ([fi_eq.h:348](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L348)))

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
fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0, &hints, &info);

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
FI_EFA_TX_IOV_LIMIT=1  # Scatter-gather limit
FI_EFA_RX_IOV_LIMIT=1

# Features
FI_EFA_ENABLE_SHM_TRANSFER=0  # Disable intra-node SHM
FI_EFA_USE_DEVICE_RDMA=1      # Enable RDMA

# Memory registration
FI_MR_CACHE_MONITOR=memhooks
FI_MR_CACHE_MAX_SIZE=unlimited
```

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
