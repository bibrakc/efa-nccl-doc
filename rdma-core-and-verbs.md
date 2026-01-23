# rdma-core and libibverbs

## Overview

**rdma-core** is the userspace component of the Linux RDMA stack, providing libraries and infrastructure for RDMA/InfiniBand communication. For EFA, rdma-core bridges the gap between libfabric and the kernel driver.

**GitHub**: [rdma-core](https://github.com/linux-rdma/rdma-core/tree/6e9643e)

## Critical Data Path Component

**rdma-core is in the hot data path for every NCCL message.** The EFA provider in libfabric directly calls `ibv_post_send()`, `ibv_post_recv()`, and `ibv_poll_cq()` for every network operation ([prov/efa/src/efa_data_path_ops.h:78](https://github.com/ofiwg/libfabric/blob/6b9e629/prov/efa/src/efa_data_path_ops.h#L78)):

```c
// From libfabric EFA provider - called for EVERY message
static inline int efa_qp_post_recv(struct efa_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad)
{
    return ibv_post_recv(qp->ibv_qp, wr, bad);  // Direct call to rdma-core
}
```

**Performance Impact:**
- `ibv_post_send()`: ~100-200 ns per call (memory writes + doorbell)
- `ibv_post_recv()`: ~100-200 ns per call (memory writes)
- `ibv_poll_cq()`: ~50-150 ns per call (memory reads)

These operations are **zero-syscall** - they only access memory-mapped hardware queues in userspace, making them extremely fast. This is why understanding rdma-core is critical for NCCL optimization.

## Architecture Position

```
┌─────────────────────────────────────┐
│   Libfabric API (fi_*)              │
├─────────────────────────────────────┤
│   Libfabric EFA Provider            │
│   - Uses verbs API                  │
├─────────────────────────────────────┤
│   rdma-core (libibverbs)            │ ← This Layer
│   - ibv_* API                       │
│   - EFA provider (rdma-core)        │
├─────────────────────────────────────┤
│   EFA Kernel Driver                 │
│   - /dev/infiniband/uverbsN         │
│   - ioctl() interface               │
├─────────────────────────────────────┤
│   EFA Hardware                      │
└─────────────────────────────────────┘
```

## Key Components

### 1. libibverbs - Core Verbs Library

**Purpose**: Provides hardware-agnostic RDMA API (verbs)

**Header**: [libibverbs/verbs.h](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h)

**Core Data Structures**:

```c
// Device context
struct ibv_context {  // ([libibverbs/verbs.h:2069-2077](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L2069-L2077))
    struct ibv_device      *device;
    struct ibv_context_ops  ops;
    int                     cmd_fd;      // /dev/infiniband/uverbsN
    int                     async_fd;    // Async events
    // ...
};

// Protection Domain (memory isolation)
struct ibv_pd {  // ([libibverbs/verbs.h:639-642](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L639-L642))
    struct ibv_context  *context;
    uint32_t             handle;
};

// Memory Region (registered memory)
struct ibv_mr {  // ([libibverbs/verbs.h:675-683](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L675-L683))
    struct ibv_context  *context;
    struct ibv_pd       *pd;
    void                *addr;           // Buffer address
    size_t               length;         // Buffer size
    uint32_t             handle;
    uint32_t             lkey;           // Local key
    uint32_t             rkey;           // Remote key
};

// Queue Pair (communication endpoint)
struct ibv_qp {  // ([libibverbs/verbs.h:1315-1325](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L1315-L1325))
    struct ibv_context  *context;
    struct ibv_pd       *pd;
    struct ibv_cq       *send_cq;        // Send completion queue
    struct ibv_cq       *recv_cq;        // Recv completion queue
    uint32_t             qp_num;
};

// Completion Queue (operation completion)
struct ibv_cq {  // ([libibverbs/verbs.h:1540-1550](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L1540-L1550))
    struct ibv_context  *context;
    int                  cqe;            // CQ depth
    uint32_t             handle;
};
```

### 2. EFA Provider (rdma-core)

**GitHub**: [providers/efa/](https://github.com/linux-rdma/rdma-core/tree/6e9643e/providers/efa)

**Key Files**:
- `verbs.c` - EFA-specific verb implementations (75KB, primary implementation)
- `efa.c` - Device initialization
- `efa.h`, `efa-abi.h` - EFA data structures
- `efadv.h` - EFA Direct Verbs (device-specific extensions)

**Critical Functions**:

```c
// From providers/efa/verbs.c

// Device open
struct ibv_context *efa_alloc_context(struct ibv_device *ibdev, int cmd_fd)  // ([providers/efa/efa.c:59](https://github.com/linux-rdma/rdma-core/blob/6e9643e/providers/efa/efa.c#L59))
{
    struct efa_context *ctx;

    ctx = calloc(1, sizeof(*ctx));
    ctx->ibv_ctx.cmd_fd = cmd_fd;

    // Query device capabilities
    efa_query_device_ex(...);

    // Set up EFA-specific operations
    ctx->ibv_ctx.ops = efa_ctx_ops;

    return &ctx->ibv_ctx;
}

// Memory registration
struct ibv_mr *efa_reg_mr(struct ibv_pd *pd, void *addr, size_t length,
                          uint64_t access_flags)  // ([providers/efa/verbs.c:338](https://github.com/linux-rdma/rdma-core/blob/6e9643e/providers/efa/verbs.c#L338))
{
    struct ibv_reg_mr cmd;
    struct ib_uverbs_reg_mr_resp resp;

    // Call kernel driver via ioctl
    cmd.addr = (uintptr_t)addr;
    cmd.length = length;
    cmd.access_flags = access_flags;

    ibv_cmd_reg_mr(pd, addr, length, access_flags, &mr, &cmd, &resp);

    // Kernel returns lkey/rkey
    mr->lkey = resp.lkey;
    mr->rkey = resp.rkey;

    return mr;
}

// QP creation
struct ibv_qp *efa_create_qp(struct ibv_pd *pd,
                             struct ibv_qp_init_attr *attr)  // ([providers/efa/verbs.c:1866](https://github.com/linux-rdma/rdma-core/blob/6e9643e/providers/efa/verbs.c#L1866))
{
    struct efa_qp *qp;
    struct ibv_create_qp cmd;

    // Allocate SQ/RQ buffers in userspace
    qp->sq_buf = mmap(...);  // Send queue
    qp->rq_buf = mmap(...);  // Receive queue

    // Tell kernel about QP
    ibv_cmd_create_qp(pd, &qp->ibv_qp, attr, &cmd, &resp);

    return &qp->ibv_qp;
}

// Post send
int efa_post_send(struct ibv_qp *ibqp, struct ibv_send_wr *wr,
                  struct ibv_send_wr **bad_wr)  // (inline function in providers/efa/verbs.c)
{
    struct efa_qp *qp = to_efa_qp(ibqp);

    while (wr) {
        // Write work request to send queue
        struct efa_io_tx_wqe *wqe = get_sq_wqe(qp, qp->sq_prod);

        wqe->buf_addr = wr->sg_list[0].addr;
        wqe->buf_len = wr->sg_list[0].length;
        wqe->lkey = wr->sg_list[0].lkey;
        // ...

        qp->sq_prod++;

        wr = wr->next;
    }

    // Ring doorbell (MMIO write to notify hardware)
    mmio_write(qp->sq_db, qp->sq_prod);

    return 0;
}

// Poll completions
int efa_poll_cq(struct ibv_cq *ibcq, int nwc, struct ibv_wc *wc)  // ([providers/efa/verbs.c:864](https://github.com/linux-rdma/rdma-core/blob/6e9643e/providers/efa/verbs.c#L864))
{
    struct efa_cq *cq = to_efa_cq(ibcq);
    int ne = 0;

    while (ne < nwc) {
        struct efa_io_cdesc *cqe = get_cqe(cq, cq->cq_cons);

        if (!cqe_is_valid(cqe))
            break;  // No more completions

        // Parse completion
        wc[ne].wr_id = cqe->req_id;
        wc[ne].status = cqe->status;
        wc[ne].byte_len = cqe->length;

        cq->cq_cons++;
        ne++;
    }

    // Update consumer index
    mmio_write(cq->cq_db, cq->cq_cons);

    return ne;
}
```

## verbs API Overview

### Device Management

**`ibv_get_device_list()` / `ibv_open_device()` / `ibv_query_device()` / `ibv_close_device()`** ([verbs.h:2291](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L2291), [verbs.h:2382](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L2382)):

```c
// List RDMA devices
struct ibv_device **ibv_get_device_list(int *num_devices);

// Open device
struct ibv_context *ibv_open_device(struct ibv_device *device);

// Query device capabilities
int ibv_query_device(struct ibv_context *context,
                     struct ibv_device_attr *device_attr);

// Close device
int ibv_close_device(struct ibv_context *context);
```

### Protection Domain

**`ibv_alloc_pd()` / `ibv_dealloc_pd()`** ([verbs.h:2539](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L2539)):

```c
// Allocate protection domain (memory isolation unit)
struct ibv_pd *ibv_alloc_pd(struct ibv_context *context);

// Deallocate PD
int ibv_dealloc_pd(struct ibv_pd *pd);
```

### Memory Registration

**`ibv_reg_mr()` / `ibv_dereg_mr()`** ([verbs.h:2644](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L2644), [verbs.h:2715](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L2715)):

```c
// Register memory region
struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length,
                          int access_flags);

// Access flags:
// IBV_ACCESS_LOCAL_WRITE  - Local writes allowed
// IBV_ACCESS_REMOTE_READ  - Remote RDMA read allowed
// IBV_ACCESS_REMOTE_WRITE - Remote RDMA write allowed

// Deregister memory
int ibv_dereg_mr(struct ibv_mr *mr);
```

### Queue Pair Operations

**`ibv_create_cq()` / `ibv_create_qp()` / `ibv_modify_qp()` / `ibv_post_send()` / `ibv_post_recv()` / `ibv_poll_cq()`** ([verbs.h:2912](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L2912), [verbs.h:3125](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L3125), [verbs.h:3458](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L3458), [verbs.h:3467](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L3467), [verbs.h:2990](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L2990)):

```c
// Create completion queue
struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe,
                             void *cq_context, ...);

// Create queue pair
struct ibv_qp *ibv_create_qp(struct ibv_pd *pd,
                             struct ibv_qp_init_attr *qp_init_attr);

// Modify QP state (RESET → INIT → RTR → RTS)
int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                  int attr_mask);

// Post send work request
int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                  struct ibv_send_wr **bad_wr);

// Post receive work request
int ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
                  struct ibv_recv_wr **bad_wr);

// Poll for completions
int ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc);
```

## libfabric → verbs Mapping

Libfabric's EFA provider uses verbs under the hood:

| Libfabric API | verbs API | Purpose |
|---------------|-----------|---------|
| `fi_domain()` | `ibv_alloc_pd()` | Create protection domain |
| `fi_endpoint()` | `ibv_create_qp()` | Create queue pair |
| `fi_mr_reg()` | `ibv_reg_mr()` | Register memory |
| `fi_send()` | `ibv_post_send()` | Post send operation |
| `fi_recv()` | `ibv_post_recv()` | Post receive operation |
| `fi_cq_read()` | `ibv_poll_cq()` | Poll completions |

**Example Mapping** (from libfabric EFA provider):

```c
// libfabric: fi_mr_reg()
static int efa_mr_reg(struct fid *fid, const void *buf, size_t len,
                      uint64_t access, uint64_t offset, uint64_t requested_key,
                      uint64_t flags, struct fid_mr **mr_fid, void *context)
{
    struct efa_domain *domain = container_of(fid, struct efa_domain, ...);
    struct efa_mr *mr;

    // Call verbs underneath
    mr->ibv_mr = ibv_reg_mr(domain->ibv_pd, (void *)buf, len,
                           IBV_ACCESS_LOCAL_WRITE |
                           IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE);

    mr->mr_fid.key = mr->ibv_mr->lkey;  // Expose verbs lkey as fi key

    *mr_fid = &mr->mr_fid;
    return 0;
}
```

### Complete Data Path Flow

Every NCCL message goes through these layers:

```
NCCL AllReduce
    ↓
OFI Plugin: nccl_net_ofi_isend()
    ↓
Libfabric: fi_send()
    ↓
EFA Provider: efa_qp_post_send()
    ↓
rdma-core: ibv_post_send() ← 100-200 ns, NO syscall
    ↓
EFA Provider (rdma-core): efa_post_send()
    ↓
Userspace: Write WQE to mmap'd send queue
    ↓
Userspace: MMIO write to doorbell register
    ↓
EFA Hardware: DMA from GPU memory, transmit packet
```

**Key Point:** From `fi_send()` to hardware notification takes only ~200-300 ns total, with rdma-core's `ibv_post_send()` consuming ~100-200 ns of that. This is achieved through:
1. **Zero syscalls** - all queue access via mmap
2. **Direct memory writes** - WQEs written to userspace buffers
3. **Single doorbell** - one MMIO write to notify hardware

## EFA-Specific Extensions (efadv.h)

EFA provides device-specific extensions via **EFA Direct Verbs**:

```c
// From rdma-core/providers/efa/efadv.h

// Query EFA-specific attributes
int efadv_query_device(struct ibv_context *context,
                       struct efadv_device_attr *attr);

struct efadv_device_attr {
    uint32_t max_sq_wr;              // Max send queue depth
    uint32_t max_rq_wr;              // Max recv queue depth
    uint16_t max_sq_sge;             // Max scatter-gather entries (send)
    uint16_t max_rq_sge;             // Max scatter-gather entries (recv)
    uint32_t max_rdma_size;          // Max RDMA transfer size
    uint16_t device_caps;            // EFA capabilities
    // ...
};

// EFA-specific QP creation
struct ibv_qp *efadv_create_qp_ex(struct ibv_context *context,
                                  struct ibv_qp_init_attr_ex *attr,
                                  struct efadv_qp_init_attr *efa_attr);
```

## Kernel Interface (uverbs)

rdma-core communicates with the kernel via `/dev/infiniband/uverbsN`:

```c
// From libibverbs/cmd.c

int ibv_cmd_reg_mr(struct ibv_pd *pd, void *addr, size_t length,
                   uint64_t hca_va, int access_flags, ...)
{
    struct ib_uverbs_reg_mr cmd = {
        .command = IB_USER_VERBS_CMD_REG_MR,
        .pd_handle = pd->handle,
        .start = (uintptr_t)addr,
        .length = length,
        .hca_va = hca_va,
        .access_flags = access_flags,
    };

    struct ib_uverbs_reg_mr_resp resp;

    // ioctl to kernel driver
    write(pd->context->cmd_fd, &cmd, sizeof(cmd));
    read(pd->context->cmd_fd, &resp, sizeof(resp));

    // Kernel returns MR handle and keys
    mr->handle = resp.mr_handle;
    mr->lkey = resp.lkey;
    mr->rkey = resp.rkey;

    return 0;
}
```

**Device Files**:
```bash
/dev/infiniband/uverbs0    # Command interface
/dev/infiniband/uverbs0a   # Async events
/dev/infiniband/rdma_cm    # Connection manager
```

## Memory Management in rdma-core

### User-Space DMA Buffers

```c
// Queue buffers are mmapped from kernel
void *sq_buf = mmap(NULL, sq_size, PROT_READ | PROT_WRITE,
                    MAP_SHARED, ctx->cmd_fd, sq_offset);

void *rq_buf = mmap(NULL, rq_size, PROT_READ | PROT_WRITE,
                    MAP_SHARED, ctx->cmd_fd, rq_offset);
```

**Why mmap?**
1. Zero-copy: Kernel and userspace share same physical pages
2. Direct hardware access: NIC can DMA directly to/from these buffers
3. No syscalls: Post send/recv by writing to mmapped memory + doorbell

### Memory Registration Flow

```
User calls ibv_reg_mr(pd, buf, size)
    ↓
libibverbs → write() to /dev/infiniband/uverbsN
    ↓
Kernel EFA driver:
    1. Pin pages (get_user_pages)
    2. Create IOMMU mapping
    3. Program EFA hardware with physical addresses
    4. Generate lkey/rkey
    ↓
Kernel returns lkey/rkey to userspace
    ↓
User gets struct ibv_mr with keys
```

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| `ibv_open_device()` | 1-5 ms | One-time setup |
| `ibv_reg_mr()` | 100-500 μs | Pins pages, creates IOMMU mapping |
| `ibv_dereg_mr()` | 50-200 μs | Unpins pages |
| `ibv_post_send()` | 100-200 ns | Just memory writes + doorbell |
| `ibv_poll_cq()` | 50-150 ns | Just memory reads |
| `ibv_create_qp()` | 500 μs - 2 ms | Allocates kernel resources |

**Key Insight**: Post/poll operations are **extremely fast** because they're just memory operations (no syscalls).

## Integration with OFI Plugin

The OFI NCCL plugin uses libfabric, which uses rdma-core for EFA:

```
NCCL Plugin Call Stack:
nccl_net_ofi_regmr()                   [OFI plugin]
  ↓
fi_mr_reg()                             [libfabric API]
  ↓
efa_mr_regattr()                        [libfabric EFA provider]
  ↓
ibv_reg_mr_iova2()                      [rdma-core libibverbs]
  ↓
ibv_cmd_reg_mr()                        [rdma-core command layer]
  ↓
write(cmd_fd, &cmd, sizeof(cmd))        [ioctl to kernel]
  ↓
efa_ib_reg_mr()                         [EFA kernel driver]
  ↓
EFA hardware                            [DMA engine setup]
```

## Debugging with rdma-core

### List Devices

```bash
# Show RDMA devices
ibv_devices
# Output:
#   device                 node GUID
#   ------              ----------------
#   efa_0               0000:00:06.0

# Device info
ibv_devinfo -d efa_0
```

### Query Device Attributes

```c
struct ibv_device_attr attr;
ibv_query_device(ctx, &attr);

printf("Max QPs: %d\n", attr.max_qp);
printf("Max CQs: %d\n", attr.max_cq);
printf("Max MR size: %llu\n", attr.max_mr_size);
```

### Check Memory Registration

```bash
# Show registered memory regions
cat /proc/$(pidof app)/maps | grep efa

# Check pinned memory
cat /proc/$(pidof app)/status | grep VmPin
```

### Trace verbs Calls

```bash
# Enable rdma-core debug
export RDMAV_DEBUG_LEVEL=1

# Run application
./app

# Output shows all verbs calls:
# ibv_reg_mr: addr=0x7f..., len=4096, access=7
# ibv_post_send: qp_num=42, wr_id=123
```

## Common Issues

### Problem: `ibv_reg_mr()` fails with ENOMEM

**Causes**:
1. Exceeded locked memory limit (`ulimit -l`)
2. Ran out of IOMMU entries
3. Fragmented physical memory

**Solutions**:
```bash
# Increase locked memory limit
ulimit -l unlimited

# Check IOMMU
dmesg | grep -i iommu

# Use huge pages for better contiguity
echo 1024 > /proc/sys/vm/nr_hugepages
```

### Problem: Slow `ibv_poll_cq()`

**Cause**: Cache misses on CQ buffer

**Solution**: Ensure CQ buffer is hot in cache
```c
// Prefetch CQ entry
__builtin_prefetch(cq_entry);
```

### Problem: Device not found

**Cause**: EFA driver not loaded or wrong permissions

**Solutions**:
```bash
# Load EFA driver
sudo modprobe efa

# Check device files
ls -l /dev/infiniband/

# Fix permissions
sudo chmod 666 /dev/infiniband/uverbs*
```

## Key Source Files

### rdma-core
- `libibverbs/verbs.h` - Main API header
- `libibverbs/cmd.c` - Kernel command interface
- `libibverbs/device.c` - Device enumeration
- `libibverbs/memory.c` - Memory registration
- `providers/efa/verbs.c` - EFA implementation (75KB)
- `providers/efa/efa.h` - EFA structures

### Kernel Headers
- `kernel-headers/rdma/ib_user_verbs.h` - Userspace ↔ kernel ABI
- `kernel-headers/rdma/ib_verbs.h` - Core kernel verbs

## Summary

| Component | Purpose | Key API |
|-----------|---------|---------|
| **libibverbs** | Hardware-agnostic RDMA API | `ibv_*` functions |
| **EFA provider** | EFA-specific implementation | `efa_*` internal functions |
| **efadv** | EFA direct verbs extensions | `efadv_*` functions |
| **uverbs** | Kernel communication | ioctl via `/dev/infiniband/uverbsN` |

**Key Takeaways**:
1. rdma-core provides the **verbs API** that libfabric uses
2. EFA provider in rdma-core implements EFA-specific optimizations
3. Post/poll operations are **extremely fast** (no syscalls, just memory ops)
4. Memory registration is **expensive** (100-500 μs) - cache it!
5. User-space buffers are **mmapped** from kernel for zero-copy
6. Device capabilities are queried via `ibv_query_device()`

**Related Documentation**:
- [efa-driver.md](efa-driver.md) - Kernel driver that rdma-core talks to
- [rdma-memreg.md](rdma-memreg.md) - Memory registration details
- [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md) - GPU memory via dmabuf
