# rdma-core and libibverbs

## Overview

**rdma-core** is the userspace component of the Linux RDMA stack, providing libraries and infrastructure for RDMA/InfiniBand communication. For EFA, rdma-core bridges the gap between libfabric and the kernel driver.

**GitHub**: [rdma-core](https://github.com/linux-rdma/rdma-core/tree/master) (this document reflects rdma-core **65.0-dev** on `master`; latest tagged release **v64.0**)

## IMPORTANT: rdma-core is no longer on the EFA data path by default

> **This is a major change as of libfabric 2.3.0, enabled by default for the RDM
> path in libfabric 2.7.0.** Historically the EFA provider called
> `ibv_post_send()` / `ibv_post_recv()` / `ibv_poll_cq()` (or their `ibv_wr_*` /
> `ibv_start_poll` extended-verbs equivalents) in rdma-core for every network
> operation. That is still supported, but it is now the **fallback path**.
>
> By default, the EFA provider uses **EFA Data Path Direct**: it writes send/recv
> Work Queue Entries (WQEs) straight into the memory-mapped hardware queues,
> rings the doorbell via its own MMIO helpers, and reads Completion Queue Entries
> (CQEs) directly out of the mapped CQ buffer — **without calling into libibverbs
> on the data path at all.** rdma-core is still used for the **control path**
> (device open, QP/CQ creation, MR registration, AH creation, and the `mmap()`
> of the queue buffers), and for the data path only when Data Path Direct is
> disabled or unavailable.
>
> The environment variable is `FI_EFA_USE_DATA_PATH_DIRECT`, default **true**
> ([prov/efa/src/efa_env.c line 41](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_env.c): `.use_data_path_direct = true`).
> Set it to `0` to force the classic libibverbs data path.
>
> Full details are in the [Data Path Direct](#efa-data-path-direct-default) section
> below and in [efa-hardware-architecture.md](efa-hardware-architecture.md). The
> libibverbs data-path description is retained in full below because it is still
> the exact fallback.

## Control Path Component (always via rdma-core)

For EFA, rdma-core is **always** used to set up communication resources, regardless
of the data-path mode:

- **Device open / query** — `efa_alloc_context()`, `efadv_query_device()`
- **QP creation** — `efadv_create_qp_ex()` / `ibv_create_qp_ex()`; kernel allocates
  and `mmap()`s the SQ/RQ buffers and doorbell pages into userspace
- **CQ creation** — kernel allocates and `mmap()`s the CQ buffer
- **MR registration** — `ibv_reg_mr()` / `ibv_reg_dmabuf_mr()` (pins pages, programs
  the device, returns lkey/rkey)
- **AH creation** — `ibv_create_ah()` (maps a destination address to a hardware AH index)

When Data Path Direct is enabled, the provider retrieves the mapped buffer pointers,
doorbell register pointers, entry sizes and queue depths from rdma-core once at QP/CQ
creation time (via `efadv_query_qp_wqs()` and `efadv_query_cq()`), then drives them
itself for every subsequent operation.

**Data-path performance (per call, either path):**
- Post send: ~100-200 ns (memory writes + doorbell)
- Post recv: ~100-200 ns (memory writes)
- CQ poll: ~50-150 ns (memory reads)

Both paths are **zero-syscall** on the data path — they only touch memory-mapped
hardware queues in userspace. Data Path Direct removes the libibverbs function-call
and dispatch overhead and writes a stack-built WQE directly into write-combining
device memory.

## Architecture Position

```
┌─────────────────────────────────────┐
│   Libfabric API (fi_*)              │
├─────────────────────────────────────┤
│   Libfabric EFA Provider            │
│   efa_qp_post_send/recv,            │
│   efa_ibv_cq_start_poll, ...        │
│      │                              │
│      ├── data_path_direct_enabled?  │
│      │        │                     │
│      │  YES   │   NO                │
│      ▼        ▼                     │
│  ┌────────┐  ┌──────────────────┐  │
│  │ DATA   │  │ rdma-core        │  │ ← dispatch in
│  │ PATH   │  │ (libibverbs)     │  │   efa_data_path_ops.h
│  │ DIRECT │  │ ibv_wr_*,        │  │
│  │ (mmio) │  │ ibv_start_poll   │  │
│  └───┬────┘  └────────┬─────────┘  │
├──────┼────────────────┼────────────┤
│      │  rdma-core (libibverbs)     │ ← control path ALWAYS
│      │  device/QP/CQ/MR/AH setup   │   here; data path here
│      │  + mmap of queue buffers    │   only when direct=off
├──────┼────────────────┼────────────┤
│   EFA Kernel Driver                 │
│   - /dev/infiniband/uverbsN         │
│   - ioctl() interface (control)     │
├─────────────────────────────────────┤
│   EFA Hardware                      │
└─────────────────────────────────────┘
```

Both data-path branches ultimately write the same hardware WQE format into the
same mmap'd SQ/RQ buffers and ring the same doorbell registers; the difference is
whether libibverbs (`ibv_wr_*` / `ibv_start_poll`) does the writing or whether the
EFA provider's own inline code does it. The branch is selected per operation in
[prov/efa/src/efa_data_path_ops.h](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_data_path_ops.h) on the
`qp->data_path_direct_enabled` / `ibv_cq->data_path_direct_enabled` flags.

## Key Components

### 1. libibverbs - Core Verbs Library

**Purpose**: Provides hardware-agnostic RDMA API (verbs)

**Header**: [libibverbs/verbs.h](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h)

**Core Data Structures**:

```c
// Device context
struct ibv_context {  // ([libibverbs/verbs.h:2069-2077](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h))
    struct ibv_device      *device;
    struct ibv_context_ops  ops;
    int                     cmd_fd;      // /dev/infiniband/uverbsN
    int                     async_fd;    // Async events
    // ...
};

// Protection Domain (memory isolation)
struct ibv_pd {  // ([libibverbs/verbs.h:639-642](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h))
    struct ibv_context  *context;
    uint32_t             handle;
};

// Memory Region (registered memory)
struct ibv_mr {  // ([libibverbs/verbs.h:675-683](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h))
    struct ibv_context  *context;
    struct ibv_pd       *pd;
    void                *addr;           // Buffer address
    size_t               length;         // Buffer size
    uint32_t             handle;
    uint32_t             lkey;           // Local key
    uint32_t             rkey;           // Remote key
};

// Queue Pair (communication endpoint)
struct ibv_qp {  // ([libibverbs/verbs.h:1315-1325](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h))
    struct ibv_context  *context;
    struct ibv_pd       *pd;
    struct ibv_cq       *send_cq;        // Send completion queue
    struct ibv_cq       *recv_cq;        // Recv completion queue
    uint32_t             qp_num;
};

// Completion Queue (operation completion)
struct ibv_cq {  // ([libibverbs/verbs.h:1540-1550](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h))
    struct ibv_context  *context;
    int                  cqe;            // CQ depth
    uint32_t             handle;
};
```

### 2. EFA Provider (rdma-core)

**GitHub**: [providers/efa/](https://github.com/linux-rdma/rdma-core/tree/master/providers/efa)

**Key Files**:
- `verbs.c` - EFA-specific verb implementations (75KB, primary implementation)
- `efa.c` - Device initialization
- `efa.h`, `efa-abi.h` - EFA data structures
- `efadv.h` - EFA Direct Verbs (device-specific extensions)

**Critical Functions**:

```c
// From providers/efa/verbs.c

// Device open
struct ibv_context *efa_alloc_context(struct ibv_device *ibdev, int cmd_fd)  // ([providers/efa/efa.c:59](https://github.com/linux-rdma/rdma-core/blob/master/providers/efa/efa.c))
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
                          uint64_t access_flags)  // ([providers/efa/verbs.c:338](https://github.com/linux-rdma/rdma-core/blob/master/providers/efa/verbs.c))
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
                             struct ibv_qp_init_attr *attr)  // ([providers/efa/verbs.c:1866](https://github.com/linux-rdma/rdma-core/blob/master/providers/efa/verbs.c))
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
int efa_poll_cq(struct ibv_cq *ibcq, int nwc, struct ibv_wc *wc)  // ([providers/efa/verbs.c:864](https://github.com/linux-rdma/rdma-core/blob/master/providers/efa/verbs.c))
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

**`ibv_get_device_list()` / `ibv_open_device()` / `ibv_query_device()` / `ibv_close_device()`** ([verbs.h:2291](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h), [verbs.h:2382](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h)):

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

**`ibv_alloc_pd()` / `ibv_dealloc_pd()`** ([verbs.h:2539](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h)):

```c
// Allocate protection domain (memory isolation unit)
struct ibv_pd *ibv_alloc_pd(struct ibv_context *context);

// Deallocate PD
int ibv_dealloc_pd(struct ibv_pd *pd);
```

### Memory Registration

**`ibv_reg_mr()` / `ibv_dereg_mr()`** ([verbs.h:2644](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h), [verbs.h:2715](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h)):

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

**`ibv_create_cq()` / `ibv_create_qp()` / `ibv_modify_qp()` / `ibv_post_send()` / `ibv_post_recv()` / `ibv_poll_cq()`** ([verbs.h:2912](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h), [verbs.h:3125](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h), [verbs.h:3458](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h), [verbs.h:3467](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h), [verbs.h:2990](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h)):

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

Every NCCL message goes through these layers. The last steps depend on whether
Data Path Direct is enabled (the default) or the classic libibverbs path is used:

```
NCCL AllReduce
    ↓
OFI Plugin: nccl_net_ofi_isend()
    ↓
Libfabric: fi_send()
    ↓
EFA Provider: efa_qp_post_send()   [efa_data_path_ops.h]
    │
    ├── data_path_direct_enabled == true  (DEFAULT)
    │      ↓
    │   efa_data_path_direct_post_send()  [efa_data_path_direct_entry.h]
    │      ↓  build efa_io_tx_wqe(_128) on the stack
    │      ↓  mmio_memcpy_x64() → write-combining SQ buffer
    │      ↓  mmio_flush_writes(); mmio_write32(sq->db, pc)  ← doorbell
    │      → NO libibverbs call
    │
    └── data_path_direct_enabled == false  (FALLBACK)
           ↓
        efa_ibv_post_send()  → ibv_wr_start/ibv_wr_send/ibv_wr_set_sge_list/
                               ibv_wr_set_ud_addr/ibv_wr_complete  [rdma-core]
           ↓
        rdma-core efa_post_send(): write WQE to mmap'd SQ, MMIO doorbell
    ↓
EFA Hardware: DMA from GPU memory, transmit packet
```

**Key Point:** From `fi_send()` to hardware notification takes only ~200-300 ns
total on either path. Data Path Direct removes the extended-verbs call chain
(`ibv_wr_start` … `ibv_wr_complete`) and instead builds the WQE as a stack
variable and copies it into write-combining device memory with `mmio_memcpy_x64`,
then rings the doorbell. Both paths achieve this through:
1. **Zero syscalls** — all queue access via mmap
2. **Direct memory writes** — WQEs written to userspace buffers
3. **Single doorbell** — one MMIO write to notify hardware (batched across a
   `FI_MORE` chain)

## EFA Data Path Direct (default)

EFA Data Path Direct was introduced in **libfabric 2.3.0** (first commit
"prov/efa: Bypass rdma-core in data path"), later extended to cover the blocking
CQ read path and enabled for `efa-rdm`, and a 128-byte wide WQE format was
onboarded. It is guarded by the build macro `HAVE_EFA_DATA_PATH_DIRECT` and the
runtime env var `FI_EFA_USE_DATA_PATH_DIRECT` (default **true**). It requires
rdma-core support for `efadv_query_qp_wqs()` and `efadv_query_cq()` to hand back
the mapped queue buffers.

**Source files** (all under [prov/efa/src/](https://github.com/ofiwg/libfabric/tree/main/prov/efa/src)):
`efa_data_path_direct.c` / `.h` (init/finalize, control-path glue),
`efa_data_path_direct_entry.h` (post_send/recv/read/write and CQ poll entry points),
`efa_data_path_direct_internal.h` (WQE building, doorbell, wrid pool, CQE decode),
`efa_data_path_direct_structs.h` (SQ/RQ/CQ/WQ mirror structs),
`efa_data_path_ops.h` (per-op dispatch to direct vs libibverbs),
`efa_mmio.h`, `efa_io_defs.h`, `efa_io_regs_defs.h`.

### Initialization (control path, once per QP/CQ)

`efa_data_path_direct_qp_initialize()` in
[efa_data_path_direct.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_data_path_direct.c)
(called from `efa_base_ep_create_qp`) queries the hardware queue attributes from
rdma-core with `efadv_query_qp_wqs(ibv_qp, &sq_attr, &rq_attr, ...)`, then records:

- `sq.desc` / `rq.buf` = the mmap'd SQ/RQ buffer (`*_attr.buffer`)
- `sq.wq.db` / `rq.wq.db` = the mmap'd doorbell register (`*_attr.doorbell`)
- `wq.wqe_size` = `*_attr.entry_size` (**64 or 128 bytes**)
- `wq.max_batch` = `*_attr.max_batch`
- number of entries = `*_attr.num_entries` (power of 2)

The SQ work queue is initialized with 64-bit request IDs when the hardware
advertises `EFADV_WQ_CAPS_64_BIT_REQ_ID` and `FI_EFA_USE_SQ_REQ_ID_64_BIT` is on
(`sq_req_id_64_bit`). `efa_data_path_direct_cq_initialize()` similarly queries
`efadv_query_cq()` to map the CQ buffer, entry size, and (where available) the CQ
doorbell. The provider then sets `data_path_direct_enabled = true` on the QP/CQ.
**No hardware buffers are unmapped by the direct finalize path** — those are owned
and torn down by rdma-core when the IBV QP/CQ is destroyed.

### Posting a send WQE directly

`efa_data_path_direct_post_send()` (in `efa_data_path_direct_entry.h`) builds the
WQE as a stack variable and copies it into the device SQ:

```c
struct efa_io_tx_wqe_128 local_wqe = {0};          // 128-byte WQE
struct efa_io_tx_meta_desc *meta_desc = &local_wqe.meta;

efa_post_send_validate(qp);                        // SQ full? (posted-completed==cnt)
if (!sq->num_wqe_pending) mmio_wc_start();         // write-combining barrier
if (sq->num_wqe_pending == sq->wq.max_batch)       // flush a full batch
    efa_data_path_direct_send_wr_ring_db(sq);

efa_data_path_direct_set_ud_addr(meta_desc, ah, qpn, qkey); // dest_qp_num, ah->ahn, qkey
efa_set_sq_comp_wrid(meta_desc, &sq->wq, wr_id);   // request-ID scheme (see below)
efa_set_common_ctrl_flags(meta_desc, sq, EFA_IO_SEND); // meta_desc, op_type, phase, first/last/comp_req
if (flags & FI_REMOTE_CQ_DATA) efa_send_wr_set_imm_data(meta_desc, data);

if (use_inline) efa_data_path_direct_set_inline_data(&local_wqe, iov_count, inline_data_list);
else            efa_data_path_direct_set_sgl(local_wqe.data.sgl, meta_desc, sge_list, iov_count);

efa_data_path_direct_send_wr_post(qp, sq, &local_wqe); // mmio_memcpy_x64 into SQ
efa_sq_advance_post_idx(sq);                       // pc++, phase flip on wrap
sq->num_wqe_pending++;
if (!(flags & FI_MORE)) efa_data_path_direct_send_wr_ring_db(sq); // doorbell
```

The WQE copy into the write-combining SQ buffer uses `mmio_memcpy_x64()` with the
runtime `wqe_size` (64 or 128), so the SQ ring stride matches the negotiated wide
or narrow WQE format (`efa_data_path_direct_send_wr_post`). RDMA read/write use the
analogous `efa_data_path_direct_post_read` / `_post_write`, which additionally fill
`efa_io_remote_mem_addr` (rkey + split 64-bit remote address) and cap the SGE count
at `EFA_IO_TX_DESC_NUM_RDMA_BUFS`. RDMA write supports inline data (copied into
`rdma_req.inline_data`) and the `FI_EFA_WR_HIGH_PPS` processing hint.

### Posting a recv WQE directly

`efa_data_path_direct_post_recv()` writes one or more `struct efa_io_rx_desc`
entries into the RQ buffer (`FIRST`/`LAST` bits on the SGE run, `req_id` from the
wrid pool, split buffer address, lkey), advances the producer counter with phase
tracking, then rings the RQ doorbell via
`efa_data_path_direct_rq_ring_doorbell()` which issues a
`udma_to_device_barrier()` before `mmio_write32(rq->wq.db, pc)`.

### Ringing the doorbell (MMIO)

Doorbell helpers live in `efa_data_path_direct_internal.h` and use the MMIO
primitives from `efa_mmio.h`:

- SQ: `efa_data_path_direct_send_wr_ring_db()` → `mmio_flush_writes()` then
  `mmio_write32(sq->wq.db, sq->wq.pc)`, and resets `num_wqe_pending = 0`.
- RQ: `efa_data_path_direct_rq_ring_doorbell()` → `udma_to_device_barrier()` then
  `mmio_write32(rq->wq.db, pc)`.
- CQ (when `HAVE_EFADV_CQ_ATTR_DB`): `efa_update_cq_doorbell()` packs the consumer
  index, `cmd_sn` and arm bit into the `EFA_IO_REGS_CQ_DB_*` fields and
  `mmio_write32`s the CQ doorbell (used by `req_notify_cq` / end-poll for
  interrupt-driven CQ notification).

### Reading completions directly

The CQ poll entry points (`efa_data_path_direct_start_poll` / `_next_poll` /
`_end_poll`) mirror rdma-core's `ibv_start_poll`/`ibv_next_poll`/`ibv_end_poll`:

1. `efa_data_path_direct_next_device_cqe_get()` reads the CQE at
   `consumed_cnt & qmask`, checks the **phase bit** (`EFA_IO_CDESC_COMMON_PHASE`)
   against the CQ's expected phase; if it matches it issues a
   `udma_from_device_barrier()` (so the rest of the CQE is not read before the
   phase is validated), advances `consumed_cnt`, and flips the phase on wrap.
2. The CQE's `qp_num` indexes `efa_domain->device->qp_table[]` to find the owning QP.
3. The completion is validated for **stale/generation** correctness (see below);
   invalid completions are dropped with an `EFA_INFO` log.
4. `efa_data_path_direct_process_ex_cqe()` fills the `ibv_cq_ex` `wr_id`/`status`
   (via `to_ibv_status()` mapping `efa_io_comp_status` → `ibv_wc_status`). The
   `wc_read_*` accessors (`opcode`, `byte_len`, `imm_data`, `src_qp`, `slid`,
   `sgid`, `wc_flags`, `is_unsolicited`) read fields straight out of the mapped
   CQE (`struct efa_io_rx_cdesc` / `efa_io_rx_cdesc_ex` / `efa_io_tx_cdesc`).
5. `efa_wq_cqe_finalize()` increments `wqe_completed` and (in non-64-bit mode)
   returns the wrid index to the pool under `wq->wqlock`.

### Request-ID scheme and QP generation

The request ID carried in the WQE and returned in the CQE is how the provider maps
a completion back to its work-request context. There are two modes:

- **wrid-index mode (default when HW/opt lacks 64-bit req-id):** a small pool
  (`wrid_idx_pool`) hands out an index; `wq->wrid[idx]` stores the caller's 64-bit
  `wr_id`. The device `req_id` is the index **OR-ed with the QP's shifted
  generation** (`efa_wq_get_dev_req_id()` → `efa_wq_get_next_wrid_idx() | wq->shifted_gen`).
  On completion the generation bits are stripped (`cqe->req_id & ~wq->gen_mask`) to
  recover the index, and the wr_id is looked up.
- **64-bit request-ID mode (`req_id_64_bit`, HW cap `EFADV_WQ_CAPS_64_BIT_REQ_ID`):**
  the full 64-bit `wr_id` is written directly into the descriptor
  (`efa_set_req_id_64()` fills `md->req_id` + `md->req_id_ex.w[0..2]`) and read back
  from the TX CQE (`efa_get_req_id_64()`), bypassing the pool entirely.

**QP generation** guards against stale completions from a destroyed-and-recreated
QP landing in the same `qp_table` slot. `efa_data_path_direct_is_valid_wrid_qp_gen()`
checks `(cqe->req_id & cur_wq->gen_mask) == cur_wq->shifted_gen`; a mismatch means
the CQE belongs to a previous generation of the QP and is dropped. (In 64-bit
req-id mode there are no generation bits in the req-id, so a full 64-bit TX
completion skips the generation check via `efa_cqe_is_64_bit_comp()`.) The QP
generation is embedded in the dp-direct request ID via `shifted_gen`.

### When Data Path Direct is disabled

If `FI_EFA_USE_DATA_PATH_DIRECT=0`, if the build lacks `HAVE_EFA_DATA_PATH_DIRECT`,
or if the device/rdma-core cannot supply the mapped queues (e.g. `efa_device_use_sub_cq()`
or missing `efadv_query_qp_wqs`/`efadv_query_cq`), `data_path_direct_enabled` stays
false and every wrapper in `efa_data_path_ops.h` falls through to the libibverbs
implementation below (`efa_ibv_post_send/read/write`, `ibv_post_recv`,
`ibv_start_poll`/`ibv_next_poll`/`ibv_end_poll` and the `ibv_wc_read_*` accessors).
The wire-level WQE/CQE format is identical either way; only who writes it differs.

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
1. rdma-core provides the **verbs API** and, for EFA, always owns the **control
   path** (device open, QP/CQ creation, MR registration, AH creation, queue mmap)
2. EFA provider in rdma-core implements EFA-specific optimizations
3. **By default rdma-core is not called per operation on the data path** — libfabric
   writes the descriptors itself into mappings rdma-core created and published via
   `efadv_query_qp_wqs()` / `efadv_query_cq()`. This is EFA Data Path Direct
   (`FI_EFA_USE_DATA_PATH_DIRECT=true`, libfabric 2.3.0+, RDM-enabled in 2.7.0);
   the libibverbs `ibv_wr_*` / `ibv_start_poll` data path is the fallback
4. Post/poll operations are **extremely fast** on either path (no syscalls, just
   memory ops + one MMIO doorbell)
5. Memory registration is **expensive** (100-500 μs) - cache it!
6. User-space buffers are **mmapped** from kernel for zero-copy
7. Device capabilities are queried via `ibv_query_device()` / `efadv_query_device()`

**Related Documentation**:
- [efa-driver.md](efa-driver.md) - Kernel driver that rdma-core talks to
- [rdma-memreg.md](rdma-memreg.md) - Memory registration details
- [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md) - GPU memory via dmabuf
