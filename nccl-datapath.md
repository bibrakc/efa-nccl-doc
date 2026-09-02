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
│  DEFAULT: data-path-direct          │   │  reads CQEs    │
│  writes WQE → mapped SQ, MMIO DB    │   │  from mapped CQ│
└────┬─────────────────────────────────┘   └────▲───────────┘
     │  (default: dp-direct bypasses rdma-core on the data path)
     │  (fallback: FI_EFA_USE_DATA_PATH_DIRECT=0 → rdma-core below)
     ├───────────────────────────┐               │
     ▼ (fallback only)           │ (default)      │
┌──────────────────────────────┐ │       ┌────────────────┐
│     rdma-core (libibverbs)   │ │       │  rdma-core     │
│     - ibv_post_send/recv     │ │       │  (fallback)    │
└────┬─────────────────────────┘ │       └────▲───────────┘
     │                           │            │
     ▼                           ▼            │
┌──────────────────────────────────────┐   ┌────────────────┐
│      EFA Driver (uverbs) / EFA NIC  │   │   EFA Driver   │
└────┬─────────────────────────────────┘   └────▲───────────┘
     │                                           │
     └────────────────► Network ────────────────┘
```

> **Default vs fallback.** As of current libfabric, the EFA provider sets
> `FI_EFA_USE_DATA_PATH_DIRECT=true` **by default** (verified in
> [libfabric/prov/efa/src/efa_env.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_env.c),
> `.use_data_path_direct = true`). In this mode the EFA provider writes work-queue
> entries (WQEs) directly into the memory-mapped send queue and reads completion
> descriptors (CQEs) directly from the memory-mapped CQ buffer, ringing the
> hardware doorbell itself via MMIO — **rdma-core/libibverbs is not on the data
> path at all**. The rdma-core `ibv_post_send`/`ibv_poll_cq` path documented below
> is now the **fallback**, used only when dp-direct is disabled
> (`FI_EFA_USE_DATA_PATH_DIRECT=0`) or unavailable (`HAVE_EFA_DATA_PATH_DIRECT`
> not compiled in). Both paths perform the same fundamental operations (WQE write
> + MMIO doorbell, CQE poll); dp-direct simply removes the libibverbs call layer.

## Detailed Data Flow

### Step-by-Step for Inter-Node Communication

#### 1. Application Issues Collective

**NCCL API** ([nccl/src/include/nccl.h](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in)):

```c
// User code
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat,
              ncclSum, comm, stream);
```

**Implementation** ([nccl/src/collectives.cc](https://github.com/NVIDIA/nccl/blob/master/src/collectives.cc)):
- Validates communicator and parameters
- Enqueues operation to CUDA stream
- Returns immediately (non-blocking)

- **Action**: Enqueue to CUDA stream
- **Timing**: ~1-2 μs (software overhead)

#### 2. NCCL Planning

**Scheduler** ([nccl/src/enqueue/enqueue.cc](https://github.com/NVIDIA/nccl/blob/master/src/enqueue/enqueue.cc)):

```c
// NCCL internal planning (simplified)
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  // Determine algorithm based on size and topology
  NCCLCHECK(selectAlgorithm(info, &algo));

  // Select protocol based on message size
  NCCLCHECK(selectProtocol(info, &proto));

  // Break into chunks and assign to channels
  NCCLCHECK(computeCollChunking(info));

  return ncclSuccess;
}
```

**Tasks**:
- Determine algorithm (Ring/Tree) based on message size and topology
- Select protocol (Simple/LL/LL128) from tuner or heuristics
- Break data into chunks for parallel processing
- Assign chunks to channels (typically 2-16 channels)
- Build operation graph for GPU kernels

**Key Functions**:
- `ncclEnqueueCheck()` - Main scheduling entry point
- `selectAlgorithm()` - Algorithm selection logic
- `selectProtocol()` - Protocol selection (may call tuner)

- **Action**: Create work items for each channel
- **Timing**: < 1 μs (usually cached)

#### 3. CUDA Kernel Launch

**NCCL Device Code** ([nccl/src/device/](https://github.com/NVIDIA/nccl/tree/master/src/device)):

```cuda
// Simplified NCCL kernel launch (from src/enqueue.cc)
for (int c = 0; c < comm->nChannels; c++) {
  if (channelHasWork[c]) {
    // Launch kernel with work descriptor
    ncclKernel<<<blocks, threads, 0, stream>>>(
      comm->devComm, channel[c].workFifo
    );
  }
}
```

**Kernel Implementation** ([nccl/src/device/common_kernel.h](https://github.com/NVIDIA/nccl/blob/master/src/device/common_kernel.h)):

**Kernel Responsibilities:**
- Read data from source GPU memory
- Perform local reduction if needed (for AllReduce)
- Write to network proxy buffers or directly to NIC
- Signal proxy thread via memory fence
- Coordinate with other GPU threads using primitives

**Key Device Functions**:
- `ncclKernel()` - Main device kernel entry point
- `runRing()` / `runTree()` - Algorithm-specific device code
- `prims.send()` / `prims.recv()` - Low-level data movement primitives

- **Timing**: ~5-10 μs (kernel launch latency)

#### 4. Proxy Thread Activation

**Proxy Service** ([nccl/src/proxy.cc](https://github.com/NVIDIA/nccl/blob/master/src/proxy.cc)):

```c
// Simplified proxy thread main loop
void* ncclProxyService(void* _args) {
  struct ncclProxyArgs* args = _args;

  while (!args->stop) {
    // Poll for work from GPU via shared memory
    ncclProxyOp* op = ncclProxyGetNextOp(args);

    if (op) {
      // Process operation through network plugin
      if (op->type == ncclProxyOpSend) {
        ncclNetSend(op);
      } else if (op->type == ncclProxyOpRecv) {
        ncclNetRecv(op);
      }

      // Poll for completion
      ncclNetTest(op->request, &done);

      // Signal GPU when complete
      ncclProxySignalGpu(op);
    }
  }
}
```

**Key Functions**:
- `ncclProxyService()` - Main proxy thread loop ([src/proxy.cc](https://github.com/NVIDIA/nccl/blob/master/src/proxy.cc))
- `ncclProxyGetNextOp()` - Dequeue work from GPU shared memory
- `ncclNetSend()` / `ncclNetRecv()` - Call into ncclNet plugin
- `ncclNetTest()` - Poll for network completion

**Why Proxy Threads:**
- Network operations are CPU-driven (EFA requires CPU polling)
- GPU focuses on compute and local data movement
- Enables async progress independent of GPU kernels
- CPU can manage complex protocols and error handling

- **Timing**: Wakeup ~1-2 μs

#### 5. OFI Plugin Send

**`ncclNet_t`** interface ([nccl/src/include/net.h](https://github.com/NVIDIA/nccl/blob/master/src/include/net.h)):

```c
// Proxy calls into OFI plugin via ncclNet_t interface
ncclNet->isend(sendComm, data, size, tag, mhandle, &request);
```

**OFI Plugin Implementation** ([aws-ofi-nccl/src/nccl_ofi_api.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp)):

```c
// aws-ofi-nccl send implementation
ncclResult_t nccl_net_ofi_isend(void* sendComm, void* data, int size,
                                int tag, void* mhandle, void** request) {
  nccl_net_ofi_send_comm_t* s_comm = (nccl_net_ofi_send_comm_t*)sendComm;
  nccl_net_ofi_mr_handle_t* mr_handle = (nccl_net_ofi_mr_handle_t*)mhandle;

  // Get libfabric memory descriptor
  void* desc = fi_mr_desc(mr_handle->mr);

  // Post send using libfabric
  ret = fi_send(s_comm->local_ep, data, size, desc,
                s_comm->remote_addr, tag);

  return ncclSuccess;
}
```

**Plugin Actions**:
- Translate NCCL request to libfabric operation
- Get memory registration handle from MR cache
- Extract libfabric memory descriptor (`fi_mr_desc()`)
- Post `fi_send()` or `fi_write()` to libfabric endpoint
- Return request handle for polling

**Key Files**:
- [nccl_ofi_api.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp) - Main ncclNet API layer (dispatches to RDMA/SendRecv transports)
- [nccl_ofi_rdma.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_rdma.cpp) - RDMA transport (P5+, default)
- [nccl_ofi_sendrecv.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_sendrecv.cpp) - SendRecv transport (P4d)
- [nccl_ofi_net.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_net.cpp) - Core plugin logic (get_domain, get_ep, object management)
- [nccl_ofi_mr.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_mr.cpp) - Memory registration cache

- **Timing**: ~1-3 μs (if memory registered)

#### 6. Libfabric Processing

**Libfabric API** ([libfabric/include/rdma/fi_endpoint.h:306-358](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h)):

```c
// Inside OFI plugin - libfabric send operation
fi_send(ep, buf, len, desc, dest_addr, context);
// or for RDMA write
fi_write(ep, buf, len, desc, dest_addr, remote_addr,
         remote_key, context);
```

**EFA Provider Implementation** ([libfabric/prov/efa/src/efa_msg.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_msg.c)):

The EFA provider's send op dispatches to one of two backends depending on
`efa_env.use_data_path_direct` (default **true**):

```c
// EFA provider fi_send implementation (conceptual)
static ssize_t efa_msg_send(struct fid_ep *ep, const void *buf,
                            size_t len, void *desc, fi_addr_t dest_addr,
                            void *context) {
  struct efa_ep *efa_ep = container_of(ep, struct efa_ep, ep_fid);

  if (efa_env.use_data_path_direct) {
    // DEFAULT: build the WQE ourselves and post straight to the mapped SQ.
    // No libibverbs call — see efa_data_path_direct_post_send().
    return efa_data_path_direct_post_send(efa_ep->qp, ...);
  }

  // FALLBACK: hand a WQE to rdma-core via libibverbs.
  struct ibv_send_wr wr = {
    .wr_id = (uintptr_t)context,
    .sg_list = &sge, .num_sge = 1, .opcode = IBV_WR_SEND,
  };
  return ibv_post_send(efa_ep->qp->ibv_qp, &wr, &bad_wr);
}
```

**Libfabric Actions**:
- Validate endpoint and parameters
- Prepare EFA-specific headers (SRD addressing)
- Build scatter-gather list from buffer descriptors
- **Default (dp-direct)**: build the tx WQE on the stack and write it into the
  mapped SQ, then ring the doorbell via MMIO (see next step)
- **Fallback**: call rdma-core `ibv_post_send()` with the work request

**Key Files**:
- [prov/efa/src/efa_msg.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_msg.c) - EFA send/recv operations
- [prov/efa/src/efa_rma.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_rma.c) - EFA RDMA write operations
- [prov/efa/src/efa_data_path_direct.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_data_path_direct.c) - dp-direct QP setup (maps SQ/RQ/CQ, doorbells, WQE entry size, 64-bit req-id)
- [prov/efa/src/efa_data_path_direct_entry.h](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_data_path_direct_entry.h) - inline WQE post + CQE poll fast path

- **Timing**: ~500-1000 ns

#### 7. EFA Provider Data-Path-Direct (DEFAULT)

Since the EFA provider defaults to `FI_EFA_USE_DATA_PATH_DIRECT=true`
([efa_env.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_env.c),
`.use_data_path_direct = true`), the send fast path **bypasses rdma-core
entirely**. The provider builds the WQE and writes it into the memory-mapped
send queue itself.

**Post send** ([efa_data_path_direct_entry.h](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_data_path_direct_entry.h) - `efa_data_path_direct_post_send`):

```c
// Build the WQE on the stack, then copy into the mapped SQ slot.
// struct efa_io_tx_wqe_128 supports the EFA driver r3.3.0 128-byte
// wide-WQE format (falls back to 64-byte WQEs on older devices).
struct efa_io_tx_wqe_128 local_wqe = {0};
struct efa_io_tx_meta_desc *meta_desc = &local_wqe.meta;

efa_post_send_validate(qp);
if (!sq->num_wqe_pending)
    mmio_wc_start();                 // write-combining barrier

efa_data_path_direct_set_ud_addr(meta_desc, ah, qpn, qkey);
efa_set_sq_comp_wrid(meta_desc, &sq->wq, wr_id);   // 64-bit req-id capable
efa_set_common_ctrl_flags(meta_desc, sq, EFA_IO_SEND);
efa_data_path_direct_set_sgl(local_wqe.data.sgl, meta_desc, sge_list, iov_count);

efa_data_path_direct_send_wr_post(qp, sq, &local_wqe);  // copy WQE into mapped SQ
efa_sq_advance_post_idx(sq);
sq->num_wqe_pending++;

if (!(flags & FI_MORE))
    efa_data_path_direct_send_wr_ring_db(sq);  // MMIO doorbell write
```

**Ring doorbell** ([efa_data_path_direct_internal.h](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_data_path_direct_internal.h) - `efa_sq_ring_doorbell`):

```c
EFA_ALWAYS_INLINE void efa_sq_ring_doorbell(struct efa_data_path_direct_sq *sq,
                                            uint32_t pc) {
    udma_to_device_barrier();   // ensure WQE bytes are visible to the device
    mmio_write32(sq->wq.db, pc); // MMIO write to the hardware doorbell register
}
```

**Direct WQE mechanics** (verified in `efa_data_path_direct.c` / `_internal.h`):
- The QP setup maps the SQ, RQ and CQ buffers and records the hardware doorbell
  addresses (`sq.wq.db`, `rq.wq.db`, and the CQ doorbell when
  `HAVE_EFADV_CQ_ATTR_DB`). WQE size is read from the device (`sq_attr.entry_size`)
  and is **64 or 128 bytes** — the EFA driver **r3.3.0** exposes 128-byte
  ("wide") SQ WQEs (`efa_device_support_wide_wqe`).
- **64-bit work request IDs**: when the device advertises
  `EFADV_WQ_CAPS_64_BIT_REQ_ID` and `FI_EFA_USE_SQ_REQ_ID_64_BIT` is enabled
  (default on — `.use_sq_req_id_64_bit = 1` in `efa_env.c`), completions carry a
  64-bit req-id (`efa_cqe_is_64_bit_comp`), removing the old 16-bit req-id table
  indirection.
- WQEs are batched: the provider only rings the doorbell when `FI_MORE` is
  clear or the batch reaches `sq->wq.max_batch`, amortizing the MMIO cost.

- **Timing**: ~80-150 ns (stack WQE build + copy to mapped SQ + doorbell; no
  libibverbs call overhead)

#### 7b. rdma-core (libibverbs) — FALLBACK PATH ONLY

**Verbs API** ([rdma-core/libibverbs/verbs.h](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h)):

```c
// Libfabric EFA provider calls verbs API
int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                  struct ibv_send_wr **bad_wr);
// or
int ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
                  struct ibv_recv_wr **bad_wr);
```

**EFA Provider Implementation** ([rdma-core/providers/efa/verbs.c](https://github.com/linux-rdma/rdma-core/blob/master/providers/efa/verbs.c)):

```c
// EFA provider ibv_post_send implementation
static int efa_post_send(struct ibv_qp *ibqp, struct ibv_send_wr *wr,
                         struct ibv_send_wr **bad_wr) {
  struct efa_qp *qp = to_efa_qp(ibqp);

  while (wr) {
    // Write work queue entry to memory-mapped send queue
    struct efa_io_tx_wqe *wqe = efa_get_next_wqe(qp);

    wqe->addr = wr->sg_list[0].addr;
    wqe->length = wr->sg_list[0].length;
    wqe->lkey = wr->sg_list[0].lkey;
    wqe->wr_id = wr->wr_id;

    // Advance producer index
    qp->sq.pc++;

    wr = wr->next;
  }

  // Ring doorbell via MMIO write
  mmio_write32(qp->sq.db, qp->sq.pc);

  return 0;
}
```

**rdma-core Actions** (fallback path — same mechanics as dp-direct, one extra call layer):
- Write work queue entry (WQE) to memory-mapped send queue buffer
- Ring doorbell register (MMIO write to notify EFA hardware)
- **Zero syscalls** - entirely userspace operation
- No kernel involvement in fast path
- Note: with dp-direct (default) the EFA provider does this WQE-write + doorbell
  itself (Step 7 above); this `efa_post_send()` inside rdma-core is only reached
  when `FI_EFA_USE_DATA_PATH_DIRECT=0`.

**Key Functions**:
- `ibv_post_send()` - Post send work request ([libibverbs/verbs.h](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h))
- `efa_post_send()` - EFA implementation ([providers/efa/verbs.c](https://github.com/linux-rdma/rdma-core/blob/master/providers/efa/verbs.c))
- `mmio_write32()` - Doorbell ring to hardware

- **Timing**: ~100-200 ns (memory writes + doorbell). dp-direct shaves the
  libibverbs call/indirection off this (~80-150 ns effective).

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

**EFA Hardware**:
- Receives packet from network
- DMAs data to registered memory buffer
- Writes completion queue entry (CQE) to memory-mapped CQ

**EFA Provider — data-path-direct CQ poll (DEFAULT)** ([efa_data_path_direct_entry.h](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_data_path_direct_entry.h)):

With dp-direct enabled (the default), `efa_cq_read()` polls the mapped CQ buffer
directly instead of calling `ibv_poll_cq()`. The provider reads the next device
CQE, validates its generation/req-id, and extracts the completion fields inline:

```c
// efa_data_path_direct_start_poll(): read next CQE straight from mapped CQ
data_path_direct->cur_cqe =
    efa_data_path_direct_next_device_cqe_get(data_path_direct);
if (!data_path_direct->cur_cqe)
    return ENOENT;                       // no completion available

// Look up the QP; accept if it's a 64-bit-req-id SQ completion or the
// wrid generation matches (guards against stale/recycled CQEs).
qpn = data_path_direct->cur_cqe->qp_num;
// ... efa_cqe_is_64_bit_comp() / efa_data_path_direct_is_valid_wrid_qp_gen() ...

// Field accessors read directly from the mapped CQE:
//   efa_data_path_direct_wc_read_byte_len(), _read_opcode(),
//   _read_src_qp(), _read_imm_data(), _read_vendor_err(), ...
```

CQ completion draining also uses an MMIO doorbell to advance the consumer index
(`efa_update_cq_doorbell` → `mmio_write32(cq->db, ...)`) when
`HAVE_EFADV_CQ_ATTR_DB` is available. No libibverbs `ibv_poll_cq` call is made on
this path.

**rdma-core — FALLBACK ONLY** ([rdma-core/providers/efa/verbs.c](https://github.com/linux-rdma/rdma-core/blob/master/providers/efa/verbs.c)):

When dp-direct is disabled, `efa_cq_read()` falls back to `ibv_poll_cq()`, which
itself reads the same mapped CQ buffer but through the libibverbs call layer:

```c
// Poll completion queue (fallback path)
int ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc) {
  struct efa_cq *efa_cq = to_efa_cq(cq);

  // Read from memory-mapped CQ buffer
  for (int i = 0; i < num_entries; i++) {
    struct efa_io_rx_cdesc *cqe = &efa_cq->buf[efa_cq->cc & efa_cq->cq_mask];

    // Check if CQE is valid
    if (cqe->phase != efa_cq->phase) break;

    // Extract completion info
    wc[i].wr_id = cqe->wr_id;
    wc[i].status = IBV_WC_SUCCESS;
    wc[i].byte_len = cqe->length;

    efa_cq->cc++;
  }

  return i;
}
```

**Libfabric** ([libfabric/prov/efa/src/efa_cq.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_cq.c)):
```c
// Poll EFA completion queue
ssize_t efa_cq_read(struct fid_cq *cq_fid, void *buf, size_t count) {
  struct efa_cq *cq = container_of(cq_fid, struct efa_cq, cq_fid);

  // DEFAULT: poll the mapped CQ directly (efa_data_path_direct_start_poll/next).
  // FALLBACK: ibv_poll_cq() when use_data_path_direct is false.
  ret = /* dp-direct poll or */ ibv_poll_cq(cq->ibv_cq, count, wc);

  // Translate to libfabric completion format
  for (i = 0; i < ret; i++) {
    comp[i].op_context = (void*)wc[i].wr_id;
    comp[i].flags = 0;
    comp[i].len = wc[i].byte_len;
  }

  return ret;
}
```

**OFI Plugin** ([aws-ofi-nccl/src/nccl_ofi_api.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_api.cpp)):
```c
// Test for completion
ncclResult_t nccl_net_ofi_itest(void* request, int* done, int* size) {
  nccl_net_ofi_req_t* req = (nccl_net_ofi_req_t*)request;

  // Poll libfabric CQ
  ret = fi_cq_read(req->comm->cq, &comp, 1);

  if (ret > 0) {
    *done = 1;
    *size = comp.len;
  }

  return ncclSuccess;
}
```

**Proxy Thread** ([nccl/src/proxy.cc](https://github.com/NVIDIA/nccl/blob/master/src/proxy.cc)):
- Polls for network completion via `ncclNet->itest()`
- Signals GPU kernel via shared memory flag
- Updates NCCL channel state

**Flow Summary**:
```
EFA Hardware
  ↓ DMA to memory, write completion to memory-mapped CQ
rdma-core
  ↓ ibv_poll_cq() reads CQE from memory
Libfabric
  ↓ fi_cq_read() translates to libfabric format
OFI Plugin
  ↓ ncclNet->itest() returns completion
Proxy Thread
  ↓ Signal GPU via shared memory
NCCL Kernel
  ↓ Copy to destination, continue operation
GPU Memory
```

#### 11. Completion

**NCCL Proxy** ([nccl/src/proxy.cc](https://github.com/NVIDIA/nccl/blob/master/src/proxy.cc)):
- Proxy polls completion queue via `ncclNet->itest()`
- Calls through libfabric `fi_cq_read()` → rdma-core `ibv_poll_cq()`
- Updates NCCL channel state when complete
- Signals GPU kernel to continue

**GPU Kernel**:
- Waits for proxy signal via shared memory polling
- Continues with next operation step
- Eventually completes, allowing CUDA stream to progress

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
# There is no NCCL_REGISTRATION_CACHE_SIZE variable. Provider-level MR cache bounds:
FI_EFA_MR_MAX_CACHED_SIZE=<bytes>    # total cached registration bytes
FI_EFA_MR_MAX_CACHED_COUNT=<n>       # number of cached registrations
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
- `fi_send()` - Send message ([fi_endpoint.h:326](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_endpoint.h))
- `fi_recv()` - Receive message
- `fi_write()` - RDMA write
- `fi_mr_reg()` - Register memory region ([fi_domain.h, ~line 413](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h))
- `fi_cq_read()` - Read completion queue ([fi_eq.h](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_eq.h))

**rdma-core (libibverbs)** - Referenced but not linked (standard RDMA API):
- `ibv_post_send()` - Post send work request ([verbs.h, ~line 2554](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h))
- `ibv_post_recv()` - Post receive work request ([verbs.h, ~line 2562](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h))
- `ibv_poll_cq()` - Poll completion queue ([verbs.h, ~line 2576](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/verbs.h))

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
- `FI_EFA_USE_DATA_PATH_DIRECT` - EFA provider builds/posts WQEs and polls CQEs itself, bypassing rdma-core on the data path (**default: true**; verified in `efa_env.c`). Set to `0` to force the libibverbs fallback.
- `FI_EFA_USE_SQ_REQ_ID_64_BIT` - Use 64-bit SQ work-request IDs when the device supports it (**default: on**; requires `EFADV_WQ_CAPS_64_BIT_REQ_ID`, EFA driver r3.3.0+).

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
