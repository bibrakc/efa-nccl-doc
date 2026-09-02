# EFA Provider in Libfabric

## Overview

The EFA (Elastic Fabric Adapter) provider in libfabric implements support for AWS's custom network adapter. EFA provides high-bandwidth, low-latency networking for EC2 instances.

**Location**: `prov/efa/` in libfabric source (this document reflects libfabric
**2.7.0rc1**, `main`).

**What changed recently (libfabric 2.3.0 → 2.7.0):**
- **EFA Data Path Direct** is now the default data path — the provider writes
  SQ/RQ WQEs and reads CQEs directly instead of calling libibverbs
  (`FI_EFA_USE_DATA_PATH_DIRECT=true`). Full mechanism in
  [rdma-core-and-verbs.md](rdma-core-and-verbs.md); the `efa_qp_post_send/recv/read/write`
  and `efa_ibv_cq_*` wrappers dispatch to it in
  [prov/efa/src/efa_data_path_ops.h](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_data_path_ops.h).
- **PEER_ERROR "receiver-decides" model** (new in 2.7) for MR-abort / receiver-side
  failure signalling — see [MR Abort and the PEER_ERROR packet](#mr-abort-and-the-peer_error-packet).
- **Extensive provider-internal locking/threading rework** — lock-free peer map
  and AV array, per-endpoint locks, thread-safety-analysis annotations. See
  [Provider-internal locking and threading](#provider-internal-locking-and-threading).
  The supported endpoint threading models are now **FI_THREAD_SAFE**,
  **FI_THREAD_COMPLETION** and **FI_THREAD_DOMAIN** for both RDM and DGRAM endpoints
  ([man/fi_efa.7.md](https://github.com/ofiwg/libfabric/blob/main/man/fi_efa.7.md)); the
  cross-stack threading discussion lives in [threading-model.md](threading-model.md).

## EFA Hardware Characteristics

### Network Interface

```
EFA Adapter (NIC)
┌─────────────────────────────────────────┐
│  PCIe Gen4 x16 Interface                │
│  ├─ To CPU/GPU: ~25 GB/s                │
│  └─ Bidirectional                       │
├─────────────────────────────────────────┤
│  Network Interface                      │
│  ├─ 100 Gbps Ethernet                   │
│  ├─ ~12.5 GB/s theoretical              │
│  └─ ~11-12 GB/s practical               │
├─────────────────────────────────────────┤
│  Hardware Features                      │
│  ├─ OS bypass (kernel bypass)           │
│  ├─ RDMA-like operations                │
│  ├─ Scalable Reliable Datagram (SRD)    │
│  ├─ Hardware retransmission             │
│  ├─ Out-of-order delivery               │
│  └─ Limited atomics                     │
└─────────────────────────────────────────┘
```

### Instance Support

**EFA-Enabled Instance Types:**

**GPU Training Instances:**
- **p4d.24xlarge**: 4×100G EFA (400 Gbps total, 8×A100 GPUs) - Send/recv only
- **p4de.24xlarge**: 4×100G EFA (400 Gbps total, 8×A100 GPUs) - Send/recv only
- **p5.48xlarge**: 32×100G EFA (3200 Gbps total, 8×H100 GPUs) - RDMA supported, EFAv2
- **p5e.48xlarge**: 32×100G EFA (3200 Gbps total, 8×H100 GPUs) - RDMA supported, EFAv2
- **p5en.48xlarge**: 32×100G EFA (3200 Gbps total, 8×H200 GPUs) - RDMA supported, EFAv3, 35% lower latency

**Older/Compute Instances:**
- **p3dn.24xlarge**: 4×100G EFA (older generation)
- **c5n.18xlarge**: 1×100G EFA
- **c6gn.16xlarge**: 1×100G EFA
- And others...

**Multi-Rail:**
- Instances can have 1-8 EFA adapters
- Each appears as separate device to libfabric
- NCCL can utilize all adapters for parallel transfers

## EFA Transport: Scalable Reliable Datagram (SRD)

### Protocol Characteristics

```
Standard UD (Unreliable Datagram):
  ├─ Connectionless
  ├─ No reliability
  └─ No ordering

SRD (Scalable Reliable Datagram):
  ├─ Connectionless
  ├─ Hardware-level reliability (retransmission)
  ├─ No strict ordering (reordering possible)
  └─ Scalable (no per-peer state)
```

**Key Properties:**
- **Reliability**: Hardware retransmits lost packets
- **No Connection Setup**: No handshake needed
- **Scalability**: O(1) state per endpoint, not O(N²)
- **Reordering**: Out-of-order delivery is possible
- **Performance**: Low latency, high throughput

### Reliability Mechanism

```
Sender                          Receiver
  │                               │
  ├─ Send Packet 1 ──────────────→│
  ├─ Send Packet 2 ──────X (lost)
  ├─ Send Packet 3 ──────────────→│
  │                               │
  │                               ├─ Receive 1, 3
  │                               ├─ Detect gap (2 missing)
  │                               │
  │←────── NAK for Packet 2 ──────┤
  │                               │
  ├─ Retransmit Packet 2 ────────→│
  │                               ├─ Receive 2
```

**Hardware-driven:**
- EFA NIC tracks sequence numbers
- Receiver NIC sends NAKs
- Sender NIC retransmits
- No CPU involvement

### Message Ordering

**No guaranteed ordering between messages:**
```
Send order:    [A] [B] [C]
Receive order: [A] [C] [B]  ← Possible!
```

**Within a message:**
- Packet order may vary
- Full message assembled before delivery
- Application sees complete messages only

**Implications for NCCL:**
- NCCL handles ordering at higher level
- Tagged messages for matching
- Flags/counters for synchronization

## EFA Provider Architecture

### Endpoint Types

EFA provider supports:

```c
FI_EP_RDM  // Reliable Datagram (primary)
FI_EP_DGRAM // Unreliable Datagram (rarely used)
```

**RDM Endpoint:**
- Built on top of SRD
- Provides reliable, unordered delivery
- Connection-less
- Supports message, tagged, and RMA

### Capabilities

```c
// EFA provider capabilities
caps = FI_MSG           // Message operations
     | FI_TAGGED        // Tagged messaging
     | FI_RMA           // RDMA read/write
     | FI_SEND
     | FI_RECV
     | FI_READ
     | FI_WRITE
     | FI_REMOTE_READ
     | FI_REMOTE_WRITE;
```

**NOT supported:**
- FI_ATOMIC (limited support)
- FI_COLLECTIVE
- FI_MULTICAST

### Data Structures

#### EFA Endpoint

**`struct efa_rdm_ep`** - EFA RDM endpoint ([prov/efa/src/rdm/efa_rdm_ep.h:46-120](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/rdm/efa_rdm_ep.h)):

```c
struct efa_rdm_ep {
  struct fid_ep ep_fid;

  // Core EFA resources
  struct efa_base_ep *base_ep;
  struct ibv_qp *qp;              // Queue pair
  struct ibv_cq *ibv_cq_ex;       // Completion queue

  // Packet management
  struct efa_rdm_pke *tx_pkt_pool;  // TX packet pool
  struct efa_rdm_pke *rx_pkt_pool;  // RX packet pool

  // Protocol state
  struct efa_rdm_peer *peer_list;   // Per-peer state
  struct dlist_entry rxe_list;      // RX entries
  struct dlist_entry txe_list;      // TX entries

  // Buffers
  void *rx_unexp_buf;              // Unexpected message buffer
  void *tx_bounce_buf;             // Bounce buffer for sends

  // Counters
  uint64_t tx_pending;
  uint64_t rx_posted;
};
```

#### Packet Entry

**`struct efa_rdm_pke`** - Packet entry ([prov/efa/src/rdm/efa_rdm_pke.h:78-100](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/rdm/efa_rdm_pke.h)):

```c
struct efa_rdm_pke {
  struct dlist_entry entry;
  void *payload;                   // Packet payload
  size_t payload_size;
  uint64_t flags;
  void *context;                   // User context

  // Protocol fields
  uint32_t msg_id;                 // Message identifier
  uint32_t seg_offset;             // Segment offset (for large msgs)
  uint16_t seg_size;               // Segment size
};
```

### Send Path

```c
// fi_send() or fi_tsend()
ssize_t efa_rdm_send(struct fid_ep *ep,
                     const void *buf, size_t len,
                     void *desc, fi_addr_t dest_addr,
                     void *context)
{
  struct efa_rdm_ep *efa_ep = container_of(ep, ...);

  // 1. Allocate TX entry
  txe = efa_rdm_get_txe(efa_ep);
  txe->context = context;
  txe->addr = dest_addr;
  txe->total_len = len;

  // 2. Determine protocol
  if (len <= EAGER_THRESHOLD) {
    // Eager: send data immediately
    return efa_rdm_send_eager(efa_ep, txe, buf, len);
  } else {
    // Rendezvous: exchange RTR/CTS first
    return efa_rdm_send_rts(efa_ep, txe);
  }
}
```

#### Eager Protocol

For small messages (< 64 KB typically):

```
Sender                          Receiver
  │                               │
  │─── DATA packet ──────────────→│
  │    (payload included)         │
  │                               ├─ Match & deliver
  │                               │
  CQ completion                   CQ completion
```

**Implementation:**
```c
int efa_rdm_send_eager(struct efa_rdm_ep *ep,
                       struct efa_rdm_txe *txe,
                       const void *buf, size_t len)
{
  // Allocate packet
  pkt = efa_rdm_get_pkt(ep);

  // Build packet header
  pkt->hdr.type = EFA_RDM_EAGER_MSGRTM;
  pkt->hdr.msg_id = txe->msg_id;
  pkt->hdr.tag = txe->tag;

  // Copy data to packet
  memcpy(pkt->payload, buf, len);

  // Post to hardware
  ibv_post_send(ep->qp, &wr, &bad_wr);

  return 0;
}
```

**Characteristics:**
- Single packet transfer
- Low latency
- Limited size (MTU constrained)

#### Rendezvous Protocol

For large messages:

```
Sender                          Receiver
  │                               │
  ├─── RTS (Ready-to-Send) ──────→│
  │    (header only)              │
  │                               ├─ Allocate buffer
  │                               │
  │←─── CTS (Clear-to-Send) ──────┤
  │    (buffer address)           │
  │                               │
  ├─── DATA (RDMA write) ────────→│
  │                               │ (DMA)
  │                               │
  ├─── RECEIPT ──────────────────→│
  │                               │
  CQ completion                   CQ completion
```

**Implementation:**
```c
// Step 1: Send RTS
int efa_rdm_send_rts(struct efa_rdm_ep *ep,
                     struct efa_rdm_txe *txe)
{
  pkt = efa_rdm_get_pkt(ep);
  pkt->hdr.type = EFA_RDM_LONGCTS_MSGRTM;
  pkt->hdr.msg_id = txe->msg_id;
  pkt->hdr.data_len = txe->total_len;
  // No payload, just header

  ibv_post_send(ep->qp, &wr, &bad_wr);
}

// Step 2: Receive CTS, send DATA
int efa_rdm_handle_cts(struct efa_rdm_ep *ep,
                       struct efa_rdm_pke *pkt)
{
  txe = find_txe(pkt->hdr.msg_id);

  // Extract remote buffer info
  remote_addr = pkt->cts.recv_addr;
  remote_key = pkt->cts.recv_key;

  // RDMA write
  ibv_post_send_write(ep->qp, txe->buf, txe->len,
                      remote_addr, remote_key);
}
```

**Characteristics:**
- Multi-round protocol
- RDMA for bulk data
- Better for large transfers
- Higher setup latency, better bandwidth

### Receive Path

#### Pre-posting

```c
ssize_t efa_rdm_recv(struct fid_ep *ep,
                     void *buf, size_t len,
                     void *desc, fi_addr_t src_addr,
                     void *context)
{
  struct efa_rdm_ep *efa_ep = container_of(ep, ...);

  // Allocate RX entry
  rxe = efa_rdm_get_rxe(efa_ep);
  rxe->context = context;
  rxe->buf = buf;
  rxe->len = len;
  rxe->addr = src_addr;

  // Add to expected receive queue
  dlist_insert_tail(&rxe->entry, &efa_ep->rxe_list);

  return 0;  // Will match later
}
```

#### Message Matching

```c
int efa_rdm_match_msg(struct efa_rdm_ep *ep,
                      struct efa_rdm_pke *pkt)
{
  // For tagged messages
  uint64_t recv_tag = rxe->tag;
  uint64_t recv_ignore = rxe->ignore;
  uint64_t send_tag = pkt->hdr.tag;

  if ((recv_tag & ~recv_ignore) == (send_tag & ~recv_ignore)) {
    // Match!
    if (rxe->addr == FI_ADDR_UNSPEC || rxe->addr == pkt->addr) {
      return 1;  // Matched
    }
  }

  return 0;  // No match
}
```

#### Processing

```c
int efa_rdm_progress_recv(struct efa_rdm_ep *ep)
{
  // Poll hardware CQ
  num = ibv_poll_cq(ep->ibv_cq_ex, WC_BATCH_SIZE, wc);

  for (int i = 0; i < num; i++) {
    pkt = wc[i].wr_id;  // Packet entry

    switch (pkt->hdr.type) {
      case EFA_RDM_EAGER_MSGRTM:
        // Match with posted recv
        rxe = match_rxe(ep, pkt);
        if (rxe) {
          // Copy data to user buffer
          memcpy(rxe->buf, pkt->payload, pkt->hdr.data_len);
          // Complete
          efa_rdm_cq_write_rx_completion(ep, rxe);
        } else {
          // Unexpected, buffer for later
          queue_unexpected(ep, pkt);
        }
        break;

      case EFA_RDM_LONGCTS_MSGRTM:
        // RTS received, send CTS
        handle_rts(ep, pkt);
        break;

      // ... other packet types
    }
  }
}
```

### RDMA Operations

#### Write

```c
ssize_t efa_rdm_write(struct fid_ep *ep,
                      const void *buf, size_t len,
                      void *desc, fi_addr_t dest_addr,
                      uint64_t addr, uint64_t key,
                      void *context)
{
  // Use ibverbs RDMA write
  struct ibv_send_wr wr = {
    .opcode = IBV_WR_RDMA_WRITE,
    .send_flags = IBV_SEND_SIGNALED,
    .wr_id = (uint64_t)context,
    .sg_list = &sge,
    .num_sge = 1,
    .wr.rdma = {
      .remote_addr = addr,
      .rkey = key,
    },
  };

  ibv_post_send(ep->qp, &wr, &bad_wr);
}
```

**EFA RDMA Write:**
- One-sided operation
- No remote CPU involvement
- Requires pre-exchanged address and key
- Used in rendezvous protocol

#### Read

```c
ssize_t efa_rdm_read(struct fid_ep *ep,
                     void *buf, size_t len,
                     void *desc, fi_addr_t src_addr,
                     uint64_t addr, uint64_t key,
                     void *context)
{
  struct ibv_send_wr wr = {
    .opcode = IBV_WR_RDMA_READ,
    .send_flags = IBV_SEND_SIGNALED,
    .wr_id = (uint64_t)context,
    .sg_list = &sge,
    .num_sge = 1,
    .wr.rdma = {
      .remote_addr = addr,
      .rkey = key,
    },
  };

  ibv_post_send(ep->qp, &wr, &bad_wr);
}
```

**Used by NCCL for:**
- Flush operations (dummy read for ordering)
- Some protocol variants

### Memory Registration

EFA uses ibverbs for memory registration:

The `struct efa_mr` ([prov/efa/src/efa_mr.h:20-32](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_mr.h)) wraps the ibverbs memory region.

```c
int efa_mr_reg(struct fid_domain *domain,
               const void *buf, size_t len,
               uint64_t access, uint64_t offset,
               uint64_t requested_key, uint64_t flags,
               struct fid_mr **mr, void *context)
{
  struct efa_domain *efa_domain = container_of(domain, ...);

  // Register with ibverbs
  int ibv_access = 0;
  if (access & FI_SEND) ibv_access |= IBV_ACCESS_LOCAL_WRITE;
  if (access & FI_RECV) ibv_access |= IBV_ACCESS_LOCAL_WRITE;
  if (access & FI_REMOTE_READ) ibv_access |= IBV_ACCESS_REMOTE_READ;
  if (access & FI_REMOTE_WRITE) ibv_access |= IBV_ACCESS_REMOTE_WRITE;

  struct ibv_mr *ibv_mr;
  if (flags & FI_HMEM_CUDA) {
    // GPU memory via dmabuf
    ibv_mr = ibv_reg_dmabuf_mr(efa_domain->pd, offset, len,
                               (uint64_t)buf, dmabuf_fd,
                               ibv_access);
  } else {
    // Host memory
    ibv_mr = ibv_reg_mr(efa_domain->pd, (void *)buf, len,
                        ibv_access);
  }

  *mr = &efa_mr->mr_fid;
  return 0;
}
```

**GPU Memory (dmabuf):**
- Modern approach using dmabuf (DMA buffer)
- Kernel 5.12+ required
- Replaces older peer-direct approach
- More efficient, better compatibility

**Memory Registration Cache:**
- EFA provider includes built-in MR cache
- Uses memory hooks (or userfaultfd) to track invalidations
- Critical for performance

```bash
# Cache configuration
FI_MR_CACHE_MONITOR=memhooks    # or userfaultfd
FI_MR_CACHE_MAX_SIZE=unlimited
FI_MR_CACHE_MAX_COUNT=unlimited
```

## MR Abort and the PEER_ERROR packet

**New in libfabric 2.7.** On an `FI_EP_RDM` `efa` endpoint an application can
**abort an in-flight transfer by closing the source memory region** (`fi_close()`
on the MR backing the local send/write/read buffer, or the remote MR of a
read/write). Closing the MR signals the provider to cancel any in-flight operation
that still references it. (This does **not** apply to MRs referenced by posted
receives — the provider fails to close such an MR; the only way to abort receives
is to close the endpoint.)

To make the *receiver* side consistent after such an abort, the provider sends an
**`EFA_RDM_PEER_ERROR_PKT`** control packet. The model is **"receiver-decides"**:

- The **rxe** (receive op) emits its `PEER_ERROR_PKT` eagerly at mark time; the
  **txe** (transmit op) defers its emit. Each ope carries a provider errno to
  attach to the packet and a flag
  (`EFA_RDM_PEER_ERROR_EMITTED_OR_SKIPPED`, `BIT_ULL(21)`) so it is emitted at most
  once (see [prov/efa/src/rdm/efa_rdm_ope.h](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/rdm/efa_rdm_ope.h)).
- The packet carries the **emitter's ope type** so the receiver can resolve the
  matching rxe/txe via the peer's `rxe_list` and unblock the reorder window.
- Aborting cancels in-flight **LONGREAD** RTMs and rebalances the read counter;
  aborted overflow packets are turned into in-place abort markers; pre-handshake
  peer-abort emits are **deferred until the handshake** completes.

**Version / capability requirement.** PEER_ERROR support is negotiated in the
handshake via an extra-feature bit
(`EFA_RDM_EXTRA_FEATURE_PEER_ERROR`, `efa_rdm_peer_support_peer_error()` in
[prov/efa/src/rdm/efa_rdm_peer.h](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/rdm/efa_rdm_peer.h)).
A peer without the bit cannot decode a `PEER_ERROR_PKT`, so the provider does not
emit one and falls back to the legacy behavior (leak the txe / write a local CQ
error). Per the [fi_efa man page](https://github.com/ofiwg/libfabric/blob/main/man/fi_efa.7.md)
**"MR ABORTING" section, MR-abort behavior is undefined unless both participants
run libfabric 2.7.0 or later** — an older peer cannot distinguish a control message
of the current transfer from one of an earlier, aborted transfer, risking
communication-state corruption.

**Scope.** Peer notification / remote-side cleanup on MR abort applies to **send and
tagged operations only**. Device RMA operations that reference a closed MR need no
peer cleanup (the initiator gets a local error completion, no provider state exists
on the peer). Emulated RMA and atomic operations are **not** supported — closing an
MR referenced by an in-flight emulated operation is undefined behavior.

## Provider-internal locking and threading

libfabric 2.7 reworked the EFA provider's locking to reduce contention and to make
the lock discipline machine-checkable. Highlights (the cross-stack model is in
[threading-model.md](threading-model.md); this section covers the provider-internal
detail):

- **Lock-free per-endpoint peer map.** The peer state is held in a per-endpoint,
  `fi_addr`-indexed map with **lock-free reads** on the fast path.
- **Lock-free `efa_av_array`.** A new lock-free address-vector array
  ([prov/efa/src/efa_av_array.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_av_array.c) / `.h`)
  replaces lock-protected AV lookups.
- **srx lock moved from domain to endpoint**, and **removed** from AV operations,
  EP enable, and SHM address lookup. Under `FI_THREAD_COMPLETION` the srx lock is
  made `OFI_LOCK_NONE` (no-op), since completion-threading already partitions work
  by CQ.
- **`util_domain` lock made a no-op under `FI_PROGRESS_CONTROL_UNIFIED`.**
- **Queued progress lists moved from domain to endpoint**, plus a `progress_ep_list`
  so CQ progress can **skip idle endpoints** rather than walking every EP on the domain.
- **`domain->num_read_msg_in_flight` made atomic** so the RDMA-read balance counter
  no longer needs a lock.
- **Thread-safety-analysis annotations.** New
  [prov/efa/src/efa_thread_annotations.h](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_thread_annotations.h)
  defines `OFI_TSA_*` macros (Clang `-Wthread-safety` `guarded_by`, `requires_capability`,
  `acquire/release_capability`, `OFI_TSA_NO_ANALYSIS`, …). Building with
  `configure --enable-thread-safety-analysis` turns these into compile-time lock-discipline
  checks. For example, the data-path-direct wrid pool documents that post and
  completion must serialize on `wq->wqlock`
  (`assert(ofi_genlock_held(wq->wqlock))`), and several CQ-path helpers are marked
  `OFI_TSA_NO_ANALYSIS` because they run under the CQ lock.

**Supported endpoint threading models** (from
[man/fi_efa.7.md](https://github.com/ofiwg/libfabric/blob/main/man/fi_efa.7.md)):
both RDM and DGRAM endpoints support **`FI_THREAD_SAFE`**, **`FI_THREAD_COMPLETION`**
and **`FI_THREAD_DOMAIN`**.

## Other provider changes (libfabric 2.7)

- **QKEY generation unified across all endpoint types** — a single QKEY-generation
  scheme is now used for RDM, DGRAM and direct endpoints.
- **Device ID included in the "EFA-ness" discriminator** — the check that decides
  whether an ibverbs device is an EFA device now also considers the device ID
  (in addition to the vendor/part ID), and a separate fix corrected a **vendor ID
  vs part ID** confusion.
- **Robust multi-device init** — if one EFA device fails to initialize, the provider
  continues initializing the remaining devices instead of aborting.

## Performance Characteristics

### Latency

```
Operation           Latency (μs)
─────────────────────────────────
Small eager send    ~10-15
Large RDM send      ~20-30 (setup) + data transfer
RDMA write          ~10-15
RDMA read           ~15-20
```

### Bandwidth

```
Message Size        Bandwidth (single connection)
──────────────────────────────────────────────────
< 4 KB              ~1-2 GB/s (latency bound)
64 KB               ~6-8 GB/s
1 MB                ~10-11 GB/s
> 4 MB              ~11-12 GB/s (line rate)
```

**Multi-connection scaling:**
- 2 connections: ~20 GB/s
- 4 connections: ~35-40 GB/s
- 8 connections: ~45-50 GB/s (PCIe limit)

### Eager vs Rendezvous Threshold

```bash
# There is no FI_EFA_RDM_LONG_MSG_SIZE knob. The real protocol-switch
# thresholds are (defaults from prov/efa/src/efa_env.c):
FI_EFA_INTER_MAX_MEDIUM_MESSAGE_SIZE=65536    # 64 KB  - max size for the medium protocol
FI_EFA_INTER_MIN_READ_MESSAGE_SIZE=1048576    # 1 MB   - min size for the read (rendezvous) protocol
FI_EFA_INTER_MAX_GDRCOPY_MESSAGE_SIZE=32768   # 32 KB  - max size that uses gdrcopy
FI_EFA_RUNT_SIZE=307200                       # 300 KB - bytes sent eagerly by the runting read protocol

# Below the read threshold: eager / medium / longcts
# At or above it:          RDMA-read based rendezvous
```

**Optimal value:**
- Smaller (32 KB): Lower latency for medium msgs
- Larger (128 KB): Fewer protocol rounds
- Default (64 KB): Good balance

## Advanced Features

### Shared Memory Optimization

EFA provider can use shared memory for intra-node:

```bash
FI_EFA_ENABLE_SHM_TRANSFER=1  # Enable (default)
```

**When enabled:**
- Intra-node traffic uses shared memory
- Bypasses EFA hardware
- Higher bandwidth, lower latency

**NCCL typically disables this:**
- NCCL has its own intra-node optimizations (NVLink, SHM)
- Avoids double-buffering
- Better integration with GPU

```bash
FI_EFA_ENABLE_SHM_TRANSFER=0  # Typical for NCCL
```

### Zero-Copy Receive

```bash
FI_EFA_USE_DEVICE_RDMA=1  # Enable RDMA (default)
```

**When enabled:**
- Uses RDMA write for large messages
- Zero-copy on receive side
- Better performance

### Endpoint Scalability

```c
// Per-endpoint resources
struct efa_rdm_ep {
  struct ibv_qp *qp;              // 1 QP per endpoint
  struct ibv_cq *ibv_cq_ex;       // 1 CQ per endpoint
  // ...
};
```

**Scalability:**
- Each endpoint = 1 QP + 1 CQ
- Hardware limit: ~thousands of QPs
- Memory overhead: ~few MB per endpoint

**NCCL typically uses:**
- Multiple endpoints per NIC
- One endpoint per connection
- Can create hundreds of endpoints

## Tuning for NCCL

### Recommended Settings

```bash
# Disable intra-node SHM (NCCL handles it)
FI_EFA_ENABLE_SHM_TRANSFER=0

# Queue depth (balance memory vs performance)
FI_EFA_TX_SIZE=256
FI_EFA_RX_SIZE=256

# RDMA operations
FI_EFA_USE_DEVICE_RDMA=1

# Memory registration cache
FI_MR_CACHE_MONITOR=memhooks
FI_MR_CACHE_MAX_SIZE=unlimited

# Eager/medium/rendezvous thresholds (tune based on workload)
FI_EFA_INTER_MAX_MEDIUM_MESSAGE_SIZE=65536  # 64 KB
FI_EFA_INTER_MIN_READ_MESSAGE_SIZE=1048576  # 1 MB

# Logging (for debugging)
FI_LOG_LEVEL=warn
FI_LOG_PROV=efa
```

### Performance Profiling

```bash
# Enable detailed logging
FI_LOG_LEVEL=debug
export FI_LOG_PROV=efa

# Check MR cache effectiveness
# Look for "mr cache hit" vs "mr cache miss" in logs
```

## Summary

**EFA Provider Characteristics:**
1. **Transport**: Scalable Reliable Datagram (SRD)
2. **Reliability**: Hardware-level retransmission
3. **Ordering**: No guaranteed order, reordering possible
4. **Endpoint**: RDM (reliable datagram message)
5. **Protocols**: Eager (small), Rendezvous (large)
6. **RDMA**: Read/write supported

**Key Features:**
- **Multi-rail**: Support for multiple EFAs
- **GPU Direct**: dmabuf-based GPU memory
- **MR Caching**: Built-in registration cache
- **SHM Fallback**: Optional shared memory for intra-node
- **Zero-copy**: RDMA write for receives

**Performance:**
- **Latency**: ~10-15 μs (small messages)
- **Bandwidth**: ~11-12 GB/s per 100G EFA
- **Scaling**: Multiple connections/rails for higher throughput

**Critical for NCCL:**
- Pre-post receives for low latency
- MR caching for repeated transfers
- Disable SHM (NCCL handles intra-node)
- Tune eager threshold based on message sizes
- Multiple endpoints for parallelism

**Next**: Threading models and how they interact across the stack.
