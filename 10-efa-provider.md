# EFA Provider in Libfabric

## Overview

The EFA (Elastic Fabric Adapter) provider in libfabric implements support for AWS's custom network adapter. EFA provides high-bandwidth, low-latency networking for EC2 instances.

**Location**: `prov/efa/` in libfabric source

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
- **p4d.24xlarge**: 4x 100 Gbps EFA (400 Gbps total)
- **p5.48xlarge**: 8x 200 Gbps EFA (1600 Gbps total - newest)
- **p3dn.24xlarge**: 4x 100 Gbps EFA
- **c5n.18xlarge**: 1x 100 Gbps EFA
- **c6gn.16xlarge**: 1x 100 Gbps EFA
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

```c
struct efa_rdm_ep {  // ([prov/efa/src/rdm/efa_rdm_ep.h:46-120](https://github.com/ofiwg/libfabric/blob/6b9e629/prov/efa/src/rdm/efa_rdm_ep.h#L46-L120))
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

```c
struct efa_rdm_pke {  // ([prov/efa/src/rdm/efa_rdm_pke.h:78-100](https://github.com/ofiwg/libfabric/blob/6b9e629/prov/efa/src/rdm/efa_rdm_pke.h#L78-L100))
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

The `struct efa_mr` ([prov/efa/src/efa_mr.h:20-32](https://github.com/ofiwg/libfabric/blob/6b9e629/prov/efa/src/efa_mr.h#L20-L32)) wraps the ibverbs memory region.

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
# Tunable threshold
FI_EFA_RDM_LONG_MSG_SIZE=65536  # 64 KB default

# Below threshold: eager
# Above threshold: rendezvous
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

# Eager/rendezvous threshold (tune based on workload)
FI_EFA_RDM_LONG_MSG_SIZE=65536  # 64 KB

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
