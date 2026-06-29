# OFI NCCL Plugin: Connection Establishment and Send/Recv Protocols

## Overview

This document provides detailed descriptions of the connection establishment mechanism and send/recv protocols used in the OFI NCCL plugin, including the actual message flows and state management.

## Connection Establishment Protocol

### High-Level Flow

The OFI plugin implements the NCCL `ncclNet_t` interface for connection management:

```
NCCL Initiates Connection:
  Receiver (Passive)              Sender (Active)
  ─────────────────               ──────────────

  1. ncclNet->listen()
     ├─ Create endpoint
     ├─ Get local address
     └─ Return handle
                                  2. ncclNet->connect(handle)
                                     ├─ Create endpoint
                                     ├─ Extract peer address
                                     └─ Insert into AV

  3. ncclNet->accept()
     └─ Return recv comm
                                  4. Ready to communicate
                                     ├─ fi_tsend()
                                     └─ fi_trecv()
```

### Detailed Connection Steps

#### Step 1: Listener (Passive Side) - `ncclNet->listen()`

**Purpose**: Receiver creates an endpoint and exports its address for the sender to connect.

```c
ncclResult_t nccl_net_ofi_listen(int dev, void* handle,
                                 void** listenComm)
{
  auto *comm = new nccl_net_ofi_listen_comm();  // C++ class, not C struct  // See struct nccl_net_ofi_listen_comm (https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi.h)

  // 1. Create libfabric endpoint
  struct fi_info* info = get_efa_info(dev);
  ret = fi_endpoint(domain, info, &comm->ep, NULL);

  // 2. Create completion queue (if not shared)
  struct fi_cq_attr cq_attr = {
    .size = 1024,
    .format = FI_CQ_FORMAT_DATA,
  };
  ret = fi_cq_open(domain, &cq_attr, &comm->cq, NULL);  // See fi_cq_open() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L363)

  // 3. Bind endpoint to resources
  fi_ep_bind(comm->ep, &comm->cq->fid, FI_TRANSMIT | FI_RECV);
  fi_ep_bind(comm->ep, &av->fid, 0);  // Address vector

  // 4. Enable endpoint
  fi_enable(comm->ep);

  // 5. Get local endpoint address
  size_t addrlen = MAX_EP_ADDR_LEN;
  ret = fi_getname(&comm->ep->fid, &local_addr, &addrlen);

  // 6. Serialize address into handle (for NCCL to send to peer)
  // struct nccl_ofi_handle - conceptual structure for handle serialization
  struct nccl_ofi_handle {
    char ep_addr[MAX_EP_ADDR_LEN];
    size_t ep_addr_len;
  };

  struct nccl_ofi_handle* h = (struct nccl_ofi_handle*)handle;
  memcpy(h->ep_addr, &local_addr, addrlen);
  h->ep_addr_len = addrlen;

  // 7. Store state
  *listenComm = comm;

  return ncclSuccess;
}
```

**What gets stored in `handle`:**
- Endpoint address (libfabric address, e.g., `<ip>:<port>` equivalent for EFA)
- Address length
- Device ID
- Any plugin-specific metadata

**NCCL's role**: NCCL will send this handle to the sender through its bootstrap network (out-of-band, typically TCP).

#### Step 2: Connector (Active Side) - `ncclNet->connect()`

**Purpose**: Sender receives the receiver's handle, creates its own endpoint, and establishes "connection" (state tracking for connectionless RDM).

```c
ncclResult_t nccl_net_ofi_connect(int dev, void* handle,
                                  void** sendComm)
{
  auto *comm = new nccl_net_ofi_send_comm();  // C++ class, not C struct  // See struct nccl_net_ofi_send_comm (https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi.h)

  // 1. Deserialize peer's address from handle
  struct nccl_ofi_handle* h = (struct nccl_ofi_handle*)handle;
  void* peer_addr = h->ep_addr;
  size_t peer_addr_len = h->ep_addr_len;

  // 2. Create local endpoint
  struct fi_info* info = get_efa_info(dev);
  ret = fi_endpoint(domain, info, &comm->ep, NULL);

  // 3. Create/share completion queue
  ret = fi_cq_open(domain, &cq_attr, &comm->cq, NULL);

  // 4. Bind endpoint
  fi_ep_bind(comm->ep, &comm->cq->fid, FI_TRANSMIT | FI_RECV);
  fi_ep_bind(comm->ep, &av->fid, 0);

  // 5. Enable endpoint
  fi_enable(comm->ep);

  // 6. Insert peer address into Address Vector (AV)
  //    This maps peer's address to fi_addr_t for use in sends
  fi_addr_t remote_fi_addr;
  ret = fi_av_insert(av, peer_addr, 1, &remote_fi_addr, 0, NULL);

  // 7. Store remote address for future sends
  comm->remote_addr = remote_fi_addr;

  // 8. Initialize request tracking
  comm->pending_sends = 0;
  comm->send_head = 0;
  comm->send_tail = 0;

  // 9. Return send comm
  *sendComm = comm;

  return ncclSuccess;
}
```

**Key Point**: For EFA (RDM endpoint), there's **no actual connection handshake** at the libfabric/network level. The "connection" is purely state tracking in the plugin.

#### Step 3: Accept - `ncclNet->accept()`

**Purpose**: Complete the connection setup on the receiver side.

```c
ncclResult_t nccl_net_ofi_accept(void* listenComm,
                                 void** recvComm)
{
  nccl_net_ofi_listen_comm *lcomm = (nccl_net_ofi_listen_comm *)listenComm;

  // For connectionless RDM (EFA):
  // No actual connection to "accept" in libfabric
  // Just transition state

  auto *comm = new nccl_net_ofi_recv_comm();  // C++ class, not C struct  // See struct nccl_net_ofi_recv_comm (https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi.h)

  // Copy endpoint and resources from listen comm
  comm->ep = lcomm->ep;
  comm->cq = lcomm->cq;

  // Initialize receive tracking
  comm->pending_recvs = 0;
  comm->recv_head = 0;
  comm->recv_tail = 0;

  // Allocate unexpected message buffer
  comm->unexp_msg_buf = malloc(UNEXP_BUF_SIZE);

  *recvComm = comm;

  return ncclSuccess;
}
```

**Important**: For EFA/RDM, accept is essentially a no-op at the libfabric level. It just creates the receive-side state structure.

### Connection State Diagram

```
Receiver (Passive)                    Sender (Active)
═══════════════════                   ══════════════

[INIT]                                [INIT]
  │                                     │
  │ listen()                            │
  ├─ Create EP                          │
  ├─ Get address                        │
  └─ Return handle                      │
  │                                     │
  ▼                                     │
[LISTENING]                             │
  │                                     │
  │ ←──── NCCL Bootstrap ───────────────┤ (Handle sent OOB)
  │                                     │
  │                                     │ connect(handle)
  │                                     ├─ Create EP
  │                                     ├─ Parse handle
  │                                     └─ Insert peer in AV
  │                                     │
  │                                     ▼
  │                                   [CONNECTED]
  │                                     │
  │ accept()                            │
  ├─ Create recv comm                   │
  └─ Ready for receives                 │
  │                                     │
  ▼                                     ▼
[CONNECTED]                           [CONNECTED]
  │                                     │
  │←────── fi_tsend/fi_trecv ──────────→│
  │                                     │
```

### What "Connection" Means for EFA/RDM

Since EFA uses **connectionless RDM** endpoints:

**No connection at network level:**
- No TCP-style 3-way handshake
- No dedicated circuit
- Endpoints can send to any peer at any time

**"Connection" is plugin state:**
- Stores peer's `fi_addr_t` (from AV insertion)
- Tracks outstanding requests
- Manages send/recv queues
- Associates NCCL channel with libfabric endpoint

**Analogy**:
```
TCP connection:     Phone call (dedicated line)
RDM "connection":   Knowing someone's phone number (can call anytime)
```

---

## Send/Recv Protocol

### Tagged Messaging Overview

The OFI plugin uses **tagged messages** to multiplex multiple NCCL channels over the same endpoint.

**Tag Structure** (plugin-specific, example):

(Note: The actual tag structure may vary by protocol implementation in aws-ofi-nccl)

**`struct nccl_ofi_tag`** - Tag encoding structure (conceptual example showing 64-bit tag layout):

```c
// Tag encoding (64-bit)
struct nccl_ofi_tag {
  uint32_t channel_id : 16;  // NCCL channel (0-15)
  uint32_t step_id    : 16;  // Algorithm step
  uint32_t msg_type   : 8;   // Data, control, etc.
  uint32_t reserved   : 24;
};

// Pack into uint64_t
uint64_t tag = (channel_id) | (step_id << 16) | (msg_type << 32);
```

### Send Protocol: `ncclNet->isend()`

**API:**
```c
ncclResult_t ncclNet->isend(void* sendComm, void* data,
                            int size, int tag,
                            void* mhandle, void** request);
```

**Detailed Flow:**

```c
ncclResult_t nccl_net_ofi_isend(void* sendComm, void* data,
                                int size, int tag,
                                void* mhandle, void** request)
{
  nccl_net_ofi_send_comm *comm = (nccl_net_ofi_send_comm *)sendComm;  // See struct nccl_net_ofi_send_comm (https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi.h)
  struct fid_mr* mr = mhandle;  // See struct fid_mr (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L131-L138)

  // === 1. Allocate Request ===
  nccl_net_ofi_req *req = alloc_request(comm);  // See struct nccl_net_ofi_req (https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi.h)
  req->comm = comm;
  req->size = size;
  req->state = REQ_PENDING;
  req->type = NCCL_OFI_SEND;

  // === 2. Get Memory Descriptor ===
  // Descriptor contains lkey for local access
  void* desc = fi_mr_desc(mr);  // See fi_mr_desc() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L147)

  // === 3. Prepare Tagged Send ===
  // Tag identifies which NCCL operation/channel this is
  uint64_t fi_tag = (uint64_t)tag;

  // === 4. Post Send to Libfabric ===
  ssize_t ret = fi_tsend(  // See fi_tsend() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_tagged.h#L121)
      comm->ep,               // Endpoint
      data,                   // Buffer to send
      size,                   // Size
      desc,                   // Memory descriptor (with lkey)
      comm->remote_addr,      // Peer's fi_addr_t (from AV)
      fi_tag,                 // Tag for matching
      req                     // Context (returned on completion)
  );

  // === 5. Handle Return ===
  if (ret == 0) {
    // Successfully posted
    comm->pending_sends++;
    req->state = REQ_POSTED;
  }
  else if (ret == -FI_EAGAIN) {
    // Queue full, need to retry
    // Either retry immediately or queue for later
    req->state = REQ_QUEUED;
    queue_request(comm, req);

    // Progress CQ to make space
    progress_sends(comm);

    // Retry
    ret = fi_tsend(comm->ep, data, size, desc,
                   comm->remote_addr, fi_tag, req);
  }
  else {
    // Error
    free_request(req);
    return ncclSystemError;
  }

  // === 6. Return Request Handle ===
  *request = req;

  return ncclSuccess;
}
```

**What Happens After `fi_tsend()`:**

```
User Space (OFI Plugin):
  fi_tsend() returns
  ↓
Libfabric (EFA Provider):
  Build WQE (Work Queue Entry)
  ├─ Source address: data
  ├─ Length: size
  ├─ lkey: from desc
  ├─ Destination: remote_addr
  ├─ Tag: fi_tag
  └─ Context: req
  ↓
  Post to Send Queue (SQ)
  ↓
  Ring doorbell (MMIO write)
  ↓
EFA Driver/Hardware:
  (async) Read WQE
  (async) DMA read from source buffer
  (async) Build network packet
      ┌────────────────────────────┐
      │ Header:                    │
      │  - Destination address     │
      │  - Tag: fi_tag             │
      │  - Length: size            │
      ├────────────────────────────┤
      │ Payload: [data...]         │
      └────────────────────────────┘
  (async) Send on network
  (async) Write completion to CQ
      ┌────────────────────────────┐
      │ CQE:                       │
      │  - Context: req            │
      │  - Status: SUCCESS         │
      │  - Length: size            │
      └────────────────────────────┘
```

### Receive Protocol: `ncclNet->irecv()`

**API:**
```c
ncclResult_t ncclNet->irecv(void* recvComm, int n,
                            void** data, int* sizes,
                            int* tags, void** mhandles,
                            void** request);
```

**Key Feature**: Can post **multiple receives** at once (batching).

**Detailed Flow:**

```c
ncclResult_t nccl_net_ofi_irecv(void* recvComm, int n,
                                void** data, int* sizes,
                                int* tags, void** mhandles,
                                void** request)
{
  nccl_net_ofi_recv_comm *comm = (nccl_net_ofi_recv_comm *)recvComm;  // See struct nccl_net_ofi_recv_comm (https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi.h)

  // === 1. Allocate Request for All Receives ===
  nccl_net_ofi_req *req = alloc_request(comm);
  req->comm = comm;
  req->type = NCCL_OFI_RECV;
  req->nrecvs = n;
  req->completed = 0;

  // === 2. Post Each Receive ===
  for (int i = 0; i < n; i++) {
    struct fid_mr* mr = mhandles[i];
    void* desc = fi_mr_desc(mr);
    uint64_t fi_tag = (uint64_t)tags[i];

    // === 3. Pre-Post Tagged Receive ===
    ssize_t ret = fi_trecv(  // See fi_trecv() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_tagged.h#L98)
        comm->ep,               // Endpoint
        data[i],                // Receive buffer
        sizes[i],               // Size
        desc,                   // Memory descriptor
        FI_ADDR_UNSPEC,         // Accept from any sender
        fi_tag,                 // Tag to match
        0,                      // Ignore mask (exact match)
        &req->sub_reqs[i]       // Sub-request context
    );

    if (ret == 0) {
      // Successfully posted
      req->sub_reqs[i].state = REQ_POSTED;
      comm->pending_recvs++;
    }
    else if (ret == -FI_EAGAIN) {
      // Queue full, retry
      progress_recvs(comm);
      i--;  // Retry this receive
    }
    else {
      // Error
      return ncclSystemError;
    }
  }

  // === 4. Return Request ===
  *request = req;

  return ncclSuccess;
}
```

**Pre-Posting Semantics:**

```
Receiver Posts fi_trecv()          Sender Posts fi_tsend()
─────────────────────────          ───────────────────────

Time T0:
  fi_trecv(tag=5, buf=A)
  ├─ Stored in RX queue
  └─ Waiting for matching send
                                   Time T1:
                                     fi_tsend(tag=5, data=X)
                                     ├─ Packet sent on network
                                     └─ Arrives at receiver

Time T2:
  Receiver NIC receives packet
  ├─ Tag=5 in packet
  ├─ Match with fi_trecv(tag=5)
  ├─ DMA write data to buf=A
  └─ Write completion to CQ

Time T3:
  fi_cq_read() → completion for fi_trecv()
```

**Unexpected Message Handling:**

If a send arrives before the matching receive is posted:

```
Option 1 (EFA Provider Default): Drop packet
  ├─ NAK sent to sender
  └─ Sender retransmits (hardware retry)

Option 2 (Plugin-Level Buffering): Buffer unexpected
  ├─ Store in unexpected message queue
  ├─ When fi_trecv() posted later, check queue
  └─ Match and deliver
```

### Tag Matching

**Exact Match (OFI Plugin Default):**
```c
fi_trecv(ep, buf, size, desc, addr, tag, ignore=0, ctx);
//                                         ↑
//                                    ignore=0 means exact match

// Matches only if: recv_tag == send_tag
```

**Range Match (if supported):**
```c
// Accept any tag in range [0x100, 0x1FF]
fi_trecv(ep, buf, size, desc, addr,
         tag=0x100,      // Base tag
         ignore=0xFF,    // Ignore lower 8 bits
         ctx);

// Matches if: (recv_tag & ~ignore) == (send_tag & ~ignore)
// Examples:
//   send_tag=0x100 → Match
//   send_tag=0x150 → Match
//   send_tag=0x1FF → Match
//   send_tag=0x200 → No match
```

**NCCL's Tag Usage:**
- Each channel has unique tag range
- Allows multiple channels to share endpoint
- Example: Channel 0 uses tags 0x0000-0x00FF, Channel 1 uses 0x0100-0x01FF

### Completion Protocol: `ncclNet->test()`

**API:**
```c
ncclResult_t ncclNet->test(void* request, int* done, int* size);
```

**Detailed Flow:**

```c
ncclResult_t nccl_net_ofi_test(void* request, int* done, int* size)
{
  struct nccl_ofi_req* req = request;
  struct nccl_ofi_comm* comm = req->comm;

  *done = 0;

  // === 1. Poll Completion Queue ===
  struct fi_cq_data_entry entries[CQ_POLL_BATCH];  // See struct fi_cq_data_entry (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L215-L220)
  int nread = fi_cq_read(comm->cq, entries, CQ_POLL_BATCH);  // See fi_cq_read() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L311)

  if (nread > 0) {
    // === 2. Process Completions ===
    for (int i = 0; i < nread; i++) {
      struct fi_cq_data_entry* entry = &entries[i];

      // Get request from context
      struct nccl_ofi_req* comp_req =
          (struct nccl_ofi_req*)entry->op_context;

      // === 3. Check if This is Our Request ===
      if (comp_req == req) {
        // Our request completed!
        *done = 1;
        *size = entry->len;

        // Update state
        req->state = REQ_COMPLETED;
        if (req->type == NCCL_OFI_SEND) {
          comm->pending_sends--;
        } else {
          comm->pending_recvs--;
        }

        // Free request
        free_request(req);

        return ncclSuccess;
      }
      else {
        // Someone else's completion, process it
        process_completion(comm, comp_req, entry);
      }
    }
  }
  else if (nread == -FI_EAGAIN) {
    // No completions ready
    *done = 0;
  }
  else {
    // Error
    struct fi_cq_err_entry err;  // See struct fi_cq_err_entry (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L233-L246)
    fi_cq_readerr(comm->cq, &err, 0);  // See fi_cq_readerr() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L348)

    // Handle error
    return ncclSystemError;
  }

  return ncclSuccess;
}
```

**Polling Loop (in NCCL Proxy Thread):**

```c
void nccl_proxy_thread() {
  while (running) {
    // Poll all pending requests
    for (req in pending_requests) {
      int done;
      ncclNet->test(req, &done, &size);

      if (done) {
        // Notify GPU that operation completed
        signal_completion_to_gpu(req);
        remove_from_pending(req);
      }
    }

    // Small yield if no work
    if (no_pending) {
      sched_yield();
    }
  }
}
```

---

## RDMA Write Protocol (Large Messages)

For large messages, the plugin may use RDMA write instead of send/recv.

### Rendezvous Protocol

```
Receiver                               Sender
────────                               ──────

1. Post receive (expecting large msg)
   fi_trecv(tag=X, ...)

                                       2. Want to send large message
                                          Decide: Use RDMA write

                                       3. Send RTS (Ready-to-Send)
   ←──────── fi_tsend(RTS) ────────────

   RTS contains:
   ├─ Message size
   ├─ Tag
   └─ Request ID

4. Receive RTS
   fi_cq_read() → RTS msg

5. Allocate/identify buffer
   Register if needed

6. Send CTS (Clear-to-Send)
   ────────→ fi_tsend(CTS) ─────────→

   CTS contains:
   ├─ Buffer address
   ├─ rkey
   └─ Request ID

                                       7. Receive CTS
                                          fi_cq_read() → CTS msg

                                       8. RDMA Write
                                          fi_write(data,  // See fi_write() (https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_rma.h#L111)
                                                   remote_addr,
                                                   remote_key)
                                          ─────────────────────→

8. Data arrives (DMA)
   (silent, no completion)

                                       9. Send RECEIPT
   ←──────── fi_tsend(RECEIPT) ────────

10. Receive RECEIPT
    fi_cq_read() → completion
    Signal to NCCL
```

**Message Types:**
- **RTS (Ready-to-Send)**: "I have large data to send"
- **CTS (Clear-to-Send)**: "Here's my buffer info, write to it"
- **DATA**: RDMA write (one-sided)
- **RECEIPT**: "I got your RDMA write"

---

## Summary

### Connection Establishment

**For EFA/RDM (Connectionless):**
1. **listen()**: Create EP, export address
2. **NCCL Bootstrap**: Exchange addresses out-of-band
3. **connect()**: Parse peer address, insert in AV
4. **accept()**: Create recv-side state
5. **Ready**: Can send/recv immediately

**No network handshake** - purely plugin state tracking.

### Send/Recv Protocol

**Send:**
1. `isend()` → `fi_tsend()` with tag
2. Posted to send queue
3. Hardware sends packet
4. Completion in sender CQ

**Receive:**
1. `irecv()` → `fi_trecv()` with tag (pre-posted)
2. Posted to receive queue
3. Incoming packet matched by tag
4. DMA to receive buffer
5. Completion in receiver CQ

**Completion:**
1. Proxy thread polls CQ via `test()`
2. `fi_cq_read()` returns completions
3. Match context to request
4. Signal to NCCL/GPU

### Key Characteristics

- **Tagged messaging**: Multiplexes channels
- **Pre-posting**: Receives posted before sends
- **Connectionless**: RDM, no connection state in libfabric
- **Async**: All operations non-blocking
- **Polling**: Manual progress via CQ polling
- **RDMA for large**: Rendezvous protocol for efficiency

### Protocol Stack Summary

```
NCCL Proxy Thread
    ↓ isend/irecv/test
OFI Plugin
    ├─ State tracking
    ├─ Tag management
    └─ Request management
    ↓ fi_tsend/fi_trecv/fi_cq_read
Libfabric EFA Provider
    ├─ WQE building
    ├─ Tag matching
    └─ Completion generation
    ↓ ibv_post_send/ibv_poll_cq
EFA Driver
    ├─ DMA operations
    ├─ Packet formation
    └─ Hardware interface
    ↓
EFA Hardware (SRD Protocol)
    ├─ Reliable delivery
    ├─ Out-of-order OK
    └─ Tag carried in packet
```

This protocol design allows NCCL to efficiently multiplex multiple channels over connectionless EFA transport with reliability and low overhead.

## Connection Manager (CM) Module

The connection establishment logic described above is now handled by a
dedicated Connection Manager module (`src/cm/`), which separates connection
state management from transport-specific logic.

**Source files:**
- `src/cm/nccl_ofi_cm.cpp` — Core CM logic
- `src/cm/nccl_ofi_cm_reqs.cpp` — CM request handling
- `src/cm/nccl_ofi_cm_resources.cpp` — CM resource management
- `include/cm/nccl_ofi_cm.h` — CM interface ([view](https://github.com/aws/aws-ofi-nccl/blob/master/include/cm/nccl_ofi_cm.h))

**Key classes:**
- `nccl_ofi_cm` — Creates connectors, receivers, and listeners
- `nccl_ofi_cm_send_connector` — Active side: sends connect request via `fi_tsend`, waits for response
- `nccl_ofi_cm_receiver` — Passive side: receives connect request via `fi_trecv`, sends response
- `nccl_ofi_cm_listener` — Listens for incoming connection requests

The CM uses libfabric tagged messaging for the handshake. Connect request
and response messages carry endpoint addresses and transport-specific
metadata (e.g., RDMA rail addresses, control mailbox info).

## GIN Protocol

GIN (Group Interconnect Network) provides one-sided put operations for
GPU-initiated networking workloads like DeepEP. It wraps the RDMA
transport and adds group communication (all-ranks-to-all-ranks).

**Source files:**
- `src/rdma/gin/nccl_ofi_gin_api.cpp` — GIN plugin API ([view](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_api.cpp))
- `src/rdma/gin/nccl_ofi_gin.cpp` — Bootstrap ring connection ([view](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin.cpp))
- `src/rdma/gin/nccl_ofi_gin_resources.cpp` — Resource management
- `src/rdma/gin/nccl_ofi_gin_reqs.cpp` — Request handling (iput, iputSignal)
- `src/rdma/gin/nccl_ofi_gin_allgather.cpp` — AllGather for handle exchange

### GIN Connection Establishment

Unlike standard Net connections (point-to-point), GIN establishes a
group communicator where all ranks can put data to all other ranks:

```
  1. listen(ctx, dev, handle, &listenComm)
     └── Creates RDMA endpoint, wraps in nccl_ofi_rdma_gin_listen_comm

  2. connect(ctx, handles[], nranks, rank, listenComm, &collComm)
     ├── Step 1: Bootstrap ring
     │   ├── RDMA connect to rank (rank+1) % nranks
     │   └── RDMA accept from rank (rank-1+nranks) % nranks
     │
     ├── Step 2: Create GIN resources on the endpoint
     │   └── nccl_ofi_gin_resources (ep_holder, gin_ep, freelists)
     │
     ├── Step 3: AllGather connection handles via bootstrap ring
     │   └── Each rank sends its GIN endpoint addresses to all others
     │
     └── Step 4: Return nccl_ofi_rdma_gin_put_comm
         └── Contains per-peer state for iput/iputSignal

  3. iput(collComm, srcOff, srcMh, size, dstOff, dstMh, rank, &req)
     └── fi_write() to remote rank's symmetric registered memory

  4. iputSignal(collComm, srcOff, srcMh, size, dstOff, dstMh,
               rank, signalOff, signalMh, value, op, &req)
     └── fi_write() for data + atomic signal operation on remote rank

  5. ginProgress(collComm)
     └── fi_cq_read() on all rails, process completions
```

### GIN vs Standard Net Protocol

| Aspect | Standard Net | GIN |
|--------|-------------|-----|
| Connection model | Point-to-point (pairs) | Group (all-to-all) |
| Data transfer | fi_tsend/fi_trecv or fi_write | fi_write only (one-sided) |
| Notification | Completion queue or tag match | Atomic signal increment |
| Bootstrap | CM handshake (2 messages) | Ring connect + AllGather |
| Use case | NCCL collectives | DeepEP MoE dispatch |
| API | ncclNet_v12_t | ncclGin_v13_t |
