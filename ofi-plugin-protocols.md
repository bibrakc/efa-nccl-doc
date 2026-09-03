# OFI NCCL Plugin: Connection Establishment and Send/Recv Protocols

> **Source lookups:** this document records mechanism, defaults, corrections and history. For current function bodies, call graphs and blast radius, query the code graph (`codegraph explore <symbol>`) rather than trusting a pasted copy here.

## Overview

This document provides detailed descriptions of the connection establishment mechanism and send/recv protocols used in the OFI NCCL plugin, including the actual message flows and state management.

### Exported plugin symbol sets

The plugin exports these versioned op-table symbols (the version *set* is the
fact; see [ofi-plugin.md](ofi-plugin.md) for the per-table contents):

- **Net**: `ncclNetPlugin_v4` .. `ncclNetPlugin_v12` (`v4` on the Neuron interface, up to `v12` on the NVIDIA interface).
- **GIN**: `ncclGinPlugin_v11`, `ncclGinPlugin_v13`, `ncclGinPlugin_v14` (**`v14` is GDAKI-only**, present iff `HAVE_GDAKI`).
- **RMA**: `ncclRmaPlugin_v14`, `ncclRmaPlugin_v15`.
- **Tuner**: `ncclTunerPlugin_v2`, `ncclTunerPlugin_v3`, `ncclTunerPlugin_v6`.

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

> Source: `src/nccl_ofi_net.cpp` (transport-specific in `nccl_ofi_sendrecv.cpp` / `nccl_ofi_rdma.cpp`) — `nccl_net_ofi_listen()`. Use `codegraph explore nccl_net_ofi_listen` for the current body.

The setup idiom is: create the endpoint (`fi_endpoint`), open a CQ
(`fi_cq_open`), bind CQ + address vector to the endpoint
(`fi_ep_bind(..., FI_TRANSMIT | FI_RECV)` and the AV), `fi_enable` it, read the
local address with `fi_getname`, and serialize that address into the opaque
`handle` NCCL will ship to the peer out-of-band.

**What gets stored in `handle`:**
- Endpoint address (libfabric address, e.g., `<ip>:<port>` equivalent for EFA)
- Address length
- Device ID
- Any plugin-specific metadata

**NCCL's role**: NCCL will send this handle to the sender through its bootstrap network (out-of-band, typically TCP).

#### Step 2: Connector (Active Side) - `ncclNet->connect()`

**Purpose**: Sender receives the receiver's handle, creates its own endpoint, and establishes "connection" (state tracking for connectionless RDM).

> Source: `src/nccl_ofi_net.cpp` (transport-specific in `nccl_ofi_sendrecv.cpp` / `nccl_ofi_rdma.cpp`) — `nccl_net_ofi_connect()`. Use `codegraph explore nccl_net_ofi_connect` for the current body.

The idiom mirrors `listen()` (create endpoint, open/share CQ, bind, enable) and
adds the one connectionless-RDM-specific step: **insert the peer's address into
the address vector** to obtain an `fi_addr_t` used as the destination on every
send:

```c
fi_av_insert(av, peer_addr, 1, &remote_fi_addr, 0, NULL);  // peer addr -> fi_addr_t
```

**Key Point**: For EFA (RDM endpoint), there's **no actual connection handshake** at the libfabric/network level. The "connection" is purely state tracking in the plugin.

#### Step 3: Accept - `ncclNet->accept()`

**Purpose**: Complete the connection setup on the receiver side.

> Source: `src/nccl_ofi_net.cpp` (transport-specific in `nccl_ofi_sendrecv.cpp` / `nccl_ofi_rdma.cpp`) — `nccl_net_ofi_accept()`. Use `codegraph explore nccl_net_ofi_accept` for the current body.

For connectionless RDM there is no libfabric-level connection to accept: the call
just creates the receive-side comm (carrying the endpoint/CQ from the listen
comm) and its receive-tracking state.

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

> Source: `src/nccl_ofi_sendrecv.cpp` / `src/nccl_ofi_rdma.cpp` — `nccl_net_ofi_isend()`. Use `codegraph explore nccl_net_ofi_isend` for the current body.

isend allocates a request, gets the MR descriptor (`fi_mr_desc(mr)` — carries the
lkey), and posts a tagged send. The `fi_tsend` argument order is the idiom worth
remembering — the request is passed as the context returned on completion:

```c
fi_tsend(comm->ep, data, size, desc, comm->remote_addr, fi_tag, /*context=*/req);
```

Return handling is the standard libfabric back-pressure idiom: `0` = posted;
**`-FI_EAGAIN`** = transmit queue full, so progress the CQ and retry (do not treat
as an error); anything else is a real error.

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

> Source: `src/nccl_ofi_sendrecv.cpp` / `src/nccl_ofi_rdma.cpp` — `nccl_net_ofi_irecv()`. Use `codegraph explore nccl_net_ofi_irecv` for the current body.

irecv allocates one request covering all `n` sub-receives and posts each with
`fi_trecv`. The request's direction is set to **`NCCL_OFI_SENDRECV_RECV`**
(`include/nccl_ofi_sendrecv.h`) — note the identifier is `NCCL_OFI_SENDRECV_RECV`,
**there is no `NCCL_OFI_RECV`**. Each `fi_trecv` accepts from any sender and matches
on tag with an exact-match ignore mask:

```c
fi_trecv(comm->ep, data[i], sizes[i], desc, FI_ADDR_UNSPEC,
         fi_tag, /*ignore=*/0, &req->sub_reqs[i]);
```

`-FI_EAGAIN` on a sub-receive means progress the CQ and retry that same index;
other non-zero returns are errors.

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

> Source: `src/nccl_ofi_sendrecv.cpp` / `src/nccl_ofi_rdma.cpp` — `nccl_net_ofi_test()`. Use `codegraph explore nccl_net_ofi_test` for the current body.

test polls the CQ and matches completions back to requests by `op_context`:

```c
int nread = fi_cq_read(comm->cq, entries, CQ_POLL_BATCH);
// per entry: req == entry->op_context  -> *done=1, *size=entry->len, free_request
// nread == -FI_EAGAIN -> no completions (*done=0)
// nread < 0           -> fi_cq_readerr(cq, &err, 0); return error
```

Completions belonging to *other* pending requests are processed in place (their
state advanced), not dropped.

**Polling Loop (in NCCL Proxy Thread):** the proxy thread repeatedly calls
`test()` on each pending request; when one reports `done`, it signals completion
to the GPU and drops it from the pending set, `sched_yield()`-ing when idle.

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
                                          fi_write(data,  // See fi_write() (https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_rma.h)
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

## Eager Protocol (Small Messages)

For small messages the rendezvous round-trip (control-message handshake before
any data moves) dominates latency. The RDMA transport therefore has an **eager**
path: messages at or below `EAGER_MAX_SIZE` are sent in a single operation that
carries the payload inline, with no prior control message from the receiver.

> **The eager path is DISABLED BY DEFAULT.** `OFI_NCCL_EAGER_MAX_SIZE`
> (env `EAGER_MAX_SIZE`) now defaults to **-1**, which disables eager entirely
> ([include/nccl_ofi_param.h:229](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_param.h),
> `OFI_NCCL_PARAM(int, eager_max_size, "EAGER_MAX_SIZE", -1)`; comment: "Default is
> disabled"). The default was flipped to disabled in master `7df78cd`
> ("param: Disable eager messages by default") and `708572b`
> ("rdma: Disable eager messages by default"), after `082120d` had briefly
> re-enabled it. To use eager, set `EAGER_MAX_SIZE` to a positive byte count
> (e.g. `8192`). The mechanics below apply **when eager is enabled**.

- Controlled by `OFI_NCCL_EAGER_MAX_SIZE` (env `EAGER_MAX_SIZE`); **default -1
  (disabled)**. Set to a positive value (e.g. 8192) to enable eager for messages at or
  below that size.
- The sender posts the payload (prefixed by an 8-byte eager header) with
  `fi_sendmsg(..., FI_REMOTE_CQ_DATA)` onto a single rail; the receiver consumes
  it from a pre-posted **eager receive (rx) buffer**, then copies it into the
  application destination.
- Control and eager rx buffers use **separate freelists** and separate posted-rx
  limits, so eager traffic cannot starve the control path.

### Eager correctness fixes

Several correctness fixes accompanied the eager work; they matter for anyone reading
older code or debugging eager (once re-enabled):

- **`1206868`** — fix eager send reporting size 0 when the ctrl msg arrives at a
  different `seq_num`.
- **`41c9c36`** — preserve eager queue matching.
- **`f50d60e`** — require full ctrl-msg arrival before consuming ctrl entries.
- **`9de49c8`** — move `msg_seq_num` to the *end* of the ctrl-msg entry (so a partially
  arrived ctrl msg is never treated as complete; pairs with `f50d60e`).

### Eager message header

Every eager send prepends an 8-byte header so the receiver can place the payload
correctly, including in a *grouped* (multi-) receive where several sub-receives
share one comm:

```c
// include/nccl_ofi_rdma.h
typedef struct nccl_ofi_eager_msg_header {
    uint8_t  eager_offset;       // position of this msg within the eager batch
    uint8_t  prev_batch_count;   // size of the previous batch (valid when eager_offset == 0)
    uint16_t eager_seq;          // per-batch sequence number (chain identity)
    int32_t  tag;                // sub-receive tag within a grouped receive
} nccl_ofi_eager_msg_header_t;   // NCCL_OFI_EAGER_HEADER_SIZE == 8
```

The receiver drains its eager queue in batch order via `drain_recv_eager_queue()`,
using `eager_offset`/`prev_batch_count` to reassemble a batch and `tag` to map each
sub-message to the right destination buffer.

### Sequence-number wrap safety (`eager_seq`)

Eager batches are chained so the receiver can detect ordering and gaps. An earlier
implementation chained them using the per-comm `msg_seq_num`, which is **10-bit
(wraps every 1024)** and is advanced by *all* traffic — including rendezvous/RDMA-write
messages. Because the control mailbox has only 256 slots, sequence numbers alias
4-to-1 per slot. Under a long run of write traffic the shared counter could wrap
while the eager chain trackers were idle, making two eager batches a full wrap apart
indistinguishable. The receiver's drain then wedged and the collective deadlocked —
observed in practice at the RING LL→LL128 transition on 16-node all-gather.

The fix (master commit `1a70558`, shipped in v1.20.0) introduces a **dedicated
`eager_seq`** carried in the header:

- It is advanced **only** for eager batches, never by rendezvous traffic, so write
  volume can no longer overrun it.
- It is 16-bit and the number of in-flight eager batches is bounded
  (`NCCL_OFI_MAX_EAGER_PENDING`, far below 65536), so two live batches can never
  share an `eager_seq` — wrap-safe by construction.
- `msg_seq_num` is still used for the receive-side message-buffer lookup (already
  window-bounded); `eager_seq` is the sole identity used to chain eager batches.

### Eager vs Rendezvous

| | Eager | Rendezvous |
|---|---|---|
| Message size | ≤ `EAGER_MAX_SIZE` (default -1 = **disabled**; e.g. 8 KB when enabled) | larger messages |
| Control handshake | none (payload sent inline) | control message before data |
| Network op | `fi_sendmsg` (+`FI_REMOTE_CQ_DATA`) | control msg + `fi_write` |
| Round trips | one-way | round trip before data |
| Optimizes for | latency (small msgs) | bandwidth (large msgs) |

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
- `src/rdma/gin/nccl_ofi_gin_gdaki.cpp` — GDAKI (kernel-initiated) plugin ([view](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_gdaki.cpp))
- `src/rdma/gin/nccl_ofi_gin_gdaki_resources.cpp` — GDAKI device-handle/queue resources

### GIN Modes: Proxy vs GDAKI

GIN has two data-path modes: **proxy** (a CPU proxy thread issues the underlying
libfabric writes) and **GDAKI / EFA-GDA** (the GPU kernel drives the EFA queues
directly). The plugin exports separate op tables for each — proxy via
`ncclGinPlugin_v11`/`v13` and `ncclRmaPlugin_v14`/`v15`, GDAKI via `ncclGinPlugin_v14`
(see [ofi-plugin.md](ofi-plugin.md) for the full symbol table and the GIN-vs-RMA split).

> **`OFI_NCCL_GIN_TYPE` has been REMOVED.** The old plugin env var
> `OFI_NCCL_GIN_TYPE` (env `GIN_TYPE`) that selected PROXY vs GDAKI **no longer exists**
> — removed in master `80f2c78` ("gin: enable GDAKI automatically, remove
> OFI_NCCL_GIN_TYPE"). There is no such knob in `include/nccl_ofi_param.h` anymore, and
> there is no `--enable-gdaki` configure flag. Do not treat `OFI_NCCL_GIN_TYPE` as a
> live setting.

**Current selection logic.** Mode selection is now split between the plugin (which
capability is *available*) and NCCL (which one is *used*):

1. **Plugin build:** GDAKI is compiled in only when `HAVE_GDAKI` holds (libfabric ≥ 2.5
   `FI_EFA_GDA_OPS` + hardware-counter ABI + CUDA device interface — see `configure.ac`).
   When compiled in, the plugin exports `ncclGinPlugin_v14` (`name = "Libfabric_GDAKI"`).
2. **Plugin runtime capability:** `nccl_ofi_gin_gdaki_capable()`
   ([include/rdma/gin/nccl_ofi_gin_gdaki.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/rdma/gin/nccl_ofi_gin_gdaki.h))
   returns true only if GDAKI is compiled in **and** the runtime satisfies: libfabric
   ≥ 2.5, DMA-BUF viable, and an `efa`-family provider. GDRCopy 2.5+ (forced PCIe copy)
   is additionally required and checked in `nccl_ofi_gin_init`.
3. **NCCL-side choice:** NCCL binds the highest op-table version it understands. Per
   `doc/gin-getting-started.md`, **NCCL 2.31 defaults to proxy for EFA**; the
   application opts into EFA-GDA with the **NCCL** env var `NCCL_GIN_TYPE=5` (plus
   `NCCL_SYM_GIN_KERNELS_ENABLE=0`). `NCCL_GIN_TYPE` here is an *NCCL* variable, not the
   removed plugin `OFI_NCCL_GIN_TYPE`.

**Fallback behaviour.** GDAKI does not silently degrade to proxy inside the plugin: if a
requested GDAKI capability (e.g. the EFA hardware completion counter, gated by
`OFI_NCCL_GDAKI_EFA_HW_COUNTER=AUTO/ON/OFF`) cannot be satisfied, GDAKI context creation
**fails** rather than falling back. Falling back to proxy is an application decision —
per the getting-started guide, "if an application needs functionality that is not
currently supported by EFA-GDA, it can use the proxy mode as a fallback." EFA-GDA also
does not support strong signals or VA signals, so the app must set
`ginStrongSignalsRequired = false` and `ginVaSignalsRequired = false`.

> **Removed knob:** `OFI_NCCL_GIN_STRONG_SIGNAL` (env `GIN_STRONG_SIGNAL`) and the
> weak-signal mode were **removed** in master `aa80b54` ("gin: remove GIN_STRONG_SIGNAL
> env variable and weak-signal mode"). The plugin's signals are now **always strong**;
> there is no such param in `include/nccl_ofi_param.h`. (On the RMA op-table, the v14
> `iputSignal` still carries a trailing `isStrongSignal` argument for NCCL ABI
> compatibility, but the plugin ignores it — see `nccl_ofi_rma_iputSignal_v14` in
> [nccl_ofi_gin_api.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_api.cpp).)

- **PROXY** (NCCL default on EFA): the GPU kernel enqueues put/putSignal operations and
  the **CPU proxy thread** issues the underlying libfabric `fi_write`/`fi_writedata`
  operations on the GPU's behalf. Works on any EFA-capable build. Proxy properties
  advertise `netDeviceType = NCCL_NET_DEVICE_GIN_PROXY`.
- **GDAKI / EFA-GDA** (opt-in via NCCL `NCCL_GIN_TYPE=5`, requires `HAVE_GDAKI` build +
  runtime prerequisites): **kernel-initiated** networking. The GPU kernel drives the EFA
  queues directly — no CPU proxy on the data path. `createContext` builds a device
  handle in **GPU memory** describing the send queue (SQ), completion queue (CQ), and
  receive queue (RQ); the kernel posts WQEs and polls CQEs using EFA's
  **ownership/phase-bit protocol**. The on-GPU queue layout is compatible with the
  `efa_cuda_qp` / `efa_cuda_cq` types from `efa-dp-direct` (vendored under
  `3rd-party/efa-gda/`).

| | Proxy GIN | GDAKI GIN (EFA-GDA) |
|---|---|---|
| Data-path driver | CPU proxy thread (`fi_write`) | GPU kernel (direct EFA queue access) |
| Op table(s) exported | `ncclGin_v11/v13`, `ncclRma_v14/v15` | `ncclGin_v14` (`Libfabric_GDAKI`) |
| Build requirement | standard EFA build | `HAVE_GDAKI` (libfabric ≥ 2.5 GDA ops + hw counter + CUDA) |
| Runtime prerequisites | EFA + GDRCopy 2.5+ | CUDA + DMA-BUF + libfabric ≥ 2.5 + GDRCopy 2.5+ + `efa` provider; AWS: P5en/P6-B200/P6-B300, EFA driver 3.3.0+, rdma-core 64.0amzn0+, NVIDIA `PeerMappingOverride=1` |
| Completion detection | libfabric CQ | EFA hardware completion counter / phase-bit CQ polling in GPU memory |
| Selected by | NCCL default on EFA | NCCL `NCCL_GIN_TYPE=5` (not a plugin env var) |
| Signals | strong + VA supported | indexed only; strong/VA **not** supported |

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
| API | ncclNet_v4..v12 | ncclGin_v11/v13 + ncclGin_v14 (GDAKI) + ncclRma_v14/v15 (host proxy) |
