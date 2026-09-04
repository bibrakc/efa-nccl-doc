# EFA Hardware Architecture: Queue Pairs, Completion Queues, and Memory Layout

> **Source lookups:** this document records mechanism, defaults, corrections and history. For current function bodies, call graphs and blast radius, query the code graph (`codegraph explore <symbol>`) rather than trusting a pasted copy here. The struct excerpts below are kept deliberately because the **byte/bitfield layout is the fact** (VALUE), not to mirror the source text.

## Overview

AWS Elastic Fabric Adapter (EFA) is a custom network interface designed for HPC and ML workloads, implementing RDMA-like semantics with kernel bypass. This document describes the low-level hardware queue architecture, memory layout, and programming interface.

**Key Components**:
- **Queue Pairs (QP)**: Send Queue (SQ) + Receive Queue (RQ)
- **Completion Queues (CQ)**: Asynchronous completion notification
- **Work Queue Entries (WQE)**: Commands submitted to hardware
- **Completion Descriptors (CQE)**: Results returned by hardware
- **Address Handles (AH)**: Destination addressing for SRD transport

**Source Locations** (versions: EFA kernel driver **r3.3.0** / amzn-drivers `master`;
libfabric **2.7.0rc1** `main`; rdma-core **65.0-dev** `master`, latest release v64.0):
- Kernel driver: [amzn-drivers/kernel/linux/efa/src/](https://github.com/amzn/amzn-drivers/tree/master/kernel/linux/efa/src)
- Key headers: [efa_io_defs.h](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h), [efa_verbs.h](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_verbs.h)
- Libfabric provider: [libfabric/prov/efa/](https://github.com/ofiwg/libfabric/tree/main/prov/efa)
- Libfabric Data Path Direct: [efa_data_path_direct*.{c,h}](https://github.com/ofiwg/libfabric/tree/main/prov/efa/src) (see [rdma-core-and-verbs.md](rdma-core-and-verbs.md))

> **`efa_io_defs.h` is duplicated across three repos.** The same device I/O
> definitions (`struct efa_io_*`, the enums and bit layouts below) are copied
> into **rdma-core**, the **EFA kernel driver** (amzn-drivers), and **libfabric**.
> They are meant to stay byte-identical; when reading one, expect the same layout
> in the others. This is a fact about *where the definitions live*, not code the
> graph can dedupe for you.

## Queue Pair Architecture

### Queue Pair Types

EFA supports multiple queue pair types.

**`enum efa_io_queue_type`** ([efa_io_defs.h:15-20](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h)):

```c
enum efa_io_queue_type {
	/* send queue (of a QP) */
	EFA_IO_SEND_QUEUE = 1,
	/* recv queue (of a QP) */
	EFA_IO_RECV_QUEUE = 2,
};
```

**Transport Types**:
- **SRD (Scalable Reliable Datagram)**: Primary transport for EFA, defined as `EFA_QPT_SRD` in [efa_verbs.h:36](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_verbs.h)
- Similar semantics to UD (Unreliable Datagram) but with reliability
- Out-of-order delivery without segmentation
- Address Handle (AH) provided with each work request

### Send Queue (SQ)

The Send Queue contains work requests (WQEs) posted by software and consumed by hardware.

**Operations Supported**:

**`enum efa_io_send_op_type`** ([efa_io_defs.h:22-33](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h)):

```c
enum efa_io_send_op_type {
	/* send message */
	EFA_IO_SEND = 0,
	/* RDMA read */
	EFA_IO_RDMA_READ = 1,
	/* RDMA write */
	EFA_IO_RDMA_WRITE = 2,
	/* Fast MR registration */
	EFA_IO_FAST_REG = 3,
	/* Fast MR invalidation */
	EFA_IO_FAST_INV = 4,
};
```

**Key Characteristics**:
- Ring buffer of WQEs in host memory
- Hardware reads WQEs via DMA
- Doorbell write to notify hardware of new entries
- Producer index managed by software, consumer index by hardware

### Receive Queue (RQ)

The Receive Queue contains pre-posted receive buffers for incoming messages.

**Characteristics**:
- Pre-posted buffers before messages arrive
- Hardware writes incoming data directly to buffers
- SRD allows out-of-order delivery (unlike RC)
- Buffers consumed as messages arrive

## Work Queue Entry (WQE) Format

### TX WQE Structure

**`struct efa_io_tx_wqe`** (64-byte WQE) ([efa_io_defs.h](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h), lines ~236-255):

```c
struct efa_io_tx_wqe {
	/* TX meta */
	struct efa_io_tx_meta_desc meta;

	union {
		/* Send buffer descriptors */
		struct efa_io_tx_buf_desc sgl[2];

		u8 inline_data[32];

		/* RDMA local and remote memory addresses */
		struct efa_io_rdma_req rdma_req;

		/* Fast registration */
		struct efa_io_fast_mr_reg_req reg_mr_req;

		/* Fast invalidation */
		struct efa_io_fast_mr_inv_req inv_mr_req;
	} data;
};
```

**Size**: 64 bytes (meta descriptor + data union).

### 128-byte (wide) TX WQE

**New in EFA kernel driver r3.3.0 / libfabric 2.7.0:** the device can advertise a
**128-byte send-queue WQE** format, and the libfabric data-path-direct code onboarded
it. The wider WQE mainly enlarges the inline-data capacity and the RDMA inline path.

**`struct efa_io_tx_wqe_128`** ([efa_io_defs.h](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h), lines ~187-201):

```c
struct efa_io_tx_wqe_128 {
	struct efa_io_tx_meta_desc meta;
	union {
		struct efa_io_tx_buf_desc sgl[2];
		uint8_t inline_data[80];              /* 32 -> 80 bytes inline */
		struct efa_io_rdma_req_128 rdma_req;  /* supports inline RDMA WRITE */
	} data;
};
```

- Inline message capacity grows from **32 bytes** (64B WQE) to **80 bytes** (128B WQE).
- `struct efa_io_rdma_req_128` adds an 80-byte inline buffer so **inline RDMA WRITE**
  is possible (kernel r3.3.0 "inline WRITE operation support"). The libfabric SQ
  entry stride (`wq.wqe_size`) is queried from the device via `efadv_query_qp_wqs()`
  and is either 64 or 128; the direct data path copies WQEs with `mmio_memcpy_x64()`
  using that runtime size. The kernel selects the SQ WQE size in
  `efa_calc_sq_wqe_size_kernel()` and enables inline write when `wqe_size > 64`
  ([efa_verbs.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_verbs.c) lines ~938-1164).

### TX Metadata Descriptor

**`struct efa_io_tx_meta_desc`** ([efa_io_defs.h](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h), meta descriptor):

```c
struct efa_io_tx_meta_desc {
	/* Verbs-generated Request ID (low 16 bits) */
	u16 req_id;

	/*
	 * control flags
	 * 3:0 : op_type - enum efa_io_send_op_type
	 * 4 : has_imm - immediate_data field carries valid data
	 * 5 : inline_msg - inline message data follows
	 * 6 : meta_extension - Extended metadata. MBZ
	 * 7 : meta_desc - Indicates metadata descriptor. Must be set.
	 */
	u8 ctrl1;

	/*
	 * control flags
	 * 0 : phase - Phase bit for ring wrap detection
	 * 2 : first - Indicates first descriptor in transaction
	 * 3 : last - Indicates last descriptor in transaction
	 * 4 : comp_req - Request completion notification
	 */
	u8 ctrl2;

	u16 dest_qp_num;      // Destination QP number
	u16 length;           // SGL length (num buffers) or inline data size
	u32 immediate_data;   // Immediate data (if has_imm set)
	u16 ah;               // Address Handle index

	/*
	 * control flags
	 * 1:0 : processing_hints - enum efa_io_processing_hint
	 * 7:2 : reserved - MBZ
	 */
	u8 ctrl3;
	u8 reserved;

	u32 qkey;             // Queue key for datagram QPs
	u8 reserved2[6];

	/* Upper 48 bits of a 64-bit request ID (see below) */
	struct efa_io_req_id_ex req_id_ex;
};
```

**Key Fields**:
- `req_id`: Software-generated ID for matching completions (low 16 bits)
- `req_id_ex`: three additional 16-bit words holding the upper 48 bits of a
  **64-bit request ID** when the QP is created with the 64-bit request-ID
  capability (`EFADV_WQ_CAPS_64_BIT_REQ_ID`, kernel r3.3.0). In the classic
  16-bit scheme `req_id` instead carries a wrid-pool index OR-ed with the QP
  generation.
- `op_type`: Operation type (send, RDMA read/write, etc.)
- `phase`: Ring buffer wrap-around detection
- `comp_req`: Whether to generate a completion event
- `dest_qp_num`: Remote QP number (SRD transport)
- `ah`: Address Handle for destination addressing
- `ctrl3.processing_hints`: `EFA_IO_PROCESSING_HINT_BURST_PPS_SENSITIVE` set by the
  provider for the `FI_EFA_WR_HIGH_PPS` hint on RDMA WRITE
- `qkey`: Protection key for datagram operations

**`struct efa_io_req_id_ex`** — three 16-bit words (`w[0..2]`) forming, together
with the 16-bit `req_id`, the full 64-bit request ID
(`efa_set_req_id_64()` / `efa_get_req_id_64()` in the libfabric data-path-direct code).

### TX Buffer Descriptor

**`struct efa_io_tx_buf_desc`** ([efa_io_defs.h:136-151](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h)):

```c
struct efa_io_tx_buf_desc {
	/* length in bytes */
	u32 length;

	/*
	 * 23:0 : lkey - local memory translation key
	 * 31:24 : reserved - MBZ
	 */
	u32 lkey;

	/* Buffer address bits[31:0] */
	u32 buf_addr_lo;

	/* Buffer address bits[63:32] */
	u32 buf_addr_hi;
};
```

**Capabilities**:
- Up to 2 SGL entries per WQE (see `EFA_IO_TX_DESC_NUM_BUFS = 2`)
- 64-bit physical addresses split into low/high 32-bit words
- 24-bit lkey for memory protection
- Inline data mode: Up to 32 bytes inline on a 64-byte WQE, or up to 80 bytes on a
  128-byte (wide) WQE (`EFA_IO_TX_DESC_INLINE_MAX_SIZE`)

### RX Descriptor

**`struct efa_io_rx_desc`** ([efa_io_defs.h:261-282](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h)):

```c
struct efa_io_rx_desc {
	/* Buffer address bits[31:0] */
	u32 buf_addr_lo;

	/* Buffer Pointer[63:32] */
	u32 buf_addr_hi;

	/* Verbs-generated request id. */
	u16 req_id;

	/* Length in bytes. */
	u16 length;

	/*
	 * LKey and control flags
	 * 23:0 : lkey
	 * 29:24 : reserved - MBZ
	 * 30 : first - Indicates first descriptor in WQE
	 * 31 : last - Indicates last descriptor in WQE
	 */
	u32 lkey_ctrl;
};
```

**Characteristics**:
- 64-bit buffer address (DMA target for incoming data)
- `req_id`: Matched in completion descriptor
- `length`: Buffer size
- `lkey`: Memory region protection key
- `first`/`last`: Multi-descriptor WQE support

## Completion Queue Architecture

### Completion Descriptors

Completion events are written by hardware to the Completion Queue (CQ).

**`struct efa_io_cdesc_common`** ([efa_io_defs.h:285-308](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h)):

```c
struct efa_io_cdesc_common {
	/*
	 * verbs-generated request ID, as provided in the completed tx or rx
	 * descriptor.
	 */
	u16 req_id;

	u8 status;   // Completion status (success or error code)

	/*
	 * flags
	 * 0 : phase - Phase bit
	 * 2:1 : q_type - enum efa_io_queue_type: send/recv
	 * 3 : has_imm - indicates that immediate data is present (RX only)
	 * 6:4 : op_type - enum efa_io_send_op_type
	 * 7 : unsolicited - no matching request (RDMA with imm. RX only)
	 */
	u8 flags;

	/* local QP number */
	u16 qp_num;
};
```

**`struct efa_io_tx_cdesc`** — just the common completion header
(`efa_io_cdesc_common`) plus a reserved 16-bit word; a TX completion carries no
extra payload beyond `req_id`/`status`/`flags`/`qp_num`.
> Source: `amzn-drivers/kernel/linux/efa/src/efa_io_defs.h` — `struct efa_io_tx_cdesc`. Use `codegraph explore efa_io_tx_cdesc` for the current definition.

**`struct efa_io_rx_cdesc`** ([efa_io_defs.h:320-334](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h)):

```c
struct efa_io_rx_cdesc {
	/* Common completion info */
	struct efa_io_cdesc_common common;

	/* Transferred length bits[15:0] */
	u16 length;

	/* Remote Address Handle FW index, 0xFFFF indicates invalid ah */
	u16 ah;

	u16 src_qp_num;  // Source QP number

	/* Immediate data */
	u32 imm;
};
```

**Key Information**:
- `req_id`: Matches original WQE request ID
- `status`: Success or specific error code (see Completion Status)
- `phase`: Ring buffer ownership bit (toggles on wrap)
- `length`: Actual bytes transferred (RX only)
- `src_qp_num`: Source QP for received messages
- `ah`: Address handle for source (RX only)
- `imm`: Immediate data if present

### Completion Status Codes

**`enum efa_io_comp_status`** ([efa_io_defs.h:35-68](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h)):

```c
enum efa_io_comp_status {
	EFA_IO_COMP_STATUS_OK = 0,                                    // Success
	EFA_IO_COMP_STATUS_FLUSHED = 1,                              // Flushed during QP destroy
	EFA_IO_COMP_STATUS_LOCAL_ERROR_QP_INTERNAL_ERROR = 2,        // Internal QP error
	EFA_IO_COMP_STATUS_LOCAL_ERROR_UNSUPPORTED_OP = 3,           // Unsupported operation
	EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_AH = 4,               // Bad AH
	EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY = 5,             // LKEY error
	EFA_IO_COMP_STATUS_LOCAL_ERROR_BAD_LENGTH = 6,               // Message too long
	EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS = 7,             // RKEY error
	EFA_IO_COMP_STATUS_REMOTE_ERROR_ABORT = 8,                   // Connection reset
	EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_DEST_QPN = 9,            // Bad dest QP
	EFA_IO_COMP_STATUS_REMOTE_ERROR_RNR = 10,                    // Receiver not ready
	EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_LENGTH = 11,             // Receiver SGL too short
	EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_STATUS = 12,             // Unexpected status
	EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE = 13,           // Unresponsive remote
	EFA_IO_COMP_STATUS_REMOTE_ERROR_UNKNOWN_PEER = 14,           // No valid AH at remote
	EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE = 15,          // Unreachable remote
};
```

**SRD-Specific Errors**:
- `BAD_DEST_QPN`: Destination QP doesn't exist or is in error state
- `RNR`: Receiver Not Ready - no posted receive WQEs (no automatic retry)
- `UNRESP_REMOTE`: Transport timeout - previously responsive destination now silent
- `UNREACH_REMOTE`: Never received response from destination

## Memory Layout and Access Patterns

### Ring Buffer Management

All queues (SQ, RQ, CQ) use **ring buffer** structure:

```
Memory Layout (Software View):
┌──────────────────────────────────────────────────┐
│ Entry 0 │ Entry 1 │ ... │ Entry N-1 │ Entry 0 │  (wraps)
└──────────────────────────────────────────────────┘
    ↑                           ↑
  Consumer (HW)              Producer (SW)

Phase Bit Tracking:
- Each entry has a phase bit
- Phase toggles on each ring wrap (0 → 1 → 0)
- Software knows entry is consumed when phase changes
```

**Producer-Consumer Model**:
- **Send Queue**: SW producer, HW consumer
- **Receive Queue**: SW producer (posts buffers), HW consumer (fills buffers)
- **Completion Queue**: HW producer, SW consumer

**Doorbell Mechanism**:
- After posting WQEs, software writes to doorbell register
- Doorbell write = MMIO write to device BAR
- Notifies hardware of new entries to process
- **No syscall required** - direct user-space access

### Memory Registration

All buffers must be registered before use:

**`struct efa_io_fast_mr_reg_req`** ([efa_io_defs.h:175-222](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h)):

```c
struct efa_io_fast_mr_reg_req {
	/* Updated local key of the MR after lkey/rkey increment */
	u32 lkey;

	/*
	 * permissions
	 * 0 : local_write_enable - Local write (RQ buffers, RDMA Read)
	 * 1 : remote_write_enable - Remote write permissions
	 * 2 : remote_read_enable - Remote read permissions
	 */
	u8 permissions;

	/*
	 * control flags
	 * 4:0 : phys_page_size_shift - page size is (1 << shift)
	 * 6:5 : pbl_mode - enum efa_io_frwr_pbl_mode
	 */
	u8 flags;

	u8 reserved[2];

	/* IO Virtual Address associated with this MR */
	u64 iova;

	/* length in bytes */
	u64 length;

	/* PBL (page list) address */
	u64 pbl_addr;

	/* number of pages in PBL */
	u16 pbl_size;
};
```

**PBL Modes**:

**`enum efa_io_frwr_pbl_mode`** ([efa_io_defs.h:70-73](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h)):
- `EFA_IO_FRWR_INLINE_PBL`: Page list inline in descriptor (small regions)
- `EFA_IO_FRWR_DIRECT_PBL`: Page list in separate buffer (large regions)

**lkey/rkey Format** ([efa_verbs.h:38-47](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_verbs.h)):

```c
#define EFA_MR_GEN_SHIFT 20
#define EFA_MR_GEN_MASK 0x00F00000

static inline void efa_inc_fast_reg_key_gen(struct ib_mr *mr)
{
	u32 mr_gen = (mr->rkey + (1 << EFA_MR_GEN_SHIFT)) & EFA_MR_GEN_MASK;

	mr->lkey = (mr->lkey & ~EFA_MR_GEN_MASK) | mr_gen;
	mr->rkey = (mr->rkey & ~EFA_MR_GEN_MASK) | mr_gen;
}
```

**Key Structure** (32 bits):
- Bits 19:0 - MR index
- Bits 23:20 - Generation count (incremented on re-registration)
- Bits 31:24 - Additional flags/index bits

### RDMA Operations

**`struct efa_io_rdma_req`** — an RDMA request is just a remote address
(`efa_io_remote_mem_addr`, below) plus one local buffer descriptor
(`efa_io_tx_buf_desc local_mem[1]`).
> Source: `amzn-drivers/kernel/linux/efa/src/efa_io_defs.h` — `struct efa_io_rdma_req`. Use `codegraph explore efa_io_rdma_req` for the current definition.

**`struct efa_io_remote_mem_addr`** ([efa_io_defs.h:153-165](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h)):

```c
struct efa_io_remote_mem_addr {
	/* length in bytes */
	u32 length;

	/* remote memory translation key */
	u32 rkey;

	/* Buffer address bits[31:0] */
	u32 buf_addr_lo;

	/* Buffer address bits[63:32] */
	u32 buf_addr_hi;
};
```

**RDMA Capabilities**:
- RDMA Read: Fetch data from remote memory
- RDMA Write: Write data to remote memory
- RDMA Write with Immediate: Write + send immediate data to CQ
- Requires valid rkey at destination
- EFA Gen 3+ supports native RDMA (p5 instances), Gen 2 emulates via send/recv (p4d)

## SRD Transport Integration

### SRD Work Request Structure

SRD work requests extend the verbs WR with **per-WR destination addressing** (the
distinctive SRD trait):

- **`struct ib_srd_wr`** = `ib_send_wr wr` + `ib_ah *ah` + `u32 remote_qpn` +
  `u32 remote_qkey`.
- **`struct ib_srd_rdma_wr`** = `ib_srd_wr wr` + `u64 remote_addr` + `u32 rkey`.

> Source: `amzn-drivers/kernel/linux/efa/src/efa_verbs.h` — `struct ib_srd_wr`, `struct ib_srd_rdma_wr`. Use `codegraph explore ib_srd_wr` for the current definitions.

**Key Differences from RC**:
- Address Handle specified per WQE (not per QP)
- Supports out-of-order delivery
- Multiple outstanding messages without head-of-line blocking
- Implicit SRD context management (no user-visible EE contexts)

### Address Handles

**Purpose**: Encapsulate destination addressing information

**Contents** (implied, not exposed in header):
- Destination MAC address
- Destination GID (Global Identifier)
- SL (Service Level), traffic class
- Path MTU
- SRD context association

**Usage**:
- Created via `ibv_create_ah()` / `fi_av_insert()`
- Stored in hardware table
- 16-bit index referenced in TX WQE (`ah` field)
- Hardware looks up full addressing info from index

## Queue Sizing and Limits

### Typical Queue Depths

From libfabric EFA provider defaults and kernel driver capabilities:

**Send Queue**:
- Default depth: 512 entries (libfabric)
- Maximum: 16384 entries (hardware limit)
- Recommendation: 128-1024 for most workloads

**Receive Queue**:
- Default depth: 512 entries
- Maximum: 16384 entries
- Must have posted RX buffers before messages arrive (or RNR error)

**Completion Queue**:
- Default depth: 1024 entries
- Maximum: 262144 entries
- Shared across multiple QPs typically
- Must be >= sum of associated SQ/RQ depths

### Resource Limits (per device)

**EFA Device Limits** (varies by instance type):
- Max QPs: 1024-8192 (p4d: ~2K, p5: ~8K)
- Max CQs: 1024-8192
- Max MRs: 1024-8192
- Max AHs: 16384
- Max QP WR: 16384 (per SQ/RQ)
- Max CQ entries: 262144
- Max SGL entries per WR: 2 (send), 1 (recv) for most operations
- Max inline data: 32 bytes (64-byte WQE) or **80 bytes (128-byte WQE)** on devices
  advertising the wide-WQE capability (r3.3.0)

**Memory Consumption** (estimated):
- SQ: 64 bytes/entry × depth
- RQ: 32 bytes/entry × depth
- CQ: **16 bytes/entry**, both TX and RX (see the completion-descriptor sizes below)

Example for 512-entry QP + 1024-entry CQ:
- SQ: 512 × 64 = 32 KB
- RQ: 512 × 32 = 16 KB
- CQ: 1024 × 16 = 16 KB
- **Total: ~64 KB per QP**

For 1000 QPs: ~64 MB of queue memory

**Completion-descriptor sizes**, summed from the struct definitions in
[amzn-drivers kernel/linux/efa/src/efa_io_defs.h](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h):

| Descriptor | Fields | Size |
| --- | --- | --- |
| `efa_io_cdesc_common` | `u16 req_id` + `u8 status` + `u8 flags` + `u16 qp_num` | **6 B** |
| `efa_io_tx_cdesc` | common (6) + `efa_io_req_id_ex` (`u16 w[3]` = 6) + `u8 reserved[4]` | **16 B** |
| `efa_io_rx_cdesc` | common (6) + `u16 length` + `u16 ah` + `u16 src_qp_num` + `u32 imm` | **16 B** |
| `efa_io_rx_cdesc_ex` | `efa_io_rx_cdesc` (16) + union { `rdma_write` \| `u8 src_addr[16]` } | **32 B** |

Note that `efa_io_tx_cdesc` is **16 bytes, not 8**. Older descriptions of this structure omit
the `req_id_ex` field and the 4-byte reserved tail; both are present in the current header. The
extended RX descriptor is used only when the source AH is unknown (`0xFFFF`) and the CQ has
`set_src_addr` enabled, so 16 B is the size to budget with for ordinary traffic.

## Programming Model

> **Who writes these descriptors.** By default (libfabric 2.3.0+, RDM-enabled in
> 2.7.0) the EFA provider's **Data Path Direct** code builds and posts these WQEs
> and reads these CQEs itself — it maps the SQ/RQ/CQ buffers and doorbell
> registers via rdma-core once at QP/CQ creation (`efadv_query_qp_wqs()` /
> `efadv_query_cq()`), then writes WQEs straight into write-combining device
> memory with `mmio_memcpy_x64()` and reads CQEs directly from the mapped CQ
> buffer. When `FI_EFA_USE_DATA_PATH_DIRECT=0` the identical descriptor writes are
> performed by rdma-core/libibverbs (`ibv_wr_*`, `ibv_start_poll`) instead. The
> pseudo-code below shows the descriptor manipulation that happens on either path.
> See [rdma-core-and-verbs.md](rdma-core-and-verbs.md) for the full dispatch detail.

### Posting Send Request

> Source: libfabric `prov/efa/src/efa_data_path_direct_*.c` (or rdma-core `ibv_wr_*` on the fallback path). Use `codegraph explore` for the current bodies.

The fill sequence is: take the WQE at the SQ producer index, set `meta` fields
(`req_id`, `ctrl1` op_type, `ctrl2` first/last/comp_req bits — see the
`efa_io_tx_meta_desc` layout above, `dest_qp_num`, `ah`, `qkey`), fill
`data.sgl[i]` with length/lkey/split 64-bit address, advance the producer index
mod depth, then **ring the doorbell with an MMIO write**:

```c
writel(sq->producer_idx, sq->doorbell_addr);  // MMIO to device BAR; no syscall
```

### Polling Completion Queue

> Source: libfabric `prov/efa/src/efa_data_path_direct_*.c` (or rdma-core `ibv_start_poll`/`ibv_next_poll` on the fallback path). Use `codegraph explore` for the current bodies.

CQ polling is driven by the **phase bit**: read the CQE at the consumer index and
compare its phase bit (`flags & 0x1`) to the expected phase. If they differ there
is no new completion. On a match, consume `req_id`/`status`/`length`, advance the
consumer index mod depth, and **toggle the expected phase on wrap**:

```c
if ((cqe->common.flags & 0x1) != cq->phase) return 0;   // no new completion
/* ... process req_id / status / length ... */
if (++cq->consumer_idx == cq->depth) { cq->consumer_idx = 0; cq->phase ^= 1; }
```

### Zero-Copy Data Path

**Complete Send Operation** (~0.5-2 μs):
1. Application: Fill WQE in mapped memory (~50 ns)
2. Application: Write doorbell register via MMIO (~50 ns)
3. Hardware: DMA read WQE from host memory (~100-200 ns)
4. Hardware: DMA read data buffers (~100-500 ns, depending on size)
5. Hardware: Packetize and transmit (~200-800 ns)
6. **No kernel involvement** - entire path in user space

**Complete Receive Operation**:
1. Hardware: Receive packet from network
2. Hardware: DMA write to pre-posted RX buffer
3. Hardware: DMA write completion to CQ
4. Application: Poll CQ in mapped memory (reads completion)
5. **No syscalls** - direct memory access throughout

## EFA Hardware Capabilities

### Protocol Support by Generation

The kernel driver (**EFA r3.3.0**) enumerates PCI VF device IDs
`0xefa0`, `0xefa1`, `0xefa2`, `0xefa3`, and **`0xefa4`** (new in r3.3.0)
([efa_main.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_main.c) lines ~27-38).

**EFA Gen 2** (p4d, p4de instances):
- SRD transport only
- Send/Recv operations native
- RDMA Write emulated via Send/Recv
- RDMA Read not supported
- MTU: 8192 bytes

**EFA Gen 3+** (p5, p5e, p5en instances):
- SRD transport
- Send/Recv operations
- **Native RDMA Write** (no emulation)
- RDMA Read supported
- MTU: 8192 bytes
- 35% lower latency on p5en (EFAv3)

**Device features exposed in r3.3.0** (see [kernel-efa-driver.md](kernel-efa-driver.md)):
- **Completion Counters** (`EFA_ADMIN_*_EVENT_COUNTER` admin opcodes 25-29) — hardware
  completion counters, used by GDAKI / GPU-initiated paths.
- **QP/SQ creation with 64-bit work request IDs** (`EFA_QUERY_DEVICE_CAPS_SQ_64_BIT_REQ_ID`,
  exposed to userspace as `EFADV_WQ_CAPS_64_BIT_REQ_ID`).
- **128-byte send-queue WQEs** and **inline WRITE** (`EFA_QUERY_DEVICE_CAPS_DATA_POLLING_128`,
  `efa_calc_sq_wqe_size_kernel()`).
- **>4 GB MR page size** via an extended page-shift field in MR registration.
- **AH device-object reuse** via an rhashtable AH cache.

### DMA-BUF GPU Memory Support

**Requirements**:
- EFA Gen 4+ (device ID 0xefa3 or higher; the r3.3.0 driver adds 0xefa4)
- Linux kernel 5.12+
- Libfabric 1.20+

**Mechanism**:
- GPU memory exported as DMA-BUF file descriptor
- Registered as MR via `fi_mr_regattr()` with `FI_HMEM` flag
- Hardware performs DMA directly to/from GPU memory
- Page merging: 4KB pages → 2MB pages (estimated 256x fewer IOMMU entries), and
  r3.3.0 adds support for MR page sizes larger than 4 GB

**Supported GPUs**:
- NVIDIA (CUDA via dmabuf)
- AMD (ROCm via dmabuf)
- AWS Neuron (Trainium/Inferentia via P2P registration)

## Performance Characteristics

### Latency Breakdown (estimated)

**Software Overhead**:
- WQE preparation: ~50-100 ns
- Doorbell write: ~50-100 ns
- CQ polling: ~50-100 ns
- **Total SW overhead: ~150-300 ns**

**Hardware Path**:
- WQE DMA read: ~100-200 ns
- Data buffer DMA: ~100-500 ns (size-dependent)
- Packet transmission: ~200-800 ns
- Network flight time: ~10-20 μs (intra-AZ)
- **Hardware + network: ~10-20 μs**

**End-to-End Latency**:
- Small message (< 1 KB): ~12-25 μs
- Medium message (4-64 KB): ~15-30 μs
- Large message (> 1 MB): Limited by bandwidth

### Throughput

**Per-EFA link speed** is reported by the device and translated by the kernel
driver. As of r3.3.0 the driver reports link speeds up to **800 Gbps** and
**1600 Gbps** (`efa_link_gbps_to_speed_and_width()` maps ≥1600 → `IB_SPEED_XDR`,
≥800 → `IB_SPEED_NDR`, in
[efa_verbs.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_verbs.c) lines ~417-439),
and exposes the raw speed in Gbps via a new query-port verb. Earlier
100-Gbps-per-EFA instances remain common:

**Single QP Limits**:
- p4d: 100 Gbps per EFA (4 EFAs total, shared across 8 GPUs)
- p5/p5e/p5en: 100 Gbps per EFA (32 EFAs total, 4 per GPU); ~400 Gbps effective per GPU
- Newer EFA generations report per-link speeds of 800 Gbps and 1600 Gbps
  (`max_link_speed_gbps` in the device attributes, `EFA_ADMIN` network attr)
- Effective per-GPU: aggregate of all EFAs bound to that GPU

**Message Rate**:
- Small messages (64B): ~10-15 Mpps per QP
- Large messages (64KB): ~200-400K msg/s per QP
- Limited by PCIe bandwidth and NIC processing

## Summary

| Aspect | Details |
|--------|---------|
| **Queue Types** | Send Queue (SQ), Receive Queue (RQ), Completion Queue (CQ) |
| **Transport** | SRD (Scalable Reliable Datagram) - reliable out-of-order delivery |
| **WQE Size** | 64 bytes (TX), or **128 bytes** with wide-WQE cap (r3.3.0); 32 bytes (RX) |
| **CQE Size** | 16 bytes (TX and RX); 32 bytes for the extended RX descriptor |
| **Max Queue Depth** | 16384 (SQ/RQ), 262144 (CQ) |
| **Typical Depth** | 512 (SQ/RQ), 1024 (CQ) |
| **Operations** | Send, RDMA Read/Write (+ inline write on 128B WQE), Fast MR Reg/Inv |
| **SGL Entries** | 2 per send WQE, 1 per recv WQE |
| **Inline Data** | 32 bytes (64B WQE) / 80 bytes (128B WQE) |
| **Data path** | **Data Path Direct by default** (provider drives SQ/RQ/CQ); libibverbs fallback |
| **Memory Layout** | Ring buffers with phase bit tracking |
| **Access Model** | Zero-copy kernel bypass (mmap + doorbell) |
| **SW Overhead** | ~150-300 ns per operation |
| **Latency** | ~12-25 μs small messages (intra-AZ) |

**Key Takeaways**:
1. **Ring buffer architecture** with phase bit ownership tracking
2. **Zero-syscall data path** via memory-mapped queues and doorbell registers
3. **SRD transport** enables scalable many-to-many communication without RC's QP explosion
4. **Hardware descriptors** are compact (64B TX WQE, 16B RX CQE) for cache efficiency
5. **Fast MR registration** via work requests allows dynamic buffer management
6. **DMA-BUF support** on Gen 4+ enables direct GPU memory access
7. **Native RDMA** on Gen 3+ (p5 instances) vs emulation on Gen 2 (p4d)

**Related Documentation**:
- [srd-protocol.md](srd-protocol.md) - SRD transport protocol details
- [kernel-efa-driver.md](kernel-efa-driver.md) - Kernel driver architecture
- [rdma-memreg.md](rdma-memreg.md) - Memory registration concepts
- [efa-provider.md](efa-provider.md) - Libfabric EFA provider implementation

---

## Code References

### Structures

**amzn-drivers (EFA Kernel Driver)**:
- `enum efa_io_queue_type` ([efa_io_defs.h:15-20](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h))
- `enum efa_io_send_op_type` ([efa_io_defs.h:22-33](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h))
- `enum efa_io_comp_status` ([efa_io_defs.h:35-68](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h))
- `enum efa_io_frwr_pbl_mode` ([efa_io_defs.h:70-73](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h))
- `struct efa_io_tx_meta_desc` ([efa_io_defs.h:75-130](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h))
- `struct efa_io_tx_buf_desc` ([efa_io_defs.h:136-151](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h))
- `struct efa_io_remote_mem_addr` ([efa_io_defs.h:153-165](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h))
- `struct efa_io_rdma_req` ([efa_io_defs.h:167-173](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h))
- `struct efa_io_fast_mr_reg_req` ([efa_io_defs.h:175-222](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h))
- `struct efa_io_fast_mr_inv_req` ([efa_io_defs.h:224-230](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h))
- `struct efa_io_tx_wqe` ([efa_io_defs.h:236-255](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h))
- `struct efa_io_rx_desc` ([efa_io_defs.h:261-282](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h))
- `struct efa_io_cdesc_common` ([efa_io_defs.h:285-308](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h))
- `struct efa_io_tx_cdesc` ([efa_io_defs.h:311-317](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h))
- `struct efa_io_rx_cdesc` ([efa_io_defs.h:320-334](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h))
- `struct ib_srd_wr` ([efa_verbs.h:13-18](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_verbs.h))
- `struct ib_srd_rdma_wr` ([efa_verbs.h:20-24](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_verbs.h))

### Functions

**amzn-drivers (EFA Kernel Driver)**:
- `efa_inc_fast_reg_key_gen()` ([efa_verbs.h:41-47](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_verbs.h))

### Constants/Defines

**amzn-drivers (EFA Kernel Driver)**:
- `EFA_QPT_SRD` ([efa_verbs.h:36](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_verbs.h))
- `EFA_MR_GEN_SHIFT`, `EFA_MR_GEN_MASK` ([efa_verbs.h:38-39](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_verbs.h))

### Total Code References
- **17 structures/enums** from amzn-drivers
- **1 inline function** from amzn-drivers
- **3 constants** from amzn-drivers

All references link to the `master` branch of the [amzn/amzn-drivers](https://github.com/amzn/amzn-drivers) repository (EFA kernel driver **r3.3.0**). Line numbers are given in the link text since branch URLs are not pinned.
