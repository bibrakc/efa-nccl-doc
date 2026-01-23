# EFA Hardware Architecture: Queue Pairs, Completion Queues, and Memory Layout

## Overview

AWS Elastic Fabric Adapter (EFA) is a custom network interface designed for HPC and ML workloads, implementing RDMA-like semantics with kernel bypass. This document describes the low-level hardware queue architecture, memory layout, and programming interface.

**Key Components**:
- **Queue Pairs (QP)**: Send Queue (SQ) + Receive Queue (RQ)
- **Completion Queues (CQ)**: Asynchronous completion notification
- **Work Queue Entries (WQE)**: Commands submitted to hardware
- **Completion Descriptors (CQE)**: Results returned by hardware
- **Address Handles (AH)**: Destination addressing for SRD transport

**Source Locations**:
- Kernel driver: [amzn-drivers/kernel/linux/efa/src/](https://github.com/amzn/amzn-drivers/tree/master/kernel/linux/efa/src)
- Key headers: [efa_io_defs.h](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h), [efa_verbs.h](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.h)
- Libfabric provider: [libfabric/prov/efa/](https://github.com/ofiwg/libfabric/tree/main/prov/efa)

## Queue Pair Architecture

### Queue Pair Types

EFA supports multiple queue pair types.

**`enum efa_io_queue_type`** ([efa_io_defs.h:15-20](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L15-L20)):

```c
enum efa_io_queue_type {
	/* send queue (of a QP) */
	EFA_IO_SEND_QUEUE = 1,
	/* recv queue (of a QP) */
	EFA_IO_RECV_QUEUE = 2,
};
```

**Transport Types**:
- **SRD (Scalable Reliable Datagram)**: Primary transport for EFA, defined as `EFA_QPT_SRD` in [efa_verbs.h:36](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.h#L36)
- Similar semantics to UD (Unreliable Datagram) but with reliability
- Out-of-order delivery without segmentation
- Address Handle (AH) provided with each work request

### Send Queue (SQ)

The Send Queue contains work requests (WQEs) posted by software and consumed by hardware.

**Operations Supported**:

**`enum efa_io_send_op_type`** ([efa_io_defs.h:22-33](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L22-L33)):

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

**`struct efa_io_tx_wqe`** ([efa_io_defs.h:236-255](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L236-L255)):

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

**Size**: Estimated 64 bytes (meta descriptor + data union)

### TX Metadata Descriptor

**`struct efa_io_tx_meta_desc`** ([efa_io_defs.h:75-130](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L75-L130)):

```c
struct efa_io_tx_meta_desc {
	/* Verbs-generated Request ID */
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
	u16 length;           // SGL length or inline data size
	u32 immediate_data;   // Immediate data (if has_imm set)
	u16 ah;               // Address Handle index
	u16 reserved;
	u32 qkey;             // Queue key for datagram QPs
	u8 reserved2[12];
};
```

**Key Fields**:
- `req_id`: Software-generated ID for matching completions (16 bits)
- `op_type`: Operation type (send, RDMA read/write, etc.)
- `phase`: Ring buffer wrap-around detection
- `comp_req`: Whether to generate a completion event
- `dest_qp_num`: Remote QP number (SRD transport)
- `ah`: Address Handle for destination addressing
- `qkey`: Protection key for datagram operations

### TX Buffer Descriptor

**`struct efa_io_tx_buf_desc`** ([efa_io_defs.h:136-151](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L136-L151)):

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
- Inline data mode: Up to 32 bytes inline (`EFA_IO_TX_DESC_INLINE_MAX_SIZE`)

### RX Descriptor

**`struct efa_io_rx_desc`** ([efa_io_defs.h:261-282](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L261-L282)):

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

**`struct efa_io_cdesc_common`** ([efa_io_defs.h:285-308](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L285-L308)):

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

**`struct efa_io_tx_cdesc`** ([efa_io_defs.h:311-317](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L311-L317)):

```c
struct efa_io_tx_cdesc {
	/* Common completion info */
	struct efa_io_cdesc_common common;

	/* MBZ */
	u16 reserved16;
};
```

**`struct efa_io_rx_cdesc`** ([efa_io_defs.h:320-334](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L320-L334)):

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

**`enum efa_io_comp_status`** ([efa_io_defs.h:35-68](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L35-L68)):

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

**`struct efa_io_fast_mr_reg_req`** ([efa_io_defs.h:175-222](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L175-L222)):

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

**`enum efa_io_frwr_pbl_mode`** ([efa_io_defs.h:70-73](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L70-L73)):
- `EFA_IO_FRWR_INLINE_PBL`: Page list inline in descriptor (small regions)
- `EFA_IO_FRWR_DIRECT_PBL`: Page list in separate buffer (large regions)

**lkey/rkey Format** ([efa_verbs.h:38-47](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.h#L38-L47)):

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

**`struct efa_io_rdma_req`** ([efa_io_defs.h:167-173](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L167-L173)):

```c
struct efa_io_rdma_req {
	/* Remote memory address */
	struct efa_io_remote_mem_addr remote_mem;

	/* Local memory address */
	struct efa_io_tx_buf_desc local_mem[1];
};
```

**`struct efa_io_remote_mem_addr`** ([efa_io_defs.h:153-165](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L153-L165)):

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

**`struct ib_srd_wr`** ([efa_verbs.h:13-18](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.h#L13-L18)):

```c
struct ib_srd_wr {
	struct ib_send_wr wr;
	struct ib_ah *ah;       // Address handle for destination
	u32 remote_qpn;         // Remote QP number
	u32 remote_qkey;        // Remote queue key
};
```

**`struct ib_srd_rdma_wr`** ([efa_verbs.h:20-24](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.h#L20-L24)):

```c
struct ib_srd_rdma_wr {
	struct ib_srd_wr wr;
	u64 remote_addr;        // Remote memory address
	u32 rkey;               // Remote key
};
```

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
- Max inline data: 32 bytes

**Memory Consumption** (estimated):
- SQ: 64 bytes/entry × depth
- RQ: 32 bytes/entry × depth
- CQ: 16 bytes/entry × depth (TX), 24 bytes/entry (RX)

Example for 512-entry QP + 1024-entry CQ:
- SQ: 512 × 64 = 32 KB
- RQ: 512 × 32 = 16 KB
- CQ: 1024 × 24 = 24 KB
- **Total: ~72 KB per QP**

For 1000 QPs: ~72 MB of queue memory

## Programming Model

### Posting Send Request

```c
// 1. Allocate WQE from send queue
struct efa_io_tx_wqe *wqe = &sq->wqes[sq->producer_idx];

// 2. Fill metadata
wqe->meta.req_id = generate_request_id();
wqe->meta.ctrl1 = (EFA_IO_SEND << 0);  // op_type
wqe->meta.ctrl2 = (1 << 2) | (1 << 3) | (1 << 4);  // first, last, comp_req
wqe->meta.dest_qp_num = remote_qp;
wqe->meta.length = num_sge;
wqe->meta.ah = address_handle_index;
wqe->meta.qkey = QP_QKEY;

// 3. Fill buffer descriptors
wqe->data.sgl[0].length = buffer_length;
wqe->data.sgl[0].lkey = mr->lkey;
wqe->data.sgl[0].buf_addr_lo = (u32)(buffer_addr & 0xFFFFFFFF);
wqe->data.sgl[0].buf_addr_hi = (u32)(buffer_addr >> 32);

// 4. Advance producer index
sq->producer_idx = (sq->producer_idx + 1) % sq->depth;

// 5. Ring doorbell (MMIO write)
writel(sq->producer_idx, sq->doorbell_addr);
```

### Polling Completion Queue

```c
// 1. Read CQ entry
struct efa_io_rx_cdesc *cqe = &cq->entries[cq->consumer_idx];

// 2. Check ownership via phase bit
u8 expected_phase = cq->phase;
u8 entry_phase = cqe->common.flags & 0x1;

if (entry_phase != expected_phase) {
	// No new completions
	return 0;
}

// 3. Process completion
u16 req_id = cqe->common.req_id;
u8 status = cqe->common.status;
u16 length = cqe->length;  // bytes received

if (status != EFA_IO_COMP_STATUS_OK) {
	// Handle error
}

// 4. Advance consumer index
cq->consumer_idx = (cq->consumer_idx + 1) % cq->depth;
if (cq->consumer_idx == 0) {
	cq->phase ^= 1;  // Toggle phase on wrap
}

// 5. Return completion to application
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

### DMA-BUF GPU Memory Support

**Requirements**:
- EFA Gen 4+ (device ID 0xefa3 or higher)
- Linux kernel 5.12+
- Libfabric 1.20+

**Mechanism**:
- GPU memory exported as DMA-BUF file descriptor
- Registered as MR via `fi_mr_regattr()` with `FI_HMEM` flag
- Hardware performs DMA directly to/from GPU memory
- Page merging: 4KB pages → 2MB pages (estimated 256x fewer IOMMU entries)

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

**Single QP Limits**:
- p4d: 100 Gbps per EFA (4 EFAs total, shared across 8 GPUs)
- p5+: 100 Gbps per EFA (32 EFAs total, 4 per GPU)
- Effective per-GPU: ~400 Gbps (50 GB/s)

**Message Rate**:
- Small messages (64B): ~10-15 Mpps per QP
- Large messages (64KB): ~200-400K msg/s per QP
- Limited by PCIe bandwidth and NIC processing

## Summary

| Aspect | Details |
|--------|---------|
| **Queue Types** | Send Queue (SQ), Receive Queue (RQ), Completion Queue (CQ) |
| **Transport** | SRD (Scalable Reliable Datagram) - reliable out-of-order delivery |
| **WQE Size** | 64 bytes (TX), 32 bytes (RX) |
| **CQE Size** | 8 bytes (TX), 16 bytes (RX) |
| **Max Queue Depth** | 16384 (SQ/RQ), 262144 (CQ) |
| **Typical Depth** | 512 (SQ/RQ), 1024 (CQ) |
| **Operations** | Send, RDMA Read/Write, Fast MR Reg/Inv |
| **SGL Entries** | 2 per send WQE, 1 per recv WQE |
| **Inline Data** | 32 bytes max |
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
- `enum efa_io_queue_type` ([efa_io_defs.h:15-20](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L15-L20))
- `enum efa_io_send_op_type` ([efa_io_defs.h:22-33](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L22-L33))
- `enum efa_io_comp_status` ([efa_io_defs.h:35-68](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L35-L68))
- `enum efa_io_frwr_pbl_mode` ([efa_io_defs.h:70-73](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L70-L73))
- `struct efa_io_tx_meta_desc` ([efa_io_defs.h:75-130](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L75-L130))
- `struct efa_io_tx_buf_desc` ([efa_io_defs.h:136-151](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L136-L151))
- `struct efa_io_remote_mem_addr` ([efa_io_defs.h:153-165](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L153-L165))
- `struct efa_io_rdma_req` ([efa_io_defs.h:167-173](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L167-L173))
- `struct efa_io_fast_mr_reg_req` ([efa_io_defs.h:175-222](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L175-L222))
- `struct efa_io_fast_mr_inv_req` ([efa_io_defs.h:224-230](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L224-L230))
- `struct efa_io_tx_wqe` ([efa_io_defs.h:236-255](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L236-L255))
- `struct efa_io_rx_desc` ([efa_io_defs.h:261-282](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L261-L282))
- `struct efa_io_cdesc_common` ([efa_io_defs.h:285-308](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L285-L308))
- `struct efa_io_tx_cdesc` ([efa_io_defs.h:311-317](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L311-L317))
- `struct efa_io_rx_cdesc` ([efa_io_defs.h:320-334](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_io_defs.h#L320-L334))
- `struct ib_srd_wr` ([efa_verbs.h:13-18](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.h#L13-L18))
- `struct ib_srd_rdma_wr` ([efa_verbs.h:20-24](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.h#L20-L24))

### Functions

**amzn-drivers (EFA Kernel Driver)**:
- `efa_inc_fast_reg_key_gen()` ([efa_verbs.h:41-47](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.h#L41-L47))

### Constants/Defines

**amzn-drivers (EFA Kernel Driver)**:
- `EFA_QPT_SRD` ([efa_verbs.h:36](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.h#L36))
- `EFA_MR_GEN_SHIFT`, `EFA_MR_GEN_MASK` ([efa_verbs.h:38-39](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.h#L38-L39))

### Total Code References
- **17 structures/enums** from amzn-drivers
- **1 inline function** from amzn-drivers
- **3 constants** from amzn-drivers

All references link to commit `8a8b6f2` in the [amzn/amzn-drivers](https://github.com/amzn/amzn-drivers) repository.
