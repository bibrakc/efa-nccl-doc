# EFA Driver Architecture and Capabilities

## Overview

The EFA (Elastic Fabric Adapter) driver is a Linux kernel driver that provides the interface between user-space applications (via libfabric) and the EFA hardware. It's based on the RDMA subsystem but with EFA-specific extensions.

**Driver Location**: `drivers/infiniband/hw/efa/` in Linux kernel (out-of-tree
build: [amzn-drivers `kernel/linux/efa/`](https://github.com/amzn/amzn-drivers/tree/master/kernel/linux/efa))

**Driver Version**: **r3.3.0** (`DRV_MODULE_VER_MAJOR/MINOR/SUBMINOR = 3/3/0`,
[efa_main.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_main.c) lines ~42-44).
See [kernel-efa-driver.md](kernel-efa-driver.md) for the detailed r3.3.0 change list.

> **Data-path note.** The `ibv_post_send()` / `ibv_poll_cq()` pseudo-code below
> illustrates the WQE/CQE mechanism the driver sets up. As of libfabric 2.3.0+
> (RDM-enabled in 2.7.0), the libfabric EFA provider by default drives these
> mapped SQ/RQ/CQ buffers itself via **Data Path Direct** rather than calling
> libibverbs on the data path — see [rdma-core-and-verbs.md](rdma-core-and-verbs.md).
> Either way the kernel driver's role is unchanged: it only sets up the queues,
> doorbells and memory mappings (control path).

## Architecture

```
User Space
┌──────────────────────────────────────┐
│  Application (NCCL + OFI plugin)     │
└─────────────┬────────────────────────┘
              │ Libfabric API
┌─────────────▼────────────────────────┐
│  Libfabric (libibverbs)              │
└─────────────┬────────────────────────┘
              │ Verbs API (ioctl, mmap)
              │ /dev/infiniband/uverbsN
──────────────┼──────────────────────────── User/Kernel boundary
              │
┌─────────────▼────────────────────────┐
│  ib_uverbs (RDMA core)               │
│  - Device abstraction                │
│  - Command dispatching               │
└─────────────┬────────────────────────┘
              │ Driver ops
┌─────────────▼────────────────────────┐
│  EFA Driver                          │
│  ├─ Device management                │
│  ├─ Memory registration              │
│  ├─ Queue pair (QP) management       │
│  ├─ Completion queue (CQ) management │
│  └─ Doorbell/MMIO handling           │
└─────────────┬────────────────────────┘
              │ PCIe
┌─────────────▼────────────────────────┐
│  EFA Hardware (NIC)                  │
│  ├─ DMA engine                       │
│  ├─ Packet processing                │
│  ├─ Reliability (SRD)                │
│  └─ Network interface (100G+)        │
└──────────────────────────────────────┘
```

## EFA Device Model

### PCIe Device

```
lspci output:
00:06.0 Ethernet controller: Amazon.com, Inc. Elastic Fabric Adapter (EFA)
    Subsystem: Amazon.com, Inc. Device efa0
    Flags: bus master, fast devsel, latency 0
    Memory at 82000000 (32-bit, non-prefetchable) [size=32M]
    Memory at 83000000 (32-bit, non-prefetchable) [size=256K]
    Capabilities: [40] Express Endpoint, MSI 00
    Capabilities: [100] Advanced Error Reporting
    Capabilities: [150] Device Serial Number
```

**PCIe Features:**
- **BAR0**: Device registers (MMIO)
- **BAR1**: Doorbell registers
- **MSI-X**: Interrupt vectors
- **Gen4 x16**: Up to 25 GB/s bidirectional

### Device Initialization

```c
// Driver probe function
static int efa_probe(struct pci_dev *pdev,
                     const struct pci_device_id *id)
{
  struct efa_dev *dev;

  // Allocate device structure
  dev = ib_alloc_device(sizeof(*dev), struct efa_dev, ibdev);

  // Enable PCIe device
  pci_enable_device(pdev);
  pci_request_regions(pdev, DRV_MODULE_NAME);
  pci_set_master(pdev);

  // Map MMIO regions
  dev->reg_bar = pci_iomap(pdev, 0, 0);  // BAR0
  dev->db_bar = pci_iomap(pdev, 2, 0);   // BAR2 (doorbells)

  // Setup DMA
  dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));

  // Initialize device
  efa_com_dev_init(&dev->edev, pdev);

  // Query device capabilities
  efa_com_get_device_attr(&dev->edev, &attr);

  // Register with RDMA core
  ib_register_device(&dev->ibdev, "efa_%d");

  return 0;
}
```

### Device Capabilities

```c
struct efa_com_get_device_attr_result {
  u32 max_qp;              // Max queue pairs (~32K)
  u32 max_cq;              // Max completion queues
  u32 max_pd;              // Max protection domains
  u32 max_mr;              // Max memory regions
  u32 max_mr_pages;        // Max pages per MR
  u32 max_qp_wr;           // Max WRs per QP (~8K)
  u32 max_cq_depth;        // Max CQ entries
  u32 max_inline_data;     // Max inline send (32 B on 64-B WQE, 80 B on 128-B WQE)
  u32 max_sge;             // Max scatter-gather entries
  u32 max_mtu;             // Max MTU (~8K)

  // EFA-specific
  u16 sub_cqs_per_cq;      // Completion sub-queues
  u32 max_rdma_size;       // Max RDMA transfer (~1 GB)
  u16 max_link_speed_gbps; // Device max link speed in Gbit/s (up to 800/1600)
  u8 device_version;       // Hardware version
  u32 device_caps;         // Capability bitmask (see below)
};
```

**Capability bits relevant to r3.3.0** (queried via `EFADV`/`EFA_QUERY_DEVICE_CAPS_*`):
- `EFA_QUERY_DEVICE_CAPS_SQ_64_BIT_REQ_ID` — SQ supports 64-bit work request IDs
  (userspace `EFADV_WQ_CAPS_64_BIT_REQ_ID`).
- `EFA_QUERY_DEVICE_CAPS_DATA_POLLING_128` — 128-byte WQE / wide-WQE support
  (enables inline RDMA WRITE).
- Hardware **completion counters** (event counters) via the
  `EFA_ADMIN_*_EVENT_COUNTER` admin opcodes; device reports `max_event_counters`,
  `event_counter_max_val`, `supported_event_counter_qp_events` (QUEUE_ATTR_2).
  These back GDAKI / GPU-initiated completion tracking.
- Extended MR page-shift field enabling **>4 GB MR page sizes**.
- Link speeds of **800 and 1600 Gbps** reported via `max_link_speed_gbps` and the
  new `efa_query_port_speed()` verb.
- New PCI device ID **0xefa4** enumerated by the driver.

## Memory Management

### Memory Registration

```c
// User calls ibv_reg_mr() → ioctl → driver
static struct ib_mr *efa_reg_mr(struct ib_pd *ibpd,
                                u64 start, u64 length,
                                u64 virt_addr,
                                int access_flags,
                                struct ib_udata *udata)
{
  struct efa_dev *dev = to_edev(ibpd->device);
  struct efa_mr *mr = kzalloc(sizeof(*mr), GFP_KERNEL);

  // Pin user pages
  mr->umem = ib_umem_get(ibpd->device, start, length,
                         access_flags);
  if (IS_ERR(mr->umem)) {
    return ERR_PTR(-ENOMEM);
  }

  // Create scatter-gather list
  npages = ib_umem_num_pages(mr->umem);
  mr->pbl = kzalloc(sizeof(*mr->pbl), GFP_KERNEL);

  // DMA map pages
  nmap = dma_map_sg_attrs(dev->pdev->dev,
                          mr->umem->sgl,
                          mr->umem->nmap,
                          DMA_BIDIRECTIONAL,
                          DMA_ATTR_SKIP_CPU_SYNC);

  // Register with device
  struct efa_com_reg_mr_params params = {
    .pd = to_epd(ibpd)->pdn,
    .mr_length = length,
    .iova = virt_addr,
    .phys_page_list = mr->pbl,
    .page_num = npages,
    .page_shift = PAGE_SHIFT,
  };

  err = efa_com_register_mr(dev->edev, &params, &mr->lkey);

  mr->ibmr.lkey = mr->lkey;
  mr->ibmr.rkey = mr->lkey;

  return &mr->ibmr;
}
```

**Steps:**
1. **Pin pages**: `ib_umem_get()` - prevents swapping
2. **Build page list**: Scatter-gather list for DMA
3. **DMA map**: Create IOMMU mappings
4. **Device registration**: Program NIC with translations
5. **Generate keys**: lkey and rkey for protection

### GPU Memory (dmabuf)

```c
static struct ib_mr *efa_reg_dmabuf_mr(struct ib_pd *ibpd,
                                       u64 offset, u64 length,
                                       u64 virt_addr,
                                       int fd,
                                       int access_flags,
                                       struct ib_udata *udata)
{
  struct efa_dev *dev = to_edev(ibpd->device);
  struct efa_mr *mr = kzalloc(sizeof(*mr), GFP_KERNEL);

  // Import dmabuf from fd
  mr->dmabuf = dma_buf_get(fd);
  if (IS_ERR(mr->dmabuf)) {
    return ERR_PTR(-EINVAL);
  }

  // Attach to dmabuf
  mr->attach = dma_buf_attach(mr->dmabuf, &dev->pdev->dev);

  // Get scatter-gather table
  mr->sgt = dma_buf_map_attachment(mr->attach, DMA_BIDIRECTIONAL);

  // Register with device (similar to regular MR)
  // ...

  return &mr->ibmr;
}
```

**dmabuf Flow:**
```
CUDA Driver
    ↓ cudaExternalMemoryGetMappedBuffer()
DMA-BUF (file descriptor)
    ↓ dma_buf_get(fd)
EFA Driver
    ↓ dma_buf_attach(), dma_buf_map_attachment()
Scatter-Gather Table
    ↓ Register with EFA device
GPU Memory ↔ NIC DMA
```

### IOMMU Translation

```
Virtual Address (App)
    ↓ CPU Page Table
Physical Address (RAM/GPU)
    ↓ IOMMU
DMA Address (NIC sees)

Example:
Virtual: 0x7fff12340000
    ↓
Physical: 0x84562000 (RAM) or GPU BAR
    ↓
DMA: 0x100000 (NIC-visible address)
```

**IOMMU Benefits:**
- **Security**: NIC can only access registered memory
- **Virtualization**: DMA address space independent of physical
- **Scatter-gather**: Fragmented physical pages appear contiguous

## Queue Pair (QP) Management

### QP Creation

```c
static struct ib_qp *efa_create_qp(struct ib_pd *ibpd,
                                   struct ib_qp_init_attr *init_attr,
                                   struct ib_udata *udata)
{
  struct efa_dev *dev = to_edev(ibpd->device);
  struct efa_qp *qp = kzalloc(sizeof(*qp), GFP_KERNEL);

  // Allocate QP number
  qp->qp_num = efa_alloc_qpn(dev);

  // Create send queue (SQ)
  qp->sq.depth = init_attr->cap.max_send_wr;
  qp->sq.wrid_map = kzalloc(qp->sq.depth * sizeof(u64), GFP_KERNEL);

  // Create receive queue (RQ)
  qp->rq.depth = init_attr->cap.max_recv_wr;
  qp->rq.wrid_map = kzalloc(qp->rq.depth * sizeof(u64), GFP_KERNEL);

  // Map queue memory to user space
  qp->sq.buf = ib_umem_get(...);  // User provides buffer
  qp->rq.buf = ib_umem_get(...);

  // Create QP on device
  struct efa_com_create_qp_params params = {
    .pd = to_epd(ibpd)->pdn,
    .qp_type = init_attr->qp_type,
    .send_cq_idx = to_ecq(init_attr->send_cq)->cq_idx,
    .recv_cq_idx = to_ecq(init_attr->recv_cq)->cq_idx,
    .sq_depth = qp->sq.depth,
    .rq_depth = qp->rq.depth,
  };

  err = efa_com_create_qp(dev->edev, &params, &qp_num);

  // Map doorbell to user space
  qp->sq_db = dev->db_bar + (qp_num * EFA_DB_SIZE);
  // User can directly write to qp->sq_db (MMIO)

  return &qp->ibqp;
}
```

### Queue Structure

```
Send Queue (SQ):
┌────────────────────────────────────┐
│ WQE 0 (Work Queue Entry)           │
│ WQE 1                              │
│ WQE 2                              │
│ ...                                │
│ WQE N                              │
└────────────────────────────────────┘
        ↑ producer (user)
        ↓ consumer (hardware)

Receive Queue (RQ):
┌────────────────────────────────────┐
│ WQE 0                              │
│ WQE 1                              │
│ ...                                │
└────────────────────────────────────┘
```

**WQE (Work Queue Entry):**
- Describes one send/recv operation
- Contains buffer address, length, flags
- Written by user, consumed by hardware

### Post Send

```c
// User calls ibv_post_send()
// → Builds WQE in user space
// → Writes to SQ ring buffer
// → Rings doorbell (MMIO write)

static int efa_post_send(struct ib_qp *ibqp,
                         const struct ib_send_wr *wr,
                         const struct ib_send_wr **bad_wr)
{
  struct efa_qp *qp = to_eqp(ibqp);

  while (wr) {
    // Check space in SQ
    if (qp->sq.tail - qp->sq.head >= qp->sq.depth) {
      return -ENOMEM;  // Queue full
    }

    // Build WQE (in user-mapped memory)
    u32 idx = qp->sq.tail & (qp->sq.depth - 1);
    struct efa_wqe *wqe = &qp->sq.wqes[idx];

    wqe->opcode = wr->opcode;
    wqe->length = wr->sg_list[0].length;
    wqe->lkey = wr->sg_list[0].lkey;
    wqe->addr = wr->sg_list[0].addr;
    wqe->wr_id = wr->wr_id;

    if (wr->opcode == IB_WR_RDMA_WRITE ||
        wr->opcode == IB_WR_RDMA_READ) {
      wqe->remote_addr = wr->wr.rdma.remote_addr;
      wqe->rkey = wr->wr.rdma.rkey;
    }

    qp->sq.tail++;
    wr = wr->next;
  }

  // Ring doorbell (MMIO write)
  writel(qp->sq.tail, qp->sq_db);

  return 0;
}
```

**Doorbell:**
- MMIO write to device register
- Notifies hardware of new WQEs
- Very fast (~100-200 ns)
- No kernel crossing

### Post Receive

```c
static int efa_post_recv(struct ib_qp *ibqp,
                         const struct ib_recv_wr *wr,
                         const struct ib_recv_wr **bad_wr)
{
  struct efa_qp *qp = to_eqp(ibqp);

  while (wr) {
    // Build receive WQE
    u32 idx = qp->rq.tail & (qp->rq.depth - 1);
    struct efa_rq_wqe *wqe = &qp->rq.wqes[idx];

    wqe->addr = wr->sg_list[0].addr;
    wqe->length = wr->sg_list[0].length;
    wqe->lkey = wr->sg_list[0].lkey;
    wqe->wr_id = wr->wr_id;

    qp->rq.tail++;
    wr = wr->next;
  }

  // Ring doorbell
  writel(qp->rq.tail, qp->rq_db);

  return 0;
}
```

## Completion Queue (CQ) Management

### CQ Creation

```c
static struct ib_cq *efa_create_cq(struct ib_device *ibdev,
                                   const struct ib_cq_init_attr *attr,
                                   struct ib_udata *udata)
{
  struct efa_dev *dev = to_edev(ibdev);
  struct efa_cq *cq = kzalloc(sizeof(*cq), GFP_KERNEL);

  cq->depth = attr->cqe;

  // Allocate CQ buffer (user-mapped)
  cq->buf = ib_umem_get(...);

  // Create CQ on device
  struct efa_com_create_cq_params params = {
    .cq_depth = cq->depth,
    .entry_size = sizeof(struct efa_cqe),
  };

  err = efa_com_create_cq(dev->edev, &params, &cq->cq_idx);

  return &cq->ibcq;
}
```

### CQ Structure

```
Completion Queue:
┌────────────────────────────────────┐
│ CQE 0 (Completion Queue Entry)     │
│ CQE 1                              │
│ CQE 2                              │
│ ...                                │
│ CQE N                              │
└────────────────────────────────────┘
        ↑ producer (hardware)
        ↓ consumer (user)
```

**CQE (Completion Queue Entry):**
```c
struct efa_cqe {
  u32 wqe_id;         // Which WQE completed
  u16 status;         // Success or error
  u16 length;         // Bytes transferred
  u32 qp_num;         // Which QP
  u8 opcode;          // Send, recv, RDMA, etc.
  u8 flags;           // Various flags
};
```

### Poll CQ (User Space)

```c
// User calls ibv_poll_cq()
// → Reads CQ ring buffer (shared memory)
// → No kernel crossing!

int ibv_poll_cq(struct ibv_cq *ibcq,
                int num_entries,
                struct ibv_wc *wc)
{
  struct efa_cq *cq = to_ecq(ibcq);
  int npolled = 0;

  while (npolled < num_entries) {
    u32 idx = cq->cons_index & (cq->depth - 1);
    struct efa_cqe *cqe = &cq->cqes[idx];

    // Check owner bit (toggles each wrap)
    u8 expected_owner = (cq->cons_index / cq->depth) & 1;
    if (cqe->owner != expected_owner) {
      break;  // No more completions
    }

    // Parse completion
    wc[npolled].wr_id = cq->wrid_map[cqe->wqe_id];
    wc[npolled].status = cqe->status;
    wc[npolled].byte_len = cqe->length;
    wc[npolled].qp_num = cqe->qp_num;
    wc[npolled].opcode = cqe->opcode;

    cq->cons_index++;
    npolled++;
  }

  return npolled;
}
```

**Key Point**: Polling is entirely in user space (no syscall)!

## Data Path

### Send Path

```
User Space:
  ibv_post_send()
    ↓ Build WQE in SQ
    ↓ Write doorbell (MMIO)
    ↓ Return to user

Hardware:
  (async) Read WQE from SQ
  (async) DMA read source buffer
  (async) Build packet
  (async) Send on network
  (async) Write CQE to CQ
  (async) Generate MSI-X interrupt (optional)

User Space:
  ibv_poll_cq()
    ↓ Read CQE from CQ
    ↓ Return completion
```

**No kernel in fast path!**

### Receive Path

```
User Space:
  ibv_post_recv()
    ↓ Build WQE in RQ
    ↓ Write doorbell
    ↓ Return

Hardware:
  (async) Packet arrives
  (async) Match with RQ WQE
  (async) DMA write to recv buffer
  (async) Write CQE to CQ

User Space:
  ibv_poll_cq()
    ↓ Read CQE
    ↓ Process received data
```

### Zero-Copy Path

```
GPU Memory                          Remote GPU Memory
    │                                     │
    ├─ Registered (fi_mr_reg)             │
    │                                     │
    ├─ DMA read by EFA NIC ──────────────→│
    │  (PCIe → NIC → Network → NIC → PCIe)│
    │                                     │
    └─ No CPU copy, no system memory!    ─┘
```

## SRD Protocol Implementation

### Reliability Layer

The EFA hardware implements Scalable Reliable Datagram (SRD) protocol:

```
Hardware Components:
┌────────────────────────────────────┐
│ TX Engine                          │
│ ├─ Packetization                   │
│ ├─ Sequence numbering              │
│ ├─ Retransmit buffer               │
│ └─ Timer management                │
├────────────────────────────────────┤
│ RX Engine                          │
│ ├─ Packet reassembly               │
│ ├─ Sequence checking               │
│ ├─ Out-of-order buffering          │
│ └─ NAK generation                  │
└────────────────────────────────────┘
```

**Sequence Numbers:**
- Per-packet sequence number
- Receiver detects gaps
- Automatic NAK for missing packets
- Sender retransmits from buffer

**No Per-Peer State:**
- Single QP to all peers
- Hardware handles reliability
- Scalable to many peers

## Interrupts vs Polling

### MSI-X Interrupts

```c
// EFA driver interrupt handler
static irqreturn_t efa_intr_handler(int irq, void *data)
{
  struct efa_eq *eq = data;

  // Read event queue
  while (efa_com_get_event(eq, &event)) {
    switch (event.type) {
      case EFA_EVENT_CQ_COMPLETION:
        // Wake up waiting threads
        wake_up(&cq->wait_queue);
        break;

      case EFA_EVENT_QP_ERROR:
        // Handle QP error
        efa_handle_qp_error(qp);
        break;
    }
  }

  return IRQ_HANDLED;
}
```

**Use Cases:**
- Blocking operations (ibv_get_cq_event)
- Error notifications
- Rare in NCCL (uses polling)

### Polling Mode

**NCCL/libfabric typical usage:**
- No interrupts for data path
- User space polls CQ continuously
- Lower latency (~10 μs vs ~50 μs)
- Higher CPU usage

```bash
# Check interrupt counts
cat /proc/interrupts | grep efa

# Low interrupt count = polling mode
```

## Performance Characteristics

### Latency Components

```
User Space Operation         Time
─────────────────────────────────────
ibv_post_send()              ~100 ns
  ├─ Build WQE               ~50 ns
  └─ MMIO write (doorbell)   ~50 ns

Hardware Processing          Time
─────────────────────────────────────
WQE fetch                    ~200 ns
DMA read (source buffer)     ~500 ns (depends on size)
Packet formation             ~100 ns
Network transmission         ~10-20 μs (one-way)

Completion                   Time
─────────────────────────────────────
Write CQE                    ~100 ns
ibv_poll_cq()                ~100 ns

Total (small message):       ~10-15 μs (one-way)
```

### Bandwidth

```
Single QP:
  Small messages (< 4 KB):    ~1-2 GB/s
  Medium (64 KB):             ~8-10 GB/s
  Large (> 1 MB):             ~11-12 GB/s

Multiple QPs (4-8):
  Aggregate:                  ~40-50 GB/s
  (limited by PCIe, not NIC)
```

### Doorbell Overhead

```
Doorbells per Second        CPU Impact
────────────────────────────────────────
1K                          Negligible
10K                         ~1% CPU
100K                        ~5-10% CPU
1M                          ~50% CPU
10M                         100% CPU (spinning)
```

**Batching helps:**
- Post multiple WQEs before doorbell
- Reduces MMIO overhead

## Tuning and Configuration

### Module Parameters

```bash
# View EFA driver parameters
cat /sys/module/efa/parameters/*

# Common parameters
echo 8192 > /sys/module/efa/parameters/max_qp_wr  # QP depth
```

### Sysfs Attributes

```bash
# Device info
ls /sys/class/infiniband/efa_0/

# Device stats
cat /sys/class/infiniband/efa_0/ports/1/counters/*

# Example counters:
# - port_xmit_packets
# - port_rcv_packets
# - port_xmit_data
# - port_rcv_data
```

### Debugging

```bash
# Enable driver debug
echo 8 > /proc/sys/kernel/printk  # Enable debug messages
echo 'module efa +p' > /sys/kernel/debug/dynamic_debug/control

# Check dmesg
dmesg | grep efa

# Check device status
ibv_devinfo -d efa_0
```

## Advanced Features

### GPUDirect RDMA

EFA supports direct GPU-to-GPU transfers:

```
GPU 0 Memory ─→ PCIe ─→ EFA NIC ─→ Network
                                      ↓
GPU 1 Memory ←─ PCIe ←─ EFA NIC ←─ Network
```

**Requirements:**
- CUDA-capable GPU
- dmabuf support (kernel 5.12+)
- Proper IOMMU configuration

**Benefits:**
- Zero-copy GPU-to-GPU
- No staging through system memory
- Lower latency, higher bandwidth

### Multi-Rail

Multiple EFAs per instance:

```
┌──────────────────────────────────┐
│  Application                     │
└───┬─────────┬─────────┬──────────┘
    │         │         │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│ EFA 0 │ │ EFA 1 │ │ EFA 2 │
└───┬───┘ └───┬───┘ └───┬───┘
    │         │         │
    └─────────┴─────────┴─→ Aggregate bandwidth
```

**Scaling:**
- Linear bandwidth increase
- NCCL automatically uses all EFAs
- 4 EFAs = ~400 Gbps (p4d.24xlarge)

## Summary

**EFA Driver Responsibilities:**
1. **Device Management**: Initialize and manage EFA hardware
2. **Memory Registration**: Pin pages, create IOMMU mappings
3. **Queue Management**: Create/manage QPs and CQs
4. **Doorbell Handling**: MMIO for user-space doorbells
5. **Completion Handling**: Manage CQ events
6. **SRD Protocol**: Hardware-accelerated reliability

**Performance Features:**
- **Zero-copy**: Direct user space to hardware
- **Kernel bypass**: No syscalls in data path
- **MMIO doorbells**: Fast work submission
- **Shared CQ memory**: Fast completion polling
- **GPUDirect**: Direct GPU-to-GPU transfers

**Key Characteristics:**
- **User-space I/O**: App directly accesses queues
- **Manual progress**: No kernel threads
- **Polling-optimized**: Low latency via polling
- **Multi-rail capable**: Support for multiple NICs
- **SRD transport**: Reliable, connectionless, scalable

**r3.3.0 additions** (see [kernel-efa-driver.md](kernel-efa-driver.md)):
- Hardware **completion counters** (used by GDAKI / GPU-initiated data paths)
- **64-bit work request IDs** and **128-byte WQEs** with **inline RDMA WRITE**
- **800 / 1600 Gbps** link-speed reporting + query-port-speed verb
- **0xefa4** device ID, admin-response **checksum validation**, **>4 GB MR page
  sizes**, and an **rhashtable AH cache** for AH device-object reuse

**Bottlenecks:**
- **PCIe bandwidth**: ~25 GB/s limit (Gen4 x16)
- **Queue depth**: Limited outstanding operations
- **Doorbell rate**: MMIO overhead at high rates
- **Memory registration**: Expensive without caching

**Next**: Passthrough modes and further optimizations.
