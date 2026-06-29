# Linux Kernel EFA Driver

## Overview

The **EFA kernel driver** is a Linux kernel module that provides the interface between userspace RDMA applications and the EFA hardware. It implements the kernel-side of the RDMA verbs API and manages hardware resources.

**GitHub**: [kernel/linux/efa/](https://github.com/amzn/amzn-drivers/tree/8a8b6f2/kernel/linux/efa)

**Driver Version**: 3.0.0 (as of this documentation)

### Firmware and Hardware Boundary (what is open vs closed)

The EFA kernel driver (`efa.ko`, ~13K LOC under `kernel/linux/efa/src`) is **open source**
(GPL-2.0 / Linux-OpenIB), as is everything above it in the stack — libfabric's EFA
provider (`prov/efa`, ~62K LOC), rdma-core's EFA userspace verbs (`providers/efa`,
~4.5K LOC), and the OFI plugin. The driver is a relatively thin **control-plane and
resource manager**: it sets up queues, registers memory, and exchanges commands with the
device over the admin queue. It does **not** implement the wire protocol.

The actual transport intelligence lives in **closed NIC firmware** running on the EFA
adapter (Nitro-class hardware):
- **SRD protocol** — packetization, multipath spraying across up to 64 paths, reordering,
  ACK/retransmit, and hardware congestion control (see `srd-protocol.md`). The driver and
  firmware communicate only through admin-queue commands and the data-path doorbells/CQs;
  the SRD state machine itself is not in any open source.
- **Hardware/ASIC** — the NIC silicon and its DMA engines.

Practical consequence: behaviors like the **Gen 1–3 DMA-BUF page-merging issue** are
firmware characteristics observable from the driver, not bugs fixable in the open driver
source. When debugging, the open code tells you *what was requested of the device*; the
device's internal handling is inferred from completions, counters, and errors.

## Architecture Position

```
┌─────────────────────────────────────┐
│   Userspace (libibverbs, libfabric) │
│   - Open /dev/infiniband/uverbsN    │
│   - ioctl() for commands            │
│   - mmap() for queue buffers        │
├─────────────────────────────────────┤
│   Kernel RDMA Core                  │
│   - ib_core module                  │
│   - /dev/infiniband/ device files   │
│   - ib_uverbs (userspace interface) │
├─────────────────────────────────────┤
│   EFA Kernel Driver (efa.ko)        │ ← This Layer
│   - PCI driver                      │
│   - Queue management                │
│   - Memory registration              │
│   - Interrupt handling               │
├─────────────────────────────────────┤
│   PCI Subsystem                     │
│   - DMA API                         │
│   - IOMMU                           │
├─────────────────────────────────────┤
│   EFA Hardware                      │
│   - PCIe device                     │
│   - Queue pairs, completion queues  │
│   - DMA engine                      │
└─────────────────────────────────────┘
```

## Key Components

### 1. Driver Initialization (`efa_main.c`)

**PCI Device Table** - Supported EFA Generations:

```c
// From efa_main.c:32
static const struct pci_device_id efa_pci_tbl[] = {
    { PCI_VDEVICE(AMAZON, PCI_DEV_ID_EFA0_VF) },  // Gen 1: 0xefa0
    { PCI_VDEVICE(AMAZON, PCI_DEV_ID_EFA1_VF) },  // Gen 2: 0xefa1
    { PCI_VDEVICE(AMAZON, PCI_DEV_ID_EFA2_VF) },  // Gen 3: 0xefa2
    { PCI_VDEVICE(AMAZON, PCI_DEV_ID_EFA3_VF) },  // Gen 4: 0xefa3
    { }
};
```

**PCI Driver Structure**:

```c
static struct pci_driver efa_pci_driver = {
    .name           = "efa",
    .id_table       = efa_pci_tbl,
    .probe          = efa_device_init,   // Called on device discovery
    .remove         = efa_device_remove, // Called on device removal
};
```

**Device Initialization Flow**:

```c
// From efa_main.c (simplified)
static int efa_device_init(struct pci_dev *pdev,
                           const struct pci_device_id *id)
{
    struct efa_dev *dev;
    int err;

    // 1. Enable PCI device
    err = pci_enable_device_mem(pdev);

    // 2. Set DMA mask (64-bit addressing)
    err = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));

    // 3. Request PCI BARs
    //    BAR 0: Register space (MMIO)
    //    BAR 2: Memory space (for doorbells)
    err = pci_request_selected_regions(pdev, EFA_BASE_BAR_MASK, "efa");

    // 4. Map BARs to kernel virtual address
    dev->reg_bar = ioremap(pci_resource_start(pdev, EFA_REG_BAR),
                           pci_resource_len(pdev, EFA_REG_BAR));
    dev->mem_bar = ioremap(pci_resource_start(pdev, EFA_MEM_BAR),
                           pci_resource_len(pdev, EFA_MEM_BAR));

    // 5. Enable bus mastering (for DMA)
    pci_set_master(pdev);

    // 6. Initialize admin queue (control path)
    err = efa_com_admin_init(&dev->edev, &aenq_handlers);

    // 7. Query device attributes
    err = efa_com_get_device_attr(&dev->edev, &dev->dev_attr);

    // 8. Set up MSI-X interrupts
    err = efa_request_mgmnt_irq(dev);

    // 9. Register with RDMA core (creates /dev/infiniband/uverbsN)
    err = ib_register_device(&dev->ibdev, "efa_%d", &pdev->dev);

    // 10. Create sysfs entries
    efa_sysfs_init(dev);

    return 0;
}
```

### 2. Memory Regions (BAR Mapping)

EFA uses two PCI Base Address Registers (BARs):

```c
#define EFA_REG_BAR 0  // Register BAR (control/status registers)
#define EFA_MEM_BAR 2  // Memory BAR (doorbells, buffers)

// BAR 0: Control Registers
//   - Admin queue control
//   - Device stats
//   - Configuration registers
//   Mapped with: ioremap()
//   Access: readl()/writel() with barriers

// BAR 2: Doorbell/Memory Space
//   - Queue doorbells (SQ, RQ, CQ)
//   - Fast path (MMIO writes to ring doorbells)
//   Mapped with: ioremap_wc() (write-combining for performance)
//   Access: Direct MMIO writes (no barriers)
```

### 3. Verbs Operations (`efa_verbs.c` - 99KB!)

This is the **largest file** in the driver, implementing all RDMA verbs.

#### Device Context

```c
// From efa_verbs.c
int efa_alloc_ucontext(struct ib_ucontext *ibucontext,
                       struct ib_udata *udata)
{
    struct efa_ucontext *ucontext = to_eucontext(ibucontext);
    struct efa_dev *dev = to_edev(ibucontext->device);

    // Allocate UAR (User Access Region) for doorbells
    ucontext->db_bar_addr = dev->mem_bar;
    ucontext->db_bar_len = dev->db_bar_len;

    // Userspace will mmap() this later
    resp.cmds_supp_udata_mask |= EFA_USER_CMDS_SUPP_UDATA_QUERY_DEVICE;

    return ib_copy_to_udata(udata, &resp, sizeof(resp));
}
```

#### Protection Domain

```c
int efa_alloc_pd(struct ib_pd *ibpd, struct ib_udata *udata)
{
    struct efa_dev *dev = to_edev(ibpd->device);
    struct efa_pd *pd = to_epd(ibpd);

    // Allocate PD from hardware
    err = efa_com_alloc_pd(&dev->edev, &result);

    pd->pdn = result.pdn;  // Protection domain number

    return 0;
}
```

#### Memory Registration

```c
// Standard memory registration (CPU memory)
struct ib_mr *efa_reg_mr(struct ib_pd *ibpd, u64 start, u64 length,
                         u64 virt_addr, int access_flags,
                         struct ib_udata *udata)  // ([kernel/linux/efa/src/efa_verbs.c:2799](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.c#L2799))
{
    struct efa_dev *dev = to_edev(ibpd->device);
    struct efa_mr *mr;
    struct efa_com_reg_mr_params params = {};
    struct efa_com_reg_mr_result result = {};

    // 1. Allocate MR structure
    mr = kzalloc(sizeof(*mr), GFP_KERNEL);

    // 2. Pin user pages in memory
    mr->umem = ib_umem_get(ibpd->device, start, length, access_flags);

    // 3. Create page list for hardware
    //    Converts umem scatter-gather list to EFA format
    params.iova = virt_addr;
    params.length = length;
    params.page_shift = order_base_2(mr->umem->page_size);
    params.page_num = ib_umem_num_pages(mr->umem);

    // Build physical address list
    err = pbl_create(dev, &mr->pbl, mr->umem, params.page_num);

    // 4. Register with EFA hardware
    err = efa_com_register_mr(&dev->edev, &params, &result);

    // 5. Hardware returns keys
    mr->ibmr.lkey = result.l_key;
    mr->ibmr.rkey = result.r_key;

    return &mr->ibmr;
}

// DMA-BUF memory registration (GPU memory)
#ifdef HAVE_IB_REG_USER_MR_DMABUF
struct ib_mr *efa_reg_user_mr_dmabuf(struct ib_pd *ibpd, u64 start, u64 length,
                                     u64 virt_addr, int dmabuf_fd,
                                     int access_flags, struct ib_udata *udata)
{
    struct efa_dev *dev = to_edev(ibpd->device);
    struct efa_mr *mr;
    struct dma_buf *dmabuf;
    struct dma_buf_attachment *attachment;
    struct sg_table *sgt;

    mr = kzalloc(sizeof(*mr), GFP_KERNEL);

    // 1. Get dmabuf object
    dmabuf = dma_buf_get(dmabuf_fd);

    // 2. Attach this device to dmabuf
    attachment = dma_buf_attach(dmabuf, &dev->pdev->dev);

    // 3. Map dmabuf pages (get scatter-gather table)
    sgt = dma_buf_map_attachment(attachment, DMA_BIDIRECTIONAL);

    // 4. Create page list from scatter-gather table
    err = pbl_create_from_sgt(dev, &mr->pbl, sgt, start, length);

    // 5. Register with hardware (same as regular MR)
    err = efa_com_register_mr(&dev->edev, &params, &result);

    mr->ibmr.lkey = result.l_key;
    mr->ibmr.rkey = result.r_key;

    // Store dmabuf info for cleanup
    mr->dmabuf = dmabuf;
    mr->dmabuf_attach = attachment;
    mr->dmabuf_sgt = sgt;

    return &mr->ibmr;
}
#endif

// Memory deregistration
int efa_dereg_mr(struct ib_mr *ibmr, struct ib_udata *udata)  // ([kernel/linux/efa/src/efa_verbs.c:3065](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.c#L3065))
{
    struct efa_dev *dev = to_edev(ibmr->device);
    struct efa_mr *mr = to_emr(ibmr);

    // 1. Deregister from hardware
    efa_com_dereg_mr(&dev->edev, &params);

    // 2. Clean up dmabuf (if applicable)
#ifdef HAVE_IB_REG_USER_MR_DMABUF
    if (mr->dmabuf) {
        dma_buf_unmap_attachment(mr->dmabuf_attach, mr->dmabuf_sgt,
                                DMA_BIDIRECTIONAL);
        dma_buf_detach(mr->dmabuf, mr->dmabuf_attach);
        dma_buf_put(mr->dmabuf);
    }
#endif

    // 3. Release pinned pages
    if (mr->umem)
        ib_umem_release(mr->umem);

    // 4. Free page list
    pbl_destroy(dev, &mr->pbl);

    kfree(mr);
    return 0;
}
```

#### Queue Pair Creation

```c
int efa_create_qp(struct ib_qp *ibqp, struct ib_qp_init_attr *init_attr,
                  struct ib_udata *udata)  // ([kernel/linux/efa/src/efa_verbs.c:1216](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.c#L1216))
{
    struct efa_dev *dev = to_edev(ibqp->pd->device);
    struct efa_qp *qp = to_eqp(ibqp);
    struct efa_com_create_qp_params params = {};
    struct efa_com_create_qp_result result = {};

    // 1. Allocate queue buffers in userspace
    //    Userspace allocates SQ/RQ buffers, tells us via udata

    err = ib_copy_from_udata(&cmd, udata, sizeof(cmd));

    // 2. Create queue pair in hardware
    params.qp_type = init_attr->qp_type;  // UD (unreliable datagram)
    params.send_cq_idx = to_ecq(init_attr->send_cq)->cq_idx;
    params.recv_cq_idx = to_ecq(init_attr->recv_cq)->cq_idx;
    params.sq_depth = init_attr->cap.max_send_wr;
    params.rq_depth = init_attr->cap.max_recv_wr;

    err = efa_com_create_qp(&dev->edev, &params, &result);

    // 3. Hardware returns QP number
    qp->qp_handle = result.qp_handle;
    qp->qp_num = result.qp_num;

    // 4. Set up doorbell addresses (for userspace mmap)
    qp->sq_db_offset = result.sq_db_offset;
    qp->rq_db_offset = result.rq_db_offset;
    qp->llq_desc_offset = result.llq_descriptors_offset;

    // 5. Return info to userspace
    resp.qp_handle = result.qp_handle;
    resp.qp_num = result.qp_num;
    resp.sq_db_offset = result.sq_db_offset;
    resp.rq_db_offset = result.rq_db_offset;

    err = ib_copy_to_udata(udata, &resp, sizeof(resp));

    return 0;
}
```

#### Completion Queue Creation

```c
int efa_create_cq(struct ib_cq *ibcq, const struct ib_cq_init_attr *attr,
                  struct ib_udata *udata)  // ([kernel/linux/efa/src/efa_verbs.c:2147](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.c#L2147))
{
    struct efa_dev *dev = to_edev(ibcq->device);
    struct efa_cq *cq = to_ecq(ibcq);
    struct efa_com_create_cq_params params = {};
    struct efa_com_create_cq_result result = {};

    // 1. Get CQ size from userspace
    err = ib_copy_from_udata(&cmd, udata, sizeof(cmd));

    // 2. Create CQ in hardware
    params.cq_depth = attr->cqe;
    params.num_sub_cqs = cmd.num_sub_cqs;

    err = efa_com_create_cq(&dev->edev, &params, &result);

    // 3. Store CQ info
    cq->cq_idx = result.cq_idx;
    cq->cq_handle = result.cq_handle;

    // 4. Set up CQ buffer info for userspace mmap
    resp.cq_idx = result.cq_idx;
    resp.cq_actual_depth = result.actual_depth;

    err = ib_copy_to_udata(udata, &resp, sizeof(resp));

    // 5. Register for interrupt notification
#ifdef HAVE_XARRAY
    xa_store(&dev->cqs_xa, cq->cq_idx, cq, GFP_KERNEL);
#else
    dev->cqs_arr[cq->cq_idx] = cq;
#endif

    return 0;
}
```

### 3b. GPU & Accelerator Peer Memory (P2P Subsystem)

Registering **device** memory (GPU/accelerator) for RDMA is fundamentally different
from host memory: the pages live behind another device's BAR, not in system RAM, so
the EFA driver cannot just `get_user_pages()`. The driver has a dedicated peer-memory
(P2P) subsystem (`efa_p2p.c`, `efa_p2p.h`) that abstracts *how* device pages are
pinned and translated to DMA addresses, with one pluggable provider per accelerator
family.

**Provider model** — each provider implements a small ops vtable
([`struct efa_p2p_ops`](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_p2p.h)):

```c
struct efa_p2p_ops {
    char *(*get_provider_string)(void);
    struct efa_p2pmem *(*try_get)(struct efa_dev *dev, u64 ticket, u64 start, u64 length);
    int   (*to_page_list)(struct efa_dev *dev, struct efa_p2pmem *p2pmem, u64 *page_list);
    void  (*release)(struct efa_dev *dev, struct efa_p2pmem *p2pmem, bool in_cb);
    unsigned int (*get_page_size)(struct efa_dev *dev, struct efa_p2pmem *p2pmem);
};
```

Registered providers (`efa_p2p_init()` → `p2p_providers_init()`):

| Provider | Backing | Source |
|----------|---------|--------|
| `EFA_P2P_PROVIDER_NVMEM_V1` | NVIDIA `nvidia_p2p_*` API, v1 (`nv-p2p.h`) | `efa_nvmem_v1.c` |
| `EFA_P2P_PROVIDER_NVMEM_V2` | NVIDIA `nvidia_p2p_*` API, v2 (`nv-p2p_v2.h`) | `efa_nvmem_v2.c` |
| `EFA_P2P_PROVIDER_NEURON` | AWS Neuron `neuron_p2p_*` API (`neuron_p2p.h`) | `efa_neuronmem.c` |

**Registration flow** (`efa_p2p_get()`): on a device-memory `reg_mr`, the driver walks
the provider array and calls each provider's `try_get()` until one claims the address
range (NVIDIA providers recognize CUDA allocations; the Neuron provider recognizes
Trainium/Inferentia memory). The winning provider pins the device pages and returns an
`efa_p2pmem` handle; `to_page_list()` then produces the DMA addresses the driver
programs into the hardware page tables via the `reg_mr` admin command.

**The ticket mechanism** — device drivers can revoke peer memory asynchronously (e.g.
the GPU frees the allocation). Each `efa_p2pmem` is tagged with a monotonic **ticket**
(`next_p2p_ticket`) and tracked on a global `p2p_list`. When the GPU driver invokes the
free callback, EFA looks the mapping up *by ticket* (`ticket_to_p2p()`) and releases it
(`efa_p2p_put(ticket, in_cb=true)`), tearing down the MR safely even though the
callback runs in the GPU driver's context. The ticket indirection avoids dereferencing
a pointer that the other driver may already be freeing.

**Relationship to DMA-BUF** — there are two independent paths to register GPU memory:

1. **DMA-BUF** (`efa_reg_user_mr_dmabuf`, see below): the modern, vendor-neutral path.
   Userspace passes a `dmabuf_fd`; the driver attaches and maps it through the kernel
   `dma_buf` framework. No EFA-specific peer provider is involved. Requires
   `HAVE_IB_REG_USER_MR_DMABUF` and a working exporter (and is gated off on EFA Gen 1–3
   due to a firmware page-merging issue — see `dmabuf-gpu-memory.md`).
2. **Peer-memory providers** (this subsystem): the legacy/vendor path used when DMA-BUF
   is unavailable — NVIDIA `nvidia_p2p` (a.k.a. nvidia-peermem) or Neuron `neuron_p2p`.

libfabric's EFA provider chooses DMA-BUF when viable and falls back to the peer-memory
path otherwise; both ultimately produce a device-page DMA list that the same `reg_mr`
admin command installs in hardware.

### 4. mmap Support (`efa_data_verbs.c`)

Userspace needs direct access to queue buffers and doorbells. This is achieved via `mmap()`:

```c
// From efa_data_verbs.c
int efa_mmap(struct ib_ucontext *ibucontext, struct vm_area_struct *vma)
{
    struct efa_ucontext *ucontext = to_eucontext(ibucontext);
    struct efa_dev *dev = to_edev(ibucontext->device);
    u64 key = vma->vm_pgoff << PAGE_SHIFT;
    size_t length = vma->vm_end - vma->vm_start;

    switch (key & EFA_MMAP_PAGE_MASK) {
    case EFA_MMAP_DB_BAR_MEMORY_FLAG:
        // Map doorbell BAR (write-combining for performance)
        return efa_mmap_io(dev, vma, dev->mem_bar, length,
                          pgprot_writecombine(vma->vm_page_prot));

    case EFA_MMAP_REG_BAR_MEMORY_FLAG:
        // Map register BAR (non-cached)
        return efa_mmap_io(dev, vma, dev->reg_bar, length,
                          pgprot_noncached(vma->vm_page_prot));

    case EFA_MMAP_IO_WC:
        // Queue buffers (write-combining)
        return efa_mmap_io(dev, vma, key & ~EFA_MMAP_PAGE_MASK, length,
                          pgprot_writecombine(vma->vm_page_prot));

    case EFA_MMAP_IO_NC:
        // Control buffers (non-cached)
        return efa_mmap_io(dev, vma, key & ~EFA_MMAP_PAGE_MASK, length,
                          pgprot_noncached(vma->vm_page_prot));

    default:
        return -EINVAL;
    }
}

static int efa_mmap_io(struct efa_dev *dev, struct vm_area_struct *vma,
                       u64 phys_addr, size_t length, pgprot_t prot)
{
    // Map physical memory into userspace
    return remap_pfn_range(vma,
                          vma->vm_start,
                          phys_addr >> PAGE_SHIFT,
                          length,
                          prot);
}
```

**Result**: Userspace can directly write to:
- Send/receive queues (no syscalls!)
- Doorbell registers (notify hardware of new work)
- Completion queues (poll for completions)

### 5. Interrupt Handling

```c
// MSI-X interrupt handler
static irqreturn_t efa_intr_msix_comp(int irq, void *data)
{
    struct efa_eq *eq = data;

    // Process event queue entries
    efa_com_eq_comp_intr_handler(eq);

    return IRQ_HANDLED;
}

// Process completion events
static void efa_process_comp_eqe(struct efa_dev *dev,
                                 struct efa_admin_eqe *eqe)
{
    u16 cqn = eqe->u.comp_event.cqn;
    struct efa_cq *cq;

    // Find CQ by index
    cq = xa_load(&dev->cqs_xa, cqn);

    // Call completion handler (wakes up polling thread)
    cq->ibcq.comp_handler(&cq->ibcq, cq->ibcq.cq_context);
}
```

### 6. Admin Queue Communication (`efa_com.c`, `efa_com_cmd.c`)

The admin queue is used for control path operations (create QP, register MR, etc.):

```c
// Admin queue descriptor
struct efa_admin_aq_entry {
    u8 opcode;          // Command opcode
    u8 flags;
    u16 command_id;
    u32 aq_common_descriptor;

    union {
        struct efa_admin_create_qp_cmd create_qp;
        struct efa_admin_reg_mr_cmd reg_mr;
        struct efa_admin_create_cq_cmd create_cq;
        // ... other commands
    } u;
};

// Send command to admin queue
static int efa_com_admin_q_comp_intr_handler(struct efa_com_admin_queue *aq)
{
    u16 comp_num = 0;
    u16 consumer = aq->cc;

    // Process completed admin commands
    while (comp_num < aq->depth) {
        struct efa_admin_acq_entry *cqe = &aq->cq.entries[consumer];

        if (!efa_com_cq_empty(cqe))
            break;

        // Wake up waiting thread
        complete(&comp_ctx->wait_event);

        consumer = (consumer + 1) % aq->depth;
        comp_num++;
    }

    aq->cc = consumer;
    aq->cq.cc = consumer;

    return comp_num;
}
```

#### Admin Queue Protocol Internals

The admin queue (AQ) is a classic **producer/consumer ring with a phase (ownership)
bit**, the same pattern EFA uses for its data-path queues — so understanding it here
explains the hardware queue model generally.

**Three rings**, each set up in `efa_com.c` with its base address and depth programmed
into registers via `writel()` on BAR 0:
- **AQ** (admin submission queue) — `EFA_REGS_AQ_BASE_{LO,HI}_OFF`, caps in `AQ_CAPS`.
- **ACQ** (admin completion queue) — `EFA_REGS_ACQ_BASE_*`.
- **AENQ** (async event notification queue) — device-initiated events (link/health).

All three start with `phase = 1` and a producer/consumer counter at 0.

**Submitting a command** (`__efa_com_submit_admin_cmd`):
```c
pi = aq->sq.pc & queue_size_mask;          // ring slot
// command_id packs ctx_id in the LSBs + entropy from pc in the MSBs
cmd_id  = ctx_id;
cmd_id |= aq->sq.pc << ilog2(aq->depth);
set_field(cmd, PHASE, aq->sq.phase);       // stamp current phase into the descriptor
// ... copy command into ring[pi] ...
aq->sq.pc++;
if ((aq->sq.pc & queue_size_mask) == 0)
    aq->sq.phase = !aq->sq.phase;          // flip phase on wrap
writel(aq->sq.pc, aq->sq.db_addr);         // ring the doorbell (MMIO write)
```
Key points:
- The **doorbell** is a single MMIO `writel` of the new producer counter to the
  queue's BAR-mapped `db_addr`; this is what tells the device "new work is posted."
  No barrier is needed because `writel` to device memory is already ordered.
- Each in-flight command gets a **`comp_ctx`** drawn from a pool (`comp_ctx_pool`),
  keyed by `command_id`. The submitter blocks on `comp_ctx->wait_event` (or polls).
  The `command_id` carries both the context-pool index (LSBs) and entropy bits from
  the producer counter (MSBs) so a stale completion can't be mismatched to a reused slot.

**Consuming completions** (`efa_com_handle_admin_completion`):
```c
phase = aq->cq.phase;
while (read_field(cqe[ci], PHASE) == phase) {   // descriptor owned by SW?
    // ... match cqe->command_id -> comp_ctx, store status, complete()
    if (++ci wrapped) phase = !phase;           // flip on wrap
}
```
The consumer never reads a length/valid flag — it reads the descriptor's **phase bit**
and compares it to the phase it expects for the current pass. Because the producer
stamps the current phase and flips on each wrap, a descriptor is "ready" exactly when
its phase matches the consumer's expected phase. This lets the device and driver share
a ring with **no extra valid-flag write and no head/tail register round-trip** — the
hardware just writes the completion (with the right phase) into the next slot. This
ownership-bit scheme is the same mechanism the **GDAKI** GPU data path uses (see
`ofi-plugin-protocols.md` → GIN Modes), where the GPU kernel polls SQ/CQ phase bits
directly.

```c
// Send command to admin queue (legacy excerpt)
static int efa_com_admin_q_comp_intr_handler(struct efa_com_admin_queue *aq)
```

## Key Data Structures

```c
// Main device structure
struct efa_dev {  // ([kernel/linux/efa/src/efa.h:53-79](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa.h#L53-L79))
    struct ib_device ibdev;               // RDMA core device
    struct pci_dev *pdev;                 // PCI device
    struct efa_com_dev edev;              // Admin queue/command interface

    void __iomem *reg_bar;                // BAR 0 (registers)
    void __iomem *mem_bar;                // BAR 2 (doorbells)

    struct efa_dev_attr dev_attr;         // Device capabilities
    struct efa_stats stats;               // Statistics

#ifdef HAVE_XARRAY
    struct xarray cqs_xa;                 // CQ index → CQ mapping
#else
    struct efa_cq **cqs_arr;
#endif

    u64 db_bar_addr;                      // Doorbell BAR physical address
    u64 db_bar_len;
};

// Queue Pair
struct efa_qp {  // ([kernel/linux/efa/src/efa.h:240-263](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa.h#L240-L263))
    struct ib_qp ibqp;
    u32 qp_handle;
    u32 qp_num;
    u32 sq_db_offset;                     // Send queue doorbell offset
    u32 rq_db_offset;                     // Recv queue doorbell offset
    // ...
};

// Completion Queue
struct efa_cq {  // ([kernel/linux/efa/src/efa.h:171-188](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa.h#L171-L188))
    struct ib_cq ibcq;
    u16 cq_idx;
    u32 cq_handle;
    // ...
};

// Memory Region
struct efa_mr {  // ([kernel/linux/efa/src/efa.h:149-157](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa.h#L149-L157))
    struct ib_mr ibmr;
    struct ib_umem *umem;                 // Pinned user pages
    struct efa_mr_pbl pbl;                // Page buffer list

#ifdef HAVE_IB_REG_USER_MR_DMABUF
    struct dma_buf *dmabuf;               // DMA-BUF object (if GPU memory)
    struct dma_buf_attachment *dmabuf_attach;
    struct sg_table *dmabuf_sgt;
#endif
};
```

## Device Capabilities

From `efa_com_get_device_attr()`:

```c
struct efa_dev_attr {
    u32 max_qp;                           // Max queue pairs (~1024)
    u32 max_cq;                           // Max completion queues (~1024)
    u32 max_mr;                           // Max memory regions (~4096)
    u32 max_mr_pages;                     // Max pages per MR (~512K)
    u32 max_qp_wr;                        // Max WRs per QP (~8192)
    u32 max_cq_depth;                     // Max CQ depth (~16384)
    u32 max_wr_send_sges;                 // Max SG entries per send (~2)
    u32 max_wr_recv_sges;                 // Max SG entries per recv (~2)
    u64 max_rdma_size;                    // Max RDMA transfer (~1GB)
    u16 device_caps;                      // Device capabilities bitmask
};
```

## Sysfs Interface

```bash
# EFA device sysfs attributes
/sys/class/infiniband/efa_0/
├── device -> ../../../0000:00:06.0    # PCI device link
├── fw_ver                              # Firmware version
├── node_guid                           # Device GUID
├── node_type                           # Node type (RNIC)
├── sys_image_guid                      # System image GUID
└── ports/1/                            # Port attributes
    ├── state                           # Port state (ACTIVE)
    ├── phys_state                      # Physical state
    ├── rate                            # Link rate
    └── gid_attrs/                      # GID attributes
```

## Debugging

### Enable Driver Debug

```bash
# Enable EFA driver debug messages
echo 'module efa +p' > /sys/kernel/debug/dynamic_debug/control

# Or via modprobe
modprobe efa debug_mask=0xffffffff

# Check dmesg for driver messages
dmesg | grep efa
```

### Check Driver Status

```bash
# Check if driver loaded
lsmod | grep efa

# Check PCI device
lspci -vv | grep -A 20 EFA

# Check device attributes
ibv_devinfo -d efa_0

# Check verbs device
ls -la /dev/infiniband/
# Should see uverbs0, uverbs0a, etc.
```

### Trace Memory Registration

```bash
# Enable MR registration tracing
echo 1 > /sys/kernel/debug/tracing/events/ib_core/ib_mr_reg/enable
echo 1 > /sys/kernel/debug/tracing/events/ib_core/ib_mr_dereg/enable

# Watch trace
cat /sys/kernel/debug/tracing/trace_pipe
```

### Monitor Interrupts

```bash
# Show EFA interrupts
cat /proc/interrupts | grep efa

# Interrupt counts should increase during traffic
watch -n 1 'cat /proc/interrupts | grep efa'
```

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| `efa_reg_mr()` | 100-500 μs | Pins pages, creates IOMMU mapping |
| `efa_create_qp()` | 200-800 μs | Allocates kernel resources |
| `efa_create_cq()` | 100-400 μs | Allocates CQ buffer |
| Admin queue command | 10-50 μs | Round-trip to hardware |
| Interrupt handling | 1-5 μs | From hardware → userspace wakeup |
| mmap() | 5-20 μs | One-time setup per region |

**Key Insight**: After initial setup (which is slow), data path operations are **entirely in userspace** (no syscalls).

## Key Source Files

```
efa_main.c (25KB)         - PCI driver, device init/cleanup
efa_verbs.c (99KB!)       - RDMA verbs implementation
efa_com.c (34KB)          - Admin queue communication
efa_com_cmd.c (26KB)      - Admin commands
efa_data_verbs.c (18KB)   - mmap, doorbell handling
efa.h (12KB)              - Main header with data structures
efa_io_defs.h (10KB)      - Hardware I/O definitions
efa_admin_cmds_defs.h (27KB) - Admin command structures
efa_p2p.c (3KB)           - Peer-to-peer (GPUDirect)
efa_neuronmem.c (4KB)     - AWS Neuron memory support
efa_sysfs.c (1KB)         - Sysfs attributes
```

## Summary

| Component | Purpose | Interface |
|-----------|---------|-----------|
| **PCI Driver** | Device discovery and initialization | `pci_driver` |
| **Verbs** | RDMA operations (reg_mr, create_qp, etc.) | `ib_device_ops` |
| **Admin Queue** | Control path (commands to hardware) | MMIO + interrupts |
| **mmap** | Zero-copy queue buffer access | `efa_mmap()` |
| **Interrupts** | Completion notifications | MSI-X |
| **DMA-BUF** | GPU memory registration | `dma_buf` API |

**Key Takeaways**:
1. EFA driver is a **PCI driver** that implements **RDMA verbs**
2. Userspace accesses queues via **mmap()** for zero-copy, zero-syscall I/O
3. Control path uses **admin queue** (slow, 10-50 μs per command)
4. Data path is **entirely userspace** (fast, no kernel involvement)
5. Supports **DMA-BUF** for GPU memory (kernel 5.12+, EFA Gen 4+)
6. Memory registration is **expensive** (100-500 μs) - userspace must cache!
7. After setup, posting sends/recvs is just **memory writes** + **doorbell ring**

**Related Documentation**:
- [rdma-core-and-verbs.md](rdma-core-and-verbs.md) - Userspace library that talks to this driver
- [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md) - DMA-BUF implementation details
- [efa-driver.md](efa-driver.md) - Higher-level EFA driver overview
- [rdma-memreg.md](rdma-memreg.md) - Memory registration from all layers
