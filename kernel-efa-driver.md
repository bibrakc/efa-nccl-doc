# Linux Kernel EFA Driver

## Overview

The **EFA kernel driver** is a Linux kernel module that provides the interface between userspace RDMA applications and the EFA hardware. It implements the kernel-side of the RDMA verbs API and manages hardware resources.

**GitHub**: [kernel/linux/efa/](https://github.com/amzn/amzn-drivers/tree/master/kernel/linux/efa)
(out-of-tree build; in-tree it lives at `drivers/infiniband/hw/efa/`).

**Driver Version**: **r3.3.0** (`DRV_MODULE_VER_MAJOR/MINOR/SUBMINOR = 3/3/0` in
[efa_main.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_main.c) lines ~42-44;
authoritative change list in
[RELEASENOTES.md](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/RELEASENOTES.md)).

> **Source lookups:** this document explains mechanism and records defaults, corrections and history. For current function bodies, call graphs and blast radius, query the code graph (`codegraph explore <symbol>`) rather than trusting a pasted copy here.

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

Practical consequence: behaviours like the **historical Gen 1–3 DMA-BUF page-merging issue**
are firmware characteristics observable from the driver, not bugs fixable in the open driver
source. (That particular one no longer gates DMA-BUF — the device-id check was removed in
aws-ofi-nccl `0f285d5` — but it remains a good example of the class.) When debugging, the open
code tells you *what was requested of the device*; the
device's internal handling is inferred from completions, counters, and errors.

## What's New in EFA Driver r3.3.0

The r3.3.0 release (change list in
[RELEASENOTES.md](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/RELEASENOTES.md))
adds several capabilities directly relevant to high-performance and GPU-initiated
networking. All items below are verified in
[kernel/linux/efa/src/](https://github.com/amzn/amzn-drivers/tree/master/kernel/linux/efa/src):

- **Completion Counters (hardware event counters).** New admin opcodes
  `EFA_ADMIN_CREATE_EVENT_COUNTER` (25), `DESTROY_EVENT_COUNTER` (26),
  `ATTACH_EVENT_COUNTER` (27), `MODIFY_EVENT_COUNTER` (28) and
  `DETACH_EVENT_COUNTER` (29) — opcodes 25-29, the new
  `EFA_ADMIN_MAX_OPCODE`, in
  [efa_admin_cmds_defs.h](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_admin_cmds_defs.h))
  and `efa_com_create_event_counter()` in
  [efa_com_cmd.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_com_cmd.c).
  Device attributes now report `max_event_counters`, `event_counter_max_val` and
  `supported_event_counter_qp_events` (QUEUE_ATTR_2). **GDAKI / GPU-initiated data
  paths use these hardware completion counters** to observe completions without a
  host CPU polling loop.
- **64-bit work request IDs for QP/SQ creation.** The device advertises
  `EFA_QUERY_DEVICE_CAPS_SQ_64_BIT_REQ_ID`, exposed to userspace as
  `EFADV_WQ_CAPS_64_BIT_REQ_ID`. When enabled, the full 64-bit `wr_id` is carried
  in the WQE (`req_id` + `req_id_ex`) and returned in the completion, removing the
  need for a wrid-index translation pool. (Consumed by libfabric Data Path Direct —
  see [rdma-core-and-verbs.md](rdma-core-and-verbs.md).)
- **128-byte send-queue WQEs.** `efa_calc_sq_wqe_size_kernel()` in
  [efa_verbs.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_verbs.c)
  can return `EFA_IO_TX_DESC_SIZE_128`; gated by `EFA_QUERY_DEVICE_CAPS_DATA_POLLING_128`.
  Wider WQEs enlarge inline capacity (32 → 80 bytes) and enable inline RDMA WRITE.
- **Inline WRITE operation support.** `sq->inline_write_enabled` is set when
  `sq->wqe_size > EFA_IO_TX_DESC_SIZE_64` (efa_verbs.c).
- **New 0xefa4 PCI device ID** (`PCI_DEV_ID_EFA4_VF`, see PCI device table below).
- **800 and 1600 Gbps link speeds.** `efa_link_gbps_to_speed_and_width()`
  (efa_verbs.c lines ~417-439) maps ≥1600 Gbps → `IB_SPEED_XDR`, ≥800 → `IB_SPEED_NDR`.
  A new **query-port-speed verb** `efa_query_port_speed()` (efa_verbs.c line ~487)
  reports the raw link speed (`max_link_speed_gbps * 10`, in units of 100 Mbps).
- **Checksum validation on admin responses.** `efa_com_calc_crc16_checksum()`
  (CRC16 over the admin SQ/CQ entry) and `efa_com_cqe_checksum_valid()` in
  [efa_com.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_com.c);
  enabled (`cq->validate_checksum = true`) when the device API version is at least
  `EFA_CRC_MIN_API_VERSION`.
- **>4 GB MR page size.** MR registration now carries an extended page-shift field,
  allowing page sizes larger than 4 GB for large physically-contiguous regions.
- **AH device-object reuse via an rhashtable AH cache.** New
  [efa_ah_cache.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_ah_cache.c)
  / `.h`: identical destination addresses share one hardware AH object, reducing AH
  table pressure and create/destroy admin traffic.
- **Admin queue v2 / generalized admin SQ.** 128-byte admin v2 SQ entry, a
  generalized admin SQ, and the admin command payload decoupled from the admin
  header (efa_com.c / efa_admin_cmds_defs.h).
- **Core-utility backports.** `umem` pinning from buffer-descriptor attributes,
  `ib_umem_get_cq_buf` for CQ buffer pinning, and the `ib_umem_get` →
  `ib_umem_get_va` rename; PBL chunk-length computation fix during MR registration;
  alignment to mainline 7.2 kernel changes.

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

## EFA Device Model (PCIe)

Example `lspci` view of an EFA function:

```
00:06.0 Ethernet controller: Amazon.com, Inc. Elastic Fabric Adapter (EFA)
    Subsystem: Amazon.com, Inc. Device efa0
    Flags: bus master, fast devsel, latency 0
    Memory at 82000000 (32-bit, non-prefetchable) [size=32M]   # BAR0 registers
    Memory at 83000000 (32-bit, non-prefetchable) [size=256K]  # BAR2 doorbells
    Capabilities: [40] Express Endpoint, MSI 00
    Capabilities: [100] Advanced Error Reporting
    Capabilities: [150] Device Serial Number
```

- **BAR0** — device registers (MMIO), mapped non-cached; admin-queue control, stats,
  configuration registers. Access via `readl()`/`writel()` with barriers.
- **BAR2** — doorbell / memory space (SQ/RQ/CQ doorbells, queue buffers), mapped
  write-combining (`ioremap_wc`) for the fast path. Doorbell rings are direct MMIO
  writes with no barrier (a `writel` to device memory is already ordered).
- **MSI-X** interrupt vectors (see Interrupts vs Polling below).
- Underlying link is **PCIe Gen4 x16** — ~25 GB/s bidirectional, which is the
  practical data-path ceiling per adapter.

**Supported EFA generations** — the PCI device table
([efa_main.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_main.c) lines ~27-38)
enumerates VF device IDs:

| Device ID | Generation | Note |
|-----------|------------|------|
| `0xefa0` (`PCI_DEV_ID_EFA0_VF`) | Gen 1 | |
| `0xefa1` (`PCI_DEV_ID_EFA1_VF`) | Gen 2 | p4d/p4de |
| `0xefa2` (`PCI_DEV_ID_EFA2_VF`) | Gen 3 | |
| `0xefa3` (`PCI_DEV_ID_EFA3_VF`) | Gen 4 | |
| `0xefa4` (`PCI_DEV_ID_EFA4_VF`) | — | **new in r3.3.0** |

> Source: `efa_main.c` — `efa_pci_tbl[]`, `efa_pci_driver`, `efa_device_init()`
> (probe) / `efa_device_remove()`. Use `codegraph explore efa_device_init` for the
> current probe flow (PCI enable → 64-bit DMA mask → request/`ioremap` BAR0 & BAR2 →
> `pci_set_master` → `efa_com_admin_init` → `efa_com_get_device_attr` →
> `efa_request_mgmnt_irq` → `ib_register_device` → `efa_sysfs_init`).

## Key Components

### 1. Verbs Operations (`efa_verbs.c`)

The largest file in the driver (~99 KB), implementing all RDMA verbs: device context
and UAR allocation, protection domains, memory registration, QP/CQ creation. All of
these are control-path setup; once queues and doorbells are mapped into userspace the
data path runs without the kernel.

> Source: `efa_verbs.c` — `efa_alloc_ucontext()`, `efa_alloc_pd()`, `efa_reg_mr()`
> (CPU memory), `efa_reg_user_mr_dmabuf()` (GPU memory), `efa_dereg_mr()`,
> `efa_create_qp()`, `efa_create_cq()`. Use `codegraph explore <symbol>` for current
> bodies.

Mechanism notes worth keeping:

- **`efa_reg_mr()`** pins user pages with `ib_umem_get()`, builds a physical-address
  list (`pbl_create`), issues the `reg_mr` admin command, and returns hardware
  `lkey`/`rkey`. Memory registration is **expensive** (100–500 μs) — userspace must
  cache registrations.
- **`efa_reg_user_mr_dmabuf()`** is the DMA-BUF (GPU memory) path: `dma_buf_get(fd)` →
  `dma_buf_attach()` → `dma_buf_map_attachment()` → build PBL from the scatter-gather
  table → same `reg_mr` admin command. Compiled under **`HAVE_MR_DMABUF`**, with
  **`HAVE_IB_UMEM_DMABUF_PINNED`** selecting the pinned-import variant. `efa_dereg_mr()`
  tears the DMA-BUF attachment down under the same `HAVE_MR_DMABUF` guard.
- **`efa_create_qp()`** — userspace allocates the SQ/RQ buffers and passes them via
  `udata`; the driver issues `create_qp` (type `EFA_ADMIN_QP_TYPE_SRD` or UD) and
  returns the QP number plus **SQ/RQ doorbell offsets** and the LLQ descriptor offset
  for userspace to `mmap()`.
- **`efa_create_cq()`** — issues `create_cq`, returns the CQ index and actual depth,
  and registers the CQ for interrupt notification (`xa_store` into `cqs_xa` when
  `HAVE_XARRAY`, else the `cqs_arr` fallback).

### 2. GPU & Accelerator Peer Memory (P2P Subsystem)

Registering **device** memory (GPU/accelerator) for RDMA is fundamentally different
from host memory: the pages live behind another device's BAR, not in system RAM, so
the EFA driver cannot just `get_user_pages()`. The driver has a dedicated peer-memory
(P2P) subsystem (`efa_p2p.c`, `efa_p2p.h`) that abstracts *how* device pages are
pinned and translated to DMA addresses, with one pluggable provider per accelerator
family.

> The deep treatment of the GPU kernel path — the NVIDIA/Neuron provider ends,
> callback timing, and how this composes with the CUDA/ROCm side — lives in
> [gpu-memory-kernel-path.md](gpu-memory-kernel-path.md). This section is the EFA-side
> overview only.

**Provider model** — each provider implements a small ops vtable
([`struct efa_p2p_ops`](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_p2p.h));
the shape of the vtable *is* the point — it defines the entire contract a peer provider
must satisfy:

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

1. **DMA-BUF** (`efa_reg_user_mr_dmabuf`, above): the modern, vendor-neutral path.
   Userspace passes a `dmabuf_fd`; the driver attaches and maps it through the kernel
   `dma_buf` framework. No EFA-specific peer provider is involved. Compiled under
   `HAVE_MR_DMABUF` (with `HAVE_IB_UMEM_DMABUF_PINNED` selecting the pinned-import variant)
   and requires a working exporter. **Not gated by EFA hardware generation** — the device-id
   check that used to disable DMA-BUF on Gen 1–3 over a firmware page-merging issue was
   removed in aws-ofi-nccl commit `0f285d5` ("dma-buf: Don't disable dma-buf on EFAv1-3").
   See [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md).
2. **Peer-memory providers** (this subsystem): the legacy/vendor path used when DMA-BUF
   is unavailable — NVIDIA `nvidia_p2p` or Neuron `neuron_p2p`.

   > **`nvidia_p2p` is not `nvidia-peermem`.** These are different mechanisms and conflating
   > them sends you to the wrong place when debugging. EFA calls the `nvidia_p2p_*` functions
   > from `nv-p2p.h` directly (or through the AWS `efa_nv_peermem` GPL shim).
   > `nvidia-peermem.ko` is NVIDIA's *separate* bridge that registers with
   > `ib_register_peer_memory_client()`, the MLNX_OFED peer-direct interface — an API that is
   > not in mainline `ib_core`. The EFA driver contains **zero** references to
   > `peer_memory_client`, and does not need `nvidia-peermem` loaded.
   > See [gpu-memory-kernel-path.md](gpu-memory-kernel-path.md).

libfabric's EFA provider chooses DMA-BUF when viable and falls back to the peer-memory
path otherwise; both ultimately produce a device-page DMA list that the same `reg_mr`
admin command installs in hardware.

### 3. mmap Support (`efa_data_verbs.c`)

Userspace needs direct access to queue buffers and doorbells. `efa_mmap()` dispatches on
a key packed into `vma->vm_pgoff` and maps the requested region with the right caching:

- **`EFA_MMAP_DB_BAR_MEMORY_FLAG`** — doorbell BAR, **write-combining**
  (`pgprot_writecombine`).
- **`EFA_MMAP_REG_BAR_MEMORY_FLAG`** — register BAR, **non-cached** (`pgprot_noncached`).
- **`EFA_MMAP_IO_WC`** — queue buffers, **write-combining**.
- **`EFA_MMAP_IO_NC`** — control buffers, **non-cached**.

The caching mode is the point: doorbells and queue buffers are write-combining so batched
MMIO stores coalesce, while register/control mappings are non-cached for ordering. The
mapping itself is a `remap_pfn_range()` of device physical memory into the process.

> Source: `efa_data_verbs.c` — `efa_mmap()`, `efa_mmap_io()`. Use
> `codegraph explore efa_mmap` for the current body.

**Result**: userspace can directly write send/receive queues (no syscalls), ring
doorbell registers, and poll completion queues.

### 4. Interrupt Handling

EFA uses **MSI-X** for admin/async events and for optional CQ completion notification.
The comp-event handler resolves the CQ by index (`xa_load(&dev->cqs_xa, cqn)`) and calls
its `comp_handler`, which wakes any thread blocked in `ibv_get_cq_event()`. See Interrupts
vs Polling below for when this fires at all.

> Source: `efa_main.c` / `efa_com.c` — `efa_intr_msix_comp()`,
> `efa_process_comp_eqe()`, `efa_com_eq_comp_intr_handler()`. Use
> `codegraph explore efa_process_comp_eqe` for the current body.

### 5. Admin Queue Communication (`efa_com.c`, `efa_com_cmd.c`)

The admin queue is the control path (create QP, register MR, etc.).

> Source: `efa_com.c` / `efa_com_cmd.c` — `efa_com_admin_init()`,
> `__efa_com_submit_admin_cmd()`, `efa_com_handle_admin_completion()`,
> `efa_com_calc_crc16_checksum()`, `efa_com_cqe_checksum_valid()`.

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

## Key Data Structures

The core structures — `struct efa_dev` (device: `ibdev`, `pdev`, `edev`, `reg_bar`
BAR0, `mem_bar` BAR2, `dev_attr`, `stats`, `cqs_xa`/`cqs_arr`, `db_bar_addr/len`),
`struct efa_qp` (adds `qp_handle`, `qp_num`, `sq_db_offset`, `rq_db_offset`),
`struct efa_cq` (`cq_idx`, `cq_handle`), and `struct efa_mr` (`umem`, `pbl`, plus
`dmabuf`/`dmabuf_attach`/`dmabuf_sgt` under **`HAVE_MR_DMABUF`**) — are defined in
[efa.h](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa.h).

> Source: `efa.h` — `struct efa_dev`, `struct efa_qp`, `struct efa_cq`,
> `struct efa_mr`. Use `codegraph explore efa_dev` for current field layouts.

Compile-guard note: the DMA-BUF fields and code paths are guarded by **`HAVE_MR_DMABUF`**
(and **`HAVE_IB_UMEM_DMABUF_PINNED`** for the pinned-import variant). There is no
`HAVE_IB_REG_USER_MR_DMABUF` macro — if you see that name it is wrong.

## Device Capabilities

`efa_com_get_device_attr()` fills a device-attributes struct with the hardware limits and
capability bitmask. Representative fields and typical values:

| Field | Meaning | Typical |
|-------|---------|---------|
| `max_qp` | Max queue pairs | ~1024 (up to ~8K on p5) |
| `max_cq` | Max completion queues | ~1024 |
| `max_mr` | Max memory regions | ~4096 |
| `max_mr_pages` | Max pages per MR | ~512K |
| `max_qp_wr` | Max WRs per QP | ~8192 |
| `max_cq_depth` | Max CQ depth | ~16384 |
| `max_wr_send_sges` | SG entries per send | ~2 |
| `max_wr_recv_sges` | SG entries per recv | ~2 |
| `max_inline_data` | Max inline send | 32 B (64-B WQE) / 80 B (128-B WQE) |
| `max_rdma_size` | Max RDMA transfer | ~1 GB |
| `max_link_speed_gbps` | Device max link speed | up to 800 / 1600 |
| `device_caps` | Capability bitmask | see r3.3.0 section |

> Source: `efa_com_cmd.c` / `efa_com_cmd.h` — `efa_com_get_device_attr()` and the
> device-attr result struct. Use `codegraph explore efa_com_get_device_attr`.

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
    ├── counters/                       # port_xmit_packets, port_rcv_packets,
    │                                   #   port_xmit_data, port_rcv_data, ...
    └── gid_attrs/                      # GID attributes
```

Module parameters are exposed under `/sys/module/efa/parameters/`:

```bash
# View EFA driver parameters
cat /sys/module/efa/parameters/*
```

## Interrupts vs Polling

EFA supports two ways to learn about completions:

**MSI-X interrupts** — the driver's completion-event handler wakes threads blocked in
`ibv_get_cq_event()`, and also delivers QP-error and async (link/health) events via the
AENQ. Used for blocking operations and error notification; **rare in NCCL**.

**Polling mode** — libfabric/NCCL typically poll the CQ continuously in userspace:
- No interrupts on the data path.
- Lower latency (~10 μs vs ~50 μs) at the cost of higher CPU usage.

```bash
# Check interrupt counts — a low, flat count indicates polling mode
cat /proc/interrupts | grep efa
```

## Debugging

### Enable Driver Debug

```bash
# Enable EFA driver debug messages (dynamic debug)
echo 'module efa +p' > /sys/kernel/debug/dynamic_debug/control

# Or via modprobe
modprobe efa debug_mask=0xffffffff

# Raise console log level if needed
echo 8 > /proc/sys/kernel/printk

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

# Interrupt counts should increase during traffic (interrupt mode)
watch -n 1 'cat /proc/interrupts | grep efa'
```

### Device counters

```bash
# Per-port hardware counters
cat /sys/class/infiniband/efa_0/ports/1/counters/*
# e.g. port_xmit_packets, port_rcv_packets, port_xmit_data, port_rcv_data
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

**Bottlenecks**:
- **PCIe bandwidth** — ~25 GB/s ceiling (Gen4 x16).
- **Queue depth** — limits outstanding operations.
- **Doorbell rate** — MMIO overhead grows with rings/sec (batch WQEs before ringing).
- **Memory registration** — expensive without a registration cache.

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
efa_ah_cache.c            - rhashtable AH cache (r3.3.0, AH device-object reuse)
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
5. Supports **DMA-BUF** for GPU memory (kernel 5.12+); no longer gated by EFA generation
6. Memory registration is **expensive** (100-500 μs) - userspace must cache!
7. After setup, posting sends/recvs is just **memory writes** + **doorbell ring**
8. **r3.3.0** adds hardware **completion counters** (for GDAKI/GPU-initiated paths),
   64-bit work request IDs, 128-byte WQEs + inline WRITE, the 0xefa4 device ID,
   800/1600 Gbps link speeds with a query-port-speed verb, admin-response checksum
   validation, >4 GB MR page sizes, and an rhashtable AH cache.

**Related Documentation**:
- [rdma-core-and-verbs.md](rdma-core-and-verbs.md) - Userspace library that talks to this driver
- [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md) - DMA-BUF implementation details
- [gpu-memory-kernel-path.md](gpu-memory-kernel-path.md) - Deep GPU kernel path (P2P provider ends, ticket revocation)
- [efa-hardware-architecture.md](efa-hardware-architecture.md) - Hardware queue/WQE/CQE model
- [rdma-memreg.md](rdma-memreg.md) - Memory registration from all layers
