# Stack Map: Repos, Locations, and the Kernel GPU-Memory Path

**Read this first.** This is the orientation doc for the whole corpus: what the stack is
made of, which GitHub repo owns each piece, where it lives on this workstation, whether it
runs in userspace or the kernel, and — the part that saves the most time — *which parts of
each repo matter and which are noise*. When you need to go read or modify source, start
here to find the right file, then jump to the deep-dive doc for that layer.

Everything below was measured on this workstation on **2026-09-02** against the HEADs in
the table. Source links are branch-form (`blob/master`, `blob/main`, `tree/master`), per
the corpus link policy in [SOURCES.md](SOURCES.md), which holds the
authoritative snapshot table — this doc does not duplicate it.

---

## 1. The seven-plus repos at a glance

Nine clones are present. Seven are the live EFA+NCCL stack; two (DeepEP, upstream-to-nvshmem)
are workload/adjacent and only matter for MoE/NVSHMEM experiments.

| Repo | GitHub | Local clone | Size (excl `.git`) | HEAD | Version | Owner class |
| --- | --- | --- | --- | --- | --- | --- |
| nccl | [NVIDIA/nccl](https://github.com/NVIDIA/nccl) | `/home/bibracha/sw-stack/nccl` | 21M | `fd168324` | v2.31.2-1 | upstream (NVIDIA) |
| aws-ofi-nccl | [aws/aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl) | **`/home/bibracha/ofi-nccl-work`** ⚠️ | 54M | `d840aa1` | v1.21.1 | **ECT-owned** |
| libfabric | [ofiwg/libfabric](https://github.com/ofiwg/libfabric) | `/home/bibracha/sw-stack/libfabric` | 29M | `cb6364e05` | 2.7.0rc1 | upstream (OFIWG) |
| rdma-core | [linux-rdma/rdma-core](https://github.com/linux-rdma/rdma-core) | `/home/bibracha/sw-stack/rdma-core` | 13M | `8b3a3e64d` | 65.0-dev (rel v64.0) | upstream (linux-rdma) |
| amzn-drivers | [amzn/amzn-drivers](https://github.com/amzn/amzn-drivers) | `/home/bibracha/sw-stack/amzn-drivers` | 8.3M | `44c91d1` | EFA driver r3.3.0 | AWS-owned, different team |
| open-gpu-kernel-modules | [NVIDIA/open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules) | `/home/bibracha/sw-stack/open-gpu-kernel-modules` | 142M | `e4a5faa2` | 610.57.04 | upstream (NVIDIA) |
| aws-ofi-nccl.wiki | [aws/aws-ofi-nccl.wiki](https://github.com/aws/aws-ofi-nccl.wiki) | `/home/bibracha/sw-stack/aws-ofi-nccl.wiki` | 36K | `c380389` | env var + tuner reference docs | ECT-owned |
| DeepEP (fork) | [aamirshafi/DeepEP](https://github.com/aamirshafi/DeepEP) | `/home/bibracha/sw-stack/DeepEP` | 1.9M | `b57e5e2` | unchanged since Dec 2025 | fork (workload) |
| upstream-to-nvshmem | [amazon-contributing/upstream-to-nvshmem](https://github.com/amazon-contributing/upstream-to-nvshmem) | `/home/bibracha/sw-stack/upstream-to-nvshmem` | 16M | `0f97cf4` | branch `devel` | AWS-contributing (workload) |

⚠️ **aws-ofi-nccl is NOT under `sw-stack/`.** Its working clone is
`/home/bibracha/ofi-nccl-work`. This trips people up constantly: every other repo is a
sibling under `sw-stack/`, but the one repo ECT actually edits sits one level up. If a
search across `sw-stack/` comes up empty for a plugin symbol, that is why.

### Ownership → where a fix goes

The owner class determines the mechanism for a change:

- **ECT-owned (`aws-ofi-nccl`, its wiki):** fix goes in a **CR** (internal code review) that
  lands on `aws/aws-ofi-nccl`. This is the repo we own end-to-end.
- **Upstream we consume (`nccl`, `libfabric`, `rdma-core`, `open-gpu-kernel-modules`):**
  a fix is an upstream **GitHub PR** (and usually a discussion first). We don't own these; we
  build against released versions.
- **AWS-owned, different team (`amzn-drivers` — the EFA kernel driver and the
  `efa_nv_peermem` shim):** AWS-owned but **out of scope for direct ECT changes**. Route
  driver changes as a **request to the owning team**, not a CR in our repo. (This doc does
  not name that team because ownership was not verified here.)
- **Workload/fork (`DeepEP`, `upstream-to-nvshmem`):** experiment scaffolding, not the
  shipping stack.

---

## 2. The spine: userspace vs kernel

The entire stack splits at one line — the `/dev/infiniband/uverbsN` character device. Above
it, four libraries are all loaded into the *same OS process* — one process per rank, typically
one rank per GPU, so eight such processes on a p5. This is not a client/server arrangement:
there is no daemon, no IPC and no context switch between NCCL, the plugin, libfabric and
libibverbs. They are shared objects in one address space calling each other through ordinary
function calls. Verified on a built plugin:

```
$ ldd libnccl-net-ofi.so
    libfabric.so.1   => /opt/libfabric/lib/libfabric.so.1
    libefa.so.1      => /lib64/libefa.so.1        # rdma-core's EFA provider
    libibverbs.so.1  => /lib64/libibverbs.so.1
```

NCCL `dlopen()`s the plugin (`src/plugin/plugin_open.cc`, `openPluginLib()`), the plugin's
dynamic dependencies pull in libfabric and libibverbs, and libfabric may itself `dlopen()`
providers (`src/fabric.c`). Four practical consequences:

- The mapped send/receive/completion queues and the doorbell page live in **this** process's
  virtual address space. That is what "OS bypass" means concretely.
- A segfault anywhere in the chain kills the rank, and takes the job down with it.
- One `gdb` attach to one PID sees every layer's frames at once, which is why the GDB workflow
  in `ect-ofi-nccl-gdb-debug` works the way it does.
- Each rank has its own queue pairs, its own MR cache and its own copy of the libraries.
  Nothing is shared between ranks on the same host except what the NIC and the kernel own.

Below it, the kernel
modules. **Read section 4 before you conclude that this layering describes the send path —
it does not.**

```
 USERSPACE  (one process per rank; all of this in that process's address space)
 ┌───────────────────────────────────────────────────────────────────────┐
 │ NCCL           libnccl.so                       NVIDIA/nccl             │
 │ aws-ofi-nccl   libnccl-net-ofi.so (dlopen'd)    aws/aws-ofi-nccl  ◄ ECT │
 │ libfabric      libfabric.so                     ofiwg/libfabric         │
 │ rdma-core      libibverbs.so + libefa.so        linux-rdma/rdma-core    │
 └───────────────────────────────────────────────────────────────────────┘
 ═══ kernel boundary: /dev/infiniband/uverbsN ═══════════════════════════════
   ioctl() for CONTROL  +  mmap() maps queues/doorbells into userspace
 ┌───────────────────────────────────────────────────────────────────────┐
 │ KERNEL                                                                  │
 │ ib_core / ib_uverbs   generic RDMA subsystem    mainline Linux (NOT     │
 │                                                 vendored in any repo)   │
 │ efa.ko                EFA RDMA driver           amzn-drivers            │
 │                                                 (also mainline:         │
 │                                                  drivers/infiniband/hw/efa/) │
 │ efa_nv_peermem.ko     GPL shim to NVIDIA P2P    amzn-drivers            │
 │ nvidia.ko (+ nv-p2p)  GPU driver                open-gpu-kernel-modules │
 │ nvidia-peermem.ko     NVIDIA's RDMA bridge      open-gpu-kernel-modules │
 │                                                 (NOT used by EFA — §5)  │
 └───────────────────────────────────────────────────────────────────────┘
 ═══ PCIe ═══════════════════════════════════════════════════════════════════
 HARDWARE: EFA NIC — firmware + ASIC, closed source.
```

`ib_core`/`ib_uverbs` are the generic Linux RDMA subsystem and are **not vendored in any
repo above** — they come from whatever kernel the instance runs. The EFA driver plugs into
them; it does not replace them.

---

## 3. What matters inside each repo (anti-noise guidance)

The clones are large and mostly irrelevant to us. This section is the part that stops an
agent from wandering into the wrong directory.

### nccl — `/home/bibracha/sw-stack/nccl` (21M)

Source lives under [`src/`](https://github.com/NVIDIA/nccl/tree/master/src). The
subdirectories: `config devcomm device diagnostics enqueue gin graph include misc
nccl_device os param plugin ras register rma scheduler transport tuning`.

**Layout notes for anyone with older NCCL knowledge — several files moved:**

| You may remember | It is now |
| --- | --- |
| `src/enqueue.cc` | [`src/enqueue/enqueue.cc`](https://github.com/NVIDIA/nccl/blob/master/src/enqueue/enqueue.cc) |
| `src/env.cc` | [`src/plugin/env.cc`](https://github.com/NVIDIA/nccl/blob/master/src/plugin/env.cc) |
| `src/graph/tuning.cc` | replaced by the [`src/tuning/`](https://github.com/NVIDIA/nccl/tree/master/src/tuning) cost-model directory |

The GIN and RMA device APIs live in
[`src/gin/`](https://github.com/NVIDIA/nccl/tree/master/src/gin) and
[`src/rma/`](https://github.com/NVIDIA/nccl/tree/master/src/rma). NCCL EP (expert-parallel
contrib) lives in [`contrib/nccl_ep/`](https://github.com/NVIDIA/nccl/tree/master/contrib/nccl_ep).
Public API header is generated from
[`src/nccl.h.in`](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in) (not the old
`src/include/nccl.h`).

### aws-ofi-nccl — `/home/bibracha/ofi-nccl-work` (54M) — **the ECT repo**

Not under `sw-stack/` (see §1). The pieces:

- [`src/`](https://github.com/aws/aws-ofi-nccl/tree/master/src) — the plugin. Transports are
  `nccl_ofi_rdma.cpp` (P5/P5en/P6, ~226K — the big one), `nccl_ofi_sendrecv.cpp` (P4d),
  `nccl_ofi_net.cpp`, plus subdirs:
  - [`src/cm/`](https://github.com/aws/aws-ofi-nccl/tree/master/src/cm) — connection manager
  - [`src/rdma/gin/`](https://github.com/aws/aws-ofi-nccl/tree/master/src/rdma/gin) — GIN + GDAKI
  - [`src/stats/`](https://github.com/aws/aws-ofi-nccl/tree/master/src/stats)
  - [`src/tuner/`](https://github.com/aws/aws-ofi-nccl/tree/master/src/tuner)
- [`include/`](https://github.com/aws/aws-ofi-nccl/tree/master/include) — mirrors `src/`
  (`include/cm/`, `include/rdma/`, `include/stats/`, `include/tuner/`).
- [`3rd-party/efa-gda`](https://github.com/aws/aws-ofi-nccl/tree/master/3rd-party/efa-gda) —
  vendored EFA-GDA CUDA sources that GDAKI needs. (`3rd-party/` also carries `efa-dp-direct`,
  `boost`, and a vendored `nccl`.)
- [`doc/gin-getting-started.md`](https://github.com/aws/aws-ofi-nccl/blob/master/doc/gin-getting-started.md) —
  GIN getting-started guide.

### libfabric — `/home/bibracha/sw-stack/libfabric` (29M)

Only the EFA provider is ours:

- [`prov/efa/`](https://github.com/ofiwg/libfabric/tree/master/prov/efa) — the provider
  (29 `.c` files under `prov/efa/src/`, ~4.0M). The env/param defaults live in
  [`prov/efa/src/efa_env.c`](https://github.com/ofiwg/libfabric/blob/master/prov/efa/src/efa_env.c)
  (field `use_data_path_direct`, default `true` — see §4).
- Public headers: [`include/rdma/`](https://github.com/ofiwg/libfabric/tree/master/include/rdma).
- Man pages: [`man/`](https://github.com/ofiwg/libfabric/tree/master/man) — the provider's own
  doc is [`man/fi_efa.7.md`](https://github.com/ofiwg/libfabric/blob/master/man/fi_efa.7.md).
- Release notes: [`NEWS.md`](https://github.com/ofiwg/libfabric/blob/master/NEWS.md).

### rdma-core — `/home/bibracha/sw-stack/rdma-core` (13M)

The EFA userspace provider is
[`providers/efa/`](https://github.com/linux-rdma/rdma-core/tree/master/providers/efa):
`efa.c`, `verbs.c` (~95K), `efa.h`, `efa-abi.h`, `efa_io_defs.h`, `efa_io_regs_defs.h`,
`efadv.h`, `efa_trace.c`, `libefa.map`, `man/`. This builds `libibverbs.so` + `libefa.so`.

**`efa_io_defs.h` is a shared hardware-descriptor definition that appears in four places** —
rdma-core, the kernel driver, libfabric, and NCCL's EFA-GDA path — kept in sync by hand:

| Copy | Path |
| --- | --- |
| rdma-core | [`providers/efa/efa_io_defs.h`](https://github.com/linux-rdma/rdma-core/blob/master/providers/efa/efa_io_defs.h) |
| kernel driver | [`kernel/linux/efa/src/efa_io_defs.h`](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_io_defs.h) |
| libfabric | [`prov/efa/src/efa_io_defs.h`](https://github.com/ofiwg/libfabric/blob/master/prov/efa/src/efa_io_defs.h) |
| nccl (EFA-GDA) | `src/transport/net_efa_gda/efa-dp-direct/include/device/efa_io_defs.h` |

If you change a descriptor layout, all four must move together.

### amzn-drivers — `/home/bibracha/sw-stack/amzn-drivers` (8.3M) — **~75% is not EFA**

Six top-level areas; **only two are ours**. Measured breakdown:

| Area | Size | Files | Lines | Relevance |
| --- | --- | --- | --- | --- |
| [`kernel/linux/efa`](https://github.com/amzn/amzn-drivers/tree/master/kernel/linux/efa) | 648K | 57 | 14,742 | **THE EFA DRIVER — this is what you want** |
| [`kernel/linux/efa_nv_peermem`](https://github.com/amzn/amzn-drivers/tree/master/kernel/linux/efa_nv_peermem) | 120K | 23 | 595 | **GPL shim to NVIDIA P2P (§5)** |
| `kernel/linux/ena` | 632K | 31 | 14,645 | Elastic **Network** Adapter = ordinary EC2 networking, **NOT EFA. Ignore.** |
| `kernel/linux/common` | 264K | 8 | — | shared build glue |
| `kernel/linux/rpm` | 76K | 11 | — | packaging |
| `userspace/dpdk` | 5.9M | 571 | — | **71% of the repo by size. Irrelevant to us. Ignore.** |

DPDK (5.9M) + ENA (632K) together are ~75% of the repo and neither is EFA. `ena` is the
plain-EC2 NIC driver — do **not** go looking there for the network driver. The two EFA
areas together are under 800K.

### open-gpu-kernel-modules — `/home/bibracha/sw-stack/open-gpu-kernel-modules` (142M) — **largest clone, almost all noise**

Almost all of the 142M is GPU driver internals we never touch. What matters for EFA lives
in `kernel-open/`:

- [`kernel-open/nvidia/nv-p2p.c`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/master/kernel-open/nvidia/nv-p2p.c)
  and [`nv-p2p.h`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/master/kernel-open/nvidia/nv-p2p.h) —
  the **GPUDirect RDMA API that EFA actually calls** (§5).
- [`kernel-open/nvidia-peermem/`](https://github.com/NVIDIA/open-gpu-kernel-modules/tree/master/kernel-open/nvidia-peermem) —
  NVIDIA's own RDMA bridge module. **EFA does NOT use it** (§5). It ships its own copy of
  `nv-p2p.h` here too.

Other kernel modules present but **not relevant** to EFA: `nvidia-uvm/` (unified memory —
by far the largest subtree), `nvidia-drm/` (display), `nvidia-modeset/` (display).

---

## 4. The data path is not the setup path (critical mental model)

The layering in §2 is the **setup / control path**. It is the wrong mental model for the
**send path**. Two independent bypasses stack on top of each other:

**Bypass 1 — OS bypass (kernel bypass).** Once queues are set up, userspace writes
work-queue entries directly into its own `mmap`'d send queue and rings a doorbell via MMIO.
**Zero syscalls on the data path; the kernel does not run on the send path at all.** The
kernel's entire job was to hand over the queue/doorbell mapping during setup.

**Bypass 2 — data-path direct (NEW, now the DEFAULT).**
`FI_EFA_USE_DATA_PATH_DIRECT` defaults to **true**
(`efa_env.c` field `use_data_path_direct`, initialized `= true`). With it on, libfabric's
EFA provider formats WQEs and reads CQEs **itself** instead of calling
`ibv_post_send()` / `ibv_poll_cq()`. So **rdma-core is CONTROL PATH ONLY by default**: it
does device open/probe, QP and CQ creation, MR registration, AH creation, and the `mmap` —
then steps out of the way.

Read "bypass" carefully here — it means *not called per operation*, not *circumvented*.
rdma-core is the only thing that can create a QP, and it deliberately hands libfabric the
pointers to write to, through its EFA direct-verbs extensions:

```c
efadv_query_qp_wqs(efa_qp->ibv_qp, &sq_attr, &rq_attr, ...);
direct_qp->sq.wq.db = sq_attr.doorbell;   /* the hardware doorbell register */
efadv_query_cq(..., &attr, ...);          /* the mapped CQ buffer + its doorbell */
```

libfabric then does, per send, `mmio_memcpy_x64()` of a 64- or 128-byte WQE into
`sq->desc + idx * wqe_size` (write-combined memory), `mmio_flush_writes()`, and
`mmio_write32(sq->wq.db, pc)`. It is writing into memory rdma-core established and
published. Declarations live in
[`providers/efa/efadv.h`](https://github.com/linux-rdma/rdma-core/blob/master/providers/efa/efadv.h)
on the rdma-core side.

**Takeaway:** "libfabric calls rdma-core which calls the kernel" is **wrong for the data
path**. On the send path, libfabric talks to the NIC directly through the mapped queues;
rdma-core set things up and the kernel is not involved. Cite
[`prov/efa/src/efa_env.c`](https://github.com/ofiwg/libfabric/blob/master/prov/efa/src/efa_env.c)
if you need the source. For the deep treatment see
[efa-provider.md](efa-provider.md) and [rdma-core-and-verbs.md](rdma-core-and-verbs.md) —
this doc deliberately does not duplicate it.

---

## 5. The kernel-side GPU-memory path (nv-p2p, efa_nv_peermem, nvidia-peermem, DMA-BUF)

For RDMA straight out of GPU memory (GPUDirect RDMA), the EFA driver needs the NVIDIA GPU
driver to pin GPU pages and expose their DMA addresses. There are two ways to bridge EFA to
NVIDIA, and EFA uses one of them.

### 5a. The path EFA uses: `efa_nv_peermem` → `nv-p2p`

`nvidia.ko` exports a GPUDirect RDMA API called **nv-p2p**
([`kernel-open/nvidia/nv-p2p.c`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/master/kernel-open/nvidia/nv-p2p.c) /
[`nv-p2p.h`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/master/kernel-open/nvidia/nv-p2p.h)):
`nvidia_p2p_get_pages()` pins GPU pages and returns their physical/DMA addresses,
`nvidia_p2p_put_pages()` releases them, plus free callbacks.

AWS ships a small **GPL shim**, `efa_nv_peermem.ko`, in
[`amzn-drivers/kernel/linux/efa_nv_peermem`](https://github.com/amzn/amzn-drivers/tree/master/kernel/linux/efa_nv_peermem)
that links the EFA driver to that nv-p2p API. Per its own README, it "serves as a link
between Nvidia opensource drivers and the EFA driver for the needs of Cuda memory GPU direct
support," supports the **NVIDIA open-source drivers only**, and refuses to load when they
are absent. It is tiny — 23 files, 595 lines. The source (under
[`efa_nv_peermem/src`](https://github.com/amzn/amzn-drivers/tree/master/kernel/linux/efa_nv_peermem/src)):

| File | Role |
| --- | --- |
| `efa_nv_peermem_main.c` | main Linux kernel module |
| `efa_nv_peermem.h` | shim interface |
| `nv-p2p.h` | the NV P2P API contract (its own copy of NVIDIA's header) |

It is not auto-loaded at boot; the EFA driver requests it on demand the first time P2P
memory registration is needed. Registration flow (setup path — see §4, this is not the send
path):

```
NCCL/plugin registers a GPU buffer
   └─ libfabric fi_mr_reg → rdma-core (verbs) → efa.ko
        └─ efa.ko asks efa_nv_peermem.ko
             └─ nvidia_p2p_get_pages() in nvidia.ko (nv-p2p)
                  └─ GPU pages pinned, DMA addresses returned → EFA MR
```

### 5b. The path EFA does NOT use: `nvidia-peermem`

NVIDIA ships **its own** RDMA bridge,
[`nvidia-peermem.ko`](https://github.com/NVIDIA/open-gpu-kernel-modules/tree/master/kernel-open/nvidia-peermem),
which plugs into the mainline `ib_core` **peer-memory** interface. **EFA does not use it** —
EFA uses AWS's `efa_nv_peermem` shim (5a) instead. Don't confuse the two: the names are one
character apart. `nvidia-peermem` is the Mellanox/mlx5-style bridge; `efa_nv_peermem` is the
EFA-specific one.

### 5c. The modern alternative: DMA-BUF

The vendor-specific peer-memory bridge is increasingly replaced by **DMA-BUF** (kernel
5.12+): CUDA exports GPU memory as a `dmabuf` file descriptor, libfabric registers it with
`FI_MR_DMABUF`, and rdma-core/`efa.ko` import it through the standard Linux DMA-BUF
interface — no NVIDIA-specific peer-memory module on the registration path. This is
vendor-neutral and produces fewer IOMMU entries. The corpus already covers this in depth in
[dmabuf-gpu-memory.md](dmabuf-gpu-memory.md); GPU memory registration generally is in
[cuda-memory.md](cuda-memory.md) and [rdma-memreg.md](rdma-memreg.md). This doc only places
DMA-BUF as the alternative to 5a/5b — it does not duplicate that material.

### Cross-references for this layer

- [kernel-efa-driver.md](kernel-efa-driver.md) — kernel EFA driver internals
- [kernel-efa-driver.md](kernel-efa-driver.md) — driver capabilities and verbs surface
- [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md) — DMA-BUF path
- [cuda-memory.md](cuda-memory.md) / [rdma-memreg.md](rdma-memreg.md) — memory registration

---

## 6. How to reproduce this layout (fresh machine)

The local paths above are **one engineer's layout**, not canonical. **Check whether a clone
already exists before cloning** (e.g. `ls /home/bibracha/sw-stack`). Note aws-ofi-nccl goes
outside `sw-stack/` in this layout (§1); place it wherever your workflow expects it.

```bash
# The seven live-stack repos
git clone https://github.com/NVIDIA/nccl.git
git clone https://github.com/aws/aws-ofi-nccl.git            # ECT repo (here: ../ofi-nccl-work)
git clone https://github.com/ofiwg/libfabric.git
git clone https://github.com/linux-rdma/rdma-core.git
git clone https://github.com/amzn/amzn-drivers.git
git clone https://github.com/NVIDIA/open-gpu-kernel-modules.git
git clone https://github.com/aws/aws-ofi-nccl.wiki.git

# Workload / adjacent (only for MoE / NVSHMEM experiments)
git clone https://github.com/aamirshafi/DeepEP.git
git clone -b devel https://github.com/amazon-contributing/upstream-to-nvshmem.git
```

To land exactly on the HEADs these docs were verified against, check out the commits in the
table in §1 (or the authoritative snapshot in [SOURCES.md](SOURCES.md)).
The corpus links are branch-form and track current upstream, so a plain clone stays valid;
the recorded HEADs matter only when you need the exact snapshot the docs describe.

---

## 7. Verification note

Every path cited here was checked against the live filesystem on this workstation on
**2026-09-02**, at the HEADs in §1. Source-link and cross-reference verification for the
whole corpus is tracked in [SOURCES.md](SOURCES.md), which is the
authoritative snapshot table — refer to it rather than any copy.
