# GPU Memory Kernel Path: nv-p2p, efa_nv_peermem, nvidia-peermem, and DMA-BUF

## Overview

This document covers the **kernel-side terminus** of GPU/accelerator memory
registration for EFA — the part that sits *below* libfabric and the OFI plugin and
*beside* the EFA driver's own P2P subsystem. The corpus already documents:

- the EFA driver's peer-memory abstraction (`efa_p2p_ops` vtable, the three providers,
  the ticket revocation mechanism) — `kernel-efa-driver.md` §"3b. GPU & Accelerator
  Peer Memory (P2P Subsystem)";
- DMA-BUF from the plugin/libfabric side — `dmabuf-gpu-memory.md`;
- the userspace CUDA/ROCm allocation and export side — `cuda-memory.md`,
  `rocm-memory.md`, `neuron-memory.md`;
- generic MR registration — `rdma-memreg.md`, `lkey-rkey-explained.md`.

What was **undocumented** is *the other end*: what NVIDIA's kernel modules actually
export, what the AWS `efa_nv_peermem` shim is (AWS-owned code sitting directly in the
GPUDirect path, previously mentioned nowhere in the corpus), how the EFA driver resolves
and chooses between these symbols at runtime, and — the single most important
correction here — **why EFA does not use `nvidia-peermem`** even though almost every
GPUDirect RDMA guide says it must. This document is dense on purpose; it is a trap map
for anyone debugging "GPUDirect not working on EFA."

Scope note: this is the *NVIDIA/CUDA and Neuron peer-memory kernel path* plus the
*DMA-BUF kernel import path*. It does not re-derive the SRD wire protocol or the reg_mr
admin command internals (see `srd-protocol.md`, `rdma-memreg.md`).

---

## The Module Cast — what each one is, and is not

| Module (`.ko`) | Repo / path | License | Role | Registers as IB peer-memory client? |
|---|---|---|---|---|
| `efa.ko` | amzn-drivers `kernel/linux/efa` | GPL-2.0 OR Linux-OpenIB | The EFA RDMA driver. Owns the P2P provider subsystem. Calls `nvidia_p2p_*` **directly** (or through the shim). | **No** |
| `efa_nv_peermem.ko` | amzn-drivers `kernel/linux/efa_nv_peermem` | Dual BSD/GPL | AWS **GPL re-export shim**. 4 pass-through functions. Optional. | No |
| `nvidia.ko` (+ `nvidia-uvm`, etc.) | open-gpu-kernel-modules `kernel-open/nvidia` | MIT/GPL (open variant) | The GPU driver. Exports the `nvidia_p2p_*` API from `nv-p2p.c`. | No (it is the *provider* of pages, not an IB client) |
| `nvidia-peermem.ko` | open-gpu-kernel-modules `kernel-open/nvidia-peermem` | Dual BSD/GPL | The **Mellanox/MLNX_OFED** GPUDirect bridge. Registers with `ib_core` via `ib_register_peer_memory_client()`. **Unused by EFA.** | **Yes** — and that is the point |
| `neuron` (Trainium/Inferentia driver) | (AWS Neuron, out of tree) | — | Exports `neuron_p2p_register_va` / `neuron_p2p_unregister_va`. | No |

The two most misunderstood entries are `efa_nv_peermem` (a shim people assume is a
driver) and `nvidia-peermem` (a module people assume EFA needs). Both are addressed in
detail below.

---

## `efa_nv_peermem` — a thin GPL re-export shim, not a driver

`efa_nv_peermem` is **595 lines total** and contains no memory-management logic of its
own. Its entire substance is four functions in
[efa_nv_peermem_main.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa_nv_peermem/src/efa_nv_peermem_main.c)
that forward straight through to NVIDIA's API and re-export the result GPL-only:

| Shim symbol (`EXPORT_SYMBOL_GPL`) | Forwards verbatim to |
|---|---|
| `efa_nv_peermem_p2p_get_pages` | `nvidia_p2p_get_pages` |
| `efa_nv_peermem_p2p_dma_map_pages` | `nvidia_p2p_dma_map_pages` |
| `efa_nv_peermem_p2p_dma_unmap_pages` | `nvidia_p2p_dma_unmap_pages` |
| `efa_nv_peermem_p2p_put_pages` | `nvidia_p2p_put_pages` |

Each body is a single-line call with identical arguments; there is no transformation,
caching, or bookkeeping. Module metadata (same file):
`MODULE_LICENSE("Dual BSD/GPL")`, `MODULE_SOFTDEP("pre: nvidia")`,
`MODULE_AUTHOR("Amazon.com, Inc. or its affiliates")`, version **1.2.3**
(`DRV_MODULE_VER_MAJOR/MINOR/SUBMINOR = 1/2/3`). The header
[efa_nv_peermem.h](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa_nv_peermem/src/efa_nv_peermem.h)
just declares those four prototypes over `nv-p2p.h`.

### It refuses to load against the proprietary NVIDIA driver

`efa_nv_peermem_init()` performs two gate checks before returning success:

1. **Taint check.** With the source comment *"Make sure that we didn't inherit taint
   from Nvidia proprietary driver"*, it tests `TAINT_PROPRIETARY_MODULE` on
   `THIS_MODULE->taints` and returns `-EINVAL` with the log line
   `efa_nv_peermem: Nvidia proprietary is unsupported`.
2. **Symbol presence.** It verifies all four `nvidia_p2p_*` pointers are non-NULL and
   otherwise returns `-ENOSYS` with `efa_nv_peermem: Nvidia P2P symbols are unavailable`.

The module README states plainly: *"EFA NV Peermem supports Nvidia opensource drivers
only and it is unable to load when such drivers are not avilable in the system."* (sic)
— [README](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa_nv_peermem/README).

**What the code demonstrably does:** it will only finish loading when the running
`nvidia.ko` is the open-source variant (which does not set `TAINT_PROPRIETARY_MODULE`)
*and* exports the `nvidia_p2p_*` symbols.

**Why (inference):** The shim exists to bridge a licensing gap. `nvidia_p2p_*` symbols
are exported by the GPU driver, but a consumer wants them re-exported as
`EXPORT_SYMBOL_GPL`. The open-source `nvidia.ko` can participate in a GPL symbol chain;
the proprietary driver taints the kernel and cannot. The taint gate is the mechanism
that enforces "open-source NVIDIA driver only." This is an inference about *motivation*
from the taint check plus the GPL re-export; the code itself only proves the gate
behavior above, not the reasoning behind it.

> Note the direction of dependence: `MODULE_SOFTDEP("pre: nvidia")` asks the loader to
> insert `nvidia` first, but this is advisory. The hard guarantee is the runtime symbol
> check in `init`.

---

## The userspace-to-hardware walk for a GPU memory registration

A single logical "register this GPU buffer" travels through many layers. The two kernel
terminal paths (DMA-BUF vs peer-memory) diverge only at the very bottom, inside the EFA
driver's `reg_mr` verb.

```
CUDA app: cudaMalloc(ptr)                                   [cuda-memory.md]
  │
NCCL / OFI plugin: regMr(ptr,len)                           [ofi-plugin.md]
  │  plugin decides dmabuf viability: nccl_ofi_dmabuf_viable()
  ▼
libfabric EFA provider: fi_mr_reg / fi_mr_regattr           [efa-provider.md, rdma-memreg.md]
  │  efa_mr_reg_ibv_mr(): dmabuf-first, else ibv_reg_mr()
  ├── dmabuf branch → ibv_reg_dmabuf_mr(fd,...)             ─┐
  └── VA branch     → ibv_reg_mr(va,...)                    ─┤
                                                             ▼
rdma-core → kernel uverbs → EFA driver                     [kernel-efa-driver.md]
  ├── efa_reg_user_mr_dmabuf()  → ib_umem_dmabuf_get_pinned()   (DMA-BUF path)
  └── efa_reg_mr()              → ib_umem_get* fails on device pages
                                   → efa_p2p_get() → provider try_get()  (peer-mem path)
                                            │
                                            ├─ NVMEM_V1/V2 → nvmem_get_fp():
                                            │     symbol_get(efa_nv_peermem_*)  [prefer shim]
                                            │     else symbol_get(nvidia_p2p_*) [direct]
                                            │        → nvidia_p2p_get_pages / dma_map_pages
                                            └─ NEURON      → neuron_p2p_register_va
                                            │
                                            ▼
                        both paths → efa_register_mr() → reg_mr admin command → NIC
```

Key structural fact, verified in
[efa_verbs.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_verbs.c):
`efa_reg_user_mr_dmabuf()` (guarded by `HAVE_MR_DMABUF`) and `efa_reg_mr()` are two
distinct verb entry points. The DMA-BUF one calls `ib_umem_dmabuf_get_pinned()` and
never touches `efa_p2p_*`. The VA one calls `ib_umem_get*()`; when that fails on device
pages (device BAR memory is not `get_user_pages()`-able), it falls through to
`efa_p2p_get()` under `HAVE_EFA_P2P`. **Both branches converge on `efa_register_mr()`**,
which issues the same `reg_mr` admin command with a device-page DMA list — the hardware
does not care which path produced the list.

---

## The three-way path choice

There are three ways GPU/accelerator memory reaches the NIC. They are **independent**:
DMA-BUF is not a provider in the `efa_p2p` array, and the peer-memory providers know
nothing about DMA-BUF.

### (a) DMA-BUF — modern, vendor-neutral, no bridge module

- **Kernel side:** `efa_reg_user_mr_dmabuf()` in `efa_verbs.c`, compiled only when
  `HAVE_MR_DMABUF` is defined (kcompat feature test). It imports the fd via
  `ib_umem_dmabuf_get_pinned()`, then hands the umem to `efa_register_mr()`.
- **Preconditions (verified in
  [nccl_ofi_dmabuf.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_dmabuf.cpp),
  `nccl_ofi_dmabuf_viable()` / `nccl_ofi_dmabuf_viable_and_supported()`):**
  libfabric ≥ 1.20 with `FI_MR_DMABUF` (`HAVE_DECL_FI_MR_DMABUF`); provider advertising
  `FI_HMEM`; `FI_VERSION_GE(api_version, 1.20)`; GPU device reporting DMA-BUF support
  (`nccl_net_ofi_gpu_have_dma_buf_attr()`); kernel ≥ 5.12 (parsed from `uname`, because
  the RDMA dmabuf import ioctls landed in 5.12); and the user not setting
  `OFI_NCCL_DISABLE_DMABUF=1`.
- **Failure mode:** if the import fails, libfabric can fall back to the VA path (see
  `efa_mr.c` below), which lands on the peer-memory providers instead.

> **Corrected fact — no hardware-generation gating.** DMA-BUF is **no longer gated by
> EFA hardware generation.** The old device-id check that disabled it on EFA Gen 1–3
> (`0xefa0` / `0xefa1` / `0xefa2`) over a firmware page-merging issue was **removed** in
> aws-ofi-nccl commit `0f285d5` ("dma-buf: Don't disable dma-buf on EFAv1-3").
> `nccl_ofi_dmabuf.cpp` contains **no device-id gating** today — verified: the file's
> viability logic is entirely libfabric-version / kernel-version / `FI_HMEM` /
> user-override based, with no reference to `0xefa0/1/2`. `dmabuf-gpu-memory.md` has this
> right. `kernel-efa-driver.md` §3b still carries the stale claim that DMA-BUF is "gated
> off on EFA Gen 1–3"; that is being corrected separately — do not trust that sentence.

### (b) NVIDIA peer memory via `nvidia_p2p_*` — the vendor path

- **Kernel side:** the `EFA_P2P_PROVIDER_NVMEM_V1` / `_V2` providers
  ([efa_nvmem_impl.h](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_nvmem_impl.h)).
  `nvmem_get()` pins pages with `get_pages` (64 KB-aligned via `GPU_PAGE_SHIFT 16`),
  DMA-maps them with `dma_map_pages`, and produces the DMA list from
  `dma_mapping->dma_addresses[]`.
- **Reached through the shim if loaded, otherwise directly** (see symbol resolution
  below).
- **Failure mode:** `get_pages` returning nonzero means "most likely not our pages"
  (`nvmem_get()` comment) — i.e. this range is not CUDA device memory — so the provider
  declines and `efa_p2p_get()` tries the next provider.

### (c) Neuron peer memory via `neuron_p2p_*` — Trainium/Inferentia

- **Kernel side:** `EFA_P2P_PROVIDER_NEURON`
  ([efa_neuronmem.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_neuronmem.c)).
  `neuronmem_get()` calls `neuron_p2p_register_va()` and builds the page list from
  `neuron_p2p_va_info.page_info[]` (physical address + page count per entry). Uses a
  4 KB page shift (`NEURON_PAGE_SHIFT 12`), unlike NVIDIA's 64 KB.
- See `neuron-memory.md` for the userspace side.

### How the choice is actually made at runtime

There are **two decision points**, one per side of the syscall boundary:

1. **libfabric / plugin side (dmabuf vs VA):** in
   [efa_mr.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_mr.c),
   `efa_mr_reg_ibv_mr()` is **dmabuf-first**. If `FI_MR_DMABUF` is set explicitly it uses
   the supplied fd. Otherwise, on the implicit VA path, if `DMABUF_IS_SUPPORTED(p_info)`
   it obtains an fd via `ofi_hmem_get_dmabuf_fd()` and calls `ibv_reg_dmabuf_mr()`; only
   if that fails (and `dmabuf_fallback_enabled`) does it fall back to `ibv_reg_mr()` on
   the VA. The per-iface state (`EFA_DMABUF_SUPPORTED` / `NOT_SUPPORTED` / `ASSUMED`) is
   probed at init in
   [efa_hmem.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_hmem.c) by
   attempting a real `ibv_reg_dmabuf_mr()` on a device buffer. So **whether the kernel
   sees a dmabuf fd or a VA is decided in libfabric**, not in the driver.
2. **EFA driver side (which peer provider):** if a VA registration reaches the driver
   and `ib_umem_get*()` fails (device pages), `efa_p2p_get()`
   ([efa_p2p.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_p2p.c))
   walks `prov_arr[]` in enum order (`NVMEM_V1`, `NVMEM_V2`, `NEURON`) calling each
   provider's `try_get()` until one claims the range. First claimant wins; the ticket is
   assigned before probing.

Both paths converge on the same `reg_mr` admin command. The generation/firmware
concerns that used to differ between them have been removed on the DMA-BUF side (see the
corrected fact above).

---

## V1 vs V2 NVMEM providers — one implementation compiled twice

`EFA_P2P_PROVIDER_NVMEM_V1` and `_V2` are **the same code built against two major
versions of the nv-p2p API**, not two implementations. Verified:

- [efa_nvmem_v1.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_nvmem_v1.c)
  is `#define NV_P2P_MAJOR_VERSION 1`, `#include "efa_nvmem_impl.h"`, and a one-line
  provider getter `nvmem_v1_get_provider()`.
- [efa_nvmem_v2.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_nvmem_v2.c)
  is identical except `#define NV_P2P_MAJOR_VERSION 2`.
- `efa_nvmem_impl.h` selects the header on that macro: `NV_P2P_MAJOR_VERSION == 1`
  includes `nv-p2p.h`, else `nv-p2p_v2.h`.

The two headers differ in structure layout — e.g. `struct nvidia_p2p_page_table` in the
v2 header carries a trailing `flags` field
([nv-p2p_v2.h](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/nv-p2p_v2.h))
that the v1 header
([nv-p2p.h](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/nv-p2p.h))
lacks. Both carry `NVIDIA_P2P_PAGE_TABLE_VERSION 0x00020000` and the same 4KB/64KB/128KB
page-size enum, so `nvmem_pgsz()` is shared. Building both providers lets a single `efa.ko`
interoperate with whichever `nvidia_p2p_*` ABI the installed GPU driver presents; at
runtime `efa_p2p_get()` tries `NVMEM_V1` first, then `NVMEM_V2`. The version-compatibility
macros (`NVIDIA_P2P_PAGE_TABLE_VERSION_COMPATIBLE`, `..._DMA_MAPPING_VERSION_COMPATIBLE`)
are checked in `nvmem_get_pages()` / `nvmem_dma_map()`; a mismatch causes that provider to
put the pages back and decline, so the other version can win.

> Note (inference): the "which NVIDIA driver generation maps to which major version" is
> an ABI-versioning question owned by NVIDIA, not encoded in the EFA source. The source
> only proves EFA builds against both and probes at runtime; it does not name specific
> driver releases. I did not find a generation→version table in these repos and do not
> assert one.

---

## Symbol resolution order — `symbol_get()`, and what "shim absent" means

The NVMEM providers do not link against NVIDIA at build time. They resolve function
pointers at runtime with the kernel's `symbol_get()`
([efa_nvmem_impl.h](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_nvmem_impl.h)):

```c
static int nvmem_get_fp(struct efa_nvmem_ops *ops)
{
    if (!nvmem_get_peermem_fp(ops))   /* prefer the AWS shim   */
        return 0;
    return nvmem_get_nvidia_fp(ops);  /* fall back to NVIDIA direct */
}
```

- `nvmem_get_peermem_fp()` calls `symbol_get(efa_nv_peermem_p2p_get_pages)` and the other
  three shim symbols; on success it sets `ops->using_peermem_fp = true`. On partial
  failure it unwinds with `symbol_put()` on whatever it already took and returns
  `-EINVAL`.
- `nvmem_get_nvidia_fp()` does the same against the raw `nvidia_p2p_*` symbols and leaves
  `using_peermem_fp` false.
- `nvmem_put_fp()` releases whichever set was taken, keyed on `ops->using_peermem_fp`.

Because these are `symbol_get()` lookups and **not** link-time dependencies,
`efa_nv_peermem.ko` is an **optional soft dependency**. Two independent operational
consequences:

1. **Shim present (loaded):** the NVMEM provider binds the `efa_nv_peermem_*` symbols and
   all GPU pin/map/unmap/put calls flow `efa.ko → efa_nv_peermem.ko → nvidia.ko`. The
   provider string reports **`"NVIDIA peermem"`** (`nvmem_provider_string()`).
2. **Shim absent (not loaded):** `nvmem_get_peermem_fp()` fails, `nvmem_get_nvidia_fp()`
   succeeds, and calls flow `efa.ko → nvidia.ko` directly, **silently**. The provider
   string reports **`"NVIDIA"`**.

The fallback is silent — no error, no warning — so *the presence or absence of
`efa_nv_peermem.ko` changes which code path runs* with no other visible signal except
the provider string. When debugging a GPUDirect problem, always establish which of the
two is active (see Debugging below).

### Auto-load: how the shim gets pulled in

`nvmem_get_provider()` (called once at `p2p_providers_init()`) does the load-time nudging,
verified in `efa_nvmem_impl.h`:

```c
err = request_module("nvidia");
if (!err) {
    err = nvmem_get_nvidia_fp(&ops);   /* are direct symbols already available? */
    if (err)
        request_module("efa_nv_peermem");  /* no → try to bring in the shim */
    else
        nvmem_put_fp(&ops);            /* yes → release, we were only probing */
}
```

So the driver first asks for `nvidia`; if the direct `nvidia_p2p_*` symbols are *not*
resolvable (which is the case when the GPU driver exports them GPL-only such that a
non-GPL consumer cannot bind, or when they are otherwise unavailable), it requests
`efa_nv_peermem`. This is why the README says the shim "will be requested by EFA driver
whenever it is needed" and need not be auto-started at boot.

---

## Why EFA does not use `nvidia-peermem` — the most important correction

**GPUDirect RDMA on Mellanox/NVIDIA NICs requires `nvidia-peermem`. EFA does not.** This
is a trap: an engineer or agent debugging "GPUDirect not working on EFA" who goes looking
for `nvidia-peermem` (loaded? symbols? version?) is investigating the wrong module
entirely.

**What `nvidia-peermem` actually is** (verified in
[nvidia-peermem.c](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/kernel-open/nvidia-peermem/nvidia-peermem.c)):

- It registers itself with `ib_core` via `ib_register_peer_memory_client()`, passing a
  `struct peer_memory_client` / `struct peer_memory_client_ex` — the **MLNX_OFED
  "peer-direct"** interface. Its client callbacks are `nv_mem_acquire`, `nv_mem_get_pages`,
  `nv_dma_map`, `nv_dma_unmap`, `nv_mem_put_pages`, `nv_mem_release`.
- `MODULE_AUTHOR("Yishai Hadas")` (a Mellanox engineer) and
  `MODULE_DESCRIPTION("NVIDIA GPU memory plug-in")`; internal `DRV_NAME "nv_mem"`.
- Module params `peerdirect_support` (whose description literally references
  *"legacy, for example MLNX_OFED 4.9 LTS"*) and `persistent_api_support`.
- The whole client body is compiled only under `#if defined(NV_MLNX_IB_PEER_MEM_SYMBOLS_PRESENT)`
  — i.e. it is inert unless the kernel's `ib_core` carries the MLNX peer-memory symbols.
  That `ib_peer_memory_client` API is **not in mainline `ib_core`**; it is an
  MLNX_OFED / patched-kernel extension.

**The evidence that EFA does not use it:** grepping the entire EFA driver source
(`amzn-drivers/kernel/linux/efa/src/`) for `peer_memory_client`, `nvidia-peermem`,
`nvidia_peermem`, and `ib_register_peer` yields **zero matches.** EFA never registers as
a peer-memory client. Instead it calls `nvidia_p2p_*` from `nv-p2p.h` **directly**, or
through the AWS `efa_nv_peermem` shim.

**Architectural contrast:**

| | Mellanox / NVIDIA NIC | EFA |
|---|---|---|
| Who drives the pin/map | `ib_core` calls into `nvidia-peermem`'s client callbacks | EFA's own NVMEM provider calls `nvidia_p2p_*` |
| Registration model | `nvidia-peermem` registers *with* `ib_core` (`ib_register_peer_memory_client`) | EFA resolves `nvidia_p2p_*` via `symbol_get` and calls it itself |
| Required extra module | `nvidia-peermem.ko` | `efa_nv_peermem.ko` (optional) or nothing |
| Required kernel patch | MLNX_OFED `ib_peer_memory` symbols | none — uses stock `symbol_get` |

Both approaches ultimately call the *same* NVIDIA API (`nvidia_p2p_get_pages` etc.); they
differ only in *who* orchestrates the calls. On EFA it is the EFA driver; on Mellanox it
is `ib_core` via the peer-direct plug-in.

---

## Revocation safety — free callbacks, tickets, and how DMA-BUF differs

GPU memory can be freed by the GPU driver **asynchronously**, from the GPU driver's own
context, while EFA still has an MR referencing it. Each of the three kernel paths handles
this hazard differently.

### Peer-memory path (NVMEM / Neuron): the ticket mechanism

When the NVMEM provider pins pages it passes a **free callback** and an opaque `data`
argument that is the **ticket** (not a pointer):

```c
/* efa_nvmem_impl.h */
static void nvmem_free_cb(void *data)
{
    pr_debug("Free callback ticket %llu\n", (u64)data);
    efa_p2p_put((u64)data, true);   /* in_cb = true */
}
```

`nvmem_get_pages()` registers this via
`ops.get_pages(0, 0, addr, size, &pgtbl, nvmem_free_cb, (void *)ticket)`.
`efa_neuronmem.c` has the identical pattern (`neuronmem_free_cb` → `efa_p2p_put(ticket, true)`).
When the GPU/Neuron driver frees the allocation, it invokes the callback; EFA looks the
mapping up **by ticket** through `ticket_to_p2p()` on the global `p2p_list`
([efa_p2p.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_p2p.c))
and tears the MR down. Passing the *ticket* rather than the `efa_p2pmem` pointer avoids
dereferencing a structure the other driver may be racing to free. The `in_cb=true`
argument tells `release()` **not** to call `put_pages`/`dma_unmap` — the GPU driver is
already tearing those down and doing so from inside its callback would double-free
(see `nvmem_release()`'s `if (!in_cb)` guard). This is the mechanism documented from the
EFA side in `kernel-efa-driver.md` §3b; this document supplies the NVIDIA/Neuron end of it.

For contrast, NVIDIA's own `nvidia-peermem` solves the same hazard in *its* world with
`nv_get_p2p_free_callback()` invoking `ib_core`'s `mem_invalidate_callback` plus a
`callback_task` guard so `nv_dma_unmap`/`put_pages` become no-ops during the invalidation —
but that path is unused by EFA (see above). It is shown here only to make clear the two
designs solve the same problem in different subsystems.

### DMA-BUF path: `move_notify` / attachment invalidation

The DMA-BUF path does **not** use the EFA ticket list at all. Revocation is handled by
the kernel `dma_buf` framework's reservation/attachment invalidation. EFA imports with
`ib_umem_dmabuf_get_pinned()` (`efa_verbs.c`), which produces a **pinned** attachment;
where the pinned import is unavailable the driver manages the `dma_resv` lock and
`dma_buf_unpin()` explicitly on teardown (see the `#ifndef HAVE_IB_UMEM_DMABUF_PINNED`
blocks in `efa_reg_user_mr_dmabuf()` and `efa_unpin_dmabuf()`). In the movable
(non-pinned) case, the exporter signals `move_notify` and `ib_core`/`ib_umem_dmabuf`
re-maps or invalidates the attachment. Either way, the invalidation is a **dma_buf-layer
event**, not an `nvidia_p2p` free callback, and it does **not** go through
`efa_p2p_put()` or the ticket list — the two revocation mechanisms are fully independent,
matching the fact that DMA-BUF is not an `efa_p2p` provider.

Practical consequence: a "GPU memory freed under an active MR" hazard is real on both
paths, but you debug it in different places — the peer-memory path through
`efa_p2p`/`nvmem_free_cb` and dmesg ticket traces; the DMA-BUF path through the dma_buf
attachment / `dma_resv` machinery.

---

## Debugging: which modules, what to grep, which path is active

**1. Which registration path is even in play?**
- If the OFI plugin logged `Attempting to use DMA-BUF capable providers` (from
  `nccl_ofi_dmabuf_viable()`), and libfabric logged `Registering dmabuf MR: fd=...` or
  `FI_MR_DMABUF: fd=...` (from `efa_mr.c`), you are on the **DMA-BUF** path — the NVMEM
  providers and both peermem modules are irrelevant.
- If you see `Fall back to ibv_reg_mr` or DMA-BUF was declared not viable
  (`Will not attempt to use DMA-BUF, ...`), the registration reaches the kernel as a VA
  and lands on the **peer-memory** path.

**2. Which peer-memory provider / shim is active?**
- `dmesg | grep -i 'Acquired peer memory using P2P'` — printed once by `efa_p2p_get()`
  when a provider first claims memory.
- The provider string (`efa_p2p_provider_string()` → first available provider):
  `"NVIDIA peermem"` means the **shim is loaded and bound**; `"NVIDIA"` means **direct**
  `nvidia_p2p_*`; `"NEURON"` means the Neuron provider; empty means none resolved.
- `lsmod | grep -E 'efa_nv_peermem|nvidia'` — is the shim actually loaded?
  `modinfo efa_nv_peermem` should show version `1.2.3`, `softdep: pre: nvidia`.
- `dmesg | grep efa_nv_peermem` — the load-time gate messages:
  `Nvidia proprietary is unsupported` (taint gate tripped → you are on the proprietary
  driver, shim will not load) or `Nvidia P2P symbols are unavailable` (nvidia not
  providing the symbols) or the success banner `EFA NV Peermem v1.2.3`.

**3. Common misdiagnoses**
- *Looking for `nvidia-peermem` on EFA.* Wrong module — EFA never uses it (see above).
  `lsmod | grep nvidia_peermem` being empty is **not** an EFA problem.
- *Assuming DMA-BUF is disabled on old EFA hardware.* No longer true; the device-id gate
  was removed. If DMA-BUF is off, look at libfabric version (≥ 1.20), kernel version
  (≥ 5.12), `FI_HMEM` capability, GPU dma_buf attribute, and `OFI_NCCL_DISABLE_DMABUF`.
- *Proprietary NVIDIA driver + expecting the shim.* The shim refuses to load against the
  proprietary driver by design; on that configuration the direct `nvidia_p2p_*` path (or
  DMA-BUF) is what carries GPUDirect.

**4. Symbol-level confirmation**
- `grep -rE 'peer_memory_client|nvidia_peermem' <efa driver src>` → expected **empty**;
  proves EFA is not a peer-memory client.
- `cat /proc/kallsyms | grep -E 'efa_nv_peermem_p2p_|nvidia_p2p_'` → shows which symbol
  set is exported and available for `symbol_get()`.

---

## Cross-references

- [kernel-efa-driver.md](kernel-efa-driver.md) §"3b. GPU & Accelerator Peer Memory (P2P
  Subsystem)" — the EFA driver's `efa_p2p_ops` vtable, provider array, and ticket revocation
  from the driver's side.
- [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md) — DMA-BUF from the plugin/libfabric side and the
  viability checks.
- [cuda-memory.md](cuda-memory.md) — CUDA allocation, pinning, and dmabuf export on the userspace side.
- `neuron-memory.md` — Trainium/Inferentia memory and the `neuron_p2p_*` userspace side.
- `rdma-memreg.md` / `lkey-rkey-explained.md` — generic MR registration and the `reg_mr`
  admin command both paths converge on.
- `rocm-memory.md` — the AMD/ROCr analogue (dmabuf path in `efa_hmem.c` for `FI_HMEM_ROCR`).
