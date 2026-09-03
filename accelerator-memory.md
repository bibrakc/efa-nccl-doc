# Non-NVIDIA Accelerator Memory Registration (ROCm + Neuron)

> **Source lookups:** this document records mechanism, defaults, corrections and history. For current function bodies, call graphs and blast radius, query the code graph (`codegraph explore <symbol>`) rather than trusting a pasted copy here.

## Overview

This document covers memory registration for the **non-NVIDIA accelerator paths** that the OFI
NCCL plugin supports:

- **AMD ROCm** — GPUs programmed via **HIP** (AMD's CUDA-compatible API). Uses **DMA-BUF**
  (ROCm 5.0+), the same framework NVIDIA uses.
- **AWS Neuron** — **Trainium** and **Inferentia** ML accelerators. Does **NOT** use DMA-BUF;
  uses a custom **`neuron_p2p_*` peer-to-peer registration** API.

For the NVIDIA CUDA path see [cuda-memory.md](cuda-memory.md); for the shared DMA-BUF framework
see [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md).

## How These Differ From the CUDA Path (shared)

Rather than re-explaining CUDA in each vendor section, here is the single comparison. The
NVIDIA path (`cudaMalloc`/driver API, `cuMemGetHandleForAddressRange` dmabuf export, GPUDirect
flush, `NCCL_PTR_CUDA`) is the baseline.

| Aspect | NVIDIA CUDA | AMD ROCm | AWS Neuron |
|--------|-------------|----------|------------|
| **Alloc API** | `cudaMalloc` / `cuMemAlloc` | `hipMalloc` | Neuron runtime |
| **Copy API** | `cudaMemcpy` / driver | `hipMemcpy` | n/a (plugin path) |
| **Pointer query** | `cuPointerGetAttributes` | `hipPointerGetAttribute` | plugin/HMEM |
| **Memory export** | dmabuf FD (`cuMemGetHandleForAddressRange`) | dmabuf FD (`hipMemGetHandleForAddressRange`) | **none — no export** |
| **Registration verb** | `ibv_reg_dmabuf_mr()` | `ibv_reg_dmabuf_mr()` | **`ibv_reg_mr()`** (standard) |
| **Kernel import** | `dma_buf_get()` | `dma_buf_get()` | **`neuron_p2p_register_va()`** |
| **libfabric iface** | `FI_HMEM_CUDA` | `FI_HMEM_ROCR` | `FI_HMEM_NEURON` |
| **DMA-BUF flag** | `FI_MR_DMABUF` | `FI_MR_DMABUF` | **not used** |
| **NCCL ptr type** | `NCCL_PTR_CUDA` | `NCCL_PTR_CUDA` (same!) | `NCCL_PTR_NEURON` |
| **GPUDirect flush** | required on some archs | **not needed** | **not needed** |
| **Page alignment of MR key** | rounded to page boundaries | rounded to page boundaries | **not rounded** (Neuron handles internally) |
| **Kernel requirement** | 5.12+ (dmabuf) | 5.12+ (dmabuf) | no dmabuf kernel dependency |

Two headline structural facts the code graph cannot express because they are *absences*:

- **Neuron uses `neuron_p2p_*` P2P registration and does NOT use DMA-BUF.** There is no dmabuf
  FD, no `FI_MR_DMABUF`, no `dma_buf_get()` on the Neuron path.
- **ROCm reuses `NCCL_PTR_CUDA`** — there is no separate "ROCm" pointer type — and needs **no
  GPUDirect flush** (`nccl_net_ofi_gpu_have_gdr_support_attr()` returns false).

---

# AMD ROCm

## Overview

**AMD ROCm** (Radeon Open Compute) is the software stack for AMD GPUs, programmed via **HIP**
(Heterogeneous-Compute Interface for Portability), AMD's CUDA-equivalent API. The **HIP API is
CUDA-compatible** — most calls are drop-in renames (`cudaMalloc` → `hipMalloc`, etc.).

**Target Hardware**: AMD MI-series GPUs (MI50, MI100, MI200, MI300), Radeon Instinct
accelerators, any ROCm-compatible AMD GPU.

**Registration approach**: **DMA-BUF export requires ROCm 5.0+**; falls back to legacy methods
on older ROCm.

## Architecture Position

```
┌─────────────────────────────────────┐
│   NCCL Application                   │
├─────────────────────────────────────┤
│   HIP Runtime                        │
│   - hipMalloc() GPU memory          │
│   - hipMemGetHandleForAddressRange()│
├─────────────────────────────────────┤
│   OFI NCCL Plugin                    │
│   - Detects NCCL_PTR_CUDA support   │
│   - Uses dmabuf or legacy methods   │
├─────────────────────────────────────┤
│   Libfabric (FI_HMEM, FI_MR_DMABUF) │
├─────────────────────────────────────┤
│   rdma-core                          │
│   - ibv_reg_dmabuf_mr() (modern)    │
│   - ibv_reg_mr() (legacy)           │
├─────────────────────────────────────┤
│   EFA Kernel Driver (DMA-BUF import) │
├─────────────────────────────────────┤
│   amdgpu Driver                      │
│   - Exports GPU memory as dmabuf    │
├─────────────────────────────────────┤
│   AMD GPU Hardware                   │
└─────────────────────────────────────┘
```

## HIP API Facts (VALUE / IDIOM)

- HIP init queries **both** driver and runtime versions (`hipDriverGetVersion`,
  `hipRuntimeGetVersion`) and sets `cuda_flush = false` — **ROCm never needs GPUDirect flush**.
- Device-for-pointer uses `hipPointerGetAttribute` with `HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL`
  and `HIP_POINTER_ATTRIBUTE_MEMORY_TYPE`; a non-`hipMemoryTypeDevice` pointer is host memory.
- The HIP↔CUDA call mapping (IDIOM):

```c
hipMalloc(&ptr, size);                                  // cudaMalloc
hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);       // cudaMemcpy
hipPointerGetAttribute(&id, HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr);
hipFree(ptr);                                           // cudaFree
// NO flush call — ROCm handles coherency in hardware.
```

> Source: `src/nccl_ofi_rocm.cpp` — `nccl_net_ofi_gpu_init`,
> `nccl_net_ofi_get_gpu_device_for_addr`, `nccl_net_ofi_gpu_mem_alloc/free`,
> `nccl_net_ofi_gpu_mem_copy_host_to_device`. Use `codegraph explore nccl_ofi_rocm`.

## DMA-BUF Export (ROCm 5.0+)

**DMA-BUF export requires ROCm 5.0+**, detected at compile time by the presence of
`HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD`. That macro **is** the version-detection logic:

```c
// IDIOM / version detection — export a HIP allocation as a dmabuf FD (offset always 0 on ROCm)
#if defined(HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD)          // => ROCm 5.0+
    hipMemGetHandleForAddressRange(fd, (uintptr_t)aligned_ptr, aligned_size,
                                   HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    *offset = 0;
#else
    return -ENOTSUP;                                       // older ROCm: no dmabuf
#endif
```

The same macro gates capability reporting:
`nccl_net_ofi_gpu_have_dma_buf_attr()` returns true iff `HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD`
is defined; `nccl_net_ofi_gpu_have_gdr_support_attr()` returns **false** (not applicable).

> Source: `src/nccl_ofi_rocm.cpp` / `include/nccl_ofi_rocm.h` —
> `nccl_net_ofi_gpu_get_dma_buf_fd`, `nccl_net_ofi_gpu_have_dma_buf_attr`,
> `nccl_net_ofi_gpu_have_gdr_support_attr`. Use `codegraph explore nccl_net_ofi_gpu_get_dma_buf_fd`.

Once a dmabuf FD is obtained, ROCm follows the **identical** registration flow as CUDA — the
plugin checks `nccl_ofi_dmabuf_viable_and_supported()`, calls `fi_mr_regattr(FI_MR_DMABUF)`,
and the EFA driver imports via `dma_buf_get()` (amdgpu supplies the scatter-gather table). See
[dmabuf-gpu-memory.md](dmabuf-gpu-memory.md) for the shared path.

## ROCm Detection and Debugging

```bash
# Check ROCm version (need 5.0+ for dmabuf)
rocm-smi --showversion
hipconfig --version

# amdgpu driver + GPU visibility
lsmod | grep amdgpu
rocm-smi

# Build the plugin with ROCm
cd aws-ofi-nccl
./autogen.sh
./configure --with-hip=/opt/rocm --enable-platform-aws
grep HAVE_ROCM config.h      # #define HAVE_ROCM 1

# Compile-time dmabuf check
cat > test_dmabuf.cpp <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
int main() {
#ifdef HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD
    printf("DMA-BUF supported\n"); return 0;
#else
    printf("DMA-BUF NOT supported\n"); return 1;
#endif
}
EOF
hipcc test_dmabuf.cpp -o test_dmabuf && ./test_dmabuf

# Debug registration
export NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=NET FI_LOG_LEVEL=debug
# Look for: "Using HIP driver version X with runtime Y", "Will use DMA-BUF"

# MR cache controls (there is NO NCCL_OFI_MR_CACHE_SIZE variable)
export OFI_NCCL_MR_CACHE_DISABLE=0   # 0 = plugin MR cache enabled (default)
export FI_EFA_MR_MAX_CACHED_SIZE=<bytes>
export FI_EFA_MR_MAX_CACHED_COUNT=<n>
```

## ROCm Summary

| Aspect | Details |
|--------|---------|
| **Hardware** | AMD MI-series GPUs, Radeon Instinct |
| **API** | HIP (CUDA-compatible) |
| **DMA-BUF support** | **ROCm 5.0+** (`HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD`) |
| **NCCL ptr type** | `NCCL_PTR_CUDA` (same as NVIDIA!) |
| **Export API** | `hipMemGetHandleForAddressRange()` (offset always 0) |
| **GPUDirect flush** | **Not required** |
| **Kernel requirement** | Linux 5.12+ (for dmabuf) |
| **Registration** | `ibv_reg_dmabuf_mr()` (modern), legacy fallback |

---

# AWS Neuron

## Overview

**AWS Neuron** is the SDK for AWS **Trainium** and **Inferentia** accelerators used for ML
training and inference. **Neuron uses `neuron_p2p_*` peer-to-peer registration and does NOT use
DMA-BUF** — this is the defining fact of the Neuron path.

**Target Hardware**:
- **AWS Trainium** (trn1, **trn2**) — training accelerators.
- **AWS Inferentia** (inf1, inf2) — inference accelerators.
- **Trn2 is CI-tested** (commit `b3af487`, *".ci/aws: Enable Trn2 testing"*), so the Neuron
  path is exercised on Trainium2 in addition to earlier Trn/Inf generations.

## Architecture Position

```
┌─────────────────────────────────────┐
│   NCCL Application                   │
├─────────────────────────────────────┤
│   Neuron Runtime                     │
│   - Allocates Neuron device memory  │
│   - Exports via neuron_p2p API      │
├─────────────────────────────────────┤
│   OFI NCCL Plugin                    │
│   - Detects NCCL_PTR_NEURON         │
│   - Uses standard regMr (no dmabuf) │
├─────────────────────────────────────┤
│   Libfabric (FI_HMEM_NEURON)         │
├─────────────────────────────────────┤
│   rdma-core                          │
│   - Standard ibv_reg_mr()           │
├─────────────────────────────────────┤
│   EFA Kernel Driver                  │
│   - neuron_p2p_register_va()        │ ← Neuron-specific!
│   - efa_neuronmem module            │
├─────────────────────────────────────┤
│   Neuron Driver                      │
│   - Physical page management        │
│   - P2P memory mapping              │
├─────────────────────────────────────┤
│   Trainium/Inferentia Hardware       │
└─────────────────────────────────────┘
```

## Page Size (VALUE)

```c
#define NEURON_PAGE_SHIFT 12
#define NEURON_PAGE_SIZE  BIT_ULL(NEURON_PAGE_SHIFT)   // 4096 bytes → 4 KB pages
```

- `NEURON_PAGE_SHIFT` = **12 → 4 KB pages**. Contrast with CUDA's `GPU_PAGE_SHIFT` = 16
  → **64 KB** page alignment. The kernel aligns Neuron VA ranges to `NEURON_PAGE_SIZE`
  (`ALIGN_DOWN(start, NEURON_PAGE_SIZE)` … `ALIGN(start+length, NEURON_PAGE_SIZE)`), and the
  page size actually used per registration is reported by the driver in
  `neuron_p2p_va_info.shift_page_size` (e.g. 12 for 4 KB).

## Neuron P2P Registration Flow

```
1. neuron_alloc(&neuron_ptr, size)          // Neuron runtime; no dmabuf export!
2. nccl_net_ofi_regMr(comm, neuron_ptr, size, NCCL_PTR_NEURON, &mhandle)
3. Plugin: FI_HMEM present → props->ptrSupport |= NCCL_PTR_NEURON
4. fi_mr_regattr(domain, {mr_iov=iovec, iface=FI_HMEM_NEURON}, 0, &mr)   // NOT FI_MR_DMABUF
5. ibv_reg_mr(pd, neuron_ptr, size, access)                             // standard verb
6. EFA kernel: neuronmem_get(dev, ticket, start, length)
7. neuron_p2p_register_va(addr, size, &va_info, free_cb, ticket)        // Neuron-specific
8. Neuron driver returns physical page list (neuron_p2p_va_info)
9. EFA expands pages → page_list, programs hardware, returns lkey/rkey
```

**FORMULA — expand the P2P page-info entries into EFA's flat page list:**

```c
// pa advances by one Neuron page (2^shift_page_size) per slot.
for (ent = 0; ent < va_info->entries; ent++) {
    pg = va_info->page_info + ent;
    pa = pg->physical_address;
    for (i = 0; i < pg->page_count; i++) {
        page_list[idx++] = pa;
        pa += BIT(va_info->shift_page_size);   // next page
    }
}
```

> Source: `amzn-drivers/kernel/linux/efa/src/efa_neuronmem.c` — `neuronmem_get`,
> `neuronmem_to_page_list`, `neuronmem_release`; `neuron_p2p.h` —
> `struct neuron_p2p_va_info`, `struct neuron_p2p_page_info`, `neuron_p2p_register_va`,
> `neuron_p2p_unregister_va`. Use `codegraph explore efa_neuronmem`.

## Plugin Interface Versions (VALUE — differs from NVIDIA)

The Neuron interface exports **`ncclNet_v4` / `v5` / `v6`** op-tables — **not** the higher
version the NVIDIA interface exports (the NVIDIA/GPU interface exports up to `ncclNet_v12`).
This version ceiling is a fact about the Neuron build, verified with
`grep -rn NCCL_OFI_EXPORT_SYMBOL src/nccl_ofi_interface_neuron.cpp`:

- `ncclNetPlugin_v6` — full table (has `.fini`, RMA `iwrite`/`iread`).
- `ncclNetPlugin_v5` — lacks `.fini` and the RMA iwrite/iread entry points.
- `ncclNetPlugin_v4` — uses `getProperties_v4` (smaller `ncclNetProperties_v4_t` field set).

Note the `v6` table still populates `.regMrDmaBuf` (present in the ABI) but **Neuron uses the
P2P path, not dmabuf** — the dmabuf entry point exists for interface completeness only.

> Source: `src/nccl_ofi_interface_neuron.cpp` — `ncclNetPlugin_v4/v5/v6`,
> `getProperties_v4/v5`. Use `codegraph explore nccl_ofi_interface_neuron`.

## Page Alignment: Neuron vs CUDA (IDIOM)

The MR cache key builder rounds to page boundaries **only for CUDA/ROCm**, never for Neuron:

```c
#if !HAVE_NEURON
    nccl_ofi_mr_ckey_round(&iov_len, &iov_base, "iovec");  // CUDA/ROCm: round to page
#endif
    // Neuron: NO rounding — register the exact address/size; the Neuron driver aligns.
```

Neuron requires **no page alignment** from the caller — the driver handles alignment
internally, so applications may register arbitrary address ranges.

> Source: `include/nccl_ofi_mr.h` — `nccl_ofi_mr_ckey_mk_vec`. Use
> `codegraph explore nccl_ofi_mr_ckey_mk_vec`.

## Neuron Detection and Debugging

```bash
# Neuron driver + devices
lsmod | grep neuron
ls -la /dev/neuron*

# EFA sees Neuron P2P
dmesg | grep -i "neuron\|NEURON"       # e.g. "efa 0000:00:06.0: P2P: NEURON"

# NCCL detection (automatic if neuron driver + FI_HMEM + efa_neuronmem present)
export NCCL_DEBUG=INFO
# Look for: "ptrSupport NCCL_PTR_HOST | NCCL_PTR_NEURON"

# Watch registration
echo 'module efa +p' > /sys/kernel/debug/dynamic_debug/control
dmesg -w | grep -i neuron
# "neuron_p2p_register_va called", "neuronmem_to_page_list: X pages"

# Confirm page size
dmesg | grep "shift_page_size"         # shift_page_size=12 → 4KB pages
```

## Neuron Summary

| Aspect | Details |
|--------|---------|
| **Hardware** | AWS Trainium (trn1, **trn2** CI-tested), Inferentia (inf1, inf2) |
| **API** | `neuron_p2p_register_va()` — **not DMA-BUF** |
| **NCCL ptr type** | `NCCL_PTR_NEURON` |
| **Kernel module** | `efa_neuronmem.c` |
| **Page size** | 4 KB (`NEURON_PAGE_SHIFT` = 12); CUDA uses 64 KB (`GPU_PAGE_SHIFT` = 16) |
| **Alignment** | **Not required** (Neuron driver aligns internally) |
| **libfabric** | `FI_HMEM_NEURON` interface |
| **Registration** | Standard `ibv_reg_mr()` (not dmabuf variant) |
| **Interface versions** | Exports `ncclNet_v4/v5/v6` (not v12 like the NVIDIA interface) |

**Key Takeaways**:
1. Neuron uses the **P2P model**, not DMA-BUF — no FD export, direct `neuron_p2p_register_va()`.
2. **No page alignment required**: applications may use arbitrary addresses (4 KB pages).
3. **Symbol-based**: uses `symbol_get()` to call Neuron driver functions.
4. Registration is still expensive (100-500 μs) — **MR caching critical**.
5. Works on **trn1 / inf1 / inf2 / trn2** (Trn2 is CI-tested).
6. Exports **`ncclNet_v4/v5/v6`** op-tables (not the v12 the NVIDIA interface exports).

---

## Shared: Related Upstream Changes

- **EFA driver r3.3.0** adds >4GB MR page-size support and an extended page-shift field in MR
  registration (amzn-drivers `bf83e44`, `a1e35dc`). ROCm registrations (dmabuf) and Neuron
  registrations (`neuron_p2p_*`) both benefit from the larger MR page-size support for large
  regions.
- **libfabric 2.7** (`main`, verified against `cb6364e05`, 2.7.0rc1) adds ROCr/hmem
  improvements paralleling the CUDA ones: async copy operations and dmabuf-fd retrieval via
  `rocr_hmem_get_dmabuf_fd()` / `rocr_hmem_put_dmabuf_fd()`
  ([src/hmem_rocr.c](https://github.com/ofiwg/libfabric/blob/main/src/hmem_rocr.c),
  used by [prov/efa/src/efa_hmem.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_hmem.c)).
- **DMA-BUF is no longer gated by EFA generation** (the Gen 1-3 device-id check was removed in
  commit `0f285d5`); this affects the ROCm dmabuf path. See
  [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md).

**Related Documentation**:
- [cuda-memory.md](cuda-memory.md) - NVIDIA CUDA path (the baseline compared against here)
- [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md) - DMA-BUF framework (used by ROCm, not Neuron)
- [mr-cache-implementation.md](mr-cache-implementation.md) - MR cache (works with all key types)
- [kernel-efa-driver.md](kernel-efa-driver.md) - EFA kernel driver integration

---

## Code References

Bodies and struct definitions live in source; query the code graph.

**AMD ROCm/HIP (external — AMD)**: `hipMalloc`, `hipMemcpy`, `hipFree`,
`hipDriverGetVersion`, `hipRuntimeGetVersion`, `hipPointerGetAttribute`,
`hipMemGetHandleForAddressRange` (ROCm 5.0+).

**aws-ofi-nccl (ROCm)** — `src/nccl_ofi_rocm.cpp`, `include/nccl_ofi_rocm.h`:
`nccl_net_ofi_gpu_init`, `nccl_net_ofi_get_gpu_device_for_addr`,
`nccl_net_ofi_gpu_get_dma_buf_fd`, `nccl_net_ofi_gpu_have_dma_buf_attr`,
`nccl_net_ofi_gpu_have_gdr_support_attr`.

**AWS Neuron SDK (external — proprietary)**: `neuron_p2p_register_va`,
`neuron_p2p_unregister_va`, `struct neuron_p2p_va_info`, `struct neuron_p2p_page_info`.

**amzn-drivers (EFA Neuron)** — `kernel/linux/efa/src/efa_neuronmem.c`, `.../neuron_p2p.h`:
`neuronmem_get`, `neuronmem_to_page_list`, `neuronmem_release`; `NEURON_PAGE_SHIFT`.

**aws-ofi-nccl (Neuron)** — `src/nccl_ofi_interface_neuron.cpp`, `include/nccl_ofi_mr.h`:
`ncclNetPlugin_v4/v5/v6`, `getProperties_v4/v5`, `nccl_ofi_mr_ckey_mk_vec`.

**rdma-core (libibverbs)**: `ibv_reg_dmabuf_mr` (ROCm, modern), `ibv_reg_mr` (both; Neuron and
ROCm legacy).

**libfabric (OFI)**: `fi_mr_regattr`; `src/hmem_rocr.c`; `prov/efa/src/efa_hmem.c`.

All aws-ofi-nccl references use branch-form links into [aws/aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl/blob/master/) (`master`).
All amzn-drivers references use branch-form links into [amzn/amzn-drivers](https://github.com/amzn/amzn-drivers/blob/master/) (`master`).
All libfabric references use branch-form links into [ofiwg/libfabric](https://github.com/ofiwg/libfabric/blob/main/) (`main`); verified against libfabric `main` `cb6364e05` (2.7.0rc1).
