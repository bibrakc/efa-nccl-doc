# DMA-BUF and GPU Memory Registration

> **Source lookups:** this document records mechanism, defaults, corrections and history. For current function bodies, call graphs and blast radius, query the code graph (`codegraph explore <symbol>`) rather than trusting a pasted copy here.

## Overview

**DMA-BUF** (DMA Buffer Sharing) is a Linux kernel framework for sharing memory buffers between different devices (GPU, NIC, camera, etc.) without copying data. For NCCL+EFA, dmabuf enables **direct GPU-to-NIC memory registration** for efficient RDMA transfers.

> Source: `src/nccl_ofi_dmabuf.cpp`. Use `codegraph explore nccl_ofi_dmabuf` for current bodies.

## Why DMA-BUF Matters

### Legacy GPU Memory Registration (Pre-dmabuf)

```
GPU Memory Registration (Old Way):
1. CUDA allocates GPU memory
2. Pin memory via nvidia driver
3. Get physical pages (nvidia-specific GDR)
4. Register with EFA via custom ioctl
5. Complex, vendor-specific, kernel 5.10 issues

Problems:
- Vendor-specific (NVIDIA GDR, AMD ROCm different APIs)
- Kernel version dependencies
- Complex pinning mechanisms
```

### Modern DMA-BUF Approach (Kernel 5.12+)

```
GPU Memory Registration (DMA-BUF):
1. CUDA allocates GPU memory
2. Export as dmabuf file descriptor (FD)
3. Pass FD to libfabric
4. libfabric/rdma-core import dmabuf
5. Standard Linux kernel interface!

Benefits:
- Vendor-neutral (works with NVIDIA, AMD, Intel)
- Standard kernel interface (since 5.12)
- Cleaner, simpler code
- Better page merging (less IOMMU entries)
```

## DMA-BUF Architecture

```
┌──────────────────────────────────────────────┐
│   NCCL Application                           │
├──────────────────────────────────────────────┤
│   CUDA/HIP Runtime                           │
│   cudaMalloc() → GPU memory                  │
│   Export as dmabuf FD                        │
├──────────────────────────────────────────────┤
│   OFI NCCL Plugin                            │
│   nccl_ofi_dmabuf_viable_and_supported()    │
├──────────────────────────────────────────────┤
│   Libfabric                                  │
│   fi_mr_regattr(..., FI_MR_DMABUF)          │
├──────────────────────────────────────────────┤
│   rdma-core                                  │
│   ibv_reg_dmabuf_mr()                        │
├──────────────────────────────────────────────┤
│   Linux Kernel (5.12+)                       │
│   DMA-BUF subsystem                          │
│   - dma_buf_get(fd) → struct dma_buf        │
│   - dma_buf_map_attachment()                │
├──────────────────────────────────────────────┤
│   GPU Driver (nvidia/amdgpu)                 │
│   - Exports GPU memory as dmabuf            │
│   - Provides scatter-gather table           │
├──────────────────────────────────────────────┤
│   EFA Driver                                 │
│   - Imports dmabuf                           │
│   - Creates IOMMU mapping                    │
│   - Provides lkey/rkey                       │
└──────────────────────────────────────────────┘
```

## DMA-BUF Data Structures

The kernel `struct dma_buf` / `struct dma_buf_attachment` / `struct sg_table` /
`struct scatterlist`, the libfabric `struct fi_mr_dmabuf`, and the plugin's
`nccl_ofi_mr_ckey` are all live definitions the code graph serves verbatim. The facts worth
recording here:

- A dmabuf MR is keyed by **(fd, offset, len, base_addr)** — the `struct fi_mr_dmabuf` fields —
  whereas a CPU MR is keyed by an `iovec` (virt addr + len). The plugin's `nccl_ofi_mr_ckey`
  is a union of the two, discriminated by `type` (`NCCL_OFI_MR_CKEY_IOVEC` vs
  `NCCL_OFI_MR_CKEY_DMABUF`).
- On the kernel side the import walks a scatter-gather table (`sg_table` → `scatterlist`) whose
  entries carry `dma_address` (the IOMMU-mapped address) after `dma_buf_map_attachment()`.

> Source: `include/rdma/fi_domain.h` — `struct fi_mr_dmabuf`; kernel
> `include/linux/dma-buf.h` — `struct dma_buf`; `aws-ofi-nccl/include/nccl_ofi_mr.h` —
> `nccl_ofi_mr_ckey`, `nccl_ofi_mr_ckey_mk_dmabuf`. Use `codegraph explore nccl_ofi_mr_ckey`.

## DMA-BUF Registration Flow

### Complete Flow (GPU Memory → EFA RDMA)

```
1. Application allocates GPU memory
   cudaMalloc(&gpu_ptr, size);  // GPU address (not CPU accessible!)

2. CUDA exports the address range as a dmabuf FD (driver API, not runtime API)
   cuMemGetHandleForAddressRange(&dmabuf_fd, aligned_ptr, aligned_size,
                                 CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags);
   // dmabuf_fd is now a file descriptor

3. NCCL calls plugin regmr with DMABUF type
   nccl_net_ofi_regmr_dmabuf(comm, data, size, type, dmabuf_fd, offset, &mhandle);

4. OFI Plugin checks if dmabuf is viable
   if (!nccl_ofi_dmabuf_viable_and_supported(provider_info)) {
       // Fall back to legacy registration
   }

5. OFI Plugin calls libfabric with FI_MR_DMABUF
   struct fi_mr_attr attr;
   attr.dmabuf = &fi_mr_dmabuf{fd, offset, len, base_addr};
   fi_mr_regattr(domain, &attr, FI_MR_DMABUF, &mr);

6. Libfabric EFA provider calls rdma-core
   ibv_reg_dmabuf_mr(pd, offset, length, dmabuf_fd, access_flags);

7. rdma-core ioctl to kernel
   struct ib_uverbs_reg_dmabuf_mr cmd = { .fd, .offset, .length, .access_flags };
   write(cmd_fd, &cmd, sizeof(cmd));

8. Kernel EFA driver:
   a. dma_buf_get(fd)
   b. dma_buf_attach(dmabuf, efa_dev)
   c. dma_buf_map_attachment(attachment, DMA_BIDIRECTIONAL)  → scatter-gather table
   d. Create IOMMU mapping per page
   e. Program EFA hardware; generate lkey/rkey

9-11. Kernel returns lkey/rkey; plugin caches MR; future sends use cached MR
      fi_send(ep, gpu_ptr, size, fi_mr_desc(mr_handle), ...);  // EFA DMAs from GPU memory
```

## DMA-BUF Viability Checks

The viability preconditions are the fact worth pinning here. `nccl_ofi_dmabuf_viable()`
returns true only when **all** of these hold:

1. **libfabric >= 1.20** with `FI_MR_DMABUF` available (`HAVE_DECL_FI_MR_DMABUF`).
2. The user has **not** disabled it — env var **`OFI_NCCL_DISABLE_DMABUF`** (`ofi_nccl_disable_dmabuf()`).
3. The **GPU reports dmabuf support** (`nccl_net_ofi_gpu_have_dma_buf_attr()`), under `HAVE_GPU`.
4. **Kernel >= 5.12** (RDMA dmabuf ioctls, added in kernel commit `bfe0cc6`).

`nccl_ofi_dmabuf_viable_and_supported(nic_prov)` additionally requires the provider to
advertise **`FI_HMEM`** and an **API version >= 1.20**:

```c
// IDIOM — the full gate (post-0f285d5): FI_HMEM + API>=1.20 + viable(); NO device-id check.
bool dmabuf_support = ((nic_prov->caps & FI_HMEM) != 0) &&
    FI_VERSION_GE(nic_prov->fabric_attr->api_version, FI_VERSION(1, 20)) &&
    nccl_ofi_dmabuf_viable();
```

The kernel-version check is a small `uname()` + `sscanf("%d.%d")` idiom requiring
`(maj == 5 && min >= 12) || maj > 5`.

> Source: `src/nccl_ofi_dmabuf.cpp` — `nccl_ofi_dmabuf_viable()`,
> `nccl_ofi_dmabuf_viable_and_supported()`, `kernel_version_rdma_dmabuf_ioctl_ok()`. Use
> `codegraph explore nccl_ofi_dmabuf_viable` for current bodies.

> **Correction (was: EFA Gen 4+ required).** Earlier revisions of this document (and older
> plugin code) disabled DMA-BUF on EFA Gen 1-3 (`0xefa0` / `0xefa1` / `0xefa2`) inside
> `nccl_ofi_dmabuf_viable_and_supported()`, citing a firmware page-merging issue. That
> device-id gating was **removed** in commit `0f285d5` (*"dma-buf: Don't disable dma-buf on
> EFAv1-3"*). DMA-BUF viability no longer depends on the EFA generation.

## EFA Hardware Generation Support

**DMA-BUF is NO LONGER gated by EFA hardware generation.** The Gen 1-3 device-id check
(`0xefa0` / `0xefa1` / `0xefa2`) was **removed** in aws-ofi-nccl commit `0f285d5`
(*"dma-buf: Don't disable dma-buf on EFAv1-3"*). This is an *absence* the code graph cannot
express, and it is the single most important fact in this file. Historically the plugin
disabled DMA-BUF on EFA Gen 1-3 due to a firmware/kernel page-merging issue; that issue was
addressed on the kernel side, so the plugin-side generation gating could be removed.

| EFA Generation | Device ID | DMA-BUF Support (current plugin) | Historical note |
|----------------|-----------|----------------------------------|-----------------|
| Gen 1 | 0xefa0 | ✅ Yes (no longer gated) | previously disabled for page merging |
| Gen 2 | 0xefa1 | ✅ Yes (no longer gated) | previously disabled for page merging |
| Gen 3 | 0xefa2 | ✅ Yes (no longer gated) | previously disabled for page merging |
| Gen 4+ | 0xefa3+ | ✅ Yes | always supported |

**Historical Page-Merging Issue**: Early EFA generations + old kernel dmabuf code (pre-kernel
patch) produced too many IOMMU page entries → communication failures. Fixed on the kernel
side, which is why the plugin-side generation gating could be removed.

**Kernel Patch**: https://web.git.kernel.org/pub/scm/linux/kernel/git/rdma/rdma.git/commit/?id=486055f5e09df9

### EFA driver r3.3.0: >4GB MR and extended page-shift

The EFA kernel driver **r3.3.0** added support for memory-region page sizes **greater than
4 GB** and an **extended page-shift field** in MR registration (amzn-drivers commits
[`bf83e44`](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/) and
[`a1e35dc`](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/)). This is
relevant to **large GPU memory registrations**: a single very large registered region can be
described with fewer, larger page entries instead of failing or fragmenting into many small
entries. EFA-GDA / GDAKI on P5en / P6 requires EFA driver ≥ 3.3.0.

## Compile Guards (VALUE)

The real compile guards for dmabuf support are:

- **`HAVE_MR_DMABUF`** — libfabric/plugin dmabuf MR support present.
- **`HAVE_IB_UMEM_DMABUF_PINNED`** — kernel `ib_umem` pinned-dmabuf support.

> **`HAVE_IB_REG_USER_MR_DMABUF` does not exist.** If you see it referenced anywhere, it is
> wrong — the two guards above are the real ones.

## DMA-BUF vs Legacy Registration

### Performance Comparison

| Metric | Legacy (GDR) | DMA-BUF |
|--------|--------------|---------|
| Kernel version required | 5.4+ | 5.12+ |
| Registration time | 100-500 μs | 100-500 μs (similar) |
| IOMMU entries (4GB GPU buffer) | ~1M entries | ~4K entries (merged!) |
| Vendor support | NVIDIA only (GDR) | All (NVIDIA, AMD, Intel) |
| Code complexity | High (vendor-specific) | Low (standard kernel API) |
| Mainline kernel | Requires out-of-tree module | Built-in |

**Key Advantage**: Better page merging → Fewer IOMMU entries → Lower memory overhead

### FORMULA: IOMMU Entry Reduction via Page Merging

```
GPU Buffer: 4 GB (1M pages @ 4KB each)

Legacy Registration:
- GPU memory fragmented into 4KB pages
- 1,048,576 IOMMU entries
- Large memory overhead for page tables

DMA-BUF Registration (with page merging):
- Kernel merges contiguous 4KB pages into 2MB huge pages
- 2,048 IOMMU entries          (4 GiB / 2 MiB = 2,048)
- 512x fewer entries           (1,048,576 / 2,048 = 512) → lower overhead

Derivation:  entries_merged = ceil(buffer_size / 2MB)
             entries_unmerged = ceil(buffer_size / 4KB)
             reduction_factor = 2MB / 4KB = 512   (best case, fully contiguous)
```

## Configuration and Tuning

### Enable DMA-BUF

```bash
# Default: enabled on supported systems
# To explicitly disable:
export OFI_NCCL_DISABLE_DMABUF=1

# Check if being used (NCCL debug output):
export NCCL_DEBUG=INFO
# Look for: "Attempting to use DMA-BUF capable providers"
```

### Check DMA-BUF Status

```bash
# Check kernel version
uname -r
# Should be 5.12 or higher for dmabuf support

# Check EFA generation (informational only; no longer gates dmabuf)
lspci -vv -s $(lspci | grep EFA | cut -d' ' -f1) | grep "Device ID"

# Check libfabric version
fi_info --version
# Should be 1.20 or higher

# Check GPU dmabuf support (NVIDIA)
nvidia-smi -q | grep "DMA-BUF"
```

### Debugging DMA-BUF Registration

```bash
# Enable detailed logging
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=NET
export FI_LOG_LEVEL=debug

# Run NCCL test
./nccl_tests/build/all_reduce_perf -b 8 -e 1G -g 8

# Look for dmabuf registration messages:
# "Going to register mr dmabuf fd=42 size=..."
# "Found MR handle 0x... for dmabuf in cache slot 5"
```

### Check IOMMU Page Tables

```bash
# Show IOMMU groups
find /sys/kernel/iommu_groups/ -type l

# EFA IOMMU mapping
cat /sys/kernel/iommu_groups/*/devices/*/iommu_group

# Kernel messages for page merging
dmesg | grep -i "ib_umem\|dmabuf\|page.*merg"
```

## DMA-BUF Memory Registration Code Path

### CUDA Export (Application Side)

**IDIOM — export a CUDA allocation as a dmabuf FD.** This is a **driver-API** call, and the
plugin's own implementation is the reference
([src/nccl_ofi_cuda.cpp, `nccl_net_ofi_gpu_get_dma_buf_fd()`, lines 487-513](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_cuda.cpp)):

```c
/* ptr and size MUST be page-aligned; the plugin asserts this. */
unsigned long long flags = 0;
#if HAVE_CUDA_DMABUF_MAPPING_TYPE_PCIE
flags = CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
#endif

CUresult ret = cuMemGetHandleForAddressRange(
        &dmabuf_fd, (uintptr_t)aligned_ptr, aligned_size,
        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags);

/* Older drivers reject the PCIe mapping-type flag; retry without it. */
if ((ret == CUDA_ERROR_INVALID_VALUE || ret == CUDA_ERROR_NOT_SUPPORTED) && flags != 0)
        ret = cuMemGetHandleForAddressRange(
                &dmabuf_fd, (uintptr_t)aligned_ptr, aligned_size,
                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
```

Three things this idiom encodes, none of them guessable:

- `cuMemGetHandleForAddressRange` requires **CUDA 11.7+** (the plugin declares it with a
  `11070` version guard) and is resolved dynamically, not linked.
- **The pointer and size must be page-aligned**, and the plugin asserts both before calling.
- The **retry without flags** is load-bearing: drivers that predate the PCIe mapping type
  return `CUDA_ERROR_INVALID_VALUE` or `CUDA_ERROR_NOT_SUPPORTED`, and the second attempt is
  what makes DMA-BUF work on them.

> **Do not use `cudaImportExternalMemory()` or `cudaExternalMemoryGetMappedBuffer()` for this.**
> Both run the *other* direction — they consume an fd you already hold and hand back a device
> pointer. In `cudaExternalMemoryHandleDesc`, `handle.fd` is an **input** field. An earlier
> revision of this document showed them as the export idiom; that was wrong. See
> [cuda-memory.md](cuda-memory.md) for the import direction.

### OFI Plugin / Kernel Registration

The plugin's `nccl_net_ofi_regmr_dmabuf()` builds a dmabuf cache key, consults the MR cache
(a C++ class living on the domain, guarded by `domain->mr_cache_lock` — see
[mr-cache-implementation.md](mr-cache-implementation.md)), and on a miss calls
`fi_mr_regattr(..., FI_MR_DMABUF)`. The EFA kernel path (`efa_reg_dmabuf_mr` /
`efa_dereg_dmabuf_mr`) does `dma_buf_get` → `dma_buf_attach` →
`dma_buf_map_attachment(DMA_BIDIRECTIONAL)` → build page list → program hardware, and unwinds
the same on deregister. **On EFA, `lkey == rkey`.**

> Source: `aws-ofi-nccl/src/nccl_ofi_dmabuf.cpp` — `nccl_net_ofi_regmr_dmabuf`;
> `amzn-drivers/kernel/linux/efa/src/efa_verbs.c` — `efa_reg_dmabuf_mr`, `efa_dereg_dmabuf_mr`.
> Use `codegraph explore nccl_net_ofi_regmr_dmabuf`.

## Page Merging in Detail

### FORMULA: Why IOMMU Entry Count Matters

```
GPU Memory: 1 GB buffer
Without page merging:
    262,144 pages @ 4KB each  →  262,144 IOMMU entries  →  ~1 MB page-table memory
With page merging:
    512 huge pages @ 2MB each →  512 IOMMU entries      →  ~2 KB page-table memory
    (512x fewer entries)
```

The kernel-side fix that enables this (kernel 5.19+) merges contiguous 4KB pages into 2MB
chunks in `ib_umem_dmabuf_map_pages()` via `sg_alloc_table_from_pages_segment(..., SZ_2M, ...)`.

> Source: kernel `drivers/infiniband/core/umem_dmabuf.c` — `ib_umem_dmabuf_map_pages()`.

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| `cudaExportMemory()` | 10-50 μs | One-time per buffer |
| `fi_mr_regattr()` (dmabuf) | 100-500 μs | Pins pages, creates IOMMU mapping |
| `fi_send()` (dmabuf MR) | 10-20 μs | Same as regular MR |
| IOMMU lookup (merged) | <10 ns | Few entries, stays in TLB |
| IOMMU lookup (unmerged) | 50-100 ns | Many entries, TLB misses |

**Key Insight**: Page merging doesn't speed up registration much, but significantly reduces IOMMU overhead during transfers.

## Summary

| Aspect | Details |
|--------|---------|
| **Kernel requirement** | Linux 5.12+ (RDMA dmabuf ioctls) |
| **Libfabric requirement** | 1.20+ (FI_MR_DMABUF flag) |
| **EFA requirement** | All EFA generations (Gen 1-3 gating removed, commit `0f285d5`) |
| **GPU requirement** | CUDA 11.7+, ROCm 5.0+, or any dmabuf-capable GPU |
| **API** | `fi_mr_regattr(..., FI_MR_DMABUF)` |
| **Compile guards** | `HAVE_MR_DMABUF`, `HAVE_IB_UMEM_DMABUF_PINNED` (not `HAVE_IB_REG_USER_MR_DMABUF`) |
| **Env var** | `OFI_NCCL_DISABLE_DMABUF` |
| **Key benefit** | Vendor-neutral, better page merging |
| **IOMMU reduction** | Up to ~512x fewer entries (4KB → 2MB pages) |

**Key Takeaways**:
1. DMA-BUF is the **modern, standard way** to register GPU memory for RDMA
2. Requires kernel 5.12+, libfabric 1.20+; **no longer gated by EFA generation** (works on
   Gen 1-3 too since commit `0f285d5`)
3. **Vendor-neutral**: Works with NVIDIA, AMD, Intel GPUs
4. **Better IOMMU efficiency**: Page merging can significantly reduce IOMMU entries
5. Same performance as legacy GDR, but cleaner and more maintainable
6. Registration is still expensive (100-500 μs) - **caching is critical**
7. **EFA driver r3.3.0** adds >4GB MR page-size support + an extended page-shift field,
   improving large GPU registrations (amzn-drivers `bf83e44`, `a1e35dc`)

### libfabric 2.7 hmem/cuda additions (relevant to DMA-BUF export)

libfabric 2.7 (`main`, verified against `cb6364e05`, 2.7.0rc1) adds several hmem/CUDA
capabilities that touch the DMA-BUF export path:

- **Async copy operations** in the CUDA hmem interface (e.g. `cuMemcpyDtoDAsync` /
  `cudaMemcpyAsync` wrappers in [prov core hmem_cuda.c](https://github.com/ofiwg/libfabric/blob/main/src/hmem_cuda.c)).
- **Exposed CUDA driver functions** for context/memory management, so the core can manage
  CUDA context and allocations via the driver API rather than the runtime.
- **`cuda_get_dmabuf_fd()` skips `cuMemGetAddressRange()`**
  ([src/hmem_cuda.c](https://github.com/ofiwg/libfabric/blob/main/src/hmem_cuda.c)): it now
  uses the caller-provided address and size directly and only page-aligns them before calling
  `cuMemGetHandleForAddressRange`. This matters for **CUDA VMM allocations** with multiple
  physical segments mapped into one contiguous VA range, where `cuMemGetAddressRange()` would
  return only the size of the individual segment containing the pointer, not the full reserved
  VA range — which would truncate the exported dmabuf.

**Related Documentation**:
- [mr-cache-implementation.md](mr-cache-implementation.md) - MR cache (works with dmabuf keys)
- [rdma-memreg.md](rdma-memreg.md) - General memory registration concepts
- [rdma-core-and-verbs.md](rdma-core-and-verbs.md) - rdma-core API that dmabuf uses
- [kernel-efa-driver.md](kernel-efa-driver.md) - Kernel driver dmabuf implementation

---

## Code References

Bodies and struct definitions live in source; query the code graph.

**Linux Kernel (DMA-BUF subsystem)** — `include/linux/dma-buf.h`,
`drivers/infiniband/core/umem_dmabuf.c`: `struct dma_buf`, `struct dma_buf_attachment`,
`struct sg_table`, `struct scatterlist`; `dma_buf_get/put/attach/detach/map_attachment`;
`ib_umem_dmabuf_map_pages`.

**libfabric (OFI)** — `include/rdma/fi_domain.h`: `struct fi_mr_dmabuf`; `fi_mr_regattr`.

**rdma-core (libibverbs)**: `ibv_reg_dmabuf_mr`.

**aws-ofi-nccl** — `include/nccl_ofi_mr.h`, `src/nccl_ofi_dmabuf.cpp`: `nccl_ofi_mr_ckey`,
`nccl_ofi_mr_ckey_mk_dmabuf`, `nccl_ofi_dmabuf_viable`,
`nccl_ofi_dmabuf_viable_and_supported`, `nccl_net_ofi_regmr_dmabuf`.

**amzn-drivers (EFA)** — `kernel/linux/efa/src/efa_verbs.c`: `efa_reg_dmabuf_mr`,
`efa_dereg_dmabuf_mr`.

**CUDA Driver API (external — NVIDIA)**: `cuMemGetHandleForAddressRange`.

All Linux kernel references link to the [torvalds/linux](https://github.com/torvalds/linux) repository (`master`).
All libfabric references use branch-form links into [ofiwg/libfabric](https://github.com/ofiwg/libfabric/blob/main/) (`main`); verified against libfabric `main` `cb6364e05` (2.7.0rc1).
