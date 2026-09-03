# NVIDIA CUDA Memory Registration

> **Source lookups:** this document records mechanism, defaults, corrections and history. For current function bodies, call graphs and blast radius, query the code graph (`codegraph explore <symbol>`) rather than trusting a pasted copy here.

## Overview

**NVIDIA CUDA** is the dominant GPU computing platform. CUDA memory registration for RDMA involves complex version handling, GPUDirect RDMA support, and optional GDRCopy for fast CPU-GPU transfers. This document covers CUDA-specific aspects beyond the general DMA-BUF framework.

**Target Hardware**:
- NVIDIA GPUs (Tesla, A100, H100, etc.)
- GPUDirect RDMA capable devices
- PCIe Gen3+ recommended

> **Note on indexing:** `.cu` / `.cuh` files are **not indexed by the code graph** (unsupported
> extension). The GDAKI device code under `3rd-party/efa-gda/CUDA/` must therefore be read
> directly from source — `codegraph explore` will not return it.

## Architecture Position

```
┌─────────────────────────────────────┐
│   NCCL Application                   │
├─────────────────────────────────────┤
│   CUDA Runtime/Driver                │
│   - cudaMalloc() / cuMemAlloc()     │
│   - cuMemGetHandleForAddressRange() │
│   - cuFlushGPUDirectRDMAWrites()    │
├─────────────────────────────────────┤
│   OFI NCCL Plugin                    │
│   - Dynamic CUDA loading            │
│   - Version-agnostic function calls │
│   - GDRCopy integration (optional)  │
├─────────────────────────────────────┤
│   Libfabric                          │
│   - FI_MR_DMABUF (CUDA 11.7+)       │
│   - FI_HMEM support                 │
├─────────────────────────────────────┤
│   rdma-core                          │
│   - ibv_reg_dmabuf_mr()             │
├─────────────────────────────────────┤
│   EFA Kernel Driver                  │
│   - DMA-BUF import                  │
├─────────────────────────────────────┤
│   NVIDIA Driver                      │
│   - GPU memory management           │
│   - DMA-BUF export                  │
├─────────────────────────────────────┤
│   NVIDIA GPU Hardware                │
└─────────────────────────────────────┘
```

## CUDA Version Handling

### The Challenge: Cross-Version Compatibility

**Problem**: CUDA evolves rapidly, adding new APIs across versions:
- **CUDA 11.0**: GPUDirect flush support
- **CUDA 11.3–11.8**: `cudaGetDriverEntryPoint` **3-argument** variant (no `driverStatus`
  out-param). Build support for CUDA 11.3–11.8 was restored in commit `aab6529`.
- **CUDA 11.7**: DMA-BUF export support
- **CUDA 12.0**: Legacy entry point (`cudaGetDriverEntryPoint`, 4-argument with `driverStatus`)
- **CUDA 13.0**: Versioned entry point (`cudaGetDriverEntryPointByVersion`)

The OFI plugin must work with **all these CUDA versions** without recompilation. Because the
runtime signature of `cudaGetDriverEntryPoint` differs between CUDA 11.3–11.8 (3 args) and
CUDA 12 (4 args), the plugin keeps **three** runtime entry-point function pointers and tries
them in order.

**VALUE — the three entry-point ABIs the plugin must straddle:**
- `cudaGetDriverEntryPointByVersion(symbol, funcPtr, cudaVersion, flags, driverStatus)` — CUDA 13+
- `cudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus)` — CUDA 12 (4-arg)
- `cudaGetDriverEntryPoint(symbol, funcPtr, flags)` — CUDA 11.3–11.8 (3-arg, **no** `driverStatus`)

`driverStatus` is declared as `void*` (not `enum cudaDriverEntryPointQueryResult*`) because
that enum only exists in CUDA 12+.

> Source: `src/nccl_ofi_cuda.cpp` — runtime entry-point function pointers
> (`pfn_cudaGetDriverEntryPointByVersion`, `pfn_cudaGetDriverEntryPoint`,
> `pfn_cudaGetDriverEntryPoint_v11030`). Use `codegraph explore nccl_ofi_cuda` for the current
> declarations.

### Driver-API function versions

The `DECLARE_CUDA_FUNCTION(fn, version)` macro embeds a **minimum driver-API version** per
symbol. The version numbers are the fact:

| Driver function | Min version | Note |
|-----------------|-------------|------|
| `cuDriverGetVersion` | 2020 | |
| `cuCtxGetDevice` | 2000 | |
| `cuCtxSetCurrent` / `cuCtxGetCurrent` | 4000 | bind calling thread to a context |
| `cuDeviceGetAttribute` | 2000 | |
| `cuMemAlloc` / `cuMemFree` | 3020 | |
| `cuPointerGetAttributes` | 7000 | |
| `cuThreadExchangeStreamCaptureMode` | 10010 | graph-capture safety |
| `cuFlushGPUDirectRDMAWrites` | 11030 | CUDA 11.3+ (`HAVE_CUDA_GDRFLUSH_SUPPORT`) |
| `cuMemGetHandleForAddressRange` | 11070 | CUDA 11.7+ (`HAVE_CUDA_DMABUF_SUPPORT`) |

> Source: `src/nccl_ofi_cuda.cpp` — `DECLARE_CUDA_FUNCTION` invocations. Use
> `codegraph explore DECLARE_CUDA_FUNCTION` for the current list.

### Initialization Flow

`nccl_net_ofi_gpu_init()` performs, in order:
1. `dlopen("libcudart.so", RTLD_NOW)` (when `ENABLE_CUDART_DYNAMIC`) — runtime-agnostic.
2. Load `cudaRuntimeGetVersion`, detect the CUDA version.
3. Select the entry-point resolver by version: **>= 13000** → `cudaGetDriverEntryPointByVersion`;
   otherwise → `cudaGetDriverEntryPoint`.
4. Resolve all driver-API functions via `RESOLVE_CUDA_FUNCTION`.
5. `cuDriverGetVersion` for logging.
6. Enable GPUDirect flush iff `HAVE_CUDA_GDRFLUSH_SUPPORT` && GDR-support attribute &&
   `ofi_nccl_cuda_flush_enable()`.

> Source: `src/nccl_ofi_cuda.cpp` — `nccl_net_ofi_gpu_init()`. Use
> `codegraph explore nccl_net_ofi_gpu_init` for the current body.

### Function Resolution Idiom

**IDIOM — resolve one driver symbol across the three entry-point ABIs, in preference order:**

```cpp
// 1. CUDA 13+ versioned entry point (preferred)
// 2. CUDA 12 legacy entry point (4-arg, with driverStatus)
// 3. CUDA 11.3-11.8 entry point (3-arg, no driverStatus)  <- restores 11.3-11.8 (commit aab6529)
if (pfn_cudaGetDriverEntryPointByVersion)      try_it(#fn, &pfn_##fn, version, cudaEnableDefault, NULL);
else if (pfn_cudaGetDriverEntryPoint)          try_it(#fn, &pfn_##fn,          cudaEnableDefault, NULL);
else if (pfn_cudaGetDriverEntryPoint_v11030)   try_it(#fn, &pfn_##fn,          cudaEnableDefault);
// if none resolved: NCCL_OFI_WARN(...) and return -ENOTSUP;
```

The third branch is what restores CUDA 11.3–11.8 support (commit `aab6529`): those runtimes
export `cudaGetDriverEntryPoint` with a 3-argument signature (no `driverStatus`), which is a
different ABI from the CUDA 12 4-argument form and would fault if called with the wrong
prototype.

> Source: `src/nccl_ofi_cuda.cpp` — `RESOLVE_CUDA_FUNCTION`, `DECLARE_CUDA_FUNCTION`,
> `LOAD_CUDA_RUNTIME_SYM`. Use `codegraph explore RESOLVE_CUDA_FUNCTION` for the current macros.

**Key Benefits**:
1. **No recompilation needed** for new CUDA versions
2. **Graceful degradation** if functions unavailable
3. **Version numbers** embedded in function names (e.g., `cuMemAlloc_v2_3020`)

## CUDA Driver API vs Runtime API

### Why Use Driver API?

The plugin moved from the CUDA **runtime** API (`cuda*`) to the **driver** API (`cu*`)
wherever possible. The reason is **version stability**: the driver API has a guaranteed-stable
ABI, whereas runtime-API signatures can change between versions.

| Aspect | Runtime API (`cuda*`) | Driver API (`cu*`) |
|--------|----------------------|-------------------|
| **Stability** | Changes between versions | **Version-stable** |
| **ABI** | May break | Guaranteed stable |
| **Precision** | High-level, abstracted | Low-level control |
| **Performance** | Slightly slower | Fastest possible |
| **Example** | `cudaMalloc()` | `cuMemAlloc()` |

### API Mapping (runtime → driver)

**IDIOM — the runtime→driver call substitutions the plugin uses:**

```text
cudaMalloc(&ptr, size)                 → cuMemAlloc(&d_ptr, size)
cudaFree(ptr)                          → cuMemFree(d_ptr)
cudaMemcpy(dst, src, size, kind)       → cuMemcpyHtoDAsync(dst, src, size, sideStream)
                                          then cuStreamSynchronize   (see graph-capture section)
cudaPointerGetAttributes(&attrs, ptr)  → cuPointerGetAttributes(num_attrs, attrs, data, ptr)
cudaGetDevice(&dev)                    → cuCtxGetDevice(&dev)
cudaDeviceFlushGPUDirectRDMAWrites(..) → cuFlushGPUDirectRDMAWrites(..)
```

> Source: `src/nccl_ofi_cuda.cpp` — `nccl_net_ofi_gpu_mem_alloc()`,
> `nccl_net_ofi_gpu_mem_free()`, `nccl_net_ofi_get_gpu_device_for_addr()`. Use
> `codegraph explore nccl_net_ofi_gpu_mem_alloc` for the current bodies.

## DMA-BUF Export (CUDA 11.7+)

**IDIOM — export a GPU range as a dmabuf FD, preferring the PCIe mapping and falling back:**

```cpp
// Try PCIe-specific mapping first (better perf), then retry with no flags.
cuMemGetHandleForAddressRange(fd, dptr, size,
    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
    CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE);   // 1st attempt
cuMemGetHandleForAddressRange(fd, dptr, size,
    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);        // fallback: no flags
// *offset is typically 0 for aligned addresses.  Guarded by HAVE_CUDA_DMABUF_SUPPORT.
```

Device-capability queries use `cuDeviceGetAttribute` with:
- `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` (attribute id **142**) → dmabuf support.
- `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED` → GDR support.

> Source: `src/nccl_ofi_cuda.cpp` — `nccl_net_ofi_gpu_get_dma_buf_fd()`,
> `nccl_net_ofi_gpu_have_dma_buf_attr()`, `nccl_net_ofi_gpu_have_gdr_support_attr()`. Use
> `codegraph explore nccl_net_ofi_gpu_get_dma_buf_fd` for the current bodies.

## GPUDirect RDMA and Flush

### Why Flush is Needed

**Problem**: On some NVIDIA GPU architectures, RDMA writes to GPU memory may be buffered in GPU L2 cache. **CPU reads** of RDMA-written data may see stale values until cache is flushed.

**Solution**: `cuFlushGPUDirectRDMAWrites()` forces cache coherency.

### When to Flush

```
Timeline without flush:
1. NIC writes data to GPU memory via RDMA  → Sits in L2 cache
2. CPU reads from GPU memory               → Reads stale data! ❌

Timeline with flush:
1. NIC writes data to GPU memory via RDMA  → Sits in L2 cache
2. cuFlushGPUDirectRDMAWrites()            → Flushes L2 to memory
3. CPU reads from GPU memory               → Reads fresh data! ✓
```

The flush call targets `CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX` with scope
`CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER`, and is compiled only under
`HAVE_CUDA_GDRFLUSH_SUPPORT` (CUDA 11.3+).

> Source: `src/nccl_ofi_cuda.cpp` — `nccl_net_ofi_gpu_flush_gpudirect_rdma_writes()`. Use
> `codegraph explore nccl_net_ofi_gpu_flush_gpudirect_rdma_writes` for the current body.

### Configuration

```bash
# Enable CUDA flush (default: disabled)
export OFI_NCCL_CUDA_FLUSH_ENABLE=1

# Check if flush is being used
export NCCL_DEBUG=WARN
# Look for: "CUDA flush enabled"
```

**When Enabled**:
- **CUDA 11.3+** required
- GPU must report `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED=1`
- Environment variable `OFI_NCCL_CUDA_FLUSH_ENABLE=1`

**Performance Impact**: 1-5 μs per flush (small overhead for correctness)

## GDRCopy Integration

### What is GDRCopy?

**GDRCopy** is a low-latency GPU memory copy library that maps GPU memory into CPU address space for fast CPU↔GPU transfers **without going through CUDA runtime**.

**Use Case**: Small message sends where CPU needs to read GPU memory.

**Performance** (estimated):
- Traditional `cudaMemcpy()`: ~10-50 μs
- **GDRCopy**: ~1-5 μs (approximately 5-10x faster)

### Architecture

```
Traditional Path (slow):
CPU → cudaMemcpy() → CUDA Runtime → GPU Driver → PCIe → GPU Memory
Total: 10-50 μs

GDRCopy Path (fast):
CPU → memcpy(gdr_mapped_ptr) → Direct PCIe BAR access → GPU Memory
Total: 1-5 μs
```

### GDRCopy version / API facts (VALUE)

- Loaded dynamically via `dlopen("libgdrapi.so", RTLD_NOW | RTLD_LOCAL)` — optional dependency.
- **GDRCopy 2.5+** exposes `gdr_pin_buffer_v2` (guarded by `HAVE_DECL_GDR_PIN_BUFFER_V2`); the
  detection is `major > 2 || (major == 2 && minor >= 5)`.
- `GDR_PIN_FLAG_FORCE_PCIE` forces PCIe BAR mapping. On CPU-GPU coherent interconnect (e.g.
  Grace Hopper / C2C) the plugin tries `FORCE_PCIE` first, then falls back to flags `0`.

### Registration FORMULA (page rounding)

GDRCopy registration rounds the region to **GPU page** boundaries before pinning:

```text
regbgn     = ROUND_DOWN(device_ptr, GPU_PAGE_SIZE)
regend     = device_ptr + size
gdr_reglen = ROUND_UP(regend - regbgn, GPU_PAGE_SIZE)
mapped_ptr = gdr_mapped_ptr + (device_ptr - regbgn)   // user-visible pointer after gdr_map()
```

**IDIOM — pin with v2 forced-PCIe, fall back to default, else v1:**

```cpp
if (gdr_pin_buffer_v2) {
    if (gdr_pin_buffer_v2(gdr, regbgn, reglen, GDR_PIN_FLAG_FORCE_PCIE, &mh) != 0)
        gdr_pin_buffer_v2(gdr, regbgn, reglen, /*flags=*/0, &mh);   // C2C fallback
} else {
    gdr_pin_buffer(gdr, regbgn, reglen, 0, 0, &mh);                 // v1
}
```

> Source: `src/nccl_ofi_gdrcopy.cpp` — `nccl_ofi_gdrcopy_ctx` (ctor, `register_region`,
> `copy_to_device`, `copy_from_device`). Use `codegraph explore nccl_ofi_gdrcopy_ctx` for the
> current bodies.

## CUDA Graph Capture Interaction (gotcha)

**Problem**: `cuMemAlloc`/`cuMemFree` and a synchronous `cuMemcpy` are **not legal inside an
active CUDA graph stream capture**. If the plugin performs a GPU memory op (e.g. allocating or
freeing a bounce/scratch buffer, or copying a signal) while the application's thread has a
stream capture in progress, the operation fails and the capture is invalidated. This was a
real, observed bug — fixed in commit `9013fa8` (*"rdma: fix GPU memory ops failing during CUDA
graph capture"*).

**Fix / IDIOM**: the plugin's GPU memory helpers **temporarily disable stream capture** on the
calling thread around the operation using `cuThreadExchangeStreamCaptureMode`, and restore the
previous mode afterward. Copies additionally run on a **dedicated non-blocking side stream** so
they never join the captured default stream:

```cpp
// Swap the calling thread into RELAXED capture mode so cuMemAlloc/cuMemFree/copy are allowed
// even if the app has a capture in progress; restore the prior mode after.
CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED;
pfn_cuThreadExchangeStreamCaptureMode(&mode);   // disable/relax capture
// ... cuMemAlloc / cuMemFree,  OR:
//   cuStreamCreate(&s, CU_STREAM_NON_BLOCKING);   // dedicated side stream
//   cuMemcpyHtoDAsync(dst, src, size, s); cuStreamSynchronize(s); cuStreamDestroy(s);
pfn_cuThreadExchangeStreamCaptureMode(&mode);   // restore prior mode
```

**Takeaway for callers**: memory registration / device copies through the plugin are safe to
call from a thread that is capturing a CUDA graph; the plugin isolates them from the capture.

> Source: `src/nccl_ofi_cuda.cpp` — `nccl_net_ofi_gpu_mem_alloc()`,
> `nccl_net_ofi_gpu_mem_free()`, `nccl_net_ofi_gpu_mem_copy_host_to_device()`. Use
> `codegraph explore nccl_net_ofi_gpu_mem_copy_host_to_device` for the current bodies.

## `nccl_ofi_device_copy.h`: the device-copy abstraction

`include/nccl_ofi_device_copy.h` defines an abstract `nccl_ofi_device_copy` interface that
decouples the rest of the plugin from *how* host↔device copies are performed. GDRCopy is one
concrete implementation (`nccl_ofi_gdrcopy_ctx`). The interface is:

- `register_region(device_ptr, size, out_handle)` — register a device region, returns an opaque `RegHandle`.
- `copy_to_device(host_ptr, handle, offset, size)` — host → device.
- `copy_from_device(handle, offset, host_ptr, size)` — device → host.
- `forced_pcie_copy()` — true if the implementation forces the PCIe path (relevant on C2C
  systems such as Grace Hopper, where the GDRCopy v2 `GDR_PIN_FLAG_FORCE_PCIE` path is used).
- `deregister_region(handle)`.

The GIN path requires an implementation whose `forced_pcie_copy()` is true (GDRCopy 2.5+),
which is why the GIN proxy signal path insists on GDRCopy 2.5+ (see
[gin-alltoall-libfabric-trace.md](gin-alltoall-libfabric-trace.md)). In **GDAKI** mode the
GDRCopy registration is **skipped entirely** (commit `5b163ed`) because the device signals the
target through EFA hardware counters instead of a CPU-mapped copy.

> Source: `include/nccl_ofi_device_copy.h` — `nccl_ofi_device_copy`. Use
> `codegraph explore nccl_ofi_device_copy` for the current interface.

## New CUDA helpers: bind a thread's GPU, inspect allocations

Commit `6014cd9` added helpers (all via the **CUDA driver API**) to bind a thread's
GPU/context and inspect allocations:

- `nccl_net_ofi_gpu_set_current_context(CUcontext)` / `nccl_net_ofi_gpu_get_current_context(CUcontext*)`
  — thin wrappers over `cuCtxSetCurrent` / `cuCtxGetCurrent`; they manage which context the
  calling thread is bound to and take **no ownership** of it. These matter for worker threads
  (e.g. the per-process gdrcopy worker) that must run on the right GPU context.
- Allocation-inspection helpers built on `cuPointerGetAttributes` /
  `cuMemGetAllocationGranularity` / `cuMemGetAllocationPropertiesFromHandle` /
  `cuMemRetainAllocationHandle` to determine device ordinal, memory type, and allocation
  properties for a pointer.

> Source: `src/nccl_ofi_cuda.cpp` — `nccl_net_ofi_gpu_set_current_context`,
> `nccl_net_ofi_gpu_get_current_context`. Use `codegraph explore nccl_net_ofi_gpu_set_current_context`.

## EFA-GDA (GDAKI) CUDA dependency and build requirements

The GDAKI (kernel-initiated) GIN device path depends on the **vendored EFA-GDA CUDA sources**
at `3rd-party/efa-gda` (the *efa-dp-direct* project, pinned at `81c4dc7`). These sources
provide the GPU-side EFA datapath: posting work requests from GPU kernels and GPU-side
completion polling. They were vendored and wired into the build across commits `27920ab`
(vendor efa-dp-direct at `3rd-party/efa-gda`), `4735ddd`, `1383e93`, `38febcc`, `2fee64f`,
`e8f6174`, `0c7eb41`.

**Build requirement (hard):** `configure.ac` **fails configuration** if the vendored CUDA
sources are missing — it checks for `3rd-party/efa-gda/CUDA/README.md` and `AC_MSG_ERROR`s
out if absent (the vendored tree is **not** a Git submodule). This is a REMOVED-if-missing
build gate, not derivable from any indexed symbol:

```m4
AC_MSG_CHECKING([for required efa-dp-direct CUDA sources])
AS_IF([test -f "$srcdir/3rd-party/efa-gda/CUDA/README.md"],
      [AC_MSG_RESULT([yes])],
      [AC_MSG_RESULT([no])
       AC_MSG_ERROR([Missing required file:
  3rd-party/efa-gda/CUDA/README.md
3rd-party/efa-gda is vendored directly in this repository and is not a Git
submodule. ...])])
```

Notes:
- `3rd-party/efa-gda` is **vendored directly**, not a Git submodule — a missing file means the
  tree is incomplete; re-checkout or build from a release tarball.
- `nvcc` is optional and only used to build the GDAKI GPU functional test.
- GDAKI is compiled in as `HAVE_GDAKI`; at runtime it also requires libfabric ≥ 2.5, DMA-BUF
  viable, and an EFA provider (see `nccl_ofi_gin_gdaki_capable()`).

### EFA-GDA glue moved to the CUDA driver API

The EFA-GDA glue was migrated from **CUDA runtime API** calls to **CUDA driver API** calls
(commit `62b63cc`, *"efa-gda: Translate cuda runtime function calls to cuda driver function
calls"*), and the GIN gdrcopy context is now bound via the **CUDA driver API** (commit
`6eb9a96`, *"cuda: Bind the GIN gdrcopy context via the CUDA driver API"*). This is consistent
with the rest of the plugin's preference for the version-stable driver API. Related dynamic
`cudart` / `CUDA_RUNTIME_LIB` handling changes: commits `323668e`, `e31de68`, `ee25228`,
`fa8d688`, `2cc20bb`.

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Dynamic CUDA library load | 5-20 ms | One-time at init |
| Function resolution | 10-50 μs | One-time per function |
| `cuMemAlloc()` | ~500 μs | Similar to `cudaMalloc()` |
| `cuMemGetHandleForAddressRange()` | 10-50 μs | DMA-BUF export |
| `cuFlushGPUDirectRDMAWrites()` | 1-5 μs | Cache flush |
| **GDRCopy `gdr_pin_buffer()`** | 100-300 μs | One-time registration |
| **GDRCopy `gdr_copy_to_mapping()`** | **~1-5 μs** | Approximately 5-10x faster than cudaMemcpy |
| Traditional `cudaMemcpy()` | 10-50 μs | Slow |

## Configuration and Tuning

### Check CUDA Support

```bash
# Check CUDA version
nvidia-smi

# Check driver version
cat /proc/driver/nvidia/version

# Check CUDA runtime
ls -la /usr/local/cuda/lib64/libcudart.so*

# Query GPU capabilities
nvidia-smi -q | grep -E "CUDA|Compute|Memory"
```

### Build Configuration

```bash
# Build OFI plugin with CUDA support
cd aws-ofi-nccl
./autogen.sh
./configure --with-cuda=/usr/local/cuda \
            --with-gdrcopy=/usr/local/gdrcopy \
            --enable-platform-aws

# Check configuration
grep HAVE_CUDA config.h
# #define HAVE_CUDA 1
# #define HAVE_CUDA_DMABUF_SUPPORT 1
# #define HAVE_CUDA_GDRFLUSH_SUPPORT 1

grep HAVE_GDRCOPY config.h
# #define HAVE_GDRCOPY 1
```

### Runtime Configuration

```bash
# Enable CUDA flush
export OFI_NCCL_CUDA_FLUSH_ENABLE=1

# Enable GDRCopy (if available)
# Automatically detected if libgdrapi.so found

# Debug CUDA initialization
export NCCL_DEBUG=INFO
# Look for:
# "Using CUDA driver version 12010 with runtime 12010"
# "CUDA flush enabled"
# "gdrcopy: Initializing"
```

### GDRCopy Installation

```bash
# Install GDRCopy
git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
make
sudo make install

# Load kernel module
sudo modprobe gdrdrv

# Verify
ls -la /dev/gdrdrv
cat /proc/modules | grep gdrdrv
```

## Debugging

### CUDA Function Resolution Failures

```bash
export NCCL_DEBUG=WARN

# Look for errors like:
# "Failed to resolve CUDA function cuMemGetHandleForAddressRange"
# → CUDA version too old (need 11.7+ for dmabuf)

# "Failed to find CUDA Runtime library"
# → libcudart.so not in LD_LIBRARY_PATH
```

### DMA-BUF Export Failures

```bash
# Check GPU supports dmabuf
python3 << 'EOF'
import ctypes
cuda = ctypes.CDLL("libcuda.so")
device = ctypes.c_int()
cuda.cuInit(0)
cuda.cuDeviceGet(ctypes.byref(device), 0)

attr = ctypes.c_int()
CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 142
cuda.cuDeviceGetAttribute(ctypes.byref(attr),
                         CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
                         device)
print(f"DMA-BUF supported: {attr.value}")
EOF
```

### GDRCopy Issues

```bash
# Check GDRCopy loaded
lsmod | grep gdrdrv

# Check device
ls -la /dev/gdrdrv

# Test GDRCopy
cd gdrcopy/tests
./copybw

# Expected output:
# BW(host->gpu): ~10-15 GB/s
# BW(gpu->host): ~12-18 GB/s
```

## Summary

| Aspect | Details |
|--------|---------|
| **Version handling** | Dynamic loading, CUDA 11-13+ support |
| **API choice** | Driver API (`cu*`) for stability |
| **DMA-BUF** | CUDA 11.7+ via `cuMemGetHandleForAddressRange()` |
| **GPUDirect flush** | CUDA 11.3+ via `cuFlushGPUDirectRDMAWrites()` |
| **GDRCopy** | Optional, approximately 5-10x faster than `cudaMemcpy()`; skipped in GDAKI mode |
| **Entry points** | CUDA 13: versioned, CUDA 12: legacy, 11.3-11.8: 3-arg |
| **Performance** | Function resolution: ~10-50 μs, GDRCopy copy: ~1-5 μs |

**Key Takeaways**:
1. **Dynamic loading** enables cross-version compatibility without recompilation
2. **Driver API** preferred over Runtime API for ABI stability
3. **GPUDirect flush** needed on some architectures for correctness (~1-5 μs overhead)
4. **GDRCopy** optional but highly recommended for small message performance; skipped in GDAKI
5. **DMA-BUF** modern approach (CUDA 11.7+), cleaner than legacy GDR
6. **Version detection** at runtime determines available features
7. **MR caching still critical** - registration expensive (100-500 μs)

**Related Documentation**:
- [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md) - DMA-BUF framework (used by CUDA)
- [accelerator-memory.md](accelerator-memory.md) - AMD ROCm and AWS Neuron (non-NVIDIA paths)
- [mr-cache-implementation.md](mr-cache-implementation.md) - MR cache (critical for CUDA)
- [kernel-efa-driver.md](kernel-efa-driver.md) - Kernel dmabuf import

---

## Code References

Bodies live in source; query the code graph rather than trusting a pasted copy.

**aws-ofi-nccl (CUDA integration)** — `src/nccl_ofi_cuda.cpp`, `src/nccl_ofi_gdrcopy.cpp`,
`include/nccl_ofi_device_copy.h`:
- `nccl_net_ofi_gpu_init()` — init/version detection.
- `RESOLVE_CUDA_FUNCTION`, `DECLARE_CUDA_FUNCTION`, `LOAD_CUDA_RUNTIME_SYM` — resolution macros.
- `nccl_net_ofi_gpu_mem_alloc/free/copy_host_to_device` — graph-capture-safe memory ops.
- `nccl_net_ofi_gpu_get_dma_buf_fd`, `nccl_net_ofi_gpu_have_dma_buf_attr`,
  `nccl_net_ofi_gpu_have_gdr_support_attr` — dmabuf export / capability queries.
- `nccl_net_ofi_gpu_flush_gpudirect_rdma_writes` — GDR flush.
- `nccl_ofi_gdrcopy_ctx` — GDRCopy implementation of `nccl_ofi_device_copy`.

**CUDA Driver API (external — NVIDIA)**: `cuDriverGetVersion`, `cuCtxGetDevice`,
`cuCtxSetCurrent`, `cuCtxGetCurrent`, `cuDeviceGetAttribute`, `cuMemAlloc`, `cuMemFree`,
`cuPointerGetAttributes`, `cuThreadExchangeStreamCaptureMode`, `cuFlushGPUDirectRDMAWrites`,
`cuMemGetHandleForAddressRange`.

**CUDA Runtime API (external — NVIDIA)**: `cudaRuntimeGetVersion`, `cudaGetDriverEntryPoint`,
`cudaGetDriverEntryPointByVersion`.

**GDRCopy API (external — NVIDIA)**: `gdr_open`, `gdr_pin_buffer`, `gdr_pin_buffer_v2`,
`gdr_map`, `gdr_copy_to_mapping`, `gdr_copy_from_mapping`.

All aws-ofi-nccl references use branch-form links into [aws/aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl/blob/master/) (`master`). Content in this document was verified against aws-ofi-nccl master `d840aa1` (tag v1.21.1). NVIDIA driver / open-gpu-kernel-modules details reflect release **610.57.04**.
