# NVIDIA CUDA Memory Registration

## Overview

**NVIDIA CUDA** is the dominant GPU computing platform. CUDA memory registration for RDMA involves complex version handling, GPUDirect RDMA support, and optional GDRCopy for fast CPU-GPU transfers. This document covers CUDA-specific aspects beyond the general DMA-BUF framework.

**Target Hardware**:
- NVIDIA GPUs (Tesla, A100, H100, etc.)
- GPUDirect RDMA capable devices
- PCIe Gen3+ recommended

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
them in order (see below).

### Solution: Dynamic Function Resolution

**Function Pointers** ([nccl_ofi_cuda.cpp, ~lines 20–35](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_cuda.cpp)):

```cpp
// CUDA Runtime function pointers - only for functions without driver equivalents.
// driverStatus is declared as void* rather than
// enum cudaDriverEntryPointQueryResult* because that enum only exists in CUDA 12+.
static cudaError_t (*pfn_cudaGetDriverEntryPointByVersion)(const char *symbol, void **funcPtr,
        unsigned int cudaVersion, unsigned long long flags, void *driverStatus) = NULL;  // CUDA 13+
static cudaError_t (*pfn_cudaGetDriverEntryPoint)(const char *symbol, void **funcPtr,
        unsigned long long flags, void *driverStatus) = NULL;                            // CUDA 12
static cudaError_t (*pfn_cudaGetDriverEntryPoint_v11030)(const char *symbol, void **funcPtr,
        unsigned long long flags) = NULL;   // CUDA 11.3 - 11.8 (3-argument variant, no driverStatus)
```

**`DECLARE_CUDA_FUNCTION` macro** ([nccl_ofi_cuda.cpp:40](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_cuda.cpp)):

```cpp
// Driver API function pointers (version-stable!)
DECLARE_CUDA_FUNCTION(cuDriverGetVersion, 2020);
DECLARE_CUDA_FUNCTION(cuCtxGetDevice, 2000);
DECLARE_CUDA_FUNCTION(cuCtxSetCurrent, 4000);   // bind calling thread to a context
DECLARE_CUDA_FUNCTION(cuCtxGetCurrent, 4000);
DECLARE_CUDA_FUNCTION(cuDeviceGetAttribute, 2000);
DECLARE_CUDA_FUNCTION(cuMemAlloc, 3020);
DECLARE_CUDA_FUNCTION(cuMemFree, 3020);
DECLARE_CUDA_FUNCTION(cuPointerGetAttributes, 7000);
// CUDA graph-capture safety: temporarily disable capture around alloc/free/copy
DECLARE_CUDA_FUNCTION(cuThreadExchangeStreamCaptureMode, 10010);
#if HAVE_CUDA_GDRFLUSH_SUPPORT
DECLARE_CUDA_FUNCTION(cuFlushGPUDirectRDMAWrites, 11030);  // CUDA 11.3+
#endif
#if HAVE_CUDA_DMABUF_SUPPORT
DECLARE_CUDA_FUNCTION(cuMemGetHandleForAddressRange, 11070);  // CUDA 11.7+
#endif
```

### Initialization Flow

**`nccl_net_ofi_gpu_init()` function** ([nccl_ofi_cuda.cpp:89-143](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_cuda.cpp)):

```cpp
int nccl_net_ofi_gpu_init(void)
{
    int driverVersion = -1;
    int runtimeVersion = -1;

#if ENABLE_CUDART_DYNAMIC
    // 1. Dynamically load libcudart.so (runtime agnostic!)
    cudaruntime_lib = dlopen("libcudart.so", RTLD_NOW);
    if (!cudaruntime_lib) {
        NCCL_OFI_WARN("Failed to find CUDA Runtime library");
        return -ENOTSUP;
    }

    // 2. Load cudaRuntimeGetVersion to detect CUDA version
    LOAD_CUDA_RUNTIME_SYM(cudaruntime_lib, cudaRuntimeGetVersion);

    pfn_cudaRuntimeGetVersion(&runtimeVersion);

    // 3. Load appropriate entry point function based on version
    if (runtimeVersion >= 13000) {
        // CUDA 13+: Use versioned entry point
        LOAD_CUDA_RUNTIME_SYM(cudaruntime_lib, cudaGetDriverEntryPointByVersion);
    } else {
        // CUDA 12: Use legacy entry point
        LOAD_CUDA_RUNTIME_SYM(cudaruntime_lib, cudaGetDriverEntryPoint);
    }
#else
    // Static linking - use compile-time functions
    pfn_cudaRuntimeGetVersion = cudaRuntimeGetVersion;
    pfn_cudaRuntimeGetVersion(&runtimeVersion);

#if CUDART_VERSION >= 13000
    pfn_cudaGetDriverEntryPointByVersion = cudaGetDriverEntryPointByVersion;
#else
    pfn_cudaGetDriverEntryPoint = cudaGetDriverEntryPoint;
#endif
#endif

    // 4. Resolve all driver API functions
    RESOLVE_CUDA_FUNCTION(cuDriverGetVersion, 2020);
    RESOLVE_CUDA_FUNCTION(cuMemAlloc, 3020);
    RESOLVE_CUDA_FUNCTION(cuMemFree, 3020);
    // ... etc

    // 5. Query driver version
    pfn_cuDriverGetVersion(&driverVersion);

    NCCL_OFI_INFO("Using CUDA driver version %d with runtime %d",
                  driverVersion, runtimeVersion);

    // 6. Enable GPUDirect flush if supported
    if (HAVE_CUDA_GDRFLUSH_SUPPORT &&
        nccl_net_ofi_gpu_have_gdr_support_attr() &&
        ofi_nccl_cuda_flush_enable()) {
        cuda_flush = true;
    }

    return 0;
}
```

### Function Resolution Macro

**`RESOLVE_CUDA_FUNCTION` macro** ([nccl_ofi_cuda.cpp, ~lines 55–75](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_cuda.cpp)):

```cpp
#define RESOLVE_CUDA_FUNCTION(function, version) do {                          \
    cudaError_t err = cudaErrorUnknown;                                        \
    bool resolved = false;                                                     \
    /* 1. CUDA 13+ versioned entry point (preferred) */                        \
    if (pfn_cudaGetDriverEntryPointByVersion != NULL) {                        \
        err = pfn_cudaGetDriverEntryPointByVersion(#function,                  \
                   (void **)&pfn_##function, version, cudaEnableDefault, NULL);\
        if (err == cudaSuccess && pfn_##function != NULL) resolved = true;     \
    }                                                                          \
    /* 2. CUDA 12 legacy entry point (4-arg, with driverStatus) */            \
    if (!resolved && pfn_cudaGetDriverEntryPoint != NULL) {                    \
        err = pfn_cudaGetDriverEntryPoint(#function,                           \
                   (void **)&pfn_##function, cudaEnableDefault, NULL);         \
        if (err == cudaSuccess && pfn_##function != NULL) resolved = true;     \
    }                                                                          \
    /* 3. CUDA 11.3-11.8 entry point (3-arg, no driverStatus) */              \
    if (!resolved && pfn_cudaGetDriverEntryPoint_v11030 != NULL) {             \
        err = pfn_cudaGetDriverEntryPoint_v11030(#function,                    \
                   (void **)&pfn_##function, cudaEnableDefault);               \
        if (err == cudaSuccess && pfn_##function != NULL) resolved = true;     \
    }                                                                          \
    if (!resolved) {                                                           \
        NCCL_OFI_WARN("Failed to resolve CUDA function %s", #function);        \
        return -ENOTSUP;                                                       \
    }                                                                          \
} while (0);
```

The third branch is what restores CUDA 11.3–11.8 support (commit `aab6529`): those runtimes
export `cudaGetDriverEntryPoint` with a 3-argument signature (no `driverStatus`), which is a
different ABI from the CUDA 12 4-argument form and would otherwise fault if called with the
wrong prototype.

**Key Benefits**:
1. **No recompilation needed** for new CUDA versions
2. **Graceful degradation** if functions unavailable
3. **Version numbers** embedded in function names (e.g., `cuMemAlloc_v2_3020`)

## CUDA Driver API vs Runtime API

### Why Use Driver API?

| Aspect | Runtime API (`cuda*`) | Driver API (`cu*`) |
|--------|----------------------|-------------------|
| **Stability** | Changes between versions | **Version-stable** |
| **ABI** | May break | Guaranteed stable |
| **Precision** | High-level, abstracted | Low-level control |
| **Performance** | Slightly slower | Fastest possible |
| **Example** | `cudaMalloc()` | `cuMemAlloc()` |

**OFI Plugin Choice**: **Driver API** wherever possible for maximum compatibility.

### API Mapping

```cpp
// Runtime API → Driver API mapping

// Memory allocation
cudaMalloc(&ptr, size)  →  cuMemAlloc(&d_ptr, size)

// Memory free
cudaFree(ptr)  →  cuMemFree(d_ptr)

// Memory copy
cudaMemcpy(dst, src, size, kind)  →  cuMemcpyHtoDAsync(dst, src, size, sideStream)
                                     (on a dedicated non-blocking side stream, then
                                      cuStreamSynchronize; see CUDA graph-capture section)

// Pointer attributes
cudaPointerGetAttributes(&attrs, ptr)  →  cuPointerGetAttributes(num_attrs, attrs, data, ptr)

// Device query
cudaGetDevice(&dev)  →  cuCtxGetDevice(&dev)

// GPUDirect flush
cudaDeviceFlushGPUDirectRDMAWrites(...)  →  cuFlushGPUDirectRDMAWrites(...)
```

### Implementation Example

```cpp
// From aws-ofi-nccl/src/nccl_ofi_cuda.cpp

int nccl_net_ofi_gpu_mem_alloc(void **ptr, size_t size)
{
    CUdeviceptr d_ptr;
    // Use driver API (cuMemAlloc, not cudaMalloc)
    CUresult ret = pfn_cuMemAlloc(&d_ptr, size);
    if (ret != CUDA_SUCCESS) {
        return -EINVAL;
    }
    *ptr = (void *)d_ptr;
    return 0;
}

int nccl_net_ofi_gpu_mem_free(void *ptr)
{
    // Use driver API (cuMemFree, not cudaFree)
    CUresult ret = pfn_cuMemFree((CUdeviceptr)ptr);
    return (ret == CUDA_SUCCESS) ? 0 : -EINVAL;
}

int nccl_net_ofi_get_gpu_device_for_addr(void *data, int *dev_id)
{
    // Use driver API cuPointerGetAttributes
    CUpointer_attribute attrs[2] = {
        CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE
    };
    void *attr_data[2];

    CUresult ret = pfn_cuPointerGetAttributes(2, attrs, attr_data, (CUdeviceptr)data);
    if (ret != CUDA_SUCCESS) {
        return -ENOTSUP;
    }

    *dev_id = *(int *)attr_data[0];
    return 0;
}
```

## DMA-BUF Export (CUDA 11.7+)

### Export GPU Memory as DMA-BUF

```cpp
// From aws-ofi-nccl/src/nccl_ofi_cuda.cpp

int nccl_net_ofi_gpu_get_dma_buf_fd(void *aligned_ptr, size_t aligned_size,
                                    int *fd, size_t *offset)
{
#if HAVE_CUDA_DMABUF_SUPPORT
    CUresult ret;

    // Try PCIe-specific mapping first (better performance)
    ret = pfn_cuMemGetHandleForAddressRange(
        fd,                                        // Output: dmabuf FD
        (CUdeviceptr)aligned_ptr,                 // GPU memory address
        aligned_size,                              // Size
        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,      // Type: dmabuf
        CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE); // PCIe mapping

    if (ret != CUDA_SUCCESS) {
        // Fallback: Try without PCIe flag (works on all systems)
        ret = pfn_cuMemGetHandleForAddressRange(
            fd,
            (CUdeviceptr)aligned_ptr,
            aligned_size,
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
            0);  // No flags
    }

    if (ret != CUDA_SUCCESS) {
        NCCL_OFI_WARN("cuMemGetHandleForAddressRange failed: %d", ret);
        return -EINVAL;
    }

    // CUDA returns offset within dmabuf
    *offset = 0;  // Typically 0 for aligned addresses

    return 0;
#else
    return -ENOTSUP;  // CUDA version < 11.7
#endif
}
```

### Device Capability Query

```cpp
bool nccl_net_ofi_gpu_have_dma_buf_attr(void)
{
    CUdevice device;
    int supports_dmabuf = 0;

    // Get current device
    if (pfn_cuCtxGetDevice(&device) != CUDA_SUCCESS) {
        return false;
    }

    // Query CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED
    CUresult ret = pfn_cuDeviceGetAttribute(
        &supports_dmabuf,
        CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
        device);

    return (ret == CUDA_SUCCESS && supports_dmabuf == 1);
}

bool nccl_net_ofi_gpu_have_gdr_support_attr(void)
{
    CUdevice device;
    int supports_gdr = 0;

    if (pfn_cuCtxGetDevice(&device) != CUDA_SUCCESS) {
        return false;
    }

    // Query CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED
    CUresult ret = pfn_cuDeviceGetAttribute(
        &supports_gdr,
        CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED,
        device);

    return (ret == CUDA_SUCCESS && supports_gdr == 1);
}
```

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

### Implementation

```cpp
// From aws-ofi-nccl/src/nccl_ofi_cuda.cpp

int nccl_net_ofi_gpu_flush_gpudirect_rdma_writes(void)
{
#if HAVE_CUDA_GDRFLUSH_SUPPORT
    CUresult ret;

    if (pfn_cuFlushGPUDirectRDMAWrites == NULL) {
        return -EPERM;  // Function not available
    }

    // Flush all RDMA writes for current context
    ret = pfn_cuFlushGPUDirectRDMAWrites(
        CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,  // Target: current GPU
        CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);           // Scope: to memory owner

    return (ret == CUDA_SUCCESS) ? 0 : -EPERM;
#else
    return -EPERM;  // CUDA version < 11.3
#endif
}
```

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

### GDRCopy Initialization

```cpp
// From aws-ofi-nccl/src/nccl_ofi_gdrcopy.cpp

class nccl_ofi_gdrcopy_ctx : public nccl_ofi_device_copy
{
public:
    nccl_ofi_gdrcopy_ctx()
    {
        pimpl = new impl();

        // 1. Dynamically load libgdrapi.so (optional dependency)
        pimpl->lib = dlopen("libgdrapi.so", RTLD_NOW | RTLD_LOCAL);
        if (pimpl->lib == nullptr) {
            throw std::runtime_error("Could not load libgdrapi.so");
        }

        // 2. Resolve GDRCopy functions
        pimpl->gdr_open_fn = dlsym(pimpl->lib, "gdr_open");
        pimpl->gdr_pin_buffer_fn = dlsym(pimpl->lib, "gdr_pin_buffer");
        pimpl->gdr_map_fn = dlsym(pimpl->lib, "gdr_map");
        pimpl->gdr_copy_to_mapping_fn = dlsym(pimpl->lib, "gdr_copy_to_mapping");
        pimpl->gdr_copy_from_mapping_fn = dlsym(pimpl->lib, "gdr_copy_from_mapping");

#if HAVE_DECL_GDR_PIN_BUFFER_V2
        // GDRCopy 2.5+: Check for v2 API (force PCIe mode)
        uint32_t major, minor;
        if (get_version(&major, &minor) == 0 && (major > 2 || (major == 2 && minor >= 5))) {
            pimpl->gdr_pin_buffer_v2_fn = dlsym(pimpl->lib, "gdr_pin_buffer_v2");
        }
#endif

        // 3. Open GDRCopy handle
        pimpl->gdr = pimpl->gdr_open_fn();
        if (pimpl->gdr == nullptr) {
            throw std::runtime_error("Failed to open gdr handle");
        }
    }
};
```

### Register and Map GPU Memory

```cpp
int nccl_ofi_gdrcopy_ctx::register_region(void *device_ptr, size_t size,
                                          RegHandle* &out_handle)
{
    auto handle = new gdrcopy_RegHandle { };
    gdr_mh_t mh;

    // 1. Round to GPU page boundaries (64KB typically)
    uintptr_t data_uint = (uintptr_t)device_ptr;
    uintptr_t regbgn = ROUND_DOWN(data_uint, GPU_PAGE_SIZE);
    uintptr_t regend = data_uint + size;
    handle->gdr_reglen = ROUND_UP(regend - regbgn, GPU_PAGE_SIZE);

    // 2. Pin GPU memory (registers with kernel driver)
#if HAVE_DECL_GDR_PIN_BUFFER_V2
    if (pimpl->gdr_pin_buffer_v2_fn != nullptr) {
        // Try PCIe mode first (better performance on most systems)
        uint32_t flags = GDR_PIN_FLAG_FORCE_PCIE;
        ret = pimpl->gdr_pin_buffer_v2_fn(pimpl->gdr, regbgn,
                                         handle->gdr_reglen, flags, &mh);
        if (ret != 0) {
            // Fallback to default mode (works on C2C systems)
            flags = 0;
            ret = pimpl->gdr_pin_buffer_v2_fn(pimpl->gdr, regbgn,
                                             handle->gdr_reglen, flags, &mh);
        }
    } else {
        ret = pimpl->gdr_pin_buffer_fn(pimpl->gdr, regbgn,
                                      handle->gdr_reglen, 0, 0, &mh);
    }
#else
    ret = pimpl->gdr_pin_buffer_fn(pimpl->gdr, regbgn,
                                  handle->gdr_reglen, 0, 0, &mh);
#endif

    handle->pin_handle = mh.h;

    // 3. Map into CPU address space
    gdr_mh_t mh_map = {handle->pin_handle};
    ret = pimpl->gdr_map_fn(pimpl->gdr, mh_map, &handle->gdr_mapped_ptr,
                           handle->gdr_reglen);

    // 4. Calculate user-visible mapped pointer (may differ from gdr_mapped_ptr)
    handle->mapped_ptr = (char *)handle->gdr_mapped_ptr +
                        (data_uint - regbgn);

    out_handle = handle;
    return 0;
}
```

### Copy Operations

```cpp
// Copy from host to GPU
int nccl_ofi_gdrcopy_ctx::copy_to_device(const void *host_ptr,
                                         const RegHandle &handle,
                                         size_t offset, size_t size)
{
    auto gdr_handle = static_cast<const gdrcopy_RegHandle*>(&handle);
    void *dst = (char *)gdr_handle->mapped_ptr + offset;

    // Direct CPU memcpy to GPU memory!
    return pimpl->gdr_copy_to_mapping_fn(pimpl->gdr->gdr_mh,
                                        dst, host_ptr, size);
}

// Copy from GPU to host
int nccl_ofi_gdrcopy_ctx::copy_from_device(const RegHandle &handle,
                                           size_t offset, void *host_ptr,
                                           size_t size)
{
    auto gdr_handle = static_cast<const gdrcopy_RegHandle*>(&handle);
    void *src = (char *)gdr_handle->mapped_ptr + offset;

    // Direct CPU memcpy from GPU memory!
    return pimpl->gdr_copy_from_mapping_fn(pimpl->gdr->gdr_mh,
                                          host_ptr, src, size);
}
```

### GDRCopy v2 (2.5+): Forced PCIe Mode

**Problem**: On systems with CPU-GPU coherent interconnect (e.g., Grace Hopper), GPU memory may use different access path.

**Solution**: `GDR_PIN_FLAG_FORCE_PCIE` forces PCIe BAR mapping for consistent performance.

```cpp
// Try PCIe first
uint32_t flags = GDR_PIN_FLAG_FORCE_PCIE;
ret = gdr_pin_buffer_v2(gdr, addr, size, flags, &mh);

if (ret != 0) {
    // Fallback to auto-detection
    flags = 0;
    ret = gdr_pin_buffer_v2(gdr, addr, size, flags, &mh);
}
```

## CUDA Graph Capture Interaction (gotcha)

**Problem**: `cuMemAlloc`/`cuMemFree` and a synchronous `cuMemcpy` are **not legal inside an
active CUDA graph stream capture**. If the plugin performs a GPU memory op (e.g. allocating or
freeing a bounce/scratch buffer, or copying a signal) while the application's thread has a
stream capture in progress, the operation fails and the capture is invalidated. This was a
real, observed bug — fixed in commit `9013fa8` (*"rdma: fix GPU memory ops failing during CUDA
graph capture"*).

**Fix**: the plugin's GPU memory helpers now **temporarily disable stream capture** on the
calling thread around the operation using `cuThreadExchangeStreamCaptureMode`, and restore the
previous mode afterward. Copies additionally run on a **dedicated non-blocking side stream**
so they never join the captured default stream. See
[nccl_ofi_cuda.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_cuda.cpp)
(`nccl_net_ofi_gpu_mem_alloc`, `nccl_net_ofi_gpu_mem_free`,
`nccl_net_ofi_gpu_mem_copy_host_to_device`):

```cpp
int nccl_net_ofi_gpu_mem_alloc(void **ptr, size_t size)
{
    CUdeviceptr d_ptr;
    // Swap the calling thread into RELAXED capture mode so cuMemAlloc is allowed
    // even if the app has a capture in progress; restore the prior mode after.
    CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED;
    pfn_cuThreadExchangeStreamCaptureMode(&mode);      // disable/relax capture
    CUresult ret = pfn_cuMemAlloc(&d_ptr, size);
    pfn_cuThreadExchangeStreamCaptureMode(&mode);      // restore prior mode
    ...
}

int nccl_net_ofi_gpu_mem_copy_host_to_device(void *dst, void *src, size_t size)
{
    CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED;
    pfn_cuThreadExchangeStreamCaptureMode(&mode);
    // Use a NON-BLOCKING side stream so as not to interfere with any
    // graph capture on the legacy default stream.
    pfn_cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
    pfn_cuMemcpyHtoDAsync((CUdeviceptr)dst, src, size, stream);
    pfn_cuStreamSynchronize(stream);
    pfn_cuStreamDestroy(stream);
    pfn_cuThreadExchangeStreamCaptureMode(&mode);      // restore
}
```

**Takeaway for callers**: memory registration / device copies through the plugin are safe to
call from a thread that is capturing a CUDA graph; the plugin isolates them from the capture.

## `nccl_ofi_device_copy.h`: the device-copy abstraction

`include/nccl_ofi_device_copy.h` (new) defines an abstract
[`nccl_ofi_device_copy`](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_device_copy.h)
interface that decouples the rest of the plugin from *how* host↔device copies are performed.
GDRCopy is one concrete implementation (`nccl_ofi_gdrcopy_ctx`). The interface is:

- `register_region(device_ptr, size, out_handle)` — register a device region for copies, returns an opaque `RegHandle`.
- `copy_to_device(host_ptr, handle, offset, size)` — host → device.
- `copy_from_device(handle, offset, host_ptr, size)` — device → host.
- `forced_pcie_copy()` — true if the implementation forces the PCIe path (relevant on C2C
  systems such as Grace Hopper, where the GDRCopy v2 `GDR_PIN_FLAG_FORCE_PCIE` path is used).
- `deregister_region(handle)`.

The GIN path requires an implementation whose `forced_pcie_copy()` is true (GDRCopy 2.5+),
which is why the GIN proxy signal path insists on GDRCopy 2.5+ (see
[gin-alltoall-libfabric-trace.md](gin-alltoall-libfabric-trace.md)). In **GDAKI** mode the
GDRCopy registration is skipped entirely (commit `5b163ed`) because the device signals the
target through EFA hardware counters instead of a CPU-mapped copy.

## New CUDA helpers: bind a thread's GPU, inspect allocations

Commit `6014cd9` added helpers to
[nccl_ofi_cuda.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_cuda.cpp) to
bind a thread's GPU/context and inspect allocations, all via the **CUDA driver API**:

- `nccl_net_ofi_gpu_set_current_context(CUcontext)` / `nccl_net_ofi_gpu_get_current_context(CUcontext*)`
  — thin wrappers over `cuCtxSetCurrent` / `cuCtxGetCurrent`; they manage which context the
  calling thread is bound to and take **no ownership** of it. These matter for worker threads
  (e.g. the per-process gdrcopy worker) that must run on the right GPU context.
- Allocation-inspection helpers built on `cuPointerGetAttributes` /
  `cuMemGetAllocationGranularity` / `cuMemGetAllocationPropertiesFromHandle` /
  `cuMemRetainAllocationHandle` to determine device ordinal, memory type, and allocation
  properties for a pointer.

## EFA-GDA (GDAKI) CUDA dependency and build requirements

The GDAKI (kernel-initiated) GIN device path depends on the **vendored EFA-GDA CUDA sources**
at `3rd-party/efa-gda` (the *efa-dp-direct* project, pinned at `81c4dc7`). These sources
provide the GPU-side EFA datapath: posting work requests from GPU kernels and GPU-side
completion polling. They were vendored and wired into the build across commits `27920ab`
(vendor efa-dp-direct at `3rd-party/efa-gda`), `4735ddd`, `1383e93`, `38febcc`, `2fee64f`,
`e8f6174`, `0c7eb41`.

**Build requirement (hard):** `configure.ac` now **fails configuration** if the vendored CUDA
sources are missing:

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
| **GDRCopy** | Optional, approximately 5-10x faster than `cudaMemcpy()` |
| **Entry points** | CUDA 13: versioned, CUDA 12: legacy |
| **Performance** | Function resolution: ~10-50 μs, GDRCopy copy: ~1-5 μs |

**Key Takeaways**:
1. **Dynamic loading** enables cross-version compatibility without recompilation
2. **Driver API** preferred over Runtime API for ABI stability
3. **GPUDirect flush** needed on some architectures for correctness (~1-5 μs overhead)
4. **GDRCopy** optional but highly recommended for small message performance (significant speedup)
5. **DMA-BUF** modern approach (CUDA 11.7+), cleaner than legacy GDR
6. **Version detection** at runtime determines available features
7. **MR caching still critical** - registration expensive (100-500 μs)

**Related Documentation**:
- [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md) - DMA-BUF framework (used by CUDA)
- [neuron-memory.md](neuron-memory.md) - AWS Neuron (different approach)
- [rocm-memory.md](rocm-memory.md) - AMD ROCm (HIP API, no flush needed)
- [mr-cache-implementation.md](mr-cache-implementation.md) - MR cache (critical for CUDA)
- [kernel-efa-driver.md](kernel-efa-driver.md) - Kernel dmabuf import

---

## Code References

### Functions

**aws-ofi-nccl (CUDA integration)**:
- `nccl_net_ofi_gpu_init()` ([nccl_ofi_cuda.cpp:89-200](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_cuda.cpp))
- Function pointers: `pfn_cudaRuntimeGetVersion`, `pfn_cudaGetDriverEntryPointByVersion`, `pfn_cudaGetDriverEntryPoint` ([nccl_ofi_cuda.cpp:20-24](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_cuda.cpp))

### Macros

**aws-ofi-nccl (CUDA integration)**:
- `DECLARE_CUDA_FUNCTION(function, version)` ([nccl_ofi_cuda.cpp:40](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_cuda.cpp))
- `RESOLVE_CUDA_FUNCTION(function, version)` ([nccl_ofi_cuda.cpp:43-65](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_cuda.cpp))
- `LOAD_CUDA_RUNTIME_SYM(handle, sym)` ([nccl_ofi_cuda.cpp:67-72](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_cuda.cpp))

### CUDA Driver API Functions (External - NVIDIA)

Referenced but not linked (standard NVIDIA CUDA Driver API):
- `cuDriverGetVersion()` - Query CUDA driver version
- `cuCtxGetDevice()` - Get current device
- `cuDeviceGetAttribute()` - Query device attributes
- `cuMemAlloc()`, `cuMemFree()`, `cuMemcpy()` - Memory management
- `cuPointerGetAttributes()` - Query pointer attributes
- `cuFlushGPUDirectRDMAWrites()` - GPUDirect flush (CUDA 11.3+)
- `cuMemGetHandleForAddressRange()` - DMA-BUF export (CUDA 11.7+)

### CUDA Runtime API Functions (External - NVIDIA)

Referenced but not linked (standard NVIDIA CUDA Runtime API):
- `cudaRuntimeGetVersion()` - Query runtime version
- `cudaGetDriverEntryPoint()` - Legacy entry point resolver (CUDA 12)
- `cudaGetDriverEntryPointByVersion()` - Versioned entry point resolver (CUDA 13+)
- `cudaMalloc()`, `cudaMemcpy()` - Runtime memory operations

### GDRCopy API Functions (External - NVIDIA GDRCopy)

Referenced but not linked (GDRCopy library API):
- `gdr_open()` - Open GDRCopy handle
- `gdr_pin_buffer()`, `gdr_pin_buffer_v2()` - Pin GPU memory
- `gdr_map()` - Map GPU memory to CPU address space
- `gdr_copy_to_mapping()`, `gdr_copy_from_mapping()` - Fast CPU↔GPU copy

### Total Code References
- **1 function** from aws-ofi-nccl (nccl_net_ofi_gpu_init)
- **3 macros** from aws-ofi-nccl (DECLARE_CUDA_FUNCTION, RESOLVE_CUDA_FUNCTION, LOAD_CUDA_RUNTIME_SYM)
- **7 CUDA Driver API functions** (external NVIDIA)
- **4 CUDA Runtime API functions** (external NVIDIA)
- **5 GDRCopy API functions** (external NVIDIA)

All aws-ofi-nccl references use branch-form links into [aws/aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl/blob/master/) (`master`). Content in this document was verified against aws-ofi-nccl master `d840aa1` (tag v1.21.1). NVIDIA driver / open-gpu-kernel-modules details reflect release **610.57.04**.
