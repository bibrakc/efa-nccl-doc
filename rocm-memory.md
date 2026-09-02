# AMD ROCm Memory Registration

## Overview

**AMD ROCm** (Radeon Open Compute) is the software stack for AMD GPUs. ROCm memory registration enables RDMA transfers to/from AMD GPU memory using **HIP** (Heterogeneous-Compute Interface for Portability), AMD's CUDA-equivalent API.

**Target Hardware**:
- AMD MI-series GPUs (MI50, MI100, MI200, MI300, etc.)
- AMD Radeon Instinct accelerators
- Any ROCm-compatible AMD GPU

**Memory Registration Approach**: ROCm supports **DMA-BUF** (kernel 5.12+) for modern registration, with fallback to legacy methods.

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
│   Libfabric                          │
│   - FI_HMEM support                 │
│   - FI_MR_DMABUF flag               │
├─────────────────────────────────────┤
│   rdma-core                          │
│   - ibv_reg_dmabuf_mr() (modern)    │
│   - ibv_reg_mr() (legacy)           │
├─────────────────────────────────────┤
│   EFA Kernel Driver                  │
│   - DMA-BUF import (kernel 5.12+)   │
│   - IOMMU mapping                   │
├─────────────────────────────────────┤
│   amdgpu Driver                      │
│   - Exports GPU memory as dmabuf    │
│   - Provides physical pages         │
├─────────────────────────────────────┤
│   AMD GPU Hardware                   │
└─────────────────────────────────────┘
```

## ROCm vs CUDA Comparison

| Aspect | NVIDIA CUDA | AMD ROCm | Notes |
|--------|-------------|----------|-------|
| **Allocation API** | `cudaMalloc()` | `hipMalloc()` | HIP mirrors CUDA API |
| **Copy API** | `cudaMemcpy()` | `hipMemcpy()` | HIP compatible |
| **Free API** | `cudaFree()` | `hipFree()` | HIP compatible |
| **Pointer query** | `cudaPointerGetAttributes()` | `hipPointerGetAttribute()` | HIP compatible |
| **DMA-BUF export** | `cudaExternalMemoryGetMappedBuffer()` | `hipMemGetHandleForAddressRange()` | Different API |
| **DMA-BUF flag** | `HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD` | Same | ROCm 5.0+ |
| **NCCL ptr type** | `NCCL_PTR_CUDA` | `NCCL_PTR_CUDA` | **Same!** |
| **GPUDirect** | `cuda_flush` required | No flush needed | Simpler |

**Key Insight**: ROCm uses **NCCL_PTR_CUDA** (not a separate type), and HIP API is CUDA-compatible.

## ROCm Memory Registration Flow

### Complete Flow (AMD GPU Memory → EFA RDMA)

```
1. Application allocates GPU memory
   hipMalloc(&gpu_ptr, size);  // HIP API (CUDA-like)

2. HIP exports as dmabuf (ROCm 5.0+)
   int dmabuf_fd;
   size_t offset;
   hipMemGetHandleForAddressRange(&dmabuf_fd, (uintptr_t)gpu_ptr, size,
                                  HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
   // offset is always 0 for ROCm

3. NCCL calls plugin regMr with NCCL_PTR_CUDA
   nccl_net_ofi_regMr(comm, gpu_ptr, size, NCCL_PTR_CUDA, &mhandle);

4. OFI Plugin checks for dmabuf support
   if (nccl_ofi_dmabuf_viable_and_supported(provider)) {
       // Use dmabuf path
   } else {
       // Fallback to legacy
   }

5. OFI Plugin calls libfabric with FI_MR_DMABUF
   struct fi_mr_attr attr;
   attr.dmabuf = &fi_mr_dmabuf{dmabuf_fd, 0, size, gpu_ptr};
   fi_mr_regattr(domain, &attr, FI_MR_DMABUF, &mr);

6. Libfabric EFA provider calls rdma-core
   ibv_reg_dmabuf_mr(pd, 0, size, dmabuf_fd, access_flags);

7. rdma-core ioctl to kernel
   struct ib_uverbs_reg_dmabuf_mr cmd = {
       .fd = dmabuf_fd,
       .offset = 0,
       .length = size,
       .access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
   };
   write(cmd_fd, &cmd, sizeof(cmd));

8. EFA Kernel Driver imports dmabuf
   dmabuf = dma_buf_get(dmabuf_fd);
   attachment = dma_buf_attach(dmabuf, &efa_dev->pdev->dev);
   sgt = dma_buf_map_attachment(attachment, DMA_BIDIRECTIONAL);

9. amdgpu driver provides scatter-gather table
   // Physical pages for GPU memory
   struct sg_table with AMD GPU physical addresses

10. EFA creates IOMMU mapping
    IOMMU maps GPU physical addresses for DMA

11. Hardware registration
    lkey/rkey generated and returned

12. Future sends use cached MR
    fi_send(ep, gpu_ptr, size, fi_mr_desc(mr), ...);
    // EFA DMA's directly from AMD GPU memory!
```

## HIP API Integration

### Basic HIP Memory Operations

```c
// From aws-ofi-nccl/src/nccl_ofi_rocm.cpp

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

/**
 * Initialize ROCm/HIP
 */
int nccl_net_ofi_gpu_init(void)
{
    int driverVersion = -1;
    int runtimeVersion = -1;

    // Query HIP driver version
    hipError_t res = hipDriverGetVersion(&driverVersion);
    if (res != hipSuccess) {
        NCCL_OFI_WARN("Failed to query HIP driver version.");
        return -EINVAL;
    }

    // Query HIP runtime version
    res = hipRuntimeGetVersion(&runtimeVersion);
    if (res != hipSuccess) {
        NCCL_OFI_WARN("Failed to query HIP runtime version.");
        return -EINVAL;
    }

    NCCL_OFI_INFO("Using HIP driver version %d with runtime %d",
                  driverVersion, runtimeVersion);

    // ROCm doesn't need GPUDirect flush
    cuda_flush = false;

    return 0;
}

/**
 * Get GPU device ID for a buffer pointer
 */
int nccl_net_ofi_get_gpu_device_for_addr(void *data, int *dev_id)
{
    unsigned int mem_type;
    unsigned int device_ordinal;

    // Query pointer attributes
    hipError_t cuda_ret_mem = hipPointerGetAttribute(
        &device_ordinal,
        HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        data);

    hipError_t cuda_ret_dev = hipPointerGetAttribute(
        &mem_type,
        HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
        data);

    if (cuda_ret_mem != hipSuccess || cuda_ret_dev != hipSuccess) {
        NCCL_OFI_WARN("Invalid buffer pointer provided");
        return -ENOTSUP;
    }

    *dev_id = device_ordinal;
    return 0;
}

/**
 * Allocate GPU memory
 */
int nccl_net_ofi_gpu_mem_alloc(void **ptr, size_t size)
{
    hipError_t ret = hipMalloc(ptr, size);
    return ret == hipSuccess ? 0 : -EINVAL;
}

/**
 * Free GPU memory
 */
int nccl_net_ofi_gpu_mem_free(void *ptr)
{
    hipError_t ret = hipFree(ptr);
    return ret == hipSuccess ? 0 : -EINVAL;
}

/**
 * Copy host to device
 */
int nccl_net_ofi_gpu_mem_copy_host_to_device(void *dst, void *src, size_t size)
{
    hipError_t ret = hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
    return ret == hipSuccess ? 0 : -EINVAL;
}
```

### DMA-BUF Export (ROCm 5.0+)

```c
// From aws-ofi-nccl/src/nccl_ofi_rocm.cpp:103

/**
 * Get DMA-BUF file descriptor for GPU memory
 * Requires ROCm 5.0+ with HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD support
 */
int nccl_net_ofi_gpu_get_dma_buf_fd(void *aligned_ptr, size_t aligned_size,
                                    int *fd, size_t *offset)
{
#if defined(HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD)
    // Export GPU memory as DMA-BUF
    hipError_t ret = hipMemGetHandleForAddressRange(
        fd,                                      // Output: dmabuf FD
        (uintptr_t)aligned_ptr,                 // GPU memory address
        aligned_size,                            // Size
        HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,   // Type: dmabuf
        0);                                      // Flags

    *offset = 0;  // ROCm always uses offset 0

    return ret == hipSuccess ? 0 : -EINVAL;
#else
    // ROCm version doesn't support dmabuf
    return -ENOTSUP;
#endif
}
```

## DMA-BUF Support Detection

### ROCm Version Requirements

```c
// From aws-ofi-nccl/include/nccl_ofi_rocm.h

/**
 * Check if GPU supports DMA-BUF
 * ROCm 5.0+ required
 */
bool nccl_net_ofi_gpu_have_dma_buf_attr(void)
{
#if defined(HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD)
    return true;   // ROCm 5.0+ with dmabuf support
#else
    return false;  // Older ROCm, no dmabuf
#endif
}

/**
 * Check if GPU supports GPUDirect RDMA
 * ROCm doesn't require special GDR support
 */
bool nccl_net_ofi_gpu_have_gdr_support_attr(void)
{
    return false;  // Not applicable for ROCm
}
```

### Runtime Detection

```bash
# Check ROCm version
rocm-smi --showversion
# Need ROCm 5.0 or later for dmabuf

# Check HIP version
hipconfig --version

# Check if dmabuf supported
rocm-smi --showmeminfo vram
```

## GPUDirect RDMA: ROCm vs CUDA

### CUDA Requires Flush

```c
// NVIDIA CUDA
int nccl_net_ofi_gpu_flush_gpudirect_rdma_writes(void)
{
    if (cuda_flush) {
        cudaError_t res = cudaDeviceFlushGPUDirectRDMAWrites(
            cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
            cudaFlushGPUDirectRDMAWritesToOwner);
        return res == cudaSuccess ? 0 : -1;
    }
    return 0;
}
```

### ROCm: No Flush Needed

```c
// AMD ROCm
int nccl_net_ofi_gpu_flush_gpudirect_rdma_writes(void)
{
    // ROCm doesn't require GPUDirect flush
    return -EPERM;  // Not implemented/not needed
}
```

**Why?** AMD GPU architecture handles memory coherency differently than NVIDIA.

## Performance Characteristics

| Operation | CUDA (NVIDIA) | ROCm (AMD) | Notes |
|-----------|---------------|------------|-------|
| `hipMalloc()` | ~500 μs | ~500 μs | Similar to cudaMalloc |
| `hipMemGetHandleForAddressRange()` | N/A | 10-50 μs | DMA-BUF export |
| Memory registration (dmabuf) | 100-500 μs | 100-500 μs | Similar performance |
| GPUDirect flush | 1-5 μs | **Not needed** | ROCm advantage |
| Data transfer latency | 10-20 μs | 10-20 μs | Same once registered |
| MR cache hit | <1 μs | <1 μs | Same caching |

**Key Insight**: ROCm performance similar to CUDA, **simpler** (no GPUDirect flush required).

## Configuration and Debugging

### Enable ROCm Support

```bash
# ROCm must be installed
dpkg -l | grep rocm

# Check amdgpu driver loaded
lsmod | grep amdgpu

# Check GPU visibility
rocm-smi

# Expected output:
# GPU[0]    : GPU ID: 0x740f
# GPU[0]    : Temperature: 35.0c
# GPU[0]    : Memory Total: 32768 MB
```

### Build OFI Plugin with ROCm

```bash
# Configure with ROCm support
cd aws-ofi-nccl
./autogen.sh
./configure --with-hip=/opt/rocm --enable-platform-aws

# Check configuration
grep HAVE_ROCM config.h
# Should see: #define HAVE_ROCM 1

make
sudo make install
```

### Debug ROCm Memory Registration

```bash
# Enable debug
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=NET
export FI_LOG_LEVEL=debug

# Run NCCL test
./nccl-tests/build/all_reduce_perf -b 8 -e 1G -g 8

# Look for:
# "Using HIP driver version X with runtime Y"
# "Will use DMA-BUF" or "DMA-BUF disabled"
# "Found MR handle for dmabuf in cache"
```

### Check DMA-BUF Support

```bash
# Check kernel version (need 5.12+)
uname -r

# Check if HIP supports dmabuf
cat > test_dmabuf.cpp <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>

int main() {
#ifdef HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD
    printf("DMA-BUF supported\n");
    return 0;
#else
    printf("DMA-BUF NOT supported\n");
    return 1;
#endif
}
EOF

hipcc test_dmabuf.cpp -o test_dmabuf
./test_dmabuf
```

### Monitor GPU Memory

```bash
# Watch GPU memory usage
watch -n 1 rocm-smi --showmeminfo vram

# Check RDMA statistics
cat /sys/class/infiniband/efa_0/ports/1/hw_counters/*
```

## ROCm Limitations and Workarounds

### 1. Older ROCm Versions (< 5.0)

**Problem**: No DMA-BUF support

**Workaround**: Use legacy GPUDirect RDMA (if available) or host bounce buffers

```c
// Fallback for old ROCm
if (!nccl_net_ofi_gpu_have_dma_buf_attr()) {
    // Copy GPU → Host → NIC → Host → GPU
    hipMemcpy(host_buf, gpu_buf, size, hipMemcpyDeviceToHost);
    fi_send(ep, host_buf, size, ...);
}
```

### 2. Memory Fragmentation

**Problem**: GPU memory fragmentation reduces page merging efficiency

**Workaround**: Allocate large contiguous regions early

```c
// Allocate all GPU memory at initialization
hipMalloc(&large_pool, total_size);
// Manually manage subregions
```

### 3. Multi-GPU Systems

**Problem**: GPU device selection

**Workaround**: Use `HIP_VISIBLE_DEVICES` or explicit device selection

```bash
# Limit to specific GPUs
export HIP_VISIBLE_DEVICES=0,1,2,3

# Or in code:
hipSetDevice(desired_gpu_id);
```

## ROCm vs CUDA Code Comparison

### CUDA Version
```c
#include <cuda_runtime.h>

cudaMalloc(&ptr, size);
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
cudaPointerGetAttributes(&attrs, ptr);
cudaFree(ptr);
cudaDeviceFlushGPUDirectRDMAWrites(...);  // Required!
```

### ROCm Version (HIP)
```c
#include <hip/hip_runtime.h>

hipMalloc(&ptr, size);
hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
hipPointerGetAttribute(&device_id, HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr);
hipFree(ptr);
// No flush needed!
```

**Difference**: Mostly drop-in compatible, **no GPUDirect flush** required for ROCm!

## Common Issues

### Problem: `hipMemGetHandleForAddressRange()` fails

**Cause**: ROCm version < 5.0 or dmabuf not enabled

**Solution**:
```bash
# Upgrade ROCm
sudo apt update
sudo apt install rocm-dev

# Check version
rocm-smi --showversion
# Should be 5.0+
```

### Problem: Poor performance compared to CUDA

**Cause**: Memory not page-aligned, causing extra IOMMU entries

**Solution**: Use MR cache aggressively
```bash
# There is no NCCL_OFI_MR_CACHE_SIZE variable. Plugin/provider MR cache controls:
export OFI_NCCL_MR_CACHE_DISABLE=0   # 0 = plugin MR cache enabled (default)
export FI_EFA_MR_MAX_CACHED_SIZE=<bytes>
export FI_EFA_MR_MAX_CACHED_COUNT=<n>
```

### Problem: "Invalid buffer pointer" error

**Cause**: Passing host memory as GPU memory

**Solution**: Verify pointer type
```c
unsigned int mem_type;
hipPointerGetAttribute(&mem_type, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr);
if (mem_type != hipMemoryTypeDevice) {
    // Not GPU memory!
}
```

## Summary

| Aspect | Details |
|--------|---------|
| **Hardware** | AMD MI-series GPUs, Radeon Instinct |
| **API** | HIP (CUDA-compatible) |
| **DMA-BUF support** | ROCm 5.0+ |
| **NCCL ptr type** | `NCCL_PTR_CUDA` (same as NVIDIA!) |
| **Export API** | `hipMemGetHandleForAddressRange()` |
| **GPUDirect flush** | **Not required** (simpler than CUDA) |
| **Kernel requirement** | Linux 5.12+ (for dmabuf) |
| **Registration** | `ibv_reg_dmabuf_mr()` (modern) |
| **Performance** | Similar to CUDA once registered |

**Key Takeaways**:
1. ROCm uses **HIP API** (CUDA-compatible, easy porting)
2. **DMA-BUF** support since ROCm 5.0 (kernel 5.12+)
3. Uses **NCCL_PTR_CUDA** type (not a separate ROCm type)
4. **No GPUDirect flush needed** - simpler than NVIDIA
5. Registration expensive (100-500 μs) - **MR caching essential**
6. Performance similar to CUDA, easier to maintain (fewer special cases)
7. Fallback to legacy methods if dmabuf unsupported
8. DMA-BUF is **no longer gated by EFA generation** in the plugin (the EFA Gen 1-3 device-id
   check was removed in commit `0f285d5`); see [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md).

### Related upstream changes

- **EFA driver r3.3.0** adds >4GB MR page-size support and an extended page-shift field in MR
  registration (amzn-drivers `bf83e44`, `a1e35dc`), improving large AMD GPU registrations.
- **libfabric 2.7** (`main`, verified against `cb6364e05`, 2.7.0rc1) adds ROCr/hmem
  improvements paralleling the CUDA ones: async copy operations and dmabuf-fd retrieval via
  `rocr_hmem_get_dmabuf_fd()` / `rocr_hmem_put_dmabuf_fd()`
  ([src/hmem_rocr.c](https://github.com/ofiwg/libfabric/blob/main/src/hmem_rocr.c),
  used by [prov/efa/src/efa_hmem.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_hmem.c)).

**Related Documentation**:
- [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md) - DMA-BUF framework (used by ROCm)
- [neuron-memory.md](neuron-memory.md) - AWS Neuron (different approach, no dmabuf)
- [mr-cache-implementation.md](mr-cache-implementation.md) - MR cache (works with ROCm)
- [kernel-efa-driver.md](kernel-efa-driver.md) - Kernel driver dmabuf import

---

## Code References

### Functions

**AMD ROCm/HIP API (External - AMD)** - Referenced but not linked:
- `hipMalloc()` - Allocate GPU memory
- `hipMemcpy()` - Copy memory (CPU ↔ GPU)
- `hipMemGetHandleForAddressRange()` - Export GPU memory as dmabuf (ROCm 5.0+)
- `hipGetDeviceProperties()` - Query device capabilities

**rdma-core (libibverbs)** - Referenced but not linked (standard RDMA API):
- `ibv_reg_dmabuf_mr()` - Register dmabuf as memory region (modern method)
- `ibv_reg_mr()` - Register memory region (legacy method)

**libfabric (OFI)** - Referenced but not linked (standard libfabric API):
- `fi_mr_regattr()` - Register MR with dmabuf attribute

### Total Code References
- **4 ROCm/HIP functions** (external AMD)
- **2 rdma-core functions** (standard API)
- **1 libfabric function** (standard API)

**Note**: ROCm/HIP is AMD's open-source GPU software stack. HIP API is designed to be CUDA-compatible for easy porting.
