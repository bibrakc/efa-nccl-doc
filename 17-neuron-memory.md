# AWS Neuron Memory Registration

## Overview

**AWS Neuron** is the SDK for AWS **Trainium** and **Inferentia** accelerators used for machine learning training and inference. The Neuron memory subsystem provides specialized memory registration for these custom ML accelerators, similar to how DMA-BUF works for GPUs.

**Target Hardware**:
- **AWS Trainium** (trn1 instances) - Training accelerator
- **AWS Inferentia** (inf1, inf2 instances) - Inference accelerator

**Key Difference from CUDA/ROCm**: Neuron devices use a **peer-to-peer (P2P) memory model** rather than DMA-BUF for RDMA operations.

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
│   Libfabric                          │
│   - FI_HMEM support                 │
│   - Neuron HMEM provider            │
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

## Key Differences: Neuron vs CUDA/ROCm

| Aspect | CUDA/ROCm (DMA-BUF) | AWS Neuron (P2P) |
|--------|---------------------|------------------|
| **Memory export** | dmabuf file descriptor | neuron_p2p API |
| **Registration** | `ibv_reg_dmabuf_mr()` | `ibv_reg_mr()` (standard) |
| **Kernel API** | `dma_buf_get()` | `neuron_p2p_register_va()` |
| **Page size** | 4KB or 2MB (huge pages) | Configurable (4KB default) |
| **NCCL ptr type** | `NCCL_PTR_CUDA` | `NCCL_PTR_NEURON` |
| **Alignment** | **No page alignment** | **No page alignment required** |
| **DMA-BUF flag** | `FI_MR_DMABUF` | Not used |

**Key Insight**: Neuron memory does **not use DMA-BUF**. Instead, it uses a custom **peer-to-peer (P2P) registration** API.

## Neuron Memory Registration Flow

### Complete Flow (Neuron Memory → EFA RDMA)

```
1. Application allocates Neuron device memory
   neuron_alloc(&neuron_ptr, size);  // Neuron runtime

2. NCCL calls plugin regMr with NCCL_PTR_NEURON
   nccl_net_ofi_regMr(comm, neuron_ptr, size, NCCL_PTR_NEURON, &mhandle);

3. OFI Plugin detects Neuron memory
   // From nccl_ofi_interface_neuron.cpp:38
   if (ofi_properties.hmem_support) {
       props->ptrSupport |= NCCL_PTR_NEURON;
   }

4. OFI Plugin calls libfabric with FI_HMEM
   struct fi_mr_attr attr;
   attr.mr_iov = &iov;  // Standard iovec (not dmabuf!)
   attr.access = FI_SEND | FI_RECV;
   attr.iface = FI_HMEM_NEURON;  // Neuron HMEM interface
   fi_mr_regattr(domain, &attr, 0, &mr);

5. Libfabric EFA provider calls rdma-core
   ibv_reg_mr(pd, neuron_ptr, size, IBV_ACCESS_LOCAL_WRITE | ...);

6. rdma-core ioctl to kernel
   struct ib_uverbs_reg_mr cmd = {
       .start = (uintptr_t)neuron_ptr,
       .length = size,
       .access_flags = access_flags,
   };
   write(cmd_fd, &cmd, sizeof(cmd));

7. EFA Kernel Driver detects Neuron memory
   // From efa_neuronmem.c:83
   neuronmem = neuronmem_get(dev, ticket, start, length);

8. Kernel calls neuron_p2p_register_va()
   // From efa_neuronmem.c:73
   err = neuronmem->ops.register_va(addr, size, &neuronmem->va_info,
                                    neuronmem_free_cb, (void *)ticket);

9. Neuron driver returns physical page list
   struct neuron_p2p_va_info {
       u64 entries;                       // Number of entries
       struct neuron_p2p_page_info *page_info;
       unsigned int shift_page_size;      // Page size (e.g., 12 for 4KB)
   };

   struct neuron_p2p_page_info {
       u64 physical_address;              // Physical address
       u32 page_count;                    // Number of pages
   };

10. EFA driver creates IOMMU mapping
    // From efa_neuronmem.c:120
    for (ent_idx = 0; ent_idx < va_info->entries; ent_idx++) {
        pg_info = va_info->page_info + ent_idx;
        pa = pg_info->physical_address;
        for (pa_idx = 0; pa_idx < pg_info->page_count; pa_idx++) {
            page_list[pg_idx++] = pa;
            pa += BIT(va_info->shift_page_size);
        }
    }

11. Hardware registration and key generation
    EFA hardware programmed with physical addresses
    lkey/rkey generated and returned to userspace

12. Future sends use registered MR
    fi_send(ep, neuron_ptr, size, fi_mr_desc(mr), ...);
    // EFA DMA's directly from Neuron device memory!
```

## Neuron P2P API

### Kernel-Side Data Structures

```c
// From amzn-drivers/kernel/linux/efa/src/neuron_p2p.h

/**
 * struct neuron_p2p_page_info - Physical page information
 * @physical_address: Physical address of the page
 * @page_count: Number of contiguous pages at this address
 */
struct neuron_p2p_page_info {
    u64 physical_address;
    u32 page_count;
};

/**
 * struct neuron_p2p_va_info - Virtual address mapping information
 * @entries: Number of page_info entries
 * @page_info: Array of physical page information
 * @shift_page_size: Page size shift (e.g., 12 for 4KB pages)
 */
struct neuron_p2p_va_info {
    u64 entries;
    struct neuron_p2p_page_info *page_info;
    unsigned int shift_page_size;
};

/**
 * Register a virtual address range for P2P access
 *
 * @param virtual_address: Virtual address to register
 * @param length: Length of the region
 * @param vainfo: Returns VA info with physical pages
 * @param free_callback: Called when pages are freed
 * @param data: Opaque data passed to callback
 * @return 0 on success, negative on error
 */
int neuron_p2p_register_va(u64 virtual_address, u64 length,
                           struct neuron_p2p_va_info **vainfo,
                           void (*free_callback)(void *data),
                           void *data);

/**
 * Unregister a previously registered VA
 *
 * @param vainfo: VA info from neuron_p2p_register_va()
 * @return 0 on success, negative on error
 */
int neuron_p2p_unregister_va(struct neuron_p2p_va_info *vainfo);
```

### EFA Neuron Memory Module

```c
// From amzn-drivers/kernel/linux/efa/src/efa_neuronmem.c

#define NEURON_PAGE_SHIFT 12
#define NEURON_PAGE_SIZE BIT_ULL(NEURON_PAGE_SHIFT)  // 4096 bytes

struct efa_neuronmem {
    struct efa_p2pmem p2pmem;                 // Base P2P memory structure
    struct efa_neuronmem_ops ops;             // Function pointers
    struct neuron_p2p_va_info *va_info;       // VA info from Neuron
    u64 virt_start;                           // Aligned start address
};

/**
 * Get Neuron memory provider
 * Called by EFA driver to handle Neuron memory registration
 */
static struct efa_p2pmem *neuronmem_get(struct efa_dev *dev, u64 ticket,
                                        u64 start, u64 length)
{
    struct efa_neuronmem *neuronmem;
    u64 virt_start;
    u64 virt_end;
    u64 pinsz;

    neuronmem = kzalloc(sizeof(*neuronmem), GFP_KERNEL);

    // Align to Neuron page size
    virt_start = ALIGN_DOWN(start, NEURON_PAGE_SIZE);
    virt_end = ALIGN(start + length, NEURON_PAGE_SIZE);
    pinsz = virt_end - virt_start;
    neuronmem->virt_start = virt_start;

    // Get function pointers (symbol_get from Neuron module)
    err = neuronmem_get_fp(neuronmem);

    // Register with Neuron driver
    err = neuronmem_register_va(dev, neuronmem, virt_start, pinsz, ticket);

    return &neuronmem->p2pmem;
}

/**
 * Convert VA info to page list for EFA hardware
 */
static int neuronmem_to_page_list(struct efa_dev *dev,
                                  struct efa_p2pmem *p2pmem,
                                  u64 *page_list)
{
    struct neuron_p2p_page_info *pg_info;
    struct neuron_p2p_va_info *va_info;
    int ent_idx, pa_idx;
    int pg_idx = 0;
    u64 pa;

    neuronmem = container_of(p2pmem, struct efa_neuronmem, p2pmem);
    va_info = neuronmem->va_info;

    // Iterate through all page info entries
    for (ent_idx = 0; ent_idx < va_info->entries; ent_idx++) {
        pg_info = va_info->page_info + ent_idx;
        pa = pg_info->physical_address;

        // Expand contiguous pages into page list
        for (pa_idx = 0; pa_idx < pg_info->page_count; pa_idx++) {
            page_list[pg_idx++] = pa;
            pa += BIT(va_info->shift_page_size);  // Next page
        }
    }

    return 0;
}

/**
 * Release Neuron memory registration
 */
static void neuronmem_release(struct efa_dev *dev,
                              struct efa_p2pmem *p2pmem,
                              bool in_cb)
{
    struct efa_neuronmem *neuronmem;

    neuronmem = container_of(p2pmem, struct efa_neuronmem, p2pmem);

    // Unregister with Neuron driver
    neuronmem->ops.unregister_va(neuronmem->va_info);

    neuronmem_put_fp();
    kfree(neuronmem);
}
```

## OFI Plugin Integration

### Neuron Interface Variant

```cpp
// From aws-ofi-nccl/src/nccl_ofi_interface_neuron.cpp

static ncclResult_t getProperties_v5(int dev_id, ncclNetProperties_v5_t* props)
{
    nccl_ofi_properties_t ofi_properties;
    ncclResult_t ret = nccl_net_ofi_get_properties(dev_id, &ofi_properties);

    props->name = ofi_properties.name;
    props->pciPath = ofi_properties.pci_path;
    props->guid = ofi_properties.guid;

    // Default to host memory support
    props->ptrSupport = NCCL_PTR_HOST;

    // Add Neuron support if HMEM is available
    if (ofi_properties.hmem_support) {
        props->ptrSupport |= NCCL_PTR_NEURON;  // Enable Neuron memory
    }

    // Global MR registration support
    props->regIsGlobal = ofi_properties.regIsGlobal;

    props->speed = ofi_properties.port_speed;
    props->latency = ofi_properties.latency;
    props->maxComms = ofi_properties.max_communicators;

    return ret;
}

// Neuron-specific plugin exports
NCCL_OFI_EXPORT_SYMBOL ncclNet_v6_t ncclNetPlugin_v6 = {
    .name = "AWS Libfabric",
    .init = nccl_net_ofi_init_v6,
    .devices = nccl_net_ofi_devices_v2,
    .getProperties = getProperties_v5,
    .regMr = nccl_net_ofi_regMr_v8,       // Standard registration
    .regMrDmaBuf = nccl_net_ofi_regMrDmaBuf_v6,  // Not used for Neuron
    // ... other functions
};
```

## Page Alignment: Neuron vs CUDA

```c
// From aws-ofi-nccl/include/nccl_ofi_mr.h:149

static inline nccl_ofi_mr_ckey_t nccl_ofi_mr_ckey_mk_vec(void *iov_base,
                                                         size_t iov_len,
                                                         nccl_net_ofi_ep_t *ep_ptr)
{
#if !HAVE_NEURON
    // CUDA/ROCm: Round to page boundaries
    nccl_ofi_mr_ckey_round(&iov_len, &iov_base, "iovec");
#endif
    // Neuron: NO rounding! Use exact address/size

    nccl_ofi_mr_ckey_t cache_key = {};
    cache_key.iovec.iov_base = iov_base;
    cache_key.iovec.iov_len = iov_len;
    cache_key.ep = ep_ptr;
    cache_key.type = NCCL_OFI_MR_CKEY_IOVEC;
    return cache_key;
}
```

**Why No Rounding for Neuron?**
- Neuron driver handles page alignment internally
- Application can register arbitrary address ranges
- Simplifies MR cache key generation

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| `neuron_p2p_register_va()` | 100-500 μs | Pins pages, creates IOMMU mapping |
| Symbol resolution (`symbol_get`) | 5-20 μs | One-time per registration |
| Page list generation | 10-50 μs | Depends on memory fragmentation |
| MR cache lookup (Neuron) | <1 μs | Same as CUDA/ROCm |
| Data transfer | 10-20 μs | Same as GPU memory |

**Key Insight**: Performance similar to CUDA/ROCm, but uses different kernel API.

## Configuration and Debugging

### Check Neuron Support

```bash
# Check if Neuron driver loaded
lsmod | grep neuron

# Check Neuron devices
ls -la /dev/neuron*

# Check if EFA supports Neuron
dmesg | grep -i "neuron\|NEURON"

# Should see: "efa 0000:00:06.0: P2P: NEURON"
```

### Enable Neuron Memory in NCCL

```bash
# Neuron memory support automatic if:
# 1. Neuron driver loaded
# 2. libfabric has FI_HMEM support
# 3. EFA has efa_neuronmem module

# Check NCCL detection
export NCCL_DEBUG=INFO
# Look for: "ptrSupport NCCL_PTR_HOST | NCCL_PTR_NEURON"
```

### Debugging Neuron Registration

```bash
# Enable EFA debug
echo 'module efa +p' > /sys/kernel/debug/dynamic_debug/control

# Watch for Neuron registration
dmesg -w | grep -i neuron

# Look for:
# "neuron_p2p_register_va called"
# "neuronmem_to_page_list: X pages"
```

### Check Neuron Page Size

```bash
# Neuron typically uses 4KB pages
# Check from dmesg:
dmesg | grep "shift_page_size"
# shift_page_size=12 → 4KB pages (2^12)
```

## Neuron vs DMA-BUF Comparison

### Registration Path Comparison

**CUDA (DMA-BUF)**:
```
cudaMalloc() → cudaExportMemory() → dmabuf_fd
  → fi_mr_regattr(FI_MR_DMABUF)
  → ibv_reg_dmabuf_mr(fd)
  → dma_buf_get(fd)
  → dma_buf_map_attachment()
```

**Neuron (P2P)**:
```
neuron_alloc() → neuron_ptr (no export!)
  → fi_mr_regattr(FI_HMEM_NEURON)
  → ibv_reg_mr(neuron_ptr)
  → neuron_p2p_register_va(neuron_ptr)
  → Returns physical pages directly
```

### Advantages of Neuron P2P Approach

1. **Simpler**: No file descriptor export/import
2. **Faster**: Direct symbol calls, no dma_buf overhead
3. **Flexible**: No kernel version dependency (5.12+)
4. **Custom**: Neuron-specific optimizations possible

## Common Issues

### Problem: Neuron registration fails with EINVAL

**Cause**: Neuron module not loaded
```bash
# Check:
lsmod | grep neuron

# Load:
sudo modprobe neuron
```

### Problem: NCCL doesn't detect Neuron support

**Cause**: libfabric built without FI_HMEM support
```bash
# Check libfabric:
fi_info -c FI_HMEM
# Should show Neuron in HMEM list

# Rebuild libfabric with --enable-neuron
```

### Problem: Slow Neuron memory operations

**Cause**: MR cache not working
```bash
# Check cache stats:
export NCCL_DEBUG=INFO
# Look for "MR cache X hits Y misses"
# Hit rate should be >95%
```

## Summary

| Aspect | Details |
|--------|---------|
| **Hardware** | AWS Trainium, Inferentia (ML accelerators) |
| **API** | `neuron_p2p_register_va()` (not DMA-BUF) |
| **NCCL ptr type** | `NCCL_PTR_NEURON` |
| **Kernel module** | `efa_neuronmem.c` (~4KB) |
| **Page size** | 4KB (configurable) |
| **Alignment** | **Not required** (Neuron handles internally) |
| **libfabric** | FI_HMEM_NEURON interface |
| **Registration** | Standard `ibv_reg_mr()` (not dmabuf variant) |

**Key Takeaways**:
1. Neuron uses **P2P model**, not DMA-BUF
2. **Simpler than CUDA**: No file descriptor export, direct kernel API
3. **No page alignment required**: Application can use arbitrary addresses
4. **Symbol-based**: Uses `symbol_get()` to call Neuron driver functions
5. Registration is still expensive (100-500 μs) - **MR caching critical**
6. Performance similar to CUDA/ROCm once registered
7. AWS-specific, works on trn1/inf1/inf2 instances

**Related Documentation**:
- [16-dmabuf-gpu-memory.md](16-dmabuf-gpu-memory.md) - CUDA/ROCm DMA-BUF approach (different!)
- [17-rocm-memory.md](17-rocm-memory.md) - AMD ROCm memory registration
- [08-mr-cache-implementation.md](08-mr-cache-implementation.md) - MR cache (works with Neuron keys)
- [13-kernel-efa-driver.md](13-kernel-efa-driver.md) - EFA kernel driver integration
