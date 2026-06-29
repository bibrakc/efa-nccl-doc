# DMA-BUF and GPU Memory Registration

## Overview

**DMA-BUF** (DMA Buffer Sharing) is a Linux kernel framework for sharing memory buffers between different devices (GPU, NIC, camera, etc.) without copying data. For NCCL+EFA, dmabuf enables **direct GPU-to-NIC memory registration** for efficient RDMA transfers.

**GitHub**: [src/nccl_ofi_dmabuf.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_dmabuf.cpp)

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

### Kernel-Side (Linux Kernel)

**`struct dma_buf`** ([linux/include/linux/dma-buf.h:294-400](https://github.com/torvalds/linux/blob/e84d960/include/linux/dma-buf.h#L294-L400)):

```c
// From linux/include/linux/dma-buf.h

/**
 * struct dma_buf - DMA buffer object
 * @file: file pointer (used for sharing via file descriptor)
 * @attachments: list of dma_buf_attachment
 * @ops: dma_buf_ops - operations for this dmabuf
 * @size: size of the buffer
 */
struct dma_buf {
    size_t size;
    struct file *file;
    struct list_head attachments;
    const struct dma_buf_ops *ops;
    void *priv;  // Private data (GPU driver specific)
    // ...
};

/**
 * struct dma_buf_attachment - attachment point for importing device
 * @dmabuf: buffer for this attachment
 * @dev: device attached to the buffer (e.g., EFA PCI device)
 * @sgt: scatter-gather table (physical pages)
 */
struct dma_buf_attachment {
    struct dma_buf *dmabuf;
    struct device *dev;
    struct sg_table *sgt;  // Physical page addresses
    void *priv;
    // ...
};

/**
 * struct sg_table - scatter-gather table
 * List of physical memory pages
 */
struct sg_table {
    struct scatterlist *sgl;  // Linked list of SG entries
    unsigned int nents;        // Number of entries
    unsigned int orig_nents;   // Original number before coalescing
};

struct scatterlist {
    unsigned long  page_link;  // Physical page address
    unsigned int   offset;     // Offset within page
    unsigned int   length;     // Length in bytes
    dma_addr_t     dma_address; // DMA address (IOMMU mapped)
    unsigned int   dma_length;
};
```

### User-Space (libfabric/rdma-core)

```c
// From libfabric rdma/fi_domain.h

**`struct fi_mr_dmabuf`** ([libfabric/include/rdma/fi_domain.h:151-156](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L151-L156)):

```c
/**
 * struct fi_mr_dmabuf - DMA-BUF memory region descriptor
 * Used with FI_MR_DMABUF flag
 */
struct fi_mr_dmabuf {
    int fd;                // DMA-BUF file descriptor
    uint64_t offset;       // Offset into dmabuf
    size_t len;            // Length to register
    void *base_addr;       // Virtual address (may be NULL)
    uint64_t flags;        // Additional flags
};

// Memory registration with dmabuf
struct fi_mr_attr {
    const struct fi_mr_dmabuf *dmabuf;  // DMA-BUF descriptor
    // OR
    const struct iovec *mr_iov;         // Traditional virtual address
    // ...
};

int fi_mr_regattr(struct fid_domain *domain,
                  const struct fi_mr_attr *attr,
                  uint64_t flags,  // Include FI_MR_DMABUF
                  struct fid_mr **mr);
```

### OFI Plugin (MR Cache Key)

```c
// From aws-ofi-nccl/include/nccl_ofi_mr.h

struct nccl_ofi_mr_ckey {
    union {
        struct iovec iovec;                // CPU memory (virt addr + len)
        struct fi_mr_dmabuf fi_mr_dmabuf;  // GPU memory (fd + offset + len)
    };
    nccl_net_ofi_ep_t *ep;
    enum nccl_ofi_mr_ckey_type type;  // IOVEC or DMABUF
};

// Create dmabuf cache key
static inline nccl_ofi_mr_ckey_t nccl_ofi_mr_ckey_mk_dmabuf(
    int fd, uint64_t offset, size_t len,
    void *base_addr, nccl_net_ofi_ep_t *ep_ptr)
{
    nccl_ofi_mr_ckey_t cache_key = {};
    cache_key.fi_mr_dmabuf.fd = fd;
    cache_key.fi_mr_dmabuf.offset = offset;
    cache_key.fi_mr_dmabuf.len = len;
    cache_key.fi_mr_dmabuf.base_addr = base_addr;
    cache_key.ep = ep_ptr;
    cache_key.type = NCCL_OFI_MR_CKEY_DMABUF;
    return cache_key;
}
```

## DMA-BUF Registration Flow

### Complete Flow (GPU Memory → EFA RDMA)

```
1. Application allocates GPU memory
   cudaMalloc(&gpu_ptr, size);  // GPU address (not CPU accessible!)

2. CUDA exports as dmabuf
   cudaExternalMemoryGetMappedBuffer(&dmabuf_fd, gpu_ptr, ...);
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
   struct ib_uverbs_reg_dmabuf_mr cmd = {
       .fd = dmabuf_fd,
       .offset = offset,
       .length = length,
       .access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
   };
   write(cmd_fd, &cmd, sizeof(cmd));

8. Kernel EFA driver:
   a. Get dmabuf object: dma_buf_get(fd)
   b. Attach to dmabuf: dma_buf_attach(dmabuf, efa_dev)
   c. Map attachment: dma_buf_map_attachment(attachment, DMA_BIDIRECTIONAL)
      → Returns scatter-gather table with physical pages
   d. Create IOMMU mapping for each page
   e. Program EFA hardware with DMA addresses
   f. Generate lkey/rkey

9. Kernel returns to userspace
   resp.lkey = ...;
   resp.rkey = ...;

10. OFI Plugin caches MR
    nccl_ofi_mr_cache_insert_entry(cache, &dmabuf_key, mr_handle);

11. Future sends use cached MR
    fi_send(ep, gpu_ptr, size, fi_mr_desc(mr_handle), ...);
    // EFA DMA's directly from GPU memory!
```

## DMA-BUF Viability Checks

From `aws-ofi-nccl/src/nccl_ofi_dmabuf.cpp`:

```cpp
/**
 * Check if DMA-BUF is viable for this system
 * Returns true if all preconditions are met
 */
int nccl_ofi_dmabuf_viable()
{
    // 1. Check libfabric version (need 1.20+)
    if (!HAVE_DECL_FI_MR_DMABUF) {
        NCCL_OFI_TRACE("Will not use DMA-BUF, requires Libfabric 1.20+");
        return false;
    }

    // 2. Check if user disabled it
    if (ofi_nccl_disable_dmabuf()) {
        NCCL_OFI_TRACE("DMA-BUF explicitly disabled by user");
        return false;
    }

    // 3. Check if GPU supports dmabuf
#if HAVE_GPU
    if (!nccl_net_ofi_gpu_have_dma_buf_attr()) {
        NCCL_OFI_TRACE("GPU device does not support DMA-BUF");
        return false;
    }
#endif

    // 4. Check kernel version (need 5.12+)
    if (!kernel_version_rdma_dmabuf_ioctl_ok()) {
        NCCL_OFI_TRACE("Will not use DMA-BUF, kernel 5.12+ not found");
        return false;
    }

    return true;
}

/**
 * Check kernel version for RDMA dmabuf support
 * dmabuf rdma ioctls added in kernel 5.12
 * Commit: github.com/torvalds/linux/commit/bfe0cc6
 */
static bool kernel_version_rdma_dmabuf_ioctl_ok(void)
{
    struct utsname buf = {};
    int maj = 0, min = 0;

    if (uname(&buf) != 0) return false;

    if (sscanf(buf.release, "%d.%d", &maj, &min) != 2)
        return false;

    // Require kernel 5.12+
    return (maj == 5 && min >= 12) || (maj > 5);
}

/**
 * Check provider support AND firmware compatibility
 */
int nccl_ofi_dmabuf_viable_and_supported(struct fi_info *nic_prov)
{
    // Check provider has FI_HMEM and API version 1.20+
    bool dmabuf_support = ((nic_prov->caps & FI_HMEM) != 0) &&
        FI_VERSION_GE(nic_prov->fabric_attr->api_version, FI_VERSION(1, 20)) &&
        nccl_ofi_dmabuf_viable();

    if (dmabuf_support && strncmp("efa", nic_prov->fabric_attr->prov_name, 3) == 0) {
        // EFA Generation check (firmware bug in Gen 1-3)
        // Only Gen 4+ (0xefa3+) supports dmabuf reliably
        if (nic_prov->nic == NULL || nic_prov->nic->device_attr == NULL) {
            NCCL_OFI_TRACE("DMA-BUF disabled due to missing nic data");
            dmabuf_support = false;
        } else if (strcmp("0xefa0", nic_prov->nic->device_attr->device_id) == 0 ||
                   strcmp("0xefa1", nic_prov->nic->device_attr->device_id) == 0 ||
                   strcmp("0xefa2", nic_prov->nic->device_attr->device_id) == 0) {
            // Gen 1-3 have page merging issues with dmabuf
            NCCL_OFI_TRACE("DMA-BUF disabled due to EFA device id %s",
                          nic_prov->nic->device_attr->device_id);
            dmabuf_support = false;
        }
    }

    return dmabuf_support;
}
```

## EFA Hardware Generation Support

| EFA Generation | Device ID | DMA-BUF Support | Notes |
|----------------|-----------|-----------------|-------|
| Gen 1 | 0xefa0 | ❌ No | Firmware page merging issue |
| Gen 2 | 0xefa1 | ❌ No | Firmware page merging issue |
| Gen 3 | 0xefa2 | ❌ No | Firmware page merging issue |
| Gen 4+ | 0xefa3+ | ✅ Yes | Full dmabuf support |

**Page Merging Issue**: Early EFA generations + kernel dmabuf code (pre-kernel patch) → too many IOMMU page entries → communication failures.

**Kernel Patch**: https://web.git.kernel.org/pub/scm/linux/kernel/git/rdma/rdma.git/commit/?id=486055f5e09df9

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

### Example: IOMMU Entry Reduction

```
GPU Buffer: 4 GB (1M pages @ 4KB each)

Legacy Registration:
- GPU memory fragmented into 4KB pages
- 1,048,576 IOMMU entries
- Large memory overhead for page tables

DMA-BUF Registration (with page merging):
- Kernel merges contiguous pages
- ~4,096 IOMMU entries (2MB huge pages)
- 256x fewer entries!
- Lower overhead, better performance
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

# Check EFA generation
lspci -vv -s $(lspci | grep EFA | cut -d' ' -f1) | grep "Device ID"
# 0xefa3 or higher → DMA-BUF supported

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

```c
// Allocate GPU memory
void *gpu_ptr;
cudaMalloc(&gpu_ptr, size);

// Export as dmabuf (CUDA 11.7+)
cudaExternalMemoryHandleDesc desc = {};
desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
desc.size = size;

cudaExternalMemory_t ext_mem;
cudaImportExternalMemory(&ext_mem, &desc);

int dmabuf_fd = desc.handle.fd;  // This is the dmabuf FD
```

### OFI Plugin Registration

```cpp
// From aws-ofi-nccl (simplified)
int nccl_net_ofi_regmr_dmabuf(void *comm, void *data, int size,
                              int type, int dmabuf_fd, uint64_t offset,
                              void **mhandle)
{
    nccl_net_ofi_ep_t *ep = (nccl_net_ofi_ep_t *)comm;

    // Create dmabuf cache key
    nccl_ofi_mr_ckey_t ckey = nccl_ofi_mr_ckey_mk_dmabuf(
        dmabuf_fd, offset, size, data, ep);

    // Try cache first
    void *mr_handle = nccl_ofi_mr_cache_lookup_entry(ep->mr_cache, &ckey, true);
    if (mr_handle) {
        *mhandle = mr_handle;
        return 0;  // Cache hit!
    }

    // Cache miss - register with libfabric
    struct fi_mr_attr attr = {};
    attr.dmabuf = &ckey.fi_mr_dmabuf;
    attr.iov_count = 1;
    attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE;

    struct fid_mr *mr;
    fi_mr_regattr(ep->domain, &attr, FI_MR_DMABUF, &mr);

    // Cache it
    nccl_ofi_mr_cache_insert_entry(ep->mr_cache, &ckey, true, mr);

    *mhandle = mr;
    return 0;
}
```

### Kernel Registration (EFA Driver)

```c
// From amzn-drivers/kernel/linux/efa/src/efa_verbs.c (simplified)
int efa_reg_dmabuf_mr(struct ib_pd *ibpd, u64 offset, u64 length,
                      u64 dmabuf_fd, u64 virt_addr, int access_flags,
                      struct ib_mr **mr)
{
    struct efa_dev *dev = to_edev(ibpd->device);
    struct efa_mr *efa_mr;
    struct dma_buf *dmabuf;
    struct dma_buf_attachment *attachment;
    struct sg_table *sgt;

    // 1. Get dmabuf object from FD
    dmabuf = dma_buf_get(dmabuf_fd);
    if (IS_ERR(dmabuf))
        return PTR_ERR(dmabuf);

    // 2. Attach EFA device to dmabuf
    attachment = dma_buf_attach(dmabuf, &dev->pdev->dev);

    // 3. Map attachment (get physical pages)
    sgt = dma_buf_map_attachment(attachment, DMA_BIDIRECTIONAL);

    // 4. Create IOMMU mappings
    efa_mr->pbl = efa_create_page_list(dev, sgt, offset, length);

    // 5. Register with EFA hardware
    efa_mr->lkey = efa_create_mr_key(dev, efa_mr);
    efa_mr->rkey = efa_mr->lkey;  // On EFA, lkey == rkey

    // 6. Store dmabuf info for later cleanup
    efa_mr->dmabuf = dmabuf;
    efa_mr->attachment = attachment;
    efa_mr->sgt = sgt;

    *mr = &efa_mr->ibmr;
    return 0;
}

// Cleanup on deregister
void efa_dereg_dmabuf_mr(struct ib_mr *ibmr)
{
    struct efa_mr *efa_mr = to_emr(ibmr);

    // Unmap from IOMMU
    dma_buf_unmap_attachment(efa_mr->attachment, efa_mr->sgt,
                            DMA_BIDIRECTIONAL);

    // Detach from dmabuf
    dma_buf_detach(efa_mr->dmabuf, efa_mr->attachment);

    // Release dmabuf reference
    dma_buf_put(efa_mr->dmabuf);

    kfree(efa_mr);
}
```

## Page Merging in Detail

### Problem: Too Many IOMMU Entries

```
GPU Memory: 1 GB buffer
Without page merging:
    262,144 pages @ 4KB each
    → 262,144 IOMMU page table entries
    → ~1 MB of page table memory
    → Slow lookups, high overhead

With page merging:
    512 huge pages @ 2MB each
    → 512 IOMMU entries
    → ~2 KB of page table memory
    → 512x fewer entries!
```

### Kernel Page Merging (Fixed in Recent Kernels)

```c
// From linux/drivers/infiniband/core/umem_dmabuf.c (kernel 5.19+)

static int ib_umem_dmabuf_map_pages(struct ib_umem_dmabuf *umem)
{
    struct sg_table *sgt;
    struct scatterlist *sg;
    int i;

    sgt = dma_buf_map_attachment(umem->attach, DMA_BIDIRECTIONAL);

    // Merge contiguous pages (the fix!)
    sg_alloc_table_from_pages_segment(
        &umem->sgt,
        sgt->sgl,
        sgt->orig_nents,
        0,  // offset
        umem->length,
        SZ_2M,  // Merge into 2MB chunks where possible
        GFP_KERNEL);

    return 0;
}
```

**Before Fix**: Each 4KB page → separate IOMMU entry
**After Fix**: Contiguous 4KB pages merged into 2MB entries

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
| **EFA requirement** | Gen 4+ (0xefa3+) |
| **GPU requirement** | CUDA 11.7+, ROCm 5.0+, or any dmabuf-capable GPU |
| **API** | `fi_mr_regattr(..., FI_MR_DMABUF)` |
| **Key benefit** | Vendor-neutral, better page merging |
| **IOMMU reduction** | Up to 256x fewer entries (4KB → 2MB pages) |

**Key Takeaways**:
1. DMA-BUF is the **modern, standard way** to register GPU memory for RDMA
2. Requires kernel 5.12+, libfabric 1.20+, EFA Gen 4+
3. **Vendor-neutral**: Works with NVIDIA, AMD, Intel GPUs
4. **Better IOMMU efficiency**: Page merging can significantly reduce IOMMU entries (estimated 100-500x reduction)
5. Same performance as legacy GDR, but cleaner and more maintainable
6. Registration is still expensive (100-500 μs) - **caching is critical**

**Related Documentation**:
- [mr-cache-implementation.md](mr-cache-implementation.md) - MR cache (works with dmabuf keys)
- [rdma-memreg.md](rdma-memreg.md) - General memory registration concepts
- [rdma-core-and-verbs.md](rdma-core-and-verbs.md) - rdma-core API that dmabuf uses
- [kernel-efa-driver.md](kernel-efa-driver.md) - Kernel driver dmabuf implementation

---

## Code References

### Structures

**Linux Kernel (DMA-BUF subsystem)**:
- `struct dma_buf` ([linux/include/linux/dma-buf.h:294-400](https://github.com/torvalds/linux/blob/e84d960/include/linux/dma-buf.h#L294-L400))
- `struct dma_buf_attachment` (referenced, defined in same file)
- `struct sg_table` (referenced, defined in linux/include/linux/scatterlist.h)

**libfabric (OFI)**:
- `struct fi_mr_dmabuf` ([include/rdma/fi_domain.h:151-156](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L151-L156))

### Functions

**Linux Kernel (DMA-BUF API)** - Referenced but not linked (kernel API):
- `dma_buf_get(int fd)` - Get dma_buf from file descriptor
- `dma_buf_put(struct dma_buf *)` - Release dma_buf reference
- `dma_buf_attach(struct dma_buf *, struct device *)` - Attach device to dmabuf
- `dma_buf_detach(struct dma_buf *, struct dma_buf_attachment *)` - Detach device
- `dma_buf_map_attachment(struct dma_buf_attachment *)` - Map attachment for DMA

**rdma-core (libibverbs)** - Referenced but not linked (standard RDMA API):
- `ibv_reg_dmabuf_mr()` - Register dmabuf as memory region

**libfabric (OFI)** - Referenced but not linked (standard libfabric API):
- `fi_mr_regattr(struct fid_domain *, struct fi_mr_attr *, flags, struct fid_mr **)` - Register MR with dmabuf

**CUDA Driver API (External - NVIDIA)** - Referenced but not linked:
- `cuMemGetHandleForAddressRange()` - Export CUDA memory as dmabuf (CUDA 11.7+)

### Total Code References
- **1 struct** from Linux kernel (dma_buf)
- **1 struct** from libfabric (fi_mr_dmabuf)
- **5 kernel functions** (DMA-BUF API, external)
- **2 userspace functions** (rdma-core/libfabric, external)
- **1 CUDA function** (external NVIDIA API)

All Linux kernel references link to commit `e84d960` in the [torvalds/linux](https://github.com/torvalds/linux) repository.
All libfabric references link to commit `6b9e629` in the [ofiwg/libfabric](https://github.com/ofiwg/libfabric) repository.
