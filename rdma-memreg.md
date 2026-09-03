# RDMA Operations and Memory Registration

> **Source lookups:** this document explains mechanism and records defaults, corrections and history. For current function bodies, call graphs and blast radius, query the code graph (`codegraph explore <symbol>`) rather than trusting a pasted copy here.

## RDMA Overview

RDMA (Remote Direct Memory Access) enables direct memory access from one computer's memory to another's without involving the operating system or CPU in the data path.

```
Traditional Network I/O:
┌─────────────────────────────────────────────────┐
│  App ─→ Copy to kernel ─→ Network stack ─→ NIC │
│  High CPU usage, multiple copies                │
└─────────────────────────────────────────────────┘

RDMA:
┌─────────────────────────────────────────────────┐
│  App ─→ NIC (direct DMA) ─→ Network             │
│  Zero CPU, zero copy                            │
└─────────────────────────────────────────────────┘
```

## EFA and RDMA

EFA provides **RDMA-like** capabilities with some differences from traditional InfiniBand RDMA:

```
InfiniBand RDMA               EFA "RDMA"
─────────────────────────────────────────────
Connected (RC)                Connectionless (SRD)
Strict ordering               Relaxed ordering
Full RDMA semantics           Limited RDMA semantics
Comprehensive atomics         Basic atomics
```

**EFA RDMA Operations:**
- ✅ RDMA Write: Supported, optimized
- ✅ RDMA Read: Supported
- ⚠️ Atomics: Limited support
- ❌ Send with Immediate: Not fully supported

## Memory Registration

### Why Register Memory?

For RDMA, memory must be registered (pinned) with the NIC:

```
Virtual Memory (App)
┌──────────────────┐
│  Buffer (4 KB)   │  Virtual address: 0x7fff1234
└────────┬─────────┘
         │ OS translation (page table)
         ▼
Physical Memory
┌──────────────────┐
│  Pages scattered │  Physical addresses: 0x8000, 0x9000, ...
└────────┬─────────┘
         │ IOMMU translation
         ▼
DMA Address (NIC sees)
┌──────────────────┐
│  Contiguous view │  DMA addresses: 0x10000, 0x11000, ...
└──────────────────┘
```

**Registration Process:**
1. **Pin pages**: Lock physical pages in RAM (prevent swapping)
2. **Create IOMMU mapping**: Translate physical → DMA addresses
3. **Generate keys**: Create protection keys for remote access
4. **Store metadata**: Track buffer info in NIC

**Without registration:**
- NIC doesn't know physical addresses
- Pages can be swapped out
- Security issues (NIC could access any memory)

### Registration Cost

```
Operation                   Time (typical)
──────────────────────────────────────────
Register 4 KB              ~50-100 μs
Register 1 MB              ~100-500 μs
Register 100 MB            ~500-2000 μs
Deregister                 ~50% of register time
Cache hit (lookup)         ~0.1-1 μs
```

**Why expensive:**
- Page table walk
- TLB flush
- IOMMU programming
- Lock kernel data structures
- Pin potentially many pages

**Implication:**
- **Registering on every transfer kills performance**
- **Caching is absolutely critical**

### EFA Kernel Driver r3.3.0 Registration Changes

The EFA kernel driver r3.3.0 (amzn-drivers) made several registration-path
changes that directly affect registration cost and the number of MRs a device
can hold:

- **>4GB MR page size support.** The driver can now register memory using a
  page size large enough that a single MR spans more than 4GB
  (`amzn-drivers/kernel/linux/efa/RELEASENOTES.md:15`,
  "Add driver support for >4GB MR page size"; commit **bf83e44**). Fewer, larger
  MRs mean fewer registrations and fewer PBL entries per byte of memory.
- **Extended page-shift field in MR registration.** The registration parameters
  gained an explicit `u8 page_shift`
  ([efa_com_cmd.h:206](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_com_cmd.h))
  computed from the MR's chosen page size rather than being hard-wired to the
  host `PAGE_SHIFT`
  ([efa_data_verbs.c:198-210](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_data_verbs.c),
  `page_shift = order_base_2(mr->ibmr.page_size)`; commit **a1e35dc**). This is
  what lets the >4GB page size be expressed to the device.
- **PBL chunk-length computation fix.** The physical buffer list (PBL) indirect
  chunk list sizing was corrected
  ([efa_verbs.c:2237-2410](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_verbs.c),
  `pbl_chunk_list_create`; commit **4eec543**), preventing miscomputed chunk
  lengths for large registrations.
- **Address-handle (AH) cache via rhashtable.** The driver now caches AH device
  objects and reuses them instead of recreating an AH per peer
  ([efa_ah_cache.c](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_ah_cache.c),
  [efa_ah_cache.h](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_ah_cache.h);
  commits **f95ece8**, **32e9eb1**). The cache is a kernel `rhashtable` keyed by
  `{pd, gid}`
  ([efa_com_cmd.c:348-422](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_com_cmd.c))
  with `get`/`put`/`lookup` refcounted entries. This does not speed up *memory*
  registration, but it removes redundant AH creation cost on the connection path
  when many peers share a protection domain.

> These are kernel-driver changes; they take effect when the r3.3.0 EFA driver
> is loaded, independent of the libfabric/plugin version.

## Memory Registration in the Stack

### Application Level (NCCL)

**NCCL's perspective:**
- Buffers are user-provided and may change between calls
- Must register on-demand
- Cache registrations when possible

> Source: `src/nccl_ofi_rdma.cpp` — registration entry points `reg_mr`, `reg_mr_on_device`. Use `codegraph explore reg_mr` for the current body and its callers.

### OFI Plugin Level

On a cache miss the plugin calls `fi_mr_reg()` (or `fi_mr_regattr()` for DMA-BUF) with access flags `FI_SEND | FI_RECV | FI_REMOTE_READ | FI_REMOTE_WRITE`. GPU memory is registered with `FI_HMEM_CUDA` (or `FI_HMEM_NEURON`). On a cache hit the refcount is incremented and the existing handle returned.

> Source: `src/nccl_ofi_rdma.cpp` — `nccl_net_ofi_rdma_domain_t::reg_mr()`. Use `codegraph explore reg_mr` for details.

### Libfabric Level

The EFA provider maintains its own internal MR cache. On a miss it calls `ibv_reg_mr()` (host) or `ibv_reg_dmabuf_mr()` (GPU via DMA-BUF) into rdma-core.

> Source: libfabric EFA provider `prov/efa/`. Use `codegraph explore efa_mr_cache_find` for provider-level caching.

### Driver Level (ibverbs → kernel)

The kernel path pins pages (`ib_umem_get()`), creates IOMMU mappings (`dma_map_sg()`), programs the EFA device, and generates lkey/rkey.

**Kernel Operations:**
1. **Pin pages**: `get_user_pages()` / `ib_umem_get()`
2. **Create DMA mapping**: `dma_map_sg()`
3. **Program IOMMU**: Hardware-specific
4. **Register with NIC**: Device command
5. **Generate keys**: lkey (local) and rkey (remote)

> Source: `amzn-drivers/kernel/linux/efa/src/efa_verbs.c` — `efa_reg_mr()`, `efa_reg_dmabuf_mr()`. Use `codegraph explore efa_reg_mr` for the current bodies.

## GPU Memory Registration

### Traditional Method (Peer Direct)

Older approach, now deprecated: NVIDIA GPUDirect RDMA via peer-direct required an out-of-tree kernel module, was complex to maintain, and had limited compatibility.

### Modern Method (dmabuf)

Current approach using Linux DMA buffer (kernel 5.12+). GPU memory is exported as a dmabuf file descriptor and registered via `ibv_reg_dmabuf_mr()`. See [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md) for full details.

```
CUDA Driver
    ↓ Export as dmabuf
DMA-BUF (kernel)
    ↓ Share file descriptor
ibverbs
    ↓ Import dmabuf
EFA Driver
    ↓ Pin GPU pages, create IOMMU mapping
GPU Memory → NIC DMA
```

## Memory Registration Cache

### Why Cache?

```
Without cache:
  AllReduce call 1: Register (500 μs) + Transfer (100 μs) = 600 μs
  AllReduce call 2: Register (500 μs) + Transfer (100 μs) = 600 μs
  AllReduce call 3: Register (500 μs) + Transfer (100 μs) = 600 μs

With cache:
  AllReduce call 1: Register (500 μs) + Transfer (100 μs) = 600 μs
  AllReduce call 2: Cache hit (1 μs) + Transfer (100 μs) = 101 μs
  AllReduce call 3: Cache hit (1 μs) + Transfer (100 μs) = 101 μs
```

**Approximately 6x speedup for subsequent calls!**

See [mr-cache-implementation.md](mr-cache-implementation.md) for the full cache implementation details (data structures, API, RAII lifecycle).

### Cache Invalidation

**Challenge**: What if registered memory is freed or modified? A stale cache entry maps the old physical pages to a recycled virtual address.

**Solutions:**

#### Memory Hooks (memhooks)

LD_PRELOAD intercepts libc `free`/`mmap`/`munmap` and invalidates the MR cache on every deallocation.

```bash
FI_MR_CACHE_MONITOR=memhooks
```

#### Userfaultfd

Linux `userfaultfd` monitors memory ranges in-kernel. On `UFFD_EVENT_UNMAP` the cache is invalidated. More reliable and lower overhead than hooks.

```bash
FI_MR_CACHE_MONITOR=userfaultfd
```

### Cache Configuration

```bash
# Maximum cache size
FI_MR_CACHE_MAX_SIZE=unlimited  # or byte limit

# Maximum number of entries
FI_MR_CACHE_MAX_COUNT=unlimited  # or count limit

# Cache monitor method
FI_MR_CACHE_MONITOR=memhooks    # or userfaultfd, disabled

# NCCL-specific (plugin level)
# There is no NCCL_IB_MR_CACHE variable, and NCCL_IB_* knobs do not apply to EFA.
# MR caching on this stack is controlled at two levels:
OFI_NCCL_MR_CACHE_DISABLE=0     # plugin-level cache (default 0 = cache enabled)
FI_EFA_MR_CACHE_ENABLE=1        # provider-level cache (default 1)

# EFA-specific size/count limits
FI_EFA_MR_MAX_CACHED_SIZE=<bytes>   # max region size to cache
FI_EFA_MR_MAX_CACHED_COUNT=<count>  # max number of cached regions
```

## RDMA Operations in Detail

### RDMA Write

```
Sender                          Receiver
──────                          ────────
1. Have remote_addr and remote_key (from prior exchange)
2. fi_write(local_buf, ..., remote_addr, remote_key)
3. NIC DMA: local_buf ─────────→ remote_addr (DMA)
4. Completion in sender CQ      (Silent on receiver)
```

**Key Points:**
- **One-sided**: Receiver CPU not involved
- **Silent**: No completion on receiver
- **Requires key exchange**: Out-of-band communication
- **Ordering**: May need fence/flush for visibility

### RDMA Read

```
Initiator                       Target
─────────                       ──────
1. fi_read(local_buf, ..., remote_addr, remote_key)
2. NIC DMA: remote_addr ───────→ local_buf (DMA)
3. Completion in initiator CQ   (Silent on target)
```

**Use in NCCL:**
- Flush operations (dummy read for ordering)
- Some algorithm variants

### Flush/Fence

RDMA writes are asynchronous and may be reordered. NCCL's `iflush` issues a dummy `fi_read` to force prior writes to be visible.

```
Problem:
  fi_write(buf1)  ─────────────→ May arrive second
  fi_write(buf2)  ─────────────→ May arrive first

Solution:
  fi_write(buf1)
  fi_write(buf2)
  fi_read(dummy)  ─────────────→ Forces ordering
```

> Source: `src/nccl_ofi_rdma.cpp` — `nccl_net_ofi_iflush()`. Use `codegraph explore iflush` for the current implementation.

## Performance Optimization

### Reduce Registration Overhead

- **Register once, reuse**: Pre-register buffers and pass the MR descriptor on every send/recv.
- **One large buffer, partition**: Register a single contiguous allocation and carve it into chunks rather than registering many small allocations.
- **Pre-register at init**: Register hot buffers during initialization so no registration cost is on the data path.

### Monitor Cache Performance

```bash
# Enable libfabric logging
export FI_LOG_LEVEL=info
export FI_LOG_PROV=efa

# Run application
./nccl_app

# Look for:
# "mr cache hit" vs "mr cache miss"
# High miss rate = poor performance
```

## Common Issues and Debugging

### Stale Cache Entries

**Symptom**: Crashes, data corruption

**Cause**: Cache not invalidated when memory freed

**Solution:**
```bash
# Enable memory monitoring
FI_MR_CACHE_MONITOR=memhooks  # or userfaultfd
```

### Registration Failures

**Symptom**: `fi_mr_reg()` returns `-FI_ENOMEM`

**Causes:**
- Too much memory registered (IOMMU limit)
- Resource exhaustion

**Solutions:**
```bash
# Increase limits
echo unlimited > /sys/module/ib_core/parameters/reg_mem_limit

# Reduce cache size
FI_MR_CACHE_MAX_SIZE=8G
```

### Slow Performance

**Symptom**: Low bandwidth, high latency

**Cause**: Registration on every transfer

**Debug:**
```bash
# Profile registration time
perf record -e 'syscalls:sys_enter_ioctl' ./app
perf report

# Look for many ibv_reg_mr calls
```

**Solution**: Ensure cache is enabled and working

### Registration-Path Fixes (aws-ofi-nccl, 2026)

Two recent correctness fixes touch the registration path and are worth knowing
when debugging hangs or leaks:

- **`gin: Fix regMrSym hang caused by duplicate detection`** (commit
  **4b778f3**). Symmetric memory registration (`regMrSym`) on the GIN path could
  hang when duplicate-detection logic mishandled a repeated registration.
- **`rdma: Fix memory leak of rx_buff_regmr_ctx in fini_rx_buffers()`** (commit
  **aaef93d**). The RDMA transport leaked the `rx_buff_regmr_ctx` context when
  tearing down receive buffers; the deregistration context is now freed in
  `fini_rx_buffers()`.

If you are on a build predating these commits and see a `regMrSym` hang or a
slow memory leak proportional to connection churn, these are the likely
culprits.

## Summary

**Key Concepts:**
1. **Memory registration** is expensive (~100-500 μs)
2. **Caching is critical** for performance
3. **GPU memory** uses dmabuf method (modern)
4. **RDMA write** for zero-copy transfers
5. **Key exchange** needed for RDMA ops

**Cache Layers:**
```
OFI Plugin Cache
    ↓
Libfabric Provider Cache
    ↓
Driver (implicit tracking)
```

**Optimization Checklist:**
- [ ] Enable MR cache (default: on)
- [ ] Use memory hooks/userfaultfd
- [ ] Pre-register hot buffers
- [ ] Reuse buffers when possible
- [ ] Monitor cache hit rate
- [ ] Tune cache size limits

**Performance Impact:**
- **Without cache**: ~500 μs overhead per transfer
- **With cache**: < 1 μs overhead per transfer
- **Speedup**: Estimated 100-500x for repeated transfers!

**Next**: EFA driver architecture and capabilities.
