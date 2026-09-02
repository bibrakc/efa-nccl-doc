# RDMA Operations and Memory Registration

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

```c
// User provides GPU memory buffer
float* gpu_buffer;
cudaMalloc(&gpu_buffer, size);

// NCCL internally registers via plugin
ncclAllReduce(gpu_buffer, gpu_buffer, count,
              ncclFloat, ncclSum, comm, stream);
```

**NCCL's perspective:**
- Buffers are user-provided
- May change between calls
- Must register on-demand
- Cache registrations when possible

### OFI Plugin Level

```c
ncclResult_t nccl_net_ofi_regMr(void* comm, void* data,
                                int size, int type,
                                void** mhandle)
{
  // Check cache
  struct mr_cache_entry* entry = mr_cache_lookup(data, size);

  if (entry) {
    // Cache hit!
    entry->refcount++;
    *mhandle = entry->mr;
    return ncclSuccess;
  }

  // Cache miss, register new
  struct fid_mr* mr;
  uint64_t access = FI_SEND | FI_RECV |
                    FI_REMOTE_READ | FI_REMOTE_WRITE;

  if (type == NCCL_PTR_CUDA) {
    // GPU memory
    ret = fi_mr_reg(domain, data, size, access,
                    0, 0, FI_HMEM_CUDA, &mr, NULL);
  } else {
    // Host memory
    ret = fi_mr_reg(domain, data, size, access,
                    0, 0, 0, &mr, NULL);
  }

  // Add to cache
  mr_cache_insert(data, size, mr);

  *mhandle = mr;
  return ncclSuccess;
}
```

### Libfabric Level

```c
int fi_mr_reg(struct fid_domain *domain,
              const void *buf, size_t len,
              uint64_t access, uint64_t offset,
              uint64_t requested_key, uint64_t flags,
              struct fid_mr **mr, void *context)
{
  // EFA provider implementation
  struct efa_domain *efa_domain = container_of(domain, ...);

  // Check provider's internal cache
  struct fid_mr *cached_mr = efa_mr_cache_find(buf, len);
  if (cached_mr) {
    *mr = cached_mr;
    return 0;
  }

  // Not cached, call ibverbs
  struct ibv_mr *ibv_mr;

  if (flags & FI_HMEM_CUDA) {
    // GPU memory registration
    ibv_mr = ibv_reg_dmabuf_mr(efa_domain->pd, offset, len,
                               (uint64_t)buf, dmabuf_fd,
                               access_flags);
  } else {
    // Host memory registration
    ibv_mr = ibv_reg_mr(efa_domain->pd, (void *)buf, len,
                        access_flags);
  }

  // Wrap in libfabric MR
  struct efa_mr *efa_mr = calloc(1, sizeof(*efa_mr));
  efa_mr->ibv_mr = ibv_mr;
  efa_mr->mr_fid.fid.fclass = FI_CLASS_MR;
  // ...

  // Add to provider cache
  efa_mr_cache_insert(efa_mr);

  *mr = &efa_mr->mr_fid;
  return 0;
}
```

### Driver Level (ibverbs → kernel)

```c
// User space (libibverbs)
struct ibv_mr* ibv_reg_mr(struct ibv_pd *pd,
                          void *addr, size_t length,
                          int access)
{
  struct ibv_reg_mr_cmd cmd = {
    .pd_handle = pd->handle,
    .addr = (uint64_t)addr,
    .length = length,
    .access_flags = access,
  };

  struct ibv_reg_mr_resp resp;

  // Ioctl to kernel
  ioctl(pd->context->cmd_fd, IBV_CMD_REG_MR, &cmd, &resp);

  struct ibv_mr *mr = malloc(sizeof(*mr));
  mr->lkey = resp.lkey;
  mr->rkey = resp.rkey;
  mr->addr = addr;
  mr->length = length;

  return mr;
}
```

```c
// Kernel space (EFA driver)
int efa_reg_mr(struct ib_pd *ibpd,
               struct ib_mr *ibmr,
               u64 start, u64 length,
               u64 virt_addr, int access_flags)
{
  struct efa_dev *dev = to_edev(ibpd->device);
  struct efa_mr *mr = to_emr(ibmr);

  // Get user pages (pin them)
  npages = ib_umem_num_pages(mr->umem);
  mr->umem = ib_umem_get(ibpd->device, start, length, access_flags);

  // Create DMA mapping (IOMMU)
  dma_map_sg(dev->dma_device, mr->umem->sg_head.sgl,
             mr->umem->nmap, DMA_BIDIRECTIONAL);

  // Register with EFA device
  struct efa_com_reg_mr_params params = {
    .pd = to_epd(ibpd)->pdn,
    .iova = virt_addr,
    .mr_length = length,
    // r3.3.0: page_shift is derived from the MR's chosen page size
    // (order_base_2(mr->ibmr.page_size)), not hard-wired to host PAGE_SHIFT.
    // This is what enables >4GB single-MR page sizes.
    .page_shift = order_base_2(mr->ibmr.page_size),
    .page_num = npages,
  };

  err = efa_com_register_mr(dev->edev, &params, &mr->key);

  // Generate keys
  mr->lkey = mr->key;
  mr->rkey = mr->key;

  return 0;
}
```

**Kernel Operations:**
1. **Pin pages**: `get_user_pages()` / `ib_umem_get()`
2. **Create DMA mapping**: `dma_map_sg()`
3. **Program IOMMU**: Hardware-specific
4. **Register with NIC**: Device command
5. **Generate keys**: lkey (local) and rkey (remote)

## GPU Memory Registration

### Traditional Method (Peer Direct)

Older approach, now deprecated:

```c
// NVIDIA GPUDirect RDMA via peer-direct
1. Get GPU BAR (Base Address Register)
2. Create peer mapping in driver
3. Register GPU physical address
```

**Issues:**
- Kernel module required
- Complex to maintain
- Limited compatibility

### Modern Method (dmabuf)

Current approach using Linux DMA buffer:

```c
// GPU memory registration via dmabuf
int register_gpu_memory(void *gpu_ptr, size_t len) {
  // 1. Export GPU memory as dmabuf
  cudaExternalMemoryHandleDesc desc = {
    .type = cudaExternalMemoryHandleTypeOpaqueFd,
    .size = len,
  };

  cudaExternalMemory_t ext_mem;
  cudaImportExternalMemory(&ext_mem, &desc);

  int dmabuf_fd;
  cudaExternalMemoryGetMappedBuffer(&dmabuf_fd, ext_mem, ...);

  // 2. Register dmabuf with ibverbs
  struct ibv_mr *mr;
  mr = ibv_reg_dmabuf_mr(pd, 0, len,
                         (uint64_t)gpu_ptr, dmabuf_fd,
                         IBV_ACCESS_LOCAL_WRITE |
                         IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_WRITE);

  close(dmabuf_fd);

  return mr;
}
```

**Benefits:**
- Standard Linux interface
- Better compatibility
- Cleaner implementation
- Required kernel 5.12+

**Under the hood:**
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

### Cache Structure

**`struct mr_cache`** - Memory registration cache (conceptual structure showing cache organization):

```c
struct mr_cache {
  // Hash table or tree
  struct mr_entry {
    void *addr;              // Buffer address
    size_t length;           // Buffer size
    struct fid_mr *mr;       // Registered MR
    int refcount;            // Reference count
    uint64_t last_used;      // LRU timestamp
    struct mr_entry *next;   // Hash chain
  } *entries[CACHE_SIZE];

  pthread_mutex_t lock;      // Protect cache

  // Statistics
  uint64_t hits;
  uint64_t misses;
  uint64_t evictions;
};
```

### Cache Lookup

```c
struct fid_mr* mr_cache_lookup(void *addr, size_t len) {
  // Hash by address
  uint32_t hash = hash_function(addr) % CACHE_SIZE;

  pthread_mutex_lock(&cache.lock);

  struct mr_entry *entry = cache.entries[hash];
  while (entry) {
    // Exact match required
    if (entry->addr == addr && entry->length == len) {
      // Hit!
      entry->refcount++;
      entry->last_used = get_timestamp();
      cache.hits++;

      pthread_mutex_unlock(&cache.lock);
      return entry->mr;
    }

    entry = entry->next;
  }

  // Miss
  cache.misses++;
  pthread_mutex_unlock(&cache.lock);
  return NULL;
}
```

### Cache Invalidation

**Challenge**: What if registered memory is freed or modified?

```
Timeline:
1. Register buffer A (0x1000) ─→ Cache
2. Free buffer A
3. Malloc buffer B ─→ Gets same address 0x1000!
4. Cache hit on 0x1000 ─→ WRONG! (stale registration)
```

**Solutions:**

#### Memory Hooks (memhooks)

```c
// Override malloc/free
void free(void *ptr) {
  // Invalidate MR cache entry
  mr_cache_invalidate(ptr);

  // Call real free
  real_free(ptr);
}

void* mmap(void *addr, size_t length, ...) {
  void *result = real_mmap(addr, length, ...);

  // If unmapping, invalidate cache
  if (flags & MAP_FIXED) {
    mr_cache_invalidate(addr);
  }

  return result;
}
```

**Implementation:**
- LD_PRELOAD intercepts libc functions
- Tracks all memory operations
- Invalidates cache on free/munmap

**Configuration:**
```bash
FI_MR_CACHE_MONITOR=memhooks
```

#### Userfaultfd

```c
// Use Linux userfaultfd to monitor memory
int uffd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK);

// Register memory range
struct uffdio_register uffdio_register = {
  .range = { .start = addr, .len = len },
  .mode = UFFDIO_REGISTER_MODE_MISSING |
          UFFDIO_REGISTER_MODE_WP,
};
ioctl(uffd, UFFDIO_REGISTER, &uffdio_register);

// Monitor thread
while (1) {
  struct uffd_msg msg;
  read(uffd, &msg, sizeof(msg));

  if (msg.event == UFFD_EVENT_UNMAP) {
    // Memory unmapped, invalidate cache
    mr_cache_invalidate(msg.arg.remove.start);
  }
}
```

**Benefits:**
- Kernel-supported
- More reliable than hooks
- Lower overhead

**Configuration:**
```bash
FI_MR_CACHE_MONITOR=userfaultfd
```

#### Reference Counting

**`struct mr_entry`** - Cache entry with reference counting (conceptual):

```c
struct mr_entry {
  int refcount;
  // ...
};

// On regMr
entry->refcount++;

// On deregMr
entry->refcount--;
if (entry->refcount == 0) {
  // Safe to evict
}
```

**Prevents eviction of in-use registrations**

### Cache Eviction

When cache is full:

```c
void mr_cache_evict() {
  struct mr_entry *victim = NULL;
  uint64_t oldest = UINT64_MAX;

  // LRU eviction
  for (int i = 0; i < CACHE_SIZE; i++) {
    struct mr_entry *entry = cache.entries[i];
    while (entry) {
      if (entry->refcount == 0 &&
          entry->last_used < oldest) {
        oldest = entry->last_used;
        victim = entry;
      }
      entry = entry->next;
    }
  }

  if (victim) {
    // Deregister
    fi_close(&victim->mr->fid);

    // Remove from cache
    remove_entry(victim);
    free(victim);

    cache.evictions++;
  }
}
```

**Eviction Policies:**
- **LRU (Least Recently Used)**: Evict oldest
- **LFU (Least Frequently Used)**: Evict least used
- **Size-aware**: Evict largest first

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
```

## RDMA Operations in Detail

### RDMA Write

```c
// Sender
ssize_t fi_write(struct fid_ep *ep,
                 const void *buf, size_t len,
                 void *desc,
                 fi_addr_t dest_addr,
                 uint64_t remote_addr,
                 uint64_t remote_key,
                 void *context);
```

**Flow:**
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

#### Key Exchange Protocol

```c
// Setup phase
void exchange_keys(struct connection *conn) {
  // 1. Register local receive buffer
  fi_mr_reg(domain, my_recv_buf, size, FI_REMOTE_WRITE, ...);

  uint64_t my_addr = (uint64_t)my_recv_buf;
  uint64_t my_key = fi_mr_key(mr);

  // 2. Send to peer (control message)
  struct key_exchange_msg msg = {
    .addr = my_addr,
    .key = my_key,
  };
  fi_send(ep, &msg, sizeof(msg), ...);

  // 3. Receive from peer
  struct key_exchange_msg peer_msg;
  fi_recv(ep, &peer_msg, sizeof(peer_msg), ...);

  // 4. Store peer's info
  conn->remote_addr = peer_msg.addr;
  conn->remote_key = peer_msg.key;
}

// Data phase
void rdma_write_to_peer(struct connection *conn,
                        void *local_buf, size_t len) {
  // Use stored remote_addr and remote_key
  fi_write(ep, local_buf, len, desc,
           conn->dest_addr,
           conn->remote_addr,
           conn->remote_key,
           context);
}
```

### RDMA Read

```c
ssize_t fi_read(struct fid_ep *ep,
                void *buf, size_t len,
                void *desc,
                fi_addr_t src_addr,
                uint64_t remote_addr,
                uint64_t remote_key,
                void *context);
```

**Flow:**
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

RDMA writes are asynchronous and may be reordered:

```
Problem:
  fi_write(buf1)  ─────────────→ May arrive second
  fi_write(buf2)  ─────────────→ May arrive first

Solution:
  fi_write(buf1)
  fi_write(buf2)
  fi_read(dummy)  ─────────────→ Forces ordering
```

**EFA Flush:**
```c
// NCCL's iflush implementation in OFI plugin
ncclResult_t nccl_net_ofi_iflush(...) {
  // Issue dummy read to ensure prior writes are visible
  fi_read(ep, dummy_buf, 0, NULL,
          remote_addr, 0, 0, context);

  // Completion of read guarantees writes are visible
}
```

## Performance Optimization

### Reduce Registration Overhead

```c
// Bad: Register on every transfer
for (int i = 0; i < 1000; i++) {
  fi_mr_reg(domain, buf, size, ...);  // 500 μs each!
  fi_send(ep, buf, size, ...);
  fi_close(&mr->fid);
}
// Total: ~500 ms just for registration!

// Good: Register once, reuse
fi_mr_reg(domain, buf, size, &mr, ...);  // 500 μs once
for (int i = 0; i < 1000; i++) {
  fi_send(ep, buf, size, fi_mr_desc(mr), ...);
}
fi_close(&mr->fid);
// Total: ~500 μs + transfer time
```

### Use Buffers Wisely

```c
// Bad: Many small buffers
for (int i = 0; i < 100; i++) {
  void *buf = malloc(4096);
  fi_mr_reg(..., buf, 4096, ...);  // 100 registrations!
}

// Good: One large buffer, partition
void *buf = malloc(100 * 4096);
fi_mr_reg(..., buf, 100 * 4096, ...);  // 1 registration

// Use chunks
for (int i = 0; i < 100; i++) {
  void *chunk = buf + i * 4096;
  fi_send(ep, chunk, 4096, ...);
}
```

### Pre-register

```c
// At initialization
void init() {
  // Pre-register common buffers
  for (int i = 0; i < NUM_BUFFERS; i++) {
    cudaMalloc(&buffers[i], BUFFER_SIZE);
    fi_mr_reg(..., buffers[i], BUFFER_SIZE, &mrs[i], ...);
  }
}

// At runtime (no registration overhead)
void transfer() {
  fi_send(ep, buffers[0], size, fi_mr_desc(mrs[0]), ...);
}
```

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
