# Memory Registration Cache Implementation

## Overview

The OFI NCCL plugin implements a **custom memory registration (MR) cache** to avoid expensive re-registration of memory regions. This is one of the most critical performance optimizations in the plugin.

**Location**:
- Header: [include/nccl_ofi_mr.h](../aws-ofi-nccl/include/nccl_ofi_mr.h)
- Implementation: [src/nccl_ofi_mr.cpp](../aws-ofi-nccl/src/nccl_ofi_mr.cpp)

## Why MR Cache Matters

**Performance Impact** (estimated):
```
Without cache (every operation registers memory):
  fi_mr_reg()    → 100-500 μs  // ([include/rdma/fi_domain.h:413](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L413))
  fi_send()      → 10-20 μs
  fi_mr_dereg()  → 50-200 μs
  Total: ~160-720 μs per operation

With cache (first operation registers, rest hit cache):
  First call:
    fi_mr_reg()       → ~500 μs (one time)
    cache_insert()    → ~1 μs
    fi_send()         → ~20 μs
    Total: ~521 μs

  Subsequent calls:
    cache_lookup()    → ~0.5 μs (fast!)
    fi_send()         → ~20 μs
    Total: ~20.5 μs
```

**Speedup**: Approximately **25x faster** for cache hits (~20.5 μs vs ~521 μs)

**Target Hit Rate**: >99% in steady state

## Data Structures

### Cache Entry

```c
/**
 * A memory registration cache entry
 * Stores one registered memory region with reference counting
 */
typedef struct nccl_ofi_reg_entry ([include/nccl_ofi_mr.h:186-192](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/include/nccl_ofi_mr.h#L186-L192)) {
    uintptr_t addr;         // Page-aligned base address
    size_t pages;           // Number of pages covered
    int refcnt;             // Reference count (how many users)
    void *handle;           // fi_mr handle from libfabric
    nccl_net_ofi_ep_t *ep;  // Endpoint that owns this MR
} nccl_ofi_reg_entry_t;
```

**Key Design Decisions**:
- **Page-aligned**: `addr` is always aligned to `system_page_size` (4KB default)
- **Page counts**: `pages` stores number of 4KB pages, not byte size
- **Reference counting**: Enables safe sharing - entry deleted only when refcnt reaches 0
- **Per-endpoint**: Different endpoints can have different MRs for same address

### Cache Structure

```c
/**
 * Device-specific memory registration cache
 * Implemented as a sorted array (by address) for fast lookup
 */
typedef struct nccl_ofi_mr_cache ([include/nccl_ofi_mr.h:197-205](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/include/nccl_ofi_mr.h#L197-L205)) {
    nccl_ofi_reg_entry_t **slots;  // Sorted array of entries
    size_t system_page_size;       // Page size (typically 4096)
    size_t size;                   // Total capacity (grows 2x on demand)
    size_t used;                   // Number of entries currently in cache
    uint32_t hit_count;            // Statistics: cache hits
    uint32_t miss_count;           // Statistics: cache misses
    pthread_mutex_t lock;          // Thread-safe access
} nccl_ofi_mr_cache_t;
```

**Architecture**:
- **Sorted Array**: Entries sorted by `addr` for binary search (O(log N) lookup)
- **Dynamic Growth**: Doubles in size when full (starts at 1024 entries typically)
- **Statistics Tracking**: Hit/miss counts for performance monitoring
- **Thread-Safe**: Mutex protects all operations

### Cache Key

```c
/**
 * Cache key for lookup - supports multiple memory types
 */
struct nccl_ofi_mr_ckey {
    union {
        struct iovec iovec;                // CPU memory (virtual address)
        struct fi_mr_dmabuf fi_mr_dmabuf;  // GPU memory (dmabuf file descriptor)
    };
    nccl_net_ofi_ep_t *ep;          // Endpoint (part of key)
    enum nccl_ofi_mr_ckey_type type; // IOVEC or DMABUF
};
```

**Key Types**:

1. **IOVEC (CPU Memory)**:
   - Uses virtual address + length
   - Rounded to page boundaries (4KB alignment)

2. **DMABUF (GPU Memory)**:
   - Uses file descriptor + offset + length
   - Modern GPU memory registration (kernel 5.12+)
   - More efficient than legacy GDR

## Cache Operations

### Initialization

```cpp
// From nccl_ofi_mr.cpp:13
nccl_ofi_mr_cache_t *nccl_ofi_mr_cache_init( // ([src/nccl_ofi_mr.cpp:13](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_mr.cpp#L13))
                                             size_t init_num_entries,
                                             size_t mr_cache_page_size)
{
    nccl_ofi_mr_cache_t *cache = calloc(1, sizeof(*cache));

    cache->slots = calloc(init_num_entries, sizeof(*cache->slots));
    cache->system_page_size = mr_cache_page_size;  // Usually 4096
    cache->size = init_num_entries;                // Start with 1024 typically
    cache->used = 0;
    cache->hit_count = 0;
    cache->miss_count = 0;
    pthread_mutex_init(&cache->lock, NULL);

    return cache;
}
```

### Lookup (Fast Path)

```cpp
// From nccl_ofi_mr.cpp:112
void *nccl_ofi_mr_cache_lookup_entry( // ([src/nccl_ofi_mr.cpp:112](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_mr.cpp#L112))
                                      nccl_ofi_mr_cache_t *cache,
                                      nccl_ofi_mr_ckey_ref ckey,
                                      bool is_endpoint_mr)
{
    // Convert address/size to page-aligned address + page count
    uintptr_t page_addr = addr & -system_page_size;
    size_t pages = (addr + size - page_addr + page_size - 1) / page_size;

    // Linear scan through sorted array (could be binary search!)
    for (size_t slot = 0; slot < cache->used; slot++) {
        if (page_addr < cache->slots[slot]->addr) {
            // Passed where entry would be - cache miss
            cache->miss_count++;
            return NULL;
        }

        // Check if this entry covers the requested range
        if (is_endpoint_mr && ckey->ep != cache->slots[slot]->ep) {
            continue;  // Different endpoint
        }

        // Check if range [page_addr, page_addr + pages) is within entry
        if ((page_addr >= cache->slots[slot]->addr) &&
            ((page_addr - cache->slots[slot]->addr) / page_size + pages)
                <= cache->slots[slot]->pages) {
            // Cache hit!
            cache->hit_count++;
            cache->slots[slot]->refcnt++;  // Increment reference
            return cache->slots[slot]->handle;
        }
    }

    cache->miss_count++;
    return NULL;
}
```

**Performance**:
- **Best case**: O(1) if entry is at beginning
- **Average case**: O(N/2) linear scan
- **Worst case**: O(N) scan entire array
- **Typical**: <500 ns with ~100 entries

**Optimization Opportunity** (P1): Replace linear scan with binary search → O(log N) lookup

### Insert (Slow Path)

```cpp
// From nccl_ofi_mr.cpp:153
int nccl_ofi_mr_cache_insert_entry( // ([src/nccl_ofi_mr.cpp:153](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_mr.cpp#L153))
                                    nccl_ofi_mr_cache_t *cache,
                                    nccl_ofi_mr_ckey_ref ckey,
                                    bool is_endpoint_mr,
                                    void *handle)
{
    uintptr_t page_addr;
    size_t pages;
    compute_page_address(ckey, &page_addr, &pages);

    // Find insertion point (maintaining sorted order)
    for (size_t slot = 0; slot <= cache->used; slot++) {
        if (slot == cache->used || page_addr < cache->slots[slot]->addr) {
            // Found insertion point

            // Grow cache if full
            if (cache->used == cache->size) {
                cache->size *= 2;  // Double capacity
                cache->slots = realloc(cache->slots,
                                      cache->size * sizeof(*cache->slots));
            }

            // Shift entries to make room
            memmove(cache->slots + slot + 1,
                   cache->slots + slot,
                   (cache->used - slot) * sizeof(nccl_ofi_reg_entry_t *));

            // Create new entry
            cache->slots[slot] = calloc(1, sizeof(nccl_ofi_reg_entry_t));
            cache->slots[slot]->addr = page_addr;
            cache->slots[slot]->pages = pages;
            cache->slots[slot]->refcnt = 1;  // Initial reference
            cache->slots[slot]->handle = handle;
            cache->slots[slot]->ep = ckey->ep;

            cache->used++;
            return 0;
        }

        // Check for duplicate
        if (page_addr == cache->slots[slot]->addr &&
            pages == cache->slots[slot]->pages) {
            return -EEXIST;  // Already in cache
        }
    }
}
```

**Performance**: O(N) due to memmove, typically 1-2 μs

### Delete (Reference Counting)

```cpp
// From nccl_ofi_mr.cpp (not shown in snippet, but typical implementation)
int nccl_ofi_mr_cache_del_entry(nccl_ofi_mr_cache_t *cache, void *handle)
{
    for (size_t slot = 0; slot < cache->used; slot++) {
        if (cache->slots[slot]->handle == handle) {
            cache->slots[slot]->refcnt--;

            if (cache->slots[slot]->refcnt == 0) {
                // Last reference - remove from cache
                free(cache->slots[slot]);

                // Shift entries to fill gap
                memmove(cache->slots + slot,
                       cache->slots + slot + 1,
                       (cache->used - slot - 1) * sizeof(*cache->slots));
                cache->used--;

                return 1;  // Entry deleted, caller should deregister
            }

            return 0;  // Entry still referenced
        }
    }

    return -ENOENT;  // Not found
}
```

## Integration with Plugin

### Usage Flow

```cpp
// Typical send path in OFI plugin
int nccl_net_ofi_regmr(void *comm, void *data, int size,
                       int type, void **mhandle)
{
    nccl_net_ofi_ep_t *ep = (nccl_net_ofi_ep_t *)comm;

    // Create cache key (page-aligned)
    nccl_ofi_mr_ckey_t ckey = nccl_ofi_mr_ckey_mk_vec(data, size, ep);

    // Try cache lookup first
    void *mr_handle = nccl_ofi_mr_cache_lookup_entry(ep->mr_cache,
                                                      &ckey,
                                                      true);
    if (mr_handle) {
        // Cache hit!
        *mhandle = mr_handle;
        return 0;
    }

    // Cache miss - register with libfabric
    struct fid_mr *mr;
    struct fi_mr_attr attrs;
    nccl_ofi_mr_ckey_fill_mr_attrs(&ckey, &attrs, &flags);

    int ret = fi_mr_regattr(ep->domain, &attrs, 0, &mr);
    if (ret != 0) return ret;

    // Insert into cache
    nccl_ofi_mr_cache_insert_entry(ep->mr_cache, &ckey, true, mr);

    *mhandle = mr;
    return 0;
}
```

### Page Alignment Logic

```c
// From nccl_ofi_mr.h:109
static inline void nccl_ofi_mr_ckey_round(size_t *len, void **base_addr)
{
    uintptr_t page_base = ROUND_DOWN((uintptr_t)*base_addr, page_size);
    size_t aligned_size = ROUND_UP(((uintptr_t)*base_addr + *len), page_size)
                          - page_base;

    *base_addr = (void *)page_base;
    *len = aligned_size;
}
```

**Example**:
```
Original: addr=0x1234, size=8192
Rounded:  addr=0x1000, size=12288

Why? Memory registration must cover full pages.
- 0x1234 is in page starting at 0x1000
- 0x1234 + 8192 = 0x3234 extends into page starting at 0x3000
- Total pages: 0x1000, 0x2000, 0x3000 = 3 pages = 12288 bytes
```

## Cache vs Freelist MR Registration

**Two Different Mechanisms**:

### 1. MR Cache (This Document)
- **Purpose**: Cache registrations of **user buffers** (NCCL sends user data)
- **Scope**: Global per-endpoint
- **Lifetime**: Lives across many operations
- **Eviction**: Reference counted, deleted when refcnt=0
- **Example**: NCCL user calls `ncclSend(user_buffer, size)`, OFI plugin caches MR for `user_buffer`

### 2. Freelist MR (Document 15)
- **Purpose**: Pre-register **internal buffers** (eager buffers, control messages)
- **Scope**: Per-freelist
- **Lifetime**: Entire block registered once at allocation
- **Eviction**: Only freed when freelist destroyed
- **Example**: Eager send buffer pool - all buffers pre-registered

**Comparison**:

| Feature | MR Cache | Freelist MR |
|---------|----------|-------------|
| Registers | User buffers | Internal buffers |
| When | On-demand (lazy) | Up-front (eager) |
| Lookup | Hash/sorted array | Freelist pop |
| Reference counting | Yes (shared) | No (exclusive) |
| Deregistration | When refcnt=0 | When freelist destroyed |
| Typical hit rate | 95-99% | 100% (always hit) |

## NUMA Awareness

The MR cache itself is **not NUMA-aware**, but the topology selection ensures correct usage:

```cpp
// From nccl_ofi_topo.cpp - device selection considers NUMA
int select_efa_device(int gpu_id) {
    // Get GPU's NUMA node
    int gpu_numa = get_gpu_numa_node(gpu_id);

    // Find EFA device on same NUMA node
    for (int dev = 0; dev < num_efa_devices; dev++) {
        if (get_efa_numa_node(dev) == gpu_numa) {
            return dev;  // Matched NUMA node
        }
    }

    // Fallback: pick any device (cross-NUMA penalty)
    return 0;
}
```

**NUMA Impact on MR Cache** (estimated):
- **Local NUMA**: Memory registration ~100-500 μs
- **Remote NUMA**: Memory registration ~150-700 μs (approximately 30-40% slower)
- **Cache benefit**: Even more important for remote NUMA (amortize slow registration)

## Performance Tuning

### Cache Size

```cpp
// Typical initialization
cache = nccl_ofi_mr_cache_init(
    1024,      // Initial entries
    4096       // Page size
);
```

**Tuning**:
- **Small (256)**: Lower memory, more growth overhead
- **Large (4096)**: Higher memory, fewer reallocations
- **Recommendation**: 1024-2048 for most workloads

### Page Size

```c
#define NCCL_OFI_CACHE_PAGE_SIZE (4096ul)  // From nccl_ofi_mr.h:20
```

**Options**:
- **4KB**: Standard page size, fine-grained caching
- **2MB**: Huge pages, coarser caching but fewer entries
- **Neuron**: Page alignment disabled (`#if !HAVE_NEURON`)

**Recommendation**: Keep at 4KB for most cases

### Monitoring Cache Effectiveness

```cpp
// From nccl_ofi_mr.cpp:69 - printed on finalize
NCCL_OFI_INFO("MR cache %d hits %d misses",
              cache->hit_count,
              cache->miss_count);
```

**Interpretation**:
```
MR cache 95432 hits 823 misses
→ Hit rate: 95432 / (95432 + 823) = 99.1% ✓ Excellent

MR cache 1234 hits 5678 misses
→ Hit rate: 1234 / (1234 + 5678) = 17.8% ✗ Poor
```

**Target**: >99% hit rate in steady state

## Optimization Opportunities

### P0: Binary Search for Lookup

**Current**: Linear scan O(N)
```cpp
for (size_t slot = 0; slot < cache->used; slot++) {
    if (page_addr >= cache->slots[slot]->addr && ...) {
        return cache->slots[slot]->handle;
    }
}
```

**Proposed**: Binary search O(log N)
```cpp
size_t lo = 0, hi = cache->used;
while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (page_addr < cache->slots[mid]->addr) {
        hi = mid;
    } else if (page_addr >= cache->slots[mid]->addr +
               cache->slots[mid]->pages * page_size) {
        lo = mid + 1;
    } else {
        // Found!
        return cache->slots[mid]->handle;
    }
}
```

**Impact**: Estimated 5-10x faster lookup for large caches (1000+ entries)

### P1: Hash Table Instead of Sorted Array

**Benefit**: O(1) lookup instead of O(log N)

**Implementation**:
```cpp
struct nccl_ofi_mr_cache {
    struct hash_bucket {
        nccl_ofi_reg_entry_t *entries;  // Linked list per bucket
    } *buckets;
    size_t num_buckets;  // Prime number, e.g., 1021
};

uint32_t hash(uintptr_t addr) {
    return (addr >> 12) % num_buckets;  // Shift off page offset
}
```

**Tradeoff**: More memory, slightly more complex

### P2: Range Tree for Overlapping Regions

**Problem**: Current cache requires exact match. User might register overlapping regions.

**Example**:
```
Cached:    [0x1000 ─────────── 0x3000)
Request:         [0x1500 ─── 0x2500)  ← Should hit but doesn't!
```

**Solution**: Use interval tree (e.g., red-black tree with interval query)

**Impact**: Potentially higher hit rate (estimated 99.5% → 99.9%), but O(log N) complexity

### P3: Lock-Free Lookup

**Current**: Mutex protects all operations
```cpp
pthread_mutex_lock(&cache->lock);
void *handle = lookup_unlocked(cache, key);
pthread_mutex_unlock(&cache->lock);
```

**Proposed**: RCU (Read-Copy-Update) for reads
```cpp
// Read-side (no lock!)
rcu_read_lock();
void *handle = lookup_rcu(cache, key);
rcu_read_unlock();

// Write-side (still uses lock)
pthread_mutex_lock(&cache->lock);
insert_and_publish_rcu(cache, key, handle);
pthread_mutex_unlock(&cache->lock);
```

**Impact**: Estimated 2-3x faster lookup under contention

## Cache Invalidation (libfabric Level)

The OFI plugin MR cache is **separate** from libfabric's internal MR cache. Libfabric EFA provider has its own cache with invalidation:

**Libfabric MR Cache** (separate from OFI plugin cache):
```bash
FI_MR_CACHE_MONITOR=memhooks      # Hook malloc/free
FI_MR_CACHE_MAX_SIZE=unlimited
FI_MR_CACHE_MAX_COUNT=unlimited
```

**Two-Level Caching**:
```
NCCL user buffer
    ↓
OFI Plugin MR Cache (this document)
    ↓ (on miss)
fi_mr_regattr()
    ↓
Libfabric MR Cache (provider level)
    ↓ (on miss)
EFA driver ioctl
```

## Debugging

### Enable MR Cache Logging

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET

# Look for:
# "Found MR handle 0x... in cache slot 42"  ← Cache hit
# "MR cache 1234 hits 567 misses"           ← Statistics
```

### Check Cache Stats

```cpp
// Add to your code:
NCCL_OFI_INFO("Cache usage: %zu / %zu entries, hit rate: %.1f%%",
              cache->used, cache->size,
              100.0 * cache->hit_count /
                  (cache->hit_count + cache->miss_count));
```

### Common Issues

**Symptom**: Low hit rate (<95%)
- **Cause**: Memory regions not reused, or different sizes
- **Fix**: Ensure NCCL uses same buffers across calls

**Symptom**: Cache grows unbounded
- **Cause**: Registrations never released (refcnt leak)
- **Fix**: Check that every `regmr` has matching `deregmr`

**Symptom**: Slow lookup (>10 μs)
- **Cause**: Cache has 10,000+ entries (linear scan)
- **Fix**: Implement binary search or hash table

## Summary

| Metric | Value |
|--------|-------|
| Data structure | Sorted array of MR entries |
| Lookup complexity | O(N) linear scan (could be O(log N)) |
| Insert complexity | O(N) due to memmove |
| Delete complexity | O(N) due to memmove |
| Typical lookup time | <500 ns (cache hit) |
| Typical insert time | 1-2 μs |
| Memory per entry | ~48 bytes |
| Target hit rate | >99% |
| Performance gain | Approximately 25x faster vs re-registration |

**Key Takeaways**:
1. MR cache is **critical** - eliminates 100-500 μs registration overhead
2. Reference counting enables safe sharing across operations
3. Page-aligned addressing required for memory registration
4. Sorted array enables fast lookup but O(N) worst case
5. Excellent target for optimization (binary search, hash table, lock-free)
6. Different from freelist MR (user buffers vs internal buffers)
7. Works in tandem with libfabric's own MR cache (two-level)
