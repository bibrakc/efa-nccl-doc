# Memory Registration Cache Implementation

> **Source lookups:** this document records mechanism, defaults, corrections and history. For current function bodies, call graphs and blast radius, query the code graph (`codegraph explore <symbol>`) rather than trusting a pasted copy here.

## Overview

The OFI NCCL plugin implements a **custom memory registration (MR) cache** to avoid expensive re-registration of memory regions. This is one of the most critical performance optimizations in the plugin.

**Location**:
- Header: [include/nccl_ofi_mr.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_mr.h)
- Implementation: [src/nccl_ofi_mr.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_mr.cpp)

> **API shape.** The cache is a C++ class, `nccl_ofi_mr_cache`
> ([include/nccl_ofi_mr.h:201-256](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_mr.h)),
> whose storage is a `std::vector<nccl_ofi_reg_entry_t *> slots` grown by the
> vector — not a hand-`realloc`-ed array. Cache lifetime follows the object:
> the constructor reserves storage and the destructor releases everything and
> logs hit/miss stats (RAII). The old C-style `nccl_ofi_mr_cache_*` free
> functions no longer exist; use the class methods described below. The former
> C API is preserved for historical reference in
> [Former C API (removed)](#former-c-api-removed) with an old→new mapping table.

## Why MR Cache Matters

**Performance Impact** (estimated):
```
Without cache (every operation registers memory):
  fi_mr_reg()    → 100-500 μs  // ([include/rdma/fi_domain.h, fi_mr_reg](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h))
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

> Source: `include/nccl_ofi_mr.h` — `struct nccl_ofi_reg_entry` / `nccl_ofi_reg_entry_t`. Use `codegraph explore nccl_ofi_reg_entry` for the current definition.

An entry stores one registered region: page-aligned base `addr`, `pages`
(page count, not bytes), `refcnt`, the `void *handle` (fi_mr handle), and the
owning `nccl_net_ofi_ep_t *ep`. **The constructor sets `refcnt = 1`** — a freshly
inserted entry already holds one reference.

**Key Design Decisions**:
- **Page-aligned**: `addr` is always aligned to the cache's `page_size`
- **Page counts**: `pages` stores number of pages, not byte size
- **Reference counting**: Enables safe sharing - entry deleted only when refcnt reaches 0. The constructor sets `refcnt = 1`, so a freshly inserted entry already holds one reference.
- **Per-endpoint**: Different endpoints can have different MRs for same address
- **Heap-allocated**: Entries are individually `new`-allocated (`new (std::nothrow) nccl_ofi_reg_entry(...)`) and stored as pointers in the cache's `std::vector`.

### Cache Structure

> Source: `include/nccl_ofi_mr.h` — `class nccl_ofi_mr_cache`. Use `codegraph explore nccl_ofi_mr_cache` for the current definition.

The public surface is a two-arg constructor `nccl_ofi_mr_cache(size_t
init_num_entries, size_t page_size_arg)`, a destructor, and three methods:
`lookup_entry`, `insert_entry`, `del_entry`. Storage is a private
`std::vector<nccl_ofi_reg_entry_t *> slots` plus `page_size`, `hit_count`,
`miss_count`. The class is `= delete`d for copy construction and copy
assignment.

**Architecture**:
- **Sorted Vector**: `slots` holds entry pointers sorted by `addr` for linear scan lookup (O(N)). Growth (`std::vector::insert`) and shrink (`std::vector::erase`) are handled by the vector; there is no manual `realloc`/`memmove`.
- **Reserved capacity**: The constructor calls `slots.reserve(init_num_entries)`; the vector still grows past that if needed. Initial size is `NCCL_OFI_MR_CACHE_INIT_SIZE` (see [include/nccl_ofi.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi.h)).
- **Statistics Tracking**: `hit_count` / `miss_count` are members, logged by the destructor.
- **Thread-Safety**: The cache class itself is **not** internally locked. Callers serialize access with an external `mr_cache_lock` held by the owning domain (see [Integration with Plugin](#integration-with-plugin)). This differs from the old C struct, which embedded its own `pthread_mutex_t`.

**RAII / lifetime**:
- **Construction acquires**: the constructor validates arguments and reserves storage; on invalid arguments (zero initial entries or zero page size) it **throws `std::runtime_error`** rather than returning an error code. Because a constructor cannot return an int, argument validation that used to be a return value is now an exception, and the "already-initialized cache pointer" the old `nccl_ofi_mr_cache_init()` produced is replaced by a fully-constructed object.
- **Destruction releases**: the destructor logs `"MR cache %d hits %d misses"`, and if any entries remain it warns (`"MR cache destroyed while %zu memory segments still held..."`) and `delete`s them. Normal teardown expects the cache to be empty because every `regmr` is matched by a `dereg` that drives refcnt to zero.
- **Non-copyable / non-movable**: copy constructor and copy assignment are `= delete`d, so a cache cannot be accidentally duplicated (which would double-free the underlying MR entries).

### Cache Key

> Source: `include/nccl_ofi_mr.h` — `struct nccl_ofi_mr_ckey`, `nccl_ofi_mr_ckey_t`, `nccl_ofi_mr_ckey_ref`. Use `codegraph explore nccl_ofi_mr_ckey` for the current definition.

The key is a tagged union: an `iovec` arm (CPU memory) and, when
`HAVE_DECL_FI_MR_DMABUF` is defined, a `fi_mr_dmabuf` arm (GPU memory), plus the
owning `ep` and a `type` tag. The critical invariant is that the union's first
member is at offset 0 so the key can be `reinterpret_cast` straight into the
matching libfabric attr member — enforced at compile time:

```c
static_assert(offsetof(struct nccl_ofi_mr_ckey, iovec) == 0,
              "Cache keys must be safe to cast to 'struct iovec'");
#if HAVE_DECL_FI_MR_DMABUF
static_assert(offsetof(struct nccl_ofi_mr_ckey, fi_mr_dmabuf) == 0,
              "Cache keys must be safe to cast to 'struct fi_mr_dmabuf'");
#endif
```

`nccl_ofi_mr_ckey_ref` (a `const`-pointer-to-`const`-key) is the argument type
used by all three cache methods, so callers pass keys by immutable reference.
The DMABUF arm of the union and the DMABUF static_assert are guarded by
`HAVE_DECL_FI_MR_DMABUF`, so on platforms without DMA-BUF support only the
`iovec` form of the key exists.

**Key Types**:

1. **IOVEC (CPU Memory)**:
   - Uses virtual address + length
   - Rounded to page boundaries via `nccl_ofi_mr_ckey_round()` (skipped on Neuron)

2. **DMABUF (GPU Memory)**:
   - Uses file descriptor + offset + length
   - Modern GPU memory registration (kernel 5.12+)
   - More efficient than legacy GDR

Keys are built with the helper factory functions
`nccl_ofi_mr_ckey_mk_vec(iov_base, iov_len, ep_ptr)` and (when available)
`nccl_ofi_mr_ckey_mk_dmabuf(fd, offset, len, base_addr, ep_ptr)`
([include/nccl_ofi_mr.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_mr.h)),
which set the `type` tag and the endpoint. Attribute filling for libfabric is
done by `nccl_ofi_mr_ckey_fill_mr_attrs()`
([include/nccl_ofi_mr.h:160](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_mr.h)),
which relies on the static_asserts above to `reinterpret_cast` the key straight
into a `struct fi_mr_attr`.

## Cache Operations

### Construction

> Source: `src/nccl_ofi_mr.cpp` — `nccl_ofi_mr_cache::nccl_ofi_mr_cache()`. Use `codegraph explore nccl_ofi_mr_cache` for the current body.

The constructor takes exactly two arguments — `init_num_entries` and
`page_size_arg` — and does not allocate any MR entries; it only `reserve`s
vector capacity. `hit_count` and `miss_count` are default-initialized to 0 in the
class body. **There is no return code: invalid arguments (zero initial entries
or zero page size) raise `std::runtime_error`**, which the caller must be
prepared to catch (see the `emplace` call site below).

### Destruction

> Source: `src/nccl_ofi_mr.cpp` — `nccl_ofi_mr_cache::~nccl_ofi_mr_cache()`. Use `codegraph explore nccl_ofi_mr_cache` for the current body.

The destructor always emits the hit/miss statistics line
(`"MR cache %d hits %d misses"`, replacing the old explicit "print stats on
finalize" step) and, if any entries remain, warns
(`"MR cache destroyed while %zu memory segments still held..."`) and `delete`s
them. Normal teardown expects the cache to be empty because every `regmr` is
matched by a `dereg` that drives refcnt to zero.

### Lookup (Fast Path)

> Source: `src/nccl_ofi_mr.cpp` — `nccl_ofi_mr_cache::lookup_entry()`. Use `codegraph explore lookup_entry` for the current body.

`lookup_entry` converts the ckey's base address/length to a page-aligned
`page_addr` + `pages` (via `compute_page_address`), then linearly scans the
sorted `slots` vector for a covering entry.

**Return value**: the `void *` MR handle on hit (**with `refcnt` incremented**),
or `nullptr` on miss. `is_endpoint_mr` controls whether the endpoint must also
match: when `true`, an entry for a different endpoint is skipped rather than
treated as a hit.

**Performance**:
- **Best case**: O(1) if entry is at beginning
- **Average case**: O(N/2) linear scan
- **Worst case**: O(N) scan entire vector
- **Typical**: <500 ns with ~100 entries

**Optimization Opportunity** (P1): Replace linear scan with binary search → O(log N) lookup

### Insert (Slow Path)

> Source: `src/nccl_ofi_mr.cpp` — `nccl_ofi_mr_cache::insert_entry()`. Use `codegraph explore insert_entry` for the current body.

`insert_entry` computes the page-aligned range, scans for the sorted insertion
point, `new`-allocates a `nccl_ofi_reg_entry` (refcnt starts at 1) and calls
`std::vector::insert` to place it in order.

**Return value**: `0` on success, `-ENOMEM` if the entry allocation fails, or
`-EEXIST` if the range is already covered. The new entry's `refcnt` is `1` (set
by the `nccl_ofi_reg_entry` constructor), representing the caller that just
registered it. Growth and shifting are done by `std::vector::insert` — there is
no manual `realloc`/`memmove` and no capacity-doubling bookkeeping in this code.

**Performance**: O(N) due to the vector shift on insert, typically 1-2 μs.

### Delete (Reference Counting)

> Source: `src/nccl_ofi_mr.cpp` — `nccl_ofi_mr_cache::del_entry()`. Use `codegraph explore del_entry` for the current body.

`del_entry` finds the slot by `handle`, decrements `refcnt`, and only removes and
frees the entry (via `std::vector::erase`) when the last reference is dropped.

**Return value**: `-ENOENT` if the handle isn't found, `0` if the entry is still
referenced (refcnt decremented but > 0, so the caller must **not** deregister),
or `1` if the last reference was dropped and the entry was removed (the caller
**should** deregister the underlying MR). `std::vector::erase` performs the shift.

**Performance**: O(N) — linear scan to find the handle plus a vector shift on removal.

## Integration with Plugin

### Ownership and Construction

The cache is owned **per-domain**, stored as a `std::optional<nccl_ofi_mr_cache>`
member on the domain object and constructed in place with `emplace`
([src/nccl_ofi_net.cpp:874-876](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_net.cpp)):

```cpp
// nccl_net_ofi_domain_t constructor
if (!ofi_nccl_mr_cache_disable()) {
    this->mr_cache.emplace(NCCL_OFI_MR_CACHE_INIT_SIZE, system_page_size);
}
```

When the cache is disabled the optional stays empty, so callers must test it
(`if (this->mr_cache)` / `domain->mr_cache.has_value()`) before use. Access is
serialized by a separate `mr_cache_lock` held by the domain — the cache class
itself has no internal mutex.

### Usage Flow (RDMA register path)

> Source: `src/nccl_ofi_rdma.cpp` — `nccl_net_ofi_rdma_domain_t::reg_mr`. Use `codegraph explore reg_mr` for the current body.

The register path takes the domain's `mr_cache_lock` **across both the lookup
and the insert** so a missing entry is registered on-device and inserted
atomically. The idiom:

```cpp
std::lock_guard cache_guard(this->mr_cache_lock);
handle = this->mr_cache->lookup_entry(ckey, endpoint_mr);   // hit → return handle
if (!handle) {
    reg_mr_on_device(ckey, type, ep, &handle);              // miss → register
    if (this->mr_cache->insert_entry(ckey, endpoint_mr, handle) != 0)
        dereg_mr_no_lock(handle);                           // insert failed → undo
}
```

When the cache is disabled (`!this->mr_cache`) the path always registers on
device. The sendrecv transport follows the same shape in
`sendrecv_comm_mr_base_reg` (`src/nccl_ofi_sendrecv.cpp`), using
`domain->mr_cache.has_value()` and `domain->mr_cache_lock`.

### Deregistration Flow

> Source: `src/nccl_ofi_rdma.cpp` — `nccl_net_ofi_rdma_domain_t::dereg_mr`; `src/nccl_ofi_sendrecv.cpp` — `sendrecv_comm_mr_base_dereg`. Use `codegraph explore dereg_mr` for the current bodies.

Deregistration is driven entirely by `del_entry`'s return contract: under the
`mr_cache_lock`, call `del_entry(mr_handle)`; on `0` (still referenced) return
without touching the device; only on `1` (last reference dropped) fall through to
`dereg_mr_on_device`. A negative return is logged as an error.

### Page Alignment Logic

Key creation rounds the region to enclosing pages before it is ever handed to
the cache, using `mr_cache_alignment` (set to `min(system_page_size,
NCCL_OFI_CACHE_PAGE_SIZE)` in
[src/nccl_ofi_net.cpp:210](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_net.cpp)):

```c
// include/nccl_ofi_mr.h - called from nccl_ofi_mr_ckey_mk_vec()
static inline void nccl_ofi_mr_ckey_round(size_t *len, void **base_addr, const char *type)
{
    uintptr_t page_base = NCCL_OFI_ROUND_DOWN((uintptr_t)*base_addr, mr_cache_alignment);
    size_t aligned_size = NCCL_OFI_ROUND_UP(((uintptr_t)*base_addr + *len), mr_cache_alignment)
                          - page_base;
    *base_addr = (void *)page_base;
    *len = aligned_size;
}
```

The cache additionally recomputes a page-aligned address and page count from the
key inside `compute_page_address()`
([src/nccl_ofi_mr.cpp:52](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_mr.cpp)),
using its own `page_size`.

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
- **Scope**: Per-domain (one cache per domain, guarded by the domain's lock)
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
| Lookup | Sorted vector scan | Freelist pop |
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
// Typical construction (per-domain, see Integration section)
domain->mr_cache.emplace(
    NCCL_OFI_MR_CACHE_INIT_SIZE,  // Initial reserved entries
    system_page_size              // Page size
);
```

**Tuning**:
- **Small (256)**: Lower memory, more vector growth overhead
- **Large (4096)**: Higher memory, fewer reallocations
- **Recommendation**: 1024-2048 for most workloads

### Page Size

```c
#define NCCL_OFI_CACHE_PAGE_SIZE (4096ul)  // From include/nccl_ofi_mr.h:19
```

**Options**:
- **4KB**: Standard page size, fine-grained caching
- **2MB**: Huge pages, coarser caching but fewer entries
- **Neuron**: Page alignment disabled (`#if !HAVE_NEURON`)

**Recommendation**: Keep at 4KB for most cases

### Monitoring Cache Effectiveness

```cpp
// From src/nccl_ofi_mr.cpp:36 - emitted by the destructor
NCCL_OFI_INFO(NCCL_NET, "MR cache %d hits %d misses",
              this->hit_count,
              this->miss_count);
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
for (size_t slot = 0; slot < this->slots.size(); slot++) {
    if (page_addr >= this->slots[slot]->addr && ...) {
        return this->slots[slot]->handle;
    }
}
```

**Proposed**: Binary search O(log N)
```cpp
size_t lo = 0, hi = this->slots.size();
while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (page_addr < this->slots[mid]->addr) {
        hi = mid;
    } else if (page_addr >= this->slots[mid]->addr +
               this->slots[mid]->pages * this->page_size) {
        lo = mid + 1;
    } else {
        // Found!
        return this->slots[mid]->handle;
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

**Current**: An external `mr_cache_lock` (held by the owning domain) serializes
all operations; the cache class has no internal lock:
```cpp
std::lock_guard cache_guard(domain->mr_cache_lock);
void *handle = domain->mr_cache->lookup_entry(ckey, endpoint_mr);
```

**Proposed**: RCU (Read-Copy-Update) for reads
```cpp
// Read-side (no lock!)
rcu_read_lock();
void *handle = lookup_rcu(cache, key);
rcu_read_unlock();

// Write-side (still uses lock)
std::lock_guard cache_guard(domain->mr_cache_lock);
insert_and_publish_rcu(cache, key, handle);
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

The `hit_count`, `miss_count` and `slots` members are `private`, so external
code cannot read them directly. The destructor already logs the hit/miss line;
to add usage introspection you would add a member function inside the class, e.g.:

```cpp
// Hypothetical member function added to nccl_ofi_mr_cache
void log_stats() const {
    NCCL_OFI_INFO(NCCL_NET, "Cache usage: %zu entries, hit rate: %.1f%%",
                  this->slots.size(),
                  100.0 * this->hit_count /
                      (this->hit_count + this->miss_count));
}
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
| Data structure | Sorted `std::vector` of MR entry pointers |
| Lookup complexity | O(N) linear scan (could be O(log N)) |
| Insert complexity | O(N) due to vector shift (`std::vector::insert`) |
| Delete complexity | O(N) due to scan + vector shift (`std::vector::erase`) |
| Typical lookup time | <500 ns (cache hit) |
| Typical insert time | 1-2 μs |
| Memory per entry | ~48 bytes |
| Target hit rate | >99% |
| Performance gain | Approximately 25x faster vs re-registration |

**Key Takeaways**:
1. MR cache is **critical** - eliminates 100-500 μs registration overhead
2. Reference counting enables safe sharing across operations
3. Page-aligned addressing required for memory registration
4. Sorted vector enables fast lookup but O(N) worst case
5. Excellent target for optimization (binary search, hash table, lock-free)
6. Different from freelist MR (user buffers vs internal buffers)
7. Works in tandem with libfabric's own MR cache (two-level)
8. RAII class: construction reserves storage (throws on bad args), destruction releases everything and logs stats; locking is external (domain-owned)

## Deployment Notes

### Per-Domain Cache

The MR cache is allocated per-domain, not per-device. Each domain
(thread scope) has its own cache, guarded by the domain's `mr_cache_lock`.
This means:
- Different threads have separate caches (no lock contention on data path)
- Cache created in the domain constructor via `mr_cache.emplace(...)`
  ([src/nccl_ofi_net.cpp:874-876](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_net.cpp))
- Cache destroyed automatically when the `std::optional<nccl_ofi_mr_cache>`
  member is destroyed with the domain (RAII), which also logs hit/miss stats
- The cache is only created when `ofi_nccl_mr_cache_disable()` is false; callers
  must therefore test `mr_cache` / `mr_cache.has_value()` before use

### Cache Key Types

The cache supports two key types (from
[include/nccl_ofi_mr.h:19-25](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_mr.h)):

1. **NCCL_OFI_MR_CKEY_IOVEC**: Standard memory — virtual address + length,
   page-aligned to `mr_cache_alignment`
2. **NCCL_OFI_MR_CKEY_DMABUF**: GPU memory via DMA-BUF — file descriptor +
   offset + length (kernel 5.12+, used on modern GPU drivers). Present only when
   `HAVE_DECL_FI_MR_DMABUF` is defined.

## Former C API (removed)

Earlier releases exposed the cache as a C struct (`nccl_ofi_mr_cache_t`) with a
set of free functions and an embedded `pthread_mutex_t`. Those free functions
**no longer exist** — the cache is now the `nccl_ofi_mr_cache` C++ class. This
section is retained only to help when reading older code, commits, or blog posts.

| Former C API (removed) | Current C++ equivalent |
|------------------------|------------------------|
| `nccl_ofi_mr_cache_init(init_num_entries, page_size)` → `nccl_ofi_mr_cache_t *` | Constructor `nccl_ofi_mr_cache(size_t init_num_entries, size_t page_size_arg)` (throws `std::runtime_error` on bad args) |
| `nccl_ofi_mr_cache_finalize(cache)` / manual `free` | Destructor `~nccl_ofi_mr_cache()` (RAII; logs stats, frees leftovers) |
| `nccl_ofi_mr_cache_lookup_entry(cache, ckey, is_endpoint_mr)` | Method `void *lookup_entry(nccl_ofi_mr_ckey_ref ckey, bool is_endpoint_mr)` |
| `nccl_ofi_mr_cache_insert_entry(cache, ckey, is_endpoint_mr, handle)` | Method `int insert_entry(nccl_ofi_mr_ckey_ref ckey, bool is_endpoint_mr, void *handle)` |
| `nccl_ofi_mr_cache_del_entry(cache, handle)` | Method `int del_entry(void *handle)` |
| `cache->slots` (`nccl_ofi_reg_entry_t **`), `cache->size`, `cache->used`, `realloc`/`memmove` | `std::vector<nccl_ofi_reg_entry_t *> slots` (private); growth/shift via `insert`/`erase` |
| `cache->system_page_size` | private member `page_size` |
| `cache->hit_count` / `cache->miss_count` (public) | private members `hit_count` / `miss_count` |
| `pthread_mutex_t cache->lock` (internal) | No internal lock — external `mr_cache_lock` on the owning domain |

Semantic changes to note:
- The old init function returned a heap pointer (or `NULL` on failure); the
  constructor cannot return a code, so argument-validation failures are thrown
  as `std::runtime_error` and object lifetime is tied to the enclosing
  `std::optional`/scope.
- Method return-value contracts are unchanged: `insert_entry` returns
  `0`/`-ENOMEM`/`-EEXIST`; `del_entry` returns `0` (still referenced),
  `1` (deleted — caller should deregister), or `-ENOENT`; `lookup_entry` returns
  the handle or `nullptr`.
- The class is non-copyable and non-movable (`= delete`d copy ctor/assignment).
