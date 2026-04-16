# Freelist Allocator

## Overview

The OFI NCCL plugin implements a custom **freelist allocator** for high-performance object pooling. This is a critical component for achieving low-latency communication by avoiding expensive malloc/free operations in the hot path.

**Location**:
- Header: [include/nccl_ofi_freelist.h](../aws-ofi-nccl/include/nccl_ofi_freelist.h)
- Implementation: [src/nccl_ofi_freelist.cpp](../aws-ofi-nccl/src/nccl_ofi_freelist.cpp)

## Purpose

Freelists solve several performance problems in the OFI plugin:

1. **Allocation Hot Path**: Avoid malloc/free overhead (1-10 μs per call) in send/recv operations
2. **Memory Registration**: Pre-register memory blocks to amortize registration cost (100-500 μs)
3. **Predictable Performance**: Eliminate non-deterministic allocator behavior
4. **Memory Fragmentation**: Reduce heap fragmentation from repeated allocations

## Architecture

### Two Freelist Types

#### 1. Simple Freelist
For objects that don't require memory registration:

```c
int nccl_ofi_freelist_init( // ([src/nccl_ofi_freelist.cpp:131](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_freelist.cpp#L131))
    size_t entry_size,              // Size of each entry
    size_t initial_entry_count,     // Initial pool size
    size_t increase_entry_count,    // Growth increment
    size_t max_entry_count,         // Max entries (0 = unlimited)
    nccl_ofi_freelist_entry_init_fn entry_init_fn,   // Optional init callback
    nccl_ofi_freelist_entry_fini_fn entry_fini_fn,   // Optional cleanup callback
    nccl_ofi_freelist_t **freelist_p);
```

**Use Cases**:
- Request structures (`nccl_net_ofi_sendrecv_req_t`)
- Schedule structures
- Non-network buffers

#### 2. MR-Registered Freelist
For buffers that require memory registration with libfabric:

```c
int nccl_ofi_freelist_init_mr( // ([src/nccl_ofi_freelist.cpp:188](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_freelist.cpp#L188))
    size_t entry_size,
    size_t initial_entry_count,
    size_t increase_entry_count,
    size_t max_entry_count,
    nccl_ofi_freelist_entry_init_fn entry_init_fn,
    nccl_ofi_freelist_entry_fini_fn entry_fini_fn,
    nccl_ofi_freelist_regmr_fn regmr_fn,      // Memory registration function
    nccl_ofi_freelist_deregmr_fn deregmr_fn,  // Deregistration function
    void *regmr_opaque,                       // Opaque data for regmr_fn
    size_t entry_alignment,                   // Entry alignment requirement
    nccl_ofi_freelist_t **freelist_p);
```

**Use Cases**:
- Eager send/recv buffers (pre-registered for RDMA)
- Control message buffers
- Connection management buffers
- GIN (GPU-initiated NCCL) metadata buffers

### Core Data Structures

```c
// Freelist element returned to users
typedef struct nccl_ofi_freelist_elem ([include/nccl_ofi_freelist.h:19-23](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/include/nccl_ofi_freelist.h#L19-L23)) {
    void *ptr;              // Pointer to actual buffer
    void *mr_handle;        // Memory registration handle (MR freelists only)
    struct nccl_ofi_freelist_elem *next;  // Internal linked list
} nccl_ofi_freelist_elem_t;

// Main freelist structure (opaque to users)
struct nccl_ofi_freelist_t ([include/nccl_ofi_freelist.h:88-109](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/include/nccl_ofi_freelist.h#L88-L109)) {
    size_t entry_size;                  // Size of each entry
    size_t num_allocated_entries;       // Total allocated
    size_t max_entry_count;             // Maximum entries
    size_t increase_entry_count;        // Growth increment

    nccl_ofi_freelist_elem_t *entries;  // Linked list of free entries
    struct nccl_ofi_freelist_block_t *blocks;  // Allocated memory blocks

    // Memory registration callbacks
    bool have_reginfo;
    nccl_ofi_freelist_regmr_fn regmr_fn;
    nccl_ofi_freelist_deregmr_fn deregmr_fn;
    void *regmr_opaque;

    // Entry callbacks
    nccl_ofi_freelist_entry_init_fn entry_init_fn;
    nccl_ofi_freelist_entry_fini_fn entry_fini_fn;

    pthread_mutex_t lock;               // Thread safety
};

// Memory block tracking
struct nccl_ofi_freelist_block_t ([include/nccl_ofi_freelist.h:28-35](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/include/nccl_ofi_freelist.h#L28-L35)) {
    struct nccl_ofi_freelist_block_t *next;
    void *memory;               // Base of allocated memory
    size_t memory_size;         // Size (rounded to page boundaries)
    void *mr_handle;            // Memory registration handle
    nccl_ofi_freelist_elem_t *entries;  // Array of entry descriptors
    size_t num_entries;
};
```

## Operation

### Allocation Flow

```
User calls: nccl_ofi_freelist_entry_alloc(freelist) // ([src/nccl_ofi_freelist.cpp:270](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_freelist.cpp#L270))
    ↓
Lock freelist mutex
    ↓
Check if entries available
    ↓ (if empty)
Allocate new block:
    1. Allocate page-aligned memory (nccl_net_ofi_alloc_mr_buffer)
    2. Call regmr_fn to register entire block (if MR freelist)
    3. Divide block into entry_size chunks
    4. Call entry_init_fn on each entry (if provided)
    5. Link all entries to freelist->entries
    ↓
Pop entry from freelist->entries
    ↓
Mark entry memory as accessible (memcheck)
    ↓
Unlock mutex
    ↓
Return nccl_ofi_freelist_elem_t* to user
```

**Key Optimizations**:
- **Batch Registration**: Entire block registered once, not per-entry
- **Page Alignment**: Memory rounded to page boundaries for efficient IOMMU mapping
- **Lazy Growth**: Only allocates when needed
- **Entry Padding**: Entries sized to fill pages completely (no wasted space)

### Deallocation Flow

```
User calls: nccl_ofi_freelist_entry_free(freelist, entry) // ([src/nccl_ofi_freelist.cpp:318](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_freelist.cpp#L318))
    ↓
Lock freelist mutex
    ↓
Mark entry memory as noaccess (memcheck)
    ↓
Push entry to head of freelist->entries linked list
    ↓
Unlock mutex
```

**Performance**: O(1) - just pointer manipulation under lock.

### Memory Layout

```
┌─────────────────────────────────────────────────────┐
│                  Block Memory                        │
│  (page-aligned, registered with fi_mr_reg)          │  // fi_mr_reg: ([include/rdma/fi_domain.h:413](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L413))
├──────────┬──────────┬──────────┬──────────┬─────────┤
│ Redzone  │  Entry 0 │ Redzone  │  Entry 1 │   ...   │
│ (guard)  │ (usable) │ (guard)  │ (usable) │         │
└──────────┴──────────┴──────────┴──────────┴─────────┘
     ^          ^
     |          |
     |          +-- entry->ptr points here
     +-- ASAN/memcheck guard region
```

Each entry has:
- **Redzone before**: Memory guard for overflow detection (ASAN/Valgrind)
- **Usable space**: `entry_size` bytes returned to user
- **Alignment**: Entries aligned to max(8, entry_alignment, MEMCHECK_GRANULARITY)

## Usage Examples

### Simple Freelist (Request Objects)

From [src/nccl_ofi_sendrecv.cpp:1364](../aws-ofi-nccl/src/nccl_ofi_sendrecv.cpp#L1364):

```cpp
// Create freelist for send/recv request structures
size_t req_size = sizeof(nccl_net_ofi_sendrecv_req_t);
nccl_ofi_freelist_t *nccl_ofi_reqs_fl;

ret = nccl_ofi_freelist_init(
    req_size,
    16,    // Start with 16 requests
    16,    // Grow by 16 when needed
    NCCL_OFI_MAX_REQUESTS,  // Max 1024 requests
    NULL,  // No init callback
    NULL,  // No fini callback
    &nccl_ofi_reqs_fl);

// Allocate request
nccl_ofi_freelist_elem_t *elem = nccl_ofi_freelist_entry_alloc(nccl_ofi_reqs_fl);
nccl_net_ofi_sendrecv_req_t *req = (nccl_net_ofi_sendrecv_req_t *)elem->ptr;

// Use request...
req->state = NCCL_OFI_SENDRECV_REQ_PENDING;

// Free when done
nccl_ofi_freelist_entry_free(nccl_ofi_reqs_fl, elem);
```

### MR-Registered Freelist (Network Buffers)

From [src/cm/nccl_ofi_cm_resources.cpp:68](../aws-ofi-nccl/src/cm/nccl_ofi_cm_resources.cpp#L68):

```cpp
// Create freelist for connection management buffers (pre-registered)
size_t buffer_size = sizeof(nccl_ofi_cm_msg_t);

ret = nccl_ofi_freelist_init_mr(
    buffer_size,
    16,    // Initial count
    16,    // Growth count
    0,     // Unlimited
    nullptr, nullptr,  // No init/fini callbacks
    endpoint::reg_mr,    // Register with fi_mr_reg
    endpoint::dereg_mr,  // Deregister with fi_close
    ep_ptr,             // Opaque data passed to reg_mr
    system_page_size,   // Page-aligned
    &buff_fl);

// Allocate buffer (already registered!)
nccl_ofi_freelist_elem_t *elem = nccl_ofi_freelist_entry_alloc(buff_fl);
void *buffer = elem->ptr;
void *mr_handle = elem->mr_handle;  // Use for fi_mr_desc(mr_handle)

// Use in fi_send without separate registration
fi_send(ep, buffer, size, fi_mr_desc(mr_handle), ...);
```

### GIN Metadata Freelist

From [src/gin/nccl_ofi_gin_resources.cpp:460](../aws-ofi-nccl/src/gin/nccl_ofi_gin_resources.cpp#L460):

```cpp
// Freelist for GPU-initiated NCCL metadata messages
ret = nccl_ofi_freelist_init_mr(
    sizeof(nccl_net_ofi_gin_signal_metadata_msg_t),
    16, 16, 0,
    nullptr, nullptr,
    endpoint::reg_mr,
    endpoint::dereg_mr,
    ep_ptr,
    system_page_size,
    &metadata_fl);
```

## Key Freelist Instances

| Freelist | Type | Entry Size | Used For | Location |
|----------|------|------------|----------|----------|
| `nccl_ofi_reqs_fl` | Simple | `sizeof(sendrecv_req_t)` | Send/recv request tracking | nccl_ofi_sendrecv.cpp:1364 |
| `schedule_fl` | Simple | Variable | Multi-rail scheduling | nccl_ofi_scheduler.cpp:268 |
| `buff_fl` | MR | `sizeof(cm_msg_t)` | Connection mgmt buffers | cm_resources.cpp:68 |
| `metadata_fl` | MR | `sizeof(gin_metadata_msg_t)` | GIN metadata messages | gin_resources.cpp:460 |
| `rx_buff_fl` | MR | Variable | GIN receive buffers | gin_resources.cpp:477 |

## Performance Impact

### Hot Path Analysis

**Without Freelist** (malloc/free per operation):
```
Send operation:
  malloc(sizeof(req)) → 1-5 μs
  ... use request ...
  free(req) → 1-5 μs

Total overhead: 2-10 μs per operation
```

**With Freelist**:
```
Send operation:
  freelist_entry_alloc() → <100 ns (pointer pop)
  ... use request ...
  freelist_entry_free() → <50 ns (pointer push)

Total overhead: <150 ns
```

**Speedup**: Estimated 20-100x reduction in allocation overhead

### Memory Registration Benefit

**Without Freelist** (register each buffer):
```
Send eager message:
  malloc(64KB) → 2-5 μs
  fi_mr_reg(buffer) → 100-500 μs
  fi_send(...)
  fi_mr_dereg() → 50-200 μs
  free(buffer) → 2-5 μs

Total: 154-710 μs
```

**With MR Freelist** (pre-registered block):
```
Send eager message:
  freelist_entry_alloc() → <100 ns (already registered!)
  fi_send(..., elem->mr_handle)
  freelist_entry_free() → <50 ns

Total: <150 ns
```

**Speedup**: Estimated 1000-5000x for small messages

## Tuning Parameters

### Initial Entry Count
- **Small (16)**: Low memory footprint, may cause early growth
- **Large (1024)**: High initial memory, fewer allocations later
- **Recommendation**: Start with 16-64 for most cases

### Increase Entry Count
- **Small (16)**: Controlled growth, more frequent allocations
- **Large (1024)**: Fewer allocations, larger memory jumps
- **Recommendation**: Match initial_entry_count or slightly larger

### Max Entry Count
- **0 (unlimited)**: Can grow forever, risk of memory exhaustion
- **Fixed (1024)**: Bounded memory, may fail under load
- **Recommendation**: Set to reasonable limit based on max connections

### Entry Alignment
- **1**: Minimal padding (simple freelists)
- **system_page_size**: Required for memory registration (MR freelists)
- **Recommendation**: Use page alignment for any buffers used with libfabric

## Thread Safety

All freelist operations are **thread-safe**:
- Uses `pthread_mutex_t lock` for all alloc/free operations
- Lock held during growth (new block allocation + registration)
- Lock **may be held** during `regmr_fn()` callback - avoid deadlocks!

**Deadlock Risk**: If `regmr_fn` tries to acquire another lock that could be held while calling freelist functions, deadlock can occur. Keep `regmr_fn` simple.

## Memory Debugging

The freelist integrates with **AddressSanitizer (ASAN)** and **Valgrind**:

```cpp
// Entry states tracked by ASAN:
NOACCESS   - Entry in freelist (freed), accesses will fault
UNDEFINED  - Entry allocated but uninitialized
DEFINED    - Entry initialized and ready to use
```

**Redzone Guards**: Each entry has guard bytes before/after to detect overflows:
```c
#define MEMCHECK_REDZONE_SIZE 16  // Typical value
```

## Optimization Opportunities

### Current Performance
- Allocation: ~50-100 ns (freelist hit)
- Deallocation: ~30-50 ns
- Growth: ~100-500 μs (when adding new block)

### P1 Optimization: Lock-Free Freelist
**Impact**: Estimated 2-3x faster allocation (~20-30 ns)

Replace mutex with atomic operations:
```cpp
// Current: mutex-protected linked list
pthread_mutex_lock(&freelist->lock);
entry = freelist->entries;
freelist->entries = entry->next;
pthread_mutex_unlock(&freelist->lock);

// Proposed: lock-free CAS
do {
    entry = atomic_load(&freelist->entries);
    if (!entry) break;
} while (!atomic_compare_exchange(&freelist->entries, &entry, entry->next));
```

**Challenges**:
- Growth still needs synchronization
- ABA problem requires versioned pointers

### P2 Optimization: Per-Thread Caches
**Impact**: Estimated 5-10x faster allocation (~5-10 ns)

Thread-local cache reduces contention:
```cpp
thread_local nccl_ofi_freelist_elem_t *local_cache[16];
thread_local int cache_count = 0;

// Allocate from local cache first
if (cache_count > 0) {
    return local_cache[--cache_count];  // No lock!
}
// Fall back to global freelist
```

### P3 Optimization: NUMA-Aware Allocation
**Impact**: 10-20% for cross-NUMA access patterns

Allocate blocks from NUMA node matching the device:
```cpp
void *memory = numa_alloc_onnode(size, numa_node);
```

## Debugging

### Check Freelist State

```cpp
// Add to nccl_ofi_freelist_t for debugging
void nccl_ofi_freelist_stats(nccl_ofi_freelist_t *freelist) {
    printf("Freelist %p:\n", freelist);
    printf("  Entry size: %zu\n", freelist->entry_size);
    printf("  Allocated: %zu / %zu\n",
           freelist->num_allocated_entries,
           freelist->max_entry_count);

    // Count free entries
    size_t free_count = 0;
    for (auto *e = freelist->entries; e; e = e->next) free_count++;
    printf("  Free: %zu\n", free_count);
    printf("  In use: %zu\n", freelist->num_allocated_entries - free_count);
}
```

### Common Issues

**Symptom**: Freelist returns NULL
**Cause**: Max entry count reached or allocation failed
**Fix**: Increase `max_entry_count` or check for leaks

**Symptom**: Crash on freelist_entry_free
**Cause**: Double-free or freeing wrong entry
**Fix**: Use ASAN, check entry ownership

**Symptom**: Slow allocation after growth
**Cause**: Memory registration taking 100s of μs
**Fix**: Increase `initial_entry_count` to avoid growth

## Related Components

- **Memory Registration Cache** ([lkey-rkey-explained.md](lkey-rkey-explained.md)): Freelist MR handles may be cached
- **Send/Recv Protocol** ([ofi-plugin-protocols.md](ofi-plugin-protocols.md)): Request structures allocated from freelist
- **RDMA & Memory Registration** ([rdma-memreg.md](rdma-memreg.md)): Freelist uses `fi_mr_reg()` for blocks

## Summary

The freelist allocator is a **critical performance component**:

| Metric | Impact |
|--------|--------|
| Allocation overhead | Estimated 20-100x faster than malloc |
| MR registration | Estimated 1000-5000x faster (pre-registered blocks) |
| Latency contribution | <100 ns (vs 2-10 μs without freelist) |
| Memory efficiency | ~95% utilization (page-padded entries) |

**Key Takeaways**:
1. Freelists eliminate malloc/free from the hot path
2. MR-registered freelists amortize expensive registration
3. Thread-safe but lock contention possible under high concurrency
4. Tuning parameters critical for memory/performance balance
5. Excellent target for lock-free optimization

## Current Status (April 2026)

**Note:** The freelist has been converted from a C-style struct with free
functions to a C++ class (`nccl_ofi_freelist`) with constructors and
methods. The code examples and line number references above are from the
older C version. The current implementation is at:
- Header: [include/nccl_ofi_freelist.h](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/include/nccl_ofi_freelist.h)
- Implementation: [src/nccl_ofi_freelist.cpp](https://github.com/aws/aws-ofi-nccl/blob/c2a27c4/src/nccl_ofi_freelist.cpp)

Key changes:
- `nccl_ofi_freelist_init()` / `nccl_ofi_freelist_init_mr()` → C++ constructors
- `nccl_ofi_freelist_entry_alloc()` / `nccl_ofi_freelist_entry_free()` → class methods
- Memory registration handled in constructor, not separate init call
- The core allocation algorithm (freelist with block growth) is unchanged
