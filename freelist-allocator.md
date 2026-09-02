# Freelist Allocator

## Overview

The OFI NCCL plugin implements a custom **freelist allocator** for high-performance object pooling. This is a critical component for achieving low-latency communication by avoiding expensive malloc/free operations in the hot path.

**Location**:
- Header: [include/nccl_ofi_freelist.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_freelist.h)
- Implementation: [src/nccl_ofi_freelist.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_freelist.cpp)

## Purpose

Freelists solve several performance problems in the OFI plugin:

1. **Allocation Hot Path**: Avoid malloc/free overhead (1-10 μs per call) in send/recv operations
2. **Memory Registration**: Pre-register memory blocks to amortize registration cost (100-500 μs)
3. **Predictable Performance**: Eliminate non-deterministic allocator behavior
4. **Memory Fragmentation**: Reduce heap fragmentation from repeated allocations

## Architecture

## API Shape

The freelist is a C++ class, `nccl_ofi_freelist`
([include/nccl_ofi_freelist.h:22](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_freelist.h)).
It is created with `new` (constructor) and destroyed with `delete` (destructor),
following RAII: the constructor allocates the initial entries and, for a
registered freelist, registers the backing memory; the destructor deregisters
and frees everything. The old `nccl_ofi_freelist_init*()` free functions and the
`nccl_ofi_freelist_t` / `nccl_ofi_freelist_elem_t` typedefs **no longer exist**;
see [Former C API (removed)](#former-c-api-removed) for the mapping.

### Two Freelist Types

Both types are the same class; which of the two constructors you call decides
whether the backing memory is registered.

#### 1. Simple Freelist
For objects that don't require memory registration
([include/nccl_ofi_freelist.h:109](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_freelist.h)):

```cpp
nccl_ofi_freelist(
    size_t entry_size,                                // Size of each entry
    size_t initial_entry_count,                       // Initial pool size
    size_t increase_entry_count,                      // Growth increment
    size_t max_entry_count,                           // Max entries (0 = unlimited)
    nccl_ofi_freelist_entry_init_fn entry_init_fn,    // Optional init callback (or NULL)
    nccl_ofi_freelist_entry_fini_fn entry_fini_fn,    // Optional cleanup callback (or NULL)
    const char *name,                                 // Debug name (must outlive freelist)
    bool enable_leak_detection);                      // Warn on in-use entries at destruction
```

**Use Cases**:
- Request structures (`nccl_net_ofi_sendrecv_req`)
- Schedule structures
- Non-network buffers

#### 2. MR-Registered Freelist
For buffers that require memory registration with libfabric
([include/nccl_ofi_freelist.h:125](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_freelist.h)):

```cpp
nccl_ofi_freelist(
    size_t entry_size,
    size_t initial_entry_count,
    size_t increase_entry_count,
    size_t max_entry_count,
    nccl_ofi_freelist_entry_init_fn entry_init_fn,
    nccl_ofi_freelist_entry_fini_fn entry_fini_fn,
    nccl_ofi_freelist_regmr_fn regmr_fn,      // Memory registration function
    nccl_ofi_freelist_deregmr_fn deregmr_fn,  // Deregistration function
    void *regmr_opaque,                       // Opaque data for regmr_fn
    size_t entry_alignment,                   // Entry alignment (power of two)
    const char *name,                         // Debug name (must outlive freelist)
    bool enable_leak_detection);
```

Both constructors delegate to a shared private helper
`nccl_ofi_freelist::init_internal(...)`
([src/nccl_ofi_freelist.cpp:19](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_freelist.cpp)),
which sets `have_reginfo` accordingly (false for the simple ctor, true for the
MR ctor).

**Error handling (RAII consequence).** A constructor cannot return an error
code. `init_internal()` calls `add(initial_entry_count)` to allocate the first
block, and if that fails it **throws `std::runtime_error("freelist initial
allocation failed")`** ([src/nccl_ofi_freelist.cpp:19-84](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_freelist.cpp)).
So the old pattern of checking an `int` return from `nccl_ofi_freelist_init*()`
is replaced by either letting the exception propagate or wrapping the `new` in a
`try`/`catch`. (Runtime growth failures during `entry_alloc()` are still
surfaced as a `NULL` return, not an exception — see below.)

**Copy/move.** The copy constructor and copy assignment are `= delete`d
([include/nccl_ofi_freelist.h:134](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_freelist.h)),
so a freelist cannot be duplicated (which would double-free the backing blocks).

**Use Cases**:
- Eager send/recv buffers (pre-registered for RDMA)
- Control message buffers
- Connection management buffers
- GIN (GPU-initiated NCCL) metadata buffers

### Core Data Structures

```cpp
class nccl_ofi_freelist {                       // [include/nccl_ofi_freelist.h:22]
public:
    // Freelist element returned to users (nested type; replaces the old
    // nccl_ofi_freelist_elem_t)
    struct fl_entry {                           // [include/nccl_ofi_freelist.h:27-31]
        void *ptr;                 // Pointer to actual buffer
        void *mr_handle;           // Memory registration handle (MR freelists only)
        struct fl_entry *next;     // Internal linked list
    };

    // ... constructors (see API Shape), destructor, entry_alloc(), entry_free() ...

protected:
    // Internal: tracking data for blocks of allocated memory
    // (now a nested/protected type; no longer a public typedef)
    struct nccl_ofi_freelist_block_t {          // [include/nccl_ofi_freelist.h:289-297]
        struct nccl_ofi_freelist_block_t *next;
        void *memory;              // Base of allocated memory
        size_t memory_size;        // Size (rounded to page boundaries)
        void *mr_handle;           // Memory registration handle
        fl_entry *entries;         // Array of entry descriptors
        size_t num_entries;
    };

    size_t entry_size;             // Size of each entry (padded, incl. redzone)

    size_t num_allocated_entries;  // Total allocated
    size_t num_in_use_entries;     // Currently handed out (used by leak detection)
    size_t max_entry_count;        // Maximum entries (0 = unlimited)
    size_t increase_entry_count;   // Growth increment

    fl_entry *entries;                          // [include/nccl_ofi_freelist.h:306: fl_entry *entries]
    struct nccl_ofi_freelist_block_t *blocks;   // Allocated memory blocks

    // Memory registration callbacks
    bool have_reginfo;
    nccl_ofi_freelist_regmr_fn regmr_fn;
    nccl_ofi_freelist_deregmr_fn deregmr_fn;
    void *regmr_opaque;

    size_t memcheck_redzone_size;

    // Entry callbacks
    nccl_ofi_freelist_entry_init_fn entry_init_fn;
    nccl_ofi_freelist_entry_fini_fn entry_fini_fn;

    const char *name;              // Debug name (user-provided, must stay valid)
    bool enable_leak_detection;
};
```

**Notes on the current layout**:
- `fl_entry` is a **public nested struct**, referenced as
  `nccl_ofi_freelist::fl_entry`. It replaced the old free-standing
  `nccl_ofi_freelist_elem_t` typedef.
- `nccl_ofi_freelist_block_t` is now a **nested, `protected`** struct
  ([include/nccl_ofi_freelist.h:289-297](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_freelist.h)),
  not a public typedef.
- There is **no `pthread_mutex_t` inside the class.** The header comments on
  `entry_alloc()` / `entry_free()` state that "the caller must ensure serialized
  access to protect the freelist" — locking is the caller's responsibility (for
  example the endpoint or communicator lock), which differs from the old
  self-locking C struct.
- The class tracks `num_in_use_entries`; when `enable_leak_detection` is set the
  destructor warns if any entries are still in use at teardown.

## Core Methods

```cpp
// Allocate a freelist item; grows the freelist on demand.
// Returns NULL if the freelist is at max size or a growth allocation failed.
fl_entry *entry_alloc();                        // [include/nccl_ofi_freelist.h:163]

// Return a freelist item to the pool. O(1) linked-list push.
void entry_free(fl_entry *entry);               // [include/nccl_ofi_freelist.h:203]

// Destructor: deregisters (if MR) and frees every block; warns on leaks.
~nccl_ofi_freelist();                           // [include/nccl_ofi_freelist.h:146]

// Internal growth routine (protected). Adds num_entries entries.
int add(size_t num_entries);                    // [include/nccl_ofi_freelist.h:245]
```

`entry_alloc()` and `entry_free()` are defined inline in the header;
`add()` and `init_internal()` live in
[src/nccl_ofi_freelist.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_freelist.cpp)
(`add` at [line 205](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_freelist.cpp)).
Note the return-type asymmetry versus the old C API: `entry_alloc()` returns a
`fl_entry *` (NULL on failure) rather than taking an out-parameter and returning
an `int`, and `entry_free()` returns `void`.

## Operation

### Allocation Flow

```
User calls: freelist->entry_alloc()            // [include/nccl_ofi_freelist.h:163]
    ↓
(Caller already holds the serializing lock)
    ↓
Check if entries available
    ↓ (if empty)
add(increase_entry_count):
    1. Allocate page-aligned memory (nccl_net_ofi_alloc_mr_buffer)
    2. Call regmr_fn to register entire block (if MR freelist)
    3. Divide block into entry_size chunks
    4. Call entry_init_fn on each entry (if provided)
    5. Link all entries to this->entries
    ↓ (if add() fails)
Return NULL
    ↓
Pop entry from this->entries
    ↓
Mark entry memory as accessible (memcheck: defined if init_fn ran, else undefined)
    ↓
num_in_use_entries++
    ↓
Return fl_entry* to user
```

**Key Optimizations**:
- **Batch Registration**: Entire block registered once, not per-entry
- **Page Alignment**: Memory rounded to page boundaries for efficient IOMMU mapping
- **Lazy Growth**: Only allocates when needed
- **Entry Padding**: Entries sized to fill pages completely (no wasted space)

### Deallocation Flow

```
User calls: freelist->entry_free(entry)        // [include/nccl_ofi_freelist.h:203]
    ↓
(Caller already holds the serializing lock)
    ↓
Push entry to head of this->entries linked list
    ↓
Mark entry memory as noaccess (memcheck)
    ↓
num_in_use_entries--
```

**Performance**: O(1) - just pointer manipulation (under the caller's lock).

### Memory Layout

```
┌─────────────────────────────────────────────────────┐
│                  Block Memory                        │
│  (page-aligned, registered with fi_mr_reg)          │  // fi_mr_reg: ([include/rdma/fi_domain.h, fi_mr_reg](https://github.com/ofiwg/libfabric/blob/main/include/rdma/fi_domain.h))
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

All examples are constructed with `new`, used via methods, and destroyed with
`delete`. The caller is responsible for serializing `entry_alloc()` /
`entry_free()`.

### Simple Freelist (Request Objects)

From `nccl_net_ofi_sendrecv.cpp` — the send/recv request freelist is created in
the recv/send comm setup (recv comm at
[src/nccl_ofi_sendrecv.cpp:1344](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_sendrecv.cpp),
send comm at
[src/nccl_ofi_sendrecv.cpp:1879](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_sendrecv.cpp)):

```cpp
// Create freelist for send/recv request structures
size_t req_size = sizeof(nccl_net_ofi_sendrecv_req_t);

r_comm->nccl_ofi_reqs_fl = new nccl_ofi_freelist(
    req_size,
    16,                       // Start with 16 requests
    16,                       // Grow by 16 when needed
    NCCL_OFI_MAX_REQUESTS,    // Max requests
    sendrecv_fl_req_entry_init,  // Per-entry init callback
    NULL,                     // No fini callback
    "Sendrecv Recv Communicator Requests",  // Debug name
    true);                    // Enable leak detection
```

Allocation and free go through `sendrecv_allocate_req()` /
`sendrecv_req_free()`
([src/nccl_ofi_sendrecv.cpp:863-881](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_sendrecv.cpp)):

```cpp
// Allocate a request (from sendrecv_allocate_req)
nccl_ofi_freelist::fl_entry *elem = fl->entry_alloc();
if (OFI_UNLIKELY(elem == NULL)) { /* freelist exhausted */ }
nccl_net_ofi_sendrecv_req *req = (nccl_net_ofi_sendrecv_req *)elem->ptr;

// ... use request ...

// Free when done (from sendrecv_req_free)
nccl_ofi_reqs_fl->entry_free(elem);

// Teardown: the comm destructor deletes the freelist
delete this->nccl_ofi_reqs_fl;   // [src/nccl_ofi_sendrecv.cpp:1044, 1798]
```

### MR-Registered Freelist (Network Buffers)

From `conn_msg_buffer_manager` in
[src/cm/nccl_ofi_cm_resources.cpp:74](https://github.com/aws/aws-ofi-nccl/blob/master/src/cm/nccl_ofi_cm_resources.cpp):

```cpp
// Create freelist for connection management buffers (pre-registered)
buff_fl = new nccl_ofi_freelist(
    buffer_size,
    16,                  // Initial count
    16,                  // Growth count
    0,                   // Unlimited
    nullptr, nullptr,    // No init/fini callbacks
    endpoint::reg_mr,    // Register with fi_mr_reg
    endpoint::dereg_mr,  // Deregister with fi_close
    &ep,                 // Opaque data passed to reg_mr
    1,                   // Entry alignment
    "Connection Message Buffer",  // Debug name
    true);               // Enable leak detection

// Allocate buffer (already registered!) — from allocate_conn_msg()
nccl_ofi_freelist::fl_entry &elem = *(buff_fl->entry_alloc());
void *buffer = elem.ptr;
void *mr_handle = elem.mr_handle;  // Use for fi_mr_desc(mr_handle)

// ... use in fi_send without separate registration ...

// Return the buffer — from free_conn_msg()
buff_fl->entry_free(&elem);

// Teardown
delete buff_fl;          // [src/cm/nccl_ofi_cm_resources.cpp:82]
```

### GIN Metadata / RX Buffer Freelist

From `nccl_ofi_gin_resources.cpp` — the RX buffer freelist for GPU-initiated
NCCL ([src/rdma/gin/nccl_ofi_gin_resources.cpp:466](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_resources.cpp)):

```cpp
rx_buff_fl_tmp = new nccl_ofi_freelist(
    sizeof(nccl_net_ofi_gin_signal_metadata_msg_t),
    num_buffers,              // Initial count
    0,                        // No growth (fixed pool)
    num_buffers,              // Max = initial
    nullptr, nullptr,
    gin_ep.freelist_regmr_fn,
    gin_ep.freelist_deregmr_fn,
    &gin_ep,
    1,                        // Entry alignment
    "GIN Rx Buffers",
    true);
this->rx_buff_fl.reset(rx_buff_fl_tmp);   // owned by a std::unique_ptr
```

The GIN request freelist is a *simple* (non-MR) freelist:
```cpp
// [src/rdma/gin/nccl_ofi_gin_resources.cpp:487]
req_fl_tmp = new nccl_ofi_freelist(
    sizeof(nccl_net_ofi_gin_union_req), 1024, 1024, 0,
    nullptr, nullptr, "GIN Requests", true);
this->req_fl.reset(req_fl_tmp);
```

## Key Freelist Instances

Entry-size and location values verified against the tree at aws-ofi-nccl master.

| Freelist | Type | Entry Size | Used For | Location |
|----------|------|------------|----------|----------|
| `nccl_ofi_reqs_fl` (sendrecv) | Simple | `sizeof(nccl_net_ofi_sendrecv_req_t)` | Send/recv request tracking | nccl_ofi_sendrecv.cpp:1344 (recv), :1879 (send) |
| `nccl_ofi_reqs_fl` (rdma) | Simple | `rdma_req_max_subclass_size` | RDMA request objects (placement new) | nccl_ofi_rdma.cpp:4909 (recv), :6349 (send) |
| `schedule_fl` | Simple | `sizeof_schedule(num_rails)` | Multi-rail scheduling | nccl_ofi_scheduler.cpp:194 |
| `buff_fl` | MR | `buffer_size` (cm msg) | Connection mgmt buffers | cm/nccl_ofi_cm_resources.cpp:74 |
| `rx_buff_reqs_fl` | Simple | `rdma_req_max_subclass_size` | RDMA rx-buffer request objects | nccl_ofi_rdma.cpp:6019 |
| `ctrl_rx_buff_fl` / `eager_rx_buff_fl` | MR | `ctrl_rx_buff_size` / `eager_rx_buff_size` | RDMA control / eager rx buffers | nccl_ofi_rdma.cpp:6034, :6044 |
| `rx_buff_fl` | MR | `sizeof(gin_signal_metadata_msg_t)` | GIN receive buffers | rdma/gin/nccl_ofi_gin_resources.cpp:466 |
| `ack_send_fl` | MR | `sizeof(gin_ack_msg_t)` | GIN ACK send buffers | rdma/gin/nccl_ofi_gin_resources.cpp:474 |
| `req_fl` (gin) | Simple | `sizeof(nccl_net_ofi_gin_union_req)` | GIN request objects | rdma/gin/nccl_ofi_gin_resources.cpp:487 |

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

The freelist class is **not internally locked**. The header explicitly states
for both `entry_alloc()` and `entry_free()` that "the caller must ensure
serialized access to protect the freelist." This is a change from the old C
struct, which embedded a `pthread_mutex_t`. Serialization is provided by the
owning component's lock (for example the endpoint or communicator lock).

- Growth (`add()`, including new block allocation + registration) runs under
  the caller's lock, since it happens inside `entry_alloc()`.
- The `regmr_fn()` / `deregmr_fn()` callbacks are invoked while the caller's
  external lock is held. The header warns that the callback **must not** try to
  acquire that same lock.

**Deadlock Risk**: If `regmr_fn` tries to acquire a lock that the caller already
holds while calling freelist functions, deadlock can occur. Keep `regmr_fn`
simple.

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

Today allocation runs under the *caller's* external lock. A lock-free freelist
would let the transport drop that serialization on the fast path:
```cpp
// Current: caller-serialized linked-list pop (inside entry_alloc)
entry = this->entries;
this->entries = entry->next;

// Proposed: lock-free CAS
do {
    entry = atomic_load(&this->entries);
    if (!entry) break;
} while (!atomic_compare_exchange(&this->entries, &entry, entry->next));
```

**Challenges**:
- Growth still needs synchronization
- ABA problem requires versioned pointers

### P2 Optimization: Per-Thread Caches
**Impact**: Estimated 5-10x faster allocation (~5-10 ns)

Thread-local cache reduces contention:
```cpp
thread_local nccl_ofi_freelist::fl_entry *local_cache[16];
thread_local int cache_count = 0;

// Allocate from local cache first
if (cache_count > 0) {
    return local_cache[--cache_count];  // No lock!
}
// Fall back to shared freelist
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
// Debugging helper (would be a member of nccl_ofi_freelist)
void nccl_ofi_freelist_stats(nccl_ofi_freelist *freelist) {
    printf("Freelist %p:\n", freelist);
    printf("  Entry size: %zu\n", freelist->entry_size);
    printf("  Allocated: %zu / %zu\n",
           freelist->num_allocated_entries,
           freelist->max_entry_count);
    printf("  In use: %zu\n", freelist->num_in_use_entries);

    // Count free entries (entries is a nccl_ofi_freelist::fl_entry *)
    size_t free_count = 0;
    for (auto *e = freelist->entries; e; e = e->next) free_count++;
    printf("  Free: %zu\n", free_count);
}
```

(Note that `entry_size`, `entries`, `num_allocated_entries` and
`num_in_use_entries` are `protected` members, so such a helper would have to be
a member/friend of `nccl_ofi_freelist` rather than a free function.)

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
3. Not internally locked — the caller serializes access (contention possible under high concurrency)
4. Tuning parameters critical for memory/performance balance
5. Excellent target for lock-free optimization
6. RAII class: `new` constructs and allocates the first block (throws `std::runtime_error` on initial-allocation failure); `delete` deregisters and frees; copy is deleted

## Former C API (removed)

Earlier releases exposed the freelist as a C struct (`nccl_ofi_freelist_t`) with
free functions and an embedded `pthread_mutex_t`. Those free functions and the
`nccl_ofi_freelist_t` / `nccl_ofi_freelist_elem_t` typedefs **no longer exist**
— the freelist is now the `nccl_ofi_freelist` C++ class. This section is kept to
help when reading older code, commits, or blog posts. The **data structure and
the block-growth algorithm are unchanged**; only the interface changed.

| Former C API (removed) | Current C++ equivalent |
|------------------------|------------------------|
| `nccl_ofi_freelist_init(entry_size, init, incr, max, init_fn, fini_fn, &fl)` → `int` | Simple constructor `nccl_ofi_freelist(entry_size, init, incr, max, init_fn, fini_fn, name, enable_leak_detection)` |
| `nccl_ofi_freelist_init_mr(..., regmr_fn, deregmr_fn, regmr_opaque, entry_alignment, &fl)` → `int` | Registered constructor `nccl_ofi_freelist(..., regmr_fn, deregmr_fn, regmr_opaque, entry_alignment, name, enable_leak_detection)` |
| `nccl_ofi_freelist_fini(fl)` / manual free | Destructor `~nccl_ofi_freelist()` (RAII: deregisters + frees) |
| `nccl_ofi_freelist_entry_alloc(fl)` → `nccl_ofi_freelist_elem_t *` | Method `fl_entry *entry_alloc()` |
| `nccl_ofi_freelist_entry_free(fl, elem)` | Method `void entry_free(fl_entry *entry)` |
| `nccl_ofi_freelist_t` (typedef) | `class nccl_ofi_freelist` |
| `nccl_ofi_freelist_elem_t` (typedef) | nested `nccl_ofi_freelist::fl_entry` |
| `struct nccl_ofi_freelist_block_t` (public) | nested, `protected` `nccl_ofi_freelist::nccl_ofi_freelist_block_t` |
| `pthread_mutex_t lock` (internal) | No internal lock — caller serializes `entry_alloc`/`entry_free` |

Semantic changes to note:
- The two init functions returned an `int` and wrote the freelist through an
  out-parameter. The constructors cannot return a code, so an initial-allocation
  failure is raised as `std::runtime_error` from `init_internal()`. Callers use
  `new` (optionally in a `try`/`catch`) and `delete`.
- New required parameters on both constructors: `name` (debug string that must
  outlive the freelist) and `enable_leak_detection`.
- `entry_alloc()` now returns the `fl_entry *` directly (NULL on
  exhaustion/growth failure) instead of an out-parameter + `int`; `entry_free()`
  returns `void`.
- The two constructors share `init_internal()`
  ([src/nccl_ofi_freelist.cpp:19](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_freelist.cpp)),
  which sets `have_reginfo` and (for the MR ctor) records the registration
  callbacks — memory registration is driven from within growth (`add()`), not a
  separate init call.

## RDMA Request Objects: Placement New out of the Freelist

The RDMA transport changed how it builds request objects, and this changes how
the freelist entry size is computed. This is important context for anyone
reasoning about the RDMA request lifecycle. (Relevant upstream commits:
**f6f6223**, **960cb56**, **5b6d45f**, **1b5b830**, **8d2b9f0**, **cc7afb2**,
**4942f12**, **aebf544**, **1b513c4**.)

### From a tagged union to a class hierarchy

Previously an RDMA request was a single base `struct` carrying a `union` of
per-type data (send/recv/flush/…). It is now a **base class with a subclass per
request type**
([include/nccl_ofi_rdma.h:863-1096](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_rdma.h)):

```cpp
class rdma_send_req        : public nccl_net_ofi_rdma_req { ... };  // :863
class rdma_recv_req        : public nccl_net_ofi_rdma_req { ... };  // :934
class rdma_flush_req       : public nccl_net_ofi_rdma_req { ... };  // :963
class rdma_rma_op_req      : public nccl_net_ofi_rdma_req { ... };  // :983
class rdma_rx_buff_req     : public nccl_net_ofi_rdma_req { ... };  // :1019
class rdma_send_close_req  : public nccl_net_ofi_rdma_req { ... };  // :1048
class rdma_eager_copy_req  : public nccl_net_ofi_rdma_req { ... };  // :1068
class rdma_recv_segms_req  : public nccl_net_ofi_rdma_req { ... };  // :1085
```

Each subclass overrides virtual methods such as `free()` and `post()`, so the
type carries both its data and its behavior (no more `switch` on a type tag into
a union member).

### Placement new into freelist storage

Because the freelist owns the raw memory, requests are constructed **in place**
with C++ placement new directly on the freelist element's buffer, rather than
allocated with `new`/`malloc`. Every request allocation site follows the same
pattern (all in
[src/nccl_ofi_rdma.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_rdma.cpp)):

```cpp
// e.g. send request (nccl_ofi_rdma.cpp:5392)
nccl_net_ofi_rdma_req *req = new (elem->ptr) rdma_send_req();
req->elem = elem;   // remember the freelist element for later free
```

Other sites: `rdma_eager_copy_req` (:701), `rdma_rx_buff_req` EAGER (:1800) and
CTRL (:1831), `rdma_recv_segms_req` (:3378), `rdma_recv_req` (:3417),
`rdma_send_close_req` (:3967), `rdma_flush_req` (:4378), and
`rdma_rma_op_req` READ (:4616) / WRITE (:5364).

**Symmetric teardown.** Because construction is a placement new, destruction must
be an explicit destructor call *before* the block goes back to the freelist —
`entry_free()` does not (and must not) run a destructor. The shared cleanup path
does exactly this
([nccl_ofi_rdma.cpp:1600-1610](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_rdma.cpp)):

```cpp
/* Call the virtual destructor before returning the block to the freelist.
 * Construction happens per-allocation via placement new, so destruction
 * must also happen per-deallocation. */
req->~nccl_net_ofi_rdma_req();      // virtual: dispatches to the real subclass
nccl_ofi_reqs_fl->entry_free(elem); // raw block returns to the pool
```

The destructor is virtual, so calling it through the base pointer runs the
correct subclass destructor.

### Sizing implication: one slot fits the largest subclass

Since any subclass can be placement-new'd into any freelist slot, every slot
must be large enough for the **largest** subclass. The entry size is therefore
computed as the max over all subclass sizes
([include/nccl_ofi_rdma.h:1096-1108](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_rdma.h)):

```cpp
/* Maximum size across all request subclasses.  Used as the freelist
 * element size so that any subclass can be placement-new'd into any
 * slot. */
static constexpr size_t rdma_req_max_subclass_size = std::max({
    sizeof(rdma_send_req),      sizeof(rdma_recv_req),
    sizeof(rdma_flush_req),     sizeof(rdma_rma_op_req),
    sizeof(rdma_rx_buff_req),   sizeof(rdma_send_close_req),
    sizeof(rdma_eager_copy_req), sizeof(rdma_recv_segms_req)
});
```

The single request freelist is created with `entry_size =
rdma_req_max_subclass_size`. Compared with the old tagged-union struct (whose
size was already the max of its variants by virtue of being a union), the
memory footprint per slot is essentially the same — the change is in *type
safety and dispatch* (real subclasses + virtual methods) rather than in layout.
The freelist itself is unchanged; it still hands out fixed-size raw blocks, and
the transport is now responsible for constructing/destructing the right subclass
in each block.

The request freelists that use this size are created in the RDMA comm/endpoint
setup: the recv-comm and send-comm request pools
([src/nccl_ofi_rdma.cpp:4909](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_rdma.cpp)
and [:6349](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_rdma.cpp))
and the endpoint rx-buffer request pool
([src/nccl_ofi_rdma.cpp:6019](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_rdma.cpp)),
each passing `rdma_req_max_subclass_size` as the freelist `entry_size`. These
request pools use per-entry init/fini callbacks (`rdma_fl_req_entry_init` /
`rdma_fl_req_entry_fini`), distinct from the placement new/destructor that the
allocation sites above perform on each individual request.
