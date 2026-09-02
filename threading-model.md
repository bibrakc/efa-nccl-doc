# Threading Model and Concurrency

## Overview

The NCCL + OFI + libfabric + EFA stack involves multiple threading layers that interact. Understanding these is critical for optimization and debugging.

```
┌──────────────────────────────────────────┐
│     Application Threads                  │
│     (PyTorch DataParallel workers)       │
└─────────────┬────────────────────────────┘
              │ Per-thread CUDA streams
┌─────────────▼────────────────────────────┐
│     NCCL Library                         │
│  ┌────────────────────────────────┐     │
│  │  Service Thread (optional)     │     │
│  │  - Async progress              │     │
│  ├────────────────────────────────┤     │
│  │  Proxy Threads (per comm)      │     │
│  │  - Network I/O                 │     │
│  │  - Per-channel threads         │     │
│  └────────────────────────────────┘     │
└─────────────┬────────────────────────────┘
              │ Plugin API calls
┌─────────────▼────────────────────────────┐
│     OFI Plugin                           │
│  - No additional threads                 │
│  - Inherits caller's thread context      │
└─────────────┬────────────────────────────┘
              │ Libfabric API calls
┌─────────────▼────────────────────────────┐
│     Libfabric (EFA Provider)             │
│  - Thread-safe operations                │
│  - Manual progress model                 │
└─────────────┬────────────────────────────┘
              │ System calls
┌─────────────▼────────────────────────────┐
│     EFA Kernel Driver                    │
│  - Kernel threads for interrupts         │
│  - Completion handling                   │
└──────────────────────────────────────────┘
```

## NCCL Threading

### Main Application Threads

```c
// Application thread per GPU
void worker_thread(int gpu_id) {
  cudaSetDevice(gpu_id);

  ncclComm_t comm;
  ncclCommInitRank(&comm, nranks, commId, gpu_id);

  // Issue collective on CUDA stream
  ncclAllReduce(sendbuf, recvbuf, count, ncclFloat,
                ncclSum, comm, stream);

  cudaStreamSynchronize(stream);
}
```

**Characteristics:**
- One thread per GPU (typical)
- Each thread has its own communicator
- Operations enqueued to CUDA streams
- Async with respect to CPU

### NCCL Proxy Threads

NCCL creates proxy threads to handle network I/O:

```c
// Simplified proxy thread structure
struct ncclProxyOp {
  int type;  // SEND, RECV, etc.
  void* buffer;
  size_t size;
  int peer;
  void* request;
};

void* ncclProxyThread(void* args) {
  struct ncclComm* comm = args;

  while (!comm->proxyState.stop) {
    // 1. Check for new operations from GPU
    poll_gpu_signals();

    // 2. Process pending network operations
    for (op in pending_ops) {
      switch (op->state) {
        case OP_SEND:
          // Call into OFI plugin
          ncclNet->isend(op->sendComm, op->buffer,
                         op->size, ...);
          break;

        case OP_RECV:
          ncclNet->irecv(op->recvComm, ...);
          break;

        case OP_TEST:
          // Poll for completion
          ncclNet->test(op->request, &done, &size);
          if (done) {
            signal_gpu_completion(op);
            free_op(op);
          }
          break;
      }
    }

    // 3. Yield if idle
    if (no_work) {
      usleep(1);  // or sched_yield()
    }
  }
}
```

**Proxy Thread Properties:**
- **Count**: Typically 1-2 proxy threads per communicator
- **Purpose**: Handle network operations (CPU-driven)
- **Affinity**: Can be pinned to specific CPUs
- **Priority**: Can be elevated for lower latency

**Why Proxies:**
- Network operations require CPU
- GPU kernels focus on compute
- Enables async overlap of compute and communication
- Better CPU utilization

### Proxy Thread Lifecycle

```c
// Communicator initialization
ncclCommInitRank() {
  // ... setup ...

  // Create proxy threads
  for (int i = 0; i < nProxyThreads; i++) {
    pthread_create(&comm->proxyThreads[i],
                   NULL,
                   ncclProxyThread,
                   comm);

    // Set affinity if configured
    if (comm->cpuAffinity[i] >= 0) {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(comm->cpuAffinity[i], &cpuset);
      pthread_setaffinity_np(comm->proxyThreads[i],
                             sizeof(cpuset), &cpuset);
    }
  }
}

// Communicator cleanup
ncclCommDestroy() {
  // Signal proxies to stop
  comm->proxyState.stop = 1;

  // Wait for completion
  for (int i = 0; i < nProxyThreads; i++) {
    pthread_join(comm->proxyThreads[i], NULL);
  }
}
```

### Proxy Thread Tuning

> **There are no `NCCL_NPROXY_THREADS`, `NCCL_PROXY_CPU_AFFINITY` or
> `NCCL_PROXY_SCHED_PRIORITY` environment variables.** Earlier revisions of this document
> listed them; none exist in the NCCL source (verified against `src/`: the only
> `PROXY`-prefixed params are `NCCL_PROXY_APPEND_BATCH_SIZE` and `NCCL_PROXY_DUMP_SIGNAL`).
> NCCL sizes its proxy threads internally from the channel/peer topology; count, affinity
> and priority are controlled from outside NCCL.

```bash
# Real proxy-related NCCL knobs (src/proxy.cc)
NCCL_PROXY_APPEND_BATCH_SIZE=16   # ops appended per proxy wakeup (default 16)
NCCL_PROXY_DUMP_SIGNAL=-1         # signal number that dumps proxy state (default -1, off)

# GIN host-proxy thread count (src/gin/gin_host.cc, default 1) -- note the plain
# NCCL_ prefix; the OFI plugin reads the same variable name
NCCL_GIN_PROXY_NTHREADS=4
NCCL_GIN_PROXY_QUEUE_SIZE=-1      # src/gin/gin_host_proxy.cc, default -1 (auto)
NCCL_GIN_PROXY_POLL_BATCH=32      # src/gin/gin_host_proxy.cc, default 32

# Whether NCCL honors the CPU affinity it inherited from the launcher
NCCL_IGNORE_CPU_AFFINITY=0        # default 0 (honor it)
```

CPU placement is done by the launcher or by `taskset`/`numactl`, not by NCCL. For this
stack the aws-ofi-nccl guidance is to give every process a mask covering **all** CPUs
(`mpirun --bind-to none`, or `srun --cpu-bind=none` with enough processors per task),
because the plugin and proxy threads need freedom to run near the right NIC.

### NCCL Service Thread

Optional global service thread:

```c
void* ncclServiceThread(void* args) {
  while (!stopService) {
    // Monitor all communicators
    for (comm in allComms) {
      // Check for errors
      check_async_errors(comm);

      // Progress operations
      progress_comm(comm);
    }

    usleep(1000);  // 1ms
  }
}
```

**When Used:**
- Asynchronous error reporting via the `ncclCommGetAsyncError()` API (there is no
  `NCCL_ASYNC_ERROR_HANDLING` variable in NCCL; the similarly named
  `TORCH_NCCL_ASYNC_ERROR_HANDLING` belongs to PyTorch, not NCCL)
- Centralized error monitoring
- Background progress

### GPU-CPU Synchronization

NCCL kernels signal proxy threads:

```
GPU Kernel                      Proxy Thread
    │                               │
    ├─ Prepare send buffer          │
    ├─ Write flag to mapped mem ────→ (poll)
    │                               │
    │                               ├─ Detect flag
    │                               ├─ Issue fi_send()
    │                               ├─ Poll completion
    │                               │
    │←──── Write completion flag ───┤
    │                               │
    ├─ Detect completion            │
    └─ Continue                     │
```

**Mechanisms:**
- **Flags in mapped memory**: GPU writes, CPU polls
- **PCIe reads**: Low latency signaling
- **Atomics**: For synchronization

## OFI Plugin Threading

### Thread Model

The OFI plugin is **thread-safe but stateless for threading**:

```c
ncclResult_t nccl_net_ofi_isend(...) {
  // Called from NCCL proxy thread

  // No internal threads
  // Just forwards to libfabric

  fi_send(comm->ep, ...);

  return ncclSuccess;
}
```

**Properties:**
- No plugin-specific threads
- Inherits caller's thread context (NCCL proxy)
- Thread-safety via libfabric's guarantees
- Locking for internal state (MR cache, etc.)

### Critical Sections

```c
// Example: MR cache access
pthread_mutex_t mr_cache_lock;

ncclResult_t nccl_net_ofi_regMr(...) {
  pthread_mutex_lock(&mr_cache_lock);

  // Search cache
  mr = find_in_cache(addr, size);
  if (!mr) {
    // Register new
    fi_mr_reg(...);
    add_to_cache(mr);
  }

  pthread_mutex_unlock(&mr_cache_lock);

  return ncclSuccess;
}
```

**Locking Granularity:**
- Coarse locks for caches
- Per-connection state usually lock-free
- Fine-grained for completion queues

### Concurrency Patterns

Multiple proxy threads may call into plugin simultaneously:

```
Proxy Thread 0              Proxy Thread 1
     │                           │
     ├─ isend(conn_0, ...)       │
     │   ├─ fi_send(ep_0, ...)   │
     │   └─ return               │
     │                           ├─ isend(conn_1, ...)
     │                           │   ├─ fi_send(ep_1, ...)
     │                           │   └─ return
     │                           │
     ├─ test(req_0, ...)         │
     │   ├─ fi_cq_read(cq_0, ...)│
     │   └─ return               │
     │                           ├─ test(req_1, ...)
     │                           │   ├─ fi_cq_read(cq_1, ...)
     │                           │   └─ return
```

**Safety:**
- Different endpoints/CQs: naturally parallel
- Shared resources (caches): protected by locks
- Libfabric handles internal concurrency

## Libfabric Threading

### Thread Safety Model

The EFA provider supports **three** `FI_THREAD_*` models, not just one. As of
libfabric 2.7 (`man/fi_efa.7.md`, "Threading" section), both the RDM and DGRAM
endpoints advertise:

- **`FI_THREAD_SAFE`** — full internal serialization; any thread may call any
  object concurrently.
- **`FI_THREAD_COMPLETION`** — the application guarantees that a given
  completion domain (a CQ and the endpoints bound to it) is only ever driven by
  one thread at a time. The provider then elides most internal locking on the
  data path. This is the model the aws-ofi-nccl RDMA transport requests, because
  each proxy thread owns its own endpoint/CQ.
- **`FI_THREAD_DOMAIN`** — the application serializes all access to a domain;
  the strongest promise and the least locking.

```c
// man/fi_efa.7.md, "Threading":
//   Both RDM and DGRAM endpoints supports FI_THREAD_SAFE,
//   FI_THREAD_COMPLETION, FI_THREAD_DOMAIN.
```

The lock type used for a given resource is *selected from the requested
threading + progress model* rather than being fixed. For example the
shared-receive-queue lock is initialized with
`efa_domain_data_progress_lock_type()`
([prov/efa/src/rdm/efa_rdm_ep_fiops.c:492](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/rdm/efa_rdm_ep_fiops.c)),
which resolves to `OFI_LOCK_NONE` under `FI_THREAD_COMPLETION` — turning the
critical section into a no-op when the application has promised single-threaded
completion-domain access. See the "EFA Provider Locking Rework" section below
for the full picture.

> **Correction (was: "EFA provider: FI_THREAD_SAFE" only).** Older revisions of
> this document claimed the EFA provider only supported `FI_THREAD_SAFE`. That
> was never fully true and is now explicitly wrong: all three models above are
> advertised.

### Progress Model

EFA provider: **`FI_PROGRESS_MANUAL`** for control and data progress on the RDM
endpoint. Under specific conditions the RDM endpoint can also support
`FI_PROGRESS_AUTO` (see `man/fi_efa.7.md:88-90`).

```c
struct fi_domain_attr {
  enum fi_progress control_progress = FI_PROGRESS_MANUAL;
  enum fi_progress data_progress = FI_PROGRESS_MANUAL;
  // ...
};
```

**Manual Progress:**
- Application must poll CQs
- No background threads in libfabric
- Calling `fi_cq_read()` drives progress

**Implications:**
- NCCL proxy threads must poll frequently
- No progress without polling
- CPU overhead proportional to poll rate

**`FI_PROGRESS_CONTROL_UNIFIED`:** When the domain is opened with unified
control progress, the `util_domain` lock is made a **no-op**
(commit cf0c4142e). This removes a domain-wide lock from the control path when
the application guarantees a single progress context, complementing the
per-endpoint locking rework below.

### Locking in EFA Provider

```c
// Example: Endpoint operations
int efa_rdm_send(...) {
  struct efa_rdm_ep *ep = ...;

  // Lock endpoint for consistency
  fastlock_acquire(&ep->lock);

  // Allocate TX entry
  txe = get_txe(ep);

  // Post to hardware
  ibv_post_send(ep->qp, &wr, ...);

  // Update state
  ep->tx_pending++;

  fastlock_release(&ep->lock);
}
```

**Locking Strategy:**
- Per-endpoint locks (not global)
- Fast locks (spinlocks for short critical sections)
- Lock-free data structures where possible

### Completion Queue Concurrency

```c
// Multiple threads polling same CQ
Thread 0:                   Thread 1:
fi_cq_read(cq, ...)         fi_cq_read(cq, ...)
  │                           │
  ├─ Lock CQ                  ├─ Wait for lock
  ├─ Poll hardware            │
  ├─ Return completions       │
  └─ Unlock                   └─ Lock, poll, unlock
```

**CQ Thread Safety:**
- Serialized access via locks
- Completions distributed to threads
- Each thread gets different entries

**NCCL typically:**
- One proxy thread per CQ
- Avoids contention
- Better cache locality

## EFA Driver Threading

### Kernel Threads

The EFA driver uses kernel threads:

```
User Space (NCCL proxy)
    │ fi_send()
    │ ioctl/write
    ▼
Kernel Space (EFA driver)
    │
    ├─ Work queue thread
    │  - Process send requests
    │  - Build packets
    │  - DMA setup
    │
    ├─ Completion thread
    │  - Poll hardware for completions
    │  - Update CQ
    │  - Wake user space
    │
    └─ Interrupt handler
       - Handle NIC interrupts
       - Schedule completions
```

**Driver Threads:**
- **Work submission**: Process user requests
- **Completion handling**: Poll hardware, update CQs
- **Interrupt handling**: Async events

### User-Kernel Interaction

```c
// NCCL proxy calls fi_send()
fi_send() {
  // Libfabric EFA provider
  ibv_post_send() {
    // Write to MMIO ring
    mmio_write(qp->sq_db, wqe);

    // Or ioctl for some ops
    ioctl(ctx->cmd_fd, IBV_POST_SEND, &cmd);
  }
}

// Kernel driver
efa_post_send() {
  // Process request
  build_wqe(qp, wr);

  // DMA setup
  setup_dma(qp, wr);

  // Ring doorbell to hardware
  writel(qp->sq_tail, qp->sq_db);
}
```

**Mechanisms:**
- **MMIO (Memory-Mapped I/O)**: Fast path for doorbells
- **ioctl**: Control operations
- **mmap**: Share memory between user/kernel

### Completion Polling

```c
// NCCL proxy polls
fi_cq_read() {
  // Libfabric
  ibv_poll_cq() {
    // Read completion queue (shared memory)
    cqe = cq->buf[cq->cons_index];

    if (cqe->owner != expected_owner) {
      return 0;  // No completion
    }

    // Parse completion
    parse_cqe(cqe, &wc);

    // Advance consumer index
    cq->cons_index++;

    return 1;  // Got completion
  }
}
```

**Polling Path:**
- **User space reads CQ directly** (shared memory)
- No kernel crossing for completions
- Very fast (~100-200 ns per poll)

### Interrupt vs Polling

```
Polling (typical for NCCL):
  User space ─► Poll CQ ─► Get completion
  (no kernel involved)

Interrupt (rare):
  Hardware ─► Interrupt ─► Kernel ─► Wake user
  (higher latency, but CPU efficient for idle)
```

**NCCL uses polling:**
- Lower latency (~10 μs vs ~50 μs)
- Higher CPU usage
- Acceptable for active communication phases

## Concurrency Bottlenecks

### Lock Contention

```
Issue: Multiple proxy threads → Same MR cache

Proxy 0 ──┐
Proxy 1 ──┼─→ [MR Cache Lock] ──→ Serialized
Proxy 2 ──┘

Solution:
- Lock-free cache (RCU)
- Per-thread caches
- Finer-grained locking
```

### False Sharing

```
Issue: Proxy threads accessing nearby memory

struct {
  int proxy0_counter;  // Cache line 0
  int proxy1_counter;  // Cache line 0 (false sharing!)
};

Solution:
- Pad to cache line boundaries (64 bytes)
- Per-thread data structures
```

```c
struct alignas(64) proxy_state {
  int counter;
  char pad[60];  // Force separate cache lines
};
```

### CQ Polling Overhead

```
Issue: Frequent polling → High CPU usage

while (1) {
  fi_cq_read(cq, ...);  // 10M+ calls/sec
}

Solutions:
- Adaptive polling (back off when idle)
- Batch operations
- Sleep when no work (trade latency for CPU)
```

## Optimization Strategies

### CPU Affinity

NCCL exposes no variable for pinning proxy threads. Placement is done by the launcher or
by `taskset`/`numactl` at process level, and NCCL threads inherit that mask.

```bash
# Give the process a mask that spans the CPUs you want NCCL to use.
# aws-ofi-nccl recommends NOT narrowly binding: every process should see all CPUs.
mpirun --bind-to none ./app
srun  --cpu-bind=none ./app

# Narrow placement (only if you have measured a benefit), at process level:
numactl --cpunodebind=0 --membind=0 ./app
taskset -c 0-31 ./app

# Let NCCL ignore an inherited mask it considers harmful:
NCCL_IGNORE_CPU_AFFINITY=1        # default 0
```

**Benefits:**
- Reduced context switching
- Better cache locality
- Predictable performance

### NUMA Awareness

```c
// Allocate buffers on same NUMA node as NIC
int nic_numa_node = get_nic_numa_node(dev);

void* buf = numa_alloc_onnode(size, nic_numa_node);
```

**Benefits:**
- Lower memory latency
- Higher bandwidth
- Important for multi-socket systems

### Priority Tuning

NCCL has no scheduling-priority variable either. Adjust the whole process:

```bash
# Real-time scheduling for the process (requires privileges; use with care --
# a runaway SCHED_FIFO thread can starve the system)
chrt --fifo 50 ./nccl_app

# Or just raise its nice level
nice -n -20 ./nccl_app
```

**Trade-offs:**
- Lower latency for network ops
- May starve other processes
- Use carefully in shared systems

### Polling vs Sleeping

```c
// Adaptive polling
int idle_count = 0;

while (1) {
  ret = fi_cq_read(cq, ...);

  if (ret > 0) {
    // Got work
    idle_count = 0;
    process_completions();
  } else {
    // No work
    idle_count++;

    if (idle_count > THRESHOLD) {
      // Back off
      usleep(1);  // or nanosleep()
    }
  }
}
```

**Benefits:**
- Low latency when active
- Lower CPU when idle
- Configurable threshold

## Debugging Threading Issues

### Deadlocks

```bash
# Enable NCCL debug
NCCL_DEBUG=TRACE
NCCL_DEBUG_SUBSYS=INIT,COLL,PROXY

# Attach debugger to hung process
gdb -p <pid>
(gdb) thread apply all bt  # Backtrace all threads
```

**Common causes:**
- Lock ordering violations
- Proxy thread waiting on GPU, GPU waiting on proxy
- Communicator destruction while ops in flight

### Race Conditions

```bash
# Thread sanitizer (compile-time)
gcc -fsanitize=thread ...

# Runtime detection
export TSAN_OPTIONS=second_deadlock_stack=1

./nccl_app
```

### Performance Profiling

```bash
# CPU profiling
perf record -g -p <pid>
perf report

# Look for:
# - Lock contention (in futex, pthread_mutex_lock)
# - Polling overhead (in fi_cq_read, ibv_poll_cq)
# - Context switches
```

## Summary

**Threading Architecture:**
```
App Threads (1 per GPU)
  ↓ Enqueue to CUDA streams
GPU Kernels (async)
  ↓ Signal via flags
Proxy Threads (1-2 per comm)
  ↓ Call OFI plugin
OFI Plugin (no threads)
  ↓ Call libfabric
Libfabric (thread-safe, manual progress)
  ↓ System calls
EFA Driver (kernel threads)
  ↓ Hardware
```

**Key Points:**
1. **NCCL proxy threads** drive network I/O
2. **Manual progress** requires frequent polling
3. **Thread safety** via locks in plugin/libfabric
4. **No background threads** in libfabric/plugin
5. **CPU affinity** critical for performance
6. **Polling vs sleeping** trade-off

**Optimization Checklist:**
- [ ] Pin proxy threads to dedicated CPUs
- [ ] Avoid NUMA remote accesses
- [ ] Tune polling frequency
- [ ] Profile for lock contention
- [ ] Monitor context switches
- [ ] Consider real-time priorities for latency

**Next**: Deep dive into RDMA and memory registration mechanisms.

## OFI Plugin Lock-Free Destruction (April 2026 Update)

The OFI plugin uses `shared_ptr`/`weak_ptr` for object lifetime management.
This eliminates the previous lock ordering issues during destruction:

### Previous Model (Manual Ref Counting)

```
  release_ep() held domain_lock
    └── cascaded to release_domain() which needed device_lock
        └── Lock ordering problem (domain_lock → device_lock)
        └── Required skip_lock / force_cleanup boolean hacks
```

### Current Model (Smart Pointers)

```
  Destruction cascade (no locks):
    comm destroyed → shared_ptr<ep> drops → ep destructor runs
      → shared_ptr<domain> drops → domain destructor runs
      → No locks needed at any point in the cascade

  Locking only during lookup (get_ep, get_domain):
    device_lock: protects domain_table during get_domain()
    domain_lock: protects ep_table during get_ep()
    get_ep() releases device_lock before calling domain->get_ep()
    → No two locks held simultaneously
```

**Key improvement:** The domain_lock and device_lock are never held during
object destruction. They are only held during table lookups (creating/finding
endpoints and domains), which are on the connection setup path, not the data path.

**Source:** [src/nccl_ofi_net.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_net.cpp) — `get_domain()`, `get_ep()`

## EFA Provider Locking Rework (2026, libfabric 2.7)

The EFA provider underwent a large locking overhaul to make the data path scale
across threads under `FI_THREAD_COMPLETION`. The theme is the same throughout:
move state that used to live on the shared *domain* down to the *endpoint* or
*CQ* so that threads owning different endpoints never contend, and replace
locked lookups with lock-free reads where the access pattern allows it.

### Per-endpoint, fi_addr-indexed peer map with lock-free reads

Each RDM endpoint now owns its own peer map instead of consulting a shared,
locked structure:

```c
// prov/efa/src/rdm/efa_rdm_ep.h:155-156
struct efa_av_array *fi_addr_to_peer_map;
struct efa_av_array *fi_addr_to_peer_map_implicit;
```

These are created in `efa_rdm_ep_construct()`
([efa_rdm_ep_fiops.c:282-286](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/rdm/efa_rdm_ep_fiops.c))
and looked up on the data path via
`efa_rdm_ep_peer_map_lookup()`
([efa_rdm_ep_utils.c:54](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/rdm/efa_rdm_ep_utils.c)),
which is a lock-free `efa_av_array_at()` call. The map is indexed directly by
`fi_addr_t`. (Relevant commits: 121baf6ec, 4a706bcdd, 2144bfd18.)

### The lock-free `efa_av_array`

The map above is built on a new data structure,
`struct efa_av_array`
([prov/efa/src/efa_av_array.c](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_av_array.c),
[efa_av_array.h](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_av_array.h);
commits 2dc25a8d9, 53a0a5118). It is a pointer array indexed by a `uint64_t`
in `[0, max_idx]`:

```
┌───────────────────────────────────────────────────────────┐
│ inline_entries[0 .. inline_size)   ← direct indexing       │
│   (INLINE_SIZE = 8192)                                      │
├───────────────────────────────────────────────────────────┤
│ chunk_table[]  → chunk (CHUNK_SIZE = 8192 entries) ...      │
│   sized once at creation, never reallocated                │
│   chunks allocated on first use, never freed/moved before  │
│   destroy                                                   │
└───────────────────────────────────────────────────────────┘
```

**Concurrency model: single writer, any number of lock-free readers.**
- `efa_av_array_insert()` publishes each newly reachable pointer *after* a write
  barrier.
- `efa_av_array_at()` consumes pointers *behind* a read barrier (`ofi_rmb()` at
  the top of the function).

The correctness argument (documented in the header) rests on a structural
invariant of `fi_addr_t` usage: **once a pointer is published it is valid for
the life of the array.** The man pages forbid `fi_av_remove()` with in-flight
packets and forbid reading an AV entry before its `fi_av_insert()` completes, so
a remove can never race a lookup. That lets a single `rmb()` at the start of
`at()` suffice — no per-slot atomics, no lock. This serves two callers:
1. the AV array itself (written only by `fi_av_insert()`, unreadable until the
   insert finishes), and
2. the per-endpoint peer maps (added to lazily, so a reader *can* run
   concurrently with the single writer — which is exactly what the barrier pair
   makes safe).

### SRX (shared receive queue) lock moved domain → endpoint

The shared-receive-queue lock used to live on the domain; it now lives on the
endpoint:

```c
// prov/efa/src/rdm/efa_rdm_ep.h:62
struct ofi_genlock srx_lock;
```

It is initialized per endpoint with a lock type chosen from the progress/
threading model
([efa_rdm_ep_fiops.c:492](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/rdm/efa_rdm_ep_fiops.c)):

```c
ret = ofi_genlock_init(&efa_rdm_ep->srx_lock,
                       efa_domain_data_progress_lock_type(efa_domain));
```

- Moved from domain to endpoint: commit **5f3aaebc2**.
- Removed from AV operations, EP enable, and SHM address lookup so those paths
  no longer serialize on it: commits **482d6bae7**, **6ae2091dc**.
- Set to `OFI_LOCK_NONE` under `FI_THREAD_COMPLETION` (via
  `efa_domain_data_progress_lock_type()`): commit **54538dd0b**. Under that
  model the srx critical section compiles to a no-op.

The srx_lock now serializes exactly two things (see the assertion comments in
[efa_rdm_cq.c:684-687](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/rdm/efa_rdm_cq.c)):
SRX recv-matching, and the per-endpoint ope/pke state.

### Queued-progress lists moved domain → endpoint + a per-CQ `progress_ep_list`

Queued work that used to hang off the domain now hangs off the endpoint
(commit **ed7966041**). On top of that, each CQ keeps a `progress_ep_list` of
the endpoints that actually have queued work
([efa_rdm_ep.h:147-149](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/rdm/efa_rdm_ep.h),
[efa_rdm_cq.c:954](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/rdm/efa_rdm_cq.c);
commits **81cf7d367**, **6c44da52a**). CQ progress iterates only that list, so
**idle endpoints are skipped entirely** instead of being walked every poll.

The enqueue/progress pair takes locks in the order
`progress_ep_list_lock → srx_lock`. The apparent inversion is safe by design
(documented at
[efa_rdm_ep_utils.c:1108-1116](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/rdm/efa_rdm_ep_utils.c)):
enqueue only takes `progress_ep_list_lock` when the EP is not already on the
list (an EP already on the list early-returns without touching that lock), and
the progress loop only takes an EP's `srx_lock` for EPs that *are* on the list.

### `domain->num_read_msg_in_flight` made atomic

The read-message-in-flight counter on the domain is now an atomic
(`ofi_atomic_*64`), removing another reason to hold a domain lock on the read
path (commit **22f16d7bf**):

```c
// prov/efa/src/rdm/efa_rdm_ope.c:997
num_read_msg_in_flight = ofi_atomic_dec64(&domain->num_read_msg_in_flight);
```

### Net effect

Under `FI_THREAD_COMPLETION` (the model aws-ofi-nccl requests), a proxy thread
that owns endpoint E:
- looks peers up lock-free in E's own `efa_av_array` peer map,
- takes a no-op `srx_lock`,
- progresses only the endpoints on its CQ's `progress_ep_list`, and
- updates `num_read_msg_in_flight` atomically.

No domain-wide lock sits on the hot path anymore.

## Clang Thread-Safety Analysis (both projects)

Both libfabric's EFA provider and aws-ofi-nccl now compile (in an analysis
build) with Clang's [Thread Safety Analysis](https://clang.llvm.org/docs/ThreadSafetyAnalysis.html).
This is high-value context for an agent because the annotations state, in the
source itself, *which lock guards which field*.

### libfabric: `efa_thread_annotations.h`

Enabled with `configure --enable-thread-safety-analysis` (which defines
`OFI_THREAD_SAFETY_ANALYSIS`); expands to nothing otherwise. Source:
[prov/efa/src/efa_thread_annotations.h](https://github.com/ofiwg/libfabric/blob/main/prov/efa/src/efa_thread_annotations.h).

Because the EFA provider locks a real `struct ofi_genlock` rather than a C++
type the compiler can reason about, each *lock role* is represented by a global
dummy capability object ("lock symbol"). Annotations reference the symbol; thin
wrappers take the real genlock and bind it to the symbol:

```c
struct OFI_TSA_CAPABILITY("mutex") ofi_tsa_lock_symbol { char dummy; };

// EFA_GENLOCK_LOCK(lock, sym) / _UNLOCK / _HELD acquire, drop, and assert
// the real lock while telling the analyzer about the symbol.
```

The declared lock symbols enumerate the provider's lock roles:

| Lock symbol | Guards (role) |
|-------------|---------------|
| `efa_qp_table_lock_sym`     | QP table |
| `efa_implicit_av_lock_sym`  | Implicit AV |
| `efa_util_ep_lock_sym`      | util endpoint |
| `efa_ctrl_lock_sym`         | control path |
| `efa_util_av_lock_sym`      | util AV |
| `efa_util_domain_lock_sym`  | util domain |

Wrapper macros (`OFI_TSA_GUARDED_BY`, `OFI_TSA_REQUIRES`, `OFI_TSA_ACQUIRE`,
`OFI_TSA_RELEASE`, `OFI_TSA_EXCLUDES`, `OFI_TSA_NO_ANALYSIS`, …) map to Clang's
`__attribute__((...))` forms.

### aws-ofi-nccl: `nccl_ofi_tsa.h`

Source:
[include/nccl_ofi_tsa.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_tsa.h).
Active when `NCCL_OFI_WANT_THREAD_ANALYSIS` is set. It defines the standard
Clang wrappers directly (`CAPABILITY`, `GUARDED_BY`, `PT_GUARDED_BY`,
`REQUIRES`, `ACQUIRE`, `RELEASE`, `TRY_ACQUIRE`, `EXCLUDES`,
`NO_THREAD_SAFETY_ANALYSIS`, …). Because the plugin locks C++ objects, the lock
*type itself* carries the capability — e.g. the spinlock (below) is declared
`class CAPABILITY("mutex") nccl_ofi_spinlock`, so a field annotated
`GUARDED_BY(my_spinlock)` is checked at compile time.

**How to read it as an agent:** a field annotated `GUARDED_BY(x)` may only be
touched while `x` is held; a function annotated `REQUIRES(x)` must be called
with `x` held; `ACQUIRE(x)` / `RELEASE(x)` mark the lock/unlock boundary. These
annotations are the ground truth for the locking discipline.

## aws-ofi-nccl Threading Changes (GIN data path)

### `NCCL_GIN_PROXY_NTHREADS` with per-thread endpoint bucketing

The GIN (GPU-Initiated Networking) proxy can now run multiple progress threads,
each driving its own endpoint, instead of contending on one shared CQ. The knob
is read via `getenv("NCCL_GIN_PROXY_NTHREADS")`
([src/rdma/gin/nccl_ofi_gin_api.cpp:79-81](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_api.cpp),
commit **e6c4eb1**), **default 1**, capped at `NCCL_GIN_MAX_CONNECTIONS`.

> Note: the environment variable is spelled `NCCL_GIN_PROXY_NTHREADS`
> (not `OFI_NCCL_GIN_PROXY_NTHREADS`); it is read with a plain `getenv`, not
> through the `OFI_NCCL_PARAM` machinery.

**Bucketing scheme** (in `nccl_ofi_gin_listen()`,
[nccl_ofi_gin_api.cpp:153-170](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_api.cpp)):
NCCL calls `listen()` once per collComm. Each call takes the next
`listen_seq` and maps the connection to a progress thread by
`thread_idx = listen_seq % nthreads`. The endpoint key folds the thread index
into the per-init sequence:

```c
const int listen_seq = context->listen_count++;
int thread_idx = listen_seq % context->nthreads;
const long endpoint_key = static_cast<long>(context->seq)
                        ^ (static_cast<long>(thread_idx) << 32);
```

- `ginCommCount == NTHREADS` (typical): each `listen()` gets a unique
  `thread_idx` → one dedicated endpoint per thread.
- `ginCommCount > NTHREADS`: connections `t, t+T, t+2T, …` round-robin onto the
  same `endpoint_key`, so several connections share thread *t*'s endpoint.
- `NCCL_GIN_PROXY_NTHREADS=1` (default): all connections share the single
  per-init-seq endpoint (legacy behavior).

The point of folding `thread_idx` into the key is that the RMA and GIN layers
end up on *different* endpoints/locks, and each progress thread owns its own
endpoint so it never contends on a shared CQ.

### One process-wide gdrcopy worker with signal coalescing

Previously signal delivery was arranged per put-comm. Now there is exactly
**one process-wide gdrcopy worker thread** (commit **ad6dcac**), a lazily-created
singleton:

```c
// src/rdma/gin/nccl_ofi_gin.cpp:1414-1425
nccl_ofi_gin_gdrcopy_worker &nccl_ofi_gin_gdrcopy_worker::get() {
    static nccl_ofi_gin_gdrcopy_worker instance;   // spawned on first use
    return instance;
}
```

With `NCCL_GIN_PROXY_NTHREADS` proxy threads each producing signal-delivery
work, running one gdrcopy worker per put-comm would oversubscribe. Instead every
comm's proxy thread feeds the *shared* worker, which **coalesces work items that
target the same signal slot** into a single PCIe read-modify-write. The worker
drains up to `MAX_BATCH` items per pass and folds same-slot entries
([nccl_ofi_gin.cpp:1436-1563](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin.cpp)).
This is "signal coalescing": a burst of updates to one counter collapses to one
device write instead of one write per update.

### The MPSC ring feeding the shared worker

Proxy threads hand work to the shared worker through a lock-free
multi-producer / single-consumer ring
([include/nccl_ofi_mpsc_ring.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_mpsc_ring.h),
commit **3e5687e**), used as
`nccl_ofi_mpsc_ring<gin_signal_work_entry> work_queue`
([nccl_ofi_gin.h:384](https://github.com/aws/aws-ofi-nccl/blob/master/include/rdma/gin/nccl_ofi_gin.h)).

**Algorithm — bounded MPMC slot ring (Dmitry Vyukov's scheme), restricted to
one consumer.** Each slot carries a `std::atomic<uint32_t> sequence` that encodes
whose turn the slot is on:
- A producer reserves a slot by **CAS-advancing** `enqueue_pos`; it owns the slot
  whose `sequence == pos`. It writes the payload, then publishes with
  `sequence = pos + 1` (release).
- The single consumer owns the slot whose `sequence == dequeue_pos + 1`; it reads
  the payload, then frees the slot with `sequence = pos + CAPACITY` (release),
  making it reusable one lap later.

**Memory ordering / progress guarantees (from the header):**
- *No torn reads:* payload is written before the publish-store and read after the
  matching acquire-load; release/acquire pair on `sequence`.
- *No lost entries:* `enqueue_pos` advances only on a successful CAS, so two
  producers never claim the same slot; the consumer only consumes after the
  producer published.
- *Full ring fails cleanly:* the code CASes rather than blindly `fetch_add`-ing,
  precisely so a full ring returns `false` without leaving a reserved-but-unusable
  hole the consumer would wait on forever.
- *No ABA:* positions are monotonic 32-bit counters; sequence comparisons use a
  signed difference so they are wrap-safe across `uint32_t` rollover.
- *Non-blocking:* `push()`/`pop()` never block — they return `false` on
  full/empty and let the caller spin, back off, or do other work.
- `CAPACITY` must be a power of two (index is a mask); `enqueue_pos` and
  `dequeue_pos` sit on separate 64-byte-aligned cache lines to avoid false
  sharing between the contending producers and the lone consumer.

### Companion primitives

- **`nccl_ofi_spsc_ring`**
  ([include/nccl_ofi_spsc_ring.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_spsc_ring.h)):
  single-producer / single-consumer lock-free ring. Producer releases `head`,
  consumer releases `tail`, each side acquires the other's index. FIFO order
  holds. One slot is always left empty to tell full from empty, so it holds
  `CAPACITY - 1` entries. `head`/`tail` are cache-line separated. `push()`/`pop()`
  return `false` rather than block.
- **`nccl_ofi_spinlock`**
  ([include/nccl_ofi_spinlock.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_spinlock.h)):
  a minimal BasicLockable spinlock over `std::atomic<bool>`. `trylock()` is a
  `compare_exchange_strong` with `acquire` success ordering; `lock()` spins with
  an architecture-specific relax hint (`pause` on x86-64, `isb` on aarch64) using
  a test-and-test-and-set loop (spins on a relaxed load while the lock looks
  taken, only attempting the CAS when it looks free); `unlock()` is a `release`
  store. The class is annotated `CAPABILITY("mutex")` so Clang thread-safety
  analysis can check `GUARDED_BY`/`REQUIRES` on fields and functions it protects.

## Removed GIN Threading Knobs

Two GIN environment variables referenced by earlier tuning advice were removed
upstream in 2026:

- **`OFI_NCCL_GIN_TYPE`** — removed (commit **80f2c78**). GDAKI is now
  auto-enabled per platform; there is no user-selectable GIN type. (An example
  still using `NCCL_GIN_TYPE=5` survives only in
  `doc/gin-getting-started.md`, which is stale.)
- **`OFI_NCCL_GIN_STRONG_SIGNAL`** — removed (commit **aa80b54**). Strong signal
  ordering is now handled internally (producer-side coalescing in the gdrcopy
  worker), so the knob no longer exists.

If you see either variable in scripts, delete it; setting it has no effect.
