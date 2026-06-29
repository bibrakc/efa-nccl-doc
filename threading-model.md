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

```bash
# Number of proxy threads per communicator
NCCL_NPROXY_THREADS=2  # Default varies

# CPU affinity for proxies
NCCL_PROXY_CPU_AFFINITY=0,1,2,3  # Pin to specific CPUs

# Priority (requires privileges)
NCCL_PROXY_SCHED_PRIORITY=99  # Real-time priority
```

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
- `NCCL_ASYNC_ERROR_HANDLING=1`
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

EFA provider: **FI_THREAD_SAFE**

```c
struct fi_domain_attr {
  enum fi_threading threading = FI_THREAD_SAFE;
  // ...
};
```

**Guarantees:**
- All libfabric calls are thread-safe
- Can call from multiple threads simultaneously
- Internal locking as needed

### Progress Model

EFA provider: **FI_PROGRESS_MANUAL**

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

```bash
# Pin proxy threads to specific CPUs
NCCL_PROXY_CPU_AFFINITY=0,1,2,3

# Avoid sharing cores with application
# Example: 64-core system
#   App threads: 0-31 (physical cores)
#   Proxy threads: 32-63 (sibling SMT threads)
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

```bash
# Elevate proxy thread priority (requires root)
NCCL_PROXY_SCHED_PRIORITY=99  # SCHED_FIFO

# Or use nice
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
