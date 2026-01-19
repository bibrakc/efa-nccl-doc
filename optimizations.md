# Passthrough Modes and Optimizations

## Overview

This document covers advanced optimizations throughout the NCCL + OFI + libfabric + EFA stack. These techniques reduce overhead and improve both latency and bandwidth.

## Software Stack Overhead

### Typical Overhead Per Layer

```
Layer                      Overhead (per operation)
─────────────────────────────────────────────────────
NCCL collective breakdown  ~1-2 μs
NCCL → OFI plugin call     ~0.5-1 μs
OFI plugin processing      ~1-2 μs
  ├─ MR cache lookup       ~0.1-1 μs (hit)
  └─ State management      ~0.5 μs
Libfabric API              ~0.5-1 μs
  ├─ Endpoint lookup       ~0.2 μs
  └─ Validation            ~0.3 μs
EFA provider processing    ~1-2 μs
  ├─ WQE building          ~0.5 μs
  └─ Doorbell              ~0.5 μs
─────────────────────────────────────────────────────
Total software overhead:   ~5-10 μs
Network/hardware:          ~10-20 μs
─────────────────────────────────────────────────────
Total end-to-end:          ~15-30 μs
```

**Goal**: Minimize software overhead to get closer to hardware limits.

## Kernel Bypass (User-Space I/O)

### Traditional Network I/O

```
Application
    ↓ write(socket, ...)
Syscall boundary
    ↓
Kernel Network Stack
    ↓ TCP/IP processing
    ↓ Copy to kernel buffers
Driver
    ↓
Hardware

Overhead: ~50-100 μs per operation
```

### EFA Kernel Bypass

```
Application
    ↓ ibv_post_send() [user space]
    ↓ Write WQE to mapped memory
    ↓ MMIO doorbell write
Hardware
    ↓ DMA directly from user memory

Overhead: ~0.1-0.5 μs (significant speedup vs syscalls)
```

**Mechanisms:**
- **Memory mapping**: Queues mapped to user space via `mmap()`
- **MMIO doorbells**: Direct hardware register access
- **Completion polling**: User reads CQ from mapped memory
- **No syscalls**: Entire data path in user space

**Already in Use:**
- EFA driver inherently supports this
- Libfabric leverages it
- NCCL benefits automatically

## Memory Registration Optimizations

### Multi-Level Caching

```
Request for buffer 0x1000, size 4MB
    │
    ▼
┌─────────────────────────────┐
│  OFI Plugin MR Cache        │ ← Exact match cache
│  (addr, size) → mr          │    ~0.1 μs lookup
└──────────┬──────────────────┘
           │ Miss
           ▼
┌─────────────────────────────┐
│  Libfabric Provider Cache   │ ← Range-based cache
│  Overlap detection          │    ~0.5 μs lookup
└──────────┬──────────────────┘
           │ Miss
           ▼
┌─────────────────────────────┐
│  Register (ibv_reg_mr)      │ ← Actual registration
│  100-500 μs                 │
└─────────────────────────────┘
```

### On-Demand Registration (ODR)

```c
// NCCL uses buffers on-demand
// Register when first used, cache for reuse

ncclResult_t optimized_send(void* buf, size_t size) {
  // Try cache
  void* mr = mr_cache_find(buf, size);

  if (!mr) {
    // Register on-demand
    mr = fi_mr_reg(domain, buf, size, ...);
    mr_cache_insert(buf, size, mr);
  }

  // Use for transfer
  fi_send(ep, buf, size, fi_mr_desc(mr), ...);
}
```

### Pre-Registration

For known buffers:

```c
// At initialization, pre-register all GPU buffers
void init_nccl() {
  for (int i = 0; i < NUM_GPUS; i++) {
    cudaMalloc(&gpu_bufs[i], BUF_SIZE);

    // Pre-register
    fi_mr_reg(domain, gpu_bufs[i], BUF_SIZE,
              access, 0, 0, FI_HMEM_CUDA, &mrs[i], NULL);
  }
}

// At runtime: zero registration overhead!
void collective() {
  // MR already registered
  fi_send(ep, gpu_bufs[0], size, fi_mr_desc(mrs[0]), ...);
}
```

### Registration Cache Monitoring

Efficient cache invalidation:

```c
// Using memhooks (LD_PRELOAD)
void* malloc(size_t size) {
  void* ptr = real_malloc(size);
  // Track allocation
  return ptr;
}

void free(void* ptr) {
  // Invalidate any cached MRs
  mr_cache_remove(ptr);
  real_free(ptr);
}

// Using userfaultfd (kernel support)
void invalidation_thread() {
  while (1) {
    struct uffd_msg msg;
    read(uffd, &msg, sizeof(msg));

    if (msg.event == UFFD_EVENT_UNMAP) {
      mr_cache_invalidate(msg.arg.remove.start,
                         msg.arg.remove.end);
    }
  }
}
```

**Choose based on:**
- **memhooks**: Simple, works everywhere, some overhead
- **userfaultfd**: Kernel 4.11+, lower overhead, more complex

## Completion Handling Optimizations

### Batching

```c
// Bad: Poll after each operation
for (int i = 0; i < N; i++) {
  fi_send(...);
  fi_cq_read(...);  // N polls
}

// Good: Batch operations
for (int i = 0; i < N; i++) {
  fi_send(...);
}
fi_cq_read(..., N);  // 1 poll for all
```

**Benefits:**
- Amortize polling overhead
- Better CPU cache utilization
- Higher throughput

### Adaptive Polling

```c
int idle_count = 0;
int backoff_threshold = 1000;

while (active) {
  ret = fi_cq_read(cq, entries, batch_size);

  if (ret > 0) {
    // Got completions
    idle_count = 0;
    process_completions(entries, ret);
  } else {
    // No completions
    idle_count++;

    if (idle_count < backoff_threshold) {
      // Busy poll (low latency)
      _mm_pause();  // CPU hint
    } else {
      // Back off (save CPU)
      usleep(1);  // or nanosleep
      backoff_threshold *= 2;  // Exponential backoff
    }
  }
}
```

**Benefits:**
- Low latency when active
- Reduced CPU usage when idle
- Configurable trade-off

### Selective Completion

```c
// Not all operations need completion events

// Normal send (generates completion)
fi_send(ep, buf, size, desc, addr, context);

// Send without completion (if supported)
fi_send(ep, buf, size, desc | FI_INJECT_COMPLETE,
        addr, NULL);  // NULL context = no completion

// Poll less frequently
```

**Use for:**
- Fire-and-forget sends
- Reduce CQ load
- Higher throughput (fewer CQEs)

## Protocol Optimizations

### Eager Send (fi_inject)

For very small messages:

```c
// Standard send (async, requires MR, generates completion)
fi_send(ep, buf, size, desc, addr, context);

// Inject send (sync, no MR, no completion)
if (size <= inject_size) {
  fi_inject(ep, buf, size, addr);
  // Data copied immediately, buf can be reused
}
```

**Characteristics:**
- **Copy overhead**: Data copied to internal buffer
- **No MR needed**: No registration required
- **No completion**: Synchronous, returns when done
- **Size limit**: Typically 32-4096 bytes (EFA: ~4KB)

**Trade-off:**
- Small messages (< 1 KB): Faster due to no MR/completion overhead
- Large messages: Slower due to copy overhead

### Rendezvous Optimization

For large messages, use RDMA write:

```c
// Traditional: fi_send/fi_recv (two-sided)
Sender: fi_send(buf, 1GB)
Receiver: fi_recv(buf, 1GB)
// Both CPUs involved, messaging overhead

// Optimized: fi_write (one-sided)
// 1. Exchange buffer info (once)
// 2. Direct RDMA write
fi_write(ep, local_buf, 1GB,
         desc, addr, remote_buf_addr, rkey, ctx);
// Only sender CPU involved
// Receiver CPU free for other work
```

**Benefits:**
- Zero-copy on receiver
- No receiver CPU overhead
- Better for large transfers (> 1 MB)

### Protocol Selection

NCCL protocol auto-selection:

```
Message Size    Protocol    Rationale
─────────────────────────────────────────
< 32 KB         LL          Low latency, flag-based
32 KB - 2 MB    LL128       Balance latency/bandwidth
> 2 MB          Simple      High bandwidth, fewer protocol overheads
```

**Tuning:**
```bash
# Override auto-selection
NCCL_PROTO=LL,LL128,Simple  # Allow all
NCCL_PROTO=Simple           # Force Simple

# Adjust thresholds (internal, not user-configurable)
# NCCL chooses based on benchmarking
```

## Channel and Parallelism Optimizations

### Channel Tuning

```bash
# Default: 4-16 channels depending on topology
NCCL_NCHANNELS=8

# More channels = more parallelism
# But diminishing returns and overhead

Channel Count   Bandwidth    Overhead
─────────────────────────────────────
1               12 GB/s      Low
2               22 GB/s      Low
4               40 GB/s      Medium
8               48 GB/s      Medium-High
16              50 GB/s      High
32              50 GB/s      Very High (not recommended)
```

**Optimal:**
- 4-8 channels for most workloads
- More channels if many NICs available

### Multi-Rail Utilization

Explicitly use all EFA devices:

```c
// NCCL auto-detects multiple EFAs
// Each EFA appears as separate device

int num_devs;
ncclNet->devices(&num_devs);  // Returns 4 for p4d.24xlarge

// NCCL distributes channels across devices
// Channel 0 → EFA 0
// Channel 1 → EFA 1
// Channel 2 → EFA 2
// Channel 3 → EFA 3
// Channel 4 → EFA 0 (wrap around)
```

**Automatic scaling:**
- 4 EFAs × 12 GB/s = ~48 GB/s aggregate
- Limited by PCIe, not network

### Chunk Size Tuning

```bash
# NCCL chunk size (data per channel per step)
NCCL_CHUNK_SIZE=131072  # 128 KB (default)

# Larger chunks:
NCCL_CHUNK_SIZE=262144  # 256 KB
# + Better bandwidth (fewer protocol rounds)
# - Higher latency (larger units)

# Smaller chunks:
NCCL_CHUNK_SIZE=65536   # 64 KB
# + Lower latency (smaller units)
# - Lower bandwidth (more overhead)
```

## CPU Optimizations

### Affinity and NUMA

```bash
# Pin proxy threads to specific CPUs
NCCL_PROXY_CPU_AFFINITY=0,1,2,3

# NUMA-aware allocation
numactl --cpunodebind=0 --membind=0 ./nccl_app

# Avoid cross-NUMA traffic:
# - Pin app to NUMA node with GPU
# - Pin proxy to same NUMA node
# - Bind memory to same NUMA node
# - Ensure NIC on same NUMA node if possible
```

**Check NUMA topology:**
```bash
lscpu | grep NUMA
numactl --hardware
lstopo  # requires hwloc
```

### CPU Frequency Scaling

```bash
# Disable CPU frequency scaling for consistent latency
sudo cpupower frequency-set -g performance

# Or per-CPU:
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance | sudo tee $cpu
done
```

### Process Priority

```bash
# Elevate priority (requires root)
nice -n -20 ./nccl_app

# Real-time scheduling (use carefully!)
chrt -f 99 ./nccl_app

# Proxy thread priority
NCCL_PROXY_SCHED_PRIORITY=99
```

## Libfabric-Specific Optimizations

### Progress Model

EFA requires manual progress:

```bash
FI_PROGRESS_MANUAL=1  # Default for EFA
```

**Implication**: Must poll frequently

```c
// Ensure frequent polling in proxy threads
while (1) {
  fi_cq_read(cq, ...);  // Drive progress
  // Do this often (microsecond frequency)
}
```

### Endpoint Scalability

Reuse endpoints when possible:

```c
// Bad: Create new endpoint for each transfer
for (int i = 0; i < 1000; i++) {
  fi_endpoint(domain, info, &ep, NULL);
  fi_send(ep, ...);
  fi_close(&ep->fid);  // Expensive!
}

// Good: Reuse endpoints
fi_endpoint(domain, info, &ep, NULL);
for (int i = 0; i < 1000; i++) {
  fi_send(ep, ...);
}
fi_close(&ep->fid);
```

### Resource Limits

```bash
# Increase libfabric resource limits
FI_EFA_TX_SIZE=512      # Larger TX queue (default: 256)
FI_EFA_RX_SIZE=512      # Larger RX queue
FI_EFA_TX_IOV_LIMIT=2   # More scatter-gather entries
```

## EFA-Specific Optimizations

### GPUDirect Configuration

```bash
# Ensure GPUDirect is enabled
FI_EFA_USE_DEVICE_RDMA=1  # Default: enabled

# Verify dmabuf support (kernel 5.12+)
uname -r  # Check kernel version

# Check IOMMU
dmesg | grep -i iommu
# Should see IOMMU enabled
```

### SRD Tuning

EFA's Scalable Reliable Datagram protocol:

```bash
# Queue depths affect outstanding operations
FI_EFA_TX_SIZE=256
FI_EFA_RX_SIZE=256

# Larger queues:
# + More pipelining
# + Higher bandwidth
# - More memory
# - Higher latency variance
```

### Disable Intra-Node SHM

```bash
# Let NCCL handle intra-node (NVLink/SHM)
FI_EFA_ENABLE_SHM_TRANSFER=0

# Why:
# - Avoid double-buffering
# - NCCL's intra-node is better optimized
# - Simpler data path
```

## NCCL-Specific Optimizations

### Algorithm Selection

```bash
# Force specific algorithm
NCCL_ALGO=Ring   # or Tree

# For large messages: Ring
# For small messages: Tree
# For latency: Tree
# For bandwidth: Ring
```

### Buffsize

```bash
# Channel buffer size
NCCL_BUFFSIZE=4194304   # 4 MB (default)

# Larger:
NCCL_BUFFSIZE=8388608   # 8 MB
# + More pipelining
# - More memory

# Smaller:
NCCL_BUFFSIZE=2097152   # 2 MB
# - Less pipelining
# + Less memory
```

### Tuning for Workload

```bash
# Deep learning training (large models)
NCCL_ALGO=Ring
NCCL_PROTO=Simple
NCCL_NCHANNELS=8
NCCL_CHUNK_SIZE=262144  # 256 KB

# Deep learning inference (small batches)
NCCL_ALGO=Tree
NCCL_PROTO=LL128
NCCL_NCHANNELS=4
NCCL_CHUNK_SIZE=65536   # 64 KB

# Large-scale training (many nodes)
NCCL_ALGO=Ring
NCCL_PROTO=Simple
NCCL_NCHANNELS=16
NCCL_CROSS_NIC=2        # Multi-rail
```

## Measuring Performance

### Latency Measurement

```c
#include <time.h>

struct timespec start, end;

clock_gettime(CLOCK_MONOTONIC, &start);

// Operation
ncclAllReduce(sendbuf, recvbuf, count, ...);
cudaStreamSynchronize(stream);

clock_gettime(CLOCK_MONOTONIC, &end);

double latency = (end.tv_sec - start.tv_sec) * 1e6 +
                 (end.tv_nsec - start.tv_nsec) / 1e3;
printf("Latency: %.2f us\n", latency);
```

### Bandwidth Measurement

```c
size_t bytes = count * sizeof(float);
double bandwidth = bytes / (latency / 1e6) / 1e9;
printf("Bandwidth: %.2f GB/s\n", bandwidth);

// Bus bandwidth (accounts for algorithm overhead)
// For AllReduce Ring: 2 * (N-1) / N
double bus_bw = bandwidth * 2 * (nranks - 1) / nranks;
printf("Bus Bandwidth: %.2f GB/s\n", bus_bw);
```

### NCCL Tests

```bash
# Clone NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make

# Run all-reduce benchmark
./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8

# Output shows:
# - Size (bytes)
# - Latency (us)
# - Algorithm bandwidth (GB/s)
# - Bus bandwidth (GB/s)
```

### Profiling

```bash
# NCCL profiling
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=COLL,INIT,ENV

# NCCL graph dump
NCCL_GRAPH_DUMP_FILE=nccl_graph.txt

# Libfabric profiling
FI_LOG_LEVEL=debug
FI_LOG_PROV=efa

# System profiling
perf record -g ./nccl_app
perf report

# GPU profiling
nsys profile --trace=cuda,nvtx ./nccl_app
```

## Optimization Checklist

### Baseline Configuration

```bash
# Memory registration
FI_MR_CACHE_MONITOR=memhooks
FI_MR_CACHE_MAX_SIZE=unlimited

# EFA settings
FI_EFA_ENABLE_SHM_TRANSFER=0
FI_EFA_USE_DEVICE_RDMA=1
FI_EFA_TX_SIZE=256
FI_EFA_RX_SIZE=256

# NCCL settings
NCCL_DEBUG=WARN
NCCL_IB_DISABLE=0
NCCL_SOCKET_IFNAME=^docker,lo
```

### Advanced Tuning

```bash
# Multi-rail
NCCL_CROSS_NIC=2

# CPU affinity
NCCL_PROXY_CPU_AFFINITY=<cpu_list>

# Algorithm override (if needed)
NCCL_ALGO=Ring,Tree

# Protocol override (if needed)
NCCL_PROTO=LL128,Simple

# Larger queues for high bandwidth
FI_EFA_TX_SIZE=512
FI_EFA_RX_SIZE=512
```

## Common Performance Issues

### Low Bandwidth

**Symptoms**: < 10 GB/s per EFA

**Possible Causes:**
1. Memory registration cache misses
   - Solution: Enable cache, use memhooks
2. Too few channels
   - Solution: Increase NCCL_NCHANNELS
3. Wrong algorithm
   - Solution: Use Ring for large messages
4. CPU bottleneck
   - Solution: Pin threads, increase priority

### High Latency

**Symptoms**: > 50 μs for small messages

**Possible Causes:**
1. Wrong protocol
   - Solution: Use LL for small messages
2. CPU scheduling delays
   - Solution: Real-time priority, CPU affinity
3. NUMA cross-traffic
   - Solution: Pin to local NUMA node
4. Infrequent polling
   - Solution: Ensure proxy threads poll continuously

### Instability

**Symptoms**: Crashes, hangs, data corruption

**Possible Causes:**
1. Stale MR cache entries
   - Solution: Enable memhooks/userfaultfd
2. Resource exhaustion
   - Solution: Increase queue sizes, limits
3. IOMMU issues
   - Solution: Check IOMMU config, dmesg errors

## Summary

**Key Optimization Areas:**

1. **Memory Registration**
   - Enable multi-level caching
   - Pre-register hot buffers
   - Use memhooks for cache coherence

2. **Completion Handling**
   - Batch operations
   - Adaptive polling
   - Selective completion

3. **Protocol Selection**
   - Eager for tiny (< 1 KB)
   - LL for small (< 32 KB)
   - LL128 for medium (32 KB - 2 MB)
   - Simple for large (> 2 MB)

4. **Parallelism**
   - Tune channel count (4-8 typical)
   - Utilize multi-rail (all EFAs)
   - Optimize chunk size

5. **CPU Optimization**
   - NUMA-aware placement
   - CPU affinity for threads
   - Disable frequency scaling
   - Elevate priority

6. **EFA-Specific**
   - GPUDirect enabled
   - Disable SHM (let NCCL handle it)
   - Tune queue sizes
   - Monitor device stats

**Performance Targets:**
- **Latency**: < 15 μs (small messages)
- **Bandwidth**: > 11 GB/s per 100G EFA
- **Multi-rail**: Linear scaling (4 EFAs → 40+ GB/s)
- **GPU-to-GPU**: Zero-copy, near line-rate

**Monitoring:**
- MR cache hit rate: > 99%
- CQ poll rate: Match operation rate
- CPU usage: High when active, low when idle
- Network utilization: > 90% for large transfers

All optimizations in this document work together to minimize software overhead and maximize the performance of NCCL over AWS EFA.
