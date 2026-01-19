# Performance Optimization Opportunities

## Executive Summary

Based on analysis of the NCCL + OFI + libfabric + EFA stack, this document identifies high-impact optimization opportunities organized by priority and expected performance gain.

## Optimization Priority Matrix

| Priority | Optimization Area | Expected Gain | Complexity | Pages |
|----------|------------------|---------------|------------|-------|
| 🔥 P0 | Memory Registration Caching | 100-500x | Medium | [Memory Reg](#1-memory-registration-bottlenecks) |
| 🔥 P0 | Reduce Software Overhead in Hot Path | 2-5x | High | [Hot Path](#2-hot-path-software-overhead) |
| 🔴 P1 | Protocol Selection Tuning | 1.5-3x | Low | [Protocol](#3-protocol-selection) |
| 🔴 P1 | Multi-Rail Load Balancing | 2-4x | Medium | [Multi-Rail](#4-multi-rail-optimization) |
| 🟡 P2 | Completion Polling Optimization | 10-30% | Medium | [Polling](#5-completion-polling) |
| 🟡 P2 | CPU/NUMA Affinity | 10-20% | Low | [NUMA](#6-cpunuma-optimization) |
| 🟢 P3 | Zero-Copy Optimizations | 5-15% | High | [Zero-Copy](#7-zero-copy-paths) |
| 🟢 P3 | Batching and Pipelining | 10-20% | Medium | [Batching](#8-batching-and-pipelining) |

## 1. Memory Registration Bottlenecks

### Current Performance Impact

From [20-rdma-memreg.md](doc/20-rdma-memreg.md):

```
Operation                   Time (typical)
──────────────────────────────────────────
Register 4 KB              ~50-100 μs
Register 1 MB              ~100-500 μs
Register 100 MB            ~500-2000 μs
Cache hit (lookup)         ~0.1-1 μs
```

**Impact**: Registering on every transfer can consume 80-95% of total transfer time for small/medium messages.

### Current State

```
Without cache (worst case):
  Transfer 1 MB:
    Register: 500 μs
    Transfer: 100 μs
    Total: 600 μs (83% overhead!)

With cache (best case):
  Transfer 1 MB:
    Cache hit: 1 μs
    Transfer: 100 μs
    Total: 101 μs (1% overhead)
```

### Optimization Opportunities

#### 1.1 Enhanced MR Cache Strategy (P0)

**Current**: Two-level cache (OFI plugin + libfabric provider)

**Opportunity**: Smarter caching strategies

```c
// Current: Exact match only
struct mr_cache_entry {
  void *addr;              // Exact address
  size_t length;           // Exact length
  struct fid_mr *mr;
};

// Proposed: Range-based caching with overlap detection
struct mr_cache_entry_v2 {
  void *base_addr;         // Base address
  size_t total_length;     // Total registered length
  struct fid_mr *mr;

  // Allow sub-range reuse
  bool allow_subrange;

  // Track usage patterns
  uint64_t access_count;
  uint64_t last_access;
  uint32_t access_pattern;  // Sequential, random, etc.
};

// Example: Register 1 GB buffer once, reuse for all sub-ranges
// Instead of registering many 1 MB chunks
```

**Expected Gain**: 50-100x for workloads with many small buffers in larger region

**Implementation**:
- Modify OFI plugin MR cache to support range queries
- Track buffer access patterns
- Pre-register large regions proactively
- Implement sub-range descriptor extraction

#### 1.2 Predictive Pre-Registration (P0)

**Opportunity**: Predict and pre-register buffers before they're needed

```c
// Pattern detection
struct buffer_pattern {
  void *base_addr;
  size_t stride;
  int count;
  uint64_t period_us;
};

// Detect pattern during first few iterations
// Example: Training iterations use same GPU buffers
void detect_pattern() {
  // Iteration 1: buf[0] = 0x1000, size = 128MB
  // Iteration 2: buf[0] = 0x1000, size = 128MB
  // Iteration 3: buf[0] = 0x1000, size = 128MB
  // Pattern detected! Pre-register for future.
}

// Pre-register in background
void* prefetch_thread() {
  while (training) {
    if (pattern_detected && !registered) {
      // Register before next iteration needs it
      fi_mr_reg(domain, predicted_buf, predicted_size, ...);
    }
    sleep_until_next_iteration();
  }
}
```

**Expected Gain**: Eliminates registration overhead for repeated patterns (common in training)

**Implementation**:
- Add pattern detection in OFI plugin
- Background thread for pre-registration
- Heuristics for deep learning workloads (gradient buffers repeat)

#### 1.3 GPU Memory Pool Registration (P0)

**Opportunity**: Register GPU memory pools instead of individual allocations

```c
// Current: NCCL gets random GPU buffers from user
cudaMalloc(&buf1, 100MB);  // Register on first use
cudaMalloc(&buf2, 100MB);  // Register on first use
cudaMalloc(&buf3, 100MB);  // Register on first use

// Proposed: Intercept cudaMalloc or use memory pools
// Register entire GPU heap at initialization
void init_gpu_pool() {
  // Get GPU memory bounds
  size_t total, free;
  cudaMemGetInfo(&free, &total);

  // Register large contiguous region
  void *gpu_base = get_gpu_heap_base();
  fi_mr_reg(domain, gpu_base, total, ...);

  // All subsequent allocations covered
}
```

**Expected Gain**: Zero registration overhead for all GPU buffers

**Challenges**:
- Requires cooperation with CUDA runtime
- May need custom allocator
- GPU memory fragmentation

**Implementation**:
- Custom CUDA allocator with pre-registered pools
- Or dmabuf registration of entire GPU BAR
- Integrate with PyTorch/TensorFlow memory allocators

#### 1.4 Registration Cache Coherency Optimization (P1)

**Current Issue**: memhooks/userfaultfd overhead

From [20-rdma-memreg.md](doc/20-rdma-memreg.md):

```bash
# memhooks: LD_PRELOAD intercepts every malloc/free
# Overhead: ~5-10% on allocation-heavy workloads

# userfaultfd: Kernel notifications
# Overhead: ~2-5%
```

**Opportunity**: Lazy invalidation for specific workloads

```c
// If NCCL knows buffers won't be freed during training:
void training_mode_start() {
  // Disable cache invalidation monitoring
  mr_cache_disable_monitoring();
  // Trust that buffers stay valid
}

void training_mode_end() {
  // Flush cache before buffers might be freed
  mr_cache_flush_all();
  mr_cache_enable_monitoring();
}
```

**Expected Gain**: 2-5% reduction in allocation overhead

**Risk**: Stale cache entries if buffers freed unexpectedly

---

## 2. Hot Path Software Overhead

### Current Latency Breakdown

From [22-optimizations.md](doc/22-optimizations.md):

```
Component                    Time (μs)
────────────────────────────────────
NCCL collective breakdown    1-2
NCCL → OFI plugin call       0.5-1
OFI plugin processing        1-2
  ├─ MR cache lookup         0.1-1
  └─ State management        0.5
Libfabric API                0.5-1
EFA provider processing      1-2
  ├─ WQE building            0.5
  └─ Doorbell                0.5
────────────────────────────────────
Total software overhead:     5-10 μs
Network/hardware:            10-20 μs
────────────────────────────────────
Total end-to-end:            15-30 μs
```

**Target**: Reduce software overhead to < 2 μs

### Optimization Opportunities

#### 2.1 OFI Plugin Direct Posting (P0)

**Current**: Multiple function calls and indirection

```c
// Current path (simplified)
ncclNet->isend(sendComm, data, size, tag, mhandle, &req)
  ├─ Validate parameters
  ├─ Get MR descriptor
  ├─ Allocate request object
  ├─ Call fi_tsend()
  └─ Track request
```

**Opportunity**: Fast path for common case

```c
// Proposed: Inline fast path
static inline ncclResult_t isend_fastpath(
    struct nccl_ofi_send_comm* comm,
    void* data, int size, void* mr) {

  // Inline MR descriptor (no function call)
  void* desc = ((struct fid_mr*)mr)->desc;

  // Direct fi_tsend (inline if possible)
  ssize_t ret = fi_tsend(comm->ep, data, size, desc,
                         comm->remote_addr, comm->tag, data);

  if (likely(ret == 0)) {
    return ncclSuccess;
  }

  // Slow path for errors
  return isend_slowpath(comm, data, size, mr, ret);
}
```

**Expected Gain**: 30-50% reduction in plugin overhead (1-2 μs → 0.5-1 μs)

**Implementation**:
- Separate fast/slow paths
- Aggressive inlining for common case
- Profile-guided optimization
- Reduce validation in hot path

#### 2.2 Request Object Pooling (P1)

**Current**: Dynamic allocation per request

```c
struct nccl_ofi_req* req = malloc(sizeof(*req));  // Expensive!
```

**Opportunity**: Pre-allocated pool with LIFO allocation

```c
// Per-thread request pool (no locking)
__thread struct {
  struct nccl_ofi_req pool[256];
  uint32_t head;
  uint32_t allocated;
} req_pool;

static inline struct nccl_ofi_req* get_req_fast() {
  if (likely(req_pool.head > 0)) {
    return &req_pool.pool[--req_pool.head];  // ~5 cycles
  }
  return get_req_slow();  // Allocate more
}
```

**Expected Gain**: 0.1-0.3 μs per operation

#### 2.3 Descriptor Caching (P1)

**Current**: fi_mr_desc() call on every send

```c
void* desc = fi_mr_desc(mr);  // Function call + lookup
fi_tsend(ep, buf, size, desc, ...);
```

**Opportunity**: Cache descriptors with MR

```c
struct mr_cache_entry {
  struct fid_mr *mr;
  void *desc_cached;  // Cache the descriptor
  // ...
};

// Fast path
void* desc = entry->desc_cached;  // No function call
```

**Expected Gain**: 0.1-0.2 μs per operation

---

## 3. Protocol Selection

### Current State

From [02-nccl-collectives.md](doc/02-nccl-collectives.md):

```
Message Size     Protocol
< 32 KB          LL          (Low latency, 8 bytes/transfer)
32 KB - 2 MB     LL128       (Balance, 128 bytes/transfer)
> 2 MB           Simple      (High bandwidth, full MTU)
```

### Optimization Opportunities

#### 3.1 Dynamic Threshold Tuning (P1)

**Current**: Fixed thresholds

**Opportunity**: Adaptive thresholds based on runtime characteristics

```c
struct protocol_selector {
  // Current thresholds
  size_t ll_threshold;      // 32 KB
  size_t ll128_threshold;   // 2 MB

  // Runtime statistics
  struct {
    uint64_t latency_samples[3];  // LL, LL128, Simple
    uint64_t bandwidth_samples[3];
    uint32_t sample_count;
  } stats;

  // Adaptive adjustment
  bool adaptive_mode;
};

void adjust_thresholds() {
  // If LL128 performs better at smaller sizes, shift threshold
  if (ll128_bandwidth > ll_bandwidth_at_threshold) {
    ll_threshold -= 4096;  // Lower by 4KB
  }
}
```

**Expected Gain**: 10-30% for workloads not matching default thresholds

**Implementation**:
- Measure protocol performance during warmup
- Adjust thresholds per-instance type
- Account for EFA generation (p4d vs p5)

#### 3.2 Protocol Fusion (P1)

**Current**: Single protocol per message

**Opportunity**: Hybrid protocols for medium messages

```
Current LL128 (32KB - 2MB):
  [128B][128B][128B]...[128B]  ← All LL128

Proposed Hybrid:
  [LL][LL][Simple][Simple]...[Simple]
  ↑        ↑
  Fast     High
  start    bandwidth
```

**Expected Gain**: 20-40% for 100KB-500KB messages (common gradient sizes)

#### 3.3 EFA-Specific Protocol (P0)

**Opportunity**: Leverage EFA's eager/rendezvous efficiently

From [10-efa-provider.md](doc/10-efa-provider.md):

```
Eager threshold: 64 KB (configurable via FI_EFA_RDM_LONG_MSG_SIZE)
Below: Single-copy eager
Above: RDMA-based rendezvous
```

**Current Mismatch**:
- NCCL switches LL128→Simple at 2MB
- EFA switches eager→rendezvous at 64KB
- Gap: 64KB-2MB using wrong underlying protocol

**Proposed**: Align NCCL protocols with EFA thresholds

```bash
# Tune EFA for NCCL's patterns
FI_EFA_RDM_LONG_MSG_SIZE=2097152  # Match NCCL's 2MB threshold

# Or tune NCCL to EFA
NCCL_PROTO_LL128_MAX_SIZE=65536   # Match EFA's 64KB eager
```

**Expected Gain**: 15-25% for 64KB-2MB range

---

## 4. Multi-Rail Optimization

### Current State

From [10-efa-provider.md](doc/10-efa-provider.md):

```
p4d.24xlarge: 4x 100 Gbps EFA (400 Gbps total)
p5.48xlarge:  8x 200 Gbps EFA (1600 Gbps total)

Current scaling:
  4 EFAs × 12 GB/s = ~48 GB/s aggregate
  Limited by PCIe, not network
```

### Optimization Opportunities

#### 4.1 Intelligent NIC Selection (P1)

**Current**: Round-robin channel assignment

```
Channel 0 → EFA 0
Channel 1 → EFA 1
Channel 2 → EFA 2
Channel 3 → EFA 3
Channel 4 → EFA 0  (wrap)
```

**Opportunity**: Topology-aware + load-balanced assignment

```c
struct nic_assignment {
  int nic_id;
  int numa_node;
  float load;           // Current utilization
  uint64_t bytes_sent;
  uint64_t bytes_recv;
};

int select_nic_smart(int channel, size_t msg_size) {
  // Prefer local NUMA NIC
  int preferred = get_numa_local_nic(channel);

  // Check load
  if (nic_load[preferred] > 0.8) {
    // Find least loaded NIC on same NUMA
    preferred = get_least_loaded_nic_numa(numa_node);
  }

  return preferred;
}
```

**Expected Gain**: 10-20% better load balancing, especially for imbalanced workloads

#### 4.2 Per-Message-Size NIC Affinity (P1)

**Opportunity**: Different NICs for different message sizes

```c
// Small messages (latency-sensitive): Use NIC closest to GPU
// Large messages (bandwidth-bound): Spread across all NICs

if (size < 64KB) {
  nic = get_closest_nic_to_gpu(gpu_id);
} else {
  nic = round_robin_all_nics();
}
```

**Expected Gain**: 5-15% latency improvement for small messages

#### 4.3 Adaptive Multi-Rail (P0)

**Current**: Static channel-to-NIC mapping

**Opportunity**: Dynamic rebalancing based on congestion

```c
void rebalance_nics() {
  // Monitor per-NIC throughput
  for (int i = 0; i < num_nics; i++) {
    uint64_t bw = measure_nic_bandwidth(i);
    if (bw < expected_bandwidth * 0.8) {
      // NIC congested, move some channels to other NICs
      migrate_channels(i, find_underutilized_nic());
    }
  }
}
```

**Expected Gain**: 20-40% in congested multi-tenant environments

---

## 5. Completion Polling

### Current State

From [19-threading-model.md](doc/19-threading-model.md):

```c
// NCCL proxy thread
void* ncclProxyThread(void* args) {
  while (!stop) {
    fi_cq_read(cq, ...);  // Poll continuously
    // Process completions
  }
}
```

**Issue**: 100% CPU usage even when idle

### Optimization Opportunities

#### 5.1 Adaptive Polling (P2)

**Opportunity**: Back off when idle

```c
#define BUSY_POLL_THRESHOLD 1000
#define MAX_BACKOFF_US 100

int idle_count = 0;
int backoff_us = 1;

while (!stop) {
  ret = fi_cq_read(cq, entries, 32);

  if (ret > 0) {
    // Active
    idle_count = 0;
    backoff_us = 1;
    process_completions(entries, ret);
  } else {
    // Idle
    idle_count++;

    if (idle_count > BUSY_POLL_THRESHOLD) {
      // Exponential backoff
      usleep(backoff_us);
      backoff_us = min(backoff_us * 2, MAX_BACKOFF_US);
    } else {
      // Busy poll with pause
      _mm_pause();  // ~40 cycles
    }
  }
}
```

**Expected Gain**:
- CPU usage: 100% → 10-30% during idle periods
- Latency impact: +1-5 μs during idle→active transition

#### 5.2 Batched Completion Processing (P2)

**Current**: Process completions one-by-one

**Opportunity**: Batch processing

```c
#define CQ_BATCH_SIZE 64

struct fi_cq_data_entry entries[CQ_BATCH_SIZE];

ret = fi_cq_read(cq, entries, CQ_BATCH_SIZE);

if (ret > 0) {
  // Process all at once
  for (int i = 0; i < ret; i++) {
    // Prefetch next entry
    __builtin_prefetch(&entries[i+1]);

    process_completion(&entries[i]);
  }
}
```

**Expected Gain**: 10-15% throughput improvement for high message rates

#### 5.3 Event-Driven for Low QPS (P3)

**Opportunity**: Use interrupts when message rate is low

```c
void smart_progress() {
  if (messages_per_sec > 10000) {
    // High rate: polling mode
    while (!stop) {
      fi_cq_read(cq, entries, 32);
    }
  } else {
    // Low rate: event-driven mode
    while (!stop) {
      fi_cq_sread(cq, entries, 32, NULL, 1000);  // Wait up to 1ms
    }
  }
}
```

**Expected Gain**: 50-90% CPU reduction for low QPS workloads

---

## 6. CPU/NUMA Optimization

### Current Issues

From [19-threading-model.md](doc/19-threading-model.md):

```
Cross-NUMA access:
  Local:  ~80ns latency, 100+ GB/s bandwidth
  Remote: ~120ns latency, 50 GB/s bandwidth

Impact: 50% performance loss for remote access
```

### Optimization Opportunities

#### 6.1 NUMA-Aware Buffer Allocation (P2)

**Current**: Buffers allocated on first-touch NUMA node (random)

**Opportunity**: Explicit NUMA placement

```c
#include <numa.h>

void* allocate_optimal_buffer(size_t size, int gpu_id) {
  // Determine GPU's NUMA node
  int numa_node = get_gpu_numa_node(gpu_id);

  // Allocate on same NUMA node
  void* buf = numa_alloc_onnode(size, numa_node);

  return buf;
}
```

**Expected Gain**: 10-20% for proxy thread buffer access

#### 6.2 Proxy Thread Affinity (P2)

**Current**: Often not pinned, may migrate

**Opportunity**: Pin to GPU-local NUMA node

```c
void pin_proxy_thread(int gpu_id) {
  // Get GPU's NUMA node
  int numa_node = get_gpu_numa_node(gpu_id);

  // Get CPUs on that NUMA node
  struct bitmask* cpus = numa_allocate_cpumask();
  numa_node_to_cpus(numa_node, cpus);

  // Pin thread
  pthread_t self = pthread_self();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  // Use least loaded CPU on NUMA node
  int cpu = find_least_loaded_cpu(cpus);
  CPU_SET(cpu, &cpuset);

  pthread_setaffinity_np(self, sizeof(cpuset), &cpuset);
}
```

**Expected Gain**: 10-15% reduction in proxy thread overhead

#### 6.3 NIC-GPU-NUMA Alignment (P1)

**Opportunity**: Ensure optimal topology

```bash
# Check current topology
nvidia-smi topo -m
lspci -vv | grep -A20 "Ethernet controller: Amazon"

# Ensure each GPU uses its closest EFA
# Current: May be misaligned
# GPU 0 (NUMA 0) → EFA 2 (NUMA 1)  ← BAD!

# Optimal:
# GPU 0 (NUMA 0) → EFA 0 (NUMA 0)  ← GOOD!
```

**Expected Gain**: 15-25% if currently misaligned

---

## 7. Zero-Copy Paths

### Current Overhead

From [03-nccl-datapath.md](doc/03-nccl-datapath.md):

```
Copies in data path:
1. GPU memory → Channel buffer (NCCL kernel)
2. Channel buffer → Network (DMA)
```

### Optimization Opportunities

#### 7.1 Direct GPU Registration (P3)

**Current**: Use NCCL channel buffers

**Opportunity**: Register user buffers directly (when safe)

```c
// Current:
GPU buffer → Copy to NCCL buffer → Register → Send

// Proposed (for large, aligned buffers):
GPU buffer → Register directly → Send (zero-copy)
```

**Challenges**:
- User buffer lifetime unknown
- Alignment requirements
- May hurt pipelining

**Expected Gain**: 5-10% for large messages if buffer reused

#### 7.2 GPUDirect Async (P3)

**Opportunity**: Overlap GPU compute with network transfer

```c
// Current: Sequential
cudaKernel<<<>>>(...);
cudaDeviceSynchronize();
ncclAllReduce(result, ...);

// Proposed: Overlap
cudaKernel<<<stream1>>>(...);
ncclAllReduce(intermediate_result, ..., stream2);  // Concurrent
cudaKernel<<<stream1>>>(next_layer);
```

**Expected Gain**: 10-30% wall-clock time reduction via overlap

---

## 8. Batching and Pipelining

### Optimization Opportunities

#### 8.1 Operation Batching (P2)

**Current**: Each collective is independent

**Opportunity**: Batch multiple small collectives

```c
// Current: 10 separate 1MB AllReduces
for (int i = 0; i < 10; i++) {
  ncclAllReduce(bufs[i], 1MB, ...);  // 10x overhead
}

// Proposed: Batch into single 10MB AllReduce
concat_buffers(bufs, 10, temp_buf, 10MB);
ncclAllReduce(temp_buf, 10MB, ...);  // 1x overhead
split_buffers(temp_buf, bufs, 10);
```

**Expected Gain**: 30-50% for many small collectives

#### 8.2 Pipeline Depth Tuning (P2)

**Current**: NCCL_BUFFSIZE controls pipeline depth

**Opportunity**: Dynamic depth based on message size

```bash
# Current: Fixed 4MB
NCCL_BUFFSIZE=4194304

# Proposed: Adaptive
# Small messages: Smaller buffers, more overlap
# Large messages: Larger buffers, better efficiency
```

**Expected Gain**: 10-20% better pipeline utilization

---

## Priority Action Items

### Immediate (P0) - Highest Impact

1. **Memory Registration Cache Enhancement**
   - Implement range-based caching
   - Add predictive pre-registration
   - Expected: 100-500x improvement

2. **OFI Plugin Fast Path**
   - Inline hot path operations
   - Reduce function call overhead
   - Expected: 30-50% latency reduction

3. **EFA Protocol Alignment**
   - Align NCCL thresholds with EFA eager/rendezvous
   - Expected: 15-25% for 64KB-2MB messages

4. **Adaptive Multi-Rail**
   - Dynamic load balancing across NICs
   - Expected: 20-40% in congested scenarios

### Short-term (P1) - High Impact, Lower Risk

5. **Dynamic Protocol Thresholds**
   - Tune per-instance type
   - Expected: 10-30% for specific workloads

6. **NUMA-Aware NIC Selection**
   - Topology-aware assignment
   - Expected: 15-25% if misaligned

7. **Request Pooling**
   - Pre-allocated request objects
   - Expected: 0.1-0.3 μs per operation

### Medium-term (P2) - Moderate Impact

8. **Adaptive Polling**
   - CPU-efficient progress
   - Expected: 10-30% CPU reduction

9. **NUMA-Aware Allocation**
   - Explicit buffer placement
   - Expected: 10-20% for memory-bound

10. **Operation Batching**
    - Reduce overhead for small ops
    - Expected: 30-50% for batched workloads

---

## Measurement and Validation

### Benchmarking Framework

```bash
# Baseline measurement
./nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 8

# Per-optimization measurement
# 1. Enable optimization
# 2. Re-run benchmark
# 3. Compare metrics:
#    - Latency (us)
#    - Bandwidth (GB/s)
#    - CPU usage (%)
#    - Memory usage

# Look for:
# - < 15 μs latency for 8-byte messages
# - > 11 GB/s per 100G EFA for large messages
# - > 95% MR cache hit rate
# - Linear scaling with multi-rail
```

### Key Metrics

```
Performance Targets:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric                  Current    Target
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Small msg latency       15-30 μs   < 15 μs
Large msg BW (per EFA)  11-12 GB/s 11.5 GB/s
Multi-rail scaling      Linear     Linear
MR cache hit rate       95%        > 99%
CPU usage (active)      100%       100%
CPU usage (idle)        100%       < 30%
Software overhead       5-10 μs    < 2 μs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Conclusion

**Highest Impact Optimizations (by expected total gain):**

1. ⭐ **Memory Registration** - 100-500x potential (P0)
2. ⭐ **Hot Path Optimization** - 2-5x latency reduction (P0)
3. ⭐ **Multi-Rail Enhancement** - 2-4x in multi-NIC setups (P1)
4. **Protocol Tuning** - 1.5-3x for specific sizes (P1)
5. **NUMA Optimization** - 10-20% overall (P2)

**Estimated Cumulative Impact**:
- **Best case**: 10-50x for cache-miss-heavy workloads
- **Typical case**: 2-5x overall performance improvement
- **Worst case**: 30-50% improvement (still significant)

Next steps: Implement P0 items first, measure, then proceed to P1/P2 based on specific workload characteristics.
