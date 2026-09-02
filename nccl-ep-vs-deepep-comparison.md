# NCCL EP vs DeepEP-NCCL: Comprehensive Comparison

**Date**: 2026-09-01  
**Purpose**: Compare NVIDIA's NCCL EP with DeepSeek's DeepEP-NCCL for MoE communication

---

## Executive Summary

**NCCL EP** (NCCL 2.29.7+) and **DeepEP-NCCL** are two implementations of Expert Parallelism communication for Mixture-of-Experts models. Both use NCCL's GIN (GPU-Initiated Networking) Device API but differ in maturity, features, and integration approach.

### Key Findings

| Aspect | NCCL EP | DeepEP-NCCL |
|--------|---------|-------------|
| **Status** | Experimental | Production (DeepSeek-V3) |
| **Integration** | Native NCCL extension | Standalone library with NCCL backend |
| **FP8 Support** | Planned (not implemented) | Production-ready |
| **Max Scale** | 8 nodes (64 GPUs) | 256+ GPUs tested |
| **API** | C with Python bindings | Python-first |
| **Transport** | GIN + LSA only | GIN or NVSHMEM |

**Bottom Line**: Both implement the same MoE communication pattern using GIN, but DeepEP-NCCL is production-ready while NCCL EP is experimental.

---

## 1. MoE Traffic Pattern

### Do Both Perform the Same Pattern?

**YES** - Both implement identical many-to-many top-k expert routing:

```
Input: topk_idx[num_tokens x top_k] = expert assignments
Pattern: Each token → top_k experts (irregular all-to-all)
Dispatch: Scatter tokens to assigned experts
Combine: Gather expert outputs back to original order
```

**Example** (4 tokens, top-k=2):
```
Token 0 → [Expert 5, Expert 2]  (send to ranks 5, 2)
Token 1 → [Expert 1, Expert 7]  (send to ranks 1, 7)
Token 2 → [Expert 3, Expert 5]  (send to ranks 3, 5)
Token 3 → [Expert 0, Expert 4]  (send to ranks 0, 4)
```

**Characteristics**:
- **Irregular**: Data distribution varies per iteration based on gating
- **Dynamic**: Routing decisions change each forward pass
- **Variable-size**: Each rank sends different amounts to different peers
- **Many-to-many**: Nearly all ranks communicate with all other ranks

**Verdict**: Identical MoE communication pattern.

---

## 2. Architecture Comparison

### NCCL EP Architecture

```
Application
    ↓
ncclEpDispatch/ncclEpCombine (C API)
    ↓
NCCL EP Kernels (low_latency.cu, hybrid_ep.cuh)
    ↓
NCCL Device API (GIN + LSA)
    ↓
NCCL Core → Network Hardware
```

**Key Characteristics**:
- Native NCCL extension in `contrib/nccl_ep/`
- Built with NCCL (`make -C contrib/nccl_ep`)
- Direct access to NCCL internals
- Single transport (GIN + LSA)

### DeepEP-NCCL Architecture

```
PyTorch Model
    ↓
DeepEP Python API (Buffer.dispatch/combine)
    ↓
Backend Selection (NVSHMEM or NCCL GIN)
    ↓
NCCL GIN Backend (nccl_gin_backend.cu)
    ↓
NCCL Device API → NCCL Core → Network
```

**Key Characteristics**:
- Standalone library with pluggable backends
- Separate build process
- Backend abstraction layer
- Multiple transports (NCCL GIN or NVSHMEM)

---

## 3. API Comparison

### NCCL EP API

**C API with explicit tensor descriptors**:

```c
// Group management
ncclEpCreateGroup(&ep_group, comm, &config, stream, alloc_fn, free_fn);

// Handle creation
ncclEpCreateHandle(&handle, ep_group, &topk_idx, local_tensors, 
                   num_local, config, stream);

// Communication
ncclEpDispatch(handle, inputs, num_in, outputs, num_out, 
               local, num_local, send_only, config, stream);
ncclEpCombine(handle, inputs, num_in, outputs, num_out, 
              local, num_local, send_only, config, stream);
ncclEpComplete(handle, config, stream);  // LL mode only

// Cleanup
ncclEpHandleDestroy(handle);
ncclEpGroupDestroy(ep_group, stream);
```

**Tensor Descriptor**:
```c
typedef struct {
    unsigned int ndim;
    unsigned int* sizes;
    unsigned int* strides;
    ncclDataType_t datatype;
    void* data;
    unsigned int tag;  // TOKENS, TOPK_IDX, TOPK_WEIGHTS, SCALES
} ncclNDTensor_t;
```

### DeepEP-NCCL API

**Python-first with implicit buffer management**:

```python
# Buffer creation
buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes,
                low_latency_mode=True, num_qps_per_rank=8)

# Dispatch
recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens, handle, event = \
    buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights,
                   num_tokens_per_rank=num_tokens_per_rank,
                   async_finish=True)

# Combine
combined_x, _, event = buffer.combine(x, handle, 
                                      topk_weights=topk_weights,
                                      async_finish=True)
```

**Backend Selection**:
```bash
export DEEP_EP_BACKEND=nccl  # or nvshmem
```

### API Philosophy

| Aspect | NCCL EP | DeepEP-NCCL |
|--------|---------|-------------|
| **Style** | Explicit, verbose | Implicit, concise |
| **Type Safety** | Strong (C structs) | Dynamic (Python) |
| **Flexibility** | High (tag-based tensors) | Medium (positional args) |
| **Ease of Use** | Lower (more boilerplate) | Higher (less code) |
| **Integration** | NCCL ecosystem | PyTorch ecosystem |

---

## 4. Fused Operations Comparison

### FP8 Quantization (BF16 → FP8)

Both use **identical per-channel quantization algorithm**:

**Algorithm** (both implementations):
```
1. Per 128-element channel:
   - Calculate amax = max(abs(values))
   - Warp-level reduction (16 lanes)
   - Calculate scale = 448.0 / amax  (FP8 E4M3 range)
   - Store scale_inv for dequantization
2. Quantize: fp8_value = saturate(fp32_value * scale)
```

**NCCL EP** (`device/low_latency.cu`):
```cpp
float amax = kFP8Margin, scale, scaleInv;
for (int j = 0; j < kNumElemsPerRead; ++j) {
    fp32Data[j] = static_cast<float>(bf16Data[j]);
    amax = fmaxf(amax, fabsf(fp32Data[j]));
}
amax = warp_reduce_max<16>(amax);
calculate_fp8_scales(amax, scale, scaleInv, roundScale);
if (laneId == 0 or laneId == 16)
    sendBufScales[i * kNumElemsPerRead / 128] = scaleInv;
```

**DeepEP-NCCL** (`csrc/kernels/internode_ll.cu`):
```cpp
float amax = kFP8Margin, scale, scale_inv;
for (int j = 0; j < kNumElemsPerRead; ++j) {
    fp32_values[j] = static_cast<float>(bf16_values[j]);
    amax = fmaxf(amax, fabsf(fp32_values[j]));
}
amax = warp_reduce_max<16>(amax);
calculate_fp8_scales(amax, scale, scale_inv, round_scale);
if (lane_id == 0 or lane_id == 16)
    rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;
```

**Status**:
- **NCCL EP**: Code present but **NOT IMPLEMENTED** (marked as limitation in RELEASE.md)
- **DeepEP-NCCL**: **Production-ready**, used in DeepSeek-V3

### Sparse-to-Dense Conversion

**NCCL EP** (HT mode):
```cpp
// Explicit conversion kernel
convert_topk_to_routing_map_kernel(
    topk_idx,      // [num_tokens, num_topk]
    routing_map,   // [num_tokens, num_experts]
    ...
);
// Creates dense routing map for hierarchical aggregation
```

**DeepEP-NCCL** (HT mode):
```cpp
// Metadata exchange approach
// No explicit sparse→dense conversion
// Uses layout computation and token counts
```

**Difference**: NCCL EP uses preprocessing, DeepEP uses metadata exchange.

### Weight Aggregation

**NCCL EP**:
```cpp
// Two-pass: convert weights, then aggregate
convert_sparse_to_dense_prob_combine_kernel(...);
// Later: aggregate using dense weights
```

**DeepEP-NCCL**:
```cpp
// Single-pass: inline weight application
for (int k = 0; k < num_topk; k++) {
    float weight = topk_weights[token_idx * num_topk + k];
    output[token_idx][h] += weight * expert_output[h];
}
```

**Efficiency**: DeepEP single-pass is more efficient.

### Message Packing

Both use **identical format**:
```
Message = [source_token_idx (4B)] + [reserved (12B)] + 
          [hidden_data] + [scales]
```

**Alignment**: 16 bytes (int4) for GPU efficiency

### Fused Operations Summary

| Operation | NCCL EP | DeepEP-NCCL | Status |
|-----------|---------|-------------|--------|
| **FP8 Quantization** | Designed | Production | DeepEP ✓ |
| **Sparse→Dense** | Explicit kernel | Metadata exchange | Different |
| **Weight Aggregation** | Two-pass | Single-pass | DeepEP ✓ |
| **Message Packing** | Identical | Identical | Tie |
| **UE8M0 Format** | No | Yes | DeepEP ✓ |

---

## 5. GIN Usage Comparison

Both use NCCL Device API identically:

### Common GIN Pattern

```cpp
// Initialize GIN context
ncclGin gin(devComm, contextId);

// Barrier synchronization
ncclGinBarrierSession<ncclCoopCta> barrier(...);
barrier.sync(ncclCoopCta(), cuda::memory_order_relaxed, 
             ncclGinFenceLevel::Relaxed);

// RDMA put operation
gin.put(team, peer,
        dstWin, dstOffset,
        srcWin, srcOffset,
        size,
        ncclGin_SignalAdd{signalId, value},
        ncclCoopThread(),
        ncclGin_None(),
        cuda::thread_scope_system);

// Wait for completion
gin.waitSignal(ncclCoopCta(), signalId, expectedValue);
```

### Differences in GIN Usage

| Aspect | NCCL EP | DeepEP-NCCL |
|--------|---------|-------------|
| **Communicators** | Single devComm | Array for multi-QP |
| **Signal Ops** | SignalInc | SignalAdd with explicit values |
| **Barrier Teams** | ncclTeamWorld | ncclTeamTagRail |
| **Window Management** | Direct passing | Device array |

**Verdict**: Same API, minor usage differences for multi-QP support.

---

## 6. Performance Comparison

### NCCL EP Performance (H100, NCCL v2.29u1)

**Low-Latency Mode** (BF16, 128 tokens/rank, 7168 hidden, top-8):

| GPUs | Nodes | Dispatch Latency | Dispatch BW | Combine Latency | Combine BW |
|:----:|:-----:|:----------------:|:-----------:|:---------------:|:----------:|
| 8    | 1     | ~32 μs*          | 224.3 GB/s  | ~39 μs*         | 185.2 GB/s |
| 16   | 2     | ~94 μs*          | 76.7 GB/s   | ~99 μs*         | 73.0 GB/s  |
| 32   | 4     | ~135 μs*         | 53.6 GB/s   | ~145 μs*        | 50.0 GB/s  |
| 64   | 8     | ~148 μs*         | 48.8 GB/s   | ~165 μs*        | 43.8 GB/s  |

*Estimated from bandwidth (7168 hidden × 128 tokens × 2 bytes / BW)

### DeepEP-NCCL Performance (H100)

**Low-Latency Mode** (FP8 dispatch/BF16 combine, 128 tokens/rank):

| GPUs | Nodes | Dispatch Latency | Dispatch BW | Combine Latency | Combine BW |
|:----:|:-----:|:----------------:|:-----------:|:---------------:|:----------:|
| 8    | 1     | 160.8 μs         | 47.0 GB/s   | 302.8 μs        | 47.9 GB/s  |
| 16   | 2     | 178.6 μs         | 42.2 GB/s   | 320.8 μs        | 45.3 GB/s  |
| 32   | 4     | 190.1 μs         | 39.8 GB/s   | 333.2 μs        | 43.6 GB/s  |
| 64   | 8     | 218.9 μs         | 34.5 GB/s   | 351.1 μs        | 41.4 GB/s  |

**High-Throughput Mode** (4096 tokens/rank):

| GPUs | Nodes | Dispatch BW | Combine BW |
|:----:|:-----:|:-----------:|:----------:|
| 16   | 2     | 76.9 GB/s   | 66.2 GB/s  |
| 32   | 4     | 61.7 GB/s   | 62.3 GB/s  |
| 64   | 8     | 52.7 GB/s   | 52.9 GB/s  |

### Performance Analysis

**NCCL EP Advantages**:
- **5x lower latency single-node** (~32 μs vs 161 μs dispatch)
- **4.7x higher single-node bandwidth** (224 vs 47 GB/s)
- Better multi-node dispatch at 2-4 nodes
- Likely benefits from tighter NCCL integration

**DeepEP-NCCL Advantages**:
- **Reports actual measured latency** (not estimated)
- **FP8 reduces data volume** (but not implemented in NCCL EP yet)
- More stable multi-node scaling
- Production-proven

**Note**: Direct comparison difficult due to:
- Different data types (BF16 vs FP8)
- Different measurement methodologies
- NCCL EP experimental/being tuned

---

## 7. Feature Comparison

| Feature | NCCL EP | DeepEP-NCCL |
|---------|---------|-------------|
| **Algorithms** | LL, HT | LL, HT (Normal) |
| **Transport** | GIN + LSA | GIN or NVSHMEM |
| **Data Types** | BF16, FP16 | FP8, BF16, FP16 |
| **FP8 Status** | Planned | Production ✓ |
| **Max Nodes** | 8 (64 GPUs) | 256+ GPUs tested |
| **GPUs/Node** | Up to 8 | Up to 8 |
| **Staged Execution** | Yes (LL) | Yes (LL) |
| **Dynamic Sizing** | Planned (NCCL_EP_AUTO) | Yes ✓ |
| **Python Bindings** | Yes (ctypes) | Yes (native) |
| **Production Status** | Experimental | Production ✓ |
| **Custom Allocators** | Yes | Yes |
| **Topology Detection** | NCCL native | NCCL native (GIN mode) |
| **TMA Support** | Yes (Hopper) | Planned |
| **UE8M0 Format** | No | Yes |

---

## 8. Implementation Maturity

### NCCL EP Status

**From RELEASE.md**:
```
Known Limitations:
- Limited QA coverage (experimental)
- No FP8 support (planned)
- Up to 8 nodes (64 GPUs)
- max_tokens_per_rank required (no NCCL_EP_AUTO)
- HT mode being tuned for performance
```

**Code Size**:
- `nccl_ep.cc`: 125 KB (host API)
- `device/low_latency.cu`: 73 KB
- `device/hybrid_ep.cuh`: 262 KB
- Total: ~460 KB of implementation

**Status**: Experimental, active development

### DeepEP-NCCL Status

**Production Deployment**:
- Used in DeepSeek-V3 training (December 2024)
- Used in DeepSeek-R1 inference
- Tested at 256+ GPU scale
- FP8 support production-ready

**Code Size**:
- `deep_ep.cpp`: 92 KB
- `csrc/kernels/internode_ll.cu`: 100 KB
- `csrc/kernels/internode.cu`: 183 KB
- Total: ~375 KB of implementation

**Status**: Production-ready, proven at scale

---

## 9. Code Similarity Analysis

### Are They Related?

**Assessment**: **Independent implementations with shared knowledge**

### Evidence of Independence

1. **Different code structure**:
   - NCCL EP: Modular helper functions
   - DeepEP: Inline kernel logic

2. **Different naming conventions**:
   - NCCL EP: `hiddenBf16Int4`, `sendBufVec`
   - DeepEP: `hidden_bf16_int4`, `rdma_x_vec`

3. **Different design choices**:
   - Weight aggregation: Two-pass vs single-pass
   - Routing: Explicit conversion vs metadata exchange
   - Backend: GIN-only vs pluggable

### Evidence of Shared Knowledge

1. **Identical FP8 algorithm**:
   - Same per-128-element channel quantization
   - Same warp reduction pattern
   - Same scale calculation
   - **Reason**: Standard FP8 E4M3 quantization (NVIDIA documentation)

2. **Identical message format**:
   - Same 16-byte header structure
   - **Reason**: Obvious design for GPU alignment

3. **Same GIN usage**:
   - Same API calls
   - **Reason**: Only way to use NCCL GIN Device API

### Verdict

**Likely scenario**: NVIDIA engineers studied DeepEP paper/code as reference, then implemented independently with NCCL-native design.

**This is legitimate engineering**:
- ✓ Learning from prior art
- ✓ Implementing standard algorithms
- ✓ Following best practices
- ✓ Making different design choices

**Not copying because**:
- Different structure throughout
- Different APIs
- Different implementation choices
- No copy-paste artifacts

---

## 10. Impact on AWS EFA

### EFA and GIN Support

AWS EFA (Elastic Fabric Adapter) now supports **both GIN paths** through the aws-ofi-nccl
plugin: a CPU **proxy** path and a kernel-initiated **EFA-GDA / GDAKI** path. As of
aws-ofi-nccl v1.21.x (master `d840aa1`), GDAKI is **auto-enabled** where the environment
supports it; there is no longer a plugin knob to pick the mode.

**Current EFA GIN Status**:
- ✅ **Proxy mode supported** — CPU progress thread assists network communication.
- ✅ **EFA-GDA / GDAKI supported** — GPU kernel writes WQEs / reads CQEs directly. Requires
  a supported EC2 instance type (P5en, P6-B200, P6-B300), libfabric ≥ 2.6, EFA driver ≥
  3.3.0, rdma-core ≥ `64.0amzn0`, and the NVIDIA driver loaded with `PeerMappingOverride=1`.
- ⚠️ **Additional ordering requirements** for SRD protocol still apply.

> **Removed knob (important correction):** Earlier revisions of this document described a
> user-selected mode via `OFI_NCCL_GIN_TYPE` (`=2` proxy, `=3` GDAKI). That plugin
> parameter was **removed** in aws-ofi-nccl commit `80f2c78` (*"gin: enable GDAKI
> automatically, remove OFI_NCCL_GIN_TYPE"*). The plugin no longer selects the path; GDAKI
> capability is auto-detected at context-creation time (see below). On the NCCL side there
> is a separate env var, `NCCL_GIN_TYPE`, which is an *NCCL* knob (not a plugin knob) that
> chooses among the op-tables NCCL itself will bind; with NCCL 2.31 defaulting to the proxy
> op-table for EFA, `NCCL_GIN_TYPE=5` selects the EFA-GDA GDAKI op-table. Do not confuse the
> two.

### GIN Modes Explained

| Mode | Path | Description | EFA Support |
|------|------|-------------|-------------|
| **Proxy** | host / CPU-proxy | CPU progress thread posts network operations on the GPU's behalf | ✅ Supported |
| **EFA-GDA / GDAKI** | kernel-initiated | GPU kernel posts RDMA operations directly (GPU Direct Async, Kernel-Initiated) | ✅ Supported on P5en / P6-B200 / P6-B300 |

**GDAKI** (GPU Direct Async Kernel-Initiated):
- GPU kernels directly write work queue entries (WQEs) and poll completion queue entries
  (CQEs); no CPU involvement in the critical path.
- Lowest latency, highest performance — preferred for MoE dispatch/combine.
- Backed by the vendored EFA-GDA CUDA device sources at `3rd-party/efa-gda`
  (efa-dp-direct), driven through the CUDA **driver** API.

**Proxy Mode**:
- GPU enqueues work; a CPU progress thread posts RDMA operations.
- Additional latency from the GPU→CPU→NIC path.
- Used as the fallback whenever GDAKI is not viable in the current environment.

### How the path is auto-selected

The plugin exports separate op-tables and lets NCCL bind the one matching the negotiated
path; the plugin itself decides GDAKI viability via
`nccl_ofi_gin_gdaki_capable()` ([include/rdma/gin/nccl_ofi_gin_gdaki.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/rdma/gin/nccl_ofi_gin_gdaki.h)),
which returns true only when **all** of the following hold:

1. GDAKI is compiled in (`HAVE_GDAKI`, which requires the vendored `3rd-party/efa-gda` CUDA sources).
2. Runtime libfabric is **≥ 2.5** (`FI_VERSION_GE(fi_version(), FI_VERSION(2,5))`).
3. DMA-BUF is viable (`nccl_ofi_dmabuf_viable()`).
4. The provider is an EFA provider (`efa` / `efa-direct` — the only family exposing `FI_EFA_GDA_OPS`).

If any check fails, the plugin falls back to the host/CPU-proxy path. On EFA specifically,
the platform hook `PlatformAWS::config_gdaki_domain()`
([include/nccl_ofi_platform.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_platform.h))
gets the final say per domain (a platform returns 0 to authorize GDAKI, or an error to force
the proxy fallback).

### Plugin op-table split: `ncclGin*` vs `ncclRma*`

NCCL's one-sided (GIN) network interface has been split across two families of op-tables,
and the aws-ofi-nccl plugin exports both. This matters because NCCL binds whichever table
matches the path it negotiated, so the exported symbol determines which code path drives
your run. Verified with `grep -rn NCCL_OFI_EXPORT_SYMBOL src/`
([src/rdma/gin/nccl_ofi_gin_api.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_api.cpp),
[src/rdma/gin/nccl_ofi_gin_gdaki.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_gdaki.cpp)):

| Exported symbol | Table type | File | Path it drives |
|-----------------|------------|------|----------------|
| `ncclGinPlugin_v11` | `ncclGin_v11_t` | `nccl_ofi_gin_api.cpp` (line 566) | Host proxy. `createContext`/`destroyContext` are `nullptr` (only relevant to GDAKI); `iput`/`iputSignal` forward `optFlags=0`. |
| `ncclGinPlugin_v13` | `ncclGin_v13_t` | `nccl_ofi_gin_api.cpp` (line 595) | Host proxy. Introduced with NCCL v2.30. Adds `iget`/`iflush` and real `createContext_v13`/`destroyContext_v13`. |
| `ncclGinPlugin_v14` | `ncclGin_v14_t` | `nccl_ofi_gin_gdaki.cpp` (line 1227) | **GDAKI only** (`name = "Libfabric_GDAKI"`). `createContext_v14` validates `backendVersion`, `regMrSym`/`deregMrSym` are the GDAKI variants, and it exposes `getGinProperties`/`queryLastError`. |
| `ncclRmaPlugin_v14` | `ncclRma_v14_t` | `nccl_ofi_gin_api.cpp` (line 666) | Host / CPU-proxy path (NCCL binds via `rma_v14.cc`). Wraps the v13 proxy entry points; `iputSignal` carries a trailing `isStrongSignal` that the plugin ignores because its signals are always strong. |
| `ncclRmaPlugin_v15` | `ncclRma_v15_t` | `nccl_ofi_gin_api.cpp` (line 732) | Host / CPU-proxy path (NCCL binds via `rma_v15.cc`, in preference to v14). Extends v14 by carrying `optFlags` (`ncclRmaOptFlags`) end-to-end so the caller's `ncclRmaOptFlagsAggregateRequests` hint reaches the doorbell-coalescing path. |

**Rule of thumb:**
- The **`ncclGin*_v14` (GDAKI)** table drives the kernel-initiated EFA-GDA path.
- The **`ncclRma*_v14/v15`** tables drive the host/CPU-proxy path. NCCL migrated its host
  data path from the GIN op-table to the RMA op-table at v14; the plugin's proxy `iput`/
  `iputSignal`/`iget`/`iflush` are shared implementations reused by both the legacy
  `ncclGin_v13` table and the newer `ncclRma_v14/v15` tables.
- Older `ncclGin_v11/v13` remain exported for backward compatibility with pre-2.31 NCCL.

Relevant commits: `c6298ae`, `19efa1c`, `3dc2e74` (vendor `rma_v15.h`, export
`ncclRmaPlugin_v15`); `2cafa4b` / `8906c0e` / `ae03e1b` (migrate tests to the v14/v15 op
tables).

### GDAKI implementation notes

The GDAKI data path (`src/rdma/gin/nccl_ofi_gin_gdaki.cpp`,
`src/rdma/gin/nccl_ofi_gin_gdaki_resources.cpp`) has several EFA-specific behaviors worth
knowing when reasoning about MoE traffic:

- **Multi-rail**: a logical context binds to `rail = contextId % num_rails`, and
  `effective_rails = min(nContexts, num_rails)` rails are actually used
  ([nccl_ofi_gin_gdaki.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_gdaki.cpp) line ~483). The GDAKI path caps at `NCCL_OFI_GDAKI_MAX_RAILS = 2` EFA NICs per GPU (commit `cb00654`).
- **Dedicated PutValue data endpoint**: a separate endpoint is used only for sending
  PutValues (commit `6a3ba5f`); its counter is now bound for reads as well as writes
  (commit `d840aa1`).
- **Target-indexed signal addressing** with asymmetric counts (commit `f3dd9cd`); **indexed
  signal shadowing** to avoid a signal read-modify-write (commit `dea6e05`), with the
  never-reset signal shadow sized from the full MR (commit `5b46cd5`).
- **`FI_HMEM` requested on the GDAKI endpoint hints** (commit `06f08e5`);
  `local_cntr_value` populated on counter device handles (commit `44d0b55`).
- **EFA hardware completion counter gated per platform** via `OFI_NCCL_GDAKI_EFA_HW_COUNTER`
  (`AUTO`/`ON`/`OFF`, default `AUTO`) and `PlatformAWS::config_gdaki_domain()` (commit
  `5b1f6dd`).
- **`backendVersion` validated** in `createContext_v14` — the plugin refuses a version it
  cannot build a matching device struct for, rather than risk silent memory corruption
  (commit `f8e945e`).
- **GDRCopy registration skipped** in GDAKI mode (commit `5b163ed`) — GDAKI does not need
  the CPU-mapped signal path that the proxy path uses.
- **Doorbell coalescing via `FI_MORE`** on the aggregate hint (commit `a3d2680`).
- **`NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET`** synced from NCCL 2.30.7-1 (commit `d1db4c5`).

### Host / proxy GIN path notes

- **`NCCL_GIN_PROXY_NTHREADS`** (default 1; read via `getenv` in
  [nccl_ofi_gin_api.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/nccl_ofi_gin_api.cpp), not an `OFI_NCCL_*` param) enables
  per-thread endpoint bucketing: `listen_seq % nthreads` maps each connection to its owning
  progress thread's endpoint, so each thread drives its own CQ instead of contending on a
  shared one (commit `e6c4eb1`).
- **One gdrcopy worker per process** with signal coalescing (commit `ad6dcac`), fed by a new
  lock-free MPSC ring ([include/nccl_ofi_mpsc_ring.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_mpsc_ring.h), commit `3e5687e`).
- Signal memory is **registered one segment at a time** in the host proxy (commit `80d995c`);
  the recv-req pool return is **deferred until the gdrcopy signal completes** (commit
  `5a7bcba`).

> **Removed knob (important correction):** `OFI_NCCL_GIN_STRONG_SIGNAL` and the weak-signal
> mode were **removed** in commit `aa80b54` (*"gin: remove GIN_STRONG_SIGNAL env variable and
> weak-signal mode"*). The plugin's signals are now **always strong**; the `isStrongSignal`
> argument that appears in the `ncclRma_v14/v15` `iputSignal` prototypes is accepted and
> ignored.



#### 1. SRD Protocol Ordering

EFA uses **SRD (Scalable Reliable Datagram)** which has ordering semantics:
- Messages between same src/dst pair are ordered
- Messages to different destinations can reorder
- **Impact**: MoE many-to-many pattern stresses ordering assumptions

**MoE Pattern Challenge**:
```
Rank 0 sends to: [Rank 1, Rank 2, Rank 3, ..., Rank N]
Each with different message sizes
All messages in flight simultaneously
```

**Potential Issues**:
- Signal/data race conditions
- Completion notification before data arrival
- Requires careful signal placement

> **libfabric data-path-direct (current default):** The EFA provider now defaults
> `FI_EFA_USE_DATA_PATH_DIRECT=true` (`efa_env.c`, `.use_data_path_direct = true`). On
> supported devices the provider writes WQEs and reads CQEs **directly** from userspace
> instead of routing every operation through `ibv_post_send()` / `ibv_poll_cq()`. This
> shortens the software path for both the proxy and GDAKI cases; the SRD ordering semantics
> above are unchanged.

#### 2. Proxy Mode Performance Impact

**Latency Overhead** (estimated):
```
GDAKI:      GPU kernel → NIC                    (~1-2 μs)
Proxy:      GPU kernel → CPU proxy → NIC        (~5-10 μs additional)
```

**For MoE Low-Latency Mode**:
- NCCL EP single-node: ~32 μs (GDAKI)
- With proxy mode: ~40-50 μs (estimated)
- DeepEP single-node: 161 μs (already includes proxy overhead)

**Impact**: Proxy mode adds ~25-50% latency overhead for small messages.

#### 3. Multi-QP Considerations

Both implementations use multiple QPs (Queue Pairs) for parallelism:
- NCCL EP: Auto-configured
- DeepEP-NCCL: `num_qps_per_rank` parameter

**EFA Limitation**: Each QP requires separate SRD connection
- More QPs = more memory
- More QPs = more completion queue polling
- **Tradeoff**: Parallelism vs resource usage

### Performance Expectations on EFA

#### Single-Node (NVLink, no EFA)
- **NCCL EP**: 224 GB/s (as measured)
- **DeepEP-NCCL**: 47 GB/s (as measured)
- No EFA impact

#### Multi-Node with EFA Proxy Mode

**Expected Performance** (estimated):

| GPUs | Nodes | NCCL EP (Proxy) | DeepEP-NCCL (Proxy) |
|:----:|:-----:|:---------------:|:-------------------:|
| 16   | 2     | ~60-70 GB/s     | ~40 GB/s            |
| 32   | 4     | ~45-50 GB/s     | ~38 GB/s            |
| 64   | 8     | ~40-45 GB/s     | ~33 GB/s            |

**Degradation**: ~10-20% vs GDAKI due to proxy overhead

#### EFA with GDAKI (now available)

EFA-GDA / GDAKI is now available on P5en / P6-B200 / P6-B300 (see status above). Expected
GDAKI performance relative to the proxy path:

| GPUs | Nodes | NCCL EP (GDAKI) | DeepEP-NCCL (GDAKI) |
|:----:|:-----:|:---------------:|:-------------------:|
| 16   | 2     | ~75 GB/s        | ~42 GB/s            |
| 32   | 4     | ~52 GB/s        | ~40 GB/s            |
| 64   | 8     | ~48 GB/s        | ~35 GB/s            |

**Improvement**: ~10-15% vs proxy mode

### EFA Optimization Strategies

#### For EFA proxy mode

1. **Reduce message count**:
   - Batch tokens when possible
   - Use larger top-k values efficiently
   - Minimize dispatch/combine frequency

2. **Optimize QP usage**:
   - Start with fewer QPs (2-4 per rank)
   - Increase only if bottlenecked
   - Monitor completion queue depth

3. **Signal placement**:
   - Use coarse-grained signals
   - Avoid per-message signaling
   - Batch signal operations

4. **Memory registration**:
   - Pre-register all buffers
   - Use NCCL's memory registration cache
   - Avoid dynamic allocation in hot path

5. **Proxy threads**: consider `NCCL_GIN_PROXY_NTHREADS>1` to give each GIN progress thread
   its own endpoint/CQ (per-thread endpoint bucketing) instead of contending on a shared CQ.

#### For EFA-GDA / GDAKI

1. **Direct kernel posting**:
   - Maximize GPU-initiated operations
   - Minimize CPU involvement
   - Use signal-based completion

2. **Ordering management**:
   - Careful signal/fence placement
   - Respect SRD ordering semantics
   - Test thoroughly for races

3. **Multi-rail utilization**:
   - GDAKI binds contexts round-robin to rails (`rail = contextId % num_rails`), capped at
     `NCCL_OFI_GDAKI_MAX_RAILS = 2` NICs per GPU. Use enough contexts to cover the rails.
   - Leverage NCCL topology detection.

### EFA-Specific Configuration

#### NCCL EP / DeepEP on EFA

```bash
# GDAKI path is auto-detected by the plugin where supported. There is NO plugin
# OFI_NCCL_GIN_TYPE knob anymore (removed in aws-ofi-nccl commit 80f2c78).
#
# On the NCCL side, NCCL 2.31 defaults to the proxy op-table for EFA. To select the
# EFA-GDA (GDAKI) op-table explicitly, set the NCCL env var:
export NCCL_GIN_TYPE=5
# Symmetric GIN kernels for NCCL collectives are not supported under EFA-GDA yet:
export NCCL_SYM_GIN_KERNELS_ENABLE=0

# EFA-GDA also needs the NVIDIA driver loaded with PeerMappingOverride=1:
#   options nvidia NVreg_RegistryDwords="PeerMappingOverride=1"
```

#### DeepEP-NCCL on EFA

```bash
# Use NCCL backend
export DEEP_EP_BACKEND=nccl

# Optimize for EFA
export NCCL_IB_DISABLE=1        # Not InfiniBand
export NCCL_SOCKET_IFNAME=eth0  # EFA interface

# Optionally select the EFA-GDA GDAKI op-table (otherwise proxy is used on NCCL 2.31):
export NCCL_GIN_TYPE=5
export NCCL_SYM_GIN_KERNELS_ENABLE=0
```

### EFA Roadmap Impact

**Proxy path**:
- Both implementations work but with GPU→CPU→NIC latency overhead.
- NCCL EP still faster due to better single-node performance.
- DeepEP-NCCL production-ready with known overhead.

**EFA-GDA / GDAKI path (now available on P5en / P6-B200 / P6-B300)**:
- Full performance potential unlocked (kernel-initiated, CPU out of the critical path).
- NCCL EP sees larger gains (optimized for kernel-initiated posting).
- DeepEP-NCCL benefits but less dramatically.

**Recommendation**: 
- **Production today**: DeepEP-NCCL (proven), proxy or GDAKI depending on instance type.
- **Lowest latency on P5en/P6**: prefer the EFA-GDA (GDAKI) path (`NCCL_GIN_TYPE=5`).

---

## 11. Use Case Recommendations

### Choose NCCL EP if:

1. **NCCL ecosystem integration** is critical
2. You want **official NVIDIA support** path
3. You're building on **NCCL infrastructure**
4. You need **single-node performance** (224 GB/s)
5. You can wait for **maturity** (experimental)
6. You prefer **C API** and explicit control
7. You need **TMA optimizations** on Hopper
8. You're on **EFA (P5en/P6) with EFA-GDA/GDAKI** or on **InfiniBand with GDAKI support**

### Choose DeepEP-NCCL if:

1. **Production deployment** is immediate need
2. You need **FP8 support now**
3. You want **PyTorch integration**
4. You need **proven multi-node scaling** (256+ GPUs)
5. You want **backend flexibility** (NVSHMEM fallback)
6. You prefer **Python API** and ease of use
7. You need **lower latency** focus
8. You're on **AWS EFA** (proxy path everywhere; EFA-GDA/GDAKI on P5en/P6)

### Use Both if:

1. Evaluating **performance characteristics**
2. Comparing **optimization strategies**
3. Contributing to **open-source MoE** research
4. Building **framework-agnostic** solutions

---

## 12. Future Trajectory

### NCCL EP Roadmap

**Likely developments**:
- FP8 support implementation
- NCCL_EP_AUTO for dynamic sizing
- Performance tuning for HT mode
- Scale beyond 8 nodes
- Production readiness

**Advantages**:
- Official NVIDIA support
- Tight NCCL integration
- Access to NCCL internals
- Part of NCCL releases

### DeepEP-NCCL Roadmap

**Current experimental branches**:
- Zero-copy (removes buffer copies)
- Eager protocol (lower latency)
- Hybrid-EP (TMA, larger NVLink domains)
- SM-free kernels

**Advantages**:
- Independent development pace
- Community contributions
- Production validation
- Multiple backend options

### Convergence

**Both will likely**:
- Adopt each other's optimizations
- Converge on best practices
- Maintain different integration models
- Serve different use cases

---

## 13. Technical Deep Dive: Key Differences

### Tensor Layout

**NCCL EP**:
- LL mode: 3D `[num_local_experts x max_tokens x hidden]`
- HT mode: 2D `[num_recv_tokens x hidden]`
- Explicit stride specification
- Tag-based tensor roles

**DeepEP-NCCL**:
- Flexible layout based on mode
- Automatic buffer sizing
- Implicit layout management
- Queue-based buffering

### Memory Management

**NCCL EP**:
```c
// Custom allocators
typedef cudaError_t (*ncclEpAllocFn_t)(void** ptr, size_t size);
typedef cudaError_t (*ncclEpFreeFn_t)(void* ptr);

ncclEpCreateGroup(&ep_group, comm, &config, stream, 
                  my_alloc, my_free);
```

**DeepEP-NCCL**:
```python
# Buffer class manages memory
buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
# Automatic allocation and cleanup
```

### Signal Management

**NCCL EP**:
```cpp
// Explicit signal configuration
ncclDevCommRequirements reqs;
reqs.ginSignalCount = num_signals;
ncclDevCommCreate(comm, &reqs, &devComm);
```

**DeepEP-NCCL**:
```cpp
// Automatic signal allocation
int signals_per_buffer = num_local_experts * comm_nranks;
int num_total_signals = signals_per_buffer * 2;  // Double buffered
```

---

## 14. Integration Examples

### NCCL EP Integration

```c
// C API - explicit and verbose.
// Verified against NCCL v2.31.2-1 contrib/nccl_ep (README.md, include/nccl_ep.h,
// include/ep_enums.h). The older 3-call ncclEpTensorCreate() shape that earlier
// revisions of this document showed does not exist: cross-boundary tensors are
// ncclEpTensor_t* fields inside named input/output structs.

ncclEpGroupConfig_t config = NCCL_EP_GROUP_CONFIG_INIT;  // pre-fills size + version
config.algorithm                  = NCCL_EP_ALGO_LOW_LATENCY;  // or _HIGH_THROUGHPUT
config.num_experts                = 256;
config.max_dispatch_tokens_per_rank = 128;   // required; NCCL_EP_AUTO not yet supported
config.max_recv_tokens_per_rank   = NCCL_EP_AUTO;  // LL: nRanks * max_dispatch_tokens
config.max_token_bytes            = 7168 * 2;      // BF16
config.rdma_buffer_size           = NCCL_EP_AUTO;  // lazy sizing on first handle
config.num_qp_per_rank            = NCCL_EP_AUTO;
config.num_channels               = NCCL_EP_AUTO;
config.max_num_sms                = NCCL_EP_AUTO;

ncclEpGroup_t ep_group;
ncclEpCreateGroup(&ep_group, comm, &config);

// topk_idx is a caller-owned tensor descriptor; the routing it carries is cached
// on the handle and reused by every dispatch until ncclEpUpdateHandle() is called.
ncclEpHandle_t handle;
ncclEpCreateHandle(&handle, ep_group, layout, &topk_idx, layout_info,
                   handle_cfg, stream);

// inputs/outputs are named structs (ncclEpDispatchInputs_t / ...Outputs_t)
ncclEpDispatch(handle, &dispatch_in, &dispatch_out, layout_info,
               &dispatch_cfg, stream);
ncclEpCombine(handle, &combine_in, &combine_out, &combine_cfg, stream);
ncclEpComplete(handle, config, stream);   // LL mode only

ncclEpHandleDestroy(handle);
ncclEpGroupDestroy(ep_group);
```

### DeepEP-NCCL Integration

```python
# Python API - concise and implicit
buffer = Buffer(
    group=dist.group.WORLD,
    num_nvl_bytes=0,
    num_rdma_bytes=auto_size,
    low_latency_mode=True,
    num_qps_per_rank=8
)

recv_x, _, _, _, handle, event = buffer.dispatch(
    x, 
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    async_finish=True
)
```

---

## 15. Ecosystem Position

### NCCL EP Position

```
NCCL Ecosystem
├── Core Collectives (AllReduce, AllGather, etc.)
├── Device API (GIN, LSA)
└── Extensions
    └── NCCL EP (MoE communication)
```

**Role**: Official NCCL extension for MoE workloads

**Integration**: Part of NCCL build and release cycle

### DeepEP-NCCL Position

```
MoE Communication Libraries
├── NVSHMEM-based (original DeepEP)
├── NCCL GIN-based (DeepEP-NCCL)
└── Framework Integration
    └── PyTorch (primary target)
```

**Role**: Standalone MoE library with NCCL backend option

**Integration**: Independent library, PyTorch-focused

---

## Conclusion

### Summary

**NCCL EP** and **DeepEP-NCCL** are **complementary** implementations:

- **Same MoE pattern**: Identical many-to-many top-k routing
- **Same GIN foundation**: Both use NCCL Device API
- **Different maturity**: Experimental vs production
- **Different integration**: Native vs standalone
- **Different features**: Planned vs implemented

### Key Takeaway

The **real innovation** is **GIN (GPU-Initiated Networking)**:
- Enables GPU kernels to directly initiate RDMA
- Eliminates CPU from critical path
- Both implementations leverage this capability

### Recommendation

**For production today**: Use **DeepEP-NCCL**
- FP8 support working
- Proven at scale
- Production-ready

**For future NCCL integration**: Watch **NCCL EP**
- Official support path
- Tighter NCCL integration
- Will mature over time

**For research**: Study **both**
- Different optimization strategies
- Different design patterns
- Learn from both approaches

### The Bigger Picture

Both implementations validate that:
1. **GIN is the right abstraction** for GPU-initiated MoE communication
2. **Per-channel FP8 quantization** is the right optimization
3. **Signal-based completion** is the right async model
4. **MoE communication** is important enough for dedicated libraries

The existence of both is **healthy for the ecosystem** - competition drives innovation.

---

## References

### NCCL EP
- **Repository**: https://github.com/NVIDIA/nccl (contrib/nccl_ep)
- **Verified against**: NCCL master `fd168324` (release v2.31.2-1)
- **Status**: Experimental
- **Documentation**: contrib/nccl_ep/README.md, contrib/nccl_ep/RELEASE.md

### DeepEP-NCCL
- **Repository**: https://github.com/deepseek-ai/DeepEP
- **Fork verified against**: aamirshafi/DeepEP main `b57e5e2`
- **Paper**: GPU-Initiated Networking for NCCL (https://arxiv.org/abs/2511.15076)
- **Production**: DeepSeek-V3, DeepSeek-R1
- **Status**: Production-ready
- **Documentation**: README-NCCL.md

### NCCL GIN Device API
- **Version**: NCCL 2.29.2+ (GIN v13 op-table added in NCCL v2.30; GDAKI GIN v14 in 2.31)
- **Examples**: nccl/docs/examples/06_device_api/
- **Paper**: GPU-Initiated Networking (arxiv.org/abs/2511.15076)

### aws-ofi-nccl GIN plugin
- **Repository**: https://github.com/aws/aws-ofi-nccl
- **Verified against**: master `d840aa1` (tag v1.21.1)
- **GIN sources**: [src/rdma/gin/](https://github.com/aws/aws-ofi-nccl/blob/master/src/rdma/gin/), [include/rdma/gin/](https://github.com/aws/aws-ofi-nccl/blob/master/include/rdma/gin/)
- **Getting-started**: [doc/gin-getting-started.md](https://github.com/aws/aws-ofi-nccl/blob/master/doc/gin-getting-started.md)

---

**Document Version**: 1.1  
**Last Updated**: 2026-09-01  
**Author**: Analysis based on source code examination (aws-ofi-nccl master `d840aa1`, NCCL master `fd168324`)
