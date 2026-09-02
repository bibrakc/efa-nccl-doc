# NCCL on AWS EFA Documentation

Comprehensive technical documentation for NCCL (NVIDIA Collective Communications Library) using the OFI plugin with libfabric and AWS EFA (Elastic Fabric Adapter).

## Purpose

This documentation provides in-depth coverage of the complete stack used for GPU-to-GPU communication on AWS EC2 instances with EFA. It's designed for engineers working on optimization, debugging, and understanding the performance characteristics of distributed deep learning workloads.

## Source Baseline

Every claim in these documents was verified against the following upstream snapshot.
Source links use branch-form URLs so they always land on current code; see
[PERMALINK_STATUS.md](PERMALINK_STATUS.md) for the link policy and verification results.

| Component | Version | Commit |
| --- | --- | --- |
| NCCL | v2.31.2-1 | `fd168324` |
| aws-ofi-nccl (OFI plugin) | v1.21.1 | `d840aa1` |
| libfabric | 2.7.0rc1 | `cb6364e05` |
| rdma-core | 65.0 dev (release v64.0) | `8b3a3e64d` |
| EFA kernel driver (amzn-drivers) | r3.3.0 | `44c91d1` |
| open-gpu-kernel-modules | 610.57.04 | `e4a5faa2` |

Verified 2026-09-01.

## Documentation Structure

### Core Concepts

- **[overview.md](overview.md)** - Architecture overview and component interactions
- **[nccl-core.md](nccl-core.md)** - NCCL fundamentals, topology, communicators
- **[nccl-usage.md](nccl-usage.md)** - Communicators, operations, threading, and practical usage patterns
- **[nccl-channels.md](nccl-channels.md)** - NCCL channels: parallelism, establishment, multi-rail support, tuning
- **[nccl-collectives.md](nccl-collectives.md)** - Collective operations, algorithms (Ring/Tree), protocols (Simple/LL/LL128)
- **[nccl-tuner.md](nccl-tuner.md)** - Cost-based algorithm/protocol selection, AWS region-based tuner, channel count tuning
- **[nccl-datapath.md](nccl-datapath.md)** - Complete data path from GPU through network

### Collective Algorithms (Detailed Analysis)

- **[algorithms/ring-algorithm.md](algorithms/ring-algorithm.md)** - Ring algorithm arranges GPUs in a circular topology for bandwidth-optimal collective operations
- **[algorithms/tree-algorithm.md](algorithms/tree-algorithm.md)** - Double binary tree algorithm provides latency-optimized communication for small to medium messages
- **[algorithms/nvls-tree-algorithm.md](algorithms/nvls-tree-algorithm.md)** - NVLS Tree combines tree-based communication with NVSwitch hardware acceleration for reduced latency
- **[algorithms/pat-algorithm.md](algorithms/pat-algorithm.md)** - Parallel Aggregated Trees uses adaptive communication patterns for AllGather and ReduceScatter operations

### EFA Hardware and Protocol

- **[efa-hardware-architecture.md](efa-hardware-architecture.md)** - EFA queue pairs, completion queues, work queue entries, memory layout, hardware capabilities
- **[srd-protocol.md](srd-protocol.md)** - Scalable Reliable Datagram protocol: multipath routing, congestion control, reliability mechanisms
- **[efa-driver.md](efa-driver.md)** - EFA driver architecture, capabilities, and verbs surface exposed to userspace
- **[kernel-efa-driver.md](kernel-efa-driver.md)** - Linux kernel EFA driver internals

### Transport Layer

- **[ofi-plugin.md](ofi-plugin.md)** - OFI NCCL plugin architecture and implementation
- **[libfabric-overview.md](libfabric-overview.md)** - Libfabric API, capabilities, data transfer operations
- **[efa-provider.md](efa-provider.md)** - EFA provider specifics, eager/rendezvous protocols
- **[rdma-core-and-verbs.md](rdma-core-and-verbs.md)** - rdma-core userspace libraries, libibverbs API, kernel uverbs interface, EFA provider implementation

### System Level

- **[threading-model.md](threading-model.md)** - Threading across the stack, proxy threads, concurrency
- **[rdma-memreg.md](rdma-memreg.md)** - RDMA operations, memory registration, caching strategies

### Performance

- **[optimizations.md](optimizations.md)** - Kernel bypass, caching, tuning parameters, best practices
- **[optimization-opportunities.md](optimization-opportunities.md)** - Detailed analysis of optimization opportunities with priorities

### Deep Dives

- **[lkey-rkey-explained.md](lkey-rkey-explained.md)** - Understanding local and remote keys in RDMA
- **[ofi-plugin-protocols.md](ofi-plugin-protocols.md)** - Connection establishment and send/recv protocols
- **[topology-and-binding.md](topology-and-binding.md)** - Topology detection and interface binding mechanisms
- **[freelist-allocator.md](freelist-allocator.md)** - Freelist allocator for high-performance object pooling (estimated 20-100x faster than malloc)
- **[mr-cache-implementation.md](mr-cache-implementation.md)** - Memory registration cache implementation details (approximately 25x speedup)
- **[nccl-buffsize-explained.md](nccl-buffsize-explained.md)** - NCCL_BUFFSIZE environment variable, channel buffers, pipelining, and performance/memory tradeoffs
- **[nccl-message-breakdown-complete.md](nccl-message-breakdown-complete.md)** - How NCCL breaks down messages for all collective operations
- **[nccl-chunk-breakdown.md](nccl-chunk-breakdown.md)** - Chunk size calculation, pipeline depth, and EFA MTU packetization

### GIN and Expert Parallelism

- **[nccl-ep-vs-deepep-comparison.md](nccl-ep-vs-deepep-comparison.md)** - NCCL EP vs DeepEP-NCCL comprehensive comparison for MoE communication
- **[gin-alltoall-libfabric-trace.md](gin-alltoall-libfabric-trace.md)** - Libfabric call trace for GIN hybrid LSA all-to-all (16 nodes, 128 ranks)

### GPU and Accelerator Memory

- **[cuda-memory.md](cuda-memory.md)** - NVIDIA CUDA memory: dynamic loading, cross-version compatibility, GPUDirect, GDRCopy
- **[dmabuf-gpu-memory.md](dmabuf-gpu-memory.md)** - DMA-BUF framework for vendor-neutral GPU memory registration
- **[neuron-memory.md](neuron-memory.md)** - AWS Trainium/Inferentia with P2P registration (not dmabuf)
- **[rocm-memory.md](rocm-memory.md)** - AMD ROCm with HIP API (CUDA-compatible)

### Documentation Metadata

- **[PERMALINK_STATUS.md](PERMALINK_STATUS.md)** - Source-link policy, the upstream snapshot these docs were verified against, and per-document link inventory
- **[PERMALINK_MAPPINGS.txt](PERMALINK_MAPPINGS.txt)** - Struct/class/function name to source location mappings

## Quick Start

### Understanding the Stack

Start with these documents in order:

1. [overview.md](overview.md) - Get the big picture
2. [nccl-core.md](nccl-core.md) - Understand NCCL basics
3. [nccl-datapath.md](nccl-datapath.md) - Follow a message through the system
4. [optimizations.md](optimizations.md) - Apply optimizations

### For Specific Topics

- **Performance tuning** → [optimizations.md](optimizations.md), [optimization-opportunities.md](optimization-opportunities.md)
- **Debugging network issues** → [ofi-plugin.md](ofi-plugin.md), [efa-provider.md](efa-provider.md)
- **Communicator usage patterns** → [nccl-usage.md](nccl-usage.md)
- **Channel configuration and tuning** → [nccl-channels.md](nccl-channels.md)
- **Algorithm selection** → [nccl-collectives.md](nccl-collectives.md), [algorithms/](algorithms/) for detailed analysis
- **Collective algorithms deep dive** → [algorithms/ring-algorithm.md](algorithms/ring-algorithm.md), [algorithms/tree-algorithm.md](algorithms/tree-algorithm.md), [algorithms/nvls-tree-algorithm.md](algorithms/nvls-tree-algorithm.md), [algorithms/pat-algorithm.md](algorithms/pat-algorithm.md)
- **Memory registration problems** → [rdma-memreg.md](rdma-memreg.md), [mr-cache-implementation.md](mr-cache-implementation.md)
- **Threading issues** → [threading-model.md](threading-model.md)
- **Connection protocols** → [ofi-plugin-protocols.md](ofi-plugin-protocols.md)
- **Topology and multi-NIC** → [topology-and-binding.md](topology-and-binding.md), [nccl-channels.md](nccl-channels.md)
- **Understanding RDMA keys** → [lkey-rkey-explained.md](lkey-rkey-explained.md)
- **Memory allocators & caching** → [freelist-allocator.md](freelist-allocator.md), [mr-cache-implementation.md](mr-cache-implementation.md)
- **GPU memory registration** → [cuda-memory.md](cuda-memory.md), [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md), [neuron-memory.md](neuron-memory.md), [rocm-memory.md](rocm-memory.md)
- **Kernel driver details** → [kernel-efa-driver.md](kernel-efa-driver.md)
- **Userspace RDMA API** → [rdma-core-and-verbs.md](rdma-core-and-verbs.md)

## Key Concepts

### Architecture Stack

```
Application (PyTorch, TensorFlow, DeepEP)
    ↓
NCCL Library (Collectives, Algorithms; Net + GIN/RMA plugin APIs)
    ↓
OFI NCCL Plugin (Network Transport: RDMA / SendRecv / GIN[GDAKI|host proxy])
    ↓
Libfabric (Fabric Abstraction)
    ↓
EFA Provider (Hardware-Specific)
    │
    ├── data path (default): Data Path Direct — provider writes WQEs into the
    │   mapped SQ, reads CQEs from the mapped CQ, rings the doorbell via MMIO
    │   ↓                                     (bypasses rdma-core entirely)
    │   │
    └── control path: rdma-core (libibverbs) ──┤  device open, QP/CQ create,
        (also the data path when                  MR registration, AH create,
         FI_EFA_USE_DATA_PATH_DIRECT=0)           queue-buffer mmap
                                               │
                                               ↓
EFA Kernel Driver (efa.ko, r3.3.0)  ─────────────  [open source boundary]
    ↓                                          everything above is open;
EFA NIC Firmware (SRD engine, congestion       firmware + ASIC below are closed
    ↓             control, multipath)
EFA Hardware (p4d: 4×100G EFAv2; p5/p5en/p6: EFAv3; up to 800/1600 Gbps
              link speeds reported by r3.3.0)
```

> **Data path direct.** Since libfabric 2.3.0, and enabled for `efa-rdm` and
> **on by default** as of 2.7.x, the EFA provider drives the mapped SQ/RQ/CQ
> buffers itself instead of calling `ibv_post_send()` / `ibv_poll_cq()`. rdma-core
> remains responsible for the control path. Set `FI_EFA_USE_DATA_PATH_DIRECT=0`
> to fall back to the libibverbs data path. See
> [efa-provider.md](efa-provider.md) and [rdma-core-and-verbs.md](rdma-core-and-verbs.md).

> The kernel driver is the lowest **open-source** layer. The SRD wire protocol,
> congestion control, and multipath routing run in **closed NIC firmware**; the
> driver communicates with it only via admin-queue commands and data-path
> doorbells/completion queues. See [kernel-efa-driver.md](kernel-efa-driver.md)
> for the firmware/hardware boundary and the admin-queue and GPU peer-memory internals.

### Data Flow

1. **Application** calls NCCL collective (e.g., AllReduce)
2. **NCCL** breaks into algorithm steps (Ring/Tree/PAT/NVLS)
3. **Proxy threads** handle network I/O
4. **OFI plugin** translates to libfabric operations
5. **Libfabric** posts to EFA provider
6. **EFA provider** builds WQEs directly into the mapped send queue and rings the
   doorbell via MMIO (Data Path Direct, the default). rdma-core is not on this path;
   it set the queues up during connection establishment
7. **EFA driver** DMAs from GPU memory
8. **EFA hardware** transmits over network with SRD reliability
9. **Remote side** receives and completes in reverse; the provider reads CQEs straight
   out of the mapped completion queue

### Critical Performance Factors

1. **Memory Registration Caching** - Estimated 100-500x speedup for repeated transfers
2. **Algorithm Selection** - Ring for large messages (>1 MB), Tree for medium (100 KB-1 MB), PAT for small-medium at scale (32 KB-1 MB), NVLS Tree for intra-node with NVSwitch. See [algorithms/](algorithms/) for complete mathematical analysis
3. **Protocol Selection** - LL for latency (<32 KB), LL128 for balanced (32 KB-1 MB), Simple for bandwidth (>1 MB)
4. **Multi-rail** - Utilize all EFA adapters (4-8 per instance)
5. **CPU Affinity** - Pin proxy threads to avoid NUMA remote access
6. **Kernel Bypass** - Direct user-space access to hardware
7. **Data Path Direct** - The EFA provider bypasses rdma-core/libibverbs on the data
   path by default, removing a call layer per operation. See
   [optimizations.md](optimizations.md)

## Additional Resources

### Official Documentation

- NCCL: https://docs.nvidia.com/deeplearning/nccl/
- OFI Plugin: https://github.com/aws/aws-ofi-nccl
- Libfabric: https://ofiwg.github.io/libfabric/
- AWS EFA: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html

### Repositories

- NCCL: https://github.com/NVIDIA/nccl
- aws-ofi-nccl: https://github.com/aws/aws-ofi-nccl
- libfabric: https://github.com/ofiwg/libfabric
- nccl-tests: https://github.com/NVIDIA/nccl-tests

## Contributing

This documentation is focused on understanding the existing implementation for optimization purposes. For improvements to the actual software components:

- NCCL issues/PRs: https://github.com/NVIDIA/nccl
- OFI plugin issues/PRs: https://github.com/aws/aws-ofi-nccl
- Libfabric issues/PRs: https://github.com/ofiwg/libfabric

## Changelog

### 2026-09-01

#### Stack refresh (data path direct, GIN/RMA API split, EFA r3.3.0, NCCL 2.31)

**Version baseline** moved to September 2026: NCCL **v2.31.2-1**, aws-ofi-nccl
**v1.21.1** (master `d840aa1`), libfabric **2.7.0rc1**, rdma-core **65.0 dev**
(release v64.0), EFA kernel driver **r3.3.0**. The full snapshot with commit SHAs is
now recorded in one place, [Source Baseline](#source-baseline) above, instead of being
scattered across documents.

- **EFA Data Path Direct is now the default, and it removes rdma-core from the data
  path.** `FI_EFA_USE_DATA_PATH_DIRECT` defaults to true, so the libfabric EFA provider
  writes work-queue entries into the mapped SQ, reads CQEs from the mapped CQ buffer,
  and rings the doorbell via MMIO itself, rather than calling `ibv_post_send()` /
  `ibv_poll_cq()`. rdma-core still owns the control path (device open, QP/CQ creation,
  MR registration, AH create/destroy, queue-buffer mmap). Documented across
  [efa-provider.md](efa-provider.md), [rdma-core-and-verbs.md](rdma-core-and-verbs.md),
  [efa-hardware-architecture.md](efa-hardware-architecture.md),
  [nccl-datapath.md](nccl-datapath.md), [libfabric-overview.md](libfabric-overview.md)
  and [optimizations.md](optimizations.md), including the 128-byte wide WQE format,
  64-bit request IDs, and the QP generation embedded in the request ID. The libibverbs
  data path is retained in full as the documented fallback.
- **The GIN plugin API split into `ncclGin_*` and `ncclRma_*` families.** The plugin now
  exports `ncclGinPlugin_v11`, `ncclGinPlugin_v13`, `ncclGinPlugin_v14` (GDAKI-only) and
  `ncclRmaPlugin_v14` / `ncclRmaPlugin_v15` (host/CPU-proxy path), alongside
  `ncclNetPlugin_v4`–`v12` and `ncclTunerPlugin_v2`/`v3`/`v6`. Corrected in
  [overview.md](overview.md), [ofi-plugin.md](ofi-plugin.md),
  [ofi-plugin-protocols.md](ofi-plugin-protocols.md) and
  [nccl-ep-vs-deepep-comparison.md](nccl-ep-vs-deepep-comparison.md).
- **`OFI_NCCL_GIN_TYPE` was removed; GDAKI auto-enables.** The previous documentation
  described a user-selected proxy-vs-GDAKI mode, which no longer exists (commit
  `80f2c78`). `OFI_NCCL_GIN_STRONG_SIGNAL` and weak-signal mode were also removed
  (`aa80b54`). Both are now documented as removed, with the real auto-detection and
  fallback logic in their place, so an agent reading older code still understands them.
- **Eager messages are disabled by default** (`7df78cd`, `708572b`). The full eager
  protocol description is retained in [ofi-plugin-protocols.md](ofi-plugin-protocols.md)
  along with the ordering fixes that landed alongside it, but the default and the
  re-enable tradeoffs are now stated correctly.
- **EFA kernel driver r3.3.0** documented in [kernel-efa-driver.md](kernel-efa-driver.md),
  [efa-driver.md](efa-driver.md) and [efa-hardware-architecture.md](efa-hardware-architecture.md):
  Completion Counters (and why they matter for GPU-initiated paths), 64-bit work request
  IDs, 128-byte SQ WQEs, inline WRITE, the new `0xefa4` PCI device ID, 800/1600 Gbps link
  speed reporting plus the query-port-speed verb, admin response checksums, >4 GB MR page
  size, the rhashtable AH cache, and the 128-byte admin v2 SQ entry.
- **libfabric 2.7 additions**: the new **XPU API** for device-initiated communication
  (`FI_XPU`, `fi_xpu.h`, `fi_xpu_device.h`, `fid_xpu_ctx`, `export_xpu` on EP/CQ/CNTR)
  documented in [libfabric-overview.md](libfabric-overview.md); the **PEER_ERROR
  receiver-decides model** for MR abort in [efa-provider.md](efa-provider.md);
  `fi_av_lookup2` default provider support; the `fi_domain_attr` ABI bump.
- **EFA provider locking rework** documented in [threading-model.md](threading-model.md)
  and [efa-provider.md](efa-provider.md): per-endpoint lock-free `fi_addr`-indexed peer
  map, lock-free `efa_av_array`, srx lock moved from domain to endpoint and set to
  `OFI_LOCK_NONE` under `FI_THREAD_COMPLETION`, `util_domain` lock made a no-op under
  `FI_PROGRESS_CONTROL_UNIFIED`, per-endpoint progress lists with a `progress_ep_list`
  that skips idle endpoints, and Clang thread-safety annotations
  (`efa_thread_annotations.h`, `nccl_ofi_tsa.h`).
- **Plugin internals**: the rdma transport's request objects became a **class hierarchy
  constructed with placement new** out of the freelist, replacing the base-struct-plus-union
  layout, with `rdma_req_max_subclass_size` driving freelist entry sizing
  ([ofi-plugin.md](ofi-plugin.md), [freelist-allocator.md](freelist-allocator.md)).
  C++20 is now mandatory. New threading machinery documented: `NCCL_GIN_PROXY_NTHREADS`
  per-thread endpoint bucketing, one gdrcopy worker per process, and the MPSC/SPSC rings
  and spinlock that feed it.
- **Removed-API corrections.** `nccl_ofi_freelist` and `nccl_ofi_mr_cache` are C++ classes
  upstream; the `nccl_ofi_freelist_*()` and `nccl_ofi_mr_cache_*()` free functions no longer
  exist. [freelist-allocator.md](freelist-allocator.md) and
  [mr-cache-implementation.md](mr-cache-implementation.md) had been presenting the removed C
  API as the current interface; they now document the real class API with real call sites,
  and keep the old API in a labelled *Former C API (removed)* section with an old→new
  mapping table.
- **NCCL 2.31 corrections.** Tuning moved out of `src/graph/tuning.cc` into a new
  `src/tuning/` cost-model directory; `enqueue.cc` moved to `src/enqueue/enqueue.cc` and
  `env.cc` to `src/plugin/env.cc`. In [nccl-channels.md](nccl-channels.md) two
  **non-existent environment variables** (`NCCL_TREE_MAX_NCHANNELS`, `NCCL_NET_BINDINGS`)
  were removed, the Simple per-channel buffer default was corrected from 1 MB to the actual
  4 MiB `DEFAULT_BUFFSIZE`, and `NCCL_NCHANNELS_PER_NET_PEER`, the `MAXCHANNELS = 64`
  ceiling and the new `NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE` were added. In
  [algorithms/tree-algorithm.md](algorithms/tree-algorithm.md) a self-contradicting N=8
  double-binary-tree example was replaced with the exact `ncclGetDtree()` output and the
  real mirror/shift derivation. All α-β derivations and DRAFT warnings preserved.
- **Tuner** ([nccl-tuner.md](nccl-tuner.md)): `ncclTunerPlugin_v6` and its `getChunkSize`
  callback, chunk-size tuning for Tree/LL128 and AllGather PAT/Simple on P5en, and PAT
  channel optimization on P6-B300. Instance tables extended to p5e/p5en/p6-b200/p6-b300.
- **Source links normalized.** All links are now branch-form (`blob/master`, `blob/main`)
  with **no commit-pinned SHAs and no `#L` anchors** — line numbers live in the link text,
  which does not rot silently. 46 pinned base URLs were converted, 66 anchors stripped, and
  **16 genuinely broken source paths** fixed (upstream file moves). Invariants are now
  machine-checked; the authoritative counts live in
  [PERMALINK_STATUS.md](PERMALINK_STATUS.md) (every source link resolves to a file that
  exists upstream, and every internal cross-reference, heading anchor and symbol mapping
  resolves).
- **Index fixes**: [efa-driver.md](efa-driver.md) was missing from the documentation index;
  the metadata files are now listed too.
- **Fabricated identifiers purged (independent audit round).** An exhaustive sweep of every
  `OFI_NCCL_*` / `FI_EFA_*` / `NCCL_*` / `FI_*` identifier in the doc set against the source
  trees found **19 environment variables that do not exist anywhere upstream** but were
  presented as usable knobs. Worst offenders for an agent to act on: `OFI_EFA_ROUND_ROBIN_HASH`,
  `OFI_NCCL_TUNER_ENABLE`, `NCCL_OFI_USE_GDRCOPY`, `NCCL_OFI_NCCL_VERSION_CHECK`,
  `FI_EFA_RDM_LONG_MSG_SIZE`, `NCCL_PROXY_CPU_AFFINITY`, `NCCL_PROXY_SCHED_PRIORITY`,
  `NCCL_NPROXY_THREADS`, `NCCL_PROXY_THREADS`, `NCCL_CPU_AFFINITY`, `NCCL_LL128_THRESHOLD`,
  `NCCL_PROTO_LL128_MAX_SIZE`, `NCCL_IB_MR_CACHE`, `NCCL_REGISTRATION_CACHE_SIZE`,
  `NCCL_OFI_MR_CACHE_SIZE`, `NCCL_TUNER_CONFIG_FILE`, `NCCL_ASYNC_ERROR_HANDLING`. Each is now
  either replaced with the real mechanism (e.g. `FI_EFA_INTER_MIN_READ_MESSAGE_SIZE` /
  `FI_EFA_INTER_MAX_MEDIUM_MESSAGE_SIZE` for protocol thresholds, `OFI_NCCL_MR_CACHE_DISABLE`
  + `FI_EFA_MR_MAX_CACHED_SIZE/COUNT` for MR caching, `NCCL_TUNER_PLUGIN` for tuner selection,
  launcher-level binding plus `NCCL_IGNORE_CPU_AFFINITY` for affinity, `NCCL_PROTO` for
  protocol control) or kept with an explicit "there is no such variable" note so an agent that
  meets the name elsewhere recognizes it as fake. Also corrected: `OFI_NCCL_GIN_PROXY_NTHREADS`
  → **`NCCL_GIN_PROXY_NTHREADS`** (plain `NCCL_` prefix; it is a real NCCL param in
  `src/gin/gin_host.cc` that the plugin reads via `getenv`), `NCCL_COMM_SPLIT_SHARE` →
  `NCCL_COMM_SPLIT_SHARE_RESOURCES`, and `FI_EFA_TX_IOV_LIMIT` / `FI_EFA_RX_IOV_LIMIT` are now
  flagged as **fatal if set** (the provider aborts at init).
- **NCCL EP API example rewritten** in
  [nccl-ep-vs-deepep-comparison.md](nccl-ep-vs-deepep-comparison.md): the old three-call
  `ncclEpTensorCreate()` / `NCCL_EP_TENSOR_TAG_TOPK_IDX` / `ncclNDTensor_t` shape does not
  exist in NCCL 2.31's `contrib/nccl_ep`; the real API passes `ncclEpTensor_t*` fields inside
  named input/output structs.
- **Structural repairs.** Three code fences left broken by the parallel refresh were silently
  swallowing whole sections into code blocks (a missing closer plus a stray fence in
  `cuda-memory.md`, an orphaned function tail in `ofi-plugin.md`, and a prose line trapped
  inside a fence in `dmabuf-gpu-memory.md`). Dead `file:///home/...` local links in
  `nccl-buffsize-explained.md` and `nccl-message-breakdown-complete.md` were replaced with
  real source links. Stale `nccl_ofi_mr_cache_*()` free-function calls remaining in
  `ofi-plugin.md` and `dmabuf-gpu-memory.md` code samples were converted to the C++ method
  form. Unattributed performance figures (the ~455 μs GIN all-to-all iteration time, the
  tuner's "measured benefits" percentages, the ~35 μs RTT) are now labelled as estimates.

### 2026-06-29

#### Stack refresh (versions, eager, GDAKI, low-level internals)
- **Version baseline updated** to June 2026: NCCL 2.30.4-1, aws-ofi-nccl v1.20.0 /
  master `f3dd9cd`, libfabric 2.6.x, rdma-core v63. Plugin API versions corrected to
  `ncclNet_v12_t` / `ncclGin_v13_t` (older versions still exported for back-compat).
- **Source permalinks unified** to `aws/aws-ofi-nccl/blob/master` (removing several
  stale pinned commits and a personal fork); line anchors dropped so links don't rot.
- **Eager protocol documented** in [ofi-plugin-protocols.md](ofi-plugin-protocols.md):
  the small-message inline path (`EAGER_MAX_SIZE`), the 8-byte eager header, grouped-recv
  batch draining, and the wrap-safe `eager_seq` that fixed the sequence-number-wrap hang
  (shipped in v1.20.0).
- **GIN proxy vs GDAKI modes** documented: `OFI_NCCL_GIN_TYPE` selecting CPU-proxy GIN
  vs kernel-initiated GDAKI (GPU drives EFA queues directly; requires `--enable-gdaki`,
  CUDA, DMA-BUF, libfabric 2.5+).
- **EFA kernel driver internals deepened** in [kernel-efa-driver.md](kernel-efa-driver.md):
  the GPU/accelerator peer-memory (P2P) provider subsystem (NVIDIA nvmem v1/v2, Neuron) and
  its revocation-safe ticket mechanism; the admin-queue producer/consumer **phase-bit**
  protocol and MMIO doorbell submission; and an explicit **firmware/hardware boundary**
  clarifying what is open source vs closed NIC firmware.
- **Architecture stack diagram** (this file and [overview.md](overview.md)) updated to show
  the firmware boundary and the GIN data path.

### 2026-01-20

#### Documentation Organization and Improvements
- **Reorganized rdma-core documentation** - Moved [rdma-core-and-verbs.md](rdma-core-and-verbs.md) from Deep Dives to Transport Layer section in README, better reflecting its role as the critical layer between libfabric and kernel driver
- **Added rdma-core to stack diagrams** - Updated [overview.md](overview.md) Architecture Stack to include rdma-core (libibverbs) layer with userspace verbs library, EFA provider plugin, and zero-copy queue operations
- **Enhanced data path documentation** - Updated [nccl-datapath.md](nccl-datapath.md) to include rdma-core layer showing ibv_post_send/recv operations (~100-200ns), memory-mapped queue writes, and doorbell rings with zero syscalls
- **Component roles expansion** - Added rdma-core component description in overview.md explaining its purpose as userspace RDMA verbs library with hardware-agnostic API and provider plugins
- **Data flow clarification** - Expanded data flow steps to explicitly show rdma-core posting work requests via verbs API and EFA kernel driver managing memory registration via uverbs interface
- **Added DRAFT warnings** - Added prominent verification warnings to all algorithm documentation files (Ring, Tree, NVLS Tree, PAT) indicating theoretical derivations need independent validation
- **Simplified algorithm descriptions** - Changed algorithm descriptions in README to simple one-sentence summaries without formulas or performance claims

### 2026-01-19

#### Algorithm Documentation
- Added **[algorithms/ring-algorithm.md](algorithms/ring-algorithm.md)** with complete mathematical latency derivation (`T = 2(N-1)×α + 2β×S×(N-1)/N`), bandwidth analysis proving ~100% utilization, O(N) scaling analysis, protocol-specific equations (Simple/LL128/LL), and concrete p5 instance examples
- Added **[algorithms/tree-algorithm.md](algorithms/tree-algorithm.md)** with double binary tree algorithm, O(log N) latency derivation (`T = 2×log₂(N)×α + log₂(N)×β×S`), ~67% bandwidth utilization analysis for N=8, tree topology construction, and comparison with Ring showing log vs linear scaling
- Added **[algorithms/nvls-tree-algorithm.md](algorithms/nvls-tree-algorithm.md)** documenting NVLink SHARP hardware acceleration, NVSwitch multicast mechanisms, in-network reduction, intra-node latency (`T = 2×α_nvls + 2×β_nvlink×S`), multi-node extensions, 2-10× speedup for small messages, and p5/p5en specific performance on Hopper/Blackwell GPUs
- Added **[algorithms/pat-algorithm.md](algorithms/pat-algorithm.md)** for Parallel Aggregated Trees (NCCL 2.23+), adaptive logarithmic-to-linear communication strategy, AllGather derivation (`T = log₂(N)×α + β×S×(1-1/N)`), achieving O(log N) latency with ~100% bandwidth, optimal for 32 KB-1 MB messages at scale, and LLM training use cases
- All algorithm files include complete α-β model derivations, step-by-step walkthroughs, protocol analysis, crossover point calculations, and AWS instance benchmarks

#### Hardware and Protocol Documentation
- Added comprehensive **[efa-hardware-architecture.md](efa-hardware-architecture.md)** documenting EFA queue pairs, completion queues, work queue entries, memory layout, and programming model
- Added **[srd-protocol.md](srd-protocol.md)** documenting AWS's Scalable Reliable Datagram protocol with multipath routing (64 paths), hardware-based congestion control, and reliability mechanisms
- Added **[nccl-tuner.md](nccl-tuner.md)** documenting NCCL's cost-based tuner, algorithm/protocol selection, channel count tuning, and AWS OFI region-based tuner with geometric selection

#### Improvements
- Updated all numerical performance claims to be properly qualified as estimates or approximations
- Removed file numbering from all documentation files to ease maintenance
- Removed Troubleshooting section from README
- Added algorithm documentation section to Quick Start guide

### 2026-01-17
- Initial comprehensive documentation
- Covered full stack: NCCL → EFA hardware
- Included optimization guide

---

**Audience**: Engineers optimizing distributed deep learning workloads on AWS EC2 with EFA
**Focus**: Technical depth, performance optimization, debugging
**Scope**: NCCL + OFI plugin + libfabric + EFA driver + EFA hardware
