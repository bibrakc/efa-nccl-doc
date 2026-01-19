# GitHub Permalink Addition Status

## Overview
Task: Add GitHub permalink references to all struct and function mentions in NCCL/EFA documentation (00-*.md through 22-*.md)

## Completed Documents

### 04-ofi-plugin.md ✅
**Status**: 42 permalinks added
**Key structs/functions linked**:
- `struct nccl_net_ofi_listen_comm`, `struct nccl_net_ofi_send_comm`, `struct nccl_net_ofi_recv_comm`
- `struct nccl_net_ofi_req`
- `struct fi_info`, `struct fid_mr`, `struct fi_cq_data_entry`, `struct fi_cq_err_entry`
- `fi_getinfo()`, `fi_endpoint()`, `fi_ep_bind()`, `fi_enable()`, `fi_getname()`
- `fi_tsend()`, `fi_trecv()`, `fi_cq_read()`, `fi_cq_readerr()`
- `fi_mr_reg()`, `fi_mr_desc()`, `fi_av_insert()`, `fi_close()`

### 05-ofi-plugin-protocols.md ✅
**Status**: 16 permalinks added
**Key structs/functions linked**:
- `struct nccl_ofi_tag`
- Same libfabric functions as 04-ofi-plugin.md

### 07-freelist-allocator.md ✅
**Status**: 8 permalinks added
**Structs linked (3)**:
- `struct nccl_ofi_freelist_elem` → [include/nccl_ofi_freelist.h:19-23](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/include/nccl_ofi_freelist.h#L19-L23)
- `struct nccl_ofi_freelist_t` → [include/nccl_ofi_freelist.h:88-109](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/include/nccl_ofi_freelist.h#L88-L109)
- `struct nccl_ofi_freelist_block_t` → [include/nccl_ofi_freelist.h:28-35](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/include/nccl_ofi_freelist.h#L28-L35)

**Functions linked (5)**:
- `nccl_ofi_freelist_init()` → [src/nccl_ofi_freelist.cpp:131](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_freelist.cpp#L131)
- `nccl_ofi_freelist_init_mr()` → [src/nccl_ofi_freelist.cpp:188](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_freelist.cpp#L188)
- `nccl_ofi_freelist_entry_alloc()` → [src/nccl_ofi_freelist.cpp:270](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_freelist.cpp#L270)
- `nccl_ofi_freelist_entry_free()` → [src/nccl_ofi_freelist.cpp:318](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_freelist.cpp#L318)
- `fi_mr_reg()` → [include/rdma/fi_domain.h:413](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L413)

### 08-mr-cache-implementation.md ✅
**Status**: 6 permalinks added
**Structs linked (2)**:
- `struct nccl_ofi_reg_entry` → [include/nccl_ofi_mr.h:186-192](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/include/nccl_ofi_mr.h#L186-L192)
- `struct nccl_ofi_mr_cache` → [include/nccl_ofi_mr.h:197-205](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/include/nccl_ofi_mr.h#L197-L205)

**Functions linked (4)**:
- `nccl_ofi_mr_cache_init()` → [src/nccl_ofi_mr.cpp:13](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_mr.cpp#L13)
- `nccl_ofi_mr_cache_lookup_entry()` → [src/nccl_ofi_mr.cpp:112](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_mr.cpp#L112)
- `nccl_ofi_mr_cache_insert_entry()` → [src/nccl_ofi_mr.cpp:153](https://github.com/sirmick/aws-ofi-nccl/blob/75240c8/src/nccl_ofi_mr.cpp#L153)
- `fi_mr_reg()` → [include/rdma/fi_domain.h:413](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L413)

### 09-libfabric-overview.md ✅
**Status**: 16 permalinks added in this session
**Key structs/functions linked**:
- `struct fi_info` → [fabric.h:198-232](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fabric.h#L198-L232)
- `struct fid_domain` → [fi_domain.h:94-104](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L94-L104)
- `struct fid_ep` → [fabric.h:168-178](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fabric.h#L168-L178)
- `struct fid_cq` → [fi_eq.h:138-148](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L138-L148)
- `struct fid_av` → [fi_cm.h:167-177](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_cm.h#L167-L177)
- `struct fid_mr` → [fi_domain.h:131-138](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_domain.h#L131-L138)
- `struct fi_cq_data_entry` → [fi_eq.h:215-220](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L215-L220)
- `struct fi_cq_err_entry` → [fi_eq.h:233-246](https://github.com/ofiwg/libfabric/blob/6b9e629/include/rdma/fi_eq.h#L233-L246)
- `fi_endpoint()`, `fi_ep_bind()`, `fi_enable()`, `fi_cq_open()`, `fi_av_insert()`
- `fi_send()`, `fi_inject()`, `fi_recv()`, `fi_tsend()`, `fi_trecv()`
- `fi_read()`, `fi_write()`, `fi_mr_reg()`, `fi_mr_desc()`, `fi_cq_read()`, `fi_cq_readerr()`

## Completed Documents (No Linkable Content)

### 00-overview.md ✅
**Status**: No structs/functions from our repos to link (high-level overview)

### 01-nccl-core.md ✅
**Status**: NCCL internal APIs only (not in our repos)

### 02-nccl-collectives.md ✅
**Status**: NCCL internal algorithms (not in our repos)

### 03-nccl-datapath.md ✅
**Status**: Conceptual overview, references libfabric functions already linked elsewhere

### 06-topology-and-binding.md ✅
**Status**: Topology detection code examples, references libfabric functions already linked

## Completed High-Value Documents (Session 2)

### 11-rdma-core-and-verbs.md ✅
**Status**: 15 permalinks added
**Key structs/functions linked**:
- `struct ibv_context` → [libibverbs/verbs.h:2069-2077](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L2069-L2077)
- `struct ibv_pd` → [libibverbs/verbs.h:639-642](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L639-L642)
- `struct ibv_mr` → [libibverbs/verbs.h:675-683](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L675-L683)
- `struct ibv_qp` → [libibverbs/verbs.h:1315-1325](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L1315-L1325)
- `struct ibv_cq` → [libibverbs/verbs.h:1540-1550](https://github.com/linux-rdma/rdma-core/blob/6e9643e/libibverbs/verbs.h#L1540-L1550)
- `efa_alloc_context()`, `efa_reg_mr()`, `efa_create_qp()`, `efa_poll_cq()`
- `ibv_get_device_list()`, `ibv_open_device()`, `ibv_alloc_pd()`, `ibv_reg_mr()`, `ibv_dereg_mr()`
- `ibv_create_cq()`, `ibv_create_qp()`, `ibv_post_send()`, `ibv_post_recv()`, `ibv_poll_cq()`

### 13-kernel-efa-driver.md ✅
**Status**: 8 permalinks added
**Key structs/functions linked**:
- `struct efa_dev` → [kernel/linux/efa/src/efa.h:53-79](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa.h#L53-L79)
- `struct efa_qp` → [kernel/linux/efa/src/efa.h:240-263](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa.h#L240-L263)
- `struct efa_cq` → [kernel/linux/efa/src/efa.h:171-188](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa.h#L171-L188)
- `struct efa_mr` → [kernel/linux/efa/src/efa.h:149-157](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa.h#L149-L157)
- `efa_reg_mr()` → [kernel/linux/efa/src/efa_verbs.c:2799](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.c#L2799)
- `efa_dereg_mr()` → [kernel/linux/efa/src/efa_verbs.c:3065](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.c#L3065)
- `efa_create_qp()` → [kernel/linux/efa/src/efa_verbs.c:1216](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.c#L1216)
- `efa_create_cq()` → [kernel/linux/efa/src/efa_verbs.c:2147](https://github.com/amzn/amzn-drivers/blob/8a8b6f2/kernel/linux/efa/src/efa_verbs.c#L2147)

### 10-efa-provider.md ✅
**Status**: 4 permalinks added
**Key structs/functions linked**:
- `struct efa_rdm_ep` → [prov/efa/src/rdm/efa_rdm_ep.h:46-120](https://github.com/ofiwg/libfabric/blob/6b9e629/prov/efa/src/rdm/efa_rdm_ep.h#L46-L120)
- `struct efa_rdm_pke` → [prov/efa/src/rdm/efa_rdm_pke.h:78-100](https://github.com/ofiwg/libfabric/blob/6b9e629/prov/efa/src/rdm/efa_rdm_pke.h#L78-L100)
- `struct efa_mr` → [prov/efa/src/efa_mr.h:20-32](https://github.com/ofiwg/libfabric/blob/6b9e629/prov/efa/src/efa_mr.h#L20-L32)

## Remaining Documents

### Documents reviewed - no new linkable content:
- 20-rdma-memreg.md - References functions already linked in other documents ✅
- 12-lkey-rkey-explained.md - References structs already linked ✅
- 14-efa-driver.md - High-level driver overview, conceptual ✅
- 19-threading-model.md - Threading concepts, no source code references ✅
- 22-optimizations.md - Optimization techniques, conceptual ✅

### Documents with limited/no linkable content:
- 15-cuda-memory.md - CUDA APIs (not in our repos)
- 16-dmabuf-gpu-memory.md - Kernel dmabuf APIs (external)
- 17-neuron-memory.md - AWS Neuron APIs (not in our repos)
- 18-rocm-memory.md - AMD ROCm APIs (not in our repos)
- 21-optimization-opportunities.md - Conceptual recommendations

## Summary Statistics

### Completed
- **Documents processed**: 18 / 23
  - 4 documents from session 1 (04, 05, 07, 08)
  - 6 documents from session 1 continuation (00, 01, 02, 03, 06, 09)
  - 3 documents from session 2 (10, 11, 13)
  - 5 documents reviewed in session 2 (12, 14, 19, 20, 22)

### New Permalinks Added in Session 2
- **11-rdma-core-and-verbs.md**: 15 permalinks
- **13-kernel-efa-driver.md**: 8 permalinks
- **10-efa-provider.md**: 4 permalinks
- **Session 2 total**: 27 permalinks

### Total Permalinks Added
- **04-ofi-plugin.md**: 42 permalinks
- **05-ofi-plugin-protocols.md**: 16 permalinks
- **07-freelist-allocator.md**: 8 permalinks
- **08-mr-cache-implementation.md**: 6 permalinks
- **09-libfabric-overview.md**: 16 permalinks
- **10-efa-provider.md**: 4 permalinks
- **11-rdma-core-and-verbs.md**: 15 permalinks
- **13-kernel-efa-driver.md**: 8 permalinks
- **Grand Total**: 115 permalinks

### Unique Structs/Functions Linked
- aws-ofi-nccl: ~15 unique structs/functions
- libfabric: ~30 unique structs/functions
- rdma-core: ~15 unique structs/functions
- amzn-drivers: ~8 unique structs/functions
- Total: ~68 unique items

### Repositories Covered
✅ aws-ofi-nccl (commit: 75240c8)
✅ libfabric (commit: 6b9e629)
✅ rdma-core (commit: 6e9643e)
✅ amzn-drivers (commit: 8a8b6f2)
⏳ linux (available if needed, commit: e84d960)

## Key Accomplishments

### Session 1
1. **Core libfabric API documented** in 09-libfabric-overview.md:
   - All major data structures (fi_info, fid_domain, fid_ep, fid_cq, fid_av, fid_mr)
   - All major operations (send, recv, tagged, RMA, completion)
   - Error handling structures

2. **OFI Plugin architecture documented** in 04-ofi-plugin.md and 05-ofi-plugin-protocols.md:
   - Connection management structures
   - Request handling
   - Protocol implementations

3. **Memory management documented** in 07-freelist-allocator.md and 08-mr-cache-implementation.md:
   - Freelist allocator structures
   - MR cache implementation

### Session 2
4. **rdma-core/libibverbs API documented** in 11-rdma-core-and-verbs.md:
   - Core data structures (ibv_context, ibv_pd, ibv_mr, ibv_qp, ibv_cq)
   - All major verbs operations (device, PD, MR, QP, CQ management)
   - EFA provider-specific implementations

5. **EFA kernel driver documented** in 13-kernel-efa-driver.md:
   - Kernel data structures (efa_dev, efa_qp, efa_cq, efa_mr)
   - Memory registration functions (efa_reg_mr, efa_dereg_mr)
   - Queue and completion management

6. **EFA provider internals documented** in 10-efa-provider.md:
   - RDM endpoint structures (efa_rdm_ep, efa_rdm_pke)
   - Memory region wrappers (efa_mr)

## Notes

- Documents 00-03, 06 reviewed and confirmed to have no linkable content (NCCL internals or conceptual overviews)
- Focus on libfabric and aws-ofi-nccl structs/functions has been completed for core documents
- Remaining work primarily involves rdma-core (ibv_*) and kernel driver (efa_*) structures
- Many documents reference functions already linked in other documents (following "first mention only" rule)

## Permalink Format Used

Structs: `struct foo` ([path/file.h:10-20](https://github.com/org/repo/blob/commit/path/file.h#L10-L20))
Functions: `function_name()` ([path/file.c:123](https://github.com/org/repo/blob/commit/path/file.c#L123))

## Repository Commit SHAs

- aws-ofi-nccl: 75240c8
- libfabric: 6b9e629
- amzn-drivers: 8a8b6f2
- rdma-core: 6e9643e
- linux: e84d960
