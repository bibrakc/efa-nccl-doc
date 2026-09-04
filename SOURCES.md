# Source-Link Status

Tracks the GitHub source links that let a reader (human or agent) jump from a claim in
these docs to the code that backs it. Regenerated **2026-09-01**.

## Link Policy

Three rules, applied uniformly across every document:

1. **Branch-form base URLs, never commit-pinned SHAs.**
   `https://github.com/aws/aws-ofi-nccl/blob/master/...`, not `.../blob/d840aa1/...`.
   A commit-pinned link freezes the reader on a snapshot that gets staler every week;
   these docs exist to point at *current* code. The snapshot the docs were written against
   is recorded under [Source Snapshot](#source-snapshot) below, so the exact permalink can
   still be reconstructed when a historical view is genuinely needed.

2. **No `#L` line anchors in URLs.** Line anchors rot silently: after any refactor the
   anchor still resolves, it just points at unrelated code, which is worse than no anchor
   at all. Line numbers go in the **link text** instead:

   ```markdown
   [include/nccl_ofi_mr.h:201-256](https://github.com/aws/aws-ofi-nccl/blob/master/include/nccl_ofi_mr.h)
   ```

3. **Every link must resolve to a file that exists** in the corresponding upstream repo
   at the recorded snapshot. This is machine-checkable; see *Verification* below.

Prior revisions of these docs mixed commit-pinned links (libfabric `6b9e629`,
amzn-drivers `8a8b6f2`, rdma-core `6e9643e`, linux `e84d960`) with branch links, and
used `#L` anchors in roughly a sixth of them. All of that was normalized in the
2026-09-01 refresh: **0 commit-pinned links and 0 `#L` anchors remain.**

## Verification (2026-09-04)

| Check | Result |
| --- | --- |
| Source links whose target file exists upstream | **525 / 525** |
| Commit-pinned `blob/<sha>` links remaining | **0** |
| `#L` line anchors remaining | **0** |
| Internal `.md` cross-references that resolve | **320 / 320** |
| Intra-doc heading anchors that resolve | **49 / 49** |

Reconciling the link counts, because two different subsets both happen to number 525:

| Set | Count |
| --- | --- |
| All `blob/<ref>/<path>` links in the corpus | **527** |
| — of those, in the 39 documents other than this one (the inventory table below sums to this) | 525 |
| — of those, inside this file (one is the `<repo>/<path>` placeholder in *Link Policy*) | 2 |
| Checkable against a local clone, and verified above | **525** |
| — not checkable: the `<repo>/<path>` placeholder in this file | 1 |
| — not checkable: one link into `nccl-tests`, which is not cloned here | 1 |

The inventory total and the checkable total are both 525 but are *not* the same 525: the
inventory includes the `nccl-tests` link and excludes this file's two, while the checkable set
excludes the `nccl-tests` link and the placeholder.

Notable broken paths corrected during this refresh (upstream moved the files). These are the
cases worth remembering, not the complete set:

| Old path in docs | Current path | Repo |
| --- | --- | --- |
| `src/include/nccl.h` | `src/nccl.h.in` | nccl |
| `src/enqueue.cc` | `src/enqueue/enqueue.cc` | nccl |
| `src/env.cc` | `src/plugin/env.cc` | nccl |
| `include/rdma/fi_msg.h` | `include/rdma/fi_endpoint.h` (msg ops are inline there) | libfabric |

## Source Snapshot

| Repo | Commit | Version | Branch |
| --- | --- | --- | --- |
| aws-ofi-nccl | `d840aa1` | v1.21.1 | master |
| libfabric | `cb6364e05` | 2.7.0rc1 | main |
| nccl | `fd168324` | v2.31.2-1 | master |
| rdma-core | `8b3a3e64d` | 65.0 dev (release v64.0) | master |
| amzn-drivers | `44c91d1` | EFA kernel driver r3.3.0 | master |
| open-gpu-kernel-modules | `e4a5faa2` | 610.57.04 | main |

## Per-Document Link Inventory

Counts are source links into upstream repos (not internal cross-references).

| Document | Links | Repos referenced |
| --- | --- | --- |
| [libfabric-overview.md](libfabric-overview.md) | 72 | libfabric 72 |
| [ofi-plugin.md](ofi-plugin.md) | 64 | aws-ofi-nccl 33, libfabric 31 |
| [efa-hardware-architecture.md](efa-hardware-architecture.md) | 43 | amzn-drivers 41, rdma-core 2 |
| [nccl-datapath.md](nccl-datapath.md) | 40 | libfabric 15, nccl 10, rdma-core 8, aws-ofi-nccl 7 |
| [nccl-channels.md](nccl-channels.md) | 33 | nccl 28, aws-ofi-nccl 5 |
| [freelist-allocator.md](freelist-allocator.md) | 25 | aws-ofi-nccl 24, libfabric 1 |
| [nccl-tuner.md](nccl-tuner.md) | 24 | aws-ofi-nccl 16, nccl 8 |
| [rdma-core-and-verbs.md](rdma-core-and-verbs.md) | 23 | rdma-core 20, libfabric 3 |
| [threading-model.md](threading-model.md) | 20 | libfabric 11, aws-ofi-nccl 9 |
| [srd-protocol.md](srd-protocol.md) | 19 | rdma-core 10, amzn-drivers 9 |
| [gpu-memory-kernel-path.md](gpu-memory-kernel-path.md) | 17 | amzn-drivers 13, libfabric 2, aws-ofi-nccl 1, open-gpu-kernel-modules 1 |
| [stack-map.md](stack-map.md) | 16 | libfabric 5, open-gpu-kernel-modules 4, nccl 3, rdma-core 2, aws-ofi-nccl 1, amzn-drivers 1 |
| [mr-cache-implementation.md](mr-cache-implementation.md) | 12 | aws-ofi-nccl 11, libfabric 1 |
| [nccl-buffsize-explained.md](nccl-buffsize-explained.md) | 12 | nccl 12 |
| [efa-provider.md](efa-provider.md) | 11 | libfabric 11 |
| [kernel-efa-driver.md](kernel-efa-driver.md) | 11 | amzn-drivers 11 |
| [algorithms/tree-algorithm.md](algorithms/tree-algorithm.md) | 10 | nccl 10 |
| [nccl-ep-vs-deepep-comparison.md](nccl-ep-vs-deepep-comparison.md) | 10 | aws-ofi-nccl 10 |
| [optimizations.md](optimizations.md) | 10 | libfabric 6, aws-ofi-nccl 4 |
| [algorithms/ring-algorithm.md](algorithms/ring-algorithm.md) | 8 | nccl 7, nccl-tests 1 |
| [gin-alltoall-libfabric-trace.md](gin-alltoall-libfabric-trace.md) | 8 | aws-ofi-nccl 7, libfabric 1 |
| [ofi-plugin-protocols.md](ofi-plugin-protocols.md) | 8 | aws-ofi-nccl 7, libfabric 1 |
| [nccl-collectives.md](nccl-collectives.md) | 6 | nccl 6 |
| [rdma-memreg.md](rdma-memreg.md) | 6 | amzn-drivers 6 |
| [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md) | 5 | amzn-drivers 2, libfabric 2, aws-ofi-nccl 1 |
| [nccl-core.md](nccl-core.md) | 4 | nccl 4 |
| [topology-and-binding.md](topology-and-binding.md) | 4 | aws-ofi-nccl 4 |
| [accelerator-memory.md](accelerator-memory.md) | 2 | libfabric 2 |
| [overview.md](overview.md) | 2 | libfabric 2 |
| **Total** | **525** | |

Documents with no source links are derivation- or concept-oriented and cite code
inline by path instead: `algorithms/nvls-tree-algorithm.md`, `algorithms/pat-algorithm.md`, `cuda-memory.md`, `lkey-rkey-explained.md`, `nccl-message-breakdown-complete.md`, `nccl-usage.md`, `optimization-opportunities.md`.

## Cross-file consistency

A fact corrected in one document must be corrected in **every** document that states it. The
2026-09-01 refresh missed this: `dmabuf-gpu-memory.md` was corrected to say DMA-BUF is no longer
gated by EFA hardware generation (the device-id check was removed in aws-ofi-nccl `0f285d5`),
but `kernel-efa-driver.md` still claimed it was "gated off on EFA Gen 1–3" in two places. Fixed
2026-09-02, along with two further errors in the same passage: a compile guard named
`HAVE_IB_REG_USER_MR_DMABUF` that does not exist anywhere in amzn-drivers (the real macro is
`HAVE_MR_DMABUF`), and a parenthetical calling `nvidia_p2p` "a.k.a. nvidia-peermem" — they are
different mechanisms and EFA uses only the former.

So: after correcting any fact, grep the whole corpus for the old claim before committing. A
single-file check will not catch this class of error, and parallel editors working on disjoint
file sets will reliably produce it.

## Re-verifying

The invariants above are checkable without network access, against local clones of the
upstream repos. Extract `(repo, path)` from every `https://github.com/<org>/<repo>/blob/<ref>/<path>`
occurrence and assert the path exists in the matching clone; separately assert that no
link contains a 7-or-40-hex `blob/<sha>/` segment or a `#L` fragment. Repo roots used for
the 2026-09-01 run:

```
aws/aws-ofi-nccl      -> /home/bibracha/ofi-nccl-work
ofiwg/libfabric       -> /home/bibracha/sw-stack/libfabric
NVIDIA/nccl           -> /home/bibracha/sw-stack/nccl
amzn/amzn-drivers     -> /home/bibracha/sw-stack/amzn-drivers
linux-rdma/rdma-core  -> /home/bibracha/sw-stack/rdma-core
```

## Symbol Mappings

Struct, class, and function name to source location mappings are cited inline in the
documents themselves, with paths and line numbers, rather than collected in a separate
index file. What follows is a **curated set of notes on the symbols most often looked
up, not a full symbol inventory** — kernel-driver types (`efa_dev`, `efa_qp`, `efa_cq`, `efa_mr`), the
`efa_io_*` I/O descriptor structs, the `ibv_*` verbs structs and the EFA provider's
`efa_rdm_ep` / `efa_rdm_pke` internals are cited inline in
[efa-hardware-architecture.md](efa-hardware-architecture.md),
[kernel-efa-driver.md](kernel-efa-driver.md),
[rdma-core-and-verbs.md](rdma-core-and-verbs.md) and
[efa-provider.md](efa-provider.md) rather than duplicated here. Two entries carry refactor notes worth reading before you trust older
documentation or blog posts:

- **`nccl_ofi_freelist`** is now a C++ class. The free functions
  `nccl_ofi_freelist_init()`, `nccl_ofi_freelist_init_mr()`,
  `nccl_ofi_freelist_entry_alloc()` and `nccl_ofi_freelist_entry_free()` no longer
  exist; they are constructors and member functions.
- **`nccl_ofi_mr_cache`** is now a C++ class. The `nccl_ofi_mr_cache_*()` free functions
  are gone, replaced by `lookup_entry()`, `insert_entry()` and `del_entry()` members.
