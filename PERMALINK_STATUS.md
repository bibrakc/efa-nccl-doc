# Source-Link Status

Tracks the GitHub source links that let a reader (human or agent) jump from a claim in
these docs to the code that backs it. Regenerated **2026-09-01**.

## Link Policy

Three rules, applied uniformly across every document:

1. **Branch-form base URLs, never commit-pinned SHAs.**
   `https://github.com/aws/aws-ofi-nccl/blob/master/...`, not `.../blob/d840aa1/...`.
   A commit-pinned link freezes the reader on a snapshot that gets staler every week;
   these docs exist to point at *current* code. The snapshot the docs were written
   against is recorded once, in [PERMALINK_MAPPINGS.txt](PERMALINK_MAPPINGS.txt), so an
   exact permalink can still be reconstructed when a historical view is genuinely needed.

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

## Verification (2026-09-01)

| Check | Result |
| --- | --- |
| Source links whose target file exists upstream | **565 / 565** |
| Commit-pinned `blob/<sha>` links remaining | **0** |
| `#L` line anchors remaining | **0** |
| Internal `.md` / `.txt` cross-references that resolve | **282 / 282** |
| Intra-doc heading anchors that resolve | **45 / 45** |
| Symbol mappings in `PERMALINK_MAPPINGS.txt` with a live target file | **74 / 74** |

Broken paths corrected during this refresh (upstream moved the files):

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
| [ofi-plugin.md](ofi-plugin.md) | 63 | aws-ofi-nccl 32, libfabric 31 |
| [efa-hardware-architecture.md](efa-hardware-architecture.md) | 44 | amzn-drivers 44 |
| [nccl-datapath.md](nccl-datapath.md) | 40 | libfabric 15, nccl 10, rdma-core 8, aws-ofi-nccl 7 |
| [nccl-channels.md](nccl-channels.md) | 31 | nccl 26, aws-ofi-nccl 5 |
| [freelist-allocator.md](freelist-allocator.md) | 25 | aws-ofi-nccl 24, libfabric 1 |
| [mr-cache-implementation.md](mr-cache-implementation.md) | 24 | aws-ofi-nccl 23, libfabric 1 |
| [nccl-tuner.md](nccl-tuner.md) | 24 | aws-ofi-nccl 16, nccl 8 |
| [ofi-plugin-protocols.md](ofi-plugin-protocols.md) | 23 | aws-ofi-nccl 13, libfabric 10 |
| [rdma-core-and-verbs.md](rdma-core-and-verbs.md) | 23 | rdma-core 20, libfabric 3 |
| [nccl-usage.md](nccl-usage.md) | 22 | nccl 21, msccl 1 |
| [threading-model.md](threading-model.md) | 20 | libfabric 11, aws-ofi-nccl 9 |
| [srd-protocol.md](srd-protocol.md) | 19 | rdma-core 10, amzn-drivers 9 |
| [kernel-efa-driver.md](kernel-efa-driver.md) | 17 | amzn-drivers 17 |
| [cuda-memory.md](cuda-memory.md) | 13 | aws-ofi-nccl 13 |
| [nccl-buffsize-explained.md](nccl-buffsize-explained.md) | 12 | nccl 12 |
| [efa-provider.md](efa-provider.md) | 11 | libfabric 11 |
| [algorithms/tree-algorithm.md](algorithms/tree-algorithm.md) | 10 | nccl 10 |
| [dmabuf-gpu-memory.md](dmabuf-gpu-memory.md) | 10 | libfabric 5, linux 2, amzn-drivers 2, aws-ofi-nccl 1 |
| [nccl-ep-vs-deepep-comparison.md](nccl-ep-vs-deepep-comparison.md) | 10 | aws-ofi-nccl 10 |
| [optimizations.md](optimizations.md) | 10 | libfabric 6, aws-ofi-nccl 4 |
| [algorithms/ring-algorithm.md](algorithms/ring-algorithm.md) | 8 | nccl 7, nccl-tests 1 |
| [gin-alltoall-libfabric-trace.md](gin-alltoall-libfabric-trace.md) | 8 | aws-ofi-nccl 7, libfabric 1 |
| [nccl-collectives.md](nccl-collectives.md) | 6 | nccl 6 |
| [rdma-memreg.md](rdma-memreg.md) | 6 | amzn-drivers 6 |
| [nccl-message-breakdown-complete.md](nccl-message-breakdown-complete.md) | 5 | nccl 5 |
| [topology-and-binding.md](topology-and-binding.md) | 5 | aws-ofi-nccl 5 |
| [nccl-core.md](nccl-core.md) | 4 | nccl 4 |
| [overview.md](overview.md) | 2 | libfabric 2 |
| [rocm-memory.md](rocm-memory.md) | 2 | libfabric 2 |
| [efa-driver.md](efa-driver.md) | 1 | amzn-drivers 1 |
| [neuron-memory.md](neuron-memory.md) | 1 | aws-ofi-nccl 1 |

Documents with no source links are derivation- or concept-oriented and cite code
inline by path instead: `lkey-rkey-explained.md`, `nccl-chunk-breakdown.md`,
`optimization-opportunities.md`, `algorithms/nvls-tree-algorithm.md`,
`algorithms/pat-algorithm.md`.

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

Struct, class, and function name to source location mappings live in
[PERMALINK_MAPPINGS.txt](PERMALINK_MAPPINGS.txt) (74 entries, all verified against the
snapshot above). This is a **curated index of the symbols most often looked up, not a full
symbol inventory** — kernel-driver types (`efa_dev`, `efa_qp`, `efa_cq`, `efa_mr`), the
`efa_io_*` I/O descriptor structs, the `ibv_*` verbs structs and the EFA provider's
`efa_rdm_ep` / `efa_rdm_pke` internals are cited inline, with paths and line numbers, in
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
