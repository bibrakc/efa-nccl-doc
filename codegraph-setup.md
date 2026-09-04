# Setting up the code graph

This corpus is one of **two** sources of truth for the stack. The other is a code graph: an
AST-derived index that answers *where is it*, *what calls it* and *what breaks if I change it*
across all five layers. This document is the procedure for getting that graph.

It is deliberately a **procedure, not a script**. Nothing executable ships with this skill.
Read the steps, decide what is appropriate for the machine you are on, and run the commands
yourself — or write your own small wrapper locally if you want idempotence. The reason is in
[Why there is no script](#why-there-is-no-script) below; read that before installing anything.

## What you are building, and why it is two indexes

| Index | Root | Answers | Do **not** use it for |
| --- | --- | --- | --- |
| **Combined** | a directory of symlinks to all five repos | cross-repo symbol *lookup* — "where is `efadv_query_qp_wqs` defined?" | cross-repo **edges** |
| **Plugin-only** | the `aws-ofi-nccl` clone | accurate callers / callees / blast radius *within the plugin* | anything outside the plugin |

Both are needed, and the split is not arbitrary. Measured on this stack: the combined index
holds ~212,600 intra-repo edges but its ~14,200 cross-repo edges are **all name-collision
artefacts**. Real cross-layer calls produce **zero** edges — `fi_send`, `fi_cq_read`,
`fi_mr_regattr` and `ibv_reg_mr` each resolve to nothing across the repo boundary, and ~141,600
references are unresolved. The cause is structural: the indexer has no preprocessor and no
linker, so it cannot follow a call through a header into another library.

**So: trust call graphs within one repo. Never across repos.** For cross-layer flow, use this
corpus — that is precisely the gap it fills.

## Step 1 — decide whether you should install anything

The tool is [`codegraph`](https://github.com/colbymchenry/codegraph), a third-party CLI. Before
installing it on a machine holding internal source, satisfy yourself on three points:

1. **Prefer an internally vended copy if one exists.** Check your organisation's approved
   tooling channel first. An internally distributed build removes both the supply-chain and the
   telemetry question below, and should be preferred whenever it is available.
2. **Pin what you install.** Install a specific tagged release, not a moving branch, and verify
   the artefact against a checksum published for that release. An unpinned install means a
   future run can fetch different code than the one you reviewed.
3. **Know the footprint.** It unpacks a self-contained bundle of roughly 280 MB under
   `~/.codegraph`. No Node runtime is required.

If any of that is unacceptable for your environment, **stop here**. The graph is an
accelerator, not a dependency: `grep` and `Read` answer the same questions more slowly, and this
corpus does not require the graph to be useful. Say you are working without it rather than
guessing at answers it would have given you.

## Step 2 — install, then turn telemetry off

Follow the upstream project's own installation instructions for the release you pinned.

The installer places the binary in `~/.local/bin`, **which is frequently absent from `PATH` in a
non-login shell** — this is the single most common reason the MCP server appears broken:

```sh
export PATH="$HOME/.local/bin:$PATH"
command -v codegraph && codegraph --version
```

Then, before indexing anything:

```sh
codegraph telemetry off
```

**Telemetry is enabled by default upstream.** This is internal source. Turn it off *before* the
first index, not after.

## Step 3 — make sure the five layers are cloned

```sh
: "${SW_STACK:=$HOME/sw-stack}"
: "${OFI_NCCL:=$HOME/ofi-nccl-work}"

git clone https://github.com/aws/aws-ofi-nccl.git         "$OFI_NCCL"
git clone https://github.com/ofiwg/libfabric.git          "$SW_STACK/libfabric"
git clone https://github.com/linux-rdma/rdma-core.git     "$SW_STACK/rdma-core"
git clone https://github.com/NVIDIA/nccl.git              "$SW_STACK/nccl"
git clone https://github.com/amzn/amzn-drivers.git        "$SW_STACK/amzn-drivers"
```

Note the asymmetry that catches people out: **the plugin clone is not under `sw-stack`.** It
lives at `$HOME/ofi-nccl-work` by convention. See [stack-map.md](stack-map.md) for which repo
owns which layer.

A partial set is fine. Index what you have and note the gap; four of five layers still answers
most questions.

## Step 4 — build the combined index

The indexer takes a single root but follows symlinks, so a directory of links yields one graph
spanning every repo. Without this you get five disconnected indexes and no cross-repo lookup at
all.

```sh
IDX="${CODEGRAPH_STACK_INDEX:-$HOME/.codegraph-stack-index}"
mkdir -p "$IDX"
ln -sfn "$OFI_NCCL"                "$IDX/aws-ofi-nccl"
ln -sfn "$SW_STACK/libfabric"      "$IDX/libfabric"
ln -sfn "$SW_STACK/rdma-core"      "$IDX/rdma-core"
ln -sfn "$SW_STACK/nccl"           "$IDX/nccl"
ln -sfn "$SW_STACK/amzn-drivers"   "$IDX/amzn-drivers"

cd "$IDX"
codegraph init -y      # first time
codegraph sync         # subsequent runs: picks up new commits, much cheaper
codegraph status       # confirm node and file counts
```

Expect roughly **3,600 files / 88,500 nodes / 227,000 edges in ~9 s** for the full five layers
on a dev desktop; a `sync` with nothing changed is well under a second. Numbers far below that
mean a layer failed to index — check the symlinks resolve.

## Step 5 — build the plugin-only index

```sh
cd "$OFI_NCCL"
codegraph init -y     # or: codegraph sync

# keep the index out of git if the repo does not already ignore it
git check-ignore -q .codegraph/ || echo ".codegraph/" >> .git/info/exclude
```

That last line matters: without it the index shows up as untracked files in every
`git status` on the plugin, which is how it ends up in someone's commit.

## Step 6 — confirm it answers

```sh
cd "$IDX"      && codegraph explore nccl_net_ofi_rdma_ep_t   # cross-repo lookup
cd "$OFI_NCCL" && codegraph impact  nccl_net_ofi_listen      # accurate blast radius
```

If the MCP server is configured, `@codegraph` should now respond. If it does not, check `PATH`
first — see Step 2.

## What the graph cannot tell you

Keep these in this corpus, because the graph provably cannot hold them:

- **Environment variables.** All 47 params declared through the `OFI_NCCL_PARAM` macro are
  invisible to the index — the names are constructed by the preprocessor, so
  `codegraph query eager_max_size` returns nothing. Env vars are the corpus's single largest
  category of error; see [ofi-plugin.md](ofi-plugin.md).
- **Values that are only initialisers.** `.use_data_path_direct = true` at
  `prov/efa/src/efa_env.c:41` is a field initialiser, not a symbol.
- **Absence.** "EFA does not use `nvidia-peermem`" is evidenced by *zero* references, and a
  graph cannot represent zero. Removals and traps live here.
- **Cross-layer flow**, for the reason measured at the top of this document.
- **GPU device code.** `.cu` / `.cuh` are unsupported, so GDAKI device code is not indexed.

## Why there is no script

An earlier version of this skill shipped `scripts/ensure-codegraph.sh`, which fetched a
third-party installer from a moving branch and ran it automatically on every agent spawn. Code
review flagged it, correctly: an unpinned installer executed automatically on a machine holding
internal source is a supply-chain exposure, and "we download it before running it" does not
help when the download is re-fetched each time.

Replacing the script with this procedure changes the shape of the problem in three ways:

- **Nothing executes automatically.** Setup happens when someone — a human, or an agent that
  has read Step 1 — decides to do it.
- **The trust decision is explicit.** The caveats sit in Step 1, ahead of any install command,
  instead of buried in a shell comment.
- **Nothing executable ships in the package**, so there is no vendored artefact to keep in sync
  with upstream, and no third-party URL in an executable position.

What it does **not** do is make the third-party dependency itself disappear. If you install the
upstream build, you have taken on an unpinned third-party binary; Step 1 tells you how to
reduce that, and the honest answer is to prefer an internal copy. The graph is optional
throughout, which is what makes declining a real option.
