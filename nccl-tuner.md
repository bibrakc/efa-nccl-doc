# NCCL Tuner: Algorithm and Protocol Selection

## Overview

NCCL includes a **cost-based tuning system** that selects the optimal algorithm (Ring, Tree, NVLS, etc.), protocol (Simple, LL, LL128), and number of channels for each collective operation based on message size, communicator size, and hardware capabilities.

The tuner can be **extended via plugins** to override NCCL's default selections. AWS provides a **region-based tuner** in the OFI plugin that uses empirically-derived performance regions optimized for specific EC2 instance types (p4d, p5, p5en, etc.).

**Key Responsibilities**:
- Algorithm selection (Ring vs Tree vs NVLS vs CollNet)
- Protocol selection (Simple vs LL vs LL128)
- Channel count selection (number of SMs to use)
- Cost estimation for algorithm/protocol combinations

**Source Locations** (verified against NCCL v2.31.2-1 and aws-ofi-nccl v1.21.1):
- NCCL core tuner cost model: [nccl/src/tuning/cost_model.cc](https://github.com/NVIDIA/nccl/blob/master/src/tuning/cost_model.cc), [nccl/src/tuning/tuning.cc](https://github.com/NVIDIA/nccl/blob/master/src/tuning/tuning.cc). In NCCL 2.31 the tuning logic was split out of `enqueue.cc`/`tuning.cc` into a dedicated `src/tuning/` directory with per-algorithm cost models (`ring.cc`, `tree.cc`, `nvls.cc`, `pat.cc`, `collnet.cc`), plus a Copy-Engine model (`ce_model.cc`) and a symmetric-memory model (`sym_model.cc`).
- Tuner plugin loader / ABI shim: [nccl/src/plugin/tuner/](https://github.com/NVIDIA/nccl/blob/master/src/plugin/tuner) — NCCL core looks up the symbol `ncclTunerPlugin_v6` (see `nccl_tuner.h`, `NCCL_TUNER_PLUGIN_SYMBOL`) and provides backward-compatible shims for v2/v3/v4/v5 plugins.
- Tuner plugin API headers: [nccl/src/include/plugin/tuner/tuner_v6.h](https://github.com/NVIDIA/nccl/blob/master/src/include/plugin/tuner/tuner_v6.h) (and `tuner_v2.h`..`tuner_v5.h`).
- AWS OFI tuner: [aws-ofi-nccl/src/tuner/](https://github.com/aws/aws-ofi-nccl/blob/master/src/tuner) — the tuner sources moved here in aws-ofi-nccl v1.21.x (previously `src/nccl_ofi_tuner*.cpp`).

## NCCL Core Tuner Architecture

### Algorithms

From [aws-ofi-nccl 3rd-party/nccl/.../tuner.h (lines 25-33)](https://github.com/aws/aws-ofi-nccl/blob/master/3rd-party/nccl/cuda/include/nccl/tuner.h):

```c
#define NCCL_ALGO_UNDEF -1
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET_DIRECT 2
#define NCCL_ALGO_COLLNET_CHAIN 3
#define NCCL_ALGO_NVLS 4
#define NCCL_ALGO_NVLS_TREE 5
#define NCCL_ALGO_PAT 6
#define NCCL_NUM_ALGORITHMS 7  // Tree/Ring/CollNet*/NVLS*/PAT
```

**Algorithm Characteristics**:

| Algorithm | Best For | Complexity | Bandwidth | Latency |
|-----------|----------|------------|-----------|---------|
| **TREE** | Small messages, low latency | O(log N) steps | Lower | Lower |
| **RING** | Large messages, high bandwidth | O(N) steps | Higher | Higher |
| **NVLS** | NVLink-connected GPUs | Hardware-accelerated | Highest (intra-node) | Lowest (intra-node) |
| **NVLS_TREE** | Multi-node with NVLS | Hybrid | High | Low |
| **COLLNET_DIRECT** | Switch/network aggregation | O(1) with switch | Highest (with switch) | Medium |
| **COLLNET_CHAIN** | Daisy-chained CollNet | O(N) with switch | High | Medium |
| **PAT** | AllGather / ReduceScatter at scale (parallel aggregated trees) | O(log N) steps | Good | Low-Medium |

**Selection Criteria**:
- Message size: Tree better for < 128 KB, Ring better for > 1 MB (typical thresholds)
- Network topology: NVLS requires NVLink, CollNet requires switch support
- Number of nodes: Tree scales better for multi-node
- PAT is a production algorithm (since NCCL 2.23) selected for AllGather and ReduceScatter; it is not applicable to AllReduce. The AWS region tuner only exposes PAT via the v3/v6 cost-table interface (not the legacy v2 algorithm/protocol interface).

### Protocols

From [aws-ofi-nccl 3rd-party/nccl/.../tuner.h (lines 35-39)](https://github.com/aws/aws-ofi-nccl/blob/master/3rd-party/nccl/cuda/include/nccl/tuner.h):

```c
#define NCCL_PROTO_UNDEF -1
#define NCCL_PROTO_LL 0         // Low Latency
#define NCCL_PROTO_LL128 1      // Low Latency 128-byte
#define NCCL_PROTO_SIMPLE 2     // Simple/Direct
#define NCCL_NUM_PROTOCOLS 3
```

**Protocol Characteristics**:

| Protocol | Packet Size | Overhead | Best For | Bandwidth | Latency |
|----------|-------------|----------|----------|-----------|---------|
| **LL (Low Latency)** | 8 bytes | High | < 32 KB messages | ~20-30% of peak | Lowest |
| **LL128** | 128 bytes | Medium | 32 KB - 1 MB | ~50-70% of peak | Low |
| **SIMPLE** | Full MTU (up to 8KB) | Low | > 1 MB messages | ~90-100% of peak | Higher |

**LL Protocol Details**:
- Uses 8-byte flag-protected transfers
- GPU spins on flags for synchronization
- Extra synchronization overhead but lower latency
- Typical use: Small AllReduce (< 32 KB)

**LL128 Protocol Details**:
- Uses 128-byte blocks with flags
- Balance between LL and Simple
- Typical use: Medium AllReduce (32 KB - 1 MB)

**SIMPLE Protocol Details**:
- Direct memory transfers with no inline flags
- Highest bandwidth utilization
- Typical use: Large AllReduce (> 1 MB)

### Cost Table

NCCL builds a **cost table** estimating the time for each algorithm/protocol combination:

**Structure** (2D array):
```c
float collCostTable[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
```

**Cost Calculation**:
```
cost(algo, proto) = baseLatency + (nBytes / bandwidth) + hwLatency
```

Where:
- `baseLatency`: Fixed protocol overhead
- `bandwidth`: Algorithm-specific bandwidth (depends on comm size, hardware)
- `hwLatency`: Hardware-specific latency (NVLink, PCIe, Network)

**Selection Logic**:
1. Build cost table for all valid algorithm/protocol pairs
2. Mark unsupported combinations as `NCCL_ALGO_PROTO_IGNORE` (-1.0)
3. Call tuner plugin (if loaded) to override costs
4. Select algorithm/protocol with minimum cost
5. Determine number of channels to use

## Tuner Plugin API

### Which version NCCL loads

NCCL core loads the **highest tuner ABI version it supports** and falls back to older ones. In **NCCL v2.31.2-1** the symbol NCCL core looks up first is `ncclTunerPlugin_v6`:

From [nccl/src/include/plugin/nccl_tuner.h (line 25)](https://github.com/NVIDIA/nccl/blob/master/src/include/plugin/nccl_tuner.h):

```c
#include "tuner/tuner_v6.h"
#define NCCL_TUNER_PLUGIN_SYMBOL "ncclTunerPlugin_v6"
```

NCCL also ships backward-compatible shims (`src/plugin/tuner/tuner_v2.cc` … `tuner_v5.cc`) that wrap an older plugin symbol into the internal v6 vtable, setting `getChunkSize = NULL` for pre-v6 plugins.

The **aws-ofi-nccl tuner (v1.21.1) exports three symbols simultaneously** so a single `.so` works across NCCL versions. Verified in [aws-ofi-nccl/src/tuner/nccl_ofi_tuner.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/tuner/nccl_ofi_tuner.cpp):

```c
NCCL_OFI_EXPORT_SYMBOL ncclTuner_v3_t ncclTunerPlugin_v3 = { ... };  // line ~185, "Tuner v3 was introduced in NCCL 2.22.3"
NCCL_OFI_EXPORT_SYMBOL ncclTuner_v6_t ncclTunerPlugin_v6 = { ... };  // line ~242, "Tuner v6 was introduced in NCCL 2.30.3"
NCCL_OFI_EXPORT_SYMBOL ncclTuner_v2_t ncclTunerPlugin_v2 = { ... };  // line ~274, "Tuner v2 was introduced in NCCL 2.21.5"
```

NCCL 2.31 therefore binds to the plugin's `ncclTunerPlugin_v6`. (Note: there is **no** `ncclTunerPlugin_v5` exported by the AWS tuner — the v5 header exists in-tree but the plugin skips straight from v3 to v6, because v6 is the first ABI that adds the `getChunkSize` callback the tuner needs.)

### Tuner v6 Interface

From [tuner_v6.h](https://github.com/NVIDIA/nccl/blob/master/src/include/plugin/tuner/tuner_v6.h). The v6 struct is identical to v5 for `init`/`getCollInfo`/`finalize` and **adds a fourth, optional callback `getChunkSize`**. The constants and NVL-domain-info structs are reused unchanged from v5 (`typedef ncclTunerConstants_v5_t ncclTunerConstants_v6_t;`).

```c
typedef struct {
  const char* name;

  // Initialize per-communicator tuner state.
  //   commId, nRanks, nNodes, logFunction, nvlDomainInfo -> in
  //   ctx -> out ; constants -> in/out
  ncclResult_t (*init)(void** ctx, uint64_t commId, size_t nRanks, size_t nNodes,
                       ncclDebugLogger_t logFunction,
                       ncclNvlDomainInfo_v6_t* nvlDomainInfo,
                       ncclTunerConstants_v6_t* constants);

  // Choose algo/proto/nChannels by editing the cost table.
  //   regBuff indicates whether the user buffer can be registered.
  //   collCostTable[algo][proto] entries NCCL wants ignored are set to -1.0.
  ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable,
                              int numAlgo, int numProto,
                              int regBuff, int* nChannels);

  // Terminate and free resources.
  ncclResult_t (*finalize)(void* context);

  // NEW IN v6: override the per-step chunk size NCCL computed.
  // Optional — if NULL, NCCL uses its own chunk size. NCCL clamps the
  // returned value to the maximum allowed by the channel buffer.
  ncclResult_t (*getChunkSize)(void* context, ncclFunc_t collType, size_t nBytes,
                               int algo, int proto, int nChannels, size_t* chunkSize);
} ncclTuner_v6_t;
```

**What changed vs v3:**

| Field | v2 (2.21.5) | v3 (2.22.3) | v6 (2.30.3+) |
|-------|-------------|-------------|--------------|
| `init` signature | `(nRanks, nNodes, logFn, ctx)` | `(nRanks, nNodes, logFn, ctx)` | `(ctx, commId, nRanks, nNodes, logFn, nvlDomainInfo, constants)` |
| collective selection | returns explicit `algorithm`/`protocol` out-params | edits `collCostTable` (set entry to 0.0 to force) | edits `collCostTable` (same as v3) + gets `regBuff` |
| teardown callback name | `destroy` | `destroy` | `finalize` |
| chunk-size control | none | none | **`getChunkSize` callback (new)** |
| PAT support | no (PAT not expressible in v2 out-params) | yes (via cost table) | yes (via cost table) |

The jump from v3 to v6 is what lets a tuner influence **chunk sizing** in addition to algorithm/protocol/channel selection. NCCL core actually invokes it: see [nccl/src/enqueue/enqueue.cc (≈line 2293)](https://github.com/NVIDIA/nccl/blob/master/src/enqueue/enqueue.cc):

```c
if (comm->tuner != nullptr && comm->tuner->getChunkSize != nullptr) {
  ...
  NCCLCHECK(comm->tuner->getChunkSize(comm->tunerContext, info->func, nBytes,
                                      info->algorithm, info->protocol, nChannels, &chunkSize));
}
```

### How the AWS tuner wires up the three ABIs

From [nccl_ofi_tuner.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/tuner/nccl_ofi_tuner.cpp), all three exported symbols share one `nccl_ofi_tuner_init()`/context. The context holds function pointers that are dispatched per ABI:

- `ncclTunerPlugin_v2` → `nccl_ofi_tuner_init_v2` → `get_coll_info_internal_v2` (declines to load if `NCCL_ALGO`/`NCCL_PROTO` are set, since v2 cannot honor them; PAT regions are skipped).
- `ncclTunerPlugin_v3` → `get_coll_info_internal_v3` (cost-table edit, no chunk size).
- `ncclTunerPlugin_v6` → `get_coll_info_internal_v6` (delegates to the v3 cost-table logic — `regBuff` is unused by the region tuner) **plus** `getChunkSize` → `get_chunk_size_internal` (only wired for the **region** tuner; the model tuner sets it to `nullptr`).

### Tuner Constants

From [tuner_v5.h](https://github.com/NVIDIA/nccl/blob/master/src/include/plugin/tuner/tuner_v5.h) (v6 reuses these verbatim via typedef):

```c
typedef struct {
  // Base latencies for each algorithm/protocol combination
  double baseLatencies[NCCL_NUM_ALGORITHMS_V5][NCCL_NUM_PROTOCOLS_V5];

  // Hardware-specific latencies (NVLink, PCIe, Network)
  double hwLatencies[NCCL_NUM_HW_LINKS_V5][NCCL_NUM_ALGORITHMS_V5][NCCL_NUM_PROTOCOLS_V5];

  // Maximum bandwidths for LL protocol
  double llMaxBws[NCCL_NUM_COMPCAPS_V5][NCCL_NUM_TUNING_SCALES_V5];

  // Per-channel max bandwidths for Ring/Tree with LL128
  double perChMaxRingLL128Bws[NCCL_NUM_COMPCAPS_V5][NCCL_NUM_TUNING_SCALES_V5];
  double perChMaxTreeLL128Bws[NCCL_NUM_COMPCAPS_V5][NCCL_NUM_TUNING_SCALES_V5];

  // Per-channel max bandwidths for Tree with Simple protocol
  double perChMaxTreeBws[NCCL_NUM_COMPCAPS_V5][NCCL_NUM_TUNING_SCALES_V5];

  // Per-channel max bandwidths for NVLS Tree
  double perChMaxNVLSTreeBws[NCCL_NUM_COMPCAPS_V5][NCCL_NUM_TUNING_SCALES_V5];
} ncclTunerConstants_v5_t;   // == ncclTunerConstants_v6_t
```

**Compute Capabilities** (`NCCL_NUM_COMPCAPS_V5 = 4`):
```c
#define NCCL_VOLTA_COMPCAP_IDX 0      // Volta (V100)
#define NCCL_AMPERE_COMPCAP_IDX 1     // Ampere (A100)
#define NCCL_HOPPER_COMPCAP_IDX 2     // Hopper (H100/H200)
#define NCCL_BLACKWELL_COMPCAP_IDX 3  // Blackwell (B200/B300)
```

**Tuning Scales** (communicator size, `NCCL_NUM_TUNING_SCALES_V5 = 3`):
```c
#define NCCL_TUNING_SCALE_1NODE 0     // Single node
#define NCCL_TUNING_SCALE_2NODES 1    // 2 nodes
#define NCCL_TUNING_SCALE_4NODES 2    // 4+ nodes
```

## Channel Selection

**Number of Channels**:
- Each channel uses one GPU SM (Streaming Multiprocessor)
- More channels = more parallelism but also more overhead
- Typical range: 1-32 channels

**Selection Criteria**:
- Message size: Small messages use fewer channels (1-4), large use more (16-32)
- Available SMs: Must leave SMs for compute kernels
- Algorithm: Ring benefits more from many channels than Tree

**Default Logic** (simplified):
```c
if (nBytes < 32 KB) {
    nChannels = min(4, maxChannels);
} else if (nBytes < 1 MB) {
    nChannels = min(16, maxChannels);
} else {
    nChannels = maxChannels;  // Typically 32
}
```

**Plugin Override**:
- Plugin can set `nChannels` output parameter
- NCCL respects plugin choice if valid
- Must be <= `comm->nChannels` (max available)

## AWS OFI Region-Based Tuner

### Overview

The AWS OFI plugin includes a **region-based tuner** that uses empirically-derived performance regions to select algorithm/protocol combinations optimized for specific EC2 instance types.

**Source Location**: [aws-ofi-nccl/src/tuner/](https://github.com/aws/aws-ofi-nccl/blob/master/src/tuner)

**Key Files**:
- [nccl_ofi_tuner.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/tuner/nccl_ofi_tuner.cpp) - Plugin entry points
- [nccl_ofi_regions.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/tuner/nccl_ofi_regions.cpp) - Region definitions
- [nccl_ofi_tuner_region.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/tuner/nccl_ofi_tuner_region.h) - Region data structures

### Region-Based Selection

**Concept**: Performance space is divided into regions where specific algorithm/protocol combinations perform best.

**2D Space**:
- X-axis: Message size (bytes)
- Y-axis: Number of ranks

**Region Definition** ([nccl_ofi_tuner_region.h:62-67](https://github.com/aws/aws-ofi-nccl/blob/master/include/tuner/nccl_ofi_tuner_region.h)):

```c
typedef struct nccl_ofi_tuner_point {
	double x;  // Message size or TUNER_MAX_SIZE
	double y;  // Number of ranks or TUNER_MAX_RANKS
} nccl_ofi_tuner_point_t;

typedef struct nccl_ofi_tuner_region {
	int algorithm;                                      // NCCL algorithm
	int protocol;                                       // NCCL protocol
	size_t num_vertices;                                // Number of polygon vertices
	nccl_ofi_tuner_point_t vertices[TUNER_MAX_NUM_VERTICES];  // Polygon vertices
} nccl_ofi_tuner_region_t;
```

**Region Limits** ([nccl_ofi_tuner_region.h:43-54](https://github.com/aws/aws-ofi-nccl/blob/master/include/tuner/nccl_ofi_tuner_region.h)):

```c
/* Maximum number of vertices per region */
#define TUNER_MAX_NUM_VERTICES 20

/* Maximum number of ranks the tuner can handle.
 * Above this, fall back to NCCL's tuner.
 */
#define TUNER_MAX_RANKS (1024.0 * 1024)  // 1 million ranks

/* Maximum message size the tuner can handle.
 * Above this, fall back to NCCL's tuner.
 */
#define TUNER_MAX_SIZE (100.0 * 1024 * 1024 * 1024)  // 100 GB
```

**Point-in-Polygon Test**:
Given a message size and rank count, determine which region contains the point:

```c
int is_inside_region(nccl_ofi_tuner_point_t point,
                     nccl_ofi_tuner_region_t *region);
```

Uses ray-casting algorithm to determine polygon containment.

### Supported Platforms

From [include/tuner/nccl_ofi_tuner_common.h (lines 18-24)](https://github.com/aws/aws-ofi-nccl/blob/master/include/tuner/nccl_ofi_tuner_common.h):

```c
enum nccl_ofi_tuner_platform {
	NCCL_OFI_TUNER_P5_P5E = 0,  // p5.48xlarge / p5e.48xlarge (H100, EFA)
	NCCL_OFI_TUNER_P5EN,        // p5en.48xlarge (H200, EFAv3)
	NCCL_OFI_TUNER_P6,          // p6.48xlarge (B200)
	NCCL_OFI_TUNER_P6_B300,     // p6 with B300
	NCCL_OFI_TUNER_UNKNOWN,
	NCCL_OFI_TUNER_PLATFORM_MAX = NCCL_OFI_TUNER_UNKNOWN
};
```

Notes on the current enum (verified in `include/tuner/nccl_ofi_tuner_common.h` and `include/tuner/nccl_ofi_tuner_process_config.h`):
- **p5 and p5e share a single platform id** (`NCCL_OFI_TUNER_P5_P5E`); they are no longer distinct entries.
- **P6-B300 is now a first-class recognized platform** (`NCCL_OFI_TUNER_P6_B300`), no longer "future". `NCCL_OFI_TUNER_P6` denotes P6 with B200.
- Both the **region** tuner (`is_region_supported`, in `src/tuner/nccl_ofi_regions.cpp`) and the **model** tuner (`is_model_supported`, in `src/tuner/nccl_ofi_model.cpp`) report support for P5/P5E, P5EN, P6 and P6-B300.

**Platform Detection & tuner selection** (in [nccl_ofi_tuner.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/tuner/nccl_ofi_tuner.cpp) `nccl_ofi_tuner_init`):
- The platform is resolved once per process by `TunerProcessConfig` (`include/tuner/nccl_ofi_tuner_process_config.h`).
- If neither region nor model tuner supports the (platform, nRanks, nNodes) combination, the plugin returns without a context and NCCL falls back to its internal tuner.
- When **both** region and model are supported, the tuner **prefers Region** unless `OFI_NCCL_TUNER_TYPE`/force-model requests the model tuner and the model tuner supports the configuration.
- Logs the choice at init: `"Region base Tuner is chosen for platform: <name>"` or `"Model base Tuner is chosen ..."` under `NCCL_DEBUG_SUBSYS=INIT,TUNING`.

### Example: p5en AllReduce Regions

From [nccl_ofi_regions.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/tuner/nccl_ofi_regions.cpp) (simplified):

**For 8 ranks per node (typical configuration)**:

```c
// AllReduce regions for p5en
const nccl_ofi_tuner_region_t regions[] = {
	// Tree + LL: Small messages (0-196 KB, 16-1024 ranks)
	{
		.algorithm = NCCL_ALGO_TREE,
		.protocol = NCCL_PROTO_LL,
		.num_vertices = 5,
		.vertices = {
			{0, 16},           // Bottom-left
			{196608, 16},      // Bottom-right (192 KB)
			{196608, 1024},    // Top-right
			{extended_point},  // Extended to max
			{0, TUNER_MAX_RANKS}  // Top-left
		}
	},

	// Tree + LL128: Medium messages (128 MB - 208 MB, 127-1024 ranks)
	{
		.algorithm = NCCL_ALGO_TREE,
		.protocol = NCCL_PROTO_LL128,
		.num_vertices = 7,
		.vertices = {
			{129499136, 127},  // ~123 MB
			{218103808, 1024}, // ~208 MB
			{extended_point},
			// ... more vertices
		}
	},

	// NVLS Tree + Simple: Large messages (7 GB+, 256-448 ranks)
	{
		.algorithm = NCCL_ALGO_NVLS_TREE,
		.protocol = NCCL_PROTO_SIMPLE,
		.num_vertices = 7,
		.vertices = {
			{7516192768, 256},   // ~7 GB
			{17179869184, 448},  // ~16 GB
			{extended_point},
			// ... more vertices
		}
	},

	// Ring + Simple: Very large messages (default for > 16 GB)
	{
		.algorithm = NCCL_ALGO_RING,
		.protocol = NCCL_PROTO_SIMPLE,
		.num_vertices = 4,
		.vertices = {
			{0, 0},
			{TUNER_MAX_SIZE, 0},
			{TUNER_MAX_SIZE, TUNER_MAX_RANKS},
			{0, TUNER_MAX_RANKS}
		}
	}
};
```

**Interpretation**:
- Small messages (< 192 KB): Use Tree + LL for low latency
- Medium messages (128-208 MB): Use Tree + LL128 for balance
- Large messages (7-16 GB): Use NVLS Tree + Simple for high bandwidth (intra-node NVLink)
- Very large messages (> 16 GB): Use Ring + Simple for maximum bandwidth

### Region Extension

**Purpose**: Extend a region boundary to infinity (max size/ranks)

From [nccl_ofi_tuner_region.h:69-71](https://github.com/aws/aws-ofi-nccl/blob/master/include/tuner/nccl_ofi_tuner_region.h):

```c
nccl_ofi_tuner_point_t extend_region(nccl_ofi_tuner_point_t a,
                                     nccl_ofi_tuner_point_t b,
                                     nccl_ofi_tuner_point_t z);
```

**Algorithm**:
Given two points `a` and `b` defining a line segment, extend it to point `z` (typically at infinity).

**Example**:
```c
// Extend Tree LL region from (196608, 192) through (196608, 1024) to infinity
nccl_ofi_tuner_point_t extended = extend_region(
	(nccl_ofi_tuner_point_t){196608, 192},
	(nccl_ofi_tuner_point_t){196608, 1024},
	(nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS}
);
// Result: extended point at maximum size/ranks maintaining line direction
```

This creates regions that extend to cover all larger message sizes or rank counts.

### Tuner Selection Logic

**Decision Flow**:

1. **Check platform support**:
   ```c
   bool is_region_supported(enum nccl_ofi_tuner_platform platform,
                            size_t nRanks, size_t nNodes)
   ```
   - Returns true if platform has tuned regions
   - Returns false → fall back to NCCL default

2. **Initialize region context**:
   ```c
   ncclResult_t region_init_internal(nccl_ofi_tuner_context_t *ctx,
                                     enum nccl_ofi_tuner_platform platform,
                                     size_t nRanks, size_t nNodes)
   ```
   - Loads platform-specific regions
   - Allocates region arrays per collective type
   - Sets up geometry (log2 of nodes)

3. **Get collective info**:
   ```c
   ncclResult_t region_get_coll_info_internal_v3(nccl_ofi_tuner_context_t *ctx,
                                                  ncclFunc_t collType,
                                                  size_t nBytes,
                                                  int numPipeOps,
                                                  float **collCostTable,
                                                  int numAlgo, int numProto,
                                                  int *nChannels)
   ```
   - Check if `(nBytes, nRanks)` point is inside any region
   - If inside region: Set cost table entry for that algo/proto to 0.0 (highest priority)
   - If outside all regions: Leave costs unchanged (NCCL default)
   - Optionally set `nChannels`

4. **NCCL selects minimum cost**:
   - NCCL core picks algorithm/protocol with lowest cost
   - If tuner set cost to 0.0, that combination is chosen
   - If tuner didn't modify, NCCL's calculated cost is used

### Cost Table Modification

**Tuner Strategy**:

```c
float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

// Iterate through regions for this collective type
for (size_t i = 0; i < num_regions; i++) {
	nccl_ofi_tuner_region_t *region = &regions[i];

	// Check if (nBytes, nRanks) is inside this region
	nccl_ofi_tuner_point_t point = {.x = nBytes, .y = nRanks};
	if (is_inside_region(point, region)) {
		// Set this algo/proto cost to 0.0 (highest priority)
		table[region->algorithm][region->protocol] = 0.0;

		// Optionally set channel count
		if (nChannels) {
			*nChannels = compute_optimal_channels(nBytes, nRanks);
		}

		return ncclSuccess;
	}
}

// No region matched: NCCL uses default tuning
return ncclSuccess;
```

**Effect**:
- Region match: Forces NCCL to use specified algorithm/protocol
- No match: NCCL uses cost-based selection from its internal model

### Region tuner guards (verified in `region_get_coll_info_internal_v3`)

Two guards in `nccl_ofi_regions.cpp` matter for correctness:
- **Skip small clusters**: if `num_nodes <= 2` the region tuner returns immediately and lets NCCL's internal tuner decide ("regions are not well defined" at that scale).
- **Validity check**: an algo/proto is only forced if it is present in the cost table (`algorithm < numAlgo && protocol < numProto`) and NCCL has not marked it `NCCL_ALGO_PROTO_IGNORE` (-1.0). This prevents forcing an unsupported combination on an older NCCL.

The `(nBytes, nRanks)` point is transformed with `p.transform_log2()` before the point-in-polygon test, i.e. the region polygons are defined in **log2(size) × log2(ranks)** space, not linear space.

### Channel-count and chunk-size overrides (new upstream behaviour)

Beyond forcing algo/proto, the region tuner now also overrides **nChannels** and **chunk size** for specific platform/collective combinations. These were added upstream since mid-2026 and are all verified in [src/tuner/nccl_ofi_regions.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/tuner/nccl_ofi_regions.cpp).

**1. PAT channel optimization (P6 and P6-B300).** After a region selects PAT/Simple for AllGather/ReduceScatter in the one-rank-per-node (0x7) topology (`num_nodes == num_ranks`) with `nBytes <= 32 MiB`, the tuner overrides `nChannels` via `calculateBestNChannelPat()` (1 channel up to `nNodes*64KiB`, 2 channels up to `nNodes*128KiB`). Originally gated to `NCCL_OFI_TUNER_P6` (B200) only; **commit 4a8ed08 extended the guard to also match `NCCL_OFI_TUNER_P6_B300`**, since B300 shows the same PAT/Simple behaviour in the 0x7 topology:

```c
if ((region_ctx->platform == NCCL_OFI_TUNER_P6 || region_ctx->platform == NCCL_OFI_TUNER_P6_B300)
    && (nBytes <= 32 * 1024 * 1024)
    && (algorithm == NCCL_ALGO_PAT) && (protocol == NCCL_PROTO_SIMPLE)
    && (region_ctx->dims.num_nodes == region_ctx->dims.num_ranks)) {
    *nChannels = calculateBestNChannelPat(nBytes, region_ctx->dims.num_nodes);
}
```

**2. Tree/LL128 channel selection (P6-B200).** For AllReduce Tree/LL128 in the 8-ranks-per-node (0x0) topology at 4–32 MiB on `NCCL_OFI_TUNER_P6`, the tuner picks the `nChannels` (16/24/32) that yields the largest LL128 chunk via `calculateBestNChannelTree()` → `calculateChunkSizeTreeLL128()`.

**3. Chunk-size tuning via the v6 `getChunkSize` callback.** Only the region tuner registers `get_chunk_size_internal` (the model tuner leaves it `nullptr`). `region_get_chunk_size_internal()` only acts in the **1-rank-per-node** case (`num_nodes == num_ranks`) and, on P5en, dispatches two heuristics:

- **AllReduce Tree/LL128 on P5en** (commit b5ed24e): `chunkSizeTuningTreeLL128P5en()` steps a ladder **288000 → 144000 → 72000 → 36000 → 18000 bytes** (~288 KB down to ~18 KB; NCCL later grains these to the sizes it uses, e.g. 36000→34560). It steps down based on how many chunks the per-channel message (`nBytes = msg_size/nChannels`) would split into, with thresholds that scale with tree depth `nsteps = 1 + log2(nNodes)` (the step-1 threshold is `2·nsteps²`). Rationale (from source): larger chunks put more bytes in flight — which EFA rewards only up to its saturation point (≈288 KB keeps in-flight near saturation) — while smaller chunks give more chunks to keep a deep tree pipeline full. (Note: this is a distinct heuristic from `calculateChunkSizeTreeLL128()`, which is used only for the P6 Tree/LL128 nChannel selection described above.)

- **AllGather PAT/Simple on P5en** (commit cb33a4c): `chunkSizeTuningAllGatherPatSimpleP5en()` steps down from a per-cluster ceiling (1 MiB for `nNodes < 16`, 512 KiB for `nNodes >= 16`) to a 32 KiB floor, choosing the largest chunk that still yields at least `T` chunks per message-size zone. Rationale documented in-source: **PAT pipelines within a phase but not across phases** (each channel's inflight budget is `NCCL_STEPS = 8`). Fewer/larger chunks keep mid/large messages in a single pipelined phase (avoiding a repeated `~log2(nNodes)*RTT` latency term); more/smaller chunks fill the recursive-doubling pipeline faster for latency-bound small messages.

NCCL core calls this back only when the plugin exposes v6 (`comm->tuner->getChunkSize != nullptr`), and clamps the returned value to the channel buffer maximum.

## Tuning Parameters

### Environment Variables

**NCCL Core**:
```bash
# Force specific algorithm (overrides tuner)
export NCCL_ALGO=TREE    # or RING, NVLS, etc.

# Force specific protocol (overrides tuner)
export NCCL_PROTO=LL     # or LL128, SIMPLE

# Set number of channels (overrides tuner)
export NCCL_NCHANNELS=16

# Enable tuner plugin
export NCCL_TUNER_PLUGIN=/path/to/libnccl-ofi-tuner.so

# Tuner configuration file (plugin-specific)
# There is no NCCL_TUNER_CONFIG_FILE variable, and the AWS tuner has no config file --
# its region tables are compiled in. NCCL selects which tuner plugin to load with:
export NCCL_TUNER_PLUGIN=ofi         # or a full .so path
# NCCL 2.31 binds the symbol ncclTunerPlugin_v6 (src/include/plugin/nccl_tuner.h)
```

**AWS OFI Tuner**:
```bash
# There is no enable/disable env var. The tuner is active whenever NCCL loads a
# plugin that exports ncclTunerPlugin_v2/v3/v6. To pick the selection strategy:
export OFI_NCCL_TUNER_TYPE=REGION   # REGION (default) | MODEL | INTERNAL

# Override platform detection
export OFI_NCCL_PLATFORM=p5en

# Tuner debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=TUNING
```

### Tuner Constants Tuning

**Customizing Latencies** (example):

```c
ncclTunerConstants_v5_t constants;

// Base latencies (μs)
constants.baseLatencies[NCCL_ALGO_TREE][NCCL_PROTO_LL] = 5.0;
constants.baseLatencies[NCCL_ALGO_TREE][NCCL_PROTO_LL128] = 7.0;
constants.baseLatencies[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 10.0;

// Hardware latencies (μs)
constants.hwLatencies[NCCL_HW_NVLINK][NCCL_ALGO_TREE][NCCL_PROTO_LL] = 1.0;
constants.hwLatencies[NCCL_HW_NET][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 15.0;

// Max bandwidths (GB/s)
constants.llMaxBws[NCCL_AMPERE_COMPCAP_IDX][NCCL_TUNING_SCALE_1NODE] = 300.0;
constants.perChMaxRingLL128Bws[NCCL_HOPPER_COMPCAP_IDX][NCCL_TUNING_SCALE_4NODES] = 50.0;
```

These constants are passed to plugin's `init()` function and can be modified by the plugin.

## Performance Impact

### Algorithm/Protocol Selection

**Impact on Latency** (estimated for AllReduce):

| Message Size | Default (NCCL) | Optimal (Tuned) | Improvement |
|--------------|----------------|-----------------|-------------|
| 4 KB         | Tree/LL        | Tree/LL         | Baseline    |
| 64 KB        | Tree/LL128     | Tree/LL128      | Baseline    |
| 1 MB         | Ring/LL128     | Ring/Simple     | ~20-30% faster |
| 128 MB       | Ring/Simple    | NVLS Tree/Simple (intra-node) | ~50-80% faster |

**Impact on Bandwidth**:

| Message Size | Protocol | Achieved BW (% of peak) |
|--------------|----------|------------------------|
| < 32 KB      | LL       | ~20-30%                |
| 32 KB - 1 MB | LL128    | ~50-70%                |
| > 1 MB       | SIMPLE   | ~90-100%               |

### Channel Count Impact

**Tradeoffs**:
- More channels: Higher bandwidth for large messages, more SM utilization
- Fewer channels: Lower overhead for small messages, more SMs for compute

**Example** (1 MB AllReduce, 8 GPUs):

| Channels | Latency (estimated) | SM Usage |
|----------|---------------------|----------|
| 4        | ~250 μs             | 4/108 SMs |
| 16       | ~150 μs             | 16/108 SMs |
| 32       | ~120 μs             | 32/108 SMs |

Diminishing returns beyond 16-32 channels for most workloads.

### Region-Based Tuner Performance

**Indicative benefits** (AWS-provided regions for p5en). These figures are illustrative
estimates of the magnitude of improvement the region tables target; no benchmark run,
instance count or date is recorded for them in this documentation, so do not quote them as
measurements:
- AllReduce 4 KB: ~15-20% lower latency vs NCCL default (Tree/LL selection)
- AllReduce 128 MB: ~30-40% higher bandwidth vs NCCL default (NVLS Tree/Simple for intra-node)
- AllReduce 1 GB: ~10-15% higher bandwidth (Ring/Simple with optimal channels)

The underlying regions themselves are empirically derived from benchmarking on specific
hardware configurations; re-measure on your target instance type before relying on any
specific number above.

## Custom Tuner Development

### Minimal Example

```c
#include <nccl/tuner.h>

static ncclResult_t myTuner_init(void** ctx, uint64_t commId, size_t nRanks, size_t nNodes,
                                 ncclDebugLogger_t logFunction,
                                 ncclNvlDomainInfo_v6_t* nvlDomainInfo,
                                 ncclTunerConstants_v6_t* constants) {
	// Allocate context
	*ctx = malloc(sizeof(int));  // Simplified
	return ncclSuccess;
}

static ncclResult_t myTuner_getCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                                        int numPipeOps, float** collCostTable,
                                        int numAlgo, int numProto, int regBuff, int* nChannels) {
	float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

	// Example: Prefer Tree/LL for messages < 64 KB
	if (collType == ncclFuncAllReduce && nBytes < 65536) {
		table[NCCL_ALGO_TREE][NCCL_PROTO_LL] = 0.0;  // Force this choice
		*nChannels = 4;  // Use 4 channels
	}

	return ncclSuccess;
}

static ncclResult_t myTuner_finalize(void* context) {
	free(context);
	return ncclSuccess;
}

// Optional in v6: override NCCL's per-step chunk size
static ncclResult_t myTuner_getChunkSize(void* context, ncclFunc_t collType, size_t nBytes,
                                         int algo, int proto, int nChannels, size_t* chunkSize) {
	// Example: cap Tree/LL128 chunk size (NCCL clamps to buffer max)
	if (algo == NCCL_ALGO_TREE && proto == NCCL_PROTO_LL128)
		*chunkSize = 131072;  // 128 KB
	return ncclSuccess;
}

// Export plugin symbol — NCCL 2.31 looks up ncclTunerPlugin_v6
const ncclTuner_v6_t ncclTunerPlugin_v6 = {
	.name = "MyCustomTuner",
	.init = myTuner_init,
	.getCollInfo = myTuner_getCollInfo,
	.finalize = myTuner_finalize,
	.getChunkSize = myTuner_getChunkSize  // set to NULL to keep NCCL's chunk size
};
```

**Build**:
```bash
gcc -shared -fPIC -o libmy_tuner.so my_tuner.c -I/path/to/nccl/include
```

**Use**:
```bash
export NCCL_TUNER_PLUGIN=/path/to/libmy_tuner.so
```

### Advanced: File-Based Configuration

**Config File** (example):
```
# Format: collective size_min size_max algorithm protocol channels
allreduce 0 65536 TREE LL 4
allreduce 65536 1048576 TREE LL128 8
allreduce 1048576 inf RING SIMPLE 32
```

**Parser** (simplified):
```c
static ncclResult_t parse_config(const char* filename, tuner_config_t* config) {
	FILE* f = fopen(filename, "r");
	char line[256];

	while (fgets(line, sizeof(line), f)) {
		char coll[32], algo[32], proto[32];
		size_t size_min, size_max;
		int channels;

		sscanf(line, "%s %zu %zu %s %s %d",
		       coll, &size_min, &size_max, algo, proto, &channels);

		// Store in config structure
		add_rule(config, coll, size_min, size_max,
		         parse_algo(algo), parse_proto(proto), channels);
	}

	fclose(f);
	return ncclSuccess;
}
```

Load in `init()`, use in `getCollInfo()` to match size ranges.

## Debugging and Profiling

### Enable Tuning Logs

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,TUNING

# Run NCCL application
mpirun -np 16 ./nccl_app

# Output shows tuner decisions:
# [INFO] TUNING: AllReduce 65536 bytes -> algo=TREE proto=LL128 nChannels=8
```

### Benchmarking Tuner Choices

**NCCL Tests with Fixed Parameters**:
```bash
# Test with default tuner
./all_reduce_perf -b 1K -e 1G -f 2 -g 8

# Test with forced algorithm
NCCL_ALGO=TREE ./all_reduce_perf -b 1K -e 1G -f 2 -g 8

# Test with custom tuner
NCCL_TUNER_PLUGIN=/path/to/tuner.so ./all_reduce_perf -b 1K -e 1G -f 2 -g 8
```

Compare latency and bandwidth across configurations.

### Profiling with nsys

```bash
nsys profile --trace=nvtx,cuda,mpi ./nccl_app

# Look for:
# - Kernel launch overhead (affected by channel count)
# - Inter-kernel gaps (protocol overhead)
# - Bandwidth utilization (algorithm efficiency)
```

## Summary

| Aspect | Details |
|--------|---------|
| **Algorithms** | Tree, Ring, NVLS, NVLS Tree, CollNet Direct/Chain, PAT (7 total) |
| **Protocols** | LL (8B), LL128 (128B), Simple (full MTU) |
| **Selection** | Cost-based model with per-algorithm/protocol cost estimation |
| **Plugin API** | v6 (`ncclTunerPlugin_v6`, loaded by NCCL 2.31): init/getCollInfo/finalize + **getChunkSize**; AWS tuner also exports v3 and v2 |
| **AWS Tuner** | Region-based (log2 point-in-polygon) or model-based selection; region preferred when both apply |
| **Platforms** | P5/P5E, P5EN (H100/H200), P6 (B200), P6-B300 |
| **Max Regions** | 20 vertices per region, unlimited regions per collective |
| **Max Capacity** | 1M ranks, 100 GB message size |
| **Channel Range** | 1-32 channels (SMs) |
| **Environment Vars** | NCCL_TUNER_PLUGIN, NCCL_ALGO, NCCL_PROTO, NCCL_NCHANNELS |

**Key Takeaways**:
1. **NCCL tuner uses cost-based selection** across algorithm/protocol matrix
2. **Plugins override costs** by setting preferred combinations to 0.0
3. **AWS OFI tuner uses geometric regions** in (size, rank) space for platform-specific optimization
4. **Empirically-derived regions** for p5/p5en provide performance improvements over NCCL defaults
5. **Channel count significantly impacts** small message latency and large message bandwidth
6. **Tuner is critical** for achieving optimal performance across varying message sizes
7. **Custom tuners** can be developed for specific workloads or hardware configurations

**Related Documentation**:
- [nccl-collectives.md](nccl-collectives.md) - Collective operation algorithms (Ring, Tree, etc.)
- [nccl-core.md](nccl-core.md) - NCCL communicators, channels, and execution model
- [ofi-plugin.md](ofi-plugin.md) - OFI plugin architecture and integration
- [optimization-opportunities.md](optimization-opportunities.md) - Performance tuning opportunities including tuner customization

**References**:
1. NCCL Source: https://github.com/NVIDIA/nccl
2. AWS OFI Plugin: https://github.com/aws/aws-ofi-nccl
3. NCCL Tuner Plugin API: [nccl/src/include/plugin/tuner/](https://github.com/NVIDIA/nccl/tree/master/src/include/plugin/tuner)
