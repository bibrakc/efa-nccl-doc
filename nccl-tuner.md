# NCCL Tuner: Algorithm and Protocol Selection

## Overview

NCCL includes a **cost-based tuning system** that selects the optimal algorithm (Ring, Tree, NVLS, etc.), protocol (Simple, LL, LL128), and number of channels for each collective operation based on message size, communicator size, and hardware capabilities.

The tuner can be **extended via plugins** to override NCCL's default selections. AWS provides a **region-based tuner** in the OFI plugin that uses empirically-derived performance regions optimized for specific EC2 instance types (p4d, p5, p5en, etc.).

**Key Responsibilities**:
- Algorithm selection (Ring vs Tree vs NVLS vs CollNet)
- Protocol selection (Simple vs LL vs LL128)
- Channel count selection (number of SMs to use)
- Cost estimation for algorithm/protocol combinations

**Source Locations**:
- NCCL core tuner: [nccl/src/include/tuner.h](https://github.com/NVIDIA/nccl), [nccl/src/enqueue.cc](https://github.com/NVIDIA/nccl)
- Tuner plugin API: [nccl/src/include/plugin/tuner/tuner_v5.h](https://github.com/NVIDIA/nccl)
- AWS OFI tuner: [aws-ofi-nccl/src/tuner/](https://github.com/aws/aws-ofi-nccl/tree/75240c8/src/tuner)

## NCCL Core Tuner Architecture

### Algorithms

From [nccl_tuner.h:25-33](https://github.com/aws/aws-ofi-nccl/blob/master/3rd-party/nccl/cuda/include/nccl/tuner.h):

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
| **PAT** | Pattern-based (experimental) | Varies | Varies | Varies |

**Selection Criteria**:
- Message size: Tree better for < 128 KB, Ring better for > 1 MB (typical thresholds)
- Network topology: NVLS requires NVLink, CollNet requires switch support
- Number of nodes: Tree scales better for multi-node

### Protocols

From [nccl_tuner.h:35-39](https://github.com/aws/aws-ofi-nccl/blob/master/3rd-party/nccl/cuda/include/nccl/tuner.h):

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

### Tuner v5 Interface

From [tuner_v5.h:38-85](https://github.com/aws/aws-ofi-nccl/blob/master/3rd-party/nccl/cuda/include/nccl/tuner_v2.h):

```c
typedef struct {
  // Name of the tuner
  const char* name;

  // Initializes tuner states.
  // Inputs:
  //   - commId: communicator identifier
  //   - nRanks: number of ranks in current communicator
  //   - nNodes: number of nodes in current communicator
  //   - logFunction: log integration with NCCL core
  //   - nvlDomainInfo: NVL domain information
  // Outputs:
  //   - context: tuner context object
  // Input/Output:
  //   - constants: tuner constants (latencies, bandwidths)
  ncclResult_t (*init)(void** ctx, uint64_t commId, size_t nRanks, size_t nNodes,
                       ncclDebugLogger_t logFunction,
                       ncclNvlDomainInfo_v5_t* nvlDomainInfo,
                       ncclTunerConstants_v5_t* constants);

  // Gets info (algo, protocol, nChannels) for a collective.
  // Inputs:
  //   - context: tuner context object
  //   - collType: collective type (allreduce, allgather, etc.)
  //   - nBytes: collective size in bytes
  //   - numPipeOps: number of operations in the group
  //   - numAlgo: number of algorithms in collCostTable
  //   - numProto: number of protocols in collCostTable
  //   - regBuff: can register user buffer
  // Outputs:
  //   - nChannels: number of channels (SMs) to be used
  // InOut:
  //   - collCostTable: cost table, NCCL sets ignored entries to -1.0
  //
  // Plugin can:
  //   - Modify cost table entries (set costs for preferred algo/proto)
  //   - Set nChannels (number of SMs)
  //   - Return ncclSuccess or error (NCCL falls back to default)
  ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable,
                              int numAlgo, int numProto,
                              int regBuff, int* nChannels);

  // Terminates the plugin and cleans up resources.
  ncclResult_t (*finalize)(void* context);
} ncclTuner_v5_t;
```

**Plugin Loading**:
```bash
# Environment variable to load plugin
export NCCL_TUNER_PLUGIN=/path/to/libnccl-ofi-tuner.so

# NCCL looks for symbol: ncclTunerPlugin_v5
```

### Tuner Constants

From [tuner_v5.h:24-35](https://github.com/aws/aws-ofi-nccl/blob/master/3rd-party/nccl/cuda/include/nccl/tuner_v2.h):

```c
typedef struct {
  // Base latencies for each algorithm/protocol combination
  double baseLatencies[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

  // Hardware-specific latencies (NVLink, PCIe, Network)
  double hwLatencies[NCCL_NUM_HW_LINKS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

  // Maximum bandwidths for LL protocol
  double llMaxBws[NCCL_NUM_COMPCAPS][NCCL_NUM_TUNING_SCALES];

  // Per-channel max bandwidths for Ring/Tree with LL128
  double perChMaxRingLL128Bws[NCCL_NUM_COMPCAPS][NCCL_NUM_TUNING_SCALES];
  double perChMaxTreeLL128Bws[NCCL_NUM_COMPCAPS][NCCL_NUM_TUNING_SCALES];

  // Per-channel max bandwidths for Tree with Simple protocol
  double perChMaxTreeBws[NCCL_NUM_COMPCAPS][NCCL_NUM_TUNING_SCALES];

  // Per-channel max bandwidths for NVLS Tree
  double perChMaxNVLSTreeBws[NCCL_NUM_COMPCAPS][NCCL_NUM_TUNING_SCALES];
} ncclTunerConstants_v5_t;
```

**Compute Capabilities**:
```c
#define NCCL_VOLTA_COMPCAP_IDX 0      // Volta (V100)
#define NCCL_AMPERE_COMPCAP_IDX 1     // Ampere (A100)
#define NCCL_HOPPER_COMPCAP_IDX 2     // Hopper (H100)
#define NCCL_BLACKWELL_COMPCAP_IDX 3  // Blackwell (B100/B200)
```

**Tuning Scales** (communicator size):
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

**Source Location**: [aws-ofi-nccl/src/tuner/](https://github.com/aws/aws-ofi-nccl/tree/75240c8/src/tuner)

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

From [nccl_ofi_tuner_common.h](https://github.com/aws/aws-ofi-nccl/blob/master/include/tuner/nccl_ofi_tuner_common.h):

```c
enum nccl_ofi_tuner_platform {
	NCCL_OFI_TUNER_P5,      // p5.48xlarge (H100, EFA Gen 3)
	NCCL_OFI_TUNER_P5E,     // p5e.48xlarge (H100, EFA Gen 3)
	NCCL_OFI_TUNER_P5EN,    // p5en.48xlarge (H200, EFAv3)
	NCCL_OFI_TUNER_P6,      // p6.48xlarge (future)
	NCCL_OFI_TUNER_P6_B300, // p6 with B300 (future)
};
```

**Platform Detection**:
- Automatic detection based on instance metadata
- Falls back to NCCL default if platform not recognized
- Logs platform selection at NCCL initialization

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
export NCCL_TUNER_CONFIG_FILE=/path/to/tuner_config.txt
```

**AWS OFI Tuner**:
```bash
# Enable/disable OFI tuner (default: enabled if plugin loaded)
export OFI_NCCL_TUNER_ENABLE=1

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

**Measured Benefits** (AWS-provided regions for p5en):
- AllReduce 4 KB: ~15-20% lower latency vs NCCL default (Tree/LL selection)
- AllReduce 128 MB: ~30-40% higher bandwidth vs NCCL default (NVLS Tree/Simple for intra-node)
- AllReduce 1 GB: ~10-15% higher bandwidth (Ring/Simple with optimal channels)

These are empirically-derived regions from benchmarking on specific hardware configurations.

## Custom Tuner Development

### Minimal Example

```c
#include <nccl/tuner.h>

static ncclResult_t myTuner_init(void** ctx, uint64_t commId, size_t nRanks, size_t nNodes,
                                 ncclDebugLogger_t logFunction,
                                 ncclNvlDomainInfo_v5_t* nvlDomainInfo,
                                 ncclTunerConstants_v5_t* constants) {
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

// Export plugin symbol
const ncclTuner_v5_t ncclTunerPlugin_v5 = {
	.name = "MyCustomTuner",
	.init = myTuner_init,
	.getCollInfo = myTuner_getCollInfo,
	.finalize = myTuner_finalize
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
| **Plugin API** | v5 interface with init/getCollInfo/finalize |
| **AWS Tuner** | Region-based tuner with geometric point-in-polygon selection |
| **Platforms** | p5, p5e, p5en (H100/H200), p6 (future) |
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
