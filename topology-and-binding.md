# Topology Detection and Interface Binding in OFI NCCL Plugin

> **Source lookups:** this document records mechanism, defaults, corrections and history. For current function bodies, call graphs and blast radius, query the code graph (`codegraph explore <symbol>`) rather than trusting a pasted copy here.

## Overview

This document describes how the OFI NCCL plugin detects network devices (EFA adapters), discovers system topology, and binds to appropriate interfaces. These mechanisms are critical for:
- Multi-NIC systems (e.g., p4d.24xlarge with 4 EFAs)
- NUMA-aware communication
- GPU-NIC affinity optimization
- Optimal channel distribution

> **Note on code style below.** Many snippets in this document are
> *illustrative pseudocode* showing the concepts (PCI parsing, affinity
> selection). The authoritative NIC-selection logic lives in
> [src/nccl_ofi_topo.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_topo.cpp)
> and the per-instance defaults in
> [src/platform-aws.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/platform-aws.cpp);
> the two verified-against-source sections that follow ("NIC Selection: the
> same-PCI-level filter" and "Recognized AWS Platforms") reflect the current
> implementation.

## NIC Selection: the same-PCI-level accelerator filter

The plugin can prefer NICs that sit at the *same PCI level* as an accelerator
(GPU), which usually gives the best GPU↔NIC path. This is the
`SKIP_NICS_WITHOUT_ACCEL_AT_SAME_PCI_LEVEL` filter in
[nccl_ofi_topo.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/nccl_ofi_topo.cpp).

**The important correctness fix (commit 91bf6ac,
"topo: Don't drop all NICs when none have a same-PCI-level accelerator"):**
the filter is only applied when there is at least one NIC that *does* have an
accelerator at the same PCI level to fall back on. If it were applied
unconditionally on a topology where *no* NIC has a same-level accelerator, the
filter would drop **every** NIC and leave the plugin with nothing to use.

The decision is computed once, up front, by
`any_nic_has_accel_at_same_level()` and passed down as `apply_skip_filter` so
the counting pass and the populate pass agree.

> Source: `src/nccl_ofi_topo.cpp` — `get_info_for_node()`, `any_nic_has_accel_at_same_level()`. Use `codegraph explore get_info_for_node` for the current body.

A NIC is filtered out (`get_info_for_node` returns early) only when
`apply_skip_filter` is set **and** the node has no accelerator at the same PCI
level; otherwise it is matched normally. `apply_skip_filter` is set only when the
param is enabled and at least one NIC has a same-level accelerator, so the filter
can never remove every NIC.

Net behavior:
- **Some NICs have a same-level accelerator:** those NICs are preferred; NICs
  without one are dropped (they have a better sibling to use instead).
- **No NIC has a same-level accelerator:** the filter is *not* applied and all
  NICs are kept (previously this case could erroneously drop them all).

## Recognized AWS Platforms

The plugin ships a table of per-instance defaults in
[platform-aws.cpp](https://github.com/aws/aws-ofi-nccl/blob/master/src/platform-aws.cpp)
(`platform_data_map[]`). `name` entries with a `regex` are matched as POSIX
regexes, evaluated top-to-bottom (first match wins). Current entries:

| Entry name | Match | Default protocol | GDR required | EFA HW completion counter (GDAKI) | Notable env defaults |
|------------|-------|------------------|--------------|-----------------------------------|----------------------|
| `p4d.24xlarge`   | exact              | SENDRECV | yes | no  | `NCCL_BUFFSIZE=8388608`, `NCCL_P2P_NET_CHUNKSIZE=524288`; topology `p4d-24xl-topo.xml` |
| `p4de.24xlarge`  | exact              | SENDRECV | yes | no  | same as p4d; topology `p4de-24xl-topo.xml` |
| `p3dn.24xlarge`  | exact              | SENDRECV | no  | no  | `default_dup_conns=4`, latency 150 |
| `p5.4xlarge`     | exact              | SENDRECV | no  | no  | — |
| `p5/p5e`         | `^p5(e?\..*)`      | RDMA     | yes | no  | BUFFSIZE/CHUNKSIZE + NVLS chunk sizes, `NCCL_NET_FORCE_FLUSH=0` |
| `p5en/p6-b200`   | `^(p5en\|p6-b200).*` | RDMA   | yes | **yes** | above + `NCCL_NETDEVS_POLICY=max:1`, latency 35 |
| `p6-b300`        | `^p6-b300.*`       | RDMA     | yes | **yes** | above (no `NETDEVS_POLICY`), latency 35 |
| `p-series`       | `^p([5-9]\|[0-9]{2,}).*` | RDMA | yes | no | catch-all for P6e-GB200 and later; NVLS chunk sizes, `NCCL_NET_FORCE_FLUSH=0`, latency 35 |
| `g5.48xlarge`    | exact              | SENDRECV | no  | no  | topology `g5.48xl-topo.xml` |
| `g7e.8xlarge`    | exact              | SENDRECV | no  | no  | BUFFSIZE/CHUNKSIZE |
| `g7e`            | `^g7e\.(12\|24\|48)xlarge` | RDMA | yes | no | BUFFSIZE/CHUNKSIZE |
| `Trainium Family`| `^trn.*`           | RDMA     | yes | no  | matches Trn1/**Trn2**/Trn2u; latency 75 |
| `inf`            | `^inf.*`           | SENDRECV | yes | no  | — |

Points worth noting for an agent:
- **P6-B300 is recognized** (`^p6-b300.*`) as its own RDMA entry with the EFA
  hardware completion counter (GDAKI) opt-in enabled. Its regex is placed
  *before* the `p-series` catch-all so P6-B300 does not fall through to it.
- **Trn2 / Trn2u** are matched by the `^trn.*` "Trainium Family" entry (RDMA,
  GDR required). There is no Trn2-specific entry; the family entry covers them.
- **`efa_hw_comp_cntr` is only `true` for `p5en/p6-b200` and `p6-b300`.** This
  is the per-platform gate for EFA hardware completion counters used by the
  GDAKI path (see optimizations.md); everything else defaults to `false`. The
  unit test `tests/unit/aws_platform_mapper.cpp` asserts exactly this opt-in set.
- **P5/P5e/P5en/P6 and later default to the RDMA protocol**; P4d/P4de/P3dn and
  the G5/G7e.8xlarge entries default to SENDRECV.
- **`p4d` is *not* a recognized OFI-tuner platform.** The table above is the
  `platform-aws.cpp` per-instance *default* map; the separate OFI tuner has its
  own, narrower platform set that does not include p4d. Recognition here (for
  BUFFSIZE/protocol defaults) does not imply the tuner recognizes it.

## Device Discovery

### Plugin Initialization and Device Enumeration

When NCCL initializes, it calls into the OFI plugin to discover available network devices.

#### `ncclNet->devices()` - Device Count

> Source: `src/nccl_ofi_net.cpp` — `nccl_net_ofi_devices()`. Use `codegraph explore nccl_net_ofi_devices` for the current body.

The count comes from a libfabric query for the `efa` provider with an RDM
endpoint. The idiom, with the version macro corrected for **libfabric 2.7**:

```c
hints->fabric_attr->prov_name = strdup("efa");
hints->ep_attr->type = FI_EP_RDM;
fi_getinfo(FI_VERSION(2, 7), NULL, NULL, 0, hints, &info_list);
// one fi_info per EFA device; count those with a domain_attr->name
```

**What libfabric returns:**
- One `fi_info` structure per EFA device
- Each contains device-specific information
- Ordered by system enumeration

**Example on p4d.24xlarge (4 EFAs):**
```
Device 0: rdmap0s0 (efa_0)
Device 1: rdmap1s0 (efa_1)
Device 2: rdmap2s0 (efa_2)
Device 3: rdmap3s0 (efa_3)
```

#### `ncclNet->getProperties()` - Device Properties

For each device, NCCL queries properties to build topology:

> Source: `src/nccl_ofi_net.cpp` — `nccl_net_ofi_getProperties()`. Use `codegraph explore nccl_net_ofi_getProperties` for the current body.

The property values the plugin reports (the fields that matter for topology and
sizing) include:
- `name` — `"AWS EFA"`
- `pciPath` — PCI address string (see below); **the key field for topology**
- `guid` — from device address / PCI path
- `ptrSupport` — `NCCL_PTR_HOST | NCCL_PTR_CUDA`
- `speed` — link speed in Mbps (see [efa-hardware-architecture.md](efa-hardware-architecture.md) for actual EFA link speeds)
- `port` — `0` (single port per EFA)
- `maxComms` — concurrent connections supported
- `latency` — base latency estimate (μs); overridden per-platform in `platform-aws.cpp`
- `maxRecvs` — receive-batching count

**Critical Field: `pciPath`**

This is the key for topology detection. Format:
```
Domain:Bus:Device.Function
0000:00:06.0
││││ ││ ││ │
││││ ││ ││ └─ Function (0-7)
││││ ││ └──── Device (0-31)
││││ └──────── Bus (0-255)
└──────────────── Domain (0-65535)
```

### Extracting PCI Path from Device

The plugin resolves a device's PCI address by one of these means (see
`src/nccl_ofi_topo.cpp`; use `codegraph explore` for current bodies):

1. **From libfabric** — when `info->nic->device_attr` is present, format
   `info->nic->device_attr->pci` as `domain:bus:device.function`:
   ```c
   snprintf(pci_path, len, "%04x:%02x:%02x.%x",
            pci->domain_id, pci->bus_id, pci->device_id, pci->function_id);
   ```
2. **From sysfs** — read the symlink for the IB device; its target ends in the
   PCI address:
   ```bash
   readlink /sys/class/infiniband/<efa_dev>/device
   # -> ../../devices/pci0000:00/0000:00:06.0
   ```
3. **From the device address** — some EFA device names encode PCI info
   (`rdmap0s0`); provider-dependent fallback.

## Topology Detection

### How NCCL Uses PCI Paths

Once NCCL has PCI paths for all devices (GPUs and NICs), it builds a topology graph:

```
NCCL Topology Building:

1. Enumerate all devices:
   - GPUs: CUDA queries (cudaDeviceGetPCIBusId)
   - NICs: ncclNet->getProperties()

2. For each device, parse PCI path:
   GPU 0: 0000:00:04.0
   GPU 1: 0000:00:05.0
   EFA 0: 0000:00:06.0
   EFA 1: 0000:00:07.0
   GPU 2: 0000:10:04.0
   GPU 3: 0000:10:05.0
   EFA 2: 0000:10:06.0
   EFA 3: 0000:10:07.0

3. Determine topology:
   - Same domain/bus → Closer (same PCIe switch)
   - Different domain → Further (different NUMA node)

4. Build affinity matrix:
          EFA0  EFA1  EFA2  EFA3
   GPU0    10    20    50    60
   GPU1    20    10    60    50
   GPU2    50    60    10    20
   GPU3    60    50    20    10

   (Lower = closer, in arbitrary units)

5. Assign channels to NICs based on affinity
```

### PCI Topology Parsing

A PCI address `domain:bus:device.function` is parsed with
`sscanf(pci_path, "%hx:%hhx:%hhx.%hhx", ...)`. Affinity is modeled as a distance
where lower means closer (conceptual tiers used to rank NIC↔GPU proximity):

```
same domain+bus+device+function  → 0    (same device)
same domain+bus                  → 10   (likely same PCIe switch)
same domain                      → 50   (same NUMA, different PCIe tree)
different domain                 → 100  (different NUMA node)
```

### NUMA Topology Integration

A device's NUMA node is read from sysfs:

```bash
cat /sys/bus/pci/devices/<XXXX:XX:XX.X>/numa_node   # -1 if unavailable
```

**Example NUMA topology (p4d.24xlarge):**
```
NUMA Node 0:
  ├─ GPU 0 (0000:00:1b.0)
  ├─ GPU 1 (0000:00:1c.0)
  ├─ GPU 2 (0000:00:1d.0)
  ├─ GPU 3 (0000:00:1e.0)
  ├─ EFA 0 (0000:00:06.0)
  └─ EFA 1 (0000:00:07.0)

NUMA Node 1:
  ├─ GPU 4 (0000:10:1b.0)
  ├─ GPU 5 (0000:10:1c.0)
  ├─ GPU 6 (0000:10:1d.0)
  ├─ GPU 7 (0000:10:1e.0)
  ├─ EFA 2 (0000:10:06.0)
  └─ EFA 3 (0000:10:07.0)
```

## Interface Binding

### Device Selection Strategies

#### 1. Round-Robin (Default)

Round-robin rail assignment is the **built-in default** — a channel is mapped to
`channel_id % num_devices`. There is **no environment knob** for it:
`OFI_EFA_ROUND_ROBIN_HASH` does **not** exist (the code graph cannot show the
absence of a param, so it is recorded here). Round-robin is simply what the
plugin does when no stronger affinity constraint applies.

**Example with 8 channels, 4 EFAs:**
```
Channel 0 → EFA 0
Channel 1 → EFA 1
Channel 2 → EFA 2
Channel 3 → EFA 3
Channel 4 → EFA 0  (wrap around)
Channel 5 → EFA 1
Channel 6 → EFA 2
Channel 7 → EFA 3
```

#### 2. GPU Affinity-Based

Pick the NIC with the smallest `compute_pci_distance` to the GPU's PCI location
(from `cudaDeviceGetPCIBusId`):

```
best = argmin over dev of pci_distance(gpu_loc, nic_loc[dev])
```

**Example result:**
```
GPU 0 (NUMA 0) → EFA 0 (NUMA 0)  ← Local
GPU 1 (NUMA 0) → EFA 1 (NUMA 0)  ← Local
GPU 2 (NUMA 1) → EFA 2 (NUMA 1)  ← Local
GPU 3 (NUMA 1) → EFA 3 (NUMA 1)  ← Local
```

#### 3. NUMA-Aware Selection

Filter NICs to those on the GPU's NUMA node, then round-robin among them; if none
are local, fall back to plain round-robin over all devices:

```
local = [dev : numa(dev) == numa(gpu)]
return local ? local[gpu_id % len(local)] : gpu_id % num_devices
```

### Environment Variable Controls

#### `NCCL_SOCKET_IFNAME`

Controls which network interfaces NCCL uses:

```bash
# Use specific interface
NCCL_SOCKET_IFNAME=eth0

# Use multiple interfaces
NCCL_SOCKET_IFNAME=eth0,eth1

# Exclude interfaces (^prefix)
NCCL_SOCKET_IFNAME=^docker,lo

# For EFA (auto-detected, but can be explicit)
NCCL_SOCKET_IFNAME=rdma0,rdma1,rdma2,rdma3
```

**Plugin implementation:** with no `NCCL_SOCKET_IFNAME` set, all interfaces are
allowed. A leading `^` means exclusion (drop names matching the pattern);
otherwise it is inclusion (keep only names matching the pattern).

#### `NCCL_IB_HCA` (for EFA)

Historically for InfiniBand HCAs, also works for EFA:

```bash
# Select specific EFA adapters
NCCL_IB_HCA=mlx5_0:1,mlx5_1:1  # Format from IB days

# For EFA (usually auto-detected)
NCCL_IB_HCA=efa_0,efa_1
```

#### `NCCL_NET_GDR_LEVEL`

Controls GPU-NIC affinity level:

```bash
# Values:
# 0 - Disable GPUDirect (force through CPU)
# 1 - PHB (PCI Host Bridge) - same PCIe root complex
# 2 - PIX (PCIe switch)
# 3 - PXB (PCIe-to-PCI bridge)
# 4 - SYS (system-wide, any NIC)
# 5 - LOC (local, same NUMA node) [DEFAULT for EFA]

NCCL_NET_GDR_LEVEL=LOC  # Prefer local NUMA NICs
```

**Plugin uses this to filter devices:** keep only NICs whose computed affinity
level is at least as close as the requested `gdr_level` (default LOC = 5):

```
keep dev where compute_affinity_level(gpu_id, dev) <= gdr_level
```

### Dynamic Device Selection During Runtime

#### Per-Connection Selection

When NCCL creates a connection, the plugin creates an endpoint on the device NCCL
selected (NCCL passes `dev` based on its own topology detection):

> Source: `src/nccl_ofi_net.cpp` — `nccl_net_ofi_connect()`. Use `codegraph explore nccl_net_ofi_connect` for the current body.

**How NCCL chooses `dev`:**
1. Build topology graph from PCI paths
2. Determine optimal GPU-NIC mapping
3. For each channel, select NIC closest to GPU
4. Pass selected device ID to plugin

#### Multi-Rail Channel Distribution

NCCL's channel distribution (conceptual): for each channel, take the channel's
GPU, filter NICs by affinity to it, then load-balance the channel across the
available NICs:

```
for ch in channels:
    avail = filter_by_affinity(gpu_of(ch), nics)
    channel_to_nic[ch] = avail[ch % len(avail)]
```

## Topology Information Export

### Plugin Exposes Topology via Properties

NCCL uses the plugin's properties to build its internal topology: it calls
`devices()` then `getProperties(nic)` for each NIC (adding a node keyed by
`pciPath`/`speed`), combines those with GPU nodes discovered via
`cudaDeviceGetPCIBusId`, and computes shortest paths / affinities across the
combined graph.

### Custom Topology XML (Advanced)

Users can override auto-detection with XML:

```xml
<!-- /tmp/topo.xml -->
<system version="1">
  <cpu numaid="0" affinity="0,1,2,3" arch="x86_64" vendor="GenuineIntel">
    <pci busid="0000:00:06.0" class="0x020000" link_speed="100000 MT/s" link_width="16">
      <net name="efa_0" dev="0"/>
    </pci>
    <pci busid="0000:00:1b.0" class="0x030200" link_speed="16 GT/s" link_width="16">
      <gpu dev="0"/>
    </pci>
  </cpu>
  <cpu numaid="1" affinity="4,5,6,7" arch="x86_64" vendor="GenuineIntel">
    <pci busid="0000:10:06.0" class="0x020000" link_speed="100000 MT/s" link_width="16">
      <net name="efa_1" dev="1"/>
    </pci>
    <pci busid="0000:10:1b.0" class="0x030200" link_speed="16 GT/s" link_width="16">
      <gpu dev="1"/>
    </pci>
  </cpu>
</system>
```

```bash
NCCL_TOPO_FILE=/tmp/topo.xml ./nccl_app
```

## Debugging Topology

### View Detected Topology

```bash
# Enable NCCL topology debug output
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH,ENV ./nccl_app
```

**Output example:**
```
NCCL INFO NET/OFI Detected 4 devices
NCCL INFO NET/OFI Device 0: efa_0, PCI: 0000:00:06.0, NUMA: 0
NCCL INFO NET/OFI Device 1: efa_1, PCI: 0000:00:07.0, NUMA: 0
NCCL INFO NET/OFI Device 2: efa_2, PCI: 0000:10:06.0, NUMA: 1
NCCL INFO NET/OFI Device 3: efa_3, PCI: 0000:10:07.0, NUMA: 1

NCCL INFO Channel 00/08 :    0   1   2   3 [NET/0]
NCCL INFO Channel 01/08 :    0   1   2   3 [NET/1]
NCCL INFO Channel 02/08 :    4   5   6   7 [NET/2]
NCCL INFO Channel 03/08 :    4   5   6   7 [NET/3]
```

### System Topology Commands

```bash
# View PCI topology
lspci -tv

# View PCI devices with addresses
lspci -nn | grep -i efa

# View NUMA topology
numactl --hardware

# View GPU-NIC affinity (NVIDIA)
nvidia-smi topo -m

# Example output:
#       GPU0  GPU1  GPU2  GPU3  mlx5_0  mlx5_1
# GPU0   X     NV4   SYS   SYS   NODE    SYS
# GPU1  NV4    X     SYS   SYS   NODE    SYS
# GPU2  SYS   SYS    X     NV4   SYS     NODE
# GPU3  SYS   SYS   NV4    X     SYS     NODE
#
# Legend: X = Self, NV# = NVLink, NODE = same NUMA, SYS = cross-NUMA

# For EFA
ls -l /sys/class/infiniband/efa_*/device
# Shows PCI symlinks

# Read NUMA node
cat /sys/class/infiniband/efa_0/device/numa_node
```

### Verify Channel-to-NIC Mapping

```bash
# Dump NCCL graph
NCCL_GRAPH_DUMP_FILE=graph.txt ./nccl_app

# View graph.txt to see channel assignments
cat graph.txt
```

## Optimization Guidelines

### Best Practices

1. **Ensure NUMA Alignment**
   ```bash
   # GPUs and NICs on same NUMA node should be paired
   # Check with: nvidia-smi topo -m
   ```

2. **Use All Available NICs**
   ```bash
   # Don't filter unnecessarily
   # Let NCCL auto-detect all EFAs
   ```

3. **Pin Processes to NUMA Nodes**
   ```bash
   numactl --cpunodebind=0 --membind=0 -- ./app_gpu0_gpu1
   numactl --cpunodebind=1 --membind=1 -- ./app_gpu2_gpu3
   ```

4. **Verify PCI Bandwidth**
   ```bash
   # Ensure PCIe Gen4 x16
   lspci -vv -s 00:06.0 | grep -i lnk
   # Should show: Speed 16GT/s, Width x16
   ```

### Common Issues

**Issue 1: Cross-NUMA NIC assignment**
```
Symptom: 50% bandwidth loss
Cause: GPU on NUMA 0 using NIC on NUMA 1

Solution:
  NCCL_NET_GDR_LEVEL=LOC  # Force local NUMA
```

**Issue 2: NICs not detected**
```
Symptom: NCCL sees fewer NICs than available
Cause: Interface filtering or driver issues

Debug:
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
  Check: ls /sys/class/infiniband/
  Check: fi_info -p efa
```

**Issue 3: Suboptimal channel distribution**
```
Symptom: Uneven NIC utilization
Cause: NCCL's topology detection incorrect

Solution:
  - Provide custom NCCL_TOPO_FILE
  - Or ensure PCI paths are correctly reported
```

## Summary

### Topology Detection Flow

```
1. Plugin Init
   ├─ fi_getinfo() → Enumerate EFA devices
   └─ Count devices

2. For Each Device
   ├─ Get PCI path (from sysfs/libfabric)
   ├─ Get NUMA node (from sysfs)
   └─ Return via getProperties()

3. NCCL Builds Topology
   ├─ Combine NIC + GPU PCI paths
   ├─ Compute affinity matrix
   └─ Assign channels to NICs

4. Plugin Receives Device Selection
   ├─ connect(dev) called with chosen device
   └─ Create endpoint on selected NIC
```

### Key Files and Paths

```
Device Info:
  /sys/class/infiniband/efa_X/device → PCI device
  /sys/bus/pci/devices/XXXX:XX:XX.X/numa_node → NUMA node

Libfabric:
  fi_getinfo() → Device enumeration
  fi_info->domain_attr->name → Device name
  fi_info->nic->device_attr→pci → PCI info (if available)

NCCL:
  ncclNet->devices() → Count
  ncclNet->getProperties() → PCI path, speed, etc.
```

### Environment Variables Summary

| Variable | Purpose | Example |
|----------|---------|---------|
| `NCCL_SOCKET_IFNAME` | Filter interfaces | `^docker,lo` |
| `NCCL_IB_HCA` | Select specific adapters | `efa_0,efa_1` |
| `NCCL_NET_GDR_LEVEL` | GPU-NIC affinity level | `LOC` (local NUMA) |
| `NCCL_TOPO_FILE` | Custom topology | `/path/to/topo.xml` |
| `NCCL_DEBUG_SUBSYS` | Debug topology | `GRAPH,INIT,NET` |

The OFI plugin's topology detection and interface binding mechanisms ensure optimal NIC selection based on PCI/NUMA topology, enabling maximum performance in multi-NIC, multi-GPU systems.
