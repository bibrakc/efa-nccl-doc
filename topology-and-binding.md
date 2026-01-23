# Topology Detection and Interface Binding in OFI NCCL Plugin

## Overview

This document describes how the OFI NCCL plugin detects network devices (EFA adapters), discovers system topology, and binds to appropriate interfaces. These mechanisms are critical for:
- Multi-NIC systems (e.g., p4d.24xlarge with 4 EFAs)
- NUMA-aware communication
- GPU-NIC affinity optimization
- Optimal channel distribution

## Device Discovery

### Plugin Initialization and Device Enumeration

When NCCL initializes, it calls into the OFI plugin to discover available network devices.

#### `ncclNet->devices()` - Device Count

```c
ncclResult_t nccl_net_ofi_devices(int* ndev)
{
  // Query libfabric for available EFA devices
  struct fi_info* hints = fi_allocinfo();
  hints->fabric_attr->prov_name = strdup("efa");
  hints->ep_attr->type = FI_EP_RDM;

  struct fi_info* info_list;
  ret = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0, hints, &info_list);

  // Count devices
  int count = 0;
  for (struct fi_info* info = info_list; info; info = info->next) {
    if (info->domain_attr && info->domain_attr->name) {
      count++;
    }
  }

  // Store globally
  g_nccl_ofi_devices = count;
  g_nccl_ofi_info_list = info_list;

  *ndev = count;
  return ncclSuccess;
}
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

```c
ncclResult_t nccl_net_ofi_getProperties(int dev,
                                        ncclNetProperties_t* props)
{
  // Get the fi_info for this device
  struct fi_info* info = get_device_info(dev);

  // === 1. Device Name ===
  props->name = strdup("AWS EFA");

  // === 2. PCI Path (Critical for Topology) ===
  // Extract PCI address from device
  // Format: "0000:00:06.0" (domain:bus:device.function)
  char pci_path[256];
  ret = get_pci_path_from_device(info, pci_path, sizeof(pci_path));
  props->pciPath = strdup(pci_path);

  // === 3. GUID (Globally Unique ID) ===
  // Use device address or generate from PCI path
  props->guid = generate_guid(info);

  // === 4. Pointer Support ===
  // EFA supports both host and GPU memory
  props->ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;

  // === 5. Speed ===
  // Link speed in Mbps
  props->speed = 100000;  // 100 Gbps for EFA

  // === 6. Port ===
  props->port = 0;  // Single port per EFA

  // === 7. Max Communicators ===
  // How many concurrent connections this device supports
  props->maxComms = 1024;

  // === 8. Latency (optional) ===
  props->latency = 10.0;  // ~10 μs base latency

  // === 9. Max Receive Size ===
  props->maxRecvs = 1;  // Receive batching support

  return ncclSuccess;
}
```

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

Different methods depending on the system:

#### Method 1: From libfabric device attributes

```c
int get_pci_path_from_device(struct fi_info* info, char* pci_path, size_t len)
{
  // Some providers expose PCI info directly
  if (info->nic && info->nic->device_attr) {
    struct fi_pci_attr* pci = &info->nic->device_attr->pci;
    snprintf(pci_path, len, "%04x:%02x:%02x.%x",
             pci->domain_id,
             pci->bus_id,
             pci->device_id,
             pci->function_id);
    return 0;
  }

  // Fallback to other methods
  return get_pci_path_from_sysfs(info, pci_path, len);
}
```

#### Method 2: From sysfs

```c
int get_pci_path_from_sysfs(struct fi_info* info, char* pci_path, size_t len)
{
  // EFA devices appear in sysfs
  // /sys/class/infiniband/efa_X/device -> ../../devices/pci0000:00/0000:00:06.0

  char device_name[256];
  snprintf(device_name, sizeof(device_name),
           "/sys/class/infiniband/%s/device",
           info->domain_attr->name);

  char link_target[512];
  ssize_t link_len = readlink(device_name, link_target, sizeof(link_target) - 1);
  if (link_len < 0) {
    return -1;
  }
  link_target[link_len] = '\0';

  // Extract PCI path from symlink target
  // Example: "../../devices/pci0000:00/0000:00:06.0"
  char* pci_ptr = strrchr(link_target, '/');
  if (pci_ptr) {
    strncpy(pci_path, pci_ptr + 1, len);
    return 0;
  }

  return -1;
}
```

#### Method 3: Parse from device address

```c
int get_pci_path_from_address(struct fi_info* info, char* pci_path, size_t len)
{
  // EFA device names often encode PCI info
  // Example: "rdmap0s0" might be on bus 0

  // Query via ioctl or parse device properties
  // Implementation varies by provider

  return -1;
}
```

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

**`struct pci_location`** - PCI device location (internal helper structure for topology parsing):

```c
struct pci_location {
  uint16_t domain;
  uint8_t bus;
  uint8_t device;
  uint8_t function;
};

int parse_pci_path(const char* pci_path, struct pci_location* loc)
{
  int ret = sscanf(pci_path, "%hx:%hhx:%hhx.%hhx",
                   &loc->domain, &loc->bus,
                   &loc->device, &loc->function);
  return (ret == 4) ? 0 : -1;
}

int compute_pci_distance(struct pci_location* a, struct pci_location* b)
{
  // Same device
  if (a->domain == b->domain && a->bus == b->bus &&
      a->device == b->device && a->function == b->function) {
    return 0;
  }

  // Same bus (likely same PCIe switch)
  if (a->domain == b->domain && a->bus == b->bus) {
    return 10;
  }

  // Same domain (same NUMA node, different PCIe tree)
  if (a->domain == b->domain) {
    return 50;
  }

  // Different domain (different NUMA node)
  return 100;
}
```

### NUMA Topology Integration

```c
int get_numa_node_from_pci(const char* pci_path)
{
  // Read from sysfs
  char path[512];
  snprintf(path, sizeof(path),
           "/sys/bus/pci/devices/%s/numa_node",
           pci_path);

  FILE* f = fopen(path, "r");
  if (!f) {
    return -1;  // NUMA info not available
  }

  int numa_node;
  fscanf(f, "%d", &numa_node);
  fclose(f);

  return numa_node;
}
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

```c
int select_device_round_robin(int channel_id, int num_devices)
{
  // Simple round-robin distribution
  return channel_id % num_devices;
}
```

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

```c
int select_device_by_gpu_affinity(int gpu_id, int num_devices)
{
  // Get GPU's PCI location
  char gpu_pci_path[256];
  cudaDeviceGetPCIBusId(gpu_pci_path, sizeof(gpu_pci_path), gpu_id);

  struct pci_location gpu_loc;
  parse_pci_path(gpu_pci_path, &gpu_loc);

  // Find closest NIC
  int best_dev = 0;
  int min_distance = INT_MAX;

  for (int dev = 0; dev < num_devices; dev++) {
    struct pci_location nic_loc;
    parse_pci_path(device_pci_paths[dev], &nic_loc);

    int distance = compute_pci_distance(&gpu_loc, &nic_loc);
    if (distance < min_distance) {
      min_distance = distance;
      best_dev = dev;
    }
  }

  return best_dev;
}
```

**Example result:**
```
GPU 0 (NUMA 0) → EFA 0 (NUMA 0)  ← Local
GPU 1 (NUMA 0) → EFA 1 (NUMA 0)  ← Local
GPU 2 (NUMA 1) → EFA 2 (NUMA 1)  ← Local
GPU 3 (NUMA 1) → EFA 3 (NUMA 1)  ← Local
```

#### 3. NUMA-Aware Selection

```c
int select_device_numa_aware(int gpu_id, int num_devices)
{
  // Get GPU's NUMA node
  int gpu_numa = get_gpu_numa_node(gpu_id);

  // Filter devices on same NUMA node
  int local_devices[MAX_DEVICES];
  int num_local = 0;

  for (int dev = 0; dev < num_devices; dev++) {
    int nic_numa = get_numa_node_from_pci(device_pci_paths[dev]);
    if (nic_numa == gpu_numa) {
      local_devices[num_local++] = dev;
    }
  }

  if (num_local > 0) {
    // Round-robin among local devices
    return local_devices[gpu_id % num_local];
  } else {
    // No local devices, fall back to round-robin
    return gpu_id % num_devices;
  }
}
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

**Plugin implementation:**
```c
int is_interface_allowed(const char* ifname)
{
  const char* filter = getenv("NCCL_SOCKET_IFNAME");
  if (!filter) {
    return 1;  // No filter, allow all
  }

  // Parse filter
  // "^docker,lo" means exclude docker*, lo
  // "eth0,eth1" means include only eth0, eth1

  if (filter[0] == '^') {
    // Exclusion mode
    return !matches_pattern(ifname, filter + 1);
  } else {
    // Inclusion mode
    return matches_pattern(ifname, filter);
  }
}
```

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

**Plugin uses this to filter devices:**
```c
int filter_devices_by_gdr_level(int gpu_id, int* devices, int num_devices)
{
  int gdr_level = get_gdr_level_from_env();  // Default: LOC (5)

  int filtered[MAX_DEVICES];
  int num_filtered = 0;

  for (int i = 0; i < num_devices; i++) {
    int dev = devices[i];

    // Check affinity level
    int affinity = compute_affinity_level(gpu_id, dev);

    if (affinity <= gdr_level) {
      filtered[num_filtered++] = dev;
    }
  }

  // Copy back
  memcpy(devices, filtered, num_filtered * sizeof(int));
  return num_filtered;
}
```

### Dynamic Device Selection During Runtime

#### Per-Connection Selection

When NCCL creates a connection, the plugin selects a device:

```c
ncclResult_t nccl_net_ofi_connect(int dev, void* handle, void** sendComm)
{
  // 'dev' is passed by NCCL based on its topology detection

  // Get the fi_info for this device
  struct fi_info* info = g_nccl_ofi_devices[dev];

  // Create endpoint on selected device
  fi_endpoint(domain, info, &ep, NULL);

  // ... rest of connection setup ...
}
```

**How NCCL chooses `dev`:**
1. Build topology graph from PCI paths
2. Determine optimal GPU-NIC mapping
3. For each channel, select NIC closest to GPU
4. Pass selected device ID to plugin

#### Multi-Rail Channel Distribution

```c
// NCCL's channel distribution logic (conceptual)
void distribute_channels_to_nics(int num_channels, int num_nics)
{
  for (int ch = 0; ch < num_channels; ch++) {
    // Get GPU associated with this channel
    int gpu_id = get_channel_gpu(ch);

    // Get NICs with good affinity to this GPU
    int available_nics[MAX_NICS];
    int num_avail = filter_by_affinity(gpu_id, available_nics, num_nics);

    // Load balance across available NICs
    int nic_idx = ch % num_avail;
    int selected_nic = available_nics[nic_idx];

    // Assign channel to NIC
    channel_to_nic[ch] = selected_nic;
  }
}
```

## Topology Information Export

### Plugin Exposes Topology via Properties

NCCL uses the properties to build its internal topology:

```c
// NCCL internal (conceptual)
void build_network_topology()
{
  int num_nics;
  ncclNet->devices(&num_nics);

  for (int nic = 0; nic < num_nics; nic++) {
    ncclNetProperties_t props;
    ncclNet->getProperties(nic, &props);

    // Add to topology graph
    add_network_node(nic, props.pciPath, props.speed);
  }

  // Combine with GPU topology
  for (int gpu = 0; gpu < num_gpus; gpu++) {
    char pci_path[256];
    cudaDeviceGetPCIBusId(pci_path, sizeof(pci_path), gpu);
    add_gpu_node(gpu, pci_path);
  }

  // Compute shortest paths, affinities, etc.
  compute_topology_graph();
}
```

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
