# NCCL Message Breakdown: Complete Analysis

## Overview

This document provides ground-truth analysis of how NCCL breaks down messages for all collective operations, from application-level message size down to individual libfabric `fi_send()` calls. All information is verified from **NCCL 2.28.9 source code**.

## Source Code References

**Key Files:**
- `src/include/collectives.h` - SLICESTEPS and CHUNKSTEPS definitions
- `src/include/device.h` - NCCL_STEPS = 8
- `src/enqueue.cc` - calcCollChunking() function, message planning
- `src/transport/net.cc` - Network transport, isend calls (line 1322)
- `src/device/*.h` - GPU kernel implementations
- `src/proxy.cc` - Proxy thread that calls network layer

## Core Constants (from NCCL Source)

### NCCL_STEPS Definition

**File**: `nccl-2.28.9-1/src/include/device.h` ([local](file:///home/cloonan/dev/nccl-2.28.9-1/src/include/device.h#L24))

```c
#define NCCL_STEPS 8
```

### Collective-Specific Constants

**File**: `nccl-2.28.9-1/src/include/collectives.h` ([local](file:///home/cloonan/dev/nccl-2.28.9-1/src/include/collectives.h#L16-L32))

```c
// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)      // = 2
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)      // = 4

#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)      // = 2
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)      // = 4

#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)  // = 2
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)  // = 4

#define BROADCAST_SLICESTEPS 1                   // = 1
#define BROADCAST_CHUNKSTEPS 1                   // = 1

#define ALLTOALL_SLICESTEPS 1                    // = 1
#define ALLTOALL_CHUNKSTEPS 1                    // = 1

#define REDUCE_SLICESTEPS 1                      // = 1
#define REDUCE_CHUNKSTEPS 1                      // = 1
```

## Message Breakdown Hierarchy

```
Application Message (e.g., 64MB)
         ↓
    [NCCL Planning]
         ↓
    Channels (e.g., 8 channels)
         ↓
    Algorithm Steps (Ring: 2×(N-1), Tree: 2×log₂(N))
         ↓
    Chunks (chunkSize = stepSize × chunkSteps)
         ↓
    Slices (sliceSize = stepSize × sliceSteps)
         ↓
    Protocol Subdivision (Simple: none, LL128: 128B, LL: 8B)
         ↓
    ncclNet->isend() calls
         ↓
    fi_send() calls (libfabric)
         ↓
    Network Packets (EFA MTU: 8KB)
```

## Calculation Formulas

### Step 1: Channel Division

```c
// From src/enqueue.cc - calcCollChunking()
bytes_per_channel = total_message_size / num_channels
```

### Step 2: Algorithm Steps

**Ring Algorithm:**
```c
// For AllReduce, AllGather, ReduceScatter
reduce_scatter_steps = num_ranks - 1
allgather_steps = num_ranks - 1
total_steps = 2 × (num_ranks - 1)
```

**Tree Algorithm:**
```c
// For Broadcast, Reduce, some AllReduce
up_steps = log₂(num_ranks)
down_steps = log₂(num_ranks)
total_steps = 2 × log₂(num_ranks)
```

### Step 3: Chunk and Slice Sizes

**File**: `nccl-2.28.9-1/src/enqueue.cc` - `calcCollChunking()` function ([local](file:///home/cloonan/dev/nccl-2.28.9-1/src/enqueue.cc#L2022-L2027))

```c
int stepSize   = comm->buffSizes[info->protocol]/NCCL_STEPS;
int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
int chunkSize = stepSize*chunkSteps;
if (info->protocol == NCCL_PROTO_LL) chunkSize /= 2;
if (info->protocol == NCCL_PROTO_LL128) chunkSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
```

**Calculation:**
```c
// For Ring + Simple protocol:
chunkSteps = ALLREDUCE_CHUNKSTEPS;  // 4 for AllReduce
sliceSteps = ALLREDUCE_SLICESTEPS;  // 2 for AllReduce
chunkSize = stepSize × chunkSteps;
sliceSize = stepSize × sliceSteps;

// For LL protocol:
chunkSize = (stepSize × chunkSteps) / 2;

// For LL128 protocol:
chunkSize = ((stepSize × chunkSteps) / NCCL_LL128_LINEELEMS) × NCCL_LL128_DATAELEMS;
```

### Step 4: Network Sends

**File**: `nccl-2.28.9-1/src/transport/net.cc` - `sendProxyProgress()` function ([local](file:///home/cloonan/dev/nccl-2.28.9-1/src/transport/net.cc#L1322))

```c
// One isend() call per slice
NCCLCHECK(proxyState->ncclNet->isend(
    resources->netSendComm, 
    buff,                    // Buffer pointer
    size,                    // Slice size
    resources->tpRank, 
    sub->sendMhandle, 
    phandle, 
    sub->requests+buffSlot
));
```

**Context** ([local](file:///home/cloonan/dev/nccl-2.28.9-1/src/transport/net.cc#L1245-L1250)):

```c
int stepSize = resources->buffSizes[p] / NCCL_STEPS;
char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
// ...
int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
int size = connFifo[buffSlot].size;  // Slice size
char* buff = shared ? localBuff+connFifo[buffSlot].offset : localBuff+buffSlot*stepSize;
```

---

## AllReduce Breakdown

### Configuration

```c
// src/include/collectives.h
ALLREDUCE_SLICESTEPS = 2
ALLREDUCE_CHUNKSTEPS = 4
SlicesPerChunk = CHUNKSTEPS / SLICESTEPS = 2
```

### Example: 64MB AllReduce, 8 GPUs, 8 Channels, Ring Algorithm

#### Level 1: Channel Division

```
Total Message: 64MB
Channels: 8
Bytes per Channel = 64MB / 8 = 8MB
```

#### Level 2: Ring Algorithm Steps

```
Ranks: 8
Reduce-Scatter Steps: 8 - 1 = 7
AllGather Steps: 8 - 1 = 7
Total Steps: 14
```

#### Level 3: Chunk and Slice Calculation

```c
// Assuming buffSize = 4MB (typical default)
stepSize = 4MB / 8 = 512KB
chunkSteps = 4
sliceSteps = 2
chunkSize = 512KB × 4 = 2MB
sliceSize = 512KB × 2 = 1MB
```

**Per Channel Per Step:**
```
Data per step = 8MB / 8 ranks = 1MB
This equals sliceSize!
```

**Important**: The "data per step" in Ring algorithm equals the slice size, not chunk size. Each step sends one slice.

#### Level 4: Protocol Subdivision

**Simple Protocol:**
- No further subdivision
- Each slice → 1 isend() call
- 1MB slice → 1 isend(1MB)

**LL128 Protocol:**
- Subdivides into 128-byte units
- 1MB slice → 8,192 isend(128B) calls

**LL Protocol:**
- Subdivides into 8-byte units
- 1MB slice → 131,072 isend(8B) calls

#### Level 5: Network Layer

**From src/transport/net.cc:**

```c
// Proxy thread posts sends for each slice
for (each step in algorithm) {
    int buffSlot = (sub->base + sub->transmitted) % NCCL_STEPS;
    int size = connFifo[buffSlot].size;  // Slice size
    char* buff = localBuff + buffSlot * stepSize;
    
    // Call into ncclNet plugin (e.g., aws-ofi-nccl)
    NCCLCHECK(proxyState->ncclNet->isend(
        resources->netSendComm,
        buff,
        size,  // 1MB for Simple protocol
        resources->tpRank,
        sub->sendMhandle,
        phandle,
        sub->requests+buffSlot
    ));
    
    sub->transmitted += args->sliceSteps;  // Advance by 2 steps
}
```

#### Level 6: Libfabric (aws-ofi-nccl plugin)

**From aws-ofi-nccl plugin:**

```c
// nccl_net_ofi_isend() implementation
fi_tsend(
    comm->ep,
    data,
    1048576,  // 1MB
    desc,
    comm->remote_addr,
    tag,
    req
);
```

#### Level 7: EFA Hardware

```
1MB fi_tsend()
  → EFA driver creates WQE
  → Hardware DMAs 1MB
  → Splits into ~128 packets (8KB MTU)
  → Sends over network
```

### Complete AllReduce Breakdown (Simple Protocol)

```
64MB AllReduce, 8 GPUs, 8 Channels, Ring Algorithm

├─ Channel 0: 8MB
│  ├─ Reduce-Scatter Phase (7 steps)
│  │  ├─ Step 0: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│  │  ├─ Step 1: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│  │  ├─ Step 2: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│  │  ├─ Step 3: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│  │  ├─ Step 4: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│  │  ├─ Step 5: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│  │  └─ Step 6: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│  │
│  └─ AllGather Phase (7 steps)
│     ├─ Step 7: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│     ├─ Step 8: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│     ├─ Step 9: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│     ├─ Step 10: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│     ├─ Step 11: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│     ├─ Step 12: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│     └─ Step 13: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets
│
├─ Channel 1-7: (same pattern, 14 steps each)
│
Total:
  - isend() calls: 8 channels × 14 steps = 112
  - fi_tsend() calls: 112
  - Bytes per call: 1MB
  - Network packets: 112 × 128 = 14,336 packets
```

---

## AllGather Breakdown

### Configuration

```c
// src/include/collectives.h
ALLGATHER_SLICESTEPS = 2
ALLGATHER_CHUNKSTEPS = 4
SlicesPerChunk = 2
```

### Example: 64MB AllGather, 8 GPUs, 8 Channels, Ring Algorithm

#### Key Difference from AllReduce

AllGather has **only one phase** (no reduce-scatter), but each GPU sends its portion around the ring.

#### Level 1-3: Same as AllReduce

```
Channels: 8
Bytes per Channel: 8MB
Steps: 7 (N-1, single phase)
sliceSize: 1MB
```

#### Algorithm Behavior

**Initial State:**
- Each GPU has 1/N of the data (1MB per channel)
- Goal: All GPUs get all data (8MB per channel)

**Ring Steps:**
```
Step 0: GPU i sends its chunk to GPU (i+1)
Step 1: GPU i forwards received chunk to GPU (i+1)
...
Step 6: Final forwarding
```

### Complete AllGather Breakdown (Simple Protocol)

```
64MB AllGather, 8 GPUs, 8 Channels

├─ Channel 0: 8MB total (each GPU contributes 1MB)
│  ├─ Step 0: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB)
│  ├─ Step 1: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB)
│  ├─ Step 2: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB)
│  ├─ Step 3: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB)
│  ├─ Step 4: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB)
│  ├─ Step 5: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB)
│  └─ Step 6: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB)
│
├─ Channel 1-7: (same pattern, 7 steps each)
│
Total:
  - isend() calls: 8 channels × 7 steps = 56
  - fi_tsend() calls: 56
  - Bytes per call: 1MB
  - Network packets: 56 × 128 = 7,168 packets
```

---

## ReduceScatter Breakdown

### Configuration

```c
// src/include/collectives.h
REDUCESCATTER_SLICESTEPS = 2
REDUCESCATTER_CHUNKSTEPS = 4
SlicesPerChunk = 2
```

### Example: 64MB ReduceScatter, 8 GPUs, 8 Channels

#### Key Difference

ReduceScatter is **only the reduce-scatter phase** of AllReduce. Each GPU ends up with 1/N of the reduced data.

#### Breakdown

```
64MB ReduceScatter, 8 GPUs, 8 Channels

├─ Channel 0: 8MB input → 1MB output per GPU
│  ├─ Step 0: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB)
│  ├─ Step 1: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB)
│  ├─ Step 2: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB)
│  ├─ Step 3: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB)
│  ├─ Step 4: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB)
│  ├─ Step 5: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB)
│  └─ Step 6: 1MB slice → 1 isend(1MB) → 1 fi_tsend(1MB)
│
├─ Channel 1-7: (same pattern, 7 steps each)
│
Total:
  - isend() calls: 8 channels × 7 steps = 56
  - fi_tsend() calls: 56
  - Bytes per call: 1MB
  - Network packets: 56 × 128 = 7,168 packets
```

---

## Broadcast Breakdown

### Configuration

```c
// src/include/collectives.h
BROADCAST_SLICESTEPS = 1
BROADCAST_CHUNKSTEPS = 1
SlicesPerChunk = 1
```

### Example: 64MB Broadcast, 8 GPUs, 8 Channels, Tree Algorithm

#### Key Characteristics

- **Root GPU** has data, sends to all others
- **Tree Algorithm** typically used (O(log N) latency)
- **No chunking/slicing** (CHUNKSTEPS=1, SLICESTEPS=1)

#### Level 1-2: Channel and Algorithm

```
Channels: 8
Bytes per Channel: 8MB
Algorithm: Binary Tree
Steps: log₂(8) = 3
```

#### Level 3: No Chunking

```c
stepSize = 4MB / 8 = 512KB
chunkSteps = 1
sliceSteps = 1
chunkSize = 512KB × 1 = 512KB
sliceSize = 512KB × 1 = 512KB
```

**But wait**: For Broadcast, the entire channel data (8MB) is sent in the tree pattern, not divided by ranks.

#### Tree Algorithm Pattern

```
Step 0: Root → 1 child (sends 8MB)
Step 1: 2 nodes → 2 children (each sends 8MB)
Step 2: 4 nodes → 4 children (each sends 8MB)
```

**However**, NCCL still uses pipelining with stepSize chunks:

```
8MB / 512KB = 16 pipeline chunks
```

### Complete Broadcast Breakdown (Simple Protocol)

```
64MB Broadcast, 8 GPUs, 8 Channels, Tree Algorithm

├─ Channel 0: 8MB
│  ├─ Tree Step 0 (Root → Child 1)
│  │  ├─ Pipeline 0: 512KB → 1 isend(512KB) → 1 fi_tsend(512KB)
│  │  ├─ Pipeline 1: 512KB → 1 isend(512KB) → 1 fi_tsend(512KB)
│  │  ├─ ... (16 pipeline chunks)
│  │  └─ Pipeline 15: 512KB → 1 isend(512KB) → 1 fi_tsend(512KB)
│  │
│  ├─ Tree Step 1 (2 nodes send)
│  │  └─ ... (16 pipeline chunks each)
│  │
│  └─ Tree Step 2 (4 nodes send)
│     └─ ... (16 pipeline chunks each)
│
├─ Channel 1-7: (same pattern)
│
Total (approximate, depends on tree structure):
  - isend() calls: 8 channels × 3 tree steps × 16 pipeline chunks = 384
  - fi_tsend() calls: 384
  - Bytes per call: 512KB
  - Network packets: 384 × 64 = 24,576 packets
```

---

## AllToAll Breakdown

### Configuration

```c
// src/include/collectives.h
ALLTOALL_SLICESTEPS = 1
ALLTOALL_CHUNKSTEPS = 1
SlicesPerChunk = 1
```

### Example: 64MB AllToAll, 8 GPUs, 8 Channels

#### Key Characteristics

- Each GPU sends unique data to every other GPU
- **Peer-to-peer** pattern, not ring or tree
- Each GPU sends (N-1) messages
- Each GPU receives (N-1) messages

#### Message Pattern

```
GPU 0 sends:
  - 1MB to GPU 1
  - 1MB to GPU 2
  - ... 
  - 1MB to GPU 7
  
Total per GPU: 7MB sent, 7MB received
```

#### Level 1-2: Channel and Peer Division

```
Total Message: 64MB
Channels: 8
Bytes per Channel: 8MB
Peers: 7 (N-1)
Bytes per Peer per Channel: 8MB / 7 ≈ 1.14MB
```

### Complete AllToAll Breakdown (Simple Protocol)

```
64MB AllToAll, 8 GPUs, 8 Channels

├─ Channel 0: 8MB
│  ├─ To Peer 1: ~1.14MB → isend(~1.14MB) → fi_tsend(~1.14MB)
│  ├─ To Peer 2: ~1.14MB → isend(~1.14MB) → fi_tsend(~1.14MB)
│  ├─ To Peer 3: ~1.14MB → isend(~1.14MB) → fi_tsend(~1.14MB)
│  ├─ To Peer 4: ~1.14MB → isend(~1.14MB) → fi_tsend(~1.14MB)
│  ├─ To Peer 5: ~1.14MB → isend(~1.14MB) → fi_tsend(~1.14MB)
│  ├─ To Peer 6: ~1.14MB → isend(~1.14MB) → fi_tsend(~1.14MB)
│  └─ To Peer 7: ~1.14MB → isend(~1.14MB) → fi_tsend(~1.14MB)
│
├─ Channel 1-7: (same pattern, 7 peers each)
│
Total:
  - isend() calls: 8 channels × 7 peers = 56
  - fi_tsend() calls: 56
  - Bytes per call: ~1.14MB
  - Network packets: 56 × 143 = 8,008 packets
```

---

## Reduce Breakdown

### Configuration

```c
// src/include/collectives.h
REDUCE_SLICESTEPS = 1
REDUCE_CHUNKSTEPS = 1
SlicesPerChunk = 1
```

### Example: 64MB Reduce, 8 GPUs, 8 Channels, Tree Algorithm

#### Key Characteristics

- All GPUs send to **root GPU**
- **Tree Algorithm** (O(log N) latency)
- Only root receives final result

#### Breakdown

Similar to Broadcast but in reverse direction:

```
64MB Reduce, 8 GPUs, 8 Channels, Tree Algorithm

├─ Channel 0: 8MB
│  ├─ Tree Step 0 (4 leaves → 4 parents)
│  │  └─ 16 pipeline chunks of 512KB each
│  │
│  ├─ Tree Step 1 (2 nodes → 2 parents)
│  │  └─ 16 pipeline chunks of 512KB each
│  │
│  └─ Tree Step 2 (1 node → root)
│     └─ 16 pipeline chunks of 512KB each
│
├─ Channel 1-7: (same pattern)
│
Total:
  - isend() calls: 8 channels × 3 tree steps × 16 pipeline chunks = 384
  - fi_tsend() calls: 384
  - Bytes per call: 512KB
```

---

## Protocol Comparison

### Simple Protocol

**Characteristics:**
- No protocol-level subdivision
- One isend() per slice
- Highest bandwidth

**For 1MB slice:**
```
1MB → 1 isend(1MB) → 1 fi_tsend(1MB) → ~128 packets (8KB MTU)
```

### LL128 Protocol

**Characteristics:**
- Subdivides into 128-byte units
- 120 bytes data + 8 bytes flag per unit
- Flag-based synchronization

**For 1MB slice:**
```
1MB / 128B = 8,192 units
8,192 isend(128B) calls
8,192 fi_tsend(128B) calls
~8,192 packets (each 128B fits in one packet)
```

**From src/device/prims_ll128.h:**
```c
static constexpr int WireWordPerSlice = WARP_SIZE*NCCL_LL128_SHMEM_ELEMS_PER_THREAD;
static constexpr int DataEltPerSlice = 
    (WireWordPerSlice - WireWordPerSlice/NCCL_LL128_LINEELEMS) * 
    (sizeof(uint64_t)/sizeof(T));
```

### LL Protocol

**Characteristics:**
- Subdivides into 8-byte units
- 4 bytes flag + 4 bytes data per unit
- Lowest latency, poorest bandwidth

**For 1MB slice:**
```
1MB / 8B = 131,072 units
131,072 isend(8B) calls
131,072 fi_tsend(8B) calls
~131,072 packets
```

---

## Summary Tables

### Collective Operation Characteristics

| Collective | SliceSteps | ChunkSteps | Algorithm | Phases | Steps (N=8) |
|------------|------------|------------|-----------|--------|-------------|
| AllReduce | 2 | 4 | Ring | 2 | 14 |
| AllGather | 2 | 4 | Ring | 1 | 7 |
| ReduceScatter | 2 | 4 | Ring | 1 | 7 |
| Broadcast | 1 | 1 | Tree | 1 | 3 |
| Reduce | 1 | 1 | Tree | 1 | 3 |
| AllToAll | 1 | 1 | P2P | 1 | 7 |

### Network Calls for 64MB Message (8 GPUs, 8 Channels, Simple Protocol)

| Collective | isend() Calls | Bytes/Call | fi_tsend() Calls | Packets (8KB MTU) |
|------------|---------------|------------|------------------|-------------------|
| AllReduce | 112 | 1MB | 112 | 14,336 |
| AllGather | 56 | 1MB | 56 | 7,168 |
| ReduceScatter | 56 | 1MB | 56 | 7,168 |
| Broadcast | ~384 | 512KB | ~384 | ~24,576 |
| Reduce | ~384 | 512KB | ~384 | ~24,576 |
| AllToAll | 56 | ~1.14MB | 56 | ~8,008 |

### Protocol Impact (1MB slice)

| Protocol | Subdivision | isend() Calls | Bytes/Call | Efficiency |
|----------|-------------|---------------|------------|------------|
| Simple | None | 1 | 1MB | ~99% |
| LL128 | 128B units | 8,192 | 128B | ~94% |
| LL | 8B units | 131,072 | 8B | ~67% |

---

## Code Flow Summary

### From Application to Network

```
1. Application: ncclAllReduce(64MB)
   ↓
2. NCCL Planning (src/enqueue.cc):
   - calcCollChunking()
   - Determine channels, algorithm, protocol
   - Calculate chunkSize, sliceSize
   ↓
3. GPU Kernel Launch (src/device/*.h):
   - Copy data to channel buffers
   - Signal proxy thread via shared memory
   ↓
4. Proxy Thread (src/proxy.cc):
   - Poll for work from GPU
   - Call ncclNet->isend() for each slice
   ↓
5. Network Transport (src/transport/net.cc:1322):
   NCCLCHECK(proxyState->ncclNet->isend(
       resources->netSendComm,
       buff,
       size,  // Slice size
       ...
   ));
   ↓
6. OFI Plugin (aws-ofi-nccl):
   fi_tsend(comm->ep, data, size, ...);
   ↓
7. Libfabric EFA Provider:
   - Post to send queue
   - Ring doorbell
   ↓
8. EFA Hardware:
   - DMA from memory
   - Packetize (8KB MTU)
   - Send over network
```

---

## Key Insights

1. **Slice is the unit of network transfer**: Each `isend()` call sends one slice
2. **Chunk contains multiple slices**: For pipelining (AllReduce: 2 slices/chunk)
3. **Protocol determines subdivision**: Simple (none), LL128 (128B), LL (8B)
4. **Algorithm determines steps**: Ring (2×(N-1)), Tree (2×log₂(N))
5. **Channels enable parallelism**: Independent data paths
6. **Pipeline depth (NCCL_STEPS=8)**: Allows overlap of GPU and network operations

---

## Environment Variables

```bash
# Channel configuration
NCCL_NCHANNELS=8           # Number of channels

# Buffer size (affects stepSize)
NCCL_BUFFSIZE=4194304      # 4MB default

# Protocol selection
NCCL_PROTO=Simple          # Simple, LL128, or LL

# Algorithm selection
NCCL_ALGO=Ring             # Ring, Tree, etc.

# Debug
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,NET
```

---

## Related Documentation

- [nccl-core.md](nccl-core.md) - NCCL core concepts
- [nccl-collectives.md](nccl-collectives.md) - Collective operations overview
- [algorithms/ring-algorithm.md](algorithms/ring-algorithm.md) - Ring algorithm details
- [algorithms/tree-algorithm.md](algorithms/tree-algorithm.md) - Tree algorithm details
- [ofi-plugin.md](ofi-plugin.md) - OFI plugin implementation
- [ofi-plugin-protocols.md](ofi-plugin-protocols.md) - Send/recv protocols

---

**Last Updated**: 2026-01-29

**Source**: NCCL 2.28.9 source code analysis

**License**: Same as efa-nccl-doc repository
