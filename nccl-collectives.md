# NCCL Collective Operations and Algorithms

## Collective Operations

NCCL implements standard MPI-style collective operations optimized for GPUs.

### AllReduce

**Operation**: All ranks contribute data, reduction is performed, result distributed to all ranks.

```
Rank 0: [A0] ─┐
Rank 1: [A1] ─┼─→ Reduce (SUM) ─→ [A0+A1+A2+A3] → All Ranks
Rank 2: [A2] ─┤
Rank 3: [A3] ─┘
```

**API:**

**`ncclAllReduce()`** - AllReduce collective operation ([src/nccl.h.in](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in) - NCCL external API):

```c
ncclAllReduce(const void* sendbuff, void* recvbuff,
              size_t count, ncclDataType_t datatype,
              ncclRedOp_t op, ncclComm_t comm,
              cudaStream_t stream);
```

**Reduction Operations:**
- `ncclSum`: Element-wise sum
- `ncclProd`: Element-wise product
- `ncclMax`: Element-wise maximum
- `ncclMin`: Element-wise minimum
- `ncclAvg`: Element-wise average

**Use Cases:**
- Gradient aggregation in distributed training
- Most common NCCL operation
- Typically 90%+ of communication in deep learning

### Broadcast

**Operation**: One rank sends data to all other ranks.

```
Rank 0 (root): [Data] ─→ All Ranks
Rank 1: receives [Data]
Rank 2: receives [Data]
Rank 3: receives [Data]
```

**API:**

**`ncclBroadcast()`** - Broadcast collective operation ([src/nccl.h.in](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in) - NCCL external API):

```c
ncclBroadcast(const void* sendbuff, void* recvbuff,
              size_t count, ncclDataType_t datatype,
              int root, ncclComm_t comm,
              cudaStream_t stream);
```

**Use Cases:**
- Broadcasting model parameters
- Distributing configuration data
- Initial weight distribution

### Reduce

**Operation**: All ranks contribute data, reduction performed, result on root only.

```
Rank 0: [A0] ─┐
Rank 1: [A1] ─┼─→ Reduce (SUM) ─→ Rank 2: [A0+A1+A2+A3]
Rank 2: [A2] ─┤                    Others: N/A
Rank 3: [A3] ─┘
```

**API:**

**`ncclReduce()`** - Reduce collective operation ([src/nccl.h.in](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in) - NCCL external API):

```c
ncclReduce(const void* sendbuff, void* recvbuff,
           size_t count, ncclDataType_t datatype,
           ncclRedOp_t op, int root, ncclComm_t comm,
           cudaStream_t stream);
```

### AllGather

**Operation**: Each rank contributes data, all ranks receive concatenated result.

```
Rank 0: [A0] ─┐
Rank 1: [A1] ─┼─→ Gather ─→ All Ranks: [A0|A1|A2|A3]
Rank 2: [A2] ─┤
Rank 3: [A3] ─┘
```

**API:**

**`ncclAllGather()`** - AllGather collective operation ([src/nccl.h.in](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in) - NCCL external API):

```c
ncclAllGather(const void* sendbuff, void* recvbuff,
              size_t sendcount, ncclDataType_t datatype,
              ncclComm_t comm, cudaStream_t stream);
```

**Use Cases:**
- Gathering embeddings from all ranks
- Collecting distributed results
- Feature gathering in recommender systems

### ReduceScatter

**Operation**: All ranks contribute data, reduction performed, result scattered.

```
Rank 0: [A0] ─┐                    Rank 0: [sum(A0_chunk0)]
Rank 1: [A1] ─┼─→ Reduce+Scatter → Rank 1: [sum(A1_chunk1)]
Rank 2: [A2] ─┤                    Rank 2: [sum(A2_chunk2)]
Rank 3: [A3] ─┘                    Rank 3: [sum(A3_chunk3)]
```

**API:**

**`ncclReduceScatter()`** - ReduceScatter collective operation ([src/nccl.h.in](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in) - NCCL external API):

```c
ncclReduceScatter(const void* sendbuff, void* recvbuff,
                  size_t recvcount, ncclDataType_t datatype,
                  ncclRedOp_t op, ncclComm_t comm,
                  cudaStream_t stream);
```

**Use Cases:**
- First step in data-parallel training
- Memory-efficient gradient aggregation
- Often paired with AllGather

### Send/Recv

**Operation**: Point-to-point communication between two ranks.

**`ncclSend()` / `ncclRecv()`** - Point-to-point send/receive operations ([src/nccl.h.in](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in) - NCCL external API):

```c
// Sender
ncclSend(const void* sendbuff, size_t count,
         ncclDataType_t datatype, int peer,
         ncclComm_t comm, cudaStream_t stream);

// Receiver
ncclRecv(void* recvbuff, size_t count,
         ncclDataType_t datatype, int peer,
         ncclComm_t comm, cudaStream_t stream);
```

**Use Cases:**
- Pipeline parallelism
- Custom communication patterns
- Tensor parallelism

## Algorithms

NCCL implements multiple algorithms for each collective. The choice depends on message size, topology, and rank count.

### Ring Algorithm

**Structure**: Ranks arranged in a logical ring.

```
Rank 0 ←→ Rank 1 ←→ Rank 2 ←→ Rank 3 ←→ Rank 0
  ↑                                        ↓
  └────────────────────────────────────────┘
```

**AllReduce via Ring:**

```
Phase 1: Reduce-Scatter
Step 0: Rank 0 → Rank 1: chunk 0
        Rank 1 → Rank 2: chunk 1
        Rank 2 → Rank 3: chunk 2
        Rank 3 → Rank 0: chunk 3

Step 1: Rank 0 → Rank 1: chunk 3 (accumulated)
        Rank 1 → Rank 2: chunk 0 (accumulated)
        Rank 2 → Rank 3: chunk 1 (accumulated)
        Rank 3 → Rank 0: chunk 2 (accumulated)

... (N-1 steps total)

Phase 2: AllGather
Similar N-1 steps to distribute fully reduced chunks
```

**Characteristics:**
- **Bandwidth**: Optimal - O(N) steps, but utilizes all links
- **Latency**: High - O(N) steps
- **Message Size**: Best for large messages
- **Scalability**: Good for many ranks
- **Algorithm Bandwidth**: `2 * (N-1) / N * LinkBandwidth`
  - Example: 8 ranks ≈ 1.75 * LinkBandwidth

**When Used:**
- Large messages (> ~2 MB)
- Many ranks (> 8)
- Balanced topology

### Tree Algorithm

**Structure**: Ranks arranged in a binary tree.

```
         Rank 0
        /      \
    Rank 1    Rank 2
      /         \
  Rank 3       Rank 4
```

**AllReduce via Tree:**

```
Phase 1: Reduce (bottom-up)
Rank 3 → Rank 1
Rank 4 → Rank 2
Rank 1 → Rank 0
Rank 2 → Rank 0
(Rank 0 has full reduction)

Phase 2: Broadcast (top-down)
Rank 0 → Rank 1, Rank 2
Rank 1 → Rank 3
Rank 2 → Rank 4
```

**Characteristics:**
- **Bandwidth**: Lower - some links more utilized
- **Latency**: Low - O(log N) steps
- **Message Size**: Best for small messages
- **Scalability**: Excellent latency scaling
- **Root Bottleneck**: Root node is bandwidth bottleneck

**When Used:**
- Small messages (< ~1 MB)
- Latency-sensitive operations
- Fewer ranks (< 8)

### Double Binary Tree

**Enhancement**: Two trees with different roots to balance load.

```
Tree 1 (root: Rank 0)         Tree 2 (root: Rank N/2)
         Rank 0                       Rank 4
        /      \                     /      \
    Rank 1    Rank 2            Rank 5    Rank 6
      /         \                 /         \
  Rank 3       Rank 4         Rank 7       Rank 0
```

**Benefits:**
- Doubles effective bandwidth
- No single root bottleneck
- Better for medium-sized messages

### CollNet Algorithm

**Structure**: Offload to switch/NIC with collective acceleration.

```
Rank 0 ─┐
Rank 1 ─┼→ [Smart Switch/NIC] → All Ranks
Rank 2 ─┤   (In-network reduction)
Rank 3 ─┘
```

**Characteristics:**
- **Latency**: Very low - single network hop
- **Bandwidth**: Depends on switch capability
- **Requirements**: Special hardware (InfiniBand SHARP, etc.)
- **Not available on EFA**: EFA doesn't support in-network collectives

### NVLS (NVLink Sharp)

**Structure**: Direct GPU-to-GPU communication via NVLink for intra-node.

**Characteristics:**
- **Scope**: Intra-node only
- **Bandwidth**: Extremely high (NVLink speeds)
- **Latency**: Very low
- **Requirements**: NVLink connectivity

**Not relevant for inter-node EFA communication**

## Protocols

NCCL implements different protocols optimizing for different scenarios.

### Simple Protocol

**Mechanism**: Standard data transfer with acknowledgments.

```
Sender                    Receiver
  │                          │
  ├─── Send Data ───────────→│
  │                          │
  │←─── ACK ─────────────────┤
  │                          │
```

**Characteristics:**
- Full payload per transfer
- Requires acknowledgment/completion
- Best for large messages
- Higher bandwidth, moderate latency

**Buffer Size**: Configurable via `NCCL_BUFFSIZE` (default 4 MB)

### LL (Low Latency) Protocol

**Mechanism**: Flag-based synchronization, no ACKs needed.

```
Sender                    Receiver
  │                          │
  ├─── Data + Flag ─────────→│
  │    (8 bytes data)         │ (polls flag)
  │                          │
```

**Characteristics:**
- 8-byte payload per transfer
- Flag embedded in data stream
- No explicit ACK needed
- Lower latency, lower bandwidth
- CPU/GPU polls for flag

**When Used:**
- Small messages (< 32 KB)
- Latency-critical operations
- Eager protocol

### LL128 Protocol

**Mechanism**: Improved LL with 128-byte payload.

```
Sender                    Receiver
  │                          │
  ├─── 128B Data + Flag ────→│
  │                          │ (polls flag)
  │                          │
```

**Characteristics:**
- 128-byte payload per transfer
- Balance between Simple and LL
- Good latency and bandwidth
- Default for many workloads

**When Used:**
- Medium messages (32 KB - 2 MB)
- General purpose
- Good all-around performance

### Protocol Selection

NCCL automatically selects protocol based on:
- Message size
- Algorithm
- Network characteristics
- User override via `NCCL_PROTO`

**Typical Selection:**
```
Message Size     Protocol
< 32 KB          LL
32 KB - 2 MB     LL128
> 2 MB           Simple
```

## Algorithm Selection Logic

NCCL's algorithm selection considers:

### Message Size
```
Size         Preferred Algorithm
< 512 KB     Tree (low latency)
512 KB - 2 MB Tree or Ring (depends on topology)
> 2 MB       Ring (high bandwidth)
```

### Topology
- **Fully connected**: Ring or Tree
- **Hierarchical** (multi-node): Tree for intra-node, Ring for inter-node
- **Asymmetric**: Custom graphs

### Rank Count
- **Small (2-8)**: Tree often wins
- **Large (> 8)**: Ring scales better

### Operation Type
- **AllReduce**: Ring or Tree
- **Broadcast**: Tree
- **AllGather**: Ring
- **ReduceScatter**: Ring

### Tuning

```bash
# Force specific algorithm
NCCL_ALGO=Ring  # or Tree, CollNet

# Force specific protocol
NCCL_PROTO=Simple  # or LL, LL128

# Allow multiple, NCCL chooses
NCCL_ALGO=Ring,Tree
NCCL_PROTO=LL,LL128,Simple
```

## Channel Assignment

Each algorithm is executed across multiple channels:

```
Message: [───────────────────]
         ↓ Split into chunks
Channel 0: [────]
Channel 1:      [────]
Channel 2:           [────]
Channel 3:                [────]
```

**Benefits:**
- Parallel data transfer
- Pipeline different stages
- Utilize multiple NICs
- Higher aggregate bandwidth

**Channel Count:**
- Default: 4-16 channels
- Configurable: `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` (clamps)
- More channels = higher bandwidth (diminishing returns)

## Performance Implications

### AllReduce Performance Model

**Ring Algorithm:**
```
Time = Latency * (N-1) + (2 * Size * (N-1)) / (N * Bandwidth)
     ≈ 2 * Size / Bandwidth  (for large N)
```

**Tree Algorithm:**
```
Time = Latency * 2 * log2(N) + (2 * Size) / (BandwidthPerLink * TreeWidth)
```

### Optimization Guidelines

1. **Large Batch AllReduce**:
   - Use Ring algorithm
   - Simple protocol
   - Maximize channels

2. **Small Frequent AllReduce**:
   - Use Tree algorithm
   - LL or LL128 protocol
   - Minimize latency

3. **Imbalanced Topology**:
   - Custom tuning needed
   - May need topology XML file
   - Consider manual algorithm selection

## Summary

**Key Takeaways:**
- NCCL provides multiple algorithms (Ring, Tree) for each collective
- Ring: high bandwidth, many ranks, large messages
- Tree: low latency, fewer ranks, small messages
- Protocols (Simple, LL, LL128) optimize for different message sizes
- Multiple channels enable parallelism
- Automatic selection usually works well
- Manual tuning available for specific workloads

**Most Common in Production:**
- AllReduce with Ring algorithm for gradient aggregation
- LL128 or Simple protocol depending on model size
- 4-8 channels per GPU

**Related Documentation**:
- [algorithms/ring-algorithm.md](algorithms/ring-algorithm.md) - Ring algorithm details
- [algorithms/tree-algorithm.md](algorithms/tree-algorithm.md) - Tree algorithm details
- [nccl-core.md](nccl-core.md) - NCCL core concepts
- [nccl-datapath.md](nccl-datapath.md) - Data flow details

---

## Code References

### Functions

**NCCL Collective API (External - NVIDIA)** - Referenced but not linked:
- `ncclAllReduce()` - Reduce and broadcast result to all ranks
- `ncclBroadcast()` - Send data from one rank to all others
- `ncclReduce()` - Reduce to single rank
- `ncclAllGather()` - Gather data from all ranks to all ranks
- `ncclReduceScatter()` - Reduce and scatter chunks to ranks
- `ncclSend()` / `ncclRecv()` - Point-to-point send/receive

### Enumerations

**NCCL Data Types (ncclDataType_t)**:
- `ncclInt8`, `ncclUint8` - 8-bit integers
- `ncclInt32`, `ncclUint32` - 32-bit integers
- `ncclInt64`, `ncclUint64` - 64-bit integers
- `ncclFloat16`, `ncclFloat32`, `ncclFloat64` - Floating point
- `ncclBfloat16` - Brain floating point

**NCCL Reduction Operations (ncclRedOp_t)**:
- `ncclSum` - Sum reduction
- `ncclProd` - Product reduction
- `ncclMax` - Maximum reduction
- `ncclMin` - Minimum reduction
- `ncclAvg` - Average reduction (NCCL 2.10+)

### Algorithms

**Algorithm Types**:
- **Ring** - Bandwidth-optimal, O(N) latency
- **Tree** - Latency-optimal, O(log N) latency
- **CollNet** - Switch-assisted collectives (IB SHARP)
- **NVLS** / **NVLS Tree** - NVLink SHARP (Hopper H100/H200+, Blackwell B200/B300)
- **PAT** - Parallel Aggregated Trees for AllGather/ReduceScatter at scale (NCCL 2.23+; used through 2.31). See [algorithms/pat-algorithm.md](algorithms/pat-algorithm.md).

### Protocols

**Protocol Types**:
- **Simple** - Standard protocol, best for large messages (>128 KB)
- **LL (Low Latency)** - Flag-based, best for small messages (<4 KB)
- **LL128** - 128-byte granularity, good for medium messages (4-128 KB)

### Environment Variables

**Algorithm/Protocol Selection**:
- `NCCL_ALGO` - Force algorithm (Ring, Tree, CollNet, NVLS)
- `NCCL_PROTO` - Force protocol (Simple, LL, LL128)
- `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` - Channel count limits

### Total Code References
- **6 collective functions** (external NVIDIA)
- **10 data types** (ncclDataType_t)
- **5 reduction operations** (ncclRedOp_t)
- **4 algorithm types**
- **3 protocol types**
- **3 environment variables**

**Note**: NCCL collectives are part of NVIDIA's NCCL library. For algorithm implementation details, see the [algorithms/](algorithms/) directory.
