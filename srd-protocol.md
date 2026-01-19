# SRD (Scalable Reliable Datagram) Protocol

## Overview

**Scalable Reliable Datagram (SRD)** is AWS's custom transport protocol designed specifically for cloud-scale HPC and ML workloads. It provides reliable, out-of-order delivery over multiple network paths, implemented in hardware on the AWS Nitro card for minimal latency and jitter.

**Key Innovation**: SRD eliminates the Queue Pair (QP) scalability problem of traditional RDMA by combining the scalability of Unreliable Datagram (UD) with the reliability of Reliable Connected (RC), while adding intelligent multipath routing and hardware-accelerated congestion control.

**Official Paper**: L. Shalev, H. Ayoub, N. Bshara and E. Sabbag, "A Cloud-Optimized Transport Protocol for Elastic and Scalable HPC," *IEEE Micro*, vol. 40, no. 6, pp. 67-73, Nov.-Dec. 2020. [DOI: 10.1109/MM.2020.3013564](https://ieeexplore.ieee.org/document/9167399)

**Source Documentation**:
- Kernel driver: [amzn-drivers/kernel/linux/efa/SRD.txt](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/SRD.txt)
- Paper PDF: [Amazon Science](https://assets.amazon.science/a6/34/41496f64421faafa1cbe301c007c/a-cloud-optimized-transport-protocol-for-elastic-and-scalable-hpc.pdf)

## The Scalability Problem

### Traditional RDMA Scalability

**Reliable Connected (RC)**:
- Each connection requires a dedicated Queue Pair (QP)
- Full all-to-all connectivity: **N × p × p QPs per node**
  - N = number of nodes in cluster
  - p = processes per node
- Modern cloud instances: p has grown 3x in 5 years (more cores)
- Example: 100 nodes × 32 processes = **3,200 QPs per node**
- Memory per QP: ~100-200 KB (queue buffers + context)
- Total: **320-640 MB just for QP contexts**

**Reliable Datagram (RD)**:
- Reduces QPs to **p** (one per process, shared across destinations)
- Uses End-to-End (EE) contexts instead of per-connection QPs
- **Fatal limitation**: Single outstanding message per EE context
- Head-of-line blocking makes it unusable for modern workloads

### SRD Solution

**Resource Savings**:
- QPs required: **p** (same as RD, not N × p × p)
- Example: 32 QPs (one per process) vs 3,200 QPs (RC model)
- Memory savings: **~99% reduction** in QP context memory

**Key Advantages**:
1. **Unlimited outstanding messages** (unlike RD's limit of 1)
2. **Out-of-order delivery** (prevents head-of-line blocking)
3. **No user-visible EE contexts** (implicit management via Address Handles)
4. **Per-message addressing** (AH specified in each Work Request)
5. **Multipath routing** (up to 64 paths simultaneously)

## Protocol Characteristics

### Transport Semantics

**Reliability**:
- **Guaranteed at-most-once delivery** (packets never duplicated)
- Lost packets detected and retransmitted automatically
- Completion status indicates whether remote accepted the request
- Similar reliability guarantees to RC/RD

**Ordering**:
- **Out-of-order delivery** (unlike RC which preserves order)
- Packets may arrive in different order than sent
- Beneficial for:
  - Preventing head-of-line blocking
  - Multipath routing efficiency
  - Multiple independent message streams
  - SCSI/NVMe-like protocols that support out-of-order execution

**Segmentation**:
- **No segmentation** (unlike RC which fragments large messages)
- Each message sent as atomic unit
- Maximum message size limited by MTU (8192 bytes for EFA)
- Multi-packet messages not supported - use application-level fragmentation

### Queue Pair Model

From [amzn-drivers/kernel/linux/efa/SRD.txt](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/SRD.txt):

**QP Type**:
- Designated as `EFA_QPT_SRD` (maps to `IB_QPT_RESERVED1`)
- Semantically similar to UD (per-WR addressing)
- State machine identical to UD QP type
- Operations: Send only (RDMA operations possible in future)

**Work Request Format**:
```c
struct ib_srd_wr {
	struct ib_send_wr wr;
	struct ib_ah *ah;       // Address Handle (destination)
	u32 remote_qpn;         // Remote QP number
	u32 remote_qkey;        // Queue key for protection
};
```

Each Send WR includes:
- **Address Handle (AH)**: Identifies destination endpoint
- **Remote QP number**: Destination QP
- **Queue Key (qkey)**: Protection key for datagram operations
- Same scatter-gather list as UD

**SRD Context**:
- Implicitly associated with each Address Handle
- Provides reliable communication to a remote node (like RD's EE context)
- No explicit user management (created/destroyed automatically)
- Controlled implicitly by AH and QP management

**Context Lifecycle**:
- QP destroyed → all pending Send WRs implicitly canceled
- AH destroyed → outstanding WRs using that AH completed with error
- QP errors do **not** affect other QPs or SRD contexts

### Error Handling

**Completion Status**:

Success is reported after WR is ACKed by responder. New SRD-specific error codes:

From [amzn-drivers/kernel/linux/efa/SRD.txt](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/SRD.txt):

**Standard Errors** (inherited from RC/RD):
- `Success`: Operation completed successfully
- `Local Length Error`: Sum of Data Segment lengths exceeds port MTU
- `Local QP Operation Error`: Internal QP consistency error
- `Local Protection Error`: Memory Region invalid for requested operation
- `Work Request Flushed Error`: WR flushed during QP error transition or AH destruction
- `Bad Response Error`: Unexpected transport opcode from responder
- `Remote Invalid Request Error`: Responder detected invalid message
- `Remote Operation Error`: Responder QP-related error or malformed RQ WQE

**SRD-Specific Errors**:
- **`Bad Dest QP Error`**: Destination QP does not exist or is in error state
- **`SRD RNR Error`**: Receiver Not Ready - Receive Queue has no posted WQEs
  - **No automatic retries** (unlike RC which retries RNR)
  - Application must handle retry logic

**Error Isolation**:
- Errors on one SRD context do **not** affect other contexts
- Errors on one QP do **not** affect other QPs
- Responder QP errors do **not** affect SRD context state
- Bad requests are dropped and ACKed with drop indication (error returned to requester)

### Asynchronous Events

**Unaffiliated Event**:
- **Remote Unresponsive Event**: Local transport timeout exceeded for a specific destination (AH)
- Indicates previously responsive remote peer has stopped responding
- SRD context state associated with AH is **not affected** (can retry)

## Multipath Routing

### Path Selection Algorithm

Based on the IEEE Micro paper and AWS documentation:

**Core Strategy**:
- SRD sends packets over **as many network paths as possible**
- Actively avoids overloaded paths
- Does **not** preserve packet order (out-of-order delivery)

**Path Diversity**:
- Modern datacenter networks provide hundreds to thousands of paths
- SRD selects **64 paths at a time** from available paths (for memory efficiency)
- Paths chosen from ECMP (Equal-Cost Multi-Path) routes
- Dynamic rerouting when paths become congested

**Path Selection Granularity**:
- Per-flow path selection (not per-packet)
- Flow = SRD context (source QP + destination AH pair)
- Each flow uses its own set of 64 active paths
- Paths can overlap between flows

### Congestion Detection

**Nitro Card Implementation**:
- SRD congestion control runs **in hardware** on Nitro networking card
- Provides sub-millisecond response to congestion
- Minimizes jitter (consistent latency)

**Congestion Signals**:

From research on SRD architecture:

1. **Per-Path Congestion**:
   - Monitored via RTT (Round-Trip Time) on each path
   - If RTT increases on a specific path → mark path as congested
   - Reroute traffic to alternate paths

2. **Connection-Wide Congestion**:
   - Detected if RTT increases on **majority of paths**
   - Indicates network-wide congestion (e.g., incast scenario)
   - Alternative trigger: Estimated rate drops below transmit rate
   - Rate control applied across all paths

**Congestion Response**:
- **Path-level**: Immediate rerouting to non-congested paths
- **Flow-level**: Rate reduction across all paths
- **Fast recovery**: ~100s of microseconds (vs milliseconds for TCP)
- **Packet drop recovery**: Hardware retransmission without software involvement

### Reliability Mechanisms

**Packet Retransmission**:
- Requester detects lost packets (timeout-based or selective ACK)
- Automatic retransmission similar to RC/RD
- Retransmission policy managed by hardware (Nitro card)
- Sub-millisecond retransmission timeouts

**ACK Protocol**:
- Responder generates ACKs for received requests
- ACK format differs from RC/RD due to shared SRD context nature:
  - Receive WQEs can be fetched from multiple QPs
  - Send WRs can target different destinations
  - ACKs include drop indication for requests that passed SRD checks but failed QP checks

**Drop Handling**:
- Request passes SRD transport checks but fails QP validation → dropped
- Responder sends ACK with **drop indication**
- Requester generates completion with appropriate error status
- Receiver QP errors don't affect SRD handling of other WRs

## Performance Characteristics

### Latency

From AWS research and measurements:

**Baseline Performance**:
- Small message latency: ~12-15 μs (intra-AZ)
- P99.9 latency improvement: **85% reduction** vs TCP
- Jitter: Minimal due to hardware implementation

**Latency Breakdown** (estimated):
- Software posting: ~100-200 ns
- Hardware DMA + packetization: ~300-500 ns
- Network flight time: ~10-15 μs
- ACK processing: ~100-200 ns
- Completion delivery: ~100-200 ns

**Comparison**:
- SRD: ~12-15 μs
- RoCEv2 (RC): ~10-12 μs (slightly faster but less scalable)
- TCP: ~50-100 μs (higher software overhead)

### Throughput

**Single-Flow Bandwidth**:
- Maximum: **25 Gbps** per flow (from AWS research)
- Achieved via multipath load balancing
- Limited by: Per-path bandwidth, packet pacing, congestion control

**Aggregate Bandwidth**:
- p4d instances: 400 Gbps total (4×100G EFAs)
- p5 instances: 3200 Gbps total (32×100G EFAs)
- Multi-flow aggregation achieves near-theoretical maximum

**Message Rate**:
- Small messages (64B): ~10-15 Mpps per QP
- Large messages (8KB MTU): ~300-400K msg/s per QP

### Comparison with Other Transports

| Aspect | SRD | RC (RoCE) | RD | UD | TCP |
|--------|-----|-----------|----|----|-----|
| **QPs Required** | p | N×p×p | p | p | N/A |
| **Reliability** | Yes | Yes | Yes | No | Yes |
| **Ordering** | Out-of-order | In-order | In-order | Out-of-order | In-order |
| **Outstanding Msgs** | Unlimited | Unlimited | 1 (fatal limit) | N/A | Unlimited |
| **Multipath** | Yes (64 paths) | No | No | No | No |
| **Latency** | ~12-15 μs | ~10-12 μs | ~12-15 μs | ~10 μs | ~50-100 μs |
| **HW Implementation** | Nitro card | NIC | NIC | NIC | OS stack |
| **Congestion Control** | HW, sub-ms | PFC/ECN | PFC/ECN | None | SW, multi-ms |

**Key Advantages of SRD**:
1. **Scalability**: O(p) QPs vs O(N×p²) for RC
2. **Performance**: Multipath routing achieves 25 Gbps single-flow
3. **Resilience**: Fast congestion response, automatic rerouting
4. **Cloud-optimized**: Designed for multi-tenant, dynamic networks

## Implementation Details

### Nitro Card Architecture

**Hardware Offloads**:
- Packet packetization and transmission
- Multipath path selection (64 active paths)
- Congestion detection (per-path RTT monitoring)
- Congestion control (rate limiting, rerouting)
- Packet retransmission (timeout detection + retry)
- Completion generation

**Why Hardware?**:
- **Low latency**: Sub-microsecond congestion response
- **Consistency**: Minimal jitter (no OS scheduling delays)
- **Scalability**: Handles millions of flows without CPU overhead
- **Efficiency**: Frees CPU for application work

### SRD vs Traditional Datagram Models

**Unreliable Datagram (UD)**:
- ✅ Scalable (one QP per process)
- ❌ No reliability (packets can be lost)
- ❌ No congestion control
- ✅ Out-of-order delivery
- Use case: Not suitable for HPC/ML (data loss unacceptable)

**Reliable Datagram (RD)** (InfiniBand spec):
- ✅ Scalable (one QP per process)
- ✅ Reliable delivery
- ❌ **Single outstanding message per EE context** (major limitation)
- ✅ Out-of-order delivery
- ❌ Rarely implemented in hardware (complexity)
- Use case: Theoretically good, but impractical due to single-message limit

**SRD (AWS's approach)**:
- ✅ Scalable (one QP per process)
- ✅ Reliable delivery
- ✅ **Unlimited outstanding messages** (solves RD's problem)
- ✅ Out-of-order delivery
- ✅ Multipath routing (unique to SRD)
- ✅ Hardware-accelerated congestion control
- ✅ Implicit context management (easier than RD)
- Use case: Ideal for cloud-scale HPC/ML

## Use Cases and Benefits

### HPC Workloads

**MPI Applications**:
- All-to-all communication patterns benefit from SRD's scalability
- Out-of-order delivery works well with MPI message matching
- Multipath routing provides resilience to network asymmetries

**Example**: 100-node cluster, 32 processes/node:
- RC: 32 × 100 × 32 = **102,400 QPs per node** (impractical)
- SRD: **32 QPs per node** (3,200x reduction)

### ML/AI Training

**Collective Operations** (AllReduce, AllGather, etc.):
- NCCL benefits from SRD's low latency and high bandwidth
- Multi-rail configurations leverage multiple EFAs
- 64-path multipath provides load balancing across rings

**Example**: 8-GPU instance (p5.48xlarge):
- 32 EFAs (4 per GPU)
- Each GPU process creates 1 SRD QP
- **8 QPs total** vs thousands with RC

### Cloud Benefits

**Multi-Tenancy**:
- Congestion control prevents noisy neighbor problems
- Per-flow fairness via multipath load balancing
- Fast congestion response (sub-ms) vs TCP (multi-ms)

**Elasticity**:
- QP scalability enables dynamic cluster sizing
- New nodes join without QP explosion
- Address Handles created on-demand (no connection setup delay)

**Reliability**:
- Automatic path rerouting on failures
- Hardware retransmission (no software involvement)
- Graceful degradation under congestion

## Programming Interface

### Creating SRD Queue Pair

Using libibverbs/rdma-core API:

```c
struct ibv_qp_init_attr qp_attr = {
	.qp_type = EFA_QPT_SRD,  // SRD transport
	.send_cq = send_cq,
	.recv_cq = recv_cq,
	.cap = {
		.max_send_wr = 512,
		.max_recv_wr = 512,
		.max_send_sge = 2,
		.max_recv_sge = 1,
	},
};

struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);
```

**State Transitions** (same as UD):
- RESET → INIT → RTR (Ready to Receive) → RTS (Ready to Send)

### Posting Send Request

```c
// Create Address Handle for destination
struct ibv_ah_attr ah_attr = {
	.dlid = dest_lid,              // Destination LID (if applicable)
	.sl = 0,                        // Service Level
	.src_path_bits = 0,
	.static_rate = 0,
	.is_global = 1,
	.grh = {
		.dgid = dest_gid,           // Destination GID
		.sgid_index = 0,
		.hop_limit = 64,
		.traffic_class = 0,
	},
};
struct ibv_ah *ah = ibv_create_ah(pd, &ah_attr);

// Post send work request
struct ib_srd_wr wr = {
	.wr = {
		.wr_id = request_id,
		.sg_list = &sge,
		.num_sge = 1,
		.opcode = IBV_WR_SEND,
		.send_flags = IBV_SEND_SIGNALED,
	},
	.ah = ah,
	.remote_qpn = dest_qp_number,
	.remote_qkey = Q_KEY,
};

ibv_post_send(qp, (struct ibv_send_wr *)&wr, &bad_wr);
```

**Key Differences from RC**:
- Address Handle specified **per WR** (not per QP)
- Remote QP number in WR (not QP attribute)
- No connection setup required (connectionless like UD)

### Receiving Messages

```c
// Post receive buffers (same as UD)
struct ibv_sge sge = {
	.addr = (uint64_t)buffer,
	.length = buffer_size,
	.lkey = mr->lkey,
};

struct ibv_recv_wr wr = {
	.wr_id = request_id,
	.sg_list = &sge,
	.num_sge = 1,
};

ibv_post_recv(qp, &wr, &bad_wr);

// Poll completion queue
struct ibv_wc wc;
int n = ibv_poll_cq(recv_cq, 1, &wc);
if (n > 0) {
	if (wc.status == IBV_WC_SUCCESS) {
		// Process received data
		uint32_t src_qp = wc.src_qp;
		uint16_t src_lid = wc.slid;
		// ...
	} else {
		// Handle error (e.g., RNR, bad dest QP)
	}
}
```

**Completion Information** (RX):
- `src_qp`: Source QP number
- `slid`: Source LID
- `byte_len`: Actual bytes received
- Status codes including SRD-specific errors

## Libfabric Integration

### FI_EP_RDM Endpoint

Libfabric EFA provider maps SRD to `FI_EP_RDM` (Reliable Datagram Message) endpoint type:

```c
struct fi_info *hints = fi_allocinfo();
hints->ep_attr->type = FI_EP_RDM;  // Reliable Datagram Message
hints->caps = FI_MSG | FI_TAGGED;  // Message and tagged messaging

struct fi_info *info;
fi_getinfo(FI_VERSION(1, 20), NULL, NULL, 0, hints, &info);

// Create endpoint
struct fid_ep *ep;
fi_endpoint(domain, info, &ep, NULL);
```

**Libfabric Abstraction**:
- Hides SRD QP details behind `FI_EP_RDM` endpoint
- Automatic Address Handle management (fi_av_insert)
- Simplified API vs raw libibverbs

### NCCL Integration

NCCL uses SRD via the OFI plugin:

```
NCCL → OFI Plugin → Libfabric → EFA Provider → SRD QPs
```

**Key Operations**:
- `fi_send()` → SRD send WR with Address Handle
- `fi_recv()` → Post receive buffer to SRD QP
- Address Vector (AV) → Maps peer addresses to Address Handles
- Multi-rail: Multiple SRD QPs per process (one per EFA)

## Performance Tuning

### Queue Depth

**Send Queue**:
- Recommendation: 256-512 entries
- Too small: Underutilization, stalls waiting for completions
- Too large: Memory waste, longer polling times

**Receive Queue**:
- Recommendation: 512-1024 entries
- Must have enough posted buffers to avoid RNR errors
- Consider message arrival rate and processing latency

### Completion Queue Polling

**Polling Strategy**:
- Busy polling: Low latency, high CPU usage
- Adaptive polling: Start busy, back off if idle
- Interrupt-driven: Low CPU, higher latency (not recommended for HPC)

**Batch Polling**:
- Poll multiple completions per call (e.g., 16-32)
- Amortizes polling overhead
- Balance batch size vs latency

### Multi-Rail Configuration

**p5 Instances** (32 EFAs, 4 per GPU):
- Create 4 SRD QPs per process (one per EFA)
- Round-robin or hash-based load balancing across QPs
- NCCL automatically uses all available EFAs

**Benefits**:
- 4× bandwidth per GPU (~400 Gbps)
- Path diversity (each EFA has independent 64-path selection)
- Fault tolerance (failure of one EFA doesn't block communication)

### Memory Registration

**Pre-Registration**:
- Register all buffers at initialization (avoid runtime overhead)
- Use freelist allocator with pre-registered blocks
- MR cache for user buffers (25x speedup)

**Fast Registration**:
- Use `EFA_IO_FAST_REG` for dynamic buffers
- Typical cost: ~10-20 μs (vs 100-500 μs for full registration)
- Deregister with `EFA_IO_FAST_INV` when done

## Limitations and Considerations

### Message Size

**MTU Limit**:
- EFA MTU: 8192 bytes
- Messages larger than MTU must be fragmented by application
- No hardware segmentation (unlike RC)

**Workaround**:
- Application-level chunking (e.g., NCCL slices large messages)
- Use pipeline protocol for large transfers

### Out-of-Order Delivery

**Implications**:
- Application must handle message reordering
- MPI/NCCL already designed for out-of-order networks
- Use message matching tags for correlation

**Benefit**:
- Prevents head-of-line blocking
- Better utilization of multiple paths
- Higher throughput for multi-flow workloads

### RNR Handling

**No Automatic Retry**:
- SRD returns RNR error immediately (no retry)
- Application responsible for retry logic

**Best Practice**:
- Pre-post sufficient receive buffers
- Monitor RNR error rate (should be near zero in steady state)
- Increase RQ depth if RNR errors occur

### Hardware Requirements

**EFA Version**:
- SRD supported on all EFA generations
- Best performance on EFA Gen 3+ (p5 instances)

**Instance Types**:
- p4d, p4de: EFA Gen 2 (SRD send/recv, emulated RDMA)
- p5, p5e, p5en: EFA Gen 3+ (SRD + native RDMA)

## Summary

| Aspect | Details |
|--------|---------|
| **Protocol Type** | Reliable, out-of-order datagram |
| **Scalability** | O(p) QPs vs O(N×p²) for RC |
| **Multipath** | Up to 64 paths simultaneously |
| **Congestion Control** | Hardware-based, sub-millisecond response |
| **Implementation** | Nitro networking card (hardware offload) |
| **Latency** | ~12-15 μs small messages (intra-AZ), 85% better P99.9 vs TCP |
| **Throughput** | 25 Gbps per flow, 400 Gbps per GPU (multi-rail) |
| **Message Rate** | ~10-15 Mpps small messages |
| **Ordering** | Out-of-order delivery (prevents head-of-line blocking) |
| **Reliability** | Guaranteed delivery, automatic retransmission |
| **Error Handling** | SRD-specific RNR and Bad Dest QP errors |
| **MTU** | 8192 bytes (no hardware segmentation) |

**Key Takeaways**:
1. **SRD solves the QP scalability problem** while maintaining reliability
2. **Multipath routing (64 paths)** provides high single-flow bandwidth and resilience
3. **Hardware implementation on Nitro card** ensures low latency and consistent performance
4. **Out-of-order delivery** prevents head-of-line blocking, ideal for HPC/ML
5. **Cloud-optimized design** with fast congestion control and multi-tenancy support
6. **Implicit context management** simplifies programming vs traditional RD
7. **85% P99.9 latency reduction** vs TCP demonstrates effectiveness for HPC workloads

**Related Documentation**:
- [efa-hardware-architecture.md](efa-hardware-architecture.md) - Queue pair and completion queue details
- [kernel-efa-driver.md](kernel-efa-driver.md) - Kernel driver SRD implementation
- [efa-provider.md](efa-provider.md) - Libfabric EFA provider using SRD
- [ofi-plugin.md](ofi-plugin.md) - NCCL OFI plugin leveraging SRD transport

**References**:
1. L. Shalev et al., "A Cloud-Optimized Transport Protocol for Elastic and Scalable HPC," IEEE Micro, vol. 40, no. 6, 2020
2. AWS EFA Documentation: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html
3. SRD Protocol Specification: https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/SRD.txt
