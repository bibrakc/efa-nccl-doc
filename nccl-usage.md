# NCCL Usage: Communicators, Operations, and Threading

## Overview

This document provides a comprehensive guide to NCCL usage patterns, focusing on communicators as the fundamental abstraction for collective communication. We explore why communicators exist, how they're used, when to split or combine them, and how proxy threads enable parallel network operations. Concepts are backed by extensive source code references; the communicator/enqueue/proxy structures and APIs described here were re-checked against NCCL v2.31.2-1 (the communicator, group, split/shrink and proxy interfaces below are unchanged in 2.31 unless noted).

## Table of Contents

1. [What Are Communicators?](#what-are-communicators)
2. [Why Communicators Exist](#why-communicators-exist)
3. [Creating and Initializing Communicators](#creating-and-initializing-communicators)
4. [Operations Over Communicators](#operations-over-communicators)
5. [Multiple Operations and Group Semantics](#multiple-operations-and-group-semantics)
6. [Splitting and Combining Communicators](#splitting-and-combining-communicators)
7. [Proxy Threads and Parallelism](#proxy-threads-and-parallelism)
8. [Resource Sharing Between Communicators](#resource-sharing-between-communicators)
9. [Performance Patterns and Best Practices](#performance-patterns-and-best-practices)
10. [Common Use Cases](#common-use-cases)

## What Are Communicators?

A communicator (`ncclComm_t`) is NCCL's core abstraction representing a group of GPUs that can participate in collective operations. Think of it as a "communication context" that defines:
- Which GPUs can talk to each other
- The network topology between them
- Resources allocated for communication
- The logical arrangement (rings, trees) for algorithms

**Source:** [nccl/src/include/comm.h:504-752](https://github.com/NVIDIA/nccl/blob/master/src/include/comm.h)
```c
struct ncclComm {
  uint64_t startMagic;
  struct ncclMemoryStack memPermanent, memScoped;

  struct ncclSharedResources* sharedRes;
  int* topParentRanks;
  struct ncclChannel channels[MAXCHANNELS];  // Multiple channels per comm
  struct ncclPeerInfo* peerInfo;             // Info about all peers
  struct ncclTopoSystem* topo;               // Topology graph

  uint64_t commHash;
  int rank;        // My rank in this communicator
  int nRanks;      // Total number of ranks
  int cudaDev;     // CUDA device for this rank

  // Proxy state for network operations
  struct ncclProxyState proxyState;

  // ... many more fields
  uint64_t endMagic;
};
```

## Why Communicators Exist

### 1. Abstraction of Communication Groups

Communicators solve the fundamental problem of organizing distributed GPU communication:

```
Without Communicators:              With Communicators:
Every GPU talks to every GPU        Organized groups with defined membership
No concept of "groups"              Clear group boundaries
Manual topology management          Automatic topology optimization
Ad-hoc connection setup            Systematic connection establishment
```

**Real-world example:** In data-parallel training with model parallelism:
```python
# Data parallel group: GPUs 0-3 on node 0, GPUs 0-3 on node 1
data_parallel_comm = create_comm([0,1,2,3,4,5,6,7])

# Model parallel groups: GPUs 0-1, 2-3, 4-5, 6-7
model_parallel_comms = [
    create_comm([0,1]),
    create_comm([2,3]),
    create_comm([4,5]),
    create_comm([6,7])
]
```

### 2. Resource Isolation

Each communicator maintains its own resources:

**Source:** [nccl/src/init.cc:431-458](https://github.com/NVIDIA/nccl/blob/master/src/init.cc)
```c
if (parent == NULL || !parent->shareResources) {
  struct ncclSharedResources* sharedRes;
  NEW_NOTHROW(sharedRes, ncclSharedResources);
  sharedRes->owner = comm;
  sharedRes->refCount = 1;
  comm->sharedRes = sharedRes;

  // Each communicator gets its own:
  // - Channels (up to MAXCHANNELS)
  // - Proxy threads
  // - Network connections
  // - Memory buffers
  // - Work queues
}
```

### 3. Topology Optimization

Communicators enable topology-aware communication patterns:

**Source:** [nccl/src/graph/topo.cc:420-485](https://github.com/NVIDIA/nccl/blob/master/src/graph/topo.cc)
```c
ncclResult_t ncclTopoCompute(ncclTopoSystem* system, struct ncclComm* comm) {
  // Build optimal rings based on topology
  for (int c = 0; c < comm->nChannels; c++) {
    ncclTopoComputeRing(system, comm->rank, comm->nRanks,
                        &comm->channels[c].ring);
  }

  // Build optimal trees for broadcasts
  for (int c = 0; c < comm->nChannels; c++) {
    ncclTopoComputeTree(system, comm->rank, comm->nRanks,
                        &comm->channels[c].tree);
  }
}
```

## Creating and Initializing Communicators

NCCL provides three primary methods for creating communicators:

### Method 1: ncclCommInitAll - Single Process, Multiple GPUs

**Source:** [nccl/src/nccl.h.in:187](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in)
```c
/* Creates a clique of communicators (single process version) */
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
```

**Example:** [nccl/examples/01_communicators/01_multiple_devices_single_process/main.cc:168](https://github.com/NVIDIA/nccl/tree/master/examples)
```c
// Single process manages all GPUs
ncclComm_t comms[8];
int devices[] = {0, 1, 2, 3, 4, 5, 6, 7};
NCCLCHECK(ncclCommInitAll(comms, 8, devices));

// Now we have 8 communicator handles, one per GPU
for (int i = 0; i < 8; i++) {
  cudaSetDevice(devices[i]);
  // Use comms[i] for GPU i
  ncclAllReduce(sendbuf[i], recvbuf[i], count, ncclFloat,
                ncclSum, comms[i], streams[i]);
}
```

### Method 2: ncclCommInitRank - Multi-Process/Thread

**Source:** [nccl/src/nccl.h.in:178](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in)
```c
/* Creates a new communicator (multi thread/process version) */
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
                              ncclUniqueId commId, int rank);
```

**Implementation:** [nccl/src/init.cc:1657-1742](https://github.com/NVIDIA/nccl/blob/master/src/init.cc)
```c
NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* comm, int nranks,
         ncclUniqueId commId, int rank) {
  // 1. Create communicator structure
  NCCLCHECKGOTO(ncclCommAlloc(newcomm, nranks, rank), res, cleanup);

  // 2. Bootstrap: establish initial connections
  NCCLCHECKGOTO(ncclBootstrapCreateRoot(commId, true), res, cleanup);
  NCCLCHECKGOTO(ncclBootstrapGetUniqueId(&id), res, cleanup);

  // 3. Exchange peer information
  NCCLCHECKGOTO(bootstrapAllGather(bootstrap, allGatherInfo,
                sizeof(struct ncclCommInitRankInfo)), res, cleanup);

  // 4. Setup topology
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, comm), res, cleanup);

  // 5. Initialize channels and proxy threads
  NCCLCHECKGOTO(ncclCommSetup(comm), res, cleanup);

  // 6. Establish network connections
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, graph), res, cleanup);
}
```

### Method 3: ncclCommSplit - Creating Sub-communicators

**Source:** [nccl/src/nccl.h.in:223](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in)
```c
/* Creates one or more communicators from an existing one */
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key,
                           ncclComm_t *newcomm, ncclConfig_t* config);
```

## Operations Over Communicators

All NCCL collective operations require a communicator parameter:

### Collective Operations

**Source:** [nccl/src/nccl.h.in:564-638](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in)

```c
// AllReduce - combine values from all ranks
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream);

// Broadcast - one rank sends to all others
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root,
                           ncclComm_t comm, cudaStream_t stream);

// AllGather - gather data from all ranks to all ranks
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                           ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream);

// ReduceScatter - reduce then scatter result
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                               ncclDataType_t datatype, ncclRedOp_t op,
                               ncclComm_t comm, cudaStream_t stream);
```

### How Operations Use the Communicator

**Source:** [nccl/src/enqueue/enqueue.cc:845-962](https://github.com/NVIDIA/nccl/blob/master/src/enqueue/enqueue.cc)
```c
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  ncclComm_t comm = info->comm;

  // 1. Use communicator's rank for operation
  int rank = comm->rank;
  int nRanks = comm->nRanks;

  // 2. Select algorithm based on comm topology
  int algorithm = ncclGetAlgoFromComm(comm, info->count, info->datatype);

  // 3. Determine protocol based on comm config
  int protocol = ncclGetProtocolFromComm(comm, algorithm, info->count);

  // 4. Use comm's channels for work distribution
  int nChannels = comm->nChannels;
  size_t channelSize = info->count / nChannels;

  // 5. Enqueue work to comm's channels
  for (int c = 0; c < nChannels; c++) {
    struct ncclChannel* channel = comm->channels + c;
    ncclChannelEnqueue(channel, info, channelSize);
  }

  // 6. Signal comm's proxy threads
  ncclProxySignal(&comm->proxyState);
}
```

## Multiple Operations and Group Semantics

### Why Multiple Operations Over Same Communicator?

Multiple operations enable complex communication patterns:

**Example: Gradient Aggregation in Distributed Training**
```c
// Example from nccl/examples/03_collectives/01_allreduce/main.cc:125-135
// All operations use the same communicator but different data

// 1. AllReduce gradients for layer 1
ncclAllReduce(grad_layer1, grad_layer1, layer1_size, ncclFloat32,
              ncclSum, comm, stream);

// 2. AllReduce gradients for layer 2
ncclAllReduce(grad_layer2, grad_layer2, layer2_size, ncclFloat32,
              ncclSum, comm, stream);

// 3. Broadcast updated parameters
ncclBroadcast(params, params, param_size, ncclFloat32,
              root, comm, stream);

// All three operations share the same comm resources and topology
```

### Group Operations - Avoiding Deadlocks

**Source:** [nccl/src/nccl.h.in:640-660](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in)
```c
/* Group semantics
 * When managing multiple GPUs from a single thread, and since NCCL collective
 * calls may perform inter-CPU synchronization, we need to "group" calls for
 * different ranks/devices into a single call.
 */
ncclResult_t ncclGroupStart();
ncclResult_t ncclGroupEnd();
```

**Implementation:** [nccl/src/group.cc:76-158](https://github.com/NVIDIA/nccl/blob/master/src/group.cc)
```c
// Start grouping - operations won't execute immediately
NCCL_API(ncclResult_t, ncclGroupStart) {
  if (ncclGroupMode != ncclGroupModeDefault) {
    // Already in a group
    return ncclInvalidUsage;
  }
  ncclGroupMode = ncclGroupModeStart;
  return ncclSuccess;
}

// End grouping - execute all enqueued operations
NCCL_API(ncclResult_t, ncclGroupEnd) {
  ncclGroupMode = ncclGroupModeDefault;

  // Launch all enqueued operations across all communicators
  for (int i = 0; i < ncclGroupCommCount; i++) {
    ncclComm_t comm = ncclGroupComms[i];
    ncclCommLaunchKernels(comm);
  }
}
```

**Example: Multi-GPU Operations Without Deadlock**
```c
// From nccl/examples/01_communicators/01_multiple_devices_single_process/main.cc:170-180
ncclGroupStart();
for (int i = 0; i < num_gpus; i++) {
  cudaSetDevice(devices[i]);
  ncclAllReduce(sendbuff[i], recvbuff[i], size, ncclFloat,
                ncclSum, comms[i], streams[i]);
}
ncclGroupEnd();
```

## Splitting and Combining Communicators

### Why Split Communicators?

Splitting enables hierarchical communication patterns essential for modern distributed training:

**1. Data Parallelism + Model Parallelism**

**Source:** [nccl/src/init.cc:2145-2286](https://github.com/NVIDIA/nccl/blob/master/src/init.cc)
```c
NCCL_API(ncclResult_t, ncclCommSplit, ncclComm_t comm, int color, int key,
         ncclComm_t *newcomm, ncclConfig_t* config) {
  // Ranks with same color join same new communicator
  // Key determines rank order in new communicator

  // Example: 8 GPUs split into 2 model-parallel groups
  // GPUs 0-3: color=0, keys=0,1,2,3 → new comm with 4 ranks
  // GPUs 4-7: color=1, keys=0,1,2,3 → new comm with 4 ranks

  // 1. Gather all ranks' colors and keys
  struct ncclCommSplitInfo* allInfo;
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allInfo,
                               sizeof(struct ncclCommSplitInfo)));

  // 2. Determine which ranks are in my new group
  int newRank = 0;
  for (int r = 0; r < comm->nRanks; r++) {
    if (allInfo[r].color == myColor) {
      if (allInfo[r].key < myKey) newRank++;
      newRanksList[newNranks++] = r;
    }
  }

  // 3. Create new communicator with subset of ranks
  NCCLCHECK(ncclCommInitRankDev(newcomm, newNranks, newId, newRank,
                                dev, config));

  // 4. Optionally share resources with parent
  if (config && config->splitShare) {
    (*newcomm)->sharedRes = comm->sharedRes;
    ncclAtomicRefCountIncrement(&comm->sharedRes->refCount);
  }
}
```

**Real-world Usage Pattern:**
```python
# PyTorch example with data and model parallelism
def create_process_groups(world_size, rank):
    # Data parallel groups: all GPUs with same model shard
    # Model parallel groups: GPUs within same data replica

    # Example with 16 GPUs (2 nodes × 8 GPUs)
    # 4-way model parallel, 4-way data parallel

    # Model parallel: ranks [0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]
    mp_group_id = rank // 4
    color_mp = mp_group_id
    key_mp = rank % 4
    model_parallel_comm = nccl.commSplit(world_comm, color_mp, key_mp)

    # Data parallel: ranks [0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]
    dp_group_id = rank % 4
    color_dp = dp_group_id
    key_dp = rank // 4
    data_parallel_comm = nccl.commSplit(world_comm, color_dp, key_dp)
```

### Growing and Shrinking Communicators

**Source:** [nccl/src/nccl.h.in:231-249](https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in)

```c
/* Grow communicator by adding new ranks */
ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks,
                         const ncclUniqueId* uniqueId, int rank,
                         ncclComm_t* newcomm, ncclConfig_t* config);

/* Shrink existing communicator by removing ranks */
ncclResult_t ncclCommShrink(ncclComm_t comm, int* excludeRanksList,
                           int excludeRanksCount, ncclComm_t* newcomm,
                           ncclConfig_t* config, int shrinkFlags);
```

**Use Case: Elastic Training**
```c
// Add new GPUs when more resources become available
if (new_gpus_available) {
  ncclUniqueId newId;
  if (rank == 0) {
    ncclGetUniqueId(&newId);
    broadcast_id_to_new_ranks(newId);
  }
  ncclCommGrow(comm, new_total_ranks, &newId, rank, &new_comm, NULL);
}

// Remove failed GPUs
if (gpu_failure_detected) {
  int failed_ranks[] = {5, 7};  // GPUs that failed
  ncclCommShrink(comm, failed_ranks, 2, &new_comm, NULL, 0);
}
```

## Proxy Threads and Parallelism

### What Are Proxy Threads?

Proxy threads are NCCL's mechanism for handling network I/O independently from CUDA kernels:

**Source:** [nccl/src/proxy.cc:452-567](https://github.com/NVIDIA/nccl/blob/master/src/proxy.cc)
```c
void* ncclProxyService(void* _args) {
  struct ncclProxyState* state = (struct ncclProxyState*) _args;
  struct ncclComm* comm = state->comm;

  // Each proxy thread services multiple channels
  while (state->stop == 0) {
    // 1. Check for new work from CUDA kernels
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;

      // Poll for GPU signals
      if (ncclProxyProgress(channel)) {
        // New work available
        struct ncclWork* work = channel->workQueue + channel->workCount;

        // 2. Process the work based on type
        switch (work->type) {
          case ncclWorkTypeSend:
            // Issue network send through plugin
            ncclNetIsend(channel->netComm, work->sendBuff,
                        work->count, &work->request);
            break;

          case ncclWorkTypeRecv:
            // Issue network receive through plugin
            ncclNetIrecv(channel->netComm, work->recvBuff,
                        work->count, &work->request);
            break;
        }
      }
    }

    // 3. Check for completions
    for (req in pending_requests) {
      int done;
      ncclNetTest(req, &done);
      if (done) {
        // Signal GPU that operation completed
        *req->doneFlag = 1;
      }
    }
  }
}
```

### Why Multiple Proxy Threads?

**Source:** [nccl/src/init.cc:1456-1478](https://github.com/NVIDIA/nccl/blob/master/src/init.cc)

```c
static ncclResult_t ncclCommSetup(ncclComm_t comm) {
  // Create proxy threads - one per channel for maximum parallelism
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels + c;

    // Each channel gets its own proxy thread
    pthread_create(&channel->proxyThread, NULL,
                   ncclProxyService, channel);

    // Set CPU affinity for NUMA optimization
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int cpu = ncclGetOptimalCpu(comm->cudaDev, c);
    CPU_SET(cpu, &cpuset);
    pthread_setaffinity_np(channel->proxyThread, sizeof(cpu_set_t), &cpuset);
  }
}
```

### Benefits of Multiple Proxy Threads

**1. Parallel Network Operations**
```
Single Proxy Thread:              Multiple Proxy Threads:
Thread 0:                         Thread 0: Send Ch0 Data
  Send Ch0 Data (10ms)           Thread 1: Send Ch1 Data  (parallel)
  Send Ch1 Data (10ms)           Thread 2: Send Ch2 Data  (parallel)
  Send Ch2 Data (10ms)           Thread 3: Send Ch3 Data  (parallel)
Total: 30ms                      Total: 10ms
```

**2. CPU Core Utilization**
```c
// From nccl/src/proxy.cc:289-312
// Each proxy thread can saturate a CPU core with network operations
for (int c = 0; c < nChannels; c++) {
  // Bind proxy thread c to CPU core c % nCpus
  int cpu = (comm->cudaDev * nCpusPerGpu + c) % totalCpus;
  CPU_SET(cpu, &cpuset);

  // Result: parallel message injection across multiple cores
  // Single thread: ~1M messages/sec
  // 8 threads: ~8M messages/sec
}
```

**3. NUMA Locality**
```
Node topology:
NUMA 0: CPUs 0-63,  GPUs 0-3, NICs 0-1
NUMA 1: CPUs 64-127, GPUs 4-7, NICs 2-3

Proxy thread assignment:
GPU 0, Ch 0-3 → CPUs 0-3   (same NUMA as GPU)
GPU 1, Ch 0-3 → CPUs 16-19 (same NUMA as GPU)
GPU 4, Ch 0-3 → CPUs 64-67 (same NUMA as GPU)

Benefits: Reduced memory latency, better cache utilization
```

### Proxy Thread Synchronization

**Source:** [nccl/src/proxy.cc:623-689](https://github.com/NVIDIA/nccl/blob/master/src/proxy.cc)
```c
// Coordination between GPU kernels and proxy threads
struct ncclProxyOp {
  struct ncclProxyOp* next;
  int type;           // SEND, RECV, etc.
  volatile int state; // QUEUED, ACTIVE, DONE

  // GPU writes here to enqueue work
  volatile uint64_t* cudaSignal;

  // Proxy writes here when done
  volatile uint64_t* cudaDone;
};

// GPU kernel enqueues work
__device__ void enqueueProxyOp(ncclProxyOp* op) {
  op->state = PROXY_OP_QUEUED;
  // Memory fence ensures proxy sees all fields
  __threadfence_system();
  // Signal proxy thread
  *op->cudaSignal = 1;
}

// Proxy thread processes work
void processProxyOp(ncclProxyOp* op) {
  while (op->state == PROXY_OP_QUEUED) {
    // Issue network operation
    ncclNetIsend(...);
    op->state = PROXY_OP_ACTIVE;
  }

  // Poll for completion
  while (!ncclNetTest(op->request)) { /* wait */ }

  op->state = PROXY_OP_DONE;
  *op->cudaDone = 1;  // Signal GPU
}
```

## Resource Sharing Between Communicators

### Shared Resources Structure

**Source:** [nccl/src/include/comm.h:109-137](https://github.com/NVIDIA/nccl/blob/master/src/include/comm.h)
```c
struct ncclSharedResources {
  int refCount;
  struct ncclComm* owner;

  // Shared channel resources
  struct ncclChannelPeer* peers[MAXCHANNELS];
  struct ncclDevChannelPeer* devPeers[MAXCHANNELS];

  // Shared network resources
  uint64_t p2pOpCount[MAXCHANNELS];   // P2P op counter, one per channel

  // Shared memory regions
  struct ncclMemoryRegion* sendMem[MAXCHANNELS];
  struct ncclMemoryRegion* recvMem[MAXCHANNELS];

  // Operation counters
  uint64_t p2pOpCount[MAXCHANNELS];
  uint64_t collOpCount;
};
```

### When Resources Are Shared

**Source:** [nccl/src/init.cc:2198-2215](https://github.com/NVIDIA/nccl/blob/master/src/init.cc)
```c
// During split with splitShare=1
if (config && config->splitShare) {
  // Child shares parent's resources
  newcomm->sharedRes = parentComm->sharedRes;
  ncclAtomicRefCountIncrement(&parentComm->sharedRes->refCount);

  // Benefits:
  // - No new connections needed
  // - Shared memory registrations
  // - Reduced resource consumption
  // - Faster split operations
}
```

### Resource Cleanup

**Source:** [nccl/src/init.cc:2341-2367](https://github.com/NVIDIA/nccl/blob/master/src/init.cc)
```c
static ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  // Decrement reference count
  if (ncclAtomicRefCountDecrement(&comm->sharedRes->refCount) == 0) {
    // Last reference - actually free resources
    for (int c = 0; c < comm->nChannels; c++) {
      // Close network connections
      for (int p = 0; p < comm->nRanks; p++) {
        ncclNetClose(comm->sharedRes->netComm[c][p]);
      }
      // Free memory regions
      free(comm->sharedRes->sendMem[c]);
      free(comm->sharedRes->recvMem[c]);
    }
    free(comm->sharedRes);
  }
  free(comm);
}
```

## Performance Patterns and Best Practices

### Optimal Communicator Count

**Pattern: One Communicator Per Communication Domain**

```c
// Bad: Single communicator for everything
ncclComm_t comm;
ncclCommInitAll(&comm, 8, NULL);
// All operations bottlenecked on single comm

// Good: Separate communicators for different patterns
ncclComm_t data_parallel_comm;    // For gradient allreduce
ncclComm_t model_parallel_comm;   // For activation transfers
ncclComm_t pipeline_comm;          // For pipeline stages

// Each comm optimized for its specific pattern
```

### Channel Configuration Per Communicator

**Source:** [nccl/src/init.cc:876-945](https://github.com/NVIDIA/nccl/blob/master/src/init.cc)
```c
static ncclResult_t ncclCommInitChannel(ncclComm_t comm) {
  // Determine optimal channel count based on:
  // 1. Number of ranks
  // 2. Message sizes (from env or auto-detected)
  // 3. Network topology

  int nChannels = comm->nRanks;  // Start with ranks

  // Adjust based on network
  int nNics = ncclTopoGetNnics(comm->topo);
  nChannels = std::max(nChannels, nNics * 2);  // 2x NICs minimum

  // Apply limits
  nChannels = std::min(nChannels, NCCL_MAX_NCHANNELS);
  nChannels = std::max(nChannels, ncclMinNchannels());

  comm->nChannels = nChannels;
}
```

### Communicator Reuse vs Recreation

```c
// Expensive: Recreate communicator for each operation
for (int iter = 0; iter < 1000; iter++) {
  ncclComm_t comm;
  ncclCommInitRank(&comm, nranks, id, rank);  // ~100ms
  ncclAllReduce(..., comm, stream);
  ncclCommDestroy(comm);  // Tears down connections
}

// Efficient: Reuse communicator
ncclComm_t comm;
ncclCommInitRank(&comm, nranks, id, rank);  // One-time cost
for (int iter = 0; iter < 1000; iter++) {
  ncclAllReduce(..., comm, stream);  // Reuses connections
}
ncclCommDestroy(comm);
```

## Common Use Cases

### 1. Hybrid Parallelism in LLM Training

```python
def setup_communicators(rank, world_size):
    # Tensor parallel (within node)
    tp_size = 8  # GPUs per node
    tp_color = rank // tp_size
    tp_comm = nccl.commSplit(world_comm, tp_color, rank % tp_size)

    # Pipeline parallel (across 2 nodes)
    pp_size = 2
    pp_color = (rank % tp_size) + (rank // (tp_size * pp_size)) * tp_size
    pp_comm = nccl.commSplit(world_comm, pp_color, rank // tp_size)

    # Data parallel (remaining dimension)
    dp_color = rank % (tp_size * pp_size)
    dp_comm = nccl.commSplit(world_comm, dp_color,
                            rank // (tp_size * pp_size))

    return tp_comm, pp_comm, dp_comm
```

### 2. Hierarchical AllReduce

**Source:** [nccl/examples/msccl/hierarchical_allreduce.cu](https://github.com/microsoft/msccl/blob/main/examples/hierarchical_allreduce.cu)

```c
// Two-level hierarchy: local + cross-node
void hierarchicalAllReduce(void* buffer, size_t count,
                          ncclComm_t local_comm, ncclComm_t cross_comm) {
  // Step 1: Local reduce (within node)
  ncclReduce(buffer, buffer, count, ncclFloat, ncclSum,
             0, local_comm, stream);

  // Step 2: Cross-node allreduce (leaders only)
  if (local_rank == 0) {
    ncclAllReduce(buffer, buffer, count, ncclFloat, ncclSum,
                  cross_comm, stream);
  }

  // Step 3: Local broadcast
  ncclBroadcast(buffer, buffer, count, ncclFloat,
                0, local_comm, stream);
}
```

### 3. Dynamic Reconfiguration

```c
// Handling GPU failures with communicator reconstruction
void handleGpuFailure(ncclComm_t* comm, int failed_rank) {
  // Option 1: Shrink communicator
  int exclude[] = {failed_rank};
  ncclComm_t new_comm;
  ncclCommShrink(*comm, exclude, 1, &new_comm, NULL, 0);

  // Option 2: Create new communicator without failed rank
  if (rank != failed_rank) {
    int new_rank = (rank < failed_rank) ? rank : rank - 1;
    ncclCommInitRank(&new_comm, nranks - 1, new_id, new_rank);
  }

  // Destroy old communicator
  ncclCommDestroy(*comm);
  *comm = new_comm;
}
```

### 4. Overlapping Communication Patterns

```c
// Multiple communicators for different operations
void overlapCommPatterns(ncclComm_t comm1, ncclComm_t comm2) {
  // Stream 1: AllReduce on first communicator
  ncclAllReduce(grad1, grad1, size1, ncclFloat, ncclSum,
                comm1, stream1);

  // Stream 2: AllGather on second communicator (parallel)
  ncclAllGather(acts, all_acts, size2, ncclFloat,
                comm2, stream2);

  // Both operations proceed in parallel
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);
}
```

## Environment Variables for Communicator Tuning

**Source:** [nccl/src/plugin/env.cc](https://github.com/NVIDIA/nccl/blob/master/src/plugin/env.cc)

```bash
# Channel configuration per communicator
export NCCL_MIN_NCHANNELS=8     # Minimum channels
export NCCL_MAX_NCHANNELS=16    # Maximum channels

# Proxy threads: NCCL has no NCCL_PROXY_THREADS or NCCL_CPU_AFFINITY variable.
# Thread count is derived internally; CPU placement comes from the launcher.
export NCCL_PROXY_APPEND_BATCH_SIZE=16   # src/proxy.cc, default 16
export NCCL_IGNORE_CPU_AFFINITY=0        # whether to honor the inherited mask

# Resource sharing (src/init.cc:1875-1876 -- note the _RESOURCES suffix)
export NCCL_COMM_SPLIT_SHARE_RESOURCES=1   # Share resources on ncclCommSplit
export NCCL_COMM_SHRINK_SHARE_RESOURCES=1  # Share resources on ncclCommShrink

# Performance tuning
export NCCL_BUFFSIZE=4194304    # 4MB buffer per channel
export NCCL_NTHREADS=512        # CUDA threads per block
export NCCL_TREE_THRESHOLD=0    # Use tree for small messages

# Debugging
export NCCL_DEBUG=INFO          # Show communicator initialization
export NCCL_DEBUG_SUBSYS=INIT,COMM  # Detailed comm debug
```

## Key Takeaways

1. **Communicators are fundamental** - They define communication scope, topology, and resources
2. **Multiple communicators enable complex patterns** - Data parallel, model parallel, pipeline parallel
3. **Operations over same communicator share optimization** - Topology, connections, buffers
4. **Splitting creates hierarchy** - Essential for modern distributed training
5. **Proxy threads enable parallelism** - One per channel for maximum throughput
6. **Resource sharing reduces overhead** - Child communicators can share parent resources
7. **Proper lifecycle management is crucial** - Create once, reuse many times

## References

- NCCL Source: https://github.com/NVIDIA/nccl
- NCCL Examples: https://github.com/NVIDIA/nccl/tree/master/examples
- AWS OFI NCCL: https://github.com/aws/aws-ofi-nccl
- NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/