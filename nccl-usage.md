# NCCL Usage: Communicators, Operations, and Threading

## Overview

This document provides a comprehensive guide to NCCL usage patterns, focusing on communicators as the fundamental abstraction for collective communication. We explore why communicators exist, how they're used, when to split or combine them, and how proxy threads enable parallel network operations. The communicator/enqueue/proxy structures and APIs described here were re-checked against NCCL v2.31.2-1 (the communicator, group, split/shrink and proxy interfaces below are unchanged in 2.31 unless noted).

> **Source lookups:** this document explains mechanism and records defaults, corrections and history. For current function bodies, call graphs and blast radius, query the code graph (`codegraph explore <symbol>`) rather than trusting a pasted copy here.

> **NCCL 2.31 source paths (corrected).** Several files moved in the 2.31 tree; use these paths:
> - `src/enqueue/enqueue.cc` — moved from `src/enqueue.cc`.
> - `src/plugin/env.cc` — moved from `src/env.cc`.
> - `src/nccl.h.in` — the public header template (not `src/include/nccl.h`).
> - Examples live at `docs/examples/` — not `examples/`.

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

> Source: `src/include/comm.h` — `struct ncclComm` (and `struct ncclSharedResources`). Use `codegraph explore ncclComm` for the current field layout.

The comm carries, among much else: its `rank`/`nRanks`/`cudaDev`, a `channels[MAXCHANNELS]` array (multiple channels per comm), per-peer info, the topology graph (`ncclTopoSystem`), a shared-resources pointer, and the proxy state used for network operations.

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

Each communicator maintains its own resources unless it explicitly shares them with a parent (see [Resource Sharing](#resource-sharing-between-communicators)). A non-sharing comm allocates its own `ncclSharedResources` — channels (up to `MAXCHANNELS`), proxy threads, network connections, memory buffers and work queues.

> Source: `src/init.cc` — shared-resource allocation in comm init. Use `codegraph explore ncclSharedResources` / `codegraph explore ncclCommInitRankDev`.

### 3. Topology Optimization

Communicators enable topology-aware communication patterns: NCCL computes optimal rings (for bandwidth-bound collectives) and trees (for latency-bound broadcasts/reductions) per channel from the detected topology.

> Source: `src/graph/topo.cc` — `ncclTopoCompute()`. Use `codegraph explore ncclTopoCompute` for the current body.

## Creating and Initializing Communicators

NCCL provides three primary methods for creating communicators. The signatures below are the subject of the discussion; see `src/nccl.h.in` for the authoritative declarations.

### Method 1: ncclCommInitAll — Single Process, Multiple GPUs

```c
/* Creates a clique of communicators (single process version) */
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
```

One process manages all GPUs; you get one communicator handle per device. Because a single thread drives all devices, `ncclCommInitAll` internally uses group semantics so per-device init cannot deadlock.

```c
// Single process manages all GPUs
ncclComm_t comms[8];
int devices[] = {0, 1, 2, 3, 4, 5, 6, 7};
NCCLCHECK(ncclCommInitAll(comms, 8, devices));

// Now we have 8 communicator handles, one per GPU
for (int i = 0; i < 8; i++) {
  cudaSetDevice(devices[i]);
  ncclAllReduce(sendbuf[i], recvbuf[i], count, ncclFloat,
                ncclSum, comms[i], streams[i]);
}
```
> Example: `docs/examples/01_communicators/01_multiple_devices_single_process/main.cc`.

### Method 2: ncclCommInitRank — Multi-Process/Thread

```c
/* Creates a new communicator (multi thread/process version) */
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
                              ncclUniqueId commId, int rank);
```

Each rank calls `ncclCommInitRank` with a shared `ncclUniqueId` (distributed out-of-band, e.g. via MPI/`torch.distributed`) and its own `rank`. Initialization bootstraps peer connections, exchanges peer info, computes the topology, sets up channels and proxy threads, and establishes transport (P2P/network) connections.

> Source: `src/init.cc` — `ncclCommInitRank` / `ncclCommInitRankDev`. Use `codegraph explore ncclCommInitRank` for the current body and call path.

### Method 3: ncclCommSplit — Creating Sub-communicators

```c
/* Creates one or more communicators from an existing one */
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key,
                           ncclComm_t *newcomm, ncclConfig_t* config);
```

Ranks that pass the same `color` join the same new communicator; `key` orders ranks within it. See [Splitting and Combining Communicators](#splitting-and-combining-communicators).

## Operations Over Communicators

All NCCL collective operations require a communicator parameter. The signatures below are the subject of the discussion; see `src/nccl.h.in` for the authoritative declarations.

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

Enqueue takes the communicator's rank/nRanks, selects an algorithm and protocol from the comm's topology and config, splits the work across the comm's channels, and signals the comm's proxy threads to drive the network side. This is why operations on the *same* comm reuse its topology, connections and buffers.

> Source: `src/enqueue/enqueue.cc` — `ncclEnqueueCheck()` and the algorithm/protocol selection helpers. Use `codegraph explore ncclEnqueueCheck`.

## Multiple Operations and Group Semantics

### Why Multiple Operations Over Same Communicator?

Multiple operations enable complex communication patterns. All operations issued on the same comm share its resources, topology and connections:

```c
// Gradient aggregation across layers, then broadcast updated params —
// all on the same comm, so all share its resources and topology.

ncclAllReduce(grad_layer1, grad_layer1, layer1_size, ncclFloat32,
              ncclSum, comm, stream);
ncclAllReduce(grad_layer2, grad_layer2, layer2_size, ncclFloat32,
              ncclSum, comm, stream);
ncclBroadcast(params, params, param_size, ncclFloat32,
              root, comm, stream);
```

### Group Operations — Avoiding Deadlocks

When managing multiple GPUs from a single thread — and because NCCL collective calls may perform inter-CPU synchronization — calls for different ranks/devices must be "grouped" into a single logical call. `ncclGroupStart()` defers execution; `ncclGroupEnd()` launches all enqueued operations across all communicators at once.

```c
/* Group semantics */
ncclResult_t ncclGroupStart();
ncclResult_t ncclGroupEnd();
```

> Source: `src/group.cc` — `ncclGroupStart` / `ncclGroupEnd`. Use `codegraph explore ncclGroupEnd` for the current launch path.

**Usage pattern — multi-GPU operations without deadlock:**
```c
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

Splitting enables hierarchical communication patterns essential for modern distributed training (e.g. combining data parallelism with model parallelism). Ranks with the same `color` join the same new communicator; `key` determines rank order within it. Optionally the child can share the parent's resources (see below and `NCCL_COMM_SPLIT_SHARE_RESOURCES`).

> Source: `src/init.cc` — `ncclCommSplit`. Use `codegraph explore ncclCommSplit` for the current body.

**Usage pattern — data + model parallelism via split:**
```python
# 16 GPUs (2 nodes x 8), 4-way model parallel, 4-way data parallel.
def create_process_groups(rank):
    # Model parallel: ranks [0..3], [4..7], [8..11], [12..15]
    model_parallel_comm = nccl.commSplit(world_comm, rank // 4, rank % 4)

    # Data parallel: ranks [0,4,8,12], [1,5,9,13], ...
    data_parallel_comm = nccl.commSplit(world_comm, rank % 4, rank // 4)
    return model_parallel_comm, data_parallel_comm
```

### Growing and Shrinking Communicators

The signatures below are the subject of the discussion; see `src/nccl.h.in` for the authoritative declarations.

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

**Usage pattern — elastic training:** grow to absorb newly available GPUs; shrink to drop failed ranks (pass the exclude list).

## Proxy Threads and Parallelism

### What Are Proxy Threads?

Proxy threads are NCCL's mechanism for handling network I/O independently from CUDA kernels. A proxy services channels: it polls for GPU-signalled work, issues network sends/receives through the net plugin (`ncclNetIsend`/`ncclNetIrecv`), then polls for completions and signals the GPU when an operation finishes. This keeps the network progress engine off the CUDA kernel's critical path.

> Source: `src/proxy.cc` — `ncclProxyService()` (the proxy progress loop) and the `ncclProxyOp` state machine. Use `codegraph explore ncclProxyService`.

### Why Multiple Proxy Threads?

Distributing channels across proxy threads lets network operations proceed in parallel instead of serially, saturating multiple CPU cores and preserving NUMA locality (a proxy for a given GPU's channels is placed on CPUs in the same NUMA node as that GPU). The result is higher aggregate message-injection rate.

**1. Parallel network operations**
```
Single Proxy Thread:              Multiple Proxy Threads:
Thread 0:                         Thread 0: Send Ch0 Data
  Send Ch0 Data (10ms)           Thread 1: Send Ch1 Data  (parallel)
  Send Ch1 Data (10ms)           Thread 2: Send Ch2 Data  (parallel)
  Send Ch2 Data (10ms)           Thread 3: Send Ch3 Data  (parallel)
Total: 30ms                      Total: 10ms
```

**2. NUMA locality**
```
Node topology:
NUMA 0: CPUs 0-63,  GPUs 0-3, NICs 0-1
NUMA 1: CPUs 64-127, GPUs 4-7, NICs 2-3

Proxy thread assignment:
GPU 0, Ch 0-3 → CPUs 0-3   (same NUMA as GPU)
GPU 4, Ch 0-3 → CPUs 64-67 (same NUMA as GPU)

Benefits: Reduced memory latency, better cache utilization
```

> **Thread count and CPU placement are not NCCL env vars.** There is **no** `NCCL_PROXY_THREADS` and **no** `NCCL_CPU_AFFINITY`. Proxy thread count is derived internally from the channel/topology layout; CPU placement is a *launcher* concern — pin with `mpirun --bind-to none` / `srun --cpu-bind=none` and let NCCL place proxies, rather than expecting a NCCL variable. The real proxy/affinity knobs are `NCCL_PROXY_APPEND_BATCH_SIZE` (default **16**) and `NCCL_IGNORE_CPU_AFFINITY` (see [Environment Variables](#environment-variables-for-communicator-tuning)).

## Resource Sharing Between Communicators

### Shared Resources Structure

`ncclSharedResources` is reference-counted and holds the resources a set of related comms can share: per-channel peer state, network resources, shared send/recv memory regions, and operation counters. Multiple comms pointing at the same `ncclSharedResources` avoid re-establishing connections and re-registering memory.

> Source: `src/include/comm.h` — `struct ncclSharedResources`. Use `codegraph explore ncclSharedResources`.

### When Resources Are Shared

During a split (or shrink) with sharing enabled, the child points at the parent's `ncclSharedResources` and bumps its refcount instead of allocating fresh resources. Benefits: no new connections, shared memory registrations, lower resource consumption, and faster split/shrink. Sharing is controlled per-call via `ncclConfig_t` and globally via `NCCL_COMM_SPLIT_SHARE_RESOURCES` / `NCCL_COMM_SHRINK_SHARE_RESOURCES`.

> Source: `src/init.cc` — split/shrink resource-sharing path and `ncclCommDestroy` (refcount decrement frees resources on the last reference). Use `codegraph explore ncclCommDestroy`.

## Performance Patterns and Best Practices

### Optimal Communicator Count

**Pattern: one communicator per communication domain.** A single comm for everything bottlenecks all traffic on one resource set; separate comms (data-parallel, model-parallel, pipeline) each get topology and connections tuned for their pattern and can progress in parallel.

### Channel Configuration Per Communicator

NCCL picks a channel count per comm from the number of ranks, message sizes, and network topology (roughly `max(nRanks, 2 x nNics)`), then clamps to `[ncclMinNchannels(), NCCL_MAX_NCHANNELS]`. Override the bounds with `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS`.

> Source: `src/init.cc` — channel-count computation during comm init. Use `codegraph explore ncclCommInitRankDev`.

### Communicator Reuse vs Recreation

Communicator creation is expensive (bootstrap, topology compute, connection setup — on the order of ~100 ms). **Create once, reuse many times.** Recreating a comm inside a training loop tears down and re-establishes connections every iteration.

```c
// Efficient: reuse the communicator across iterations
ncclComm_t comm;
ncclCommInitRank(&comm, nranks, id, rank);  // one-time cost
for (int iter = 0; iter < 1000; iter++) {
  ncclAllReduce(..., comm, stream);          // reuses connections
}
ncclCommDestroy(comm);
```

## Common Use Cases

### 1. Hybrid Parallelism in LLM Training

Split the world communicator along each parallelism axis (tensor / pipeline / data), using `color` to select the group and `key` to order ranks within it:

```python
def setup_communicators(rank, world_comm):
    tp_size = 8  # GPUs per node
    tp_comm = nccl.commSplit(world_comm, rank // tp_size, rank % tp_size)

    pp_size = 2
    pp_color = (rank % tp_size) + (rank // (tp_size * pp_size)) * tp_size
    pp_comm = nccl.commSplit(world_comm, pp_color, rank // tp_size)

    dp_comm = nccl.commSplit(world_comm, rank % (tp_size * pp_size),
                             rank // (tp_size * pp_size))
    return tp_comm, pp_comm, dp_comm
```

### 2. Hierarchical AllReduce

Two-level hierarchy: local reduce within a node, cross-node allreduce among node leaders, then local broadcast — using a `local_comm` and a `cross_comm`:

```c
void hierarchicalAllReduce(void* buffer, size_t count,
                          ncclComm_t local_comm, ncclComm_t cross_comm) {
  ncclReduce(buffer, buffer, count, ncclFloat, ncclSum, 0, local_comm, stream);
  if (local_rank == 0) {
    ncclAllReduce(buffer, buffer, count, ncclFloat, ncclSum, cross_comm, stream);
  }
  ncclBroadcast(buffer, buffer, count, ncclFloat, 0, local_comm, stream);
}
```

### 3. Dynamic Reconfiguration

On GPU failure, either `ncclCommShrink` the existing comm (passing the failed ranks as the exclude list) or build a fresh comm without the failed rank, then destroy the old one.

### 4. Overlapping Communication Patterns

Issue independent collectives on different comms and different streams so they progress in parallel (e.g. an AllReduce on `comm1`/`stream1` overlapped with an AllGather on `comm2`/`stream2`), then synchronize both streams.

## Environment Variables for Communicator Tuning

> Source: `src/plugin/env.cc` (env parsing; moved from `src/env.cc` in 2.31). Use `codegraph explore` for where a given variable is read.

```bash
# Channel configuration per communicator
export NCCL_MIN_NCHANNELS=8     # Minimum channels
export NCCL_MAX_NCHANNELS=16    # Maximum channels

# Proxy / affinity: NCCL has NO NCCL_PROXY_THREADS and NO NCCL_CPU_AFFINITY.
# Proxy thread count is derived internally; CPU placement comes from the launcher
# (mpirun --bind-to none / srun --cpu-bind=none).
export NCCL_PROXY_APPEND_BATCH_SIZE=16   # default 16 (src/proxy.cc)
export NCCL_IGNORE_CPU_AFFINITY=0        # whether to ignore the inherited CPU mask

# Resource sharing (note the _RESOURCES suffix)
export NCCL_COMM_SPLIT_SHARE_RESOURCES=1    # share resources on ncclCommSplit
export NCCL_COMM_SHRINK_SHARE_RESOURCES=1   # share resources on ncclCommShrink

# Performance tuning
export NCCL_BUFFSIZE=4194304    # 4MB buffer per channel
export NCCL_NTHREADS=512        # CUDA threads per block
export NCCL_TREE_THRESHOLD=0    # Use tree for small messages

# Debugging
export NCCL_DEBUG=INFO              # Show communicator initialization
export NCCL_DEBUG_SUBSYS=INIT,COMM  # Detailed comm debug
```

## Key Takeaways

1. **Communicators are fundamental** — they define communication scope, topology, and resources.
2. **Multiple communicators enable complex patterns** — data parallel, model parallel, pipeline parallel.
3. **Operations over the same communicator share optimization** — topology, connections, buffers.
4. **Splitting creates hierarchy** — essential for modern distributed training.
5. **Proxy threads enable parallelism** — distributed across channels for throughput; thread count is internal, CPU placement is a launcher concern.
6. **Resource sharing reduces overhead** — child communicators can share parent resources via the `*_SHARE_RESOURCES` knobs.
7. **Proper lifecycle management is crucial** — create once, reuse many times.

## References

- NCCL Source: https://github.com/NVIDIA/nccl
- NCCL Examples: https://github.com/NVIDIA/nccl/tree/master/docs/examples
- AWS OFI NCCL: https://github.com/aws/aws-ofi-nccl
- NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
