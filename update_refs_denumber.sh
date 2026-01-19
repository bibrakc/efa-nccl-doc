#!/bin/bash
# Update all cross-references to remove numbering

for file in *.md; do
    echo "Updating $file..."
    sed -i 's/00-overview\.md/overview.md/g' "$file"
    sed -i 's/01-nccl-core\.md/nccl-core.md/g' "$file"
    sed -i 's/02-nccl-collectives\.md/nccl-collectives.md/g' "$file"
    sed -i 's/03-nccl-datapath\.md/nccl-datapath.md/g' "$file"
    sed -i 's/04-ofi-plugin\.md/ofi-plugin.md/g' "$file"
    sed -i 's/05-ofi-plugin-protocols\.md/ofi-plugin-protocols.md/g' "$file"
    sed -i 's/06-topology-and-binding\.md/topology-and-binding.md/g' "$file"
    sed -i 's/07-freelist-allocator\.md/freelist-allocator.md/g' "$file"
    sed -i 's/08-mr-cache-implementation\.md/mr-cache-implementation.md/g' "$file"
    sed -i 's/09-libfabric-overview\.md/libfabric-overview.md/g' "$file"
    sed -i 's/10-efa-provider\.md/efa-provider.md/g' "$file"
    sed -i 's/11-rdma-core-and-verbs\.md/rdma-core-and-verbs.md/g' "$file"
    sed -i 's/12-lkey-rkey-explained\.md/lkey-rkey-explained.md/g' "$file"
    sed -i 's/13-kernel-efa-driver\.md/kernel-efa-driver.md/g' "$file"
    sed -i 's/14-efa-driver\.md/efa-driver.md/g' "$file"
    sed -i 's/15-cuda-memory\.md/cuda-memory.md/g' "$file"
    sed -i 's/16-dmabuf-gpu-memory\.md/dmabuf-gpu-memory.md/g' "$file"
    sed -i 's/17-neuron-memory\.md/neuron-memory.md/g' "$file"
    sed -i 's/18-rocm-memory\.md/rocm-memory.md/g' "$file"
    sed -i 's/19-threading-model\.md/threading-model.md/g' "$file"
    sed -i 's/20-rdma-memreg\.md/rdma-memreg.md/g' "$file"
    sed -i 's/21-optimization-opportunities\.md/optimization-opportunities.md/g' "$file"
    sed -i 's/22-optimizations\.md/optimizations.md/g' "$file"
done
