#!/bin/bash
# Renumber documentation files to logical progression

# Create mapping (old -> new)
declare -A renames=(
    # LAYER 1: Overview & Foundation (00-03) - keep same
    ["00-overview.md"]="00-overview.md"
    ["01-nccl-core.md"]="01-nccl-core.md"
    ["02-nccl-collectives.md"]="02-nccl-collectives.md"
    ["03-nccl-datapath.md"]="03-nccl-datapath.md"
    
    # LAYER 2: OFI Plugin (04-08) - PRIMARY DEVELOPMENT AREA
    ["04-ofi-plugin.md"]="04-ofi-plugin.md"
    ["13-ofi-plugin-protocols.md"]="05-ofi-plugin-protocols.md"
    ["14-topology-and-binding.md"]="06-topology-and-binding.md"
    ["15-freelist-allocator.md"]="07-freelist-allocator.md"
    ["16-mr-cache-implementation.md"]="08-mr-cache-implementation.md"
    
    # LAYER 3: Libfabric (09-10)
    ["05-libfabric-overview.md"]="09-libfabric-overview.md"
    ["06-efa-provider.md"]="10-efa-provider.md"
    
    # LAYER 4: rdma-core (11-12)
    ["17-rdma-core-and-verbs.md"]="11-rdma-core-and-verbs.md"
    ["12-lkey-rkey-explained.md"]="12-lkey-rkey-explained.md"
    
    # LAYER 5: Kernel & Hardware (13-14)
    ["19-kernel-efa-driver.md"]="13-kernel-efa-driver.md"
    ["09-efa-driver.md"]="14-efa-driver.md"
    
    # LAYER 6: GPU/Accelerator Memory (15-17)
    ["18-dmabuf-gpu-memory.md"]="15-dmabuf-gpu-memory.md"
    ["16-neuron-memory.md"]="16-neuron-memory.md"
    ["17-rocm-memory.md"]="17-rocm-memory.md"
    
    # LAYER 7: Cross-Cutting Concerns (18-19)
    ["07-threading-model.md"]="18-threading-model.md"
    ["08-rdma-memreg.md"]="19-rdma-memreg.md"
    
    # LAYER 8: Performance & Optimization (20-21)
    ["10-optimizations.md"]="20-optimizations.md"
    ["11-optimization-opportunities.md"]="21-optimization-opportunities.md"
)

# Print what we're going to do
echo "=== Renumbering Plan ==="
for old in "${!renames[@]}"; do
    new="${renames[$old]}"
    if [ "$old" != "$new" ]; then
        echo "$old -> $new"
    fi
done

echo ""
read -p "Proceed with renaming? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted"
    exit 1
fi

# Rename to temp names first (avoid conflicts)
for old in "${!renames[@]}"; do
    new="${renames[$old]}"
    if [ "$old" != "$new" ] && [ -f "$old" ]; then
        mv "$old" "tmp_$old"
    fi
done

# Rename from temp to final
for old in "${!renames[@]}"; do
    new="${renames[$old]}"
    if [ "$old" != "$new" ] && [ -f "tmp_$old" ]; then
        mv "tmp_$old" "$new"
        echo "Renamed: $old -> $new"
    fi
done

echo ""
echo "=== Renaming complete! ==="
echo "Files now in logical order:"
ls -1 [0-2][0-9]-*.md
