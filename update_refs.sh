#!/bin/bash

# Update all cross-references in markdown files
# Process in reverse order of old numbers to avoid conflicts

for file in *.md; do
    echo "Updating references in $file..."
    
    # Process from highest to lowest old numbers to avoid conflicts
    sed -i 's/19-kernel-efa-driver\.md/13-kernel-efa-driver.md/g' "$file"
    sed -i 's/18-dmabuf-gpu-memory\.md/16-dmabuf-gpu-memory.md/g' "$file"
    sed -i 's/17-rdma-core-and-verbs\.md/11-rdma-core-and-verbs.md/g' "$file"
    sed -i 's/16-mr-cache-implementation\.md/08-mr-cache-implementation.md/g' "$file"
    sed -i 's/15-freelist-allocator\.md/07-freelist-allocator.md/g' "$file"
    sed -i 's/14-topology-and-binding\.md/06-topology-and-binding.md/g' "$file"
    sed -i 's/13-ofi-plugin-protocols\.md/05-ofi-plugin-protocols.md/g' "$file"
    sed -i 's/11-optimization-opportunities\.md/21-optimization-opportunities.md/g' "$file"
    sed -i 's/10-optimizations\.md/22-optimizations.md/g' "$file"
    sed -i 's/09-efa-driver\.md/14-efa-driver.md/g' "$file"
    sed -i 's/08-rdma-memreg\.md/20-rdma-memreg.md/g' "$file"
    sed -i 's/07-threading-model\.md/19-threading-model.md/g' "$file"
    sed -i 's/06-efa-provider\.md/10-efa-provider.md/g' "$file"
    sed -i 's/05-libfabric-overview\.md/09-libfabric-overview.md/g' "$file"
done

echo "Cross-reference update complete!"
