# Understanding lkey and rkey (Local Key and Remote Key)

## Overview

**lkey** (Local Key) and **rkey** (Remote Key) are protection keys used in RDMA operations to control access to registered memory regions. They are security and validation tokens that allow NICs to verify they're accessing the correct memory with proper permissions.

## What Are They?

### Definition

```c
struct ibv_mr {
  void *addr;        // Virtual address of registered memory
  size_t length;     // Length of region
  uint32_t lkey;     // Local key (L-Key)
  uint32_t rkey;     // Remote key (R-Key)
};
```

**lkey (Local Key)**: Token used for **local** operations
- Used when **this** NIC accesses **this** node's memory
- Send operations, receive operations on local side

**rkey (Remote Key)**: Token used for **remote** operations
- Shared with **remote** node
- Used when **remote** NIC accesses **this** node's memory
- RDMA Read/Write from remote side

### Generation

From [08-rdma-memreg.md](doc/08-rdma-memreg.md):

```c
// When memory is registered
fi_mr_reg(domain, buffer, size, access_flags, 0, 0, 0, &mr, NULL);

// Driver generates keys
mr->lkey = generate_key();  // Local key
mr->rkey = generate_key();  // Remote key (often same as lkey)
```

**On EFA:**
```c
// EFA driver (simplified)
mr->lkey = mr->key;  // Same value
mr->rkey = mr->key;  // Same value
```

**On EFA, lkey == rkey** (same value for both)

On some other fabrics (InfiniBand), they might be different or have additional encoding.

## How They Work

### The Problem They Solve

Without keys, any RDMA operation could write to any memory:

```
Bad scenario (no protection):
  Remote NIC: Write to 0x1000  ← Could be any memory!
  Local NIC: "OK, writing..."  ← No validation!
  Result: Security hole, corruption possible
```

With keys:

```
Good scenario (with keys):
  Remote NIC: Write to 0x1000 with key=0x12345
  Local NIC: Check if key 0x12345 is valid for address 0x1000
             ├─ Valid? Proceed with write
             └─ Invalid? Reject with error
```

### Key as Access Token

Think of keys like a ticket system:

```
Memory Registration = Issuing a Ticket
┌─────────────────────────────────────┐
│ Buffer: 0x1000 - 0x2000            │
│ Ticket (lkey): 0x4567              │
│ Ticket (rkey): 0x4567              │
│ Permissions: READ | WRITE          │
└─────────────────────────────────────┘

Using the Ticket:
  "I want to access 0x1500 with ticket 0x4567"
     ↓
  NIC checks: Is 0x4567 valid for 0x1500?
     ├─ YES → Allow access
     └─ NO  → Reject (error)
```

## When Each Key Is Used

### Local Operations (lkey)

```c
// Send operation (local NIC reads local memory)
fi_send(ep, local_buffer, size, fi_mr_desc(mr), dest, ctx);
                                     ↑
                        Contains lkey internally

// Receive operation (local NIC writes to local memory)
fi_recv(ep, local_buffer, size, fi_mr_desc(mr), src, ctx);
                                     ↑
                        Contains lkey internally
```

**What happens:**
```
Application: Send from buffer 0x1000
    ↓
Libfabric: Get descriptor (contains lkey)
    ↓
EFA Driver: Build send request
    ├─ Source address: 0x1000
    ├─ lkey: 0x4567
    └─ Post to NIC
    ↓
NIC: Validate lkey 0x4567 for address 0x1000
    ├─ Valid? DMA read from 0x1000
    └─ Invalid? Error
    ↓
Data sent on network
```

### Remote Operations (rkey)

```c
// RDMA Write (remote NIC writes to remote memory)
fi_write(ep, local_buffer, size, desc,
         dest_addr,
         remote_addr,    // Address on REMOTE node
         remote_key,     // rkey from REMOTE node
         ctx);

// RDMA Read (remote NIC reads from remote memory)
fi_read(ep, local_buffer, size, desc,
        src_addr,
        remote_addr,     // Address on REMOTE node
        remote_key,      // rkey from REMOTE node
        ctx);
```

**What happens (RDMA Write):**
```
Node A (Sender)                     Node B (Receiver)
──────────────────────────────────────────────────────

1. Have Node B's info:
   remote_addr = 0x5000
   remote_key = 0x9999

2. fi_write(local_buf,
            remote_addr=0x5000,
            remote_key=0x9999)

3. NIC sends packet:
   ┌──────────────────┐
   │ Target: 0x5000   │
   │ Key: 0x9999      │          ──────────→
   │ Data: [...]      │
   └──────────────────┘

                                    4. Node B's NIC receives packet
                                       ├─ Target address: 0x5000
                                       ├─ Key: 0x9999
                                       └─ Validate!

                                    5. Validation:
                                       Is 0x9999 valid for 0x5000?
                                       ├─ YES → DMA write to 0x5000
                                       └─ NO → Reject packet

                                    6. Silent completion
                                       (no CPU involved!)
```

## Key Exchange Protocol

Since RDMA operations need the remote key, nodes must exchange this information:

### Setup Phase (Out-of-Band)

```c
// Each node does this:

// 1. Register receive buffer
fi_mr_reg(domain, my_recv_buf, size, FI_REMOTE_WRITE, 0, 0, 0, &mr, NULL);

uint64_t my_addr = (uint64_t)my_recv_buf;
uint64_t my_key = fi_mr_key(mr);  // Get rkey to share

// 2. Exchange via control message (using fi_send/fi_recv)
struct handshake_msg {
  uint64_t recv_buf_addr;  // Where I want to receive data
  uint64_t recv_buf_key;   // rkey to access my buffer
  size_t recv_buf_size;
};

// Send my info to peer
handshake_msg my_info = {
  .recv_buf_addr = my_addr,
  .recv_buf_key = my_key,
  .recv_buf_size = size,
};
fi_send(ep, &my_info, sizeof(my_info), ...);

// Receive peer's info
handshake_msg peer_info;
fi_recv(ep, &peer_info, sizeof(peer_info), ...);

// Store peer's info for RDMA operations
conn->remote_addr = peer_info.recv_buf_addr;
conn->remote_key = peer_info.recv_buf_key;
```

### Data Phase (RDMA)

```c
// Now can do RDMA write to peer's buffer
fi_write(ep, my_send_buf, size, desc,
         peer_addr,
         conn->remote_addr,   // From handshake
         conn->remote_key,    // From handshake
         ctx);
```

## EFA Provider Implementation

### Key Generation in EFA Driver

From [09-efa-driver.md](doc/09-efa-driver.md):

```c
// EFA kernel driver
int efa_reg_mr(struct ib_pd *ibpd, struct ib_mr *ibmr,
               u64 start, u64 length, u64 virt_addr,
               int access_flags)
{
  // ... pin pages, create DMA mapping ...

  // Register with EFA device
  struct efa_com_reg_mr_params params = {
    .pd = to_epd(ibpd)->pdn,
    .iova = virt_addr,
    .mr_length = length,
    .page_num = npages,
  };

  // Device generates a key
  err = efa_com_register_mr(dev->edev, &params, &mr->key);

  // Both lkey and rkey are the same on EFA
  mr->lkey = mr->key;
  mr->rkey = mr->key;

  return 0;
}
```

**EFA Key Characteristics:**
- Single key value for both local and remote
- Generated by hardware during registration
- Unique per memory region
- Used for validation at NIC level

### Key Validation in Hardware

```
EFA NIC Hardware Lookup Table:
┌────────────────────────────────────────┐
│ Key     │ DMA Address │ Permissions   │
├────────────────────────────────────────┤
│ 0x4567  │ 0x10000     │ READ|WRITE    │
│ 0x9999  │ 0x20000     │ REMOTE_WRITE  │
│ 0xABCD  │ 0x30000     │ REMOTE_READ   │
└────────────────────────────────────────┘

On RDMA Write packet arrival:
  Packet says: "Write to address X with key K"
  ↓
  NIC looks up key K in table
  ↓
  Validates:
    ├─ Does key K exist?
    ├─ Does key K allow access to address X?
    ├─ Does key K have REMOTE_WRITE permission?
    └─ Is address X within registered region?
  ↓
  If all checks pass: Execute DMA
  Else: Drop packet, generate error
```

## Practical Example: NCCL Data Transfer

### Scenario: Node A sends 1GB to Node B using RDMA

```c
// === NODE B (Receiver) ===

// 1. Allocate and register receive buffer
void *recv_buf;
cudaMalloc(&recv_buf, 1GB);

struct fid_mr *mr;
fi_mr_reg(domain, recv_buf, 1GB,
          FI_REMOTE_WRITE,  // Allow remote to write
          0, 0, FI_HMEM_CUDA, &mr, NULL);

uint64_t my_rkey = fi_mr_key(mr);  // e.g., 0xAABBCCDD

// 2. Send rkey and address to Node A (via control channel)
struct rdma_info {
  uint64_t addr = (uint64_t)recv_buf;  // 0x7f1234567000
  uint64_t rkey = my_rkey;              // 0xAABBCCDD
};
fi_send(control_ep, &rdma_info, sizeof(rdma_info), ...);


// === NODE A (Sender) ===

// 3. Receive Node B's info
struct rdma_info peer_info;
fi_recv(control_ep, &peer_info, sizeof(peer_info), ...);

// Now have:
//   peer_info.addr = 0x7f1234567000
//   peer_info.rkey = 0xAABBCCDD

// 4. Do RDMA write
void *send_buf;
cudaMalloc(&send_buf, 1GB);
// ... fill with data ...

fi_write(data_ep,
         send_buf,              // Local buffer (Node A)
         1GB,                   // Size
         my_desc,               // My lkey (for local access)
         node_b_addr,           // Node B's fi_addr_t
         peer_info.addr,        // Remote address (0x7f1234567000)
         peer_info.rkey,        // Remote key (0xAABBCCDD)
         ctx);

// Node A's NIC sends packet to Node B with:
//   - Target address: 0x7f1234567000
//   - Validation key: 0xAABBCCDD
//   - Data: 1GB


// === NODE B (Receiver - Hardware) ===

// 5. Node B's EFA NIC receives packet
//    Validates:
//      Key 0xAABBCCDD is valid? YES (registered earlier)
//      Key allows REMOTE_WRITE? YES (registered with FI_REMOTE_WRITE)
//      Address 0x7f1234567000 is within registered region? YES
//
//    Validation passes → DMA write directly to GPU memory
//    No CPU involvement!


// === NODE A (Sender) ===

// 6. Poll for completion
fi_cq_read(cq, &entry, 1);  // Write completed

// Node B has data, never touched CPU!
```

## Security and Protection

### What Keys Protect Against

1. **Incorrect Address Access**
   ```
   Buffer registered: 0x1000 - 0x2000, key=0xAAAA

   RDMA Write to 0x3000 with key=0xAAAA
   ↓
   REJECTED: Address 0x3000 not in range for key 0xAAAA
   ```

2. **Use-After-Free**
   ```
   Register buffer, get key=0xBBBB
   Deregister buffer (key invalidated)

   Later: RDMA Write with key=0xBBBB
   ↓
   REJECTED: Key 0xBBBB no longer valid
   ```

3. **Unauthorized Access**
   ```
   Buffer registered with: FI_REMOTE_READ only

   RDMA Write with correct key
   ↓
   REJECTED: Key doesn't have WRITE permission
   ```

### What Keys DON'T Protect Against

- **If attacker has valid key**: Full access to that region
- **Key prediction**: If keys are sequential/predictable (hardware uses random)
- **Network interception**: Keys sent over network could be captured

**Security relies on:**
- Control channel security (key exchange)
- Network isolation (trusted fabric)
- Hardware key generation (unpredictable)

## Performance Implications

### Key Lookup Overhead

```
Fast Path (NIC hardware):
  Packet arrives
  ↓
  Key lookup in NIC table (hardware)  ← ~10-20 ns
  ↓
  Validation (hardware)               ← ~10-20 ns
  ↓
  DMA operation

Total overhead: ~20-40 ns (negligible!)
```

**Keys are designed for hardware validation** - very fast!

### Key Caching

The memory registration cache (discussed earlier) caches:
- Memory region
- lkey/rkey pair
- Memory descriptor

```c
struct mr_cache_entry {
  void *addr;
  size_t size;
  struct fid_mr *mr;  // Contains lkey/rkey
  void *desc;         // Descriptor with lkey encoded
};
```

Cache hit avoids:
- Re-registration (~500 μs)
- Key generation (free)
- Key lookup (< 1 μs)

## Summary

### Key Points

| Aspect | lkey | rkey |
|--------|------|------|
| **Used By** | Local NIC | Remote NIC |
| **Operations** | Send, Recv | RDMA Read, RDMA Write |
| **Scope** | Local memory access | Remote memory access |
| **Shared?** | No (stays local) | Yes (sent to peer) |
| **On EFA** | Same value as rkey | Same value as lkey |

### Memory Registration Process

```
1. Application registers memory
   ↓
2. Driver pins pages, creates IOMMU mapping
   ↓
3. Hardware/driver generates key(s)
   ↓
4. Keys returned to application
   ↓
5. lkey used for local ops (send/recv)
   rkey shared with peers for RDMA ops
```

### RDMA Operation Flow

```
Setup (Once):
  1. Register memory → Get lkey, rkey
  2. Exchange rkey with peer (via control message)

Data Transfer (Many times):
  3. RDMA Write/Read using peer's rkey
  4. Hardware validates key
  5. DMA operation proceeds
```

### Why Two Keys?

Historical reasons and flexibility:
- **Conceptually**: Local vs remote access
- **Practically (EFA)**: Same value, different purpose
- **Other fabrics**: May have different encoding/protections

**On EFA**: `lkey == rkey` (implementation detail)

**Purpose**: Security and validation tokens for RDMA operations, enforced at hardware level with minimal overhead (~20-40 ns).

---

## Further Reading

- [05-libfabric-overview.md](doc/05-libfabric-overview.md) - Memory registration in libfabric
- [08-rdma-memreg.md](doc/08-rdma-memreg.md) - Complete RDMA and memory registration details
- [09-efa-driver.md](doc/09-efa-driver.md) - EFA driver implementation of key generation
