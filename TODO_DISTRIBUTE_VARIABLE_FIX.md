# TODO: Fix PyTorch distribute_variable to shard tensors immediately

## Problem Statement
The PyTorch backend's `distribute_variable()` function only tags tensors with metadata (`_distributed_layout` attribute) without actually partitioning them. This causes OOM errors on larger models since full tensors stay in GPU memory.

JAX backend automatically partitions tensors using `jax.device_put` with sharding layout.

## Solution
Modify `distribute_variable()` in `keras/src/backend/torch/distribution_lib.py` to:
1. Identify the layout early when called
2. Initialize tensors on CPU or Meta device to avoid GPU memory
3. Use `torch.split()` or `torch.chunk()` to partition the tensor
4. Move only the required shard to the specific target GPU
5. Store the full layout information for potential All-Gather operations

## Implementation Plan

### Step 1: Create helper functions for tensor sharding
- [x] Add `_get_current_rank()` - Get current distributed rank
- [x] Add `_get_current_device()` - Get current device for this rank
- [x] Add `_shard_tensor()` - Partition tensor and return specific shard

### Step 2: Modify `distribute_variable()` function
- [x] Extract axes and device mesh from layout
- [x] Check if sharding is needed (any axis is not None)
- [x] Create full tensor on CPU if not already there
- [x] Partition tensor using torch.split/torch.chunk
- [x] Return only the shard for current rank/device
- [x] Store full layout info for All-Gather operations

### Step 3: Update debug logging
- [x] Add debug output for sharding decisions
- [x] Log tensor sizes before/after partitioning
- [x] Log device placement of shards

### Step 4: Add helper function for All-Gather (for getting full tensor)
- [x] Add `all_gather_variable()` function
- [ ] Used when full tensor is needed (e.g., for saving/checkpointing)

## Files to Modify
- `keras/src/backend/torch/distribution_lib.py`

## Testing
- Verify tensors are sharded correctly on GPU
- Verify OOM is avoided for large models
- Verify All-Gather works to reconstruct full tensor

## Progress
Status: Ready to implement

