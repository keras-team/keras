# Backend-Agnostic Distribution Implementation Plan

## Overview
Make Keras distribution strategies (DataParallel, ModelParallel) work with PyTorch backend by adding a backend-specific `distribution_lib.py` implementation that delegates tensor operations to PyTorch's distributed primitives.

## Files to Create/Modify

### 1. Create: `keras/src/backend/torch/distribution_lib.py`
**Purpose**: Backend-specific implementation for PyTorch distribution operations

**Key Functions**:
- `list_devices(device_type=None)` - Return available CUDA devices
- `get_device_count(device_type=None)` - Count available GPUs
- `distribute_tensor(tensor, layout)` - Apply layout/sharding to tensor
- `distribute_variable(value, layout)` - Distribute variable according to layout
- `_to_backend_mesh(device_mesh)` - Convert DeviceMesh to PyTorch representation
- `_to_backend_layout(tensor_layout)` - Convert TensorLayout to PyTorch representation
- `initialize(job_addresses, num_processes, process_id)` - Setup torch.distributed
- `num_processes()` - Get number of processes
- `process_id()` - Get current process ID

**Design Approach**:
- For DataParallel: Use `torch.distributed` primitives for gradient synchronization
- For ModelParallel: Use tensor partitioning via `torch.split()` and manual all-gather
- Maintain compatibility with JAX's automatic sharding semantics

### 2. Modify: `keras/src/backend/__init__.py`
**Change**:
```python
# From:
elif backend() == "torch":
    from keras.src.backend.torch import *  # noqa: F403
    from keras.src.backend.torch.core import Variable as BackendVariable

    distribution_lib = None

# To:
elif backend() == "torch":
    from keras.src.backend.torch import *  # noqa: F403
    from keras.src.backend.torch.core import Variable as BackendVariable
    from keras.src.backend.torch import distribution_lib
```

### 3. Modify: `keras/src/distribution/distribution_lib.py`
**Changes Required**:
- Update `DataParallel.distribute_dataset()` to handle PyTorch datasets
- Update `ModelParallel.distribute_dataset()` to handle PyTorch datasets
- Add backend delegation for tensor operations (if needed)
- Ensure backward compatibility with JAX

## Implementation Steps

### Step 1: Create PyTorch distribution_lib.py
1. Implement device enumeration (`list_devices`, `get_device_count`)
2. Implement basic tensor distribution (`distribute_tensor`)
3. Implement variable distribution (`distribute_variable`)
4. Implement mesh/layout conversion (`_to_backend_mesh`, `_to_backend_layout`)
5. Implement multi-process initialization (`initialize`)

### Step 2: Update backend/__init__.py
1. Import torch distribution_lib instead of setting to None

### Step 3: Update distribution_lib.py (high-level)
1. Update `DataParallel.distribute_dataset()` to support PyTorch DataLoader
2. Update `ModelParallel.distribute_dataset()` to support PyTorch DataLoader
3. Add any necessary backend delegation methods

### Step 4: Testing (optional)
1. Create basic test to verify distribution works with PyTorch
2. Verify DataParallel works with multi-GPU setup
3. Verify ModelParallel works with tensor partitioning

## Key Design Decisions

### PyTorch DataParallel Implementation
- Use `torch.distributed` primitives for multi-process training
- For single-process multi-GPU, use `torch.nn.DataParallel` wrapper
- Simulate JAX's automatic sharding via explicit operations

### PyTorch ModelParallel Implementation
- Partition weights using `torch.split()` based on layout
- Store sharded weights on appropriate devices
- All-gather weights when needed for operations
- Re-shard outputs after computation

### Backward Compatibility
- JAX implementation remains unchanged
- Existing code continues to work without modification
- New PyTorch support is additive

## Files Reference

### Input Files
- `/Users/suhanaaa/keras/keras/src/distribution/distribution_lib.py` - High-level distribution API
- `/Users/suhanaaa/keras/keras/src/backend/jax/distribution_lib.py` - JAX implementation (reference)
- `/Users/suhanaaa/keras/keras/src/backend/tensorflow/distribution_lib.py` - TF DTensor implementation (reference)
- `/Users/suhanaaa/keras/keras/src/backend/__init__.py` - Backend initialization
- `/Users/suhanaaa/keras/keras/src/backend/torch/core.py` - Torch backend core
- `/Users/suhanaaa/keras/examples/demo_torch_multi_gpu.py` - PyTorch DDP example

### Output Files
- `/Users/suhanaaa/keras/keras/src/backend/torch/distribution_lib.py` - New file
- Modified: `/Users/suhanaaa/keras/keras/src/backend/__init__.py`
- Modified: `/Users/suhanaaa/keras/keras/src/distribution/distribution_lib.py`

## Implementation Notes

### Why This Approach?
1. **Minimal Changes**: Keep high-level API unchanged, add backend-specific code
2. **Composition over Inheritance**: Delegate to backend implementations
3. **Backward Compatible**: Existing JAX code continues to work
4. **PyTorch Idiomatic**: Use PyTorch's distributed primitives where possible

### Challenges Addressed
1. **No Automatic Sharding in PyTorch**: Must manually handle tensor partitioning
2. **Gradient Synchronization**: Use torch.distributed primitives for DDP-style sync
3. **Multi-Process Setup**: Initialize torch.distributed based on Keras distribution config

### Future Extensibility
- This pattern allows adding new backends easily
- FSDP can be added later without breaking current implementation
- Tensor parallelism can be added following same pattern

