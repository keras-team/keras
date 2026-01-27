# Model Parallel Logging Implementation

## Goal
Add logging to verify that model parallel is working correctly by checking:
1. Variables are being sharded across 2+ devices
2. Devices are storing and performing operations on sharded data

## Implementation Steps

### Step 1: Add logging to Torch Backend Distribution (`keras/src/backend/torch/distribution_lib.py`)
- [ ] Import logging module
- [ ] Add logger configuration
- [ ] Add logging to `_shard_tensor()` function
- [ ] Add logging to `distribute_variable()` function
- [ ] Add logging to `distribute_tensor()` function
- [ ] Add logging to `all_gather_variable()` function
- [ ] Add logging to `all_reduce()` function
- [ ] Add logging to `all_gather()` function

### Step 2: Add logging to High-Level Distribution API (`keras/src/distribution/distribution_lib.py`)
- [ ] Import logging module
- [ ] Add logger configuration
- [ ] Add logging to `ModelParallel.get_variable_layout()` method
- [ ] Add logging to `ModelParallel.get_data_layout()` method
- [ ] Create utility function `verify_model_parallel()` for comprehensive verification

### Step 3: Create verification script
- [ ] Create `test_model_parallel_logging.py` script
- [ ] Demonstrate logging output for model parallel setup

## Expected Log Output Format

### Tensor Sharding Log
```
[ModelParallel] Sharding tensor 'kernel' | Original shape: (1024, 512) | Shard axis: 1 | Devices: cuda:0, cuda:1
```

### Variable Distribution Log
```
[ModelParallel] Distributing variable 'dense_1.kernel' | Shape: (512,) | Device: cuda:1 | Is sharded: True
```

### Verification Output
```
[ModelParallel] Verification Summary:
  - Total devices: 2
  - Sharded variables: 5
  - Replicated variables: 3
  - Model parallel is ACTIVE
```

## Files to Modify
1. `keras/src/backend/torch/distribution_lib.py`
2. `keras/src/distribution/distribution_lib.py`
3. `keras/test_model_parallel_logging.py` (new file)

## Testing
After implementation, run the verification script to ensure logs are generated correctly.
