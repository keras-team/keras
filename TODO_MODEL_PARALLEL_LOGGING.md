# Model Parallel Logging Implementation

## Goal
Add logging to verify that model parallel is working correctly by checking:
1. Variables are being sharded across 2+ devices
2. Devices are storing and performing operations on sharded data

## Implementation Steps

### Step 1: Add logging to Torch Backend Distribution (`keras/src/backend/torch/distribution_lib.py`)
- [x] Import logging module
- [x] Add logger configuration
- [x] Add logging to `_shard_tensor()` function
- [x] Add logging to `distribute_variable()` function
- [x] Add logging to `all_gather_variable()` function

### Step 2: Add logging to High-Level Distribution API (`keras/src/distribution/distribution_lib.py`)
- [x] Import logging module
- [x] Add logger configuration
- [x] Add logging to `ModelParallel.get_variable_layout()` method
- [x] Create utility function `verify_model_parallel()` for comprehensive verification

### Step 3: Update exports
- [x] Update `keras/src/distribution/__init__.py` to export `verify_model_parallel`

### Step 4: Create verification script
- [x] Create `test_model_parallel_logging.py` script
- [x] Demonstrate logging output for model parallel setup

## Expected Log Output Format

### Tensor Sharding Log
```
[INFO] keras.distribution - [ModelParallel] Sharding tensor | Original shape: (1024, 512) | Shard axis: 1 | Shard shape: (512, 512) | Rank: 0 | Device: cuda:0 | Mesh devices: 2
```

### Variable Distribution Log
```
[INFO] keras.distribution - [ModelParallel] Distributing variable | Full shape: (1024, 512) | Shard shape: (512, 512) | Axes: (None, 'model') | Device mesh: (2, 4) | Device: cuda:0 | Is sharded: True
```

### Verification Output
```
[INFO] keras.distribution - [ModelParallel] Verification starting... | Device mesh shape: (2, 4) | Total devices: 8 | Axis names: ['batch', 'model']
[INFO] keras.distribution - [ModelParallel] Verification Summary: | Total devices: 8 | Sharded variables: 4 | Replicated variables: 2 | Model parallel is ACTIVE
```

## Files Modified
1. ✅ `keras/src/backend/torch/distribution_lib.py` - Added logging to core distribution functions
2. ✅ `keras/src/distribution/distribution_lib.py` - Added logging and `verify_model_parallel()` utility
3. ✅ `keras/src/distribution/__init__.py` - Exported new utility function

## Testing
After implementation, run the verification script to ensure logs are generated correctly.
