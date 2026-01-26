# TODO: Debuggers for Torch Model Parallel Distribution

## Phase 1: Add Debugging Infrastructure to PyTorch Backend Distribution Lib

### 1.1 Add debugging utilities to `keras/src/backend/torch/distribution_lib.py`
- [x] Add environment variable control for debug mode
- [x] Create debug logging functions
- [x] Implement conditional debug prints

### 1.2 Add debuggers to key functions in torch distribution lib
- [x] Debug `initialize()` function - log process setup and environment
- [x] Debug `distribute_tensor()` function - log sharding decisions
- [x] Debug `distribute_variable()` function - log variable layouts
- [x] Debug `distribute_data_input()` function - log data distribution
- [x] Debug `all_reduce()`, `all_gather()`, `broadcast()` - log collective ops
- [ ] Debug `num_processes()` and `process_id()` - log process info
- [ ] Debug device mesh and layout conversion functions

## Phase 2: Add Debugging to High-Level Distribution APIs

### 2.1 Add debuggers to distribution classes in `keras/src/distribution/distribution_lib.py`
- [x] Debug `DataParallel` initialization and setup
- [x] Debug `ModelParallel` initialization and setup  
- [x] Debug `get_data_layout()` method in both classes
- [x] Debug `get_variable_layout()` method in both classes
- [x] Debug `distribute_dataset()` method in both classes

### 2.2 Add debuggers to LayoutMap
- [x] Debug layout lookup and matching logic
- [x] Debug regex matching for variable paths
- [x] Debug layout assignment operations

## Phase 3: Testing and Validation

### 3.1 Create test cases
- [ ] Test debugging functionality with environment variables
- [ ] Verify debug output is clear and helpful
- [ ] Test enable/disable of debug mode

### 3.2 Performance and functionality verification
- [ ] Ensure debugging doesn't affect normal operation
- [ ] Verify debug output provides useful information
- [ ] Test with both DataParallel and ModelParallel scenarios

## Progress Tracking

### Phase 1: In Progress
- [ ] 1.1 Add debugging utilities
- [ ] 1.2 Add debuggers to key functions

### Phase 2: Pending
- [ ] 2.1 Add debuggers to distribution classes
- [ ] 2.2 Add debuggers to LayoutMap

### Phase 3: Pending
- [ ] 3.1 Create test cases
- [ ] 3.2 Performance verification

## Notes
- Use environment variable `KERAS_TORCH_DISTRIBUTION_DEBUG` to enable debug mode
- Make debug output clear and structured
- Avoid performance impact when debug mode is disabled
- Provide useful information for troubleshooting distribution issues
