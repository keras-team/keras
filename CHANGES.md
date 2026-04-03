# Performance Optimizations — `performance-optimizations` branch

Branch: `pctablet505/keras:performance-optimizations`  
Base: `keras-team/keras:master` (merged at commit `cccbefd1d`)  
Benchmark machine: RTX 4050 Laptop GPU, PyTorch 2.6.0+cu124, Python 3.13

## Results

| Model | Baseline (pip Keras) | This branch | Pure PyTorch |
|-------|----------------------|-------------|--------------|
| ResNet-style CNN (batch=4) | 1.861 ms | 0.249 ms | 0.132 ms |
| LLM forward (batch=4) | 3.961 ms | 0.321 ms | 0.243 ms |
| LLM generate (100 tokens) | 31.870 ms | 2.866 ms | 2.063 ms |

CNN overhead vs pure PyTorch: **1.89×**  
LLM overhead vs pure PyTorch: **1.32–1.39×**

---

## Changes

### 1. `keras/src/ops/function.py` — Pre-compiled forward plan

**What:** `Function._compile_forward_plan()` pre-computes a list of 6-tuples at graph-build time:
```python
(call_fn, in_slots, out_slot, pattern, static_kw, multi_out)
```
`_execute_forward_plan()` iterates this list during every forward pass.

**Why:** The original `run_nodes()` loop did `id()`-based dict lookups into `tensor_dict`, called `symbolic_arguments.fill_in()` with a tree-map (`replace_object()`), and had per-node dispatch logic. For a 50-layer model this is ~300 Python function calls per forward pass. The forward plan replaces all of this with a flat list iteration and direct index access (`tensor_list[slot]`).

**Key details:**
- `pattern` encodes the argument shape at build time (single tensor, dual tensor, etc.)
- For layers with `_fast_call=True`, `call_fn = layer.call` — this bypasses `Layer.__call__` entirely for eligible layers (Dense, LayerNorm, etc.)
- `in_slots` / `out_slot` are integer indices into a flat `tensor_list`, replacing `dict[id(tensor)]` lookups
- Falls back to `fill_in()` for complex argument shapes (pattern 0)

---

### 2. `keras/src/models/functional.py` — Ultra-fast functional model `__call__`

**What:** `Functional.__call__()` override that skips `Layer.__call__` for the common inference case (no training kwarg, built model, single input, eager tensor). `Functional.call()` uses the forward plan directly.

**Why:** `Layer.__call__` has ~40 lines of Python overhead: `_call_context`, `in_stateless_scope()` checks, `call_context.training`, `any_symbolic_tensors()`, `_loss_tracker`, etc. For inference with a frozen model none of this is needed. The fast path calls `_execute_forward_plan()` directly.

**Key details:**
- `_single_input_model` flag set at build time to avoid `len(self._inputs)` every call
- Guard: only activates when not in symbolic scope and model is built

---

### 3. `keras/src/layers/layer.py` — `_fast_call` flag and `__call__` fast path

**What:** Layers eligible for fast dispatch set `self._fast_call = True` (Dense, LayerNorm, Embedding, etc. — stateless pointwise layers). `Layer.__call__` has a fast path that calls `self.call()` directly when `_fast_call` is True and we are in an eager, non-training, non-symbolic context.

**Why:** `Layer.__call__` ordinarily runs mask propagation, activity regularization, `call_context`, loss tracking, and metric tracking on every forward pass. For layers that have none of these, the overhead is pure waste.

**Key details:**
- `_UNTRACKED_ATTRS` frozenset to avoid unnecessary `_tracker.update()` calls on common attributes
- `_accepts_context_arg` cached at first call to avoid `inspect.signature()` on hot path

---

### 4. `keras/src/backend/torch/nn.py` — Inlined channels-last Conv2D

**What:** `_conv2d_channels_last_fast()` is an inlined implementation of 2D convolution for the common case: `data_format="channels_last"`, `groups=1`, `dilation_rate=1`. Called from `conv()` as a fast path.

**Why:** The original `conv()` for channels-last called 6+ helper functions: `compute_conv_output_spec()`, `_compute_conv_padding()`, `convert_to_tensor()` on weights and inputs, a transpose to channels-first, `F.conv2d()`, then transpose back. Most of these are avoidable in the common case. The inlined version does one transpose each direction and one `F.conv2d()` call.

**Note:** This is active only when `groups=1` and `dilation_rate=1`. All other combinations fall back to the original path.

---

### 5. `keras/src/backend/torch/core.py` — Tensor fast paths

#### `convert_to_tensor` fast path
**What:** Check `type(x) is torch.Tensor and dtype is None` first. If the tensor is already on the right device, return it immediately, avoiding `_is_sparse()`, `_is_ragged()`, and the full conversion path.

**Why:** During inference, `convert_to_tensor` is called on activations that are already `torch.Tensor` on the right device. The original code ran isinstance checks, sparse/ragged guards, and dtype normalization on every call. The fast path exits in ~5 Python bytecodes.

**Key detail:** `x.device.type == device` avoids constructing a device string when `device` is a simple type like `"cuda"`.

#### `cast` fast path
**What:** When `type(x) is torch.Tensor` and `x.dtype == dtype`, return `x` immediately.

**Why:** `cast()` is called by layers that normalize dtypes. If the tensor already has the right dtype, the original code still went through `isinstance` and type normalization.

#### `get_device()` direct attribute access
**What:** `getattr(global_state.GLOBAL_STATE_TRACKER, "torch_device", None)` instead of `global_state.get_global_attribute("torch_device")`.

**Why:** `get_global_attribute` is a function call with a `threading.local()` getattr inside. Direct access to the thread-local is slightly faster on the hot path. This is safe because `GLOBAL_STATE_TRACKER` is the only store for this attribute.

#### `slice` / `slice_update` fast path for list/tuple indices
**What:** When `start_indices` and `shape` are already `list`/`tuple`, skip `convert_to_tensor()` and iterate directly.

**Why:** Shape indices in graph tracing are usually Python lists. The original code converted them to tensors just to iterate their values.

---

### 6. `keras/src/backend/torch/trainer.py` — Training loop optimizations

#### `_cached_trainable_weights`
**What:** Cache `self.trainable_variables` into `self._cached_trainable_weights` using `object.__setattr__` to bypass `Layer.__setattr__` and `torch.nn.Module` parameter tracking.

**Why:** `trainable_variables` traverses `_flatten_layers()` on every call, which is expensive for large models. During a training loop, the set of trainable variables does not change between steps. The cache is invalidated on `compile()`.

**Why `object.__setattr__`:** `Layer.__setattr__` calls `self._tracker.update(name, value)` which tries to track the list as a parameter. `torch.nn.Module.__setattr__` would try to register it as a submodule. Using `object.__setattr__` stores the value as a plain Python attribute.

#### `v._value.grad` instead of `v.value.grad`
**What:** Access `.grad` on `v._value` (the raw `torch.Tensor`) rather than `v.value` (which goes through `_maybe_autocast()`).

**Why:** `_maybe_autocast()` may return a new tensor object (the autocasted view). The gradient lives on the original `v._value`, not the autocasted copy. Accessing `v.value.grad` would always return `None` when AMP is active.

#### `make_predict_function` — compile `self.forward`
**What:** When `jit_compile=True`, wrap `self.forward` (the `torch.nn.Module` forward method) with `torch.compile` for predict instead of wrapping `predict_step`.

**Why:** `predict_step` includes Python overhead (data unpacking, dict construction) that Dynamo cannot optimize. Compiling `self.forward` gives Dynamo a clean graph that represents exactly the model computation.

---

### 7. `keras/src/trainers/trainer.py` — Metrics and loss caching

**What:** Three per-step caches:
- `_cached_metrics`: cached list from `self.metrics`
- `_cached_metric_names`: cached list of metric name strings
- `_has_extra_losses`: cached bool for whether model has `add_loss()` losses
- `_symbolic_build_done`: flag to skip `_build_with_dtypes()` after first build

**Why:** `self.metrics` calls `_flatten_layers()` which is an `O(n_layers)` traversal that also calls `getattr()` on every layer. For a 50-layer model this is ~100 getattr calls per training step, executed 3-4 times per step (once in `_update_loss_metrics`, once in `_get_metric_result_or_none`, once in `get_metrics_result()`). The cached versions do this traversal once at compile-time and reuse the result every step.

**Invalidation:** All caches are reset in `compile()` via `object.__setattr__`.

---

### 8. `keras/src/ops/symbolic_arguments.py` — Dual-tensor fast paths

**What:** Two new fast paths in `SymbolicArguments.__init__` and `fill_in()`:
- `_dual_positional_tensors`: for binary ops (Add, Multiply, etc.) with two tensor args and no kwargs
- `_dual_tensors_static_kwargs`: for ops like MultiHeadAttention with two tensor args + static kwargs

**Why:** The existing `_single_positional_tensor` fast path (70× speedup comment in original code) only covered unary ops. Binary ops (which are very common in ResNets, transformers) went through the full tree-map. These paths cover the next most common cases.

---

### 9. `keras/src/backend/torch/optimizers/torch_adam.py` — Adam step without ops dispatch

**What:** The `update_step()` method uses native PyTorch ops (`torch.pow`, `torch.sqrt`, `torch.as_tensor`) directly instead of routing through `ops.cast`, `ops.power`, `ops.sqrt`.

**Why:** `ops.*` functions are backend-dispatched — they go through `keras.src.ops.numpy` which calls `backend.numpy.*` which calls the torch backend. For a function as simple as `lr * sqrt(1-β2^t) / (1-β1^t)`, this dispatch chain adds more overhead than the computation itself. Using `torch.*` directly allows `torch.compile` to see a clean computation graph.

---

### 10. Bug fixes (this session)

#### Removed `_IN_STATELESS_SCOPE` / `_IN_SYMBOLIC_SCOPE` module-level flags

**Files:** `global_state.py`, `stateless_scope.py`, `symbolic_scope.py`, `torch/core.py`

**The bug:** A previous optimization attempt added:
```python
# global_state.py
_IN_STATELESS_SCOPE = False
_IN_SYMBOLIC_SCOPE = False
```
These module-level Python booleans caused `torch.compile` (Dynamo) to create **constant guards**: "this must be `False`". Every time `StatelessScope` was entered during metric updates (every training step), the boolean became `True` → guard broke → Dynamo recompiled the entire training step from scratch. This cancelled out any performance gain from `torch.compile`.

**The fix:** The original thread-local approach via `get_global_attribute("stateless_scope")` uses `getattr(threading.local(), ...)` which Dynamo treats as a dynamic value — it never constant-folds thread-locals — so no guard is created.

**Symptom:** Model recompiles on every training step when `jit_compile=True`. Can be observed with `TORCH_LOGS=recompiles python train.py`.

#### Reverted duplicate TPU transpose block in `jax/nn.py`

**File:** `keras/src/backend/jax/nn.py`

**The bug:** A modification to `dot_product_attention` added an extra `if is_tpu and flash_attention:` block that transposed `query`, `key`, `value` into TPU layout. The immediately following block (the real TPU flash-attention path) did the same transpose again before running the flash attention kernel. Result: tensors transposed twice before the attention kernel, producing wrong outputs on TPU.

**The fix:** Revert the entire flash attention compatibility logic to the original: `_can_use_flash_attention()` with auto-detect for `None` and `raise_error=True` for explicit `True`.

#### Removed `__setattr__` override for `generate_function` in `torch/trainer.py`

**What:** An override of `TorchTrainer.__setattr__` that transparently wrapped `generate_function` in `torch.compile` when `jit_compile=True`.

**Why removed:** Side effects in `__setattr__` are unexpected and hard to debug. keras-hub's `make_generate_function` wraps in `torch.no_grad()` and then our code would wrap that in `torch.compile`, creating an undebuggable double-wrapper. If keras-hub needs compiled generation, it should do so explicitly.

---

## Files changed summary

| File | Change type |
|------|------------|
| `keras/src/ops/function.py` | Forward plan: `_compile_forward_plan`, `_execute_forward_plan` |
| `keras/src/models/functional.py` | Ultra-fast `__call__`, forward plan in `call()` |
| `keras/src/layers/layer.py` | `_fast_call` flag, `_UNTRACKED_ATTRS`, `__call__` fast path |
| `keras/src/backend/torch/nn.py` | `_conv2d_channels_last_fast()` inlined Conv2D |
| `keras/src/backend/torch/core.py` | `convert_to_tensor`/`cast`/`slice` fast paths, `get_device` direct access |
| `keras/src/backend/torch/trainer.py` | `_cached_trainable_weights`, `v._value.grad`, predict compile |
| `keras/src/trainers/trainer.py` | `_cached_metrics`, `_cached_metric_names`, `_has_extra_losses`, `_symbolic_build_done` |
| `keras/src/ops/symbolic_arguments.py` | `_dual_positional_tensors`, `_dual_tensors_static_kwargs` fast paths |
| `keras/src/backend/torch/optimizers/torch_adam.py` | Native `torch.*` ops in Adam update step |
| `keras/src/backend/common/global_state.py` | **Bugfix:** Remove module-level scope flags |
| `keras/src/backend/common/stateless_scope.py` | **Bugfix:** Revert to thread-local `in_stateless_scope()` |
| `keras/src/backend/common/symbolic_scope.py` | **Bugfix:** Revert to thread-local `in_symbolic_scope()` |
| `keras/src/backend/jax/nn.py` | **Bugfix:** Revert duplicate TPU transpose block |
| `benchmarks/bench.py` | New: reproducible benchmark harness |
| `benchmarks/compare.sh` | New: automated comparison script |
