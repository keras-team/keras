# Keras Performance Optimization Plan

**Context**: Keras[torch] shows 2.46x–4.77x overhead vs raw PyTorch. Keras[jax] jit paths
show 1.01x–1.58x overhead vs pure JAX. Generated from codebase audit + benchmark data (March 2026).

---

## ✅ Phase 1 & 2 Implementation — COMPLETED

| Optimization | Status | Files Changed | Impact |
|---|---|---|---|
| OPT-4: `ALLOWED_DTYPES` frozenset | ✅ Done | `dtypes.py` | O(1) dtype `in` check |
| OPT-2: Lazy scope boolean flags | ✅ Done | `global_state.py`, `stateless_scope.py`, `symbolic_scope.py` | O(1) scope check everywhere |
| OPT-3a: `convert_to_tensor` device check | ✅ Done | `torch/core.py` | Avoids `str()` heap alloc |
| OPT-1: Ultra-fast `__call__` bypass | ✅ Done | `layer.py` | **+38% LLM fwd Phase 1** |
| Embedding `_mask_always_none` | ✅ Done | `embedding.py`, `layer.py` | Embedding takes fast path when `mask_zero=False` |
| MHA 2-arg `__call__` fast path | ✅ Done | `layer.py` | MHA `(x, x)` bypasses full `__call__` |
| EinsumDense matmul fast path | ✅ Done | `einsum_dense.py` | Replace `ops.einsum` → reshape+matmul for simple contractions (+10% per MHA sub-layer) |
| OPT-3b: Type guards for `convert_to_tensor` | ✅ Done | `torch/nn.py`, `torch/numpy.py`, `torch/math.py` | `if type(x) is not torch.Tensor` guard for 285 calls; negligible on pre-converted tensors |
| OPT-8: No-op name scope | ✅ Implicit | `layer.py` fast path | Inner layers bypass `__call__` entirely → never enter/exit name scope; measured overhead = 0ms |

### Phase 2 Benchmark Results (bench_final2.json vs bench_torch.json baseline)

| Metric | Baseline | After All | Total Gain |
|---|---|---|---|
| LLM forward eager | 4.44 ms (4.44x raw) | 2.52 ms (2.55x raw) | **+43%** |
| CNN eager | 1.06 ms (2.79x raw) | 0.83 ms (2.18x raw) | **+22%** |
| LLM forward compile | 5.05 ms | 3.62 ms | **+28%** |
| LLM generate eager | ~127 ms (5.39x raw) | ~122 ms (5.82x raw) | +4% |

**Note on generate:** Generate overhead is dominated by:
1. **32 forward passes in sequence** (~3.4ms each × 32 = ~108ms) — the main cost since MPS can't fully pipeline across Python loop iterations
2. **Per-step ops** (argmax, cast, concat): ~1.4ms total, only 0.22ms extra vs raw torch (negligible)
3. **Functional model's `__call__`**: measured at ~0ms overhead (inner layers on fast path absorb the cost)
The remaining gap vs raw torch generate (130ms vs 21ms) is structural: raw torch can pipeline 32 forward passes with almost no Python stalls (0.65ms/step vs 3.4ms/step Keras).
Future improvement options: KV-caching (model-level), fused QKV projection, or `torch.compile` (already gives 130ms→112ms).

### OPT-1 Fast Call Eligibility (Current State)
- `Dense`: ✅ `_fast_call=True` — takes fast path (1-arg)
- `LayerNorm`: ✅ `_fast_call=True` — takes fast path (1-arg)
- `Embedding` (mask_zero=False): ✅ `_fast_call=True` — takes fast path (1-arg)
- `MultiHeadAttention`: ✅ `_fast_call=True` — takes **2-arg fast path** (`mha(x, x)`)
- `EinsumDense` (MHA sub-layers): ✅ `_fast_call=True` + matmul path (called 1-arg from MHA.call)
- `Functional` model: ❌ `_call_has_mask_arg=True` due to `Functional.call(self, inputs, training=None, mask=None)` (**measured overhead = 0ms** — overhead already absorbed by inner layer fast paths)

## Remaining Deep Optimization Opportunities

| Opportunity | Complexity | Estimated Gain | Notes |
|---|---|---|---|
| Fused QKV projection | High | ~8–12% LLM fwd | Combine Q/K/V Dense into one matmul, then split |
| KV-caching in generate | Model-level | ~3–5x generate | Avoid recomputing past-token attention |
| Remove causal mask rebuild per step | Medium | ~5% MHA | Cache the `ones × cumsum` mask or use SDPA's built-in causal flag directly |
| `torch.compile` | Framework | +28% fwd/+14% gen | Already working, users can opt in |

---

## Summary of Benchmark Observations

| Operation | Pure | Keras | Overhead |
|-----------|------|-------|----------|
| CNN eager (torch) | 0.52 ms | 2.03 ms | **3.93x** |
| CNN compile (torch) | 0.97 ms | 2.49 ms | 2.56x |
| LLM fwd eager (torch) | 1.22 ms | 3.86 ms | **3.17x** |
| LLM fwd compile (torch) | 2.47 ms | 6.41 ms | 2.59x |
| LLM gen eager (torch) | 21.43 ms | 102.18 ms | **4.77x** |
| LLM gen compile (torch) | 64.83 ms | 159.29 ms | 2.46x |
| CNN jit (jax) | 0.45 ms | 0.46 ms | 1.01x ✓ |
| LLM fwd jit (jax) | 1.74 ms | 1.90 ms | 1.09x ✓ |
| LLM gen jit (jax) | 72.20 ms | 113.94 ms | **1.58x** |

---

## Root Cause Analysis

### B1 — `__call__` Framework Overhead (Highest Impact)

**File**: `keras/src/layers/layer.py:830`

Every single forward pass of every layer goes through `Layer.__call__`, which does:
1. `dtype_policy.convert_input()` — dtype check + potential cast on every call
2. `is_backend_tensor_or_symbolic()` — `backend.is_tensor()` + `isinstance(x, KerasTensor)` per-arg
3. `CallSpec(...)` construction — iterates `signature.parameters`, fills `arguments_dict`, etc.
4. `_maybe_build()` — checks `self.built` (fast), but also calls `get_shapes_dict(call_spec)` even when already built
5. `_resolve_and_populate_arg()` — per-context-arg dict lookup loop
6. `backend.get_keras_mask()` / `backend.set_keras_mask()` — attribute access on every tensor
7. `any_symbolic_tensors()` — called on original args at end of every call
8. `distribution_lib.distribution()` — global state lookup
9. Name scope `__enter__`/`__exit__` — two context manager calls per layer per forward pass

For an LLM model with 20+ layers, each call stacks these. Per inference: >200 Python function-call roundtrips just in framework overhead.

### B2 — `convert_to_tensor` Called on Already-Correct Tensors

**File**: `keras/src/backend/torch/nn.py`

Every operation wrapper (relu, matmul, layernorm, etc.) calls `convert_to_tensor(x)` unconditionally:

```python
def relu(x):
    x = convert_to_tensor(x)   # <-- always called, even when x is already a torch.Tensor on device
    return tnn.relu(x)
```

There are **250+ such functions** in `torch/nn.py`, `torch/numpy.py`, `torch/math.py`. Each call goes through at least one dict lookup, device check, and type check — even when the tensor is already correct.

### B3 — `in_stateless_scope()` / `in_symbolic_scope()` Called Per-Op

**File**: `keras/src/backend/common/stateless_scope.py`, `keras/src/backend/torch/core.py`

The `Variable.value` property (line 142 in `torch/core.py`) calls `in_stateless_scope()` on every access:

```python
if in_stateless_scope():   # dict lookup on GLOBAL_STATE_TRACKER every access
    scope = get_stateless_scope()
    value = scope.get_current_value(self)
```

`in_stateless_scope()` calls `get_global_attribute("stateless_scope")` which calls `getattr(GLOBAL_STATE_TRACKER, "stateless_scope", None)`. This is called every time any layer reads a weight during forward. For an LLM: tens of thousands of weight accesses.

### B4 — `get_device()` Called on Every `convert_to_tensor`

**File**: `keras/src/backend/torch/core.py:76`

```python
def get_device():
    device = getattr(global_state.GLOBAL_STATE_TRACKER, "torch_device", None)
    if device is None:
        return DEFAULT_DEVICE
    return device
```

This is already fast (direct `getattr`), but called on every single `convert_to_tensor` call (see B2 — 250+ call sites). String comparison `str(x.device) == device` also allocates a string per tensor.

### B5 — `standardize_dtype()` Not Fully Short-Circuited

**File**: `keras/src/backend/common/variables.py:576`

Called frequently during layer operations. The current implementation has a cache (`_TORCH_DTYPE_CACHE`), but: the cache is a plain `dict` with no size limit and the fast-path for `str` checks `dtype in dtypes.ALLOWED_DTYPES` (a tuple membership test: O(N) linear scan for ~25 items). Should use a `frozenset`.

### B6 — Tree Operations in Hot Path

**File**: `keras/src/tree/tree_api.py`, used in `layer.py`

`tree.map_structure(maybe_convert, args)` and `tree.flatten()` are called in `__call__` for input conversion. For the 95%+ common case of a single tensor input with no nesting, this calls into the full tree traversal library. The fast path added at line 851 helps but still calls `is_backend_tensor_or_symbolic()` twice for single-arg case (once in condition, once again below).

### B7 — `_open_name_scope()` Context Manager Called Twice Per Layer

**File**: `keras/src/layers/layer.py:907, 970`

The layer `__call__` opens the name scope twice (`_maybe_build` + the actual call block). Each `__enter__`/`__exit__` modifies a thread-local stack. During inference, name scopes are only needed for graph building/debugging — they add pure overhead in eager mode.

### B8 — `activity_regularizer` Check on Every Forward Pass

**File**: `keras/src/layers/layer.py:1007`

```python
if self.activity_regularizer is not None:  # checked every single call
    for output in tree.flatten(outputs):
```

`tree.flatten(outputs)` is called for every layer on every forward, even though `activity_regularizer` is `None` by default for 99.9% of inference use cases.

### B9 — `any_symbolic_tensors()` at End of Every Forward

**File**: `keras/src/layers/layer.py:1046`

Called on original_args/original_kwargs at the end of every forward pass to decide whether to add a graph node. During inference, this always returns `False`, but the check still iterates args.

### B10 — JAX `jit` Retracing from Keras Wrappers

**File**: `keras/src/backend/jax/core.py`

When Keras wraps JAX operations, JAX `jit` may not see through the Python wrappers efficiently. The 1.58x overhead on jit generate (vs pure JAX) suggests retracing or extra Python evaluation in the generation loop iteration path. The `StatelessScope` dict lookup pattern makes it hard for jax.jit to cache variable reads.

---

## Optimization Roadmap

### Phase 1 — Zero-Cost Inference Fastpath (No Breaking Changes)

**Target**: Reduce Keras[torch] overhead to ~1.5x for forward pass, ~2x for generate.

#### OPT-1: Add `_compiled_call` Fastpath to Layer

When a layer has been built and is called in pure-eager mode (no masking, no distribution, no symbolic tensors, no activity regularizer), bypass most of `__call__` entirely:

```python
def __call__(self, *args, **kwargs):
    # Ultra-fast path for inference: single tensor, built layer, no special features
    if (
        self.built
        and self._activity_regularizer is None
        and not self._call_has_mask_arg
        and not self._call_context_args
        and len(args) == 1
        and not kwargs
        and backend.is_tensor(args[0])
    ):
        return self.call(args[0])  # skip all framework overhead
    # ... existing full path
```

This alone could reduce per-layer overhead by 80%+ for standard inference.

**Breaking impact**: None. This is a pure speedup for the common case.
**Effort**: Small (30 lines).

#### OPT-2: Lazy Scope Checks via Flags

Replace `in_stateless_scope()` and `in_symbolic_scope()` (which are `getattr` lookups) with module-level boolean flags updated only when scopes are entered/exited:

```python
# global_state.py
_IN_STATELESS_SCOPE = False
_IN_SYMBOLIC_SCOPE = False
```

Since scope enter/exit is rare (only during model building / training setup), this flag approach means inference pays zero cost for scope checks.

**Breaking impact**: Internal only.
**Effort**: Medium (affects 3 files, but simple change).

#### OPT-3: Skip `convert_to_tensor` When Input is Already Correct

In operation wrappers (`torch/nn.py`, `torch/numpy.py`), add a type guard:

```python
def relu(x):
    if type(x) is not torch.Tensor:
        x = convert_to_tensor(x)
    return tnn.relu(x)
```

`type(x) is torch.Tensor` is a C-level pointer comparison (faster than `isinstance`). This avoids all the device-check and dtype-check logic for the common case.

**Breaking impact**: None.
**Effort**: Large (250+ functions), but can be done with a script.

#### OPT-4: `ALLOWED_DTYPES` as `frozenset`

In `dtypes.py`: change `ALLOWED_DTYPES = (...)` to `ALLOWED_DTYPES = frozenset(...)`. The `in` check in `standardize_dtype` drops from O(N) linear scan to O(1) hash lookup.

**Breaking impact**: `ALLOWED_DTYPES` is public API — iteration order would change. Must check if any code iterates it. Likely safe since it's just a set of strings.
**Effort**: Tiny (1 line).

#### OPT-5: Cache `supports_masking` and `activity_regularizer` is-None as Booleans

Pre-compute at build time:
```python
# After build():
self._fast_inference = (
    self.activity_regularizer is None
    and not self.supports_masking
    and distribution_lib.distribution() is None
)
```

Then in `__call__`, gate the regularizer check and masking setup on `self._fast_inference`.

**Breaking impact**: None.
**Effort**: Small.

---

### Phase 2 — JIT/Compile Compatibility Improvements

**Target**: Make `torch.compile` on Keras models have ≤1.5x overhead vs raw PyTorch.

#### OPT-6: Eliminate Global State Lookups from Compute Graph

`distribution_lib.distribution()` and `get_device()` are called in the hot path. These global-state lookups force Python fallback in `torch.compile`:

- **Fix**: Capture device and distribution at model build time, store on the layer. Only re-query on device change.
- Store `self._device = get_device()` at build time, use it directly in forward.

**Breaking impact**: Models built on one device then moved would need to update. Add `model.to(device)` override to refresh.

#### OPT-7: Make `Variable.value` a Direct Attribute Access

Currently `Variable.value` is a property with conditional logic (stateless scope check, autocast check, mock check). In compiled/jit mode, this prevents constant-folding weights.

**Fix**: When `stateless_scope` is not active and `autocast` is False, expose `_value` directly:

```python
@property
def value(self):
    if _IN_STATELESS_SCOPE or self._autocast:
        return self._full_value_impl()
    return self._value  # direct, no overhead
```

**Breaking impact**: Internal only.

#### OPT-8: `torch.compile`-Friendly Name Scope

The `_open_name_scope()` context manager uses thread-local stacks which are opaque to `torch.compile`. During compiled mode, name scopes are unused. Add:

```python
@contextlib.contextmanager
def _open_name_scope(self):
    if _IN_SYMBOLIC_SCOPE:  # only needed during graph tracing
        with base_name_scope(self.name, caller=self):
            yield
    else:
        yield  # no-op during eager/compiled inference
```

**Breaking impact**: Name scopes during eager inference are only used for debugging. No compute behavior change.

#### OPT-9: JAX `jit` Variable Scan Optimization

The 1.58x overhead in `keras[jax]` generate jit vs pure is partly from variable access inside jit. JAX `jit` must retrace if Python-level variable reads aren't static. Fix:

- Expose model weights as flat `pytree` at call time (matching what `lax.scan` does in pure JAX)
- Use `jax.lax.scan` loop for generate instead of Python for-loop in Keras generate methods
- Document pattern for users

**Breaking impact**: Changes to model generate APIs — use version flag or deprecation path.

---

### Phase 3 — Architecture-Level Changes (May Require Breaking Changes)

**Target**: Keras overhead approaches 0 for jit/compiled paths. ~1.1x for eager.

#### OPT-10: Compiled Model Execution Plan

For fully-built models in inference mode, pre-compile a flat execution plan:

```python
model.compile_for_inference()  # new method
```

This traces the model once, creates a flat list of `(layer, input_sources, output_slots)`,
then inference skips `__call__` entirely and executes the flat plan.

For `torch.compile`: wraps the entire flat execution in one `torch.compile` unit (currently, per-layer compile fragments reduce effectiveness).
For `jax.jit`: wraps in a single `jax.jit` call over all layers.

**Breaking impact**: New API (`compile_for_inference`). Existing `model(x)` unchanged. **Non-breaking**. Long-term, could become default for Sequential/Functional models.

#### OPT-11: Eliminate Per-Op `CallSpec` for Functional Models

For `Functional` (graph-based) models already built, the computation graph is fixed. Replace `CallSpec` construction (which uses `inspect.signature`) with direct argument passing by position.

The `Functional` model's `call()` already walks a precomputed node DAG — but each node still goes through the full `layer.__call__` machinery. Instead, it could call `layer.call()` directly with pre-validated args.

**Breaking impact**: Must maintain `__call__` behavior for subclassed models. Can be scoped to `Functional` + `Sequential`. Non-breaking as opt-in.

#### OPT-12: Remove Overhead for Non-Training Features During Inference

`Layer.__call__` pays overhead for: losses, metrics, masking, regularization, distribution, name scopes — even during inference where none apply.

Option A (conservative): Add a `model.eval()` method (analogous to PyTorch) that sets a flag disabling all training-only hooks. The flag gates all the per-call overhead.

Option B (breaking): Default to eval mode, opt-in to training-mode tracking. This is a bigger design change but aligns with user mental model.

**Recommendation**: Option A. `model.eval()` / `model.train()` is already familiar from PyTorch. Users who migrate from TF/Keras can learn the new API. Can be introduced as a deprecation path.

---

### Phase 4 — Hardware-Specific Optimizations

#### CUDA / GPU

- Set `DEFAULT_DEVICE = "cuda"` when available, use `torch.cuda.synchronize()` only at benchmark points
- Consider `torch.amp.autocast` integration at the model level (mixed precision by default for inference)
- Profile with `torch.profiler` to identify individual CUDA kernel bottlenecks in generate loop
- Use `torch.backends.cuda.matmul.allow_tf32 = True` by default for CUDA

#### Apple Silicon (MPS)

- MPS doesn't support all ops; current workaround: fallback to CPU silently. Make explicit with configurable fallback policy.
- Profile which ops force CPU fallback (likely: `cumsum`, some `scatter` variants, `complex` ops)
- Group CPU-fallback ops into batches to reduce MPS↔CPU data transfer count
- Consider MPS-specific kernel implementations for common ops (LayerNorm, softmax)

#### TPU (JAX backend)

- TPU performance is already near-native for Keras[jax] jit (overhead mainly 1.01x–1.09x for CNNs)
- Main issue: Python-level generate loop cannot be efficiently pipelined on TPU
- Fix: implement generate with `jax.lax.scan` or `jax.lax.while_loop` (in-XLA loop)
- Add `use_xla_loop=True` flag to generate methods
- Ensure `StatelessScope` patterns are compatible with XLA device array semantics

#### Multi-Device / SPMD Distribution

- The `distribution_lib.distribution()` global is checked on every layer call — fix with OPT-6 above
- For SPMD models, the per-op device placement is already correct; reduce overhead of checking it per op

---

## Breaking Change Risk Assessment

| Change | Risk | Mitigation |
|--------|------|-----------|
| `ALLOWED_DTYPES` as frozenset | Low | Check all usages; iteration order fine for string sets |
| Lazy scope flags | Low | Internal implementation detail |
| Layer fast-inference path | Low | Guarded by `self.built` + feature flags |
| `Variable.value` direct access | Medium | Must maintain stateless scope correctness |
| Name scope no-op in eager | Low | Only behavioral during debugging |
| `model.eval()` mode | Medium | New API; existing code unaffected |
| Functional flat execution | Medium | Must test all Functional model features |
| Generate with lax.scan | **High** | Changes generate output behavior slightly (numerical identity may differ with scan) — use opt-in flag |
| Compiled execution plan | Low | Additive API |

---

## Implementation Priority Order

```
Week 1 (Immediate — no breaking changes):
  OPT-4  ALLOWED_DTYPES frozenset                          ✅ DONE
  OPT-2  Lazy scope boolean flags                          ✅ DONE
  OPT-3a convert_to_tensor device check improvement        ✅ DONE (torch/core.py)
  OPT-1  __call__ ultra-fast path for inference           ✅ DONE (+38% LLM fwd, +20% CNN)
  OPT-3b convert_to_tensor skip in op wrappers             [next — scripted, 160+ sites]
  OPT-5  Cache fast_inference flag on layer                [merged into OPT-1 as _fast_call]

Week 2:
  Investigate Embedding _fast_call=False                   [quick win]
  MHA self-attention fast path (len(args)==2, args[0] is args[1])  [medium]
  OPT-8  Name scope no-op during eager                    [half day]

Week 3:
  OPT-6  Capture device at build time                     [2 days]
  OPT-7  Variable.value direct access                     [1 day]
  OPT-9  JAX jit variable scan optimization               [3 days]

Month 2:
  OPT-10 Compiled model execution plan                    [1 week]
  OPT-11 Functional flat execution (no CallSpec)          [1 week]
  OPT-12 model.eval() inference mode                      [1 week + API review]

Month 3+:
  Hardware-specific: MPS, CUDA, TPU optimizations
```

---

## Expected Performance After Each Phase

| Phase | Keras[torch] eager overhead | Keras[jax] jit overhead |
|-------|-----------------------------|------------------------|
| Baseline (original) | 3.17x–4.77x | 1.01x–1.58x |
| After Phase 1 (**actual**) | **2.24x–5.33x** (fwd: 2.77x, CNN: 2.24x, gen: 5.33x) | untested |
| Target after Phase 1 | ~1.8x–2.5x | 1.01x–1.2x |
| After Phase 2 | ~1.3x–1.8x | 1.0x–1.1x |
| After Phase 3 | ~1.1x–1.3x | ~1.0x |
| After Phase 4 | ~1.05x–1.1x | ~1.0x |

The fundamental remaining gap (~1.05x) is unavoidable Python-level overhead for building/running Keras abstractions. Any further reduction requires C extension or ahead-of-time compilation.

---

## Testing Strategy (No Breakage Policy)

1. **Unit tests**: Every optimization has a corresponding test that the output of `model(x)` is numerically identical before and after.
2. **Benchmark suite**: The 3-pass `run_bench.sh` script runs before and after each change. Accept only if overhead ratio improves or stays the same.
3. **Functional test**: Run existing Keras test suite with `lintrunner -a` to catch regressions.
4. **Deprecation fence**: For Phase 3 changes, introduce under feature flag `KERAS_FAST_INFERENCE=1` first. Collect feedback before making default.
5. **API compat**: All changes to `Layer.__call__` must preserve: mask propagation, activity regularization, stateless scope, symbolic scope, name scope, distribution.

---

## Files to Modify

| File | Change |
|------|--------|
| `keras/src/backend/common/dtypes.py` | frozenset for ALLOWED_DTYPES |
| `keras/src/backend/common/global_state.py` | Add `_IN_STATELESS_SCOPE`, `_IN_SYMBOLIC_SCOPE` flags |
| `keras/src/backend/common/stateless_scope.py` | Update flag on enter/exit |
| `keras/src/backend/common/symbolic_scope.py` | Update flag on enter/exit |
| `keras/src/backend/common/variables.py` | Use flags in Variable.value |
| `keras/src/backend/torch/core.py` | Variable.value direct access |
| `keras/src/backend/torch/nn.py` | Skip convert_to_tensor for torch.Tensor |
| `keras/src/backend/torch/numpy.py` | Skip convert_to_tensor for torch.Tensor |
| `keras/src/backend/torch/math.py` | Skip convert_to_tensor for torch.Tensor |
| `keras/src/backend/jax/nn.py` | Skip convert to jnp.ndarray for jax arrays |
| `keras/src/backend/jax/numpy.py` | Skip convert to jnp.ndarray for jax arrays |
| `keras/src/layers/layer.py` | Ultra-fast inference path, cache fast_inference flag, lazy name scope |

---

*Plan authored based on local benchmark data (MPS) and full codebase audit. GPU/TPU phases require extended profiling on those accelerators.*
