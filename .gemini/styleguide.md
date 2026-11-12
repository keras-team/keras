# Keras Pull Request Code Review Guidelines

Use these guidelines when reviewing pull requests in `keras-team/keras`. Focus on catching concrete correctness bugs, multi-backend/symbolic/dtype mismatches, regressions, and clear violations of the scoped conventions below, while strictly respecting every **Do NOT flag** boundary.

---

## 1. Review Scope, Context Verification & High-Signal Bar

- **Check end-to-end consistency of the PR's change, without critiquing untouched code**:
  - **DO check**: When a PR updates a contract (such as changing `compute_output_spec` dtype/shape promotion, decoupling a component from an optional dependency, or adding an argument to `Layer.add_weight`), verify that sibling methods and helpers in that class or module (`_<op>()` fallback, `call()`, public aliases like `Layer.add_variable`, or other lifecycle hooks) stay synchronized and do not crash under the new contract. Also flag genuine spelling typos in newly added identifiers, docstrings, or test descriptions, and stray debug `print`s/scripts.
  - **Do NOT flag**: Untouched context lines, moved/renamed code, or pre-existing logic adjacent to a `+`/`-` line. Never widen `start_line..end_line` across a diff hunk just to critique pre-existing code when the PR only made a narrow fix or migration nearby.
- **PR scope**: Flag notebooks, guides, and demo scripts (these go to keras-io), and intentional breaking changes mixed into an otherwise unrelated PR. A bug in one backend is fixed in that backend, not worked around in shared code, and disabling a feature is not a fix.
- **Verify file imports and base classes before claiming `NameError` or `AttributeError`**:
  - Large files like `keras/src/ops/numpy.py` define imports thousands of lines above the diff, and unchanged `*_test.py` files plus `keras/src/testing/test_case.py` and `test_utils.py` are omitted from auxiliary `<FILES>` context.
  - **Do NOT flag**: Unverified `NameError` or missing-method/attribute claims unless verified against full file imports and base classes (`keras.src.testing.TestCase` defines `assertAllClose`, `assertNotAllClose`, `assertAllEqual`, `assertSparse`, `run_layer_test`, and `get_temp_dir`).
- **Real type/state bugs vs. speculative defensive programming**:
  - **DO check**: (1) Uninitialized attributes across `__init__`/control-flow paths, or attributes initialized to `None` and dereferenced before assignment; (2) calling `issubclass(x, ...)` without `inspect.isclass(x)` when `x` can be a function/callable; and (3) reading private `_*` attributes on user-supplied objects instead of public attributes or `getattr`.
  - **Do NOT flag**: Hypothetical `None` returns from standard constructors or factory classmethods, extra runtime type checks that contradict a hook's documented return signature, state rollback on objects or scopes discarded upon an unhandled exception, `RuntimeWarning` wrappers in tests (`pyproject.toml` ignores `RuntimeWarning`), auto-filling mandatory user arguments, expanding an API's contract beyond its docstring, English i18n string nags, or micro-optimizations outside per-step hot paths (`Layer.__call__`, `convert_to_tensor`, dtype resolution) such as one-time setup, host data loading, or standalone benchmark scripts.
- **Skip checks already enforced by CI**: Pre-commit and CI run `ruff check` (`E`, `F`, `I`, `S101`), `ruff format --check`, `shell/api_gen.sh`, and `zizmor`. Do NOT flag formatting, line length, import sorting, `assert` (`S101`), or unused imports (`F401`).

---

## 2. Multi-Backend, Symbolic, Shape & Dtype Invariants

Keras supports JAX, TensorFlow, PyTorch, NumPy, and pluggable backends across eager and symbolic (`KerasTensor`, `jax.jit`, `tf.function`, `torch.compile`) execution.

### Ops, Config & Backend Semantics
- **Calling ops inside vs. outside `keras/src/ops/`**:
  - **Inside `keras/src/ops/`**, backend-agnostic fallbacks MUST call backend ops directly: `backend.ops.<op>` for core ops (`cast`, `convert_to_tensor`, `convert_to_numpy`, `shape`, `cond`, `while_loop`—note `backend.ops.cast` accepts Python scalars too) and `backend.ops.<submodule>.<op>` for submodule ops (`backend.ops.numpy.where`, `backend.ops.numpy.equal`, `backend.ops.nn.conv`). If a dependency has its own `_<dep>()` helper (e.g. `_divide_no_nan`), call that instead. Never call public `ops.*` inside a `keras/src/ops/` fallback, and never suggest non-existent flat names like `backend.ops.divide_no_nan`, `backend.ops.equal`, or `backend.convert_to_numpy`.
  - **Outside `keras/src/ops/`** (layers, losses, metrics, optimizers, trainers, tests), **both** `ops.*` and `backend.ops.*` are valid—**Do NOT** ask authors to revert `backend.ops.*` to `ops.*` or vice versa.
  - **Config & `backend.common`**: Both `config.<fn>()` (`from keras.src.backend import config`, including `config._use_backend_agnostic_ops()`) and `backend.<fn>()` (`backend.backend()`, `backend.floatx()`, `backend.epsilon()`) are valid—**Do NOT** flag `backend.backend()` or `backend.floatx()`. However, `standardize_dtype`, `standardize_shape`, `is_int_dtype`, and `is_float_dtype` live in `keras.src.backend.common` (not `keras.src.backend`), and `ml_dtypes.finfo` supports both standard float dtypes and `bfloat16`.
- **Backend op contracts**:
  - In `keras/src/backend/numpy/`, NumPy reductions and `np.squeeze` raise `TypeError` when `axis` is a Python `list`—normalize list `axes` to `tuple` there. PyTorch ops (`torch.sum`, `torch.mean`, `torch.var`, `torch.squeeze`) accept a list `dim`, so **Do NOT flag** list `dim` in `keras/src/backend/torch/`.
  - Shared modeling code (`layers/`, `losses/`, `metrics/`, `trainers/`) must not branch on backend names or hardcode `("tensorflow", "jax", "torch")`; use capability flags (`backend.SUPPORTS_GRADIENT`, `backend.SUPPORTS_SPARSE_TENSORS`).
  - **Do NOT flag**: backend dispatch in `keras/src/backend/**`, `keras/src/export/**`, `keras/src/saving/**`, `keras/src/trainers/data_adapters/**`, `keras/src/distribute/**`, or `keras/src/callbacks/tensorboard.py`, nor a bugfix in a single backend (`keras/src/backend/<backend>/...`) when other backends already pass the shared test.
  - **Same behavior on every backend**: When a PR adds a feature or behavior change to one backend, ask whether the other backends (including NumPy) need it too, or for a reason why not. When backends disagree, JAX is the reference.
  - **Ops run natively**: Op implementations in `keras/src/backend/{jax,tensorflow,torch,openvino}/` and backend-agnostic fallbacks must not forward tensors to NumPy or SciPy; they must work on GPU/TPU and under `jax.jit`, `tf.function`, and `torch.compile`. **Do NOT flag** NumPy in `keras/src/backend/numpy/**`, in tests, or on static shapes and Python values.

### Shapes & Dynamic Dimensions
- **Dynamic shapes & `isinstance(d, int)`**:
  - In shared code (`keras/src/ops/`, layers, metrics) and `keras/src/backend/tensorflow/**`, use `ops.shape(x)` (or `backend.ops.shape(x)` in `keras/src/ops/`, `tf.shape(x)` in `backend/tensorflow/`) whenever an axis length is used in runtime tensor computation, padding, or slicing (`reshape`, `broadcast_to`, `split`, `conv` padding), because `x.shape[i]` is `None` on symbolic TensorFlow tensors. On TensorFlow, `ops.shape(x)` keeps static dims as Python ints and only returns tensors for unknown dims, so **Do NOT flag** `ops.shape(x)[axis]` as losing static shape information.
  - Use `isinstance(d, int)` (not `d is not None`) when checking whether a dimension obtained from `ops.shape(x)` or a layer `input_shape` tuple is static vs. dynamic.
  - Use plain Python arithmetic (`+`, `-`, `*`, `//`, `math.prod`, `builtins.abs`) on dimensions, never Keras ops, and never use Python `if`/`else` on a TensorFlow dynamic tensor dimension from `ops.shape(x)[i]`.
  - **Do NOT flag `x.shape`, `x.ndim`, or `is not None` when**:
    1. Inside `keras/src/backend/{jax,torch,numpy}/**`—in JAX, Torch, and NumPy backend modules, `.shape` dimensions are concrete integers (never claim `.shape` arithmetic/comparisons fail under tracing/export there, and remember JAX ops like `jnp.repeat` require static integer sizes).
    2. Checking rank (`len(x.shape)` or `x.ndim`) in shared code, or reading `KerasTensor.shape` inside `compute_output_spec`, `compute_output_shape`, or `build(self, input_shape)`.
    3. Guarding an optional static bounds check on a raw `.shape` tuple (`size = x.shape[axis]; if size is not None:`), where unknown `None` dims are intentionally skipped.

### Tensor Comparisons & Dtype Promotion
- **Keras comparison ops on tensors**: Use `ops.less`, `ops.less_equal`, `ops.greater`, `ops.greater_equal`, `ops.equal`, `ops.not_equal` (or `backend.ops.numpy.<op>` inside `keras/src/ops/`) instead of `<`, `<=`, `>`, `>=`, `==`, `!=` on tensors in shared code and TF/agnostic ops. **Do NOT flag** Python comparisons on scalars, `axis`, `ndim`, `len(...)`, static shape ints, dtype strings, or NumPy arrays in `*_test.py`.
- **Dtype promotion & weak Python scalars**: When an op requires float computation, promote *only* integer and `bool` inputs (`"int" in dtype or dtype == "bool"` or `dtypes.promote_to_float_dtype(dtype)`), keeping `float16`, `bfloat16`, and `float64` unchanged, and ensure `_<op>()` matches `compute_output_spec`. Never write `dtype not in (...)` or `np.dtype(dtype).kind == "f"` (`ml_dtypes.bfloat16` has `kind == "V"`).
  - **Do NOT flag weak Python scalars or `float` in `dtypes.result_type`**: Python `int` and `float` (`0`, `1`, `2.0`, `float`) are **weak types** in `dtypes.result_type` and tensor ops (`result_type("float16", float) == "float16"`, `power(2.0, x)` keeps `bfloat16`). Never suggest replacing `float` with `"float32"` in `result_type`, and never call `convert_to_tensor` on Python `(int, float)` scalars before `result_type`.

### Numeric Stability & Validation Placement
- **Division by zero & masking**: Use `ops.divide_no_nan` (or `_divide_no_nan` / `backend.ops.numpy.divide_no_nan` in `keras/src/ops/`) where zero divisors can occur, and `ops.where(mask, x, 0)` for boolean masking. **Do NOT flag** `x * mask` for continuous weights, and never simplify the gradient-safe "double `where`" pattern (`safe_x2 = where(equal(x2, 0), 1, x2)`) into a single `where` (causes `0 * inf = NaN` gradients).
- **Validation in backend ops vs. `Layer` build-time normalization**:
  - In `keras/src/ops/` & `keras/src/backend/<backend>/ops/`, input validation and argument normalization (such as converting list `axes` to `tuple` or `canonicalize_axes`) must run in both `compute_output_spec` and eager/backend op implementations—not *only* in the public `keras/src/ops/` wrapper or *only* in `compute_output_spec`.
  - Conversely, in `Layer` classes, normalize static axis/reduction attributes (`TrackedList` -> `tuple`) **once in `__init__` or `build()`**, never via per-call `list(...)`/`tuple(...)` conversions inside `Layer.call()`.

---

## 3. TensorFlow Decoupling (`keras/src` & Tests)

- Flag new `import tensorflow`, `tf.*`, or `tensorflow.python.*` usage **outside**: (1) `keras/src/backend/tensorflow/**`, (2) `tf.data` handling in `data_adapters/` and `keras/src/utils/*dataset_utils.py`, (3) `keras/src/export/{saved_model,saved_model_export_archive,tfsm_layer,litert,tf2onnx_lib,xla_saved_model_archive}.py` and `keras/src/legacy/saving/**`, and (4) TF-specific tests, including lazy `from keras.src.utils.module_utils import tensorflow as tf` in `*_test.py` (+ `.github/workflows/config/tf_free_ignored_tests.txt`). Outside these paths, the lazy `module_utils` TensorFlow import is still a TensorFlow dependency. When a module adds a TF-free execution path, ensure all methods reachable without TensorFlow guard any remaining `tf.*` calls.

---

## 4. Backend-Agnostic Implementation for Ops (`keras/src/ops/`)

**Scope**: Applies **only** when adding a new op or adding a backend-agnostic fallback `_<op>()` to an existing op under `keras/src/ops/`.
- Flag a new op under `keras/src/ops/` that only forwards to `backend.ops.*` without a backend-agnostic fallback (`CONTRIBUTING.md`: "Any new op that is implemented must include a backend-agnostic implementation").
- Centralize dispatch (`if not config._use_backend_agnostic_ops() and hasattr(...):`) and `backend.ops.*` fallback in a private `_my_op()` helper called by both the public function and `Operation.call()` (citing "Backend-agnostic implementation for new ops" in `CONTRIBUTING.md` and `_vsplit` in `keras/src/ops/numpy.py`).
- Parameterize existing correctness tests with `@parameterized.named_parameters(named_product(BACKEND_AGNOSTIC_OPS))` (above any `@pytest.mark.skipif`) and immediately register `self.addCleanup(backend.config._set_use_backend_agnostic_ops, False)`.
- **Do NOT flag**: When adding a fallback `_<op>()` to an **existing** op, do NOT ask to add new test methods, new dtype cases, or extra `Operation()(x)` assertions not already present in the existing test.

---

## 5. Imports, Comments, Public API & Testing

- **Imports**: Inside library code (`keras/src/**/*.py`, excluding `*_test.py`), never `import keras` (use `keras.src.*`). Place standard imports at module top, **except** (Do NOT flag) circular-import breaks, `module_utils` `LazyModule`, public `keras.*` imports inside `*_test.py`, or local imports of backend/optional packages (`jax`, `torch`, `tensorflow`, `scipy`, `onnx`, `orbax`, `pil`, `pandas`) in shared tests, exporters, callbacks, or adapters.
- **Simplicity**: Flag dead/unreachable code, `hasattr`/`getattr` for attributes always set in `__init__`, broad `try`/`except` used for control flow when an explicit check exists, unmeasured hot-path performance hacks, redundant conversions or copies on hot paths (a no-op `np.asarray`, a discarded `convert_to_tensor`, converting the same value twice), caches without a measured benefit, deeply nested or hard-to-read control flow where an early `return`/`continue` or a named intermediate value would do, and reinvented helpers (`canonicalize_axis`, `standardize_tuple`, `ops.convert_to_numpy`, `keras.src.tree`, `serialization_lib`). **Do NOT flag** dummy test subclasses overriding `super()` to test serialization/override detection.
- **Names**: Flag names that overclaim (e.g. `fused_*` when fusion isn't guaranteed) or use a different term than the one already used nearby for the same concept (e.g. `world_size`/`rank` next to `num_processes`/`process_id`).
- **Comments, docstrings & error messages**:
  - Flag comments that narrate git/PR history ("previously...", old behavior, issue links) or trivial comments that literally restate obvious code (including redundant `# Test ...` comments in unit tests). **Do NOT flag** concise multi-line "why" comments explaining non-obvious workarounds, invariants, or subtle edge-case test setups.
  - Document class constructor arguments in `Args:` on the class docstring (not `__init__`), keep docstrings matched to behavior, and **Do NOT flag** missing `Examples:` on internal helpers or `Operation` subclasses.
  - In `ValueError` messages, use `.name` on objects instead of raw `<object at 0x...>` strings and format shapes as tuples `(2, 2)` rather than `[2 2]`. **Do NOT** nitpick wording or demand a rigid `"Received: arg=..."` format when an error message already clearly states the invalid input.
- **Public API & serialization**: Keep public aliases (e.g. `Layer.add_variable` and `Layer.add_weight`) synchronized in arguments, preserve backward compatibility on `@keras_export` APIs, exception types, default numerics, and `get_config()` / legacy config round trips (renames need a backward-compatible alias, and intentional breaking changes need a release-notes entry), and avoid dynamic-output-shape ops (`ops.nonzero`, `ops.unique` without `size`) in layers/losses/metrics. **Do NOT flag** argument counts/names dictated by NumPy/JAX/PyTorch/ONNX specs.
- **Tests (`*_test.py`)**:
  - **Tests must be able to fail**: Assert numerical values (`assertAllClose`) on non-trivial inputs, not just shapes, dtypes, `assertIsNotNone`, or "does not raise". Every bug fix and new code branch needs a test that fails without the change. **Do NOT flag** missing tests for docs-only, comment-only, `.gemini/`, or `.github/` changes.
  - **Dtype coverage**: Dtype tests for new ops or dtype changes include `bfloat16`, `float64`, integer, and `bool` inputs, not only `FLOAT_DTYPES` (see Section 4 for fallbacks to existing ops).
  - **Extend, don't duplicate**: Extend existing `*_test.py` classes with `absl.testing.parameterized` (not `pytest.mark.parametrize`, `self.subTest`, or loops over cases), and flag tests that duplicate existing coverage (e.g. the multi-backend tests in `keras/src/ops/*_test.py`). A new `foo_test.py` for a new `foo.py` and separate `test_*_invalid_*` error methods are fine.
  - **Shared tests stay shared**: A backend-independent test goes in the shared test module without a backend skip, even when the fixed bug was backend-specific. Shared helpers go in `test_utils`, not imported from another `*_test.py`.
  - **Skips**: Guard backend-specific API tests (`torch.compile`, `jax.jit`, `tf.function`) with `@pytest.mark.skipif`, not `self.skipTest` or an early `return` on `backend.backend()`. Exclude tests for a pluggable backend in its `excluded_tests.txt`. Don't guard packages from the requirements files with `pytest.importorskip` or `try`/`except ImportError`, except optional TensorFlow via `module_utils.tensorflow` and `.github/workflows/config/tf_free_ignored_tests.txt`.
  - **Isolation**: Prefer real arrays and objects over mocks. Call `super().setUp()` first, restore mutated global config (image data format, distribution, env vars, flags) with `self.addCleanup`, never modify `sys.modules` or call `os.environ.clear()`, and write files only under `self.get_temp_dir()`.
  - **Do NOT** require `assertRaisesRegex` in shared multi-backend op tests when native backend error messages differ.
