# MLX backend — test status & how to run

This is the Keras MLX backend (`mlx.core`), targeting Apple Silicon unified
memory (Metal GPU + CPU). It is **training-capable** (`model.fit` via MLX
functional autograd `mx.value_and_grad`), unlike the numpy backend which is
inference-only.

This document records the verified test status and the exact commands to
reproduce it, so the numbers below are always re-checkable.

> Last verified: 2026-06-28 · mlx 0.31.2 · tensorflow 2.21.0 · numpy 2.5.0 ·
> Python 3.12 (Apple Silicon). Re-verify against the installed `mlx.core` if a
> number here drifts.

---

## TL;DR

**Full `pytest keras` suite: 0 MLX-specific failures.** Every remaining
failure/error is an *environment* limitation (a library not installed, or a
native lib that segfaults on this Python), and fails identically on other
backends in the same environment.

---

## Requirements

- **Python:** 3.12 (recommended). MLX installs on macOS arm64 only
  (`sys_platform == 'darwin' and platform_machine == 'arm64'`).
  TensorFlow has no wheel for Python 3.14 yet, so 3.12 is used for the
  with-tensorflow run.
- **Core deps:** `mlx`, `numpy`, `scipy`, `optree`, `h5py`, `ml-dtypes`,
  `orbax-checkpoint`, `protobuf`, `absl-py`, `rich`, `pytest`, `pytest-xdist`.
- **For full coverage (recommended):** also install `tensorflow` (makes the
  preprocessing layers, tf.data dataset adapters, and image-ops tests
  *runnable* — they hard-import tensorflow and otherwise skip/error at
  collection). Install from the repo requirements:
  ```bash
  python3.12 -m venv .venv312
  .venv312/bin/pip install "tensorflow" "mlx" -r requirements-common.txt
  # requirements-common pulls keras transitively as a PyPI package; uninstall
  # it so this source tree (3.16.0) wins on sys.path:
  .venv312/bin/pip uninstall -y keras
  ```
  > Known env snag: `array-record` (a `grain` dependency) **segfaults** during
  > data loading on Python 3.12. It is unrelated to MLX but crashes the `grain`
  > data-adapter tests. `.venv312/bin/pip uninstall -y array-record` removes the
  > crash (those tests then fail with a clean import error instead).

Keras is run from source (the repo root is on `sys.path`); it is **not**
installed as a package.

---

## How to run

```bash
cd /Users/deepak/Working/keras-mlx

# Full suite (with tensorflow) — recommended:
KERAS_BACKEND=mlx TF_CPP_MIN_LOG_LEVEL=3 .venv312/bin/python -m pytest keras \
    --ignore=keras/src/wrappers/sklearn_test.py \
    -n auto -q

# sklearn wrapper tests separately (single process): their parametrized IDs
# embed a function object whose repr contains a per-process memory address,
# which breaks pytest-xdist's collection-consistency check. Run without xdist:
KERAS_BACKEND=mlx .venv312/bin/python -m pytest \
    keras/src/wrappers/sklearn_test.py -p no:xdist -q

# Without tensorflow (e.g. on the py3.14 venv): same commands with .venv/bin/python.
# Preprocessing / dataset / image-op tests that hard-import tensorflow then
# fail at *collection* with ModuleNotFoundError — see "Status" below.
```

`TF_CPP_MIN_LOG_LEVEL=3` silences TensorFlow's oneDNN/CPU log spam. Add
`--junit-xml=out.xml -o junit_logging=verbose` to capture structured
pass/fail/error for classification.

---

## Status

### With tensorflow installed (Python 3.12 `.venv312`) — the real end-to-end check

| Suite | Passed | Failed | Skipped | Errors |
|---|---|---|---|---|
| bulk (`pytest keras`, xdist) | 12607 | 48 | 1520 | 17 |
| sklearn_test (single proc) | 129 | 0 | 9 | 0 |
| **Combined** | **12736** | **48** | **1529** | **17** |

**All 48 failures + 17 errors are environment, not MLX:**

| Cause | Count | Why it's not MLX |
|---|---|---|
| `grain` / `array-record` native segfault on py3.12 | ~38 | The grain data-loading C++ extension crashes during import/use on Python 3.12. Reproduces under any backend; not an MLX-array issue. |
| `torch` not installed | ~27 | These tests exercise the **torch** backend / torch data adapters (`torch_data_loader_adapter_test`, `torch_utils_test`, `export/torch_test`, …). They import `torch`, which isn't installed here. |

**0 MLX-specific failures.** The `+852` passed vs. the no-tensorflow run is the
headline: full application models (ResNet, ConvNeXt, NASNet, …), all
preprocessing layers, tf.data adapters, image ops, and the tensorboard callback
now actually execute on the MLX backend.

### Without tensorflow (e.g. Python 3.14 `.venv`)

| Suite | Passed | Failed | Skipped | Errors |
|---|---|---|---|---|
| bulk (`pytest keras`, xdist) | 11730 | 168 | 1357 | 66 |
| sklearn_test (single proc) | 129 | 0 | 9 | 0 |
| **Combined** | **11859** | **168** | **1366** | **66** |

Here the 168 failures + 66 errors are **all** `ModuleNotFoundError: No module
named 'tensorflow'` (preprocessing modules hard-import tf, tf.data datasets,
StringLookup, gfile). Proven non-MLX: the identical tests fail the same way
under the **numpy** backend without tensorflow. Installing tensorflow moves
almost all of these to **pass** (see the table above) and surfaced the real bugs
listed below.

---

## Known limitations (skip-listed in `excluded_concrete_tests.txt`)

These are honest MLX constraints, excluded via substring match on the pytest
node id (see `conftest.py`):

| Entry | Reason |
|---|---|
| `float8` | MLX has no float8 dtype (`float8_e4m3fn`/`e5m2`). Catches all float8-named tests. |
| `test_standardize_dtype_complex128` | MLX has no complex128. |
| `test_standardize_dtype_float64` | MLX has no float64 on GPU; float64 maps to float32. |
| `test_argmin_negative_zero`, `test_argmax_negative_zero` | GPU flushes subnormals to zero (FTZ) → results differ from numpy. (Skipped for openvino / JAX-TPU too.) |
| `test_comprehensive_model_state_restoration` | MLX's compiled `__matmul__` ignores the `__mlx_array__` coercion protocol that `+ - *` honor, so `mlx_array @ KerasVariable` raises `TypeError` instead of reaching `Variable.__rmatmul__`. Affects custom layers using raw `inputs @ self.kernel`; use `ops.matmul` instead. |
| `test_quantize_and_dequantize`, `test_quantized_layer_with_remat` | float8-only quantization tests. |
| `steps_per_execution > 1` (orbax `test_training_resumption_2`) | Not supported by the MLX trainer (same as torch); the test now skips mlx alongside torch. |

---

## Bugs found and fixed (notable)

The with-tensorflow run is the meaningful verification: tensorflow made
previously-blocked tests **runnable**, which surfaced genuine MLX bugs that
would otherwise have shipped unnoticed. All fixed and individually re-verified:

- **`nn.py` `_pool` strided no-op** — `MaxPooling2D(pool_size=1, strides=2)`
  (ResNetV2/NASNet identity shortcut) returned the input unchanged, dropping the
  stride → residual `Add` shape mismatch. Now processes any axis with
  `strides[i] != 1`.
- **`image.py` `compute_homography_matrix`** — `mx.linalg.solve` is CPU-only;
  wrapped the solve in a CPU stream.
- **`image.py` `rgb_to_hsv`** — `np.finfo` rejects bfloat16 → `ml_dtypes.finfo`.
- **`core.py` `convert_to_numpy`** — routing non-MLX inputs through
  `convert_to_tensor` silently downcast float64→float32 on the `.keras` save
  round-trip; now short-circuits non-MLX arrays with `np.asarray`.
- **`trainer.py`** — `tree.flatten(x)[0].shape[0]` crashed on `None` optional
  inputs; now `next(i for i in tree.flatten(x) if i is not None)`.
- **`export.py`** — added `track_and_add_endpoint` raising the
  `NotImplementedError` `test_export_error` asserts.
- **Shared `numpy/image.py`** — `elastic_transform` seed: `draw_seed` returns an
  MLX array under this backend; `np.random.default_rng` rejected it → `np.asarray(seed)`.
- **Shared `utils/tf_utils.py` `ensure_tensor`** — MLX arrays hit
  `tf.convert_to_tensor` raw (`len() 0-d`); extended the torch pre-conversion branch to mlx.
- **Shared `random_crop.py`** — MLX int32 scalars aren't valid slice indices; coerce
  with `int()`, guarded so tf `SymbolicTensor` (tf.data graph path) is left for slicing.
- **Shared `layers/core/embedding.py` `enable_lora`** — set `.trainable` on the
  underlying `_embeddings` Variable, not the `embeddings` property (which returns a
  raw immutable MLX array for int4 layers).

Plus two test-side fixes: `ops.cross` reference `np.cross` fails on numpy ≥ 2.5
(dropped the dim-2 fallback — fails on numpy/jax backends too; MLX impl correct) →
version guard; `tensorboard_test` monkeypatches a `.numpy` attr MLX arrays lack →
skip mlx like torch.

---

## Non-obvious MLX behaviors

See the project memory (`mlx-backend-quirks`) for the full list. Highlights that
affect testing:

- MLX arrays are immutable (functional scatter via `.at[...]`); no `__array__`,
  no settable attributes, no `.numpy()` — use `backend.convert_to_numpy`.
- `linalg.solve`/`cholesky`/`inv` are **CPU-only** — wrap in a CPU stream.
- No float8/complex128, no float64 on GPU.
- `IS_THREAD_SAFE = False` (callbacks run on the main thread).
- `mx.tensordot(axes=...)` wants `[[a],[b]]`, not a tuple; `mx.conv_general`
  supports `groups != 1` only for 1D/2D (3D grouped convs use a split+concat fallback).
