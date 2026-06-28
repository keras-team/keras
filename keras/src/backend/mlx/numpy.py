import builtins
import math

import mlx.core as mx
import numpy as np

from keras.src import tree
from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import standardize_axis_for_numpy
from keras.src.backend.mlx.core import _mlx_dtype
from keras.src.backend.mlx.core import convert_to_numpy
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.core import is_tensor


def _cast(x, dtype):
    """Cast to a keras dtype string (handles mlx/numpy/tf dtype objects)."""
    return x.astype(_mlx_dtype(dtype))


def _to_np(x):
    """Materialize an mlx tensor (or pass through python/numpy) to numpy."""
    if is_tensor(x):
        return np.asarray(convert_to_numpy(x))
    return np.asarray(x)


def _from_np(arr, dtype=None):
    out = convert_to_tensor(arr)
    if dtype is not None:
        out = _cast(out, dtype)
    return out


# ---------------------------------------------------------------------------
# Elementwise arithmetic (with keras dtype promotion)
# ---------------------------------------------------------------------------


def add(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.add(x1, x2)


def subtract(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.subtract(x1, x2)


def multiply(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.multiply(x1, x2)


def divide(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
        float,
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.divide(x1, x2)


def divide_no_nan(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
        float,
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    safe = mx.divide(x1, mx.where(x2 == 0, 1, x2))
    return mx.where(x2 == 0, mx.zeros_like(x1), safe)


def true_divide(x1, x2):
    return divide(x1, x2)


def power(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.power(x1, x2)


def matmul(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    x1_dtype = standardize_dtype(x1.dtype)
    x2_dtype = standardize_dtype(x2.dtype)
    if x1_dtype == "int8" and x2_dtype == "int8":
        dtype = "int32"
    else:
        dtype = dtypes.result_type(x1.dtype, x2.dtype)
    # MLX matmul supports inexact (float/complex) types only. For integer
    # inputs fall back to numpy (which matches JAX's integer matmul dtype).
    # Cast to the wider result dtype first so numpy accumulates without
    # overflow (e.g. int8 @ int8 must accumulate in int32, not int8).
    if dtype in dtypes.INT_TYPES or dtype == "bool":
        out = np.matmul(_to_np(x1).astype(dtype), _to_np(x2).astype(dtype))
        return _cast(convert_to_tensor(out), dtype)
    x1 = _cast(x1, dtype)
    x2 = _cast(x2, dtype)
    return _cast(mx.matmul(x1, x2), dtype)


def einsum(subscripts, *operands, **kwargs):
    operands = tree.map_structure(convert_to_tensor, operands)
    dtypes_to_resolve = list(set(standardize_dtype(x.dtype) for x in operands))
    if len(dtypes_to_resolve) == 1 and dtypes_to_resolve[0] == "int8":
        compute_dtype = "int32"
        result_dtype = "int32"
    else:
        result_dtype = dtypes.result_type(*dtypes_to_resolve)
        compute_dtype = result_dtype
    # mx.einsum is backed by matmul and rejects integer types.
    # Cast to the wider result dtype first so numpy accumulates without
    # overflow (e.g. int8 einsum must accumulate in int32, not int8).
    if result_dtype in dtypes.INT_TYPES or result_dtype == "bool":
        np_ops = [_to_np(x).astype(result_dtype) for x in operands]
        out = np.einsum(subscripts, *np_ops, **kwargs)
        return _cast(convert_to_tensor(out), result_dtype)
    # mx.einsum does not support bfloat16.
    if compute_dtype == "bfloat16":
        compute_dtype = "float32"
    operands = tree.map_structure(lambda x: _cast(x, compute_dtype), operands)
    return _cast(mx.einsum(subscripts, *operands, **kwargs), result_dtype)


# ---------------------------------------------------------------------------
# Reductions / statistics
# ---------------------------------------------------------------------------


def _accum_dtype(dtype):
    """JAX-aligned accumulation dtype widening."""
    if dtype in ("bool", "int8", "int16"):
        return "int32"
    elif dtype in ("uint8", "uint16"):
        return "uint32"
    return dtype


def sum(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtype = _accum_dtype(standardize_dtype(x.dtype))
    return _cast(mx.sum(x, axis=axis, keepdims=keepdims), dtype)


def prod(x, axis=None, keepdims=False, dtype=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    if dtype is None:
        dtype = _accum_dtype(standardize_dtype(x.dtype))
    return _cast(mx.prod(x, axis=axis, keepdims=keepdims), dtype)


def mean(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        result_dtype = dtypes.result_type(x.dtype, "float32")
        compute_dtype = result_dtype
    elif ori_dtype in ("float16", "bfloat16"):
        # Avoid float16/bfloat16 accumulation overflow (numpy upcasts).
        result_dtype = ori_dtype
        compute_dtype = "float32"
    else:
        result_dtype = ori_dtype
        compute_dtype = ori_dtype
    return _cast(
        mx.mean(_cast(x, compute_dtype), axis=axis, keepdims=keepdims),
        result_dtype,
    )


def max(x, axis=None, keepdims=False, initial=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    if initial is None:
        return mx.max(x, axis=axis, keepdims=keepdims)
    init = _cast(mx.array(initial), standardize_dtype(x.dtype))
    total = 1
    for d in x.shape:
        total *= d
    if total == 0:
        # Reduction over zero elements: numpy returns `initial`.
        if axis is None:
            return mx.broadcast_to(init, ())
        axes = axis if isinstance(axis, (list, tuple)) else [axis]
        out_shape = [s for i, s in enumerate(x.shape) if i not in axes]
        return mx.broadcast_to(init, out_shape)
    return mx.maximum(mx.max(x, axis=axis, keepdims=keepdims), init)


def min(x, axis=None, keepdims=False, initial=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    if initial is None:
        return mx.min(x, axis=axis, keepdims=keepdims)
    init = _cast(mx.array(initial), standardize_dtype(x.dtype))
    total = 1
    for d in x.shape:
        total *= d
    if total == 0:
        if axis is None:
            return mx.broadcast_to(init, ())
        axes = axis if isinstance(axis, (list, tuple)) else [axis]
        out_shape = [s for i, s in enumerate(x.shape) if i not in axes]
        return mx.broadcast_to(init, out_shape)
    return mx.minimum(mx.min(x, axis=axis, keepdims=keepdims), init)


def amax(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    return mx.max(convert_to_tensor(x), axis=axis, keepdims=keepdims)


def amin(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    return mx.min(convert_to_tensor(x), axis=axis, keepdims=keepdims)


def std(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = _cast(x, config.floatx())
    return mx.std(x, axis=axis, keepdims=keepdims)


def var(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    result_dtype = dtypes.result_type(x.dtype, float)
    return _cast(
        mx.var(_cast(x, compute_dtype), axis=axis, keepdims=keepdims),
        result_dtype,
    )


def cumsum(x, axis=None, dtype=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    return _cast(mx.cumsum(_cast(x, dtype), axis=axis), dtype)


def cumprod(x, axis=None, dtype=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    return _cast(mx.cumprod(_cast(x, dtype), axis=axis), dtype)


def average(x, axis=None, weights=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtypes_to_resolve = [x.dtype, float]
    if weights is not None:
        weights = convert_to_tensor(weights)
        dtypes_to_resolve.append(weights.dtype)
    dtype = dtypes.result_type(*dtypes_to_resolve)
    x = _cast(x, dtype)
    if weights is None:
        return mx.mean(x, axis=axis, keepdims=False)
    weights = _cast(weights, dtype)
    # numpy allows 1-D weights applied along `axis` for N-D inputs.
    if weights.ndim < x.ndim:
        shape = [1] * x.ndim
        shape[axis if isinstance(axis, int) else 0] = weights.shape[0]
        weights = mx.reshape(weights, shape)
    num = mx.sum(mx.multiply(x, weights), axis=axis, keepdims=False)
    den = mx.sum(weights, axis=axis, keepdims=False)
    return mx.divide(num, den)


# ---------------------------------------------------------------------------
# NaN-handling reductions (compose with mx.isnan + plain reductions)
# ---------------------------------------------------------------------------


def nanmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if axis == ():
        return x
    axis = standardize_axis_for_numpy(axis)
    d = standardize_dtype(x.dtype)
    mask = mx.isnan(x)
    neg_inf = _cast(mx.full(x.shape, -np.inf), d)
    reduced = mx.max(mx.where(mask, neg_inf, x), axis=axis, keepdims=keepdims)
    all_nan = mx.all(mask, axis=axis, keepdims=keepdims)
    return mx.where(all_nan, _cast(mx.full(reduced.shape, np.nan), d), reduced)


def nanmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if axis == ():
        return x
    axis = standardize_axis_for_numpy(axis)
    d = standardize_dtype(x.dtype)
    mask = mx.isnan(x)
    pos_inf = _cast(mx.full(x.shape, np.inf), d)
    reduced = mx.min(mx.where(mask, pos_inf, x), axis=axis, keepdims=keepdims)
    all_nan = mx.all(mask, axis=axis, keepdims=keepdims)
    return mx.where(all_nan, _cast(mx.full(reduced.shape, np.nan), d), reduced)


def nansum(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtype = _accum_dtype(standardize_dtype(x.dtype))
    zeros = _cast(mx.zeros_like(x), dtype)
    x = mx.where(mx.isnan(x), zeros, x)
    return _cast(mx.sum(_cast(x, dtype), axis=axis, keepdims=keepdims), dtype)


def nanprod(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtype = _accum_dtype(standardize_dtype(x.dtype))
    ones = _cast(mx.ones_like(x), dtype)
    x = mx.where(mx.isnan(x), ones, x)
    return _cast(mx.prod(_cast(x, dtype), axis=axis, keepdims=keepdims), dtype)


def _squeeze_reduced(out, axis, keepdims):
    """Squeeze the reduced axes when the caller did not request keepdims."""
    if keepdims:
        return out
    if axis is None:
        return mx.squeeze(out)
    axes = axis if isinstance(axis, (list, tuple)) else [axis]
    for ax in sorted(axes, reverse=True):
        out = mx.squeeze(out, axis=ax)
    return out


def nanmean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = standardize_axis_for_numpy(axis)
    dtype = dtypes.result_type(standardize_dtype(x.dtype), float)
    mask = mx.isnan(x)
    zeros = _cast(mx.zeros_like(x), dtype)
    xc = mx.where(mask, zeros, _cast(x, dtype))
    valid = mx.where(mask, zeros, _cast(mx.ones_like(x), dtype))
    count = mx.sum(valid, axis=axis, keepdims=True)
    meanv = mx.divide(mx.sum(xc, axis=axis, keepdims=True), count)
    return _cast(_squeeze_reduced(meanv, axis, keepdims), dtype)


def nanstd(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = standardize_axis_for_numpy(axis)
    result_dtype = dtypes.result_type(x.dtype, float)
    mask = mx.isnan(x)
    zeros = _cast(mx.zeros_like(x), result_dtype)
    xc = mx.where(mask, zeros, _cast(x, result_dtype))
    valid = mx.where(mask, zeros, _cast(mx.ones_like(x), result_dtype))
    count = mx.sum(valid, axis=axis, keepdims=True)
    meanv = mx.divide(mx.sum(xc, axis=axis, keepdims=True), count)
    dev = mx.where(mask, zeros, xc - meanv)
    var = mx.divide(mx.sum(mx.square(dev), axis=axis, keepdims=True), count)
    out = mx.sqrt(var)
    return _cast(_squeeze_reduced(out, axis, keepdims), result_dtype)


def nanvar(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = standardize_axis_for_numpy(axis)
    result_dtype = dtypes.result_type(x.dtype, float)
    mask = mx.isnan(x)
    zeros = _cast(mx.zeros_like(x), result_dtype)
    xc = mx.where(mask, zeros, _cast(x, result_dtype))
    valid = mx.where(mask, zeros, _cast(mx.ones_like(x), result_dtype))
    count = mx.sum(valid, axis=axis, keepdims=True)
    meanv = mx.divide(mx.sum(xc, axis=axis, keepdims=True), count)
    dev = mx.where(mask, zeros, xc - meanv)
    var = mx.divide(mx.sum(mx.square(dev), axis=axis, keepdims=True), count)
    return _cast(_squeeze_reduced(var, axis, keepdims), result_dtype)


def nancumsum(x, axis=None, dtype=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    zeros = _cast(mx.zeros_like(x), dtype)
    x = mx.where(mx.isnan(x), zeros, _cast(x, dtype))
    return _cast(mx.cumsum(x, axis=axis), dtype)


def nancumprod(x, axis=None, dtype=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    ones = _cast(mx.ones_like(x), dtype)
    x = mx.where(mx.isnan(x), ones, _cast(x, dtype))
    return _cast(mx.cumprod(x, axis=axis), dtype)


def nanargmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if "float" not in standardize_dtype(x.dtype):
        return argmax(x, axis=axis, keepdims=keepdims)
    nan_mask = mx.isnan(x)
    return mx.where(
        mx.all(nan_mask, axis=axis, keepdims=keepdims),
        _cast(mx.full((), -1), mx.int32),
        argmax(
            mx.where(
                nan_mask,
                _cast(mx.full(x.shape, -np.inf), standardize_dtype(x.dtype)),
                x,
            ),
            axis=axis,
            keepdims=keepdims,
        ),
    )


def nanargmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if "float" not in standardize_dtype(x.dtype):
        return argmin(x, axis=axis, keepdims=keepdims)
    nan_mask = mx.isnan(x)
    return mx.where(
        mx.all(nan_mask, axis=axis, keepdims=keepdims),
        _cast(mx.full((), -1), mx.int32),
        argmin(
            mx.where(
                nan_mask,
                _cast(mx.full(x.shape, np.inf), standardize_dtype(x.dtype)),
                x,
            ),
            axis=axis,
            keepdims=keepdims,
        ),
    )


# ---------------------------------------------------------------------------
# Extrema
# ---------------------------------------------------------------------------


def maximum(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.maximum(x1, x2)


def minimum(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.minimum(x1, x2)


def fmax(x1, x2):
    # NaNs propagate to the non-nan operand; mx.maximum already propagates nan,
    # so emulate fmax by treating nan as ignored.
    x1 = convert_to_tensor(x1) if not isinstance(x1, (int, float)) else x1
    x2 = convert_to_tensor(x2) if not isinstance(x2, (int, float)) else x2
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    out = mx.maximum(x1, x2)
    out = mx.where(mx.isnan(x1), x2, out)
    out = mx.where(mx.isnan(x2), x1, out)
    return out


def fmin(x1, x2):
    x1 = convert_to_tensor(x1) if not isinstance(x1, (int, float)) else x1
    x2 = convert_to_tensor(x2) if not isinstance(x2, (int, float)) else x2
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    out = mx.minimum(x1, x2)
    out = mx.where(mx.isnan(x1), x2, out)
    out = mx.where(mx.isnan(x2), x1, out)
    return out


# ---------------------------------------------------------------------------
# Unary math / trig (with int->float upcast)
# ---------------------------------------------------------------------------


def _unary_float(x, fn):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        dtype = dtypes.result_type(x.dtype, float)
    return fn(_cast(x, dtype))


# A CPU stream for ops that need float64 precision (not available on the GPU).
_CPU = mx.Device(mx.DeviceType.cpu, 0)


def _unary_float64(x, fn):
    """Compute a unary op in float64 on the CPU and cast back.

    MLX's float32 transcendentals can round less accurately than numpy's
    correctly-rounded float32 (e.g. ``expm1(1)`` is off by ~4e-6, exceeding the
    1e-6 test tolerance). float64 is CPU-only in MLX but computes a value we can
    round back to the target dtype for a correct result.
    """
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        dtype = dtypes.result_type(x.dtype, float)
    target = _mlx_dtype(dtype)
    # Keep the float64 value AND the cast back to the target dtype on the CPU
    # stream: MLX rejects float64 arrays on the GPU stream entirely.
    with mx.stream(_CPU):
        r = fn(x.astype(mx.float64))
        r = r.astype(target)
    return r


def absolute(x):
    return mx.abs(convert_to_tensor(x))


def abs(x):
    return absolute(x)


def fabs(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        x = _cast(x, config.floatx())
    return mx.abs(x)


def exp(x):
    return _unary_float(x, mx.exp)


def exp2(x):
    return _unary_float(x, lambda v: mx.power(2, v))


def expm1(x):
    return _unary_float64(x, mx.expm1)


def log(x):
    return _unary_float(x, mx.log)


def log1p(x):
    return _unary_float(x, mx.log1p)


def log2(x):
    return _unary_float(x, mx.log2)


def log10(x):
    return _unary_float(x, mx.log10)


def sqrt(x):
    return _unary_float(x, mx.sqrt)


def rsqrt(x):
    return _unary_float(x, mx.rsqrt)


def square(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "bool":
        x = _cast(x, "int32")
    return mx.square(x)


def cbrt(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype in ["bool", "int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        dtype = config.floatx()
    elif dtype == "int64":
        dtype = "float32"
    x = _cast(x, dtype)
    # numpy's `cbrt` keeps the sign for negative inputs; `power(x, 1/3)` would
    # return NaN there. Use `sign(x) * abs(x) ** (1/3)` (also numpy's reference
    # formula in the op tests), which is differentiable for x != 0.
    return mx.sign(x) * mx.power(mx.abs(x), 1.0 / 3.0)


def sin(x):
    return _unary_float(x, mx.sin)


def cos(x):
    return _unary_float(x, mx.cos)


def tan(x):
    return _unary_float(x, mx.tan)


def sinh(x):
    return _unary_float(x, mx.sinh)


def cosh(x):
    return _unary_float(x, mx.cosh)


def tanh(x):
    return _unary_float(x, mx.tanh)


def arcsin(x):
    return _unary_float(x, mx.arcsin)


def arccos(x):
    return _unary_float(x, mx.arccos)


def arctan(x):
    return _unary_float(x, mx.arctan)


def arcsinh(x):
    return _unary_float(x, mx.arcsinh)


def arccosh(x):
    return _unary_float(x, mx.arccosh)


def arctanh(x):
    return _unary_float(x, mx.arctanh)


def arctan2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    return mx.arctan2(_cast(x1, dtype), _cast(x2, dtype))


def sigmoid(x):
    return mx.sigmoid(convert_to_tensor(x))


def sign(x):
    return mx.sign(convert_to_tensor(x))


def negative(x):
    return mx.negative(convert_to_tensor(x))


def reciprocal(x):
    x = convert_to_tensor(x)
    return mx.reciprocal(x)


def clip(x, x_min, x_max):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype == "bool":
        dtype = "int32"
    return _cast(mx.clip(x, x_min, x_max), dtype)


def ceil(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    return mx.ceil(_cast(x, dtype))


def floor(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    return mx.floor(_cast(x, dtype))


def round(x, decimals=0):
    x = convert_to_tensor(x)
    if decimals == 0:
        return mx.round(x)
    # `10 ** decimals` as a python float avoids integer power (which yields 0
    # for negative exponents and then NaNs the division).
    factor = float(10**decimals)
    return mx.round(x * factor) / factor


def trunc(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        return x
    return mx.where(x >= 0, mx.floor(x), mx.ceil(x))


def deg2rad(x):
    x = convert_to_tensor(x)
    dtype = _angle_dtype(x)
    return _cast(mx.multiply(_cast(x, dtype), mx.array(np.pi / 180.0)), dtype)


def rad2deg(x):
    x = convert_to_tensor(x)
    dtype = _angle_dtype(x)
    return _cast(mx.multiply(_cast(x, dtype), mx.array(180.0 / np.pi)), dtype)


def _angle_dtype(x):
    d = standardize_dtype(x.dtype)
    if d in ["int64", "float64"]:
        return "float32"  # mlx has no float64 on GPU
    if d in ["bfloat16", "float16"]:
        return d
    return config.floatx()


def angle(x):
    # MLX has only complex64. For real input, the phase is arctan2(0, x)
    # (pi for negative reals, 0 otherwise) — matching numpy.
    x = convert_to_tensor(x)
    if str(x.dtype) == "mlx.core.complex64":
        return mx.atan2(mx.imag(x), mx.real(x))
    d = standardize_dtype(x.dtype)
    out_dtype = config.floatx() if (d in dtypes.INT_TYPES or d == "bool") else d
    xf = _cast(x, out_dtype)
    return mx.arctan2(mx.zeros(xf.shape, dtype=_mlx_dtype(out_dtype)), xf)


def real(x):
    x = convert_to_tensor(x)
    if str(x.dtype) == "mlx.core.complex64":
        return mx.real(x)
    return x


def imag(x):
    x = convert_to_tensor(x)
    if str(x.dtype) == "mlx.core.complex64":
        return mx.imag(x)
    return mx.zeros_like(x)


def conjugate(x):
    x = convert_to_tensor(x)
    return mx.conjugate(x)


def conj(x):
    return conjugate(x)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if posinf is None:
        posinf = (
            np.finfo(np.float32).max
            if dtype != "float64"
            else np.finfo(np.float64).max
        )
    if neginf is None:
        neginf = (
            np.finfo(np.float32).min
            if dtype != "float64"
            else np.finfo(np.float64).min
        )
    out = mx.where(mx.isnan(x), _cast(mx.array(nan), dtype), x)
    out = mx.where(mx.isposinf(out), _cast(mx.array(posinf), dtype), out)
    out = mx.where(mx.isneginf(out), _cast(mx.array(neginf), dtype), out)
    return out


# ---------------------------------------------------------------------------
# Log-sum-exp / softplus
# ---------------------------------------------------------------------------


def logaddexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    return _cast(mx.logaddexp(_cast(x1, dtype), _cast(x2, dtype)), dtype)


def logaddexp2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    a = mx.power(2, _cast(x1, dtype))
    b = mx.power(2, _cast(x2, dtype))
    return _cast(mx.log2(a + b), dtype)


# ---------------------------------------------------------------------------
# Logical / comparison / classification predicates
# ---------------------------------------------------------------------------


def all(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    return mx.all(convert_to_tensor(x), axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    return mx.any(convert_to_tensor(x), axis=axis, keepdims=keepdims)


def allclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.allclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


def equal(x1, x2):
    return mx.equal(convert_to_tensor(x1), convert_to_tensor(x2))


def not_equal(x1, x2):
    return mx.not_equal(convert_to_tensor(x1), convert_to_tensor(x2))


def greater(x1, x2):
    return mx.greater(convert_to_tensor(x1), convert_to_tensor(x2))


def greater_equal(x1, x2):
    return mx.greater_equal(convert_to_tensor(x1), convert_to_tensor(x2))


def less(x1, x2):
    return mx.less(convert_to_tensor(x1), convert_to_tensor(x2))


def less_equal(x1, x2):
    return mx.less_equal(convert_to_tensor(x1), convert_to_tensor(x2))


def logical_and(x1, x2):
    return mx.logical_and(convert_to_tensor(x1), convert_to_tensor(x2))


def logical_or(x1, x2):
    return mx.logical_or(convert_to_tensor(x1), convert_to_tensor(x2))


def logical_not(x):
    return mx.logical_not(convert_to_tensor(x))


def logical_xor(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.not_equal(x1, x2)


def count_nonzero(x, axis=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    nonzero = mx.not_equal(
        x, _cast(mx.zeros_like(x), standardize_dtype(x.dtype))
    )
    return _cast(mx.sum(nonzero, axis=axis, keepdims=False), "int32")


def isfinite(x):
    return mx.isfinite(convert_to_tensor(x))


def isinf(x):
    return mx.isinf(convert_to_tensor(x))


def isnan(x):
    return mx.isnan(convert_to_tensor(x))


def isneginf(x):
    return mx.isneginf(convert_to_tensor(x))


def isposinf(x):
    return mx.isposinf(convert_to_tensor(x))


def isreal(x):
    x = convert_to_tensor(x)
    if str(x.dtype) == "mlx.core.complex64":
        return mx.equal(
            mx.imag(x),
            _cast(
                mx.zeros_like(mx.imag(x)),
                standardize_dtype(mx.imag(x).dtype),
            ),
        )
    return mx.ones_like(x).astype(mx.bool_)


def isin(x1, x2, assume_unique=False, invert=False):
    # numpy fallback (data-dependent membership is rarely on the hot path).
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result = np.isin(
        _to_np(x1),
        _to_np(x2),
        assume_unique=assume_unique,
        invert=invert,
    )
    return convert_to_tensor(result)


def signbit(x):
    # numpy fallback: signbit inspects the sign bit (incl. -0.0), which MLX
    # cannot express via value comparison.
    x = convert_to_tensor(x)
    return convert_to_tensor(np.signbit(_to_np(x)))


def where(condition, x1=None, x2=None):
    if x1 is not None and x2 is not None:
        if not isinstance(x1, (int, float)):
            x1 = convert_to_tensor(x1)
        if not isinstance(x2, (int, float)):
            x2 = convert_to_tensor(x2)
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        x1 = convert_to_tensor(x1, dtype)
        x2 = convert_to_tensor(x2, dtype)
        return mx.where(condition, x1, x2)
    else:
        # Index form (data-dependent shape) -> numpy fallback.
        result = np.nonzero(_to_np(convert_to_tensor(condition)))
        return tuple(
            convert_to_tensor(indices).astype(mx.int32) for indices in result
        )


def select(condlist, choicelist, default=0):
    # Chain of mx.where, right fold from default.
    condlist = [convert_to_tensor(c) for c in condlist]
    choicelist = [convert_to_tensor(c) for c in choicelist]
    dtype = dtypes.result_type(
        *[c.dtype for c in choicelist],
        type(default) if not isinstance(default, (int, float)) else float,
    )
    out = _cast(mx.full((), default), dtype)
    out = mx.broadcast_to(out, condlist[0].shape) if condlist else out
    for cond, choice in zip(condlist, choicelist):
        out = mx.where(cond, _cast(choice, dtype), out)
    return out


def nonzero(x):
    result = np.nonzero(_to_np(convert_to_tensor(x)))
    return tuple(
        convert_to_tensor(indices).astype(mx.int32) for indices in result
    )


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------


def reshape(x, newshape):
    x = convert_to_tensor(x)
    if isinstance(newshape, int):
        newshape = (newshape,)
    return mx.reshape(x, newshape)


def ravel(x):
    return mx.reshape(convert_to_tensor(x), (-1,))


def transpose(x, axes=None):
    axes = tuple(axes) if isinstance(axes, list) else axes
    return mx.transpose(convert_to_tensor(x), axes=axes)


def moveaxis(x, source, destination):
    x = convert_to_tensor(x)
    # MLX moveaxis takes scalar source/destination; compose the permutation
    # following numpy's algorithm (normalize negative axes first).
    nd = x.ndim
    if isinstance(source, int):
        source = [source]
        destination = [destination]
    else:
        source = list(source)
        destination = list(destination)
    source = [s % nd for s in source]
    destination = [d % nd for d in destination]
    order = [i for i in range(nd) if i not in source]
    for dest, src in sorted(zip(destination, source), key=lambda t: t[0]):
        order.insert(dest, src)
    return mx.transpose(x, axes=order)


def swapaxes(x, axis1, axis2):
    return mx.swapaxes(convert_to_tensor(x), axis1, axis2)


def _flip(x, axis):
    """Reverse `x` along each axis in `axis` (MLX has no `flip` primitive)."""
    if axis is None:
        axis = list(range(x.ndim))
    elif isinstance(axis, int):
        axis = [axis]
    for ax in axis:
        sl = [slice(None)] * x.ndim
        sl[ax] = slice(None, None, -1)
        x = x[tuple(sl)]
    return x


def flip(x, axis=None):
    axis = standardize_axis_for_numpy(axis)
    return _flip(convert_to_tensor(x), axis)


def fliplr(x):
    return _flip(convert_to_tensor(x), 1)


def flipud(x):
    return _flip(convert_to_tensor(x), 0)


def roll(x, shift, axis=None):
    axis = standardize_axis_for_numpy(axis)
    return mx.roll(convert_to_tensor(x), shift, axis=axis)


def squeeze(x, axis=None):
    axis = standardize_axis_for_numpy(axis)
    return mx.squeeze(convert_to_tensor(x), axis=axis)


def expand_dims(x, axis):
    axis = standardize_axis_for_numpy(axis)
    return mx.expand_dims(convert_to_tensor(x), axis)


def broadcast_to(x, shape):
    return mx.broadcast_to(convert_to_tensor(x), shape)


def tile(x, repeats):
    return mx.tile(convert_to_tensor(x), repeats)


def repeat(x, repeats, axis=None):
    x = convert_to_tensor(x)
    axis = standardize_axis_for_numpy(axis)
    # mx.repeat only accepts an integer `repeats`; numpy also allows arrays.
    if isinstance(repeats, int):
        return mx.repeat(x, repeats, axis=axis)
    return convert_to_tensor(np.repeat(_to_np(x), repeats, axis=axis))


def concatenate(xs, axis=0):
    axis = standardize_axis_for_numpy(axis)
    dtype_set = set([getattr(x, "dtype", type(x)) for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = tree.map_structure(
            lambda x: _cast(convert_to_tensor(x), dtype), xs
        )
    else:
        xs = [convert_to_tensor(x) for x in xs]
    return mx.concatenate(xs, axis=axis)


def stack(x, axis=0):
    axis = standardize_axis_for_numpy(axis)
    dtype_set = set([getattr(a, "dtype", type(a)) for a in x])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        x = tree.map_structure(lambda a: _cast(convert_to_tensor(a), dtype), x)
    else:
        x = [convert_to_tensor(a) for a in x]
    return mx.stack(x, axis=axis)


def append(x1, x2, axis=None):
    axis = standardize_axis_for_numpy(axis)
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    return mx.concatenate([_cast(x1, dtype), _cast(x2, dtype)], axis=axis)


def split(x, indices_or_sections, axis=0):
    axis = standardize_axis_for_numpy(axis)
    return mx.split(convert_to_tensor(x), indices_or_sections, axis=axis)


def array_split(x, indices_or_sections, axis=0):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    n = x.shape[axis]
    if isinstance(indices_or_sections, int):
        # np.array_split semantics: uneven sizes allowed.
        sections = indices_or_sections
        base = n // sections
        rem = n % sections
        sizes = [base + 1 if i < rem else base for i in range(sections)]
        indices = []
        acc = 0
        for s in sizes[:-1]:
            acc += s
            indices.append(acc)
        return mx.split(x, indices, axis=axis)
    return mx.split(x, indices_or_sections, axis=axis)


def vsplit(x, indices_or_sections):
    return split(convert_to_tensor(x), indices_or_sections, axis=0)


def hsplit(x, indices_or_sections):
    # numpy hsplit: along axis=1 for ndim>=2, along axis=0 for 1-D inputs.
    x = convert_to_tensor(x)
    axis = 1 if x.ndim >= 2 else 0
    return split(x, indices_or_sections, axis=axis)


def dsplit(x, indices_or_sections):
    return split(convert_to_tensor(x), indices_or_sections, axis=2)


def vstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    xs = [mx.atleast_2d(x) for x in xs]
    return concatenate(xs, axis=0)


def hstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    xs = [mx.atleast_1d(x) for x in xs]
    if builtins.all(x.ndim == 1 for x in xs):
        return concatenate(xs, axis=0)
    return concatenate(xs, axis=1)


def dstack(xs):
    dtype_set = set([getattr(x, "dtype", type(x)) for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = tree.map_structure(
            lambda x: _cast(convert_to_tensor(x), dtype), xs
        )
    else:
        xs = [convert_to_tensor(x) for x in xs]
    xs = [mx.atleast_3d(x) for x in xs]
    return mx.concatenate(xs, axis=2)


def flatten(x):
    return mx.flatten(convert_to_tensor(x))


def unflatten(x, axis, shape):
    return mx.unflatten(convert_to_tensor(x), axis, shape)


def rot90(array, k=1, axes=(0, 1)):
    array = convert_to_tensor(array)
    if array.ndim < 2:
        raise ValueError(
            "Input array must have at least 2 dimensions. "
            f"Received: array.ndim={array.ndim}"
        )
    if len(axes) != 2 or axes[0] == axes[1]:
        raise ValueError(
            f"Invalid axes: {axes}. Axes must be a tuple "
            "of two different dimensions."
        )
    axes = (
        canonicalize_axis(axes[0], array.ndim),
        canonicalize_axis(axes[1], array.ndim),
    )
    k = k % 4
    out = array
    for _ in range(k):
        out = mx.swapaxes(out, axes[0], axes[1])
        out = _flip(out, axes[0])
    return out


def squeeze_dims(x, axis):
    return squeeze(x, axis=axis)


# ---------------------------------------------------------------------------
# Array creation
# ---------------------------------------------------------------------------


def array(x, dtype=None):
    return convert_to_tensor(x, dtype=dtype)


def zeros(shape, dtype=None):
    dtype = dtype or config.floatx()
    return mx.zeros(shape, dtype=_mlx_dtype(dtype))


def ones(shape, dtype=None):
    dtype = dtype or config.floatx()
    return mx.ones(shape, dtype=_mlx_dtype(dtype))


def full(shape, fill_value, dtype=None):
    dtype = dtype or config.floatx()
    # mx.full can ignore `dtype` for scalar shapes with int fill_value; cast.
    out = mx.full(shape, fill_value, dtype=_mlx_dtype(dtype))
    return out.astype(_mlx_dtype(dtype))


def zeros_like(x, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        return mx.zeros_like(x)
    return mx.zeros(x.shape, dtype=_mlx_dtype(dtype))


def ones_like(x, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        return mx.ones_like(x)
    return mx.ones(x.shape, dtype=_mlx_dtype(dtype))


def full_like(x, fill_value, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        out = mx.full(x.shape, fill_value, dtype=x.dtype)
        return out.astype(x.dtype)
    out = mx.full(x.shape, fill_value, dtype=_mlx_dtype(dtype))
    return out.astype(_mlx_dtype(dtype))


def empty(shape, dtype=None):
    dtype = dtype or config.floatx()
    return mx.zeros(shape, dtype=_mlx_dtype(dtype))


def empty_like(x, dtype=None):
    return zeros_like(x, dtype=dtype)


def eye(N, M=None, k=0, dtype=None):
    dtype = dtype or config.floatx()
    M = N if M is None else M
    out = mx.zeros((N, M), dtype=_mlx_dtype(dtype))
    # k offset along rows/cols.
    if k >= 0:
        n_diag = builtins.min(N, M - k)
        idx = mx.arange(n_diag)
        out = out.at[(idx, idx + k)].add(_cast(mx.ones((n_diag,)), dtype))
    else:
        n_diag = builtins.min(N + k, M)
        idx = mx.arange(n_diag)
        out = out.at[(idx - k, idx)].add(_cast(mx.ones((n_diag,)), dtype))
    return out


def identity(n, dtype=None):
    return eye(n, dtype=dtype)


def arange(start, stop=None, step=None, dtype=None):
    if dtype is None:
        dtypes_to_resolve = [getattr(start, "dtype", type(start))]
        if stop is not None:
            dtypes_to_resolve.append(getattr(stop, "dtype", type(stop)))
        if step is not None:
            dtypes_to_resolve.append(getattr(step, "dtype", type(step)))
        dtype = dtypes.result_type(*dtypes_to_resolve)
    if stop is None:
        start, stop = 0, start
    if step is None:
        step = 1
    # MLX arange only accepts python int/float scalars for start/stop/step,
    # not mlx.core.array or numpy scalars. Coerce them to python scalars.
    start = _arange_scalar(start)
    stop = _arange_scalar(stop)
    step = _arange_scalar(step)
    # MLX arange takes (start, stop, step) positionally; keyword form differs.
    return mx.arange(start, stop, step, dtype=_mlx_dtype(dtype))


def _arange_scalar(x):
    if x is None or isinstance(x, (int, float)):
        return x
    if isinstance(x, mx.array):
        x = convert_to_numpy(x)
    if isinstance(x, (np.ndarray, np.number)):
        return x.item()
    return x


def linspace(
    start,
    stop,
    num=50,
    endpoint=True,
    retstep=False,
    dtype=None,
    axis=0,
):
    # `num` may arrive as an MLX/numpy tensor (e.g. convert_to_tensor(5)).
    if isinstance(num, mx.array):
        num = int(convert_to_numpy(num))
    elif isinstance(num, np.ndarray):
        num = int(num)
    if dtype is None:
        dtype = dtypes.result_type(
            getattr(start, "dtype", type(start)),
            getattr(stop, "dtype", type(stop)),
            float,
        )
    start_np = (
        start
        if isinstance(start, (int, float))
        else _to_np(convert_to_tensor(start))
    )
    stop_np = (
        stop
        if isinstance(stop, (int, float))
        else _to_np(convert_to_tensor(stop))
    )
    result = np.linspace(
        start_np, stop_np, num, endpoint=endpoint, retstep=retstep, axis=axis
    )
    if retstep:
        out, step = result
        return _cast(convert_to_tensor(out), dtype), step
    return _cast(convert_to_tensor(result), dtype)


def meshgrid(*x, indexing="xy"):
    return mx.meshgrid(*[convert_to_tensor(a) for a in x], indexing=indexing)


# ---------------------------------------------------------------------------
# Copy / view
# ---------------------------------------------------------------------------


def copy(x):
    return mx.array(convert_to_tensor(x))


_NP_VIEW_DTYPES = {
    "bool": np.bool_,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "complex64": np.complex64,
}


def _np_view_dtype(dtype):
    """Map a keras dtype string to a numpy dtype for bit-reinterpretation.
    `bfloat16`/`float8_*` require `ml_dtypes`."""
    if dtype in _NP_VIEW_DTYPES:
        return _NP_VIEW_DTYPES[dtype]
    import ml_dtypes

    return getattr(ml_dtypes, dtype)


def view(x, dtype=None):
    # MLX arrays are immutable and have no bit-reinterpret; use numpy. The
    # bit-reinterpret target may be `bfloat16` (via `ml_dtypes`).
    if dtype is None:
        return convert_to_tensor(x)
    target = standardize_dtype(dtype)
    np_x = _to_np(convert_to_tensor(x))
    viewed = np_x.view(_np_view_dtype(target))
    if target in _NP_VIEW_DTYPES:
        return convert_to_tensor(viewed)
    # `bfloat16`/float8 numpy arrays: feed raw values via float, then cast.
    return mx.array(viewed).astype(_mlx_dtype(target))


# ---------------------------------------------------------------------------
# Diagonal / triangular
# ---------------------------------------------------------------------------


def diag(x, k=0):
    return mx.diag(convert_to_tensor(x), k=k)


def diagflat(x, k=0):
    x = convert_to_tensor(x)
    flat = mx.reshape(x, (-1,))
    n = flat.shape[0]
    # numpy makes the matrix (n + |k|) on a side so the offset diagonal fits.
    size = n + abs(k)
    out = mx.zeros((size, size), dtype=x.dtype)
    idx = mx.arange(n)
    if k >= 0:
        out = out.at[(idx, idx + k)].add(flat)
    else:
        out = out.at[(idx - k, idx)].add(flat)
    return out


def diagonal(x, offset=0, axis1=0, axis2=1):
    axis1 = standardize_axis_for_numpy(axis1)
    axis2 = standardize_axis_for_numpy(axis2)
    return mx.diagonal(
        convert_to_tensor(x), offset=offset, axis1=axis1, axis2=axis2
    )


def trace(x, offset=0, axis1=0, axis2=1):
    axis1 = standardize_axis_for_numpy(axis1)
    axis2 = standardize_axis_for_numpy(axis2)
    x = convert_to_tensor(x)
    dtype = _accum_dtype(standardize_dtype(x.dtype))
    return _cast(
        mx.trace(_cast(x, dtype), offset=offset, axis1=axis1, axis2=axis2),
        dtype,
    )


def tri(N, M=None, k=0, dtype=None):
    dtype = dtype or config.floatx()
    M = N if M is None else M
    # Element [i, j] is 1 iff j <= i + k, i.e. (i + k) >= j.
    i = mx.arange(N, dtype=mx.int32)[:, None]
    j = mx.arange(M, dtype=mx.int32)[None, :]
    mask = (i + k) >= j
    return _cast(mask, dtype)


def tril(x, k=0):
    return mx.tril(convert_to_tensor(x), k=k)


def triu(x, k=0):
    return mx.triu(convert_to_tensor(x), k=k)


# ---------------------------------------------------------------------------
# Dot products
# ---------------------------------------------------------------------------


def dot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    # MLX has no `dot`; compose it, and fall back to numpy for integer types
    # (matmul is float-only).
    if dtype in dtypes.INT_TYPES or dtype == "bool":
        return _cast(convert_to_tensor(np.dot(_to_np(x1), _to_np(x2))), dtype)
    x1 = _cast(x1, dtype)
    x2 = _cast(x2, dtype)
    if x1.ndim == 0 or x2.ndim == 0:
        return mx.multiply(x1, x2)
    if x1.ndim == 1 and x2.ndim == 1:
        return mx.sum(mx.multiply(x1, x2))
    if x1.ndim <= 2 and x2.ndim <= 2:
        return mx.matmul(x1, x2)
    # N-D dot: contract last axis of x1 with second-to-last of x2.
    return mx.tensordot(x1, x2, axes=[[x1.ndim - 1], [x2.ndim - 2]])


def matvec(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.matmul(x1, x2[..., None])[..., 0]


def vecmat(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.matmul(x1[..., None, :], x2)[..., 0, :]


def tensordot(x1, x2, axes=2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype in dtypes.INT_TYPES or dtype == "bool":
        out = np.tensordot(_to_np(x1), _to_np(x2), axes=axes)
        return _cast(convert_to_tensor(out), dtype)
    # `mx.tensordot` wants `axes` as an int or list[list[int]] (no tuples).
    # numpy also accepts the (axes_a, axes_b) shorthand where each is an int.
    if not isinstance(axes, int):
        a_axes, b_axes = axes[0], axes[1]
        a_axes = [a_axes] if isinstance(a_axes, int) else list(a_axes)
        b_axes = [b_axes] if isinstance(b_axes, int) else list(b_axes)
        axes = [a_axes, b_axes]
    return mx.tensordot(_cast(x1, dtype), _cast(x2, dtype), axes=axes)


def inner(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype in dtypes.INT_TYPES or dtype == "bool":
        out = np.inner(_to_np(x1), _to_np(x2))
        return _cast(convert_to_tensor(out), dtype)
    return mx.inner(_cast(x1, dtype), _cast(x2, dtype))


def outer(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype in dtypes.INT_TYPES or dtype == "bool":
        out = np.outer(_to_np(x1), _to_np(x2))
        return _cast(convert_to_tensor(out), dtype)
    return mx.outer(_cast(x1, dtype), _cast(x2, dtype))


def vdot(x1, x2):
    # MLX has only complex64; conjugation is a no-op for real inputs.
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype in dtypes.INT_TYPES or dtype == "bool":
        return _cast(convert_to_tensor(np.vdot(_to_np(x1), _to_np(x2))), dtype)
    a = _cast(mx.flatten(x1), dtype)
    b = _cast(mx.flatten(x2), dtype)
    if str(a.dtype) == "mlx.core.complex64":
        a = mx.conjugate(a)
    return mx.sum(mx.multiply(a, b))


def kron(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    return _cast(mx.kron(_cast(x1, dtype), _cast(x2, dtype)), dtype)


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    # numpy fallback for the general (axisa/axisb/axisc) form.
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    result = np.cross(
        _to_np(x1),
        _to_np(x2),
        axisa=axisa,
        axisb=axisb,
        axisc=axisc,
        axis=axis,
    )
    return _cast(convert_to_tensor(result), dtype)


def correlate(x1, x2, mode="valid"):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    # JAX promotes integer/bool inputs to float32 for correlate.
    if dtype in dtypes.INT_TYPES or dtype == "bool":
        dtype = "float32"
    result = np.correlate(_to_np(x1), _to_np(x2), mode)
    return _cast(convert_to_tensor(result), dtype)


def convolve(x1, x2, mode="full"):
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = _to_np(convert_to_tensor(x1, dtype))
    x2 = _to_np(convert_to_tensor(x2, dtype))
    return convert_to_tensor(np.convolve(x1, x2, mode))


# ---------------------------------------------------------------------------
# Indexing / gather
# ---------------------------------------------------------------------------


def take(x, indices, axis=None):
    axis = standardize_axis_for_numpy(axis)
    return mx.take(convert_to_tensor(x), convert_to_tensor(indices), axis=axis)


def take_along_axis(x, indices, axis=None):
    axis = standardize_axis_for_numpy(axis)
    return mx.take_along_axis(
        convert_to_tensor(x), convert_to_tensor(indices), axis=axis
    )


def put_along_axis(x, indices, values, axis):
    axis = standardize_axis_for_numpy(axis)
    return mx.put_along_axis(
        convert_to_tensor(x),
        convert_to_tensor(indices),
        convert_to_tensor(values),
        axis,
    )


# ---------------------------------------------------------------------------
# Sorting / partitioning
# ---------------------------------------------------------------------------


def sort(x, axis=-1):
    axis = standardize_axis_for_numpy(axis)
    return mx.sort(convert_to_tensor(x), axis=axis)


def argsort(x, axis=-1):
    x = convert_to_tensor(x)
    if x.ndim == 0:
        # numpy returns shape (1,) for a 0-D input.
        return _cast(mx.argsort(mx.reshape(x, (1,)), axis=0), "int32")
    axis = standardize_axis_for_numpy(axis)
    return _cast(mx.argsort(x, axis=axis), "int32")


def argmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = standardize_axis_for_numpy(axis)
    dtype = standardize_dtype(x.dtype)
    if "float" not in dtype or x.ndim == 0:
        return _cast(mx.argmax(x, axis=axis, keepdims=keepdims), "int32")
    dtype = dtypes.result_type(dtype, "float32")
    xc = _cast(x, dtype)
    # Flip negative zero to break ties the same way numpy does.
    is_neg_zero = mx.equal(xc, _cast(mx.zeros_like(xc), dtype)) & mx.less(
        xc, _cast(mx.zeros_like(xc), dtype)
    )
    xc = mx.where(
        is_neg_zero,
        _cast(mx.full(xc.shape, -np.finfo(np.float32).tiny), dtype),
        xc,
    )
    return _cast(mx.argmax(xc, axis=axis, keepdims=keepdims), "int32")


def argmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = standardize_axis_for_numpy(axis)
    dtype = standardize_dtype(x.dtype)
    if "float" not in dtype or x.ndim == 0:
        return _cast(mx.argmin(x, axis=axis, keepdims=keepdims), "int32")
    dtype = dtypes.result_type(dtype, "float32")
    xc = _cast(x, dtype)
    is_neg_zero = mx.equal(xc, _cast(mx.zeros_like(xc), dtype)) & mx.less(
        xc, _cast(mx.zeros_like(xc), dtype)
    )
    xc = mx.where(
        is_neg_zero,
        _cast(mx.full(xc.shape, -np.finfo(np.float32).tiny), dtype),
        xc,
    )
    return _cast(mx.argmin(xc, axis=axis, keepdims=keepdims), "int32")


def top_k(x, k, sorted=True):
    # `mx.topk` returns flat indices in some shapes; emulate the numpy
    # semantics exactly with argsort/take_along_axis.
    x = convert_to_tensor(x)
    if sorted:
        sorted_indices = mx.argsort(x, axis=-1)[..., ::-1]
        sorted_values = mx.take_along_axis(x, sorted_indices, axis=-1)
        top_k_values = sorted_values[..., :k]
        top_k_indices = sorted_indices[..., :k]
    else:
        top_k_indices = mx.argpartition(x, -k, axis=-1)[..., -k:]
        top_k_values = mx.take_along_axis(x, top_k_indices, axis=-1)
    return top_k_values, _cast(top_k_indices, "int32")


def partition(x, kth, axis=-1):
    axis = standardize_axis_for_numpy(axis)
    return mx.partition(convert_to_tensor(x), kth, axis=axis)


def argpartition(x, kth, axis=-1):
    axis = standardize_axis_for_numpy(axis)
    return _cast(mx.argpartition(convert_to_tensor(x), kth, axis=axis), "int32")


# ---------------------------------------------------------------------------
# Percentile / quantile / median (sort + interpolation)
# ---------------------------------------------------------------------------


def median(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(x.dtype, float)
    return _quantile_core(
        _cast(x, dtype), 0.5, axis=axis, method="linear", keepdims=keepdims
    )


def _quantile_core(x, q, axis=None, method="linear", keepdims=False):
    """Sort-based quantile matching numpy semantics.

    numpy places the ``q`` axis at the front of the output and supports
    reducing over a tuple of axes simultaneously. We move all reduced axes to
    the tail, fuse them into a single axis, sort along it, then gather by the
    interpolated rank and reshape back.
    """
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(x.dtype, float)
    x = _cast(x, dtype)
    orig_ndim = x.ndim

    if axis is None:
        axes = tuple(range(orig_ndim))
    elif isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(axis)
    axes = tuple(a % orig_ndim for a in axes)

    # Move kept axes to the front, reduced axes to the tail, then fuse.
    kept_axes = [i for i in range(orig_ndim) if i not in axes]
    perm = kept_axes + list(axes)
    x = mx.transpose(x, perm) if perm != list(range(orig_ndim)) else x
    kept_shape = list(x.shape[: len(kept_axes)])
    red_shape = list(x.shape[len(kept_axes) :])
    n = builtins.max(1, math.prod(red_shape))
    m = builtins.max(1, math.prod(kept_shape))
    x = mx.reshape(x, (m, n))
    x_sorted = mx.sort(x, axis=1)  # (m, n)

    q = convert_to_tensor(q, dtype="float32")
    q_shape = list(q.shape)
    q_flat = mx.reshape(q, (-1,))  # (p,)
    p = builtins.max(1, math.prod(q_shape))
    rank = mx.multiply(q_flat, mx.array(float(n - 1), dtype=mx.float32))
    lo_f = mx.floor(rank)
    frac = rank - lo_f
    lo_idx = _cast(lo_f, mx.int32)
    hi_idx = _cast(mx.ceil(rank), mx.int32)

    lo_b = mx.broadcast_to(mx.reshape(lo_idx, (1, p)), (m, p))
    hi_b = mx.broadcast_to(mx.reshape(hi_idx, (1, p)), (m, p))
    frac_b = mx.broadcast_to(mx.reshape(frac, (1, p)), (m, p))
    v_lo = mx.take_along_axis(x_sorted, lo_b, axis=1)  # (m, p)
    v_hi = mx.take_along_axis(x_sorted, hi_b, axis=1)  # (m, p)

    if method == "lower":
        result = v_lo
    elif method == "higher":
        result = v_hi
    elif method == "nearest":
        result = mx.where(mx.less_equal(frac_b, 0.5), v_lo, v_hi)
    elif method == "midpoint":
        result = (v_lo + v_hi) / 2
    elif method == "linear":
        result = v_lo + frac_b * (v_hi - v_lo)
    else:
        raise ValueError(f"Unsupported quantile method: {method}")

    # (m, p) -> (p, m) -> q-shape + kept-shape, with the q axis at the front.
    result = mx.transpose(result)  # (p, m)
    out_shape = q_shape + kept_shape
    result = mx.reshape(result, out_shape)  # empty -> scalar ()

    if keepdims:
        full_shape = []
        ki = 0
        for i in range(orig_ndim):
            if i in axes:
                full_shape.append(1)
            else:
                full_shape.append(kept_shape[ki])
                ki += 1
        result = mx.reshape(result, q_shape + full_shape)
    return _cast(result, dtype)


def percentile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype == "bool":
        x = _cast(x, config.floatx())
    q = convert_to_tensor(q) * 0.01
    return _quantile_core(x, q, axis=axis, method=method, keepdims=keepdims)


def quantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype == "bool":
        x = _cast(x, config.floatx())
    q = convert_to_tensor(q)
    return _quantile_core(x, q, axis=axis, method=method, keepdims=keepdims)


def nanmedian(x, axis=None, keepdims=False):
    # numpy fallback (nan handling is intricate; rarely on the autograd path).
    dtype = dtypes.result_type(
        standardize_dtype(convert_to_tensor(x).dtype), float
    )
    res = np.nanmedian(
        _to_np(convert_to_tensor(x)), axis=axis, keepdims=keepdims
    )
    return _cast(convert_to_tensor(res), dtype)


def nanpercentile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype == "bool":
        x = _cast(x, config.floatx())
    dtype = dtypes.result_type(x.dtype, float)
    res = np.nanpercentile(
        _to_np(x),
        _to_np(convert_to_tensor(q)),
        axis=axis,
        method=method,
        keepdims=keepdims,
    )
    return _cast(convert_to_tensor(res), dtype)


def nanquantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype == "bool":
        x = _cast(x, config.floatx())
    dtype = dtypes.result_type(x.dtype, float)
    res = np.nanquantile(
        _to_np(x),
        _to_np(convert_to_tensor(q)),
        axis=axis,
        method=method,
        keepdims=keepdims,
    )
    return _cast(convert_to_tensor(res), dtype)


# ---------------------------------------------------------------------------
# Mod / floor / shifts
# ---------------------------------------------------------------------------


def mod(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype == "bool":
        dtype = "int32"
    # `mx.remainder` matches `np.mod` (Python-style, divisor sign) exactly.
    return _cast(mx.remainder(_cast(x1, dtype), _cast(x2, dtype)), dtype)


def fmod(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype == "bool":
        dtype = "int32"
    # C-style: result sign follows the dividend. `mx.remainder` gives the
    # Python-style mod; subtract the divisor when the signs differ.
    a = _cast(x1, dtype)
    b = _cast(x2, dtype)
    r = mx.remainder(a, b)
    flip = mx.not_equal(r, mx.zeros_like(r)) & mx.not_equal(
        mx.sign(a), mx.sign(b)
    )
    return _cast(mx.where(flip, r - b, r), dtype)


def remainder(x1, x2):
    return mod(x1, x2)


def floor_divide(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)), getattr(x2, "dtype", type(x2))
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.floor_divide(x1, x2)


def divmod(x1, x2):
    return floor_divide(x1, x2), mod(x1, x2)


def bitwise_and(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    return mx.bitwise_and(_cast(x, dtype), _cast(y, dtype))


def bitwise_or(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    return mx.bitwise_or(_cast(x, dtype), _cast(y, dtype))


def bitwise_xor(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    return mx.bitwise_xor(_cast(x, dtype), _cast(y, dtype))


def bitwise_invert(x):
    return mx.bitwise_invert(convert_to_tensor(x))


def bitwise_not(x):
    return bitwise_invert(x)


def bitwise_left_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, int):
        y = convert_to_tensor(y)
        dtype = dtypes.result_type(x.dtype, y.dtype)
        x = _cast(x, dtype)
        y = _cast(y, dtype)
    return mx.left_shift(x, y)


def left_shift(x, y):
    return bitwise_left_shift(x, y)


def bitwise_right_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, int):
        y = convert_to_tensor(y)
        dtype = dtypes.result_type(x.dtype, y.dtype)
        x = _cast(x, dtype)
        y = _cast(y, dtype)
    return mx.right_shift(x, y)


def right_shift(x, y):
    return bitwise_right_shift(x, y)


def heaviside(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype in ["int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        dtype = config.floatx()
    elif dtype in ["int64"]:
        dtype = "float32"
    x1 = _cast(x1, dtype)
    x2 = _cast(x2, dtype)
    return _cast(
        mx.where(
            mx.greater(x1, mx.zeros_like(x1)),
            mx.ones_like(x1),
            mx.where(
                mx.equal(x1, mx.zeros_like(x1)),
                x2,
                mx.zeros_like(x1),
            ),
        ),
        dtype,
    )


def hypot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype in ["int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        dtype = config.floatx()
    elif dtype in ["int64"]:
        dtype = "float32"
    return _cast(
        mx.sqrt(mx.square(_cast(x1, dtype)) + mx.square(_cast(x2, dtype))),
        dtype,
    )


def sinc(x):
    x = convert_to_tensor(x)
    dtype = _angle_dtype(x)
    x = _cast(x, dtype)
    pi_x = mx.multiply(x, mx.array(np.pi))
    return _cast(
        mx.where(
            mx.equal(x, mx.zeros_like(x)),
            mx.ones_like(x),
            mx.divide(mx.sin(pi_x), pi_x),
        ),
        dtype,
    )


def ldexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    if standardize_dtype(x2.dtype) not in dtypes.INT_TYPES:
        raise TypeError(
            "ldexp exponent must be an integer type. "
            f"Received: x2 dtype={x2.dtype}"
        )
    return _cast(
        mx.multiply(_cast(x1, dtype), mx.power(2, _cast(x2, dtype))),
        dtype,
    )


def nextafter(x1, x2):
    # numpy fallback (mlx has no nextafter). JAX promotes integer inputs to
    # float32, and preserves float dtypes (incl. bfloat16).
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if result_dtype in dtypes.INT_TYPES or result_dtype == "bool":
        result_dtype = "float32"
    # numpy cannot hold bfloat16; compute in float32 then cast back.
    a = _to_np(_cast(x1, "float32"))
    b = _to_np(_cast(x2, "float32"))
    out = np.nextafter(a, b)
    return _cast(convert_to_tensor(out), result_dtype)


def gcd(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    return _cast(convert_to_tensor(np.gcd(_to_np(x1), _to_np(x2))), dtype)


def lcm(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    return _cast(convert_to_tensor(np.lcm(_to_np(x1), _to_np(x2))), dtype)


def ptp(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    return mx.subtract(
        mx.max(x, axis=axis, keepdims=keepdims),
        mx.min(x, axis=axis, keepdims=keepdims),
    )


def diff(a, n=1, axis=-1):
    axis = standardize_axis_for_numpy(axis)
    a = convert_to_tensor(a)
    for _ in range(n):
        a = mx.subtract(
            mx.take(a, mx.arange(1, a.shape[axis]), axis=axis),
            mx.take(a, mx.arange(0, a.shape[axis] - 1), axis=axis),
        )
    return a


def ndim(x):
    if isinstance(x, (int, float)):
        return 0
    return convert_to_tensor(x).ndim


def size(x):
    if isinstance(x, (int, float)):
        return 1
    x = convert_to_tensor(x)
    s = 1
    for d in x.shape:
        s *= d
    return s


# ---------------------------------------------------------------------------
# Window functions (closed-form over arange)
# ---------------------------------------------------------------------------


def bartlett(x):
    x = int(_to_np(convert_to_tensor(x)))
    n = mx.arange(x, dtype=_mlx_dtype(config.floatx()))
    if x == 1:
        return mx.ones_like(n)
    return _cast(1 - mx.abs(2 * n - (x - 1)) / (x - 1), config.floatx())


def hamming(x):
    x = int(_to_np(convert_to_tensor(x)))
    n = mx.arange(x, dtype=_mlx_dtype(config.floatx()))
    if x == 1:
        return mx.ones_like(n)
    return _cast(0.54 - 0.46 * mx.cos(2 * np.pi * n / (x - 1)), config.floatx())


def hanning(x):
    x = int(_to_np(convert_to_tensor(x)))
    n = mx.arange(x, dtype=_mlx_dtype(config.floatx()))
    if x == 1:
        return mx.ones_like(n)
    return _cast(0.5 - 0.5 * mx.cos(2 * np.pi * n / (x - 1)), config.floatx())


def blackman(x):
    x = int(_to_np(convert_to_tensor(x)))
    n = mx.arange(x, dtype=_mlx_dtype(config.floatx()))
    if x == 1:
        return mx.ones_like(n)
    return _cast(
        0.42
        - 0.5 * mx.cos(2 * np.pi * n / (x - 1))
        + 0.08 * mx.cos(4 * np.pi * n / (x - 1)),
        config.floatx(),
    )


def kaiser(x, beta):
    x = int(_to_np(convert_to_tensor(x)))
    beta = float(_to_np(convert_to_tensor(beta)))
    # numpy fallback for the Bessel I0 dependence.
    return _cast(convert_to_tensor(np.kaiser(x, beta)), config.floatx())


def i0(x):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(x.dtype, float)
    return _cast(convert_to_tensor(np.i0(_to_np(_cast(x, dtype)))), dtype)


# ---------------------------------------------------------------------------
# Misc numpy ops (numpy fallback for the long tail)
# ---------------------------------------------------------------------------


def digitize(x, bins):
    result = np.digitize(
        _to_np(convert_to_tensor(x)), _to_np(convert_to_tensor(bins))
    )
    return convert_to_tensor(result).astype(mx.int32)


def bincount(x, weights=None, minlength=0, sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with mlx backend")
    x = convert_to_tensor(x)
    if weights is not None:
        weights = convert_to_tensor(weights)
        dtype = dtypes.result_type(x.dtype, weights.dtype)
    else:
        dtype = "int32"
    if len(x.shape) == 2:
        if weights is None:
            bincounts = [
                np.bincount(arr, minlength=minlength) for arr in _to_np(x)
            ]
        else:
            wnp = _to_np(weights)
            bincounts = [
                np.bincount(a, weights=b, minlength=minlength)
                for a, b in zip(_to_np(x), wnp)
            ]
        return _cast(convert_to_tensor(np.stack(bincounts)), dtype)
    return _cast(
        convert_to_tensor(
            np.bincount(
                _to_np(x),
                _to_np(weights) if weights is not None else None,
                minlength,
            )
        ),
        dtype,
    )


def histogram(x, bins=10, range=None):
    counts, edges = np.histogram(
        _to_np(convert_to_tensor(x)), bins=bins, range=range
    )
    return convert_to_tensor(counts), convert_to_tensor(edges)


def searchsorted(sorted_sequence, values, side="left"):
    if ndim(sorted_sequence) != 1:
        raise ValueError(
            "`searchsorted` only supports 1-D sorted sequences. "
            "You can use `keras.ops.vectorized_map` "
            "to extend it to N-D sequences. Received: "
            f"sorted_sequence.shape={sorted_sequence.shape}"
        )
    out_type = (
        "int32"
        if sorted_sequence.shape[0] <= np.iinfo(np.int32).max
        else "int64"
    )
    result = np.searchsorted(
        _to_np(convert_to_tensor(sorted_sequence)),
        _to_np(convert_to_tensor(values)),
        side=side,
    )
    return convert_to_tensor(result).astype(_mlx_dtype(out_type))


def unravel_index(indices, shape):
    dtype = dtypes.result_type(convert_to_tensor(indices).dtype)
    result = np.unravel_index(_to_np(convert_to_tensor(indices)), shape)
    return tuple(_cast(convert_to_tensor(idx), dtype) for idx in result)


def corrcoef(x):
    x = convert_to_tensor(x)
    d = standardize_dtype(x.dtype)
    if d in ["int64", "float64"]:
        dtype = "float32"
    elif d in ["bfloat16", "float16"]:
        dtype = d
    else:
        dtype = config.floatx()
    return _cast(convert_to_tensor(np.corrcoef(_to_np(_cast(x, dtype)))), dtype)


def trapezoid(y, x=None, dx=1.0, axis=-1):
    y = convert_to_tensor(y)
    result_dtype = dtypes.result_type(y.dtype, float)
    if x is not None:
        x = _to_np(convert_to_tensor(x))
    dxn = _to_np(convert_to_tensor(dx))
    res = np.trapezoid(_to_np(_cast(y, result_dtype)), x, dx=dxn, axis=axis)
    return _cast(convert_to_tensor(res), result_dtype)


def vander(x, N=None, increasing=False):
    x = convert_to_tensor(x)
    result_dtype = dtypes.result_type(x.dtype)
    compute_dtype = dtypes.result_type(x.dtype, config.floatx())
    res = np.vander(_to_np(_cast(x, compute_dtype)), N=N, increasing=increasing)
    return _cast(convert_to_tensor(res), result_dtype)


def slogdet(x):
    sign, logabs = np.linalg.slogdet(_to_np(convert_to_tensor(x)))
    return convert_to_tensor(sign), convert_to_tensor(logabs)


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    dtype = dtype or config.floatx()
    res = np.geomspace(start, stop, num=num, endpoint=endpoint)
    res = convert_to_tensor(res).astype(_mlx_dtype(dtype))
    if axis != 0:
        res = mx.moveaxis(res, 0, axis)
    return res


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    if dtype is None:
        dtype = dtypes.result_type(
            getattr(start, "dtype", type(start)),
            getattr(stop, "dtype", type(stop)),
            float,
        )
    res = np.logspace(start, stop, num=num, endpoint=endpoint, base=base)
    res = convert_to_tensor(res).astype(_mlx_dtype(dtype))
    if axis != 0:
        res = mx.moveaxis(res, 0, axis)
    return res


def vectorize(pyfunc, *, excluded=None, signature=None):
    # numpy's vectorize is a python loop wrapper; reuse it via numpy fallback.
    return np.vectorize(pyfunc, excluded=excluded, signature=signature)


def unique(
    x,
    sorted=True,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    axis=None,
    size=None,
    fill_value=None,
):
    x = convert_to_tensor(x)
    output = np.unique(
        _to_np(x),
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
        equal_nan=False,
    )

    if not (return_index or return_inverse or return_counts):
        output = [output]
    else:
        output = list(output)

    values = convert_to_tensor(output[0])

    if size is not None:
        if axis is None:
            dim = 0
        else:
            dim = canonicalize_axis(axis, x.ndim)
        values_count = values.shape[dim]

        if values_count > size:
            indices = [slice(None)] * values.ndim
            indices[dim] = slice(0, size)
            values = values[tuple(indices)]
            if return_counts:
                output[-1] = output[-1][indices[dim]]
            if return_index:
                output[1] = output[1][indices[dim]]
        elif values_count < size:
            pad_width = [(0, 0)] * values.ndim
            pad_width[dim] = (0, size - values_count)
            fill = 0 if fill_value is None else fill_value
            values = mx.pad(values, pad_width, constant_values=fill)
            if return_counts:
                output[-1] = np.pad(
                    output[-1], pad_width[dim], constant_values=0
                )
            if return_index:
                output[1] = np.pad(output[1], pad_width[dim], constant_values=1)

    output[0] = values
    if len(output) == 1:
        return output[0]
    return tuple(convert_to_tensor(o) for o in output)


def pad(x, pad_width, mode="constant", constant_values=None):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    kwargs = {}
    if constant_values is not None:
        if mode != "constant":
            raise ValueError(
                "Argument `constant_values` can only be "
                "provided when `mode == 'constant'`. "
                f"Received: mode={mode}"
            )
        kwargs["constant_values"] = constant_values
    if mode in ("constant", "edge"):
        out = mx.pad(x, pad_width, mode=mode, **kwargs)
    else:
        # Modes mlx lacks (reflect, symmetric, wrap, ...) -> numpy fallback.
        out = convert_to_tensor(
            np.pad(_to_np(x), pad_width, mode=mode, **kwargs)
        )
    # mx.pad can upcast bfloat16; restore the input dtype.
    return _cast(out, dtype)


def scalar_mul(x, val):
    return mx.multiply(convert_to_tensor(x), convert_to_tensor(val))
