import builtins

import numpy as np
import paddle

from keras.src.backend.common import standardize_dtype
from keras.src.backend.paddle.core import _weak_tensors
from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.backend.paddle.core import shape
from keras.src.backend.paddle.core import to_paddle_dtype

_CPU_UNSUPPORTED_DTYPES = {paddle.float16, paddle.bfloat16}
_CPU_UNSUPPORTED_INT = {"int8", "int16", "uint8", "bool"}
_FLOAT_TYPES = {
    "float16",
    "float32",
    "float64",
    "bfloat16",
    "float8_e4m3fn",
    "float8_e5m2",
}


def _maybe_upcast(x):
    """Cast float16/bfloat16 to float32 on CPU."""
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        return x.cast("float32"), True
    return x, False


def _maybe_downcast(result, needs_downcast, original_dtype):
    """Cast result back to original dtype if it was upcast."""
    if needs_downcast:
        return result.cast(original_dtype)
    return result


def _get_promoted_dtype(x1, x2):
    """Compute the promoted dtype for two tensors without casting.

    Returns the target dtype string. Used by _promote_dtypes_list
    where we need the logical target dtype without CPU upcasting.
    """
    dt1 = standardize_dtype(x1.dtype)
    dt2 = standardize_dtype(x2.dtype)
    if dt1 == dt2:
        return dt1
    float_types = {"float16", "float32", "float64", "bfloat16"}
    complex_types = {"complex64", "complex128"}
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    is_f1 = dt1 in float_types
    is_f2 = dt2 in float_types
    is_c1 = dt1 in complex_types
    is_c2 = dt2 in complex_types
    is_i1 = dt1 in int_types
    is_i2 = dt2 in int_types
    w1 = x1 in _weak_tensors
    w2 = x2 in _weak_tensors
    if w1 and not w2 and ((is_f1 and is_f2) or (is_i1 and is_i2)):
        return "int32" if dt2 == "bool" and is_i1 else dt2
    elif w2 and not w1 and ((is_f1 and is_f2) or (is_i1 and is_i2)):
        return "int32" if dt1 == "bool" and is_i2 else dt1
    elif is_f1 and not is_f2:
        return dt1
    elif is_f2 and not is_f1:
        return dt2
    elif (is_c1 or is_c2) and not (is_c1 and is_c2):
        return dt1 if is_c1 else dt2
    else:
        try:
            common = np.result_type(
                np.zeros(1, dtype=dt1), np.zeros(1, dtype=dt2)
            )
            return standardize_dtype(common)
        except (TypeError, np.exceptions.DTypePromotionError):
            return "float32"


def _promote_dtypes(x1, x2):
    """Cast two tensors to a common dtype for cross-type operations.

    Matches Keras/JAX behavior: when mixing float and int/bool types,
    promote to the float type (not widen to float64).
    Python scalars are treated as "weak" types (JAX semantics):
    int8 + python_int → int8, int8 + python_float → float32.
    """
    dt1 = standardize_dtype(x1.dtype)
    dt2 = standardize_dtype(x2.dtype)
    if dt1 == dt2:
        # CPU doesn't support float16/bfloat16 for most ops
        if dt1 in ("float16", "bfloat16"):
            x1 = x1.cast("float32")
            x2 = x2.cast("float32")
        return x1, x2
    common_dtype = _get_promoted_dtype(x1, x2)
    if dt1 != common_dtype:
        x1 = paddle.cast(x1, to_paddle_dtype(common_dtype))
    if dt2 != common_dtype:
        x2 = paddle.cast(x2, to_paddle_dtype(common_dtype))
    # CPU doesn't support float16/bfloat16 for most ops
    if common_dtype in ("float16", "bfloat16"):
        x1 = x1.cast("float32")
        x2 = x2.cast("float32")
    return x1, x2


def _binary_op_with_dtype(op, x1, x2):
    """Run a binary op with _promote_dtypes and cast result back if needed.

    Handles float16/bfloat16 CPU upcasting.
    """
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dt1 = standardize_dtype(x1.dtype)
    dt2 = standardize_dtype(x2.dtype)
    w1 = x1 in _weak_tensors
    w2 = x2 in _weak_tensors
    low_precision = {"float16", "bfloat16"}
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    # Determine if result should be low precision float
    target = None
    if w1 and not w2:
        if dt2 in low_precision:
            target = dt2
    elif w2 and not w1:
        if dt1 in low_precision:
            target = dt1
    elif not w1 and not w2:
        if dt1 in low_precision and dt2 in int_types:
            target = dt1
        elif dt2 in low_precision and dt1 in int_types:
            target = dt2
        elif dt1 in low_precision and dt2 in low_precision:
            target = dt1 if dt1 == dt2 else None
    x1, x2 = _promote_dtypes(x1, x2)
    result = op(x1, x2)
    if target in low_precision:
        result_dtype = standardize_dtype(result.dtype)
        if result_dtype != target:
            result = result.cast(to_paddle_dtype(target))
    return result


def _unary_op(op, x):
    """Run a unary op with CPU upcasting for unsupported dtypes."""
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
        needs_cast = True
    result = op(x)
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def _unary_math_op(op, x):
    """Run a unary math op that returns float for int inputs."""
    x = convert_to_tensor(x)
    orig_dtype = standardize_dtype(x.dtype)
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
    }
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif orig_dtype in int_types:
        x = x.cast("float32")
    result = op(x)
    # Cast back float16/bfloat16, but not int types (they stay float)
    if needs_cast and orig_dtype not in int_types:
        result = result.cast(to_paddle_dtype(orig_dtype))
    return result


_CPU_UNSUPPORTED_INT = {"int8", "int16", "uint8", "bool"}


def _cpu_binary_target(x1, x2):
    """Compute target dtype from original types before CPU casting.

    Returns (target_dtype_string, x1_cast, x2_cast) where x1_cast and x2_cast
    are the CPU-compatible versions of x1 and x2.
    """
    dt1 = standardize_dtype(x1.dtype)
    dt2 = standardize_dtype(x2.dtype)
    w1 = x1 in _weak_tensors
    w2 = x2 in _weak_tensors
    float_types = {"float16", "float32", "float64", "bfloat16"}
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    is_f1 = dt1 in float_types
    is_f2 = dt2 in float_types
    is_i1 = dt1 in int_types
    is_i2 = dt2 in int_types
    # Determine target using same logic as _promote_dtypes
    if w1 and not w2 and ((is_f1 and is_f2) or (is_i1 and is_i2)):
        target = "int32" if dt2 == "bool" and is_i1 else dt2
    elif w2 and not w1 and ((is_f1 and is_f2) or (is_i1 and is_i2)):
        target = "int32" if dt1 == "bool" and is_i2 else dt1
    elif is_f1 and not is_f2:
        target = dt1
    elif is_f2 and not is_f1:
        target = dt2
    elif dt1 == dt2:
        target = dt1
    else:
        try:
            common = np.result_type(
                np.zeros(1, dtype=dt1), np.zeros(1, dtype=dt2)
            )
            target = standardize_dtype(common)
        except (TypeError, np.exceptions.DTypePromotionError):
            target = "float32"
    # CPU cast
    if x1.dtype in _CPU_UNSUPPORTED_DTYPES:
        x1 = x1.cast("float32")
    if x2.dtype in _CPU_UNSUPPORTED_DTYPES:
        x2 = x2.cast("float32")
    if standardize_dtype(x1.dtype) in _CPU_UNSUPPORTED_INT:
        x1 = x1.cast("int32")
    if standardize_dtype(x2.dtype) in _CPU_UNSUPPORTED_INT:
        x2 = x2.cast("int32")
    x1, x2 = _promote_dtypes(x1, x2)
    return target, x1, x2


def _binary_op_with_int(op, x1, x2, bool_to_int32=False):
    """Run a binary op with full CPU dtype support (float16 + int types).

    Handles CPU-unsupported int types (int8/int16/uint8/bool) by upcasting
    to int32, and ensures mixed float+int operations work by promoting
    to the float type (matching JAX weak-type semantics).

    Args:
        bool_to_int32: If True, promote bool result to int32 (for power).
    """
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dt1 = standardize_dtype(x1.dtype)
    dt2 = standardize_dtype(x2.dtype)
    w1 = x1 in _weak_tensors
    w2 = x2 in _weak_tensors
    float_types = {"float16", "float32", "float64", "bfloat16"}
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    is_f1 = dt1 in float_types
    is_f2 = dt2 in float_types
    is_i1 = dt1 in int_types
    is_i2 = dt2 in int_types
    # Determine target dtype using same logic as _promote_dtypes
    if w1 and not w2 and ((is_f1 and is_f2) or (is_i1 and is_i2)):
        target = "int32" if dt2 == "bool" and is_i1 else dt2
    elif w2 and not w1 and ((is_f1 and is_f2) or (is_i1 and is_i2)):
        target = "int32" if dt1 == "bool" and is_i2 else dt1
    elif is_f1 and not is_f2:
        target = dt1
    elif is_f2 and not is_f1:
        target = dt2
    elif dt1 == dt2:
        target = dt1
    else:
        try:
            common = np.result_type(
                np.zeros(1, dtype=dt1), np.zeros(1, dtype=dt2)
            )
            target = standardize_dtype(common)
        except (TypeError, np.exceptions.DTypePromotionError):
            target = "float32"
    # JAX promotes bool to int32 for power
    if bool_to_int32 and target == "bool":
        target = "int32"
    # Cast unsupported types to CPU-supported equivalents
    if dt1 in _CPU_UNSUPPORTED_INT or dt1 in ("float16", "bfloat16"):
        cast_to = "float32" if dt1 in ("float16", "bfloat16") else "int32"
        x1 = x1.cast(cast_to)
    if dt2 in _CPU_UNSUPPORTED_INT or dt2 in ("float16", "bfloat16"):
        cast_to = "float32" if dt2 in ("float16", "bfloat16") else "int32"
        x2 = x2.cast(cast_to)
    # Ensure both tensors have the same dtype (paddle ops don't auto-promote)
    x1, x2 = _promote_dtypes(x1, x2)
    result = op(x1, x2)
    result_dtype = standardize_dtype(result.dtype)
    if target != result_dtype:
        result = result.cast(to_paddle_dtype(target))
    return result


def add(x1, x2):
    return _binary_op_with_dtype(paddle.add, x1, x2)


def subtract(x1, x2):
    return _binary_op_with_int(paddle.subtract, x1, x2)


def multiply(x1, x2):
    return _binary_op_with_int(paddle.multiply, x1, x2)


def divide(x1, x2):
    return _binary_op_with_dtype(paddle.divide, x1, x2)


def true_divide(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    target_dtype = _get_promoted_dtype(x1, x2)
    # true_divide always returns float for int inputs
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
    }
    if target_dtype in int_types:
        target_dtype = "float32"
    # Cast to float for computation (paddle.divide needs float inputs)
    if x1.dtype in _CPU_UNSUPPORTED_DTYPES:
        x1 = x1.cast("float32")
    if x2.dtype in _CPU_UNSUPPORTED_DTYPES:
        x2 = x2.cast("float32")
    x1_dt = standardize_dtype(x1.dtype)
    x2_dt = standardize_dtype(x2.dtype)
    if x1_dt in int_types:
        x1 = x1.cast("float32")
    if x2_dt in int_types:
        x2 = x2.cast("float32")
    result = paddle.divide(x1, x2)
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target_dtype:
        result = result.cast(to_paddle_dtype(target_dtype))
    return result


def floor_divide(x1, x2):
    return _binary_op_with_dtype(paddle.floor_divide, x1, x2)


def mod(x1, x2):
    return _binary_op_with_int(paddle.remainder, x1, x2)


def negative(x):
    return _unary_op(paddle.neg, x)


def abs(x):
    return _unary_op(paddle.abs, x)


def absolute(x):
    return abs(x)


def sign(x):
    return _unary_op(paddle.sign, x)


def log(x):
    return _unary_math_op(paddle.log, x)


def log2(x):
    return _unary_math_op(paddle.log2, x)


def log10(x):
    return _unary_math_op(paddle.log10, x)


def log1p(x):
    return _unary_math_op(paddle.log1p, x)


def exp(x):
    return _unary_math_op(paddle.exp, x)


def expm1(x):
    return _unary_math_op(paddle.expm1, x)


def sqrt(x):
    return _unary_math_op(paddle.sqrt, x)


def square(x):
    x = convert_to_tensor(x)
    orig_dtype = standardize_dtype(x.dtype)
    # JAX promotes bool to int32 for square, don't cast back
    if orig_dtype == "bool":
        x = x.cast("int32")
        return paddle.square(x)
    return _unary_op(paddle.square, x)


def pow(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return _binary_op_with_int(paddle.pow, x1, x2)


def power(x1, x2):
    return _binary_op_with_int(paddle.pow, x1, x2, bool_to_int32=True)


def maximum(x1, x2):
    return _binary_op_with_int(paddle.maximum, x1, x2)


def minimum(x1, x2):
    return _binary_op_with_int(paddle.minimum, x1, x2)


def round(x, decimals=0):
    if decimals == 0:
        return _unary_op(lambda t: paddle.round(t), x)
    # For non-zero decimals, scale, round, scale back
    factor = 10.0**decimals
    return _unary_op(lambda t: paddle.round(t * factor) / factor, x)


def clip(x, x_min, x_max):
    x = convert_to_tensor(x)
    orig_dtype = standardize_dtype(x.dtype)
    # JAX promotes bool to int32 for clip
    if orig_dtype == "bool":
        x = x.cast("int32")
    return _unary_op(lambda t: paddle.clip(t, x_min, x_max), x)


def clip_by_value(x, clip_value_min, clip_value_max):
    return clip(x, clip_value_min, clip_value_max)


def floor(x):
    return _unary_math_op(paddle.floor, x)


def ceil(x):
    return _unary_math_op(paddle.ceil, x)


def dot(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    # paddle.dot only supports 1D tensors
    # Scalar case: use element-wise multiply
    if x.ndim == 0 or y.ndim == 0:
        return multiply(x, y)
    if x.ndim <= 1 and y.ndim <= 1:
        return _binary_op_with_int(paddle.dot, x, y)
    # For multi-dimensional inputs, use matmul
    return matmul(x, y)


def tensordot(x1, x2, axes=2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    # Determine target dtype
    target = _get_promoted_dtype(x1, x2)
    # Use numpy fallback for correct axes interpretation
    x1_np = x1.numpy()
    x2_np = x2.numpy()
    result_np = np.tensordot(x1_np, x2_np, axes=axes)
    result = paddle.to_tensor(result_np, dtype=to_paddle_dtype(target))
    return result


def einsum(subscripts, *operands, **kwargs):
    operands = [convert_to_tensor(x) for x in operands]
    # Determine target dtype using promotion (without CPU casting)
    if len(operands) >= 2:
        target = _get_promoted_dtype(operands[0], operands[1])
    else:
        target = standardize_dtype(operands[0].dtype)
    # einsum int8+int8 widens to int32 (JAX multiply+sum overflow)
    if len(operands) >= 2:
        dt1 = standardize_dtype(operands[0].dtype)
        dt2 = standardize_dtype(operands[1].dtype)
        if dt1 == "int8" and dt2 == "int8":
            target = "int32"
    # Cast unsupported dtypes to float32 (einsum only supports float on CPU)
    for i, op in enumerate(operands):
        dt = standardize_dtype(op.dtype)
        if dt not in ("float32", "float64", "complex64", "complex128"):
            operands[i] = op.cast("float32")
    result = paddle.einsum(subscripts, *operands)
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def get_item(x, i):
    x = convert_to_tensor(x)
    return x[i]


def slice_count(x):
    x = convert_to_tensor(x)
    return len(x.shape)


def shape_equal(x, y):
    x_shape = shape(x)
    y_shape = shape(y)
    return x_shape == y_shape


def where(condition, x1, x2):
    condition = convert_to_tensor(condition, dtype="bool")
    if x1 is not None and x2 is not None:
        x1 = convert_to_tensor(x1)
        x2 = convert_to_tensor(x2)
        dt1 = standardize_dtype(x1.dtype)
        dt2 = standardize_dtype(x2.dtype)
        w1 = x1 in _weak_tensors
        w2 = x2 in _weak_tensors
        low_precision = {"float16", "bfloat16"}
        int_types = {
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        }
        target = None
        if w1 and not w2:
            if dt2 in low_precision:
                target = dt2
        elif w2 and not w1:
            if dt1 in low_precision:
                target = dt1
        elif not w1 and not w2:
            if dt1 in low_precision and dt2 in int_types:
                target = dt1
            elif dt2 in low_precision and dt1 in int_types:
                target = dt2
            elif dt1 in low_precision and dt2 in low_precision:
                target = dt1 if dt1 == dt2 else None
        x1, x2 = _promote_dtypes(x1, x2)
        dt = standardize_dtype(x1.dtype)
        needs_cast = False
        orig_dtype = None
        if dt in ("int8", "int16", "uint8", "bool"):
            x1 = x1.cast("int32")
            x2 = x2.cast("int32")
            needs_cast = True
            orig_dtype = dt
        elif dt in ("float16", "bfloat16"):
            x1 = x1.cast("float32")
            x2 = x2.cast("float32")
            needs_cast = True
            orig_dtype = dt
        result = paddle.where(condition, x1, x2)
        if needs_cast and orig_dtype:
            result = result.cast(to_paddle_dtype(orig_dtype))
        if target is not None:
            result_dtype = standardize_dtype(result.dtype)
            if result_dtype != target:
                result = result.cast(to_paddle_dtype(target))
        return result
    return paddle.where(condition)


def mean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    orig_dtype = standardize_dtype(x.dtype)
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
        needs_cast = True
    # JAX returns float32 for int/bool inputs, float64 for float64
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    result = paddle.mean(x, axis=axis, keepdim=keepdims)
    if orig_dtype in int_types:
        if result.dtype != paddle.float32:
            result = result.cast("float32")
    elif needs_cast:
        result = result.cast(to_paddle_dtype(orig_dtype))
    return result


def variance(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return paddle.zeros_like(x, dtype="float32")
    orig_dtype = standardize_dtype(x.dtype)
    # Cast to float for computation (variance of ints is float)
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif orig_dtype in int_types or orig_dtype in _CPU_UNSUPPORTED_INT:
        x = x.cast("float32")
    # paddle.var defaults to unbiased=True (ddof=1), numpy/JAX uses ddof=0
    result = paddle.var(x, axis=axis, unbiased=False, keepdim=keepdims)
    # JAX returns float32 for int/bool inputs, float64 for float64
    if orig_dtype in int_types and standardize_dtype(result.dtype) != "float32":
        result = result.cast("float32")
    elif needs_cast:
        result = result.cast(to_paddle_dtype(orig_dtype))
    return result


def std(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return paddle.zeros_like(x, dtype="float32")
    orig_dtype = standardize_dtype(x.dtype)
    # Cast to float for computation (std of ints is float)
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif orig_dtype in int_types or orig_dtype in _CPU_UNSUPPORTED_INT:
        x = x.cast("float32")
    # paddle.std defaults to unbiased=True (ddof=1), numpy/JAX uses ddof=0
    result = paddle.std(x, axis=axis, unbiased=False, keepdim=keepdims)
    # JAX returns float32 for int/bool inputs, float64 for float64
    if orig_dtype in int_types and standardize_dtype(result.dtype) != "float32":
        result = result.cast("float32")
    elif needs_cast:
        result = result.cast(to_paddle_dtype(orig_dtype))
    return result


def sum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    orig_dtype = standardize_dtype(x.dtype)
    # Paddle preserves small int types (int8, int16, uint8, bool) for sum,
    # but JAX promotes them to int32/uint32. Also, Paddle promotes int32 to
    # int64 for sum, but JAX keeps it as int32.
    target_dtype = None
    if orig_dtype in ("int8", "int16"):
        x = x.cast("int32")
        target_dtype = "int32"
    elif orig_dtype == "uint8":
        x = x.cast("int32")
        target_dtype = "uint32"
    elif orig_dtype == "bool":
        x = x.cast("int32")
        target_dtype = "int32"
    elif orig_dtype == "int32":
        # Paddle promotes int32 to int64 for sum, but JAX keeps int32
        target_dtype = "int32"
    result = paddle.sum(x, axis=axis, keepdim=keepdims)
    if target_dtype and standardize_dtype(result.dtype) != target_dtype:
        result = result.cast(to_paddle_dtype(target_dtype))
    return result


def prod(x, axis=None, keepdims=False, dtype=None):
    if dtype is not None:
        x = convert_to_tensor(x, dtype=dtype)
    else:
        x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    orig_dtype = standardize_dtype(x.dtype)
    # Cast int/bool to int32 for computation (matching JAX without x64)
    # Paddle preserves int32 for prod, matching JAX behavior
    needs_cast = False
    if orig_dtype in ("int8", "int16", "uint8", "bool"):
        x = x.cast("int32")
        needs_cast = True
    elif x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    result = paddle.prod(x, axis=axis, keepdim=keepdims)
    if needs_cast and orig_dtype in ("float16", "bfloat16"):
        result = result.cast(to_paddle_dtype(orig_dtype))
    return result


def logsumexp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return paddle.logsumexp(x, axis=axis, keepdim=keepdims)


def cumsum(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    dtype = standardize_dtype(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    if standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    result = paddle.cumsum(x, axis=axis, dtype=x.dtype)
    return paddle.cast(result, dtype)


def cumprod(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    dtype = standardize_dtype(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    if standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    result = paddle.cumprod(x, dim=axis, dtype=x.dtype)
    return paddle.cast(result, dtype)


def argmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    # JAX returns int32 for argmax, paddle returns int64
    return paddle.argmax(x, axis=axis, keepdim=keepdims).cast("int32")


def argmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    # JAX returns int32 for argmin, paddle returns int64
    return paddle.argmin(x, axis=axis, keepdim=keepdims).cast("int32")


def argsort(x, axis=-1):
    x = convert_to_tensor(x)
    if axis is None:
        axis = -1
        x = x.reshape([-1])
    # Paddle CPU does not support argsort for many dtypes
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    # Paddle returns int64, but JAX returns int32 — match JAX
    return paddle.argsort(x, axis=axis).cast(paddle.int32)


def sort(x, axis=-1):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    orig_dtype = x.dtype
    # Paddle CPU does not support sort (uses argsort internally) for many dtypes
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    result = paddle.sort(x, axis=axis)
    if orig_dtype != result.dtype:
        result = result.cast(orig_dtype)
    return result


def searchsorted(sorted_sequence, values, side="left"):
    sorted_sequence = convert_to_tensor(sorted_sequence)
    values = convert_to_tensor(values)
    # Paddle CPU searchsorted only supports float32/float64/int32/int64
    _supported = {paddle.float32, paddle.float64, paddle.int32, paddle.int64}
    if sorted_sequence.dtype not in _supported:
        sorted_sequence = sorted_sequence.cast(paddle.float32)
    if values.dtype not in _supported:
        values = values.cast(paddle.float32)
    right = side == "right"
    # Paddle returns int64, but JAX/NumPy return int32 — match them
    return paddle.searchsorted(sorted_sequence, values, right=right).cast(
        paddle.int32
    )


def top_k(x, k, sorted=False):
    return paddle.topk(convert_to_tensor(x), k)


def in_top_k(targets, predictions, k):
    targets = convert_to_tensor(targets, "int64")
    predictions = convert_to_tensor(predictions)
    topk_indices = paddle.topk(predictions, k, axis=-1)[1]
    return paddle.any(topk_indices == targets.unsqueeze(-1), axis=-1)


def flip(x, axis=None):
    x = convert_to_tensor(x)
    if axis is None:
        axis = list(range(x.ndim))
    orig_dtype = x.dtype
    # Paddle CPU does not support flip for many dtypes — cast to float32/int32
    _float_unsupported = {paddle.float16, paddle.bfloat16}
    _int_unsupported = {paddle.int8, paddle.int16, paddle.uint8}
    if orig_dtype in _float_unsupported:
        x = x.cast(paddle.float32)
    elif orig_dtype in _int_unsupported:
        x = x.cast(paddle.int32)
    result = paddle.flip(x, axis=axis)
    if result.dtype != orig_dtype:
        result = result.cast(orig_dtype)
    return result


def roll(x, shift, axis=None):
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
        needs_cast = True
    result = paddle.roll(x, shift, axis=axis)
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def pad(x, pad_width, mode="constant", constant_values=None):
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    # Validate constant_values for non-constant modes
    if mode != "constant" and constant_values is not None:
        raise ValueError(
            "Argument `constant_values` can only be provided when "
            "`mode == 'constant'`. Received: "
            f"mode={mode}, constant_values={constant_values}"
        )
    # Cast unsupported dtypes for CPU
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in ("int8", "int16", "uint8", "bool"):
        x = x.cast("int32")
    pad_list = []
    for left, right in pad_width:
        pad_list.extend([left, right])
    if mode == "constant":
        if constant_values is None:
            constant_values = 0
        result = paddle.nn.functional.pad(
            x, pad_list, mode="constant", value=constant_values
        )
    elif mode == "reflect":
        # paddle only supports reflect for 3D with exactly 6 padding values
        # Use numpy for general case
        x_np = x.numpy()
        pad_width_np = [(int(l), int(r)) for l, r in pad_width]
        result_np = np.pad(x_np, pad_width_np, mode="reflect")
        result = paddle.to_tensor(result_np, dtype=x.dtype)
    elif mode == "symmetric":
        x_np = x.numpy()
        pad_width_np = [(int(l), int(r)) for l, r in pad_width]
        result_np = np.pad(x_np, pad_width_np, mode="symmetric")
        result = paddle.to_tensor(result_np, dtype=x.dtype)
    elif mode == "edge":
        # paddle replicate mode only supports 3D/4D/5D tensors
        if x.ndim < 3:
            x_np = x.numpy()
            pad_width_np = [(int(l), int(r)) for l, r in pad_width]
            result_np = np.pad(x_np, pad_width_np, mode="edge")
            result = paddle.to_tensor(result_np, dtype=x.dtype)
        else:
            result = paddle.nn.functional.pad(x, pad_list, mode="replicate")
    else:
        raise NotImplementedError(
            f"`pad` with mode='{mode}' is not supported with paddle backend"
        )
    return result.cast(orig_dtype)


def _promote_dtypes_list(tensors):
    """Promote a list of tensors to a common dtype."""
    if len(tensors) <= 1:
        return tensors
    # Determine target dtype by pairwise promotion (without CPU casting)
    target_dtype = standardize_dtype(tensors[0].dtype)
    for t in tensors[1:]:
        dt2 = standardize_dtype(t.dtype)
        if target_dtype == dt2:
            continue
        t1_tmp = paddle.zeros((1,), dtype=to_paddle_dtype(target_dtype))
        target_dtype = _get_promoted_dtype(t1_tmp, t)
    # Cast all tensors to the target dtype
    result = []
    for t in tensors:
        t_dt = standardize_dtype(t.dtype)
        if t_dt != target_dtype:
            result.append(t.cast(to_paddle_dtype(target_dtype)))
        else:
            result.append(t)
    return result


def concatenate(xs, axis=0):
    xs = [convert_to_tensor(x) for x in xs]
    xs = _promote_dtypes_list(xs)
    return paddle.concat(xs, axis=axis)


def append(x, values, axis=None):
    x = convert_to_tensor(x)
    values = convert_to_tensor(values)
    target_dtype = _get_promoted_dtype(x, values)
    x_dt = standardize_dtype(x.dtype)
    v_dt = standardize_dtype(values.dtype)
    if x_dt != target_dtype:
        x = x.cast(to_paddle_dtype(target_dtype))
    if v_dt != target_dtype:
        values = values.cast(to_paddle_dtype(target_dtype))
    if axis is None:
        return paddle.concat([x.flatten(), values.flatten()])
    return paddle.concat([x, values], axis=axis)


def stack(x, axis=0):
    x = [convert_to_tensor(xi) for xi in x]
    x = _promote_dtypes_list(x)
    return paddle.stack(x, axis=axis)


def unstack(x, num=None, axis=0):
    return paddle.unbind(x, axis)


def split(x, indices_or_sections, axis=0):
    x = convert_to_tensor(x)
    # Cast unsupported dtypes for paddle.split
    orig_dtype = x.dtype
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
        needs_cast = True
    # Convert tensor indices_or_sections to Python types
    if isinstance(indices_or_sections, paddle.Tensor):
        indices_or_sections = indices_or_sections.numpy().tolist()
    # paddle.split takes section sizes, numpy.split takes split indices
    if isinstance(indices_or_sections, (list, tuple)):
        # Convert split indices to section sizes
        split_points = [int(p) for p in indices_or_sections]
        axis_size = x.shape[axis]
        sizes = []
        prev = 0
        for pt in split_points:
            sizes.append(pt - prev)
            prev = pt
        sizes.append(axis_size - prev)
        result = paddle.split(x, sizes, axis=axis)
    else:
        result = paddle.split(x, int(indices_or_sections), axis=axis)
    if needs_cast:
        result = [r.cast(orig_dtype) for r in result]
    return result


def swapaxes(x, axis1, axis2):
    perm = list(range(len(shape(x))))
    perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
    return paddle.transpose(convert_to_tensor(x), perm)


def moveaxis(x, source, destination):
    x = convert_to_tensor(x)
    ndim = x.ndim
    # Handle tuple/list source and destination
    if isinstance(source, (list, tuple)):
        source = list(source)
    else:
        source = [source]
    if isinstance(destination, (list, tuple)):
        destination = list(destination)
    else:
        destination = [destination]
    # Normalize negative indices
    source = [s + ndim if s < 0 else s for s in source]
    destination = [d + ndim if d < 0 else d for d in destination]
    # Build permutation
    perm = list(range(ndim))
    # Remove sources in reverse order of their position to keep indices valid
    for s in sorted(source, reverse=True):
        perm.pop(s)
    # Insert destinations
    for d, s in sorted(zip(destination, source)):
        perm.insert(d, s)
    return paddle.transpose(x, perm)


def transpose(x, axes=None):
    x = convert_to_tensor(x)
    if axes is None:
        axes = list(range(len(x.shape)))[::-1]
    return paddle.transpose(x, axes)


def squeeze(x, axis=None):
    return paddle.squeeze(convert_to_tensor(x), axis=axis)


def expand_dims(x, axis):
    return paddle.unsqueeze(convert_to_tensor(x), axis)


def reshape(x, newshape):
    return paddle.reshape(convert_to_tensor(x), newshape)


def eye(N, M=None, k=0, dtype="float32"):
    if isinstance(N, float):
        raise TypeError(
            f"An integer is required for N, but received: {N} of type "
            f"{type(N).__name__}"
        )
    if M is not None and isinstance(M, float):
        raise TypeError(
            f"An integer is required for M, but received: {M} of type "
            f"{type(M).__name__}"
        )
    if M is None:
        M = N
    dtype = to_paddle_dtype(dtype)
    # Use numpy for k parameter support, then convert
    result_np = np.eye(N, M, k=k)
    result = paddle.to_tensor(result_np, dtype=dtype)
    return result


def linspace(
    start, stop, num, dtype=None, endpoint=True, retstep=False, axis=0
):
    if dtype is not None:
        dtype = to_paddle_dtype(dtype)
    needs_downcast = dtype in _CPU_UNSUPPORTED_DTYPES
    compute_dtype = (
        paddle.float32
        if needs_downcast
        else (dtype if dtype is not None else paddle.float32)
    )
    # Always convert to float for computation
    start_t = paddle.to_tensor(start, dtype=compute_dtype)
    stop_t = paddle.to_tensor(stop, dtype=compute_dtype)
    is_scalar = start_t.ndim == 0 and stop_t.ndim == 0
    if endpoint and is_scalar:
        result = paddle.linspace(
            start_t.item(), stop_t.item(), num, dtype=compute_dtype
        )
    else:
        # For endpoint=False or array start/stop, compute manually
        denom = (num - 1) if endpoint else num
        if denom == 0:
            denom = 1
        step_val = (stop_t - start_t) / denom
        indices = paddle.arange(num, dtype=compute_dtype)
        if is_scalar:
            result = start_t + indices * step_val
        else:
            result = start_t.unsqueeze(-1) + indices * step_val.unsqueeze(-1)
    if needs_downcast:
        result = result.cast(dtype)
    if retstep:
        step = (stop_t - start_t) / max(num - 1 if endpoint else num, 1)
        return result, step
    return result


def arange(start, stop=None, step=1, dtype=None):
    if dtype is not None:
        dtype = to_paddle_dtype(dtype)
    else:
        # Default: int for integer inputs, float for float inputs
        if builtins.all(
            isinstance(v, (int, np.integer))
            for v in [start, stop, step]
            if v is not None
        ):
            dtype = paddle.int32
    # paddle.arange only supports int32/int64/float32/float64 on CPU
    needs_cast = False
    orig_dtype = dtype
    if dtype is not None:
        supported = {paddle.int32, paddle.int64, paddle.float32, paddle.float64}
        if dtype not in supported:
            needs_cast = True
            if dtype in (paddle.float16, paddle.bfloat16):
                dtype = paddle.float32
            elif dtype == paddle.int16:
                dtype = paddle.int32
            elif dtype == paddle.int8:
                dtype = paddle.int32
            elif dtype == paddle.uint8:
                dtype = paddle.int32
    if step is None:
        step = 1
    if stop is None:
        result = paddle.arange(start, dtype=dtype)
    else:
        result = paddle.arange(start, stop, step, dtype=dtype)
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def full(shape, fill_value, dtype=None):
    if dtype is not None:
        dtype = to_paddle_dtype(dtype)
    # paddle.full doesn't accept numpy array, convert to tensor
    if isinstance(fill_value, np.ndarray):
        fill_value = paddle.to_tensor(fill_value, dtype=dtype)
    # For multi-element tensor fill_value, use broadcast
    if isinstance(fill_value, paddle.Tensor) and fill_value.numel() > 1:
        result = paddle.full(shape, 0, dtype=dtype or fill_value.dtype)
        return result + fill_value.broadcast_to(shape)
    return paddle.full(shape, fill_value, dtype=dtype)


def full_like(x, fill_value, dtype=None):
    if dtype is not None:
        dtype = to_paddle_dtype(dtype)
    if isinstance(fill_value, np.ndarray):
        fill_value = fill_value.flat[0]
    return paddle.full_like(convert_to_tensor(x), fill_value, dtype=dtype)


def zeros(shape, dtype="float32"):
    dtype = to_paddle_dtype(dtype)
    return paddle.zeros(shape, dtype=dtype)


def zeros_like(x, dtype=None):
    if dtype is not None:
        dtype = to_paddle_dtype(dtype)
    return paddle.zeros_like(convert_to_tensor(x), dtype=dtype)


def ones(shape, dtype="float32"):
    dtype = to_paddle_dtype(dtype)
    return paddle.ones(shape, dtype=dtype)


def ones_like(x, dtype=None):
    if dtype is not None:
        dtype = to_paddle_dtype(dtype)
    return paddle.ones_like(convert_to_tensor(x), dtype=dtype)


def identity(n, dtype="float32"):
    return eye(n, dtype=dtype)


def tri(N, M=None, k=0, dtype="float32"):
    if M is None:
        M = N
    dtype = to_paddle_dtype(dtype)
    std_dtype = standardize_dtype(dtype)
    if dtype in _CPU_UNSUPPORTED_DTYPES:
        result = paddle.tril(
            paddle.ones([N, M], dtype=paddle.float32), diagonal=k
        )
        return result.cast(dtype)
    if std_dtype in _CPU_UNSUPPORTED_INT:
        result = paddle.tril(
            paddle.ones([N, M], dtype=paddle.int32), diagonal=k
        )
        return result.cast(dtype)
    return paddle.tril(paddle.ones([N, M], dtype=dtype), diagonal=k)


def tril(x, k=0):
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    if standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
        needs_cast = True
    result = paddle.tril(x, diagonal=k)
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def triu(x, k=0):
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    if standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
        needs_cast = True
    result = paddle.triu(x, diagonal=k)
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def diagonal(x, offset=0, axis1=0, axis2=1):
    return paddle.diagonal(convert_to_tensor(x), offset, axis1, axis2)


def trace(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    orig_dtype = standardize_dtype(x.dtype)
    needs_cast = False
    # JAX promotes bool/int8/int16/uint8 to int32/uint32 for trace
    target_dtype = None
    if orig_dtype in ("bool", "int8", "int16"):
        x = x.cast("int32")
        target_dtype = "int32"
    elif orig_dtype == "uint8":
        x = x.cast("int32")
        target_dtype = "uint32"
    # Cast fp16/bf16 for numpy fallback
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    # Use numpy fallback for multi-dimensional trace support
    x_np = x.numpy()
    result_np = np.trace(x_np, offset=offset, axis1=axis1, axis2=axis2)
    result = paddle.to_tensor(result_np, dtype=x.dtype)
    if target_dtype and standardize_dtype(result.dtype) != target_dtype:
        result = result.cast(to_paddle_dtype(target_dtype))
    if needs_cast:
        result = result.cast(to_paddle_dtype(orig_dtype))
    return result


def meshgrid(*x, indexing="xy"):
    x = [convert_to_tensor(xi) for xi in x]
    orig_dtype = x[0].dtype if x else None
    needs_cast = False
    for i, t in enumerate(x):
        if t.dtype in _CPU_UNSUPPORTED_DTYPES:
            x[i] = t.cast("float32")
            needs_cast = True
        elif standardize_dtype(t.dtype) in _CPU_UNSUPPORTED_INT:
            x[i] = t.cast("int32")
            needs_cast = True
    result = paddle.meshgrid(*x, indexing=indexing)
    if needs_cast and orig_dtype is not None:
        result = [r.cast(orig_dtype) for r in result]
    return result


def histogram(x, bins=10, range=None):
    x = convert_to_tensor(x)
    x_np = x.numpy()
    if range is not None:
        hist, bin_edges = np.histogram(x_np, bins=bins, range=range)
    else:
        hist, bin_edges = np.histogram(x_np, bins=bins)
    return (
        paddle.to_tensor(hist, dtype="int64"),
        paddle.to_tensor(bin_edges, dtype=x.dtype),
    )


def tile(x, repeats):
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    result = paddle.tile(x, repeats)
    return result.cast(orig_dtype)


def repeat(x, repeats, axis=None):
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
        needs_cast = True
    # paddle.repeat_interleave requires int or Tensor for repeats
    if isinstance(repeats, np.ndarray):
        if repeats.size == 1:
            repeats = int(repeats.flat[0])
        else:
            repeats = paddle.to_tensor(repeats)
    result = paddle.repeat_interleave(x, repeats, axis=axis)
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def take(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices, dtype="int64")
    orig_dtype = x.dtype
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    if axis is None:
        return paddle.gather(x.flatten(), indices.flatten()).cast(orig_dtype)
    # Paddle's gather requires indices to be 1D or have last dim == 1
    # Reshape indices to work with paddle.gather
    axis = axis + x.ndim if axis < 0 else axis
    indices_flat = indices.flatten()
    result = paddle.gather(x, indices_flat, axis=axis)
    new_shape = (
        list(x.shape[:axis]) + list(indices.shape) + list(x.shape[axis + 1 :])
    )
    return paddle.reshape(result, new_shape).cast(orig_dtype)


def take_along_axis(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices, dtype="int64")
    orig_dtype = x.dtype
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    if standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    if axis is None:
        result = paddle.take_along_axis(x.flatten(), indices.flatten(), 0)
    else:
        # Ensure indices has same ndim as x
        while indices.ndim < x.ndim:
            indices = indices.unsqueeze(-1)
        result = paddle.take_along_axis(x, indices, axis)
    return result.cast(orig_dtype)


def put_along_axis(x, indices, values, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices, dtype="int64")
    values = convert_to_tensor(values, dtype=x.dtype)
    if axis is None:
        return paddle.put_along_axis(
            x.flatten(), indices.flatten(), values.flatten(), axis=0
        ).reshape(x.shape)
    return paddle.put_along_axis(x, indices, values, axis=axis)


def block_diag(inputs):
    inputs = [convert_to_tensor(x) for x in inputs]
    rows = []
    for i, x in enumerate(inputs):
        row = []
        for j, y in enumerate(inputs):
            if i == j:
                row.append(x)
            else:
                row.append(
                    paddle.zeros([x.shape[0], y.shape[1]], dtype=x.dtype)
                )
        rows.append(paddle.concat(row, axis=1))
    return paddle.concat(rows, axis=0)


def conjugate(x):
    return paddle.conj(convert_to_tensor(x))


def conj(x):
    return conjugate(x)


def real(x):
    return paddle.real(convert_to_tensor(x))


def imag(x):
    return paddle.imag(convert_to_tensor(x))


def angle(x):
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    if standardize_dtype(x.dtype) not in _FLOAT_TYPES:
        x = x.cast("float32")
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    result = paddle.angle(x)
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    # paddle.isclose only supports float types on CPU
    if x1.dtype in _CPU_UNSUPPORTED_DTYPES:
        x1 = x1.cast("float32")
    if x2.dtype in _CPU_UNSUPPORTED_DTYPES:
        x2 = x2.cast("float32")
    dt1 = standardize_dtype(x1.dtype)
    dt2 = standardize_dtype(x2.dtype)
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
    }
    if dt1 in int_types:
        x1 = x1.cast("float64")
    if dt2 in int_types:
        x2 = x2.cast("float64")
    # Ensure both have the same dtype
    if x1.dtype != x2.dtype:
        x1 = x1.cast(x2.dtype)
    return paddle.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


def allclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    # paddle.allclose only supports float types on CPU
    if x1.dtype in _CPU_UNSUPPORTED_DTYPES:
        x1 = x1.cast("float32")
    if x2.dtype in _CPU_UNSUPPORTED_DTYPES:
        x2 = x2.cast("float32")
    dt1 = standardize_dtype(x1.dtype)
    dt2 = standardize_dtype(x2.dtype)
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
    }
    if dt1 in int_types:
        x1 = x1.cast("float64")
    if dt2 in int_types:
        x2 = x2.cast("float64")
    return paddle.allclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


def _comparison_op(op, x1, x2):
    """Run a comparison op with CPU dtype handling."""
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    if x1.dtype in _CPU_UNSUPPORTED_DTYPES:
        x1 = x1.cast("float32")
    if x2.dtype in _CPU_UNSUPPORTED_DTYPES:
        x2 = x2.cast("float32")
    x1, x2 = _promote_dtypes(x1, x2)
    return op(x1, x2)


def equal(x1, x2):
    return _comparison_op(paddle.equal, x1, x2)


def not_equal(x1, x2):
    return _comparison_op(paddle.not_equal, x1, x2)


def greater(x1, x2):
    return _comparison_op(paddle.greater_than, x1, x2)


def greater_equal(x1, x2):
    return _comparison_op(paddle.greater_equal, x1, x2)


def less(x1, x2):
    return _comparison_op(paddle.less_than, x1, x2)


def less_equal(x1, x2):
    return _comparison_op(paddle.less_equal, x1, x2)


def all(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    return paddle.all(x, axis=axis, keepdim=keepdims)


def any(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    return paddle.any(x, axis=axis, keepdim=keepdims)


def logical_and(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    if x1.dtype in _CPU_UNSUPPORTED_DTYPES:
        x1 = x1.cast("float32")
    if x2.dtype in _CPU_UNSUPPORTED_DTYPES:
        x2 = x2.cast("float32")
    if standardize_dtype(x1.dtype) in _CPU_UNSUPPORTED_INT:
        x1 = x1.cast("int32")
    if standardize_dtype(x2.dtype) in _CPU_UNSUPPORTED_INT:
        x2 = x2.cast("int32")
    x1, x2 = _promote_dtypes(x1, x2)
    return paddle.logical_and(x1, x2)


def logical_or(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    if x1.dtype in _CPU_UNSUPPORTED_DTYPES:
        x1 = x1.cast("float32")
    if x2.dtype in _CPU_UNSUPPORTED_DTYPES:
        x2 = x2.cast("float32")
    if standardize_dtype(x1.dtype) in _CPU_UNSUPPORTED_INT:
        x1 = x1.cast("int32")
    if standardize_dtype(x2.dtype) in _CPU_UNSUPPORTED_INT:
        x2 = x2.cast("int32")
    x1, x2 = _promote_dtypes(x1, x2)
    return paddle.logical_or(x1, x2)


def logical_not(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    elif x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    return paddle.logical_not(x)


def logical_xor(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    if x1.dtype in _CPU_UNSUPPORTED_DTYPES:
        x1 = x1.cast("float32")
    if x2.dtype in _CPU_UNSUPPORTED_DTYPES:
        x2 = x2.cast("float32")
    if standardize_dtype(x1.dtype) in _CPU_UNSUPPORTED_INT:
        x1 = x1.cast("int32")
    if standardize_dtype(x2.dtype) in _CPU_UNSUPPORTED_INT:
        x2 = x2.cast("int32")
    x1, x2 = _promote_dtypes(x1, x2)
    return paddle.logical_xor(x1, x2)


def bitwise_and(x1, x2):
    return _binary_op_with_dtype(paddle.bitwise_and, x1, x2)


def bitwise_or(x1, x2):
    return _binary_op_with_dtype(paddle.bitwise_or, x1, x2)


def bitwise_xor(x1, x2):
    return _binary_op_with_dtype(paddle.bitwise_xor, x1, x2)


def bitwise_not(x):
    return paddle.bitwise_not(convert_to_tensor(x))


def isfinite(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) not in _FLOAT_TYPES:
        x = x.cast("float32")
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    return paddle.isfinite(x)


def isinf(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) not in _FLOAT_TYPES:
        x = x.cast("float32")
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    return paddle.isinf(x)


def isnan(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) not in _FLOAT_TYPES:
        x = x.cast("float32")
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    return paddle.isnan(x)


def isneginf(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    elif x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    return paddle.logical_and(paddle.isinf(x), x < 0)


def isposinf(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    elif x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    return paddle.logical_and(paddle.isinf(x), x > 0)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    # paddle.nan_to_num only supports float types
    if standardize_dtype(x.dtype) not in _FLOAT_TYPES:
        return x.clone()
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    result = paddle.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    target, x1, x2 = _cpu_binary_target(x1, x2)
    if axis is not None:
        x1 = paddle.moveaxis(x1, axisa, axis)
        x2 = paddle.moveaxis(x2, axisb, axis)
    # Broadcast to same shape (paddle.cross doesn't support broadcasting)
    target_shape = paddle.broadcast_shape(x1.shape, x2.shape)
    x1 = paddle.broadcast_to(x1, target_shape)
    x2 = paddle.broadcast_to(x2, target_shape)
    result = paddle.cross(x1, x2, axis=axisc if axis is not None else axisa)
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids, "int64")
    orig_dtype = data.dtype
    needs_cast = False
    if data.dtype in _CPU_UNSUPPORTED_DTYPES:
        data = data.cast("float32")
        needs_cast = True
    elif standardize_dtype(data.dtype) in _CPU_UNSUPPORTED_INT:
        data = data.cast("int32")
        needs_cast = True
    if num_segments is None:
        num_segments = paddle.max(segment_ids) + 1
    zeros = paddle.zeros(
        [num_segments] + list(data.shape[1:]), dtype=data.dtype
    )
    result = paddle.scatter_nd_add(zeros, segment_ids.unsqueeze(-1), data)
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids, "int64")
    orig_dtype = data.dtype
    needs_cast = False
    if data.dtype in _CPU_UNSUPPORTED_DTYPES:
        data = data.cast("float32")
        needs_cast = True
    elif standardize_dtype(data.dtype) in _CPU_UNSUPPORTED_INT:
        data = data.cast("int32")
        needs_cast = True
    if num_segments is None:
        num_segments = int(segment_ids.max().item()) + 1
    result = paddle.zeros(
        [num_segments] + list(data.shape[1:]), dtype=data.dtype
    )
    for i in range(num_segments):
        mask = segment_ids == i
        if mask.any():
            result[i] = paddle.max(data[mask], axis=0)
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def gamma(shape, alpha, dtype=None, seed=None):
    alpha = convert_to_tensor(alpha)
    return paddle.distribution.gamma.Gamma(
        alpha, paddle.ones_like(alpha)
    ).sample(shape)


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    counts = convert_to_tensor(counts)
    probabilities = convert_to_tensor(probabilities)
    return paddle.distribution.binomial.Binomial(counts, probabilities).sample(
        shape
    )


def beta(shape, alpha, beta_param, dtype=None, seed=None):
    alpha = convert_to_tensor(alpha)
    beta_param = convert_to_tensor(beta_param)
    return paddle.distribution.beta.Beta(alpha, beta_param).sample(shape)


# --- Additional ops needed for layers/models ---


def matmul(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    # matmul involves multiply+sum, int8 results widen to int32 (JAX behavior)
    if (
        standardize_dtype(x1.dtype) == "int8"
        and standardize_dtype(x2.dtype) == "int8"
    ):
        x1 = x1.cast("int32")
        x2 = x2.cast("int32")
    return _binary_op_with_int(paddle.matmul, x1, x2)


def copy(x):
    return convert_to_tensor(x).clone()


def broadcast_to(x, shape):
    return paddle.broadcast_to(convert_to_tensor(x), shape)


def array(x, dtype=None):
    if dtype is not None:
        return convert_to_tensor(x, dtype=dtype)
    return convert_to_tensor(x)


def ndim(x):
    return len(convert_to_tensor(x).shape)


def size(x):
    return convert_to_tensor(x).size


def ravel(x):
    return paddle.flatten(convert_to_tensor(x))


def trunc(x):
    return _unary_op(paddle.trunc, x)


def inner(a, b):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    target, a, b = _cpu_binary_target(a, b)
    # inner product: sum over last axis
    # output shape: a.shape[:-1] + b.shape[:-1]
    if a.ndim == 1 and b.ndim == 1:
        result = paddle.dot(a, b)
    else:
        # Reshape: a -> [..., 1, K], b -> [1, ..., K], then multiply and sum
        a_expanded = a.reshape(list(a.shape[:-1]) + [1] + [a.shape[-1]])
        b_expanded = b.reshape([1] * (a.ndim - 1) + list(b.shape))
        result = paddle.sum(a_expanded * b_expanded, axis=-1)
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def outer(a, b):
    a = convert_to_tensor(a).flatten()
    b = convert_to_tensor(b).flatten()
    target, a, b = _cpu_binary_target(a, b)
    result = paddle.mm(a.unsqueeze(1), b.unsqueeze(0))
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def reciprocal(x):
    return _unary_math_op(paddle.reciprocal, x)


def cos(x):
    return _unary_math_op(paddle.cos, x)


def sin(x):
    return _unary_math_op(paddle.sin, x)


def tan(x):
    return _unary_math_op(paddle.tan, x)


def cosh(x):
    return _unary_math_op(paddle.cosh, x)


def sinh(x):
    return _unary_math_op(paddle.sinh, x)


def arccos(x):
    return _unary_math_op(paddle.acos, x)


def arcsin(x):
    return _unary_math_op(paddle.asin, x)


def arctan(x):
    return _unary_math_op(paddle.atan, x)


def arctan2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    target, x1, x2 = _cpu_binary_target(x1, x2)
    # arctan2 always returns float for int inputs
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
    }
    if target in int_types:
        target = "float32"
    result = paddle.atan2(x1, x2)
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def arccosh(x):
    return _unary_math_op(paddle.acosh, x)


def arcsinh(x):
    return _unary_math_op(paddle.asinh, x)


def arctanh(x):
    return _unary_math_op(paddle.atanh, x)


def deg2rad(x):
    return _unary_math_op(paddle.deg2rad, x)


def rad2deg(x):
    return _unary_math_op(paddle.rad2deg, x)


def hypot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    target, x1, x2 = _cpu_binary_target(x1, x2)
    # hypot always returns float for int inputs
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
    }
    if target in int_types:
        target = "float32"
    result = paddle.hypot(x1, x2)
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def fmod(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    # numpy.fmod uses truncation division, not floor division
    target = _get_promoted_dtype(x1, x2)
    # Cast to float64 for computation
    x1f = x1.cast("float64")
    x2f = x2.cast("float64")
    result = x1f - paddle.trunc(x1f / x2f) * x2f
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def ldexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dt1 = standardize_dtype(x1.dtype)
    float_types = {"float16", "float32", "float64", "bfloat16"}
    # Determine target dtype
    if dt1 in float_types:
        target = dt1
    else:
        # JAX ldexp always returns float32 for int inputs
        target = "float32"
    x1 = x1.cast("float32")
    x2 = x2.cast("float32")
    result = x1 * paddle.pow(paddle.to_tensor(2.0, dtype="float32"), x2)
    return result.cast(to_paddle_dtype(target))


def left_shift(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    target, x1, x2 = _cpu_binary_target(x1, x2)
    result = paddle.bitwise_left_shift(x1, x2)
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def right_shift(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    target, x1, x2 = _cpu_binary_target(x1, x2)
    result = paddle.bitwise_right_shift(x1, x2)
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def bitwise_left_shift(x1, x2):
    return left_shift(x1, x2)


def bitwise_right_shift(x1, x2):
    return right_shift(x1, x2)


def bitwise_invert(x):
    return _unary_op(paddle.bitwise_not, x)


def signbit(x):
    x = convert_to_tensor(x)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    # Use numpy to detect signbit (handles -0.0 correctly)
    return convert_to_tensor(np.signbit(x.numpy()), dtype="bool")


def heaviside(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    target, x1, x2 = _cpu_binary_target(x1, x2)
    # heaviside returns float32 for int inputs
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
    }
    if target in int_types:
        target = "float32"
    result = paddle.where(
        x1 > 0,
        paddle.ones_like(x1),
        paddle.where(x1 == 0, x2, paddle.zeros_like(x1)),
    )
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def i0(x):
    x = convert_to_tensor(x, "float32")
    return paddle.i0(x)


def sinc(x):
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif standardize_dtype(x.dtype) not in (
        "float32",
        "float64",
        "complex64",
        "complex128",
    ):
        x = x.cast("float32")
    result = paddle.where(
        x == 0, paddle.ones_like(x), paddle.sin(np.pi * x) / (np.pi * x)
    )
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def count_nonzero(x, axis=None):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        # axis=() means no reduction, return per-element nonzero count
        return paddle.cast(x != 0, "int32")
    return paddle.sum(paddle.cast(x != 0, "int32"), axis=axis).cast("int32")


def nanargmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    if standardize_dtype(x.dtype) in _FLOAT_TYPES:
        mask = paddle.isnan(x)
        # Check for all-NaN slices and return -1 for them
        all_nan = paddle.all(mask, axis=axis, keepdim=False)
        x = paddle.where(mask, paddle.full_like(x, float("-inf")), x)
    else:
        all_nan = None
    # JAX returns int32 for argmax, paddle returns int64
    result = paddle.argmax(x, axis=axis, keepdim=keepdims).cast("int32")
    if all_nan is not None:
        # Replace result with -1 where all values were NaN
        if keepdims:
            all_nan_expanded = (
                paddle.unsqueeze(all_nan, axis=axis)
                if axis is not None
                else all_nan
            )
            result = paddle.where(
                all_nan_expanded, paddle.to_tensor(-1, dtype="int32"), result
            )
        else:
            result = paddle.where(
                all_nan, paddle.to_tensor(-1, dtype="int32"), result
            )
    return result


def nanargmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    if standardize_dtype(x.dtype) in _FLOAT_TYPES:
        mask = paddle.isnan(x)
        all_nan = paddle.all(mask, axis=axis, keepdim=False)
        x = paddle.where(mask, paddle.full_like(x, float("inf")), x)
    else:
        all_nan = None
    # JAX returns int32 for argmin, paddle returns int64
    result = paddle.argmin(x, axis=axis, keepdim=keepdims).cast("int32")
    if all_nan is not None:
        if keepdims:
            all_nan_expanded = (
                paddle.unsqueeze(all_nan, axis=axis)
                if axis is not None
                else all_nan
            )
            result = paddle.where(
                all_nan_expanded, paddle.to_tensor(-1, dtype="int32"), result
            )
        else:
            result = paddle.where(
                all_nan, paddle.to_tensor(-1, dtype="int32"), result
            )
    return result


def nanmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    orig_dtype = x.dtype
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
        needs_cast = True
    if standardize_dtype(x.dtype) in _FLOAT_TYPES:
        mask = paddle.isnan(x)
        all_nan = paddle.all(mask, axis=axis, keepdim=keepdims)
        x = paddle.where(mask, paddle.full_like(x, float("-inf")), x)
    else:
        all_nan = None
    result = paddle.max(x, axis=axis, keepdim=keepdims)
    if all_nan is not None:
        result = paddle.where(
            all_nan, paddle.to_tensor(float("nan"), dtype=result.dtype), result
        )
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def nanmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    orig_dtype = x.dtype
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
        needs_cast = True
    if standardize_dtype(x.dtype) in _FLOAT_TYPES:
        mask = paddle.isnan(x)
        all_nan = paddle.all(mask, axis=axis, keepdim=keepdims)
        x = paddle.where(mask, paddle.full_like(x, float("inf")), x)
    else:
        all_nan = None
    result = paddle.min(x, axis=axis, keepdim=keepdims)
    if all_nan is not None:
        result = paddle.where(
            all_nan, paddle.to_tensor(float("nan"), dtype=result.dtype), result
        )
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def nansum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        # axis=() means no reduction, but NaN still replaced with 0
        if standardize_dtype(x.dtype) in _FLOAT_TYPES:
            return paddle.where(paddle.isnan(x), paddle.zeros_like(x), x)
        return x.clone()
    orig_dtype = standardize_dtype(x.dtype)
    needs_cast = False
    # Cast unsupported dtypes before where (paddle.where needs fp32)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    # Replace NaN with 0 for float types
    if standardize_dtype(x.dtype) in _FLOAT_TYPES:
        x = paddle.where(paddle.isnan(x), paddle.zeros_like(x), x)
    # JAX promotes small ints to int32/uint32 for nansum (same as sum)
    target_dtype = None
    if orig_dtype in ("int8", "int16"):
        x = x.cast("int32")
        target_dtype = "int32"
    elif orig_dtype == "uint8":
        x = x.cast("int32")
        target_dtype = "uint32"
    elif orig_dtype == "bool":
        x = x.cast("int32")
        target_dtype = "int32"
    elif orig_dtype == "int32":
        target_dtype = "int32"
    elif orig_dtype in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    result = paddle.sum(x, axis=axis, keepdim=keepdims)
    if target_dtype and standardize_dtype(result.dtype) != target_dtype:
        result = result.cast(to_paddle_dtype(target_dtype))
    if needs_cast:
        result = result.cast(to_paddle_dtype(orig_dtype))
    return result


def nanmean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return (
            x.clone().cast("float32")
            if standardize_dtype(x.dtype)
            in (
                "bool",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
            )
            else x.clone()
        )
    orig_dtype = standardize_dtype(x.dtype)
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif orig_dtype in int_types:
        x = x.cast("float32")
    mask = ~paddle.isnan(x)
    x = paddle.where(mask, x, paddle.zeros_like(x))
    count = paddle.sum(
        paddle.cast(mask, "float32"), axis=axis, keepdim=keepdims
    )
    result = paddle.sum(x, axis=axis, keepdim=keepdims) / count
    # JAX returns float32 for int/bool inputs (already handled by casting above)
    if needs_cast:
        result = result.cast(to_paddle_dtype(orig_dtype))
    return result


def nanvar(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        # axis=() means no reduction, var of single element is 0, NaN preserved
        result = paddle.zeros_like(x, dtype="float32")
        if standardize_dtype(x.dtype) in _FLOAT_TYPES:
            result = paddle.where(
                paddle.isnan(x),
                paddle.to_tensor(float("nan"), dtype="float32").expand_as(x),
                result,
            )
        return result
    orig_dtype = standardize_dtype(x.dtype)
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif orig_dtype in int_types:
        x = x.cast("float32")
    mask = ~paddle.isnan(x)
    m = nanmean(x, axis=axis, keepdims=True)
    x = paddle.where(mask, x, paddle.zeros_like(x))
    diff = (x - m) * mask.cast(x.dtype)
    count = paddle.sum(mask.cast("float32"), axis=axis, keepdim=True)
    result = paddle.sum(diff**2, axis=axis, keepdim=True) / count
    if not keepdims:
        result = result.squeeze(axis) if axis is not None else result.squeeze()
    # JAX returns float32 for int/bool inputs (already handled by casting above)
    if needs_cast:
        result = result.cast(to_paddle_dtype(orig_dtype))
    return result


def nanstd(x, axis=None, keepdims=False):
    result = nanvar(x, axis=axis, keepdims=keepdims)
    # paddle.sqrt doesn't support fp16/bf16, cast to float32
    orig_dtype = result.dtype
    if result.dtype in _CPU_UNSUPPORTED_DTYPES:
        result = result.cast("float32")
    result = paddle.sqrt(result)
    if orig_dtype in _CPU_UNSUPPORTED_DTYPES:
        result = result.cast(orig_dtype)
    return result


def nanprod(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        # axis=() means no reduction, but NaN still replaced with 1
        if standardize_dtype(x.dtype) in _FLOAT_TYPES:
            return paddle.where(paddle.isnan(x), paddle.ones_like(x), x)
        return x.clone()
    orig_dtype = standardize_dtype(x.dtype)
    needs_cast = False
    # Cast unsupported dtypes before where (paddle.where needs fp32)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    # Replace NaN with 1 for float types
    if standardize_dtype(x.dtype) in _FLOAT_TYPES:
        x = paddle.where(paddle.isnan(x), paddle.ones_like(x), x)
    # Cast int/bool to int32 for computation
    if orig_dtype in ("int8", "int16", "uint8", "bool"):
        x = x.cast("int32")
    elif orig_dtype in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    result = paddle.prod(x, axis=axis, keepdim=keepdims)
    if needs_cast:
        result = result.cast(to_paddle_dtype(orig_dtype))
    return result


def nancumsum(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    orig_std = standardize_dtype(orig_dtype)
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
        needs_cast = True
    # paddle.cumsum promotes int32→int64, need to cast back
    # But bool→int32 should NOT be cast back (JAX returns int32 for bool)
    if orig_std == "int32":
        needs_cast = True
    x = paddle.where(paddle.isnan(x), paddle.zeros_like(x), x)
    result = paddle.cumsum(x, axis=axis)
    # Cast int64 back to int32 if needed
    if standardize_dtype(result.dtype) == "int64" and orig_std in (
        "int32",
        "bool",
    ):
        result = result.cast("int32")
    if dtype is not None:
        result = result.cast(to_paddle_dtype(dtype))
    elif needs_cast and orig_std != "bool":
        result = result.cast(orig_dtype)
    return result


def nancumprod(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    orig_std = standardize_dtype(orig_dtype)
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
        needs_cast = True
    # paddle.cumprod promotes int32→int64, need to cast back
    # But bool→int32 should NOT be cast back (JAX returns int32 for bool)
    if orig_std == "int32":
        needs_cast = True
    x = paddle.where(paddle.isnan(x), paddle.ones_like(x), x)
    result = paddle.cumprod(x, dim=axis)
    # Cast int64 back to int32 if needed
    if standardize_dtype(result.dtype) == "int64" and orig_std in (
        "int32",
        "bool",
    ):
        result = result.cast("int32")
    if dtype is not None:
        result = result.cast(to_paddle_dtype(dtype))
    elif needs_cast and orig_std != "bool":
        result = result.cast(orig_dtype)
    return result


def select(condlist, choicelist, default=0):
    # Determine common dtype from choicelist
    tensors = [convert_to_tensor(c) for c in choicelist]
    if tensors:
        target_dtype = tensors[0].dtype
        orig_dtype = target_dtype
        needs_cast = False
        if target_dtype in _CPU_UNSUPPORTED_DTYPES:
            target_dtype = paddle.float32
            needs_cast = True
        elif standardize_dtype(target_dtype) in _CPU_UNSUPPORTED_INT:
            target_dtype = paddle.int32
            needs_cast = True
        tensors = [
            t.cast(target_dtype) if t.dtype != target_dtype else t
            for t in tensors
        ]
    else:
        needs_cast = False
        orig_dtype = None
    result = paddle.full_like(
        tensors[-1] if tensors else convert_to_tensor(0), default
    )
    for cond, choice in reversed(list(zip(condlist, tensors))):
        result = paddle.where(convert_to_tensor(cond), choice, result)
    if needs_cast and orig_dtype is not None:
        result = result.cast(orig_dtype)
    return result


def unique(x, **kwargs):
    x = convert_to_tensor(x)
    # paddle.unique doesn't support 'size' and 'fill_value' kwargs
    kwargs.pop("size", None)
    kwargs.pop("fill_value", None)
    return paddle.unique(x, **kwargs)


def unravel_index(indices, shape):
    orig_indices = convert_to_tensor(indices)
    orig_dtype = orig_indices.dtype
    indices = orig_indices.cast("int64")
    result = []
    for s in reversed(shape):
        result.append((indices % s).cast("int32"))
        indices = indices // s
    result = list(reversed(result))
    # Cast back to original dtype
    return tuple(r.cast(orig_dtype) for r in result)


def kron(a, b):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    target = _get_promoted_dtype(a, b)
    a_dt = standardize_dtype(a.dtype)
    b_dt = standardize_dtype(b.dtype)
    if a_dt != target:
        a = a.cast(to_paddle_dtype(target))
    if b_dt != target:
        b = b.cast(to_paddle_dtype(target))
    # CPU doesn't support kron for int8/int16/uint8/bool
    if target in _CPU_UNSUPPORTED_INT:
        a = a.cast("int32")
        b = b.cast("int32")
    result = paddle.kron(a, b)
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def vdot(a, b):
    a = convert_to_tensor(a).flatten()
    b = convert_to_tensor(b).flatten()
    target, a, b = _cpu_binary_target(a, b)
    if a.is_complex():
        result = paddle.dot(a.conj(), b)
    else:
        result = paddle.dot(a, b)
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def vectorize(pyfunc, **kwargs):
    import functools

    @functools.wraps(pyfunc)
    def wrapper(*args):
        return pyfunc(*args)

    return wrapper


def view(x, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        return x
    return x.view(dtype=to_paddle_dtype(dtype))


def diff(x, n=1, axis=-1, prepend=None, append=None):
    x = convert_to_tensor(x)
    if prepend is not None:
        prepend = convert_to_tensor(prepend)
        x = paddle.concat([prepend, x], axis=axis)
    if append is not None:
        append = convert_to_tensor(append)
        x = paddle.concat([x, append], axis=axis)
    orig_dtype = x.dtype
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    for _ in range(n):
        x = paddle.diff(x, axis=axis)
    return x.cast(orig_dtype)


def digitize(x, bins):
    x = convert_to_tensor(x)
    bins = convert_to_tensor(bins)
    # numpy.digitize is equivalent to searchsorted with side='right'
    return searchsorted(bins, x, side="right")


def bincount(x, weights=None, minlength=0, sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with paddle backend")
    x = convert_to_tensor(x, "int64")
    # Handle batch dimensions: process each row along the last axis
    if x.ndim > 1:
        batch_shape = x.shape[:-1]
        x_flat = x.reshape([-1, x.shape[-1]])
        if weights is not None:
            weights = convert_to_tensor(weights)
            weights_flat = weights.reshape([-1, weights.shape[-1]])
        else:
            weights_flat = None
        results = []
        for i in range(x_flat.shape[0]):
            row_result = bincount(
                x_flat[i],
                weights=weights_flat[i] if weights_flat is not None else None,
                minlength=minlength,
                sparse=False,
            )
            results.append(row_result)
        result = paddle.stack(results).reshape(
            batch_shape + [results[0].shape[0]]
        )
        return result
    if weights is not None:
        weights = convert_to_tensor(weights)
    n = paddle.maximum(x.max() + 1, paddle.to_tensor(minlength, dtype="int64"))
    if weights is None:
        ones = paddle.ones([x.shape[0]], dtype="int64")
        result = paddle.scatter_nd_add(
            paddle.zeros([n], dtype="int64"), x.unsqueeze(-1), ones
        )
        # JAX returns int32 for unweighted bincount
        return result.cast("int32")
    orig_w_dtype = weights.dtype
    needs_cast = False
    if weights.dtype in _CPU_UNSUPPORTED_DTYPES:
        weights = weights.cast("float32")
        needs_cast = True
    elif standardize_dtype(weights.dtype) in _CPU_UNSUPPORTED_INT:
        weights = weights.cast("int32")
        needs_cast = True
    result = paddle.scatter_nd_add(
        paddle.zeros([n], dtype=weights.dtype), x.unsqueeze(-1), weights
    )
    if needs_cast:
        result = result.cast(orig_w_dtype)
    return result


def corrcoef(x):
    x = convert_to_tensor(x)
    orig_dtype = standardize_dtype(x.dtype)
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    needs_cast_back = False
    if orig_dtype in int_types or orig_dtype == "bool":
        x = x.cast("float32")
        # JAX returns float32 for int/bool inputs, don't cast back
    elif orig_dtype in ("float16", "bfloat16"):
        x = x.cast("float32")
        needs_cast_back = True
    elif orig_dtype == "int64":
        x = x.cast("float64")
        needs_cast_back = True
    # Flatten to 2D if needed
    if x.ndim < 2:
        x = x.unsqueeze(0)
    # Center the data
    x = x - x.mean(axis=1, keepdim=True)
    # Compute covariance
    n = x.shape[1]
    c = paddle.matmul(x, x.t()) / n
    # Compute correlation
    d = paddle.diag(c).clip(min=1e-12).sqrt()
    c = c / (d.unsqueeze(1) * d.unsqueeze(0))
    # Cast back to original dtype if needed (float16/bfloat16)
    if needs_cast_back and c.dtype != orig_dtype:
        c = c.cast(orig_dtype)
    return c


def correlate(x1, x2, mode="valid"):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    # Determine target dtype using promotion (without CPU casting)
    target_dtype = _get_promoted_dtype(x1, x2)
    # JAX correlate returns float32 for int inputs
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
    }
    if target_dtype in int_types:
        target_dtype = "float32"
    # Use numpy fallback for correct mode handling
    x1_np = x1.numpy().astype("float64")
    x2_np = x2.numpy().astype("float64")
    result_np = np.correlate(x1_np, x2_np, mode=mode)
    result = paddle.to_tensor(result_np, dtype=to_paddle_dtype(target_dtype))
    return result


def median(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    needs_cast = False
    if standardize_dtype(x.dtype) not in _FLOAT_TYPES:
        x = x.cast("float32")
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    # paddle.median only supports int axis, not tuple
    if isinstance(axis, tuple):
        axis = axis[0] if len(axis) == 1 else axis
    # Use numpy fallback for tuple axis
    if isinstance(axis, tuple):
        x_np = x.numpy()
        result_np = np.median(x_np, axis=axis, keepdims=keepdims)
        result = paddle.to_tensor(result_np, dtype=x.dtype)
    else:
        result = paddle.median(x, axis=axis, keepdim=keepdims)
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def quantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q, "float32")
    orig_dtype = x.dtype
    needs_cast = False
    if standardize_dtype(x.dtype) not in _FLOAT_TYPES:
        x = x.cast("float32")
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    if isinstance(axis, tuple):
        axis = list(axis)
    result = paddle.quantile(
        x, q, axis=axis, keepdim=keepdims, interpolation=method
    )
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def percentile(x, q, axis=None, method="linear", keepdims=False):
    return quantile(x, q / 100.0, axis=axis, method=method, keepdims=keepdims)


def nanmedian(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    orig_dtype = standardize_dtype(x.dtype)
    # Use numpy fallback for correct NaN handling
    # Cast int/bool types to float32 (numpy can't handle bool subtract)
    x_np = x.numpy()
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    if orig_dtype in int_types:
        x_np = x_np.astype("float32")
        result_dtype = "float32"
    else:
        result_dtype = orig_dtype
    result_np = np.nanmedian(x_np, axis=axis, keepdims=keepdims)
    result = paddle.to_tensor(result_np, dtype=to_paddle_dtype(result_dtype))
    return result


def nanquantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    orig_dtype = standardize_dtype(x.dtype)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    # Use numpy fallback for correct NaN handling and method support
    # Cast int/bool types to float32 (numpy can't handle bool subtract)
    x_np = x.numpy()
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    if orig_dtype in int_types:
        x_np = x_np.astype("float32")
        result_dtype = "float32"
    else:
        result_dtype = orig_dtype
    q_np = q.numpy() if hasattr(q, "numpy") else np.asarray(q)
    result_np = np.nanquantile(
        x_np, q_np, axis=axis, method=method, keepdims=keepdims
    )
    result = paddle.to_tensor(result_np, dtype=to_paddle_dtype(result_dtype))
    return result


def nanpercentile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    orig_dtype = standardize_dtype(x.dtype)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    # Use numpy fallback for correct NaN handling and method support
    # Cast int/bool types to float32 (numpy can't handle bool subtract)
    x_np = x.numpy()
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    if orig_dtype in int_types:
        x_np = x_np.astype("float32")
        result_dtype = "float32"
    else:
        result_dtype = orig_dtype
    q_np = q.numpy() if hasattr(q, "numpy") else np.asarray(q)
    result_np = np.nanpercentile(
        x_np, q_np, axis=axis, method=method, keepdims=keepdims
    )
    result = paddle.to_tensor(result_np, dtype=to_paddle_dtype(result_dtype))
    return result


def ptp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return paddle.zeros_like(x)
    orig_dtype = x.dtype
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
        needs_cast = True
    result = paddle.max(x, axis=axis, keepdim=keepdims) - paddle.min(
        x, axis=axis, keepdim=keepdims
    )
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def logaddexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    target, x1, x2 = _cpu_binary_target(x1, x2)
    # logaddexp always returns float for int inputs
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
    }
    if target in int_types:
        target = "float32"
    result = paddle.logaddexp(x1, x2)
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def logaddexp2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    target, x1, x2 = _cpu_binary_target(x1, x2)
    # logaddexp2 always returns float for int inputs
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
    }
    if target in int_types:
        target = "float32"
    # Ensure float computation
    if standardize_dtype(x1.dtype) in int_types:
        x1 = x1.cast("float32")
        x2 = x2.cast("float32")
    # logaddexp2(x1, x2) = log2(2^x1 + 2^x2)
    # = log(e^(x1*ln2) + e^(x2*ln2)) / ln2
    # = logaddexp(x1*ln2, x2*ln2) / ln2
    ln2 = paddle.to_tensor(0.6931471805599453, dtype="float32")
    result = paddle.logaddexp(x1 * ln2, x2 * ln2) / ln2
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def logspace(start, stop, num, base=10.0, dtype=None, endpoint=True, axis=0):
    result = linspace(start, stop, num, endpoint=endpoint, dtype=dtype)
    orig_dtype = result.dtype
    if result.dtype in _CPU_UNSUPPORTED_DTYPES:
        result = result.cast("float32")
    base_t = paddle.to_tensor(base, dtype=result.dtype)
    result = paddle.pow(base_t, result).cast(orig_dtype)
    # linspace puts num dimension at the end (-1); move to requested axis
    if axis != -1 and axis != result.ndim - 1:
        # Move last axis to the requested position
        result = paddle.moveaxis(result, -1, axis)
    return result


def geomspace(start, stop, num, endpoint=True, dtype=None, axis=0):
    # paddle.linspace doesn't support array start/stop, use numpy fallback
    start_np = convert_to_tensor(start, "float32").numpy()
    stop_np = convert_to_tensor(stop, "float32").numpy()
    result_np = np.geomspace(
        start_np, stop_np, num, endpoint=endpoint, axis=axis
    )
    result = paddle.to_tensor(result_np, dtype=paddle.float32)
    if dtype is not None:
        target_dtype = to_paddle_dtype(dtype)
        result = result.cast(target_dtype)
    return result


def empty(shape, dtype="float32"):
    return zeros(shape, dtype=dtype)


def empty_like(x, dtype=None):
    return zeros_like(x, dtype=dtype)


def nextafter(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dt1 = standardize_dtype(x1.dtype)
    dt2 = standardize_dtype(x2.dtype)
    # Determine target dtype (nextafter always returns float)
    float_types = {"float16", "float32", "float64", "bfloat16"}
    if dt1 in float_types and dt2 in float_types:
        target = dt1 if dt1 == dt2 else "float32"
    elif dt1 in float_types:
        target = dt1
    elif dt2 in float_types:
        target = dt2
    else:
        target = "float32"
    # Cast to float for computation
    x1 = x1.cast("float32")
    x2 = x2.cast("float32")
    # Approximate nextafter using bit manipulation
    eps = np.finfo(np.float32).eps
    direction = paddle.sign(x2 - x1)
    result = x1 + direction * eps
    return result.cast(to_paddle_dtype(target))


def isreal(x):
    x = convert_to_tensor(x)
    return paddle.isreal(x)


def isin(elements, test_elements, assume_unique=False, invert=False):
    elements = convert_to_tensor(elements)
    test_elements = convert_to_tensor(test_elements).flatten()
    elements, test_elements = _promote_dtypes(elements, test_elements)
    # Broadcasting comparison
    result = paddle.any(elements.unsqueeze(-1) == test_elements, axis=-1)
    if invert:
        result = ~result
    return result


def nonzero(x):
    x = convert_to_tensor(x)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    indices = paddle.nonzero(x)
    # JAX returns int32 indices, paddle returns int64
    return tuple(idx.cast("int32") for idx in indices.T)


def array_split(x, indices_or_sections, axis=0):
    x = convert_to_tensor(x)
    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        size = x.shape[axis]
        base_size = size // n
        remainder = size % n
        splits = []
        start = 0
        for i in range(n):
            end = start + base_size + (1 if i < remainder else 0)
            splits.append(paddle.slice(x, [axis], [start], [end]))
            start = end
        return splits
    else:
        indices = list(indices_or_sections)
        splits = []
        start = 0
        for idx in indices:
            splits.append(paddle.slice(x, [axis], [start], [idx]))
            start = idx
        splits.append(paddle.slice(x, [axis], [start], [x.shape[axis]]))
        return splits


def dsplit(x, indices_or_sections):
    return split(x, indices_or_sections, axis=2)


def hsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    if x.ndim == 1:
        x = paddle.reshape(x, [1, -1])
        result = split(x, indices_or_sections, axis=1)
        return [r.squeeze(0) for r in result]
    return split(x, indices_or_sections, axis=1)


def vsplit(x, indices_or_sections):
    return split(x, indices_or_sections, axis=0)


def dstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    xs = _promote_dtypes_list(xs)
    # Ensure all tensors are at least 3D
    result = []
    for x in xs:
        if x.ndim == 0:
            x = x.reshape(1, 1, 1)
        elif x.ndim == 1:
            x = x.reshape(1, -1, 1)
        elif x.ndim == 2:
            x = x.unsqueeze(2)
        result.append(x)
    return paddle.concat(result, axis=2)


def hstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    xs = _promote_dtypes_list(xs)
    if len(xs[0].shape) == 1:
        return paddle.concat(xs, axis=0)
    return paddle.concat(xs, axis=1)


def vstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    xs = _promote_dtypes_list(xs)
    return paddle.concat(xs, axis=0)


def diagflat(x, k=0):
    x = convert_to_tensor(x).flatten()
    orig_dtype = x.dtype
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    result = paddle.diag(x, offset=k)
    return result.cast(orig_dtype)


def fliplr(x):
    return paddle.flip(convert_to_tensor(x), axis=[1])


def flipud(x):
    return paddle.flip(convert_to_tensor(x), axis=[0])


def rot90(x, k=1, axes=(0, 1)):
    x = convert_to_tensor(x)
    if x.ndim < 2:
        raise ValueError(f"rot90 requires at least 2 dimensions, got {x.ndim}")
    if axes[0] == axes[1] or axes[0] % x.ndim == axes[1] % x.ndim:
        raise ValueError("axes must be different")
    k = k % 4
    if k == 0:
        return x
    # Build the permutation that swaps axes[0] and axes[1]
    perm = list(range(x.ndim))
    perm[axes[0]], perm[axes[1]] = axes[1], axes[0]
    if k == 1:
        return paddle.transpose(paddle.flip(x, [axes[1]]), perm)
    elif k == 2:
        return paddle.flip(x, [axes[0], axes[1]])
    elif k == 3:
        return paddle.transpose(paddle.flip(x, [axes[0]]), perm)
    return x


def average(x, axis=None, weights=None, returned=False, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        if returned:
            return x.clone(), paddle.ones_like(x)
        return x.clone()
    if weights is None:
        result = paddle.mean(x, axis=axis, keepdim=keepdims)
    else:
        weights = convert_to_tensor(weights)
        target_dtype = _get_promoted_dtype(x, weights)
        # Division of ints produces float in JAX
        int_types = {
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
        }
        if target_dtype in int_types:
            target_dtype = "float32"
        x, weights = _promote_dtypes(x, weights)
        # CPU doesn't support int8/int16/uint8 for multiply
        x_dt = standardize_dtype(x.dtype)
        if x_dt in _CPU_UNSUPPORTED_INT:
            x = x.cast("int32")
            weights = weights.cast("int32")
        # Broadcast weights to match x shape for the reduction axis
        if axis is not None and weights.ndim < x.ndim:
            # Reshape weights to broadcast along the reduction axis
            broadcast_shape = [1] * x.ndim
            if isinstance(axis, (list, tuple)):
                ax = axis[0]
            else:
                ax = axis
            ax = ax + x.ndim if ax < 0 else ax
            broadcast_shape[ax] = weights.shape[0]
            weights = weights.reshape(broadcast_shape)
        result = paddle.sum(
            x * weights, axis=axis, keepdim=keepdims
        ) / paddle.sum(weights, axis=axis, keepdim=keepdims)
        result_dt = standardize_dtype(result.dtype)
        if result_dt != target_dtype:
            result = result.cast(to_paddle_dtype(target_dtype))
    if returned:
        weights_sum = paddle.sum(
            weights if weights is not None else paddle.ones_like(x),
            axis=axis,
            keepdim=keepdims,
        )
        return result, weights_sum
    return result


def cbrt(x):
    x = convert_to_tensor(x)
    orig_dtype = standardize_dtype(x.dtype)
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif orig_dtype in int_types:
        x = x.cast("float32")
    elif x.dtype not in (paddle.float32, paddle.float64):
        x = x.cast("float32")
    # paddle.pow with negative base and fractional exponent returns NaN,
    # so compute sign(x) * |x|^(1/3) to handle negative inputs correctly
    result = paddle.sign(x) * paddle.pow(paddle.abs(x), 1.0 / 3.0)
    # JAX returns float32 for int inputs (don't cast back to int)
    if needs_cast:
        result = result.cast(to_paddle_dtype(orig_dtype))
    return result


def exp2(x):
    x = convert_to_tensor(x)
    orig_dtype = standardize_dtype(x.dtype)
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif orig_dtype in int_types:
        x = x.cast("float32")
    elif x.dtype not in (paddle.float32, paddle.float64):
        x = x.cast("float32")
    result = paddle.pow(paddle.to_tensor(2.0, dtype=x.dtype), x)
    # JAX returns float32 for int inputs (don't cast back to int)
    if needs_cast:
        result = result.cast(to_paddle_dtype(orig_dtype))
    return result


def divide_no_nan(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    x1, x2 = _promote_dtypes(x1, x2)
    # CPU casting for paddle.where
    orig_dtype = x1.dtype
    needs_cast = False
    if x1.dtype in _CPU_UNSUPPORTED_DTYPES:
        x1 = x1.cast("float32")
        x2 = x2.cast("float32")
        needs_cast = True
    elif standardize_dtype(x1.dtype) in _CPU_UNSUPPORTED_INT:
        x1 = x1.cast("int32")
        x2 = x2.cast("int32")
        needs_cast = True
    safe_x2 = paddle.where(x2 == 0, paddle.ones_like(x2), x2)
    result = paddle.where(x2 == 0, paddle.zeros_like(x1), x1 / safe_x2)
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def slogdet(x):
    x = convert_to_tensor(x)
    sign, logabsdet = paddle.linalg.slogdet(x)
    return sign, logabsdet


def argpartition(x, kth, axis=-1):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    # Paddle CPU does not support argsort for many dtypes
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    sorted_indices = paddle.argsort(x, axis=axis)
    # JAX returns int32 for argsort/argpartition
    return sorted_indices.cast("int32")


def gcd(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    target = _get_promoted_dtype(x1, x2)
    x1_dt = standardize_dtype(x1.dtype)
    x2_dt = standardize_dtype(x2.dtype)
    if x1_dt != target:
        x1 = x1.cast(to_paddle_dtype(target))
    if x2_dt != target:
        x2 = x2.cast(to_paddle_dtype(target))
    # CPU doesn't support gcd for int8/int16/uint8/bool
    if target in _CPU_UNSUPPORTED_INT:
        x1 = x1.cast("int32")
        x2 = x2.cast("int32")
    result = paddle.gcd(x1, x2)
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def lcm(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    target = _get_promoted_dtype(x1, x2)
    x1_dt = standardize_dtype(x1.dtype)
    x2_dt = standardize_dtype(x2.dtype)
    if x1_dt != target:
        x1 = x1.cast(to_paddle_dtype(target))
    if x2_dt != target:
        x2 = x2.cast(to_paddle_dtype(target))
    # CPU doesn't support lcm for int8/int16/uint8/bool
    if target in _CPU_UNSUPPORTED_INT:
        x1 = x1.cast("int32")
        x2 = x2.cast("int32")
    result = paddle.lcm(x1, x2)
    result_dt = standardize_dtype(result.dtype)
    if result_dt != target:
        result = result.cast(to_paddle_dtype(target))
    return result


def max(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    orig_dtype = x.dtype
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    if standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
        needs_cast = True
    # Handle empty axis (size 0) - use numpy fallback
    if axis is not None and x.shape[axis if axis >= 0 else axis + x.ndim] == 0:
        result_np = np.max(
            x.numpy(), axis=axis, keepdims=keepdims, initial=initial
        )
        result = paddle.to_tensor(result_np, dtype=x.dtype)
        if needs_cast:
            result = result.cast(orig_dtype)
        return result
    result = paddle.max(x, axis=axis, keepdim=keepdims)
    if initial is not None:
        result = paddle.maximum(
            result, convert_to_tensor(initial, dtype=result.dtype)
        )
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def min(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    orig_dtype = x.dtype
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    if standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
        needs_cast = True
    # Handle empty axis (size 0) - use numpy fallback
    if axis is not None and x.shape[axis if axis >= 0 else axis + x.ndim] == 0:
        result_np = np.min(
            x.numpy(), axis=axis, keepdims=keepdims, initial=initial
        )
        result = paddle.to_tensor(result_np, dtype=x.dtype)
        if needs_cast:
            result = result.cast(orig_dtype)
        return result
    result = paddle.min(x, axis=axis, keepdim=keepdims)
    if initial is not None:
        result = paddle.minimum(
            result, convert_to_tensor(initial, dtype=result.dtype)
        )
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def amin(x, axis=None, keepdims=False, initial=None):
    return min(x, axis=axis, keepdims=keepdims, initial=initial)


def amax(x, axis=None, keepdims=False, initial=None):
    return max(x, axis=axis, keepdims=keepdims, initial=initial)


def var(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return paddle.zeros_like(x, dtype="float32")
    orig_dtype = standardize_dtype(x.dtype)
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    elif orig_dtype in int_types or orig_dtype in _CPU_UNSUPPORTED_INT:
        x = x.cast("float32")
    result = paddle.var(x, axis=axis, unbiased=False, keepdim=keepdims)
    if orig_dtype in int_types and standardize_dtype(result.dtype) != "float32":
        result = result.cast("float32")
    elif needs_cast:
        result = result.cast(to_paddle_dtype(orig_dtype))
    return result


def tanh(x):
    return _unary_math_op(paddle.tanh, x)


def fabs(x):
    return _unary_math_op(paddle.abs, x)


def diag(x, k=0):
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    needs_cast = False
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
        needs_cast = True
    if standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
        needs_cast = True
    result = paddle.diag(x, offset=k)
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


def trapezoid(y, x=None, dx=1.0, axis=-1):
    y = convert_to_tensor(y)
    orig_dtype = standardize_dtype(y.dtype)
    int_types = {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    needs_cast = False
    if y.dtype in _CPU_UNSUPPORTED_DTYPES:
        y = y.cast("float32")
        needs_cast = True
    elif orig_dtype in int_types:
        y = y.cast("float32")
    ndim = y.ndim
    axis = axis + ndim if axis < 0 else axis
    slice_left = [builtins.slice(None)] * ndim
    slice_left[axis] = builtins.slice(None, -1)
    slice_right = [builtins.slice(None)] * ndim
    slice_right[axis] = builtins.slice(1, None)
    if x is not None:
        x = convert_to_tensor(x)
        if x.dtype in _CPU_UNSUPPORTED_DTYPES:
            x = x.cast("float32")
        elif standardize_dtype(x.dtype) in _CPU_UNSUPPORTED_INT:
            x = x.cast("float32")
        dx_tensor = x[tuple(slice_right)] - x[tuple(slice_left)]
        avg = (y[tuple(slice_right)] + y[tuple(slice_left)]) / 2.0
        result = paddle.sum(avg * dx_tensor, axis=axis)
        if needs_cast:
            result = result.cast(to_paddle_dtype(orig_dtype))
        return result
    avg = (y[tuple(slice_right)] + y[tuple(slice_left)]) / 2.0
    result = paddle.sum(avg * dx, axis=axis)
    if needs_cast:
        result = result.cast(to_paddle_dtype(orig_dtype))
    return result


def bartlett(M):
    if M < 1:
        return paddle.zeros([0])
    if M == 1:
        return paddle.ones([1])
    n = paddle.arange(M, dtype="float32")
    return paddle.where(
        n <= (M - 1) / 2.0, 2 * n / (M - 1), 2 - 2 * n / (M - 1)
    )


def blackman(M):
    if hasattr(M, "numpy"):
        M = int(M.numpy())
    elif hasattr(M, "item"):
        M = int(M.item())
    if M < 1:
        return paddle.zeros([0])
    if M == 1:
        return paddle.ones([1], dtype="float32")
    n = paddle.arange(M, dtype="float32")
    return (
        0.42
        - 0.5 * paddle.cos(2 * np.pi * n / (M - 1))
        + 0.08 * paddle.cos(4 * np.pi * n / (M - 1))
    )


def hamming(M):
    if hasattr(M, "numpy"):
        M = int(M.numpy())
    elif hasattr(M, "item"):
        M = int(M.item())
    if M < 1:
        return paddle.zeros([0])
    if M == 1:
        return paddle.ones([1], dtype="float32")
    n = paddle.arange(M, dtype="float32")
    return 0.54 - 0.46 * paddle.cos(2 * np.pi * n / (M - 1))


def hanning(M):
    if hasattr(M, "numpy"):
        M = int(M.numpy())
    elif hasattr(M, "item"):
        M = int(M.item())
    if M < 1:
        return paddle.zeros([0])
    if M == 1:
        return paddle.ones([1], dtype="float32")
    n = paddle.arange(M, dtype="float32")
    return 0.5 - 0.5 * paddle.cos(2 * np.pi * n / (M - 1))


def kaiser(M, beta):
    if hasattr(M, "numpy"):
        M = int(M.numpy())
    elif hasattr(M, "item"):
        M = int(M.item())
    if M < 1:
        return paddle.zeros([0])
    if M == 1:
        return paddle.ones([1], dtype="float32")
    n = paddle.arange(M, dtype="float32")
    alpha = (M - 1) / 2.0
    return paddle.i0(
        beta * paddle.sqrt(1 - ((n - alpha) / alpha) ** 2)
    ) / paddle.i0(paddle.to_tensor(beta, dtype="float32"))


def vander(x, N=None, increasing=False):
    x = convert_to_tensor(x)
    orig_dtype = standardize_dtype(x.dtype)
    # JAX doesn't support bool for vander, cast to int32
    if orig_dtype == "bool":
        x = x.cast("int32")
    elif x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif orig_dtype in _CPU_UNSUPPORTED_INT:
        x = x.cast("int32")
    if N is None:
        N = x.shape[0]
    # Use int32 for arange (CPU doesn't support int8/int16/uint8)
    arange_dtype = (
        x.dtype if x.dtype not in _CPU_UNSUPPORTED_DTYPES else paddle.int32
    )
    if not increasing:
        powers = paddle.arange(N - 1, -1, -1, dtype=arange_dtype)
    else:
        powers = paddle.arange(0, N, dtype=arange_dtype)
    result = x.unsqueeze(1) ** powers.unsqueeze(0)
    # Cast back to original dtype if needed
    result_dt = standardize_dtype(result.dtype)
    if result_dt != orig_dtype:
        result = result.cast(to_paddle_dtype(orig_dtype))
    return result
