import numpy as np

from keras.src import tree
from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.common.backend_utils import standardize_axis_for_numpy
from keras.src.backend.numpy.core import convert_to_tensor


def rot90(array, k=1, axes=(0, 1)):
    """Rotate an array by 90 degrees in the specified plane."""
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
    return np.rot90(array, k=k, axes=axes)


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
    return np.add(x1, x2)


def einsum(subscripts, *operands, **kwargs):
    operands = tree.map_structure(convert_to_tensor, operands)
    dtypes_to_resolve = list(set(standardize_dtype(x.dtype) for x in operands))
    # When operands are of int8, we cast the result to int32 to align with
    # the behavior of jax.
    if len(dtypes_to_resolve) == 1 and dtypes_to_resolve[0] == "int8":
        compute_dtype = "int32"  # prevent overflow
        result_dtype = "int32"
    else:
        result_dtype = dtypes.result_type(*dtypes_to_resolve)
        compute_dtype = result_dtype
        # TODO: np.einsum doesn't support bfloat16
        if compute_dtype == "bfloat16":
            compute_dtype = "float32"
    operands = tree.map_structure(lambda x: x.astype(compute_dtype), operands)
    return np.einsum(subscripts, *operands, **kwargs).astype(result_dtype)


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
    return np.subtract(x1, x2)


def matmul(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    # When both x1 and x2 are of int8, we cast the outputs to int32 to align
    # with jax
    x1_dtype = standardize_dtype(x1.dtype)
    x2_dtype = standardize_dtype(x2.dtype)
    if x1_dtype == "int8" and x2_dtype == "int8":
        dtype = "int32"
    else:
        dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = x1.astype(dtype)
    x2 = x2.astype(dtype)
    return np.matmul(x1, x2).astype(dtype)


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
    return np.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        result_dtype = dtypes.result_type(x.dtype, "float32")
    else:
        result_dtype = ori_dtype
    return np.mean(x, axis=axis, keepdims=keepdims).astype(result_dtype)


def max(x, axis=None, keepdims=False, initial=None):
    axis = standardize_axis_for_numpy(axis)
    return np.max(x, axis=axis, keepdims=keepdims, initial=initial)


def ones(shape, dtype=None):
    dtype = dtype or config.floatx()
    return np.ones(shape, dtype=dtype)


def zeros(shape, dtype=None):
    dtype = dtype or config.floatx()
    return np.zeros(shape, dtype=dtype)


def absolute(x):
    return np.absolute(x)


def abs(x):
    return absolute(x)


def all(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    return np.all(x, axis=axis, keepdims=keepdims)


def angle(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(dtype)
    return np.angle(x)


def any(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    return np.any(x, axis=axis, keepdims=keepdims)


def amax(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    return np.amax(x, axis=axis, keepdims=keepdims)


def amin(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    return np.amin(x, axis=axis, keepdims=keepdims)


def append(x1, x2, axis=None):
    axis = standardize_axis_for_numpy(axis)
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = x1.astype(dtype)
    x2 = x2.astype(dtype)
    return np.append(x1, x2, axis=axis)


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
    return np.arange(start, stop, step=step, dtype=dtype)


def arccos(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(dtype)
    return np.arccos(x)


def arccosh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(dtype)
    return np.arccosh(x)


def arcsin(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(dtype)
    return np.arcsin(x)


def arcsinh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(dtype)
    return np.arcsinh(x)


def arctan(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(dtype)
    return np.arctan(x)


def arctan2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = x1.astype(dtype)
    x2 = x2.astype(dtype)
    return np.arctan2(x1, x2)


def arctanh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(dtype)
    return np.arctanh(x)


def argmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = standardize_axis_for_numpy(axis)
    dtype = standardize_dtype(x.dtype)
    if "float" not in dtype or x.ndim == 0:
        return np.argmax(x, axis=axis, keepdims=keepdims).astype("int32")

    dtype = dtypes.result_type(dtype, "float32")
    x = x.astype(dtype)
    is_negative_zero = (x == 0.0) & np.signbit(x)
    x = np.where(is_negative_zero, -np.finfo(x.dtype).tiny, x)
    return np.argmax(x, axis=axis, keepdims=keepdims).astype("int32")


def argmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = standardize_axis_for_numpy(axis)
    dtype = standardize_dtype(x.dtype)
    if "float" not in dtype or x.ndim == 0:
        return np.argmin(x, axis=axis, keepdims=keepdims).astype("int32")

    dtype = dtypes.result_type(dtype, "float32")
    x = x.astype(dtype)
    is_negative_zero = (x == 0.0) & np.signbit(x)
    x = np.where(is_negative_zero, -np.finfo(x.dtype).tiny, x)
    return np.argmin(x, axis=axis, keepdims=keepdims).astype("int32")


def argsort(x, axis=-1):
    axis = standardize_axis_for_numpy(axis)
    return np.argsort(x, axis=axis).astype("int32")


def array(x, dtype=None):
    return convert_to_tensor(x, dtype=dtype)


def view(x, dtype=None):
    x = convert_to_tensor(x)
    return x.view(dtype=dtype)


def average(x, axis=None, weights=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtypes_to_resolve = [x.dtype, float]
    if weights is not None:
        weights = convert_to_tensor(weights)
        dtypes_to_resolve.append(weights.dtype)
    dtype = dtypes.result_type(*dtypes_to_resolve)
    x = x.astype(dtype)
    if weights is not None:
        weights = weights.astype(dtype)
    return np.average(x, weights=weights, axis=axis)


def bartlett(x):
    x = convert_to_tensor(x)
    return np.bartlett(x).astype(config.floatx())


def hamming(x):
    x = convert_to_tensor(x)
    return np.hamming(x).astype(config.floatx())


def hanning(x):
    x = convert_to_tensor(x)
    return np.hanning(x).astype(config.floatx())


def heaviside(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype in ["int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        dtype = config.floatx()
    elif dtype in ["int64"]:
        dtype = "float64"

    return np.heaviside(x1, x2).astype(dtype)


def kaiser(x, beta):
    x = convert_to_tensor(x)
    return np.kaiser(x, beta).astype(config.floatx())


def bincount(x, weights=None, minlength=0, sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with numpy backend")
    x = convert_to_tensor(x)
    dtypes_to_resolve = [x.dtype]
    if weights is not None:
        weights = convert_to_tensor(weights)
        dtypes_to_resolve.append(weights.dtype)
        dtype = dtypes.result_type(*dtypes_to_resolve)
    else:
        dtype = "int32"
    if len(x.shape) == 2:
        if weights is None:

            def bincount_fn(arr):
                return np.bincount(arr, minlength=minlength)

            bincounts = list(map(bincount_fn, x))
        else:

            def bincount_fn(arr_w):
                return np.bincount(
                    arr_w[0], weights=arr_w[1], minlength=minlength
                )

            bincounts = list(map(bincount_fn, zip(x, weights)))

        return np.stack(bincounts).astype(dtype)
    return np.bincount(x, weights, minlength).astype(dtype)


def bitwise_and(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    x = x.astype(dtype)
    y = y.astype(dtype)
    return np.bitwise_and(x, y)


def bitwise_invert(x):
    x = convert_to_tensor(x)
    return np.bitwise_not(x)


def bitwise_not(x):
    return bitwise_invert(x)


def bitwise_or(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    x = x.astype(dtype)
    y = y.astype(dtype)
    return np.bitwise_or(x, y)


def bitwise_xor(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    x = x.astype(dtype)
    y = y.astype(dtype)
    return np.bitwise_xor(x, y)


def bitwise_left_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, int):
        y = convert_to_tensor(y)
        dtype = dtypes.result_type(x.dtype, y.dtype)
        x = x.astype(dtype)
        y = y.astype(dtype)
    return np.left_shift(x, y)


def left_shift(x, y):
    return bitwise_left_shift(x, y)


def bitwise_right_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, int):
        y = convert_to_tensor(y)
        dtype = dtypes.result_type(x.dtype, y.dtype)
        x = x.astype(dtype)
        y = y.astype(dtype)
    return np.right_shift(x, y)


def right_shift(x, y):
    return bitwise_right_shift(x, y)


def blackman(x):
    x = convert_to_tensor(x)
    return np.blackman(x).astype(config.floatx())


def broadcast_to(x, shape):
    return np.broadcast_to(x, shape)


def cbrt(x):
    x = convert_to_tensor(x)

    dtype = standardize_dtype(x.dtype)
    if dtype in ["bool", "int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        dtype = config.floatx()
    elif dtype == "int64":
        dtype = "float64"

    return np.cbrt(x).astype(dtype)


def ceil(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(dtype)
    return np.ceil(x)


def clip(x, x_min, x_max):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype == "bool":
        dtype = "int32"
    return np.clip(x, x_min, x_max).astype(dtype)


def concatenate(xs, axis=0):
    axis = standardize_axis_for_numpy(axis)
    dtype_set = set([getattr(x, "dtype", type(x)) for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = tree.map_structure(
            lambda x: convert_to_tensor(x).astype(dtype), xs
        )
    return np.concatenate(xs, axis=axis)


def conjugate(x):
    return np.conjugate(x)


def conj(x):
    return conjugate(x)


def copy(x):
    return np.copy(x)


def cos(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(dtype)
    return np.cos(x)


def cosh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(dtype)
    return np.cosh(x)


def count_nonzero(x, axis=None):
    axis = standardize_axis_for_numpy(axis)
    # np.count_nonzero will return python int when axis=None, so we need
    # to convert_to_tensor
    return convert_to_tensor(np.count_nonzero(x, axis=axis)).astype("int32")


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    axis = standardize_axis_for_numpy(axis)
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = x1.astype(dtype)
    x2 = x2.astype(dtype)
    return np.cross(
        x1,
        x2,
        axisa=axisa,
        axisb=axisb,
        axisc=axisc,
        axis=axis,
    )


def cumprod(x, axis=None, dtype=None):
    axis = standardize_axis_for_numpy(axis)
    dtype = dtypes.result_type(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    return np.cumprod(x, axis=axis, dtype=dtype)


def cumsum(x, axis=None, dtype=None):
    axis = standardize_axis_for_numpy(axis)
    dtype = dtypes.result_type(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    return np.cumsum(x, axis=axis, dtype=dtype)


def deg2rad(x):
    x = convert_to_tensor(x)

    if x.dtype in ["int64", "float64"]:
        dtype = "float64"
    elif x.dtype in ["bfloat16", "float16"]:
        dtype = x.dtype
    else:
        dtype = config.floatx()

    return np.deg2rad(x).astype(dtype)


def diag(x, k=0):
    return np.diag(x, k=k)


def diagflat(x, k=0):
    return np.diagflat(x, k=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    axis1 = standardize_axis_for_numpy(axis1)
    axis2 = standardize_axis_for_numpy(axis2)
    return np.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


def diff(a, n=1, axis=-1):
    return np.diff(a, n=n, axis=axis)


def digitize(x, bins):
    return np.digitize(x, bins).astype(np.int32)


def dot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = x1.astype(dtype)
    x2 = x2.astype(dtype)
    return np.dot(x1, x2)


def empty(shape, dtype=None):
    dtype = dtype or config.floatx()
    return np.empty(shape, dtype=dtype)


def empty_like(x, dtype=None):
    return np.empty_like(x, dtype=dtype)


def equal(x1, x2):
    return np.equal(x1, x2)


def exp(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = x.astype(config.floatx())
    return np.exp(x)


def exp2(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = x.astype(config.floatx())
    return np.exp2(x)


def expand_dims(x, axis):
    axis = standardize_axis_for_numpy(axis)
    return np.expand_dims(x, axis)


def expm1(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = x.astype(config.floatx())
    return np.expm1(x)


def flip(x, axis=None):
    axis = standardize_axis_for_numpy(axis)
    return np.flip(x, axis=axis)


def floor(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = x.astype(dtype)
    return np.floor(x)


def full(shape, fill_value, dtype=None):
    dtype = dtype or config.floatx()
    return np.full(shape, fill_value, dtype=dtype)


def full_like(x, fill_value, dtype=None):
    return np.full_like(x, fill_value, dtype=dtype)


def gcd(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    return np.gcd(x1, x2).astype(dtype)


def greater(x1, x2):
    return np.greater(x1, x2)


def greater_equal(x1, x2):
    return np.greater_equal(x1, x2)


def hstack(xs):
    dtype_set = set([getattr(x, "dtype", type(x)) for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = tree.map_structure(
            lambda x: convert_to_tensor(x).astype(dtype), xs
        )
    return np.hstack(xs)


def hypot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype in ["int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        dtype = config.floatx()
    elif dtype in ["int64"]:
        dtype = "float64"

    return np.hypot(x1, x2).astype(dtype)


def identity(n, dtype=None):
    dtype = dtype or config.floatx()
    return np.identity(n, dtype=dtype)


def imag(x):
    return np.imag(x)


def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    return np.isclose(x1, x2, rtol, atol, equal_nan)


def isfinite(x):
    return np.isfinite(x)


def isin(x1, x2, assume_unique=False, invert=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return np.isin(x1, x2, assume_unique=assume_unique, invert=invert)


def isinf(x):
    return np.isinf(x)


def isnan(x):
    return np.isnan(x)


def isneginf(x):
    x = convert_to_tensor(x)
    return np.isneginf(x)


def isposinf(x):
    x = convert_to_tensor(x)
    return np.isposinf(x)


def isreal(x):
    x = convert_to_tensor(x)
    return np.isreal(x)


def kron(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    return np.kron(x1, x2).astype(dtype)


def lcm(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    return np.lcm(x1, x2).astype(dtype)


def ldexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)

    if standardize_dtype(x2.dtype) not in dtypes.INT_TYPES:
        raise TypeError(
            f"ldexp exponent must be an integer type. "
            f"Received: x2 dtype={x2.dtype}"
        )
    return np.ldexp(x1, x2).astype(dtype)


def less(x1, x2):
    return np.less(x1, x2)


def less_equal(x1, x2):
    return np.less_equal(x1, x2)


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    axis = standardize_axis_for_numpy(axis)
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(stop, "dtype", type(stop)),
            float,
        ]
        dtype = dtypes.result_type(*dtypes_to_resolve)
    return np.linspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        retstep=retstep,
        dtype=dtype,
        axis=axis,
    )


def log(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    return np.log(x, dtype=dtype)


def log10(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    return np.log10(x, dtype=dtype)


def log1p(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    return np.log1p(x, dtype=dtype)


def log2(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    return np.log2(x, dtype=dtype)


def logaddexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = x1.astype(dtype)
    x2 = x2.astype(dtype)
    return np.logaddexp(x1, x2)


def logaddexp2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    return np.logaddexp2(x1, x2).astype(dtype)


def logical_and(x1, x2):
    return np.logical_and(x1, x2)


def logical_not(x):
    return np.logical_not(x)


def logical_or(x1, x2):
    return np.logical_or(x1, x2)


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(stop, "dtype", type(stop)),
            float,
        ]
        dtype = dtypes.result_type(*dtypes_to_resolve)
    return np.logspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        base=base,
        dtype=dtype,
        axis=axis,
    )


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
    return np.maximum(x1, x2)


def median(x, axis=None, keepdims=False):
    dtype = dtypes.result_type(x.dtype, float)
    return np.median(x, axis=axis, keepdims=keepdims).astype(dtype)


def meshgrid(*x, indexing="xy"):
    return np.meshgrid(*x, indexing=indexing)


def min(x, axis=None, keepdims=False, initial=None):
    axis = standardize_axis_for_numpy(axis)
    return np.min(x, axis=axis, keepdims=keepdims, initial=initial)


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
    return np.minimum(x1, x2)


def mod(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype == "bool":
        dtype = "int32"
    x1 = x1.astype(dtype)
    x2 = x2.astype(dtype)
    return np.mod(x1, x2)


def moveaxis(x, source, destination):
    return np.moveaxis(x, source=source, destination=destination)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    return np.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def ndim(x):
    return np.ndim(x)


def nonzero(x):
    return tuple(indices.astype("int32") for indices in np.nonzero(x))


def not_equal(x1, x2):
    return np.not_equal(x1, x2)


def zeros_like(x, dtype=None):
    return np.zeros_like(x, dtype=dtype)


def ones_like(x, dtype=None):
    return np.ones_like(x, dtype=dtype)


def outer(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = x1.astype(dtype)
    x2 = x2.astype(dtype)
    return np.outer(x1, x2)


def pad(x, pad_width, mode="constant", constant_values=None):
    kwargs = {}
    if constant_values is not None:
        if mode != "constant":
            raise ValueError(
                "Argument `constant_values` can only be "
                "provided when `mode == 'constant'`. "
                f"Received: mode={mode}"
            )
        kwargs["constant_values"] = constant_values
    return np.pad(x, pad_width, mode=mode, **kwargs)


def prod(x, axis=None, keepdims=False, dtype=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    if dtype is None:
        dtype = dtypes.result_type(x.dtype)
        if dtype in ("bool", "int8", "int16"):
            dtype = "int32"
        elif dtype in ("uint8", "uint16"):
            dtype = "uint32"
    return np.prod(x, axis=axis, keepdims=keepdims, dtype=dtype)


def quantile(x, q, axis=None, method="linear", keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)

    ori_dtype = standardize_dtype(x.dtype)
    # np.quantile doesn't support bool
    if ori_dtype == "bool":
        x = x.astype(config.floatx())
    if ori_dtype == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    return np.quantile(
        x, q, axis=axis, method=method, keepdims=keepdims
    ).astype(dtype)


def ravel(x):
    return np.ravel(x)


def unravel_index(indices, shape):
    dtype = dtypes.result_type(indices.dtype)
    return tuple(
        indices.astype(dtype) for indices in np.unravel_index(indices, shape)
    )


def real(x):
    return np.real(x)


def reciprocal(x):
    return np.reciprocal(x)


def repeat(x, repeats, axis=None):
    return np.repeat(x, repeats, axis=axis)


def reshape(x, newshape):
    return np.reshape(x, newshape)


def roll(x, shift, axis=None):
    return np.roll(x, shift, axis=axis)


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
    return np.searchsorted(sorted_sequence, values, side=side).astype(out_type)


def sign(x):
    return np.sign(x)


def signbit(x):
    return np.signbit(x)


def sin(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(dtype)
    return np.sin(x)


def sinh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(dtype)
    return np.sinh(x)


def size(x):
    return np.size(x)


def sort(x, axis=-1):
    axis = standardize_axis_for_numpy(axis)
    return np.sort(x, axis=axis)


def split(x, indices_or_sections, axis=0):
    axis = standardize_axis_for_numpy(axis)
    return np.split(x, indices_or_sections, axis=axis)


def array_split(x, indices_or_sections, axis=0):
    axis = standardize_axis_for_numpy(axis)
    return np.array_split(x, indices_or_sections, axis=axis)


def stack(x, axis=0):
    axis = standardize_axis_for_numpy(axis)
    dtype_set = set([getattr(a, "dtype", type(a)) for a in x])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        x = tree.map_structure(lambda a: convert_to_tensor(a).astype(dtype), x)
    return np.stack(x, axis=axis)


def std(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = x.astype(config.floatx())
    return np.std(x, axis=axis, keepdims=keepdims)


def swapaxes(x, axis1, axis2):
    return np.swapaxes(x, axis1=axis1, axis2=axis2)


def take(x, indices, axis=None):
    axis = standardize_axis_for_numpy(axis)
    return np.take(x, indices, axis=axis)


def take_along_axis(x, indices, axis=None):
    axis = standardize_axis_for_numpy(axis)
    return np.take_along_axis(x, indices, axis=axis)


def tan(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(dtype)
    return np.tan(x)


def tanh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(dtype)
    return np.tanh(x)


def tensordot(x1, x2, axes=2):
    axes = tuple(axes) if isinstance(axes, list) else axes
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = x1.astype(dtype)
    x2 = x2.astype(dtype)
    return np.tensordot(x1, x2, axes=axes)


def round(x, decimals=0):
    return np.round(x, decimals=decimals)


def tile(x, repeats):
    return np.tile(x, repeats)


def trace(x, offset=0, axis1=0, axis2=1):
    axis1 = standardize_axis_for_numpy(axis1)
    axis2 = standardize_axis_for_numpy(axis2)
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype in ("bool", "int8", "int16"):
        dtype = "int32"
    elif dtype in ("uint8", "uint16"):
        dtype = "uint32"
    return np.trace(x, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)


def tri(N, M=None, k=0, dtype=None):
    dtype = dtype or config.floatx()
    return np.tri(N, M=M, k=k, dtype=dtype)


def tril(x, k=0):
    return np.tril(x, k=k)


def triu(x, k=0):
    return np.triu(x, k=k)


def trunc(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or "bool" == dtype:
        return x
    return np.trunc(x)


def vdot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = x1.astype(dtype)
    x2 = x2.astype(dtype)
    return np.vdot(x1, x2)


def inner(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = x1.astype(dtype)
    x2 = x2.astype(dtype)
    return np.inner(x1, x2)


def vstack(xs):
    dtype_set = set([getattr(x, "dtype", type(x)) for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = tree.map_structure(
            lambda x: convert_to_tensor(x).astype(dtype), xs
        )
    return np.vstack(xs)


def vectorize(pyfunc, *, excluded=None, signature=None):
    return np.vectorize(pyfunc, excluded=excluded, signature=signature)


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
        return np.where(condition, x1, x2)
    else:
        return np.where(condition)


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
    return np.divide(x1, x2)


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
    # No need for the double-where trick since we don't calculate gradients in
    # numpy backend.
    return np.where(x2 == 0, np.array(0, dtype=dtype), np.divide(x1, x2))


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
    return np.power(x1, x2)


def negative(x):
    return np.negative(x)


def square(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "bool":
        x = x.astype("int32")
    return np.square(x)


def sqrt(x):
    x = convert_to_tensor(x)
    # upcast to float64 for int64 which matches JAX's behavior
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    return np.sqrt(x, dtype=dtype)


def squeeze(x, axis=None):
    axis = standardize_axis_for_numpy(axis)
    return np.squeeze(x, axis=axis)


def transpose(x, axes=None):
    axes = tuple(axes) if isinstance(axes, list) else axes
    return np.transpose(x, axes=axes)


def trapezoid(y, x=None, dx=1.0, axis=-1):
    y = convert_to_tensor(y)
    result_dtype = dtypes.result_type(y.dtype, float)
    if x is not None:
        x = convert_to_tensor(x)
    dx = convert_to_tensor(dx)
    return np.trapezoid(y, x, dx=dx, axis=axis).astype(result_dtype)


def vander(x, N=None, increasing=False):
    x = convert_to_tensor(x)
    result_dtype = dtypes.result_type(x.dtype)
    compute_dtype = dtypes.result_type(x.dtype, config.floatx())
    x = x.astype(compute_dtype)
    return np.vander(x, N=N, increasing=increasing).astype(result_dtype)


def var(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    result_dtype = dtypes.result_type(x.dtype, float)
    return np.var(x, axis=axis, keepdims=keepdims, dtype=compute_dtype).astype(
        result_dtype
    )


def sum(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    dtype = standardize_dtype(x.dtype)
    # follow jax's rule
    if dtype in ("bool", "int8", "int16"):
        dtype = "int32"
    elif dtype in ("uint8", "uint16"):
        dtype = "uint32"
    return np.sum(x, axis=axis, keepdims=keepdims).astype(dtype)


def eye(N, M=None, k=0, dtype=None):
    dtype = dtype or config.floatx()
    return np.eye(N, M=M, k=k, dtype=dtype)


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
    return np.floor_divide(x1, x2)


def logical_xor(x1, x2):
    return np.logical_xor(x1, x2)


def corrcoef(x):
    if x.dtype in ["int64", "float64"]:
        dtype = "float64"
    elif x.dtype in ["bfloat16", "float16"]:
        dtype = x.dtype
    else:
        dtype = config.floatx()

    x = convert_to_tensor(x)

    return np.corrcoef(x).astype(dtype)


def correlate(x1, x2, mode="valid"):
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    if dtype == "int64":
        dtype = "float64"
    elif dtype not in ["bfloat16", "float16", "float64"]:
        dtype = "float32"

    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return np.correlate(x1, x2, mode)


def select(condlist, choicelist, default=0):
    return np.select(condlist, choicelist, default=default)


def slogdet(x):
    return tuple(np.linalg.slogdet(x))


def argpartition(x, kth, axis=-1):
    return np.argpartition(x, kth, axis).astype("int32")


def histogram(x, bins=10, range=None):
    return np.histogram(x, bins=bins, range=range)
