import jax.numpy as jnp

from keras.backend import config
from keras.backend.common import dtypes
from keras.backend.common.variables import standardize_dtype
from keras.backend.jax.core import cast
from keras.backend.jax.core import convert_to_tensor


def add(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.add(x1, x2)


def bincount(x, weights=None, minlength=0):
    if len(x.shape) == 2:
        if weights is None:

            def bincount_fn(arr):
                return jnp.bincount(arr, minlength=minlength)

            bincounts = list(map(bincount_fn, x))
        else:

            def bincount_fn(arr_w):
                return jnp.bincount(
                    arr_w[0], weights=arr_w[1], minlength=minlength
                )

            bincounts = list(map(bincount_fn, zip(x, weights)))

        return jnp.stack(bincounts)
    return jnp.bincount(x, weights=weights, minlength=minlength)


def einsum(subscripts, *operands, **kwargs):
    operands = [convert_to_tensor(x) for x in operands]
    return jnp.einsum(subscripts, *operands, **kwargs)


def subtract(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.subtract(x1, x2)


def matmul(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.matmul(x1, x2)


def multiply(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    # `jnp.mean` does not handle low precision (e.g., float16) overflow
    # correctly, so we compute with float32 and cast back to the original type.
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    if "int" in ori_dtype or ori_dtype == "bool":
        result_dtype = compute_dtype
    else:
        result_dtype = ori_dtype
    outputs = jnp.mean(x, axis=axis, keepdims=keepdims, dtype=compute_dtype)
    return cast(outputs, result_dtype)


def max(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    return jnp.max(x, axis=axis, keepdims=keepdims, initial=initial)


def ones(shape, dtype=None):
    dtype = dtype or config.floatx()
    return jnp.ones(shape, dtype=dtype)


def zeros(shape, dtype=None):
    dtype = dtype or config.floatx()
    return jnp.zeros(shape, dtype=dtype)


def absolute(x):
    return jnp.absolute(x)


def abs(x):
    return absolute(x)


def all(x, axis=None, keepdims=False):
    return jnp.all(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    return jnp.any(x, axis=axis, keepdims=keepdims)


def amax(x, axis=None, keepdims=False):
    return jnp.amax(x, axis=axis, keepdims=keepdims)


def amin(x, axis=None, keepdims=False):
    return jnp.amin(x, axis=axis, keepdims=keepdims)


def append(x1, x2, axis=None):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.append(x1, x2, axis=axis)


def arange(start, stop=None, step=1, dtype=None):
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(step, "dtype", type(step)),
        ]
        if stop is not None:
            dtypes_to_resolve.append(getattr(stop, "dtype", type(stop)))
        dtype = dtypes.result_type(*dtypes_to_resolve)
    dtype = standardize_dtype(dtype)
    return jnp.arange(start, stop, step=step, dtype=dtype)


def arccos(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.arccos(x)


def arccosh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.arccosh(x)


def arcsin(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.arcsin(x)


def arcsinh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.arcsinh(x)


def arctan(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.arctan(x)


def arctan2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = cast(x1, dtype)
    x2 = cast(x2, dtype)
    return jnp.arctan2(x1, x2)


def arctanh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.arctanh(x)


def argmax(x, axis=None):
    return jnp.argmax(x, axis=axis)


def argmin(x, axis=None):
    return jnp.argmin(x, axis=axis)


def argsort(x, axis=-1):
    x = convert_to_tensor(x)
    if x.ndim == 0:
        return jnp.argsort(x, axis=None)
    return jnp.argsort(x, axis=axis)


def array(x, dtype=None):
    return jnp.array(x, dtype=dtype)


def average(x, axis=None, weights=None):
    x = convert_to_tensor(x)
    dtypes_to_resolve = [x.dtype, float]
    if weights is not None:
        weights = convert_to_tensor(weights)
        dtypes_to_resolve.append(weights.dtype)
    dtype = dtypes.result_type(*dtypes_to_resolve)
    x = cast(x, dtype)
    if weights is not None:
        weights = cast(weights, dtype)
    return jnp.average(x, weights=weights, axis=axis)


def broadcast_to(x, shape):
    return jnp.broadcast_to(x, shape)


def ceil(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    return cast(jnp.ceil(x), dtype)


def clip(x, x_min, x_max):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "bool":
        x = cast(x, "int32")
    return jnp.clip(x, x_min, x_max)


def concatenate(xs, axis=0):
    return jnp.concatenate(xs, axis=axis)


def conjugate(x):
    return jnp.conjugate(x)


def conj(x):
    return conjugate(x)


def copy(x):
    return jnp.copy(x)


def cos(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.cos(x)


def cosh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.cosh(x)


def count_nonzero(x, axis=None):
    return cast(jnp.count_nonzero(x, axis=axis), "int32")


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    return jnp.cross(
        x1,
        x2,
        axisa=axisa,
        axisb=axisb,
        axisc=axisc,
        axis=axis,
    )


def cumprod(x, axis=None, dtype=None):
    return jnp.cumprod(x, axis=axis, dtype=dtype)


def cumsum(x, axis=None, dtype=None):
    return jnp.cumsum(x, axis=axis, dtype=dtype)


def diag(x, k=0):
    x = convert_to_tensor(x)
    return jnp.diag(x, k=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    return jnp.diagonal(
        x,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


def diff(a, n=1, axis=-1):
    return jnp.diff(a, n=n, axis=axis)


def digitize(x, bins):
    x = convert_to_tensor(x)
    bins = convert_to_tensor(bins)
    return jnp.digitize(x, bins)


def dot(x, y):
    return jnp.dot(x, y)


def empty(shape, dtype=None):
    dtype = dtype or config.floatx()
    return jnp.empty(shape, dtype=dtype)


def equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.equal(x1, x2)


def exp(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = cast(x, config.floatx())
    return jnp.exp(x)


def expand_dims(x, axis):
    return jnp.expand_dims(x, axis)


def expm1(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = cast(x, config.floatx())
    return jnp.expm1(x)


def flip(x, axis=None):
    return jnp.flip(x, axis=axis)


def floor(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())
    return jnp.floor(x)


def full(shape, fill_value, dtype=None):
    dtype = dtype or config.floatx()
    return jnp.full(shape, fill_value, dtype=dtype)


def full_like(x, fill_value, dtype=None):
    return jnp.full_like(x, fill_value, dtype=dtype)


def greater(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.greater(x1, x2)


def greater_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.greater_equal(x1, x2)


def hstack(xs):
    return jnp.hstack(xs)


def identity(n, dtype=None):
    dtype = dtype or config.floatx()
    return jnp.identity(n, dtype=dtype)


def imag(x):
    return jnp.imag(x)


def isclose(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.isclose(x1, x2)


def isfinite(x):
    return jnp.isfinite(x)


def isinf(x):
    return jnp.isinf(x)


def isnan(x):
    return jnp.isnan(x)


def less(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.less(x1, x2)


def less_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.less_equal(x1, x2)


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    return jnp.linspace(
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
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())
    return jnp.log(x)


def log10(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())
    return jnp.log10(x)


def log1p(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())
    return jnp.log1p(x)


def log2(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())
    return jnp.log2(x)


def logaddexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = cast(x1, dtype)
    x2 = cast(x2, dtype)
    return jnp.logaddexp(x1, x2)


def logical_and(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.logical_and(x1, x2)


def logical_not(x):
    return jnp.logical_not(x)


def logical_or(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.logical_or(x1, x2)


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    return jnp.logspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        base=base,
        dtype=dtype,
        axis=axis,
    )


def maximum(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.maximum(x1, x2)


def median(x, axis=None, keepdims=False):
    # axis of jnp.median must be hashable
    if isinstance(axis, list):
        axis = tuple(axis)
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())

    result = jnp.median(x, axis=axis, keepdims=keepdims)

    # TODO: jnp.median failed to keepdims when axis is None
    if keepdims is True and axis is None:
        for _ in range(x.ndim - 1):
            result = jnp.expand_dims(result, axis=-1)
    return result


def meshgrid(*x, indexing="xy"):
    return jnp.meshgrid(*x, indexing=indexing)


def min(x, axis=None, keepdims=False, initial=None):
    return jnp.min(x, axis=axis, keepdims=keepdims, initial=initial)


def minimum(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.minimum(x1, x2)


def mod(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.mod(x1, x2)


def moveaxis(x, source, destination):
    return jnp.moveaxis(x, source=source, destination=destination)


def nan_to_num(x):
    return jnp.nan_to_num(x)


def ndim(x):
    return jnp.ndim(x)


def nonzero(x):
    return jnp.nonzero(x)


def not_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.not_equal(x1, x2)


def ones_like(x, dtype=None):
    return jnp.ones_like(x, dtype=dtype)


def zeros_like(x, dtype=None):
    return jnp.zeros_like(x, dtype=dtype)


def outer(x1, x2):
    return jnp.outer(x1, x2)


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
    return jnp.pad(x, pad_width, mode=mode, **kwargs)


def prod(x, axis=None, keepdims=False, dtype=None):
    return jnp.prod(x, axis=axis, keepdims=keepdims, dtype=dtype)


def quantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())

    result = jnp.quantile(x, q, axis=axis, method=method, keepdims=keepdims)

    # TODO: jnp.quantile failed to keepdims when axis is None
    if keepdims is True and axis is None:
        for _ in range(x.ndim - 1):
            result = jnp.expand_dims(result, axis=-1)
    return result


def ravel(x):
    return jnp.ravel(x)


def real(x):
    return jnp.real(x)


def reciprocal(x):
    return jnp.reciprocal(x)


def repeat(x, repeats, axis=None):
    return jnp.repeat(x, repeats, axis=axis)


def reshape(x, new_shape):
    return jnp.reshape(x, new_shape)


def roll(x, shift, axis=None):
    return jnp.roll(x, shift, axis=axis)


def sign(x):
    return jnp.sign(x)


def sin(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.sin(x)


def sinh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.sinh(x)


def size(x):
    return jnp.size(x)


def sort(x, axis=-1):
    return jnp.sort(x, axis=axis)


def split(x, indices_or_sections, axis=0):
    return jnp.split(x, indices_or_sections, axis=axis)


def stack(x, axis=0):
    return jnp.stack(x, axis=axis)


def std(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())
    return jnp.std(x, axis=axis, keepdims=keepdims)


def swapaxes(x, axis1, axis2):
    return jnp.swapaxes(x, axis1=axis1, axis2=axis2)


def take(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices)
    return jnp.take(x, indices, axis=axis)


def take_along_axis(x, indices, axis=None):
    return jnp.take_along_axis(x, indices, axis=axis)


def tan(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.tan(x)


def tanh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.tanh(x)


def tensordot(x1, x2, axes=2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.tensordot(x1, x2, axes=axes)


def round(x, decimals=0):
    return jnp.round(x, decimals=decimals)


def tile(x, repeats):
    return jnp.tile(x, repeats)


def trace(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    dtype = None
    if standardize_dtype(x.dtype) == "bool":
        dtype = "int32"
    return jnp.trace(x, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)


def tri(N, M=None, k=0, dtype=None):
    dtype = dtype or config.floatx()
    return jnp.tri(N, M=M, k=k, dtype=dtype)


def tril(x, k=0):
    return jnp.tril(x, k=k)


def triu(x, k=0):
    return jnp.triu(x, k=k)


def vdot(x1, x2):
    return jnp.vdot(x1, x2)


def vstack(xs):
    return jnp.vstack(xs)


def where(condition, x1, x2):
    return jnp.where(condition, x1, x2)


def divide(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.divide(x1, x2)


def true_divide(x1, x2):
    return divide(x1, x2)


def power(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.power(x1, x2)


def negative(x):
    return jnp.negative(x)


def square(x):
    return jnp.square(x)


def sqrt(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())
    return jnp.sqrt(x)


def squeeze(x, axis=None):
    return jnp.squeeze(x, axis=axis)


def transpose(x, axes=None):
    x = convert_to_tensor(x)
    return jnp.transpose(x, axes=axes)


def var(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    # `jnp.var` does not handle low precision (e.g., float16) overflow
    # correctly, so we compute with float32 and cast back to the original type.
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    result_dtype = dtypes.result_type(x.dtype, float)
    return cast(
        jnp.var(x, axis=axis, keepdims=keepdims, dtype=compute_dtype),
        result_dtype,
    )


def sum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return jnp.sum(x, axis=axis, keepdims=keepdims)


def eye(N, M=None, k=0, dtype=None):
    dtype = dtype or config.floatx()
    return jnp.eye(N, M=M, k=k, dtype=dtype)


def floor_divide(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.floor_divide(x1, x2)


def logical_xor(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return jnp.logical_xor(x1, x2)
