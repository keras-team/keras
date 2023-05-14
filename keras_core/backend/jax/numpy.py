import jax.numpy as jnp


def add(x1, x2):
    return jnp.add(x1, x2)


def bincount(x, weights=None, minlength=0):
    if len(x.shape) == 2:
        bincounts = [
            jnp.bincount(arr, weights=weights, minlength=minlength)
            for arr in list(x)
        ]
        return jnp.stack(bincounts)
    return jnp.bincount(x, weights=weights, minlength=minlength)


def einsum(subscripts, *operands, **kwargs):
    return jnp.einsum(subscripts, *operands, **kwargs)


def subtract(x1, x2):
    return jnp.subtract(x1, x2)


def matmul(x1, x2):
    return jnp.matmul(x1, x2)


def multiply(x1, x2):
    return jnp.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    return jnp.mean(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False, initial=None):
    return jnp.max(x, axis=axis, keepdims=keepdims, initial=initial)


def ones(shape, dtype="float32"):
    return jnp.ones(shape, dtype=dtype)


def zeros(shape, dtype="float32"):
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


def append(
    x1,
    x2,
    axis=None,
):
    return jnp.append(x1, x2, axis=axis)


def arange(start, stop=None, step=None, dtype=None):
    return jnp.arange(start, stop, step=step, dtype=dtype)


def arccos(x):
    return jnp.arccos(x)


def arcsin(x):
    return jnp.arcsin(x)


def arctan(x):
    return jnp.arctan(x)


def arctan2(x1, x2):
    return jnp.arctan2(x1, x2)


def argmax(x, axis=None):
    return jnp.argmax(x, axis=axis)


def argmin(x, axis=None):
    return jnp.argmin(x, axis=axis)


def argsort(x, axis=-1):
    return jnp.argsort(x, axis=axis)


def array(x, dtype=None):
    return jnp.array(x, dtype=dtype)


def average(x, axis=None, weights=None):
    return jnp.average(x, weights=weights, axis=axis)


def broadcast_to(x, shape):
    return jnp.broadcast_to(x, shape)


def ceil(x):
    return jnp.ceil(x)


def clip(x, x_min, x_max):
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
    return jnp.cos(x)


def count_nonzero(x, axis=None):
    return jnp.count_nonzero(x, axis=axis)


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    return jnp.cross(
        x1,
        x2,
        axisa=axisa,
        axisb=axisb,
        axisc=axisc,
        axis=axis,
    )


def cumprod(x, axis=None):
    return jnp.cumprod(x, axis=axis)


def cumsum(x, axis=None):
    return jnp.cumsum(x, axis=axis)


def diag(x, k=0):
    return jnp.diag(x, k=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    return jnp.diagonal(
        x,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


def dot(x, y):
    return jnp.dot(x, y)


def empty(shape, dtype="float32"):
    return jnp.empty(shape, dtype=dtype)


def equal(x1, x2):
    return jnp.equal(x1, x2)


def exp(x):
    return jnp.exp(x)


def expand_dims(x, axis):
    return jnp.expand_dims(x, axis)


def expm1(x):
    return jnp.expm1(x)


def flip(x, axis=None):
    return jnp.flip(x, axis=axis)


def floor(x):
    return jnp.floor(x)


def full(shape, fill_value, dtype=None):
    return jnp.full(shape, fill_value, dtype=dtype)


def full_like(x, fill_value, dtype=None):
    return jnp.full_like(x, fill_value, dtype=dtype)


def greater(x1, x2):
    return jnp.greater(x1, x2)


def greater_equal(x1, x2):
    return jnp.greater_equal(x1, x2)


def hstack(xs):
    return jnp.hstack(xs)


def identity(n, dtype="float32"):
    return jnp.identity(n, dtype=dtype)


def imag(x):
    return jnp.imag(x)


def isclose(x1, x2):
    return jnp.isclose(x1, x2)


def isfinite(x):
    return jnp.isfinite(x)


def isinf(x):
    return jnp.isinf(x)


def isnan(x):
    return jnp.isnan(x)


def less(x1, x2):
    return jnp.less(x1, x2)


def less_equal(x1, x2):
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
    return jnp.log(x)


def log10(x):
    return jnp.log10(x)


def log1p(x):
    return jnp.log1p(x)


def log2(x):
    return jnp.log2(x)


def logaddexp(x1, x2):
    return jnp.logaddexp(x1, x2)


def logical_and(x1, x2):
    return jnp.logical_and(x1, x2)


def logical_not(x):
    return jnp.logical_not(x)


def logical_or(x1, x2):
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
    return jnp.maximum(x1, x2)


def meshgrid(*x, indexing="xy"):
    return jnp.meshgrid(*x, indexing=indexing)


def min(x, axis=None, keepdims=False, initial=None):
    return jnp.min(x, axis=axis, keepdims=keepdims, initial=initial)


def minimum(x1, x2):
    return jnp.minimum(x1, x2)


def mod(x1, x2):
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
    return jnp.not_equal(x1, x2)


def ones_like(x, dtype=None):
    return jnp.ones_like(x, dtype=dtype)


def zeros_like(x, dtype=None):
    return jnp.zeros_like(x, dtype=dtype)


def outer(x1, x2):
    return jnp.outer(x1, x2)


def pad(x, pad_width, mode="constant"):
    return jnp.pad(x, pad_width, mode=mode)


def prod(x, axis=None, keepdims=False, dtype=None):
    return jnp.prod(x, axis=axis, keepdims=keepdims, dtype=dtype)


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
    return jnp.sin(x)


def size(x):
    return jnp.size(x)


def sort(x, axis=-1):
    return jnp.sort(x, axis=axis)


def split(x, indices_or_sections, axis=0):
    return jnp.split(x, indices_or_sections, axis=axis)


def stack(x, axis=0):
    return jnp.stack(x, axis=axis)


def std(x, axis=None, keepdims=False):
    return jnp.std(x, axis=axis, keepdims=keepdims)


def swapaxes(x, axis1, axis2):
    return jnp.swapaxes(x, axis1=axis1, axis2=axis2)


def take(x, indices, axis=None):
    return jnp.take(x, indices, axis=axis)


def take_along_axis(x, indices, axis=None):
    return jnp.take_along_axis(x, indices, axis=axis)


def tan(x):
    return jnp.tan(x)


def tensordot(x1, x2, axes=2):
    return jnp.tensordot(x1, x2, axes=axes)


def round(x, decimals=0):
    return jnp.round(x, decimals=decimals)


def tile(x, repeats):
    return jnp.tile(x, repeats)


def trace(x, offset=0, axis1=0, axis2=1):
    return jnp.trace(x, offset=offset, axis1=axis1, axis2=axis2)


def tri(N, M=None, k=0, dtype="float32"):
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
    return jnp.divide(x1, x2)


def true_divide(x1, x2):
    return jnp.true_divide(x1, x2)


def power(x1, x2):
    return jnp.power(x1, x2)


def negative(x):
    return jnp.negative(x)


def square(x):
    return jnp.square(x)


def sqrt(x):
    return jnp.sqrt(x)


def squeeze(x, axis=None):
    return jnp.squeeze(x, axis=axis)


def transpose(x, axes=None):
    return jnp.transpose(x, axes=axes)


def var(x, axis=None, keepdims=False):
    return jnp.var(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    return jnp.sum(x, axis=axis, keepdims=keepdims)


def eye(N, M=None, k=0, dtype="float32"):
    return jnp.eye(N, M=M, k=k, dtype=dtype)
