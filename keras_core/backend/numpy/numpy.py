import numpy as np


def add(x1, x2):
    return np.add(x1, x2)


def einsum(subscripts, *operands, **kwargs):
    return np.einsum(subscripts, *operands, **kwargs)


def subtract(x1, x2):
    return np.subtract(x1, x2)


def matmul(x1, x2):
    return np.matmul(x1, x2)


def multiply(x1, x2):
    return np.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.mean(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False, initial=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.max(x, axis=axis, keepdims=keepdims, initial=initial)


def ones(shape, dtype="float32"):
    return np.ones(shape, dtype=dtype)


def zeros(shape, dtype="float32"):
    return np.zeros(shape, dtype=dtype)


def absolute(x):
    return np.absolute(x)


def abs(x):
    return absolute(x)


def all(x, axis=None, keepdims=False):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.all(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.any(x, axis=axis, keepdims=keepdims)


def amax(x, axis=None, keepdims=False):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.amax(x, axis=axis, keepdims=keepdims)


def amin(x, axis=None, keepdims=False):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.amin(x, axis=axis, keepdims=keepdims)


def append(
    x1,
    x2,
    axis=None,
):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.append(x1, x2, axis=axis)


def arange(start, stop=None, step=None, dtype=None):
    return np.arange(start, stop, step=step, dtype=dtype)


def arccos(x):
    return np.arccos(x)


def arccosh(x):
    return np.arccosh(x)


def arcsin(x):
    return np.arcsin(x)


def arcsinh(x):
    return np.arcsinh(x)


def arctan(x):
    return np.arctan(x)


def arctan2(x1, x2):
    return np.arctan2(x1, x2)


def arctanh(x):
    return np.arctanh(x)


def argmax(x, axis=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.argmax(x, axis=axis)


def argmin(x, axis=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.argmin(x, axis=axis)


def argsort(x, axis=-1):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.argsort(x, axis=axis)


def array(x, dtype=None):
    return np.array(x, dtype=dtype)


def average(x, axis=None, weights=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.average(x, weights=weights, axis=axis)


def bincount(x, weights=None, minlength=0):
    return np.bincount(x, weights, minlength)


def broadcast_to(x, shape):
    return np.broadcast_to(x, shape)


def ceil(x):
    return np.ceil(x)


def clip(x, x_min, x_max):
    return np.clip(x, x_min, x_max)


def concatenate(xs, axis=0):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.concatenate(xs, axis=axis)


def conjugate(x):
    return np.conjugate(x)


def conj(x):
    return conjugate(x)


def copy(x):
    return np.copy(x)


def cos(x):
    return np.cos(x)


def cosh(x):
    return np.cosh(x)


def count_nonzero(x, axis=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.count_nonzero(x, axis=axis)


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.cross(
        x1,
        x2,
        axisa=axisa,
        axisb=axisb,
        axisc=axisc,
        axis=axis,
    )


def cumprod(x, axis=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.cumprod(x, axis=axis)


def cumsum(x, axis=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.cumsum(x, axis=axis)


def diag(x, k=0):
    return np.diag(x, k=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    axis1 = tuple(axis1) if isinstance(axis1, list) else axis1
    axis2 = tuple(axis2) if isinstance(axis2, list) else axis2
    return np.diagonal(
        x,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


def dot(x, y):
    return np.dot(x, y)


def empty(shape, dtype="float32"):
    return np.empty(shape, dtype=dtype)


def equal(x1, x2):
    return np.equal(x1, x2)


def exp(x):
    return np.exp(x)


def expand_dims(x, axis):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.expand_dims(x, axis)


def expm1(x):
    return np.expm1(x)


def flip(x, axis=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.flip(x, axis=axis)


def floor(x):
    return np.floor(x)


def full(shape, fill_value, dtype=None):
    return np.full(shape, fill_value, dtype=dtype)


def full_like(x, fill_value, dtype=None):
    return np.full_like(x, fill_value, dtype=dtype)


def greater(x1, x2):
    return np.greater(x1, x2)


def greater_equal(x1, x2):
    return np.greater_equal(x1, x2)


def hstack(xs):
    return np.hstack(xs)


def identity(n, dtype="float32"):
    return np.identity(n, dtype=dtype)


def imag(x):
    return np.imag(x)


def isclose(x1, x2):
    return np.isclose(x1, x2)


def isfinite(x):
    return np.isfinite(x)


def isinf(x):
    return np.isinf(x)


def isnan(x):
    return np.isnan(x)


def less(x1, x2):
    return np.less(x1, x2)


def less_equal(x1, x2):
    return np.less_equal(x1, x2)


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    axis = tuple(axis) if isinstance(axis, list) else axis
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
    return np.log(x)


def log10(x):
    return np.log10(x)


def log1p(x):
    return np.log1p(x)


def log2(x):
    return np.log2(x)


def logaddexp(x1, x2):
    return np.logaddexp(x1, x2)


def logical_and(x1, x2):
    return np.logical_and(x1, x2)


def logical_not(x):
    return np.logical_not(x)


def logical_or(x1, x2):
    return np.logical_or(x1, x2)


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
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
    return np.maximum(x1, x2)


def meshgrid(*x, indexing="xy"):
    return np.meshgrid(*x, indexing=indexing)


def min(x, axis=None, keepdims=False, initial=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.min(x, axis=axis, keepdims=keepdims, initial=initial)


def minimum(x1, x2):
    return np.minimum(x1, x2)


def mod(x1, x2):
    return np.mod(x1, x2)


def moveaxis(x, source, destination):
    return np.moveaxis(x, source=source, destination=destination)


def nan_to_num(x):
    return np.nan_to_num(x)


def ndim(x):
    return np.ndim(x)


def nonzero(x):
    return np.nonzero(x)


def not_equal(x1, x2):
    return np.not_equal(x1, x2)


def zeros_like(x, dtype=None):
    return np.zeros_like(x, dtype=dtype)


def ones_like(x, dtype=None):
    return np.ones_like(x, dtype=dtype)


def outer(x1, x2):
    return np.outer(x1, x2)


def pad(x, pad_width, mode="constant"):
    return np.pad(x, pad_width, mode=mode)


def prod(x, axis=None, keepdims=False, dtype=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.prod(x, axis=axis, keepdims=keepdims, dtype=dtype)


def ravel(x):
    return np.ravel(x)


def real(x):
    return np.real(x)


def reciprocal(x):
    return np.reciprocal(x)


def repeat(x, repeats, axis=None):
    return np.repeat(x, repeats, axis=axis)


def reshape(x, new_shape):
    return np.reshape(x, new_shape)


def roll(x, shift, axis=None):
    return np.roll(x, shift, axis=axis)


def sign(x):
    return np.sign(x)


def sin(x):
    return np.sin(x)


def sinh(x):
    return np.sinh(x)


def size(x):
    return np.size(x)


def sort(x, axis=-1):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.sort(x, axis=axis)


def split(x, indices_or_sections, axis=0):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.split(x, indices_or_sections, axis=axis)


def stack(x, axis=0):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.stack(x, axis=axis)


def std(x, axis=None, keepdims=False):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.std(x, axis=axis, keepdims=keepdims)


def swapaxes(x, axis1, axis2):
    return np.swapaxes(x, axis1=axis1, axis2=axis2)


def take(x, indices, axis=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.take(x, indices, axis=axis)


def take_along_axis(x, indices, axis=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.take_along_axis(x, indices, axis=axis)


def tan(x):
    return np.tan(x)


def tanh(x):
    return np.tanh(x)


def tensordot(x1, x2, axes=2):
    axes = tuple(axes) if isinstance(axes, list) else axes
    return np.tensordot(x1, x2, axes=axes)


def round(x, decimals=0):
    return np.round(x, decimals=decimals)


def tile(x, repeats):
    return np.tile(x, repeats)


def trace(x, offset=0, axis1=0, axis2=1):
    axis1 = tuple(axis1) if isinstance(axis1, list) else axis1
    axis2 = tuple(axis2) if isinstance(axis2, list) else axis2
    return np.trace(x, offset=offset, axis1=axis1, axis2=axis2)


def tri(N, M=None, k=0, dtype="float32"):
    return np.tri(N, M=M, k=k, dtype=dtype)


def tril(x, k=0):
    return np.tril(x, k=k)


def triu(x, k=0):
    return np.triu(x, k=k)


def vdot(x1, x2):
    return np.vdot(x1, x2)


def vstack(xs):
    return np.vstack(xs)


def where(condition, x1, x2):
    return np.where(condition, x1, x2)


def divide(x1, x2):
    return np.divide(x1, x2)


def true_divide(x1, x2):
    return np.true_divide(x1, x2)


def power(x1, x2):
    return np.power(x1, x2)


def negative(x):
    return np.negative(x)


def square(x):
    return np.square(x)


def sqrt(x):
    return np.sqrt(x)


def squeeze(x, axis=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.squeeze(x, axis=axis)


def transpose(x, axes=None):
    axes = tuple(axes) if isinstance(axes, list) else axes
    return np.transpose(x, axes=axes)


def var(x, axis=None, keepdims=False):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.var(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    axis = tuple(axis) if isinstance(axis, list) else axis
    return np.sum(x, axis=axis, keepdims=keepdims)


def eye(N, M=None, k=0, dtype="float32"):
    return np.eye(N, M=M, k=k, dtype=dtype)


def floor_divide(x1, x2):
    return np.floor_divide(x1, x2)


def logical_xor(x1, x2):
    return np.logical_xor(x1, x2)
