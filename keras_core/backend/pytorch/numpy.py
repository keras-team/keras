import torch

def add(x1, x2):
    pass
    #return tfnp.add(x1, x2)


def einsum(subscripts, *operands, **kwargs):
    pass
    #return tfnp.einsum(subscripts, *operands, **kwargs)


def subtract(x1, x2):
    pass
    #return tfnp.subtract(x1, x2)


def matmul(x1, x2):
    pass
    #return tfnp.matmul(x1, x2)


def multiply(x1, x2):
    pass
    #return tfnp.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    pass
    #return tfnp.mean(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False, initial=None):
    pass
    # The TensorFlow numpy API implementation doesn't support `initial` so we
    # handle it manually here.
    #if initial is not None:
    #    return tf.math.maximum(
    #        tfnp.max(x, axis=axis, keepdims=keepdims), initial
    #    )

    # TensorFlow returns -inf by default for an empty list, but for consistency
    # with other backends and the numpy API we want to throw in this case.
    #tf.assert_greater(
    #    size(x),
    #    tf.constant(0, dtype=tf.int64),
    #    message="Cannot compute the max of an empty tensor.",
    #)

    #return tfnp.max(x, axis=axis, keepdims=keepdims)


def ones(shape, dtype="float32"):
    pass
    #return tf.ones(shape, dtype=dtype)


def zeros(shape, dtype="float32"):
    pass
    #return tf.zeros(shape, dtype=dtype)


def absolute(x):
    pass
    #return tfnp.absolute(x)


def abs(x):
    pass
    #return absolute(x)


def all(x, axis=None, keepdims=False):
    pass
    #return tfnp.all(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    pass
    #return tfnp.any(x, axis=axis, keepdims=keepdims)


def amax(x, axis=None, keepdims=False):
    pass
    #return tfnp.amax(x, axis=axis, keepdims=keepdims)


def amin(x, axis=None, keepdims=False):
    pass
    #return tfnp.amin(x, axis=axis, keepdims=keepdims)


def append(
    x1,
    x2,
    axis=None,
):
    pass
    #return tfnp.append(x1, x2, axis=axis)


def arange(start, stop=None, step=None, dtype=None):
    pass
    #return tfnp.arange(start, stop, step=step, dtype=dtype)


def arccos(x):
    pass
    #return tfnp.arccos(x)


def arcsin(x):
    pass
    #return tfnp.arcsin(x)


def arctan(x):
    pass
    #return tfnp.arctan(x)


def arctan2(x1, x2):
    pass
    #return tfnp.arctan2(x1, x2)


def argmax(x, axis=None):
    pass
    #return tfnp.argmax(x, axis=axis)


def argmin(x, axis=None):
    pass
    #return tfnp.argmin(x, axis=axis)


def argsort(x, axis=-1):
    pass
    #return tfnp.argsort(x, axis=axis)


def array(x, dtype=None):
    pass
    #return tfnp.array(x, dtype=dtype)


def average(x, axis=None, weights=None):
    pass
    #return tfnp.average(x, weights=weights, axis=axis)


def broadcast_to(x, shape):
    pass
    #return tfnp.broadcast_to(x, shape)


def ceil(x):
    pass
    #return tfnp.ceil(x)


def clip(x, x_min, x_max):
    pass
    #return tfnp.clip(x, x_min, x_max)


def concatenate(xs, axis=0):
    pass
    #return tfnp.concatenate(xs, axis=axis)


def conjugate(x):
    pass
    #return tfnp.conjugate(x)


def conj(x):
    pass
    #return conjugate(x)


def copy(x):
    pass
    #return tfnp.copy(x)


def cos(x):
    pass
    #return tfnp.cos(x)


def count_nonzero(x, axis=None):
    pass
    #return tfnp.count_nonzero(x, axis=axis)


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    pass
    #return tfnp.cross(
    #    x1,
    #    x2,
    #    axisa=axisa,
    #    axisb=axisb,
    #    axisc=axisc,
    #    axis=axis,
    #)


def cumprod(x, axis=None):
    pass
    #return tfnp.cumprod(x, axis=axis)


def cumsum(x, axis=None):
    pass
    #return tfnp.cumsum(x, axis=axis)


def diag(x, k=0):
    pass
    #return tfnp.diag(x, k=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    pass
    #return tfnp.diagonal(
    #    x,
    #    offset=offset,
    #    axis1=axis1,
    #    axis2=axis2,
    #)


def dot(x, y):
    pass
    #return tfnp.dot(x, y)


def empty(shape, dtype="float32"):
    pass
    #return tfnp.empty(shape, dtype=dtype)


def equal(x1, x2):
    pass
    #return tfnp.equal(x1, x2)


def exp(x):
    pass
    #return tfnp.exp(x)


def expand_dims(x, axis):
    pass
    #return tfnp.expand_dims(x, axis)


def expm1(x):
    pass
    #return tfnp.expm1(x)


def flip(x, axis=None):
    pass
    #return tfnp.flip(x, axis=axis)


def floor(x):
    pass
    #return tfnp.floor(x)


def full(shape, fill_value, dtype=None):
    pass
    return tfnp.full(shape, fill_value, dtype=dtype)


def full_like(x, fill_value, dtype=None):
    pass
    return tfnp.full_like(x, fill_value, dtype=dtype)


def greater(x1, x2):
    pass
    return tfnp.greater(x1, x2)


def greater_equal(x1, x2):
    pass
    return tfnp.greater_equal(x1, x2)


def hstack(xs):
    pass
    return tfnp.hstack(xs)


def identity(n, dtype="float32"):
    pass
    return tfnp.identity(n, dtype=dtype)


def imag(x):
    pass
    return tfnp.imag(x)


def isclose(x1, x2):
    pass
    return tfnp.isclose(x1, x2)


def isfinite(x):
    pass
    return tfnp.isfinite(x)


def isinf(x):
    pass
    return tfnp.isinf(x)


def isnan(x):
    pass
    return tfnp.isnan(x)


def less(x1, x2):
    pass
    return tfnp.less(x1, x2)


def less_equal(x1, x2):
    pass
    return tfnp.less_equal(x1, x2)


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    pass
    #return tfnp.linspace(
    #    start,
    #    stop,
    #    num=num,
    #    endpoint=endpoint,
    #    retstep=retstep,
    #    dtype=dtype,
    #    axis=axis,
    #)


def log(x):
    pass
    #return tfnp.log(x)


def log10(x):
    pass
    #return tfnp.log10(x)


def log1p(x):
    pass
    #return tfnp.log1p(x)


def log2(x):
    pass
    #return tfnp.log2(x)


def logaddexp(x1, x2):
    pass
    #return tfnp.logaddexp(x1, x2)


def logical_and(x1, x2):
    pass
    #return tfnp.logical_and(x1, x2)


def logical_not(x):
    pass
    #return tfnp.logical_not(x)


def logical_or(x1, x2):
    pass
    #return tfnp.logical_or(x1, x2)


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    pass
    #return tfnp.logspace(
    #    start,
    #    stop,
    #    num=num,
    #    endpoint=endpoint,
    #    base=base,
    #    dtype=dtype,
    #    axis=axis,
    #)


def maximum(x1, x2):
    pass
    #return tfnp.maximum(x1, x2)


def meshgrid(*x, indexing="xy"):
    pass
    #return tfnp.meshgrid(*x, indexing=indexing)


def min(x, axis=None, keepdims=False, initial=None):
    pass
    ## The TensorFlow numpy API implementation doesn't support `initial` so we
    ## handle it manually here.
    #if initial is not None:
    #    return tf.math.minimum(
    #        tfnp.min(x, axis=axis, keepdims=keepdims), initial
    #    )

    ## TensorFlow returns inf by default for an empty list, but for consistency
    ## with other backends and the numpy API we want to throw in this case.
    #tf.assert_greater(
    #    size(x),
    #    tf.constant(0, dtype=tf.int64),
    #    message="Cannot compute the min of an empty tensor.",
    #)

    return tfnp.min(x, axis=axis, keepdims=keepdims)


def minimum(x1, x2):
    pass
    #return tfnp.minimum(x1, x2)


def mod(x1, x2):
    pass
    #return tfnp.mod(x1, x2)


def moveaxis(x, source, destination):
    pass
    #return tfnp.moveaxis(x, source=source, destination=destination)


def nan_to_num(x):
    pass
    ## Replace NaN with 0
    #x = tf.where(tf.math.is_nan(x), 0, x)

    ## Replace positive infinitiy with dtype.max
    #x = tf.where(tf.math.is_inf(x) & (x > 0), x.dtype.max, x)

    ## Replace negative infinity with dtype.min
    #x = tf.where(tf.math.is_inf(x) & (x < 0), x.dtype.min, x)

    #return x


def ndim(x):
    pass
    #return tfnp.ndim(x)


def nonzero(x):
    pass
    #return tfnp.nonzero(x)


def not_equal(x1, x2):
    pass
    #return tfnp.not_equal(x1, x2)


def ones_like(x, dtype=None):
    pass
    #return tfnp.ones_like(x, dtype=dtype)


def outer(x1, x2):
    pass
    #return tfnp.outer(x1, x2)


def pad(x, pad_width, mode="constant"):
    pass
    #return tfnp.pad(x, pad_width, mode=mode)


def prod(x, axis=None, keepdims=False, dtype=None):
    pass
    #return tfnp.prod(x, axis=axis, keepdims=keepdims, dtype=dtype)


def ravel(x):
    pass
    #return tfnp.ravel(x)


def real(x):
    pass
    #return tfnp.real(x)


def reciprocal(x):
    pass
    #return tfnp.reciprocal(x)


def repeat(x, repeats, axis=None):
    pass
    #return tfnp.repeat(x, repeats, axis=axis)


def reshape(x, new_shape):
    pass
    #return tfnp.reshape(x, new_shape)


def roll(x, shift, axis=None):
    pass
    #return tfnp.roll(x, shift, axis=axis)


def sign(x):
    pass
    #return tfnp.sign(x)


def sin(x):
    pass
    #return tfnp.sin(x)


def size(x):
    pass
    #return tfnp.size(x)


def sort(x, axis=-1):
    pass
    #return tfnp.sort(x, axis=axis)


def split(x, indices_or_sections, axis=0):
    pass
    #return tfnp.split(x, indices_or_sections, axis=axis)


def stack(x, axis=0):
    pass
    #return tfnp.stack(x, axis=axis)


def std(x, axis=None, keepdims=False):
    pass
    #return tfnp.std(x, axis=axis, keepdims=keepdims)


def swapaxes(x, axis1, axis2):
    pass
    #return tfnp.swapaxes(x, axis1=axis1, axis2=axis2)


def take(x, indices, axis=None):
    pass
    #return tfnp.take(x, indices, axis=axis)


def take_along_axis(x, indices, axis=None):
    pass
    #return tfnp.take_along_axis(x, indices, axis=axis)


def tan(x):
    pass
    #return tfnp.tan(x)


def tensordot(x1, x2, axes=2):
    pass
    #return tfnp.tensordot(x1, x2, axes=axes)


def round(x, decimals=0):
    pass
    #return tfnp.round(x, decimals=decimals)


def tile(x, repeats):
    pass
    #return tfnp.tile(x, repeats)


def trace(x, offset=0, axis1=0, axis2=1):
    pass
    #return tfnp.trace(x, offset=offset, axis1=axis1, axis2=axis2)


def tri(N, M=None, k=0, dtype="float32"):
    pass
    #return tfnp.tri(N, M=M, k=k, dtype=dtype)


def tril(x, k=0):
    pass
    #return tfnp.tril(x, k=k)


def triu(x, k=0):
    pass
    return tfnp.triu(x, k=k)


def vdot(x1, x2):
    pass
    #return tfnp.vdot(x1, x2)


def vstack(xs):
    pass
    #return tfnp.vstack(xs)


def where(condition, x1, x2):
    pass
    #return tfnp.where(condition, x1, x2)


def divide(x1, x2):
    pass
    #return tfnp.divide(x1, x2)


def true_divide(x1, x2):
    pass
    #return tfnp.true_divide(x1, x2)


def power(x1, x2):
    pass
    #return tfnp.power(x1, x2)


def negative(x):
    pass
    #return tfnp.negative(x)


def square(x):
    pass
    #return tfnp.square(x)


def sqrt(x):
    pass
    #return tfnp.sqrt(x)


def squeeze(x, axis=None):
    pass
    #return tfnp.squeeze(x, axis=axis)


def transpose(x, axes=None):
    pass
    #return tfnp.transpose(x, axes=axes)


def var(x, axis=None, keepdims=False):
    pass
    #return tfnp.var(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    pass
    #return tfnp.sum(x, axis=axis, keepdims=keepdims)


def eye(N, M=None, k=0, dtype="float32"):
    pass
    #return tfnp.eye(N, M=M, k=k, dtype=dtype)
