from itertools import product
import numpy as np
from torch.autograd import Variable
from torch import ByteTensor, Tensor

from .shape import repeat_to_shape, squeeze
from .variable import ndim, tensor_from_numpy


def _normalize_axis(axis, num_dims):
    if axis is None:
        return None
    elif isinstance(axis, int):
        axes = [axis]
    elif isinstance(axis, tuple):
        axes = list(axis)
    elif isinstance(axis, list):
        axes = axis
    else:
        assert False
    return list(map(lambda axis: axis % num_dims, axes))


def _reduce_builtin(reduce_func_name, x, axis, keepdims=False):
    reduce_func = getattr(x, reduce_func_name)
    axes = _normalize_axis(axis, ndim(x))
    if axes is None:
        reduced = reduce_func()
        x = tensor_from_numpy(np.full(x.size(), reduced,
                                      dtype=x.numpy().dtype))
    else:
        for axis in axes:
            x = reduce_func(axis)[0]
            x.unsqueeze_(axis)
    if not keepdims:
        x.squeeze_()
    return x


def max(x, axis=None, keepdims=False):
    return _reduce_builtin('max', x, axis, keepdims)


def min(x, axis=None, keepdims=False):
    return _reduce_builtin('min', x, axis, keepdims)


def sum(x, axis=None, keepdims=False):
    return _reduce_builtin('sum', x, axis, keepdims)


def prod(x, axis=None, keepdims=False):
    return _reduce_builtin('prod', x, axis, keepdims)


def cumsum(x, axis=0):
    return _reduce_builtin('cumsum', x, axis, keepdims)


def cumprod(x, axis=0):
    return _reduce_builtin('cumprod', x, axis, keepdims)


def var(x, axis=None, keepdims=False):
    return _reduce_builtin('var', x, axis, keepdims)


def std(x, axis=None, keepdims=False):
    return _reduce_builtin('std', x, axis, keepdims)


def mean(x, axis=None, keepdims=False):
    return _reduce_builtin('mean', x, axis, keepdims)


def _reduce_custom(func, x, axis=None, keepdims=False):
    axes = _normalize_axis(axis, ndim(x))
    if axes is None:
        return func(x)
    shape = x.size()
    num_dims = len(shape)
    slices_per_dim = [None] * num_dims
    out_shape = list(shape)
    for axis in axes:
        slices_per_dim[axis] = list(range(shape[axis]))
        out_shape[axis] = 1
    for i, slices in enumerate(slices_per_dim):
        if slices is None:
            slices = [Slice(None)]
    out = torch.zeros(out_shape)
    for slices in product(slices_per_dim):
        output_slices = map(lambda n: 0 if isinstance(n, slice) else n, slices)
        out[output_slices] = func(x[slices])
    if not keepdims:
        out.squeeze_()
    return out


def _any(x):
    x = x.ceil()
    x.clamp_(0, 1)
    x.sum_()
    return x.clamp(0, 1)


def any(x, axis=None, keepdims=False):
    return _reduce_custom(_any, x, axis, keepdims)


def _all(x):
    x = x.ceil()
    x.clamp_(0, 1)
    x = 1 - x
    x.sum_()
    return x.clamp(0, 1)


def all(x, axis=None, keepdims=False):
    return _reduce_custom(_all, x, axis, keepdims)


def argmax(self, axis=-1):
    axis = _normalize_axis(axis, ndim(x))
    _, indices = x.max(axis)
    return indices


def argmin(self, axis=-1):
    axis = _normalize_axis(axis, ndim(x))
    _, indices = x.min(axis)
    return indices


def square(x):
    return x.pow(2)


def abs(x):
    return x.abs()


def sqrt(x):
    x.clamp_(x, 0.0, np.inf)
    return x.sqrt()


def exp(x):
    return x.exp()


def log(x):
    return x.log()


def logsumexp(x, axis=None, keepdims=False):
    x = x.exp()
    x = sum(x, axis, keepdims)
    return x.sum().log()


def round(x):
    return x.round()


def sign(x):
    return x.sign()


def pow(x, a):
    return x.pow(a)


def clip(x, min_value, max_value):
    if max_value is not None and max_value < min_value:
        max_value = min_value
    if max_value is None:
        max_value = np.inf
    return x.clamp(min_value, max_value)


def equal(x, y):
    return x == y


def not_equal(x, y):
    return x != y


def greater(x, y):
    return x > y


def greater_equal(x, y):
    return x >= y


def less(x, y):
    return x < y


def less_equal(x, y):
    return x <= y


def greater(x, y):
    return x > y


def greater_equal(x, y):
    return x >= y


def maximum(x, y):
    return torch.maximum(x, y)


def minimum(x, y):
    return torch.minimum(x, y)


def sin(x):
    return x.sin()


def cos(x):
    return x.cos()


def _moments(x, axes=None, shift=None, keep_dims=False):
    if shift is None:
        shift = mean(x, axes, keepdims=True)
        shift = repeat_to_shape(shift, x.size())

    x -= shift
    shifted_mean = x - shift
    shifted_mean = mean(shifted_mean, axes, keepdims=True)

    variance_mean = square(x - shift)
    variance_mean = mean(variance_mean, axes, keepdims=True)

    variance = variance_mean - square(shifted_mean)
    variance = repeat_to_shape(variance, x.size())
    
    shifted_mean = repeat_to_shape(shifted_mean, x.size())
    the_mean = shifted_mean + shift

    if not keep_dims:
        axes = _normalize_axis(axes, ndim(x))
        the_mean = squeeze(the_mean, axes)
        variance = squeeze(variance, axes)

    return the_mean, variance


def normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=1e-3):
    if gamma is None:
        if beta is None:
            gamma = torch.ones(x.size())
        else:
            gamma = torch.ones(beta.size())
    if beta is None:
        if gamma is None:
            beta = torch.zeros(x.size())
        else:
            beta = torch.zeros(gamma.size())

    axes = _normalize_axis(reduction_axes, ndim(x))
    mean, var = _moments(x, axes)
    raise NotImplementedError


def batch_normalization(x, mean, var, beta, gamma, epsilon=1e-03):
    if gamma is None:
        gamma = torch.ones(x.size())
    if beta is None:
        beta = torch.ones(x.size())
    return gamma * ((x - mean) / (sqrt(var) + epsilon)) + beta
