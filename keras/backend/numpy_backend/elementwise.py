import numpy as np
import scipy as sp


def prod(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.prod(x, axis=axis, keepdims=keepdims)


def cumsum(x, axis=0):
    return np.cumsum(x, axis=axis)


def cumprod(x, axis=0):
    return np.cumprod(x, axis=axis)


def mean(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.mean(x, axis=axis, keepdims=keepdims)


def std(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.std(x, axis=axis, keepdims=keepdims)


def var(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.var(x, axis=axis, keepdims=keepdims)


def argmax(x, axis=-1):
    return np.argmax(x, axis=axis)


def argmin(x, axis=-1):
    return np.argmin(x, axis=axis)


def sqrt(x):
    y = np.sqrt(x)
    y[np.isnan(y)] = 0.
    return y


def logsumexp(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return sp.misc.logsumexp(x, axis=axis, keepdims=keepdims)


def pow(x, a=1.):
    return np.power(x, a)


def clip(x, min_value, max_value):
    return np.clip(x, min_value, max_value)


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


def maximum(x, y):
    return np.maximum(x, y)


def minimum(x, y):
    return np.minimum(x, y)


square = np.square
exp = np.exp
log = np.log
sign = np.sign
cos = np.cos
sin = np.sin
