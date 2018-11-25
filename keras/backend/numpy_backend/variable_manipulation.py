import numpy as np

from ..common import floatx


def variable(value, dtype=None, name=None, constraint=None):
    if constraint is not None:
        raise TypeError("Constraint must be None when "
                        "using the NumPy backend.")
    return np.array(value, dtype)


def constant(value, dtype=None, shape=None, name=None):
    if dtype is None:
        dtype = floatx()
    if shape is None:
        shape = ()
    np_value = value * np.ones(shape)
    np_value.astype(dtype)
    return np_value


def dtype(x):
    return x.dtype.name


def eval(x):
    return x


def zeros(shape, dtype=floatx(), name=None):
    return np.zeros(shape, dtype=dtype)


def zeros_like(x, dtype=floatx(), name=None):
    return np.zeros_like(x, dtype=dtype)


def ones(shape, dtype=floatx(), name=None):
    return np.ones(shape, dtype=dtype)


def ones_like(x, dtype=floatx(), name=None):
    return np.ones_like(x, dtype=dtype)


def eye(size, dtype=None, name=None):
    return np.eye(size, dtype=dtype)


def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
    return (high - low) * np.random.random(shape).astype(dtype) + low


def random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None):
    return scale * np.random.randn(*shape).astype(dtype) + mean


def count_params(x):
    return x.size
