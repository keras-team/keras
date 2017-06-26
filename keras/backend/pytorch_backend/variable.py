from contextlib import contextmanager
import numpy as np
import torch
from torch.autograd import Variable
from torch import Tensor

from ..common import floatx


NAME2CPU_TENSOR_TYPE = {
    'float16': None,
    'float32': torch.FloatTensor,
    'float64': torch.DoubleTensor,
    'int8': torch.CharTensor,
    'int16': torch.ShortTensor,
    'int32': torch.IntTensor,
    'int64': torch.LongTensor,
    'uint8': torch.ByteTensor,
}


NAME_SCOPE_STACK = []


def is_sparse(tensor):
    raise NotImplementedError


def to_dense(tensor):
    raise NotImplementedError


@contextmanager
def name_scope(name):
    global NAME_SCOPE_STACK
    NAME_SCOPE_STACK.append(name)
    yield
    NAME_SCOPE_STACK.pop()


def _str_from_dtype(dtype):
    if isinstance(dtype, np.dtype):
        return dtype.name
    elif isinstance(dtype, str):
        return dtype
    else:
        assert False


def tensor_from_numpy(arr, override_dtype=None):
    if override_dtype:
        dtype = override_dtype
    else:
        dtype = arr.dtype
    dtype = _str_from_dtype(dtype)
    tensor_type = NAME2CPU_TENSOR_TYPE[dtype]
    return tensor_type(arr)


def variable(value, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    if isinstance(value, (int, float)):
        value = np.array([value], dtype)
        value = tensor_from_numpy(value, dtype)
    elif isinstance(value, (tuple, list)):
        value = np.array(value, dtype=dtype)
        value = tensor_from_numpy(value, dtype)
    elif isinstance(value, Tensor):
        pass
    elif isinstance(value, Constant):
        value = tensor_from_numpy(value.value)
    else:
        assert False
    v = Variable(value)
    v._keras_shape = value.numpy().shape
    v._uses_learning_phase = False
    return v


class Constant(object):
    def __init__(self, value, dtype, shape):
        self.value = value.astype(dtype)
        self.dtype = dtype
        self.shape = shape
        self._keras_shape = value.shape
        self._uses_learning_phase = False


def constant(value, dtype=None, shape=None, name=None):
    if dtype is None:
        dtype = floatx()
    if shape is None:
        shape = ()
    value = value * np.ones(shape)
    v = Constant(value, dtype, shape)
    return v


def is_keras_tensor(x):
    print('is_keras_tensor', type(x))
    #assert isinstance(x, Tensor)
    return hasattr(x, '_keras_history')


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    if dtype is None:
        dtype = floatx()
    else:
        dtype = _str_from_dtype(dtype)
    if not shape:
        if ndim:
            shape = tuple([None for _ in range(ndim)])
    fake_shape = tuple(map(lambda dim: 1 if dim is None else dim, shape))
    fake_tensor = tensor_from_numpy(np.ones(fake_shape, dtype))
    v = Variable(fake_tensor)
    v._keras_shape = shape
    v._uses_learning_phase = False
    return v


def shape(x):
    x = int_shape(x)
    return tensor_from_numpy(x)


def int_shape(x):
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    return tuple(x.size())


def ndim(x):
    return len(x.size())


def dtype(x):
    if isinstance(x, Variable):
        return x.data.numpy().dtype
    elif isinstance(x, Tensor):
        return x.numpy().dtype
    else:
        assert False


def eval(x):
    return x.data


def zeros(shape, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    shape = tuple(map(int, shape))
    value = np.zeros(shape)
    return variable(value, dtype, name)


def ones(shape, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    shape = tuple(map(int, shape))
    value = np.zeros(shape)
    return variable(value, dtype, name)


def ones(size, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    value = np.eye(size)
    return variable(value, dtype, name)


def zeros_like(x, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    value = np.zeros(int_shape(x))
    return variable(value, dtype, name)


def ones_like(x, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    value = np.zeros(int_shape(x))
    return variable(value, dtype, name)


def identity(x):
    return x.clone()


def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
    data = random_uniform(shape, low, high, dtype, seed)
    return Variable(data)


def random_normal_variable(shape, mean, scale, dtype=None, name=None,
                           seed=None):
    data = random_normal(shape, mean, scale, dtype, seed)
    return Variable(data)


def count_params(x):
    return np.prod(x.size())


def cast(x, dtype):
    dtype = _str_from_dtype(dtype)
    tensor_type = NAME2CPU_TENSOR_TYPE[dtype]
    return x.type(tensor_type)
