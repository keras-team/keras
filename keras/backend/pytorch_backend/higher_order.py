from functools import reduce
from torch import Tensor


def map_fn(fn, elems, name=None, dtype=None):
    x = list(map(fn, elems))
    x = torch.cat(x)
    return Tensor(x).type(dtype)


def foldl(fn, elems, initializer=None, name=None):
    x = reduce(fn, elems, initializer)
    return Tensor(x).type(dype)


def foldr(fn, elems, initializer=None, name=None):
    x = initializer
    for elem in reversed(elems):
        x = fn(elem, x)
    return Tensor(x).type(dtype)


def local_conv1d(inputs, kernel, kernel_size, strides, data_format=None):
    raise NotImplementedError


def local_conv2d(inputs, kernel, kernel_size, strides, output_shape,
                 data_format=None):
    raise NotImplementedError
