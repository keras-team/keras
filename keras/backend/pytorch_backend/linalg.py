from .variable import ndim


def dot(x, y):
    return x @ y


def batch_dot(x, y, axes=None):
    raise NotImplementedError


def transpose(x):
    num_dims = ndim(x)
    if num_dims == 1:
        x = x.unsqueeze(1)
    x.tranpose_(0, 1)
    return x


def gather(reference, indices):
    raise NotImplementedError
