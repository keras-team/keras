import numpy as np

from .variable import cast, ndim


def concatenate(tensors, axis=-1):
    return torch.cat(tensors, axis)


def reshape(x, shape):
    return x.view(*shape)


def permute_dimensions(x, pattern):
    return x.permute(pattern)


def resize_images(x, height_factor, width_factor, data_format):
    raise NotImplementedError


def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    raise NotImplementedError


def repeat_elements(x, rep, axis):
    sizes = [1] * ndim(x)
    sizes[axis] = rep
    return x.repeat(*sizes)


def repeat(x, n):
    assert ndim(x) == 2
    x = x.unsqueeze(1)
    return repeat_elements(x, n, 1)


def repeat_to_shape(x, new_shape):
    old_shape = x.size()
    assert len(old_shape) == len(new_shape)
    multiples = []
    for i in range(len(old_shape)):
        assert not new_shape[i] % old_shape[i]
        mul = new_shape[i] // old_shape[i]
        multiples.append(mul)
    return x.repeat(*multiples)


def arange(start, stop=None, step=1, dtype='int32'):
    if stop is None and start < 0:
        start = 0

    result = torch.arange(start, stop, step)
    return cast(result, dtype)


def tile(x, n):
    if isinstance(n, int):
        n = [n]
    raise NotImplementedError


def flatten(x):
    return x.view(-1)


def batch_flatten(x):
    return x.view(x.size()[0], -1)


def expand_dims(x, axis=-1):
    return x.unsqueeze(axis)


def squeeze(x, axis):
    if isinstance(axis, int):
        return x.squeeze(axis)
    else:
        for dim in axis:
            x = x.squeeze(dim)
        return x


def _padded(x, pattern):
    assert len(pattern) == x.dim() - 1
    for pair in pattern:
        a, b = pair
        assert 0 <= a
        assert 0 <= b
    for i, pair in enumerate(pattern):
        if pair == (0, 0):
            continue
        dim = i + 1
        top, bottom = pair
        to_cat = []
        if top:
            shape = list(x.size())
            shape[dim] = top
            to_cat.append(torch.zeros(shape))
        to_cat.append(x)
        if bottom:
            shape = list(x.size())
            shape[dim] = bottom
            to_cat.append(torch.zeros(shape))
        x = torch.cat(to_cat, dim)
    return x


def temporal_padding(x, padding=(1, 1)):
    assert len(padding) == 2
    pattern = [
        (0, 0),
        padding,
        (0, 0),
    ]
    return _padded(x, pattern)


def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    assert len(padding) == 2
    for pair in padding:
        assert len(pair) == 2
    if data_format is None:
        data_format = image_data_format()
    if data_format == 'channels_first':
        pattern = [
            (0, 0),
            (0, 0),
            padding[0],
            padding[1],
        ]
    elif data_format == 'channels_last':
        pattern = [
            (0, 0),
            padding[0],
            padding[1],
            (0, 0),
        ]
    else:
        assert False
    return _padded(x, pattern)


def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    assert len(padding) == 2
    for pair in padding:
        assert len(pair) == 2
    if data_format is None:
        data_format = image_data_format()
    if data_format == 'channels_first':
        pattern = [
            (0, 0),
            (0, 0),
            padding[0],
            padding[1],
            padding[2],
        ]
    elif data_format == 'channels_last':
        pattern = [
            (0, 0),
            padding[0],
            padding[1],
            padding[2],
            (0, 0),
        ]
    else:
        assert False
    return _padded(x, pattern)


def stack(x, axis=0):
    raise NotImplementedError


def one_hot(indices, num_classes):
    raise NotImplementedError


def reverse(x, axes):
    raise NotImplementedError
