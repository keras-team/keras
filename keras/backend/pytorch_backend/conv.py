from ..common import image_data_format
from .shape import reshape
from .variable import ndim


def bias_add(x, bias, data_format=None):
    if data_format is None:
        data_format = image_data_format()
    assert data_format in {'channels_first', 'channels_last'}
    assert not (ndim(bias) != 1 and ndim(bias) != ndim(x) - 1)
    bias_shape = bias.size()
    if ndim(x) == 5:
        if data_format == 'channels_first':
            if ndim(bias) == 1:
                x += reshape(bias, (1, bias_shape[0], 1, 1, 1))
            else:
                x += reshape(bias, (1, bias_shape[3]) + bias_shape[:3])
        elif data_format == 'channels_last':
            if ndim(bias) == 1:
                x += reshape(bias, (1, 1, 1, 1, bias_shape[0]))
            else:
                x += reshape(bias, (1,) + bias_shape)
    elif ndim(x) == 4:
        if data_format == 'channels_first':
            if ndim(bias) == 1:
                x += reshape(bias, (1, bias_shape[0], 1, 1))
            else:
                x += reshape(bias, (1, bias_shape[2]) + bias_shape[:2])
        elif data_format == 'channels_last':
            if ndim(bias) == 1:
                x += reshape(bias, (1, 1, 1, bias_shape[0]))
            else:
                x += reshape(bias, (1,) + bias_shape)
    elif ndim(x) == 3:
        if data_format == 'channels_first':
            if ndim(bias) == 1:
                x += reshape(bias, (1, bias_shape[0], 1))
            else:
                x += reshape(bias, (1, bias_shape[1], bias_shape[0]))
        elif data_format == 'channels_last':
            if ndim(bias) == 1:
                x += reshape(bias, (1, 1, bias_shape[0]))
            else:
                x += reshape(bias, (1,) + bias_shape)
    else:
        x += bias
    return x
