import numpy as np
from keras.utils.generic_utils import transpose_shape
from keras.utils import to_categorical


def int_shape(x):
    return x.shape


def ndim(x):
    return x.ndim


def get_variable_shape(x):
    return int_shape(x)


def concatenate(tensors, axis=-1):
    return np.concatenate(tensors, axis)


def reshape(x, shape):
    return np.reshape(x, shape)


def permute_dimensions(x, pattern):
    return np.transpose(x, pattern)


def repeat_elements(x, rep, axis):
    return np.repeat(x, rep, axis=axis)


def resize_images(x, height_factor, width_factor, data_format):
    if data_format == 'channels_first':
        x = repeat_elements(x, height_factor, axis=2)
        x = repeat_elements(x, width_factor, axis=3)
    elif data_format == 'channels_last':
        x = repeat_elements(x, height_factor, axis=1)
        x = repeat_elements(x, width_factor, axis=2)
    return x


def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    if data_format == 'channels_first':
        x = repeat_elements(x, depth_factor, axis=2)
        x = repeat_elements(x, height_factor, axis=3)
        x = repeat_elements(x, width_factor, axis=4)
    elif data_format == 'channels_last':
        x = repeat_elements(x, depth_factor, axis=1)
        x = repeat_elements(x, height_factor, axis=2)
        x = repeat_elements(x, width_factor, axis=3)
    return x


def repeat(x, n):
    y = np.expand_dims(x, 1)
    y = np.repeat(y, n, axis=1)
    return y


def arange(start, stop=None, step=1, dtype='int32'):
    return np.arange(start, stop, step, dtype)


def tile(x, n):
    return np.tile(x, n)


def flatten(x):
    return np.reshape(x, (-1,))


def batch_flatten(x):
    return np.reshape(x, (x.shape[0], -1))


def temporal_padding(x, padding=(1, 1)):
    return np.pad(x, [(0, 0), padding, (0, 0)], mode='constant')


def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    all_dims_padding = ((0, 0),) + padding + ((0, 0),)
    all_dims_padding = transpose_shape(all_dims_padding, data_format,
                                       spatial_axes=(1, 2))
    return np.pad(x, all_dims_padding, mode='constant')


def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    all_dims_padding = ((0, 0),) + padding + ((0, 0),)
    all_dims_padding = transpose_shape(all_dims_padding, data_format,
                                       spatial_axes=(1, 2, 3))
    return np.pad(x, all_dims_padding, mode='constant')


def one_hot(indices, num_classes):
    return to_categorical(indices, num_classes)


def reverse(x, axes):
    if isinstance(axes, list):
        axes = tuple(axes)
    return np.flip(x, axes)


expand_dims = np.expand_dims
squeeze = np.squeeze
