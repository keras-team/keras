# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .. import backend as K
from ..engine import Layer
from ..engine import InputSpec
from ..utils.np_utils import conv_output_length


class _Pooling1D(Layer):
    """Abstract class for different pooling 1D layers.
    """

    def __init__(self, pool_length=2, stride=None,
                 border_mode='valid', **kwargs):
        super(_Pooling1D, self).__init__(**kwargs)
        if stride is None:
            stride = pool_length
        self.pool_length = pool_length
        self.stride = stride
        self.st = (self.stride, 1)
        self.pool_size = (pool_length, 1)
        if border_mode not in {'valid', 'same'}:
            raise ValueError('`border_mode` must be in {valid, same}.')
        self.border_mode = border_mode
        self.input_spec = [InputSpec(ndim=3)]

    def get_output_shape_for(self, input_shape):
        length = conv_output_length(input_shape[1], self.pool_length,
                                    self.border_mode, self.stride)
        return (input_shape[0], length, input_shape[2])

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        raise NotImplementedError

    def call(self, x, mask=None):
        x = K.expand_dims(x, 2)   # add dummy last dimension
        output = self._pooling_function(inputs=x, pool_size=self.pool_size,
                                        strides=self.st,
                                        border_mode=self.border_mode,
                                        dim_ordering='tf')
        return K.squeeze(output, 2)  # remove dummy last dimension

    def get_config(self):
        config = {'stride': self.stride,
                  'pool_length': self.pool_length,
                  'border_mode': self.border_mode}
        base_config = super(_Pooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPooling1D(_Pooling1D):
    """Max pooling operation for temporal data.

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.

    # Output shape
        3D tensor with shape: `(samples, downsampled_steps, features)`.

    # Arguments
        pool_length: size of the region to which max pooling is applied
        stride: integer, or None. factor by which to downscale.
            2 will halve the input.
            If None, it will default to `pool_length`.
        border_mode: 'valid' or 'same'.
    """

    def __init__(self, pool_length=2, stride=None,
                 border_mode='valid', **kwargs):
        super(MaxPooling1D, self).__init__(pool_length, stride,
                                           border_mode, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='max')
        return output


class AveragePooling1D(_Pooling1D):
    """Average pooling for temporal data.

    # Arguments
        pool_length: factor by which to downscale. 2 will halve the input.
        stride: integer, or None. Stride value.
            If None, it will default to `pool_length`.
        border_mode: 'valid' or 'same'.

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.

    # Output shape
        3D tensor with shape: `(samples, downsampled_steps, features)`.
    """

    def __init__(self, pool_length=2, stride=None,
                 border_mode='valid', **kwargs):
        super(AveragePooling1D, self).__init__(pool_length, stride,
                                               border_mode, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='avg')
        return output


class _Pooling2D(Layer):
    """Abstract class for different pooling 2D layers.
    """

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='default', **kwargs):
        super(_Pooling2D, self).__init__(**kwargs)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.pool_size = tuple(pool_size)
        if strides is None:
            strides = self.pool_size
        self.strides = tuple(strides)
        if border_mode not in {'valid', 'same'}:
            raise ValueError('`border_mode` must be in {valid, same}.')
        self.border_mode = border_mode
        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('`dim_ordering` must be in {tf, th}.')
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

        rows = conv_output_length(rows, self.pool_size[0],
                                  self.border_mode, self.strides[0])
        cols = conv_output_length(cols, self.pool_size[1],
                                  self.border_mode, self.strides[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, input_shape[3])

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        raise NotImplementedError

    def call(self, x, mask=None):
        output = self._pooling_function(inputs=x,
                                        pool_size=self.pool_size,
                                        strides=self.strides,
                                        border_mode=self.border_mode,
                                        dim_ordering=self.dim_ordering)
        return output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'border_mode': self.border_mode,
                  'strides': self.strides,
                  'dim_ordering': self.dim_ordering}
        base_config = super(_Pooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPooling2D(_Pooling2D):
    """Max pooling operation for spatial data.

    # Arguments
        pool_size: tuple of 2 integers,
            factors by which to downscale (vertical, horizontal).
            (2, 2) will halve the image in each dimension.
        strides: tuple of 2 integers, or None. Strides values.
            If None, it will default to `pool_size`.
        border_mode: 'valid' or 'same'.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.
    """

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='default', **kwargs):
        super(MaxPooling2D, self).__init__(pool_size, strides, border_mode,
                                           dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering,
                          pool_mode='max')
        return output


class AveragePooling2D(_Pooling2D):
    """Average pooling operation for spatial data.

    # Arguments
        pool_size: tuple of 2 integers,
            factors by which to downscale (vertical, horizontal).
            (2, 2) will halve the image in each dimension.
        strides: tuple of 2 integers, or None. Strides values.
            If None, it will default to `pool_size`.
        border_mode: 'valid' or 'same'.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.
    """

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='default', **kwargs):
        super(AveragePooling2D, self).__init__(pool_size, strides, border_mode,
                                               dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='avg')
        return output


class _Pooling3D(Layer):
    """Abstract class for different pooling 3D layers.
    """

    def __init__(self, pool_size=(2, 2, 2), strides=None, border_mode='valid',
                 dim_ordering='default', **kwargs):
        super(_Pooling3D, self).__init__(**kwargs)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.pool_size = tuple(pool_size)
        if strides is None:
            strides = self.pool_size
        self.strides = tuple(strides)
        if border_mode not in {'valid', 'same'}:
            raise ValueError('`border_mode` must be in {valid, same}.')
        self.border_mode = border_mode
        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('`dim_ordering` must be in {tf, th}.')
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=5)]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            len_dim1 = input_shape[2]
            len_dim2 = input_shape[3]
            len_dim3 = input_shape[4]
        elif self.dim_ordering == 'tf':
            len_dim1 = input_shape[1]
            len_dim2 = input_shape[2]
            len_dim3 = input_shape[3]
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

        len_dim1 = conv_output_length(len_dim1, self.pool_size[0],
                                      self.border_mode, self.strides[0])
        len_dim2 = conv_output_length(len_dim2, self.pool_size[1],
                                      self.border_mode, self.strides[1])
        len_dim3 = conv_output_length(len_dim3, self.pool_size[2],
                                      self.border_mode, self.strides[2])
        if self.dim_ordering == 'th':
            return (input_shape[0],
                    input_shape[1],
                    len_dim1, len_dim2, len_dim3)
        elif self.dim_ordering == 'tf':
            return (input_shape[0],
                    len_dim1, len_dim2, len_dim3,
                    input_shape[4])

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        raise NotImplementedError

    def call(self, x, mask=None):
        output = self._pooling_function(inputs=x, pool_size=self.pool_size,
                                        strides=self.strides,
                                        border_mode=self.border_mode,
                                        dim_ordering=self.dim_ordering)
        return output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'border_mode': self.border_mode,
                  'strides': self.strides,
                  'dim_ordering': self.dim_ordering}
        base_config = super(_Pooling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPooling3D(_Pooling3D):
    """Max pooling operation for 3D data (spatial or spatio-temporal).

    # Arguments
        pool_size: tuple of 3 integers,
            factors by which to downscale (dim1, dim2, dim3).
            (2, 2, 2) will halve the size of the 3D input in each dimension.
        strides: tuple of 3 integers, or None. Strides values.
        border_mode: 'valid' or 'same'.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 4.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".

    # Input shape
        5D tensor with shape:
        `(samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, len_pool_dim1, len_pool_dim2, len_pool_dim3, channels)` if dim_ordering='tf'.

    # Output shape
        5D tensor with shape:
        `(nb_samples, channels, pooled_dim1, pooled_dim2, pooled_dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, pooled_dim1, pooled_dim2, pooled_dim3, channels)` if dim_ordering='tf'.
    """

    def __init__(self, pool_size=(2, 2, 2), strides=None, border_mode='valid',
                 dim_ordering='default', **kwargs):
        super(MaxPooling3D, self).__init__(pool_size, strides, border_mode,
                                           dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool3d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='max')
        return output


class AveragePooling3D(_Pooling3D):
    """Average pooling operation for 3D data (spatial or spatio-temporal).

    # Arguments
        pool_size: tuple of 3 integers,
            factors by which to downscale (dim1, dim2, dim3).
            (2, 2, 2) will halve the size of the 3D input in each dimension.
        strides: tuple of 3 integers, or None. Strides values.
        border_mode: 'valid' or 'same'.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 4.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".

    # Input shape
        5D tensor with shape:
        `(samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, len_pool_dim1, len_pool_dim2, len_pool_dim3, channels)` if dim_ordering='tf'.

    # Output shape
        5D tensor with shape:
        `(nb_samples, channels, pooled_dim1, pooled_dim2, pooled_dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, pooled_dim1, pooled_dim2, pooled_dim3, channels)` if dim_ordering='tf'.
    """

    def __init__(self, pool_size=(2, 2, 2), strides=None, border_mode='valid',
                 dim_ordering='default', **kwargs):
        super(AveragePooling3D, self).__init__(pool_size, strides, border_mode,
                                               dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool3d(inputs, pool_size, strides,
                          border_mode, dim_ordering,
                          pool_mode='avg')
        return output


class _GlobalPooling1D(Layer):
    """Abstract class for different global pooling 1D layers.
    """

    def __init__(self, **kwargs):
        super(_GlobalPooling1D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3)]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        raise NotImplementedError


class GlobalAveragePooling1D(_GlobalPooling1D):
    """Global average pooling operation for temporal data.

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.

    # Output shape
        2D tensor with shape: `(samples, features)`.
    """

    def call(self, x, mask=None):
        return K.mean(x, axis=1)


class GlobalMaxPooling1D(_GlobalPooling1D):
    """Global max pooling operation for temporal data.

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.

    # Output shape
        2D tensor with shape: `(samples, features)`.
    """

    def call(self, x, mask=None):
        return K.max(x, axis=1)


class _GlobalPooling2D(Layer):
    """Abstract class for different global pooling 2D layers.
    """

    def __init__(self, dim_ordering='default', **kwargs):
        super(_GlobalPooling2D, self).__init__(**kwargs)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'tf':
            return (input_shape[0], input_shape[3])
        else:
            return (input_shape[0], input_shape[1])

    def call(self, x, mask=None):
        raise NotImplementedError

    def get_config(self):
        config = {'dim_ordering': self.dim_ordering}
        base_config = super(_GlobalPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalAveragePooling2D(_GlobalPooling2D):
    """Global average pooling operation for spatial data.

    # Arguments
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        2D tensor with shape:
        `(nb_samples, channels)`
    """

    def call(self, x, mask=None):
        if self.dim_ordering == 'tf':
            return K.mean(x, axis=[1, 2])
        else:
            return K.mean(x, axis=[2, 3])


class GlobalMaxPooling2D(_GlobalPooling2D):
    """Global max pooling operation for spatial data.

    # Arguments
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        2D tensor with shape:
        `(nb_samples, channels)`
    """

    def call(self, x, mask=None):
        if self.dim_ordering == 'tf':
            return K.max(x, axis=[1, 2])
        else:
            return K.max(x, axis=[2, 3])


class _GlobalPooling3D(Layer):
    """Abstract class for different global pooling 3D layers.
    """

    def __init__(self, dim_ordering='default', **kwargs):
        super(_GlobalPooling3D, self).__init__(**kwargs)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=5)]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'tf':
            return (input_shape[0], input_shape[4])
        else:
            return (input_shape[0], input_shape[1])

    def call(self, x, mask=None):
        raise NotImplementedError

    def get_config(self):
        config = {'dim_ordering': self.dim_ordering}
        base_config = super(_GlobalPooling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalAveragePooling3D(_GlobalPooling3D):
    """Global Average pooling operation for 3D data.

    # Arguments
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 4.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".

    # Input shape
        5D tensor with shape:
        `(samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, len_pool_dim1, len_pool_dim2, len_pool_dim3, channels)` if dim_ordering='tf'.

    # Output shape
        2D tensor with shape:
        `(nb_samples, channels)`
    """

    def call(self, x, mask=None):
        if self.dim_ordering == 'tf':
            return K.mean(x, axis=[1, 2, 3])
        else:
            return K.mean(x, axis=[2, 3, 4])


class GlobalMaxPooling3D(_GlobalPooling3D):
    """Global Max pooling operation for 3D data.

    # Arguments
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 4.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".

    # Input shape
        5D tensor with shape:
        `(samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, len_pool_dim1, len_pool_dim2, len_pool_dim3, channels)` if dim_ordering='tf'.

    # Output shape
        2D tensor with shape:
        `(nb_samples, channels)`
    """

    def call(self, x, mask=None):
        if self.dim_ordering == 'tf':
            return K.max(x, axis=[1, 2, 3])
        else:
            return K.max(x, axis=[2, 3, 4])
