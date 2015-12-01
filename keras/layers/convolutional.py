# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .. import backend as K
from .. import activations, initializations, regularizers, constraints
from ..layers.core import Layer


def conv_output_length(input_length, filter_size, border_mode, stride):
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - filter_size + 1
    return (output_length + stride - 1) // stride


class Convolution1D(Layer):
    input_ndim = 3

    def __init__(self, nb_filter, filter_length,
                 init='uniform', activation='linear', weights=None,
                 border_mode='valid', subsample_length=1,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 input_dim=None, input_length=None, **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution1D:', border_mode)
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample_length = subsample_length

        self.subsample = (subsample_length, 1)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        self.input = K.placeholder(ndim=3)
        super(Convolution1D, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.W_shape = (self.nb_filter, input_dim, self.filter_length, 1)
        self.W = self.init(self.W_shape)
        self.b = K.zeros((self.nb_filter,))
        self.params = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        length = conv_output_length(self.input_shape[1],
                                    self.filter_length,
                                    self.border_mode,
                                    self.subsample[0])
        return (self.input_shape[0], length, self.nb_filter)

    def get_output(self, train=False):
        X = self.get_input(train)
        X = K.expand_dims(X, -1)  # add a dimension of the right
        X = K.permute_dimensions(X, (0, 2, 1, 3))
        conv_out = K.conv2d(X, self.W, strides=self.subsample,
                            border_mode=self.border_mode, dim_ordering='th')

        output = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        output = self.activation(output)
        output = K.squeeze(output, 3)  # remove the dummy 3rd dimension
        output = K.permute_dimensions(output, (0, 2, 1))
        return output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "nb_filter": self.nb_filter,
                  "filter_length": self.filter_length,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__,
                  "border_mode": self.border_mode,
                  "subsample_length": self.subsample_length,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                  "b_constraint": self.b_constraint.get_config() if self.b_constraint else None,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(Convolution1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Convolution2D(Layer):
    input_ndim = 4

    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering='th',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights
        self.input = K.placeholder(ndim=4)
        super(Convolution2D, self).__init__(**kwargs)

    def build(self):
        if self.dim_ordering == 'th':
            stack_size = self.input_shape[1]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = self.input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        self.W = self.init(self.W_shape)
        self.b = K.zeros((self.nb_filter,))
        self.params = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_output(self, train=False):
        X = self.get_input(train)
        conv_out = K.conv2d(X, self.W, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering)

        output = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        output = self.activation(output)
        return output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "nb_filter": self.nb_filter,
                  "nb_row": self.nb_row,
                  "nb_col": self.nb_col,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__,
                  "border_mode": self.border_mode,
                  "subsample": self.subsample,
                  "dim_ordering": self.dim_ordering,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                  "b_constraint": self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(Convolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPooling1D(Layer):
    input_ndim = 3

    def __init__(self, pool_length=2, stride=None,
                 border_mode='valid', **kwargs):
        super(MaxPooling1D, self).__init__(**kwargs)
        if stride is None:
            stride = pool_length
        self.pool_length = pool_length
        self.stride = stride
        self.st = (self.stride, 1)
        self.input = K.placeholder(ndim=3)
        self.pool_size = (pool_length, 1)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode

    @property
    def output_shape(self):
        input_shape = self.input_shape
        length = conv_output_length(input_shape[1], self.pool_length,
                                    self.border_mode, self.stride)
        return (input_shape[0], length, input_shape[2])

    def get_output(self, train=False):
        X = self.get_input(train)
        X = K.expand_dims(X, -1)   # add dummy last dimension
        X = K.permute_dimensions(X, (0, 2, 1, 3))
        output = K.maxpool2d(X, pool_size=self.pool_size, strides=self.st,
                             border_mode=self.border_mode,
                             dim_ordering='th')
        output = K.permute_dimensions(output, (0, 2, 1, 3))
        return K.squeeze(output, 3)  # remove dummy last dimension

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "stride": self.stride,
                  "pool_length": self.pool_length,
                  "border_mode": self.border_mode}
        base_config = super(MaxPooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPooling2D(Layer):
    input_ndim = 4

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(MaxPooling2D, self).__init__(**kwargs)
        self.input = K.placeholder(ndim=4)
        self.pool_size = tuple(pool_size)
        if strides is None:
            strides = self.pool_size
        self.strides = tuple(strides)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.pool_size[0],
                                  self.border_mode, self.strides[0])
        cols = conv_output_length(cols, self.pool_size[1],
                                  self.border_mode, self.strides[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = K.maxpool2d(X, pool_size=self.pool_size,
                             strides=self.strides,
                             border_mode=self.border_mode,
                             dim_ordering=self.dim_ordering)
        return output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "pool_size": self.pool_size,
                  "border_mode": self.border_mode,
                  "strides": self.strides,
                  "dim_ordering": self.dim_ordering}
        base_config = super(MaxPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UpSampling1D(Layer):
    input_ndim = 3

    def __init__(self, length=2, **kwargs):
        super(UpSampling1D, self).__init__(**kwargs)
        self.length = length
        self.input = K.placeholder(ndim=3)

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], self.length * input_shape[1], input_shape[2])

    def get_output(self, train=False):
        X = self.get_input(train)
        output = K.concatenate([X] * self.length, axis=1)
        return output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "length": self.length}
        base_config = super(UpSampling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UpSampling2D(Layer):
    input_ndim = 4

    def __init__(self, size=(2, 2), dim_ordering='th', **kwargs):
        super(UpSampling2D, self).__init__(**kwargs)
        self.input = K.placeholder(ndim=4)
        self.size = tuple(size)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            return (input_shape[0],
                    input_shape[1],
                    self.size[0] * input_shape[2],
                    self.size[1] * input_shape[3])
        elif self.dim_ordering == 'tf':
            return (input_shape[0],
                    self.size[0] * input_shape[1],
                    self.size[1] * input_shape[2],
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.dim_ordering == 'th':
            output = K.concatenate([X] * self.size[0], axis=2)
            output = K.concatenate([output] * self.size[1], axis=3)
        elif self.dim_ordering == 'tf':
            output = K.concatenate([X] * self.size[0], axis=1)
            output = K.concatenate([output] * self.size[1], axis=2)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        return output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "size": self.size}
        base_config = super(UpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ZeroPadding1D(Layer):
    """Zero-padding layer for 1D input (e.g. temporal sequence).

    Input shape
    -----------
    3D tensor with shape (samples, axis_to_pad, features)

    Output shape
    ------------
    3D tensor with shape (samples, padded_axis, features)

    Arguments
    ---------
    padding: int
        How many zeros to add at the beginning and end of
        the padding dimension (axis 1).
    """
    input_ndim = 3

    def __init__(self, padding=1, **kwargs):
        super(ZeroPadding1D, self).__init__(**kwargs)
        self.padding = padding
        self.input = K.placeholder(ndim=3)

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0],
                input_shape[1] + self.padding * 2,
                input_shape[2])

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.temporal_padding(X, padding=self.padding)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "padding": self.padding}
        base_config = super(ZeroPadding1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ZeroPadding2D(Layer):
    """Zero-padding layer for 2D input (e.g. picture).

    Input shape
    -----------
    4D tensor with shape:
        (samples, depth, first_axis_to_pad, second_axis_to_pad)

    Output shape
    ------------
    4D tensor with shape:
        (samples, depth, first_padded_axis, second_padded_axis)

    Arguments
    ---------
    padding: tuple of int (length 2)
        How many zeros to add at the beginning and end of
        the 2 padding dimensions (axis 3 and 4).
    """
    input_ndim = 4

    def __init__(self, padding=(1, 1), dim_ordering='th', **kwargs):
        super(ZeroPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)
        self.input = K.placeholder(ndim=4)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            return (input_shape[0],
                    input_shape[1],
                    input_shape[2] + 2 * self.padding[0],
                    input_shape[3] + 2 * self.padding[1])
        elif self.dim_ordering == 'tf':
            return (input_shape[0],
                    input_shape[1] + 2 * self.padding[0],
                    input_shape[2] + 2 * self.padding[1],
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.spatial_2d_padding(X, padding=self.padding,
                                    dim_ordering=self.dim_ordering)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "padding": self.padding}
        base_config = super(ZeroPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
