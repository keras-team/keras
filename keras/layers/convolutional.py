# -*- coding: utf-8 -*-
from __future__ import absolute_import

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv3d2d
from .. import activations, initializations, regularizers, constraints
from ..utils.theano_utils import shared_zeros, on_gpu
from ..layers.core import Layer

if on_gpu():
    from theano.sandbox.cuda import dnn


def conv_output_length(input_length, filter_size, border_mode, stride):
    if input_length is None:
        return None
    assert border_mode in {'same', 'full', 'valid'}
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'full':
        output_length = input_length + filter_size - 1
    elif border_mode == 'valid':
        output_length = input_length - filter_size + 1
    return (output_length + stride - 1) // stride


def pool_output_length(input_length, pool_size, ignore_border, stride):
    if input_length is None:
        return None
    if ignore_border:
        output_length = input_length - pool_size + 1
        output_length = (output_length + stride - 1) // stride
    else:
        if pool_size == input_length:
            output_length = min(input_length, stride - stride % 2)
            if output_length <= 0:
                output_length = 1
        elif stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = (input_length - pool_size + stride - 1) // stride
            if output_length <= 0:
                output_length = 1
            else:
                output_length += 1
    return output_length


class Convolution1D(Layer):
    input_ndim = 3

    def __init__(self, nb_filter, filter_length,
                 init='uniform', activation='linear', weights=None,
                 border_mode='valid', subsample_length=1,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, input_length=None, **kwargs):

        if border_mode not in {'valid', 'full', 'same'}:
            raise Exception('Invalid border mode for Convolution1D:', border_mode)
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
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
        super(Convolution1D, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()
        self.W_shape = (self.nb_filter, input_dim, self.filter_length, 1)
        self.W = self.init(self.W_shape)
        self.b = shared_zeros((self.nb_filter,))
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
        length = conv_output_length(self.input_shape[1], self.filter_length, self.border_mode, self.subsample[0])
        return (self.input_shape[0], length, self.nb_filter)

    def get_output(self, train=False):
        X = self.get_input(train)
        X = T.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1)).dimshuffle(0, 2, 1, 3)

        border_mode = self.border_mode
        if on_gpu() and dnn.dnn_available():
            if border_mode == 'same':
                assert(self.subsample_length == 1)
                pad_x = (self.filter_length - self.subsample_length) // 2
                conv_out = dnn.dnn_conv(img=X,
                                        kerns=self.W,
                                        border_mode=(pad_x, 0))
            else:
                conv_out = dnn.dnn_conv(img=X,
                                        kerns=self.W,
                                        border_mode=border_mode,
                                        subsample=self.subsample)
        else:
            if border_mode == 'same':
                assert(self.subsample_length == 1)
                border_mode = 'full'

            input_shape = self.input_shape
            image_shape = (input_shape[0], input_shape[2], input_shape[1], 1)
            conv_out = T.nnet.conv.conv2d(X, self.W,
                                          border_mode=border_mode,
                                          subsample=self.subsample,
                                          image_shape=image_shape,
                                          filter_shape=self.W_shape)
            if self.border_mode == 'same':
                shift_x = (self.filter_length - 1) // 2
                conv_out = conv_out[:, :, shift_x:X.shape[2] + shift_x, :]

        output = self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        output = T.reshape(output, (output.shape[0], output.shape[1], output.shape[2])).dimshuffle(0, 2, 1)
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
                 border_mode='valid', subsample=(1, 1),
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, **kwargs):

        if border_mode not in {'valid', 'full', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.border_mode = border_mode
        self.subsample = tuple(subsample)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights
        super(Convolution2D, self).__init__(**kwargs)

    def build(self):
        stack_size = self.input_shape[1]
        self.input = T.tensor4()
        self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        self.W = self.init(self.W_shape)
        self.b = shared_zeros((self.nb_filter,))
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
        rows = input_shape[2]
        cols = input_shape[3]
        rows = conv_output_length(rows, self.nb_row, self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col, self.border_mode, self.subsample[1])
        return (input_shape[0], self.nb_filter, rows, cols)

    def get_output(self, train=False):
        X = self.get_input(train)
        border_mode = self.border_mode
        if on_gpu() and dnn.dnn_available():
            if border_mode == 'same':
                assert(self.subsample == (1, 1))
                pad_x = (self.nb_row - self.subsample[0]) // 2
                pad_y = (self.nb_col - self.subsample[1]) // 2
                conv_out = dnn.dnn_conv(img=X,
                                        kerns=self.W,
                                        border_mode=(pad_x, pad_y))
            else:
                conv_out = dnn.dnn_conv(img=X,
                                        kerns=self.W,
                                        border_mode=border_mode,
                                        subsample=self.subsample)
        else:
            if border_mode == 'same':
                border_mode = 'full'
                assert(self.subsample == (1, 1))

            conv_out = T.nnet.conv.conv2d(X, self.W,
                                          border_mode=border_mode,
                                          subsample=self.subsample,
                                          image_shape=self.input_shape,
                                          filter_shape=self.W_shape)
            if self.border_mode == 'same':
                shift_x = (self.nb_row - 1) // 2
                shift_y = (self.nb_col - 1) // 2
                conv_out = conv_out[:, :, shift_x:X.shape[2] + shift_x, shift_y:X.shape[3] + shift_y]

        return self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "nb_filter": self.nb_filter,
                  "nb_row": self.nb_row,
                  "nb_col": self.nb_col,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__,
                  "border_mode": self.border_mode,
                  "subsample": self.subsample,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                  "b_constraint": self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(Convolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Convolution3D(Layer):
    input_ndim = 5

    def __init__(self, nb_filter, nb_depth, nb_row, nb_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1, 1),
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, **kwargs):

        if border_mode not in {'valid', 'full', 'same'}:
            raise Exception('Invalid border mode for Convolution3D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_depth = nb_depth
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.border_mode = border_mode
        self.subsample = tuple(subsample)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights
        super(Convolution3D, self).__init__(**kwargs)

    def build(self):
        stack_size = self.input_shape[1]
        dtensor5 = T.TensorType('float32', (0,)*5)
        self.input = dtensor5()
        self.W_shape = (self.nb_filter, stack_size, self.nb_depth, self.nb_row, self.nb_col)
        self.W = self.init(self.W_shape)
        self.b = shared_zeros((self.nb_filter,))
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
        depth = input_shape[2]
        rows = input_shape[3]
        cols = input_shape[4]
        depth = conv_output_length(depth, self.nb_depth, self.border_mode, self.subsample[0])
        rows = conv_output_length(rows, self.nb_row, self.border_mode, self.subsample[1])
        cols = conv_output_length(cols, self.nb_col, self.border_mode, self.subsample[2])
        return (input_shape[0], self.nb_filter, depth, rows, cols)

    def get_output(self, train):
        X = self.get_input(train)
        border_mode = self.border_mode

        # Both conv3d2d.conv3d and nnet.conv3D only support the 'valid' border mode
        if border_mode != 'valid':
            if border_mode == 'same':
                assert(self.subsample == (1, 1, 1))
                pad_z = (self.nb_depth - self.subsample[0])
                pad_x = (self.nb_row - self.subsample[1])
                pad_y = (self.nb_col - self.subsample[2])
            else: #full
                pad_z = (self.nb_depth - 1) * 2
                pad_x = (self.nb_row - 1) * 2
                pad_y = (self.nb_col - 1) * 2

            input_shape = X.shape
            output_shape = (input_shape[0], input_shape[1],
                            input_shape[2] + pad_z,
                            input_shape[3] + pad_x,
                            input_shape[4] + pad_y)
            output = T.zeros(output_shape)
            indices = (slice(None), slice(None),
                       slice(pad_z//2, input_shape[2] + pad_z//2),
                       slice(pad_x//2, input_shape[3] + pad_x//2),
                       slice(pad_y//2, input_shape[4] + pad_y//2))
            X = T.set_subtensor(output[indices], X)


        border_mode = 'valid'

        if on_gpu():
            # Shuffle the dimensions as per the input parameter order, restore it once done
            W_shape = (self.W_shape[0], self.W_shape[2], self.W_shape[1],
                       self.W_shape[3],self.W_shape[4])

            conv_out = conv3d2d.conv3d(signals=X.dimshuffle(0, 2, 1, 3, 4),
                                       filters=self.W.dimshuffle(0, 2, 1, 3, 4),
                                       filters_shape=W_shape,
                                       border_mode=border_mode)

            conv_out = conv_out.dimshuffle(0, 2, 1, 3, 4)
            self.W = self.W.dimshuffle(0, 2, 1, 3, 4)
        else:
            # Shuffle the dimensions as per the input parameter order, restore it once done
            self.W = self.W.dimshuffle(0, 2, 3, 4 , 1)
            conv_out = T.nnet.conv3D(V=X.dimshuffle(0, 2, 3, 4, 1),
                                     W=self.W,
                                     b=self.b, d=self.subsample)
            conv_out = conv_out.dimshuffle(0, 4, 1, 2, 3)
            self.W = self.W.dimshuffle(0, 4, 1, 2, 3)

        output = self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x', 'x'))
        return output

    def get_config(self):
          return {"name": self.__class__.__name__,
                   "nb_filter": self.nb_filter,
                   "stack_size": self.stack_size,
                   "nb_depth": self.nb_depth,
                   "nb_row": self.nb_row,
                   "nb_col": self.nb_col,
                   "init": self.init.__name__,
                   "activation": self.activation.__name__,
                   "border_mode": self.border_mode,
                   "subsample": self.subsample,
                   "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                   "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                   "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                   "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                   "b_constraint": self.b_constraint.get_config() if self.b_constraint else None}


class MaxPooling1D(Layer):
    input_ndim = 3

    def __init__(self, pool_length=2, stride=None, ignore_border=True, **kwargs):
        super(MaxPooling1D, self).__init__(**kwargs)
        if stride is None:
            stride = pool_length
        self.pool_length = pool_length
        self.stride = stride
        self.st = (self.stride, 1)

        self.input = T.tensor3()
        self.pool_size = (pool_length, 1)
        self.ignore_border = ignore_border

    @property
    def output_shape(self):
        input_shape = self.input_shape
        length = pool_output_length(input_shape[1], self.pool_length, self.ignore_border, self.stride)
        return (input_shape[0], length, input_shape[2])

    def get_output(self, train=False):
        X = self.get_input(train)
        X = T.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1)).dimshuffle(0, 2, 1, 3)
        output = downsample.max_pool_2d(X, ds=self.pool_size, st=self.st, ignore_border=self.ignore_border)
        output = output.dimshuffle(0, 2, 1, 3)
        return T.reshape(output, (output.shape[0], output.shape[1], output.shape[2]))

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "stride": self.stride,
                  "pool_length": self.pool_length,
                  "ignore_border": self.ignore_border}
        base_config = super(MaxPooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPooling2D(Layer):
    input_ndim = 4

    def __init__(self, pool_size=(2, 2), stride=None, ignore_border=True, **kwargs):
        super(MaxPooling2D, self).__init__(**kwargs)
        self.input = T.tensor4()
        self.pool_size = tuple(pool_size)
        if stride is None:
            stride = self.pool_size
        self.stride = tuple(stride)
        self.ignore_border = ignore_border

    @property
    def output_shape(self):
        input_shape = self.input_shape
        rows = pool_output_length(input_shape[2], self.pool_size[0], self.ignore_border, self.stride[0])
        cols = pool_output_length(input_shape[3], self.pool_size[1], self.ignore_border, self.stride[1])
        return (input_shape[0], input_shape[1], rows, cols)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = downsample.max_pool_2d(X, ds=self.pool_size, st=self.stride, ignore_border=self.ignore_border)
        return output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "pool_size": self.pool_size,
                  "ignore_border": self.ignore_border,
                  "stride": self.stride}
        base_config = super(MaxPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPooling3D(Layer):
    input_ndim = 5

    def __init__(self, pool_size=(2, 2, 2), stride=None, ignore_border=True, **kwargs):
        super(MaxPooling3D, self).__init__(**kwargs)
        self.mode = 'max'
        self.pool_size = tuple(pool_size)
        self.ignore_border = ignore_border
        if stride is None:
            stride = self.pool_size
        self.stride = tuple(stride)
        self.ignore_border = ignore_border

        dtensor5 = T.TensorType('float32', (0,)*5)
        self.input = dtensor5()
        self.params = []

    @property
    def output_shape(self):
        input_shape = self.input_shape
        depth = pool_output_length(input_shape[2], self.pool_size[0], self.ignore_border, self.stride[0])
        rows = pool_output_length(input_shape[3], self.pool_size[1], self.ignore_border, self.stride[1])
        cols = pool_output_length(input_shape[4], self.pool_size[2], self.ignore_border, self.stride[2])
        return (input_shape[0], input_shape[1], depth, rows, cols)

    def get_output(self, train):
        X = self.get_input(train)

        # pooling over X, Z (last two channels)
        output = downsample.max_pool_2d(input=X.dimshuffle(0, 1, 4, 3, 2),
                                        ds=(self.pool_size[1], self.pool_size[0]),
                                        ignore_border=self.ignore_border)

        # max_pool_2d X and Y, X constant
        output = downsample.max_pool_2d(input=output.dimshuffle(0, 1, 4, 3, 2),
                                        ds=(1, self.pool_size[2]),
                                        ignore_border=self.ignore_border)
        return output

    def get_config(self):
        return {"name": self.__class__.__name__,
                "pool_size": self.pool_size,
                "ignore_border": self.ignore_border,
                "stride": self.stride}
        base_config = super(MaxPooling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UpSample1D(Layer):
    input_ndim = 3

    def __init__(self, length=2, **kwargs):
        super(UpSample1D, self).__init__(**kwargs)
        self.length = length
        self.input = T.tensor3()

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], self.length * input_shape[1], input_shape[2])

    def get_output(self, train=False):
        X = self.get_input(train)
        output = theano.tensor.extra_ops.repeat(X, self.length, axis=1)
        return output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "length": self.length}
        base_config = super(UpSample1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UpSample2D(Layer):
    input_ndim = 4

    def __init__(self, size=(2, 2), **kwargs):
        super(UpSample2D, self).__init__(**kwargs)
        self.input = T.tensor4()
        self.size = tuple(size)

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1], self.size[0] * input_shape[2], self.size[1] * input_shape[3])

    def get_output(self, train=False):
        X = self.get_input(train)
        Y = theano.tensor.extra_ops.repeat(X, self.size[0], axis=2)
        output = theano.tensor.extra_ops.repeat(Y, self.size[1], axis=3)
        return output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "size": self.size}
        base_config = super(UpSample2D, self).get_config()
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
        self.input = T.tensor3()

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1] + self.padding * 2, input_shape[2])

    def get_output(self, train=False):
        X = self.get_input(train)
        input_shape = X.shape
        output_shape = (input_shape[0],
                        input_shape[1] + 2 * self.padding,
                        input_shape[2])
        output = T.zeros(output_shape)
        return T.set_subtensor(output[:, self.padding:X.shape[1] + self.padding, :], X)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "padding": self.padding}
        base_config = super(ZeroPadding1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ZeroPadding2D(Layer):
    """Zero-padding layer for 1D input (e.g. temporal sequence).

    Input shape
    -----------
    4D tensor with shape (samples, depth, first_axis_to_pad, second_axis_to_pad)

    Output shape
    ------------
    4D tensor with shape (samples, depth, first_padded_axis, second_padded_axis)

    Arguments
    ---------
    padding: tuple of int (length 2)
        How many zeros to add at the beginning and end of
        the 2 padding dimensions (axis 3 and 4).
    """
    input_ndim = 4

    def __init__(self, padding=(1, 1), **kwargs):
        super(ZeroPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)
        self.input = T.tensor4()

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0],
                input_shape[1],
                input_shape[2] + 2 * self.padding[0],
                input_shape[3] + 2 * self.padding[1])

    def get_output(self, train=False):
        X = self.get_input(train)
        input_shape = X.shape
        output_shape = (input_shape[0],
                        input_shape[1],
                        input_shape[2] + 2 * self.padding[0],
                        input_shape[3] + 2 * self.padding[1])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(None),
                   slice(self.padding[0], input_shape[2] + self.padding[0]),
                   slice(self.padding[1], input_shape[3] + self.padding[1]))
        return T.set_subtensor(output[indices], X)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "padding": self.padding}
        base_config = super(ZeroPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ZeroPadding3D(Layer):
    """Zero-padding layer for 3D input (e.g. 3D voxel points of hand).

    Input shape
    -----------
    5D tensor with shape (samples, channels, depth, first_axis_to_pad, second_axis_to_pad)

    Output shape
    ------------
    5D tensor with shape (samples, channels, depth, first_padded_axis, second_padded_axis)

    Arguments
    ---------
    padding: tuple of int (length 3)
        How many zeros to add at the beginning and end of
        the 3 padding dimensions (axis 3 and 4 and 5).
    """
    input_ndim = 5

    def __init__(self, padding=(1, 1, 1), **kwargs):
        super(ZeroPadding3D, self).__init__(**kwargs)
        self.padding = tuple(padding)
        dtensor5 = T.TensorType('float32', (0,)*5)
        self.input = dtensor5()

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0],
                input_shape[1],
                input_shape[2] + 2 * self.padding[0],
                input_shape[3] + 2 * self.padding[1],
                input_shape[4] + 2 * self.padding[2])

    def get_output(self, train=False):
        X = self.get_input(train)
        input_shape = X.shape
        output_shape = (input_shape[0],
                        input_shape[1],
                        input_shape[2] + 2 * self.padding[0],
                        input_shape[3] + 2 * self.padding[1],
                        input_shape[4] + 2 * self.padding[2])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(None),
                   slice(self.padding[0], input_shape[2] + self.padding[0]),
                   slice(self.padding[1], input_shape[3] + self.padding[1]),
                   slice(self.padding[2], input_shape[4] + self.padding[2]))
        return T.set_subtensor(output[indices], X)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "padding": self.padding}
        base_config = super(ZeroPadding3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))