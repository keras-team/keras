# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .. import backend as K
from .. import activations
from .. import initializations
from .. import regularizers
from .. import constraints
from ..engine import Layer
from ..engine import InputSpec
from ..utils.np_utils import conv_output_length


class LocallyConnected1D(Layer):
    """Locally-connected layer for 1D inputs.

    The `LocallyConnected1D` layer works similarly to
    the `Convolution1D` layer, except that weights are unshared,
    that is, a different set of filters is applied at each different patch
    of the input.
    When using this layer as the first layer in a model,
    either provide the keyword argument `input_dim`
    (int, e.g. 128 for sequences of 128-dimensional vectors), or `input_shape`
    (tuple of integers, e.g. `input_shape=(10, 128)`
    for sequences of 10 vectors of 128-dimensional vectors).
    Also, note that this layer can only be used with
    a fully-specified input shape (`None` dimensions not allowed).

    # Example
    ```python
        # apply a unshared weight convolution 1d of length 3 to a sequence with
        # 10 timesteps, with 64 output filters
        model = Sequential()
        model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
        # now model.output_shape == (None, 8, 64)
        # add a new conv1d on top
        model.add(LocallyConnected1D(32, 3))
        # now model.output_shape == (None, 6, 32)
    ```

    # Arguments
        nb_filter: Dimensionality of the output.
        filter_length: The extension (spatial or temporal) of each filter.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: Only support 'valid'. Please make good use of
            ZeroPadding1D to achieve same output length.
        subsample_length: factor by which to subsample output.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias (i.e. make the layer affine rather than linear).
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.
        input_length: Length of input sequences, when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).

    # Input shape
        3D tensor with shape: `(samples, steps, input_dim)`.

    # Output shape
        3D tensor with shape: `(samples, new_steps, nb_filter)`.
        `steps` value might have changed due to padding.
    """

    def __init__(self, nb_filter, filter_length,
                 init='glorot_uniform', activation=None, weights=None,
                 border_mode='valid', subsample_length=1,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, input_length=None, **kwargs):
        if border_mode != 'valid':
            raise ValueError('Invalid border mode for LocallyConnected1D '
                             '(only "valid" is supported):', border_mode)
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.init = initializations.get(init, dim_ordering='th')
        self.activation = activations.get(activation)

        self.border_mode = border_mode
        self.subsample_length = subsample_length

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=3)]
        self.initial_weights = weights
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(LocallyConnected1D, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[2]
        _, output_length, nb_filter = self.get_output_shape_for(input_shape)
        self.W_shape = (output_length,
                        self.filter_length * input_dim,
                        nb_filter)
        self.W = self.add_weight(self.W_shape,
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((output_length, self.nb_filter),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def get_output_shape_for(self, input_shape):
        length = conv_output_length(input_shape[1],
                                    self.filter_length,
                                    self.border_mode,
                                    self.subsample_length)
        return (input_shape[0], length, self.nb_filter)

    def call(self, x, mask=None):
        stride = self.subsample_length
        output_length, feature_dim, nb_filter = self.W_shape

        xs = []
        for i in range(output_length):
            slice_length = slice(i * stride, i * stride + self.filter_length)
            xs.append(K.reshape(x[:, slice_length, :], (1, -1, feature_dim)))
        x_aggregate = K.concatenate(xs, axis=0)
        # (output_length, batch_size, nb_filter)
        output = K.batch_dot(x_aggregate, self.W)
        output = K.permute_dimensions(output, (1, 0, 2))

        if self.bias:
            output += K.reshape(self.b, (1, output_length, nb_filter))

        output = self.activation(output)
        return output

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'filter_length': self.filter_length,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample_length': self.subsample_length,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(LocallyConnected1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LocallyConnected2D(Layer):
    """Locally-connected layer for 2D inputs.

    The `LocallyConnected2D` layer works similarly
    to the `Convolution2D` layer, except that weights are unshared,
    that is, a different set of filters is applied at each
    different patch of the input.
    When using this layer as the
    first layer in a model, provide the keyword argument `input_shape` (tuple
    of integers, does not include the sample axis), e.g.
    `input_shape=(3, 128, 128)` for 128x128 RGB pictures.
    Also, note that this layer can only be used with
    a fully-specified input shape (`None` dimensions not allowed).

    # Examples
    ```python
        # apply a 3x3 unshared weights convolution with 64 output filters on a 32x32 image:
        model = Sequential()
        model.add(LocallyConnected2D(64, 3, 3, input_shape=(3, 32, 32)))
        # now model.output_shape == (None, 64, 30, 30)
        # notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64 parameters

        # add a 3x3 unshared weights convolution on top, with 32 output filters:
        model.add(LocallyConnected2D(32, 3, 3))
        # now model.output_shape == (None, 32, 28, 28)
    ```

    # Arguments
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: Only support 'valid'. Please make good use of
            ZeroPadding2D to achieve same output shape.
        subsample: tuple of length 2. Factor by which to subsample output.
            Also called strides elsewhere.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
        bias: whether to include a bias (i.e. make the layer affine rather than linear).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', activation=None, weights=None,
                 border_mode='valid', subsample=(1, 1),
                 dim_ordering='default',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if border_mode != 'valid':
            raise ValueError('Invalid border mode for LocallyConnected2D '
                             '(only "valid" is supported):', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)

        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('`dim_ordering` must be in {tf, th}.')
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        super(LocallyConnected2D, self).__init__(**kwargs)

    def build(self, input_shape):
        output_shape = self.get_output_shape_for(input_shape)
        if self.dim_ordering == 'th':
            _, nb_filter, output_row, output_col = output_shape
            input_filter = input_shape[1]
        elif self.dim_ordering == 'tf':
            _, output_row, output_col, nb_filter = output_shape
            input_filter = input_shape[3]
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

        self.output_row = output_row
        self.output_col = output_col
        self.W_shape = (output_row * output_col,
                        self.nb_row * self.nb_col * input_filter,
                        nb_filter)
        self.W = self.add_weight(self.W_shape,
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((output_row, output_col, nb_filter),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)

    def call(self, x, mask=None):
        stride_row, stride_col = self.subsample
        _, feature_dim, nb_filter = self.W_shape

        if self.dim_ordering == 'th':
            if K.backend() == 'theano':
                output = []
                for i in range(self.output_row):
                    for j in range(self.output_col):
                        slice_row = slice(i * stride_row,
                                          i * stride_row + self.nb_row)
                        slice_col = slice(j * stride_col,
                                          j * stride_col + self.nb_col)
                        x_flatten = K.reshape(x[:, :, slice_row, slice_col], (1, -1, feature_dim))
                        output.append(K.dot(x_flatten, self.W[i * self.output_col + j, :, :]))
                output = K.concatenate(output, axis=0)
            else:
                xs = []
                for i in range(self.output_row):
                    for j in range(self.output_col):
                        slice_row = slice(i * stride_row,
                                          i * stride_row + self.nb_row)
                        slice_col = slice(j * stride_col,
                                          j * stride_col + self.nb_col)
                        xs.append(K.reshape(x[:, :, slice_row, slice_col], (1, -1, feature_dim)))
                x_aggregate = K.concatenate(xs, axis=0)
                output = K.batch_dot(x_aggregate, self.W)
            output = K.reshape(output, (self.output_row, self.output_col, -1, nb_filter))
            output = K.permute_dimensions(output, (2, 3, 0, 1))
        elif self.dim_ordering == 'tf':
            xs = []
            for i in range(self.output_row):
                for j in range(self.output_col):
                    slice_row = slice(i * stride_row,
                                      i * stride_row + self.nb_row)
                    slice_col = slice(j * stride_col,
                                      j * stride_col + self.nb_col)
                    xs.append(K.reshape(x[:, slice_row, slice_col, :], (1, -1, feature_dim)))
            x_aggregate = K.concatenate(xs, axis=0)
            output = K.batch_dot(x_aggregate, self.W)
            output = K.reshape(output, (self.output_row, self.output_col, -1, nb_filter))
            output = K.permute_dimensions(output, (2, 0, 1, 3))
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, nb_filter, self.output_row, self.output_col))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, self.output_row, self.output_col, nb_filter))

        output = self.activation(output)
        return output

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias}
        base_config = super(LocallyConnected2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
