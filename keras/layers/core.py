# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np

import copy
import inspect
import types as python_types
import warnings

from .. import backend as K
from .. import activations, initializations, regularizers, constraints
from ..engine import InputSpec, Layer, Merge
from ..regularizers import ActivityRegularizer
from ..utils.generic_utils import func_dump, func_load



class Masking(Layer):
    '''Masks an input sequence by using a mask value to
    identify timesteps to be skipped.

    For each timestep in the input tensor (dimension #1 in the tensor),
    if all values in the input tensor at that timestep
    are equal to `mask_value`, then the timestep will masked (skipped)
    in all downstream layers (as long as they support masking).

    If any downstream layer does not support masking yet receives such
    an input mask, an exception will be raised.

    # Example

    Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
    to be fed to a LSTM layer.
    You want to mask timestep #3 and #5 because you lack data for
    these timesteps. You can:

        - set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
        - insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

    ```python
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
        model.add(LSTM(32))
    ```
    '''
    def __init__(self, mask_value=0., **kwargs):
        self.supports_masking = True
        self.mask_value = mask_value
        super(Masking, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return K.any(K.not_equal(input, self.mask_value), axis=-1)

    def call(self, x, mask=None):
        boolean_mask = K.any(K.not_equal(x, self.mask_value),
                             axis=-1, keepdims=True)
        return x * K.cast(boolean_mask, K.floatx())

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(Masking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dropout(Layer):
    '''Applies Dropout to the input. Dropout consists in randomly setting
    a fraction `p` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    '''
    def __init__(self, p, **kwargs):
        self.p = p
        if 0. < self.p < 1.:
            self.uses_learning_phase = True
        self.supports_masking = True
        super(Dropout, self).__init__(**kwargs)

    def _get_noise_shape(self, x):
        return None

    def call(self, x, mask=None):

        if 0. < self.p < 1.:
            noise_shape = self._get_noise_shape(x)
            x = K.in_train_phase(K.dropout(x, self.p, noise_shape), x)
        return x

    def get_config(self):
        config = {'p': self.p}
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SpatialDropout1D(Dropout):
    '''This version performs the same function as Dropout, however it drops
    entire 1D feature maps instead of individual elements. If adjacent frames
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout1D will help promote independence
    between feature maps and should be used instead.

    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.

    # Input shape
        3D tensor with shape:
        `(samples, timesteps, channels)`

    # Output shape
        Same as input

    # References
        - [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/pdf/1411.4280.pdf)
    '''
    def __init__(self, p, **kwargs):
        super(SpatialDropout1D, self).__init__(p, **kwargs)

    def _get_noise_shape(self, x):
        input_shape = K.shape(x)
        noise_shape = (input_shape[0], 1, input_shape[2])
        return noise_shape


class SpatialDropout2D(Dropout):
    '''This version performs the same function as Dropout, however it drops
    entire 2D feature maps instead of individual elements. If adjacent pixels
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout2D will help promote independence
    between feature maps and should be used instead.

    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.
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
        Same as input

    # References
        - [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/pdf/1411.4280.pdf)
    '''
    def __init__(self, p, dim_ordering='default', **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        super(SpatialDropout2D, self).__init__(p, **kwargs)

    def _get_noise_shape(self, x):
        input_shape = K.shape(x)
        if self.dim_ordering == 'th':
            noise_shape = (input_shape[0], input_shape[1], 1, 1)
        elif self.dim_ordering == 'tf':
            noise_shape = (input_shape[0], 1, 1, input_shape[3])
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        return noise_shape


class SpatialDropout3D(Dropout):
    '''This version performs the same function as Dropout, however it drops
    entire 3D feature maps instead of individual elements. If adjacent voxels
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout3D will help promote independence
    between feature maps and should be used instead.

    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 4.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".

    # Input shape
        5D tensor with shape:
        `(samples, channels, dim1, dim2, dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, dim1, dim2, dim3, channels)` if dim_ordering='tf'.

    # Output shape
        Same as input

    # References
        - [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/pdf/1411.4280.pdf)
    '''
    def __init__(self, p, dim_ordering='default', **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        super(SpatialDropout3D, self).__init__(p, **kwargs)

    def _get_noise_shape(self, x):
        input_shape = K.shape(x)
        if self.dim_ordering == 'th':
            noise_shape = (input_shape[0], input_shape[1], 1, 1, 1)
        elif self.dim_ordering == 'tf':
            noise_shape = (input_shape[0], 1, 1, 1, input_shape[4])
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        return noise_shape


class Activation(Layer):
    '''Applies an activation function to an output.

    # Arguments
        activation: name of activation function to use
            (see: [activations](../activations.md)),
            or alternatively, a Theano or TensorFlow operation.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    '''
    def __init__(self, activation, **kwargs):
        self.supports_masking = True
        self.activation = activations.get(activation)
        super(Activation, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return self.activation(x)

    def get_config(self):
        config = {'activation': self.activation.__name__}
        base_config = super(Activation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Reshape(Layer):
    '''Reshapes an output to a certain shape.

    # Arguments
        target_shape: target shape. Tuple of integers,
            does not include the samples dimension (batch size).

    # Input shape
        Arbitrary, although all dimensions in the input shaped must be fixed.
        Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        `(batch_size,) + target_shape`

    # Example

    ```python
        # as first layer in a Sequential model
        model = Sequential()
        model.add(Reshape((3, 4), input_shape=(12,)))
        # now: model.output_shape == (None, 3, 4)
        # note: `None` is the batch dimension

        # as intermediate layer in a Sequential model
        model.add(Reshape((6, 2)))
        # now: model.output_shape == (None, 6, 2)
    ```
    '''
    def __init__(self, target_shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def _fix_unknown_dimension(self, input_shape, output_shape):
        '''Find and replace a single missing dimension in an output shape
        given an input shape.

        A near direct port of the internal Numpy function
        _fix_unknown_dimension in numpy/core/src/multiarray/shape.c

        # Arguments
            input_shape: shape of array being reshaped

            output_shape: desired shape of the array with at most
                a single -1 which indicates a dimension that should be
                derived from the input shape.

        # Returns
            The new output shape with a -1 replaced with its computed value.

            Raises a ValueError if the total array size of the output_shape is
            different then the input_shape, or more then one unknown dimension
            is specified.
        '''
        output_shape = list(output_shape)

        msg = 'total size of new array must be unchanged'

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('can only specify one unknown dimension')
            else:
                known *= dim

        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)

        return tuple(output_shape)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],) + self._fix_unknown_dimension(input_shape[1:],
                                                               self.target_shape)

    def call(self, x, mask=None):
        # In case the target shape is not fully defined,
        # we need access to the shape of x.
        # solution:
        # 1) rely on x._keras_shape
        # 2) fallback: K.int_shape
        target_shape = self.target_shape
        if -1 in target_shape:
            # target shape not fully defined
            input_shape = None
            if hasattr(x, '_keras_shape'):
                input_shape = x._keras_shape
            elif hasattr(K, 'int_shape'):
                input_shape = K.int_shape(x)
            if input_shape is not None:
                target_shape = self.get_output_shape_for(input_shape)
        return K.reshape(x, (-1,) + target_shape)

    def get_config(self):
        config = {'target_shape': self.target_shape}
        base_config = super(Reshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Permute(Layer):
    '''Permutes the dimensions of the input according to a given pattern.

    Useful for e.g. connecting RNNs and convnets together.

    # Example

    ```python
        model = Sequential()
        model.add(Permute((2, 1), input_shape=(10, 64)))
        # now: model.output_shape == (None, 64, 10)
        # note: `None` is the batch dimension
    ```

    # Arguments
        dims: Tuple of integers. Permutation pattern, does not include the
            samples dimension. Indexing starts at 1.
            For instance, `(2, 1)` permutes the first and second dimension
            of the input.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same as the input shape, but with the dimensions re-ordered according
        to the specified pattern.
    '''
    def __init__(self, dims, **kwargs):
        self.dims = tuple(dims)
        super(Permute, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        input_shape = list(input_shape)
        output_shape = copy.copy(input_shape)
        for i, dim in enumerate(self.dims):
            target_dim = input_shape[dim]
            output_shape[i+1] = target_dim
        return tuple(output_shape)

    def call(self, x, mask=None):
        return K.permute_dimensions(x, (0,) + self.dims)

    def get_config(self):
        config = {'dims': self.dims}
        base_config = super(Permute, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Flatten(Layer):
    '''Flattens the input. Does not affect the batch size.

    # Example

    ```python
        model = Sequential()
        model.add(Convolution2D(64, 3, 3,
                                border_mode='same',
                                input_shape=(3, 32, 32)))
        # now: model.output_shape == (None, 64, 32, 32)

        model.add(Flatten())
        # now: model.output_shape == (None, 65536)
    ```
    '''
    def __init__(self, **kwargs):
        self.input_spec = [InputSpec(ndim='3+')]
        super(Flatten, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "Flatten" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '. '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:]))

    def call(self, x, mask=None):
        return K.batch_flatten(x)


class RepeatVector(Layer):
    '''Repeats the input n times.

    # Example

    ```python
        model = Sequential()
        model.add(Dense(32, input_dim=32))
        # now: model.output_shape == (None, 32)
        # note: `None` is the batch dimension

        model.add(RepeatVector(3))
        # now: model.output_shape == (None, 3, 32)
    ```

    # Arguments
        n: integer, repetition factor.

    # Input shape
        2D tensor of shape `(nb_samples, features)`.

    # Output shape
        3D tensor of shape `(nb_samples, n, features)`.
    '''
    def __init__(self, n, **kwargs):
        self.n = n
        self.input_spec = [InputSpec(ndim=2)]
        super(RepeatVector, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.n, input_shape[1])

    def call(self, x, mask=None):
        return K.repeat(x, self.n)

    def get_config(self):
        config = {'n': self.n}
        base_config = super(RepeatVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RepeatMatrix(Layer):
    '''Repeats the input n times.
       Applies the same procedure as RepeatVector() but for inputs of any dimenions.
       The new dimension will be introduced in the position defined by the user.

    # Arguments
        n: integer, repetition factor.
        dim: integer, dimension along which the input will be repeated (default = 1)

    # Input shape
        R-dimensional tensor of shape `(nb_samples, dim1, dim2, ..., dimR-1)`.

    # Output shape
        R+1-dimensional tensor of shape `(nb_samples, n, dim2, dim3, ..., dimR)` if dim==1.
    '''
    def __init__(self, n, dim=1, **kwargs):
        self.supports_masking = True
        self.n = n
        self.dim = dim
        super(RepeatMatrix, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape[:self.dim]) + [self.n] + list(input_shape[self.dim:])
        return tuple(output_shape)

    def compute_mask(self, input, input_mask=None):
        return input_mask

    def call(self, x, mask=None):
        return K.repeatRdim(x, self.n, axis=self.dim)

    def get_config(self):
        config = {'n': self.n,
                  'dim': self.dim}
        base_config = super(RepeatMatrix, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Lambda(Layer):
    '''Used for evaluating an arbitrary Theano / TensorFlow expression
    on the output of the previous layer.

    # Examples

    ```python
        # add a x -> x^2 layer
        model.add(Lambda(lambda x: x ** 2))
    ```
    ```python
        # add a layer that returns the concatenation
        # of the positive part of the input and
        # the opposite of the negative part

        def antirectifier(x):
            x -= K.mean(x, axis=1, keepdims=True)
            x = K.l2_normalize(x, axis=1)
            pos = K.relu(x)
            neg = K.relu(-x)
            return K.concatenate([pos, neg], axis=1)

        def antirectifier_output_shape(input_shape):
            shape = list(input_shape)
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)

        model.add(Lambda(antirectifier,
                         output_shape=antirectifier_output_shape))
    ```

    # Arguments
        function: The function to be evaluated.
            Takes input tensor as first argument.
        output_shape: Expected output shape from function.
            Can be a tuple or function.
            If a tuple, it only specifies the first dimension onward;
                 sample dimension is assumed either the same as the input:
                 `output_shape = (input_shape[0], ) + output_shape`
                 or, the input is `None` and the sample dimension is also `None`:
                 `output_shape = (None, ) + output_shape`
            If a function, it specifies the entire shape as a function of the
            input shape: `output_shape = f(input_shape)`
        arguments: optional dictionary of keyword arguments to be passed
            to the function.

    # Input shape
        Arbitrary. Use the keyword argument input_shape
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Specified by `output_shape` argument.
    '''
    def __init__(self, function, output_shape=None, arguments=None, **kwargs):
        self.function = function
        self.arguments = arguments if arguments else {}
        self.supports_masking = False

        if output_shape is None:
            self._output_shape = None
        elif isinstance(output_shape, (tuple, list)):
            self._output_shape = tuple(output_shape)
        else:
            if not callable(output_shape):
                raise TypeError('In Lambda, `output_shape` '
                                'must be a list, a tuple, or a function.')
            self._output_shape = output_shape
        super(Lambda, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self._output_shape is None:
            # With TensorFlow, we can infer the output shape directly:
            if K.backend() == 'tensorflow':
                if isinstance(input_shape, list):
                    xs = [K.placeholder(shape=shape) for shape in input_shape]
                    x = self.call(xs)
                else:
                    x = K.placeholder(shape=input_shape)
                    x = self.call(x)
                if isinstance(x, list):
                    return [K.int_shape(x_elem) for x_elem in x]
                else:
                    return K.int_shape(x)
            # Otherwise, we default to the input shape.
            warnings.warn('`output_shape` argument not specified for layer {} '
                          'and cannot be automatically inferred with the Theano backend. '
                          'Defaulting to output shape `{}` (same as input shape). '
                          'If the expected output shape is different, specify it via the `output_shape` argument.'
                          .format(self.name, input_shape))
            return input_shape
        elif isinstance(self._output_shape, (tuple, list)):
            if isinstance(input_shape, list):
                nb_samples = input_shape[0][0]
            else:
                nb_samples = input_shape[0] if input_shape else None
            return (nb_samples,) + tuple(self._output_shape)
        else:
            shape = self._output_shape(input_shape)
            if not isinstance(shape, (list, tuple)):
                raise ValueError('output_shape function must return a tuple')
            return tuple(shape)

    def call(self, x, mask=None):
        arguments = self.arguments
        arg_spec = inspect.getargspec(self.function)
        if 'mask' in arg_spec.args:
            arguments['mask'] = mask
        return self.function(x, **arguments)

    def get_config(self):
        if isinstance(self.function, python_types.LambdaType):
            function = func_dump(self.function)
            function_type = 'lambda'
        else:
            function = self.function.__name__
            function_type = 'function'

        if isinstance(self._output_shape, python_types.LambdaType):
            output_shape = func_dump(self._output_shape)
            output_shape_type = 'lambda'
        elif callable(self._output_shape):
            output_shape = self._output_shape.__name__
            output_shape_type = 'function'
        else:
            output_shape = self._output_shape
            output_shape_type = 'raw'

        config = {'function': function,
                  'function_type': function_type,
                  'output_shape': output_shape,
                  'output_shape_type': output_shape_type,
                  'arguments': self.arguments}
        base_config = super(Lambda, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects={}):
        # Insert custom objects into globals.
        if custom_objects:
            globs = globals().copy()
            globs.update(custom_objects)
        else:
            globs = globals()

        function_type = config.pop('function_type')
        if function_type == 'function':
            function = globs[config['function']]
        elif function_type == 'lambda':
            function = func_load(config['function'], globs=globs)
        else:
            raise TypeError('Unknown function type:', function_type)

        output_shape_type = config.pop('output_shape_type')
        if output_shape_type == 'function':
            output_shape = globs[config['output_shape']]
        elif output_shape_type == 'lambda':
            output_shape = func_load(config['output_shape'], globs=globs)
        else:
            output_shape = config['output_shape']

        config['function'] = function
        config['output_shape'] = output_shape
        return cls(**config)


class Dense(Layer):
    '''Just your regular fully connected NN layer.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential(Dense(32, input_dim=16))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # this is equivalent to the above:
        model = Sequential(Dense(32, input_shape=(16,)))

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
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
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        nD tensor with shape: `(nb_samples, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(nb_samples, input_dim)`.

    # Output shape
        nD tensor with shape: `(nb_samples, ..., output_dim)`.
        For instance, for a 2D input with shape `(nb_samples, input_dim)`,
        the output would have shape `(nb_samples, output_dim)`.
    '''
    def __init__(self, output_dim, init='glorot_uniform',
                 activation=None, weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None,
                 W_learning_rate_multiplier=None, b_learning_rate_multiplier=None,**kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.W_learning_rate_multiplier = W_learning_rate_multiplier
        self.b_learning_rate_multiplier = b_learning_rate_multiplier
        self.learning_rate_multipliers = [self.W_learning_rate_multiplier, self.b_learning_rate_multiplier]

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim='2+')]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Dense, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.W = self.add_weight((input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.output_dim,),
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

    def call(self, x, mask=None):
        return self.activation(K.dot(x, self.W) + self.b)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'W_learning_rate_multiplier': self.W_learning_rate_multiplier,
                  'b_learning_rate_multiplier': self.b_learning_rate_multiplier,
                  'input_dim': self.input_dim}
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def set_lr_multipliers(self, W_learning_rate_multiplier, b_learning_rate_multiplier):
        self.W_learning_rate_multiplier = W_learning_rate_multiplier
        self.b_learning_rate_multiplier = b_learning_rate_multiplier
        self.learning_rate_multipliers = [self.W_learning_rate_multiplier,
                                          self.b_learning_rate_multiplier]


class ActivityRegularization(Layer):
    '''Layer that returns its input unchanged, but applies an update
    to the cost function based on the activity of the input.

    # Arguments
        l1: L1 regularization factor (positive float).
        l2: L2 regularization factor (positive float).

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    '''
    def __init__(self, l1=0., l2=0., **kwargs):
        self.supports_masking = True
        self.l1 = l1
        self.l2 = l2

        super(ActivityRegularization, self).__init__(**kwargs)
        self.activity_regularizer = regularizers.L1L2Regularizer(l1=l1, l2=l2)
        self.regularizers = [self.activity_regularizer]

    def get_config(self):
        config = {'l1': self.l1,
                  'l2': self.l2}
        base_config = super(ActivityRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxoutDense(Layer):
    '''A dense maxout layer.

    A `MaxoutDense` layer takes the element-wise maximum of
    `nb_feature` `Dense(input_dim, output_dim)` linear layers.
    This allows the layer to learn a convex,
    piecewise linear activation function over the inputs.

    Note that this is a *linear* layer;
    if you wish to apply activation function
    (you shouldn't need to --they are universal function approximators),
    an `Activation` layer must be added after.

    # Arguments
        output_dim: int > 0.
        nb_feature: number of Dense layers to use internally.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
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
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.

    # References
        - [Maxout Networks](http://arxiv.org/pdf/1302.4389.pdf)
    '''
    def __init__(self, output_dim,
                 nb_feature=4,
                 init='glorot_uniform',
                 weights=None,
                 W_regularizer=None,
                 b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True,
                 input_dim=None,
                 **kwargs):
        self.output_dim = output_dim
        self.nb_feature = nb_feature
        self.init = initializations.get(init)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MaxoutDense, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight((self.nb_feature, input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.nb_feature, self.output_dim,),
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
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def call(self, x, mask=None):
        # no activation, this layer is only linear.
        output = K.max(K.dot(x, self.W) + self.b, axis=1)
        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'nb_feature': self.nb_feature,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MaxoutDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Highway(Layer):
    '''Densely connected highway network,
    a natural extension of LSTMs to feedforward networks.

    # Arguments
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        transform_bias: value for the bias to take on initially (default -2)
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
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
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # References
        - [Highway Networks](http://arxiv.org/pdf/1505.00387v2.pdf)
    '''
    def __init__(self,
                 init='glorot_uniform',
                 transform_bias=-2,
                 activation=None,
                 weights=None,
                 W_regularizer=None,
                 b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True,
                 input_dim=None,
                 **kwargs):
        self.init = initializations.get(init)
        self.transform_bias = transform_bias
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight((input_dim, input_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.W_carry = self.add_weight((input_dim, input_dim),
                                       initializer=self.init,
                                       name='{}_W_carry'.format(self.name))
        if self.bias:
            self.b = self.add_weight((input_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.b_carry = self.add_weight((input_dim,),
                                           initializer='one',
                                           name='{}_b_carry'.format(self.name))
        else:
            self.b_carry = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        transform_weight = activations.sigmoid(K.dot(x, self.W_carry) + self.b_carry)
        act = self.activation(K.dot(x, self.W) + self.b)
        act *= transform_weight
        output = act + (1 - transform_weight) * x
        return output

    def get_config(self):
        config = {'init': self.init.__name__,
                  'transform_bias': self.transform_bias,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(Highway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TimeDistributedDense(Layer):
    '''Apply a same Dense layer for each dimension[1] (time_dimension) input.
    Especially useful after a recurrent network with 'return_sequence=True'.

    Note: this layer is deprecated, prefer using the `TimeDistributed` wrapper:
    ```python
        model.add(TimeDistributed(Dense(32)))
    ```

    # Input shape
        3D tensor with shape `(nb_sample, time_dimension, input_dim)`.
    # Output shape
        3D tensor with shape `(nb_sample, time_dimension, output_dim)`.
    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
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
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    '''

    def __init__(self, output_dim,
                 init='glorot_uniform',
                 activation=None,
                 weights=None,
                 W_regularizer=None,
                 b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True,
                 input_dim=None,
                 input_length=None,
                 **kwargs):
        warnings.warn('`TimeDistributedDense` is deprecated, '
                      'And will be removed on May 1st, 2017. '
                      'Please use a `Dense` layer instead.')
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=3)]
        self.supports_masking = True

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(TimeDistributedDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None,) + input_shape[1:])]
        input_dim = input_shape[2]

        self.W = self.add_weight((input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.output_dim,),
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
        return (input_shape[0], input_shape[1], self.output_dim)

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        # x has shape (samples, timesteps, input_dim)
        input_length = input_shape[1]
        if not input_length:
            if hasattr(K, 'int_shape'):
                input_length = K.int_shape(x)[1]
                if not input_length:
                    raise ValueError(
                        'Layer ' + self.name +
                        ' requires to know the length of its input, '
                        'but it could not be inferred automatically. '
                        'Specify it manually by passing an input_shape '
                        'argument to the first layer in your model.')
            else:
                input_length = K.shape(x)[1]

        # Squash samples and timesteps into a single axis
        x = K.reshape(x, (-1, input_shape[-1]))  # (samples * timesteps, input_dim)
        y = K.dot(x, self.W) + self.b  # (samples * timesteps, output_dim)
        # We have to reshape Y to (samples, timesteps, output_dim)
        y = K.reshape(y, (-1, input_length, self.output_dim))  # (samples, timesteps, output_dim)
        y = self.activation(y)
        return y

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(TimeDistributedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaskedMean(Layer):
    """
    This layer is called after an Embedding layer.
    It averages all of the masked-out embeddings.
    The mask is discarded
    """
    def __init__(self, **kwargs):
        self.support_mask = True
        super(MaskedMean, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.mean(mask[:, :, None] * x, axis=1)

    def compute_mask(self, input_shape, input_mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        base_config = super(MaskedMean, self).get_config()
        return dict(list(base_config.items()))

class WeightedSum(Layer):
    ''' Applies a weighted sum over a set of vectors input[0] and their respective weights input[1].
        First, the weights are tiled for matching the length of the input vectors on dim=1.
        Second, an element-wise multiplication is applied over the inputs.
        Third, the output tensor is summed over the defined set of dimensions if
        the input parameter sum_dims is provided.

    # Arguments
        sum_dims: dimensions on which the final summation will be applied after the respective multiplication

    # Input shape
        List with two tensors:
            input[0]: vectors
            input[1]: weights
        Both tensors must have a matching number of dimensions and lengths, except
        dim=1, which must be 1 for the set of weights.

    # Output shape
        Vector with the same number of dimensions and length as input[0] but having removed the dimensions
        specified in sum_dims (if any).
    '''
    def __init__(self, sum_dims=[], **kwargs):
        assert isinstance(sum_dims, list)
        self.sum_dims = sorted(sum_dims)[::-1]
        self.supports_masking = True
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2

    def call(self, x, mask=None):
        # get input values and weights
        values = x[0]
        weights = x[1]

        # tile weights before summing
        K.repeatRdim(weights, K.shape(values)[1], axis=1)

        #x = K.dot(values, weights)
        x = values*weights

        for d in self.sum_dims:
            x = K.sum(x, axis=d)
        return x

    def get_output_shape_for(self, input_shape):
        out_dim = []
        num_dim = len(input_shape[0])
        for d in range(num_dim):
            if d not in self.sum_dims:
                out_dim.append(max(input_shape[0][d], input_shape[1][d]))
        return tuple(out_dim)

    def compute_mask(self, input, input_mask=None):
        if not any(input_mask):
            return None
        else:
            not_None_masks = [m for m in input_mask if m is not None]
            if len(not_None_masks) == 1:
                out_mask = input_mask[not_None_masks[0]]
            else:
                out_mask = input_mask[not_None_masks[0]] * input_mask[not_None_masks[1]]

            return out_mask


    def get_config(self):
        config = {'sum_dims': self.sum_dims}
        base_config = super(WeightedSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightedMerge(Layer):
    ''' Applies a weighted merge over a set of tensors.
        This layer learns a set of lambda weights for applying a weighted sum
        for merging the input tensors.

    # Parameters
        :param mode: merge mode used. Possible values are 'sum' (default) or 'mul'.

    # Input shape
        List of tensors of any dimensions but with the same shape.

    # Output shape
        Tensor with the same number of dimensions as the input tensors.
    '''
    def __init__(self, mode='sum', init='glorot_uniform', lambdas_regularizer=None, weights=None, **kwargs):
        #self.out_shape = out_shape
        self._valid_modes = ['sum', 'mul']

        if mode not in self._valid_modes:
            raise NotImplementedError("Merge mode of type '"+ mode +"' is not valid. Valid modes are: "+str(self._valid_modes))
        self.mode = mode

        self.init = initializations.get(init)
        self.lambdas_regularizer = regularizers.get(lambdas_regularizer)
        self.initial_weights = weights

        self.supports_masking = True
        super(WeightedMerge, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        s = input_shape[0]
        for i in range(1,len(input_shape)):
            for s1,s2 in zip(input_shape[i], s):
                assert s1 == s2 or s1 is None or s2 is None, 'The shapes of some input tensors do not match ' \
                                                             '(' + str(input_shape[i]) + ' vs ' + str(s) + ').'
            #assert input_shape[i] == s, 'The shapes of some input tensors do not match ' \
            #                        '('+str(input_shape[i])+' vs '+str(s)+').'

        self.lambdas = self.init((len(input_shape),), name='{}_lambdas'.format(self.name))
        self.trainable_weights = [self.lambdas]
        self.regularizers = []

        if self.lambdas_regularizer:
            self.lambdas_regularizer.set_param(self.lambdas)
            self.regularizers.append(self.lambdas_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        if not isinstance(x, list):
            x = [x]

        # sum inputs after weighting by the learnt lambda weights
        s = x[0] * self.lambdas[0]
        for i in range(1, len(x)):
            if self.mode == 'sum':
                s += x[i] * self.lambdas[i]
            elif self.mode == 'mul':
                s *= x[i] * self.lambdas[i]

        return s

        """
        s = x[0][:,:,:self.out_shape[0],:self.out_shape[1]] * self.lambdas[0]
        for i in range(1, len(x)):
            s += x[i][:,:,:self.out_shape[0],:self.out_shape[1]] * self.lambdas[i]
        return s
        """

    def get_output_shape_for(self, input_shape):
        #return tuple(list(input_shape[0][:2]) + self.out_shape)

        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        return tuple(input_shape[0])

    def compute_mask(self, input, input_mask=None):
        if not isinstance(input_mask, list):
            input_mask = [input_mask]
        if not any(input_mask):
            return None
        else:
            return input_mask[0]

    def get_config(self):
        config = {'mode': self.mode,
                  'init': self.init.__name__,
                  'lambdas_regularizer': self.lambdas_regularizer.get_config() if self.lambdas_regularizer else None}
        base_config = super(WeightedMerge, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SetSubtensor(Layer):
    """
    This layer performs a set_subtensor operation over two layers
    # Arguments
        indices: list of strings specifying the indexation over the two input layers

    # Input shape
        List with two tensors:
            input[0]: Tensor to overwrite
            input[1]: Tensor that overwrites
    # Output shape
        K.set_subtensor(input[0][indices[0], input[1][indices[1]])
    # Supports masking: The mask of the first input layer
    """
    def __init__(self, indices, **kwargs):
        self.supports_masking = True
        self.indices = indices
        super(SetSubtensor, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2

    def call(self, x, mask=None):
        return K.set_subtensor(eval('x[0]' + self.indices[0]), eval('x[1]' + self.indices[1]))

    def compute_mask(self, input_shape, input_mask=None):
        return input_mask[0]

    def get_output_shape_for(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {'indices': self.indices}
        base_config = super(SetSubtensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
