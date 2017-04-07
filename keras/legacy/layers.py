import inspect
import types as python_types
import warnings

from ..engine.topology import Layer, InputSpec
from .. import backend as K
from ..utils.generic_utils import func_dump, func_load
from .. import regularizers
from .. import constraints
from .. import activations
from .. import initializers


class Merge(Layer):
    """A `Merge` layer can be used to merge a list of tensors
    into a single tensor, following some merge `mode`.
    # Example
    ```python
    model1 = Sequential()
    model1.add(Dense(32, input_dim=32))
    model2 = Sequential()
    model2.add(Dense(32, input_dim=32))
    merged_model = Sequential()
    merged_model.add(Merge([model1, model2], mode='concat', concat_axis=1))
    ```
    # Arguments
        layers: Can be a list of Keras tensors or
            a list of layer instances. Must be more
            than one layer/tensor.
        mode: String or lambda/function. If string, must be one
            of: 'sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'.
            If lambda/function, it should take as input a list of tensors
            and return a single tensor.
        concat_axis: Integer, axis to use in mode `concat`.
        dot_axes: Integer or tuple of integers,
            axes to use in mode `dot` or `cos`.
        output_shape: Either a shape tuple (tuple of integers),
            or a lambda/function
            to compute `output_shape`
            (only if merge mode is a lambda/function).
            If the argument is a tuple,
            it should be expected output shape, *not* including the batch size
            (same convention as the `input_shape` argument in layers).
            If the argument is callable,
            it should take as input a list of shape tuples
            (1:1 mapping to input tensors)
            and return a single shape tuple, including the
            batch size (same convention as the
            `compute_output_shape` method of layers).
        node_indices: Optional list of integers containing
            the output node index for each input layer
            (in case some input layers have multiple output nodes).
            will default to an array of 0s if not provided.
        tensor_indices: Optional list of indices of output tensors
            to consider for merging
            (in case some input layer node returns multiple tensors).
        output_mask: Mask or lambda/function to compute the output mask (only
            if merge mode is a lambda/function). If the latter case, it should
            take as input a list of masks and return a single mask.
    """

    def __init__(self, layers=None, mode='sum', concat_axis=-1,
                 dot_axes=-1, output_shape=None, output_mask=None,
                 arguments=None, node_indices=None, tensor_indices=None,
                 name=None):
        warnings.warn('The `Merge` layer is deprecated '
                      'and will be removed after 08/2017. '
                      'Use instead layers from `keras.layers.merge`, '
                      'e.g. `add`, `concatenate`, etc.', stacklevel=2)
        self.layers = layers
        self.mode = mode
        self.concat_axis = concat_axis
        self.dot_axes = dot_axes
        self._output_shape = output_shape
        self.node_indices = node_indices
        self._output_mask = output_mask
        self.arguments = arguments if arguments else {}
        self._initial_weights = None
        self._updates = []
        self._losses = []
        self._per_input_updates = {}
        self._per_input_losses = {}

        # Layer parameters.
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self._trainable_weights = []
        self._non_trainable_weights = []
        self.supports_masking = True
        self.uses_learning_phase = False
        self.input_spec = None  # Compatible with anything.
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        if layers:
            # This exists for backwards compatibility.
            # equivalent to:
            # merge = Merge(layers=None)
            # output = merge([input_tensor_1, input_tensor_2])
            if not node_indices:
                # By default we connect to
                # the 1st output stream in the input layer.
                node_indices = [0 for _ in range(len(layers))]
            if not tensor_indices:
                tensor_indices = [0 for _ in range(len(layers))]
            self._arguments_validation(layers, mode,
                                       concat_axis, dot_axes,
                                       node_indices, tensor_indices)
            self.built = True
            input_tensors = []
            input_masks = []
            for i, layer in enumerate(layers):
                node_index = node_indices[i]
                tensor_index = tensor_indices[i]
                inbound_node = layer.inbound_nodes[node_index]
                input_tensors.append(inbound_node.output_tensors[tensor_index])
                input_masks.append(inbound_node.output_masks[tensor_index])
            self(input_tensors, mask=input_masks)
        else:
            self.built = False

    def _arguments_validation(self, layers, mode, concat_axis, dot_axes,
                              node_indices, tensor_indices):
        """Validates user-passed arguments and raises exceptions
        as appropriate.
        """
        if not callable(mode):
            if mode not in {'sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'}:
                raise ValueError('Invalid merge mode: ' + str(mode))
        if not isinstance(layers, (list, tuple)) or len(layers) < 2:
            raise TypeError('A Merge should only be applied to a list of '
                            'layers with at least 2 elements. Found: ' +
                            str(layers))

        if tensor_indices is None:
            tensor_indices = [None for _ in range(len(layers))]

        input_shapes = []
        for i, layer in enumerate(layers):
            layer_output_shape = layer.get_output_shape_at(node_indices[i])
            if isinstance(layer_output_shape, list):
                # Case: the layer has multiple output tensors
                # and we only need a specific one.
                layer_output_shape = layer_output_shape[tensor_indices[i]]
            input_shapes.append(layer_output_shape)

        if mode in {'sum', 'mul', 'ave', 'cos', 'max'}:
            input_shapes_set = set(input_shapes)
            if len(input_shapes_set) > 1:
                raise ValueError('Only layers of same output shape can '
                                 'be merged using ' + mode + ' mode. ' +
                                 'Layer shapes: %s' % input_shapes)
        if mode in {'cos', 'dot'}:
            if len(layers) > 2:
                raise ValueError(mode + ' merge takes exactly 2 layers')
            shape1 = input_shapes[0]
            shape2 = input_shapes[1]
            n1 = len(shape1)
            n2 = len(shape2)
            if isinstance(dot_axes, int):
                if dot_axes < 0:
                    self.dot_axes = [dot_axes % n1, dot_axes % n2]
                else:
                    self.dot_axes = [dot_axes, ] * 2
            if not isinstance(self.dot_axes, (list, tuple)):
                raise TypeError('Invalid type for dot_axes - '
                                'should be a list.')
            if len(self.dot_axes) != 2:
                raise ValueError('Invalid format for dot_axes - '
                                 'should contain two elements.')
            if not isinstance(self.dot_axes[0], int) or not isinstance(self.dot_axes[1], int):
                raise ValueError('Invalid format for dot_axes - '
                                 'list elements should be "int".')
            if shape1[self.dot_axes[0]] != shape2[self.dot_axes[1]]:
                raise ValueError('Dimension incompatibility using dot mode: '
                                 '%s != %s. ' % (shape1[self.dot_axes[0]], shape2[self.dot_axes[1]]) +
                                 'Layer shapes: %s, %s' % (shape1, shape2))
        elif mode == 'concat':
            reduced_inputs_shapes = [list(shape) for shape in input_shapes]
            shape_set = set()
            for i in range(len(reduced_inputs_shapes)):
                del reduced_inputs_shapes[i][self.concat_axis]
                shape_set.add(tuple(reduced_inputs_shapes[i]))
            if len(shape_set) > 1:
                raise ValueError('"concat" mode can only merge '
                                 'layers with matching '
                                 'output shapes except for the concat axis. '
                                 'Layer shapes: %s' % (input_shapes))

    def call(self, inputs, mask=None):
        if not isinstance(inputs, list) or len(inputs) <= 1:
            raise TypeError('Merge must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))
        # Case: "mode" is a lambda or function.
        if callable(self.mode):
            arguments = self.arguments
            arg_spec = inspect.getargspec(self.mode)
            if 'mask' in arg_spec.args:
                arguments['mask'] = mask
            return self.mode(inputs, **arguments)

        if self.mode == 'sum' or self.mode == 'ave':
            s = inputs[0]
            for i in range(1, len(inputs)):
                s += inputs[i]
            if self.mode == 'ave':
                s /= len(inputs)
            return s

        elif self.mode == 'concat':
            return K.concatenate(inputs, axis=self.concat_axis)

        elif self.mode == 'mul':
            s = inputs[0]
            for i in range(1, len(inputs)):
                s *= inputs[i]
            return s
        elif self.mode == 'max':
            s = inputs[0]
            for i in range(1, len(inputs)):
                s = K.maximum(s, inputs[i])
            return s
        elif self.mode == 'dot':
            l1 = inputs[0]
            l2 = inputs[1]
            output = K.batch_dot(l1, l2, self.dot_axes)
            return output

        elif self.mode == 'cos':
            l1 = inputs[0]
            l2 = inputs[1]
            denominator = K.sqrt(K.batch_dot(l1, l1, self.dot_axes) *
                                 K.batch_dot(l2, l2, self.dot_axes))
            denominator = K.maximum(denominator, K.epsilon())
            output = K.batch_dot(l1, l2, self.dot_axes) / denominator
            output = K.expand_dims(output, 1)
            return output
        else:
            raise ValueError('Unknown merge mode.')

    def compute_output_shape(self, input_shape):
        # Must have multiple input shape tuples.
        assert isinstance(input_shape, list)
        # Case: callable self._output_shape.
        if callable(self.mode):
            if callable(self._output_shape):
                output_shape = self._output_shape(input_shape)
                return output_shape
            elif self._output_shape is not None:
                return (input_shape[0][0],) + tuple(self._output_shape)
            else:
                raise ValueError('The Merge layer ' + self.name +
                                 ' has a callable `mode` argument, '
                                 'and we cannot infer its output shape '
                                 'because no `output_shape` '
                                 'argument was provided. '
                                 'Make sure to pass a shape tuple '
                                 '(or callable) '
                                 '`output_shape` to Merge.')
        # Pre-defined merge modes.
        input_shapes = input_shape
        if self.mode in ['sum', 'mul', 'ave', 'max']:
            # All tuples in input_shapes should be the same.
            return input_shapes[0]
        elif self.mode == 'concat':
            output_shape = list(input_shapes[0])
            for shape in input_shapes[1:]:
                if output_shape[self.concat_axis] is None or shape[self.concat_axis] is None:
                    output_shape[self.concat_axis] = None
                    break
                output_shape[self.concat_axis] += shape[self.concat_axis]
            return tuple(output_shape)
        elif self.mode in ['dot', 'cos']:
            shape1 = list(input_shapes[0])
            shape2 = list(input_shapes[1])
            shape1.pop(self.dot_axes[0])
            shape2.pop(self.dot_axes[1])
            shape2.pop(0)
            output_shape = shape1 + shape2
            if len(output_shape) == 1:
                output_shape += [1]
            return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        if mask is None or all([m is None for m in mask]):
            return None

        assert hasattr(mask, '__len__') and len(mask) == len(inputs)

        if self.mode in ['sum', 'mul', 'ave', 'max']:
            masks = [K.expand_dims(m, 0) for m in mask if m is not None]
            return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)
        elif self.mode == 'concat':
            # Make a list of masks while making sure
            # the dimensionality of each mask
            # is the same as the corresponding input.
            masks = []
            for input_i, mask_i in zip(inputs, mask):
                if mask_i is None:
                    # Input is unmasked. Append all 1s to masks,
                    # but cast it to bool first
                    masks.append(K.cast(K.ones_like(input_i), 'bool'))
                elif K.ndim(mask_i) < K.ndim(input_i):
                    # Mask is smaller than the input, expand it
                    masks.append(K.expand_dims(mask_i))
                else:
                    masks.append(mask_i)
            concatenated = K.concatenate(masks, axis=self.concat_axis)
            return K.all(concatenated, axis=-1, keepdims=False)
        elif self.mode in ['cos', 'dot']:
            return None
        elif callable(self.mode):
            if callable(self._output_mask):
                return self._output_mask(mask)
            else:
                return self._output_mask
        else:
            # This should have been caught earlier.
            raise ValueError('Invalid merge mode: {}'.format(self.mode))

    def get_config(self):
        if isinstance(self.mode, python_types.LambdaType):
            mode = func_dump(self.mode)
            mode_type = 'lambda'
        elif callable(self.mode):
            mode = self.mode.__name__
            mode_type = 'function'
        else:
            mode = self.mode
            mode_type = 'raw'

        if isinstance(self._output_shape, python_types.LambdaType):
            output_shape = func_dump(self._output_shape)
            output_shape_type = 'lambda'
        elif callable(self._output_shape):
            output_shape = self._output_shape.__name__
            output_shape_type = 'function'
        else:
            output_shape = self._output_shape
            output_shape_type = 'raw'

        if isinstance(self._output_mask, python_types.LambdaType):
            output_mask = func_dump(self._output_mask)
            output_mask_type = 'lambda'
        elif callable(self._output_mask):
            output_mask = self._output_mask.__name__
            output_mask_type = 'function'
        else:
            output_mask = self._output_mask
            output_mask_type = 'raw'

        return {'name': self.name,
                'mode': mode,
                'mode_type': mode_type,
                'concat_axis': self.concat_axis,
                'dot_axes': self.dot_axes,
                'output_shape': output_shape,
                'output_shape_type': output_shape_type,
                'output_mask': output_mask,
                'output_mask_type': output_mask_type,
                'arguments': self.arguments}

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        mode_type = config.pop('mode_type')
        if mode_type == 'function':
            mode = globals()[config['mode']]
        elif mode_type == 'lambda':
            mode = func_load(config['mode'], globs=globals())
        else:
            mode = config['mode']

        output_shape_type = config.pop('output_shape_type', None)
        if output_shape_type == 'function':
            output_shape = globals()[config['output_shape']]
        elif output_shape_type == 'lambda':
            output_shape = func_load(config['output_shape'],
                                     globs=globals())
        else:
            output_shape = config.get('output_shape')

        output_mask_type = config.pop('output_mask_type', None)
        if output_mask_type == 'function':
            output_mask = globals()[config['output_mask']]
        elif output_mask_type == 'lambda':
            output_mask = func_load(config['output_mask'],
                                    globs=globals())
        else:
            output_mask = config.get('output_mask')

        config['mode'] = mode
        config['output_shape'] = output_shape
        config['output_mask'] = output_mask
        return super(Merge, cls).from_config(config)


def merge(inputs, mode='sum', concat_axis=-1,
          dot_axes=-1, output_shape=None, output_mask=None,
          arguments=None, name=None):
    """Functional merge, to apply to Keras tensors (NOT layers).
    Returns a Keras tensor.
    # Example
    ```python
    tensor_a = Input(shape=(32,))
    tensor_b = Input(shape=(32,))
    merged_tensor = merge([tensor_a, tensor_b], mode='concat', concat_axis=1)
    ```
    # Arguments
        mode: String or lambda/function. If string, must be one
            of: 'sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'.
            If lambda/function, it should take as input a list of tensors
            and return a single tensor.
        concat_axis: Integer, axis to use in mode `concat`.
        dot_axes: Integer or tuple of integers,
            axes to use in mode `dot` or `cos`.
        output_shape: Shape tuple (tuple of integers), or lambda/function
            to compute output_shape (only if merge mode is a lambda/function).
            If the latter case, it should take as input a list of shape tuples
            (1:1 mapping to input tensors) and return a single shape tuple,
            including the batch size
            (same convention as the `compute_output_shape` method of layers).
        node_indices: Optional list of integers containing
            the output node index for each input layer
            (in case some input layers have multiple output nodes).
            will default to an array of 0s if not provided.
        tensor_indices: Optional list of indices of output tensors
            to consider for merging
            (in case some input layer node returns multiple tensors).
    """
    warnings.warn('The `merge` function is deprecated '
                  'and will be removed after 08/2017. '
                  'Use instead layers from `keras.layers.merge`, '
                  'e.g. `add`, `concatenate`, etc.', stacklevel=2)
    all_keras_tensors = True
    for x in inputs:
        if not hasattr(x, '_keras_history'):
            all_keras_tensors = False
            break
    if all_keras_tensors:
        input_layers = []
        node_indices = []
        tensor_indices = []
        for x in inputs:
            input_layer, node_index, tensor_index = x._keras_history
            input_layers.append(input_layer)
            node_indices.append(node_index)
            tensor_indices.append(tensor_index)
        merge_layer = Merge(input_layers, mode=mode,
                            concat_axis=concat_axis,
                            dot_axes=dot_axes,
                            output_shape=output_shape,
                            output_mask=output_mask,
                            arguments=arguments,
                            node_indices=node_indices,
                            tensor_indices=tensor_indices,
                            name=name)
        return merge_layer.inbound_nodes[0].output_tensors[0]
    else:
        merge_layer = Merge(mode=mode,
                            concat_axis=concat_axis,
                            dot_axes=dot_axes,
                            output_shape=output_shape,
                            output_mask=output_mask,
                            arguments=arguments,
                            name=name)
        return merge_layer(inputs)


class MaxoutDense(Layer):
    """A dense maxout layer.
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
        weights: list of Numpy arrays to set as initial weights.
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
        - [Maxout Networks](http://arxiv.org/abs/1302.4389)
    """

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
        warnings.warn('The `MaxoutDense` layer is deprecated '
                      'and will be removed after 06/2017.')
        self.output_dim = output_dim
        self.nb_feature = nb_feature
        self.init = initializers.get(init)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MaxoutDense, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    shape=(None, input_dim))

        self.W = self.add_weight((self.nb_feature, input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='W',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.nb_feature, self.output_dim,),
                                     initializer='zero',
                                     name='b',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def call(self, x):
        # no activation, this layer is only linear.
        output = K.dot(x, self.W)
        if self.bias:
            output += self.b
        output = K.max(output, axis=1)
        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': initializers.serialize(self.init),
                  'nb_feature': self.nb_feature,
                  'W_regularizer': regularizers.serialize(self.W_regularizer),
                  'b_regularizer': regularizers.serialize(self.b_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'W_constraint': constraints.serialize(self.W_constraint),
                  'b_constraint': constraints.serialize(self.b_constraint),
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(MaxoutDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Highway(Layer):
    """Densely connected highway network.
    Highway layers are a natural extension of LSTMs to feedforward networks.
    # Arguments
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
        weights: list of Numpy arrays to set as initial weights.
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
        - [Highway Networks](http://arxiv.org/abs/1505.00387v2)
    """

    def __init__(self,
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
                 **kwargs):
        warnings.warn('The `Highway` layer is deprecated '
                      'and will be removed after 06/2017.')
        if 'transform_bias' in kwargs:
            kwargs.pop('transform_bias')
            warnings.warn('`transform_bias` argument is deprecated and '
                          'has been removed.')
        self.init = initializers.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    shape=(None, input_dim))

        self.W = self.add_weight((input_dim, input_dim),
                                 initializer=self.init,
                                 name='W',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.W_carry = self.add_weight((input_dim, input_dim),
                                       initializer=self.init,
                                       name='W_carry')
        if self.bias:
            self.b = self.add_weight((input_dim,),
                                     initializer='zero',
                                     name='b',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.b_carry = self.add_weight((input_dim,),
                                           initializer='one',
                                           name='b_carry')
        else:
            self.b_carry = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x):
        y = K.dot(x, self.W_carry)
        if self.bias:
            y += self.b_carry
        transform_weight = activations.sigmoid(y)
        y = K.dot(x, self.W)
        if self.bias:
            y += self.b
        act = self.activation(y)
        act *= transform_weight
        output = act + (1 - transform_weight) * x
        return output

    def get_config(self):
        config = {'init': initializers.serialize(self.init),
                  'activation': activations.serialize(self.activation),
                  'W_regularizer': regularizers.serialize(self.W_regularizer),
                  'b_regularizer': regularizers.serialize(self.b_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'W_constraint': constraints.serialize(self.W_constraint),
                  'b_constraint': constraints.serialize(self.b_constraint),
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(Highway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def AtrousConvolution1D(*args, **kwargs):
    from ..layers import Conv1D
    if 'atrous_rate' in kwargs:
        rate = kwargs.pop('atrous_rate')
    else:
        rate = 1
    kwargs['dilation_rate'] = rate
    warnings.warn('The `AtrousConvolution1D` layer '
                  ' has been deprecated. Use instead '
                  'the `Conv1D` layer with the `dilation_rate` '
                  'argument.')
    return Conv1D(*args, **kwargs)


def AtrousConvolution2D(*args, **kwargs):
    from ..layers import Conv2D
    if 'atrous_rate' in kwargs:
        rate = kwargs.pop('atrous_rate')
    else:
        rate = 1
    kwargs['dilation_rate'] = rate
    warnings.warn('The `AtrousConvolution2D` layer '
                  ' has been deprecated. Use instead '
                  'the `Conv2D` layer with the `dilation_rate` '
                  'argument.')
    return Conv2D(*args, **kwargs)
