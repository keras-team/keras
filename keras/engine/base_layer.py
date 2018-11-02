"""Contains the base Layer class, from which all layers inherit.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import copy
import re
from six.moves import zip

from .. import backend as K
from .. import initializers
from ..utils.layer_utils import count_params
from ..utils.generic_utils import has_arg
from ..utils.generic_utils import object_list_uid
from ..utils.generic_utils import to_list
from ..utils.generic_utils import unpack_singleton
from ..utils.generic_utils import is_all_none
from ..legacy import interfaces


class Layer(object):
    """Abstract base layer class.

    # Properties
        input, output: Input/output tensor(s). Note that if the layer
            is used more than once (shared layer), this is ill-defined
            and will raise an exception. In such cases, use
            `layer.get_input_at(node_index)`.
        input_mask, output_mask: Mask tensors. Same caveats apply as
            input, output.
        input_shape: Shape tuple. Provided for convenience, but note
            that there may be cases in which this attribute is
            ill-defined (e.g. a shared layer with multiple input
            shapes), in which case requesting `input_shape` will raise
            an Exception. Prefer using
            `layer.get_input_shape_at(node_index)`.
        input_spec: List of InputSpec class instances
            each entry describes one required input:
                - ndim
                - dtype
            A layer with `n` input tensors must have
            an `input_spec` of length `n`.
        name: String, must be unique within a model.
        non_trainable_weights: List of variables.
        output_shape: Shape tuple. See `input_shape`.
        stateful: Boolean indicating whether the layer carries
            additional non-weight state. Used in, for instance, RNN
            cells to carry information between batches.
        supports_masking: Boolean indicator of whether the layer
            supports masking, typically for unused timesteps in a
            sequence.
        trainable: Boolean, whether the layer weights
            will be updated during training.
        trainable_weights: List of variables.
        uses_learning_phase: Whether any operation
            of the layer uses `K.in_training_phase()`
            or `K.in_test_phase()`.
        weights: The concatenation of the lists trainable_weights and
            non_trainable_weights (in this order).


    # Methods
        call(x, mask=None): Where the layer's logic lives.
        __call__(x, mask=None): Wrapper around the layer logic (`call`).
            If x is a Keras tensor:
                - Connect current layer with last layer from tensor:
                    `self._add_inbound_node(last_layer)`
                - Add layer to tensor history
            If layer is not built:
                - Build from x._keras_shape
        compute_mask(x, mask)
        compute_output_shape(input_shape)
        count_params()
        get_config()
        get_input_at(node_index)
        get_input_mask_at(node_index)
        get_input_shape_at(node_index)
        get_output_at(node_index)
        get_output_mask_at(node_index)
        get_output_shape_at(node_index)
        get_weights()
        set_weights(weights)

    # Class Methods
        from_config(config)

    # Internal methods:
        _add_inbound_node(layer, index=0)
        assert_input_compatibility()
        build(input_shape)
    """

    def __init__(self, **kwargs):
        self.input_spec = None
        self.supports_masking = False
        self.stateful = False

        # These properties will be set upon call of self.build()
        self._trainable_weights = []
        self._non_trainable_weights = []
        self._losses = []
        self._updates = []
        self._per_input_losses = {}
        self._per_input_updates = {}
        self._built = False

        # These lists will be filled via successive calls
        # to self._add_inbound_node().
        self._inbound_nodes = []
        self._outbound_nodes = []

        # These properties should be set by the user via keyword arguments.
        # note that 'dtype', 'input_shape' and 'batch_input_shape'
        # are only applicable to input layers: do not pass these keywords
        # to non-input layers.
        allowed_kwargs = {'input_shape',
                          'batch_input_shape',
                          'batch_size',
                          'dtype',
                          'name',
                          'trainable',
                          'weights',
                          'input_dtype',  # legacy
                          }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)
        name = kwargs.get('name')
        if not name:
            prefix = self.__class__.__name__
            name = _to_snake_case(prefix) + '_' + str(K.get_uid(prefix))
        self.name = name

        self.trainable = kwargs.get('trainable', True)
        if 'input_shape' in kwargs or 'batch_input_shape' in kwargs:
            # In this case we will later create an input layer
            # to insert before the current layer
            if 'batch_input_shape' in kwargs:
                batch_input_shape = tuple(kwargs['batch_input_shape'])
            elif 'input_shape' in kwargs:
                if 'batch_size' in kwargs:
                    batch_size = kwargs['batch_size']
                else:
                    batch_size = None
                batch_input_shape = (
                    batch_size,) + tuple(kwargs['input_shape'])
            self.batch_input_shape = batch_input_shape

            # Set dtype.
            dtype = kwargs.get('dtype')
            if dtype is None:
                dtype = kwargs.get('input_dtype')
            if dtype is None:
                dtype = K.floatx()
            self.dtype = dtype

        if 'weights' in kwargs:
            self._initial_weights = kwargs['weights']
        else:
            self._initial_weights = None

    @staticmethod
    def _node_key(layer, node_index):
        """Converts a layer and its index to a unique (immutable type) name.

        This function is used internally with `self._network_nodes`.

        # Arguments
            layer: The layer.
            node_index: The layer's position (e.g. via enumerate) in a list of
                nodes.

        # Returns
            The unique name.
        """
        return layer.name + '_ib-' + str(node_index)

    @property
    def losses(self):
        return self._losses

    @property
    def updates(self):
        if not self.trainable and not self.stateful:
            return []
        return self._updates

    @property
    def built(self):
        return self._built

    @built.setter
    def built(self, value):
        self._built = value

    @property
    def trainable_weights(self):
        trainable = getattr(self, 'trainable', True)
        if trainable:
            return self._trainable_weights
        else:
            return []

    @trainable_weights.setter
    def trainable_weights(self, weights):
        self._trainable_weights = weights

    @property
    def non_trainable_weights(self):
        trainable = getattr(self, 'trainable', True)
        if not trainable:
            return self._trainable_weights + self._non_trainable_weights
        else:
            return self._non_trainable_weights

    @non_trainable_weights.setter
    def non_trainable_weights(self, weights):
        self._non_trainable_weights = weights

    @interfaces.legacy_add_weight_support
    def add_weight(self,
                   name,
                   shape,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   constraint=None):
        """Adds a weight variable to the layer.

        # Arguments
            name: String, the name for the weight variable.
            shape: The shape tuple of the weight.
            dtype: The dtype of the weight.
            initializer: An Initializer instance (callable).
            regularizer: An optional Regularizer instance.
            trainable: A boolean, whether the weight should
                be trained via backprop or not (assuming
                that the layer itself is also trainable).
            constraint: An optional Constraint instance.

        # Returns
            The created weight variable.
        """
        initializer = initializers.get(initializer)
        if dtype is None:
            dtype = K.floatx()
        weight = K.variable(initializer(shape),
                            dtype=dtype,
                            name=name,
                            constraint=constraint)
        if regularizer is not None:
            with K.name_scope('weight_regularizer'):
                self.add_loss(regularizer(weight))
        if trainable:
            self._trainable_weights.append(weight)
        else:
            self._non_trainable_weights.append(weight)
        return weight

    def assert_input_compatibility(self, inputs):
        """Checks compatibility between the layer and provided inputs.

        This checks that the tensor(s) `input`
        verify the input assumptions of the layer
        (if any). If not, exceptions are raised.

        # Arguments
            inputs: input tensor or list of input tensors.

        # Raises
            ValueError: in case of mismatch between
                the provided inputs and the expectations of the layer.
        """
        inputs = to_list(inputs)
        for x in inputs:
            try:
                K.is_keras_tensor(x)
            except ValueError:
                raise ValueError('Layer ' + self.name + ' was called with '
                                 'an input that isn\'t a symbolic tensor. '
                                 'Received type: ' +
                                 str(type(x)) + '. Full input: ' +
                                 str(inputs) + '. All inputs to the layer '
                                 'should be tensors.')

        if not self.input_spec:
            return
        if not isinstance(self.input_spec, (list, tuple)):
            input_spec = to_list(self.input_spec)
        else:
            input_spec = self.input_spec
        if len(inputs) != len(input_spec):
            raise ValueError('Layer ' + self.name + ' expects ' +
                             str(len(input_spec)) + ' inputs, '
                             'but it received ' + str(len(inputs)) +
                             ' input tensors. Input received: ' +
                             str(inputs))
        for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
            if spec is None:
                continue

            # Check ndim.
            if spec.ndim is not None:
                if K.ndim(x) != spec.ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected ndim=' +
                                     str(spec.ndim) + ', found ndim=' +
                                     str(K.ndim(x)))
            if spec.max_ndim is not None:
                ndim = K.ndim(x)
                if ndim is not None and ndim > spec.max_ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected max_ndim=' +
                                     str(spec.max_ndim) + ', found ndim=' +
                                     str(K.ndim(x)))
            if spec.min_ndim is not None:
                ndim = K.ndim(x)
                if ndim is not None and ndim < spec.min_ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected min_ndim=' +
                                     str(spec.min_ndim) + ', found ndim=' +
                                     str(K.ndim(x)))
            # Check dtype.
            if spec.dtype is not None:
                if K.dtype(x) != spec.dtype:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected dtype=' +
                                     str(spec.dtype) + ', found dtype=' +
                                     str(K.dtype(x)))
            # Check specific shape axes.
            if spec.axes:
                try:
                    x_shape = K.int_shape(x)
                except TypeError:
                    x_shape = None
                if x_shape is not None:
                    for axis, value in spec.axes.items():
                        if (value is not None and
                                x_shape[int(axis)] not in {value, None}):
                            raise ValueError(
                                'Input ' + str(input_index) +
                                ' is incompatible with layer ' +
                                self.name + ': expected axis ' +
                                str(axis) + ' of input shape to have '
                                'value ' + str(value) +
                                ' but got shape ' + str(x_shape))
            # Check shape.
            if spec.shape is not None:
                try:
                    x_shape = K.int_shape(x)
                except TypeError:
                    x_shape = None
                if x_shape is not None:
                    for spec_dim, dim in zip(spec.shape, x_shape):
                        if spec_dim is not None and dim is not None:
                            if spec_dim != dim:
                                raise ValueError(
                                    'Input ' + str(input_index) +
                                    ' is incompatible with layer ' +
                                    self.name + ': expected shape=' +
                                    str(spec.shape) + ', found shape=' +
                                    str(x_shape))

    def call(self, inputs, **kwargs):
        """This is where the layer's logic lives.

        # Arguments
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        # Returns
            A tensor or list/tuple of tensors.
        """
        return inputs

    def __call__(self, inputs, **kwargs):
        """Wrapper around self.call(), for handling internal references.

        If a Keras tensor is passed:
            - We call self._add_inbound_node().
            - If necessary, we `build` the layer to match
                the _keras_shape of the input(s).
            - We update the _keras_shape of every input tensor with
                its new shape (obtained via self.compute_output_shape).
                This is done as part of _add_inbound_node().
            - We update the _keras_history of the output tensor(s)
                with the current layer.
                This is done as part of _add_inbound_node().

        # Arguments
            inputs: Can be a tensor or list/tuple of tensors.
            **kwargs: Additional keyword arguments to be passed to `call()`.

        # Returns
            Output of the layer's `call` method.

        # Raises
            ValueError: in case the layer is missing shape information
                for its `build` call.
        """
        if isinstance(inputs, list):
            inputs = inputs[:]
        with K.name_scope(self.name):
            # Handle laying building (weight creating, input spec locking).
            if not self.built:
                # Raise exceptions in case the input is not compatible
                # with the input_spec specified in the layer constructor.
                self.assert_input_compatibility(inputs)

                # Collect input shapes to build layer.
                input_shapes = []
                for x_elem in to_list(inputs):
                    if hasattr(x_elem, '_keras_shape'):
                        input_shapes.append(x_elem._keras_shape)
                    elif hasattr(K, 'int_shape'):
                        input_shapes.append(K.int_shape(x_elem))
                    else:
                        raise ValueError('You tried to call layer "' +
                                         self.name +
                                         '". This layer has no information'
                                         ' about its expected input shape, '
                                         'and thus cannot be built. '
                                         'You can build it manually via: '
                                         '`layer.build(batch_input_shape)`')
                self.build(unpack_singleton(input_shapes))
                self.built = True

                # Load weights that were specified at layer instantiation.
                if self._initial_weights is not None:
                    self.set_weights(self._initial_weights)

            # Raise exceptions in case the input is not compatible
            # with the input_spec set at build time.
            self.assert_input_compatibility(inputs)

            # Handle mask propagation.
            previous_mask = _collect_previous_mask(inputs)
            user_kwargs = copy.copy(kwargs)
            if not is_all_none(previous_mask):
                # The previous layer generated a mask.
                if has_arg(self.call, 'mask'):
                    if 'mask' not in kwargs:
                        # If mask is explicitly passed to __call__,
                        # we should override the default mask.
                        kwargs['mask'] = previous_mask
            # Handle automatic shape inference (only useful for Theano).
            input_shape = _collect_input_shape(inputs)

            # Actually call the layer,
            # collecting output(s), mask(s), and shape(s).
            output = self.call(inputs, **kwargs)
            output_mask = self.compute_mask(inputs, previous_mask)

            # If the layer returns tensors from its inputs, unmodified,
            # we copy them to avoid loss of tensor metadata.
            output_ls = to_list(output)
            inputs_ls = to_list(inputs)
            output_ls_copy = []
            for x in output_ls:
                if x in inputs_ls:
                    x = K.identity(x)
                output_ls_copy.append(x)
            output = unpack_singleton(output_ls_copy)

            # Inferring the output shape is only relevant for Theano.
            if all([s is not None
                    for s in to_list(input_shape)]):
                output_shape = self.compute_output_shape(input_shape)
            else:
                if isinstance(input_shape, list):
                    output_shape = [None for _ in input_shape]
                else:
                    output_shape = None

            if (not isinstance(output_mask, (list, tuple)) and
                    len(output_ls) > 1):
                # Augment the mask to match the length of the output.
                output_mask = [output_mask] * len(output_ls)

            # Add an inbound node to the layer, so that it keeps track
            # of the call and of all new variables created during the call.
            # This also updates the layer history of the output tensor(s).
            # If the input tensor(s) had not previous Keras history,
            # this does nothing.
            self._add_inbound_node(input_tensors=inputs,
                                   output_tensors=output,
                                   input_masks=previous_mask,
                                   output_masks=output_mask,
                                   input_shapes=input_shape,
                                   output_shapes=output_shape,
                                   arguments=user_kwargs)

            # Apply activity regularizer if any:
            if (hasattr(self, 'activity_regularizer') and
                    self.activity_regularizer is not None):
                with K.name_scope('activity_regularizer'):
                    regularization_losses = [
                        self.activity_regularizer(x)
                        for x in to_list(output)]
                self.add_loss(regularization_losses,
                              inputs=to_list(inputs))
        return output

    def _add_inbound_node(self, input_tensors, output_tensors,
                          input_masks, output_masks,
                          input_shapes, output_shapes, arguments=None):
        """Internal method to create an inbound node for the layer.

        # Arguments
            input_tensors: list of input tensors.
            output_tensors: list of output tensors.
            input_masks: list of input masks (a mask can be a tensor, or None).
            output_masks: list of output masks
                (a mask can be a tensor, or None).
            input_shapes: list of input shape tuples.
            output_shapes: list of output shape tuples.
            arguments: dictionary of keyword arguments that were passed to the
                `call` method of the layer at the call that created the node.
        """
        input_tensors = to_list(input_tensors)
        output_tensors = to_list(output_tensors)
        input_masks = to_list(input_masks)
        output_masks = to_list(output_masks)
        input_shapes = to_list(input_shapes)
        output_shapes = to_list(output_shapes)

        # Collect input tensor(s) coordinates.
        inbound_layers = []
        node_indices = []
        tensor_indices = []
        for x in input_tensors:
            if hasattr(x, '_keras_history'):
                inbound_layer, node_index, tensor_index = x._keras_history
                inbound_layers.append(inbound_layer)
                node_indices.append(node_index)
                tensor_indices.append(tensor_index)
            else:
                inbound_layers.append(None)
                node_indices.append(None)
                tensor_indices.append(None)

        # Create node, add it to inbound nodes.
        Node(
            self,
            inbound_layers=inbound_layers,
            node_indices=node_indices,
            tensor_indices=tensor_indices,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            input_masks=input_masks,
            output_masks=output_masks,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            arguments=arguments
        )

        # Update tensor history, _keras_shape and _uses_learning_phase.
        for i in range(len(output_tensors)):
            output_tensors[i]._keras_shape = output_shapes[i]
            uses_lp = any(
                [getattr(x, '_uses_learning_phase', False)
                 for x in input_tensors])
            uses_lp = getattr(self, 'uses_learning_phase', False) or uses_lp
            output_tensors[i]._uses_learning_phase = getattr(
                output_tensors[i], '_uses_learning_phase', False) or uses_lp
            output_tensors[i]._keras_history = (self,
                                                len(self._inbound_nodes) - 1,
                                                i)

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Assumes that the layer will be built
        to match that input shape provided.

        # Arguments
            input_shape: Shape tuple (tuple of integers)
                or list of shape tuples (one per output tensor of the layer).
                Shape tuples can include None for free dimensions,
                instead of an integer.

        # Returns
            An input shape tuple.
        """
        return input_shape

    def compute_mask(self, inputs, mask=None):
        """Computes an output mask tensor.

        # Arguments
            inputs: Tensor or list of tensors.
            mask: Tensor or list of tensors.

        # Returns
            None or a tensor (or list of tensors,
                one per output tensor of the layer).
        """
        if not self.supports_masking:
            if mask is not None:
                if isinstance(mask, list):
                    if any(m is not None for m in mask):
                        raise TypeError('Layer ' + self.name +
                                        ' does not support masking, '
                                        'but was passed an input_mask: ' +
                                        str(mask))
                else:
                    raise TypeError('Layer ' + self.name +
                                    ' does not support masking, '
                                    'but was passed an input_mask: ' +
                                    str(mask))
            # masking not explicitly supported: return None as mask
            return None
        # if masking is explicitly supported, by default
        # carry over the input mask
        return mask

    def build(self, input_shape):
        """Creates the layer weights.

        Must be implemented on all layers that have weights.

        # Arguments
            input_shape: Keras tensor (future input to layer)
                or list/tuple of Keras tensors to reference
                for weight shape computations.
        """
        self.built = True

    def _get_node_attribute_at_index(self, node_index, attr, attr_name):
        """Retrieves an attribute (e.g. input_tensors) from a node.

        This is used to implement the methods:
            - get_input_shape_at
            - get_output_shape_at
            - get_input_at
            etc...

        # Arguments
            node_index: Integer index of the node from which
                to retrieve the attribute.
            attr: Exact node attribute name.
            attr_name: Human-readable attribute name, for error messages.

        # Returns
            The layer's attribute `attr` at the node of index `node_index`.

        # Raises
            RuntimeError: If the layer has no inbound nodes.
            ValueError: If the index is does not match any node.
        """
        if not self._inbound_nodes:
            raise RuntimeError('The layer has never been called '
                               'and thus has no defined ' + attr_name + '.')
        if not len(self._inbound_nodes) > node_index:
            raise ValueError('Asked to get ' + attr_name +
                             ' at node ' + str(node_index) +
                             ', but the layer has only ' +
                             str(len(self._inbound_nodes)) + ' inbound nodes.')
        values = getattr(self._inbound_nodes[node_index], attr)
        return unpack_singleton(values)

    def get_input_shape_at(self, node_index):
        """Retrieves the input shape(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A shape tuple
            (or list of shape tuples if the layer has multiple inputs).
        """
        return self._get_node_attribute_at_index(node_index,
                                                 'input_shapes',
                                                 'input shape')

    def get_output_shape_at(self, node_index):
        """Retrieves the output shape(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A shape tuple
            (or list of shape tuples if the layer has multiple outputs).
        """
        return self._get_node_attribute_at_index(node_index,
                                                 'output_shapes',
                                                 'output shape')

    def get_input_at(self, node_index):
        """Retrieves the input tensor(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A tensor (or list of tensors if the layer has multiple inputs).
        """
        return self._get_node_attribute_at_index(node_index,
                                                 'input_tensors',
                                                 'input')

    def get_output_at(self, node_index):
        """Retrieves the output tensor(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A tensor (or list of tensors if the layer has multiple outputs).
        """
        return self._get_node_attribute_at_index(node_index,
                                                 'output_tensors',
                                                 'output')

    def get_input_mask_at(self, node_index):
        """Retrieves the input mask tensor(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A mask tensor
            (or list of tensors if the layer has multiple inputs).
        """
        return self._get_node_attribute_at_index(node_index,
                                                 'input_masks',
                                                 'input mask')

    def get_output_mask_at(self, node_index):
        """Retrieves the output mask tensor(s) of a layer at a given node.

        # Arguments
            node_index: Integer, index of the node
                from which to retrieve the attribute.
                E.g. `node_index=0` will correspond to the
                first time the layer was called.

        # Returns
            A mask tensor
            (or list of tensors if the layer has multiple outputs).
        """
        return self._get_node_attribute_at_index(node_index,
                                                 'output_masks',
                                                 'output mask')

    @property
    def input(self):
        """Retrieves the input tensor(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        # Returns
            Input tensor or list of input tensors.

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        """
        if len(self._inbound_nodes) > 1:
            raise AttributeError('Layer ' + self.name +
                                 ' has multiple inbound nodes, '
                                 'hence the notion of "layer input" '
                                 'is ill-defined. '
                                 'Use `get_input_at(node_index)` instead.')
        elif not self._inbound_nodes:
            raise AttributeError('Layer ' + self.name +
                                 ' is not connected, no input to return.')
        return self._get_node_attribute_at_index(0, 'input_tensors',
                                                 'input')

    @property
    def output(self):
        """Retrieves the output tensor(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        # Returns
            Output tensor or list of output tensors.

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        """
        if not self._inbound_nodes:
            raise AttributeError('Layer ' + self.name +
                                 ' has no inbound nodes.')
        if len(self._inbound_nodes) > 1:
            raise AttributeError('Layer ' + self.name +
                                 ' has multiple inbound nodes, '
                                 'hence the notion of "layer output" '
                                 'is ill-defined. '
                                 'Use `get_output_at(node_index)` instead.')
        return self._get_node_attribute_at_index(0, 'output_tensors',
                                                 'output')

    @property
    def input_mask(self):
        """Retrieves the input mask tensor(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        # Returns
            Input mask tensor (potentially None) or list of input
            mask tensors.

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        """
        if len(self._inbound_nodes) != 1:
            raise AttributeError('Layer ' + self.name +
                                 ' has multiple inbound nodes, ' +
                                 'hence the notion of "layer input mask" '
                                 'is ill-defined. '
                                 'Use `get_input_mask_at(node_index)` '
                                 'instead.')
        return self._get_node_attribute_at_index(0, 'input_masks',
                                                 'input mask')

    @property
    def output_mask(self):
        """Retrieves the output mask tensor(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        # Returns
            Output mask tensor (potentially None) or list of output
            mask tensors.

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        """
        if len(self._inbound_nodes) != 1:
            raise AttributeError('Layer ' + self.name +
                                 ' has multiple inbound nodes, '
                                 'hence the notion of "layer output mask" '
                                 'is ill-defined. '
                                 'Use `get_output_mask_at(node_index)` '
                                 'instead.')
        return self._get_node_attribute_at_index(0, 'output_masks',
                                                 'output mask')

    @property
    def input_shape(self):
        """Retrieves the input shape tuple(s) of a layer.

        Only applicable if the layer has exactly one inbound node,
        i.e. if it is connected to one incoming layer.

        # Returns
            Input shape tuple
            (or list of input shape tuples, one tuple per input tensor).

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        """
        if not self._inbound_nodes:
            raise AttributeError('The layer has never been called '
                                 'and thus has no defined input shape.')
        all_input_shapes = set(
            [str(node.input_shapes) for node in self._inbound_nodes])
        if len(all_input_shapes) == 1:
            input_shapes = self._inbound_nodes[0].input_shapes
            return unpack_singleton(input_shapes)
        else:
            raise AttributeError('The layer "' + str(self.name) +
                                 ' has multiple inbound nodes, '
                                 'with different input shapes. Hence '
                                 'the notion of "input shape" is '
                                 'ill-defined for the layer. '
                                 'Use `get_input_shape_at(node_index)` '
                                 'instead.')

    @property
    def output_shape(self):
        """Retrieves the output shape tuple(s) of a layer.

        Only applicable if the layer has one inbound node,
        or if all inbound nodes have the same output shape.

        # Returns
            Output shape tuple
            (or list of input shape tuples, one tuple per output tensor).

        # Raises
            AttributeError: if the layer is connected to
            more than one incoming layers.
        """
        if not self._inbound_nodes:
            raise AttributeError('The layer has never been called '
                                 'and thus has no defined output shape.')
        all_output_shapes = set(
            [str(node.output_shapes) for node in self._inbound_nodes])
        if len(all_output_shapes) == 1:
            output_shapes = self._inbound_nodes[0].output_shapes
            return unpack_singleton(output_shapes)
        else:
            raise AttributeError('The layer "' + str(self.name) +
                                 ' has multiple inbound nodes, '
                                 'with different output shapes. Hence '
                                 'the notion of "output shape" is '
                                 'ill-defined for the layer. '
                                 'Use `get_output_shape_at(node_index)` '
                                 'instead.')

    def add_loss(self, losses, inputs=None):
        """Adds losses to the layer.

        The loss may potentially be conditional on some inputs tensors,
        for instance activity losses are conditional on the layer's inputs.

        # Arguments
            losses: loss tensor or list of loss tensors
                to add to the layer.
            inputs: input tensor or list of inputs tensors to mark
                the losses as conditional on these inputs.
                If None is passed, the loss is assumed unconditional
                (e.g. L2 weight regularization, which only depends
                on the layer's weights variables, not on any inputs tensors).
        """
        if losses is None or losses == []:
            return
        # Update self.losses
        losses = to_list(losses)
        if hasattr(self, '_losses'):
            self._losses += losses
        # Update self._per_input_updates
        if isinstance(inputs, list) and inputs == []:
            inputs = None
        if inputs is not None:
            inputs_hash = object_list_uid(inputs)
        else:
            # Updates indexed by None are unconditional
            # rather than input-dependent
            inputs_hash = None
        if inputs_hash not in self._per_input_losses:
            self._per_input_losses[inputs_hash] = []
        self._per_input_losses[inputs_hash] += losses

    def add_update(self, updates, inputs=None):
        """Adds updates to the layer.

        The updates may potentially be conditional on some inputs tensors,
        for instance batch norm updates are conditional on the layer's inputs.

        # Arguments
            updates: update op or list of update ops
                to add to the layer.
            inputs: input tensor or list of inputs tensors to mark
                the updates as conditional on these inputs.
                If None is passed, the updates are assumed unconditional.
        """
        if updates is None or updates == []:
            return
        # Update self.updates
        updates = to_list(updates)
        if hasattr(self, '_updates'):
            self._updates += updates
        # Update self._per_input_updates
        if isinstance(inputs, list) and inputs == []:
            inputs = None
        if inputs is not None:
            inputs_hash = object_list_uid(inputs)
        else:
            # Updates indexed by None are unconditional
            # rather than input-dependent
            inputs_hash = None
        if inputs_hash not in self._per_input_updates:
            self._per_input_updates[inputs_hash] = []
        self._per_input_updates[inputs_hash] += updates

    def get_updates_for(self, inputs):
        if not self.trainable and not self.stateful:
            return []
        if inputs is not None:
            inputs_hash = object_list_uid(inputs)
        else:
            inputs_hash = None
        if inputs_hash in self._per_input_updates:
            return self._per_input_updates[inputs_hash]
        return []

    def get_losses_for(self, inputs):
        if inputs is not None:
            inputs_hash = object_list_uid(inputs)
        else:
            inputs_hash = None
        if inputs_hash in self._per_input_losses:
            return self._per_input_losses[inputs_hash]
        return []

    @property
    def weights(self):
        return self.trainable_weights + self.non_trainable_weights

    def set_weights(self, weights):
        """Sets the weights of the layer, from Numpy arrays.

        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the layer (i.e. it should match the
                output of `get_weights`).

        # Raises
            ValueError: If the provided weights list does not match the
                layer's specifications.
        """
        params = self.weights
        if len(params) != len(weights):
            raise ValueError('You called `set_weights(weights)` on layer "' +
                             self.name +
                             '" with a  weight list of length ' +
                             str(len(weights)) +
                             ', but the layer was expecting ' +
                             str(len(params)) +
                             ' weights. Provided weights: ' +
                             str(weights)[:50] + '...')
        if not params:
            return
        weight_value_tuples = []
        param_values = K.batch_get_value(params)
        for pv, p, w in zip(param_values, params, weights):
            if pv.shape != w.shape:
                raise ValueError('Layer weight shape ' +
                                 str(pv.shape) +
                                 ' not compatible with '
                                 'provided weight shape ' + str(w.shape))
            weight_value_tuples.append((p, w))
        K.batch_set_value(weight_value_tuples)

    def get_weights(self):
        """Returns the current weights of the layer.

        # Returns
            Weights values as a list of numpy arrays.
        """
        params = self.weights
        return K.batch_get_value(params)

    def get_config(self):
        """Returns the config of the layer.

        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.

        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by `Network` (one layer of abstraction above).

        # Returns
            Python dictionary.
        """
        config = {'name': self.name,
                  'trainable': self.trainable}
        if hasattr(self, 'batch_input_shape'):
            config['batch_input_shape'] = self.batch_input_shape
        if hasattr(self, 'dtype'):
            config['dtype'] = self.dtype
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a layer from its config.

        This method is the reverse of `get_config`,
        capable of instantiating the same layer from the config
        dictionary. It does not handle layer connectivity
        (handled by Network), nor weights (handled by `set_weights`).

        # Arguments
            config: A Python dictionary, typically the
                output of get_config.

        # Returns
            A layer instance.
        """
        return cls(**config)

    def count_params(self):
        """Counts the total number of scalars composing the weights.

        # Returns
            An integer count.

        # Raises
            RuntimeError: if the layer isn't yet built
                (in which case its weights aren't yet defined).
        """
        if not self.built:
            if self.__class__.__name__ == 'Sequential':
                self.build()
            else:
                raise RuntimeError('You tried to call `count_params` on ' +
                                   self.name + ', but the layer isn\'t built. '
                                   'You can build it manually via: `' +
                                   self.name + '.build(batch_input_shape)`.')
        return count_params(self.weights)


class InputSpec(object):
    """Specifies the ndim, dtype and shape of every input to a layer.

    Every layer should expose (if appropriate) an `input_spec` attribute:
    a list of instances of InputSpec (one per input tensor).

    A None entry in a shape is compatible with any dimension,
    a None shape is compatible with any shape.

    # Arguments
        dtype: Expected datatype of the input.
        shape: Shape tuple, expected shape of the input
            (may include None for unchecked axes).
        ndim: Integer, expected rank of the input.
        max_ndim: Integer, maximum rank of the input.
        min_ndim: Integer, minimum rank of the input.
        axes: Dictionary mapping integer axes to
            a specific dimension value.
    """

    def __init__(self, dtype=None,
                 shape=None,
                 ndim=None,
                 max_ndim=None,
                 min_ndim=None,
                 axes=None):
        self.dtype = dtype
        self.shape = shape
        if shape is not None:
            self.ndim = len(shape)
        else:
            self.ndim = ndim
        self.max_ndim = max_ndim
        self.min_ndim = min_ndim
        self.axes = axes or {}

    def __repr__(self):
        spec = [('dtype=' + str(self.dtype)) if self.dtype else '',
                ('shape=' + str(self.shape)) if self.shape else '',
                ('ndim=' + str(self.ndim)) if self.ndim else '',
                ('max_ndim=' + str(self.max_ndim)) if self.max_ndim else '',
                ('min_ndim=' + str(self.min_ndim)) if self.min_ndim else '',
                ('axes=' + str(self.axes)) if self.axes else '']
        return 'InputSpec(%s)' % ', '.join(x for x in spec if x)


class Node(object):
    """A `Node` describes the connectivity between two layers.

    Each time a layer is connected to some new input,
    a node is added to `layer._inbound_nodes`.
    Each time the output of a layer is used by another layer,
    a node is added to `layer._outbound_nodes`.

    # Arguments
        outbound_layer: the layer that takes
            `input_tensors` and turns them into `output_tensors`
            (the node gets created when the `call`
            method of the layer was called).
        inbound_layers: a list of layers, the same length as `input_tensors`,
            the layers from where `input_tensors` originate.
        node_indices: a list of integers, the same length as `inbound_layers`.
            `node_indices[i]` is the origin node of `input_tensors[i]`
            (necessary since each inbound layer might have several nodes,
            e.g. if the layer is being shared with a different data stream).
        tensor_indices: a list of integers,
            the same length as `inbound_layers`.
            `tensor_indices[i]` is the index of `input_tensors[i]` within the
            output of the inbound layer
            (necessary since each inbound layer might
            have multiple tensor outputs, with each one being
            independently manipulable).
        input_tensors: list of input tensors.
        output_tensors: list of output tensors.
        input_masks: list of input masks (a mask can be a tensor, or None).
        output_masks: list of output masks (a mask can be a tensor, or None).
        input_shapes: list of input shape tuples.
        output_shapes: list of output shape tuples.
        arguments: dictionary of keyword arguments that were passed to the
            `call` method of the layer at the call that created the node.

    `node_indices` and `tensor_indices` are basically fine-grained coordinates
    describing the origin of the `input_tensors`, verifying the following:

    origin_node = inbound_layers[i]._inbound_nodes[node_indices[i]]
    input_tensors[i] == origin_node.output_tensors[tensor_indices[i]]

    A node from layer A to layer B is added to:
        A._outbound_nodes
        B._inbound_nodes
    """

    def __init__(self, outbound_layer,
                 inbound_layers, node_indices, tensor_indices,
                 input_tensors, output_tensors,
                 input_masks, output_masks,
                 input_shapes, output_shapes,
                 arguments=None):
        # Layer instance (NOT a list).
        # this is the layer that takes a list of input tensors
        # and turns them into a list of output tensors.
        # the current node will be added to
        # the inbound_nodes of outbound_layer.
        self.outbound_layer = outbound_layer

        # The following 3 properties describe where
        # the input tensors come from: which layers,
        # and for each layer, which node and which
        # tensor output of each node.

        # List of layer instances.
        self.inbound_layers = inbound_layers
        # List of integers, 1:1 mapping with inbound_layers.
        self.node_indices = node_indices
        # List of integers, 1:1 mapping with inbound_layers.
        self.tensor_indices = tensor_indices

        # Following 2 properties:
        # tensor inputs and outputs of outbound_layer.

        # List of tensors. 1:1 mapping with inbound_layers.
        self.input_tensors = input_tensors
        # List of tensors, created by outbound_layer.call().
        self.output_tensors = output_tensors

        # Following 2 properties: input and output masks.
        # List of tensors, 1:1 mapping with input_tensor.
        self.input_masks = input_masks
        # List of tensors, created by outbound_layer.compute_mask().
        self.output_masks = output_masks

        # Following 2 properties: input and output shapes.

        # List of shape tuples, shapes of input_tensors.
        self.input_shapes = input_shapes
        # List of shape tuples, shapes of output_tensors.
        self.output_shapes = output_shapes

        # Optional keyword arguments to layer's `call`.
        self.arguments = arguments

        # Add nodes to all layers involved.
        for layer in inbound_layers:
            if layer is not None:
                layer._outbound_nodes.append(self)
        outbound_layer._inbound_nodes.append(self)

    def get_config(self):
        inbound_names = []
        for layer in self.inbound_layers:
            if layer:
                inbound_names.append(layer.name)
            else:
                inbound_names.append(None)
        if self.outbound_layer:
            outbound_layer = self.outbound_layer.name
        else:
            outbound_layer = None
        return {'outbound_layer': outbound_layer,
                'inbound_layers': inbound_names,
                'node_indices': self.node_indices,
                'tensor_indices': self.tensor_indices}


def _collect_previous_mask(input_tensors):
    """Retrieves the output mask(s) of the previous node.

    # Arguments
        input_tensors: A tensor or list of tensors.

    # Returns
        A mask tensor or list of mask tensors.
    """
    input_tensors = to_list(input_tensors)
    masks = []
    for x in input_tensors:
        if hasattr(x, '_keras_history'):
            inbound_layer, node_index, tensor_index = x._keras_history
            node = inbound_layer._inbound_nodes[node_index]
            mask = node.output_masks[tensor_index]
            masks.append(mask)
        else:
            masks.append(None)
    return unpack_singleton(masks)


def _to_snake_case(name):
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != '_':
        return insecure
    return 'private' + insecure


def _collect_input_shape(input_tensors):
    """Collects the output shape(s) of a list of Keras tensors.

    # Arguments
        input_tensors: list of input tensors (or single input tensor).

    # Returns
        List of shape tuples (or single tuple), one tuple per input.
    """
    input_tensors = to_list(input_tensors)
    shapes = []
    for x in input_tensors:
        try:
            shapes.append(K.int_shape(x))
        except TypeError:
            shapes.append(None)
    return unpack_singleton(shapes)
