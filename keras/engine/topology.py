# -*- coding: utf-8 -*-
"""Topology-related part of the Keras engine.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import json
import yaml
import warnings
import copy
import os
import re
from six.moves import zip

from .. import backend as K
from .. import initializers
from ..utils.io_utils import ask_to_proceed_with_overwrite
from ..utils.layer_utils import print_summary as print_layer_summary
from ..utils.layer_utils import count_params
from ..utils.generic_utils import has_arg
from ..utils import conv_utils
from ..legacy import interfaces

try:
    import h5py
except ImportError:
    h5py = None


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

    `input_tensors[i] == inbound_layers[i]._inbound_nodes[node_indices[i]].output_tensors[tensor_indices[i]]`

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
        return {'outbound_layer': self.outbound_layer.name if self.outbound_layer else None,
                'inbound_layers': inbound_names,
                'node_indices': self.node_indices,
                'tensor_indices': self.tensor_indices}


class Layer(object):
    """Abstract base layer class.

    # Properties
        name: String, must be unique within a model.
        input_spec: List of InputSpec class instances
            each entry describes one required input:
                - ndim
                - dtype
            A layer with `n` input tensors must have
            an `input_spec` of length `n`.
        trainable: Boolean, whether the layer weights
            will be updated during training.
        uses_learning_phase: Whether any operation
            of the layer uses `K.in_training_phase()`
            or `K.in_test_phase()`.
        input_shape: Shape tuple. Provided for convenience,
            but note that there may be cases in which this
            attribute is ill-defined (e.g. a shared layer
            with multiple input shapes), in which case
            requesting `input_shape` will raise an Exception.
            Prefer using `layer.get_input_shape_for(input_shape)`,
            or `layer.get_input_shape_at(node_index)`.
        output_shape: Shape tuple. See above.
        inbound_nodes: List of nodes.
        outbound_nodes: List of nodes.
        input, output: Input/output tensor(s). Note that if the layer is used
            more than once (shared layer), this is ill-defined
            and will raise an exception. In such cases, use
            `layer.get_input_at(node_index)`.
        input_mask, output_mask: Same as above, for masks.
        trainable_weights: List of variables.
        non_trainable_weights: List of variables.
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
        get_weights()
        set_weights(weights)
        get_config()
        count_params()
        compute_output_shape(input_shape)
        compute_mask(x, mask)
        get_input_at(node_index)
        get_output_at(node_index)
        get_input_shape_at(node_index)
        get_output_shape_at(node_index)
        get_input_mask_at(node_index)
        get_output_mask_at(node_index)

    # Class Methods
        from_config(config)

    # Internal methods:
        build(input_shape)
        _add_inbound_node(layer, index=0)
        assert_input_compatibility()
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
                batch_input_shape = (batch_size,) + tuple(kwargs['input_shape'])
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
        This function is used internally with `self._container_nodes`.

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
        inputs = _to_list(inputs)
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
            input_spec = _to_list(self.input_spec)
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
                        if value is not None and x_shape[int(axis)] not in {value, None}:
                            raise ValueError('Input ' + str(input_index) +
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
                for x_elem in _to_list(inputs):
                    if hasattr(x_elem, '_keras_shape'):
                        input_shapes.append(x_elem._keras_shape)
                    elif hasattr(K, 'int_shape'):
                        input_shapes.append(K.int_shape(x_elem))
                    else:
                        raise ValueError('You tried to call layer "' + self.name +
                                         '". This layer has no information'
                                         ' about its expected input shape, '
                                         'and thus cannot be built. '
                                         'You can build it manually via: '
                                         '`layer.build(batch_input_shape)`')
                if len(input_shapes) == 1:
                    self.build(input_shapes[0])
                else:
                    self.build(input_shapes)
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
            if not _is_all_none(previous_mask):
                # The previous layer generated a mask.
                if has_arg(self.call, 'mask'):
                    if 'mask' not in kwargs:
                        # If mask is explicitly passed to __call__,
                        # we should override the default mask.
                        kwargs['mask'] = previous_mask
            # Handle automatic shape inference (only useful for Theano).
            input_shape = _collect_input_shape(inputs)

            # Actually call the layer, collecting output(s), mask(s), and shape(s).
            output = self.call(inputs, **kwargs)
            output_mask = self.compute_mask(inputs, previous_mask)

            # If the layer returns tensors from its inputs, unmodified,
            # we copy them to avoid loss of tensor metadata.
            output_ls = _to_list(output)
            inputs_ls = _to_list(inputs)
            output_ls_copy = []
            for x in output_ls:
                if x in inputs_ls:
                    x = K.identity(x)
                output_ls_copy.append(x)
            if len(output_ls_copy) == 1:
                output = output_ls_copy[0]
            else:
                output = output_ls_copy

            # Inferring the output shape is only relevant for Theano.
            if all([s is not None for s in _to_list(input_shape)]):
                output_shape = self.compute_output_shape(input_shape)
            else:
                if isinstance(input_shape, list):
                    output_shape = [None for _ in input_shape]
                else:
                    output_shape = None

            if not isinstance(output_mask, (list, tuple)) and len(output_ls) > 1:
                # Augment the mask to match the length of the output.
                output_mask = [output_mask] * len(output_ls)

            # Add an inbound node to the layer, so that it keeps track
            # of the call and of all new variables created during the call.
            # This also updates the layer history of the output tensor(s).
            # If the input tensor(s) had not previous Keras history,
            # this does nothing.
            self._add_inbound_node(input_tensors=inputs, output_tensors=output,
                                   input_masks=previous_mask, output_masks=output_mask,
                                   input_shapes=input_shape, output_shapes=output_shape,
                                   arguments=user_kwargs)

            # Apply activity regularizer if any:
            if hasattr(self, 'activity_regularizer') and self.activity_regularizer is not None:
                regularization_losses = [self.activity_regularizer(x) for x in _to_list(output)]
                self.add_loss(regularization_losses, _to_list(inputs))
        return output

    def _add_inbound_node(self, input_tensors, output_tensors,
                          input_masks, output_masks,
                          input_shapes, output_shapes, arguments=None):
        """Internal method to create an inbound node for the layer.

        # Arguments
            input_tensors: list of input tensors.
            output_tensors: list of output tensors.
            input_masks: list of input masks (a mask can be a tensor, or None).
            output_masks: list of output masks (a mask can be a tensor, or None).
            input_shapes: list of input shape tuples.
            output_shapes: list of output shape tuples.
            arguments: dictionary of keyword arguments that were passed to the
                `call` method of the layer at the call that created the node.
        """
        input_tensors = _to_list(input_tensors)
        output_tensors = _to_list(output_tensors)
        input_masks = _to_list(input_masks)
        output_masks = _to_list(output_masks)
        input_shapes = _to_list(input_shapes)
        output_shapes = _to_list(output_shapes)

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
            uses_lp = any([getattr(x, '_uses_learning_phase', False) for x in input_tensors])
            uses_lp = getattr(self, 'uses_learning_phase', False) or uses_lp
            output_tensors[i]._uses_learning_phase = getattr(output_tensors[i], '_uses_learning_phase', False) or uses_lp
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
        if hasattr(self, 'get_output_shape_for'):
            msg = "Class `{}.{}` defines `get_output_shape_for` but does not override `compute_output_shape`. " + \
                  "If this is a Keras 1 layer, please implement `compute_output_shape` to support Keras 2."
            warnings.warn(msg.format(type(self).__module__, type(self).__name__), stacklevel=2)
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
        if len(values) == 1:
            return values[0]
        else:
            return values

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
        all_input_shapes = set([str(node.input_shapes) for node in self._inbound_nodes])
        if len(all_input_shapes) == 1:
            input_shapes = self._inbound_nodes[0].input_shapes
            if len(input_shapes) == 1:
                return input_shapes[0]
            else:
                return input_shapes
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
        all_output_shapes = set([str(node.output_shapes) for node in self._inbound_nodes])
        if len(all_output_shapes) == 1:
            output_shapes = self._inbound_nodes[0].output_shapes
            if len(output_shapes) == 1:
                return output_shapes[0]
            else:
                return output_shapes
        else:
            raise AttributeError('The layer "' + str(self.name) +
                                 ' has multiple inbound nodes, '
                                 'with different output shapes. Hence '
                                 'the notion of "output shape" is '
                                 'ill-defined for the layer. '
                                 'Use `get_output_shape_at(node_index)` '
                                 'instead.')

    def add_loss(self, losses, inputs=None):
        """Add losses to the layer.

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
        losses = _to_list(losses)
        if hasattr(self, '_losses'):
            self._losses += losses
        # Update self._per_input_updates
        if isinstance(inputs, list) and inputs == []:
            inputs = None
        if inputs is not None:
            inputs_hash = _object_list_uid(inputs)
        else:
            # Updates indexed by None are unconditional
            # rather than input-dependent
            inputs_hash = None
        if inputs_hash not in self._per_input_losses:
            self._per_input_losses[inputs_hash] = []
        self._per_input_losses[inputs_hash] += losses

    def add_update(self, updates, inputs=None):
        """Add updates to the layer.

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
        updates = _to_list(updates)
        if hasattr(self, '_updates'):
            self._updates += updates
        # Update self._per_input_updates
        if isinstance(inputs, list) and inputs == []:
            inputs = None
        if inputs is not None:
            inputs_hash = _object_list_uid(inputs)
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
            inputs_hash = _object_list_uid(inputs)
        else:
            inputs_hash = None
        if inputs_hash in self._per_input_updates:
            return self._per_input_updates[inputs_hash]
        return []

    def get_losses_for(self, inputs):
        if inputs is not None:
            inputs_hash = _object_list_uid(inputs)
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
        by `Container` (one layer of abstraction above).

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
        (handled by Container), nor weights (handled by `set_weights`).

        # Arguments
            config: A Python dictionary, typically the
                output of get_config.

        # Returns
            A layer instance.
        """
        return cls(**config)

    def count_params(self):
        """Count the total number of scalars composing the weights.

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


class InputLayer(Layer):
    """Layer to be used as an entry point into a graph.

    It can either wrap an existing tensor (pass an `input_tensor` argument)
    or create its a placeholder tensor (pass arguments `input_shape`
    or `batch_input_shape` as well as `dtype`).

    # Arguments
        input_shape: Shape tuple, not including the batch axis.
        batch_size: Optional input batch size (integer or None).
        batch_input_shape: Shape tuple, including the batch axis.
        dtype: Datatype of the input.
        input_tensor: Optional tensor to use as layer input
            instead of creating a placeholder.
        sparse: Boolean, whether the placeholder created
            is meant to be sparse.
        name: Name of the layer (string).
    """

    @interfaces.legacy_input_support
    def __init__(self, input_shape=None, batch_size=None,
                 batch_input_shape=None,
                 dtype=None, input_tensor=None, sparse=False, name=None):
        if not name:
            prefix = 'input'
            name = prefix + '_' + str(K.get_uid(prefix))
        super(InputLayer, self).__init__(dtype=dtype, name=name)

        self.trainable = False
        self.built = True
        self.sparse = sparse

        if input_shape and batch_input_shape:
            raise ValueError('Only provide the input_shape OR '
                             'batch_input_shape argument to '
                             'InputLayer, not both at the same time.')
        if input_tensor is not None and batch_input_shape is None:
            # If input_tensor is set, and batch_input_shape is not set:
            # Attempt automatic input shape inference.
            try:
                batch_input_shape = K.int_shape(input_tensor)
            except TypeError:
                if not input_shape and not batch_input_shape:
                    raise ValueError('InputLayer was provided '
                                     'an input_tensor argument, '
                                     'but its input shape cannot be '
                                     'automatically inferred. '
                                     'You should pass an input_shape or '
                                     'batch_input_shape argument.')
        if not batch_input_shape:
            if not input_shape:
                raise ValueError('An Input layer should be passed either '
                                 'a `batch_input_shape` or an `input_shape`.')
            else:
                batch_input_shape = (batch_size,) + tuple(input_shape)
        else:
            batch_input_shape = tuple(batch_input_shape)

        if not dtype:
            if input_tensor is None:
                dtype = K.floatx()
            else:
                dtype = K.dtype(input_tensor)

        self.batch_input_shape = batch_input_shape
        self.dtype = dtype

        if input_tensor is None:
            self.is_placeholder = True
            input_tensor = K.placeholder(shape=batch_input_shape,
                                         dtype=dtype,
                                         sparse=self.sparse,
                                         name=self.name)
        else:
            self.is_placeholder = False
            input_tensor._keras_shape = batch_input_shape
        # Create an input node to add to self.outbound_node
        # and set output_tensors' _keras_history.
        input_tensor._uses_learning_phase = False
        input_tensor._keras_history = (self, 0, 0)
        Node(self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=[input_tensor],
             output_tensors=[input_tensor],
             input_masks=[None],
             output_masks=[None],
             input_shapes=[batch_input_shape],
             output_shapes=[batch_input_shape])

    def get_config(self):
        config = {'batch_input_shape': self.batch_input_shape,
                  'dtype': self.dtype,
                  'sparse': self.sparse,
                  'name': self.name}
        return config


def Input(shape=None, batch_shape=None,
          name=None, dtype=None, sparse=False,
          tensor=None):
    """`Input()` is used to instantiate a Keras tensor.

    A Keras tensor is a tensor object from the underlying backend
    (Theano, TensorFlow or CNTK), which we augment with certain
    attributes that allow us to build a Keras model
    just by knowing the inputs and outputs of the model.

    For instance, if a, b and c are Keras tensors,
    it becomes possible to do:
    `model = Model(input=[a, b], output=c)`

    The added Keras attributes are:
        `_keras_shape`: Integer shape tuple propagated
            via Keras-side shape inference.
        `_keras_history`: Last layer applied to the tensor.
            the entire layer graph is retrievable from that layer,
            recursively.

    # Arguments
        shape: A shape tuple (integer), not including the batch size.
            For instance, `shape=(32,)` indicates that the expected input
            will be batches of 32-dimensional vectors.
        batch_shape: A shape tuple (integer), including the batch size.
            For instance, `batch_shape=(10, 32)` indicates that
            the expected input will be batches of 10 32-dimensional vectors.
            `batch_shape=(None, 32)` indicates batches of an arbitrary number
            of 32-dimensional vectors.
        name: An optional name string for the layer.
            Should be unique in a model (do not reuse the same name twice).
            It will be autogenerated if it isn't provided.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
        sparse: A boolean specifying whether the placeholder
            to be created is sparse.
        tensor: Optional existing tensor to wrap into the `Input` layer.
            If set, the layer will not create a placeholder tensor.

    # Returns
        A tensor.

    # Example

        ```python
        # this is a logistic regression in Keras
        x = Input(shape=(32,))
        y = Dense(16, activation='softmax')(x)
        model = Model(x, y)
        ```
    """
    if not batch_shape and tensor is None:
        assert shape is not None, ('Please provide to Input either a `shape`'
                                   ' or a `batch_shape` argument. Note that '
                                   '`shape` does not include the batch '
                                   'dimension.')
    if shape is not None and not batch_shape:
        batch_shape = (None,) + tuple(shape)
    if not dtype:
        dtype = K.floatx()
    input_layer = InputLayer(batch_input_shape=batch_shape,
                             name=name, dtype=dtype,
                             sparse=sparse,
                             input_tensor=tensor)
    # Return tensor including _keras_shape and _keras_history.
    # Note that in this case train_output and test_output are the same pointer.
    outputs = input_layer._inbound_nodes[0].output_tensors
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


class Container(Layer):
    """A Container is a directed acyclic graph of layers.

    It is the topological form of a "model". A Model
    is simply a Container with added training routines.

    # Properties
        name
        inputs
        outputs
        input_layers
        output_layers
        input_spec (list of class instances)
            each entry describes one required input:
                - ndim
                - dtype
        trainable (boolean)
        input_shape
        output_shape
        inbound_nodes: list of nodes
        outbound_nodes: list of nodes
        trainable_weights (list of variables)
        non_trainable_weights (list of variables)

    # Methods
        summary
        get_layer
        get_weights
        set_weights
        get_config
        compute_output_shape

    # Class Methods
        from_config

    # Raises
        TypeError: if input tensors are not Keras tensors from InputLayer objects
    """

    @interfaces.legacy_model_constructor_support
    def __init__(self, inputs, outputs, name=None):
        # Handle `name` argument.
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        self.supports_masking = False
        self.trainable = True
        self._per_input_losses = {}
        self._per_input_updates = {}

        # Container-specific properties.
        if isinstance(inputs, (list, tuple)):
            self.inputs = list(inputs)  # Tensor or list of tensors.
        else:
            self.inputs = [inputs]
        if isinstance(outputs, (list, tuple)):
            self.outputs = list(outputs)
        else:
            self.outputs = [outputs]

        # Check for redundancy in inputs.
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError('The list of inputs passed to the model '
                             'is redundant. '
                             'All inputs should only appear once.'
                             ' Found: ' + str(self.inputs))

        # Check for redundancy in outputs.
        if len(set(self.outputs)) != len(self.outputs):
            warnings.warn('The list of outputs passed to the model '
                          'is redundant. '
                          'All outputs should only appear once.'
                          ' Found: ' + str(self.outputs))

        # List of initial layers (1 to 1 mapping with self.inputs,
        # hence the same layer might appear twice)
        self.input_layers = []
        self.input_layers_node_indices = []
        self.input_layers_tensor_indices = []
        # list of layers (1 to 1 mapping with self.inputs,
        # hence the same layer might appear twice)
        self.output_layers = []
        self.output_layers_node_indices = []
        self.output_layers_tensor_indices = []
        # all layers in order of horizontal graph traversal.
        # Entries are unique. Includes input and output layers.
        self.layers = []

        # This is for performance optimization
        # when calling the Container on new inputs.
        # every time the Container is called on a set on input tensors,
        # we compute the output tensors,
        # output masks and output shapes in one pass,
        # then cache them here. When one of these output is queried later,
        # we retrieve it from there instead of recomputing it.
        self._output_mask_cache = {}
        self._output_tensor_cache = {}
        self._output_shape_cache = {}

        # User-provided arguments validation.
        for x in self.inputs:
            # Check that x is a Keras tensor.
            if not hasattr(x, '_keras_history'):
                cls_name = self.__class__.__name__
                raise TypeError('Input tensors to a ' + cls_name + ' ' +
                                'must be Keras tensors. Found: ' + str(x) +
                                ' (missing Keras metadata).')
            # Check that x is an input tensor.
            layer, node_index, tensor_index = x._keras_history
            if len(layer._inbound_nodes) > 1 or (layer._inbound_nodes and layer._inbound_nodes[0].inbound_layers):
                cls_name = self.__class__.__name__
                warnings.warn(cls_name + ' inputs must come from '
                              'a Keras Input layer, '
                              'they cannot be the output of '
                              'a previous non-Input layer. '
                              'Here, a tensor specified as '
                              'input to "' + self.name +
                              '" was not an Input tensor, '
                              'it was generated by layer ' +
                              layer.name + '.\n'
                              'Note that input tensors are '
                              'instantiated via `tensor = Input(shape)`.\n'
                              'The tensor that caused the issue was: ' +
                              str(x.name))
        for x in self.outputs:
            if not hasattr(x, '_keras_history'):
                cls_name = self.__class__.__name__
                raise TypeError('Output tensors to a ' + cls_name + ' must be '
                                'Keras tensors. Found: ' + str(x))
        # Build self.output_layers:
        for x in self.outputs:
            layer, node_index, tensor_index = x._keras_history
            self.output_layers.append(layer)
            self.output_layers_node_indices.append(node_index)
            self.output_layers_tensor_indices.append(tensor_index)

        # Fill in the output mask cache.
        masks = []
        for x in self.inputs:
            layer, node_index, tensor_index = x._keras_history
            node = layer._inbound_nodes[node_index]
            mask = node.output_masks[tensor_index]
            masks.append(mask)
        mask_cache_key = ','.join([str(id(x)) for x in self.inputs])
        mask_cache_key += '_' + ','.join([str(id(x)) for x in masks])
        masks = []
        for x in self.outputs:
            layer, node_index, tensor_index = x._keras_history
            node = layer._inbound_nodes[node_index]
            mask = node.output_masks[tensor_index]
            masks.append(mask)
        if len(masks) == 1:
            mask = masks[0]
        else:
            mask = masks
        self._output_mask_cache[mask_cache_key] = mask

        # Build self.input_layers:
        for x in self.inputs:
            layer, node_index, tensor_index = x._keras_history
            # It's supposed to be an input layer, so only one node
            # and one tensor output.
            assert node_index == 0
            assert tensor_index == 0
            self.input_layers.append(layer)
            self.input_layers_node_indices.append(node_index)
            self.input_layers_tensor_indices.append(tensor_index)

        # Build self.input_names and self.output_names.
        self.input_names = []
        self.output_names = []
        self._feed_input_names = []
        self._feed_inputs = []
        self._feed_input_shapes = []
        for i, layer in enumerate(self.input_layers):
            # Check that layer is an InputLayer.
            if not isinstance(layer, InputLayer):
                raise TypeError(
                    'Input layers to a `Model` must be `InputLayer` objects. '
                    'Received inputs: {}. '
                    'Input {} (0-based) originates '
                    'from layer type `{}`.'.format(inputs,
                                                   i,
                                                   layer.__class__.__name__))
            self.input_names.append(layer.name)
            if layer.is_placeholder:
                self._feed_input_names.append(layer.name)
                self._feed_inputs.append(layer.input)
                self._feed_input_shapes.append(self.inputs[i]._keras_shape)
        for layer in self.output_layers:
            self.output_names.append(layer.name)

        self._internal_input_shapes = [x._keras_shape for x in self.inputs]
        self._internal_output_shapes = [x._keras_shape for x in self.outputs]

        # Container_nodes: set of nodes included in the graph
        # (not all nodes included in the layers
        # are relevant to the current graph).
        container_nodes = set()  # ids of all nodes relevant to the Container
        nodes_depths = {}  # dict {node: depth value}
        layers_depths = {}  # dict {layer: depth value}
        layer_indices = {}  # dict {layer: index in traversal}
        nodes_in_decreasing_depth = []

        def build_map_of_graph(tensor, finished_nodes, nodes_in_progress,
                               layer=None, node_index=None, tensor_index=None):
            """Builds a map of the graph of layers.

            This recursively updates the map `layer_indices`,
            the list `nodes_in_decreasing_depth` and the set `container_nodes`.

            # Arguments
                tensor: Some tensor in a graph.
                finished_nodes: Set of nodes whose subgraphs have been traversed
                    completely. Useful to prevent duplicated work.
                nodes_in_progress: Set of nodes that are currently active on the
                    recursion stack. Useful to detect cycles.
                layer: Layer from which `tensor` comes from. If not provided,
                    will be obtained from `tensor._keras_history`.
                node_index: Node index from which `tensor` comes from.
                tensor_index: Tensor_index from which `tensor` comes from.

            # Raises
                RuntimeError: if a cycle is detected.
            """
            if not layer or node_index is None or tensor_index is None:
                layer, node_index, tensor_index = tensor._keras_history
            node = layer._inbound_nodes[node_index]

            # Prevent cycles.
            if node in nodes_in_progress:
                raise RuntimeError(
                    'The tensor ' + str(tensor) + ' at layer "' +
                    layer.name + '" is part of a cycle.')

            # Don't repeat work for shared subgraphs
            if node in finished_nodes:
                return

            # Update container_nodes.
            container_nodes.add(self._node_key(layer, node_index))

            # Store the traversal order for layer sorting.
            if layer not in layer_indices:
                layer_indices[layer] = len(layer_indices)

            nodes_in_progress.add(node)

            # Propagate to all previous tensors connected to this node.
            for i in range(len(node.inbound_layers)):
                x = node.input_tensors[i]
                layer = node.inbound_layers[i]
                node_index = node.node_indices[i]
                tensor_index = node.tensor_indices[i]
                build_map_of_graph(x, finished_nodes, nodes_in_progress,
                                   layer, node_index, tensor_index)

            finished_nodes.add(node)
            nodes_in_progress.remove(node)

            nodes_in_decreasing_depth.append(node)

        finished_nodes = set()
        nodes_in_progress = set()
        for x in self.outputs:
            build_map_of_graph(x, finished_nodes, nodes_in_progress)

        for node in reversed(nodes_in_decreasing_depth):
            # If the depth is not set, the node has no outbound nodes (depth 0).
            depth = nodes_depths.setdefault(node, 0)

            # Update the depth of the corresponding layer
            previous_depth = layers_depths.get(node.outbound_layer, 0)
            # If we've seen this layer before at a higher depth, we should use that depth instead
            # of the node depth.  This is necessary for shared layers that have inputs at different
            # depth levels in the graph.
            depth = max(depth, previous_depth)
            layers_depths[node.outbound_layer] = depth
            nodes_depths[node] = depth

            # Update the depth of inbound nodes.
            for i in range(len(node.inbound_layers)):
                inbound_layer = node.inbound_layers[i]
                node_index = node.node_indices[i]
                inbound_node = inbound_layer._inbound_nodes[node_index]
                previous_depth = nodes_depths.get(inbound_node, 0)
                nodes_depths[inbound_node] = max(depth + 1, previous_depth)

        # Build a dict {depth: list of nodes with this depth}
        nodes_by_depth = {}
        for node, depth in nodes_depths.items():
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(node)

        # Build a dict {depth: list of layers with this depth}
        layers_by_depth = {}
        for layer, depth in layers_depths.items():
            if depth not in layers_by_depth:
                layers_by_depth[depth] = []
            layers_by_depth[depth].append(layer)

        # Get sorted list of layer depths.
        depth_keys = list(layers_by_depth.keys())
        depth_keys.sort(reverse=True)

        # Set self.layers and self.layers_by_depth.
        layers = []
        for depth in depth_keys:
            layers_for_depth = layers_by_depth[depth]
            # Container.layers needs to have a deterministic order:
            # here we order them by traversal order.
            layers_for_depth.sort(key=lambda x: layer_indices[x])
            for layer in layers_for_depth:
                layers.append(layer)
        self.layers = layers
        self.layers_by_depth = layers_by_depth

        # Get sorted list of node depths.
        depth_keys = list(nodes_by_depth.keys())
        depth_keys.sort(reverse=True)

        # Check that all tensors required are computable.
        # computable_tensors: all tensors in the graph
        # that can be computed from the inputs provided.
        computable_tensors = []
        for x in self.inputs:
            computable_tensors.append(x)

        layers_with_complete_input = []  # To provide a better error msg.
        for depth in depth_keys:
            for node in nodes_by_depth[depth]:
                layer = node.outbound_layer
                if layer:
                    for x in node.input_tensors:
                        if x not in computable_tensors:
                            raise RuntimeError(
                                'Graph disconnected: '
                                'cannot obtain value for tensor ' +
                                str(x) + ' at layer "' + layer.name + '". '
                                'The following previous layers '
                                'were accessed without issue: ' +
                                str(layers_with_complete_input))
                    for x in node.output_tensors:
                        computable_tensors.append(x)
                    layers_with_complete_input.append(layer.name)

        # Set self._container_nodes and self._nodes_by_depth.
        self._container_nodes = container_nodes
        self._nodes_by_depth = nodes_by_depth

        # Ensure name unicity, which will be crucial for serialization
        # (since serialized nodes refer to layers by their name).
        all_names = [layer.name for layer in self.layers]
        for name in all_names:
            if all_names.count(name) != 1:
                raise RuntimeError('The name "' + name + '" is used ' +
                                   str(all_names.count(name)) +
                                   ' times in the model. '
                                   'All layer names should be unique. '
                                   'Layer names: ', all_names)

        # Layer parameters.
        # The new container starts with a single inbound node
        # for its inputs, and no outbound nodes.
        self._outbound_nodes = []  # Will be appended to by future calls to __call__
        self._inbound_nodes = []  # Will be appended to below, and by future calls to __call__
        # Create the node linking internal inputs to internal outputs.
        Node(outbound_layer=self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=self.inputs,
             output_tensors=self.outputs,
             # No container-level masking for now.
             input_masks=[None for _ in self.inputs],
             output_masks=[None for _ in self.outputs],
             input_shapes=[x._keras_shape for x in self.inputs],
             output_shapes=[x._keras_shape for x in self.outputs])
        self.built = True

        # The following are implemented as property functions:
        # self.trainable_weights
        # self.non_trainable_weights
        # self.input_spec

    def get_layer(self, name=None, index=None):
        """Retrieves a layer based on either its name (unique) or index.

        Indices are based on order of horizontal graph traversal (bottom-up).

        # Arguments
            name: String, name of layer.
            index: Integer, index of layer.

        # Returns
            A layer instance.

        # Raises
            ValueError: In case of invalid layer name or index.
        """
        # It would be unreliable to build a dictionary
        # based on layer names, because names can potentially
        # be changed at any point by the user
        # without the container being notified of it.
        if index is not None:
            if len(self.layers) <= index:
                raise ValueError('Was asked to retrieve layer at index ' +
                                 str(index) + ' but model only has ' +
                                 str(len(self.layers)) + ' layers.')
            else:
                return self.layers[index]
        else:
            if not name:
                raise ValueError('Provide either a layer name or layer index.')

        for layer in self.layers:
            if layer.name == name:
                return layer

        raise ValueError('No such layer: ' + name)

    @property
    def updates(self):
        """Retrieve the model's updates.

        Will only include updates that are either
        inconditional, or conditional on inputs to this model
        (e.g. will not include updates that depend on tensors
        that aren't inputs to this model).

        # Returns
            A list of update ops.
        """
        if not self.trainable and not self.stateful:
            return []
        updates = []
        for layer in self.layers:
            if hasattr(layer, 'updates'):
                # Collect updates that are dependent on inputs
                # that are part of the model.
                for node_index, node in enumerate(layer._inbound_nodes):
                    node_key = self._node_key(layer, node_index)
                    if node_key in self._container_nodes:
                        # The model owns this layer node.
                        inputs = node.input_tensors
                        updates += layer.get_updates_for(inputs)
                # Collect unconditional updates.
                updates += layer.get_updates_for(None)
        return updates

    @property
    def losses(self):
        """Retrieve the model's losses.

        Will only include losses that are either
        inconditional, or conditional on inputs to this model
        (e.g. will not include losses that depend on tensors
        that aren't inputs to this model).

        # Returns
            A list of loss tensors.
        """
        losses = []
        # Retrieve losses for all internal layers.
        for layer in self.layers:
            if hasattr(layer, 'losses'):
                # Collect losses that are dependent on inputs
                # that are part of the model.
                for node_index, node in enumerate(layer._inbound_nodes):
                    node_key = self._node_key(layer, node_index)
                    if node_key in self._container_nodes:
                        # The model owns this layer node.
                        inputs = node.input_tensors
                        losses += layer.get_losses_for(inputs)
                # Collect unconditional losses.
                losses += layer.get_losses_for(None)
        # Add any potential unconditional model-level loss.
        losses += self.get_losses_for(None)

        unique_tensors = list(set(x for x in losses if not isinstance(x, (float, int))))
        non_tensors = [x for x in losses if isinstance(x, (float, int))]
        return unique_tensors + non_tensors

    @property
    def uses_learning_phase(self):
        return any([x._uses_learning_phase for x in self.outputs])

    @property
    def stateful(self):
        return any([(hasattr(layer, 'stateful') and layer.stateful) for layer in self.layers])

    def reset_states(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_states') and getattr(layer, 'stateful', False):
                layer.reset_states()

    @property
    def state_updates(self):
        """Returns the `updates` from all layers that are stateful.

        This is useful for separating training updates and
        state updates, e.g. when we need to update a layer's internal state
        during prediction.

        # Returns
            A list of update ops.
        """
        state_updates = []
        for layer in self.layers:
            if layer.stateful:
                state_updates += layer.updates
        return state_updates

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        weights = []
        for layer in self.layers:
            weights += layer.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.non_trainable_weights
        if not self.trainable:
            trainable_weights = []
            for layer in self.layers:
                trainable_weights += layer.trainable_weights
            return trainable_weights + weights
        return weights

    def get_weights(self):
        """Retrieves the weights of the model.

        # Returns
            A flat list of Numpy arrays.
        """
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return K.batch_get_value(weights)

    def set_weights(self, weights):
        """Sets the weights of the model.

        # Arguments
            weights: A list of Numpy arrays with shapes and types matching
                the output of `model.get_weights()`.
        """
        tuples = []
        for layer in self.layers:
            num_param = len(layer.weights)
            layer_weights = weights[:num_param]
            for sw, w in zip(layer.weights, layer_weights):
                tuples.append((sw, w))
            weights = weights[num_param:]
        K.batch_set_value(tuples)

    @property
    def input_spec(self):
        """Gets the model's input specs.

        # Returns
            A list of `InputSpec` instances (one per input to the model)
                or a single instance if the model has only one input.
        """
        specs = []
        for layer in getattr(self, 'input_layers', []):
            if layer.input_spec is None:
                specs.append(None)
            else:
                if not isinstance(layer.input_spec, list):
                    raise TypeError('Layer ' + layer.name +
                                    ' has an input_spec attribute that '
                                    'is not a list. We expect a list. '
                                    'Found input_spec = ' +
                                    str(layer.input_spec))
                specs += layer.input_spec
        if len(specs) == 1:
            return specs[0]
        return specs

    def call(self, inputs, mask=None):
        """Call the model on new inputs.

        In this case `call` just reapplies
        all ops in the graph to the new inputs
        (e.g. build a new computational graph from the provided inputs).

        A model is callable on non-Keras tensors.

        # Arguments
            inputs: A tensor or list of tensors.
            mask: A mask or list of masks. A mask can be
                either a tensor or None (no mask).

        # Returns
            A tensor if there is a single output, or
            a list of tensors if there are more than one outputs.
        """
        inputs = _to_list(inputs)
        if mask is None:
            masks = [None for _ in range(len(inputs))]
        else:
            masks = _to_list(mask)
        cache_key = ','.join([str(id(x)) for x in inputs])
        cache_key += '_' + ','.join([str(id(x)) for x in masks])
        if cache_key in self._output_tensor_cache:
            return self._output_tensor_cache[cache_key]
        else:
            output_tensors, _, _ = self.run_internal_graph(inputs, masks)
            return output_tensors

    def compute_mask(self, inputs, mask):
        inputs = _to_list(inputs)
        if mask is None:
            masks = [None for _ in range(len(inputs))]
        else:
            masks = _to_list(mask)
        cache_key = ','.join([str(id(x)) for x in inputs])
        cache_key += '_' + ','.join([str(id(x)) for x in masks])
        if cache_key in self._output_mask_cache:
            return self._output_mask_cache[cache_key]
        else:
            _, output_masks, _ = self.run_internal_graph(inputs, masks)
            return output_masks

    def compute_output_shape(self, input_shape):
        input_shapes = _to_list(input_shape)
        if len(input_shapes) != len(self.input_layers):
            raise ValueError('Invalid input_shape argument ' +
                             str(input_shape) + ': model has ' +
                             str(len(self.input_layers)) + ' tensor inputs.')

        cache_key = ','.join([str(x) for x in input_shapes])
        if cache_key in self._output_shape_cache:
            output_shapes = self._output_shape_cache[cache_key]
            if isinstance(output_shapes, list) and len(output_shapes) == 1:
                return output_shapes[0]
            return output_shapes
        else:
            # Bad luck, we have to run the graph manually.
            layers_to_output_shapes = {}
            for i in range(len(input_shapes)):
                layer = self.input_layers[i]
                input_shape = input_shapes[i]
                # It's an input layer: compute_output_shape is identity,
                # and there is only one node and one tensor output.
                shape_key = layer.name + '_0_0'
                layers_to_output_shapes[shape_key] = input_shape

            depth_keys = list(self._nodes_by_depth.keys())
            depth_keys.sort(reverse=True)
            # Iterate over nodes, by depth level.
            if len(depth_keys) > 1:
                for depth in depth_keys:
                    nodes = self._nodes_by_depth[depth]
                    for node in nodes:
                        # This is always a single layer, never a list.
                        layer = node.outbound_layer
                        if layer in self.input_layers:
                            # We've already covered the input layers
                            # a few lines above.
                            continue
                        # Potentially redundant list,
                        # same size of node.input_tensors.
                        input_shapes = []
                        for j in range(len(node.inbound_layers)):
                            inbound_layer = node.inbound_layers[j]
                            node_index = node.node_indices[j]
                            tensor_index = node.tensor_indices[j]
                            shape_key = inbound_layer.name + '_%s_%s' % (node_index, tensor_index)
                            input_shape = layers_to_output_shapes[shape_key]
                            input_shapes.append(input_shape)

                        if len(input_shapes) == 1:
                            output_shape = layer.compute_output_shape(input_shapes[0])
                        else:
                            output_shape = layer.compute_output_shape(input_shapes)

                        output_shapes = _to_list(output_shape)
                        node_index = layer._inbound_nodes.index(node)
                        for j in range(len(output_shapes)):
                            shape_key = layer.name + '_%s_%s' % (node_index, j)
                            layers_to_output_shapes[shape_key] = output_shapes[j]

            # Read final output shapes from layers_to_output_shapes.
            output_shapes = []
            output_shape_keys = []
            for i in range(len(self.output_layers)):
                layer = self.output_layers[i]
                node_index = self.output_layers_node_indices[i]
                tensor_index = self.output_layers_tensor_indices[i]
                shape_key = layer.name + '_%s_%s' % (node_index, tensor_index)
                output_shape_keys.append(shape_key)

            for i, key in enumerate(output_shape_keys):
                assert key in layers_to_output_shapes
                output_shapes.append(layers_to_output_shapes[key])
            # Store in cache.
            self._output_shape_cache[cache_key] = output_shapes
            if isinstance(output_shapes, list) and len(output_shapes) == 1:
                return output_shapes[0]
            return output_shapes

    def run_internal_graph(self, inputs, masks=None):
        """Computes output tensors for new inputs.

        # Note:
            - Expects `inputs` to be a list (potentially with 1 element).
            - Can be run on non-Keras tensors.

        # Arguments
            inputs: List of tensors
            masks: List of masks (tensors or None).

        # Returns
            Three lists: output_tensors, output_masks, output_shapes
        """
        if masks is None:
            masks = [None for _ in range(len(inputs))]

        # Dictionary mapping reference tensors to tuples
        # (computed tensor, compute mask)
        # we assume a 1:1 mapping from tensor to mask
        # TODO: raise exception when a `.compute_mask()` call
        # does not return a list the same size as `call`
        tensor_map = {}
        for x, y, mask in zip(self.inputs, inputs, masks):
            tensor_map[str(id(x))] = (y, mask)

        depth_keys = list(self._nodes_by_depth.keys())
        depth_keys.sort(reverse=True)
        for depth in depth_keys:
            nodes = self._nodes_by_depth[depth]
            for node in nodes:
                # This is always a single layer, never a list.
                layer = node.outbound_layer

                reference_input_tensors = node.input_tensors
                reference_output_tensors = node.output_tensors

                # If all previous input tensors are available in tensor_map,
                # then call node.inbound_layer on them.
                computed_data = []  # List of tuples (input, mask).
                for x in reference_input_tensors:
                    if str(id(x)) in tensor_map:
                        computed_data.append(tensor_map[str(id(x))])

                if len(computed_data) == len(reference_input_tensors):
                    # call layer
                    with K.name_scope(layer.name):
                        if node.arguments:
                            kwargs = node.arguments
                        else:
                            kwargs = {}
                        if len(computed_data) == 1:
                            computed_tensor, computed_mask = computed_data[0]
                            if has_arg(layer.call, 'mask'):
                                if 'mask' not in kwargs:
                                    kwargs['mask'] = computed_mask
                            output_tensors = _to_list(layer.call(computed_tensor, **kwargs))
                            output_masks = _to_list(layer.compute_mask(computed_tensor,
                                                                       computed_mask))
                            computed_tensors = [computed_tensor]
                            computed_masks = [computed_mask]
                        else:
                            computed_tensors = [x[0] for x in computed_data]
                            computed_masks = [x[1] for x in computed_data]
                            if has_arg(layer.call, 'mask'):
                                if 'mask' not in kwargs:
                                    kwargs['mask'] = computed_masks
                            output_tensors = _to_list(layer.call(computed_tensors, **kwargs))
                            output_masks = _to_list(layer.compute_mask(computed_tensors,
                                                                       computed_masks))

                        # Apply activity regularizer if any:
                        if hasattr(layer, 'activity_regularizer') and layer.activity_regularizer is not None:
                            regularization_losses = [layer.activity_regularizer(x) for x in computed_tensors]
                            layer.add_loss(regularization_losses, computed_tensors)

                    # Update model updates and losses:
                    # Keep track of updates that depend on the inputs
                    # (e.g. BN updates).
                    self.add_update(layer.get_updates_for(computed_tensors), inputs)
                    # Keep track of unconditional updates (e.g. a counter).
                    self.add_update(layer.get_updates_for(None), None)
                    # Keep track of losses that depend on the inputs
                    # (e.g. activity regularizers).
                    self.add_loss(layer.get_losses_for(computed_tensors), inputs)
                    # Keep track of unconditional losses
                    # (e.g. weight regularizers).
                    self.add_loss(layer.get_losses_for(None), None)

                    # Update _keras_shape.
                    if all([hasattr(x, '_keras_shape') for x in computed_tensors]):
                        if len(computed_tensors) == 1:
                            shapes = _to_list(layer.compute_output_shape(computed_tensors[0]._keras_shape))
                            uses_learning_phase = computed_tensors[0]._uses_learning_phase
                        else:
                            shapes = _to_list(layer.compute_output_shape([x._keras_shape for x in computed_tensors]))
                            uses_learning_phase = any([x._uses_learning_phase for x in computed_tensors])
                        for x, s in zip(output_tensors, shapes):
                            x._keras_shape = s
                            x._uses_learning_phase = getattr(x, '_uses_learning_phase', False) or uses_learning_phase

                    # Update tensor_map.
                    for x, y, mask in zip(reference_output_tensors, output_tensors, output_masks):
                        tensor_map[str(id(x))] = (y, mask)

        output_tensors = []
        output_masks = []
        output_shapes = []
        for x in self.outputs:
            assert str(id(x)) in tensor_map, 'Could not compute output ' + str(x)
            tensor, mask = tensor_map[str(id(x))]
            if hasattr(tensor, '_keras_shape') and output_shapes is not None:
                shape = tensor._keras_shape
                output_shapes.append(shape)
            else:
                output_shapes = None
            output_tensors.append(tensor)
            output_masks.append(mask)

        # Update cache;
        # keys are based on ids on input tensors and inputs masks.
        cache_key = ','.join([str(id(x)) for x in inputs])
        cache_key += '_' + ','.join([str(id(x)) for x in masks])

        if len(output_tensors) == 1:
            output_tensors = output_tensors[0]
            self._output_tensor_cache[cache_key] = output_tensors
        else:
            self._output_tensor_cache[cache_key] = output_tensors

        if len(output_masks) == 1:
            output_masks = output_masks[0]
            self._output_mask_cache[cache_key] = output_masks
        else:
            self._output_mask_cache[cache_key] = output_masks

        if output_shapes is not None:
            input_shapes = [x._keras_shape for x in inputs]
            cache_key = ','.join([str(x) for x in input_shapes])
            if len(output_shapes) == 1:
                output_shapes = output_shapes[0]
                self._output_shape_cache[cache_key] = output_shapes
            else:
                self._output_shape_cache[cache_key] = output_shapes
        return output_tensors, output_masks, output_shapes

    def get_config(self):
        config = {
            'name': self.name,
        }

        # Build a map from a layer unique name (self._node_key)
        # to the index of the nodes that are saved in the config.
        # Only nodes in container_nodes are saved.
        node_conversion_map = {}
        for layer in self.layers:
            if issubclass(layer.__class__, Container):
                # Containers start with a pre-existing node
                # linking their input to output.
                kept_nodes = 1
            else:
                kept_nodes = 0
            for original_node_index, node in enumerate(layer._inbound_nodes):
                node_key = self._node_key(layer, original_node_index)
                if node_key in self._container_nodes:
                    # i.e. we mark it to be saved
                    node_conversion_map[node_key] = kept_nodes
                    kept_nodes += 1

        # serialize and save the layers in layer_configs
        layer_configs = []
        for layer in self.layers:  # From the earliest layers on.
            layer_class_name = layer.__class__.__name__
            layer_config = layer.get_config()
            filtered_inbound_nodes = []
            for original_node_index, node in enumerate(layer._inbound_nodes):
                node_key = self._node_key(layer, original_node_index)
                if node_key in self._container_nodes:
                    # The node is relevant to the model:
                    # add to filtered_inbound_nodes.
                    if node.arguments:
                        try:
                            json.dumps(node.arguments)
                            kwargs = node.arguments
                        except TypeError:
                            warnings.warn(
                                'Layer ' + layer.name +
                                ' was passed non-serializable keyword arguments: ' +
                                str(node.arguments) + '. They will not be included '
                                'in the serialized model (and thus will be missing '
                                'at deserialization time).')
                            kwargs = {}
                    else:
                        kwargs = {}
                    if node.inbound_layers:
                        node_data = []
                        for i in range(len(node.inbound_layers)):
                            inbound_layer = node.inbound_layers[i]
                            node_index = node.node_indices[i]
                            tensor_index = node.tensor_indices[i]

                            new_node_index = node_conversion_map.get(
                                self._node_key(inbound_layer, node_index), 0)
                            node_data.append([inbound_layer.name,
                                              new_node_index,
                                              tensor_index,
                                              kwargs])
                        filtered_inbound_nodes.append(node_data)
            layer_configs.append({
                'name': layer.name,
                'class_name': layer_class_name,
                'config': layer_config,
                'inbound_nodes': filtered_inbound_nodes,
            })
        config['layers'] = layer_configs

        # Gather info about inputs and outputs.
        model_inputs = []
        for i in range(len(self.input_layers)):
            layer = self.input_layers[i]
            node_index = self.input_layers_node_indices[i]

            node_key = self._node_key(layer, node_index)
            if node_key not in self._container_nodes:
                continue
            new_node_index = node_conversion_map[node_key]
            tensor_index = self.input_layers_tensor_indices[i]
            model_inputs.append([layer.name, new_node_index, tensor_index])
        config['input_layers'] = model_inputs
        model_outputs = []
        for i in range(len(self.output_layers)):
            layer = self.output_layers[i]
            node_index = self.output_layers_node_indices[i]

            node_key = self._node_key(layer, node_index)
            if node_key not in self._container_nodes:
                continue
            new_node_index = node_conversion_map[node_key]
            tensor_index = self.output_layers_tensor_indices[i]
            model_outputs.append([layer.name, new_node_index, tensor_index])
        config['output_layers'] = model_outputs
        return copy.deepcopy(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Instantiates a Model from its config (output of `get_config()`).

        # Arguments
            config: Model config dictionary.
            custom_objects: Optional dictionary mapping names
                (strings) to custom classes or functions to be
                considered during deserialization.

        # Returns
            A model instance.

        # Raises
            ValueError: In case of improperly formatted config dict.
        """
        # Layer instances created during
        # the graph reconstruction process
        created_layers = {}

        # Dictionary mapping layer instances to
        # node data that specifies a layer call.
        # It acts as a queue that maintains any unprocessed
        # layer call until it becomes possible to process it
        # (i.e. until the input tensors to the call all exist).
        unprocessed_nodes = {}

        def add_unprocessed_node(layer, node_data):
            if layer not in unprocessed_nodes:
                unprocessed_nodes[layer] = [node_data]
            else:
                unprocessed_nodes[layer].append(node_data)

        def process_node(layer, node_data):
            input_tensors = []
            for input_data in node_data:
                inbound_layer_name = input_data[0]
                inbound_node_index = input_data[1]
                inbound_tensor_index = input_data[2]
                if len(input_data) == 3:
                    kwargs = {}
                elif len(input_data) == 4:
                    kwargs = input_data[3]
                else:
                    raise ValueError('Improperly formatted model config.')
                if inbound_layer_name not in created_layers:
                    add_unprocessed_node(layer, node_data)
                    return
                inbound_layer = created_layers[inbound_layer_name]
                if len(inbound_layer._inbound_nodes) <= inbound_node_index:
                    add_unprocessed_node(layer, node_data)
                    return
                inbound_node = inbound_layer._inbound_nodes[inbound_node_index]
                input_tensors.append(inbound_node.output_tensors[inbound_tensor_index])
            # Call layer on its inputs, thus creating the node
            # and building the layer if needed.
            if input_tensors:
                if len(input_tensors) == 1:
                    layer(input_tensors[0], **kwargs)
                else:
                    layer(input_tensors, **kwargs)

        def process_layer(layer_data):
            """Deserialize a layer, then call it on appropriate inputs.

            # Arguments
                layer_data: layer config dict.

            # Raises
                ValueError: In case of improperly formatted `layer_data` dict.
            """
            layer_name = layer_data['name']

            # Instantiate layer.
            from ..layers import deserialize as deserialize_layer

            layer = deserialize_layer(layer_data,
                                      custom_objects=custom_objects)
            created_layers[layer_name] = layer

            # Gather layer inputs.
            inbound_nodes_data = layer_data['inbound_nodes']
            for node_data in inbound_nodes_data:
                # We don't process nodes (i.e. make layer calls)
                # on the fly because the inbound node may not yet exist,
                # in case of layer shared at different topological depths
                # (e.g. a model such as A(B(A(B(x)))))
                add_unprocessed_node(layer, node_data)

        # First, we create all layers and enqueue nodes to be processed
        for layer_data in config['layers']:
            process_layer(layer_data)
        # Then we process nodes in order of layer depth.
        # Nodes that cannot yet be processed (if the inbound node
        # does not yet exist) are re-enqueued, and the process
        # is repeated until all nodes are processed.
        while unprocessed_nodes:
            for layer_data in config['layers']:
                layer = created_layers[layer_data['name']]
                if layer in unprocessed_nodes:
                    for node_data in unprocessed_nodes.pop(layer):
                        process_node(layer, node_data)

        name = config.get('name')
        input_tensors = []
        output_tensors = []
        for layer_data in config['input_layers']:
            layer_name, node_index, tensor_index = layer_data
            assert layer_name in created_layers
            layer = created_layers[layer_name]
            layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
            input_tensors.append(layer_output_tensors[tensor_index])
        for layer_data in config['output_layers']:
            layer_name, node_index, tensor_index = layer_data
            assert layer_name in created_layers
            layer = created_layers[layer_name]
            layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
            output_tensors.append(layer_output_tensors[tensor_index])
        return cls(inputs=input_tensors, outputs=output_tensors, name=name)

    def save(self, filepath, overwrite=True, include_optimizer=True):
        """Save the model to a single HDF5 file.

        The savefile includes:
            - The model architecture, allowing to re-instantiate the model.
            - The model weights.
            - The state of the optimizer, allowing to resume training
                exactly where you left off.

        This allows you to save the entirety of the state of a model
        in a single file.

        Saved models can be reinstantiated via `keras.models.load_model`.
        The model returned by `load_model`
        is a compiled model ready to be used (unless the saved model
        was never compiled in the first place).

        # Arguments
            filepath: String, path to the file to save the weights to.
            overwrite: Whether to silently overwrite any existing file at the
                target location, or provide the user with a manual prompt.
            include_optimizer: If True, save optimizer's state together.

        # Example

        ```python
        from keras.models import load_model

        model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
        del model  # deletes the existing model

        # returns a compiled model
        # identical to the previous one
        model = load_model('my_model.h5')
        ```
        """
        from ..models import save_model
        save_model(self, filepath, overwrite, include_optimizer)

    def save_weights(self, filepath, overwrite=True):
        """Dumps all layer weights to a HDF5 file.

        The weight file has:
            - `layer_names` (attribute), a list of strings
                (ordered names of model layers).
            - For every layer, a `group` named `layer.name`
                - For every such layer group, a group attribute `weight_names`,
                    a list of strings
                    (ordered names of weights tensor of the layer).
                - For every weight in the layer, a dataset
                    storing the weight value, named after the weight tensor.

        # Arguments
            filepath: String, path to the file to save the weights to.
            overwrite: Whether to silently overwrite any existing file at the
                target location, or provide the user with a manual prompt.

        # Raises
            ImportError: If h5py is not available.
        """
        if h5py is None:
            raise ImportError('`save_weights` requires h5py.')
        # If file exists and should not be overwritten:
        if not overwrite and os.path.isfile(filepath):
            proceed = ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return
        with h5py.File(filepath, 'w') as f:
            save_weights_to_hdf5_group(f, self.layers)
            f.flush()

    def load_weights(self, filepath, by_name=False,
                     skip_mismatch=False, reshape=False):
        """Loads all layer weights from a HDF5 save file.

        If `by_name` is False (default) weights are loaded
        based on the network's topology, meaning the architecture
        should be the same as when the weights were saved.
        Note that layers that don't have weights are not taken
        into account in the topological ordering, so adding or
        removing layers is fine as long as they don't have weights.

        If `by_name` is True, weights are loaded into layers
        only if they share the same name. This is useful
        for fine-tuning or transfer-learning models where
        some of the layers have changed.

        # Arguments
            filepath: String, path to the weights file to load.
            by_name: Boolean, whether to load weights by name
                or by topological order.
            skip_mismatch: Boolean, whether to skip loading of layers
                where there is a mismatch in the number of weights,
                or a mismatch in the shape of the weight
                (only valid when `by_name`=True).
        reshape: Reshape weights to fit the layer when the correct number
            of values are present but the shape does not match.


        # Raises
            ImportError: If h5py is not available.
        """
        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        with h5py.File(filepath, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']
            if by_name:
                load_weights_from_hdf5_group_by_name(
                    f, self.layers, skip_mismatch=skip_mismatch,
                    reshape=reshape)
            else:
                load_weights_from_hdf5_group(
                    f, self.layers, reshape=reshape)

    def _updated_config(self):
        """Util hared between different serialization methods.

        # Returns
            Model config with Keras version information added.
        """
        from .. import __version__ as keras_version

        config = self.get_config()
        model_config = {
            'class_name': self.__class__.__name__,
            'config': config,
            'keras_version': keras_version,
            'backend': K.backend()
        }
        return model_config

    def to_json(self, **kwargs):
        """Returns a JSON string containing the network configuration.

        To load a network from a JSON save file, use
        `keras.models.model_from_json(json_string, custom_objects={})`.

        # Arguments
            **kwargs: Additional keyword arguments
                to be passed to `json.dumps()`.

        # Returns
            A JSON string.
        """
        def get_json_type(obj):
            # If obj is any numpy type
            if type(obj).__module__ == np.__name__:
                return obj.item()

            # If obj is a python 'type'
            if type(obj).__name__ == type.__name__:
                return obj.__name__

            raise TypeError('Not JSON Serializable:', obj)

        model_config = self._updated_config()
        return json.dumps(model_config, default=get_json_type, **kwargs)

    def to_yaml(self, **kwargs):
        """Returns a yaml string containing the network configuration.

        To load a network from a yaml save file, use
        `keras.models.model_from_yaml(yaml_string, custom_objects={})`.

        `custom_objects` should be a dictionary mapping
        the names of custom losses / layers / etc to the corresponding
        functions / classes.

        # Arguments
            **kwargs: Additional keyword arguments
                to be passed to `yaml.dump()`.

        # Returns
            A YAML string.
        """
        return yaml.dump(self._updated_config(), **kwargs)

    def summary(self, line_length=None, positions=None, print_fn=None):
        """Prints a string summary of the network.

        # Arguments
            line_length: Total length of printed lines
                (e.g. set this to adapt the display to different
                terminal window sizes).
            positions: Relative or absolute positions of log elements
                in each line. If not provided,
                defaults to `[.33, .55, .67, 1.]`.
            print_fn: Print function to use.
                It will be called on each line of the summary.
                You can set it to a custom function
                in order to capture the string summary.
                It defaults to `print` (prints to stdout).
        """
        return print_layer_summary(self,
                                   line_length=line_length,
                                   positions=positions,
                                   print_fn=print_fn)


def get_source_inputs(tensor, layer=None, node_index=None):
    """Returns the list of input tensors necessary to compute `tensor`.

    Output will always be a list of tensors
    (potentially with 1 element).

    # Arguments
        tensor: The tensor to start from.
        layer: Origin layer of the tensor. Will be
            determined via tensor._keras_history if not provided.
        node_index: Origin node index of the tensor.

    # Returns
        List of input tensors.
    """
    if not hasattr(tensor, '_keras_history'):
        return tensor

    if layer is None or node_index:
        layer, node_index, _ = tensor._keras_history
    if not layer._inbound_nodes:
        return [tensor]
    else:
        node = layer._inbound_nodes[node_index]
        if not node.inbound_layers:
            # Reached an Input layer, stop recursion.
            return node.input_tensors
        else:
            source_tensors = []
            for i in range(len(node.inbound_layers)):
                x = node.input_tensors[i]
                layer = node.inbound_layers[i]
                node_index = node.node_indices[i]
                previous_sources = get_source_inputs(x,
                                                     layer,
                                                     node_index)
                # Avoid input redundancy.
                for x in previous_sources:
                    if x not in source_tensors:
                        source_tensors.append(x)
            return source_tensors


def _to_list(x):
    """Normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.

    # Arguments
        x: target object to be normalized.

    # Returns
        A list.
    """
    if isinstance(x, list):
        return x
    return [x]


def _object_list_uid(object_list):
    object_list = _to_list(object_list)
    return ', '.join([str(abs(id(x))) for x in object_list])


def _is_all_none(iterable_or_element):
    if not isinstance(iterable_or_element, (list, tuple)):
        iterable = [iterable_or_element]
    else:
        iterable = iterable_or_element
    for element in iterable:
        if element is not None:
            return False
    return True


def _collect_previous_mask(input_tensors):
    """Retrieves the output mask(s) of the previous node.

    # Arguments
        input_tensors: A tensor or list of tensors.

    # Returns
        A mask tensor or list of mask tensors.
    """
    input_tensors = _to_list(input_tensors)
    masks = []
    for x in input_tensors:
        if hasattr(x, '_keras_history'):
            inbound_layer, node_index, tensor_index = x._keras_history
            node = inbound_layer._inbound_nodes[node_index]
            mask = node.output_masks[tensor_index]
            masks.append(mask)
        else:
            masks.append(None)
    if len(masks) == 1:
        return masks[0]
    return masks


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
    input_tensors = _to_list(input_tensors)
    shapes = []
    for x in input_tensors:
        try:
            shapes.append(K.int_shape(x))
        except TypeError:
            shapes.append(None)
    if len(shapes) == 1:
        return shapes[0]
    return shapes


def save_weights_to_hdf5_group(f, layers):
    from .. import __version__ as keras_version

    f.attrs['layer_names'] = [layer.name.encode('utf8') for layer in layers]
    f.attrs['backend'] = K.backend().encode('utf8')
    f.attrs['keras_version'] = str(keras_version).encode('utf8')

    for layer in layers:
        g = f.create_group(layer.name)
        symbolic_weights = layer.weights
        weight_values = K.batch_get_value(symbolic_weights)
        weight_names = []
        for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
            if hasattr(w, 'name') and w.name:
                name = str(w.name)
            else:
                name = 'param_' + str(i)
            weight_names.append(name.encode('utf8'))
        g.attrs['weight_names'] = weight_names
        for name, val in zip(weight_names, weight_values):
            param_dset = g.create_dataset(name, val.shape,
                                          dtype=val.dtype)
            if not val.shape:
                # scalar
                param_dset[()] = val
            else:
                param_dset[:] = val


def preprocess_weights_for_loading(layer, weights,
                                   original_keras_version=None,
                                   original_backend=None,
                                   reshape=False):
    """Converts layers weights from Keras 1 format to Keras 2.

    # Arguments
        layer: Layer instance.
        weights: List of weights values (Numpy arrays).
        original_keras_version: Keras version for the weights, as a string.
        original_backend: Keras backend the weights were trained with,
            as a string.
        reshape: Reshape weights to fit the layer when the correct number
            of values are present but the shape does not match.

    # Returns
        A list of weights values (Numpy arrays).
    """
    if layer.__class__.__name__ == 'Bidirectional':
        num_weights_per_layer = len(weights) // 2
        forward_weights = preprocess_weights_for_loading(layer.forward_layer,
                                                         weights[:num_weights_per_layer],
                                                         original_keras_version,
                                                         original_backend)
        backward_weights = preprocess_weights_for_loading(layer.backward_layer,
                                                          weights[num_weights_per_layer:],
                                                          original_keras_version,
                                                          original_backend)
        weights = forward_weights + backward_weights

    if original_keras_version == '1':
        if layer.__class__.__name__ == 'TimeDistributed':
            weights = preprocess_weights_for_loading(layer.layer,
                                                     weights,
                                                     original_keras_version,
                                                     original_backend)

        if layer.__class__.__name__ == 'Conv1D':
            shape = weights[0].shape
            # Handle Keras 1.1 format
            if shape[:2] != (layer.kernel_size[0], 1) or shape[3] != layer.filters:
                # Legacy shape:
                # (filters, input_dim, filter_length, 1)
                assert shape[0] == layer.filters and shape[2:] == (layer.kernel_size[0], 1)
                weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
            weights[0] = weights[0][:, 0, :, :]

        if layer.__class__.__name__ == 'Conv2D':
            if layer.data_format == 'channels_first':
                # old: (filters, stack_size, kernel_rows, kernel_cols)
                # new: (kernel_rows, kernel_cols, stack_size, filters)
                weights[0] = np.transpose(weights[0], (2, 3, 1, 0))

        if layer.__class__.__name__ == 'Conv2DTranspose':
            if layer.data_format == 'channels_last':
                # old: (kernel_rows, kernel_cols, stack_size, filters)
                # new: (kernel_rows, kernel_cols, filters, stack_size)
                weights[0] = np.transpose(weights[0], (0, 1, 3, 2))
            if layer.data_format == 'channels_first':
                # old: (filters, stack_size, kernel_rows, kernel_cols)
                # new: (kernel_rows, kernel_cols, filters, stack_size)
                weights[0] = np.transpose(weights[0], (2, 3, 0, 1))

        if layer.__class__.__name__ == 'Conv3D':
            if layer.data_format == 'channels_first':
                # old: (filters, stack_size, ...)
                # new: (..., stack_size, filters)
                weights[0] = np.transpose(weights[0], (2, 3, 4, 1, 0))

        if layer.__class__.__name__ == 'GRU':
            if len(weights) == 9:
                kernel = np.concatenate([weights[0],
                                         weights[3],
                                         weights[6]], axis=-1)
                recurrent_kernel = np.concatenate([weights[1],
                                                   weights[4],
                                                   weights[7]], axis=-1)
                bias = np.concatenate([weights[2],
                                       weights[5],
                                       weights[8]], axis=-1)
                weights = [kernel, recurrent_kernel, bias]

        if layer.__class__.__name__ == 'LSTM':
            if len(weights) == 12:
                # old: i, c, f, o
                # new: i, f, c, o
                kernel = np.concatenate([weights[0],
                                         weights[6],
                                         weights[3],
                                         weights[9]], axis=-1)
                recurrent_kernel = np.concatenate([weights[1],
                                                   weights[7],
                                                   weights[4],
                                                   weights[10]], axis=-1)
                bias = np.concatenate([weights[2],
                                       weights[8],
                                       weights[5],
                                       weights[11]], axis=-1)
                weights = [kernel, recurrent_kernel, bias]

        if layer.__class__.__name__ == 'ConvLSTM2D':
            if len(weights) == 12:
                kernel = np.concatenate([weights[0],
                                         weights[6],
                                         weights[3],
                                         weights[9]], axis=-1)
                recurrent_kernel = np.concatenate([weights[1],
                                                   weights[7],
                                                   weights[4],
                                                   weights[10]], axis=-1)
                bias = np.concatenate([weights[2],
                                       weights[8],
                                       weights[5],
                                       weights[11]], axis=-1)
                if layer.data_format == 'channels_first':
                    # old: (filters, stack_size, kernel_rows, kernel_cols)
                    # new: (kernel_rows, kernel_cols, stack_size, filters)
                    kernel = np.transpose(kernel, (2, 3, 1, 0))
                    recurrent_kernel = np.transpose(recurrent_kernel,
                                                    (2, 3, 1, 0))
                weights = [kernel, recurrent_kernel, bias]

        if layer.__class__.__name__ in ['Model', 'Sequential']:
            new_weights = []
            # trainable weights
            for sublayer in layer.layers:
                num_weights = len(sublayer.trainable_weights)
                if num_weights > 0:
                    new_weights.extend(preprocess_weights_for_loading(
                        layer=sublayer,
                        weights=weights[:num_weights],
                        original_keras_version=original_keras_version,
                        original_backend=original_backend))
                    weights = weights[num_weights:]

            # non-trainable weights
            for sublayer in layer.layers:
                num_weights = len([l for l in sublayer.weights if l not in sublayer.trainable_weights])
                if num_weights > 0:
                    new_weights.extend(preprocess_weights_for_loading(
                        layer=sublayer,
                        weights=weights[:num_weights],
                        original_keras_version=original_keras_version,
                        original_backend=original_backend))
                    weights = weights[num_weights:]
            weights = new_weights

    conv_layers = ['Conv1D',
                   'Conv2D',
                   'Conv3D',
                   'Conv2DTranspose',
                   'ConvLSTM2D']
    if layer.__class__.__name__ in conv_layers:
        layer_weights_shape = K.int_shape(layer.weights[0])
        if _need_convert_kernel(original_backend):
            weights[0] = conv_utils.convert_kernel(weights[0])
            if layer.__class__.__name__ == 'ConvLSTM2D':
                weights[1] = conv_utils.convert_kernel(weights[1])
        if reshape and layer_weights_shape != weights[0].shape:
            if weights[0].size != np.prod(layer_weights_shape):
                raise ValueError('Weights must be of equal size to ' +
                                 'apply a reshape operation. ' +
                                 'Layer ' + layer.name +
                                 '\'s weights have shape ' +
                                 str(layer_weights_shape) + ' and size ' +
                                 str(np.prod(layer_weights_shape)) + '. ' +
                                 'The weights for loading have shape ' +
                                 str(weights[0].shape) + ' and size ' +
                                 str(weights[0].size) + '. ')
            weights[0] = np.reshape(weights[0], layer_weights_shape)
        elif layer_weights_shape != weights[0].shape:
            weights[0] = np.transpose(weights[0], (3, 2, 0, 1))
            if layer.__class__.__name__ == 'ConvLSTM2D':
                weights[1] = np.transpose(weights[1], (3, 2, 0, 1))

    # convert the weights of CuDNNLSTM so that they could be loaded into LSTM
    if layer.__class__.__name__ == 'LSTM' and len(weights) == 3:
        # determine if we're loading a CuDNNLSTM layer from the number of bias weights:
        # CuDNNLSTM has (units * 8) weights; while LSTM has (units * 4)
        # if there's no bias weight in the file, skip this conversion
        units = weights[1].shape[0]
        bias = weights[2]
        if len(bias) == units * 8:
            # reshape the kernels
            kernels = np.split(weights[0], 4, axis=1)
            kernels = [kernel.reshape(-1).reshape(kernel.shape, order='F') for kernel in kernels]
            weights[0] = np.concatenate(kernels, axis=1)

            # transpose the recurrent kernels
            recurrent_kernels = np.split(weights[1], 4, axis=1)
            recurrent_kernels = [kernel.T for kernel in recurrent_kernels]
            weights[1] = np.concatenate(recurrent_kernels, axis=1)

            # split the bias into half and merge
            weights[2] = bias[:units * 4] + bias[units * 4:]

    return weights


def _need_convert_kernel(original_backend):
    """Check if conversion on kernel matrices is required during weight loading.

    The convolution operation is implemented differently in different backends.
    While TH implements convolution, TF and CNTK implement the correlation operation.
    So the channel axis needs to be flipped when we're loading TF weights onto a TH model,
    or vice verca. However, there's no conversion required between TF and CNTK.

    # Arguments
        original_backend: Keras backend the weights were trained with, as a string.

    # Returns
        `True` if conversion on kernel matrices is required, otherwise `False`.
    """
    if original_backend is None:
        # backend information not available
        return False
    uses_correlation = {'tensorflow': True,
                        'theano': False,
                        'cntk': True}
    return uses_correlation[original_backend] != uses_correlation[K.backend()]


def load_weights_from_hdf5_group(f, layers, reshape=False):
    """Implements topological (order-based) weight loading.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: a list of target layers.
        reshape: Reshape weights to fit the layer when the correct number
            of values are present but the shape does not match.

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file.
    """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    filtered_layers = []
    for layer in layers:
        weights = layer.weights
        if weights:
            filtered_layers.append(layer)

    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        if weight_names:
            filtered_layer_names.append(name)
    layer_names = filtered_layer_names
    if len(layer_names) != len(filtered_layers):
        raise ValueError('You are trying to load a weight file '
                         'containing ' + str(len(layer_names)) +
                         ' layers into a model with ' +
                         str(len(filtered_layers)) + ' layers.')

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]
        layer = filtered_layers[k]
        symbolic_weights = layer.weights
        weight_values = preprocess_weights_for_loading(layer,
                                                       weight_values,
                                                       original_keras_version,
                                                       original_backend,
                                                       reshape=reshape)
        if len(weight_values) != len(symbolic_weights):
            raise ValueError('Layer #' + str(k) +
                             ' (named "' + layer.name +
                             '" in the current model) was found to '
                             'correspond to layer ' + name +
                             ' in the save file. '
                             'However the new layer ' + layer.name +
                             ' expects ' + str(len(symbolic_weights)) +
                             ' weights, but the saved weights have ' +
                             str(len(weight_values)) +
                             ' elements.')
        weight_value_tuples += zip(symbolic_weights, weight_values)
    K.batch_set_value(weight_value_tuples)


def load_weights_from_hdf5_group_by_name(f, layers, skip_mismatch=False,
                                         reshape=False):
    """Implements name-based weight loading.

    (instead of topological weight loading).

    Layers that have no matching name are skipped.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: A list of target layers.
        skip_mismatch: Boolean, whether to skip loading of layers
            where there is a mismatch in the number of weights,
            or a mismatch in the shape of the weights.
        reshape: Reshape weights to fit the layer when the correct number
            of values are present but the shape does not match.

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file and skip_mismatch=False.
    """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    # New file format.
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            weight_values = preprocess_weights_for_loading(
                layer,
                weight_values,
                original_keras_version,
                original_backend,
                reshape=reshape)
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:
                    warnings.warn('Skipping loading of weights for layer {}'.format(layer.name) +
                                  ' due to mismatch in number of weights' +
                                  ' ({} vs {}).'.format(len(symbolic_weights), len(weight_values)))
                    continue
                else:
                    raise ValueError('Layer #' + str(k) +
                                     ' (named "' + layer.name +
                                     '") expects ' +
                                     str(len(symbolic_weights)) +
                                     ' weight(s), but the saved weights' +
                                     ' have ' + str(len(weight_values)) +
                                     ' element(s).')
            # Set values.
            for i in range(len(weight_values)):
                if skip_mismatch:
                    if K.int_shape(symbolic_weights[i]) != weight_values[i].shape:
                        warnings.warn('Skipping loading of weights for layer {}'.format(layer.name) +
                                      ' due to mismatch in shape' +
                                      ' ({} vs {}).'.format(
                                          symbolic_weights[i].shape,
                                          weight_values[i].shape))
                        continue

                weight_value_tuples.append((symbolic_weights[i],
                                            weight_values[i]))

    K.batch_set_value(weight_value_tuples)
