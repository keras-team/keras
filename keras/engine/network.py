"""A `Network` is way to compose layers: the topological form of a `Model`.
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
from six.moves import zip

from . import saving
from .base_layer import Layer
from .base_layer import Node
from .input_layer import InputLayer
from .. import backend as K
from ..utils.io_utils import ask_to_proceed_with_overwrite
from ..utils.layer_utils import print_summary as print_layer_summary
from ..utils.layer_utils import get_source_inputs
from ..utils.generic_utils import has_arg
from ..utils.generic_utils import to_list
from ..utils.generic_utils import object_list_uid
from ..utils.generic_utils import unpack_singleton
from ..legacy import interfaces

try:
    import h5py
except ImportError:
    h5py = None


class Network(Layer):
    """A Network is a directed acyclic graph of layers.

    It is the topological form of a "model". A Model
    is simply a Network with added training routines.

    # Properties
        name
        inputs
        outputs
        layers
        input_spec (list of class instances)
            each entry describes one required input:
                - ndim
                - dtype
        trainable (boolean)
        input_shape
        output_shape
        weights (list of variables)
        trainable_weights (list of variables)
        non_trainable_weights (list of variables)
        losses
        updates
        state_updates
        stateful

    # Methods
        __call__
        summary
        get_layer
        get_weights
        set_weights
        get_config
        compute_output_shape
        save
        add_loss
        add_update
        get_losses_for
        get_updates_for
        to_json
        to_yaml
        reset_states

    # Class Methods
        from_config

    # Raises
        TypeError: if input tensors are not Keras tensors
            (tensors returned by `Input`).
    """

    @interfaces.legacy_model_constructor_support
    def __init__(self, *args, **kwargs):
        # Signature detection
        if (len(args) == 2 or
            len(args) == 1 and 'outputs' in kwargs or
                'inputs' in kwargs and 'outputs' in kwargs):
            # Graph network
            self._init_graph_network(*args, **kwargs)
        else:
            # Subclassed network
            self._init_subclassed_network(**kwargs)

    def _base_init(self, name=None):
        # The following are implemented as property functions:
        # self.trainable_weights
        # self.non_trainable_weights
        # self.input_spec
        # self.losses
        # self.updates

        # Handle `name` argument.
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        # This acts just like the `trainable` attribute of any layer instance.
        # It does not affect users of the underlying layers, only users of the
        # Network instance.
        self.trainable = True
        self._is_compiled = False
        self._expects_training_arg = False
        self._initial_weights = None

        self.supports_masking = False
        if not hasattr(self, 'optimizer'):
            # Don't reset optimizer if already set.
            self.optimizer = None

        # Private attributes to implement compatibility with Layer.
        self._updates = []
        self._losses = []
        self._per_input_losses = {}
        self._per_input_updates = {}

        # All layers in order of horizontal graph traversal.
        # Entries are unique. Includes input and output layers.
        self._layers = []

        # Used only in conjunction with graph-networks
        self._outbound_nodes = []
        self._inbound_nodes = []

    def _init_graph_network(self, inputs, outputs, name=None):
        self._uses_inputs_arg = True
        # Normalize and set self.inputs, self.outputs.
        if isinstance(inputs, (list, tuple)):
            self.inputs = list(inputs)  # Tensor or list of tensors.
        else:
            self.inputs = [inputs]
        if isinstance(outputs, (list, tuple)):
            self.outputs = list(outputs)
        else:
            self.outputs = [outputs]

        # User-provided argument validation.
        # Check for redundancy in inputs.
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError('The list of inputs passed to the model '
                             'is redundant. '
                             'All inputs should only appear once.'
                             ' Found: ' + str(self.inputs))
        for x in self.inputs:
            # Check that x has appropriate `_keras_history` metadata.
            if not hasattr(x, '_keras_history'):
                cls_name = self.__class__.__name__
                raise ValueError('Input tensors to a ' + cls_name + ' ' +
                                 'must come from `keras.layers.Input`. '
                                 'Received: ' + str(x) +
                                 ' (missing previous layer metadata).')
            # Check that x is an input tensor.
            layer, node_index, tensor_index = x._keras_history
            if (len(layer._inbound_nodes) > 1 or
                    (layer._inbound_nodes and
                     layer._inbound_nodes[0].inbound_layers)):
                cls_name = self.__class__.__name__
                warnings.warn(cls_name + ' inputs must come from '
                              '`keras.layers.Input` '
                              '(thus holding past layer metadata), '
                              'they cannot be the output of '
                              'a previous non-Input layer. '
                              'Here, a tensor specified as '
                              'input to your model '
                              'was not an Input tensor, '
                              'it was generated by layer ' +
                              layer.name + '.\n'
                              'Note that input tensors are '
                              'instantiated via '
                              '`tensor = keras.layers.Input(shape)`.\n'
                              'The tensor that caused the issue was: ' +
                              str(x.name))
        for x in self.outputs:
            if not hasattr(x, '_keras_history'):
                cls_name = self.__class__.__name__
                raise ValueError('Output tensors to a ' + cls_name +
                                 ' must be '
                                 'the output of a Keras `Layer` '
                                 '(thus holding past layer metadata). '
                                 'Found: ' + str(x))
        self._base_init(name=name)
        self._compute_previous_mask = (
            has_arg(self.call, 'mask') or
            hasattr(self, 'compute_mask'))
        # A Network does not create weights of its own,
        # thus it is already built.
        self.built = True
        self._is_graph_network = True

        self._input_layers = []
        self._output_layers = []
        self._input_coordinates = []
        self._output_coordinates = []

        # This is for performance optimization when calling the Network on new
        # inputs. Every time the Network is called on a set on input tensors,
        # we compute the output tensors,
        # output masks and output shapes in one pass,
        # then cache them here. When any of these outputs is queried later, we
        # retrieve it from there instead of recomputing it.
        self._output_mask_cache = {}
        self._output_tensor_cache = {}
        self._output_shape_cache = {}

        # Build self._output_layers:
        for x in self.outputs:
            layer, node_index, tensor_index = x._keras_history
            self._output_layers.append(layer)
            self._output_coordinates.append((layer, node_index, tensor_index))

        # Build self._input_layers:
        for x in self.inputs:
            layer, node_index, tensor_index = x._keras_history
            # It's supposed to be an input layer, so only one node
            # and one tensor output.
            assert node_index == 0
            assert tensor_index == 0
            self._input_layers.append(layer)
            self._input_coordinates.append((layer, node_index, tensor_index))

        # Keep track of the network's nodes and layers.
        nodes, nodes_by_depth, layers, layers_by_depth = _map_graph_network(
            self.inputs, self.outputs)
        self._network_nodes = nodes
        self._nodes_by_depth = nodes_by_depth
        self._layers = layers
        self._layers_by_depth = layers_by_depth

        # Create the node linking internal inputs to internal outputs.
        Node(outbound_layer=self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=self.inputs,
             output_tensors=self.outputs,
             # No network-level masking for now.
             input_masks=[None for _ in self.inputs],
             output_masks=[None for _ in self.outputs],
             input_shapes=[x._keras_shape for x in self.inputs],
             output_shapes=[x._keras_shape for x in self.outputs])

        # Fill in the output mask cache.
        masks = []
        for x in self.inputs:
            layer, node_index, tensor_index = x._keras_history
            node = layer._inbound_nodes[node_index]
            mask = node.output_masks[tensor_index]
            masks.append(mask)
        mask_cache_key = object_list_uid(inputs)
        mask_cache_key += '_' + object_list_uid(masks)
        masks = []
        for x in self.outputs:
            layer, node_index, tensor_index = x._keras_history
            node = layer._inbound_nodes[node_index]
            mask = node.output_masks[tensor_index]
            masks.append(mask)
        mask = unpack_singleton(masks)
        self._output_mask_cache[mask_cache_key] = mask

        # Build self.input_names and self.output_names.
        self.input_names = []
        self.output_names = []
        self._feed_input_names = []
        self._feed_inputs = []
        self._feed_input_shapes = []
        for i, layer in enumerate(self._input_layers):
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
                self._feed_inputs.append(layer.input)
                self._feed_input_names.append(layer.name)
                self._feed_input_shapes.append(self.inputs[i]._keras_shape)

        for layer in self._output_layers:
            self.output_names.append(layer.name)

    def _init_subclassed_network(self, name=None):
        self._base_init(name=name)
        self._is_graph_network = False
        self._expects_training_arg = has_arg(self.call, 'training')
        self._uses_inputs_arg = has_arg(self.call, 'inputs')
        self.outputs = None
        self.inputs = None
        self.built = False

    def __setattr__(self, name, value):
        # Automatically track layers set as Model
        # attributes for subclassed Models.
        if isinstance(value, (Layer, Network)):
            try:
                is_graph_network = self._is_graph_network
            except AttributeError:
                raise RuntimeError(
                    'It looks like you are subclassing `Model` and you '
                    'forgot to call `super(YourClass, self).__init__()`.'
                    ' Always start with this line.')
            if not is_graph_network:
                if value not in self._layers:
                    self._layers.append(value)
        super(Network, self).__setattr__(name, value)

    @property
    def layers(self):
        return self._layers

    def get_layer(self, name=None, index=None):
        """Retrieves a layer based on either its name (unique) or index.

        If `name` and `index` are both provided, `index` will take precedence.

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
        # without the network being notified of it.
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
        """Retrieves the model's updates.

        Will only include updates that are either
        unconditional, or conditional on inputs to this model
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
                if self._is_graph_network:
                    # Collect updates that are dependent on inputs
                    # that are part of the model.
                    for node_index, node in enumerate(layer._inbound_nodes):
                        node_key = self._node_key(layer, node_index)
                        if node_key in self._network_nodes:
                            # The model owns this layer node.
                            inputs = node.input_tensors
                            updates += layer.get_updates_for(inputs)
                    # Collect unconditional updates.
                    updates += layer.get_updates_for(None)
                else:
                    updates += layer.updates
        return updates

    @property
    def losses(self):
        """Retrieves the model's losses.

        Will only include losses that are either
        unconditional, or conditional on inputs to this model
        (e.g. will not include losses that depend on tensors
        that aren't inputs to this model).

        # Returns
            A list of loss tensors.
        """
        losses = []
        for layer in self.layers:
            if hasattr(layer, 'losses'):
                if self._is_graph_network:
                    # Collect losses that are dependent on inputs
                    # that are part of the model.
                    for node_index, node in enumerate(layer._inbound_nodes):
                        node_key = self._node_key(layer, node_index)
                        if node_key in self._network_nodes:
                            # The model owns this layer node.
                            inputs = node.input_tensors
                            losses += layer.get_losses_for(inputs)
                    # Collect unconditional losses.
                    losses += layer.get_losses_for(None)
                else:
                    losses += layer.losses

        # Add any potential unconditional model-level loss.
        losses += self.get_losses_for(None)

        unique_tensors = list(
            set(x for x in losses if not isinstance(x, (float, int))))
        non_tensors = [x for x in losses if isinstance(x, (float, int))]
        return unique_tensors + non_tensors

    @property
    def uses_learning_phase(self):
        if not self.outputs:
            return False
        return any([x._uses_learning_phase for x in self.outputs])

    @property
    def stateful(self):
        return any([(hasattr(layer, 'stateful') and
                    layer.stateful) for layer in self.layers])

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
        if not self._is_graph_network:
            # TODO: support it in subclassed networks after inputs are set.
            return None

        specs = []
        for layer in getattr(self, '_input_layers', []):
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
        return unpack_singleton(specs)

    def call(self, inputs, mask=None):
        """Calls the model on new inputs.

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
        inputs = to_list(inputs)
        if mask is None:
            masks = [None for _ in range(len(inputs))]
        else:
            masks = to_list(mask)
        cache_key = object_list_uid(inputs)
        cache_key += '_' + object_list_uid(masks)
        if cache_key in self._output_tensor_cache:
            return self._output_tensor_cache[cache_key]
        else:
            output_tensors, _, _ = self.run_internal_graph(inputs, masks)
            return output_tensors

    def compute_mask(self, inputs, mask):
        if not self._is_graph_network:
            return None

        inputs = to_list(inputs)
        if mask is None:
            masks = [None for _ in range(len(inputs))]
        else:
            masks = to_list(mask)
        cache_key = object_list_uid(inputs)
        cache_key += '_' + object_list_uid(masks)
        if cache_key in self._output_mask_cache:
            return self._output_mask_cache[cache_key]
        else:
            _, output_masks, _ = self.run_internal_graph(inputs, masks)
            return output_masks

    def compute_output_shape(self, input_shape):
        if not self._is_graph_network:
            # Must be implemented by subclasses.
            raise NotImplementedError

        input_shapes = to_list(input_shape)
        if len(input_shapes) != len(self._input_layers):
            raise ValueError('Invalid input_shape argument ' +
                             str(input_shape) + ': model has ' +
                             str(len(self._input_layers)) + ' tensor inputs.')

        cache_key = ', '.join([str(x) for x in input_shapes])
        if cache_key in self._output_shape_cache:
            output_shapes = self._output_shape_cache[cache_key]
            if isinstance(output_shapes, list):
                return unpack_singleton(output_shapes)
            return output_shapes
        else:
            # Bad luck, we have to run the graph manually.
            layers_to_output_shapes = {}
            for i in range(len(input_shapes)):
                layer = self._input_layers[i]
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
                        if layer in self._input_layers:
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

                        output_shape = layer.compute_output_shape(unpack_singleton(input_shapes))

                        output_shapes = to_list(output_shape)
                        node_index = layer._inbound_nodes.index(node)
                        for j in range(len(output_shapes)):
                            shape_key = layer.name + '_%s_%s' % (node_index, j)
                            layers_to_output_shapes[shape_key] = output_shapes[j]

            # Read final output shapes from layers_to_output_shapes.
            output_shapes = []
            output_shape_keys = []
            for i in range(len(self._output_layers)):
                layer = self._output_layers[i]
                node_index = self._output_coordinates[i][1]
                tensor_index = self._output_coordinates[i][2]
                shape_key = layer.name + '_%s_%s' % (node_index, tensor_index)
                output_shape_keys.append(shape_key)

            for i, key in enumerate(output_shape_keys):
                assert key in layers_to_output_shapes
                output_shapes.append(layers_to_output_shapes[key])
            # Store in cache.
            self._output_shape_cache[cache_key] = output_shapes
            if isinstance(output_shapes, list):
                return unpack_singleton(output_shapes)
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
                            output_tensors = to_list(layer.call(computed_tensor, **kwargs))
                            output_masks = layer.compute_mask(computed_tensor,
                                                              computed_mask)
                            if output_masks is None:
                                output_masks = [None for _ in output_tensors]
                            else:
                                output_masks = to_list(output_masks)
                            computed_tensors = [computed_tensor]
                            computed_masks = [computed_mask]
                        else:
                            computed_tensors = [x[0] for x in computed_data]
                            computed_masks = [x[1] for x in computed_data]
                            if has_arg(layer.call, 'mask'):
                                if 'mask' not in kwargs:
                                    kwargs['mask'] = computed_masks
                            output_tensors = to_list(
                                layer.call(computed_tensors, **kwargs))
                            output_masks = layer.compute_mask(computed_tensors,
                                                              computed_masks)
                            if output_masks is None:
                                output_masks = [None for _ in output_tensors]
                            else:
                                output_masks = to_list(output_masks)
                        # Apply activity regularizer if any:
                        if hasattr(layer, 'activity_regularizer') and layer.activity_regularizer is not None:
                            with K.name_scope('activity_regularizer'):
                                regularization_losses = [
                                    layer.activity_regularizer(x)
                                    for x in output_tensors]
                            layer.add_loss(regularization_losses,
                                           inputs=computed_tensors)

                        if len(output_masks) != len(output_tensors):
                            raise Exception(
                                'Layers should have equal number of output tensors '
                                'and output masks. Layer ' + str(layer.name) + ' has'
                                ' ' + str(len(output_tensors)) + ' output tensors and'
                                ' ' + str(len(output_masks)) + ' output masks.')
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
                        input_shapes = unpack_singleton([x._keras_shape for x in computed_tensors])
                        shapes = to_list(layer.compute_output_shape(input_shapes))
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
        cache_key = object_list_uid(inputs)
        cache_key += '_' + object_list_uid(masks)

        output_tensors = unpack_singleton(output_tensors)
        self._output_tensor_cache[cache_key] = output_tensors

        output_masks = unpack_singleton(output_masks)
        self._output_mask_cache[cache_key] = output_masks

        if output_shapes is not None:
            input_shapes = [x._keras_shape for x in inputs]
            cache_key = ', '.join([str(x) for x in input_shapes])

            output_shapes = unpack_singleton(output_shapes)
            self._output_shape_cache[cache_key] = output_shapes
        return output_tensors, output_masks, output_shapes

    def get_config(self):
        if not self._is_graph_network:
            # Subclassed networks are not serializable
            # (unless serialization is implemented by
            # the author of the subclassed network).
            raise NotImplementedError

        config = {
            'name': self.name,
        }

        # Build a map from a layer unique name (self._node_key)
        # to the index of the nodes that are saved in the config.
        # Only nodes in network_nodes are saved.
        node_conversion_map = {}
        for layer in self.layers:
            if issubclass(layer.__class__, Network):
                # Networks start with a pre-existing node
                # linking their input to output.
                kept_nodes = 1
            else:
                kept_nodes = 0
            for original_node_index, node in enumerate(layer._inbound_nodes):
                node_key = self._node_key(layer, original_node_index)
                if node_key in self._network_nodes:
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
                if node_key in self._network_nodes:
                    # The node is relevant to the model:
                    # add to filtered_inbound_nodes.
                    if node.arguments:
                        try:
                            json.dumps(node.arguments)
                            kwargs = node.arguments
                        except TypeError:
                            warnings.warn(
                                'Layer ' + layer.name +
                                ' was passed non-serializable '
                                'keyword arguments: ' +
                                str(node.arguments) +
                                '. They will not be included '
                                'in the serialized model '
                                '(and thus will be missing '
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
        for i in range(len(self._input_layers)):
            layer = self._input_layers[i]
            node_index = self._input_coordinates[i][1]

            node_key = self._node_key(layer, node_index)
            if node_key not in self._network_nodes:
                continue
            new_node_index = node_conversion_map[node_key]
            tensor_index = self._input_coordinates[i][2]
            model_inputs.append([layer.name, new_node_index, tensor_index])
        config['input_layers'] = model_inputs
        model_outputs = []
        for i in range(len(self._output_layers)):
            layer = self._output_layers[i]
            node_index = self._output_coordinates[i][1]

            node_key = self._node_key(layer, node_index)
            if node_key not in self._network_nodes:
                continue
            new_node_index = node_conversion_map[node_key]
            tensor_index = self._output_coordinates[i][2]
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
                input_tensors.append(
                    inbound_node.output_tensors[inbound_tensor_index])
            # Call layer on its inputs, thus creating the node
            # and building the layer if needed.
            if input_tensors:
                layer(unpack_singleton(input_tensors), **kwargs)

        def process_layer(layer_data):
            """Deserializes a layer, then call it on appropriate inputs.

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
        """Saves the model to a single HDF5 file.

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
        if not self._is_graph_network:
            raise NotImplementedError
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
            saving.save_weights_to_hdf5_group(f, self.layers)
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
                of weight arrays is present but their shape does not match.


        # Raises
            ImportError: If h5py is not available.
        """
        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        with h5py.File(filepath, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']
            if by_name:
                saving.load_weights_from_hdf5_group_by_name(
                    f, self.layers, skip_mismatch=skip_mismatch,
                    reshape=reshape)
            else:
                saving.load_weights_from_hdf5_group(
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
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
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
        if not self.built:
            raise ValueError(
                'This model has never been called, thus its weights '
                'have not yet been created, so no summary can be displayed. '
                'Build the model first '
                '(e.g. by calling it on some test data).')
        return print_layer_summary(self,
                                   line_length=line_length,
                                   positions=positions,
                                   print_fn=print_fn)


def _make_node_key(layer_name, node_index):
    return layer_name + '_ib-' + str(node_index)


def _map_graph_network(inputs, outputs):
    """Validates a network's topology and gather its layers and nodes.

    # Arguments
        inputs: List of input tensors.
        outputs: List of outputs tensors.

    # Returns
        A tuple `(nodes, nodes_by_depth, layers, layers_by_depth)`.
        - nodes: list of Node instances.
        - nodes_by_depth: dict mapping ints (depth) to lists of node instances.
        - layers: list of Layer instances.
        - layers_by_depth: dict mapping ints (depth)
            to lists of layer instances.

    # Raises
        ValueError: In case the network is not valid (e.g. disconnected graph).
    """
    # Network_nodes: set of nodes included in the graph of layers
    # (not all nodes included in the layers are relevant to the current graph).
    network_nodes = set()  # ids of all nodes relevant to the Network
    nodes_depths = {}  # dict {node: depth value}
    layers_depths = {}  # dict {layer: depth value}
    layer_indices = {}  # dict {layer: index in traversal}
    nodes_in_decreasing_depth = []

    def build_map(tensor,
                  finished_nodes,
                  nodes_in_progress,
                  layer,
                  node_index,
                  tensor_index):
        """Builds a map of the graph of layers.

        This recursively updates the map `layer_indices`,
        the list `nodes_in_decreasing_depth` and the set `network_nodes`.

        # Arguments:
            tensor: Some tensor in a graph.
            finished_nodes: Set of nodes whose subgraphs have been traversed
                completely. Useful to prevent duplicated work.
            nodes_in_progress: Set of nodes that are currently active on the
                recursion stack. Useful to detect cycles.
            layer: Layer from which `tensor` comes from. If not provided,
                will be obtained from `tensor._keras_history`.
            node_index: Node index from which `tensor` comes from.
            tensor_index: Tensor_index from which `tensor` comes from.

        # Raises:
            ValueError: if a cycle is detected.
        """
        node = layer._inbound_nodes[node_index]

        # Prevent cycles.
        if node in nodes_in_progress:
            raise ValueError('The tensor ' + str(tensor) + ' at layer "' +
                             layer.name + '" is part of a cycle.')

        # Don't repeat work for shared subgraphs
        if node in finished_nodes:
            return

        node_key = _make_node_key(layer.name, node_index)
        # Update network_nodes.
        network_nodes.add(node_key)

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
            build_map(x, finished_nodes, nodes_in_progress, layer,
                      node_index, tensor_index)

        finished_nodes.add(node)
        nodes_in_progress.remove(node)
        nodes_in_decreasing_depth.append(node)

    finished_nodes = set()
    nodes_in_progress = set()
    for x in outputs:
        layer, node_index, tensor_index = x._keras_history
        build_map(x, finished_nodes, nodes_in_progress,
                  layer=layer,
                  node_index=node_index,
                  tensor_index=tensor_index)

    for node in reversed(nodes_in_decreasing_depth):
        # If the depth is not set, the node has no outbound nodes (depth 0).
        depth = nodes_depths.setdefault(node, 0)

        # Update the depth of the corresponding layer
        previous_depth = layers_depths.get(node.outbound_layer, 0)
        # If we've seen this layer before at a higher depth,
        # we should use that depth instead of the node depth.
        # This is necessary for shared layers that have inputs at different
        # depth levels in the graph.
        depth = max(depth, previous_depth)
        layers_depths[node.outbound_layer] = depth
        nodes_depths[node] = depth

        # Update the depth of inbound nodes.
        # The "depth" of a node is the max of the depths
        # of all layers it is connected to.
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

    # Set self.layers and self._layers_by_depth.
    layers = []
    for depth in depth_keys:
        layers_for_depth = layers_by_depth[depth]
        # Network.layers needs to have a deterministic order:
        # here we order them by traversal order.
        layers_for_depth.sort(key=lambda x: layer_indices[x])
        layers.extend(layers_for_depth)

    # Get sorted list of node depths.
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    # Check that all tensors required are computable.
    # computable_tensors: all tensors in the graph
    # that can be computed from the inputs provided.
    computable_tensors = []
    for x in inputs:
        computable_tensors.append(x)

    layers_with_complete_input = []  # To provide a better error msg.
    for depth in depth_keys:
        for node in nodes_by_depth[depth]:
            layer = node.outbound_layer
            if layer:
                for x in node.input_tensors:
                    if x not in computable_tensors:
                        raise ValueError('Graph disconnected: '
                                         'cannot obtain value for tensor ' +
                                         str(x) + ' at layer "' +
                                         layer.name + '". '
                                         'The following previous layers '
                                         'were accessed without issue: ' +
                                         str(layers_with_complete_input))
                for x in node.output_tensors:
                    computable_tensors.append(x)
                layers_with_complete_input.append(layer.name)

    # Ensure name unicity, which will be crucial for serialization
    # (since serialized nodes refer to layers by their name).
    all_names = [layer.name for layer in layers]
    for name in all_names:
        if all_names.count(name) != 1:
            raise ValueError('The name "' + name + '" is used ' +
                             str(all_names.count(name)) +
                             ' times in the model. '
                             'All layer names should be unique.')
    return network_nodes, nodes_by_depth, layers, layers_by_depth
