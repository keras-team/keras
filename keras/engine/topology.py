# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

import sys
import marshal
import types as python_types
import warnings
import copy
from six.moves import zip

from keras import backend as K


def to_list(x):
    '''This normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.
    '''
    if type(x) is list:
        return x
    return [x]


class InputSpec(object):
    '''This specifies the ndim, dtype and shape of every input to a layer.
    Every layer should expose (if appropriate) an `input_spec` attribute:
    a list of instances of InputSpec (one per input tensor).

    A None entry in a shape is compatible with any dimension,
    a None shape is compatible with any shape.
    '''
    def __init__(self, dtype=None, shape=None, ndim=None):
        if type(ndim) is str:
            assert '+' in ndim, 'When passing a str "ndim", it should have the form "2+", "3+", etc.'
            int_ndim = ndim[:ndim.find('+')]
            assert int_ndim.isdigit(), 'When passing a str "ndim", it should have the form "2+", "3+", etc.'
        if shape is not None:
            self.ndim = len(shape)
        else:
            self.ndim = ndim
        self.dtype = dtype
        self.shape = shape


class Node(object):
    '''A `Node` describes the connectivity between two layers.

    Each time a layer is connected to some new input,
    a node is added to `layer.inbound_nodes`.
    Each time the output of a layer is used by another layer,
    a node is added to `layer.outbound_nodes`.

    # Attributes
        outbound_layer: the layer that takes
            `input_tensors` and turns them into `output_tensors`.
        inbound_layers: a list of layers, the same length as `input_tensors`,
            the layers from where `input_tensors` originate.
        node_indices: a list of integers, the same length as `inbound_layers`.
            `node_indices[i]` is the origin node of `input_tensors[i]`
            (necessary since each inbound layer might have several nodes,
            e.g. if the layer is being shared with a different data stream).
        tensor_indices: a list of integers, the same length as `inbound_layers`.
            `tensor_indices[i]` is the index of `input_tensors[i]` within the
            output of the inbound layer (necessary since each inbound layer might
            have multiple tensor outputs, with each one being
            independently manipulable).
        input_tensors: list of input tensors.
        output_tensors: list of output tensors.
        input_masks: list of input masks (a mask can be a tensor, or None).
        output_masks: list of output masks (a mask can be a tensor, or None).
        input_shapes: list of input shape tuples.
        output_shapes: list of output shape tuples.

    `node_indices` and `tensor_indices` are basically fine-grained coordinates
    describing the origin of the `input_tensors`, verifying the following:

    `input_tensors[i] == inbound_layers[i].inbound_nodes[node_indices[i]].output_tensors[tensor_indices[i]]`

    A node from layer A to layer B is added to:
        A.outbound_nodes
        B.inbound_nodes
    '''
    def __init__(self, outbound_layer,
                 inbound_layers, node_indices, tensor_indices,
                 input_tensors, output_tensors,
                 input_masks, output_masks,
                 input_shapes, output_shapes):
        # layer instance (NOT a list).
        # this is the layer that takes a list of input tensors
        # and turns them into a list of output tensors.
        # the current node will be added to the inbound_nodes of outbound_layer
        self.outbound_layer = outbound_layer

        # the following 3 properties describe where
        # the input tensors come from: which layers,
        # and for each layer, which node and which
        # tensor output of each node.
        self.inbound_layers = inbound_layers  # list of layer instances
        self.node_indices = node_indices  # list of integers, 1:1 mapping with inbound_layers
        self.tensor_indices = tensor_indices  # list of integers, 1:1 mapping with inbound_layers

        # tensor inputs and outputs of outbound_layer
        self.input_tensors = input_tensors  # list of tensors. 1:1 mapping with inbound_layers
        self.output_tensors = output_tensors  # list of tensors, created by outbound_layer.call()

        # input and output masks
        self.input_masks = input_masks  # list of tensors, 1:1 mapping with input_tensor
        self.output_masks = output_masks  # list of tensors, created by outbound_layer.compute_mask()

        # input and output shapes
        self.input_shapes = input_shapes  # list of shape tuples, shapes of input_tensors
        self.output_shapes = output_shapes  # list of shape tuples, shapes of output_tensors

        # add nodes to all layers involved.
        for layer in inbound_layers:
            if layer is not None:
                layer.outbound_nodes.append(self)
        outbound_layer.inbound_nodes.append(self)

    @classmethod
    def create_node(cls, outbound_layer,
                    inbound_layers, node_indices=None, tensor_indices=None):
        if not node_indices:
            node_indices = [0 for _ in range(len(inbound_layers))]
        else:
            assert len(node_indices) == len(inbound_layers)
        if not tensor_indices:
            tensor_indices = [0 for _ in range(len(inbound_layers))]

        input_tensors = []
        input_masks = []
        input_shapes = []

        for inbound_layer, node_index, tensor_index in zip(inbound_layers, node_indices, tensor_indices):
            inbound_node = inbound_layer.inbound_nodes[node_index]
            input_tensors.append(inbound_node.output_tensors[tensor_index])
            input_masks.append(inbound_node.output_masks[tensor_index])
            input_shapes.append(inbound_node.output_shapes[tensor_index])

        assert len(input_shapes) == len(input_tensors) == len(input_masks)

        if len(input_tensors) == 1:
            output_tensors = to_list(outbound_layer.call(input_tensors[0], mask=input_masks[0]))
            output_masks = to_list(outbound_layer.compute_mask(input_tensors[0], input_masks[0]))
            # TODO: try to auto-infer shape if exception is raised by get_output_shape_for
            output_shapes = to_list(outbound_layer.get_output_shape_for(input_shapes[0]))
        else:
            output_tensors = to_list(outbound_layer.call(input_tensors, mask=input_masks))
            output_masks = to_list(outbound_layer.compute_mask(input_tensors, input_masks))
            output_shapes = to_list(outbound_layer.get_output_shape_for(input_shapes))

        if not output_tensors or output_tensors[0] is None:
            raise Exception('The `call` method of layer "' +
                            outbound_layer.name +
                            '" should return a tensor. Found: ' +
                            str(output_tensors[0]))
        if len(output_tensors) != len(output_shapes):
            raise Exception('The `get_output_shape_for` method of layer "' +
                            outbound_layer.name +
                            '"" should return one shape tuple per '
                            'output tensor of the layer. Found: ' +
                            str(output_shapes))
        if len(output_tensors) != len(output_masks):
            raise Exception('The `compute_mask` method of layer "' +
                            outbound_layer.name +
                            '" should return one mask tensor per '
                            'output tensor of the layer. Found: ' +
                            str(output_masks))

        for i in range(len(output_tensors)):
            output_tensors[i]._keras_shape = output_shapes[i]
            output_tensors[i]._uses_learning_phase = any([x._uses_learning_phase for x in input_tensors]) or outbound_layer.uses_learning_phase
            output_tensors[i]._keras_history = (outbound_layer, len(outbound_layer.inbound_nodes), i)

        return cls(outbound_layer,
                   inbound_layers, node_indices, tensor_indices,
                   input_tensors, output_tensors,
                   input_masks, output_masks,
                   input_shapes, output_shapes)

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
    '''Abstract base layer class.

    # Properties
        name: string, must be unique within a model.
        input_spec: list of InputSpec class instances
            each entry describes one required input:
                - ndim
                - dtype
            A layer with `n` input tensors must have
            an `input_spec` of length `n`.
        trainable: boolean, whether the layer weights
            will be updated during training.
        uses_learning_phase: whether any operation
            of the layer uses `K.in_training_phase()`
            or `K.in_test_phase()`.
        input_shape: shape tuple. Provided for convenience,
            but note that there may be cases in which this
            attribute is ill-defined (e.g. a shared layer
            with multiple input shapes), in which case
            requesting `input_shape` will raise an Exception.
            Prefer using `layer.get_input_shape_for(input_shape)`,
            or `layer.get_input_shape_at(node_index)`.
        output_shape: shape tuple. See above.
        inbound_nodes: list of nodes.
        outbound_nodes: list of nodes.
        supports_masking: boolean
        input, output: input/output tensor(s). Note that if the layer is used
            more than once (shared layer), this is ill-defined
            and will raise an exception. In such cases, use
            `layer.get_input_at(node_index)`.
        input_mask, output_mask: same as above, for masks.

        trainable_weights: list of variables.
        non_trainable_weights: list of variables.
        regularizers: list of regularizers.
        constraints: dict mapping weights to constraints.

    # Methods
        call(x, mask=None): where the layer's logic lives.
        __call__(x, mask=None): wrapper around the layer logic (`call`).
            if x is a Keras tensor:
                - connect current layer with last layer from tensor:
                    `self.add_inbound_node(last_layer)`
                - add layer to tensor history
            if layer is not built:
                - build from x._keras_shape
        get_weights()
        set_weights(weights)
        get_config()
        count_params()
        get_output_shape_for(input_shape)
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
        add_inbound_node(layer, index=0)
        create_input_layer()
        assert_input_compatibility()
    '''
    def __init__(self, **kwargs):
        # these properties should have been set
        # by the child class, as appropriate.
        if not hasattr(self, 'input_spec'):
            self.input_spec = None
        if not hasattr(self, 'supports_masking'):
            self.supports_masking = False
        if not hasattr(self, 'uses_learning_phase'):
            self.uses_learning_phase = False

        # these lists will be filled via successive calls
        # to self.add_inbound_node()
        self.inbound_nodes = []
        self.outbound_nodes = []

        # these properties will be set upon call of self.build(),
        # which itself will be called upon self.add_inbound_node if necessary.
        self.trainable_weights = []
        self.non_trainable_weights = []
        self.regularizers = []
        self.constraints = {}  # dict {tensor: constraint instance}
        self.built = False

        # these properties should be set by the user via keyword arguments.
        # note that 'input_dtype', 'input_shape' and 'batch_input_shape'
        # are only applicable to input layers: do not pass these keywords
        # to non-input layers.
        allowed_kwargs = {'input_shape',
                          'batch_input_shape',
                          'input_dtype',
                          'name',
                          'trainable',
                          'create_input_layer'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Keyword argument not understood: ' + kwarg

        name = kwargs.get('name')
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        self.trainable = kwargs.get('trainable', True)
        if 'batch_input_shape' in kwargs or 'input_shape' in kwargs:
            # in this case we will create an input layer
            # to insert before the current layer
            if 'batch_input_shape' in kwargs:
                batch_input_shape = tuple(kwargs['batch_input_shape'])
            elif 'input_shape' in kwargs:
                batch_input_shape = (None,) + tuple(kwargs['input_shape'])
            self.batch_input_shape = batch_input_shape
            input_dtype = kwargs.get('input_dtype', K.floatx())
            self.input_dtype = input_dtype
            if 'create_input_layer' in kwargs:
                self.create_input_layer(batch_input_shape, input_dtype)

    def create_input_layer(self, batch_input_shape,
                           input_dtype=None, name=None):
        if not name:
            prefix = self.__class__.__name__.lower() + '_input_'
            name = prefix + str(K.get_uid(prefix))
        if not input_dtype:
            input_dtype = K.floatx()

        self.batch_input_shape = batch_input_shape
        self.input_dtype = input_dtype

        # instantiate the input layer
        x = Input(batch_shape=batch_input_shape,
                  dtype=input_dtype, name=name)
        # this will build the current layer
        # and create the node connecting the current layer
        # to the input layer we just created.
        self(x)

    def assert_input_compatibility(self, input):
        '''This checks that the tensor(s) `input`
        verify the input assumptions of the layer
        (if any). If not, exceptions are raised.
        '''
        if not self.input_spec:
            return True
        assert type(self.input_spec) is list, ('input_spec must be a list of ' +
                                               'InputSpec instances. Found: ' +
                                               str(self.input_spec))
        inputs = to_list(input)
        if len(self.input_spec) > 1:
            if len(inputs) != len(self.input_spec):
                raise Exception('Layer ' + self.name + ' expects ' +
                                str(len(self.input_spec)) + ' inputs, '
                                'but it received ' + str(len(inputs)) +
                                ' input tensors. Input received: ' +
                                str(input))
        for input_index, (x, spec) in enumerate(zip(inputs, self.input_spec)):
            if spec is None:
                continue

            # check ndim
            if spec.ndim is not None:
                if type(spec.ndim) is str:
                    int_ndim = spec.ndim[:spec.ndim.find('+')]
                    ndim = int(int_ndim)
                    if K.ndim(x) < ndim:
                        raise Exception('Input ' + str(input_index) +
                                        ' is incompatible with layer ' +
                                        self.name + ': expected ndim >= ' +
                                        str(ndim) + ', found ndim=' +
                                        str(K.ndim(x)))
                else:
                    if K.ndim(x) != spec.ndim:
                        raise Exception('Input ' + str(input_index) +
                                        ' is incompatible with layer ' +
                                        self.name + ': expected ndim=' +
                                        str(spec.ndim) + ', found ndim=' +
                                        str(K.ndim(x)))
            if spec.dtype is not None:
                if K.dtype(x) != spec.dtype:
                    raise Exception('Input ' + str(input_index) +
                                    ' is incompatible with layer ' +
                                    self.name + ': expected dtype=' +
                                    str(spec.dtype) + ', found dtype=' +
                                    str(K.dtype(x)))
            if spec.shape is not None:
                if hasattr(x, '_keras_shape'):
                    x_shape = x._keras_shape
                elif hasattr(K, 'int_shape'):
                    # tensorflow shape inference
                    x_shape = K.int_shape(x)
                else:
                    continue
                for spec_dim, dim in zip(spec.shape, x_shape):
                    if spec_dim is not None:
                        if spec_dim != dim:
                            raise Exception('Input ' + str(input_index) +
                                            ' is incompatible with layer ' +
                                            self.name + ': expected shape=' +
                                            str(spec.shape) + ', found shape=' +
                                            str(x_shape))

    def call(self, x, mask=None):
        '''This is where the layer's logic lives.

        # Arguments
            x: input tensor, or list/tuple of input tensors.
            mask: a masking tensor (or list of tensors). Used mainly in RNNs.

        # Returns:
            A tensor or list/tuple of tensors.
        '''
        return x

    def __call__(self, x, mask=None):
        '''Wrapper around self.call(), for handling
        internal Keras references.

        If a Keras tensor is passed:
            - we call self.add_inbound_node()
            - if necessary, we `build` the layer to match
                the _keras_shape of the input(s)
            - we update the _keras_shape of every input tensor with
                its new shape (obtained via self.get_output_shape_for).
                This is done as part of add_inbound_node().
            - we update the _keras_history of the output tensor(s)
                with the current layer.
                This is done as part of add_inbound_node().

        # Arguments
            x: can be a tensor or list/tuple of tensors.
            mask: tensor or list/tuple of tensors.
        '''
        if not self.built:
            # raise exceptions in case the input is not compatible
            # with the input_spec specified in the layer constructor
            self.assert_input_compatibility(x)

            # collect input shapes to build layer
            input_shapes = []
            for x_elem in to_list(x):
                if hasattr(x_elem, '_keras_shape'):
                    input_shapes.append(x_elem._keras_shape)
                elif hasattr(K, 'int_shape'):
                    input_shapes.append(K.int_shape(x_elem))
                else:
                    raise Exception('You tried to call layer "' + self.name +
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

        # raise exceptions in case the input is not compatible
        # with the input_spec set at build time
        self.assert_input_compatibility(x)
        # build and connect layer
        input_added = False
        input_tensors = to_list(x)

        inbound_layers = []
        node_indices = []
        tensor_indices = []
        for input_tensor in input_tensors:
            if hasattr(input_tensor, '_keras_history') and input_tensor._keras_history:
                # this is a Keras tensor
                previous_layer, node_index, tensor_index = input_tensor._keras_history
                inbound_layers.append(previous_layer)
                node_indices.append(node_index)
                tensor_indices.append(tensor_index)
            else:
                inbound_layers = None
                break
        if inbound_layers:
            # this will call layer.build() if necessary
            self.add_inbound_node(inbound_layers, node_indices, tensor_indices)
            input_added = True

        # get the output tensor to be returned
        if input_added:
            # output was already computed when calling self.add_inbound_node
            outputs = self.inbound_nodes[-1].output_tensors
            # if single output tensor: return it,
            # else return a list (at least 2 elements)
            if len(outputs) == 1:
                return outputs[0]
            else:
                return outputs
        else:
            # this case appears if the input was not a Keras tensor
            return self.call(x, mask)

    def add_inbound_node(self, inbound_layers,
                         node_indices=None, tensor_indices=None):
        '''
        # Arguments:
            inbound_layers: can be a layer instance
                or a list/tuple of layer instances.
            node_indices: integer (or list of integers).
                The input layer might have a number of
                parallel output streams;
                this is the index of the stream (in the input layer)
                where to connect the current layer.
            tensor_indices: integer or list of integers.
                The output of the inbound node might be a list/tuple
                of tensor, and we might only be interested in one specific entry.
                This index allows you to specify the index of the entry in the output list
                (if applicable). "None" means that we take all outputs (as a list).
        '''
        inbound_layers = to_list(inbound_layers)
        if not node_indices:
            node_indices = [0 for _ in range(len(inbound_layers))]
        else:
            node_indices = to_list(node_indices)
            assert len(node_indices) == len(inbound_layers)
        if not tensor_indices:
            tensor_indices = [0 for _ in range(len(inbound_layers))]
        else:
            tensor_indices = to_list(tensor_indices)

        if not self.built:
            # collect input_shapes for call to build()
            input_shapes = []
            for layer, node_index, tensor_index in zip(inbound_layers, node_indices, tensor_indices):
                input_shapes.append(layer.inbound_nodes[node_index].output_shapes[tensor_index])
            # call build()
            if len(input_shapes) == 1:
                self.build(input_shape=input_shapes[0])
            else:
                self.build(input_shape=input_shapes)
            self.built = True
        # creating the node automatically updates self.inbound_nodes
        # as well as outbound_nodes on inbound layers.
        Node.create_node(self, inbound_layers, node_indices, tensor_indices)

    def get_output_shape_for(self, input_shape):
        '''Computes the output shape of the layer given
        an input shape (assumes that the layer will be built
        to match that input shape).

        # Arguments
            input_shape: shape tuple (tuple of integers)
                or list of shape tuples (one per output tensor of the layer).
                Shape tuples can include None for free dimensions,
                instead of an integer.
        '''
        return input_shape

    def compute_mask(self, input, input_mask=None):
        '''Computes an output masking tensor, given an input tensor
        (or list thereof) and an input mask (or list thereof).

        # Arguments
            input: tensor or list of tensors.
            input_mask: tensor or list of tensors.

        # Returns
            None or a tensor (or list of tensors,
                one per output tensor of the layer).
        '''
        if not hasattr(self, 'supports_masking') or not self.supports_masking:
            if input_mask is not None:
                if type(input_mask) is list:
                    if any(input_mask):
                        raise Exception('Layer ' + self.name + ' does not support masking, ' +
                                        'but was passed an input_mask: ' + str(input_mask))
                else:
                    raise Exception('Layer ' + self.name + ' does not support masking, ' +
                                    'but was passed an input_mask: ' + str(input_mask))
            # masking not explicitly supported: return None as mask
            return None
        # if masking is explictly supported, by default
        # carry over the input mask
        return input_mask

    def build(self, input_shape):
        '''Creates the layer weights.
        Must be implemented on all layers that have weights.

        # Arguments
            input_shape: Keras tensor (future input to layer)
                or list/tuple of Keras tensors to reference
                for weight shape computations.
        '''
        self.built = True

    def _get_node_attribute_at_index(self, node_index, attr, attr_name):
        '''Retrieves an attribute (e.g. input_tensors) from a node.

        # Arguments
            node_index: integer index of the node from which
                to retrieve the attribute
            attr: exact node attribute name
            attr_name: human-readable attribute name, for error messages
        '''
        if not self.inbound_nodes:
            raise Exception('The layer has never been called ' +
                            'and thus has no defined ' + attr_name + '.')
        if not len(self.inbound_nodes) > node_index:
            raise Exception('Asked to get ' + attr_name +
                            ' at node ' + str(node_index) +
                            ', but the layer has only ' +
                            str(len(self.inbound_nodes)) + ' inbound nodes.')
        values = getattr(self.inbound_nodes[node_index], attr)
        if len(values) == 1:
            return values[0]
        else:
            return values

    def get_input_shape_at(self, node_index):
        '''Retrieves the input shape(s) of a layer at a given node.
        '''
        return self._get_node_attribute_at_index(node_index,
                                                 'input_shapes',
                                                 'input shape')

    def get_output_shape_at(self, node_index):
        '''Retrieves the output shape(s) of a layer at a given node.
        '''
        return self._get_node_attribute_at_index(node_index,
                                                 'output_shapes',
                                                 'output shape')

    def get_input_at(self, node_index):
        '''Retrieves the input tensor(s) of a layer at a given node.
        '''
        return self._get_node_attribute_at_index(node_index,
                                                 'input_tensors',
                                                 'input')

    def get_output_at(self, node_index):
        '''Retrieves the output tensor(s) of a layer at a given node.
        '''
        return self._get_node_attribute_at_index(node_index,
                                                 'output_tensors',
                                                 'output')

    def get_input_mask_at(self, node_index):
        '''Retrieves the input mask tensor(s) of a layer at a given node.
        '''
        return self._get_node_attribute_at_index(node_index,
                                                 'input_masks',
                                                 'input mask')

    def get_output_mask_at(self, node_index):
        '''Retrieves the output mask tensor(s) of a layer at a given node.
        '''
        return self._get_node_attribute_at_index(node_index,
                                                 'output_masks',
                                                 'output mask')

    @property
    def input(self):
        '''Retrieves the input tensor(s) of a layer (only applicable if
        the layer has exactly one inbound node, i.e. if it is connected
        to one incoming layer).
        '''
        if len(self.inbound_nodes) > 1:
            raise Exception('Layer ' + self.name +
                            ' has multiple inbound nodes, ' +
                            'hence the notion of "layer input" '
                            'is ill-defined. '
                            'Use `get_input_at(node_index)` instead.')
        elif not self.inbound_nodes:
            raise Exception('Layer ' + self.name +
                            ' is not connected, no input to return.')
        return self._get_node_attribute_at_index(0, 'input_tensors',
                                                 'input')

    def set_input(self, input_tensor, shape=None):
        if len(self.inbound_nodes) > 1:
            raise Exception('Cannot `set_input` for layer ' + self.name +
                            ' because it has more than one inbound connection.')
        if len(self.inbound_nodes) == 1:
            # check that the inbound node is an Input node
            if self.inbound_nodes[0].inbound_layers:
                warnings.warn('You are manually setting the input for layer ' +
                              self.name + ' but it is not an Input layer. '
                              'This will cause part of your model '
                              'to be disconnected.')
        if self.outbound_nodes:
            warnings.warn('You are manually setting the input for layer ' +
                          self.name + ' but it has ' +
                          str(len(self.outbound_nodes)) +
                          ' outbound layers. '
                          'This will cause part of your model '
                          'to be disconnected.')
        if not shape:
            if hasattr(K, 'int_shape'):
                shape = K.int_shape(input_tensor)
            else:
                raise Exception('`set_input` needs to know the shape '
                                'of the `input_tensor` it receives, but '
                                'Keras was not able to infer it automatically.'
                                ' Specify it via: '
                                '`model.set_input(input_tensor, shape)`')
        # reset layer connections
        self.inbound_nodes = []
        self.outbound_nodes = []
        input_shape = tuple(shape)
        self.build(input_shape=input_shape)

        # set Keras tensor metadata
        input_tensor._uses_learning_phase = False
        input_tensor._keras_history = (None, 0, 0)
        input_tensor._keras_shape = input_shape

        output_tensors = to_list(self.call(input_tensor))
        output_shapes = to_list(self.get_output_shape_for(input_shape))
        output_masks = to_list(self.compute_mask(input_tensor, None))

        for i, output_tensor in enumerate(output_tensors):
            output_tensor._keras_history = (self, 0, i)
            output_tensor._keras_shape = output_shapes[i]
            output_tensor._uses_learning_phase = self.uses_learning_phase

        # create node
        Node(self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=[input_tensor],
             output_tensors=output_tensors,
             input_masks=[None],
             output_masks=output_masks,
             input_shapes=[input_shape],
             output_shapes=output_shapes)

    @property
    def output(self):
        '''Retrieves the output tensor(s) of a layer (only applicable if
        the layer has exactly one inbound node, i.e. if it is connected
        to one incoming layer).
        '''
        if len(self.inbound_nodes) != 1:
            raise Exception('Layer ' + self.name +
                            ' has multiple inbound nodes, ' +
                            'hence the notion of "layer output" '
                            'is ill-defined. '
                            'Use `get_output_at(node_index)` instead.')
        return self._get_node_attribute_at_index(0, 'output_tensors',
                                                 'output')

    @property
    def input_mask(self):
        '''Retrieves the input mask tensor(s) of a layer (only applicable if
        the layer has exactly one inbound node, i.e. if it is connected
        to one incoming layer).
        '''
        if len(self.inbound_nodes) != 1:
            raise Exception('Layer ' + self.name +
                            ' has multiple inbound nodes, ' +
                            'hence the notion of "layer input mask" '
                            'is ill-defined. '
                            'Use `get_input_mask_at(node_index)` instead.')
        return self._get_node_attribute_at_index(0, 'input_masks',
                                                 'input mask')

    @property
    def output_mask(self):
        '''Retrieves the output mask tensor(s) of a layer (only applicable if
        the layer has exactly one inbound node, i.e. if it is connected
        to one incoming layer).
        '''
        if len(self.inbound_nodes) != 1:
            raise Exception('Layer ' + self.name +
                            ' has multiple inbound nodes, ' +
                            'hence the notion of "layer output mask" '
                            'is ill-defined. '
                            'Use `get_output_mask_at(node_index)` instead.')
        return self._get_node_attribute_at_index(0, 'output_masks',
                                                 'output mask')

    @property
    def input_shape(self):
        '''Retrieves the input shape tuple(s) of a layer. Only applicable
        if the layer has one inbound node,
        or if all inbound nodes have the same input shape.
        '''
        if not self.inbound_nodes:
            raise Exception('The layer has never been called ' +
                            'and thus has no defined input shape.')
        all_input_shapes = set([str(node.input_shapes) for node in self.inbound_nodes])
        if len(all_input_shapes) == 1:
            input_shapes = self.inbound_nodes[0].input_shapes
            if len(input_shapes) == 1:
                return input_shapes[0]
            else:
                return input_shapes
        else:
            raise Exception('The layer "' + str(self.name) +
                            ' has multiple inbound nodes, ' +
                            'with different input shapes. Hence ' +
                            'the notion of "input shape" is ' +
                            'ill-defined for the layer. ' +
                            'Use `get_input_shape_at(node_index)` instead.')

    @property
    def output_shape(self):
        '''Retrieves the output shape tuple(s) of a layer. Only applicable
        if the layer has one inbound node,
        or if all inbound nodes have the same output shape.
        '''
        if not self.inbound_nodes:
            raise Exception('The layer has never been called ' +
                            'and thus has no defined output shape.')
        all_output_shapes = set([str(node.output_shapes) for node in self.inbound_nodes])
        if len(all_output_shapes) == 1:
            output_shapes = self.inbound_nodes[0].output_shapes
            if len(output_shapes) == 1:
                return output_shapes[0]
            else:
                return output_shapes
        else:
            raise Exception('The layer "' + str(self.name) +
                            ' has multiple inbound nodes, ' +
                            'with different output shapes. Hence ' +
                            'the notion of "output shape" is ' +
                            'ill-defined for the layer. ' +
                            'Use `get_output_shape_at(node_index)` instead.')

    def set_weights(self, weights):
        '''Sets the weights of the layer, from Numpy arrays.

        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the layer (i.e. it should match the
                output of `get_weights`).
        '''
        params = self.trainable_weights + self.non_trainable_weights
        if len(params) != len(weights):
            raise Exception('You called `set_weights(weights)` on layer "' + self.name +
                            '" with a  weight list of length ' + str(len(weights)) +
                            ', but the layer was expecting ' + str(len(params)) +
                            ' weights. Provided weights: ' + str(weights))
        if not params:
            return
        weight_value_tuples = []
        param_values = K.batch_get_value(params)
        for pv, p, w in zip(param_values, params, weights):
            if pv.shape != w.shape:
                raise Exception('Layer weight shape ' +
                                str(pv.shape) +
                                ' not compatible with '
                                'provided weight shape ' + str(w.shape))
            weight_value_tuples.append((p, w))
        K.batch_set_value(weight_value_tuples)

    def get_weights(self):
        '''Returns the current weights of the layer,
        as a list of numpy arrays.
        '''
        params = self.trainable_weights + self.non_trainable_weights
        return K.batch_get_value(params)

    def get_config(self):
        '''Returns a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.

        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by Container (one layer of abstraction above).
        '''
        config = {'name': self.name,
                  'trainable': self.trainable}
        if hasattr(self, 'batch_input_shape'):
            config['batch_input_shape'] = self.batch_input_shape
        if hasattr(self, 'input_dtype'):
            config['input_dtype'] = self.input_dtype
        return config

    @classmethod
    def from_config(cls, config):
        '''This method is the reverse of get_config,
        capable of instantiating the same layer from the config
        dictionary. It does not handle layer connectivity
        (handled by Container), nor weights (handled by `set_weights`).

        # Arguments
            config: a Python dictionary, typically the
                output of get_config.
        '''
        return cls(**config)

    def count_params(self):
        '''Returns the total number of floats (or ints)
        composing the weights of the layer.
        '''
        if not self.built:
            if self.__class__.__name__ in {'Sequential', 'Graph'}:
                self.build()
            else:
                raise Exception('You tried to call `count_params` on ' +
                                self.name + ', but the layer isn\'t built. '
                                'You can build it manually via: `' +
                                self.name + '.build(batch_input_shape)`.')
        return sum([K.count_params(p) for p in self.trainable_weights])


class InputLayer(Layer):
    '''TODO: dosctring
    '''
    def __init__(self, input_shape=None, batch_input_shape=None,
                 input_dtype=None, name=None):
        self.input_spec = None
        self.supports_masking = False
        self.uses_learning_phase = False
        self.trainable = False
        self.built = True

        self.inbound_nodes = []
        self.outbound_nodes = []

        self.trainable_weights = []
        self.non_trainable_weights = []
        self.regularizers = []
        self.constraints = {}

        if not name:
            prefix = 'input'
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        if not batch_input_shape:
            assert input_shape, 'An Input layer should be passed either a `batch_input_shape` or an `input_shape`.'
            batch_input_shape = (None,) + tuple(input_shape)
        else:
            batch_input_shape = tuple(batch_input_shape)
        if not input_dtype:
            input_dtype = K.floatx()

        self.batch_input_shape = batch_input_shape
        self.input_dtype = input_dtype

        input_tensor = K.placeholder(shape=batch_input_shape,
                                     dtype=input_dtype,
                                     name=self.name)
        # create an input node to add to self.outbound_node
        # and set output_tensors' _keras_history
        input_tensor._uses_learning_phase = False
        input_tensor._keras_history = (self, 0, 0)
        shape = input_tensor._keras_shape
        Node(self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=[input_tensor],
             output_tensors=[input_tensor],
             input_masks=[None],
             output_masks=[None],
             input_shapes=[shape],
             output_shapes=[shape])

    def get_config(self):
        config = {'batch_input_shape': self.batch_input_shape,
                  'input_dtype': self.input_dtype,
                  'name': self.name}
        return config


def Input(shape=None, batch_shape=None,
          name=None, dtype=K.floatx()):
    '''`Input()` is used to instantiate a Keras tensor.
    A Keras tensor is a tensor object from the underlying backend
    (Theano or TensorFlow), which we augment with certain
    attributes that allow us to build a Keras model
    just by knowing the inputs and outputs of the model.

    For instance, if a, b and c and Keras tensors,
    it becomes possible to do:
    `model = Model(input=[a, b], output=c)`

    The added Keras attributes are:
        ._keras_shape: integer shape tuple propagated
            via Keras-side shape inference.
        ._keras_history: last layer applied to the tensor.
            the entire layer graph is retrievable from that layer,
            recursively.

    # Arguments
        shape: a shape tuple (integer), not including the batch size.
            For instance, `shape=(32,)` indicates that the expected input
            will be batches of 32-dimensional vectors.
        batch_shape: a shape tuple (integer), including the batch size.
            For instance, `batch_shape=(10, 32)` indicates that
            the expected input will be batches of 10 32-dimensional vectors.
            `batch_shape=(None, 32)` indicates batches of an arbitrary number
            of 32-dimensional vectors.
        name: An optional name string for the layer.
            Should be unique in a model (do not reuse the same name twice).
            It will be autogenerated if it isn't provided.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Example usage

        ```python
        # this is a logistic regression in Keras
        a = Input(shape=(32,))
        b = Dense(16, activation='softmax')(a)
        model = Model(input=a, output=b)
        ```
    '''
    if not batch_shape:
        assert shape, ('Please provide to Input either a `shape`' +
                       ' or a `batch_shape` argument. Note that ' +
                       '`shape` does not include the batch '
                       'dimension.')
        batch_shape = (None,) + tuple(shape)
    input_layer = InputLayer(batch_input_shape=batch_shape,
                             name=name, input_dtype=dtype)
    # return tensor including _keras_shape and _keras_history
    # note that in this case train_output and test_output are the same pointer.
    outputs = input_layer.inbound_nodes[0].output_tensors
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


class Merge(Layer):
    '''A `Merge` layer can be used to merge a list of tensors
    into a single tensor, following some merge `mode`.

    # Example usage

    ```python
    model1 = Sequential()
    model1.add(Dense(32))

    model2 = Sequential()
    model2.add(Dense(32))

    merged_model = Sequential()
    merged_model.add(Merge([model1, model2], mode='concat', concat_axis=1)
    # TODO: would this actually work? it needs to.
    # achieve this with get_source_inputs in Sequential.
    ```

    # Arguments
        layers: can be a list of Keras tensors or
            a list of layer instances. Must be more
            than one layer/tensor.
        mode: string or lambda/function. If string, must be one
            of: 'sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'.
            If lambda/function, it should take as input a list of tensors
            and return a single tensor.
        concat_axis: integer, axis to use in mode `concat`.
        dot_axes: integer or tuple of integers, axes to use in mode `dot`.
        output_shape: either a shape tuple (tuple of integers), or a lambda/function
            to compute `output_shape` (only if merge mode is a lambda/function).
            If the argument is a tuple,
            it should be expected output shape, *not* including the batch size
            (same convention as the `input_shape` argument in layers).
            If the argument is callable, it should take as input a list of shape tuples
            (1:1 mapping to input tensors) and return a single shape tuple, including the
            batch size (same convention as the `get_output_shape_for` method of layers).
        node_indices: optional list of integers containing
            the output node index for each input layer
            (in case some input layers have multiple output nodes).
            will default to an array of 0s if not provided.
        tensor_indices: optional list of indices of output tensors
            to consider for merging
            (in case some input layer node returns multiple tensors).
        output_mask: mask or lambda/function to compute the output mask (only
            if merge mode is a lambda/function). If the latter case, it should
            take as input a list of masks and return a single mask.
    '''
    def __init__(self, layers=None, mode='sum', concat_axis=-1,
                 dot_axes=-1, output_shape=None, output_mask=None,
                 node_indices=None, tensor_indices=None, name=None):
        self.layers = layers
        self.mode = mode
        self.concat_axis = concat_axis
        self.dot_axes = dot_axes
        if type(self.dot_axes) == int:
            self.dot_axes = [self.dot_axes, ] * 2
        self._output_shape = output_shape
        self.node_indices = node_indices
        self._output_mask = output_mask

        # layer parameters
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self.regularizers = []
        self.trainable_weights = []
        self.non_trainable_weights = []
        self.supports_masking = True
        self.uses_learning_phase = False
        self.input_spec = None  # compatible with whatever
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        if layers:
            # this exists for backwards compatibility.
            # equivalent to:
            # merge = Merge(layers=None)
            # output = merge([input_tensor_1, input_tensor_2])
            if not node_indices:
                # by default we connect to
                # the 1st output stream in the input layer
                node_indices = [0 for _ in range(len(layers))]
            self._arguments_validation(layers, mode,
                                       concat_axis, dot_axes,
                                       node_indices, tensor_indices)
            self.built = True
            self.add_inbound_node(layers, node_indices, tensor_indices)
        else:
            self.built = False

    def _arguments_validation(self, layers, mode, concat_axis, dot_axes,
                              node_indices, tensor_indices):
        '''Validates user-passed arguments and raises exceptions
        as appropriate.
        '''
        if not hasattr(mode, '__call__'):
            if mode not in {'sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'}:
                raise Exception('Invalid merge mode: ' + str(mode))
        if type(layers) not in {list, tuple} or len(layers) < 2:
            raise Exception('A Merge should only be applied to a list of '
                            'layers with at least 2 elements. Found: ' + str(layers))

        if tensor_indices is None:
            tensor_indices = [None for _ in range(len(layers))]

        input_shapes = []
        for i, layer in enumerate(layers):
            layer_output_shape = layer.get_output_shape_at(node_indices[i])
            if type(layer_output_shape) is list:
                # case: the layer has multiple output tensors
                # and we only need a specific one
                layer_output_shape = layer_output_shape[tensor_indices[i]]
            input_shapes.append(layer_output_shape)

        if mode in {'sum', 'mul', 'ave', 'cos', 'max'}:
            input_shapes_set = set(input_shapes)
            if len(input_shapes_set) > 1:
                raise Exception('Only layers of same output shape can '
                                'be merged using ' + mode + ' mode. ' +
                                'Layer shapes: %s' % input_shapes)
        if mode in {'cos', 'dot'}:
            if len(layers) > 2:
                raise Exception(mode + ' merge takes exactly 2 layers')
            shape1 = input_shapes[0]
            shape2 = input_shapes[1]
            n1 = len(shape1)
            n2 = len(shape2)
            if type(dot_axes) == int:
                if dot_axes < 0:
                    dot_axes = [dot_axes % n1, dot_axes % n2]
                else:
                    dot_axes = [n1 - dot_axes, n2-dot_axes]
            if type(dot_axes) not in [list, tuple]:
                raise Exception('Invalid type for dot_axes - should be a list.')
            if len(dot_axes) != 2:
                raise Exception('Invalid format for dot_axes - should contain two elements.')
            if type(dot_axes[0]) is not int or type(dot_axes[1]) is not int:
                raise Exception('Invalid format for dot_axes - list elements should be "int".')
            if shape1[dot_axes[0]] != shape2[dot_axes[1]]:
                raise Exception('Dimension incompatibility using dot mode: ' +
                                '%s != %s. ' % (shape1[dot_axes[0]], shape2[dot_axes[1]]) +
                                'Layer shapes: %s, %s' % (shape1, shape2))
        elif mode == 'concat':
            reduced_inputs_shapes = [list(shape) for shape in input_shapes]
            shape_set = set()
            for i in range(len(reduced_inputs_shapes)):
                del reduced_inputs_shapes[i][self.concat_axis]
                shape_set.add(tuple(reduced_inputs_shapes[i]))
            if len(shape_set) > 1:
                raise Exception('"concat" mode can only merge layers with matching ' +
                                'output shapes except for the concat axis. ' +
                                'Layer shapes: %s' % (input_shapes))

    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('Merge must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))
        # case: "mode" is a lambda or function.
        if hasattr(self.mode, '__call__'):
            # TODO: consider making it possible to
            # pass custom arguments to lambda.
            arguments = {}
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
            raise Exception('Unknown merge mode.')

    def __call__(self, inputs, mask=None):
        '''We disable successive calls to __call__ for Merge layers.
        Although there is no technical obstacle to
        making it possible to __call__ a Merge instance many times
        (it is just a layer), it would make for a rather inelegant API.
        '''
        if type(inputs) is not list:
            raise Exception('Merge can only be called on a list of tensors, '
                            'not a single tensor. Received: ' + str(inputs))
        if self.built:
            raise Exception('A Merge layer cannot be used more than once, '
                            'please use ' +
                            'the "merge" function instead: ' +
                            '`merged_tensor = merge([tensor_1, tensor2])`.')

        all_keras_tensors = True
        for x in inputs:
            if not hasattr(x, '_keras_history'):
                all_keras_tensors = False
                break

        if all_keras_tensors:
            layers = []
            node_indices = []
            tensor_indices = []
            for x in inputs:
                layer, node_index, tensor_index = x._keras_history
                layers.append(layer)
                node_indices.append(node_index)
                tensor_indices.append(tensor_index)
            self._arguments_validation(layers, self.mode,
                                       self.concat_axis, self.dot_axes,
                                       node_indices, tensor_indices)
            self.built = True
            self.add_inbound_node(layers, node_indices, tensor_indices)

            outputs = self.inbound_nodes[-1].output_tensors
            return outputs[0]  # merge only returns a single tensor
        else:
            return self.call(inputs, mask)

    def get_output_shape_for(self, input_shape):
        assert type(input_shape) is list  # must have multiple input shape tuples
        # case: callable self._output_shape
        if hasattr(self.mode, '__call__'):
            if hasattr(self._output_shape, '__call__'):
                output_shape = self._output_shape(input_shape)
                return output_shape
            elif self._output_shape is not None:
                return (input_shape[0][0],) + tuple(self._output_shape)
            else:
                # TODO: consider shape auto-inference with TF
                raise Exception('The Merge layer ' + self.name +
                                ' has a callable `mode` argument, ' +
                                'and we cannot infer its output shape because ' +
                                'no `output_shape` argument was provided.' +
                                'Make sure to pass a shape tuple (or a callable) ' +
                                '`output_shape` to Merge.')
        # pre-defined merge modes
        input_shapes = input_shape
        if self.mode in ['sum', 'mul', 'ave', 'max']:
            # all tuples in input_shapes should be the same
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
            dot_axes = [a - 1 for a in self.dot_axes]
            tensordot_output = np.tensordot(np.zeros(tuple(shape1[1:])),
                                            np.zeros(tuple(shape2[1:])),
                                            axes=dot_axes)
            if len(tensordot_output.shape) == 0:
                shape = (1,)
            else:
                shape = tensordot_output.shape
            return (shape1[0],) + shape

    def compute_mask(self, inputs, mask=None):
        if mask is None or all([m is None for m in mask]):
            return None

        assert hasattr(mask, '__len__') and len(mask) == len(inputs)

        if self.mode in ['sum', 'mul', 'ave']:
            masks = [K.expand_dims(m, 0) for m in mask if m is not None]
            return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)
        elif self.mode == 'concat':
            masks = [K.ones_like(inputs[i][:-1]) if m is None else m for i, m in zip(inputs, mask)]
            expanded_dims = [K.expand_dims(m) for m in masks]
            concatenated = K.concatenate(expanded_dims, axis=self.concat_axis)
            return K.all(concatenated, axis=-1, keepdims=False)
        elif self.mode in ['cos', 'dot']:
            return None
        elif hasattr(self.mode, '__call__'):
            if hasattr(self._output_mask, '__call__'):
                return self._output_mask(mask)
            else:
                return self._output_mask
        else:
            # this should have been caught earlier
            raise Exception('Invalid merge mode: {}'.format(self.mode))

    def get_config(self):
        py3 = sys.version_info[0] == 3

        if isinstance(self.mode, python_types.LambdaType):
            if py3:
                mode = marshal.dumps(self.mode.__code__).decode('raw_unicode_escape')
            else:
                mode = marshal.dumps(self.mode.func_code).decode('raw_unicode_escape')
            mode_type = 'lambda'
        elif callable(self.mode):
            mode = self.mode.__name__
            mode_type = 'function'
        else:
            mode = self.mode
            mode_type = 'raw'

        if isinstance(self._output_shape, python_types.LambdaType):
            if py3:
                output_shape = marshal.dumps(self._output_shape.__code__).decode('raw_unicode_escape')
            else:
                output_shape = marshal.dumps(self._output_shape.func_code).decode('raw_unicode_escape')
            output_shape_type = 'lambda'
        elif callable(self._output_shape):
            output_shape = self._output_shape.__name__
            output_shape_type = 'function'
        else:
            output_shape = self._output_shape
            output_shape_type = 'raw'

        return {'name': self.name,
                'mode': mode,
                'mode_type': mode_type,
                'concat_axis': self.concat_axis,
                'dot_axes': self.dot_axes,
                'output_shape': output_shape,
                'output_shape_type': output_shape_type}

    @classmethod
    def from_config(cls, config):
        mode_type = config.pop('mode_type')
        if mode_type == 'function':
            mode = globals()[config['mode']]
        elif mode_type == 'lambda':
            mode = marshal.loads(config['mode'].encode('raw_unicode_escape'))
            mode = python_types.FunctionType(mode, globals())
        else:
            mode = config['mode']

        output_shape_type = config.pop('output_shape_type')
        if output_shape_type == 'function':
            output_shape = globals()[config['output_shape']]
        elif output_shape_type == 'lambda':
            output_shape = marshal.loads(config['output_shape'].encode('raw_unicode_escape'))
            output_shape = python_types.FunctionType(output_shape, globals())
        else:
            output_shape = config['output_shape']

        config['mode'] = mode
        config['output_shape'] = output_shape
        return super(Merge, cls).from_config(config)


def merge(inputs, mode='sum', concat_axis=-1,
          dot_axes=-1, output_shape=None, output_mask=None, name=None):
    '''Functional merge, to apply to Keras tensors (NOT layers).
    Returns a Keras tensor.

    # Example usage:

    ```python
    tensor_a = Input(shape=(32,))
    tensor_b = Input(shape=(32,))
    merged_tensor = merge([tensor_a, tensor_b], mode='concat', concat_axis=1)
    ```

    # Arguments
        mode: string or lambda/function. If string, must be one
            of: 'sum', 'mul', 'concat', 'ave', 'cos', 'dot'.
            If lambda/function, it should take as input a list of tensors
            and return a single tensor.
        concat_axis: integer, axis to use in mode `concat`.
        dot_axes: integer or tuple of integers, axes to use in mode `dot`.
        output_shape: shape tuple (tuple of integers), or lambda/function
            to compute output_shape (only if merge mode is a lambda/function).
            If the latter case, it should take as input a list of shape tuples
            (1:1 mapping to input tensors) and return a single shape tuple, including the
            batch size (same convention as the `get_output_shape_for` method of layers).
        node_indices: optional list of integers containing
            the output node index for each input layer
            (in case some input layers have multiple output nodes).
            will default to an array of 0s if not provided.
        tensor_indices: optional list of indices of output tensors
            to consider for merging
            (in case some input layer node returns multiple tensors).
    '''
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
                            name=name)
        return merge_layer(inputs)


class Container(Layer):
    '''TODO: dosctring

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

        (supports_masking (boolean))

        trainable_weights (list of variables)
        non_trainable_weights (list of variables)
        regularizers (list of regularizers)
        constraints (list of tuples (weight, constraint))

    # Methods
        summary
        get_layer
        get_weights
        set_weights
        get_config
        get_output_shape_for

    # Class Methods
        from_config
    '''
    def __init__(self, input, output, name=None):
        # handle name argument
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        # Container-specific properties
        if type(input) in {list, tuple}:
            self.inputs = list(input)  # tensor or list of tensors
        else:
            self.inputs = [input]
        if type(output) in {list, tuple}:
            self.outputs = list(output)
        else:
            self.outputs = [output]

        # check for redundancy in inputs:
        inputs_set = set(self.inputs)
        if len(inputs_set) != len(self.inputs):
            raise Exception('The list of inputs passed to the model '
                            'is redundant. All inputs should only appear once.'
                            ' Found: ' + str(self.inputs))

        # list of initial layers (1 to 1 mapping with self.inputs,
        # hence the same layer might appear twice)
        self.input_layers = []
        # TODO: probably useless because input layers must be Input layers (node_indices = [0], tensor_indices = [0])
        self.input_layers_node_indices = []
        self.input_layers_tensor_indices = []
        # list of layers (1 to 1 mapping with self.inputs,
        # hence the same layer might appear twice)
        self.output_layers = []
        # TODO: probably useless
        self.output_layers_node_indices = []
        self.output_layers_tensor_indices = []
        # all layers in order of horizontal graph traversal.
        # Entries are unique. Includes input and output layers.
        self.layers = []

        # this is for performance optimization
        # when calling the Container on new inputs.
        # every time the Container is called on a set on input tensors,
        # we compute the output tensors,
        # output masks and output shapes in one pass,
        # then cache them here. When of of these output is queried later,
        # we retrieve it from there instead of recomputing it.
        self._output_mask_cache = {}
        self._output_tensor_cache = {}
        self._output_shape_cache = {}

        # arguments validation
        for x in self.inputs:
            # check that x is a Keras tensor
            if not hasattr(x, '_keras_history'):
                cls_name = self.__class__.__name__
                raise Exception('Input tensors to a ' + cls_name + ' ' +
                                'must be Keras tensors. Found: ' + str(x) +
                                ' (missing Keras metadata).')
            # check that x is an input tensor
            layer, node_index, tensor_index = x._keras_history
            if len(layer.inbound_nodes) > 1 or (layer.inbound_nodes and layer.inbound_nodes[0].inbound_layers):
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
                raise Exception('Output tensors to a ' + cls_name + ' must be '
                                'Keras tensors. Found: ' + str(x))
        # build self.output_layers:
        for x in self.outputs:
            layer, node_index, tensor_index = x._keras_history
            self.output_layers.append(layer)
            self.output_layers_node_indices.append(node_index)
            self.output_layers_tensor_indices.append(tensor_index)

        # fill in the output mask cache
        masks = []
        for x in self.inputs:
            layer, node_index, tensor_index = x._keras_history
            node = layer.inbound_nodes[node_index]
            mask = node.output_masks[tensor_index]
            masks.append(mask)
        mask_cache_key = ','.join([str(id(x)) for x in self.inputs])
        mask_cache_key += '_' + ','.join([str(id(x)) for x in masks])
        masks = []
        for x in self.outputs:
            layer, node_index, tensor_index = x._keras_history
            node = layer.inbound_nodes[node_index]
            mask = node.output_masks[tensor_index]
            masks.append(mask)
        if len(masks) == 1:
            mask = masks[0]
        else:
            mask = masks
        self._output_mask_cache[mask_cache_key] = mask

        # build self.input_layers:
        for x in self.inputs:
            layer, node_index, tensor_index = x._keras_history
            # it's supposed to be an input layer, so only one node
            # and one tensor output
            assert node_index == 0
            assert tensor_index == 0
            self.input_layers.append(layer)
            self.input_layers_node_indices.append(node_index)
            self.input_layers_tensor_indices.append(tensor_index)

        # build self.input_names and self.output_names
        self.input_names = []
        self.output_names = []
        for layer in self.input_layers:
            self.input_names.append(layer.name)
        for layer in self.output_layers:
            self.output_names.append(layer.name)

        self.internal_input_shapes = [x._keras_shape for x in self.inputs]
        self.internal_output_shapes = [x._keras_shape for x in self.outputs]

        # container_nodes: set of nodes included in the graph
        # (not all nodes included in the layers are relevant to the current graph).
        container_nodes = set()  # ids of all nodes relevant to the Container
        nodes_depths = {}  # map {node: depth value}
        layers_depths = {}  # map {layer: depth value}

        def make_node_marker(node, depth):
            return str(id(node)) + '-' + str(depth)

        def build_map_of_graph(tensor, seen_nodes=set(), depth=0,
                               layer=None, node_index=None, tensor_index=None):
            '''This recursively updates the maps nodes_depths,
            layers_depths and the set container_nodes.
            Does not try to detect cycles in graph (TODO?)

            # Arguments
                tensor: some tensor in a graph
                seen_nodes: set of node ids ("{layer.name}_ib-{node_index}")
                    of nodes seen so far. Useful to prevent infinite loops.
                depth: current depth in the graph (0 = last output).
                layer: layer from which `tensor` comes from. If not provided,
                    will be obtained from `tensor._keras_history`.
                node_index: node index from which `tensor` comes from.
                tensor_index: tensor_index from which `tensor` comes from.
            '''
            if not layer or node_index is None or tensor_index is None:
                layer, node_index, tensor_index = tensor._keras_history
            node = layer.inbound_nodes[node_index]

            # prevent cycles
            seen_nodes.add(make_node_marker(node, depth))

            node_key = layer.name + '_ib-' + str(node_index)
            # update container_nodes
            container_nodes.add(node_key)
            # update nodes_depths
            node_depth = nodes_depths.get(node)
            if node_depth is None:
                nodes_depths[node] = depth
            else:
                nodes_depths[node] = max(depth, node_depth)
            # update layers_depths
            previously_seen_depth = layers_depths.get(layer)
            if previously_seen_depth is None:
                current_depth = depth
            else:
                current_depth = max(depth, previously_seen_depth)
            layers_depths[layer] = current_depth

            # propagate to all previous tensors connected to this node
            for i in range(len(node.inbound_layers)):
                x = node.input_tensors[i]
                layer = node.inbound_layers[i]
                node_index = node.node_indices[i]
                tensor_index = node.tensor_indices[i]
                next_node = layer.inbound_nodes[node_index]
                # use node_marker to prevent cycles
                node_marker = make_node_marker(next_node, current_depth + 1)
                if node_marker not in seen_nodes:
                    build_map_of_graph(x, seen_nodes, current_depth + 1,
                                       layer, node_index, tensor_index)

        for x in self.outputs:
            seen_nodes = set()
            build_map_of_graph(x, seen_nodes, depth=0)

        # build a map {depth: list of nodes with this depth}
        nodes_by_depth = {}
        for node, depth in nodes_depths.items():
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(node)

        # build a map {depth: list of layers with this depth}
        layers_by_depth = {}
        for layer, depth in layers_depths.items():
            if depth not in layers_by_depth:
                layers_by_depth[depth] = []
            layers_by_depth[depth].append(layer)

        # get sorted list of layer depths
        depth_keys = list(layers_by_depth.keys())
        depth_keys.sort(reverse=True)

        # set self.layers and self.layers_by_depth
        layers = []
        for depth in depth_keys:
            layers_for_depth = layers_by_depth[depth]
            # container.layers needs to have a deterministic order
            layers_for_depth.sort(key=lambda x: x.name)
            for layer in layers_for_depth:
                layers.append(layer)
        self.layers = layers
        self.layers_by_depth = layers_by_depth

        # get sorted list of node depths
        depth_keys = list(nodes_by_depth.keys())
        depth_keys.sort(reverse=True)

        # check that all tensors required are computable.
        # computable_tensors: all tensors in the graph
        # that can be computed from the inputs provided
        computable_tensors = []
        for x in self.inputs:
            computable_tensors.append(x)

        layers_with_complete_input = []  # to provide a better error msg
        for depth in depth_keys:
            for node in nodes_by_depth[depth]:
                layer = node.outbound_layer
                if layer:
                    for x in node.input_tensors:
                        if x not in computable_tensors:
                            raise Exception(
                                'Graph disconnected: '
                                'cannot obtain value for tensor ' +
                                str(x) + ' at layer "' + layer.name + '". '
                                'The following previous layers '
                                'were accessed without issue: ' +
                                str(layers_with_complete_input))
                    for x in node.output_tensors:
                        computable_tensors.append(x)
                    layers_with_complete_input.append(layer.name)

        # set self.nodes and self.nodes_by_depth
        self.container_nodes = container_nodes
        self.nodes_by_depth = nodes_by_depth

        # ensure name unicity, which will be crucial for serialization
        # (since serialized nodes refer to layers by their name).
        all_names = [layer.name for layer in self.layers]
        for name in all_names:
            if all_names.count(name) != 1:
                raise Exception('The name "' + name + '" is used ' +
                                str(all_names.count(name)) +
                                ' times in the model. ' +
                                'All layer names should be unique.')

        # layer parameters
        # the new container starts with a single inbound node
        # for its inputs, and no outbound nodes.
        self.outbound_nodes = []  # will be appended to by future calls to __call__
        self.inbound_nodes = []  # will be appended to below, and by future calls to __call__
        # create the node linking internal inputs to internal outputs
        Node(outbound_layer=self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=self.inputs,
             output_tensors=self.outputs,
             # no container-level masking for now
             input_masks=[None for _ in self.inputs],
             output_masks=[None for _ in self.outputs],
             input_shapes=[x._keras_shape for x in self.inputs],
             output_shapes=[x._keras_shape for x in self.outputs])
        self.built = True
        self.supports_masking = False
        # the following are implemented as property functions:
        # self.constraints
        # self.regularizers
        # self.trainable_weights
        # self.non_trainable_weights
        # self.input_spec

    def get_layer(self, name=None, index=None):
        '''Returns a layer based on either its name (unique)
        or its index in the graph. Indices are based on
        order of horizontal graph traversal (bottom-up).

        # Arguments
            name: string, name of layer.
            index: integer, index of layer.

        # Returns
            A layer instance.
        '''
        # it would be unreliable to build a dictionary
        # based on layer names, because names can potentially
        # be changed at any point by the user
        # without the container being notified of it
        if index:
            if len(self.layers) <= index:
                raise Exception('Was asked to retrieve layer at index ' +
                                str(index) + ' but model only has ' +
                                str(len(self.layers)) + ' layers.')
        else:
            assert name, 'Provide either a layer name or layer index.'
        layer = None
        for layer in self.layers:
            if layer.name == name:
                return layer
        if not layer:
            raise Exception('No such layer: ' + name)

    @property
    def updates(self):
        updates = []
        for layer in self.layers:
            if hasattr(layer, 'updates'):
                updates += layer.updates
        return updates

    @property
    def stateful(self):
        return any([(hasattr(layer, 'stateful') and layer.stateful) for layer in self.layers])

    def reset_states(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_states') and getattr(layer, 'stateful', False):
                layer.reset_states()

    @property
    def state_updates(self):
        '''Returns the `updates` from all layers that are
        stateful.  This is useful for separating training updates and
        state updates, e.g. when we need to update a layer's internal state
        during prediction.
        '''
        state_updates = []
        for layer in self.layers:
            if getattr(layer, 'stateful', False):
                if hasattr(layer, 'updates'):
                    state_updates += layer.updates
        return state_updates

    @property
    def constraints(self):
        cons = {}
        for layer in self.layers:
            for key, value in layer.constraints.items():
                if key in cons:
                    raise Exception('Received multiple constraints '
                                    'for one weight tensor: ' + str(key))
                cons[key] = value
        return cons

    @property
    def regularizers(self):
        regs = []
        for layer in self.layers:
            regs += layer.regularizers
        return regs

    @property
    def trainable_weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.non_trainable_weights
        return weights

    @property
    def input_spec(self):
        specs = []
        for layer in getattr(self, 'input_layers', []):
            if layer.input_spec is None:
                specs.append(None)
            else:
                if type(layer.input_spec) is not list:
                    raise Exception('Layer ' + layer.name +
                                    ' has an input_spec attribute that '
                                    'is not a list. We expect a list. '
                                    'Found input_spec = ' +
                                    str(layer.input_spec))
                specs += layer.input_spec
        return specs

    @property
    def uses_learning_phase(self):
        '''True if any layer in the graph uses it.
        '''
        layers_learning_phase = any([layer.uses_learning_phase for layer in self.layers])
        regs_learning_phase = any([reg.uses_learning_phase for reg in self.regularizers])
        return layers_learning_phase or regs_learning_phase

    def call(self, input, mask=None):
        '''`call` just reapplies all ops in the graph to the new inputs
        (e.g. build a new computational graph from the provided inputs).

        It is callable on non-Keras tensors.

        # Arguments
            input: a tensor or list of tensors.
            mask: a mask or list of masks. A mask can be
                either a tensor or None (no mask).

        # Returns
            A tensor if there is a single output, or
            a list of tensors if there are more than one outputs.
        '''
        inputs = to_list(input)
        if mask is None:
            masks = [None for _ in range(len(inputs))]
        else:
            masks = to_list(mask)
        cache_key = ','.join([str(id(x)) for x in inputs])
        cache_key += '_' + ','.join([str(id(x)) for x in masks])
        if cache_key in self._output_tensor_cache:
            return self._output_tensor_cache[cache_key]
        else:
            output_tensors, output_masks, output_shapes = self.run_internal_graph(inputs, masks)
            return output_tensors

    def compute_mask(self, input, mask):
        inputs = to_list(input)
        if mask is None:
            masks = [None for _ in range(len(inputs))]
        else:
            masks = to_list(mask)
        cache_key = ','.join([str(id(x)) for x in inputs])
        cache_key += '_' + ','.join([str(id(x)) for x in masks])
        if cache_key in self._output_mask_cache:
            return self._output_mask_cache[cache_key]
        else:
            output_tensors, output_masks, output_shapes = self.run_internal_graph(inputs, masks)
            return output_masks

    def get_output_shape_for(self, input_shape):
        input_shapes = to_list(input_shape)
        if len(input_shapes) != len(self.input_layers):
            raise Exception('Invalid input_shape argument ' +
                            str(input_shape) + ': model has ' +
                            str(len(self.input_layers)) + ' tensor inputs.')

        cache_key = ','.join([str(x) for x in input_shapes])
        if cache_key in self._output_shape_cache:
            output_shapes = self._output_shape_cache[cache_key]
            if type(output_shapes) is list and len(output_shapes) == 1:
                return output_shapes[0]
            return output_shapes
        else:
            # bad luck, have to run the graph manually
            layers_to_output_shapes = {}
            for i in range(len(input_shapes)):
                layer = self.input_layers[i]
                input_shape = input_shapes[i]
                # it's an input layer: get_output_shape_for is identity,
                # and there is only one node and one tensor output.
                shape_key = layer.name + '_0_0'
                layers_to_output_shapes[shape_key] = input_shape

            depth_keys = list(self.nodes_by_depth.keys())
            depth_keys.sort(reverse=True)
            # iterate over nodes, by depth level
            if len(depth_keys) > 1:
                for depth in depth_keys:
                    nodes = self.nodes_by_depth[depth]
                    for node in nodes:
                        # this is always a single layer, never a list
                        layer = node.outbound_layer
                        if layer in self.input_layers:
                            # we've already covered the input layers
                            # a few lines above
                            continue
                        # potentially redundant list,
                        # same size of node.input_tensors
                        input_shapes = []
                        for j in range(len(node.inbound_layers)):
                            inbound_layer = node.inbound_layers[j]
                            node_index = node.node_indices[j]
                            tensor_index = node.tensor_indices[j]
                            shape_key = inbound_layer.name + '_%s_%s' % (node_index, tensor_index)
                            input_shape = layers_to_output_shapes[shape_key]
                            input_shapes.append(input_shape)

                        if len(input_shapes) == 1:
                            output_shape = layer.get_output_shape_for(input_shapes[0])
                        else:
                            output_shape = layer.get_output_shape_for(input_shapes)

                        output_shapes = to_list(output_shape)
                        node_index = layer.inbound_nodes.index(node)
                        for j in range(len(output_shapes)):
                            shape_key = layer.name + '_%s_%s' % (node_index, j)
                            layers_to_output_shapes[shape_key] = output_shapes[j]

            # read final output shapes from layers_to_output_shapes
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
            # store in cache
            self._output_shape_cache[cache_key] = output_shapes
            if type(output_shapes) is list and len(output_shapes) == 1:
                return output_shapes[0]
            return output_shapes

    def run_internal_graph(self, inputs, masks=None):
        '''Computes output tensors for new inputs.

        # Note:
            - expects `inputs` to be a list (potentially with 1 element).
            - can be run on non-Keras tensors.

        # Arguments
            inputs: list of tensors
            masks: list of masks (tensors or None).

        # Returns
            Three lists: output_tensors, output_masks, output_shapes
        '''
        assert type(inputs) is list
        if masks is None:
            masks = [None for _ in range(len(inputs))]
        assert type(masks) is list

        # dictionary mapping reference tensors to tuples (computed tensor, compute mask)
        # we assume a 1:1 mapping from tensor to mask
        # TODO: raise exception when a .compute_mask does not return a list the same size as call
        tensor_map = {}
        for x, y, mask in zip(self.inputs, inputs, masks):
            tensor_map[str(id(x))] = (y, mask)

        depth_keys = list(self.nodes_by_depth.keys())
        depth_keys.sort(reverse=True)
        for depth in depth_keys:
            nodes = self.nodes_by_depth[depth]
            for node in nodes:
                # this is always a single layer, never a list
                layer = node.outbound_layer

                reference_input_tensors = node.input_tensors
                reference_output_tensors = node.output_tensors

                # if all previous input tensors are available in tensor_map,
                # then call node.inbound_layer on them
                computed_data = []  # list of tuples (input, mask)
                for x in reference_input_tensors:
                    if str(id(x)) in tensor_map:
                        computed_data.append(tensor_map[str(id(x))])
                if len(computed_data) == len(reference_input_tensors):
                    # call layer
                    if len(computed_data) == 1:
                        computed_tensor, computed_mask = computed_data[0]
                        output_tensors = to_list(layer.call(computed_tensor, computed_mask))
                        output_masks = to_list(layer.compute_mask(computed_tensor, computed_mask))
                        computed_tensors = [computed_tensor]
                        computed_masks = [computed_mask]
                    else:
                        computed_tensors = [x[0] for x in computed_data]
                        computed_masks = [x[1] for x in computed_data]
                        output_tensors = to_list(layer.call(computed_tensors, computed_masks))
                        output_masks = to_list(layer.compute_mask(computed_tensors, computed_masks))

                    # update _keras_shape
                    if all([hasattr(x, '_keras_shape') for x in computed_tensors]):
                        if len(computed_tensors) == 1:
                            shapes = to_list(layer.get_output_shape_for(computed_tensors[0]._keras_shape))
                            uses_learning_phase = computed_tensors[0]._uses_learning_phase or layer.uses_learning_phase
                        else:
                            shapes = to_list(layer.get_output_shape_for([x._keras_shape for x in computed_tensors]))
                            uses_learning_phase = any([x._uses_learning_phase for x in computed_tensors]) or layer.uses_learning_phase
                        for x, s in zip(output_tensors, shapes):
                            x._keras_shape = s
                            x._uses_learning_phase = uses_learning_phase

                    # update tensor_map
                    for x, y, mask in zip(reference_output_tensors, output_tensors, output_masks):
                        tensor_map[str(id(x))] = (y, mask)

        output_tensors = []
        output_masks = []
        output_shapes = []
        for x in self.outputs:
            # todo: better error msg
            assert str(id(x)) in tensor_map, 'Could not compute output ' + str(x)
            tensor, mask = tensor_map[str(id(x))]
            if hasattr(tensor, '_keras_shape') and output_shapes is not None:
                shape = tensor._keras_shape
                output_shapes.append(shape)
            else:
                output_shapes = None
            output_tensors.append(tensor)
            output_masks.append(mask)

        # update cache; keys are based on ids on input tensors and inputs masks
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
        node_conversion_map = {}
        for layer in self.layers:
            if issubclass(layer.__class__, Container):
                # containers start with a pre-existing node
                # linking their input to output
                kept_nodes = 1
            else:
                kept_nodes = 0
            for original_node_index, node in enumerate(layer.inbound_nodes):
                node_key = layer.name + '_ib-' + str(original_node_index)
                if node_key in self.container_nodes:
                    node_conversion_map[node_key] = kept_nodes
                    kept_nodes += 1
        layer_configs = []
        for layer in self.layers:  # from the earliest layers on
            layer_class_name = layer.__class__.__name__
            layer_config = layer.get_config()
            filtered_inbound_nodes = []
            for original_node_index, node in enumerate(layer.inbound_nodes):
                node_key = layer.name + '_ib-' + str(original_node_index)
                if node_key in self.container_nodes:
                    # the node is relevant to the model:
                    # add to filtered_inbound_nodes
                    if node.inbound_layers:
                        node_data = []
                        for i in range(len(node.inbound_layers)):
                            inbound_layer = node.inbound_layers[i]
                            node_index = node.node_indices[i]
                            tensor_index = node.tensor_indices[i]
                            node_key = inbound_layer.name + '_ib-' + str(node_index)
                            # assert node_key in node_conversion_map, 'Node never seen before: %s' % node_key
                            new_node_index = node_conversion_map.get(node_key, 0)
                            node_data.append([inbound_layer.name,
                                              new_node_index,
                                              tensor_index])
                        filtered_inbound_nodes.append(node_data)
            layer_configs.append({
                'name': layer.name,
                'class_name': layer_class_name,
                'config': layer_config,
                'inbound_nodes': filtered_inbound_nodes,
            })
        config['layers'] = layer_configs

        # gather info about inputs and outputs
        model_inputs = []
        for i in range(len(self.input_layers)):
            layer = self.input_layers[i]
            node_index = self.input_layers_node_indices[i]
            node_key = layer.name + '_ib-' + str(node_index)
            new_node_index = node_conversion_map[node_key]
            tensor_index = self.input_layers_tensor_indices[i]
            model_inputs.append([layer.name, new_node_index, tensor_index])
        config['input_layers'] = model_inputs
        model_outputs = []
        for i in range(len(self.output_layers)):
            layer = self.output_layers[i]
            node_index = self.output_layers_node_indices[i]
            node_key = layer.name + '_ib-' + str(node_index)
            new_node_index = node_conversion_map[node_key]
            tensor_index = self.output_layers_tensor_indices[i]
            model_outputs.append([layer.name, new_node_index, tensor_index])
        config['output_layers'] = model_outputs
        return copy.deepcopy(config)

    @classmethod
    def from_config(cls, config, custom_objects={}):
        '''Instantiates a Model from its config (output of `get_config()`).

        TODO: support for custom objects
        '''
        from keras.utils.layer_utils import layer_from_config

        # layer instances created during
        # the graph reconstruction process
        created_layers = {}

        def process_layer(layer_data):
            # iterate over saved layers, instantiate them,
            # then call them on appropriate inputs to create graph nodes
            layer_name = layer_data['name']

            # instantiate layer
            layer = layer_from_config(layer_data,
                                      custom_objects=custom_objects)
            created_layers[layer_name] = layer

            # gather layer inputs
            inbound_nodes_data = layer_data['inbound_nodes']
            for node_data in inbound_nodes_data:
                input_tensors = []
                for input_data in node_data:
                    inbound_layer_name, inbound_node_index, inbound_tensor_index = input_data
                    assert inbound_layer_name in created_layers, 'Missing layer: %s' % inbound_layer_name
                    inbound_layer = created_layers[inbound_layer_name]
                    inbound_node = inbound_layer.inbound_nodes[inbound_node_index]
                    input_tensors.append(inbound_node.output_tensors[inbound_tensor_index])
                # call layer on its inputs, thus creating the node
                # and building the layer if needed
                if input_tensors:
                    if len(input_tensors) == 1:
                        layer(input_tensors[0])
                    else:
                        layer(input_tensors)

        for layer_data in config['layers']:
            process_layer(layer_data)

        name = config.get('name')
        input_tensors = []
        output_tensors = []
        for layer_data in config['input_layers']:
            layer_name, node_index, tensor_index = layer_data
            assert layer_name in created_layers
            layer = created_layers[layer_name]
            layer_output_tensors = layer.inbound_nodes[node_index].output_tensors
            input_tensors.append(layer_output_tensors[tensor_index])
        for layer_data in config['output_layers']:
            layer_name, node_index, tensor_index = layer_data
            assert layer_name in created_layers
            layer = created_layers[layer_name]
            layer_output_tensors = layer.inbound_nodes[node_index].output_tensors
            output_tensors.append(layer_output_tensors[tensor_index])
        return cls(input=input_tensors, output=output_tensors, name=name)

    def save_weights(self, filepath, overwrite=False):
        '''Dumps all layer weights to a HDF5 file.

        The weight file has:
            - `layer_names` (attribute), a list of strings
                (ordered names of model layers)
            - for every layer, a `group` named `layer.name`
                - for every such layer group, a group attribute `weight_names`,
                    a list of strings (ordered names of weights tensor of the layer)
                - for every weight in the layer, a dataset
                    storing the weight value, named after the weight tensor
        '''
        import h5py
        import os.path
        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(filepath):
            import sys
            get_input = input
            if sys.version_info[:2] <= (2, 7):
                get_input = raw_input
            overwrite = get_input('[WARNING] %s already exists - overwrite? '
                                  '[y/n]' % (filepath))
            while overwrite not in ['y', 'n']:
                overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
            if overwrite == 'n':
                return
            print('[TIP] Next time specify overwrite=True in save_weights!')

        if hasattr(self, 'flattened_layers'):
            # support for legacy Sequential/Merge behavior
            flattened_layers = self.flattened_layers
        else:
            flattened_layers = self.layers

        f = h5py.File(filepath, 'w')
        f.attrs['layer_names'] = [layer.name.encode('utf8') for layer in flattened_layers]

        for layer in flattened_layers:
            g = f.create_group(layer.name)
            symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
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
                param_dset[:] = val
        f.flush()
        f.close()

    def load_weights(self, filepath):
        '''Load all layer weights from a HDF5 save file.
        '''
        import h5py
        f = h5py.File(filepath, mode='r')

        if hasattr(self, 'flattened_layers'):
            # support for legacy Sequential/Merge behavior
            flattened_layers = self.flattened_layers
        else:
            flattened_layers = self.layers

        if 'nb_layers' in f.attrs:
            # legacy format
            nb_layers = f.attrs['nb_layers']
            if nb_layers != len(flattened_layers):
                raise Exception('You are trying to load a weight file '
                                'containing ' + str(nb_layers) +
                                ' layers into a model with ' +
                                str(len(flattened_layers)) + '.')

            for k in range(nb_layers):
                g = f['layer_{}'.format(k)]
                weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
                flattened_layers[k].set_weights(weights)
        else:
            # new file format
            layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
            if len(layer_names) != len(flattened_layers):
                raise Exception('You are trying to load a weight file '
                                'containing ' + str(len(layer_names)) +
                                ' layers into a model with ' +
                                str(len(flattened_layers)) + ' layers.')

            # we batch weight value assignments in a single backend call
            # which provides a speedup in TensorFlow.
            weight_value_tuples = []
            for k, name in enumerate(layer_names):
                g = f[name]
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                if len(weight_names):
                    weight_values = [g[weight_name] for weight_name in weight_names]
                    layer = flattened_layers[k]
                    symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                    if len(weight_values) != len(symbolic_weights):
                        raise Exception('Layer #' + str(k) +
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
        f.close()

    def _updated_config(self):
        '''shared between different serialization methods'''
        from keras import __version__ as keras_version

        config = self.get_config()
        model_config = {
            'class_name': self.__class__.__name__,
            'config': config,
            'keras_version': keras_version
        }

        if hasattr(self, 'optimizer'):
            model_config['optimizer'] = self.optimizer.get_config()
            model_config['loss'] = getattr(self.loss, '__name__', self.loss)
            model_config['sample_weight_mode'] = self.sample_weight_mode

        if hasattr(self, 'loss_weights'):
            model_config['loss_weights'] = self.loss_weights
        return model_config

    def to_json(self, **kwargs):
        '''Returns a JSON string containing the network configuration.

        To load a network from a JSON save file, use
        `keras.models.model_from_json(json_string, custom_objects={})`.
        '''
        import json

        def get_json_type(obj):
            # if obj is any numpy type
            if type(obj).__module__ == np.__name__:
                return obj.item()

            # if obj is a python 'type'
            if type(obj).__name__ == type.__name__:
                return obj.__name__

            raise TypeError('Not JSON Serializable')

        model_config = self._updated_config()
        return json.dumps(model_config, default=get_json_type, **kwargs)

    def to_yaml(self, **kwargs):
        '''Returns a yaml string containing the network configuration.

        To load a network from a yaml save file, use
        `keras.models.model_from_yaml(yaml_string, custom_objects={})`.

        `custom_objects` should be a dictionary mapping
        the names of custom losses / layers / etc to the corresponding
        functions / classes.
        '''
        import yaml
        return yaml.dump(self._updated_config(), **kwargs)

    def summary(self, line_length=100, positions=[.33, .55, .67, 1.]):
        from keras.utils.layer_utils import print_summary

        if hasattr(self, 'flattened_layers'):
            # support for legacy Sequential/Merge behavior
            flattened_layers = self.flattened_layers
        else:
            flattened_layers = self.layers

        print_summary(flattened_layers, getattr(self, 'container_nodes', None), line_length=line_length, positions=positions)


def get_source_inputs(tensor, layer=None, node_index=None):
    '''Returns the list of input tensors
    necessary to compute `tensor`.

    Output will always be a list of tensors
    (potentially with 1 element).

    # Arguments
        tensor: the tensor to start from.
        layer: origin layer of the tensor. Will be
            determined via tensor._keras_history if not provided.
        node_index: origin node index of the tensor.
    '''
    if not hasattr(tensor, '_keras_history'):
        raise Exception('Tensor must be a Keras tensor. Found: ' + str(tensor))

    if layer is None or node_index:
        layer, node_index, _ = tensor._keras_history
    if not layer.inbound_nodes:
        return [tensor]
    else:
        node = layer.inbound_nodes[node_index]
        if not node.inbound_layers:
            # reached an Input layer, stop recursion
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
                # avoid input redundancy
                for x in previous_sources:
                    if x not in source_tensors:
                        source_tensors.append(x)
            return source_tensors
