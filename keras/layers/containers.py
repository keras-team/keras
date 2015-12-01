# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from collections import OrderedDict
from .. import backend as K
from ..layers.core import Layer, Merge, Siamese, SiameseHead
from six.moves import range


class Sequential(Layer):
    '''
        Simple linear stack of layers.

        inherited from Layer:
        - get_params
        - get_output_mask
        - supports_masked_input
    '''

    def __init__(self, layers=[]):
        self.layers = []
        for layer in layers:
            self.add(layer)

    def set_previous(self, layer):
        self.layers[0].previous = layer

    def add(self, layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].set_previous(self.layers[-2])
            if not hasattr(self.layers[0], 'input'):
                self.set_input()

    @property
    def params(self):
        params = []
        for l in self.layers:
            if l.trainable:
                params += l.get_params()[0]
        return params

    @property
    def regularizers(self):
        regularizers = []
        for l in self.layers:
            if l.trainable:
                regularizers += l.get_params()[1]
        return regularizers

    @property
    def constraints(self):
        constraints = []
        for l in self.layers:
            if l.trainable:
                constraints += l.get_params()[2]
        return constraints

    @property
    def updates(self):
        updates = []
        for l in self.layers:
            if l.trainable:
                updates += l.get_params()[3]
        return updates

    @property
    def output_shape(self):
        return self.layers[-1].output_shape

    def get_output(self, train=False):
        return self.layers[-1].get_output(train)

    def set_input(self):
        for l in self.layers:
            if hasattr(l, 'input'):
                ndim = len(K.get_shape(l.input))
                self.layers[0].input = K.placeholder(ndim=ndim)
                break

    def get_input(self, train=False):
        if not hasattr(self.layers[0], 'input'):
            self.set_input()
        return self.layers[0].get_input(train)

    @property
    def input_shape(self):
        return self.layers[0].input_shape

    @property
    def input(self):
        return self.get_input()

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.get_weights()
        return weights

    def set_weights(self, weights):
        for i in range(len(self.layers)):
            nb_param = len(self.layers[i].params)
            self.layers[i].set_weights(weights[:nb_param])
            weights = weights[nb_param:]

    def get_config(self):
        return {"name": self.__class__.__name__,
                "layers": [layer.get_config() for layer in self.layers]}

    def count_params(self):
        return sum([layer.count_params() for layer in self.layers])


class Graph(Layer):
    '''
        Implement a NN graph with arbitrary layer connections,
        arbitrary number of inputs and arbitrary number of outputs.

        Note: Graph can only be used as a layer
        (connect, input, get_input, get_output)
        when it has exactly one input and one output.

        inherited from Layer:
            - get_output_mask
            - supports_masked_input
            - get_weights
            - set_weights
    '''
    def __init__(self):
        self.namespace = set()  # strings
        self.nodes = OrderedDict()  # layer-like
        self.inputs = {}  # layer-like
        self.input_order = []  # strings
        self.outputs = {}  # layer-like
        self.output_order = []  # strings
        self.input_config = []  # dicts
        self.output_config = []  # dicts
        self.node_config = []  # dicts

    @property
    def nb_input(self):
        return len(self.inputs)

    @property
    def nb_output(self):
        return len(self.outputs)

    @property
    def params(self):
        params = []
        for l in self.nodes.values():
            if l.trainable:
                params += l.get_params()[0]
        return params

    @property
    def regularizers(self):
        regularizers = []
        for l in self.nodes.values():
            if l.trainable:
                regularizers += l.get_params()[1]
        return regularizers

    @property
    def constraints(self):
        constraints = []
        for l in self.nodes.values():
            if l.trainable:
                constraints += l.get_params()[2]
        return constraints

    @property
    def updates(self):
        updates = []
        for l in self.nodes.values():
            if l.trainable:
                updates += l.get_params()[3]
        return updates

    def set_previous(self, layer, connection_map={}):
        if self.nb_input != layer.nb_output:
            raise Exception('Cannot connect layers: input count does not match output count.')
        if self.nb_input == 1:
            self.inputs[self.input_order[0]].set_previous(layer)
        else:
            if not connection_map:
                raise Exception('Cannot attach multi-input layer: no connection_map provided.')
            for k, v in connection_map.items():
                if k in self.inputs and v in layer.outputs:
                    self.inputs[k].set_previous(layer.outputs[v])
                else:
                    raise Exception('Invalid connection map.')

    def get_input(self, train=False):
        if len(self.inputs) == len(self.outputs) == 1:
            return self.inputs[self.input_order[0]].get_input(train)
        else:
            return dict([(k, v.get_input(train)) for k, v in self.inputs.items()])

    @property
    def input(self):
        return self.get_input()

    @property
    def output_shape(self):
        if self.nb_output == 1:
            # return tuple
            return self.outputs[self.output_order[0]].output_shape
        else:
            # return dictionary mapping output names to shape tuples
            return dict([(k, v.output_shape) for k, v in self.outputs.items()])

    def get_output(self, train=False):
        if len(self.inputs) == len(self.outputs) == 1:
            return self.outputs[self.output_order[0]].get_output(train)
        else:
            return dict([(k, v.get_output(train)) for k, v in self.outputs.items()])

    def add_input(self, name, input_shape=None, batch_input_shape=None, dtype='float'):
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)
        self.namespace.add(name)
        self.input_order.append(name)
        layer = Layer()  # empty layer
        if input_shape:
            layer.set_input_shape((None,) + tuple(input_shape))
        elif batch_input_shape:
            layer.set_input_shape(batch_input_shape)
        if dtype == 'float':
            layer.input = K.placeholder(shape=layer.input_shape, name=name)
        else:
            if len(input_shape) == 1:
                layer.input = K.placeholder(shape=layer.input_shape,
                                            dtype='int32',
                                            name=name)
            else:
                raise Exception('Type "int" can only be used with ndim==2 (Embedding).')
        self.inputs[name] = layer
        self.input_config.append({'name': name,
                                  'input_shape': input_shape,
                                  'dtype': dtype})

    def add_node(self, layer, name, input=None, inputs=[],
                 merge_mode='concat', concat_axis=-1, dot_axes=-1,
                 create_output=False):
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)
        if input:
            if input not in self.namespace:
                raise Exception('Unknown node/input identifier: ' + input)
            if input in self.nodes:
                layer.set_previous(self.nodes[input])
            elif input in self.inputs:
                layer.set_previous(self.inputs[input])
        if inputs:
            to_merge = []
            for n in inputs:
                if n in self.nodes:
                    to_merge.append(self.nodes[n])
                elif n in self.inputs:
                    to_merge.append(self.inputs[n])
                else:
                    raise Exception('Unknown identifier: ' + n)
            merge = Merge(to_merge, mode=merge_mode,
                          concat_axis=concat_axis, dot_axes=dot_axes)
            layer.set_previous(merge)

        self.namespace.add(name)
        self.nodes[name] = layer
        self.node_config.append({'name': name,
                                 'input': input,
                                 'inputs': inputs,
                                 'merge_mode': merge_mode,
                                 'concat_axis': concat_axis,
                                 'dot_axes': dot_axes,
                                 'create_output': create_output})

        if create_output:
            self.add_output(name, input=name)

    def add_shared_node(self, layer, name, inputs=[], merge_mode=None,
                        concat_axis=-1, dot_axes=-1, outputs=[],
                        create_output=False):
        '''
        Used to shared / multi input-multi output node

        Arguments
        ------------
        layer - The layer to be shared across multiple inputs
        name - Name of the shared layer
        inputs - List of names of input nodes
        merge_mode - Similar to merge_mode argument of add_node()
        concat_axis - Similar to concat_axis argument of add_node()
        dot_axes - Similar to dot_axes argument of add_node()
        outputs - Names for output nodes. Used when merge_mode = None
        create_output -  Similar to create_output argument of add_node().
            Output will be created only if merge_mode is given
        '''
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)
        for o in outputs:
            if o in self.namespace:
                raise Exception('Duplicate node identifier: ' + o)
        if merge_mode:
            if merge_mode not in {'sum', 'ave', 'mul', 'dot', 'cos', 'concat', 'join'}:
                raise Eception("Invalid merge mode")
        layers = []
        for i in range(len(inputs)):
            input = inputs[i]
            if input in self.nodes:
                n = self.nodes[input]
                if n.__class__.__name__ == 'Siamese':
                    if n.merge_mode is None:
                        for j in range(len(n.inputs)):
                            sh = SiameseHead(j)
                            sh.previous = n
                            layers.append(sh)
                    else:
                        layers.append(n)
                else:
                    layers.append(n)
            elif input in self.inputs:
                n = self.inputs[input]
                layers.append(n)
            else:
                raise Exception('Unknown identifier: ' + input)
        s = Siamese(layer, layers, merge_mode, concat_axis=concat_axis, dot_axes=dot_axes)
        s.set_name(name)
        self.namespace.add(name)
        self.nodes[name] = s
        self.node_config.append({'name': name,
                                 'inputs': inputs,
                                 'merge_mode': merge_mode,
                                 'concat_axis': concat_axis,
                                 'dot_axes': dot_axes,
                                 'create_output': create_output if merge_mode else False})
        if not merge_mode:
            for i in range(len(outputs)):
                sh = SiameseHead(i)
                sh.previous = s
                sh_name = outputs[i]
                sh.set_name(sh_name)
                self.namespace.add(sh_name)
                self.nodes[sh_name] = sh
                self.node_config.append({'name': sh_name,
                                         'inputs': [s],
                                         'create_output': create_output})
                if create_output:
                    self.add_output(sh_name, input=sh_name)

        if create_output and merge_mode:
            if merge_mode == 'join':
                raise Exception("Output can not be of type OrderedDict")
            self.add_output(name, input=name)

    def add_output(self, name, input=None, inputs=[],
                   merge_mode='concat', concat_axis=-1, dot_axes=-1):
        if name in self.output_order:
            raise Exception('Duplicate output identifier: ' + name)
        if input:
            if input not in self.namespace:
                raise Exception('Unknown node/input identifier: ' + input)
            if input in self.nodes:
                self.outputs[name] = self.nodes[input]
            elif input in self.inputs:
                self.outputs[name] = self.inputs[input]
        if inputs:
            to_merge = []
            for n in inputs:
                if n not in self.nodes:
                    raise Exception('Unknown identifier: ' + n)
                to_merge.append(self.nodes[n])
            merge = Merge(to_merge, mode=merge_mode,
                          concat_axis=concat_axis, dot_axes=dot_axes)
            self.outputs[name] = merge

        self.output_order.append(name)
        self.output_config.append({'name': name,
                                   'input': input,
                                   'inputs': inputs,
                                   'merge_mode': merge_mode,
                                   'concat_axis': concat_axis,
                                   'dot_axes': dot_axes})

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_config": self.input_config,
                "node_config": self.node_config,
                "output_config": self.output_config,
                "input_order": self.input_order,
                "output_order": self.output_order,
                "nodes": dict([(c["name"], self.nodes[c["name"]].get_config()) for c in self.node_config])}

    def count_params(self):
        return sum([layer.count_params() for layer in self.nodes.values()])

    def get_weights(self):
        weights = []
        for layer in self.nodes.values():
            weights += layer.get_weights()
        return weights

    def set_weights(self, weights):
        for layer in self.nodes.values():
            nb_param = len(layer.get_weights())
            layer.set_weights(weights[:nb_param])
            weights = weights[nb_param:]
