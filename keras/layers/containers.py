# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import theano.tensor as T
from ..layers.core import Layer, Merge
from six.moves import range

def ndim_tensor(ndim):
    if ndim == 2:
        return T.matrix()
    elif ndim == 3:
        return T.tensor3()
    elif ndim == 4:
        return T.tensor4()
    return T.matrix()

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
        self.params = []
        self.regularizers = []
        self.constraints = []

        for layer in layers:
            self.add(layer)

    def set_previous(self, layer):
        self.layers[0].previous = layer

    def add(self, layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].set_previous(self.layers[-2])
        
        params, regularizers, constraints = layer.get_params()
        self.params += params
        self.regularizers += regularizers
        self.constraints += constraints

    def get_output(self, train=False):
        return self.layers[-1].get_output(train)

    def set_input(self):
        for l in self.layers:
            if hasattr(l, 'input'):
                ndim = l.input.ndim 
                self.layers[0].input = ndim_tensor(ndim)
                break

    def get_input(self, train=False):
        if not hasattr(self.layers[0], 'input'):
            self.set_input()
        return self.layers[0].get_input(train)

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
        return {"name":self.__class__.__name__,
            "layers":[layer.get_config() for layer in self.layers]}


class Graph(Layer):
    '''
        Implement a NN graph with arbitrary layer connections,
        arbitrary number of inputs and arbitrary number of outputs.

        Note: Graph can only be used as a layer
        (connect, input, get_input, get_output) 
        when it has exactly one input and one output.

        inherited from Layer:
            - get_params
            - get_output_mask
            - supports_masked_input
            - get_weights
            - set_weights
    '''
    def __init__(self):
        self.namespace = set() # strings
        self.nodes = {} # layer-like
        self.inputs = {} # layer-like
        self.input_order = [] # strings
        self.outputs = {} # layer-like
        self.output_order = [] # strings

        self.params = []
        self.regularizers = []
        self.constraints = []

    def set_previous(self, layer):
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise Exception('The Graph container can only be used as a layer \
                when it has exactly one input and one output.')
        self.inputs[self.input_order[0]].set_previous(layer)

    def get_input(self, train=False):
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise Exception('The Graph container can only be used as a layer \
                when it has exactly one input and one output.')
        return self.inputs[self.input_order[0]].get_input(train)

    @property
    def input(self):
        return self.get_input()

    def get_output(self, train=False):
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise Exception('The Graph container can only be used as a layer \
                when it has exactly one input and one output.')
        return self.outputs[self.output_order[0]].get_output(train)

    def add_input(self, name, ndim=2):
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)
        self.namespace.add(name)
        self.input_order.append(name)
        layer = Layer() # empty layer 
        layer.input = ndim_tensor(ndim)
        layer.input.name = name
        self.inputs[name] = layer

    def add_node(self, layer, name, input=None, inputs=[], merge_mode='concat'):
        layer.set_name(name)
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)
        if input:
            if input not in self.namespace:
                raise Exception('Unknown identifier: ' + input)
            if input in self.nodes:
                layer.set_previous(self.nodes[input])
            elif input in self.inputs:
                layer.set_previous(self.inputs[input])
        if inputs:
            to_merge = []
            for n in inputs:
                if n not in self.nodes:
                    raise Exception('Unknown identifier: ' + n)
                to_merge.append(self.nodes[n])
            merge = Merge(to_merge, mode=merge_mode)
            layer.set_previous(merge)

        self.namespace.add(name)
        self.nodes[name] = layer
        params, regularizers, constraints = layer.get_params()
        self.params += params
        self.regularizers += regularizers
        self.constraints += constraints

    def add_output(self, name, input=None, inputs=[], merge_mode='concat'):
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)
        if input:
            if input not in self.namespace:
                raise Exception('Unknown identifier: ' + input)
            if input in self.nodes:
                self.outputs[name] = self.nodes[input]
            elif input in self.inputs:
                self.ouputs[name] = self.inputs[input]
        if inputs:
            to_merge = []
            for n in inputs:
                if n not in self.nodes:
                    raise Exception('Unknown identifier: ' + n)
                to_merge.append(self.nodes[n])
            merge = Merge(to_merge, mode=merge_mode)
            self.outputs[name] = merge
        self.namespace.add(name)
        self.output_order.append(name)

    def get_config(self):
        pass



