# -*- coding: utf-8 -*-
from __future__ import absolute_import

import theano.tensor as T
from ..layers.core import Layer
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

    def connect(self, layer):
        self.layers[0].previous = layer

    def add(self, layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])
        
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
        Implement a NN graph with arbitrary layer connections.

        Small difference with 'classical' layer API:
        input, get_input, get_output are lists of tensors instead of single tensors.

        inherited from Layer:
            - get_params
            - get_output_mask
            - supports_masked_input

        not implemented:
            - connect
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

    def connect(self):
        raise Exception('The Graph container does not implement the connect method.')

    def add_input(self, name, ndim=2):
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)
        self.namespace.add(name)
        layer = Layer() # empty layer 
        layer.input = ndim_tensor(ndim)
        self.inputs[name] = layer

    def get_input(self, train=False):
        # return list of tensors
        inputs = []
        for name in self.input_order:
            inputs.append(self.inputs[name].get_output(train))
        return inputs

    @property
    def input(self):
        return self.get_input()

    def add_node(self, layer, name, input=None, inputs=[], merge_mode='concat'):
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)
        if input:
            if input not in self.namespace:
                raise Exception('Unknown identifier: ' + input)
            if input in self.nodes:
                layer.connect(self.nodes(input))
            elif input in self.inputs:
                layer.input = self.inputs[input]
        if inputs:
            to_merge = []
            for n in inputs:
                if n not in self.nodes:
                    raise Exception('Unknown identifier: ' + n)
                to_merge.append(self.nodes[n])
            merge = Merge(to_merge, merge_mode=merge_mode)
            layer.connect(merge)

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
                self.outputs[name] = self.nodes[inputs]
            elif input in self.inputs:
                layer = Layer()
                layer.input = self.inputs[input]
                self.ouputs[name] = layer
            
        if inputs:
            to_merge = []
            for n in inputs:
                if n not in self.nodes:
                    raise Exception('Unknown identifier: ' + n)
                to_merge.append(self.nodes[n])
            merge = Merge(to_merge, merge_mode=merge_mode)
            self.outputs[name] = merge.get_output()

    def get_ouput(self, train=False):
        # return list of tensors
        outputs = []
        for name in self.output_order:
            outputs.append(self.outputs[name].get_output(train))
        return ouputs

    def get_weights(self):
        pass

    def set_weights(self):
        pass



