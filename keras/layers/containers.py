# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from collections import OrderedDict
from .. import backend as K
from ..layers.core import Layer, Merge, Siamese, SiameseHead
from six.moves import range


class Sequential(Layer):
    '''The Sequential container is a linear stack of layers.
    Apart from the `add` methods and the `layers` constructor argument,
    the API is identical to that of the `Layer` class.

    This class is also the basis for the `keras.models.Sequential` model.

    # Arguments
        layers: list of layers to be added to the container.
    '''
    def __init__(self, layers=[]):
        self.layers = []
        self.layer_cache = {}
        self.shape_cache = {}
        for layer in layers:
            self.add(layer)
        self._cache_enabled = True

    @property
    def cache_enabled(self):
        return self._cache_enabled

    @cache_enabled.setter
    def cache_enabled(self, value):
        self._cache_enabled = value
        for l in self.layers:
            l.cache_enabled = value

    @property
    def layer_cache(self):
        return super(Sequential, self).layer_cache

    @layer_cache.setter
    def layer_cache(self, value):
        self._layer_cache = value
        for layer in self.layers:
            layer.layer_cache = self._layer_cache

    @property
    def shape_cache(self):
        return super(Sequential, self).shape_cache

    @shape_cache.setter
    def shape_cache(self, value):
        self._shape_cache = value
        for layer in self.layers:
            layer.shape_cache = self._shape_cache

    def set_previous(self, layer, reset_weights=True):
        self.layers[0].set_previous(layer, reset_weights)

    def clear_previous(self, reset_weights=True):
        self.layers[0].clear_previous(reset_weights)

    def add(self, layer):
        layer.layer_cache = self.layer_cache
        layer.shape_cache = self.shape_cache
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].set_previous(self.layers[-2])
            if not hasattr(self.layers[0], 'input'):
                self.set_input()

    @property
    def trainable_weights(self):
        weights = []
        for l in self.layers:
            if l.trainable:
                weights += l.get_params()[0]
        return weights

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
    def learning_rate_multipliers(self):
        learning_rate_multipliers = []
        for l in self.layers:
            if l.trainable:
                learning_rate_multipliers += l.get_params()[3]
        return learning_rate_multipliers

    @property
    def updates(self):
        updates = []
        for l in self.layers:
            if l.trainable:
                updates += l.get_params()[4]
        return updates

    @property
    def state_updates(self):
        """
        Return the `updates` from all layers in the sequence that are
        stateful.  This is useful for separating _training_ updates and
        _prediction_ updates for when we need to update a layers internal state
        during a stateful prediction.
        """
        state_updates = []
        for l in self.layers:
            if getattr(l, 'stateful', False):
                state_updates += l.get_params()[4]
        return state_updates

    def reset_states(self):
        for l in self.layers:
            if hasattr(l, 'reset_states') and getattr(l, 'stateful', False):
                l.reset_states()

    @property
    def output_shape(self):
        return self.layers[-1].output_shape

    def get_output(self, train=False):
        return self.layers[-1].get_output(train)

    def set_input(self):
        for l in self.layers:
            if hasattr(l, 'input'):
                ndim = K.ndim(l.input)
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
        for layer in self.layers:
            nb_param = len(layer.get_weights())
            layer.set_weights(weights[:nb_param])
            weights = weights[nb_param:]

    def get_config(self):
        return {'name': self.__class__.__name__,
                'layers': [layer.get_config() for layer in self.layers]}

    def count_params(self):
        return sum([layer.count_params() for layer in self.layers])


class Graph(Layer):
    '''Implement a NN graph with arbitrary layer connections,
    arbitrary number of inputs and arbitrary number of outputs.

    This class is also the basis for the `keras.models.Graph` model.

    Note: `Graph` can only be used as a layer
    (connect, input, get_input, get_output)
    when it has exactly one input and one output.
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
        self.layer_cache = {}
        self.shape_cache = {}
        self._cache_enabled = True

    def __call__(self, X, mask=None, train=False):
        if type(X) != dict:
            return super(Graph, self).__call__(X, mask, train)
        else:
            # turn off layer cache temporarily
            tmp_cache_enabled = self.cache_enabled
            self.cache_enabled = False
            # create a temporary layer for each input
            tmp_previous = {}
            for name, input in self.inputs.items():
                layer = Layer(batch_input_shape=input.input_shape)
                layer.input = X[name]
                if hasattr(self, 'get_input_mask'):
                    layer.get_input_mask = lambda _: mask[name]
                # set temporary previous
                if hasattr(input, 'previous'):
                    tmp_previous[name] = input.previous
                input.set_previous(layer, False)
            Y = self.get_output(train=train)
            # return previous to what it was
            for name, input in self.inputs.items():
                if name in tmp_previous:
                    input.set_previous(tmp_previous[name], False)
                else:
                    input.clear_previous(False)
            self.cache_enabled = tmp_cache_enabled
            return Y

    @property
    def cache_enabled(self):
        return self._cache_enabled

    @cache_enabled.setter
    def cache_enabled(self, value):
        self._cache_enabled = value
        for l in self.nodes.values():
            l.cache_enabled = value
        for l in self.inputs.values():
            l.cache_enabled = value

    @property
    def layer_cache(self):
        return super(Graph, self).layer_cache

    @layer_cache.setter
    def layer_cache(self, value):
        self._layer_cache = value
        for layer in self.nodes.values():
            layer.layer_cache = self._layer_cache
        for layer in self.inputs.values():
            layer.layer_cache = self._layer_cache

    @property
    def shape_cache(self):
        return super(Graph, self).shape_cache

    @shape_cache.setter
    def shape_cache(self, value):
        self._shape_cache = value
        for layer in self.nodes.values():
            layer.shape_cache = self._shape_cache
        for layer in self.inputs.values():
            layer.shape_cache = self._shape_cache

    @property
    def nb_input(self):
        return len(self.inputs)

    @property
    def nb_output(self):
        return len(self.outputs)

    @property
    def trainable_weights(self):
        weights = []
        for l in self.nodes.values():
            if l.trainable:
                weights += l.get_params()[0]
        return weights

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
    def learning_rate_multipliers(self):
        learning_rate_multipliers = []
        for l in self.nodes.values():
            if l.trainable:
                learning_rate_multipliers += l.get_params()[3]
        return learning_rate_multipliers

    @property
    def updates(self):
        updates = []
        for l in self.nodes.values():
            if l.trainable:
                updates += l.get_params()[4]
        return updates

    @property
    def state_updates(self):
        """
        Return the `updates` from all nodes in that graph for nodes that are
        stateful.  This is useful for separating _training_ updates and
        _prediction_ updates for when we need to update a layers internal state
        during a stateful prediction.
        """
        state_updates = []
        for l in self.nodes.values():
            if getattr(l, 'stateful', False):
                state_updates += l.get_params()[4]
        return state_updates

    def reset_states(self):
        for l in self.nodes.values():
            if hasattr(l, 'reset_states') and getattr(l, 'stateful', False):
                l.reset_states()

    def set_previous(self, layer, connection_map={}, reset_weights=True):
        if self.nb_input != layer.nb_output:
            raise Exception('Cannot connect layers: '
                            'input count does not match output count.')
        if self.nb_input == 1:
            self.inputs[self.input_order[0]].set_previous(layer, reset_weights)
        else:
            if not connection_map:
                raise Exception('Cannot attach multi-input layer: '
                                'no connection_map provided.')
            for k, v in connection_map.items():
                if k in self.inputs and v in layer.outputs:
                    self.inputs[k].set_previous(layer.outputs[v], reset_weights)
                else:
                    raise Exception('Invalid connection map.')

    def clear_previous(self, reset_weights=True):
        for k in self.inputs.values():
            k.clear_previous(reset_weights)

    @property
    def input_shape(self):
        if self.nb_input == 1:
            # return tuple
            return self.inputs[self.input_order[0]].input_shape
        else:
            # return dictionary mapping input names to shape tuples
            return dict([(k, v.input_shape) for k, v in self.inputs.items()])

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

    def add_input(self, name, input_shape=None,
                  batch_input_shape=None, dtype='float'):
        '''Add an input to the graph.

        # Arguments:
            name: string. The name of the new input. Must be unique in the graph.
            input_shape: a tuple of integers, the expected shape of the input samples.
                Does not include the batch size.
            batch_input_shape: a tuple of integers, the expected shape of the
                whole input batch, including the batch size.
            dtype: 'float' or 'int'.
        '''
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)
        self.namespace.add(name)
        self.input_order.append(name)
        layer = Layer(name=name)  # empty layer
        if input_shape:
            layer.set_input_shape((None,) + tuple(input_shape))
        elif batch_input_shape:
            layer.set_input_shape(batch_input_shape)
        if dtype == 'float':
            layer.input = K.placeholder(shape=layer.input_shape, name=name)
        else:
            if (input_shape and len(input_shape) == 1) or (batch_input_shape and len(batch_input_shape) == 2):
                layer.input = K.placeholder(shape=layer.input_shape,
                                            dtype='int32',
                                            name=name)
            else:
                raise Exception('Type "int" can only be used with ndim==2 (Embedding).')
        self.inputs[name] = layer
        config = {'name': name, 'dtype': dtype}
        if batch_input_shape:
            config['batch_input_shape'] = batch_input_shape
        else:
            config['input_shape'] = input_shape
        self.input_config.append(config)

    def add_node(self, layer, name, input=None, inputs=[],
                 merge_mode='concat', concat_axis=-1, dot_axes=-1,
                 create_output=False):
        '''Add a node in the graph. It can be connected to multiple
        inputs, which will first be merged into one tensor
        according to the mode specified.

        # Arguments
            layer: the layer at the node.
            name: name for the node.
            input: when connecting the layer to a single input,
                this is the name of the incoming node.
            inputs: when connecting the layer to multiple inputs,
                this is a list of names of incoming nodes.
            merge_mode: one of {concat, sum, dot, ave, mul}
            concat_axis: when `merge_mode=='concat'`, this is the
                input concatenation axis.
            dot_axes: when `merge_mode='dot'`, this is the contraction axes
                specification; see the `Merge layer for details.
            create_output: boolean. Set this to `True` if you want the output
                of your node to be an output of the graph.
        '''
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)
        layer.name = name
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
        layer.layer_cache = self.layer_cache
        layer.shape_cache = self.shape_cache
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
        '''Used to share a same layer across multiple nodes.

        Supposed, for instance, that you want to apply one same `Dense`
        layer after to the output of two different nodes.
        You can then add the `Dense` layer as a shared node.

        # Arguments
            layer: The layer to be shared across multiple inputs
            name: Name of the shared node
            inputs: List of names of input nodes
            merge_mode: Same meaning as `merge_mode` argument of `add_node()`
            concat_axis: Same meaning as `concat_axis` argument of `add_node()`
            dot_axes: Same meaning as `dot_axes` argument of `add_node()`
            outputs: Used when `merge_mode=None`. Names for the output nodes.
            create_output: Same meaning as `create_output` argument of `add_node()`.
        '''
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)
        for o in outputs:
            if o in self.namespace:
                raise Exception('Duplicate node identifier: ' + o)
        if merge_mode:
            if merge_mode not in {'sum', 'ave', 'mul', 'dot', 'cos', 'concat', 'join'}:
                raise Exception('Invalid merge mode')
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
        s = Siamese(layer, layers, merge_mode,
                    concat_axis=concat_axis,
                    dot_axes=dot_axes,
                    is_graph=True)
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
                sh.name = sh_name
                self.namespace.add(sh_name)
                self.nodes[sh_name] = sh
                self.node_config.append({'name': sh_name,
                                         'inputs': [name],
                                         'create_output': create_output})
                if create_output:
                    self.add_output(sh_name, input=sh_name)

        if create_output and merge_mode:
            if merge_mode == 'join':
                raise Exception('Output can not be of type OrderedDict')
            self.add_output(name, input=name)

    def add_output(self, name, input=None, inputs=[],
                   merge_mode='concat', concat_axis=-1, dot_axes=-1):
        '''Add an output to the graph.

        This output can merge several node outputs into a single output.

        # Arguments
            name: name of the output.
            input: when connecting the layer to a single input,
                this is the name of the incoming node.
            inputs: when connecting the layer to multiple inputs,
                this is a list of names of incoming nodes.
            merge_mode: one of {concat, sum, dot, ave, mul}
            concat_axis: when `merge_mode=='concat'`, this is the
                input concatenation axis.
            dot_axes: when `merge_mode='dot'`, this is the contraction axes
                specification; see the `Merge layer for details.
        '''
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
        return {'name': self.__class__.__name__,
                'input_config': self.input_config,
                'node_config': self.node_config,
                'output_config': self.output_config,
                'input_order': self.input_order,
                'output_order': self.output_order,
                'nodes': dict([(c['name'], self.nodes[c['name']].get_config()) for c in self.node_config])}

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
