from __future__ import print_function
import inspect
import numpy as np
import copy

from ..layers.advanced_activations import *
from ..layers.core import *
from ..layers.convolutional import *
from ..layers.embeddings import *
from ..layers.noise import *
from ..layers.normalization import *
from ..layers.recurrent import *
from ..layers import containers
from .. import regularizers
from .. import constraints


def container_from_config(original_layer_dict, custom_objects={}):
    layer_dict = copy.deepcopy(original_layer_dict)
    name = layer_dict.get('name')

    # Insert custom layers into globals so they can
    # be accessed by `get_from_module`.
    for cls_key in custom_objects:
        globals()[cls_key] = custom_objects[cls_key]

    if name == 'Merge':
        mode = layer_dict.get('mode')
        concat_axis = layer_dict.get('concat_axis')
        dot_axes = layer_dict.get('dot_axes')
        layers = layer_dict.get('layers')
        layer_list = []
        for layer in layers:
            init_layer = container_from_config(layer)
            layer_list.append(init_layer)
        merge_layer = Merge(layer_list, mode, concat_axis, dot_axes)
        return merge_layer

    elif name == 'Sequential':
        layers = layer_dict.get('layers')
        layer_list = []
        for layer in layers:
            init_layer = container_from_config(layer)
            layer_list.append(init_layer)
        seq_layer = containers.Sequential(layer_list)
        return seq_layer

    elif name == 'Graph':
        graph_layer = containers.Graph()
        inputs = layer_dict.get('input_config')

        for input in inputs:
            graph_layer.add_input(**input)

        nodes = layer_dict.get('node_config')
        for node in nodes:
            layer = container_from_config(layer_dict['nodes'].get(node['name']))
            node['layer'] = layer
            graph_layer.add_node(**node)

        outputs = layer_dict.get('output_config')
        for output in outputs:
            graph_layer.add_output(**output)
        return graph_layer

    elif name == 'AutoEncoder':
        kwargs = {'encoder': container_from_config(layer_dict.get('encoder_config')),
                  'decoder': container_from_config(layer_dict.get('decoder_config'))}
        for kwarg in ['output_reconstruction', 'weights']:
            if kwarg in layer_dict:
                kwargs[kwarg] = layer_dict[kwarg]
        return AutoEncoder(**kwargs)

    else:  # this is a non-topological layer (e.g. Dense, etc.)
        layer_dict.pop('name')

        for k, v in layer_dict.items():
            # a dictionary argument may be a regularizer or constraint
            if isinstance(v, dict):
                vname = v.pop('name')
                if vname in [x for x, y in inspect.getmembers(constraints, predicate=inspect.isclass)]:
                    layer_dict[k] = constraints.get(vname, v)
                elif vname in [x for x, y in inspect.getmembers(regularizers, predicate=inspect.isclass)]:
                    layer_dict[k] = regularizers.get(vname, v)
                else:
                    # not a regularizer of constraint, don't touch it
                    v['name'] = vname

        # the "name" keyword argument of layers is saved as "custom_name"
        if 'custom_name' in layer_dict:
            layer_dict['name'] = layer_dict.pop('custom_name')
        base_layer = get_layer(name, layer_dict)
        return base_layer


def model_summary(model):
    param_count = 0  # param count in the model

    def display(objects, positions):
        line = ''
        for i in range(len(objects)):
            line += str(objects[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print(line)

    def display_layer_info(layer, name, positions):
        layer_type = layer.__class__.__name__
        output_shape = layer.output_shape
        params = layer.count_params()
        to_display = ['%s (%s)' % (layer_type, name), output_shape, params]
        display(to_display, positions)

    line_length = 80  # total length of printed lines
    positions = [30, 60, 80]  # absolute positions of log elements in each line
    # header names for the different log elements
    to_display = ['Layer (name)', 'Output Shape', 'Param #']

    # for sequential models, we start by printing
    # the expect input shape
    if model.__class__.__name__ == 'Sequential':
        print('-' * line_length)
        print('Initial input shape: ' + str(model.input_shape))

    # print header
    print('-' * line_length)
    display(to_display, positions)
    print('-' * line_length)

    if model.__class__.__name__ == 'Sequential':
        for layer in model.layers:
            name = getattr(layer, 'name', 'Unnamed')
            display_layer_info(layer, name, positions)
            param_count += layer.count_params()

    elif model.__class__.__name__ == 'Graph':
        for name in model.input_order:
            layer = model.inputs[name]
            display_layer_info(layer, name, positions)

        for name in model.nodes:
            layer = model.nodes[name]
            display_layer_info(layer, name, positions)
            param_count += layer.count_params()

        for name in model.output_order:
            layer = model.outputs[name]
            display_layer_info(layer, name, positions)

    print('-' * line_length)
    print('Total params: %s' % param_count)
    print('-' * line_length)


from .generic_utils import get_from_module
def get_layer(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'layer',
                           instantiate=True, kwargs=kwargs)
