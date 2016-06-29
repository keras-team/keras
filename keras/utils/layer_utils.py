from __future__ import print_function

from .generic_utils import get_from_module
from ..layers import *
from ..models import Model, Sequential, Graph
from .. import backend as K


def layer_from_config(config, custom_objects={}):
    '''
    # Arguments
        config: dict of the form {'class_name': str, 'config': dict}
        custom_objects: dict mapping class names (or function names)
            of custom (non-Keras) objects to class/functions

    # Returns
        Layer instance (may be Model, Sequential, Graph, Layer...)
    '''
    # Insert custom layers into globals so they can
    # be accessed by `get_from_module`.
    for cls_key in custom_objects:
        globals()[cls_key] = custom_objects[cls_key]

    class_name = config['class_name']

    if class_name == 'Sequential':
        layer_class = Sequential
    elif class_name == 'Graph':
        layer_class = Graph
    elif class_name in ['Model', 'Container']:
        layer_class = Model
    else:
        layer_class = get_from_module(class_name, globals(), 'layer',
                                      instantiate=False)
    return layer_class.from_config(config['config'])


def print_summary(layers, relevant_nodes=None, line_length=100, positions=[.33, .55, .67, 1.]):
    # line_length: total length of printed lines
    # positions: relative or absolute positions of log elements in each line
    if positions[-1] <= 1:
        positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Connected to']

    def print_row(fields, positions):
        line = ''
        for i in range(len(fields)):
            line += str(fields[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print(line)

    print('_' * line_length)
    print_row(to_display, positions)
    print('=' * line_length)

    def print_layer_summary(layer):
        try:
            output_shape = layer.output_shape
        except:
            output_shape = 'multiple'
        connections = []
        for node_index, node in enumerate(layer.inbound_nodes):
            if relevant_nodes:
                node_key = layer.name + '_ib-' + str(node_index)
                if node_key not in relevant_nodes:
                    # node is node part of the current network
                    continue
            for i in range(len(node.inbound_layers)):
                inbound_layer = node.inbound_layers[i].name
                inbound_node_index = node.node_indices[i]
                inbound_tensor_index = node.tensor_indices[i]
                connections.append(inbound_layer + '[' + str(inbound_node_index) + '][' + str(inbound_tensor_index) + ']')

        name = layer.name
        cls_name = layer.__class__.__name__
        if not connections:
            first_connection = ''
        else:
            first_connection = connections[0]
        fields = [name + ' (' + cls_name + ')', output_shape, layer.count_params(), first_connection]
        print_row(fields, positions)
        if len(connections) > 1:
            for i in range(1, len(connections)):
                fields = ['', '', '', connections[i]]
                print_row(fields, positions)

    total_params = 0
    for i in range(len(layers)):
        print_layer_summary(layers[i])
        if i == len(layers) - 1:
            print('=' * line_length)
        else:
            print('_' * line_length)
        total_params += layers[i].count_params()

    print('Total params: %s' % total_params)
    print('_' * line_length)
