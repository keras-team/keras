from __future__ import print_function
import inspect

from .generic_utils import get_custom_objects
from .generic_utils import deserialize_keras_object
from .conv_utils import convert_kernel
py_sum = sum
from ..layers import *
from ..models import Model, Sequential
from .. import backend as K


def layer_from_config(config, custom_objects=None):
    """Instantiate a layer from a config dictionary.

    # Arguments
        config: dict of the form {'class_name': str, 'config': dict}
        custom_objects: dict mapping class names (or function names)
            of custom (non-Keras) objects to class/functions

    # Returns
        Layer instance (may be Model, Sequential, Layer...)

    # TODO: rename to "deserialize" and move to "layers.__init__.py"
    """
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='layer')


def print_summary(layers, relevant_nodes=None,
                  line_length=100, positions=None):
    """Prints a summary of a layer.

    # Arguments
        layers: list of layers to print summaries of
        relevant_nodes: list of relevant nodes
        line_length: total length of printed lines
        positions: relative or absolute positions of log elements in each line.
            If not provided, defaults to `[.33, .55, .67, 1.]`.

    # TODO: don't print connectivity for sequential models
    maybe change API to accept a model instance
    """
    positions = positions or [.33, .55, .67, 1.]
    if positions[-1] <= 1:
        positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Connected to']

    def print_row(fields, positions):
        line = ''
        for i in range(len(fields)):
            if i > 0:
                line = line[:-1] + ' '
            line += str(fields[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print(line)

    print('_' * line_length)
    print_row(to_display, positions)
    print('=' * line_length)

    def print_layer_summary(layer):
        """Prints a summary for a single layer.

        # Arguments
            layer: target layer.
        """
        try:
            output_shape = layer.output_shape
        except AttributeError:
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

    for i in range(len(layers)):
        print_layer_summary(layers[i])
        if i == len(layers) - 1:
            print('=' * line_length)
        else:
            print('_' * line_length)

    trainable_count, non_trainable_count = count_total_params(layers, layer_set=None)

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    print('_' * line_length)


def count_total_params(layers, layer_set=None):
    """Counts the number of parameters in a list of layers.

    # Arguments
        layers: list of layers.
        layer_set: set of layers already seen
            (so that we don't count their weights twice).

    # Returns
        A tuple (count of trainable weights, count of non-trainable weights.)
    """
    if layer_set is None:
        layer_set = set()
    trainable_count = 0
    non_trainable_count = 0
    for layer in layers:
        if layer in layer_set:
            continue
        layer_set.add(layer)
        if isinstance(layer, (Model, Sequential)):
            t, nt = count_total_params(layer.layers, layer_set)
            trainable_count += t
            non_trainable_count += nt
        else:
            trainable_count += py_sum([K.count_params(p) for p in layer.trainable_weights])
            non_trainable_count += py_sum([K.count_params(p) for p in layer.non_trainable_weights])
    return trainable_count, non_trainable_count


def convert_all_kernels_in_model(model):
    """Converts all convolution kernels in a model from Theano to TensorFlow.

    Also works from TensorFlow to Theano.

    # Arguments
        model: target model for the conversion.
    """
    # Note: SeparableConvolution not included
    # since only supported by TF.
    conv_classes = {
        'Convolution1D',
        'Convolution2D',
        'Convolution3D',
        'AtrousConvolution2D',
        'Deconvolution2D',
    }
    to_assign = []
    for layer in model.layers:
        if layer.__class__.__name__ in conv_classes:
            original_w = K.get_value(layer.W)
            converted_w = convert_kernel(original_w)
            to_assign.append((layer.W, converted_w))
    K.batch_set_value(to_assign)
