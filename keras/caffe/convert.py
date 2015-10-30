from ..layers.advanced_activations import *
from ..layers.convolutional import *
from ..layers.core import *
from ..layers.normalization import *

from ..models import Graph

import caffe_pb2 as caffe
import google.protobuf
from caffe_utils import *

import numpy as np


def caffe_to_keras(prototext=None, caffemodel=None, phase='train'):
        '''
            prototext: model description file in caffe
            caffemodel: stored weights file
            phase: train or test
            Usage:
                model_data = caffe_to_keras(prototext='VGG16.prototxt',
                                            caffemodel='VGG16_700iter.caffemodel')
                graph = model_data.get('network') # loaded with with weights is caffemodel is provided, else randomly initialized
                inputs = model_data.get('inputs')
                outputs = model_data.get('outputs')
                weights = model_data.get('weights') # useful for embedding networks
        '''
        model_data = {}

        if prototext:
            config = caffe.NetParameter()
            google.protobuf.text_format.Merge(open(prototext).read(), config)
        elif caffemodel:
            config = caffe.NetParameter()
            config.MergeFromString(open(caffemodel, 'rb').read())

        if len(config.layers) != 0:
            # prototext V1
            layers = config.layers[:]
        elif len(config.layer) != 0:
            # prototext V2
            layers = config.layer[:]
        else:
            raise Exception('could not load any layers from prototext')

        model = model_from_config(layers,
                                  0 if phase == 'train' else 1,
                                  config.input_dim[1:])
        return model


def flip90(W):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = np.rot90(W[i, j], 2)
    return W


def model_from_config(layers, phase, input_dim):
    '''
        layers: a list of all the layers in the model
        phase: parameter to specify which network to extract: training or test
        input_dim: input dimensions of the configuration (if in model is in deploy mode)
    '''
    # DEPLOY MODE: first layer is computation
    # NON DEPLOY MODE: first layer is data input
    if input_dim == []:
        in_deploy_mode = False
    else:
        in_deploy_mode = True

    network = parse_network(layers, phase)  # obtain the nodes that make up the graph
    if len(network) == 0:
        raise Exception('failed to construct network from the prototext')
    network_inputs = get_inputs(network)  # inputs of the network - 'in-order' is zero
    network_outputs = get_outputs(network)  # outputs of the network - 'out-order' is zero
    network = remove_label_paths(layers, network, network_inputs, network_outputs)  # path from input to loss layers removed
    reverse_network = reverse(network)  # stores the 'input' for each layer

    model = Graph()

    output_dim = {}

    # add input nodes
    for network_input in network_inputs:
        name = layers[network_input].name
        model.add_input(name=name, ndim=4)
        if in_deploy_mode:
            output_dim[name] = input_dim
        else:
            data_dim = get_data_dim(layers[network_input])
            output_dim[name] = data_dim

    # parse all the layers and build equivalent keras graph
    for layer_nb in network:
        layer = layers[layer_nb]
        name = layer.name
        type_of_layer = layer_type(layer)

        if is_data_input(layer):
            # not including data layers
            continue

        if layer_nb in network_inputs:
            # in deploy mode since data layers can't reach this line and condition holds.
            new_name = 'graph_input_' + name
            input_layer_names = [name]
            name = new_name
        else:
            input_layers = reverse_network[layer_nb]  # inputs to current layer, in the form of layer numbers
            input_layer_names = []  # list of strings identifying the input layer
            for input_layer in input_layers:
                if input_layer in network_inputs and in_deploy_mode:
                    input_layer_names.append('graph_input_' + layers[input_layer].name)
                else:
                    input_layer_names.append(layers[input_layer].name)

        if layer_nb in network_outputs:
            # outputs nodes are marked with 'output_' prefix from which output is derived later in 'add_output'
            name = 'output_' + name

        layer_input_dims = []
        for input_layer_name in input_layer_names:
            layer_input_dims.append(output_dim[input_layer_name])

        if len(input_layer_names) == 1:
            # single input. since concatenation is explicit,
            # all layers can be thought of single input layers
            # (except loss layers, which is handled anyway)
            input_layer_name = input_layer_names[0]
            layer_input_dim = layer_input_dims[0]

        if type_of_layer == 'concat':
            # emulation of just concatenation
            axis = layer.concat_param.axis  # 0 for batch, 1 for stack
            model.add_node(Activation('linear'), name=name, inputs=input_layer_names, concat_axis=axis)

            layer_output_dim = [layer_input_dims[0][0], layer_input_dims[0][1], layer_input_dims[0][2]]
            for dim in layer_input_dims[1:]:
                layer_output_dim[0] += dim[0]

        elif type_of_layer == 'convolution':
            if layer.blobs:
                blobs = layer.blobs
                nb_filter, temp_stack_size, nb_col, nb_row = blobs[0].num, blobs[0].channels, blobs[0].height, blobs[0].width

                # model parallel network
                group = layer.convolution_param.group
                stack_size = temp_stack_size * group

                # maybe not all synapses are existant
                weights_p = np.zeros((nb_filter, stack_size, nb_col, nb_row))
                weights_b = np.array(blobs[1].data)

                chunk_data_size = len(blobs[0].data) // group
                stacks_size_per_chunk = stack_size // group
                nb_filter_per_chunk = nb_filter // group

                for i in range(group):
                    chunk_weights = weights_p[i * nb_filter_per_chunk: (i + 1) * nb_filter_per_chunk,
                                              i * stacks_size_per_chunk: (i + 1) * stacks_size_per_chunk, :, :]
                    chunk_weights[:] = np.array(blobs[0].data[i * chunk_data_size:(i + 1) * chunk_data_size]).reshape(chunk_weights.shape)

                weights = [flip90(weights_p.astype(dtype=np.float32)), weights_b.astype(dtype=np.float32)]
            else:
                weights = None

            nb_col = (layer.convolution_param.kernel_size or [layer.convolution_param.kernel_h])[0]
            nb_row = (layer.convolution_param.kernel_size or [layer.convolution_param.kernel_w])[0]
            nb_filter = layer.convolution_param.num_output

            stride_h = (layer.convolution_param.stride or [layer.convolution_param.stride_h])[0] or 1
            stride_w = (layer.convolution_param.stride or [layer.convolution_param.stride_w])[0] or 1

            pad_h = (layer.convolution_param.pad or [layer.convolution_param.pad_h])[0]
            pad_w = (layer.convolution_param.pad or [layer.convolution_param.pad_w])[0]

            if pad_h + pad_w > 0:
                model.add_node(ZeroPadding2D(pad=(pad_h, pad_w)), name=name + '_zeropadding', input=input_layer_name)
                input_layer_name = name + '_zeropadding'

            stack_size = layer_input_dim[0]

            model.add_node(Convolution2D(nb_filter, stack_size, nb_row, nb_col, subsample=(stride_h, stride_w), weights=weights), name=name, input=input_layer_name)

            layer_output_dim_padding = [layer_input_dim[0], layer_input_dim[1] + 2 * pad_h, layer_input_dim[2] + 2 * pad_w]
            layer_output_dim = [nb_filter, (layer_output_dim_padding[1] - nb_row) / stride_h + 1, (layer_output_dim_padding[2] - nb_col) / stride_w + 1]

        elif type_of_layer == 'dropout':
            prob = layer.dropout_param.dropout_ratio
            model.add_node(Dropout(prob), name=name, input=input_layer_name)
            layer_output_dim = layer_input_dim

        elif type_of_layer == 'flatten':
            model.add_node(Flatten(), name=name, input=input_layer_name)
            layer_output_dim = np.prod(layer_input_dim)

        elif type_of_layer == 'innerproduct':
            if layer.blobs:
                blobs = layer.blobs
                nb_filter, stack_size, nb_col, nb_row = blobs[0].num, blobs[0].channels, blobs[0].height, blobs[0].width

                weights_p = np.array(blobs[0].data).reshape(nb_filter, stack_size, nb_col, nb_row)[0, 0, :, :].T
                weights_b = np.array(blobs[1].data)
                weights = [weights_p.astype(dtype=np.float32), weights_b.astype(dtype=np.float32)]
            else:
                weights = None

            layer_output_dim = layer.inner_product_param.num_output

            if len(layer_input_dim) > 1:
                model.add_node(Flatten(), name=name + '_flatten', input=input_layer_name)
                layer_input_dim = [np.prod(layer_input_dim)]
                model.add_node(Dense(layer_input_dim[0], layer_output_dim, weights=weights), name=name, input=name + '_flatten')
            else:
                model.add_node(Dense(layer_input_dim[0], layer_output_dim), name=name, input=input_layer_name)

            layer_output_dim = [layer_output_dim]

        elif type_of_layer == 'lrn':
            alpha = layer.lrn_param.alpha
            k = layer.lrn_param.k
            beta = layer.lrn_param.beta
            n = layer.lrn_param.local_size

            model.add_node(LRN2D(alpha=alpha, k=k, beta=beta, n=n), name=name, input=input_layer_name)

            layer_output_dim = layer_input_dim

        elif type_of_layer == 'pooling':
            kernel_h = layer.pooling_param.kernel_size or layer.pooling_param.kernel_h
            kernel_w = layer.pooling_param.kernel_size or layer.pooling_param.kernel_w

            stride_h = layer.pooling_param.stride or layer.pooling_param.stride_h or 1
            stride_w = layer.pooling_param.stride or layer.pooling_param.stride_w or 1

            pad_h = layer.pooling_param.pad or layer.pooling_param.pad_h
            pad_w = layer.pooling_param.pad or layer.pooling_param.pad_w

            if pad_h + pad_w > 0:
                model.add_node(ZeroPadding2D(pad=(pad_h, pad_w)), name=name + '_zeropadding', input=input_layer_name)
                input_layer_name = name + '_zeropadding'

            model.add_node(MaxPooling2D(poolsize=(kernel_h, kernel_w), stride=(stride_h, stride_w)), name=name, input=input_layer_name)

            layer_output_dim_padding = [layer_input_dim[0],
                                        layer_input_dim[1] + 2 * pad_h,
                                        layer_input_dim[2] + 2 * pad_w]
            layer_output_dim = [layer_output_dim_padding[0],
                                (layer_output_dim_padding[1] - kernel_h) / stride_h + 1,
                                (layer_output_dim_padding[2] - kernel_w) / stride_w + 1]

        elif type_of_layer == 'relu':
            model.add_node(Activation('relu'), name=name, input=input_layer_name)
            layer_output_dim = layer_input_dim

        elif type_of_layer == 'sigmoid':
            model.add_node(Activation('sigmoid'), name=name, input=input_layer_name)
            layer_output_dim = layer_input_dim

        elif type_of_layer == 'softmax' or type_of_layer == 'softmaxwithloss':
            model.add_node(Activation('softmax'), name=name, input=input_layer_name)
            layer_output_dim = layer_input_dim

        elif type_of_layer == 'split':
            model.add_node(Activation('linear'), name=name, inputs=input_layer_name)
            layer_output_dim = layer_input_dim

        elif type_of_layer == 'tanh':
            model.add_node(Activation('tanh'), name=name, input=input_layer_name)
            layer_output_dim = layer_input_dim

        else:
            raise RuntimeError('layer type', type_of_layer, 'used in this model is not currently supported')

        output_dim[name] = layer_output_dim

    for network_output in network_outputs:
        input_layer_name = 'output_' + layers[network_output].name
        model.add_output(name=layers[network_output].name, input=input_layer_name)
    return model
