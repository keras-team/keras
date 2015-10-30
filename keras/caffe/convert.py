from ..layers.advanced_activations import *
from ..layers.convolutional import *
from ..layers.core import *
from ..layers.normalization import *

from ..models import Graph

import caffe_pb2 as caffe
import google.protobuf
from caffe_utils import *

import numpy as np


def caffe_to_keras(prototext, caffemodel, phase='train'):
        '''
            Converts a Caffe Graph into a Keras Graph
            prototext: model description file in caffe
            caffemodel: stored weights file
            phase: train or test
            
            Usage:
                model = caffe_to_keras('VGG16.prototxt', 'VGG16_700iter.caffemodel')
        '''

        config = caffe.NetParameter()
        google.protobuf.text_format.Merge(open(prototext).read(), config)

        if len(config.layers) != 0:
            layers = config.layers[:]   # prototext V1
        elif len(config.layer) != 0:
            layers = config.layer[:]    # prototext V2
        else:
            raise Exception('could not load any layers from prototext')


        model = create_model(layers,
                                  0 if phase == 'train' else 1,
                                  config.input_dim[1:])
        
        params = caffe.NetParameter()
        params.MergeFromString(open(caffemodel, 'rb').read())
        
        if len(params.layers) != 0:
            param_layers = params.layers[:]    # V1
        elif len(params.layer) != 0:
            param_layers = params.layer[:]     # V2
        else:
            raise Exception('could not load any layers from caffemodel')

        weights = convert_weights(param_layers)

        load_weights(model, weights)

        return model


def create_model(layers, phase, input_dim):
    '''
        layers: 
            a list of all the layers in the model
        phase: 
            parameter to specify which network to extract: training or test
        input_dim: 
            `input dimensions of the configuration (if in model is in deploy mode)
    '''
    # DEPLOY MODE: first layer is computation (conv, dense, etc..)
    # NON DEPLOY MODE: first layer is data input
    if input_dim == []:
        in_deploy_mode = False
    else:
        in_deploy_mode = True

    # obtain the nodes that make up the graph
    # returned in linked list (not matrix) representation (dictionary here)
    network = parse_network(layers, phase)  
    
    if len(network) == 0:
        raise Exception('failed to construct network from the prototext')

    # inputs of the network - 'in-order' is zero
    inputs = get_inputs(network)
    # outputs of the network - 'out-order' is zero
    network_outputs = get_outputs(network)
    
    # path from input to loss layers (label) removed
    network = remove_label_paths(layers, network, inputs, network_outputs)

    # while network contains what nodes follow a perticular node.
    # we need to know what feeds a given node, hence reverse it.
    inputs_to = reverse(network)

    model = Graph()

    # add input nodes
    for network_input in inputs:
        name = layers[network_input].name
        if in_deploy_mode:
            dim = input_dim
        else:
            dim = get_data_dim(layers[network_input])
        model.add_input(name=name, input_shape=dim)
       
    # parse all the layers and build equivalent keras graph
    for layer_nb in network:
        layer = layers[layer_nb]
        name = layer.name
        type_of_layer = layer_type(layer)

        if is_data_input(layer):
            continue

        # DEPLOY MODE: this layer takes in input as per the model. 
        # No data layer preceding it.
        # we use the actual name of the layer in 'add_input'
        # so users can see the first layer as input
        # Here we prefix the layer name by 'input_'
        # OR, in case, if this layer takes input from a layer 
        # that is now pefixed with 'input_', we need to adjust input layer names 
        if layer_nb in inputs:
            input_layer_names = [name]
            name = 'input_' + name
        else:
            input_layers = inputs_to[layer_nb]
            input_layer_names = []
            for input_layer in input_layers:
                if input_layer in inputs and in_deploy_mode:
                    input_layer_names.append('input_' + layers[input_layer].name)
                else:
                    input_layer_names.append(layers[input_layer].name)

        # outputs nodes are marked with 'output_' prefix 
        # from which output is derived later in 'add_output'
        if layer_nb in network_outputs:
            name = 'output_' + name

        # since concatenation is explicit,
        # all other layers can be thought of single input layers
        # (except loss layers, which is handled before, while creating the DAG)
        if len(input_layer_names) == 1:
            input_layer_name = input_layer_names[0]
           
        if type_of_layer == 'concat':
            # emulation of just concatenation
            axis = layer.concat_param.axis  # 0 for batch, 1 for stack
            model.add_node(Activation('linear'), name=name, inputs=input_layer_names, concat_axis=axis)

        elif type_of_layer == 'convolution':
            nb_filter = layer.convolution_param.num_output
            nb_col = (layer.convolution_param.kernel_size or [layer.convolution_param.kernel_h])[0]
            nb_row = (layer.convolution_param.kernel_size or [layer.convolution_param.kernel_w])[0]
            stride_h = (layer.convolution_param.stride or [layer.convolution_param.stride_h])[0] or 1
            stride_w = (layer.convolution_param.stride or [layer.convolution_param.stride_w])[0] or 1
            pad_h = (layer.convolution_param.pad or [layer.convolution_param.pad_h])[0]
            pad_w = (layer.convolution_param.pad or [layer.convolution_param.pad_w])[0]

            if pad_h + pad_w > 0:
                model.add_node(ZeroPadding2D(padding=(pad_h, pad_w)), name=name + '_zeropadding', input=input_layer_name)
                input_layer_name = name + '_zeropadding'

            model.add_node(Convolution2D(nb_filter, nb_row, nb_col, subsample=(stride_h, stride_w)), name=name, input=input_layer_name)

        elif type_of_layer == 'dropout':
            prob = layer.dropout_param.dropout_ratio
            model.add_node(Dropout(prob), name=name, input=input_layer_name)

        elif type_of_layer == 'flatten':
            model.add_node(Flatten(), name=name, input=input_layer_name)

        elif type_of_layer == 'innerproduct':
            output_dim = layer.inner_product_param.num_output
            if len(model.nodes[input_layer_name].output_shape[1:]) > 1:
                model.add_node(Flatten(), name=name + '_flatten', input=input_layer_name)
                input_layer_name = name + '_flatten'
            
            model.add_node(Dense(output_dim), name=name, input=input_layer_name)
           
        elif type_of_layer == 'lrn':
            alpha = layer.lrn_param.alpha
            k = layer.lrn_param.k
            beta = layer.lrn_param.beta
            n = layer.lrn_param.local_size

            model.add_node(LRN2D(alpha=alpha, k=k, beta=beta, n=n), name=name, input=input_layer_name)

        elif type_of_layer == 'pooling':
            kernel_h = layer.pooling_param.kernel_size or layer.pooling_param.kernel_h
            kernel_w = layer.pooling_param.kernel_size or layer.pooling_param.kernel_w

            # caffe defaults to 1, hence both of the params can be zero. 'or 1'
            stride_h = layer.pooling_param.stride or layer.pooling_param.stride_h or 1
            stride_w = layer.pooling_param.stride or layer.pooling_param.stride_w or 1

            pad_h = layer.pooling_param.pad or layer.pooling_param.pad_h
            pad_w = layer.pooling_param.pad or layer.pooling_param.pad_w

            if pad_h + pad_w > 0:
                model.add_node(ZeroPadding2D(padding=(pad_h, pad_w)), name=name + '_zeropadding', input=input_layer_name)
                input_layer_name = name + '_zeropadding'

            model.add_node(MaxPooling2D(pool_size=(kernel_h, kernel_w), stride=(stride_h, stride_w)), name=name, input=input_layer_name)

        elif type_of_layer == 'relu':
            model.add_node(Activation('relu'), name=name, input=input_layer_name)

        elif type_of_layer == 'sigmoid':
            model.add_node(Activation('sigmoid'), name=name, input=input_layer_name)

        elif type_of_layer == 'softmax' or type_of_layer == 'softmaxwithloss':
            model.add_node(Activation('softmax'), name=name, input=input_layer_name)

        elif type_of_layer == 'split':
            model.add_node(Activation('linear'), name=name, inputs=input_layer_name)

        elif type_of_layer == 'tanh':
            model.add_node(Activation('tanh'), name=name, input=input_layer_name)

        else:
            raise RuntimeError('layer type', type_of_layer, 'used in this model is not currently supported')

    # add the output nodes. The actual nodes are in interior with prefix 'output_'
    for network_output in network_outputs:
        input_layer_name = 'output_' + layers[network_output].name
        model.add_output(name=layers[network_output].name, input=input_layer_name)
    
    return model


def rot90(W):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = np.rot90(W[i, j], 2)
    return W


def convert_weights(param_layers):
    weights = {}

    for layer in param_layers:
        typ = layer_type(layer)
        if typ == 'innerproduct':
            blobs = layer.blobs
            nb_filter = blobs[0].num
            stack_size = blobs[0].channels
            nb_col = blobs[0].height
            nb_row = blobs[0].width

            weights_p = np.array(blobs[0].data).reshape(nb_filter, stack_size, nb_col, nb_row)[0, 0, :, :]
            weights_p = weights_p.T     # need to swapaxes here, hence transpose. See comment in conv
            weights_b = np.array(blobs[1].data)
            layer_weights = [weights_p.astype(dtype=np.float32), weights_b.astype(dtype=np.float32)]
            
            weights[layer.name] = layer_weights

        elif typ == 'convolution':
            blobs = layer.blobs
            nb_filter = blobs[0].num
            temp_stack_size = blobs[0].channels
            nb_col = blobs[0].height
            nb_row = blobs[0].width

            # NOTE: on model parallel networks
            # if group is > 1, that means the conv filters are split up
            # into a number of 'groups' and each group lies on a seperate GPU.
            # Each group only acts on the select group of outputs from pervious layer
            # that was in the same GPU (not the entire stack)
            # Here, we add zeros to simulate the same effect
            # This was famously used in AlexNet and few other models from 2012-14

            group = layer.convolution_param.group
            stack_size = temp_stack_size * group

            weights_p = np.zeros((nb_filter, stack_size, nb_col, nb_row))
            weights_b = np.array(blobs[1].data)

            group_data_size = len(blobs[0].data) // group
            stacks_size_per_group = stack_size // group
            nb_filter_per_group = nb_filter // group

            for i in range(group):
                group_weights = weights_p[i * nb_filter_per_group: (i + 1) * nb_filter_per_group,
                                          i * stacks_size_per_group: (i + 1) * stacks_size_per_group, :, :]
                group_weights[:] = np.array(blobs[0].data[i * group_data_size:
                                            (i + 1) * group_data_size]).reshape(group_weights.shape)

            # caffe, unlike theano, does correlation not convolution. We need to flip the weights 180 deg
            weights_p = rot90(weights_p)
            layer_weights = [weights_p.astype(dtype=np.float32), weights_b.astype(dtype=np.float32)]

            weights[layer.name] = layer_weights

    return weights


def load_weights(model, weights):
    for layer in model.nodes:
        if weights.has_key(layer):
            model.nodes[layer].set_weights(weights[layer])


