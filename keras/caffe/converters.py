from ..layers.advanced_activations import *
from ..layers.convolutional import *
from ..layers.core import *
from ..layers.normalization import *

from ..models import Graph

import caffe_pb2 as caffe
import google.protobuf
from caffe_utils import *

import numpy as np
import math

def ConvertModel(layers, phase, input_dim):
	'''
		layers: a list of all the layers in the model
		phase: parameter to specify which network to extract: training or test
		input_dim: input dimensions of the configuration (if in model is in deploy mode)
	'''
	network = make_network(layers, phase)	# obtain the nodes that make up the graph
	network = acyclic(network)	# Convert it to be truly acyclic
	network = merge_layer_blob(network)	# eliminate 'blobs', just have layers
	reverse_network = reverse(network)	# reverse, to obtain the start
	starts = get_starts(reverse_network)	# inputs of the network - 'in-order' is zero
	ends = get_ends(network)	# outputs of the network - 'out-order' is zero
	network = remove_label_paths(network, starts, ends)	# path from input to loss layers removed
	reverse_network = reverse(network) # stores the 'input' for each layer

	model = Graph()

	output_dim = {}	# for dimensionality inference (layer_nb: output_dim)

	# add input nodes
	for start in starts:
		name = str(layers[start].name) 
		model.add_input(name=name, ndim=4)
		if input_dim == []:
			print 'true'
			# not in deploy mode - need to get input DATA dimensions from layers and their transformations
			data_dim = get_data_dim(layers[start])
			output_dim[start] = data_dim # input dimensions (dimensions of DATA)

	# parse all the layers and build equivalent keras graph
	for layer_nb in network:
		if layer_nb in ends or layer_nb in starts:
			continue	# an output layer is skipped. added later with 'add_output'

		layer = layers[layer_nb]
		input_layers = reverse_network[layer_nb]	# inputs to current layer, in the form of layer numbers
		input_layer_names = []	# list of strings identifying the input layer
		for input_layer in input_layers:
			input_layer_names.append(str(layers[input_layer].name))
		
		if len(output_dim) == 0:
			# caffe model in deploy mode, and this is the first layer
			layer_input_dim = input_dim
		else:
			layer_input_dims = []
			for input_layer in input_layers:
				layer_input_dims.append(output_dim[input_layer])

		if len(input_layer_names) == 1:
			input_layer_name = input_layer_names[0] # single input. since concatenation is explicit, 
			layer_input_dim = layer_input_dims[0]	# all layers can be thought of single input layers 
													# (except loss layers, which is handled anyway)

		if layer.type == 3:
			# CONCAT
			# emulation of just concatenation
			axis = layer.concat_param.axis

			model.add_node(ZeroPadding2D(pad=(0, 0)), name=layer.name, inputs=input_layer_names)
			
			layer_output_dim = np.concatenate((layer_input_dims), axis=axis)

		elif layer.type == 4:
			# CONVOLUTION
			nb_col = max(layer.convolution_param.kernel_h, layer.convolution_param.kernel_size)
			nb_row = max(layer.convolution_param.kernel_w, layer.convolution_param.kernel_size)
			nb_filter = layer.convolution_param.num_output

			stride_h = max(layer.convolution_param.kernel_h, layer.convolution_param.stride)
			stride_w = max(layer.convolution_param.kernel_w, layer.convolution_param.stride)

			pad_h = max(layer.convolution_param.pad, layer.convolution_param.pad_h)
			pad_w = max(layer.convolution_param.pad, layer.convolution_param.pad_w)
			model.add_node(ZeroPadding2D(pad=(pad_h, pad_w)), name=layer.name + 'zeropadding', input=input_layer_name)

			stack_size = layer_input_dim[0]
			model.add_node(Convolution2D(nb_filter, stack_size, nb_row, nb_col, subsample=(stride_h, stride_w)), name=layer.name, input=layer.name + 'zeropadding')

			layer_output_dim_padding = [layer_input_dim[0], layer_input_dim[1] + 2 * pad_h, layer_input_dim[2] + 2 * pad_w]
			layer_output_dim = [nb_filter, (layer_output_dim_padding[1] - nb_row + 1) / stride_h, (layer_output_dim_padding[2] - nb_col + 1) / stride_w]

		elif layer.type == 5:
			# IMAGEDATA
			continue

		elif layer.type == 6:
			# DROPOUT
			prob = layer.dropout_param.dropout_ratio
			model.add_node(Dropout(prob), name=layer.name, input=input_layer_name)
			layer_output_dim = layer_input_dim

		elif layer.type == 8:
			# FLATTEN
			model.add_node(Flatten(), name=layer.name, input=input_layer_name)
			layer_output_dim = np.prod(layer_input_dim)

		elif layer.type == 12:
			continue

		elif layer.type == 14:
			# INNER PRODUCT OR DENSE
			layer_output_dim = layer.inner_product_param.num_output
			if len(layer_input_dim) > 1:
				model.add_node(Flatten(), name=layer.name + 'flatten', input=input_layer_name)
				layer_input_dim = [np.prod(layer_input_dim)]
				model.add_node(Dense(layer_input_dim[0], layer_output_dim), name=layer.name, input=layer.name + 'flatten')
			else:
				model.add_node(Dense(layer_input_dim[0], layer_output_dim), name=layer.name, input=input_layer_name)

			layer_output_dim = [layer_output_dim]

		elif layer.type == 15:
			# LOCAL RESPONSE NORMALIZATION
			alpha = layer.lrn_param.alpha
			k = layer.lrn_param.k
			beta = layer.lrn_param.beta
			n = layer.lrn_param.local_size

			model.add_node(LRN2D(alpha=alpha, k=k, beta=beta, n=n), name=layer.name, input=input_layer_name)

			layer_output_dim = layer_input_dim

		elif layer.type == 17:
			# POOLING
			kernel_h = max(layer.pooling_param.kernel_h, layer.pooling_param.kernel_size)
			kernel_w = max(layer.pooling_param.kernel_w, layer.pooling_param.kernel_size)

			stride_h = max(layer.pooling_param.kernel_h, layer.pooling_param.stride)
			stride_w = max(layer.pooling_param.kernel_w, layer.pooling_param.stride)

			pad_h = max(layer.pooling_param.pad, layer.pooling_param.pad_h)
			pad_w = max(layer.pooling_param.pad, layer.pooling_param.pad_w)
			model.add_node(ZeroPadding2D(pad=(pad_h, pad_w)), name=layer.name + 'zeropadding', input=input_layer_name)

			model.add_node(MaxPooling2D(poolsize=(kernel_h, kernel_w), stride=(stride_h, stride_w)), name=layer.name, input=layer.name + 'zeropadding')

			layer_output_dim_padding = [layer_input_dim[0], layer_input_dim[1] + 2 * pad_h, layer_input_dim[2] + 2 * pad_w]
			layer_output_dim = [layer_output_dim_padding[0], (layer_output_dim_padding[1] + kernel_h - 1) / stride_h, (layer_output_dim_padding[2] + kernel_w - 1) / stride_w]

		elif layer.type == 18:
			# ReLU
			model.add_node(Activation('relu'), name=layer.name, input=input_layer_name)
			layer_output_dim = layer_input_dim

		elif layer.type == 19:
			# SIGMOID
			model.add_node(Activation('sigmoid'), name=layer.name, input=input_layer_name)
			layer_output_dim = layer_input_dim

		elif layer.type == 20:
			# SOFTMAX
			model.add_node(Activation('softmax'), name=layer.name, input=input_layer_name)
			layer_output_dim = layer_input_dim

		elif layer.type == 23:
			# TANH
			model.add_node(Activation('tanh'), name=layer.name, input=input_layer_name)
			layer_output_dim = layer_input_dim

		else:
			print "The Layer is not currently Supported"
			return

		output_dim[layer_nb] = layer_output_dim

	for end in ends:
		input_layer_name = layers[reverse_network[end][0]].name
		model.add_output(name=layers[end].name, input=input_layer_name)

	return model

def ConvertWeights(layers):
	pass

def ConvertSolver(caffe_solver):
	pass

def ConvertMeanFile(caffe_mean):
	pass