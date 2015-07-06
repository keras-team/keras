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

def model_from_config(layers, phase, input_dim):
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
		name = layers[start].name
		model.add_input(name='input_' + name, ndim=4) #input points to input layers - marked with 'input_' prefix
		if input_dim == []:
			# not in deploy mode - need to get input DATA dimensions from layers and their transformations
			data_dim = get_data_dim(layers[start])
			output_dim[start] = data_dim # input dimensions (dimensions of DATA)

	# parse all the layers and build equivalent keras graph
	for layer_nb in network:
		layer = layers[layer_nb]
		name = layer.name
		input_layers = reverse_network[layer_nb]	# inputs to current layer, in the form of layer numbers
		input_layer_names = []	# list of strings identifying the input layer
		for input_layer in input_layers:
			input_layer_names.append(layers[input_layer].name)

		layer_input_dims = []
		if len(output_dim) == 0:
			# caffe model in deploy mode, and this is the first layer
			layer_input_dims = [input_dim]
		else:
			for input_layer in input_layers:
				layer_input_dims.append(output_dim[input_layer])

		if len(input_layer_names) == 1:
			input_layer_name = input_layer_names[0] # single input. since concatenation is explicit, 
			layer_input_dim = layer_input_dims[0]	# all layers can be thought of single input layers 
													# (except loss layers, which is handled anyway)

		if layer_nb in starts:
			input_layer_name = 'input_' + input_layer_name	# the respective input node has that prefix. 
															# always single input as its the input layer
		elif layer_nb in ends:
			name = 'output_' + name 	# outputs nodes are marked with 'output_' prefix from which output is derived


		if layer.type == 3:
			# CONCAT
			# emulation of just concatenation
			axis = layer.concat_param.axis # 0 for batch, 1 for stack
			model.add_node(ZeroPadding2D(pad=(0, 0)), name=name, inputs=input_layer_names, concat_axis=axis)
			
			layer_output_dim = [layer_input_dims[0][0], layer_input_dims[0][1], layer_input_dims[0][2]]
			for dim in layer_input_dims[1:]:
				layer_output_dim[0] += dim[0]

		elif layer.type == 4:
			# CONVOLUTION
			nb_col = max(layer.convolution_param.kernel_h, layer.convolution_param.kernel_size)
			nb_row = max(layer.convolution_param.kernel_w, layer.convolution_param.kernel_size)
			nb_filter = layer.convolution_param.num_output

			stride_h = max(layer.convolution_param.kernel_h, layer.convolution_param.stride)
			stride_w = max(layer.convolution_param.kernel_w, layer.convolution_param.stride)

			pad_h = max(layer.convolution_param.pad, layer.convolution_param.pad_h)
			pad_w = max(layer.convolution_param.pad, layer.convolution_param.pad_w)
			if pad_h + pad_w > 0:
				model.add_node(ZeroPadding2D(pad=(pad_h, pad_w)), name=name + '_zeropadding', input=input_layer_name)

			stack_size = layer_input_dim[0]
			model.add_node(Convolution2D(nb_filter, stack_size, nb_row, nb_col, subsample=(stride_h, stride_w)), name=name, input=name + '_zeropadding')

			layer_output_dim_padding = [layer_input_dim[0], layer_input_dim[1] + 2 * pad_h, layer_input_dim[2] + 2 * pad_w]
			layer_output_dim = [nb_filter, (layer_output_dim_padding[1] - nb_row) / stride_h + 1, (layer_output_dim_padding[2] - nb_col) / stride_w + 1]

		elif layer.type == 5:
			# IMAGEDATA
			continue

		elif layer.type == 6:
			# DROPOUT
			prob = layer.dropout_param.dropout_ratio
			model.add_node(Dropout(prob), name=name, input=input_layer_name)
			layer_output_dim = layer_input_dim

		elif layer.type == 8:
			# FLATTEN
			model.add_node(Flatten(), name=name, input=input_layer_name)
			layer_output_dim = np.prod(layer_input_dim)

		elif layer.type == 12:
			continue

		elif layer.type == 14:
			# INNER PRODUCT OR DENSE
			layer_output_dim = layer.inner_product_param.num_output
			if len(layer_input_dim) > 1:
				model.add_node(Flatten(), name=name + '_flatten', input=input_layer_name)
				layer_input_dim = [np.prod(layer_input_dim)]
				model.add_node(Dense(layer_input_dim[0], layer_output_dim), name=name, input=name + '_flatten')
			else:
				model.add_node(Dense(layer_input_dim[0], layer_output_dim), name=name, input=input_layer_name)

			layer_output_dim = [layer_output_dim]

		elif layer.type == 15:
			# LOCAL RESPONSE NORMALIZATION
			alpha = layer.lrn_param.alpha
			k = layer.lrn_param.k
			beta = layer.lrn_param.beta
			n = layer.lrn_param.local_size

			model.add_node(LRN2D(alpha=alpha, k=k, beta=beta, n=n), name=name, input=input_layer_name)

			layer_output_dim = layer_input_dim

		elif layer.type == 17:
			# POOLING
			kernel_h = max(layer.pooling_param.kernel_h, layer.pooling_param.kernel_size)
			kernel_w = max(layer.pooling_param.kernel_w, layer.pooling_param.kernel_size)

			stride_h = max(layer.pooling_param.kernel_h, layer.pooling_param.stride)
			stride_w = max(layer.pooling_param.kernel_w, layer.pooling_param.stride)

			pad_h = max(layer.pooling_param.pad, layer.pooling_param.pad_h)
			pad_w = max(layer.pooling_param.pad, layer.pooling_param.pad_w)
			if pad_h + pad_w > 0:
				model.add_node(ZeroPadding2D(pad=(pad_h, pad_w)), name=name + '_zeropadding', input=input_layer_name)

			model.add_node(MaxPooling2D(poolsize=(kernel_h, kernel_w), stride=(stride_h, stride_w)), name=name, input=name + '_zeropadding')

			layer_output_dim_padding = [layer_input_dim[0], layer_input_dim[1] + 2 * pad_h, layer_input_dim[2] + 2 * pad_w]
			layer_output_dim = [layer_output_dim_padding[0], (layer_output_dim_padding[1] - kernel_h) / stride_h + 1, (layer_output_dim_padding[2] - kernel_w) / stride_w + 1]

		elif layer.type == 18:
			# ReLU
			model.add_node(Activation('relu'), name=name, input=input_layer_name)
			layer_output_dim = layer_input_dim

		elif layer.type == 19:
			# SIGMOID
			model.add_node(Activation('sigmoid'), name=name, input=input_layer_name)
			layer_output_dim = layer_input_dim

		elif layer.type == 20:
			# SOFTMAX
			model.add_node(Activation('softmax'), name=name, input=input_layer_name)
			layer_output_dim = layer_input_dim

		elif layer.type == 23:
			# TANH
			model.add_node(Activation('tanh'), name=name, input=input_layer_name)
			layer_output_dim = layer_input_dim

		else:
			print "The Layer is not currently Supported"
			return

		output_dim[layer_nb] = layer_output_dim

	for end in ends:
		input_layer_name = 'output_' + layers[reverse_network[end][0]].name
		model.add_output(name=layers[end].name, input=input_layer_name)

	return model

def model_from_param(layers):
	'''
		layers: a list of all the layers in the model with the assocaited parameters
	'''
	network = make_network(layers, 0)	# obtain the nodes that make up the graph
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
		name = layers[start].name
		model.add_input(name='input_' + name, ndim=4) #input points to input layers - marked with 'input_' prefix

	# parse all the layers and build equivalent keras graph
	for layer_nb in network:
		layer = layers[layer_nb]
		name = layer.name
		input_layers = reverse_network[layer_nb]	# inputs to current layer, in the form of layer numbers
		input_layer_names = []	# list of strings identifying the input layer
		for input_layer in input_layers:
			input_layer_names.append(layers[input_layer].name)

		if len(input_layer_names) == 1:
			input_layer_name = input_layer_names[0] # single input. since concatenation is explicit, 
													# all layers can be thought of single input layers 
													# (except loss layers, which is handled anyway)
		if layer_nb in starts:
			input_layer_name = 'input_' + input_layer_name	# the respective input node has that prefix. 
															# always single input as its the input layer
		elif layer_nb in ends:
			name = 'output_' + name 	# outputs nodes are marked with 'output_' prefix from which output is derived


		if layer.type == 3:
			# CONCAT
			# emulation of just concatenation
			axis = layer.concat_param.axis # 0 for batch, 1 for stack
			model.add_node(ZeroPadding2D(pad=(0, 0)), name=name, inputs=input_layer_names, concat_axis=axis)

		elif layer.type == 4:
			# CONVOLUTION

			blobs = layer.blobs
			nb_filter, stack_size, nb_col, nb_row = blobs[0].num, blobs[0].channels, blobs[0].height, blobs[0].width
			
			weights_p = np.array(blobs[0].data).reshape(nb_filter, stack_size, nb_col, nb_row)[:,:,::-1,::-1]
			weights_b = np.array(blobs[1].data)
			weights = [weights_p, weights_b]

			stride_h = max(layer.convolution_param.kernel_h, layer.convolution_param.stride)
			stride_w = max(layer.convolution_param.kernel_w, layer.convolution_param.stride)

			pad_h = max(layer.convolution_param.pad, layer.convolution_param.pad_h)
			pad_w = max(layer.convolution_param.pad, layer.convolution_param.pad_w)
			if pad_h + pad_w > 0:
				model.add_node(ZeroPadding2D(pad=(pad_h, pad_w)), name=name + '_zeropadding', input=input_layer_name)

			model.add_node(Convolution2D(nb_filter, stack_size, nb_row, nb_col, subsample=(stride_h, stride_w), weights=weights), name=name, input=name + '_zeropadding')

		elif layer.type == 5:
			# IMAGEDATA
			continue

		elif layer.type == 6:
			# DROPOUT
			prob = layer.dropout_param.dropout_ratio
			model.add_node(Dropout(prob), name=name, input=input_layer_name)
			
		elif layer.type == 8:
			# FLATTEN
			model.add_node(Flatten(), name=name, input=input_layer_name)

		elif layer.type == 12:
			# DATA
			continue

		elif layer.type == 14:
			# INNER PRODUCT OR DENSE
			blobs = layer_param.blobs
			nb_filter, stack_size, nb_col, nb_row = blobs[0].num, blobs[0].channels, blobs[0].height, blobs[0].width
			weights_p = np.array(blobs[0].data).reshape(nb_filter, stack_size, nb_col, nb_row)[0,0,:,:].T
			weights_b = np.array(blobs[1].data)
			weights = [weights_p, weights_b]

			model.add_node(Flatten(), name=name + '_flatten', input=input_layer_name)
			model.add_node(Dense(nb_col, nb_row, weights=weights), name=name, input=name + '_flatten')

		elif layer.type == 15:
			# LOCAL RESPONSE NORMALIZATION
			alpha = layer.lrn_param.alpha
			k = layer.lrn_param.k
			beta = layer.lrn_param.beta
			n = layer.lrn_param.local_size

			model.add_node(LRN2D(alpha=alpha, k=k, beta=beta, n=n), name=name, input=input_layer_name)

		elif layer.type == 17:
			# POOLING
			kernel_h = max(layer.pooling_param.kernel_h, layer.pooling_param.kernel_size)
			kernel_w = max(layer.pooling_param.kernel_w, layer.pooling_param.kernel_size)

			stride_h = max(layer.pooling_param.kernel_h, layer.pooling_param.stride)
			stride_w = max(layer.pooling_param.kernel_w, layer.pooling_param.stride)

			pad_h = max(layer.pooling_param.pad, layer.pooling_param.pad_h)
			pad_w = max(layer.pooling_param.pad, layer.pooling_param.pad_w)
			if pad_h + pad_w > 0:
				model.add_node(ZeroPadding2D(pad=(pad_h, pad_w)), name=name + '_zeropadding', input=input_layer_name)

			model.add_node(MaxPooling2D(poolsize=(kernel_h, kernel_w), stride=(stride_h, stride_w)), name=name, input=name + '_zeropadding')

		elif layer.type == 18:
			# ReLU
			model.add_node(Activation('relu'), name=name, input=input_layer_name)

		elif layer.type == 19:
			# SIGMOID
			model.add_node(Activation('sigmoid'), name=name, input=input_layer_name)

		elif layer.type == 20:
			# SOFTMAX
			model.add_node(Activation('softmax'), name=name, input=input_layer_name)

		elif layer.type == 23:
			# TANH
			model.add_node(Activation('tanh'), name=name, input=input_layer_name)

		else:
			print "The Layer is not currently Supported"
			return

	for end in ends:
		input_layer_name = 'output_' + layers[reverse_network[end][0]].name
		model.add_output(name=layers[end].name, input=input_layer_name)

	return model

def convert_weights(layers):
	pass

def convert_solver(caffe_solver):
	pass

def convert_meanfile(caffe_mean):
	pass