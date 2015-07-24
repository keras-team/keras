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

def is_data_input(layer):
	return layer.type in [5, 12, 25, 29, 9]


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

	network = make_network(layers, phase)	# obtain the nodes that make up the graph
	if len(network) == 0:
		raise Exception('failed to construct network from the prototext')
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
		model.add_input(name=name, ndim=4)
		if in_deploy_mode:
			output_dim[name] = input_dim
		else:
			data_dim = get_data_dim(layers[start])	# not in deploy mode - need to get input DATA dimensions from layers and their transformations
			output_dim[name] = data_dim
			

	# parse all the layers and build equivalent keras graph
	for layer_nb in network:
		layer = layers[layer_nb]
		name = layer.name

		if is_data_input(layer):
			# not including data layers
			continue

		if layer_nb in starts:
			# in deploy mode since data layers can't reach this line and condition holds.
			new_name = 'graph_input_' + name
			input_layer_names = [name]
			name = new_name
		else:
			input_layers = reverse_network[layer_nb]	# inputs to current layer, in the form of layer numbers
			input_layer_names = []	# list of strings identifying the input layer
			for input_layer in input_layers:
				if input_layer in starts and in_deploy_mode:
					input_layer_names.append('graph_input_' + layers[input_layer].name)
				else:
					input_layer_names.append(layers[input_layer].name)

		if layer_nb in ends:
			name = 'output_' + name 	# outputs nodes are marked with 'output_' prefix from which output is derived later in 'add_output'

		layer_input_dims = []
		for input_layer_name in input_layer_names:
			layer_input_dims.append(output_dim[input_layer_name])
		
		if len(input_layer_names) == 1:
			input_layer_name = input_layer_names[0] # single input. since concatenation is explicit, 
			layer_input_dim = layer_input_dims[0]	# all layers can be thought of single input layers 
													# (except loss layers, which is handled anyway)

		if layer.type == 3:
			# CONCAT
			# emulation of just concatenation
			axis = layer.concat_param.axis # 0 for batch, 1 for stack
			model.add_node(Activation('linear'), name=name, inputs=input_layer_names, concat_axis=axis)
			
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
				input_layer_name = name + '_zeropadding'
			
			stack_size = layer_input_dim[0]
			
			model.add_node(Convolution2D(nb_filter, stack_size, nb_row, nb_col, subsample=(stride_h, stride_w)), name=name, input=input_layer_name)

			layer_output_dim_padding = [layer_input_dim[0], layer_input_dim[1] + 2 * pad_h, layer_input_dim[2] + 2 * pad_w]
			layer_output_dim = [nb_filter, (layer_output_dim_padding[1] - nb_row) / stride_h + 1, (layer_output_dim_padding[2] - nb_col) / stride_w + 1]

		elif layer.type == 6:
			# DROPOUT
			prob = layer.dropout_param.dropout_ratio
			model.add_node(Dropout(prob), name=name, input=input_layer_name)
			layer_output_dim = layer_input_dim

		elif layer.type == 8:
			# FLATTEN
			model.add_node(Flatten(), name=name, input=input_layer_name)
			layer_output_dim = np.prod(layer_input_dim)

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
				input_layer_name = name + '_zeropadding'

			model.add_node(MaxPooling2D(poolsize=(kernel_h, kernel_w), stride=(stride_h, stride_w)), name=name, input=input_layer_name)

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

		elif layer.type == 20 or layer.type == 21:
			# SOFTMAX
			model.add_node(Activation('softmax'), name=name, input=input_layer_name)
			layer_output_dim = layer_input_dim

		elif layer.type == 22:
			# SPLIT
			model.add_node(ZeroPadding2D(pad=(0, 0)), name=name, inputs=input_layer_name)
			layer_output_dim = layer_input_dim

		elif layer.type == 23:
			# TANH
			model.add_node(Activation('tanh'), name=name, input=input_layer_name)
			layer_output_dim = layer_input_dim

		else:
			raise RuntimeError("one or more layers used in this model is not currently supported")

		output_dim[name] = layer_output_dim

	for end in ends:
		input_layer_name = 'output_' + layers[end].name
		model.add_output(name=layers[end].name, input=input_layer_name)

	starts_names = []
	for start in starts:
		starts_names.append(layers[start].name)
	ends_names = []
	for end in ends:
		ends_names.append(layers[end].name)

	return model, starts_names, ends_names

def model_from_param(layers):
	'''
		layers: a list of all the layers in the model with the assocaited parameters
	'''
	# DEPLOY MODE: first layer is computation
	# NON DEPLOY MODE: first layer is data input

	network = make_network(layers, 0)	# obtain the nodes that make up the graph
	if len(network) == 0:
		raise Exception('failed to construct network from the caffemodel')
	network = acyclic(network)	# Convert it to be truly acyclic
	network = merge_layer_blob(network)	# eliminate 'blobs', just have layers
	reverse_network = reverse(network)	# reverse, to obtain the start
	starts = get_starts(reverse_network)	# inputs of the network - 'in-order' is zero
	ends = get_ends(network)	# outputs of the network - 'out-order' is zero
	network = remove_label_paths(network, starts, ends)	# path from input to loss layers removed
	reverse_network = reverse(network) # stores the 'input' for each layer

	model = Graph()

	# add input nodes
	for start in starts:
		name = layers[start].name
		model.add_input(name=name, ndim=4)

	# parse all the layers and build equivalent keras graph
	for layer_nb in network:
		layer = layers[layer_nb]
		name = layer.name

		if is_data_input(layer):
			continue

		if layer_nb in starts:
			input_layer_names = [name]	# this is both an input layer and computation layer. the name has been added as input layer
			name = 'graph_input_' + name # use a prefix to create a new name, which is used on from hereon to take input from this layer(see below)
		else:
			input_layers = reverse_network[layer_nb]	# inputs to current layer, in the form of layer numbers
			input_layer_names = []	# list of strings identifying the input layer
			for input_layer in input_layers:
				if input_layer in starts and not is_data_input(layers[input_layer]):	# taking input from first layer and that's not a data layer, implies use the prefix 
					input_layer_names.append('graph_input_' + layers[input_layer].name)
				else:
					input_layer_names.append(layers[input_layer].name)

		if len(input_layer_names) == 1:
			input_layer_name = input_layer_names[0] # single input. since concatenation is explicit, 
													# all layers can be thought of single input layers 
													# (except loss layers, which is handled anyway)
		if layer_nb in ends:
			name = 'output_' + name 	# outputs nodes are marked with 'output_' prefix from which output is derived
										# since the graph output is not model output


		if layer.type == 3:
			# CONCAT
			# emulation of just concatenation
			axis = layer.concat_param.axis # 0 for batch, 1 for stack
			model.add_node(Activation('linear'), name=name, inputs=input_layer_names, concat_axis=axis)

		elif layer.type == 4:
			# CONVOLUTION
			blobs = layer.blobs
			nb_filter, temp_stack_size, nb_col, nb_row = blobs[0].num, blobs[0].channels, blobs[0].height, blobs[0].width
			
			group = layer.convolution_param.group # weird old trick used in paralellizing networks
			stack_size = temp_stack_size * group
			
			weights_p = np.zeros((nb_filter, stack_size, nb_col, nb_row)) # maybe not all synapses are existant
			weights_b = np.array(blobs[1].data)

			chunk_data_size = len(blobs[0].data) // group
			stacks_size_per_chunk = stack_size // group
			nb_filter_per_chunk = nb_filter // group
			
			for i in range(group):
				chunk_weights = weights_p[i * nb_filter_per_chunk: (i + 1) * nb_filter_per_chunk, i * stacks_size_per_chunk: (i + 1) * stacks_size_per_chunk, :, :]
				chunk_weights[:] = np.array(blobs[0].data[i * chunk_data_size:(i + 1) * chunk_data_size]).reshape(chunk_weights.shape)

			weights = [weights_p, weights_b]

			stride_h = max(layer.convolution_param.kernel_h, layer.convolution_param.stride)
			stride_w = max(layer.convolution_param.kernel_w, layer.convolution_param.stride)

			pad_h = max(layer.convolution_param.pad, layer.convolution_param.pad_h)
			pad_w = max(layer.convolution_param.pad, layer.convolution_param.pad_w)
			
			if pad_h + pad_w > 0:
				model.add_node(ZeroPadding2D(pad=(pad_h, pad_w)), name=name + '_zeropadding', input=input_layer_name)
				input_layer_name = name + '_zeropadding'

			model.add_node(Convolution2D(nb_filter, stack_size, nb_row, nb_col, subsample=(stride_h, stride_w), weights=weights), name=name, input=input_layer_name)

		elif layer.type == 6:
			# DROPOUT
			prob = layer.dropout_param.dropout_ratio
			model.add_node(Dropout(prob), name=name, input=input_layer_name)
			
		elif layer.type == 8:
			# FLATTEN
			model.add_node(Flatten(), name=name, input=input_layer_name)

		elif layer.type == 14:
			# INNER PRODUCT OR DENSE
			blobs = layer.blobs
			nb_filter, stack_size, nb_col, nb_row = blobs[0].num, blobs[0].channels, blobs[0].height, blobs[0].width
			
			weights_p = np.array(blobs[0].data).reshape(nb_filter, stack_size, nb_col, nb_row)[0,0,:,:].T
			weights_b = np.array(blobs[1].data)
			weights = [weights_p, weights_b]

			model.add_node(Flatten(), name=name + '_flatten', input=input_layer_name)
			model.add_node(Dense(nb_row, nb_col, weights=weights), name=name, input=name + '_flatten')

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
				input_layer_name = name + '_zeropadding'

			model.add_node(MaxPooling2D(poolsize=(kernel_h, kernel_w), stride=(stride_h, stride_w)), name=name, input=input_layer_name)

		elif layer.type == 18:
			# ReLU
			model.add_node(Activation('relu'), name=name, input=input_layer_name)

		elif layer.type == 19:
			# SIGMOID
			model.add_node(Activation('sigmoid'), name=name, input=input_layer_name)

		elif layer.type == 20 or layer.type == 21:
			# SOFTMAX
			model.add_node(Activation('softmax'), name=name, input=input_layer_name)

		elif layer.type == 22:
			# SPLIT
			model.add_node(ZeroPadding2D(pad=(0, 0)), name=name, input=input_layer_name)

		elif layer.type == 23:
			# TANH
			model.add_node(Activation('tanh'), name=name, input=input_layer_name)

		else:
			raise RuntimeError("the layer: ", layer.name, "used in this model is not currently supported")

	for end in ends:
		input_layer_name = 'output_' + layers[end].name
		model.add_output(name=layers[end].name, input=input_layer_name)

	starts_names = []
	for start in starts:
		starts_names.append(layers[start].name)
	ends_names = []
	for end in ends:
		ends_names.append(layers[end].name)

	return model, starts_names, ends_names

def convert_weights(layers):
	'''
		layers: a list of all the layers in the model with the assocaited parameters
	'''
	# DEPLOY MODE: first layer is computation
	# NON DEPLOY MODE: first layer is data input

	weights = {}

	# parse all the layers and build equivalent keras graph
	for layer_nb in range(len(layers)):
		layer = layers[layer_nb]
		name = layer.name
		
		if layer.type == 4:
			# CONVOLUTION
			blobs = layer.blobs
			nb_filter, temp_stack_size, nb_col, nb_row = blobs[0].num, blobs[0].channels, blobs[0].height, blobs[0].width
			
			group = layer.convolution_param.group # weird old trick used in paralellizing networks
			stack_size = temp_stack_size * group
			
			weights_p = np.zeros((nb_filter, stack_size, nb_col, nb_row)) # maybe not all synapses are existant
			weights_b = np.array(blobs[1].data)

			chunk_data_size = len(blobs[0].data) // group
			stacks_size_per_chunk = stack_size // group
			nb_filter_per_chunk = nb_filter // group
			
			for i in range(group):
				chunk_weights = weights_p[i * nb_filter_per_chunk: (i + 1) * nb_filter_per_chunk, i * stacks_size_per_chunk: (i + 1) * stacks_size_per_chunk, :, :]
				chunk_weights[:] = np.array(blobs[0].data[i * chunk_data_size:(i + 1) * chunk_data_size]).reshape(chunk_weights.shape)

			layer_weights = [weights_p, weights_b]
			weights[name] = layer_weights

		elif layer.type == 14:
			# INNER PRODUCT OR DENSE
			blobs = layer.blobs
			nb_filter, stack_size, nb_col, nb_row = blobs[0].num, blobs[0].channels, blobs[0].height, blobs[0].width
			
			weights_p = np.array(blobs[0].data).reshape(nb_filter, stack_size, nb_col, nb_row)[0,0,:,:].T
			weights_b = np.array(blobs[1].data)
			
			layer_weights = [weights_p, weights_b]
			weights[name] = layer_weights

	return weights

def convert_solver(caffe_solver):
	pass

def convert_meanfile(caffe_mean):
	pass