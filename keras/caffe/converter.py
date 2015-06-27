from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam

def ConvertModel(layers):
	nb_layers = len(layers)

	model = Sequential()

	for l in range(nb_layers):
		layer = layers[l]
		layer_type = layer.type

		if layer_type == 4:
			# CONVOLUTION
			if layer.convolution_param.kernel_h > 0:
				nb_col = layer.convolution_param.kernel_h
			else:
				nb_col = layer.convolution_param.kernel_size

			if layer.convolution_param.kernel_w > 0:
				nb_row = layer.convolution_param.kernel_w
			else:
				nb_row = layer.convolution_param.kernel_size

			nb_filter = layer.convolution_param.num_output

			if pad > 0:
				model.add(ZeroPadding2D(width=pad))

			model.add(Convolution2D(nb_filter, stack_size, nb_row, nb_col))
			pad = layer_config.convolution_param.pad

		elif layer_type == 6:
			# DROPOUT
			model.add(Dropout(layer_config.dropout_param.dropout_ratio))
		elif layer_type == 8:
			# FLATTEN
			model.add(Flatten())
		elif layer_type == 14:
			# INNER PRODUCT OR DENSE
			model.add(Flatten())
			blobs = layer_param.blobs
			nb_filter, stack_size, nb_col, nb_row = blobs[0].num, blobs[0].channels, blobs[0].height, blobs[0].width
			weights_p = np.array(blobs[0].data).reshape(nb_filter, stack_size, nb_col, nb_row)[0,0,:,:].T
			weights_b = np.array(blobs[1].data)
			weights = [weights_p, weights_b]
			model.add(Dense(nb_col, nb_row, weights=weights))
		elif layer_type == 15:
			# LOCAL RESPONSE NORMALIZATION
			print "LRN is being implemented"
		elif layer_type == 17:
			# POOLING
			# TODO: Caffe allows pooling with variable strides. max_pool_2d should be modified appropriately.
			pool_size = (layer_config.pooling_param.kernel_size, layer_config.pooling_param.kernel_size)
			model.add(MaxPooling2D(poolsize=pool_size))
		elif layer_type == 18:
			# ReLU
			model.add(Activation('relu'))
		elif layer_type == 19:
			# SIGMOID
			model.add(Activation('sigmoid'))
		elif layer_type == 20:
			# SOFTMAX
			model.add(Activation('softmax'))
		elif layer_type == 23:
			# TANH
			model.add(Activation('tanh'))
		else:
			print "The Layer ", layer_type_nb[layer_config.type], " is not currently supported or is out of context"

def ConvertWeights():

def ConvertSolver():

def ConvertMean():