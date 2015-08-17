from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential, Graph
from keras.layers.core import Layer, Activation, Dense, Flatten, Reshape, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import keras.utils.layer_utils as layer_utils

print('-- Sequential model')
left = Sequential()
left.add(Convolution2D(32, 1, 3, 3, border_mode='valid'))
left.add(MaxPooling2D(poolsize=(2, 2)))
left.add(Flatten())
left.add(Dense(32 * 13 * 13, 50))
left.add(Activation('relu'))

right = Sequential()
right.add(Dense(784, 30))
right.add(Activation('relu'))

model = Sequential()
model.add(Merge([left, right], mode='concat'))

model.add(Dense(80, 10))
model.add(Activation('softmax'))

layer_utils.print_layer_shapes(model, [(1, 1, 28, 28), (1, 784)])

print('-- Graph model')
graph = Graph()
graph.add_input(name='input1', ndim=2)
graph.add_input(name='input2', ndim=4)
graph.add_node(Dense(32, 16), name='dense1', input='input1')
graph.add_node(Dense(16, 4), name='dense3', input='dense1')

graph.add_node(Convolution2D(32, 1, 3, 3), name='conv1', input='input2')
graph.add_node(Flatten(), name='flatten1', input='conv1')
graph.add_node(Dense(32 * 13 * 13, 10), name='dense4', input='flatten1')

graph.add_output(name='output1', inputs=['dense1', 'dense3'], merge_mode='sum')
graph.add_output(name='output2', inputs=['dense1', 'dense4'], merge_mode='concat')

layer_utils.print_layer_shapes(graph, {'input1': (1, 32), 'input2': (1, 1, 28, 28)})

print('Test script complete')
