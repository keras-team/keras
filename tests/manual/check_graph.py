from keras.models import Graph, Sequential
from keras.layers import containers
from keras.layers.core import Dense
import numpy as np

X = np.random.random((100, 32))
X2 = np.random.random((100, 32))
y = np.random.random((100, 4))
y2 = np.random.random((100, 4))

print 'test a non-sequential graph with 1 input and 1 output'
graph = Graph()
graph.add_input(name='input1', ndim=2)

graph.add_node(Dense(32, 16), name='dense1', input='input1')
graph.add_node(Dense(32, 4), name='dense2', input='input1')
graph.add_node(Dense(16, 4), name='dense3', input='dense1')

graph.add_output(name='output1', inputs=['dense2', 'dense3'], merge_mode='sum')
graph.compile('rmsprop', {'output1':'mse'})

out = graph.predict({'input1':X})
loss = graph.test({'input1':X, 'output1':y})
loss = graph.train({'input1':X, 'output1':y})
print loss


print 'test a more complex non-sequential graph with 1 input and 1 output'
graph = Graph()
graph.add_input(name='input1', ndim=2)

graph.add_node(Dense(32, 16), name='dense1', input='input1')
graph.add_node(Dense(32, 4), name='dense2', input='input1')

graph.add_node(Dense(4, 16), name='dense3', input='dense2')
graph.add_node(Dense(16, 4), name='dense4', inputs=['dense1', 'dense3'], merge_mode='sum')

graph.add_output(name='output1', inputs=['dense2', 'dense4'], merge_mode='sum')
graph.compile('rmsprop', {'output1':'mse'})

out = graph.predict({'input1':X})
loss = graph.test({'input1':X, 'output1':y})
loss = graph.train({'input1':X, 'output1':y})
print loss


print 'test a non-sequential graph with 2 inputs and 1 output'
graph = Graph()
graph.add_input(name='input1', ndim=2)
graph.add_input(name='input2', ndim=2)

graph.add_node(Dense(32, 16), name='dense1', input='input1')
graph.add_node(Dense(32, 4), name='dense2', input='input2')
graph.add_node(Dense(16, 4), name='dense3', input='dense1')

graph.add_output(name='output1', inputs=['dense2', 'dense3'], merge_mode='sum')
graph.compile('rmsprop', {'output1':'mse'})

out = graph.predict({'input1':X, 'input2':X2})
loss = graph.test({'input1':X, 'input2':X2, 'output1':y})
loss = graph.train({'input1':X, 'input2':X2, 'output1':y})
print loss


print 'test a non-sequential graph with 1 input and 2 outputs'
graph = Graph()
graph.add_input(name='input1', ndim=2)

graph.add_node(Dense(32, 16), name='dense1', input='input1')
graph.add_node(Dense(32, 4), name='dense2', input='input1')
graph.add_node(Dense(16, 4), name='dense3', input='dense1')

graph.add_output(name='output1', input='dense2')
graph.add_output(name='output2', input='dense3')
graph.compile('rmsprop', {'output1':'mse', 'output2':'mse'})

out = graph.predict({'input1':X})
loss = graph.test({'input1':X, 'output1':y, 'output2':y2})
loss = graph.train({'input1':X, 'output1':y, 'output2':y2})
print loss


print 'test layer-like API'

graph = containers.Graph()
graph.add_input(name='input1', ndim=2)
graph.add_node(Dense(32, 16), name='dense1', input='input1')
graph.add_node(Dense(32, 4), name='dense2', input='input1')
graph.add_node(Dense(16, 4), name='dense3', input='dense1')
graph.add_output(name='output1', inputs=['dense2', 'dense3'], merge_mode='sum')

seq = Sequential()
seq.add(Dense(32, 32, name='first_seq_dense'))
seq.add(graph)
seq.add(Dense(4, 4, name='last_seq_dense'))

print seq.params
print seq.layers
print 'input:'
print seq.get_input()
print 'output:'
print seq.get_output()
seq.compile('rmsprop', 'mse')

loss = seq.fit(X, y, batch_size=10, nb_epoch=1)
print loss