from __future__ import print_function
import unittest
import numpy as np
np.random.seed(1337)
from keras.models import Graph, Sequential
from keras.layers import containers
from keras.layers.core import Dense, Activation
from keras.utils.test_utils import get_test_data

X = np.random.random((100, 32))
X2 = np.random.random((100, 32))
y = np.random.random((100, 4))
y2 = np.random.random((100,))

(X_train, y_train), (X_test, y_test) = get_test_data(nb_train=1000, nb_test=200, input_shape=(32,),
    classification=False, output_shape=(4,))
(X2_train, y2_train), (X2_test, y2_test) = get_test_data(nb_train=1000, nb_test=200, input_shape=(32,),
    classification=False, output_shape=(1,))

class TestGraph(unittest.TestCase):
    def test_1o_1i(self):
        print('test a non-sequential graph with 1 input and 1 output')
        graph = Graph()
        graph.add_input(name='input1', ndim=2)

        graph.add_node(Dense(32, 16), name='dense1', input='input1')
        graph.add_node(Dense(32, 4), name='dense2', input='input1')
        graph.add_node(Dense(16, 4), name='dense3', input='dense1')

        graph.add_output(name='output1', inputs=['dense2', 'dense3'], merge_mode='sum')
        graph.compile('rmsprop', {'output1':'mse'})

        history = graph.fit({'input1':X_train, 'output1':y_train}, nb_epoch=10)
        out = graph.predict({'input1':X_test})
        assert(type(out == dict))
        assert(len(out) == 1)
        loss = graph.test_on_batch({'input1':X_test, 'output1':y_test})
        loss = graph.train_on_batch({'input1':X_test, 'output1':y_test})
        loss = graph.evaluate({'input1':X_test, 'output1':y_test})
        print(loss)
        assert(loss < 1.4)

    def test_1o_1i_2(self):
        print('test a more complex non-sequential graph with 1 input and 1 output')
        graph = Graph()
        graph.add_input(name='input1', ndim=2)

        graph.add_node(Dense(32, 16), name='dense1', input='input1')
        graph.add_node(Dense(32, 4), name='dense2-0', input='input1')
        graph.add_node(Activation('relu'), name='dense2', input='dense2-0')

        graph.add_node(Dense(4, 16), name='dense3', input='dense2')
        graph.add_node(Dense(16, 4), name='dense4', inputs=['dense1', 'dense3'], merge_mode='sum')

        graph.add_output(name='output1', inputs=['dense2', 'dense4'], merge_mode='sum')
        graph.compile('rmsprop', {'output1':'mse'})

        history = graph.fit({'input1':X_train, 'output1':y_train}, nb_epoch=10)
        out = graph.predict({'input1':X_train})
        assert(type(out == dict))
        assert(len(out) == 1)
        loss = graph.test_on_batch({'input1':X_test, 'output1':y_test})
        loss = graph.train_on_batch({'input1':X_test, 'output1':y_test})
        loss = graph.evaluate({'input1':X_test, 'output1':y_test})
        print(loss)
        assert(loss < 1.4)
        graph.get_config(verbose=1)

    def test_1o_2i(self):
        print('test a non-sequential graph with 2 inputs and 1 output')
        graph = Graph()
        graph.add_input(name='input1', ndim=2)
        graph.add_input(name='input2', ndim=2)

        graph.add_node(Dense(32, 16), name='dense1', input='input1')
        graph.add_node(Dense(32, 4), name='dense2', input='input2')
        graph.add_node(Dense(16, 4), name='dense3', input='dense1')

        graph.add_output(name='output1', inputs=['dense2', 'dense3'], merge_mode='sum')
        graph.compile('rmsprop', {'output1':'mse'})

        history = graph.fit({'input1':X_train, 'input2':X2_train, 'output1':y_train}, nb_epoch=10)
        out = graph.predict({'input1':X_test, 'input2':X2_test})
        assert(type(out == dict))
        assert(len(out) == 1)
        loss = graph.test_on_batch({'input1':X_test, 'input2':X2_test, 'output1':y_test})
        loss = graph.train_on_batch({'input1':X_test, 'input2':X2_test, 'output1':y_test})
        loss = graph.evaluate({'input1':X_test, 'input2':X2_test, 'output1':y_test})
        print(loss)
        assert(loss < 1.4)
        graph.get_config(verbose=1)

    def test_2o_1i_weights(self):
        print('test a non-sequential graph with 1 input and 2 outputs')
        graph = Graph()
        graph.add_input(name='input1', ndim=2)

        graph.add_node(Dense(32, 16), name='dense1', input='input1')
        graph.add_node(Dense(32, 4), name='dense2', input='input1')
        graph.add_node(Dense(16, 1), name='dense3', input='dense1')

        graph.add_output(name='output1', input='dense2')
        graph.add_output(name='output2', input='dense3')
        graph.compile('rmsprop', {'output1':'mse', 'output2':'mse'})

        history = graph.fit({'input1':X_train, 'output1':y_train, 'output2':y2_train}, nb_epoch=10)
        out = graph.predict({'input1':X_test})
        assert(type(out == dict))
        assert(len(out) == 2)
        loss = graph.test_on_batch({'input1':X_test, 'output1':y_test, 'output2':y2_test})
        loss = graph.train_on_batch({'input1':X_test, 'output1':y_test, 'output2':y2_test})
        loss = graph.evaluate({'input1':X_test, 'output1':y_test, 'output2':y2_test})
        print(loss)
        assert(loss < 2.7)

        print('test weight saving')
        graph.save_weights('temp.h5', overwrite=True)
        graph = Graph()
        graph.add_input(name='input1', ndim=2)
        graph.add_node(Dense(32, 16), name='dense1', input='input1')
        graph.add_node(Dense(32, 4), name='dense2', input='input1')
        graph.add_node(Dense(16, 1), name='dense3', input='dense1')
        graph.add_output(name='output1', input='dense2')
        graph.add_output(name='output2', input='dense3')
        graph.compile('rmsprop', {'output1':'mse', 'output2':'mse'})
        graph.load_weights('temp.h5')
        nloss = graph.evaluate({'input1':X_test, 'output1':y_test, 'output2':y2_test})
        print(nloss)
        assert(loss == nloss)

    def test_recursive(self):
        print('test layer-like API')

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

        seq.compile('rmsprop', 'mse')

        history = seq.fit(X_train, y_train, batch_size=10, nb_epoch=10)
        loss = seq.evaluate(X_test, y_test)
        print(loss)
        assert(loss < 1.4)

        loss = seq.evaluate(X_test, y_test, show_accuracy=True)
        pred = seq.predict(X_test)
        seq.get_config(verbose=1)


if __name__ == '__main__':
    print('Test graph model')
    unittest.main()