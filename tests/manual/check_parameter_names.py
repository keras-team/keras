

from __future__ import print_function
import unittest
import numpy as np
np.random.seed(1337)

from keras.models import Graph, Sequential
from keras.layers import containers
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Convolution1D, Convolution2D


def print_model_parameter_names(model):

    assert type(model) in [Graph, Sequential]

    for p in model.params:
        print(p.name)
        # all parameters should be named
        assert p.name is not None and p.name != ""

    return


class TestParameterNames(unittest.TestCase):


    def test_named_nodes(self):
        print("\nTesting when all nodes are named")
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

        print_model_parameter_names(seq)
        return

    def test_sequential_defaults(self):
        print("\nTesting default parameter names (for sequential model)")

        model = Sequential()
        model.add(Dense(10, 512))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 32, 32, 32))
        model.add(Activation('relu'))
        model.add(Dense(512, 2))
        model.add(Activation('softmax'))

        print_model_parameter_names(model)
        return

    def test_graph_defaults(self):
        print("\nTesting default parameter names (for graph model)")

        graph = Graph()
        graph.add_input(name='input1', ndim=2)
        graph.add_node(Dense(32, 16), name='dense1', input='input1')
        graph.add_node(Dense(32, 4), name='dense2', input='input1')
        graph.add_node(Dense(16, 4), name='dense3', input='dense1')
        graph.add_output(name='output1', inputs=['dense2', 'dense3'], merge_mode='sum')

        print_model_parameter_names(graph)
        return


    def test_mix(self):
        print("\nTesting when some nodes have names, and others do not")
        graph = containers.Graph()
        graph.add_input(name='input1', ndim=2)
        graph.add_node(Dense(32, 16), name='dense1', input='input1')
        graph.add_node(Dense(32, 4), name='dense2', input='input1')
        graph.add_node(Dense(16, 4), name='dense3', input='dense1')
        graph.add_output(name='output1', inputs=['dense2', 'dense3'], merge_mode='sum')

        seq = Sequential()
        seq.add(Dense(32, 32, name='first_seq_sense'))
        seq.add(Dense(32, 32, name=None))
        seq.add(Convolution2D(32, 32, 32, 32, name=None))
        seq.add(graph)
        seq.add(Dense(4, 4, name='last_seq_dense'))

        print_model_parameter_names(seq)
        return


if __name__ == "__main__":
    t = TestParameterNames()
    t.test_sequential_defaults()
    t.test_graph_defaults()
    t.test_named_nodes()
    t.test_mix()
