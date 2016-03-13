from __future__ import absolute_import
from __future__ import print_function
import pytest
import os
import numpy as np
np.random.seed(1337)

from keras import backend as K
from keras.models import Graph, Sequential, model_from_json, model_from_yaml
from keras.layers.core import Dense, Activation, Merge, Lambda, LambdaMerge, Siamese, add_shared_layer
from keras.layers import containers
from keras.utils.test_utils import get_test_data


batch_size = 32

(X_train_graph, y_train_graph), (X_test_graph, y_test_graph) = get_test_data(nb_train=1000,
                                                                             nb_test=200,
                                                                             input_shape=(32,),
                                                                             classification=False,
                                                                             output_shape=(4,))
(X2_train_graph, y2_train_graph), (X2_test_graph, y2_test_graph) = get_test_data(nb_train=1000,
                                                                                 nb_test=200,
                                                                                 input_shape=(32,),
                                                                                 classification=False,
                                                                                 output_shape=(1,))


def test_graph_fit_generator():
    def data_generator_graph(train):
        while 1:
            if train:
                yield {'input1': X_train_graph, 'output1': y_train_graph}
            else:
                yield {'input1': X_test_graph, 'output1': y_test_graph}

    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))

    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input1')
    graph.add_node(Dense(4), name='dense3', input='dense1')

    graph.add_output(name='output1',
                     inputs=['dense2', 'dense3'],
                     merge_mode='sum')
    graph.compile('rmsprop', {'output1': 'mse'})

    graph.fit_generator(data_generator_graph(True), 1000, nb_epoch=4)
    graph.fit_generator(data_generator_graph(True), 1000, nb_epoch=4,
                        validation_data={'input1': X_test_graph, 'output1': y_test_graph})
    graph.fit_generator(data_generator_graph(True), 1000, nb_epoch=4,
                        validation_data=data_generator_graph(False), nb_val_samples=batch_size * 3)
    graph.fit_generator(data_generator_graph(True), 1000, nb_epoch=4,
                        validation_data=data_generator_graph(False), nb_val_samples=batch_size * 3)
    gen_loss = graph.evaluate_generator(data_generator_graph(True), 128, verbose=0)
    assert(gen_loss < 3.)

    loss = graph.evaluate({'input1': X_test_graph, 'output1': y_test_graph}, verbose=0)
    assert(loss < 3.)

    # test show_accuracy
    graph.fit_generator(data_generator_graph(True), 1000, nb_epoch=4, show_accuracy=True)
    graph.fit_generator(data_generator_graph(True), 1000, nb_epoch=4,
                        validation_data={'input1': X_test_graph, 'output1': y_test_graph}, show_accuracy=True)
    graph.fit_generator(data_generator_graph(True), 1000, nb_epoch=4,
                        validation_data=data_generator_graph(False), nb_val_samples=batch_size * 3, show_accuracy=True)
    graph.fit_generator(data_generator_graph(True), 1000, nb_epoch=4,
                        validation_data=data_generator_graph(False), nb_val_samples=batch_size * 3, show_accuracy=True)
    gen_loss = graph.evaluate_generator(data_generator_graph(True), 128, verbose=0, show_accuracy=True)


def test_1o_1i():
    # test a non-sequential graph with 1 input and 1 output
    np.random.seed(1337)

    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))

    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input1')
    graph.add_node(Dense(4), name='dense3', input='dense1')

    graph.add_output(name='output1',
                     inputs=['dense2', 'dense3'],
                     merge_mode='sum')
    graph.compile('rmsprop', {'output1': 'mse'})

    graph.fit({'input1': X_train_graph, 'output1': y_train_graph},
              nb_epoch=10)
    out = graph.predict({'input1': X_test_graph})
    assert(type(out == dict))
    assert(len(out) == 1)
    loss = graph.test_on_batch({'input1': X_test_graph, 'output1': y_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'output1': y_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'output1': y_test_graph}, verbose=0)
    assert(loss < 2.5)

    # test show_accuracy:
    graph.fit({'input1': X_train_graph, 'output1': y_train_graph},
              nb_epoch=1, show_accuracy=True)
    loss, acc = graph.test_on_batch({'input1': X_test_graph, 'output1': y_test_graph}, accuracy=True)
    loss, acc = graph.train_on_batch({'input1': X_test_graph, 'output1': y_test_graph}, accuracy=True)
    loss, acc = graph.evaluate({'input1': X_test_graph, 'output1': y_test_graph}, verbose=0, show_accuracy=True)

    # test validation split
    graph.fit({'input1': X_train_graph, 'output1': y_train_graph},
              validation_split=0.2, nb_epoch=1)
    # test validation data
    graph.fit({'input1': X_train_graph, 'output1': y_train_graph},
              validation_data={'input1': X_train_graph, 'output1': y_train_graph},
              nb_epoch=1)


def test_1o_1i_2():
    # test a more complex non-sequential graph with 1 input and 1 output
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))

    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2-0', input='input1')
    graph.add_node(Activation('relu'), name='dense2', input='dense2-0')

    graph.add_node(Dense(16), name='dense3', input='dense2')
    graph.add_node(Dense(4), name='dense4', inputs=['dense1', 'dense3'],
                   merge_mode='sum')

    graph.add_output(name='output1', inputs=['dense2', 'dense4'],
                     merge_mode='sum')
    graph.compile('rmsprop', {'output1': 'mse'})

    graph.fit({'input1': X_train_graph, 'output1': y_train_graph},
              nb_epoch=10)
    out = graph.predict({'input1': X_train_graph})
    assert(type(out == dict))
    assert(len(out) == 1)

    loss = graph.test_on_batch({'input1': X_test_graph, 'output1': y_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'output1': y_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'output1': y_test_graph})
    assert(loss < 2.5)

    graph.get_config(verbose=1)
    graph.summary()


def test_1o_2i():
    # test a non-sequential graph with 2 inputs and 1 output
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))
    graph.add_input(name='input2', input_shape=(32,))

    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input2')
    graph.add_node(Dense(4), name='dense3', input='dense1')

    graph.add_output(name='output1', inputs=['dense2', 'dense3'],
                     merge_mode='sum')
    graph.compile('rmsprop', {'output1': 'mse'})

    graph.fit({'input1': X_train_graph, 'input2': X2_train_graph, 'output1': y_train_graph},
              nb_epoch=10)
    out = graph.predict({'input1': X_test_graph, 'input2': X2_test_graph})
    assert(type(out == dict))
    assert(len(out) == 1)

    loss = graph.test_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    assert(loss < 3.0)

    graph.get_config(verbose=1)


def test_siamese_3():
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))
    graph.add_input(name='input2', input_shape=(32,))

    graph.add_shared_node(Dense(16), name='shared', inputs=['input1', 'input2'], merge_mode='sum')
    graph.add_node(Dense(4), name='dense1', input='shared')
    graph.add_node(Dense(4), name='dense2', input='dense1')

    graph.add_output(name='output1', input='dense2')
    graph.compile('rmsprop', {'output1': 'mse'})

    graph.fit({'input1': X_train_graph, 'input2': X2_train_graph, 'output1': y_train_graph},
              nb_epoch=10)
    out = graph.predict({'input1': X_test_graph, 'input2': X2_test_graph})
    assert(type(out == dict))
    assert(len(out) == 1)

    loss = graph.test_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    assert(loss < 3.0)

    graph.get_config(verbose=1)


def test_siamese_4():
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))
    graph.add_input(name='input2', input_shape=(32,))

    graph.add_shared_node(Dense(16), name='shared1', inputs=['input1', 'input2'])
    graph.add_shared_node(Dense(4), name='shared2', inputs=['shared1'])
    graph.add_shared_node(Dense(4), name='shared3', inputs=['shared2'], merge_mode='sum')
    graph.add_node(Dense(4), name='dense', input='shared3')

    graph.add_output(name='output1', input='dense',
                     merge_mode='sum')
    graph.compile('rmsprop', {'output1': 'mse'})

    graph.fit({'input1': X_train_graph, 'input2': X2_train_graph, 'output1': y_train_graph},
              nb_epoch=10)
    out = graph.predict({'input1': X_test_graph, 'input2': X2_test_graph})
    assert(type(out == dict))
    assert(len(out) == 1)

    loss = graph.test_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    assert(loss < 3.0)

    graph.get_config(verbose=1)


def test_siamese_5():
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))
    graph.add_input(name='input2', input_shape=(32,))

    graph.add_shared_node(Dense(16), name='shared1', inputs=['input1', 'input2'])
    graph.add_shared_node(Dense(4), name='shared2', inputs=['shared1'])
    graph.add_shared_node(Dense(4), name='shared3', inputs=['shared2'], outputs=['shared_output1','shared_output2'])
    graph.add_node(Dense(4), name='dense1',  input='shared_output1')
    graph.add_node(Dense(4), name='dense2',  input='shared_output2')

    graph.add_output(name='output1', inputs=['dense1', 'dense2'],
                     merge_mode='sum')
    graph.compile('rmsprop', {'output1': 'mse'})

    graph.fit({'input1': X_train_graph, 'input2': X2_train_graph, 'output1': y_train_graph},
              nb_epoch=10)
    out = graph.predict({'input1': X_test_graph, 'input2': X2_test_graph})
    assert(type(out == dict))
    assert(len(out) == 1)

    loss = graph.test_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'input2': X2_test_graph, 'output1': y_test_graph})
    assert(loss < 3.0)

    graph.get_config(verbose=1)


def test_2o_1i_weights():
    # test a non-sequential graph with 1 input and 2 outputs
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))

    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input1')
    graph.add_node(Dense(1), name='dense3', input='dense1')

    graph.add_output(name='output1', input='dense2')
    graph.add_output(name='output2', input='dense3')
    graph.compile('rmsprop', {'output1': 'mse', 'output2': 'mse'})

    graph.fit({'input1': X_train_graph, 'output1': y_train_graph, 'output2': y2_train_graph},
              nb_epoch=10)
    out = graph.predict({'input1': X_test_graph})
    assert(type(out == dict))
    assert(len(out) == 2)
    loss = graph.test_on_batch({'input1': X_test_graph, 'output1': y_test_graph, 'output2': y2_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'output1': y_test_graph, 'output2': y2_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'output1': y_test_graph, 'output2': y2_test_graph})
    assert(loss < 4.)

    # test weight saving
    fname = 'test_2o_1i_weights_temp.h5'
    graph.save_weights(fname, overwrite=True)

    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))
    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input1')
    graph.add_node(Dense(1), name='dense3', input='dense1')
    graph.add_output(name='output1', input='dense2')
    graph.add_output(name='output2', input='dense3')
    graph.compile('rmsprop', {'output1': 'mse', 'output2': 'mse'})
    graph.load_weights('test_2o_1i_weights_temp.h5')
    os.remove(fname)

    nloss = graph.evaluate({'input1': X_test_graph, 'output1': y_test_graph, 'output2': y2_test_graph})
    assert(loss == nloss)

    # test loss weights
    graph.compile('rmsprop', {'output1': 'mse', 'output2': 'mse'},
                  loss_weights={'output1': 1., 'output2': 2.})
    graph.fit({'input1': X_train_graph, 'output1': y_train_graph, 'output2': y2_train_graph},
              nb_epoch=1)


def test_2o_1i_sample_weights():
    # test a non-sequential graph with 1 input and 2 outputs with sample weights
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))

    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input1')
    graph.add_node(Dense(1), name='dense3', input='dense1')

    graph.add_output(name='output1', input='dense2')
    graph.add_output(name='output2', input='dense3')

    weights1 = np.random.uniform(size=y_train_graph.shape[0])
    weights2 = np.random.uniform(size=y2_train_graph.shape[0])
    weights1_test = np.random.uniform(size=y_test_graph.shape[0])
    weights2_test = np.random.uniform(size=y2_test_graph.shape[0])

    graph.compile('rmsprop', {'output1': 'mse', 'output2': 'mse'})

    graph.fit({'input1': X_train_graph, 'output1': y_train_graph, 'output2': y2_train_graph},
              nb_epoch=10,
              sample_weight={'output1': weights1, 'output2': weights2})
    out = graph.predict({'input1': X_test_graph})
    assert(type(out == dict))
    assert(len(out) == 2)
    loss = graph.test_on_batch({'input1': X_test_graph, 'output1': y_test_graph, 'output2': y2_test_graph},
                               sample_weight={'output1': weights1_test, 'output2': weights2_test})
    loss = graph.train_on_batch({'input1': X_train_graph, 'output1': y_train_graph, 'output2': y2_train_graph},
                                sample_weight={'output1': weights1, 'output2': weights2})
    loss = graph.evaluate({'input1': X_train_graph, 'output1': y_train_graph, 'output2': y2_train_graph},
                          sample_weight={'output1': weights1, 'output2': weights2})


def test_recursive():
    # test layer-like API

    graph = containers.Graph()
    graph.add_input(name='input1', input_shape=(32,))
    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input1')
    graph.add_node(Dense(4), name='dense3', input='dense1')
    graph.add_output(name='output1', inputs=['dense2', 'dense3'],
                     merge_mode='sum')

    seq = Sequential()
    seq.add(Dense(32, input_shape=(32,)))
    seq.add(graph)
    seq.add(Dense(4))

    seq.compile('rmsprop', 'mse')

    seq.fit(X_train_graph, y_train_graph, batch_size=10, nb_epoch=10)
    loss = seq.evaluate(X_test_graph, y_test_graph)
    assert(loss < 2.5)

    loss = seq.evaluate(X_test_graph, y_test_graph, show_accuracy=True)
    seq.predict(X_test_graph)
    seq.get_config(verbose=1)


def test_create_output():
    # test create_output argument
    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))

    graph.add_node(Dense(16), name='dense1', input='input1')
    graph.add_node(Dense(4), name='dense2', input='input1')
    graph.add_node(Dense(4), name='dense3', input='dense1')
    graph.add_node(Dense(4), name='output1', inputs=['dense2', 'dense3'],
                   merge_mode='sum', create_output=True)
    graph.compile('rmsprop', {'output1': 'mse'})

    history = graph.fit({'input1': X_train_graph, 'output1': y_train_graph},
                        nb_epoch=10)
    out = graph.predict({'input1': X_test_graph})
    assert(type(out == dict))
    assert(len(out) == 1)

    loss = graph.test_on_batch({'input1': X_test_graph, 'output1': y_test_graph})
    loss = graph.train_on_batch({'input1': X_test_graph, 'output1': y_test_graph})
    loss = graph.evaluate({'input1': X_test_graph, 'output1': y_test_graph})
    assert(loss < 2.5)

    # test serialization
    config = graph.to_json()
    del graph
    graph = model_from_json(config)


def test_count_params():
    # test count params

    nb_units = 100
    nb_classes = 2

    graph = Graph()
    graph.add_input(name='input1', input_shape=(32,))
    graph.add_input(name='input2', input_shape=(32,))
    graph.add_node(Dense(nb_units),
                   name='dense1', input='input1')
    graph.add_node(Dense(nb_classes),
                   name='dense2', input='input2')
    graph.add_node(Dense(nb_classes),
                   name='dense3', input='dense1')
    graph.add_output(name='output', inputs=['dense2', 'dense3'],
                     merge_mode='sum')

    n = 32 * nb_units + nb_units
    n += 32 * nb_classes + nb_classes
    n += nb_units * nb_classes + nb_classes

    assert(n == graph.count_params())

    graph.compile('rmsprop', {'output': 'binary_crossentropy'})

    assert(n == graph.count_params())


if __name__ == '__main__':
    pytest.main([__file__])
