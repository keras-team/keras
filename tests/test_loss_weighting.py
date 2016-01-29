from __future__ import absolute_import
from __future__ import print_function
import pytest
import numpy as np
np.random.seed(1337)

from keras.utils.test_utils import get_test_data
from keras.models import Sequential, Graph
from keras.layers import Dense, Activation, RepeatVector, TimeDistributedDense, GRU
from keras.utils import np_utils

nb_classes = 10
batch_size = 128
nb_epoch = 15
weighted_class = 5
standard_weight = 1
high_weight = 10
train_samples = 5000
test_samples = 1000
timesteps = 3
input_dim = 10
loss = 'mse'

(X_train, y_train), (X_test, y_test) = get_test_data(nb_train=train_samples,
                                                     nb_test=test_samples,
                                                     input_shape=(input_dim,),
                                                     classification=True,
                                                     nb_class=nb_classes)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
test_ids = np.where(y_test == np.array(weighted_class))[0]

class_weight = dict([(i, standard_weight) for i in range(nb_classes)])
class_weight[weighted_class] = high_weight

sample_weight = np.ones((y_train.shape[0])) * standard_weight
sample_weight[y_train == weighted_class] = high_weight

temporal_X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
temporal_X_train = np.repeat(temporal_X_train, timesteps, axis=1)
temporal_X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))
temporal_X_test = np.repeat(temporal_X_test, timesteps, axis=1)

temporal_Y_train = np.reshape(Y_train, (len(Y_train), 1, Y_train.shape[1]))
temporal_Y_train = np.repeat(temporal_Y_train, timesteps, axis=1)
temporal_Y_test = np.reshape(Y_test, (len(Y_test), 1, Y_test.shape[1]))
temporal_Y_test = np.repeat(temporal_Y_test, timesteps, axis=1)

temporal_sample_weight = np.reshape(sample_weight, (len(sample_weight), 1))
temporal_sample_weight = np.repeat(temporal_sample_weight, timesteps, axis=1)


def create_sequential_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model


def create_graph_model():
    model = Graph()
    model.add_input(name='input', input_shape=(input_dim,))
    model.add_node(Dense(32, activation='relu'), name='d1', input='input')
    model.add_node(Dense(nb_classes, activation='softmax'), name='d2', input='d1')
    model.add_output(name='output', input='d2')
    return model


def create_temporal_sequential_model():
    model = Sequential()
    model.add(GRU(32, input_shape=(timesteps, input_dim), return_sequences=True))
    model.add(TimeDistributedDense(nb_classes))
    model.add(Activation('softmax'))
    return model


def create_temporal_graph_model():
    model = Graph()
    model.add_input(name='input', input_shape=(timesteps, input_dim))
    model.add_node(GRU(32, return_sequences=True),
                   name='d1', input='input')
    model.add_node(TimeDistributedDense(nb_classes, activation='softmax'),
                   name='d2', input='d1')
    model.add_output(name='output', input='d2')
    return model


def _test_weights_sequential(model, class_weight=None, sample_weight=None,
                             X_train=X_train, Y_train=Y_train,
                             X_test=X_test, Y_test=Y_test):
    if sample_weight is not None:
        model.fit(X_train, Y_train, batch_size=batch_size,
                  nb_epoch=nb_epoch // 3, verbose=0,
                  class_weight=class_weight, sample_weight=sample_weight)
        model.fit(X_train, Y_train, batch_size=batch_size,
                  nb_epoch=nb_epoch // 3, verbose=0,
                  class_weight=class_weight, sample_weight=sample_weight,
                  validation_split=0.1)
        model.fit(X_train, Y_train, batch_size=batch_size,
                  nb_epoch=nb_epoch // 3, verbose=0,
                  class_weight=class_weight, sample_weight=sample_weight,
                  validation_data=(X_train, Y_train, sample_weight))
    else:
        model.fit(X_train, Y_train, batch_size=batch_size,
                  nb_epoch=nb_epoch // 2, verbose=0,
                  class_weight=class_weight, sample_weight=sample_weight)
        model.fit(X_train, Y_train, batch_size=batch_size,
                  nb_epoch=nb_epoch // 2, verbose=0,
                  class_weight=class_weight, sample_weight=sample_weight,
                  validation_split=0.1)

    model.train_on_batch(X_train[:32], Y_train[:32],
                         class_weight=class_weight,
                         sample_weight=sample_weight[:32] if sample_weight is not None else None)
    model.test_on_batch(X_train[:32], Y_train[:32],
                        sample_weight=sample_weight[:32] if sample_weight is not None else None)
    score = model.evaluate(X_test[test_ids, :], Y_test[test_ids, :], verbose=0)
    return score


def _test_weights_graph(model, class_weight=None, sample_weight=None,
                        X_train=X_train, Y_train=Y_train,
                        X_test=X_test, Y_test=Y_test):
    model.fit({'input': X_train, 'output': Y_train},
              batch_size=batch_size, nb_epoch=nb_epoch // 2, verbose=0,
              class_weight={'output': class_weight},
              sample_weight={'output': sample_weight})
    model.fit({'input': X_train, 'output': Y_train},
              batch_size=batch_size, nb_epoch=nb_epoch // 2, verbose=0,
              class_weight={'output': class_weight},
              sample_weight={'output': sample_weight}, validation_split=0.1)

    model.train_on_batch({'input': X_train[:32], 'output': Y_train[:32]},
                         class_weight={'output': class_weight},
                         sample_weight={'output': sample_weight[:32] if sample_weight is not None else None})
    model.test_on_batch({'input': X_train[:32], 'output': Y_train[:32]},
                        sample_weight={'output': sample_weight[:32] if sample_weight is not None else None})
    score = model.evaluate({'input': X_test[test_ids, :],
                            'output': Y_test[test_ids, :]},
                           verbose=0)
    return score


# no weights: reference point
model = create_sequential_model()
model.compile(loss=loss, optimizer='rmsprop')
standard_score_sequential = _test_weights_sequential(model)

model = create_graph_model()
model.compile(loss={'output': loss}, optimizer='rmsprop')
standard_score_graph = _test_weights_graph(model)


def test_sequential_class_weights():
    model = create_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop')
    score = _test_weights_sequential(model, class_weight=class_weight)
    assert(score < standard_score_sequential)


def test_sequential_sample_weights():
    model = create_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop')
    score = _test_weights_sequential(model, sample_weight=sample_weight)
    assert(score < standard_score_sequential)


def test_sequential_temporal_sample_weights():
    model = create_temporal_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop',
                  sample_weight_mode='temporal')
    score = _test_weights_sequential(model,
                                     sample_weight=temporal_sample_weight,
                                     X_train=temporal_X_train,
                                     X_test=temporal_X_test,
                                     Y_train=temporal_Y_train,
                                     Y_test=temporal_Y_test)
    assert(score < standard_score_sequential)

    # a twist: sample-wise weights with temporal output
    model = create_temporal_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop',
                  sample_weight_mode=None)
    score = _test_weights_sequential(model,
                                     sample_weight=sample_weight,
                                     X_train=temporal_X_train,
                                     X_test=temporal_X_test,
                                     Y_train=temporal_Y_train,
                                     Y_test=temporal_Y_test)
    assert(score < standard_score_sequential)


def test_graph_class_weights():
    model = create_graph_model()
    model.compile(loss={'output': loss}, optimizer='rmsprop')
    score = _test_weights_graph(model, class_weight=class_weight)
    assert(score < standard_score_graph)


def test_graph_sample_weights():
    model = create_graph_model()
    model.compile(loss={'output': loss}, optimizer='rmsprop')
    score = _test_weights_graph(model, sample_weight=sample_weight)
    assert(score < standard_score_graph)


def test_graph_temporal_sample_weight():
    model = create_temporal_graph_model()
    model.compile(loss={'output': loss}, optimizer='rmsprop',
                  sample_weight_modes={'output': 'temporal'})
    score = _test_weights_graph(model,
                                sample_weight=temporal_sample_weight,
                                X_train=temporal_X_train,
                                X_test=temporal_X_test,
                                Y_train=temporal_Y_train,
                                Y_test=temporal_Y_test)
    assert(score < standard_score_graph)


if __name__ == '__main__':
    pytest.main([__file__])
