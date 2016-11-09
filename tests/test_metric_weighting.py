from __future__ import absolute_import
from __future__ import print_function
import pytest
import numpy as np
np.random.seed(1337)

from keras.utils.test_utils import get_test_data
from keras.models import Sequential
from keras.layers import Dense, Activation, RepeatVector, TimeDistributedDense, GRU
from keras.utils import np_utils
from keras.utils.test_utils import keras_test

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
weight_ids = np.where(y_test == np.array(weighted_class))[0]
unweight_ids = np.where(y_test != np.array(weighted_class))[0]
test_ids = np.hstack(zip(unweight_ids, weight_ids))

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


def create_temporal_sequential_model():
    model = Sequential()
    model.add(GRU(32, input_shape=(timesteps, input_dim), return_sequences=True))
    model.add(TimeDistributedDense(nb_classes))
    model.add(Activation('softmax'))
    return model


def _fit_weights_sequential(model, class_weight=None, sample_weight=None,
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

def setup_module():
    # no evaluation weights: reference point
    model = create_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop', metrics=['acc',
                                                           {"metric": "accuracy",
                                                            "weighted": "True"}])
    _fit_weights_sequential(model, sample_weight=sample_weight)
    (loss_val, acc, weighted_acc) = model.evaluate(X_test[test_ids, :], Y_test[test_ids, :])

@keras_test
def test_sequential_sample_weights_unchanged_acc():
    model = create_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop', metrics=['acc'])
    _fit_weights_sequential(model,sample_weight=sample_weight)
    test_weights = np.ones(X_test.shape[:-1])
    (test_loss_val, test_simple_acc) = model.evaluate(X_test[test_ids, :], Y_test[test_ids, :])
    (test_loss, test_acc) = model.evaluate(X_test[test_ids, :], Y_test[test_ids, :],
                                           sample_weight=test_weights[test_ids])
    assert(np.isclose(test_loss_val, test_loss))
    assert(np.isclose(test_simple_acc, test_acc))


@keras_test
def test_sequential_sample_weights_ones_match_acc():
    test_weights = np.ones(X_test.shape[:-1])
    (test_loss, test_acc, test_weighted_acc) = model.evaluate(X_test[test_ids, :],
                                                              Y_test[test_ids, :],
                                                              sample_weight=test_weights[test_ids])
    assert(np.isclose(loss_val, test_loss))
    assert(np.isclose(acc, test_acc))
    assert(np.isclose(acc, test_weighted_acc))
    assert(np.isclose(weighted_acc, test_weighted_acc))


@keras_test
def test_sequential_sample_weights_change_acc_with_fewer():
    test_weights = np.ones(X_test.shape[:-1])
    # prioritize on weighted
    test_weights[test_weights==unweight_ids] = 0
    (test_loss, test_acc, test_weighted_acc) = model.evaluate(X_test[test_ids, :],
                                                              Y_test[test_ids, :],
                                                              sample_weight=test_weights[test_ids])
    assert(test_acc != test_weighted_acc)


@keras_test
def test_sequential_sample_weights_change_acc_with_fewer_unweighted():
    test_weights = np.ones(X_test.shape[:-1])
    # prioritize on unweighted
    test_weights[test_weights==weight_ids] = 0
    (test_loss, test_acc, test_weighted_acc) = model.evaluate(X_test[test_ids, :],
                                                              Y_test[test_ids, :],
                                                              sample_weight=test_weights[test_ids])
    assert(test_acc != test_weighted_acc)


@keras_test
def test_sequential_temporal_sample_weights():
    model = create_temporal_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop', metrics=['acc',
                                                           {"metric": "accuracy",
                                                            "weighted": "True"}],
                  sample_weight_mode='temporal')
    _fit_weights_sequential(model, sample_weight=temporal_sample_weight,
                            X_train=temporal_X_train,
                            X_test=temporal_X_test,
                            Y_train=temporal_Y_train,
                            Y_test=temporal_Y_test)
    test_weights = np.ones(temporal_X_test.shape[:-1])
    (test_loss, test_acc, test_weighted_acc) = model.evaluate(temporal_X_test[test_ids, :],
                                                              temporal_Y_test[test_ids, :],
                                                              sample_weight=test_weights[test_ids, :])
    assert(np.isclose(test_acc, test_weighted_acc))


@keras_test
def test_sequential_temporal_sample_weights_change_acc_with_fewer():
    model = create_temporal_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop', metrics=['acc',
                                                           {"metric": "accuracy",
                                                            "weighted": "True"}],
                  sample_weight_mode='temporal')
    _fit_weights_sequential(model, sample_weight=temporal_sample_weight,
                            X_train=temporal_X_train,
                            X_test=temporal_X_test,
                            Y_train=temporal_Y_train,
                            Y_test=temporal_Y_test)
    test_weights = np.ones(temporal_X_test.shape[:-1])
    test_weights[test_weights==unweight_ids, :] = 0
    (test_loss, test_acc, test_weighted_acc) = model.evaluate(temporal_X_test[test_ids, :],
                                                              temporal_Y_test[test_ids, :],
                                                              sample_weight=test_weights[test_ids, :])
    assert(test_acc != test_weighted_acc)


@keras_test
def test_sequential_temporal_sample_weights_change_acc_with_subvector():
    model = create_temporal_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop', metrics=['acc',
                                                           {"metric": "accuracy",
                                                            "weighted": "True"}],
                  sample_weight_mode='temporal')
    _fit_weights_sequential(model, sample_weight=temporal_sample_weight,
                            X_train=temporal_X_train,
                            X_test=temporal_X_test,
                            Y_train=temporal_Y_train,
                            Y_test=temporal_Y_test)
    test_weights = np.ones(temporal_X_test.shape[:-1])
    # this tests that we can weigh subsequences in temporal inputs selectively
    test_weights[test_weights==unweight_ids, 1:] = 0
    (test_loss, test_acc, test_weighted_acc) = model.evaluate(temporal_X_test[test_ids, :],
                                                              temporal_Y_test[test_ids, :],
                                                              sample_weight=test_weights[test_ids, :])
    assert(test_acc != test_weighted_acc)


@keras_test
def test_sequential_temporal_sample_weights_with_none_mode():
    model = create_temporal_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop', metrics=['acc',
                                                           {"metric": "accuracy",
                                                            "weighted": "True"}],
                  sample_weight_mode=None)
    _fit_weights_sequential(model, sample_weight=sample_weight,
                            X_train=temporal_X_train,
                            X_test=temporal_X_test,
                            Y_train=temporal_Y_train,
                            Y_test=temporal_Y_test)
    test_weights = np.ones(X_test.shape[:-1])
    (test_loss, test_acc, test_weighted_acc) = model.evaluate(temporal_X_test[test_ids, :],
                                                              temporal_Y_test[test_ids, :],
                                                              sample_weight=test_weights[test_ids])
    assert(np.isclose(test_acc, test_weighted_acc))

if __name__ == '__main__':
    pytest.main([__file__])
