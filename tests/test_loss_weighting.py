from __future__ import absolute_import
from __future__ import print_function
import pytest
import numpy as np

from keras import backend as K
from keras.utils.test_utils import get_test_data
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, GRU, TimeDistributed, Input
from keras.utils import np_utils
from keras.utils.test_utils import keras_test
from numpy.testing import assert_almost_equal, assert_array_almost_equal

num_classes = 10
batch_size = 128
epochs = 15
weighted_class = 5
high_weight = 10
train_samples = 5000
test_samples = 1000
timesteps = 3
input_dim = 10
loss = 'mse'
loss_full_name = 'mean_squared_error'
standard_weight = 1
standard_score_sequential = 0.5

decimal_precision = {
    'cntk': 2,
    'theano': 6,
    'tensorflow': 6
}


def _get_test_data():
    np.random.seed(1337)
    (x_train, y_train), (x_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=num_classes)
    int_y_test = y_test.copy()
    int_y_train = y_train.copy()
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    test_ids = np.where(int_y_test == np.array(weighted_class))[0]

    class_weight = dict([(i, standard_weight) for i in range(num_classes)])
    class_weight[weighted_class] = high_weight

    sample_weight = np.ones((y_train.shape[0])) * standard_weight
    sample_weight[int_y_train == weighted_class] = high_weight

    return (x_train, y_train), (x_test, y_test), (sample_weight, class_weight, test_ids)


def create_sequential_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def create_temporal_sequential_model():
    model = Sequential()
    model.add(GRU(32, input_shape=(timesteps, input_dim), return_sequences=True))
    model.add(TimeDistributed(Dense(num_classes)))
    model.add(Activation('softmax'))
    return model


@keras_test
def test_sequential_class_weights():
    model = create_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop')

    (x_train, y_train), (x_test, y_test), (sample_weight, class_weight, test_ids) = _get_test_data()

    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs // 3, verbose=0,
              class_weight=class_weight,
              validation_data=(x_train, y_train, sample_weight))
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs // 2, verbose=0,
              class_weight=class_weight)
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs // 2, verbose=0,
              class_weight=class_weight,
              validation_split=0.1)

    model.train_on_batch(x_train[:32], y_train[:32],
                         class_weight=class_weight)
    score = model.evaluate(x_test[test_ids, :], y_test[test_ids, :], verbose=0)
    assert(score < standard_score_sequential)


@keras_test
def test_sequential_sample_weights():
    model = create_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop')

    (x_train, y_train), (x_test, y_test), (sample_weight, class_weight, test_ids) = _get_test_data()

    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs // 3, verbose=0,
              sample_weight=sample_weight)
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs // 3, verbose=0,
              sample_weight=sample_weight,
              validation_split=0.1)

    model.train_on_batch(x_train[:32], y_train[:32],
                         sample_weight=sample_weight[:32])
    model.test_on_batch(x_train[:32], y_train[:32],
                        sample_weight=sample_weight[:32])
    score = model.evaluate(x_test[test_ids, :], y_test[test_ids, :], verbose=0)
    assert(score < standard_score_sequential)


@keras_test
def test_sequential_temporal_sample_weights():
    (x_train, y_train), (x_test, y_test), (sample_weight, class_weight, test_ids) = _get_test_data()

    temporal_x_train = np.reshape(x_train, (len(x_train), 1, x_train.shape[1]))
    temporal_x_train = np.repeat(temporal_x_train, timesteps, axis=1)
    temporal_x_test = np.reshape(x_test, (len(x_test), 1, x_test.shape[1]))
    temporal_x_test = np.repeat(temporal_x_test, timesteps, axis=1)

    temporal_y_train = np.reshape(y_train, (len(y_train), 1, y_train.shape[1]))
    temporal_y_train = np.repeat(temporal_y_train, timesteps, axis=1)
    temporal_y_test = np.reshape(y_test, (len(y_test), 1, y_test.shape[1]))
    temporal_y_test = np.repeat(temporal_y_test, timesteps, axis=1)

    temporal_sample_weight = np.reshape(sample_weight, (len(sample_weight), 1))
    temporal_sample_weight = np.repeat(temporal_sample_weight, timesteps, axis=1)

    model = create_temporal_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop',
                  sample_weight_mode='temporal')

    model.fit(temporal_x_train, temporal_y_train, batch_size=batch_size,
              epochs=epochs // 3, verbose=0,
              sample_weight=temporal_sample_weight)
    model.fit(temporal_x_train, temporal_y_train, batch_size=batch_size,
              epochs=epochs // 3, verbose=0,
              sample_weight=temporal_sample_weight,
              validation_split=0.1)

    model.train_on_batch(temporal_x_train[:32], temporal_y_train[:32],
                         sample_weight=temporal_sample_weight[:32])
    model.test_on_batch(temporal_x_train[:32], temporal_y_train[:32],
                        sample_weight=temporal_sample_weight[:32])
    score = model.evaluate(temporal_x_test[test_ids], temporal_y_test[test_ids], verbose=0)
    assert(score < standard_score_sequential)


@keras_test
def test_weighted_metrics_with_sample_weight():
    decimal = decimal_precision[K.backend()]

    model = create_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop', metrics=[loss], weighted_metrics=[loss])

    (x_train, y_train), (x_test, y_test), (sample_weight, class_weight, test_ids) = _get_test_data()

    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs // 3, verbose=0,
                        sample_weight=sample_weight)

    h = history.history
    assert_array_almost_equal(h['loss'], h['weighted_' + loss_full_name], decimal=decimal)

    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs // 3, verbose=0,
                        sample_weight=sample_weight,
                        validation_split=0.1)

    h = history.history
    assert_almost_equal(h['val_loss'], h['val_weighted_' + loss_full_name], decimal=decimal)

    model.train_on_batch(x_train[:32], y_train[:32],
                         sample_weight=sample_weight[:32])
    model.test_on_batch(x_train[:32], y_train[:32],
                        sample_weight=sample_weight[:32])

    test_sample_weight = np.ones((y_test.shape[0])) * standard_weight
    test_sample_weight[test_ids] = high_weight

    scores = model.evaluate(x_test, y_test, verbose=0, sample_weight=test_sample_weight)
    loss_score, metric_score, weighted_metric_score = scores

    assert loss_score < standard_score_sequential
    assert loss_score != metric_score
    assert_almost_equal(loss_score, weighted_metric_score, decimal=decimal)


@keras_test
def test_weighted_metrics_with_no_sample_weight():
    decimal = decimal_precision[K.backend()]

    model = create_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop', metrics=[loss], weighted_metrics=[loss])

    (x_train, y_train), (x_test, y_test), _ = _get_test_data()

    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs // 3, verbose=0)

    h = history.history
    assert_array_almost_equal(h['loss'], h[loss_full_name], decimal=decimal)
    assert_array_almost_equal(h['loss'], h['weighted_' + loss_full_name], decimal=decimal)

    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs // 3, verbose=0, validation_split=0.1)

    h = history.history
    assert_array_almost_equal(h['val_loss'], h['val_' + loss_full_name], decimal=decimal)
    assert_array_almost_equal(h['val_loss'], h['val_weighted_' + loss_full_name], decimal=decimal)

    model.train_on_batch(x_train[:32], y_train[:32])
    model.test_on_batch(x_train[:32], y_train[:32])

    scores = model.evaluate(x_test, y_test, verbose=0)
    loss_score, metric_score, weighted_metric_score = scores

    assert_almost_equal(loss_score, metric_score, decimal=decimal)
    assert_almost_equal(loss_score, weighted_metric_score, decimal=decimal)


@keras_test
def test_weighted_metrics_with_weighted_accuracy_metric():
    model = create_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop', metrics=['acc'], weighted_metrics=['acc'])

    (x_train, y_train), _, (sample_weight, _, _) = _get_test_data()

    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs // 3, verbose=0,
                        sample_weight=sample_weight)

    assert history.history['acc'] != history.history['weighted_acc']


@keras_test
def test_weighted_metrics_with_multiple_outputs():
    decimal = decimal_precision[K.backend()]

    inputs = Input(shape=(5,))
    x = Dense(5)(inputs)
    output1 = Dense(1, name='output1')(x)
    output2 = Dense(1, name='output2')(x)

    model = Model(inputs=inputs, outputs=[output1, output2])

    metrics = {'output1': [loss], 'output2': [loss]}
    weighted_metrics = {'output2': [loss]}
    loss_map = {'output1': loss, 'output2': loss}

    model.compile(loss=loss_map, optimizer='sgd', metrics=metrics, weighted_metrics=weighted_metrics)

    x = np.array([[1, 1, 1, 1, 1]])
    y = {'output1': np.array([0]), 'output2': np.array([1])}
    weight = 5

    history = model.fit(x, y, sample_weight={'output2': np.array([weight])})

    unweighted_metric = history.history['output2_' + loss_full_name][0]
    weighted_metric = history.history['output2_weighted_' + loss_full_name][0]

    assert_almost_equal(unweighted_metric * weight, weighted_metric, decimal=decimal)


@keras_test
def test_class_weight_wrong_classes():
    model = create_sequential_model()
    model.compile(loss=loss, optimizer='rmsprop')

    (x_train, y_train), (x_test, y_test), (sample_weight, class_weight, test_ids) = _get_test_data()

    del class_weight[1]
    with pytest.raises(ValueError):
        model.fit(x_train, y_train,
                  epochs=0, verbose=0, class_weight=class_weight)


if __name__ == '__main__':
    pytest.main([__file__])
