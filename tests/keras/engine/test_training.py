import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.layers import Dense, Dropout
from keras.engine.topology import merge, Input
from keras.engine.training import Model, check_loss_and_target_compatibility
from keras.models import Sequential
from keras import backend as K
from keras.utils.test_utils import keras_test
from keras.callbacks import LambdaCallback


@keras_test
def test_model_methods():
    a = Input(shape=(3,), name='input_a')
    b = Input(shape=(3,), name='input_b')

    a_2 = Dense(4, name='dense_1')(a)
    dp = Dropout(0.5, name='dropout')
    b_2 = dp(b)

    model = Model([a, b], [a_2, b_2])

    optimizer = 'rmsprop'
    loss = 'mse'
    loss_weights = [1., 0.5]
    model.compile(optimizer, loss, metrics=[], loss_weights=loss_weights,
                  sample_weight_mode=None)

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_a_np = np.random.random((10, 4))
    output_b_np = np.random.random((10, 3))

    # test train_on_batch
    out = model.train_on_batch([input_a_np, input_b_np],
                               [output_a_np, output_b_np])
    out = model.train_on_batch({'input_a': input_a_np, 'input_b': input_b_np},
                               [output_a_np, output_b_np])
    out = model.train_on_batch({'input_a': input_a_np, 'input_b': input_b_np},
                               {'dense_1': output_a_np, 'dropout': output_b_np})

    # test fit
    out = model.fit([input_a_np, input_b_np],
                    [output_a_np, output_b_np], nb_epoch=1, batch_size=4)
    out = model.fit({'input_a': input_a_np, 'input_b': input_b_np},
                    [output_a_np, output_b_np], nb_epoch=1, batch_size=4)
    out = model.fit({'input_a': input_a_np, 'input_b': input_b_np},
                    {'dense_1': output_a_np, 'dropout': output_b_np},
                    nb_epoch=1, batch_size=4)

    # test validation_split
    out = model.fit([input_a_np, input_b_np],
                    [output_a_np, output_b_np],
                    nb_epoch=1, batch_size=4, validation_split=0.5)
    out = model.fit({'input_a': input_a_np, 'input_b': input_b_np},
                    [output_a_np, output_b_np],
                    nb_epoch=1, batch_size=4, validation_split=0.5)
    out = model.fit({'input_a': input_a_np, 'input_b': input_b_np},
                    {'dense_1': output_a_np, 'dropout': output_b_np},
                    nb_epoch=1, batch_size=4, validation_split=0.5)

    # test validation data
    out = model.fit([input_a_np, input_b_np],
                    [output_a_np, output_b_np],
                    nb_epoch=1, batch_size=4,
                    validation_data=([input_a_np, input_b_np], [output_a_np, output_b_np]))
    out = model.fit({'input_a': input_a_np, 'input_b': input_b_np},
                    [output_a_np, output_b_np],
                    nb_epoch=1, batch_size=4, validation_split=0.5,
                    validation_data=({'input_a': input_a_np, 'input_b': input_b_np}, [output_a_np, output_b_np]))
    out = model.fit({'input_a': input_a_np, 'input_b': input_b_np},
                    {'dense_1': output_a_np, 'dropout': output_b_np},
                    nb_epoch=1, batch_size=4, validation_split=0.5,
                    validation_data=({'input_a': input_a_np, 'input_b': input_b_np}, {'dense_1': output_a_np, 'dropout': output_b_np}))

    # test_on_batch
    out = model.test_on_batch([input_a_np, input_b_np],
                              [output_a_np, output_b_np])
    out = model.test_on_batch({'input_a': input_a_np, 'input_b': input_b_np},
                              [output_a_np, output_b_np])
    out = model.test_on_batch({'input_a': input_a_np, 'input_b': input_b_np},
                              {'dense_1': output_a_np, 'dropout': output_b_np})

    # predict_on_batch
    out = model.predict_on_batch([input_a_np, input_b_np])
    out = model.predict_on_batch({'input_a': input_a_np, 'input_b': input_b_np})

    # predict, evaluate
    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_a_np = np.random.random((10, 4))
    output_b_np = np.random.random((10, 3))

    out = model.evaluate([input_a_np, input_b_np], [output_a_np, output_b_np], batch_size=4)
    out = model.predict([input_a_np, input_b_np], batch_size=4)

    # with sample_weight
    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_a_np = np.random.random((10, 4))
    output_b_np = np.random.random((10, 3))

    sample_weight = [None, np.random.random((10,))]
    out = model.train_on_batch([input_a_np, input_b_np],
                               [output_a_np, output_b_np],
                               sample_weight=sample_weight)

    out = model.test_on_batch([input_a_np, input_b_np],
                              [output_a_np, output_b_np],
                              sample_weight=sample_weight)

    # test accuracy metric
    model.compile(optimizer, loss, metrics=['acc'],
                  sample_weight_mode=None)

    out = model.train_on_batch([input_a_np, input_b_np],
                               [output_a_np, output_b_np])
    assert len(out) == 5
    out = model.test_on_batch([input_a_np, input_b_np],
                              [output_a_np, output_b_np])
    assert len(out) == 5

    # this should also work
    model.compile(optimizer, loss, metrics={'dense_1': 'acc'},
                  sample_weight_mode=None)

    out = model.train_on_batch([input_a_np, input_b_np],
                               [output_a_np, output_b_np])
    assert len(out) == 4
    out = model.test_on_batch([input_a_np, input_b_np],
                              [output_a_np, output_b_np])
    assert len(out) == 4

    # and this as well
    model.compile(optimizer, loss, metrics={'dense_1': ['acc']},
                  sample_weight_mode=None)

    out = model.train_on_batch([input_a_np, input_b_np],
                               [output_a_np, output_b_np])
    assert len(out) == 4
    out = model.test_on_batch([input_a_np, input_b_np],
                              [output_a_np, output_b_np])
    assert len(out) == 4

    # test starting from non-zero initial epoch
    trained_epochs = []

    def on_epoch_begin(epoch, logs):
        trained_epochs.append(epoch)
    tracker_cb = LambdaCallback(on_epoch_begin=on_epoch_begin)
    out = model.fit([input_a_np, input_b_np],
                    [output_a_np, output_b_np], nb_epoch=5, batch_size=4,
                    initial_epoch=2, callbacks=[tracker_cb])
    assert trained_epochs == [2, 3, 4]

    # test starting from non-zero initial epoch for generator too
    trained_epochs = []

    def gen_data(batch_sz):
        while True:
            yield ([np.random.random((batch_sz, 3)), np.random.random((batch_sz, 3))],
                   [np.random.random((batch_sz, 4)), np.random.random((batch_sz, 3))])
    out = model.fit_generator(gen_data(4), samples_per_epoch=10, nb_epoch=5,
                              initial_epoch=2, callbacks=[tracker_cb])
    assert trained_epochs == [2, 3, 4]

    # test with a custom metric function
    mse = lambda y_true, y_pred: K.mean(K.pow(y_true - y_pred, 2))

    def mse_powers(y_true, y_pred):
        m = mse(y_true, y_pred)
        return {
            'mse_squared': K.pow(m, 2),
            'mse_cubed': K.pow(m, 3)
        }

    model.compile(optimizer, loss, metrics=[mse, mse_powers],
                  sample_weight_mode=None)

    out = model.train_on_batch([input_a_np, input_b_np],
                               [output_a_np, output_b_np])
    out_len = 1 + 2 * 4  # total loss, per layer: loss + 3 metrics
    assert len(out) == out_len
    out = model.test_on_batch([input_a_np, input_b_np],
                              [output_a_np, output_b_np])
    assert len(out) == out_len

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_a_np = np.random.random((10, 4))
    output_b_np = np.random.random((10, 3))

    out = model.fit([input_a_np, input_b_np], [output_a_np, output_b_np], batch_size=4, nb_epoch=1)
    out = model.evaluate([input_a_np, input_b_np], [output_a_np, output_b_np], batch_size=4)
    out = model.predict([input_a_np, input_b_np], batch_size=4)


@keras_test
def test_trainable_argument():
    x = np.random.random((5, 3))
    y = np.random.random((5, 2))

    model = Sequential()
    model.add(Dense(2, input_dim=3, trainable=False))
    model.compile('rmsprop', 'mse')
    out = model.predict(x)
    model.train_on_batch(x, y)
    out_2 = model.predict(x)
    assert_allclose(out, out_2)

    # test with nesting
    input = Input(shape=(3,))
    output = model(input)
    model = Model(input, output)
    model.compile('rmsprop', 'mse')
    out = model.predict(x)
    model.train_on_batch(x, y)
    out_2 = model.predict(x)
    assert_allclose(out, out_2)


@keras_test
def test_check_not_failing():
    a = np.random.random((2, 1, 3))
    check_loss_and_target_compatibility([a], [K.categorical_crossentropy], [a.shape])
    check_loss_and_target_compatibility([a], [K.categorical_crossentropy], [(2, None, 3)])


@keras_test
def test_check_last_is_one():
    a = np.random.random((2, 3, 1))
    with pytest.raises(Exception) as exc:
        check_loss_and_target_compatibility([a], [K.categorical_crossentropy], [a.shape])

    assert "You are passing a target array" in str(exc)


@keras_test
def test_check_bad_shape():
    a = np.random.random((2, 3, 5))
    with pytest.raises(Exception) as exc:
        check_loss_and_target_compatibility([a], [K.categorical_crossentropy], [(2, 3, 6)])

    assert "targets to have the same shape" in str(exc)


if __name__ == '__main__':
    pytest.main([__file__])
