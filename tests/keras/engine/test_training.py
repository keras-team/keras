import pytest
import numpy as np
from numpy.testing import assert_allclose
import sys
import scipy.sparse as sparse

from keras.layers import Dense, Dropout
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.engine.training import Model
from keras.engine.training import _check_loss_and_target_compatibility
from keras.engine.training import _weighted_masked_objective
from keras.engine.training import _check_array_lengths
from keras.engine.training import _slice_arrays
from keras.models import Sequential
from keras import backend as K
from keras.utils import Sequence
from keras.utils.test_utils import keras_test
from keras.callbacks import LambdaCallback


class RandomSequence(Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __len__(self):
        return 12

    def __getitem__(self, idx):
        return [np.random.random((self.batch_size, 3)), np.random.random((self.batch_size, 3))], [
            np.random.random((self.batch_size, 4)),
            np.random.random((self.batch_size, 3))]

    def on_epoch_end(self):
        pass


@keras_test
def test_check_array_lengths():
    _check_array_lengths(None, None, None)
    a_np = np.random.random((4, 3, 3))
    _check_array_lengths(a_np, a_np, a_np)
    _check_array_lengths([a_np, a_np], [a_np, a_np], [a_np, a_np])
    _check_array_lengths([None], [None], [None])

    b_np = np.random.random((3, 4))
    with pytest.raises(ValueError):
        _check_array_lengths(a_np, None, None)
    with pytest.raises(ValueError):
        _check_array_lengths(a_np, a_np, None)
    with pytest.raises(ValueError):
        _check_array_lengths([a_np], [None], None)
    with pytest.raises(ValueError):
        _check_array_lengths([a_np], [b_np], None)
    with pytest.raises(ValueError):
        _check_array_lengths([a_np], None, [b_np])


@keras_test
def test_slice_arrays():
    input_a = np.random.random((10, 3))
    _slice_arrays(None)
    _slice_arrays(input_a, 0)
    _slice_arrays(input_a, 0, 1)
    _slice_arrays(input_a, stop=2)
    input_a = [None, [1, 1], None, [1, 1]]
    _slice_arrays(input_a, 0)
    _slice_arrays(input_a, 0, 1)
    _slice_arrays(input_a, stop=2)
    input_a = [None]
    _slice_arrays(input_a, 0)
    _slice_arrays(input_a, 0, 1)
    _slice_arrays(input_a, stop=2)
    input_a = None
    _slice_arrays(input_a, 0)
    _slice_arrays(input_a, 0, 1)
    _slice_arrays(input_a, stop=2)


@keras_test
def test_weighted_masked_objective():
    a = Input(shape=(3,), name='input_a')

    # weighted_masked_objective
    def mask_dummy(y_true=None, y_pred=None, weight=None):
        return K.placeholder(y_true.shape)

    weighted_function = _weighted_masked_objective(K.categorical_crossentropy)
    weighted_function(a, a, None)


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

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_a_np = np.random.random((10, 4))
    output_b_np = np.random.random((10, 3))

    # training/testing doesn't work before compiling.
    with pytest.raises(RuntimeError):
        model.train_on_batch([input_a_np, input_b_np], [output_a_np, output_b_np])

    model.compile(optimizer, loss, metrics=[], loss_weights=loss_weights,
                  sample_weight_mode=None)

    # test train_on_batch
    out = model.train_on_batch([input_a_np, input_b_np],
                               [output_a_np, output_b_np])
    out = model.train_on_batch({'input_a': input_a_np, 'input_b': input_b_np},
                               [output_a_np, output_b_np])
    out = model.train_on_batch({'input_a': input_a_np, 'input_b': input_b_np},
                               {'dense_1': output_a_np, 'dropout': output_b_np})

    # test fit
    out = model.fit([input_a_np, input_b_np],
                    [output_a_np, output_b_np], epochs=1, batch_size=4)
    out = model.fit({'input_a': input_a_np, 'input_b': input_b_np},
                    [output_a_np, output_b_np], epochs=1, batch_size=4)
    out = model.fit({'input_a': input_a_np, 'input_b': input_b_np},
                    {'dense_1': output_a_np, 'dropout': output_b_np},
                    epochs=1, batch_size=4)

    # test validation_split
    out = model.fit([input_a_np, input_b_np],
                    [output_a_np, output_b_np],
                    epochs=1, batch_size=4, validation_split=0.5)
    out = model.fit({'input_a': input_a_np, 'input_b': input_b_np},
                    [output_a_np, output_b_np],
                    epochs=1, batch_size=4, validation_split=0.5)
    out = model.fit({'input_a': input_a_np, 'input_b': input_b_np},
                    {'dense_1': output_a_np, 'dropout': output_b_np},
                    epochs=1, batch_size=4, validation_split=0.5)

    # test validation data
    out = model.fit([input_a_np, input_b_np],
                    [output_a_np, output_b_np],
                    epochs=1, batch_size=4,
                    validation_data=([input_a_np, input_b_np], [output_a_np, output_b_np]))
    out = model.fit({'input_a': input_a_np, 'input_b': input_b_np},
                    [output_a_np, output_b_np],
                    epochs=1, batch_size=4, validation_split=0.5,
                    validation_data=({'input_a': input_a_np, 'input_b': input_b_np}, [output_a_np, output_b_np]))
    out = model.fit({'input_a': input_a_np, 'input_b': input_b_np},
                    {'dense_1': output_a_np, 'dropout': output_b_np},
                    epochs=1, batch_size=4, validation_split=0.5,
                    validation_data=(
                        {'input_a': input_a_np, 'input_b': input_b_np},
                        {'dense_1': output_a_np, 'dropout': output_b_np}))

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

    # define tracer callback
    def on_epoch_begin(epoch, logs):
        trained_epochs.append(epoch)

    tracker_cb = LambdaCallback(on_epoch_begin=on_epoch_begin)

    out = model.fit([input_a_np, input_b_np],
                    [output_a_np, output_b_np], epochs=5, batch_size=4,
                    initial_epoch=2, callbacks=[tracker_cb])
    assert trained_epochs == [2, 3, 4]

    # test starting from non-zero initial epoch for generator too
    trained_epochs = []

    def gen_data(batch_sz):
        while True:
            yield ([np.random.random((batch_sz, 3)), np.random.random((batch_sz, 3))],
                   [np.random.random((batch_sz, 4)), np.random.random((batch_sz, 3))])

    out = model.fit_generator(gen_data(4), steps_per_epoch=3, epochs=5,
                              initial_epoch=2, callbacks=[tracker_cb])
    assert trained_epochs == [2, 3, 4]

    # test with a custom metric function
    def mse(y_true, y_pred):
        return K.mean(K.pow(y_true - y_pred, 2))

    model.compile(optimizer, loss, metrics=[mse],
                  sample_weight_mode=None)

    out = model.train_on_batch([input_a_np, input_b_np],
                               [output_a_np, output_b_np])
    out_len = 1 + 2 * (1 + 1)  # total loss + 2 outputs * (loss + metric)
    assert len(out) == out_len
    out = model.test_on_batch([input_a_np, input_b_np],
                              [output_a_np, output_b_np])
    assert len(out) == out_len

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_a_np = np.random.random((10, 4))
    output_b_np = np.random.random((10, 3))

    out = model.fit([input_a_np, input_b_np], [output_a_np, output_b_np], batch_size=4, epochs=1)
    out = model.evaluate([input_a_np, input_b_np], [output_a_np, output_b_np], batch_size=4)
    out = model.predict([input_a_np, input_b_np], batch_size=4)

    # empty batch
    with pytest.raises(ValueError):
        def gen_data():
            while True:
                yield (np.asarray([]), np.asarray([]))
        out = model.evaluate_generator(gen_data(), steps=1)

    # x is not a list of numpy arrays.
    with pytest.raises(ValueError):
        out = model.predict([None])

    # x does not match _feed_input_names.
    with pytest.raises(ValueError):
        out = model.predict([input_a_np, None, input_b_np])
    with pytest.raises(ValueError):
        out = model.predict([None, input_a_np, input_b_np])

    # all input/output/weight arrays should have the same number of samples.
    with pytest.raises(ValueError):
        out = model.train_on_batch([input_a_np, input_b_np[:2]],
                                   [output_a_np, output_b_np],
                                   sample_weight=sample_weight)
    with pytest.raises(ValueError):
        out = model.train_on_batch([input_a_np, input_b_np],
                                   [output_a_np, output_b_np[:2]],
                                   sample_weight=sample_weight)
    with pytest.raises(ValueError):
        out = model.train_on_batch([input_a_np, input_b_np],
                                   [output_a_np, output_b_np],
                                   sample_weight=[sample_weight[1], sample_weight[1][:2]])

    # `sample_weight` is neither a dict nor a list.
    with pytest.raises(TypeError):
        out = model.train_on_batch([input_a_np, input_b_np],
                                   [output_a_np, output_b_np],
                                   sample_weight=tuple(sample_weight))

    # `validation_data` is neither a tuple nor a triple.
    with pytest.raises(ValueError):
        out = model.fit([input_a_np, input_b_np],
                        [output_a_np, output_b_np],
                        epochs=1, batch_size=4,
                        validation_data=([input_a_np, input_b_np],))

    # `loss` does not match outputs.
    with pytest.raises(ValueError):
        model.compile(optimizer, loss=['mse', 'mae', 'mape'])

    # `loss_weights` does not match output_names.
    with pytest.raises(ValueError):
        model.compile(optimizer, loss='mse', loss_weights={'lstm': 0.5})

    # `loss_weights` does not match outputs.
    with pytest.raises(ValueError):
        model.compile(optimizer, loss='mse', loss_weights=[0.5])

    # `loss_weights` is invalid type.
    with pytest.raises(TypeError):
        model.compile(optimizer, loss='mse', loss_weights=(0.5, 0.5))

    # `sample_weight_mode` does not match output_names.
    with pytest.raises(ValueError):
        model.compile(optimizer, loss='mse', sample_weight_mode={'lstm': 'temporal'})

    # `sample_weight_mode` does not match output_names.
    with pytest.raises(ValueError):
        model.compile(optimizer, loss='mse', sample_weight_mode=['temporal'])

    # `sample_weight_mode` matches output_names partially.
    with pytest.raises(ValueError):
        model.compile(optimizer, loss='mse', sample_weight_mode={'dense_1': 'temporal'})

    # `loss` does not exist.
    with pytest.raises(RuntimeError):
        model.compile(optimizer, loss=[])

    model.compile(optimizer, loss=['mse', 'mae'])
    model.compile(optimizer, loss='mse', loss_weights={'dense_1': 0.2, 'dropout': 0.8})
    model.compile(optimizer, loss='mse', loss_weights=[0.2, 0.8])

    # the rank of weight arrays should be 1.
    with pytest.raises(ValueError):
        out = model.train_on_batch([input_a_np, input_b_np],
                                   [output_a_np, output_b_np],
                                   sample_weight=[None, np.random.random((10, 20, 30))])

    model.compile(optimizer, loss='mse', sample_weight_mode={'dense_1': None, 'dropout': 'temporal'})
    model.compile(optimizer, loss='mse', sample_weight_mode=[None, 'temporal'])

    # the rank of output arrays should be at least 3D.
    with pytest.raises(ValueError):
        out = model.train_on_batch([input_a_np, input_b_np],
                                   [output_a_np, output_b_np],
                                   sample_weight=sample_weight)

    model.compile(optimizer, loss, metrics=[], loss_weights=loss_weights,
                  sample_weight_mode=None)
    trained_epochs = []
    out = model.fit_generator(generator=RandomSequence(3), steps_per_epoch=4, epochs=5,
                              initial_epoch=0, validation_data=RandomSequence(4),
                              validation_steps=3, callbacks=[tracker_cb])
    assert trained_epochs == [0, 1, 2, 3, 4]


@pytest.mark.skipif(sys.version_info < (3,), reason='Cannot catch warnings in python 2')
@keras_test
def test_warnings():
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

    def gen_data(batch_sz):
        while True:
            yield ([np.random.random((batch_sz, 3)), np.random.random((batch_sz, 3))],
                   [np.random.random((batch_sz, 4)), np.random.random((batch_sz, 3))])

    with pytest.warns(Warning) as w:
        out = model.fit_generator(gen_data(4), steps_per_epoch=10, use_multiprocessing=True, workers=2)
    warning_raised = any(['Sequence' in str(w_.message) for w_ in w])
    assert warning_raised, 'No warning raised when using generator with processes.'

    with pytest.warns(None) as w:
        out = model.fit_generator(RandomSequence(3), steps_per_epoch=4, use_multiprocessing=True, workers=2)
    assert all(['Sequence' not in str(w_.message) for w_ in w]), 'A warning was raised for Sequence.'


@pytest.mark.skipif(K.backend() != 'tensorflow', reason='sparse operations supported only by TF')
@keras_test
def test_sparse_input_validation_split():
    test_input = sparse.random(6, 3, density=0.25).tocsr()
    in1 = Input(shape=(3,), sparse=True)
    out1 = Dense(4)(in1)
    test_output = np.random.random((6, 4))
    model = Model(in1, out1)
    model.compile('rmsprop', 'mse')
    model.fit(test_input, test_output, epochs=1, batch_size=2, validation_split=0.2)


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
    inputs = Input(shape=(3,))
    outputs = model(inputs)
    model = Model(inputs, outputs)
    model.compile('rmsprop', 'mse')
    out = model.predict(x)
    model.train_on_batch(x, y)
    out_2 = model.predict(x)
    assert_allclose(out, out_2)


@keras_test
def test_check_not_failing():
    a = np.random.random((2, 1, 3))
    _check_loss_and_target_compatibility([a], [K.categorical_crossentropy], [a.shape])
    _check_loss_and_target_compatibility([a], [K.categorical_crossentropy], [(2, None, 3)])


@keras_test
def test_check_last_is_one():
    a = np.random.random((2, 3, 1))
    with pytest.raises(ValueError) as exc:
        _check_loss_and_target_compatibility([a], [K.categorical_crossentropy], [a.shape])

    assert 'You are passing a target array' in str(exc)


@keras_test
def test_check_bad_shape():
    a = np.random.random((2, 3, 5))
    with pytest.raises(ValueError) as exc:
        _check_loss_and_target_compatibility([a], [K.categorical_crossentropy], [(2, 3, 6)])

    assert 'targets to have the same shape' in str(exc)


@pytest.mark.skipif(K.backend() != 'tensorflow', reason='Requires TF backend')
@keras_test
def test_model_with_input_feed_tensor():
    """We test building a model with a TF variable as input.
    We should be able to call fit, evaluate, predict,
    by only passing them data for the placeholder inputs
    in the model.
    """
    import tensorflow as tf

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_a_np = np.random.random((10, 4))
    output_b_np = np.random.random((10, 3))

    a = Input(tensor=tf.Variable(input_a_np, dtype=tf.float32))
    b = Input(shape=(3,), name='input_b')

    a_2 = Dense(4, name='dense_1')(a)
    dp = Dropout(0.5, name='dropout')
    b_2 = dp(b)

    model = Model([a, b], [a_2, b_2])
    model.summary()

    optimizer = 'rmsprop'
    loss = 'mse'
    loss_weights = [1., 0.5]
    model.compile(optimizer, loss, metrics=['mean_squared_error'],
                  loss_weights=loss_weights,
                  sample_weight_mode=None)

    # test train_on_batch
    out = model.train_on_batch(input_b_np,
                               [output_a_np, output_b_np])
    out = model.train_on_batch({'input_b': input_b_np},
                               [output_a_np, output_b_np])
    out = model.test_on_batch({'input_b': input_b_np},
                              [output_a_np, output_b_np])
    out = model.predict_on_batch({'input_b': input_b_np})

    # test fit
    out = model.fit({'input_b': input_b_np},
                    [output_a_np, output_b_np], epochs=1, batch_size=10)
    out = model.fit(input_b_np,
                    [output_a_np, output_b_np], epochs=1, batch_size=10)

    # test evaluate
    out = model.evaluate({'input_b': input_b_np},
                         [output_a_np, output_b_np], batch_size=10)
    out = model.evaluate(input_b_np,
                         [output_a_np, output_b_np], batch_size=10)

    # test predict
    out = model.predict({'input_b': input_b_np}, batch_size=10)
    out = model.predict(input_b_np, batch_size=10)
    assert len(out) == 2

    # Now test a model with a single input
    # i.e. we don't pass any data to fit the model.
    a = Input(tensor=tf.Variable(input_a_np, dtype=tf.float32))
    a_2 = Dense(4, name='dense_1')(a)
    a_2 = Dropout(0.5, name='dropout')(a_2)
    model = Model(a, a_2)
    model.summary()

    optimizer = 'rmsprop'
    loss = 'mse'
    model.compile(optimizer, loss, metrics=['mean_squared_error'])

    # test train_on_batch
    out = model.train_on_batch(None,
                               output_a_np)
    out = model.train_on_batch(None,
                               output_a_np)
    out = model.test_on_batch(None,
                              output_a_np)
    out = model.predict_on_batch(None)
    out = model.train_on_batch([],
                               output_a_np)
    out = model.train_on_batch({},
                               output_a_np)

    # test fit
    out = model.fit(None,
                    output_a_np, epochs=1, batch_size=10)
    out = model.fit(None,
                    output_a_np, epochs=1, batch_size=10)

    # test evaluate
    out = model.evaluate(None,
                         output_a_np, batch_size=10)
    out = model.evaluate(None,
                         output_a_np, batch_size=10)

    # test predict
    out = model.predict(None, batch_size=10)
    out = model.predict(None, batch_size=10)
    assert out.shape == (10, 4)

    # Same, without learning phase
    # i.e. we don't pass any data to fit the model.
    a = Input(tensor=tf.Variable(input_a_np, dtype=tf.float32))
    a_2 = Dense(4, name='dense_1')(a)
    model = Model(a, a_2)
    model.summary()

    optimizer = 'rmsprop'
    loss = 'mse'
    model.compile(optimizer, loss, metrics=['mean_squared_error'])

    # test train_on_batch
    out = model.train_on_batch(None,
                               output_a_np)
    out = model.train_on_batch(None,
                               output_a_np)
    out = model.test_on_batch(None,
                              output_a_np)
    out = model.predict_on_batch(None)
    out = model.train_on_batch([],
                               output_a_np)
    out = model.train_on_batch({},
                               output_a_np)

    # test fit
    out = model.fit(None,
                    output_a_np, epochs=1, batch_size=10)
    out = model.fit(None,
                    output_a_np, epochs=1, batch_size=10)

    # test evaluate
    out = model.evaluate(None,
                         output_a_np, batch_size=10)
    out = model.evaluate(None,
                         output_a_np, batch_size=10)

    # test predict
    out = model.predict(None, batch_size=10)
    out = model.predict(None, batch_size=10)
    assert out.shape == (10, 4)


@keras_test
def test_model_with_partial_loss():
    a = Input(shape=(3,), name='input_a')
    a_2 = Dense(4, name='dense_1')(a)
    dp = Dropout(0.5, name='dropout')
    a_3 = dp(a_2)
    model = Model(a, [a_2, a_3])

    optimizer = 'rmsprop'
    loss = {'dropout': 'mse'}
    model.compile(optimizer, loss, metrics=['mae'])

    input_a_np = np.random.random((10, 3))
    output_a_np = np.random.random((10, 4))

    # test train_on_batch
    out = model.train_on_batch(input_a_np, output_a_np)
    out = model.test_on_batch(input_a_np, output_a_np)
    # fit
    out = model.fit(input_a_np, [output_a_np])
    # evaluate
    out = model.evaluate(input_a_np, [output_a_np])

    # Same without dropout.
    a = Input(shape=(3,), name='input_a')
    a_2 = Dense(4, name='dense_1')(a)
    a_3 = Dense(4, name='dense_2')(a_2)
    model = Model(a, [a_2, a_3])

    optimizer = 'rmsprop'
    loss = {'dense_2': 'mse'}
    model.compile(optimizer, loss, metrics={'dense_1': 'mae'})

    # test train_on_batch
    out = model.train_on_batch(input_a_np, output_a_np)
    out = model.test_on_batch(input_a_np, output_a_np)
    # fit
    out = model.fit(input_a_np, [output_a_np])
    # evaluate
    out = model.evaluate(input_a_np, [output_a_np])


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support external loss yet")
def test_model_with_external_loss():
    # None loss, only regularization loss.
    a = Input(shape=(3,), name='input_a')
    a_2 = Dense(4, name='dense_1',
                kernel_regularizer='l1',
                bias_regularizer='l2')(a)
    dp = Dropout(0.5, name='dropout')
    a_3 = dp(a_2)

    model = Model(a, [a_2, a_3])

    optimizer = 'rmsprop'
    loss = None
    model.compile(optimizer, loss, metrics=['mae'])

    input_a_np = np.random.random((10, 3))

    # test train_on_batch
    out = model.train_on_batch(input_a_np, None)
    out = model.test_on_batch(input_a_np, None)
    # fit
    out = model.fit(input_a_np, None)
    # evaluate
    out = model.evaluate(input_a_np, None)

    # No dropout, external loss.
    a = Input(shape=(3,), name='input_a')
    a_2 = Dense(4, name='dense_1')(a)
    a_3 = Dense(4, name='dense_2')(a)

    model = Model(a, [a_2, a_3])
    model.add_loss(K.mean(a_3 + a_2))

    optimizer = 'rmsprop'
    loss = None
    model.compile(optimizer, loss, metrics=['mae'])

    # test train_on_batch
    out = model.train_on_batch(input_a_np, None)
    out = model.test_on_batch(input_a_np, None)
    # fit
    out = model.fit(input_a_np, None)
    # evaluate
    out = model.evaluate(input_a_np, None)

    # Test fit with no external data at all.
    if K.backend() == 'tensorflow':
        import tensorflow as tf

        a = Input(tensor=tf.Variable(input_a_np, dtype=tf.float32))
        a_2 = Dense(4, name='dense_1')(a)
        a_2 = Dropout(0.5, name='dropout')(a_2)
        model = Model(a, a_2)
        model.add_loss(K.mean(a_2))

        model.compile(optimizer='rmsprop',
                      loss=None,
                      metrics=['mean_squared_error'])

        # test train_on_batch
        out = model.train_on_batch(None, None)
        out = model.test_on_batch(None, None)
        out = model.predict_on_batch(None)

        # test fit
        out = model.fit(None, None, epochs=1, batch_size=10)

        # test evaluate
        out = model.evaluate(None, None, batch_size=10)

        # test predict
        out = model.predict(None, batch_size=10)
        assert out.shape == (10, 4)


if __name__ == '__main__':
    pytest.main([__file__])
