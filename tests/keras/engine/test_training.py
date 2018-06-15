import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
import sys
import scipy.sparse as sparse

import keras
from keras import losses
from keras.layers import Activation, Dense, Dropout, Conv2D, Concatenate
from keras.engine import Input
from keras.engine.training import Model
from keras.engine import training_utils
from keras.utils.generic_utils import slice_arrays
from keras.models import Sequential
from keras import backend as K
from keras.utils import Sequence
from keras.utils.test_utils import keras_test
from keras.callbacks import LambdaCallback


class RandomSequence(Sequence):
    def __init__(self, batch_size, sequence_length=12):
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def __len__(self):
        return self.sequence_length

    def __getitem__(self, idx):
        return ([np.random.random((self.batch_size, 3)),
                 np.random.random((self.batch_size, 3))],
                [np.random.random((self.batch_size, 4)),
                 np.random.random((self.batch_size, 3))])

    def on_epoch_end(self):
        pass


@keras_test
def test_check_array_length_consistency():
    training_utils.check_array_length_consistency(None, None, None)
    a_np = np.random.random((4, 3, 3))
    training_utils.check_array_length_consistency(a_np, a_np, a_np)
    training_utils.check_array_length_consistency(
        [a_np, a_np], [a_np, a_np], [a_np, a_np])
    training_utils.check_array_length_consistency([None], [None], [None])

    b_np = np.random.random((3, 4))
    with pytest.raises(ValueError):
        training_utils.check_array_length_consistency(a_np, None, None)
    with pytest.raises(ValueError):
        training_utils.check_array_length_consistency(a_np, a_np, None)
    with pytest.raises(ValueError):
        training_utils.check_array_length_consistency([a_np], [None], None)
    with pytest.raises(ValueError):
        training_utils.check_array_length_consistency([a_np], [b_np], None)
    with pytest.raises(ValueError):
        training_utils.check_array_length_consistency([a_np], None, [b_np])


@keras_test
def testslice_arrays():
    input_a = np.random.random((10, 3))
    slice_arrays(None)
    slice_arrays(input_a, 0)
    slice_arrays(input_a, 0, 1)
    slice_arrays(input_a, stop=2)
    input_a = [None, [1, 1], None, [1, 1]]
    slice_arrays(input_a, 0)
    slice_arrays(input_a, 0, 1)
    slice_arrays(input_a, stop=2)
    input_a = [None]
    slice_arrays(input_a, 0)
    slice_arrays(input_a, 0, 1)
    slice_arrays(input_a, stop=2)
    input_a = None
    slice_arrays(input_a, 0)
    slice_arrays(input_a, 0, 1)
    slice_arrays(input_a, stop=2)


@keras_test
def test_weighted_masked_objective():
    a = Input(shape=(3,), name='input_a')

    # weighted_masked_objective
    def mask_dummy(y_true=None, y_pred=None, weight=None):
        return K.placeholder(y_true.shape)

    weighted_function = training_utils.weighted_masked_objective(
        losses.categorical_crossentropy)
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
        model.train_on_batch([input_a_np, input_b_np],
                             [output_a_np, output_b_np])

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

    # test validation data
    out = model.fit([input_a_np, input_b_np],
                    [output_a_np, output_b_np],
                    epochs=1, batch_size=4,
                    validation_data=([input_a_np, input_b_np],
                                     [output_a_np, output_b_np]))
    out = model.fit({'input_a': input_a_np, 'input_b': input_b_np},
                    [output_a_np, output_b_np],
                    epochs=1, batch_size=4, validation_split=0.5,
                    validation_data=({'input_a': input_a_np,
                                      'input_b': input_b_np},
                                     [output_a_np, output_b_np]))
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
    out = model.predict_on_batch({'input_a': input_a_np,
                                  'input_b': input_b_np})

    # predict, evaluate
    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_a_np = np.random.random((10, 4))
    output_b_np = np.random.random((10, 3))

    out = model.evaluate([input_a_np, input_b_np],
                         [output_a_np, output_b_np],
                         batch_size=4)
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
    trained_batches = []

    # define tracer callback
    def on_epoch_begin(epoch, logs):
        trained_epochs.append(epoch)

    def on_batch_begin(batch, logs):
        trained_batches.append(batch)

    tracker_cb = LambdaCallback(on_epoch_begin=on_epoch_begin,
                                on_batch_begin=on_batch_begin)

    out = model.fit([input_a_np, input_b_np],
                    [output_a_np, output_b_np], epochs=5, batch_size=4,
                    initial_epoch=2, callbacks=[tracker_cb])
    assert trained_epochs == [2, 3, 4]

    # test starting from non-zero initial epoch for generator too
    trained_epochs = []

    def gen_data(batch_sz):
        while True:
            yield ([np.random.random((batch_sz, 3)),
                    np.random.random((batch_sz, 3))],
                   [np.random.random((batch_sz, 4)),
                    np.random.random((batch_sz, 3))])

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

    out = model.fit([input_a_np, input_b_np],
                    [output_a_np, output_b_np],
                    batch_size=4, epochs=1)
    out = model.evaluate([input_a_np, input_b_np],
                         [output_a_np, output_b_np],
                         batch_size=4)
    out = model.predict([input_a_np, input_b_np], batch_size=4)

    # enable verbose for evaluate_generator
    out = model.evaluate_generator(gen_data(4), steps=3, verbose=1)

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
                                   sample_weight=[sample_weight[1],
                                                  sample_weight[1][:2]])

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
        model.compile(optimizer, loss='mse',
                      sample_weight_mode={'lstm': 'temporal'})

    # `sample_weight_mode` does not match output_names.
    with pytest.raises(ValueError):
        model.compile(optimizer, loss='mse', sample_weight_mode=['temporal'])

    # `sample_weight_mode` matches output_names partially.
    with pytest.raises(ValueError):
        model.compile(optimizer, loss='mse',
                      sample_weight_mode={'dense_1': 'temporal'})

    # `loss` does not exist.
    with pytest.raises(ValueError):
        model.compile(optimizer, loss=[])

    model.compile(optimizer, loss=['mse', 'mae'])
    model.compile(optimizer, loss='mse', loss_weights={'dense_1': 0.2,
                                                       'dropout': 0.8})
    model.compile(optimizer, loss='mse', loss_weights=[0.2, 0.8])

    # the rank of weight arrays should be 1.
    with pytest.raises(ValueError):
        out = model.train_on_batch([input_a_np, input_b_np],
                                   [output_a_np, output_b_np],
                                   sample_weight=[None, np.random.random((10, 20, 30))])

    model.compile(optimizer, loss='mse',
                  sample_weight_mode={'dense_1': None, 'dropout': 'temporal'})
    model.compile(optimizer, loss='mse', sample_weight_mode=[None, 'temporal'])

    # the rank of output arrays should be at least 3D.
    with pytest.raises(ValueError):
        out = model.train_on_batch([input_a_np, input_b_np],
                                   [output_a_np, output_b_np],
                                   sample_weight=sample_weight)

    model.compile(optimizer, loss, metrics=[], loss_weights=loss_weights,
                  sample_weight_mode=None)
    trained_epochs = []
    trained_batches = []
    out = model.fit_generator(generator=RandomSequence(3),
                              steps_per_epoch=3,
                              epochs=5,
                              initial_epoch=0,
                              validation_data=RandomSequence(4),
                              validation_steps=3,
                              callbacks=[tracker_cb])
    assert trained_epochs == [0, 1, 2, 3, 4]
    assert trained_batches == list(range(3)) * 5

    # steps_per_epoch will be equal to len of sequence if it's unspecified
    trained_epochs = []
    trained_batches = []
    out = model.fit_generator(generator=RandomSequence(3),
                              epochs=5,
                              initial_epoch=0,
                              validation_data=RandomSequence(4),
                              callbacks=[tracker_cb])
    assert trained_epochs == [0, 1, 2, 3, 4]
    assert trained_batches == list(range(12)) * 5

    # fit_generator will throw an exception
    # if steps is unspecified for regular generator
    with pytest.raises(ValueError):
        def gen_data():
            while True:
                yield (np.asarray([]), np.asarray([]))
        out = model.fit_generator(generator=gen_data(), epochs=5,
                                  initial_epoch=0, validation_data=gen_data(),
                                  callbacks=[tracker_cb])

    # Check if generator is only accessed an expected number of times
    gen_counters = [0, 0]

    def gen_data(i):
        while True:
            gen_counters[i] += 1
            yield ([np.random.random((1, 3)), np.random.random((1, 3))],
                   [np.random.random((1, 4)), np.random.random((1, 3))])
    out = model.fit_generator(generator=gen_data(0), epochs=3,
                              steps_per_epoch=2,
                              validation_data=gen_data(1),
                              validation_steps=1,
                              max_queue_size=2,
                              workers=2)

    # Need range check here as filling
    # of the queue depends on sleep in the enqueuers
    assert 6 <= gen_counters[0] <= 8
    # 12 = (epoch * workers * validation steps * max_queue_size)
    assert 3 <= gen_counters[1] <= 12

    gen_counters = [0]
    out = model.fit_generator(generator=RandomSequence(3), epochs=3,
                              validation_data=gen_data(0),
                              validation_steps=1,
                              max_queue_size=2,
                              workers=2)

    # 12 = (epoch * workers * validation steps * max_queue_size)
    # Need range check here as filling
    # of the queue depends on sleep in the enqueuers
    assert 3 <= gen_counters[0] <= 12

    # predict_generator output shape behavior should be consistent
    def expected_shape(batch_size, n_batches):
        return (batch_size * n_batches, 4), (batch_size * n_batches, 3)

    # Multiple outputs and one step.
    batch_size = 5
    sequence_length = 1
    shape_0, shape_1 = expected_shape(batch_size, sequence_length)
    out = model.predict_generator(
        RandomSequence(batch_size, sequence_length=sequence_length))
    assert np.shape(out[0]) == shape_0 and np.shape(out[1]) == shape_1

    # Multiple outputs and multiple steps.
    batch_size = 5
    sequence_length = 2
    shape_0, shape_1 = expected_shape(batch_size, sequence_length)
    out = model.predict_generator(
        RandomSequence(batch_size, sequence_length=sequence_length))
    assert np.shape(out[0]) == shape_0 and np.shape(out[1]) == shape_1

    # Create a model with a single output.
    single_output_model = Model([a, b], a_2)
    single_output_model.compile(optimizer, loss,
                                metrics=[], sample_weight_mode=None)

    # Single output and one step.
    batch_size = 5
    sequence_length = 1
    shape_0, _ = expected_shape(batch_size, sequence_length)
    out = single_output_model.predict_generator(
        RandomSequence(batch_size, sequence_length=sequence_length))
    assert np.shape(out) == shape_0

    # Single output and multiple steps.
    batch_size = 5
    sequence_length = 2
    shape_0, _ = expected_shape(batch_size, sequence_length)
    out = single_output_model.predict_generator(
        RandomSequence(batch_size, sequence_length=sequence_length))
    assert np.shape(out) == shape_0


@pytest.mark.skipif(sys.version_info < (3,),
                    reason='Cannot catch warnings in python 2')
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
            yield ([np.random.random((batch_sz, 3)),
                    np.random.random((batch_sz, 3))],
                   [np.random.random((batch_sz, 4)),
                    np.random.random((batch_sz, 3))])

    with pytest.warns(Warning) as w:
        out = model.fit_generator(gen_data(4),
                                  steps_per_epoch=10,
                                  use_multiprocessing=True,
                                  workers=2)
    warning_raised = any(['Sequence' in str(w_.message) for w_ in w])
    assert warning_raised, 'No warning raised when using generator with processes.'

    with pytest.warns(None) as w:
        out = model.fit_generator(RandomSequence(3),
                                  steps_per_epoch=4,
                                  use_multiprocessing=True,
                                  workers=2)
    assert all(['Sequence' not in str(w_.message) for w_ in w]), 'A warning was raised for Sequence.'


@keras_test
def test_sparse_inputs_targets():
    test_inputs = [sparse.random(6, 3, density=0.25).tocsr() for _ in range(2)]
    test_outputs = [sparse.random(6, i, density=0.25).tocsr() for i in range(3, 5)]
    in1 = Input(shape=(3,))
    in2 = Input(shape=(3,))
    out1 = Dropout(0.5, name='dropout')(in1)
    out2 = Dense(4, name='dense_1')(in2)
    model = Model([in1, in2], [out1, out2])
    model.predict(test_inputs, batch_size=2)
    model.compile('rmsprop', 'mse')
    model.fit(test_inputs, test_outputs,
              epochs=1, batch_size=2, validation_split=0.5)
    model.evaluate(test_inputs, test_outputs, batch_size=2)


@pytest.mark.skipif(K.backend() != 'tensorflow', reason='sparse operations supported only by TensorFlow')
@keras_test
def test_sparse_placeholder_fit():
    test_inputs = [sparse.random(6, 3, density=0.25).tocsr() for _ in range(2)]
    test_outputs = [sparse.random(6, i, density=0.25).tocsr() for i in range(3, 5)]
    in1 = Input(shape=(3,))
    in2 = Input(shape=(3,), sparse=True)
    out1 = Dropout(0.5, name='dropout')(in1)
    out2 = Dense(4, name='dense_1')(in2)
    model = Model([in1, in2], [out1, out2])
    model.predict(test_inputs, batch_size=2)
    model.compile('rmsprop', 'mse')
    model.fit(test_inputs, test_outputs,
              epochs=1, batch_size=2, validation_split=0.5)
    model.evaluate(test_inputs, test_outputs, batch_size=2)


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
def test_with_list_as_targets():
    model = Sequential()
    model.add(Dense(1, input_dim=3, trainable=False))
    model.compile('rmsprop', 'mse')

    x = np.random.random((2, 3))
    y = [0, 1]
    model.train_on_batch(x, y)


@keras_test
def test_check_not_failing():
    a = np.random.random((2, 1, 3))
    training_utils.check_loss_and_target_compatibility(
        [a], [losses.categorical_crossentropy], [a.shape])
    training_utils.check_loss_and_target_compatibility(
        [a], [losses.categorical_crossentropy], [(2, None, 3)])


@keras_test
def test_check_last_is_one():
    a = np.random.random((2, 3, 1))
    with pytest.raises(ValueError) as exc:
        training_utils.check_loss_and_target_compatibility(
            [a], [losses.categorical_crossentropy], [a.shape])

    assert 'You are passing a target array' in str(exc)


@keras_test
def test_check_bad_shape():
    a = np.random.random((2, 3, 5))
    with pytest.raises(ValueError) as exc:
        training_utils.check_loss_and_target_compatibility(
            [a], [losses.categorical_crossentropy], [(2, 3, 6)])

    assert 'targets to have the same shape' in str(exc)


@pytest.mark.skipif(K.backend() != 'tensorflow', reason='Requires TensorFlow backend')
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
    out = model.predict(None, steps=3)
    out = model.predict(None, steps=3)
    assert out.shape == (10 * 3, 4)

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
    out = model.predict(None, steps=3)
    out = model.predict(None, steps=3)
    assert out.shape == (10 * 3, 4)


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
                    reason='cntk does not support external loss yet')
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
        with pytest.raises(ValueError):
            out = model.fit(None, None, epochs=1, batch_size=10)
        out = model.fit(None, None, epochs=1, steps_per_epoch=1)

        # define a generator to produce x=None and y=None
        def data_tensors_generator():
            while True:
                yield (None, None)

        generator = data_tensors_generator()

        # test fit_generator for framework-native data tensors
        out = model.fit_generator(generator, epochs=1,
                                  steps_per_epoch=3)

        # test evaluate_generator for framework-native data tensors
        out = model.evaluate_generator(generator, steps=3)

        # test fit with validation data
        with pytest.raises(ValueError):
            out = model.fit(None, None,
                            epochs=1,
                            steps_per_epoch=None,
                            validation_steps=2)
        out = model.fit(None, None,
                        epochs=1,
                        steps_per_epoch=2,
                        validation_steps=2)

        # test evaluate
        with pytest.raises(ValueError):
            out = model.evaluate(None, None, batch_size=10)
        out = model.evaluate(None, None, steps=3)

        # test predict
        with pytest.raises(ValueError):
            out = model.predict(None, batch_size=10)
        out = model.predict(None, steps=3)
        assert out.shape == (10 * 3, 4)

        # Test multi-output model without external data.
        a = Input(tensor=tf.Variable(input_a_np, dtype=tf.float32))
        a_1 = Dense(4, name='dense_1')(a)
        a_2 = Dropout(0.5, name='dropout')(a_1)
        model = Model(a, [a_1, a_2])
        model.add_loss(K.mean(a_2))
        model.compile(optimizer='rmsprop',
                      loss=None,
                      metrics=['mean_squared_error'])

        # test train_on_batch
        out = model.train_on_batch(None, None)
        out = model.test_on_batch(None, None)
        out = model.predict_on_batch(None)

        # test fit
        with pytest.raises(ValueError):
            out = model.fit(None, None, epochs=1, batch_size=10)
        out = model.fit(None, None, epochs=1, steps_per_epoch=1)

        # test fit with validation data
        with pytest.raises(ValueError):
            out = model.fit(None, None,
                            epochs=1,
                            steps_per_epoch=None,
                            validation_steps=2)
        out = model.fit(None, None,
                        epochs=1,
                        steps_per_epoch=2,
                        validation_steps=2)

        # test evaluate
        with pytest.raises(ValueError):
            out = model.evaluate(None, None, batch_size=10)
        out = model.evaluate(None, None, steps=3)

        # test predict
        with pytest.raises(ValueError):
            out = model.predict(None, batch_size=10)
        out = model.predict(None, steps=3)
        assert len(out) == 2
        assert out[0].shape == (10 * 3, 4)
        assert out[1].shape == (10 * 3, 4)


@keras_test
def test_target_tensors():
    # single-output, as list
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(4, input_shape=(4,), name='dense'))
    input_val = np.random.random((10, 4))
    target_val = np.random.random((10, 4))
    target = keras.backend.variable(target_val)
    model.compile(optimizer='rmsprop', loss='mse', target_tensors=[target])
    model.train_on_batch(input_val, None)

    # single-output, as dict
    model.compile(optimizer='rmsprop', loss='mse',
                  target_tensors={'dense': target})
    model.train_on_batch(input_val, None)

    # test invalid arguments
    with pytest.raises(TypeError):
        model.compile(optimizer='rmsprop', loss='mse',
                      target_tensors=set())
    with pytest.raises(ValueError):
        model.compile(optimizer='rmsprop', loss='mse',
                      target_tensors=[target, target])
    with pytest.raises(ValueError):
        model.compile(optimizer='rmsprop', loss='mse',
                      target_tensors={'dense2': None})
    with pytest.raises(ValueError):
        model.compile(optimizer='rmsprop', loss='mse',
                      target_tensors=[target])
        model.train_on_batch(input_val, target_val)

    # multi-output, as list
    input_val = np.random.random((10, 4))
    target_val_a = np.random.random((10, 4))
    target_val_b = np.random.random((10, 4))
    target_a = keras.backend.variable(target_val_a)
    target_b = keras.backend.variable(target_val_b)

    inputs = keras.layers.Input(shape=(4,))
    output_a = keras.layers.Dense(4, name='dense_a')(inputs)
    output_b = keras.layers.Dense(4, name='dense_b')(inputs)
    model = keras.models.Model(inputs, [output_a, output_b])
    model.compile(optimizer='rmsprop', loss='mse',
                  target_tensors=[target_a, target_b])
    model.train_on_batch(input_val, None)

    # multi-output, as dict
    model.compile(optimizer='rmsprop', loss='mse',
                  target_tensors={'dense_a': target_a,
                                  'dense_b': target_b})
    model.train_on_batch(input_val, None)

    # test with sample weights
    model.compile(optimizer='rmsprop', loss='mse',
                  target_tensors=[target_a, target_b])
    model.train_on_batch(input_val, None,
                         sample_weight={'dense_a': np.random.random((10,))})


@keras_test
def test_model_custom_target_tensors():
    a = Input(shape=(3,), name='input_a')
    b = Input(shape=(3,), name='input_b')

    a_2 = Dense(4, name='dense_1')(a)
    dp = Dropout(0.5, name='dropout')
    b_2 = dp(b)

    y = K.placeholder([10, 4], name='y')
    y1 = K.placeholder([10, 3], name='y1')
    y2 = K.placeholder([7, 5], name='y2')
    model = Model([a, b], [a_2, b_2])

    optimizer = 'rmsprop'
    loss = 'mse'
    loss_weights = [1., 0.5]

    # test list of target tensors
    with pytest.raises(ValueError):
        model.compile(optimizer, loss, metrics=[], loss_weights=loss_weights,
                      sample_weight_mode=None, target_tensors=[y, y1, y2])
    model.compile(optimizer, loss, metrics=[], loss_weights=loss_weights,
                  sample_weight_mode=None, target_tensors=[y, y1])
    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_a_np = np.random.random((10, 4))
    output_b_np = np.random.random((10, 3))

    out = model.train_on_batch([input_a_np, input_b_np],
                               [output_a_np, output_b_np],
                               {y: np.random.random((10, 4)),
                                y1: np.random.random((10, 3))})
    # test dictionary of target_tensors
    with pytest.raises(ValueError):
        model.compile(optimizer, loss,
                      metrics=[],
                      loss_weights=loss_weights,
                      sample_weight_mode=None,
                      target_tensors={'does_not_exist': y2})
    # test dictionary of target_tensors
    model.compile(optimizer, loss,
                  metrics=[],
                  loss_weights=loss_weights,
                  sample_weight_mode=None,
                  target_tensors={'dense_1': y, 'dropout': y1})
    out = model.train_on_batch([input_a_np, input_b_np],
                               [output_a_np, output_b_np],
                               {y: np.random.random((10, 4)),
                                y1: np.random.random((10, 3))})

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        # test with custom TF placeholder as target
        pl_target_a = tf.placeholder('float32', shape=(None, 4))
        model.compile(optimizer='rmsprop', loss='mse',
                      target_tensors={'dense_1': pl_target_a})
        model.train_on_batch([input_a_np, input_b_np],
                             [output_a_np, output_b_np])


@pytest.mark.skipif(sys.version_info < (3,), reason='Cannot catch warnings in python 2')
@keras_test
def test_trainable_weights_count_consistency():
    """Tests the trainable weights consistency check of Model.

    This verifies that a warning is shown if model.trainable is modified
    and the model is summarized/run without a new call to .compile()

    Reproduce issue #8121
    """
    a = Input(shape=(3,), name='input_a')
    model1 = Model(inputs=a, outputs=Dense(1)(a))

    model1.trainable = False
    b = Input(shape=(3,), name='input_b')
    y = model1(b)
    model2 = Model(inputs=b, outputs=Dense(1)(y))

    model2.compile(optimizer='adam', loss='mse')

    model1.trainable = True

    # Should warn on .summary()
    with pytest.warns(UserWarning) as w:
        model2.summary()
    warning_raised = any(['Discrepancy' in str(w_.message) for w_ in w])
    assert warning_raised, 'No warning raised when trainable is modified without .compile.'

    # And on .fit()
    with pytest.warns(UserWarning) as w:
        model2.fit(x=np.zeros((5, 3)), y=np.zeros((5, 1)))
    warning_raised = any(['Discrepancy' in str(w_.message) for w_ in w])
    assert warning_raised, 'No warning raised when trainable is modified without .compile.'

    # And shouldn't warn if we recompile
    model2.compile(optimizer='adam', loss='mse')
    with pytest.warns(None) as w:
        model2.summary()
    assert len(w) == 0, "Warning raised even when .compile() is called after modifying .trainable"


@keras_test
def test_pandas_dataframe():
    input_a = Input(shape=(3,), name='input_a')
    input_b = Input(shape=(3,), name='input_b')

    x = Dense(4, name='dense_1')(input_a)
    y = Dense(3, name='desne_2')(input_b)

    model_1 = Model(inputs=input_a, outputs=x)
    model_2 = Model(inputs=[input_a, input_b], outputs=[x, y])

    optimizer = 'rmsprop'
    loss = 'mse'

    model_1.compile(optimizer=optimizer, loss=loss)
    model_2.compile(optimizer=optimizer, loss=loss)

    input_a_df = pd.DataFrame(np.random.random((10, 3)))
    input_b_df = pd.DataFrame(np.random.random((10, 3)))

    output_a_df = pd.DataFrame(np.random.random((10, 4)))
    output_b_df = pd.DataFrame(np.random.random((10, 3)))

    model_1.fit(input_a_df,
                output_a_df)
    model_2.fit([input_a_df, input_b_df],
                [output_a_df, output_b_df])
    model_1.fit([input_a_df],
                [output_a_df])
    model_1.fit({'input_a': input_a_df},
                output_a_df)
    model_2.fit({'input_a': input_a_df, 'input_b': input_b_df},
                [output_a_df, output_b_df])

    model_1.predict(input_a_df)
    model_2.predict([input_a_df, input_b_df])
    model_1.predict([input_a_df])
    model_1.predict({'input_a': input_a_df})
    model_2.predict({'input_a': input_a_df, 'input_b': input_b_df})

    model_1.predict_on_batch(input_a_df)
    model_2.predict_on_batch([input_a_df, input_b_df])
    model_1.predict_on_batch([input_a_df])
    model_1.predict_on_batch({'input_a': input_a_df})
    model_2.predict_on_batch({'input_a': input_a_df, 'input_b': input_b_df})

    model_1.evaluate(input_a_df,
                     output_a_df)
    model_2.evaluate([input_a_df, input_b_df],
                     [output_a_df, output_b_df])
    model_1.evaluate([input_a_df],
                     [output_a_df])
    model_1.evaluate({'input_a': input_a_df},
                     output_a_df)
    model_2.evaluate({'input_a': input_a_df, 'input_b': input_b_df},
                     [output_a_df, output_b_df])

    model_1.train_on_batch(input_a_df,
                           output_a_df)
    model_2.train_on_batch([input_a_df, input_b_df],
                           [output_a_df, output_b_df])
    model_1.train_on_batch([input_a_df],
                           [output_a_df])
    model_1.train_on_batch({'input_a': input_a_df},
                           output_a_df)
    model_2.train_on_batch({'input_a': input_a_df, 'input_b': input_b_df},
                           [output_a_df, output_b_df])

    model_1.test_on_batch(input_a_df,
                          output_a_df)
    model_2.test_on_batch([input_a_df, input_b_df],
                          [output_a_df, output_b_df])
    model_1.test_on_batch([input_a_df],
                          [output_a_df])
    model_1.test_on_batch({'input_a': input_a_df},
                          output_a_df)
    model_2.test_on_batch({'input_a': input_a_df, 'input_b': input_b_df},
                          [output_a_df, output_b_df])


@keras_test
@pytest.mark.skipif(K.backend() != 'tensorflow', reason='Requires TensorFlow')
@pytest.mark.skipif((K.backend() == 'tensorflow' and
                     not hasattr(K.get_session(),
                                 '_make_callable_from_options')),
                    reason='Requires TF 1.8 or higher')
def test_training_and_eval_methods_on_symbolic_tensors_single_io():
    x = keras.layers.Input(shape=(3,), name='input')
    y = keras.layers.Dense(4, name='dense')(x)
    model = keras.Model(x, y)

    optimizer = 'rmsprop'
    loss = 'mse'
    metrics = ['mae']
    model.compile(optimizer, loss, metrics=metrics)

    inputs = keras.backend.zeros(shape=(10, 3))
    targets = keras.backend.zeros(shape=(10, 4))

    model.fit(inputs, targets, epochs=1, steps_per_epoch=2, verbose=0)
    model.evaluate(inputs, targets, steps=2, verbose=0)
    model.predict(inputs, steps=2)
    model.train_on_batch(inputs, targets)
    model.test_on_batch(inputs, targets)
    model.fit(inputs, targets,
              epochs=1, steps_per_epoch=2, verbose=1,
              validation_data=(inputs, targets), validation_steps=2)


@keras_test
@pytest.mark.skipif(K.backend() != 'tensorflow', reason='Requires TensorFlow')
@pytest.mark.skipif((K.backend() == 'tensorflow' and
                     not hasattr(K.get_session(),
                                 '_make_callable_from_options')),
                    reason='Requires TF 1.8 or higher')
def test_training_and_eval_methods_on_symbolic_tensors_multi_io():
    a = keras.layers.Input(shape=(3,), name='input_a')
    b = keras.layers.Input(shape=(3,), name='input_b')

    dense = keras.layers.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = keras.layers.Dropout(0.5, name='dropout')(c)

    model = keras.models.Model([a, b], [d, e])

    optimizer = 'rmsprop'
    loss = 'mse'
    loss_weights = [1., 0.5]
    metrics = ['mae']
    model.compile(optimizer, loss, metrics=metrics, loss_weights=loss_weights)

    input_a_tf = keras.backend.zeros(shape=(10, 3))
    input_b_tf = keras.backend.zeros(shape=(10, 3))

    output_d_tf = keras.backend.zeros(shape=(10, 4))
    output_e_tf = keras.backend.zeros(shape=(10, 4))

    model.fit(
        [input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
        epochs=1,
        steps_per_epoch=2,
        verbose=0)
    with pytest.raises(ValueError) as excinfo:
        model.fit(
            [input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
            epochs=1,
            batch_size=5,
            verbose=0)
    assert 'should specify the `steps_per_epoch`' in str(excinfo.value)
    model.train_on_batch([input_a_tf, input_b_tf], [output_d_tf, output_e_tf])

    # Test with dictionary inputs
    model.fit(
        {'input_a': input_a_tf,
         'input_b': input_b_tf},
        {'dense': output_d_tf,
         'dropout': output_e_tf},
        epochs=1,
        steps_per_epoch=2,
        verbose=0)
    model.fit(
        {'input_a': input_a_tf,
         'input_b': input_b_tf},
        {'dense': output_d_tf,
         'dropout': output_e_tf},
        validation_data=({'input_a': input_a_tf,
                          'input_b': input_b_tf},
                         {'dense': output_d_tf,
                          'dropout': output_e_tf}),
        epochs=1,
        steps_per_epoch=2,
        validation_steps=2,
        verbose=0)
    model.train_on_batch(
        {'input_a': input_a_tf,
         'input_b': input_b_tf},
        {'dense': output_d_tf,
         'dropout': output_e_tf})

    # Test with validation data
    model.fit(
        [input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
        validation_data=([input_a_tf, input_b_tf],
                         [output_d_tf, output_e_tf]),
        epochs=1,
        steps_per_epoch=2,
        validation_steps=2,
        verbose=0)
    # Test with validation split
    with pytest.raises(ValueError) as excinfo:
        model.fit(
            [input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
            epochs=2,
            steps_per_epoch=2,
            verbose=0,
            validation_split=0.2,
            validation_steps=2)
    assert 'you cannot use `validation_split`' in str(excinfo.value)

    # Test evaluation / prediction methods
    model.evaluate([input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
                   steps=2, verbose=0)
    model.predict([input_a_tf, input_b_tf], steps=2)
    model.test_on_batch([input_a_tf, input_b_tf], [output_d_tf, output_e_tf])


@keras_test
def test_model_with_crossentropy_losses_channels_first():
    """Tests use of all crossentropy losses with `channels_first`.

    Tests `sparse_categorical_crossentropy`, `categorical_crossentropy`,
    and `binary_crossentropy`.
    Verifies that evaluate gives the same result with either
    `channels_first` or `channels_last` image_data_format.
    Tests PR #9715.
    """
    def prepare_simple_model(input_tensor, loss_name, target):
        axis = 1 if K.image_data_format() == 'channels_first' else -1
        if loss_name == 'sparse_categorical_crossentropy':
            loss = lambda y_true, y_pred: K.sparse_categorical_crossentropy(
                y_true, y_pred, axis=axis)
            num_channels = np.amax(target) + 1
            activation = 'softmax'
        elif loss_name == 'categorical_crossentropy':
            loss = lambda y_true, y_pred: K.categorical_crossentropy(
                y_true, y_pred, axis=axis)
            num_channels = target.shape[axis]
            activation = 'softmax'
        elif loss_name == 'binary_crossentropy':
            loss = lambda y_true, y_pred: K.binary_crossentropy(y_true, y_pred)
            num_channels = target.shape[axis]
            activation = 'sigmoid'
        predictions = Conv2D(num_channels, 1, activation=activation,
                             kernel_initializer='ones',
                             bias_initializer='ones')(input_tensor)
        simple_model = Model(inputs=input_tensor, outputs=predictions)
        simple_model.compile(optimizer='rmsprop', loss=loss)
        return simple_model

    losses_to_test = ['sparse_categorical_crossentropy',
                      'categorical_crossentropy', 'binary_crossentropy']

    data_channels_first = np.array([[[[8., 7.1, 0.], [4.5, 2.6, 0.55],
                                      [0.9, 4.2, 11.2]]]], dtype=np.float32)
    # Labels for testing 4-class sparse_categorical_crossentropy, 4-class
    # categorical_crossentropy, and 2-class binary_crossentropy:
    labels_channels_first = [np.array([[[[0, 1, 3], [2, 1, 0], [2, 2, 1]]]]),
                             np.array([[[[0, 1, 0], [0, 1, 0], [0, 0, 0]],
                                        [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                                        [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
                                        [[0, 0, 1], [0, 0, 0], [1, 0, 0]]]]),
                             np.array([[[[0, 1, 0], [0, 1, 0], [0, 0, 1]],
                                        [[1, 0, 1], [1, 0, 1], [1, 1, 0]]]])]
    # Compute one loss for each loss function in the list `losses_to_test`:
    loss_channels_last = [0., 0., 0.]
    loss_channels_first = [0., 0., 0.]

    old_data_format = K.image_data_format()

    # Evaluate a simple network with channels last, with all three loss
    # functions:
    K.set_image_data_format('channels_last')
    data = np.moveaxis(data_channels_first, 1, -1)
    for index, loss_function in enumerate(losses_to_test):
        labels = np.moveaxis(labels_channels_first[index], 1, -1)
        inputs = Input(shape=(3, 3, 1))
        model = prepare_simple_model(inputs, loss_function, labels)
        loss_channels_last[index] = model.evaluate(x=data, y=labels,
                                                   batch_size=1, verbose=0)

    # Evaluate the same network with channels first, with all three loss
    # functions:
    K.set_image_data_format('channels_first')
    data = data_channels_first
    for index, loss_function in enumerate(losses_to_test):
        labels = labels_channels_first[index]
        inputs = Input(shape=(1, 3, 3))
        model = prepare_simple_model(inputs, loss_function, labels)
        loss_channels_first[index] = model.evaluate(x=data, y=labels,
                                                    batch_size=1, verbose=0)

    K.set_image_data_format(old_data_format)

    assert_allclose(loss_channels_first, loss_channels_last,
                    err_msg='{}{}'.format('Computed different losses for ',
                                          'channels_first and channels_last.'))


@keras_test
def test_dynamic_set_inputs():
    model = Sequential()
    model.add(Dense(16, input_dim=32))
    model.add(Activation('relu'))

    model2 = Sequential()
    model2.add(model.layers[-1])
    model2.add(Dense(8))
    preds2 = model2.predict([np.random.random((1, 32))])
    assert preds2.shape == (1, 8)

    model3 = Model(inputs=model.inputs, outputs=model.outputs)
    with pytest.raises(ValueError):
        model3._set_inputs(model.inputs)

    model3.inputs = None
    model3._set_inputs(model.inputs)
    preds3 = model3.predict([np.random.random((1, 32))])
    assert preds3.shape == (1, 16)

    model3.inputs = None
    model3._set_inputs(model.input)
    preds3 = model3.predict(np.random.random((1, 32)))
    assert preds3.shape == (1, 16)

    aux_input = Input(shape=(5,), name='aux_input')
    aux_model = Dense(3)(aux_input)
    model4 = Model(inputs=model.inputs + [aux_input],
                   outputs=Concatenate()(model.outputs + [aux_model]))
    model4.inputs = None
    model4._set_inputs(model.inputs + [aux_input])
    preds4 = model4.predict([np.random.random((1, 32)),
                             np.random.random((1, 5))])
    assert preds4.shape == (1, 19)


if __name__ == '__main__':
    pytest.main([__file__])
