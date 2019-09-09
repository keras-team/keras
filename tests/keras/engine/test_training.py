import threading

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
import sys
import scipy.sparse as sparse
from flaky import flaky

import keras
from keras import losses
from keras import metrics
from keras.layers import Layer, Activation, Dense, Dropout, Conv2D, Concatenate
from keras.engine import Input
from keras.engine.training import Model
from keras.engine import training_utils
from keras.utils.generic_utils import slice_arrays
from keras.models import Sequential
from keras import backend as K
from keras.utils import Sequence
from keras.callbacks import Callback

if K.backend() == 'tensorflow':
    import tensorflow as tf


class RandomSequence(Sequence):
    def __init__(self, batch_size, sequence_length=12):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.logs = []  # It will work for use_multiprocessing=False

    def __len__(self):
        return self.sequence_length

    def __getitem__(self, idx):
        self.logs.append(idx)
        return ([np.random.random((self.batch_size, 3)),
                 np.random.random((self.batch_size, 3))],
                [np.random.random((self.batch_size, 4)),
                 np.random.random((self.batch_size, 3))])

    def on_epoch_end(self):
        pass


class IncreaseBatchSizeRandomSequence(Sequence):
    def __init__(self, initial_batch_size, initial_sequence_length=12,
                 batch_size_func=lambda x: x + 2):
        self.batch_size = initial_batch_size
        self.initial_sequence_length = initial_sequence_length
        self.batch_size_func = batch_size_func
        self.logs = []

    def __len__(self):
        return int(np.ceil(self.initial_sequence_length / float(self.batch_size)))

    def __getitem__(self, idx):
        self.logs.append(idx)
        return ([np.random.random((self.batch_size, 3)),
                 np.random.random((self.batch_size, 3))],
                [np.random.random((self.batch_size, 4)),
                 np.random.random((self.batch_size, 3))])

    def on_epoch_end(self):
        self.batch_size = self.batch_size_func(self.batch_size)


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


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


def test_weighted_masked_objective():
    a = Input(shape=(3,), name='input_a')

    # weighted_masked_objective
    def mask_dummy(y_true=None, y_pred=None, weight=None):
        return K.placeholder(y_true.shape)

    weighted_function = training_utils.weighted_masked_objective(
        losses.categorical_crossentropy)
    weighted_function(a, a, None)


def get_model(num_outputs=1):
    a = Input(shape=(3,), name='input_a')
    b = Input(shape=(3,), name='input_b')

    a_2 = Dense(4, name='dense_1')(a)
    dp = Dropout(0.5, name='dropout')
    b_2 = dp(b)

    if num_outputs == 1:
        model = Model([a, b], a_2)
    else:
        model = Model([a, b], [a_2, b_2])
    return model


class TrackerCallback(Callback):

    def __init__(self):
        # test starting from non-zero initial epoch
        self.trained_epochs = []
        self.trained_batches = []
        self.steps_per_epoch_log = []
        super(TrackerCallback, self).__init__()

    def set_params(self, params):
        super(TrackerCallback, self).set_params(params)
        self.steps_per_epoch_log.append(params['steps'])

    # define tracer callback
    def on_epoch_begin(self, epoch, logs):
        self.trained_epochs.append(epoch)

    def on_batch_begin(self, batch, logs):
        self.trained_batches.append(batch)


# TODO: resolve flakyness issue. Tracked with #11560
@flaky(rerun_filter=lambda err, *args: issubclass(err[0], AssertionError))
def test_model_methods():
    model = get_model(num_outputs=2)

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

    tracker_cb = TrackerCallback()

    out = model.fit([input_a_np, input_b_np],
                    [output_a_np, output_b_np], epochs=5, batch_size=4,
                    initial_epoch=2, callbacks=[tracker_cb])
    assert tracker_cb.trained_epochs == [2, 3, 4]

    # test starting from non-zero initial epoch for generator too
    tracker_cb = TrackerCallback()

    @threadsafe_generator
    def gen_data(batch_sz):
        while True:
            yield ([np.random.random((batch_sz, 3)),
                    np.random.random((batch_sz, 3))],
                   [np.random.random((batch_sz, 4)),
                    np.random.random((batch_sz, 3))])

    out = model.fit_generator(gen_data(4), steps_per_epoch=3, epochs=5,
                              initial_epoch=2, callbacks=[tracker_cb])
    assert tracker_cb.trained_epochs == [2, 3, 4]

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
    # pass generator directly so `is_generator_or_sequence`
    # doesn't get confused.
    out = model.evaluate(gen_data(4).it, steps=3, verbose=1)

    # empty batch
    with pytest.raises(ValueError):
        @threadsafe_generator
        def gen_data():
            while True:
                yield (np.asarray([]), np.asarray([]))

        out = model.evaluate_generator(gen_data(), steps=1)
    with pytest.raises(ValueError):
        @threadsafe_generator
        def gen_data():
            while True:
                yield (np.asarray([]), np.asarray([]))

        out = model.evaluate(gen_data().it, steps=1)

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
        out = model.train_on_batch(
            [input_a_np, input_b_np],
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


# TODO: resolve flakyness issue. Tracked with #11560
@flaky(rerun_filter=lambda err, *args: issubclass(err[0], AssertionError))
def test_fit_generator():
    model = get_model(num_outputs=2)
    optimizer = 'rmsprop'
    loss = 'mse'
    loss_weights = [1., 0.5]

    model.compile(optimizer, loss, metrics=[], loss_weights=loss_weights,
                  sample_weight_mode=None)
    tracker_cb = TrackerCallback()
    val_seq = RandomSequence(4)
    out = model.fit_generator(generator=RandomSequence(3),
                              steps_per_epoch=3,
                              epochs=5,
                              initial_epoch=0,
                              validation_data=val_seq,
                              validation_steps=3,
                              max_queue_size=1,
                              callbacks=[tracker_cb])
    assert tracker_cb.trained_epochs == [0, 1, 2, 3, 4]
    assert tracker_cb.trained_batches == list(range(3)) * 5
    assert len(val_seq.logs) <= 4 * 5

    tracker_cb = TrackerCallback()
    val_seq = RandomSequence(4)
    out = model.fit(RandomSequence(3),
                    steps_per_epoch=3,
                    epochs=5,
                    initial_epoch=0,
                    validation_data=val_seq,
                    validation_steps=3,
                    max_queue_size=1,
                    callbacks=[tracker_cb])
    assert tracker_cb.trained_epochs == [0, 1, 2, 3, 4]
    assert tracker_cb.trained_batches == list(range(3)) * 5
    assert len(val_seq.logs) <= 4 * 5

    # steps_per_epoch will be equal to len of sequence if it's unspecified
    tracker_cb = TrackerCallback()
    val_seq = RandomSequence(4)
    out = model.fit_generator(generator=RandomSequence(3),
                              epochs=5,
                              initial_epoch=0,
                              validation_data=val_seq,
                              callbacks=[tracker_cb],
                              max_queue_size=1)
    assert tracker_cb.trained_epochs == [0, 1, 2, 3, 4]
    assert tracker_cb.trained_batches == list(range(12)) * 5
    assert 12 * 5 <= len(val_seq.logs) <= (12 * 5) + 2  # the queue may be full.

    tracker_cb = TrackerCallback()
    val_seq = RandomSequence(4)
    out = model.fit(RandomSequence(3),
                    epochs=5,
                    initial_epoch=0,
                    validation_data=val_seq,
                    callbacks=[tracker_cb],
                    max_queue_size=1)
    assert tracker_cb.trained_epochs == [0, 1, 2, 3, 4]
    assert tracker_cb.trained_batches == list(range(12)) * 5
    assert 12 * 5 <= len(val_seq.logs) <= (12 * 5) + 2  # the queue may be full.

    # test for workers = 0
    tracker_cb = TrackerCallback()
    val_seq = RandomSequence(4)
    out = model.fit_generator(generator=RandomSequence(3),
                              epochs=5,
                              validation_data=val_seq,
                              callbacks=[tracker_cb],
                              workers=0)
    assert tracker_cb.trained_epochs == [0, 1, 2, 3, 4]
    assert tracker_cb.trained_batches == list(range(12)) * 5
    assert len(val_seq.logs) == 12 * 5

    tracker_cb = TrackerCallback()
    val_seq = RandomSequence(4)
    out = model.fit(RandomSequence(3),
                    steps_per_epoch=3,
                    epochs=5,
                    initial_epoch=0,
                    validation_data=val_seq,
                    validation_steps=3,
                    max_queue_size=1,
                    callbacks=[tracker_cb])
    assert tracker_cb.trained_epochs == [0, 1, 2, 3, 4]
    assert tracker_cb.trained_batches == list(range(3)) * 5
    assert len(val_seq.logs) <= 4 * 5

    # fit_generator will throw an exception
    # if steps is unspecified for regular generator
    with pytest.raises(ValueError):
        @threadsafe_generator
        def gen_data():
            while True:
                yield (np.asarray([]), np.asarray([]))

        out = model.fit_generator(generator=gen_data(), epochs=5,
                                  initial_epoch=0, validation_data=gen_data(),
                                  callbacks=[tracker_cb])

    # Check if generator is only accessed an expected number of times
    gen_counters = [0, 0]

    @threadsafe_generator
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
    max_train = 3 * 2 + 2 * 2
    min_train = 2 * 3
    assert min_train <= gen_counters[0] <= max_train
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


def test_fit_generator_dynamic_size_sequence_with_workers():
    model = get_model(num_outputs=2)
    optimizer = 'rmsprop'
    loss = 'mse'
    loss_weights = [1., 0.5]

    model.compile(optimizer, loss, metrics=[], loss_weights=loss_weights,
                  sample_weight_mode=None)
    tracker_cb = TrackerCallback()
    val_seq = RandomSequence(4)
    train_seq = IncreaseBatchSizeRandomSequence(3, 20)
    out = model.fit_generator(generator=train_seq,
                              epochs=5,
                              initial_epoch=0,
                              validation_data=val_seq,
                              validation_steps=3,
                              max_queue_size=1,
                              callbacks=[tracker_cb])
    assert tracker_cb.trained_epochs == [0, 1, 2, 3, 4]
    assert tracker_cb.trained_batches == [
        0, 1, 2, 3, 4, 5, 6,  # 1st epoch -> ceil(20 / 3) = 7 batches
        0, 1, 2, 3,           # 2nd epoch -> ceil(20 / 5) = 4 batches
        0, 1, 2,              # 3d  epoch -> ceil(20 / 7) = 3 batches
        0, 1, 2,              # 4th epoch -> ceil(20 / 9) = 3 batches
        0, 1,                 # 5th epoch -> ceil(20 /11) = 2 batches
    ]
    assert tracker_cb.steps_per_epoch_log[0:5] == [7, 4, 3, 3, 2]

    tracker_cb = TrackerCallback()
    val_seq = RandomSequence(4)
    train_seq = IncreaseBatchSizeRandomSequence(3, 30)
    out = model.fit_generator(generator=train_seq,
                              epochs=5,
                              initial_epoch=0,
                              validation_data=val_seq,
                              validation_steps=3,
                              max_queue_size=1,
                              callbacks=[tracker_cb])
    assert tracker_cb.trained_epochs == [0, 1, 2, 3, 4]
    assert tracker_cb.trained_batches == [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  # 1st epoch -> ceil(30 / 3) = 10 batches
        0, 1, 2, 3, 4, 5,              # 2nd epoch -> ceil(30 / 5) =  6 batches
        0, 1, 2, 3, 4,                 # 3d  epoch -> ceil(30 / 7) =  5 batches
        0, 1, 2, 3,                    # 4th epoch -> ceil(30 / 9) =  4 batches
        0, 1, 2,                       # 5th epoch -> ceil(30 /11) =  3 batches
    ]
    assert tracker_cb.steps_per_epoch_log[0:5] == [10, 6, 5, 4, 3]

    tracker_cb = TrackerCallback()
    val_seq = RandomSequence(4)
    train_seq = IncreaseBatchSizeRandomSequence(2, 404, lambda x: x * 2)
    out = model.fit_generator(generator=train_seq,
                              epochs=5,
                              initial_epoch=0,
                              validation_data=val_seq,
                              validation_steps=3,
                              max_queue_size=1,
                              callbacks=[tracker_cb])
    assert tracker_cb.trained_epochs == [0, 1, 2, 3, 4]
    # number of trained batches should match sum of steps per each epoch
    assert len(tracker_cb.trained_batches) == 202 + 101 + 51 + 26 + 13
    assert tracker_cb.steps_per_epoch_log[0:5] == [202, 101, 51, 26, 13]


def test_fit_generator_dynamic_size_sequence_main_thread():
    model = get_model(num_outputs=2)
    optimizer = 'rmsprop'
    loss = 'mse'
    loss_weights = [1., 0.5]

    model.compile(optimizer, loss, metrics=[], loss_weights=loss_weights,
                  sample_weight_mode=None)
    tracker_cb = TrackerCallback()
    val_seq = RandomSequence(4)
    train_seq = IncreaseBatchSizeRandomSequence(3, 20)
    out = model.fit_generator(generator=train_seq,
                              epochs=5,
                              initial_epoch=0,
                              validation_data=val_seq,
                              validation_steps=3,
                              workers=0,
                              callbacks=[tracker_cb])
    assert tracker_cb.trained_epochs == [0, 1, 2, 3, 4]
    assert tracker_cb.trained_batches == [
        0, 1, 2, 3, 4, 5, 6,  # 1st epoch -> ceil(20 / 3) = 7 batches
        0, 1, 2, 3,           # 2nd epoch -> ceil(20 / 5) = 4 batches
        0, 1, 2,              # 3d  epoch -> ceil(20 / 7) = 3 batches
        0, 1, 2,              # 4th epoch -> ceil(20 / 9) = 3 batches
        0, 1,                 # 5th epoch -> ceil(20 /11) = 2 batches
    ]
    assert tracker_cb.steps_per_epoch_log[0:5] == [7, 4, 3, 3, 2]

    tracker_cb = TrackerCallback()
    val_seq = RandomSequence(4)
    train_seq = IncreaseBatchSizeRandomSequence(3, 30)
    out = model.fit_generator(generator=train_seq,
                              epochs=5,
                              initial_epoch=0,
                              validation_data=val_seq,
                              validation_steps=3,
                              workers=0,
                              callbacks=[tracker_cb])
    assert tracker_cb.trained_epochs == [0, 1, 2, 3, 4]
    assert tracker_cb.trained_batches == [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  # 1st epoch -> ceil(30 / 3) = 10 batches
        0, 1, 2, 3, 4, 5,              # 2nd epoch -> ceil(30 / 5) =  6 batches
        0, 1, 2, 3, 4,                 # 3d  epoch -> ceil(30 / 7) =  5 batches
        0, 1, 2, 3,                    # 4th epoch -> ceil(30 / 9) =  4 batches
        0, 1, 2,                       # 5th epoch -> ceil(30 /11) =  3 batches
    ]
    assert tracker_cb.steps_per_epoch_log[0:5] == [10, 6, 5, 4, 3]

    tracker_cb = TrackerCallback()
    val_seq = RandomSequence(4)
    train_seq = IncreaseBatchSizeRandomSequence(2, 404, lambda x: x * 2)
    out = model.fit_generator(generator=train_seq,
                              epochs=5,
                              initial_epoch=0,
                              validation_data=val_seq,
                              validation_steps=3,
                              workers=0,
                              callbacks=[tracker_cb])
    assert tracker_cb.trained_epochs == [0, 1, 2, 3, 4]
    # number of trained batches should match sum of steps per each epoch
    assert len(tracker_cb.trained_batches) == 202 + 101 + 51 + 26 + 13
    assert tracker_cb.steps_per_epoch_log[0:5] == [202, 101, 51, 26, 13]


def test_fit_generator_shape():
    # predict_generator output shape behavior should be consistent
    def expected_shape(batch_size, n_batches):
        return (batch_size * n_batches, 4), (batch_size * n_batches, 3)

    model = get_model(num_outputs=2)
    optimizer = 'rmsprop'
    loss = 'mse'

    # Multiple outputs and one step.
    batch_size = 5
    sequence_length = 1
    shape_0, shape_1 = expected_shape(batch_size, sequence_length)
    out = model.predict_generator(
        RandomSequence(batch_size, sequence_length=sequence_length))
    assert np.shape(out[0]) == shape_0 and np.shape(out[1]) == shape_1

    out = model.predict(
        RandomSequence(batch_size, sequence_length=sequence_length))
    assert np.shape(out[0]) == shape_0 and np.shape(out[1]) == shape_1

    # Multiple outputs and multiple steps.
    batch_size = 5
    sequence_length = 2
    shape_0, shape_1 = expected_shape(batch_size, sequence_length)
    out = model.predict_generator(
        RandomSequence(batch_size, sequence_length=sequence_length))
    assert np.shape(out[0]) == shape_0 and np.shape(out[1]) == shape_1

    out = model.predict(
        RandomSequence(batch_size, sequence_length=sequence_length))
    assert np.shape(out[0]) == shape_0 and np.shape(out[1]) == shape_1

    # Create a model with a single output.
    single_output_model = get_model(num_outputs=1)
    single_output_model.compile(optimizer, loss,
                                metrics=[], sample_weight_mode=None)

    # Single output and one step.
    batch_size = 5
    sequence_length = 1
    shape_0, _ = expected_shape(batch_size, sequence_length)
    out = single_output_model.predict_generator(
        RandomSequence(batch_size, sequence_length=sequence_length))
    assert np.shape(out) == shape_0

    out = single_output_model.predict(
        RandomSequence(batch_size, sequence_length=sequence_length))
    assert np.shape(out) == shape_0

    # Single output and multiple steps.
    batch_size = 5
    sequence_length = 2
    shape_0, _ = expected_shape(batch_size, sequence_length)
    out = single_output_model.predict_generator(
        RandomSequence(batch_size, sequence_length=sequence_length))
    assert np.shape(out) == shape_0

    out = single_output_model.predict(
        RandomSequence(batch_size, sequence_length=sequence_length))
    assert np.shape(out) == shape_0


def test_training_with_loss_instance():
    a = Input(shape=(3,), name='input_a')
    b = Input(shape=(3,), name='input_b')

    dense = Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = Dropout(0.5, name='dropout')(c)

    model = Model([a, b], [d, e])
    loss_weights = [1., 0.5]
    model.compile(
        'sgd',
        loss=losses.MeanSquaredError(),
        metrics=['mae'],
        loss_weights=loss_weights)

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_d_np = np.random.random((10, 4))
    output_e_np = np.random.random((10, 4))

    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              epochs=1,
              batch_size=5)


@pytest.mark.skipif(sys.version_info < (3,),
                    reason='Cannot catch warnings in python 2')
def DISABLED_test_warnings():
    """This test hangs Travis."""
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

    @threadsafe_generator
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
    assert all(['Sequence' not in str(w_.message) for w_ in w]), (
        'A warning was raised for Sequence.')


@pytest.mark.skipif(K.backend() == 'tensorflow',
                    reason='Must for for tf.keras to support sparse ops.')
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


@pytest.mark.skipif(K.backend() != 'tensorflow',
                    reason='sparse operations supported only by TensorFlow')
def DISABLED_test_sparse_placeholder_fit():
    """Must wait for tf.keras to support sparse operations."""
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


def test_with_list_as_targets():
    model = Sequential()
    model.add(Dense(1, input_dim=3, trainable=False))
    model.compile('rmsprop', 'mse')

    x = np.random.random((2, 3))
    y = [0, 1]
    model.train_on_batch(x, y)


def test_check_not_failing():
    a = np.random.random((2, 1, 3))
    training_utils.check_loss_and_target_compatibility(
        [a], [losses.categorical_crossentropy], [a.shape])
    training_utils.check_loss_and_target_compatibility(
        [a], [losses.categorical_crossentropy], [(2, None, 3)])


def test_check_last_is_one():
    a = np.random.random((2, 3, 1))
    with pytest.raises(ValueError,
                       match='You are passing a target array'):
        training_utils.check_loss_and_target_compatibility(
            [a], [losses.CategoricalCrossentropy()], [a.shape])


def test_check_bad_shape():
    a = np.random.random((2, 3, 5))
    with pytest.raises(ValueError,
                       match='targets to have the same shape'):
        training_utils.check_loss_and_target_compatibility(
            [a], [losses.CategoricalCrossentropy()], [(2, 3, 6)])


@pytest.mark.skipif(K.backend() != 'tensorflow',
                    reason='Requires TensorFlow backend')
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
        @threadsafe_generator
        def data_tensors_generator():
            while True:
                yield (None, None)

        generator = data_tensors_generator()

        # test fit_generator for framework-native data tensors
        out = model.fit_generator(generator, epochs=1,
                                  steps_per_epoch=3)

        # test evaluate_generator for framework-native data tensors
        out = model.evaluate_generator(generator, steps=3)
        out = model.evaluate(generator, steps=3)

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

    # single-output, as tensor
    model.compile(optimizer='rmsprop', loss='mse',
                  target_tensors=target)
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

    # multi-output, not enough target tensors when `target_tensors` is not a dict
    with pytest.raises(ValueError,
                       match='When passing a list as `target_tensors`, it should '
                             'have one entry per model output. The model has \\d '
                             'outputs, but you passed target_tensors='):
        model.compile(optimizer='rmsprop', loss='mse',
                      target_tensors=[target_a])
    with pytest.raises(ValueError,
                       match='The model has \\d outputs, but you passed a single '
                             'tensor as `target_tensors`. Expected a list or '
                             'a dict of tensors.'):
        model.compile(optimizer='rmsprop', loss='mse',
                      target_tensors=target_a)

    # test with sample weights
    model.compile(optimizer='rmsprop', loss='mse',
                  target_tensors=[target_a, target_b])
    model.train_on_batch(input_val, None,
                         sample_weight={'dense_a': np.random.random((10,))})


@pytest.mark.skipif(K.backend() == 'tensorflow' and
                    tf.__version__.startswith('2'),
                    reason='Cannot have tensors as dict keys in TF2')
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

    # test with custom placeholder as target
    pl_target_a = K.placeholder(shape=(None, 4))
    model.compile(optimizer='rmsprop', loss='mse',
                  target_tensors={'dense_1': pl_target_a})
    model.train_on_batch([input_a_np, input_b_np],
                         [output_a_np, output_b_np])


@pytest.mark.skipif(sys.version_info < (3,),
                    reason='Cannot catch warnings in python 2')
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
    assert warning_raised, (
        'No warning raised when trainable is modified without .compile.')

    # And on .fit()
    with pytest.warns(UserWarning) as w:
        model2.fit(x=np.zeros((5, 3)), y=np.zeros((5, 1)))
    warning_raised = any(['Discrepancy' in str(w_.message) for w_ in w])
    assert warning_raised, (
        'No warning raised when trainable is modified without .compile.')

    # And shouldn't warn if we recompile
    model2.compile(optimizer='adam', loss='mse')
    with pytest.warns(None) as w:
        model2.summary()
    assert len(w) == 0, (
        'Warning raised even when .compile() is called after modifying .trainable')


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


@pytest.mark.skipif(K.backend() != 'tensorflow', reason='Requires TensorFlow')
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


@pytest.mark.skipif(K.backend() != 'tensorflow', reason='Requires TensorFlow')
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
    with pytest.raises(ValueError,
                       match='should specify the `steps_per_epoch`'):
        model.fit(
            [input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
            epochs=1,
            batch_size=5,
            verbose=0)

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
    with pytest.raises(ValueError,
                       match='you cannot use `validation_split`'):
        model.fit(
            [input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
            epochs=2,
            steps_per_epoch=2,
            verbose=0,
            validation_split=0.2,
            validation_steps=2)

    # Test evaluation / prediction methods
    model.evaluate([input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
                   steps=2, verbose=0)
    model.predict([input_a_tf, input_b_tf], steps=2)
    model.test_on_batch([input_a_tf, input_b_tf], [output_d_tf, output_e_tf])


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
    assert K.image_data_format() == 'channels_first'
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


def test_sample_weights():
    y = np.array([0, 1, 0, 0, 2])
    sample_weights = np.array([0.5, 1., 1., 0., 2.])
    class_weights = {0: 0.5, 1: 1., 2: 1.5}

    # Only `sample_weights`.
    weights = training_utils.standardize_weights(y, sample_weights)
    assert np.allclose(weights, sample_weights)

    # Only `class_weights`.
    weights = training_utils.standardize_weights(y, class_weight=class_weights)
    assert np.allclose(weights, np.array([0.5, 1., 0.5, 0.5, 1.5]))

    # Both 'sample_weights` and 'class_weights`.
    weights = training_utils.standardize_weights(y, sample_weights,
                                                 class_weights)
    expected = sample_weights * np.array([0.5, 1., 0.5, 0.5, 1.5])
    assert np.allclose(weights, expected)


def test_validation_freq():
    model = Sequential([Dense(1)])
    model.compile('sgd', 'mse')

    def _gen():
        while True:
            yield np.ones((2, 10)), np.ones((2, 1))

    x, y = np.ones((10, 10)), np.ones((10, 1))

    class ValCounter(Callback):

        def __init__(self):
            self.val_runs = 0

        def on_test_begin(self, logs=None):
            self.val_runs += 1

    # Test in training_arrays.py
    val_counter = ValCounter()
    model.fit(
        x,
        y,
        batch_size=2,
        epochs=4,
        validation_data=(x, y),
        validation_freq=2,
        callbacks=[val_counter])
    assert val_counter.val_runs == 2

    # Test in training_generator.py
    val_counter = ValCounter()
    model.fit_generator(
        _gen(),
        epochs=4,
        steps_per_epoch=5,
        validation_data=(x, y),
        validation_freq=[4, 2, 2, 1],
        callbacks=[val_counter])
    assert val_counter.val_runs == 3


def test_loss_correctness():
    class Bias(Layer):

        def build(self, input_shape):
            self.bias = self.add_weight('bias', (1,), initializer='zeros')

        def call(self, inputs):
            return inputs + self.bias

    inp = Input(shape=(1,))
    out = Bias()(inp)
    model = Model(inp, out)
    model.compile(
        keras.optimizers.SGD(lr=0.1),
        loss=keras.losses.MeanAbsoluteError())

    x = np.array([[0.], [1.], [2.]])
    y = np.array([[0.5], [2.], [3.5]])
    history = model.fit(x, y, batch_size=3, epochs=5)
    np.allclose(history.history['loss'], [1., 0.9, 0.8, 0.7, 0.6])


def test_model_metrics_list():

    class LayerWithAddMetric(Layer):

        def __init__(self):
            super(LayerWithAddMetric, self).__init__()
            self.dense = keras.layers.Dense(1, kernel_initializer='ones')

        def __call__(self, inputs):
            outputs = self.dense(inputs)
            return outputs

    class LayerWithNestedAddMetricLayer(Layer):

        def __init__(self):
            super(LayerWithNestedAddMetricLayer, self).__init__()
            self.layer = LayerWithAddMetric()

        def call(self, inputs):
            outputs = self.layer(inputs)
            self.add_metric(K.sum(outputs), name='metric_4')
            return outputs

    x = Input(shape=(1,))
    y = LayerWithNestedAddMetricLayer()(x)

    model = keras.models.Model(x, y)
    model.add_metric(K.sum(y), name='metric_2')
    model.add_metric(metrics.Mean(name='metric_3')(y))

    model.compile(
        'sgd',
        loss='mse',
        metrics=[metrics.MeanSquaredError('metric_1')])

    # Verify that the metrics added using `compile` and `add_metric` API are
    # included
    for m1, m2 in zip([m.name for m in model._compile_metrics], ['metric_1']):
        assert m1 == m2

    for m1, m2 in zip(
            [m.name for m in model.metrics],
            ['metric_1', 'metric_2', 'metric_3', 'metric_4']):
        assert m1 == m2


def test_model_metrics_list_in_call():

    class TestModel(Model):

        def __init__(self):
            super(TestModel, self).__init__(name='test_model')
            self.dense1 = keras.layers.Dense(2)

        def call(self, x):
            self.add_metric(K.sum(x), name='metric_2')
            return self.dense1(x)

    model = TestModel()
    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=[metrics.MeanSquaredError('metric_1')])
    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

    # Verify that the metrics added using `compile` and `add_metric` API are
    # included
    for m1, m2 in zip([m.name for m in model._compile_metrics], ['metric_1']):
        assert m1 == m2

    for m1, m2 in zip(
            [m.name for m in model.metrics],
            ['metric_1', 'metric_2']):
        assert m1 == m2


def test_duplicate_metric_name_in_add_metric():

    class TestModel(Model):

        def __init__(self):
            super(TestModel, self).__init__(name='test_model')
            self.dense1 = keras.layers.Dense(2, kernel_initializer='ones')
            self.mean = metrics.Mean(name='metric_1')
            self.mean2 = metrics.Mean(name='metric_1')

        def call(self, x):
            self.add_metric(self.mean(x), name='metric_1')
            return self.dense1(x)

    model = TestModel()
    model.compile(loss='mse', optimizer='adam')

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    with pytest.raises(ValueError):
        model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))


def test_add_metric_on_model():
    x = Input(shape=(1,))
    y = Dense(1, kernel_initializer='ones', trainable=False)(x)
    model = Model(x, y)
    model.add_metric(K.sum(y), name='metric_1')
    model.add_metric(metrics.Mean(name='metric_2')(y))
    model.compile('sgd', loss='mse', metrics=['mse'])

    inputs = np.ones(shape=(10, 1))
    targets = np.zeros(shape=(10, 1))
    history = model.fit(
        inputs,
        targets,
        epochs=2,
        batch_size=5,
        validation_data=(inputs, targets))
    assert history.history['metric_1'][-1] == 5
    assert history.history['val_metric_1'][-1] == 5

    assert history.history['metric_2'][-1] == 1
    assert history.history['val_metric_2'][-1] == 1

    eval_results = model.evaluate(inputs, targets, batch_size=5)
    assert eval_results[-2] == 5
    assert eval_results[-1] == 1

    model.predict(inputs, batch_size=5)
    model.train_on_batch(inputs, targets)
    model.test_on_batch(inputs, targets)


def test_add_metric_in_model_call():

    class TestModel(Model):

        def __init__(self):
            super(TestModel, self).__init__(name='test_model')
            self.dense1 = keras.layers.Dense(2, kernel_initializer='ones')
            self.mean = metrics.Mean(name='metric_1')

        def call(self, x):
            self.add_metric(K.sum(x), name='metric_2')
            # Provide same name as in the instance created in __init__
            # for eager mode
            self.add_metric(self.mean(x), name='metric_1')
            return self.dense1(x)

    model = TestModel()
    model.compile(loss='mse', optimizer='sgd')

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    history = model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))
    assert np.isclose(history.history['metric_1'][-1], 1, 0)
    assert np.isclose(history.history['val_metric_1'][-1], 1, 0)
    assert np.isclose(history.history['metric_2'][-1], 5, 0)
    assert np.isclose(history.history['val_metric_2'][-1], 5, 0)

    eval_results = model.evaluate(x, y, batch_size=5)
    assert np.isclose(eval_results[1], 1, 0)
    assert np.isclose(eval_results[2], 5, 0)

    model.predict(x, batch_size=5)
    model.train_on_batch(x, y)
    model.test_on_batch(x, y)


def test_multiple_add_metric_calls():

    class TestModel(Model):

        def __init__(self):
            super(TestModel, self).__init__(name='test_model')
            self.dense1 = keras.layers.Dense(2, kernel_initializer='ones')
            self.mean1 = metrics.Mean(name='metric_1')
            self.mean2 = metrics.Mean(name='metric_2')

        def call(self, x):
            self.add_metric(self.mean2(x), name='metric_2')
            self.add_metric(self.mean1(x), name='metric_1')
            self.add_metric(K.sum(x), name='metric_3')
            return self.dense1(x)

    model = TestModel()
    model.compile(loss='mse', optimizer='sgd')

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    history = model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))
    assert np.isclose(history.history['metric_1'][-1], 1, 0)
    assert np.isclose(history.history['metric_2'][-1], 1, 0)
    assert np.isclose(history.history['metric_3'][-1], 5, 0)

    eval_results = model.evaluate(x, y, batch_size=5)
    assert np.allclose(eval_results[1:4], [1, 1, 5], 0.1)

    model.predict(x, batch_size=5)
    model.train_on_batch(x, y)
    model.test_on_batch(x, y)


def test_add_metric_in_layer_call():

    class TestLayer(Layer):

        def build(self, input_shape):
            self.a = self.add_weight(
                'a', (1, 1), initializer='ones', trainable=False)
            self.built = True

        def call(self, inputs):
            self.add_metric(K.sum(inputs), name='metric_1')
            return inputs + 1

    inp = Input(shape=(1,))
    x = TestLayer(input_shape=(1,))(inp)
    x = keras.layers.Dense(2, kernel_initializer='ones')(x)

    model = Model(inp, x)
    model.compile('adam', loss='mse')

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    history = model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))
    assert np.isclose(history.history['metric_1'][-1], 5, 0)
    assert np.isclose(history.history['val_metric_1'][-1], 5, 0)


if __name__ == '__main__':
    pytest.main([__file__])
