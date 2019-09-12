import os
import multiprocessing

import numpy as np
import pytest
from numpy.testing import assert_allclose
from csv import reader
from csv import Sniffer
import shutil
from collections import defaultdict
from keras import optimizers
from keras import initializers
from keras import callbacks
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, add, dot, Lambda, Layer
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.utils.test_utils import get_test_data
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import unpack_singleton
from keras import backend as K
from keras.utils import np_utils

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


input_dim = 2
num_hidden = 4
num_classes = 2
batch_size = 5
train_samples = 20
test_samples = 20


def data_generator(x, y, batch_size):
    x = to_list(x)
    y = to_list(y)
    max_batch_index = len(x[0]) // batch_size
    i = 0
    while 1:
        x_batch = [array[i * batch_size: (i + 1) * batch_size] for array in x]
        x_batch = unpack_singleton(x_batch)

        y_batch = [array[i * batch_size: (i + 1) * batch_size] for array in y]
        y_batch = unpack_singleton(y_batch)
        yield x_batch, y_batch
        i += 1
        i = i % max_batch_index


# Changing the default arguments of get_test_data.
def get_data_callbacks(num_train=train_samples,
                       num_test=test_samples,
                       input_shape=(input_dim,),
                       classification=True,
                       num_classes=num_classes):
    return get_test_data(num_train=num_train,
                         num_test=num_test,
                         input_shape=input_shape,
                         classification=classification,
                         num_classes=num_classes)


class Counter(callbacks.Callback):
    """Counts the number of times each callback method was run.

    # Arguments
        method_counts: dict, contains the counts of time
            each callback method was run.
    """

    def __init__(self):
        self.method_counts = defaultdict(int)
        methods_to_count = [
            'on_batch_begin', 'on_batch_end', 'on_epoch_begin', 'on_epoch_end',
            'on_train_batch_begin', 'on_train_batch_end',
            'on_test_batch_begin', 'on_test_batch_end',
            'on_predict_batch_begin', 'on_predict_batch_end',
            'on_train_begin', 'on_train_end',
            'on_predict_begin', 'on_predict_end',
            'on_test_begin', 'on_test_end',
        ]
        for method_name in methods_to_count:
            setattr(self, method_name,
                    self.wrap_with_counts(
                        method_name, getattr(self, method_name)))

    def wrap_with_counts(self, method_name, method):

        def _call_and_count(*args, **kwargs):
            self.method_counts[method_name] += 1
            return method(*args, **kwargs)

        return _call_and_count


class TestCallbackCounts(object):

    def _check_counts(self, counter, expected_counts):
        """Checks that counts registered by `counter` are those expected."""
        for method_name, expected_count in expected_counts.items():
            count = counter.method_counts[method_name]
            assert count == expected_count, \
                'For method {}: expected {}, got: {}'.format(
                    method_name, expected_count, count)

    def _get_model(self):
        layers = [
            Dense(10, activation='relu', input_dim=input_dim),
            Dense(num_classes, activation='softmax')
        ]
        model = Sequential(layers=layers)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def test_callback_hooks_are_called_in_fit(self):
        np.random.seed(1337)
        (X_train, y_train), (X_test, y_test) = get_data_callbacks(num_train=10,
                                                                  num_test=4)
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        model = self._get_model()
        counter = Counter()
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  batch_size=2, epochs=5, callbacks=[counter])

        self._check_counts(
            counter, {
                'on_batch_begin': 25,
                'on_batch_end': 25,
                'on_epoch_begin': 5,
                'on_epoch_end': 5,
                'on_predict_batch_begin': 0,
                'on_predict_batch_end': 0,
                'on_predict_begin': 0,
                'on_predict_end': 0,
                'on_test_batch_begin': 10,
                'on_test_batch_end': 10,
                'on_test_begin': 5,
                'on_test_end': 5,
                'on_train_batch_begin': 25,
                'on_train_batch_end': 25,
                'on_train_begin': 1,
                'on_train_end': 1,
            })

    def test_callback_hooks_are_called_in_evaluate(self):
        np.random.seed(1337)
        (_, _), (X_test, y_test) = get_data_callbacks(num_test=10)

        y_test = np_utils.to_categorical(y_test)

        model = self._get_model()
        counter = Counter()
        model.evaluate(X_test, y_test, batch_size=2, callbacks=[counter])
        self._check_counts(
            counter, {
                'on_test_batch_begin': 5,
                'on_test_batch_end': 5,
                'on_test_begin': 1,
                'on_test_end': 1,
                'on_batch_begin': 0,
                'on_batch_end': 0,
                'on_epoch_begin': 0,
                'on_epoch_end': 0,
                'on_predict_batch_begin': 0,
                'on_predict_batch_end': 0,
                'on_predict_begin': 0,
                'on_predict_end': 0,
                'on_train_batch_begin': 0,
                'on_train_batch_end': 0,
                'on_train_begin': 0,
                'on_train_end': 0,
            })

    def test_callback_hooks_are_called_in_predict(self):
        np.random.seed(1337)
        (_, _), (X_test, _) = get_data_callbacks(num_test=10)

        model = self._get_model()
        counter = Counter()
        model.predict(X_test, batch_size=2, callbacks=[counter])
        self._check_counts(
            counter, {
                'on_predict_batch_begin': 5,
                'on_predict_batch_end': 5,
                'on_predict_begin': 1,
                'on_predict_end': 1,
                'on_batch_begin': 0,
                'on_batch_end': 0,
                'on_epoch_begin': 0,
                'on_epoch_end': 0,
                'on_test_batch_begin': 0,
                'on_test_batch_end': 0,
                'on_test_begin': 0,
                'on_test_end': 0,
                'on_train_batch_begin': 0,
                'on_train_batch_end': 0,
                'on_train_begin': 0,
                'on_train_end': 0,
            })

    def test_callback_hooks_are_called_in_fit_generator(self):
        np.random.seed(1337)
        (X_train, y_train), (X_test, y_test) = get_data_callbacks(num_train=10,
                                                                  num_test=4)
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        train_generator = data_generator(X_train, y_train, batch_size=2)
        validation_generator = data_generator(X_test, y_test, batch_size=2)

        model = self._get_model()
        counter = Counter()
        model.fit_generator(train_generator,
                            steps_per_epoch=len(X_train) // 2,
                            epochs=5,
                            validation_data=validation_generator,
                            validation_steps=len(X_test) // 2,
                            callbacks=[counter])

        self._check_counts(
            counter, {
                'on_batch_begin': 25,
                'on_batch_end': 25,
                'on_epoch_begin': 5,
                'on_epoch_end': 5,
                'on_predict_batch_begin': 0,
                'on_predict_batch_end': 0,
                'on_predict_begin': 0,
                'on_predict_end': 0,
                'on_test_batch_begin': 10,
                'on_test_batch_end': 10,
                'on_test_begin': 5,
                'on_test_end': 5,
                'on_train_batch_begin': 25,
                'on_train_batch_end': 25,
                'on_train_begin': 1,
                'on_train_end': 1,
            })

    def test_callback_hooks_are_called_in_evaluate_generator(self):
        np.random.seed(1337)
        (_, _), (X_test, y_test) = get_data_callbacks(num_test=10)
        y_test = np_utils.to_categorical(y_test)

        model = self._get_model()
        counter = Counter()
        model.evaluate_generator(data_generator(X_test, y_test, batch_size=2),
                                 steps=len(X_test) // 2, callbacks=[counter])
        self._check_counts(
            counter, {
                'on_test_batch_begin': 5,
                'on_test_batch_end': 5,
                'on_test_begin': 1,
                'on_test_end': 1,
                'on_batch_begin': 0,
                'on_batch_end': 0,
                'on_epoch_begin': 0,
                'on_epoch_end': 0,
                'on_predict_batch_begin': 0,
                'on_predict_batch_end': 0,
                'on_predict_begin': 0,
                'on_predict_end': 0,
                'on_train_batch_begin': 0,
                'on_train_batch_end': 0,
                'on_train_begin': 0,
                'on_train_end': 0,
            })

    def test_callback_hooks_are_called_in_predict_generator(self):
        np.random.seed(1337)
        (_, _), (X_test, _) = get_data_callbacks(num_test=10)

        def data_generator(x, batch_size):
            x = to_list(x)
            max_batch_index = len(x[0]) // batch_size
            i = 0
            while 1:
                x_batch = [
                    array[i * batch_size: (i + 1) * batch_size] for array in x]
                x_batch = unpack_singleton(x_batch)

                yield x_batch
                i += 1
                i = i % max_batch_index

        model = self._get_model()
        counter = Counter()
        model.predict_generator(data_generator(X_test, batch_size=2),
                                steps=len(X_test) // 2, callbacks=[counter])
        self._check_counts(
            counter, {
                'on_predict_batch_begin': 5,
                'on_predict_batch_end': 5,
                'on_predict_begin': 1,
                'on_predict_end': 1,
                'on_batch_begin': 0,
                'on_batch_end': 0,
                'on_epoch_begin': 0,
                'on_epoch_end': 0,
                'on_test_batch_begin': 0,
                'on_test_batch_end': 0,
                'on_test_begin': 0,
                'on_test_end': 0,
                'on_train_batch_begin': 0,
                'on_train_batch_end': 0,
                'on_train_begin': 0,
                'on_train_end': 0,
            })

    def test_callback_list_methods(self):
        counter = Counter()
        callback_list = callbacks.CallbackList([counter])

        batch = 0
        callback_list.on_test_batch_begin(batch)
        callback_list.on_test_batch_end(batch)
        callback_list.on_predict_batch_begin(batch)
        callback_list.on_predict_batch_end(batch)

        self._check_counts(
            counter, {
                'on_test_batch_begin': 1,
                'on_test_batch_end': 1,
                'on_predict_batch_begin': 1,
                'on_predict_batch_end': 1,
                'on_predict_begin': 0,
                'on_predict_end': 0,
                'on_batch_begin': 0,
                'on_batch_end': 0,
                'on_epoch_begin': 0,
                'on_epoch_end': 0,
                'on_test_begin': 0,
                'on_test_end': 0,
                'on_train_batch_begin': 0,
                'on_train_batch_end': 0,
                'on_train_begin': 0,
                'on_train_end': 0,
            })


def test_TerminateOnNaN():
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_data_callbacks()

    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    cbks = [callbacks.TerminateOnNaN()]
    model = Sequential()
    initializer = initializers.Constant(value=1e5)
    for _ in range(5):
        model.add(Dense(num_hidden, input_dim=input_dim, activation='relu',
                        kernel_initializer=initializer))
    model.add(Dense(num_classes, activation='linear'))
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop')

    # case 1 fit
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=cbks,
                        epochs=20)
    loss = history.history['loss']
    assert len(loss) == 1
    assert loss[0] == np.inf

    history = model.fit_generator(data_generator(X_train, y_train, batch_size),
                                  len(X_train),
                                  validation_data=(X_test, y_test),
                                  callbacks=cbks,
                                  epochs=20)
    loss = history.history['loss']
    assert len(loss) == 1
    assert loss[0] == np.inf or np.isnan(loss[0])


def test_stop_training_csv(tmpdir):
    np.random.seed(1337)
    fp = str(tmpdir / 'test.csv')
    (X_train, y_train), (X_test, y_test) = get_data_callbacks()

    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    cbks = [callbacks.TerminateOnNaN(), callbacks.CSVLogger(fp)]
    model = Sequential()
    for _ in range(5):
        model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_classes, activation='linear'))
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop')

    def data_generator():
        i = 0
        max_batch_index = len(X_train) // batch_size
        tot = 0
        while 1:
            if tot > 3 * len(X_train):
                yield (np.ones([batch_size, input_dim]) * np.nan,
                       np.ones([batch_size, num_classes]) * np.nan)
            else:
                yield (X_train[i * batch_size: (i + 1) * batch_size],
                       y_train[i * batch_size: (i + 1) * batch_size])
            i += 1
            tot += 1
            i = i % max_batch_index

    history = model.fit_generator(data_generator(),
                                  len(X_train) // batch_size,
                                  validation_data=(X_test, y_test),
                                  callbacks=cbks,
                                  epochs=20)
    loss = history.history['loss']
    assert len(loss) > 1
    assert loss[-1] == np.inf or np.isnan(loss[-1])

    values = []
    with open(fp) as f:
        for x in reader(f):
            values.append(x)

    assert 'nan' in values[-1], 'The last epoch was not logged.'
    os.remove(fp)


def test_ModelCheckpoint(tmpdir):
    np.random.seed(1337)
    filepath = str(tmpdir / 'checkpoint.h5')
    (X_train, y_train), (X_test, y_test) = get_data_callbacks()
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    # case 1
    monitor = 'val_loss'
    save_best_only = False
    mode = 'auto'

    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    cbks = [callbacks.ModelCheckpoint(filepath, monitor=monitor,
                                      save_best_only=save_best_only, mode=mode)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=1)
    assert os.path.isfile(filepath)
    os.remove(filepath)

    # case 2
    mode = 'min'
    cbks = [callbacks.ModelCheckpoint(filepath, monitor=monitor,
                                      save_best_only=save_best_only, mode=mode)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=1)
    assert os.path.isfile(filepath)
    os.remove(filepath)

    # case 3
    mode = 'max'
    monitor = 'val_accuracy'
    cbks = [callbacks.ModelCheckpoint(filepath,
                                      monitor=monitor,
                                      save_best_only=save_best_only,
                                      mode=mode)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=1)
    assert os.path.isfile(filepath)
    os.remove(filepath)

    # case 4
    save_best_only = True
    cbks = [callbacks.ModelCheckpoint(filepath,
                                      monitor=monitor,
                                      save_best_only=save_best_only,
                                      mode=mode)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=1)
    assert os.path.isfile(filepath)
    os.remove(filepath)

    # case 5
    save_best_only = False
    period = 2
    mode = 'auto'
    filepath = 'checkpoint.{epoch:02d}.h5'
    cbks = [callbacks.ModelCheckpoint(filepath,
                                      monitor=monitor,
                                      save_best_only=save_best_only,
                                      mode=mode,
                                      period=period)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=4)
    assert os.path.isfile(filepath.format(epoch=2))
    assert os.path.isfile(filepath.format(epoch=4))
    assert not os.path.exists(filepath.format(epoch=1))
    assert not os.path.exists(filepath.format(epoch=3))
    os.remove(filepath.format(epoch=2))
    os.remove(filepath.format(epoch=4))
    assert not tmpdir.listdir()


def test_EarlyStopping():
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_data_callbacks()
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    mode = 'max'
    monitor = 'val_acc'
    patience = 0
    cbks = [callbacks.EarlyStopping(patience=patience,
                                    monitor=monitor,
                                    mode=mode)]
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=cbks,
                        epochs=20)
    mode = 'auto'
    monitor = 'val_acc'
    patience = 2
    cbks = [callbacks.EarlyStopping(patience=patience,
                                    monitor=monitor,
                                    mode=mode)]
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=cbks,
                        epochs=20)


def test_EarlyStopping_reuse():
    np.random.seed(1337)
    patience = 3
    data = np.random.random((100, 1))
    labels = np.where(data > 0.5, 1, 0)
    model = Sequential((
        Dense(1, input_dim=1, activation='relu'),
        Dense(1, activation='sigmoid'),
    ))
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    stopper = callbacks.EarlyStopping(monitor='acc', patience=patience)
    weights = model.get_weights()

    hist = model.fit(data, labels, callbacks=[stopper], epochs=20)
    assert len(hist.epoch) >= patience

    # This should allow training to go for at least `patience` epochs
    model.set_weights(weights)
    hist = model.fit(data, labels, callbacks=[stopper], epochs=20)
    assert len(hist.epoch) >= patience


def test_EarlyStopping_patience():
    class DummyModel(object):
        def __init__(self):
            self.stop_training = False

        def get_weights(self):
            return []

        def set_weights(self, weights):
            pass

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=2)
    early_stop.model = DummyModel()

    losses = [0.0860, 0.1096, 0.1040, 0.1019]

    # Should stop after epoch 3,
    # as the loss has not improved after patience=2 epochs.
    epochs_trained = 0
    early_stop.on_train_begin()

    for epoch in range(len(losses)):
        epochs_trained += 1
        early_stop.on_epoch_end(epoch, logs={'val_loss': losses[epoch]})

        if early_stop.model.stop_training:
            break

    assert epochs_trained == 3


def test_EarlyStopping_baseline():
    class DummyModel(object):
        def __init__(self):
            self.stop_training = False

        def get_weights(self):
            return []

        def set_weights(self, weights):
            pass

    def baseline_tester(acc_levels):
        early_stop = callbacks.EarlyStopping(monitor='val_acc', baseline=0.75,
                                             patience=2)
        early_stop.model = DummyModel()
        epochs_trained = 0
        early_stop.on_train_begin()
        for epoch in range(len(acc_levels)):
            epochs_trained += 1
            early_stop.on_epoch_end(epoch, logs={'val_acc': acc_levels[epoch]})
            if early_stop.model.stop_training:
                break
        return epochs_trained

    acc_levels = [0.55, 0.76, 0.81, 0.81]
    baseline_met = baseline_tester(acc_levels)
    acc_levels = [0.55, 0.74, 0.81, 0.81]
    baseline_not_met = baseline_tester(acc_levels)

    # All epochs should run because baseline was met in second epoch
    assert baseline_met == 4
    # Baseline was not met by second epoch and should stop
    assert baseline_not_met == 2


def test_EarlyStopping_final_weights():
    class DummyModel(object):
        def __init__(self):
            self.stop_training = False
            self.weights = -1

        def get_weights(self):
            return self.weights

        def set_weights(self, weights):
            self.weights = weights

        def set_weight_to_epoch(self, epoch):
            self.weights = epoch

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=2)
    early_stop.model = DummyModel()

    losses = [0.2, 0.15, 0.1, 0.11, 0.12]

    epochs_trained = 0
    early_stop.on_train_begin()

    for epoch in range(len(losses)):
        epochs_trained += 1
        early_stop.model.set_weight_to_epoch(epoch=epoch)
        early_stop.on_epoch_end(epoch, logs={'val_loss': losses[epoch]})

        if early_stop.model.stop_training:
            break

    # The best configuration is in the epoch 2 (loss = 0.1000),
    # so with patience=2 we need to end up at epoch 4
    assert early_stop.model.get_weights() == 4


def test_EarlyStopping_final_weights_when_restoring_model_weights():
    class DummyModel(object):
        def __init__(self):
            self.stop_training = False
            self.weights = -1

        def get_weights(self):
            return self.weights

        def set_weights(self, weights):
            self.weights = weights

        def set_weight_to_epoch(self, epoch):
            self.weights = epoch

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=2,
                                         restore_best_weights=True)
    early_stop.model = DummyModel()

    losses = [0.2, 0.15, 0.1, 0.11, 0.12]

    # The best configuration is in the epoch 2 (loss = 0.1000).

    epochs_trained = 0
    early_stop.on_train_begin()

    for epoch in range(len(losses)):
        epochs_trained += 1
        early_stop.model.set_weight_to_epoch(epoch=epoch)
        early_stop.on_epoch_end(epoch, logs={'val_loss': losses[epoch]})

        if early_stop.model.stop_training:
            break

    # The best configuration is in epoch 2 (loss = 0.1000),
    # and while patience = 2, we're restoring the best weights,
    # so we end up at the epoch with the best weights, i.e. epoch 2
    assert early_stop.model.get_weights() == 2


def test_LearningRateScheduler():
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_data_callbacks()
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    cbks = [callbacks.LearningRateScheduler(lambda x: 1. / (1. + x))]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=5)
    assert (float(K.get_value(model.optimizer.lr)) - 0.2) < K.epsilon()


def test_ReduceLROnPlateau():
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_data_callbacks()
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    def make_model():
        np.random.seed(1337)
        model = Sequential()
        model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=0.1),
                      metrics=['accuracy'])
        return model

    model = make_model()

    # This should reduce the LR after the first epoch (due to high epsilon).
    cbks = [callbacks.ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.1,
                                        min_delta=10,
                                        patience=1,
                                        cooldown=5)]
    model.fit(X_train, y_train,
              batch_size=batch_size,
              validation_data=(X_test, y_test),
              callbacks=cbks,
              epochs=5,
              verbose=2)
    assert_allclose(
        float(K.get_value(model.optimizer.lr)), 0.01, atol=K.epsilon())

    model = make_model()
    cbks = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                        min_delta=0, patience=1, cooldown=5)]
    model.fit(X_train, y_train,
              batch_size=batch_size,
              validation_data=(X_test, y_test),
              callbacks=cbks,
              epochs=5,
              verbose=2)
    assert_allclose(
        float(K.get_value(model.optimizer.lr)), 0.1, atol=K.epsilon())


def test_ReduceLROnPlateau_patience():
    class DummyOptimizer(object):
        def __init__(self):
            self.lr = K.variable(1.0)

    class DummyModel(object):
        def __init__(self):
            self.optimizer = DummyOptimizer()

    reduce_on_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    patience=2)
    reduce_on_plateau.model = DummyModel()

    losses = [0.0860, 0.1096, 0.1040]
    lrs = []

    for epoch in range(len(losses)):
        reduce_on_plateau.on_epoch_end(epoch, logs={'val_loss': losses[epoch]})
        lrs.append(K.get_value(reduce_on_plateau.model.optimizer.lr))

    # The learning rates should be 1.0 except the last one
    assert all([lr == 1.0 for lr in lrs[:-1]]) and lrs[-1] < 1.0


def test_ReduceLROnPlateau_backwards_compatibility():
    import warnings
    with warnings.catch_warnings(record=True) as ws:
        reduce_on_plateau = callbacks.ReduceLROnPlateau(epsilon=1e-13)
        # Check if warnings are disabled
        if os.environ.get("PYTHONWARNINGS") != "ignore":
            assert "`epsilon` argument is deprecated" in str(ws[0].message)
    assert not hasattr(reduce_on_plateau, 'epsilon')
    assert hasattr(reduce_on_plateau, 'min_delta')
    assert reduce_on_plateau.min_delta == 1e-13


def test_CSVLogger(tmpdir):
    np.random.seed(1337)
    filepath = str(tmpdir / 'log.tsv')
    sep = '\t'
    (X_train, y_train), (X_test, y_test) = get_data_callbacks()
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    def make_model():
        np.random.seed(1337)
        model = Sequential()
        model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=0.1),
                      metrics=['accuracy'])
        return model

    # case 1, create new file with defined separator
    model = make_model()
    cbks = [callbacks.CSVLogger(filepath, separator=sep)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=1)

    assert os.path.isfile(filepath)
    with open(filepath) as csvfile:
        dialect = Sniffer().sniff(csvfile.read())
    assert dialect.delimiter == sep
    del model
    del cbks

    # case 2, append data to existing file, skip header
    model = make_model()
    cbks = [callbacks.CSVLogger(filepath, separator=sep, append=True)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=1)

    # case 3, reuse of CSVLogger object
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=2)

    import re
    with open(filepath) as csvfile:
        list_lines = csvfile.readlines()
        for line in list_lines:
            assert line.count(sep) == 4
        assert len(list_lines) == 5
        output = " ".join(list_lines)
        assert len(re.findall('epoch', output)) == 1

    os.remove(filepath)
    assert not tmpdir.listdir()


def test_CallbackValData():
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_data_callbacks()
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    cbk = callbacks.LambdaCallback(on_train_end=lambda x: 1)
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=[cbk], epochs=1)

    cbk2 = callbacks.LambdaCallback(on_train_end=lambda x: 1)
    train_generator = data_generator(X_train, y_train, batch_size)
    model.fit_generator(train_generator, len(X_train), epochs=1,
                        validation_data=(X_test, y_test),
                        callbacks=[cbk2])

    # callback validation data should always have x, y, and sample weights
    assert len(cbk.validation_data) == len(cbk2.validation_data) == 3
    assert cbk.validation_data[0] is cbk2.validation_data[0]
    assert cbk.validation_data[1] is cbk2.validation_data[1]
    assert cbk.validation_data[2].shape == cbk2.validation_data[2].shape


def test_LambdaCallback():
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_data_callbacks()
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # Start an arbitrary process that should run during model training and
    # be terminated after training has completed.
    def f():
        while True:
            pass

    p = multiprocessing.Process(target=f)
    p.start()
    cleanup_callback = callbacks.LambdaCallback(
        on_train_end=lambda logs: p.terminate())

    cbks = [cleanup_callback]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=5)
    p.join()
    assert not p.is_alive()


@pytest.mark.skipif(K.backend() != 'tensorflow', reason='Uses TensorBoard')
def test_TensorBoard_with_ReduceLROnPlateau(tmpdir):
    import shutil
    np.random.seed(np.random.randint(1, 1e7))
    filepath = str(tmpdir / 'logs')

    (X_train, y_train), (X_test, y_test) = get_data_callbacks()
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    cbks = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            verbose=1),
        callbacks.TensorBoard(
            log_dir=filepath)]

    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=2)

    assert os.path.isdir(filepath)
    shutil.rmtree(filepath)
    assert not tmpdir.listdir()


def tests_RemoteMonitor():
    (X_train, y_train), (X_test, y_test) = get_data_callbacks()
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    cbks = [callbacks.RemoteMonitor()]

    with patch('requests.post'):
        model.fit(X_train, y_train, batch_size=batch_size,
                  validation_data=(X_test, y_test), callbacks=cbks, epochs=1)


def tests_RemoteMonitorWithJsonPayload():
    (X_train, y_train), (X_test, y_test) = get_data_callbacks()
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    cbks = [callbacks.RemoteMonitor(send_as_json=True)]

    with patch('requests.post'):
        model.fit(X_train, y_train, batch_size=batch_size,
                  validation_data=(X_test, y_test), callbacks=cbks, epochs=1)


if __name__ == '__main__':
    pytest.main([__file__])
