import os
import multiprocessing

import numpy as np
import pytest
from numpy.testing import assert_allclose
from csv import reader
from csv import Sniffer
import shutil
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
except:
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
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=cbks, epochs=20)
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
    monitor = 'val_acc'
    cbks = [callbacks.ModelCheckpoint(filepath, monitor=monitor,
                                      save_best_only=save_best_only, mode=mode)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=1)
    assert os.path.isfile(filepath)
    os.remove(filepath)

    # case 4
    save_best_only = True
    cbks = [callbacks.ModelCheckpoint(filepath, monitor=monitor,
                                      save_best_only=save_best_only, mode=mode)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=1)
    assert os.path.isfile(filepath)
    os.remove(filepath)

    # case 5
    save_best_only = False
    period = 2
    mode = 'auto'
    filepath = 'checkpoint.{epoch:02d}.h5'
    cbks = [callbacks.ModelCheckpoint(filepath, monitor=monitor,
                                      save_best_only=save_best_only, mode=mode,
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
    cbks = [callbacks.EarlyStopping(patience=patience, monitor=monitor, mode=mode)]
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=cbks, epochs=20)

    mode = 'auto'
    monitor = 'val_acc'
    patience = 2
    cbks = [callbacks.EarlyStopping(patience=patience, monitor=monitor, mode=mode)]
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=cbks, epochs=20)


def test_EarlyStopping_reuse():
    np.random.seed(1337)
    patience = 3
    data = np.random.random((100, 1))
    labels = np.where(data > 0.5, 1, 0)
    model = Sequential((
        Dense(1, input_dim=1, activation='relu'),
        Dense(1, activation='sigmoid'),
    ))
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
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
    cbks = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                        min_delta=10, patience=1, cooldown=5)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=5, verbose=2)
    assert_allclose(float(K.get_value(model.optimizer.lr)), 0.01, atol=K.epsilon())

    model = make_model()
    cbks = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                        min_delta=0, patience=1, cooldown=5)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=5, verbose=2)
    assert_allclose(float(K.get_value(model.optimizer.lr)), 0.1, atol=K.epsilon())


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


@pytest.mark.parametrize('update_freq', ['batch', 'epoch', 9])
def test_TensorBoard(tmpdir, update_freq):
    np.random.seed(np.random.randint(1, 1e7))
    filepath = str(tmpdir / 'logs')

    (X_train, y_train), (X_test, y_test) = get_data_callbacks()
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    class DummyStatefulMetric(Layer):

        def __init__(self, name='dummy_stateful_metric', **kwargs):
            super(DummyStatefulMetric, self).__init__(name=name, **kwargs)
            self.stateful = True
            self.state = K.variable(value=0, dtype='int32')

        def reset_states(self):
            pass

        def __call__(self, y_true, y_pred):
            return self.state

    inp = Input((input_dim,))
    hidden = Dense(num_hidden, activation='relu')(inp)
    hidden = Dropout(0.1)(hidden)
    hidden = BatchNormalization()(hidden)
    output = Dense(num_classes, activation='softmax')(hidden)
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy', DummyStatefulMetric()])

    # we must generate new callbacks for each test, as they aren't stateless
    def callbacks_factory(histogram_freq, embeddings_freq=1):
        return [callbacks.TensorBoard(log_dir=filepath,
                                      histogram_freq=histogram_freq,
                                      write_images=True, write_grads=True,
                                      embeddings_freq=embeddings_freq,
                                      embeddings_layer_names=['dense_1'],
                                      embeddings_data=X_test,
                                      batch_size=5,
                                      update_freq=update_freq)]

    # fit without validation data
    model.fit(X_train, y_train, batch_size=batch_size,
              callbacks=callbacks_factory(histogram_freq=0, embeddings_freq=0),
              epochs=3)

    # fit with validation data and accuracy
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test),
              callbacks=callbacks_factory(histogram_freq=0), epochs=2)

    # fit generator without validation data
    train_generator = data_generator(X_train, y_train, batch_size)
    model.fit_generator(train_generator, len(X_train), epochs=2,
                        callbacks=callbacks_factory(histogram_freq=0,
                                                    embeddings_freq=0))

    # fit generator with validation data and accuracy
    train_generator = data_generator(X_train, y_train, batch_size)
    model.fit_generator(train_generator, len(X_train), epochs=2,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks_factory(histogram_freq=1))

    assert os.path.isdir(filepath)
    shutil.rmtree(filepath)
    assert not tmpdir.listdir()


@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
def test_TensorBoard_histogram_freq_must_have_validation_data(tmpdir):
    np.random.seed(np.random.randint(1, 1e7))
    filepath = str(tmpdir / 'logs')

    (X_train, y_train), (X_test, y_test) = get_data_callbacks()
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    inp = Input((input_dim,))
    hidden = Dense(num_hidden, activation='relu')(inp)
    hidden = Dropout(0.1)(hidden)
    output = Dense(num_classes, activation='softmax')(hidden)
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # we must generate new callbacks for each test, as they aren't stateless
    def callbacks_factory(histogram_freq, embeddings_freq=1):
        return [callbacks.TensorBoard(log_dir=filepath,
                                      histogram_freq=histogram_freq,
                                      write_images=True, write_grads=True,
                                      embeddings_freq=embeddings_freq,
                                      embeddings_layer_names=['dense_1'],
                                      embeddings_data=X_test,
                                      batch_size=5)]

    # fit without validation data should raise ValueError if histogram_freq > 0
    with pytest.raises(ValueError) as raised_exception:
        model.fit(X_train, y_train, batch_size=batch_size,
                  callbacks=callbacks_factory(histogram_freq=1), epochs=3)
    assert 'validation_data must be provided' in str(raised_exception.value)

    train_generator = data_generator(X_train, y_train, batch_size)
    validation_generator = data_generator(X_test, y_test, batch_size)

    # fit generator without validation data should raise ValueError if
    # histogram_freq > 0
    with pytest.raises(ValueError) as raised_exception:
        model.fit_generator(train_generator,
                            len(X_train), epochs=2,
                            callbacks=callbacks_factory(histogram_freq=1))
    assert 'validation_data must be provided' in str(raised_exception.value)

    # fit generator with validation data generator should raise ValueError if
    # histogram_freq > 0
    with pytest.raises(ValueError) as raised_exception:
        model.fit_generator(train_generator, len(X_train), epochs=2,
                            validation_data=validation_generator,
                            validation_steps=1,
                            callbacks=callbacks_factory(histogram_freq=1))
    assert 'validation_data must be provided' in str(raised_exception.value)


def test_TensorBoard_multi_input_output(tmpdir):
    np.random.seed(np.random.randint(1, 1e7))
    filepath = str(tmpdir / 'logs')

    (X_train, y_train), (X_test, y_test) = get_data_callbacks(
        input_shape=(input_dim, input_dim))

    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    inp1 = Input((input_dim, input_dim))
    inp2 = Input((input_dim, input_dim))
    inp_3d = add([inp1, inp2])
    inp_2d = GlobalAveragePooling1D()(inp_3d)
    # test a layer with a list of output tensors
    inp_pair = Lambda(lambda x: x)([inp_3d, inp_2d])
    hidden = dot(inp_pair, axes=-1)
    hidden = Dense(num_hidden, activation='relu')(hidden)
    hidden = Dropout(0.1)(hidden)
    output1 = Dense(num_classes, activation='softmax')(hidden)
    output2 = Dense(num_classes, activation='softmax')(hidden)
    model = Model(inputs=[inp1, inp2], outputs=[output1, output2])
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # we must generate new callbacks for each test, as they aren't stateless
    def callbacks_factory(histogram_freq, embeddings_freq=1):
        return [callbacks.TensorBoard(log_dir=filepath,
                                      histogram_freq=histogram_freq,
                                      write_images=True, write_grads=True,
                                      embeddings_freq=embeddings_freq,
                                      embeddings_layer_names=['dense_1'],
                                      embeddings_data=[X_test] * 2,
                                      batch_size=5)]

    # fit without validation data
    model.fit([X_train] * 2, [y_train] * 2, batch_size=batch_size,
              callbacks=callbacks_factory(histogram_freq=0, embeddings_freq=0),
              epochs=3)

    # fit with validation data and accuracy
    model.fit([X_train] * 2, [y_train] * 2, batch_size=batch_size,
              validation_data=([X_test] * 2, [y_test] * 2),
              callbacks=callbacks_factory(histogram_freq=1), epochs=2)

    train_generator = data_generator([X_train] * 2, [y_train] * 2, batch_size)

    # fit generator without validation data
    model.fit_generator(train_generator, len(X_train), epochs=2,
                        callbacks=callbacks_factory(histogram_freq=0,
                                                    embeddings_freq=0))

    # fit generator with validation data and accuracy
    model.fit_generator(train_generator, len(X_train), epochs=2,
                        validation_data=([X_test] * 2, [y_test] * 2),
                        callbacks=callbacks_factory(histogram_freq=1))

    assert os.path.isdir(filepath)
    shutil.rmtree(filepath)
    assert not tmpdir.listdir()


def test_TensorBoard_convnet(tmpdir):
    np.random.seed(np.random.randint(1, 1e7))
    filepath = str(tmpdir / 'logs')

    input_shape = (16, 16, 3)
    (x_train, y_train), (x_test, y_test) = get_data_callbacks(
        num_train=500,
        num_test=200,
        input_shape=input_shape)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    model = Sequential([
        Conv2D(filters=8, kernel_size=3,
               activation='relu',
               input_shape=input_shape),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=4, kernel_size=(3, 3),
               activation='relu', padding='same'),
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    tsb = callbacks.TensorBoard(log_dir=filepath, histogram_freq=1,
                                write_images=True, write_grads=True,
                                batch_size=16)
    cbks = [tsb]
    model.summary()
    history = model.fit(x_train, y_train, epochs=2, batch_size=16,
                        validation_data=(x_test, y_test),
                        callbacks=cbks,
                        verbose=0)
    assert os.path.isdir(filepath)
    shutil.rmtree(filepath)
    assert not tmpdir.listdir()


def test_TensorBoard_display_float_from_logs(tmpdir):
    filepath = str(tmpdir / 'logs')

    input_shape = (3,)
    (x_train, y_train), _ = get_data_callbacks(num_train=10,
                                               num_test=0,
                                               input_shape=input_shape)
    y_train = np_utils.to_categorical(y_train)

    model = Sequential([
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop')

    class CustomCallback(callbacks.Callback):

        def on_epoch_end(self, epoch, logs=None):
            logs['test'] = 0.

    tsb = callbacks.TensorBoard(log_dir=filepath,
                                batch_size=16)
    cbks = [CustomCallback(), tsb]
    model.fit(x_train, y_train, epochs=2, batch_size=16,
              callbacks=cbks,
              verbose=0)
    assert os.path.isdir(filepath)
    shutil.rmtree(filepath)
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
