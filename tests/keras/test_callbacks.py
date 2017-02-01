import os
import sys
import multiprocessing

import numpy as np
import pytest
from csv import Sniffer
from keras import optimizers

np.random.seed(1337)

from keras import callbacks
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils.test_utils import get_test_data
from keras import backend as K
from keras.utils import np_utils

input_dim = 2
nb_hidden = 4
nb_class = 2
batch_size = 5
train_samples = 20
test_samples = 20


def test_ModelCheckpoint():
    filepath = 'checkpoint.h5'
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=train_samples,
                                                         nb_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         nb_class=nb_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    # case 1
    monitor = 'val_loss'
    save_best_only = False
    mode = 'auto'

    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    cbks = [callbacks.ModelCheckpoint(filepath, monitor=monitor,
                                      save_best_only=save_best_only, mode=mode)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=1)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # case 2
    mode = 'min'
    cbks = [callbacks.ModelCheckpoint(filepath, monitor=monitor,
                                      save_best_only=save_best_only, mode=mode)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=1)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # case 3
    mode = 'max'
    monitor = 'val_acc'
    cbks = [callbacks.ModelCheckpoint(filepath, monitor=monitor,
                                      save_best_only=save_best_only, mode=mode)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=1)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # case 4
    save_best_only = True
    cbks = [callbacks.ModelCheckpoint(filepath, monitor=monitor,
                                      save_best_only=save_best_only, mode=mode)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=1)
    assert os.path.exists(filepath)
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
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=4)
    assert os.path.exists(filepath.format(epoch=1))
    assert os.path.exists(filepath.format(epoch=3))
    assert not os.path.exists(filepath.format(epoch=0))
    assert not os.path.exists(filepath.format(epoch=2))
    os.remove(filepath.format(epoch=1))
    os.remove(filepath.format(epoch=3))


def test_EarlyStopping():
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=train_samples,
                                                         nb_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         nb_class=nb_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    mode = 'max'
    monitor = 'val_acc'
    patience = 0
    cbks = [callbacks.EarlyStopping(patience=patience, monitor=monitor, mode=mode)]
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=20)

    mode = 'auto'
    monitor = 'val_acc'
    patience = 2
    cbks = [callbacks.EarlyStopping(patience=patience, monitor=monitor, mode=mode)]
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=20)


def test_EarlyStopping_reuse():
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

    hist = model.fit(data, labels, callbacks=[stopper])
    assert len(hist.epoch) >= patience

    # This should allow training to go for at least `patience` epochs
    model.set_weights(weights)
    hist = model.fit(data, labels, callbacks=[stopper])
    assert len(hist.epoch) >= patience


def test_LearningRateScheduler():
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=train_samples,
                                                         nb_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         nb_class=nb_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    cbks = [callbacks.LearningRateScheduler(lambda x: 1. / (1. + x))]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=5)
    assert (float(K.get_value(model.optimizer.lr)) - 0.2) < K.epsilon()


def test_ReduceLROnPlateau():
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=train_samples,
                                                         nb_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         nb_class=nb_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    def make_model():
        np.random.seed(1337)
        model = Sequential()
        model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
        model.add(Dense(nb_class, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=0.1),
                      metrics=['accuracy'])
        return model

    model = make_model()

    # This should reduce the LR after the first epoch (due to high epsilon).
    cbks = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=10, patience=1, cooldown=5)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=5, verbose=2)
    assert np.allclose(float(K.get_value(model.optimizer.lr)), 0.01, atol=K.epsilon())

    model = make_model()
    cbks = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=0, patience=1, cooldown=5)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=5, verbose=2)
    assert np.allclose(float(K.get_value(model.optimizer.lr)), 0.1, atol=K.epsilon())


def test_CSVLogger():
    filepath = 'log.tsv'
    sep = '\t'
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=train_samples,
                                                         nb_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         nb_class=nb_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    def make_model():
        np.random.seed(1337)
        model = Sequential()
        model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
        model.add(Dense(nb_class, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=0.1),
                      metrics=['accuracy'])
        return model

    # case 1, create new file with defined separator
    model = make_model()
    cbks = [callbacks.CSVLogger(filepath, separator=sep)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=1)

    assert os.path.exists(filepath)
    with open(filepath) as csvfile:
        dialect = Sniffer().sniff(csvfile.read())
    assert dialect.delimiter == sep
    del model
    del cbks

    # case 2, append data to existing file, skip header
    model = make_model()
    cbks = [callbacks.CSVLogger(filepath, separator=sep, append=True)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=1)

    # case 3, reuse of CSVLogger object
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=1)

    import re
    with open(filepath) as csvfile:
        output = " ".join(csvfile.readlines())
        assert len(re.findall('epoch', output)) == 1

    os.remove(filepath)


@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason="Requires tensorflow backend")
def test_TensorBoard():
    import shutil

    filepath = './logs'
    (X_train, y_train), (X_test, y_test) = get_test_data(
        nb_train=train_samples,
        nb_test=test_samples,
        input_shape=(input_dim,),
        classification=True,
        nb_class=nb_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    def data_generator(train):
        if train:
            max_batch_index = len(X_train) // batch_size
        else:
            max_batch_index = len(X_test) // batch_size
        i = 0
        while 1:
            if train:
                yield (X_train[i * batch_size: (i + 1) * batch_size],
                       y_train[i * batch_size: (i + 1) * batch_size])
            else:
                yield (X_test[i * batch_size: (i + 1) * batch_size],
                       y_test[i * batch_size: (i + 1) * batch_size])
            i += 1
            i = i % max_batch_index

    def data_generator_graph(train):
        while 1:
            if train:
                yield {'X_vars': X_train, 'output': y_train}
            else:
                yield {'X_vars': X_test, 'output': y_test}

    # case 1 Sequential
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    tsb = callbacks.TensorBoard(log_dir=filepath, histogram_freq=1)
    cbks = [tsb]

    # fit with validation data
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=3)

    # fit with validation data and accuracy
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=2)

    # fit generator with validation data
    model.fit_generator(data_generator(True), len(X_train), nb_epoch=2,
                        validation_data=(X_test, y_test),
                        callbacks=cbks)

    # fit generator without validation data
    model.fit_generator(data_generator(True), len(X_train), nb_epoch=2,
                        callbacks=cbks)

    # fit generator with validation data and accuracy
    model.fit_generator(data_generator(True), len(X_train), nb_epoch=2,
                        validation_data=(X_test, y_test),
                        callbacks=cbks)

    # fit generator without validation data and accuracy
    model.fit_generator(data_generator(True), len(X_train), nb_epoch=2,
                        callbacks=cbks)

    assert os.path.exists(filepath)
    shutil.rmtree(filepath)


def test_LambdaCallback():
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=train_samples,
                                                         nb_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         nb_class=nb_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # Start an arbitrary process that should run during model training and be terminated after training has completed.
    def f():
        while True:
            pass

    p = multiprocessing.Process(target=f)
    p.start()
    cleanup_callback = callbacks.LambdaCallback(on_train_end=lambda logs: p.terminate())

    cbks = [cleanup_callback]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=5)
    p.join()
    assert not p.is_alive()


@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason="Requires tensorflow backend")
def test_TensorBoard_with_ReduceLROnPlateau():
    import shutil
    filepath = './logs'
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=train_samples,
                                                         nb_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         nb_class=nb_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
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
              validation_data=(X_test, y_test), callbacks=cbks, nb_epoch=2)

    assert os.path.exists(filepath)
    shutil.rmtree(filepath)


if __name__ == '__main__':
    pytest.main([__file__])
