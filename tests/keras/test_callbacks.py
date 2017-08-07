import os
import multiprocessing

import numpy as np
import pytest
from csv import reader
from csv import Sniffer
import shutil
from keras import optimizers
from keras import initializers
from keras import callbacks
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, add
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.utils.test_utils import get_test_data
from keras.utils.test_utils import keras_test
from keras import backend as K
from keras.utils import np_utils

input_dim = 2
num_hidden = 4
num_class = 2
batch_size = 5
train_samples = 20
test_samples = 20


@keras_test
def test_TerminateOnNaN():
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=num_class)

    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    cbks = [callbacks.TerminateOnNaN()]
    model = Sequential()
    initializer = initializers.Constant(value=1e5)
    for _ in range(5):
        model.add(Dense(num_hidden, input_dim=input_dim, activation='relu',
                        kernel_initializer=initializer))
    model.add(Dense(num_class, activation='linear'))
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop')

    # case 1 fit
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        validation_data=(X_test, y_test), callbacks=cbks, epochs=20)
    loss = history.history['loss']
    assert len(loss) == 1
    assert loss[0] == np.inf

    # case 2 fit_generator
    def data_generator():
        max_batch_index = len(X_train) // batch_size
        i = 0
        while 1:
            yield (X_train[i * batch_size: (i + 1) * batch_size],
                   y_train[i * batch_size: (i + 1) * batch_size])
            i += 1
            i = i % max_batch_index
    history = model.fit_generator(data_generator(),
                                  len(X_train),
                                  validation_data=(X_test, y_test),
                                  callbacks=cbks,
                                  epochs=20)
    loss = history.history['loss']
    assert len(loss) == 1
    assert loss[0] == np.inf or np.isnan(loss[0])


@keras_test
def test_stop_training_csv(tmpdir):
    np.random.seed(1337)
    fp = str(tmpdir / 'test.csv')
    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=num_class)

    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    cbks = [callbacks.TerminateOnNaN(), callbacks.CSVLogger(fp)]
    model = Sequential()
    for _ in range(5):
        model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_class, activation='linear'))
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop')

    def data_generator():
        i = 0
        max_batch_index = len(X_train) // batch_size
        tot = 0
        while 1:
            if tot > 3 * len(X_train):
                yield np.ones([batch_size, input_dim]) * np.nan, np.ones([batch_size, num_class]) * np.nan
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


@keras_test
def test_ModelCheckpoint(tmpdir):
    np.random.seed(1337)
    filepath = str(tmpdir / 'checkpoint.h5')
    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=num_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    # case 1
    monitor = 'val_loss'
    save_best_only = False
    mode = 'auto'

    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))
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
    assert os.path.isfile(filepath.format(epoch=1))
    assert os.path.isfile(filepath.format(epoch=3))
    assert not os.path.exists(filepath.format(epoch=0))
    assert not os.path.exists(filepath.format(epoch=2))
    os.remove(filepath.format(epoch=1))
    os.remove(filepath.format(epoch=3))
    assert not tmpdir.listdir()


@keras_test
def test_EarlyStopping():
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=num_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))
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


@keras_test
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

    hist = model.fit(data, labels, callbacks=[stopper])
    assert len(hist.epoch) >= patience

    # This should allow training to go for at least `patience` epochs
    model.set_weights(weights)
    hist = model.fit(data, labels, callbacks=[stopper])
    assert len(hist.epoch) >= patience


@keras_test
def test_LearningRateScheduler():
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=num_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    cbks = [callbacks.LearningRateScheduler(lambda x: 1. / (1. + x))]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=5)
    assert (float(K.get_value(model.optimizer.lr)) - 0.2) < K.epsilon()


@keras_test
def test_ReduceLROnPlateau():
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=num_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    def make_model():
        np.random.seed(1337)
        model = Sequential()
        model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
        model.add(Dense(num_class, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=0.1),
                      metrics=['accuracy'])
        return model

    model = make_model()

    # This should reduce the LR after the first epoch (due to high epsilon).
    cbks = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=10, patience=1, cooldown=5)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=5, verbose=2)
    assert np.allclose(float(K.get_value(model.optimizer.lr)), 0.01, atol=K.epsilon())

    model = make_model()
    cbks = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=0, patience=1, cooldown=5)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=5, verbose=2)
    assert np.allclose(float(K.get_value(model.optimizer.lr)), 0.1, atol=K.epsilon())


@keras_test
def test_CSVLogger(tmpdir):
    np.random.seed(1337)
    filepath = str(tmpdir / 'log.tsv')
    sep = '\t'
    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=num_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    def make_model():
        np.random.seed(1337)
        model = Sequential()
        model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
        model.add(Dense(num_class, activation='softmax'))

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
              validation_data=(X_test, y_test), callbacks=cbks, epochs=1)

    import re
    with open(filepath) as csvfile:
        output = " ".join(csvfile.readlines())
        assert len(re.findall('epoch', output)) == 1

    os.remove(filepath)
    assert not tmpdir.listdir()


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='Requires tensorflow backend')
def test_TensorBoard(tmpdir):
    np.random.seed(np.random.randint(1, 1e7))
    filepath = str(tmpdir / 'logs')

    (X_train, y_train), (X_test, y_test) = get_test_data(
        num_train=train_samples,
        num_test=test_samples,
        input_shape=(input_dim,),
        classification=True,
        num_classes=num_class)
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
                # simulate multi-input/output models
                yield (X_train[i * batch_size: (i + 1) * batch_size],
                       y_train[i * batch_size: (i + 1) * batch_size])
            else:
                yield (X_test[i * batch_size: (i + 1) * batch_size],
                       y_test[i * batch_size: (i + 1) * batch_size])
            i += 1
            i = i % max_batch_index

    inp = Input((input_dim,))
    hidden = Dense(num_hidden, activation='relu')(inp)
    hidden = Dropout(0.1)(hidden)
    output = Dense(num_class, activation='softmax')(hidden)
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    tsb = callbacks.TensorBoard(log_dir=filepath, histogram_freq=1,
                                write_images=True, write_grads=True,
                                embeddings_freq=1,
                                embeddings_layer_names=['dense_1'],
                                batch_size=5)
    cbks = [tsb]

    # fit without validation data
    model.fit(X_train, y_train, batch_size=batch_size,
              callbacks=cbks, epochs=3)

    # fit with validation data and accuracy
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test),
              callbacks=cbks, epochs=2)

    # fit generator without validation data
    model.fit_generator(data_generator(True), len(X_train), epochs=2,
                        callbacks=cbks)

    # fit generator with validation data and accuracy
    model.fit_generator(data_generator(True), len(X_train), epochs=2,
                        validation_data=(X_test, y_test),
                        callbacks=cbks)

    assert os.path.isdir(filepath)
    shutil.rmtree(filepath)
    assert not tmpdir.listdir()


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='Requires tensorflow backend')
def test_TensorBoard_multi_input_output(tmpdir):
    np.random.seed(np.random.randint(1, 1e7))
    filepath = str(tmpdir / 'logs')

    (X_train, y_train), (X_test, y_test) = get_test_data(
        num_train=train_samples,
        num_test=test_samples,
        input_shape=(input_dim,),
        classification=True,
        num_classes=num_class)
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
                # simulate multi-input/output models
                yield ([X_train[i * batch_size: (i + 1) * batch_size]] * 2,
                       [y_train[i * batch_size: (i + 1) * batch_size]] * 2)
            else:
                yield ([X_test[i * batch_size: (i + 1) * batch_size]] * 2,
                       [y_test[i * batch_size: (i + 1) * batch_size]] * 2)
            i += 1
            i = i % max_batch_index

    inp1 = Input((input_dim,))
    inp2 = Input((input_dim,))
    inp = add([inp1, inp2])
    hidden = Dense(num_hidden, activation='relu')(inp)
    hidden = Dropout(0.1)(hidden)
    output1 = Dense(num_class, activation='softmax')(hidden)
    output2 = Dense(num_class, activation='softmax')(hidden)
    model = Model(inputs=[inp1, inp2], outputs=[output1, output2])
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    tsb = callbacks.TensorBoard(log_dir=filepath, histogram_freq=1,
                                write_images=True, write_grads=True,
                                embeddings_freq=1,
                                embeddings_layer_names=['dense_1'],
                                batch_size=5)
    cbks = [tsb]

    # fit without validation data
    model.fit([X_train] * 2, [y_train] * 2, batch_size=batch_size,
              callbacks=cbks, epochs=3)

    # fit with validation data and accuracy
    model.fit([X_train] * 2, [y_train] * 2, batch_size=batch_size,
              validation_data=([X_test] * 2, [y_test] * 2),
              callbacks=cbks, epochs=2)

    # fit generator without validation data
    model.fit_generator(data_generator(True), len(X_train), epochs=2,
                        callbacks=cbks)

    # fit generator with validation data and accuracy
    model.fit_generator(data_generator(True), len(X_train), epochs=2,
                        validation_data=([X_test] * 2, [y_test] * 2),
                        callbacks=cbks)

    assert os.path.isdir(filepath)
    shutil.rmtree(filepath)
    assert not tmpdir.listdir()


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='Requires tensorflow backend')
def test_TensorBoard_convnet(tmpdir):
    np.random.seed(np.random.randint(1, 1e7))
    filepath = str(tmpdir / 'logs')

    input_shape = (16, 16, 3)
    (x_train, y_train), (x_test, y_test) = get_test_data(num_train=500,
                                                         num_test=200,
                                                         input_shape=input_shape,
                                                         classification=True,
                                                         num_classes=4)
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
        Dense(y_test.shape[-1], activation='softmax')
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


@keras_test
def test_CallbackValData():
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=num_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    cbk = callbacks.LambdaCallback(on_train_end=lambda x: 1)
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=[cbk], epochs=1)

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

    cbk2 = callbacks.LambdaCallback(on_train_end=lambda x: 1)
    model.fit_generator(data_generator(True), len(X_train), epochs=1,
                        validation_data=(X_test, y_test),
                        callbacks=[cbk2])

    # callback validation data should always have x, y, and sample weights
    assert len(cbk.validation_data) == len(cbk2.validation_data) == 3
    assert cbk.validation_data[0] is cbk2.validation_data[0]
    assert cbk.validation_data[1] is cbk2.validation_data[1]
    assert cbk.validation_data[2].shape == cbk2.validation_data[2].shape


@keras_test
def test_LambdaCallback():
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=num_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))
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
              validation_data=(X_test, y_test), callbacks=cbks, epochs=5)
    p.join()
    assert not p.is_alive()


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason="Requires tensorflow backend")
def test_TensorBoard_with_ReduceLROnPlateau(tmpdir):
    import shutil
    np.random.seed(np.random.randint(1, 1e7))
    filepath = str(tmpdir / 'logs')

    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=num_class)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))
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


if __name__ == '__main__':
    pytest.main([__file__])
