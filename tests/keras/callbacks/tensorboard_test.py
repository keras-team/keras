import os
import numpy as np
import pytest
import shutil

from keras import callbacks
from keras.models import Sequential, Model
from keras import layers
from keras import backend as K
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import unpack_singleton


input_dim = 2
num_hidden = 4
num_classes = 2
batch_size = 5
train_samples = 20
test_samples = 20


if K.backend() != 'tensorflow':
    pytestmark = pytest.mark.skip


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


@pytest.mark.parametrize('update_freq', ['batch', 'epoch', 9])
def test_TensorBoard(tmpdir, update_freq):
    np.random.seed(np.random.randint(1, 1e7))
    filepath = str(tmpdir / 'logs')

    (X_train, y_train), (X_test, y_test) = get_data_callbacks()
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    class DummyStatefulMetric(layers.Layer):

        def __init__(self, name='dummy_stateful_metric', **kwargs):
            super(DummyStatefulMetric, self).__init__(name=name, **kwargs)
            self.stateful = True
            self.state = K.variable(value=0, dtype='int32')

        def reset_states(self):
            pass

        def __call__(self, y_true, y_pred):
            return self.state

    inp = layers.Input((input_dim,))
    hidden = layers.Dense(num_hidden, activation='relu')(inp)
    hidden = layers.Dropout(0.1)(hidden)
    hidden = layers.BatchNormalization()(hidden)
    output = layers.Dense(num_classes, activation='softmax')(hidden)
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy', DummyStatefulMetric()])

    # we must generate new callbacks for each test, as they aren't stateless
    def callbacks_factory(histogram_freq=0,
                          embeddings_freq=0,
                          write_images=False,
                          write_grads=False):
        if embeddings_freq:
            embeddings_layer_names = ['dense_1']
            embeddings_data = X_test
        else:
            embeddings_layer_names = None
            embeddings_data = None
        return [callbacks.TensorBoard(log_dir=filepath,
                                      histogram_freq=histogram_freq,
                                      write_images=write_images,
                                      write_grads=write_grads,
                                      embeddings_freq=embeddings_freq,
                                      embeddings_layer_names=embeddings_layer_names,
                                      embeddings_data=embeddings_data,
                                      update_freq=update_freq)]

    # fit without validation data
    model.fit(X_train, y_train, batch_size=batch_size,
              callbacks=callbacks_factory(),
              epochs=2)

    # fit with validation data and accuracy
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test),
              callbacks=callbacks_factory(),
              epochs=2)

    # fit generator without validation data
    train_generator = data_generator(X_train, y_train, batch_size)
    model.fit_generator(train_generator, len(X_train), epochs=2,
                        callbacks=callbacks_factory())

    # fit generator with validation data and accuracy
    train_generator = data_generator(X_train, y_train, batch_size)
    model.fit_generator(train_generator, len(X_train), epochs=2,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks_factory(histogram_freq=1))

    assert os.path.isdir(filepath)
    shutil.rmtree(filepath)
    assert not tmpdir.listdir()


def test_TensorBoard_multi_input_output(tmpdir):
    np.random.seed(np.random.randint(1, 1e7))
    filepath = str(tmpdir / 'logs')

    (X_train, y_train), (X_test, y_test) = get_data_callbacks(
        input_shape=(input_dim, input_dim))

    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    inp1 = layers.Input((input_dim, input_dim))
    inp2 = layers.Input((input_dim, input_dim))
    inp_3d = layers.add([inp1, inp2])
    inp_2d = layers.GlobalAveragePooling1D()(inp_3d)
    # test a layer with a list of output tensors
    inp_pair = layers.Lambda(lambda x: x)([inp_3d, inp_2d])
    hidden = layers.dot(inp_pair, axes=-1)
    hidden = layers.Dense(num_hidden, activation='relu')(hidden)
    hidden = layers.Dropout(0.1)(hidden)
    output1 = layers.Dense(num_classes, activation='softmax')(hidden)
    output2 = layers.Dense(num_classes, activation='softmax')(hidden)
    model = Model(inputs=[inp1, inp2], outputs=[output1, output2])
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # we must generate new callbacks for each test, as they aren't stateless
    def callbacks_factory(histogram_freq=0,
                          embeddings_freq=0,
                          write_images=False,
                          write_grads=False):
        if embeddings_freq:
            embeddings_layer_names = ['dense_1']
            embeddings_data = [X_test] * 2
        else:
            embeddings_layer_names = None
            embeddings_data = None
        return [callbacks.TensorBoard(log_dir=filepath,
                                      histogram_freq=histogram_freq,
                                      write_images=write_images,
                                      write_grads=write_grads,
                                      embeddings_freq=embeddings_freq,
                                      embeddings_layer_names=embeddings_layer_names,
                                      embeddings_data=embeddings_data)]

    # fit without validation data
    model.fit([X_train] * 2, [y_train] * 2, batch_size=batch_size,
              callbacks=callbacks_factory(),
              epochs=3)

    # fit with validation data and accuracy
    model.fit([X_train] * 2, [y_train] * 2, batch_size=batch_size,
              validation_data=([X_test] * 2, [y_test] * 2),
              callbacks=callbacks_factory(histogram_freq=1),
              epochs=2)

    train_generator = data_generator([X_train] * 2, [y_train] * 2, batch_size)

    # fit generator without validation data
    model.fit_generator(train_generator, len(X_train), epochs=2,
                        callbacks=callbacks_factory())

    # fit generator with validation data and accuracy
    model.fit_generator(train_generator, len(X_train), epochs=2,
                        validation_data=([X_test] * 2, [y_test] * 2),
                        callbacks=callbacks_factory())

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
        layers.Conv2D(filters=8, kernel_size=3,
                      activation='relu',
                      input_shape=input_shape),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filters=4, kernel_size=(3, 3),
                      activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    tsb = callbacks.TensorBoard(filepath, histogram_freq=1)
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
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop')

    class CustomCallback(callbacks.Callback):

        def on_epoch_end(self, epoch, logs=None):
            logs['test'] = 0.

    tsb = callbacks.TensorBoard(log_dir=filepath)
    cbks = [CustomCallback(), tsb]
    model.fit(x_train, y_train, epochs=2, batch_size=16,
              callbacks=cbks,
              verbose=0)
    assert os.path.isdir(filepath)
    shutil.rmtree(filepath)
    assert not tmpdir.listdir()
