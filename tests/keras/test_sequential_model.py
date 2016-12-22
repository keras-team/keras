from __future__ import absolute_import
from __future__ import print_function
import pytest
import os
import numpy as np
np.random.seed(1337)

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Lambda
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data, keras_test
from keras.models import model_from_json, model_from_yaml
from keras import objectives
from keras.engine.training import make_batches


input_dim = 16
nb_hidden = 8
nb_class = 4
batch_size = 32
nb_epoch = 1


@keras_test
def test_sequential_pop():
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim))
    model.add(Dense(nb_class))
    model.compile(loss='mse', optimizer='sgd')
    x = np.random.random((batch_size, input_dim))
    y = np.random.random((batch_size, nb_class))
    model.fit(x, y, nb_epoch=1)
    model.pop()
    assert len(model.layers) == 1
    assert model.output_shape == (None, nb_hidden)
    model.compile(loss='mse', optimizer='sgd')
    y = np.random.random((batch_size, nb_hidden))
    model.fit(x, y, nb_epoch=1)


def _get_test_data():
    np.random.seed(1234)

    train_samples = 100
    test_samples = 50

    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=train_samples,
                                                         nb_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         nb_class=4)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    return (X_train, y_train), (X_test, y_test)


@keras_test
def test_sequential_fit_generator():
    (X_train, y_train), (X_test, y_test) = _get_test_data()

    def data_generator(train):
        if train:
            max_batch_index = len(X_train) // batch_size
        else:
            max_batch_index = len(X_test) // batch_size
        i = 0
        while 1:
            if train:
                yield (X_train[i * batch_size: (i + 1) * batch_size], y_train[i * batch_size: (i + 1) * batch_size])
            else:
                yield (X_test[i * batch_size: (i + 1) * batch_size], y_test[i * batch_size: (i + 1) * batch_size])
            i += 1
            i = i % max_batch_index

    model = Sequential()
    model.add(Dense(nb_hidden, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_class))
    model.pop()
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit_generator(data_generator(True), len(X_train), nb_epoch)
    model.fit_generator(data_generator(True), len(X_train), nb_epoch, validation_data=(X_test, y_test))
    model.fit_generator(data_generator(True), len(X_train), nb_epoch,
                        validation_data=data_generator(False), nb_val_samples=batch_size * 3)
    model.fit_generator(data_generator(True), len(X_train), nb_epoch, max_q_size=2)
    model.evaluate(X_train, y_train)


@keras_test
def test_sequential():
    (X_train, y_train), (X_test, y_test) = _get_test_data()

    # TODO: factor out
    def data_generator(x, y, batch_size=50):
        index_array = np.arange(len(x))
        while 1:
            batches = make_batches(len(X_test), batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                x_batch = x[batch_ids]
                y_batch = y[batch_ids]
                yield (x_batch, y_batch)

    model = Sequential()
    model.add(Dense(nb_hidden, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, y_test))
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_split=0.1)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=False)

    model.train_on_batch(X_train[:32], y_train[:32])

    loss = model.evaluate(X_test, y_test)

    prediction = model.predict_generator(data_generator(X_test, y_test), X_test.shape[0], max_q_size=2)
    gen_loss = model.evaluate_generator(data_generator(X_test, y_test, 50), X_test.shape[0], max_q_size=2)
    pred_loss = K.eval(K.mean(objectives.get(model.loss)(K.variable(y_test), K.variable(prediction))))

    assert(np.isclose(pred_loss, loss))
    assert(np.isclose(gen_loss, loss))

    model.predict(X_test, verbose=0)
    model.predict_classes(X_test, verbose=0)
    model.predict_proba(X_test, verbose=0)

    fname = 'test_sequential_temp.h5'
    model.save_weights(fname, overwrite=True)
    model = Sequential()
    model.add(Dense(nb_hidden, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.load_weights(fname)
    os.remove(fname)

    nloss = model.evaluate(X_test, y_test, verbose=0)
    assert(loss == nloss)

    # test serialization
    config = model.get_config()
    Sequential.from_config(config)

    model.summary()
    json_str = model.to_json()
    model_from_json(json_str)

    yaml_str = model.to_yaml()
    model_from_yaml(yaml_str)


@keras_test
def test_nested_sequential():
    (X_train, y_train), (X_test, y_test) = _get_test_data()

    inner = Sequential()
    inner.add(Dense(nb_hidden, input_shape=(input_dim,)))
    inner.add(Activation('relu'))
    inner.add(Dense(nb_class))

    middle = Sequential()
    middle.add(inner)

    model = Sequential()
    model.add(middle)
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, y_test))
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_split=0.1)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=False)

    model.train_on_batch(X_train[:32], y_train[:32])

    loss = model.evaluate(X_test, y_test, verbose=0)

    model.predict(X_test, verbose=0)
    model.predict_classes(X_test, verbose=0)
    model.predict_proba(X_test, verbose=0)

    fname = 'test_nested_sequential_temp.h5'
    model.save_weights(fname, overwrite=True)

    inner = Sequential()
    inner.add(Dense(nb_hidden, input_shape=(input_dim,)))
    inner.add(Activation('relu'))
    inner.add(Dense(nb_class))

    middle = Sequential()
    middle.add(inner)

    model = Sequential()
    model.add(middle)
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.load_weights(fname)
    os.remove(fname)

    nloss = model.evaluate(X_test, y_test, verbose=0)
    assert(loss == nloss)

    # test serialization
    config = model.get_config()
    Sequential.from_config(config)

    model.summary()
    json_str = model.to_json()
    model_from_json(json_str)

    yaml_str = model.to_yaml()
    model_from_yaml(yaml_str)


@keras_test
def test_merge_sum():
    (X_train, y_train), (X_test, y_test) = _get_test_data()
    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(nb_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left, right], mode='sum'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, validation_data=([X_test, X_test], y_test))
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, validation_split=0.1)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

    loss = model.evaluate([X_test, X_test], y_test, verbose=0)

    model.predict([X_test, X_test], verbose=0)
    model.predict_classes([X_test, X_test], verbose=0)
    model.predict_proba([X_test, X_test], verbose=0)

    # test weight saving
    fname = 'test_merge_sum_temp.h5'
    model.save_weights(fname, overwrite=True)
    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))
    right = Sequential()
    right.add(Dense(nb_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))
    model = Sequential()
    model.add(Merge([left, right], mode='sum'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.load_weights(fname)
    os.remove(fname)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    nloss = model.evaluate([X_test, X_test], y_test, verbose=0)
    assert(loss == nloss)

    # test serialization
    config = model.get_config()
    Sequential.from_config(config)

    model.summary()
    json_str = model.to_json()
    model_from_json(json_str)

    yaml_str = model.to_yaml()
    model_from_yaml(yaml_str)


@keras_test
def test_merge_dot():
    (X_train, y_train), (X_test, y_test) = _get_test_data()

    left = Sequential()
    left.add(Dense(input_dim=input_dim, output_dim=nb_hidden))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(input_dim=input_dim, output_dim=nb_hidden))
    right.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left, right], mode='dot', dot_axes=1))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    left = Sequential()
    left.add(Dense(input_dim=input_dim, output_dim=nb_hidden))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(input_dim=input_dim, output_dim=nb_hidden))
    right.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left, right], mode='dot', dot_axes=[1, 1]))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


@keras_test
def test_merge_concat():
    (X_train, y_train), (X_test, y_test) = _get_test_data()

    left = Sequential(name='branch_1')
    left.add(Dense(nb_hidden, input_shape=(input_dim,), name='dense_1'))
    left.add(Activation('relu', name='relu_1'))

    right = Sequential(name='branch_2')
    right.add(Dense(nb_hidden, input_shape=(input_dim,), name='dense_2'))
    right.add(Activation('relu', name='relu_2'))

    model = Sequential(name='merged_branches')
    model.add(Merge([left, right], mode='concat', name='merge'))
    model.add(Dense(nb_class, name='final_dense'))
    model.add(Activation('softmax', name='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, validation_data=([X_test, X_test], y_test))
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, validation_split=0.1)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

    loss = model.evaluate([X_test, X_test], y_test, verbose=0)

    model.predict([X_test, X_test], verbose=0)
    model.predict_classes([X_test, X_test], verbose=0)
    model.predict_proba([X_test, X_test], verbose=0)
    model.get_config()

    fname = 'test_merge_concat_temp.h5'
    model.save_weights(fname, overwrite=True)
    model.fit([X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.load_weights(fname)
    os.remove(fname)

    nloss = model.evaluate([X_test, X_test], y_test, verbose=0)
    assert(loss == nloss)


@keras_test
def test_merge_recursivity():
    (X_train, y_train), (X_test, y_test) = _get_test_data()
    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    right = Sequential()
    right.add(Dense(nb_hidden, input_shape=(input_dim,)))
    right.add(Activation('relu'))

    righter = Sequential()
    righter.add(Dense(nb_hidden, input_shape=(input_dim,)))
    righter.add(Activation('relu'))

    intermediate = Sequential()
    intermediate.add(Merge([left, right], mode='sum'))
    intermediate.add(Dense(nb_hidden))
    intermediate.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([intermediate, righter], mode='sum'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, validation_data=([X_test, X_test, X_test], y_test))
    model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, validation_split=0.1)
    model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.fit([X_train, X_train, X_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

    loss = model.evaluate([X_test, X_test, X_test], y_test, verbose=0)

    model.predict([X_test, X_test, X_test], verbose=0)
    model.predict_classes([X_test, X_test, X_test], verbose=0)
    model.predict_proba([X_test, X_test, X_test], verbose=0)

    fname = 'test_merge_recursivity_temp.h5'
    model.save_weights(fname, overwrite=True)
    model.load_weights(fname)
    os.remove(fname)

    nloss = model.evaluate([X_test, X_test, X_test], y_test, verbose=0)
    assert(loss == nloss)

    # test serialization
    config = model.get_config()
    Sequential.from_config(config)

    model.summary()
    json_str = model.to_json()
    model_from_json(json_str)

    yaml_str = model.to_yaml()
    model_from_yaml(yaml_str)


@keras_test
def test_merge_overlap():
    (X_train, y_train), (X_test, y_test) = _get_test_data()
    left = Sequential()
    left.add(Dense(nb_hidden, input_shape=(input_dim,)))
    left.add(Activation('relu'))

    model = Sequential()
    model.add(Merge([left, left], mode='sum'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, y_test))
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_split=0.1)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=False)

    model.train_on_batch(X_train[:32], y_train[:32])

    loss = model.evaluate(X_test, y_test, verbose=0)
    model.predict(X_test, verbose=0)
    model.predict_classes(X_test, verbose=0)
    model.predict_proba(X_test, verbose=0)

    fname = 'test_merge_overlap_temp.h5'
    print(model.layers)
    model.save_weights(fname, overwrite=True)
    print(model.trainable_weights)

    model.load_weights(fname)
    os.remove(fname)

    nloss = model.evaluate(X_test, y_test, verbose=0)
    assert(loss == nloss)

    # test serialization
    config = model.get_config()
    Sequential.from_config(config)

    model.summary()
    json_str = model.to_json()
    model_from_json(json_str)

    yaml_str = model.to_yaml()
    model_from_yaml(yaml_str)


@keras_test
def test_sequential_count_params():
    input_dim = 20
    nb_units = 10
    nb_classes = 2

    n = input_dim * nb_units + nb_units
    n += nb_units * nb_units + nb_units
    n += nb_units * nb_classes + nb_classes

    model = Sequential()
    model.add(Dense(nb_units, input_shape=(input_dim,)))
    model.add(Dense(nb_units))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.build()

    assert(n == model.count_params())

    model.compile('sgd', 'binary_crossentropy')
    assert(n == model.count_params())


if __name__ == '__main__':
    pytest.main([__file__])
