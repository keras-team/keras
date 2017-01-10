from __future__ import print_function
import numpy as np
from numpy.testing import assert_allclose
np.random.seed(1337)
import pytest
from keras.utils.test_utils import keras_test
from keras.layers import ChainCRF
from keras.models import Sequential
from keras.optimizers import SGD
from keras import backend as K


@keras_test
def test_tag_sequence():
    # Generate data

    n_samples, n_steps, n_classes = 1000, 16, 3
    U_true = get_test_transition_matrix(n_classes)
    (X_train, y_train), (X_test, y_test) = get_test_sequences(n_samples,
                                                              n_steps,
                                                              U_true)
    model = Sequential()
    crf = ChainCRF(input_shape=(n_steps, n_classes))
    model.add(crf)

    sgd = SGD(lr=0.2, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss=crf.loss, optimizer=sgd, metrics=['accuracy'])
    history = model.fit(X_train, y_train, nb_epoch=1, batch_size=32,
                        validation_data=(X_test, y_test))

    assert(history.history['val_acc'][-1] >= 0.94)


@keras_test
def test_sparse_tag_sequence():
    # Generate data

    n_samples, n_steps, n_classes = 1000, 16, 3
    U_true = get_test_transition_matrix(n_classes)
    (X_train, y_train), (X_test, y_test) = get_test_sequences(n_samples,
                                                              n_steps,
                                                              U_true,
                                                              sparse_tags=True)
    model = Sequential()
    crf = ChainCRF(input_shape=(n_steps, n_classes))
    model.add(crf)

    sgd = SGD(lr=0.2, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss=crf.sparse_loss, optimizer=sgd, metrics=['sparse_categorical_accuracy'])
    history = model.fit(X_train, y_train, nb_epoch=1, batch_size=32,
                        validation_data=(X_test, y_test))

    assert(history.history['val_sparse_categorical_accuracy'][-1] >= 0.94)


@keras_test
def test_generate_transition_matrix():
    # Generate data

    n_samples, n_steps, n_classes = 20000, 16, 3
    U_true = get_test_transition_matrix(n_classes)
    (X_train, y_train), (X_test, y_test) = get_test_sequences(n_samples=n_samples,
                                                              n_steps=n_steps,
                                                              U=U_true)
    model = Sequential()
    crf = ChainCRF(input_shape=X_train[0].shape)
    model.add(crf)

    sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss=crf.loss, optimizer=sgd, metrics=['accuracy'])

    model.fit(X_train, y_train, nb_epoch=1, batch_size=32,
              validation_data=(X_test, y_test))

    print('Example predictions:')
    y_pred = model.predict_classes(X_test)
    for i in range(10):
        print(i)
        print('y_true', np.argmax(y_test[i], axis=1))
        print('y_pred', y_pred[i])
    U_pred = K.get_value(crf.U)
    print('U:\n', U_pred)
    print('b_start:\n', K.get_value(crf.b_start))
    print('b_end:\n', K.get_value(crf.b_end))

    U_pred = np.exp(U_pred)
    U_pred /= np.sum(U_pred, axis=1, keepdims=True)
    print('transitions_true:\n', U_true)
    print('transitions_pred:\n', U_pred)
    assert_allclose(U_pred, U_true, atol=5e-2)


def get_test_transition_matrix(n_classes=2):
    perm = np.roll(np.arange(n_classes), -1)
    return np.eye(n_classes, dtype=np.float32)[:, np.argsort(perm)]


def get_test_sequences(n_samples, n_steps, U, test_split=0.1, verbose=1, sparse_tags=False):
    n_classes = U.shape[0]
    X = np.random.uniform(size=(n_samples, n_steps, n_classes)).astype(np.float32)
    X[:, 0] *= 100

    y = np.zeros((n_samples, n_steps), dtype=np.int32)
    for i in range(n_samples):
        y[i] = _sample_tag_sequence(X[i], U)

    if sparse_tags:
        y = np.expand_dims(y, -1)
    else:
        y = np.eye(n_classes)[y]

    n_train = int(1.0 - n_samples * test_split)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    if verbose:
        print('X_train.shape:', X_train.shape)
        print('y_train.shape:', y_train.shape)
        print('X_test.shape:', X_test.shape)
        print('y_test.shape:', y_test.shape)

    if verbose:
        if sparse_tags:
            print('Example (first 10 test tag sequences):')
            print(y_test[:10])
        else:
            print('Example (first 10 test elements with argmax(axis=2)):')
            print(np.argmax(y_test[:10], axis=2))

    return (X_train, y_train), (X_test, y_test)


def _sample_tag_sequence(obs, U):
    """Sample a chain of ids with given transition probabilities"""
    n_classes, _ = U.shape
    n_steps, _ = obs.shape
    y = np.zeros((n_steps), dtype=np.int32)
    for t in range(n_steps):
        p = obs[t] * U[y[t-1]] if t >= 1 else obs[t]
        y[t] = np.argmax(p)
    return y


if __name__ == '__main__':
    pytest.main([__file__])
