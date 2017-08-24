'''Tests for functions in np_utils.py.
'''
import os
import pytest
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import batchyield_choice, batchyield_shuffle
import numpy as np
import warnings
import h5py


@pytest.fixture
def in_tmpdir(tmpdir):
    """Runs a function in a temporary directory.

    Checks that the directory is empty afterwards.
    """
    with tmpdir.as_cwd():
        yield None
    assert not tmpdir.listdir()


def create_dataset(npy_path='test'):
    X = np.random.randn(200, 10).astype('float32')
    y = np.random.randint(0, 2, size=(200, 1))
    np.save(npy_path + '_X.npy', X)
    np.save(npy_path + '_y.npy', y)


def test_np_utils_batchyield_shuffle(in_tmpdir):
    '''Test for memmap generators in utils/np_utils.py
    '''
    npy_path = 'test'
    create_dataset(npy_path)

    # Memmap numpy file
    X = np.load(npy_path + '_X.npy', mmap_mode='r')
    y = np.load(npy_path + '_y.npy', mmap_mode='r')

    # Shuffle only first 150 indices
    shuffle_train = np.arange(150)

    fitgen = batchyield_shuffle(X, y, shuffle_train, batchsize=32)

    X_test = X[150:]
    y_test = y[150:]

    # HDF5Matrix behave more or less like Numpy matrices with regards to
    # indexing

    model = Sequential()
    model.add(Dense(64, input_shape=(10,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd')

    model.fit_generator(fitgen, 10, verbose=False)
    # test that evalutation and prediction don't crash and return reasonable
    # results
    out_pred = model.predict(X_test, batch_size=32, verbose=False)
    out_eval = model.evaluate(X_test, y_test, batch_size=32, verbose=False)

    assert out_pred.shape == (50, 1), 'Prediction shape does not match'
    assert out_eval.shape == (), 'Shape of evaluation does not match'
    assert out_eval > 0, 'Evaluation value does not meet criteria: {}'.format(
        out_eval)

    os.remove(npy_path + '_X.npy')
    os.remove(npy_path + '_y.npy')


def test_np_utils_batchyield_choice(in_tmpdir):
    '''Test for memmap generators in utils/np_utils.py
    '''
    npy_path = 'test'
    create_dataset(npy_path)

    # Memmap numpy file
    X = np.load(npy_path + '_X.npy', mmap_mode='r')
    y = np.load(npy_path + '_y.npy', mmap_mode='r')

    # Creates generator to use in fit_generator
    fitgen = batchyield_choice(X, y, batchsize=32)

    X_test = X[150:]
    y_test = y[150:]

    model = Sequential()
    model.add(Dense(64, input_shape=(10,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd')

    model.fit_generator(fitgen, 10, verbose=False)
    # test that evalutation and prediction don't crash and return reasonable
    # results
    out_pred = model.predict(X_test, batch_size=32, verbose=False)
    out_eval = model.evaluate(X_test, y_test, batch_size=32, verbose=False)

    assert out_pred.shape == (50, 1), 'Prediction shape does not match'
    assert out_eval.shape == (), 'Shape of evaluation does not match'
    assert out_eval > 0, 'Evaluation value does not meet criteria: {}'.format(
        out_eval)

    os.remove(npy_path + '_X.npy')
    os.remove(npy_path + '_y.npy')


if __name__ == '__main__':
    pytest.main([__file__])
