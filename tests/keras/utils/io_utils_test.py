'''Tests for functions in io_utils.py.
'''
import os
import pytest
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.io_utils import HDF5Matrix
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


def create_dataset(h5_path='test.h5'):
    X = np.random.randn(200, 10).astype('float32')
    y = np.random.randint(0, 2, size=(200, 1))
    f = h5py.File(h5_path, 'w')
    # Creating dataset to store features
    X_dset = f.create_dataset('my_data', (200, 10), dtype='f')
    X_dset[:] = X
    # Creating dataset to store labels
    y_dset = f.create_dataset('my_labels', (200, 1), dtype='i')
    y_dset[:] = y
    f.close()


def test_io_utils(in_tmpdir):
    '''Tests the HDF5Matrix code using the sample from @jfsantos at
    https://gist.github.com/jfsantos/e2ef822c744357a4ed16ec0c885100a3
    '''
    h5_path = 'test.h5'
    create_dataset(h5_path)

    # Instantiating HDF5Matrix for the training set, which is a slice of the first 150 elements
    X_train = HDF5Matrix(h5_path, 'my_data', start=0, end=150)
    y_train = HDF5Matrix(h5_path, 'my_labels', start=0, end=150)

    # Likewise for the test set
    X_test = HDF5Matrix(h5_path, 'my_data', start=150, end=200)
    y_test = HDF5Matrix(h5_path, 'my_labels', start=150, end=200)

    # HDF5Matrix behave more or less like Numpy matrices with regards to indexing
    assert y_train.shape == (150, 1), 'HDF5Matrix shape should match input array'
    # But they do not support negative indices, so don't try print(X_train[-1])

    assert y_train.dtype == np.dtype('i'), 'HDF5Matrix dtype should match input array'
    assert y_train.ndim == 2, 'HDF5Matrix ndim should match input array'
    assert y_train.size == 150, 'HDF5Matrix ndim should match input array'

    model = Sequential()
    model.add(Dense(64, input_shape=(10,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd')

    # Note: you have to use shuffle='batch' or False with HDF5Matrix
    model.fit(X_train, y_train, batch_size=32, shuffle='batch', verbose=False)
    # test that evalutation and prediction don't crash and return reasonable results
    out_pred = model.predict(X_test, batch_size=32, verbose=False)
    out_eval = model.evaluate(X_test, y_test, batch_size=32, verbose=False)

    assert out_pred.shape == (50, 1), 'Prediction shape does not match'
    assert out_eval.shape == (), 'Shape of evaluation does not match'
    assert out_eval > 0, 'Evaluation value does not meet criteria: {}'.format(out_eval)

    # test slicing for shortened array
    assert len(X_train[0:]) == len(X_train), 'Incorrect shape for sliced data'

    os.remove(h5_path)


if __name__ == '__main__':
    pytest.main([__file__])
