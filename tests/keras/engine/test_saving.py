import os
import io
import h5py
import pytest
import tempfile

from contextlib import contextmanager

import numpy as np
from numpy.testing import assert_array_equal

from keras.engine.saving import save_to_binary_h5py, load_from_binary_h5py

# NOTE tests for saving of models are found in tests/test_model_saving.py


@contextmanager
def temp_filename(filename):
    _, temp_filename = tempfile.mkstemp(filename)
    yield temp_filename
    if os.path.exists(temp_filename):
        os.remove(temp_filename)


def test_save_to_binary_h5py_direct_to_file():
    data = np.random.random((3, 5))

    def save_function(h5file_):
        h5file_['data'] = data

    with temp_filename('.h5') as fname:
        with open(fname, 'wb') as f:
            save_to_binary_h5py(save_function, f)

        with h5py.File(fname) as h5file:
            data_rec = h5file['data'][:]

    assert_array_equal(data_rec, data)


def test_save_to_binary_h5py_to_bytes_io():
    data = np.random.random((3, 5))

    def save_function(h5file_):
        h5file_['data'] = data

    file_like = io.BytesIO()
    save_to_binary_h5py(save_function, file_like)

    file_like.seek(0)

    with temp_filename('.h5') as fname:
        with open(fname, 'wb') as f:
            f.write(file_like.read())

        with h5py.File(fname) as h5file:
            data_rec = h5file['data'][:]

    assert_array_equal(data_rec, data)


def test_load_from_binary_h5py_direct_from_file():
    data = np.random.random((3, 5))

    def load_function(h5file_):
        return h5file_['data'][:]

    with temp_filename('.h5') as fname:
        with h5py.File(fname, 'w') as h5file:
            h5file['data'] = data

        with open(fname, 'rb') as f:
            data_rec = load_from_binary_h5py(load_function, f)

    assert_array_equal(data_rec, data)


def test_load_from_binary_h5py_from_bytes_io():
    data = np.random.random((3, 5))

    def load_function(h5file_):
        return h5file_['data'][:]

    with temp_filename('.h5') as fname:
        with h5py.File(fname, 'w') as h5file:
            h5file['data'] = data

        file_like = io.BytesIO()
        with open(fname, 'rb') as f:
            file_like.write(f.read())

    file_like.seek(0)
    data_rec = load_from_binary_h5py(load_function, file_like)

    assert_array_equal(data_rec, data)


def test_save_load_binary_h5py():

    data1 = np.random.random((3, 5))
    data2 = np.random.random((2, 3, 5))
    attr = 1
    datas = [data1, data2, attr]

    def save_function(h5file_):
        h5file_['data1'] = data1
        h5file_['subgroup/data2'] = data2
        h5file_['data1'].attrs['attr'] = attr

    def load_function(h5file_):
        d1 = h5file_['data1'][:]
        d2 = h5file_['subgroup/data2'][:]
        a = h5file_['data1'].attrs['attr']
        return d1, d2, a

    file_like = io.BytesIO()
    save_to_binary_h5py(save_function, file_like)
    file_like.seek(0)
    datas_rec = load_from_binary_h5py(load_function, file_like)
    for d_rec, d in zip(datas_rec, datas):
        assert_array_equal(d_rec, d)


if __name__ == '__main__':
    pytest.main([__file__])
