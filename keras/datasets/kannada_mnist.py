"""Kannada-MNIST dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as joinpath
from os.path import split as splitpath

from ..utils.data_utils import get_file
import numpy as np


def load_data():
    """Loads the Kannada-MNIST dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = joinpath('datasets', 'kannada-mnist')
    baseurl = 'https://unifyid-public-datasets.s3-us-west-2.amazonaws.com/kannada-mnist/'
    fname = 'kannada-mnist.tar.bz'
    fhash = 'c1cd9953366bb42c06f1205691d399b4'

    # Download and Unpack
    basepath = splitpath(get_file(fname, baseurl + fname, cache_subdir=dirname, file_hash=fhash, extract=True))[0]

    x_train = _read_kmnist(basepath + 'X_kannada_MNIST_train-idx3-ubyte', xory='x')
    y_train = _read_kmnist(basepath + 'y_kannada_MNIST_train-idx1-ubyte', xory='y')
    x_test = _read_kmnist(basepath + 'X_kannada_MNIST_test-idx3-ubyte', xory='x')
    y_test = _read_kmnist(basepath + 'y_kannada_MNIST_test-idx1-ubyte', xory='y')

    return (x_train, y_train), (x_test, y_test)

def _read_kmnist(filepath, xory='x'):
    """Read and deserialize kmnist files containing raw binary tensors

    # Returns
        Numpy tensors
    """
    offset, shape = (16, (-1, 28, 28)) if xory == 'x' else (8, -1)

    with open(filepath, 'rb') as fp:
        return np.frombuffer(fp.read(), np.uint8, offset=offset).reshape(shape)
