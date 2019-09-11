"""Kannada-MNIST dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as joinpath
from os.path import split as splitpath

from ..utils.data_utils import get_file
import numpy as np


def load_data(path='kannada-mnist.tar.bz'):
    """Loads the Kannada-MNIST dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    # Download and Unpack
    filepath = get_file(
        path, 'https://unifyid-public-datasets.s3-us-west-2.amazonaws.com/kannada-mnist/kannada-mnist.tar.bz',
        cache_subdir=joinpath('datasets', 'kannada-mnist'),
        file_hash='c1cd9953366bb42c06f1205691d399b4',
        extract=True,
    )

    basedir = splitpath(filepath)[0]

    x_train = _read_kannada_mnist(joinpath(basedir, 'X_kannada_MNIST_train-idx3-ubyte'), xory='x')
    y_train = _read_kannada_mnist(joinpath(basedir, 'y_kannada_MNIST_train-idx1-ubyte'), xory='y')
    x_test = _read_kannada_mnist(joinpath(basedir, 'X_kannada_MNIST_test-idx3-ubyte'), xory='x')
    y_test = _read_kannada_mnist(joinpath(basedir, 'y_kannada_MNIST_test-idx1-ubyte'), xory='y')

    return (x_train, y_train), (x_test, y_test)

def _read_kannada_mnist(filepath, xory='x'):
    """Read and deserialize kannada mnist files containing raw binary tensors

    # Returns
        Numpy tensors
    """
    offset, shape = (16, (-1, 28, 28)) if xory == 'x' else (8, -1)

    with open(filepath, 'rb') as fp:
        return np.frombuffer(fp.read(), np.uint8, offset=offset).reshape(shape)
