"""Numpy-related utilities."""
from __future__ import absolute_import

import numpy as np


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def normalize(x, axis=-1, order=2):
    """Normalizes a Numpy array.

    # Arguments
        x: Numpy array to normalize.
        axis: axis along which to normalize.
        order: Normalization order (e.g. 2 for L2 norm).

    # Returns
        A normalized copy of the array.
    """
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)


def batchyield_shuffle(x, y, shufflearr=None, batchsize=256, stopiter=None):
    """Creates a generator over numpy arrays with
    shuffling that can iterate over memmaped arrays

    # Arguments
        x: Feature data (numpy array or memmap)
            to iterate over.
        y: Target data (numpy array or memmap)
            to iterate over.
        shufflearr: (optional) if only you want to
            iterate over a subset of indices
        batchsize: size of returned batchsize
        stopiter: total number of batches

    # Returns
        generator that returns batches of data
    """
    if shufflearr is not None:
        if isinstance(x, np.ndarray):
            n_samp = x.shape[0]
        else:
            n_samp = x[0].shape[0]
        shufflearr = np.arange(n_samp)
    np.random.shuffle(shufflearr)
    dataind = 0
    iterations = 0
    while True:
        if stopiter:
            if iterations >= stopiter:
                break
        endind = dataind + batchsize
        if endind > (shufflearr.shape[0] - 1):
            np.random.shuffle(shufflearr)
            dataind = 0
            endind = batchsize
        batchinds = shufflearr[dataind:endind]
        dataind += batchsize
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            yield (x[batchinds], y[batchinds])
        elif isinstance(x, list) and isinstance(y, list):
            yield ([kk[batchinds] for kk in x], [ss[batchinds] for ss in y])
        iterations += 1


def batchyield_choice(x, y, batchsize=256, stopiter=None):
    """Creates a generator over numpy arrays a
    random subset of indices that can iterate
    over memmaped arrays

    # Arguments
        x: Feature data (numpy array or memmap)
            to iterate over.
        y: Target data (numpy array or memmap)
            to iterate over.
        batchsize: size of returned batchsize
        stopiter: total number of batches

    # Returns
        generator that returns batches of data
    """
    dataind = 0
    iterations = 0
    if isinstance(x, np.ndarray):
        n_samp = x.shape[0]
    else:
        n_samp = x[0].shape[0]
    while True:
        if stopiter:
            if iterations >= stopiter:
                break
        batchinds = np.random.choice(n_samp, batchsize, replace=False)
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            yield (x[batchinds], y[batchinds])
        elif isinstance(x, list) and isinstance(y, list):
            yield ([kk[batchinds] for kk in x], [ss[batchinds] for ss in y])
        iterations += 1
