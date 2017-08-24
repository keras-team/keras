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


def batchyield_shuffle(data,shufflearr=None,batchsize=256,stopiter=None):
    """Creates a generator over numpy arrays with
    shuffling that can iterate over memmaped arrays

    # Arguments
        data: Arraylike (numpy array or memmap)
            to iterate over.
        shufflearr: (optional) if only you want to
            iterate over a subset of indices
        batchsize: size of returned batchsize
        stopiter: total number of batches

    # Returns
        generator that returns batches of data
    """
    if shufflearr is not None:
        shufflearr=np.arange(data.shape[0])
    np.random.shuffle(shufflearr)
    dataind=0
    iterations=0
    while 1:
        if stopiter:
            if iterations>=stopiter:
                break
        endind=dataind+batchsize
        if endind>(shufflearr.shape[0]-1):
            np.random.shuffle(shufflearr)
            dataind=0
            endind=batchsize
        inds=shufflearr[dataind:endind]
        dataind+=batchsize
        yield data[inds]
        iterations+=1


def batchyield_choice(data,batchsize=256,stopiter=None):
    """Creates a generator over numpy arrays a
    random subset of indices that can iterate
    over memmaped arrays

    # Arguments
        data: Arraylike (numpy array or memmap)
            to iterate over.
        batchsize: size of returned batchsize
        stopiter: total number of batches

    # Returns
        generator that returns batches of data
    """
    dataind=0
    iterations=0
    while 1:
        if stopiter:
            if iterations>=stopiter:
                break
        batchinds=np.random.choice(data.shape[0],batchsize,replace=False)
        yield data[batchinds]
        iterations+=1
