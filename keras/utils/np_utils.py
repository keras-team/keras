"""Numpy-related utilities."""
from __future__ import absolute_import

import numpy as np
from six.moves import zip
import collections
import itertools


def to_categorical(y, nb_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector (single label) or iterable of class vectors (multi
           label) to be converted into a matrix (integers from 0 to nb_classes).
        nb_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    n = len(y)
    if n > 0 and isinstance(y[0], collections.Iterable):
        if not nb_classes:
            nb_classes = max(itertools.chain(*y)) + 1
        categorical = np.zeros((n, nb_classes))
        for i in range(n):
            categorical[i, list(y[i])] = 1.
    else:
        y = np.array(y, dtype='int').ravel()
        if not nb_classes:
            nb_classes = np.max(y) + 1
        categorical = np.zeros((n, nb_classes))
        categorical[np.arange(n), y] = 1
    return categorical


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def accuracy(p, y):
    return np.mean([a == b for a, b in zip(p, y)])


def probas_to_classes(y_pred):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        return categorical_probas_to_classes(y_pred)
    return np.array([1 if p > 0.5 else 0 for p in y_pred])


def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)
