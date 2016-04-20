from __future__ import absolute_import
import numpy as np
import scipy as sp
from six.moves import range
from six.moves import zip


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def binary_logloss(p, y):
    epsilon = 1e-15
    p = sp.maximum(epsilon, p)
    p = sp.minimum(1-epsilon, p)
    res = sum(y * sp.log(p) + sp.subtract(1, y) * sp.log(sp.subtract(1, p)))
    res *= -1.0/len(y)
    return res


def multiclass_logloss(P, Y):
    npreds = [P[i][Y[i]-1] for i in range(len(Y))]
    score = -(1. / len(Y)) * np.sum(np.log(npreds))
    return score


def accuracy(p, y):
    return np.mean([a == b for a, b in zip(p, y)])


def probas_to_classes(y_pred):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        return categorical_probas_to_classes(y_pred)
    return np.array([1 if p > 0.5 else 0 for p in y_pred])


def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)


def convert_kernel(kernel, dim_ordering='th'):
    '''Converts a kernel matrix (numpy array)
    from Theano format to TensorFlow format
    (or reciprocally, since the transformation
    is its own inverse).
    '''
    new_kernel = np.copy(kernel)
    if dim_ordering == 'th':
        w = kernel.shape[2]
        h = kernel.shape[3]
        for i in range(w):
            for j in range(h):
                new_kernel[:, :, i, j] = kernel[:, :, w - i - 1, h - j - 1]
    elif dim_ordering == 'tf':
        w = kernel.shape[0]
        h = kernel.shape[1]
        for i in range(w):
            for j in range(h):
                new_kernel[i, j, :, :] = kernel[w - i - 1, h - j - 1, :, :]
    else:
        raise Exception('Invalid dim_ordering: ' + str(dim_ordering))
    return new_kernel
