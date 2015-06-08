from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
from six.moves import range

epsilon = 1.0e-15

def mean_squared_error(y_true, y_pred, weight=None):
    return T.sqr(y_pred - y_true).mean()

def mean_absolute_error(y_true, y_pred, weight=None):
    return T.abs_(y_pred - y_true).mean()

def squared_hinge(y_true, y_pred, weight=None):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean()

def hinge(y_true, y_pred, weight=None):
    return T.maximum(1. - y_true * y_pred, 0.).mean()

def categorical_crossentropy(y_true, y_pred, weight=None):
    '''Expects a binary class matrix instead of a vector of scalar classes
    '''
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= y_pred.sum(axis=1, keepdims=True) 
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    if weight is not None:
        # return avg. of scaled cat. crossentropy
        return (weight*cce).mean()
    else:
        return cce.mean()

def binary_crossentropy(y_true, y_pred, weight=None):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    bce = T.nnet.binary_crossentropy(y_pred, y_true).mean()
    if weight is not None:
        return (weight*bce).mean()
    else:
        return bce.mean()

# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')

def to_categorical(y):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y
