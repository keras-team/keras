from __future__ import absolute_import

import numpy as np
import theano.tensor as T

from keras import backend as K
from keras.backend.common import _EPSILON as epsilon


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def mean_absolute_percentage_error(y_true, y_pred):
    return T.abs_((y_true - y_pred) / T.clip(T.abs_(y_true), epsilon, np.inf)).mean(axis=-1) * 100.


def mean_squared_logarithmic_error(y_true, y_pred):
    return K.mean(K.square(K.log(K.clip(y_pred, epsilon, np.inf) + 1.) - K.log(K.clip(y_true, epsilon, np.inf) + 1.)))


def squared_hinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean(axis=-1)


def hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean(axis=-1)


def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes
    '''
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    return cce


def binary_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    bce = T.nnet.binary_crossentropy(y_pred, y_true).mean(axis=-1)
    return bce


def poisson_loss(y_true, y_pred):
    return T.mean(y_pred - y_true * T.log(y_pred + epsilon), axis=-1)


# aliases
mse = MSE = mean_squared_error
rmse = RMSE = root_mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error

from .utils.generic_utils import get_from_module


def get(identifier):
    return get_from_module(identifier, globals(), 'objective')
