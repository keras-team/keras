from __future__ import absolute_import
from functools import wraps
import theano
import theano.tensor as T
import numpy as np


if theano.config.floatX == 'float64':
    epsilon = 1.0e-9
else:
    epsilon = 1.0e-7


def objectivefy(func):
    @wraps(func)
    def wrapper(y_true, y_pred, sample_weight, mask):
        cost = func(y_true, y_pred)
        cost = T.switch(T.eq(sample_weight, 0.), 0., cost)  # makes sure 0 * inf == 0, not NaN
        weighted_cost = cost * sample_weight
        if mask is None:
            return weighted_cost.sum() / sample_weight.sum()
        else:
            mask = T.switch(T.eq(sample_weight, 0.), 0., mask)
            masked_cost = T.switch(T.eq(mask, 0.), 0., weighted_cost)  # Make sure we mask out that cost
            return masked_cost.sum() / (mask * sample_weight).sum()
    return wrapper


@objectivefy
def mean_squared_error(y_true, y_pred):
    return T.sqr(y_pred - y_true)


@objectivefy
def mean_absolute_error(y_true, y_pred):
    return T.abs_(y_pred - y_true)


@objectivefy
def mean_absolute_percentage_error(y_true, y_pred):
    return T.abs_((y_true - y_pred) / T.clip(T.abs_(y_true), epsilon,
                                             np.inf)) * 100


@objectivefy
def mean_squared_logarithmic_error(y_true, y_pred):
    return T.sqr(T.log(T.clip(y_pred, epsilon, np.inf) + 1.) -
                 T.log(T.clip(y_true, epsilon, np.inf) + 1.))


@objectivefy
def squared_hinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.))


@objectivefy
def hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean(axis=-1)


@objectivefy
def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes
    '''
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    cce = T.nnet.categorical_crossentropy(y_pred, y_true).dimshuffle(0, 'x')
    return cce


@objectivefy
def binary_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    bce = T.nnet.binary_crossentropy(y_pred, y_true)
    return bce


@objectivefy
def poisson_loss(y_true, y_pred):
    return y_pred - y_true * T.log(y_pred + epsilon)

# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')
