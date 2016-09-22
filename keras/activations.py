#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from . import backend as K


def softmax(x):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim == 3:
        e = K.exp(x - K.max(x, axis=-1, keepdims=True))
        s = K.sum(e, axis=-1, keepdims=True)
        return e / s
    else:
        raise Exception('Cannot apply softmax to a tensor that is not 2D or 3D. ' +
                        'Here, ndim=' + str(ndim))


def softplus(x):
    return K.softplus(x)


def softsign(x):
    return K.softsign(x)


def relu(x, alpha=0., max_value=None):
    return K.relu(x, alpha=alpha, max_value=max_value)


def softexp(x, alpha=0., max_value=None):
    """Soft Exponential activation function by Godfrey and Gashler

    See: https://arxiv.org/pdf/1602.01321.pdf

    α == 0:  f(α, x) = x
    α  > 0:  f(α, x) = (exp(αx)-1) / α + α
    α  < 0:  f(α, x) = -ln(1-α(x + α)) / α
    """
    if alpha == 0:
        return x
    elif alpha > 0:
        return alpha + (K.exp(alpha * x) - 1.) / alpha
    else:
        return - K.log(1 - alpha * (x + alpha)) / alpha


def tanh(x):
    return K.tanh(x)


def sigmoid(x):
    return K.sigmoid(x)


def hard_sigmoid(x):
    return K.hard_sigmoid(x)


def linear(x):
    '''
    The function returns the variable that is passed in, so all types work.
    '''
    return x


from .utils.generic_utils import get_from_module
def get(identifier):
    if identifier is None:
        return linear
    return get_from_module(identifier, globals(), 'activation function')
