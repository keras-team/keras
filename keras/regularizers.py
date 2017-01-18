from __future__ import absolute_import
from . import backend as K
from .utils.generic_utils import get_from_module


class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, x):
        return 0.


class L1L2(Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(x))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l1': float(self.l1),
                'l2': float(self.l2)}


# Aliases.


def l1(l=0.01):
    return L1L2(l1=l)


def l2(l=0.01):
    return L1L2(l2=l)


def l1l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2)


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'regularizer',
                           instantiate=True, kwargs=kwargs)
