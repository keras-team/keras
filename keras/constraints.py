from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np


class Constraint(object):
    def __call__(self, p):
        return p

    def get_config(self):
        return {"name": self.__class__.__name__}


class MaxNorm(Constraint):
    def __init__(self, m=2):
        self.m = m

    def __call__(self, p):
        norms = T.sqrt(T.sum(T.sqr(p), axis=0))
        desired = T.clip(norms, 0, self.m)
        p = p * (desired / (1e-7 + norms))
        return p

    def get_config(self):
        return {"name": self.__class__.__name__,
                "m": self.m}


class NonNeg(Constraint):
    def __call__(self, p):
        p *= T.ge(p, 0)
        return p


class UnitNorm(Constraint):
    def __call__(self, p):
        return p / T.sqrt(T.sum(p**2, axis=-1, keepdims=True))

identity = Constraint
maxnorm = MaxNorm
nonneg = NonNeg
unitnorm = UnitNorm

from .utils.generic_utils import get_from_module
def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'constraint', instantiate=True, kwargs=kwargs)
