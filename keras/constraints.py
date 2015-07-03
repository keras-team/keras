from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

class Constraint(object):
    def __call__(self, p):
        return p

class MaxNorm(Constraint):
    def __init__(self, m=2):
        self.m = m

    def __call__(self, p):
        norms = T.sqrt(T.sum(T.sqr(p), axis=0))
        desired = T.clip(norms, 0, self.m)
        p = p * (desired / (1e-7 + norms))
        return p

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
