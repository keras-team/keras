from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

def maxnorm(m=2):
    def maxnorm_wrap(p):
        norms = T.sqrt(T.sum(T.sqr(p), axis=0))
        desired = T.clip(norms, 0, m)
        p = p * (desired / (1e-7 + norms))
        return p
    return maxnorm_wrap

def nonneg(p):
    p *= T.ge(p, 0)
    return p

def identity(g):
    return g