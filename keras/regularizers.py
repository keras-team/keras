from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

def l1(l=.01):
    def l1wrap(g, p):
        g += T.sgn(p) * l
        return g
    return l1wrap

def l2(l=.01):
    def l2wrap(g, p):
        g += p * l
        return g
    return l2wrap

def l1l2(l1=.01, l2=.01):
    def l1l2wrap(g, p):
        g += T.sgn(p) * l1
        g += p * l2
        return g
    return l1l2wrap

def identity(g, p):
    return g