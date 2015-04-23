from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

def l1(lam=.01):
    def l1wrap(g,p):
        g += T.sgn(p) * lam
        return g
    return l1wrap

def l2(lam=.01):
    def l2wrap(g,p):
        g += p * lam
        return g
    return l2wrap

def ident(g,*l):
    return g