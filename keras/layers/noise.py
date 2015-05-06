# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import theano
import theano.tensor as T

from .. import activations, initializations
from ..regularizers import identity
from ..utils.theano_utils import shared_zeros, floatX
from ..utils.generic_utils import make_tuple
from ..layers.core import Layer

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from six.moves import zip
srng = RandomStreams()


class Dropout(Layer):
    '''
        Hinton's dropout.
    '''
    def __init__(self, p):
        super(Dropout,self).__init__()
        self.p = p

    def output(self, train):
        X = self.get_input(train)
        if self.p > 0.:
            retain_prob = 1. - self.p
            if train:
                X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            else:
                X *= retain_prob
        return X

    def get_config(self):
        return {"name":self.__class__.__name__,
            "p":self.p}
            
            
class GaussianNoise(Layer):
    '''
        Multiplicative Gaussian Noise
        Reference: 
            Dropout: A Simple Way to Prevent Neural Networks from Overfitting
            Srivastava, Hinton, et al. 2014
    '''
    def __init__(self, p):
        super(GaussianNoise,self).__init__()
        self.p = p

    def output(self, train):
        X = self.get_input(train)
        if train:
            # self.p refers to drop probability rather than retain probability (as in paper) to match Dropout layer syntax
            X *= srng.normal(X.shape, avg = 1., std=T.sqrt(self.p/(1-self.p)), dtype=theano.config.floatX)
        return X
                    