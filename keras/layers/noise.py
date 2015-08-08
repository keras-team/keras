from __future__ import absolute_import
import numpy as np
from .core import MaskedLayer
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class GaussianNoise(MaskedLayer):
    '''
        Corruption process with GaussianNoise
    '''
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
        self.srng = RandomStreams(seed=np.random.randint(10e6))

    def get_output(self, train=False):
        X = self.get_input(train)
        if not train or self.sigma == 0:
            return X
        else:
            return X + self.srng.normal(size=X.shape, avg=0.0, std=self.sigma,
                                        dtype=theano.config.floatX)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "sigma": self.sigma}


class GaussianDropout(MaskedLayer):
    '''
        Multiplicative Gaussian Noise
        Reference:
            Dropout: A Simple Way to Prevent Neural Networks from Overfitting
            Srivastava, Hinton, et al. 2014
            http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
    '''
    def __init__(self, p):
        super(GaussianDropout, self).__init__()
        self.p = p
        self.srng = RandomStreams(seed=np.random.randint(10e6))

    def get_output(self, train):
        X = self.get_input(train)
        if train:
            # self.p refers to drop probability rather than retain probability (as in paper) to match Dropout layer syntax
            X *= self.srng.normal(size=X.shape, avg=1.0, std=T.sqrt(self.p / (1.0 - self.p)), dtype=theano.config.floatX)
        return X

    def get_config(self):
        return {"name": self.__class__.__name__,
                "p": self.p}
