from __future__ import absolute_import
from .core import MaskedLayer
from .. import backend as K


class GaussianNoise(MaskedLayer):
    '''
        Corruption process with GaussianNoise
    '''
    def __init__(self, sigma, **kwargs):
        super(GaussianNoise, self).__init__(**kwargs)
        self.sigma = sigma

    def get_output(self, train=False):
        X = self.get_input(train)
        if not train or self.sigma == 0:
            return X
        else:
            return X + K.random_normal(shape=K.shape(X),
                                       mean=0.,
                                       std=self.sigma)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "sigma": self.sigma}
        base_config = super(GaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GaussianDropout(MaskedLayer):
    '''
        Multiplicative Gaussian Noise
        Reference:
            Dropout: A Simple Way to Prevent Neural Networks from Overfitting
            Srivastava, Hinton, et al. 2014
            http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
    '''
    def __init__(self, p, **kwargs):
        super(GaussianDropout, self).__init__(**kwargs)
        self.p = p

    def get_output(self, train):
        X = self.get_input(train)
        if train:
            # self.p refers to drop probability rather than
            # retain probability (as in paper), for consistency
            X *= K.random_normal(shape=K.shape(X), mean=1.0,
                                 std=self.p / (1.0 - self.p))
        return X

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "p": self.p}
        base_config = super(GaussianDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
