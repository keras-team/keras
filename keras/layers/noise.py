from __future__ import absolute_import
from .core import srng, MaskedLayer
import theano

class GaussianNoise(MaskedLayer):
    '''
        Corruption process with GaussianNoise
    '''
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def get_output(self, train=False):
        X = self.get_input(train)
        if not train or self.sigma == 0:
            return X
        else:
            return X + srng.normal(size=X.shape, avg=0.0, std=self.sigma,
                             dtype=theano.config.floatX)

    def get_config(self):
        return {"name":self.__class__.__name__,
            "sigma":self.sigma}