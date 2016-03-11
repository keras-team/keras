from __future__ import absolute_import
from . import backend as K


class Regularizer(object):
    def set_param(self, p):
        self.p = p

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__}


class WeightRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        loss += K.sum(K.abs(self.p)) * self.l1
        loss += K.sum(K.square(self.p)) * self.l2
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2}


class ActivityRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        output = self.layer.get_output(True)
        loss += self.l1 * K.sum(K.mean(K.abs(output), axis=0))
        loss += self.l2 * K.sum(K.mean(K.square(output), axis=0))
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2}


class GaussianKL(Regularizer):
    """ KL-divergence between two gaussians.
    Useful for Variational AutoEncoders. Use this as an activation regularizer

    Parameters:
    -----------
    mean, logsigma: parameters of the input distributions.
    prior_mean, prior_logsigma: paramaters of the desired distribution (note the
    log on logsigma)
    regularizer_scale: Rescales the regularization cost. Keep this 1 for most cases.

    Notes:
    ------
    See keras.layers.variational.VariationalDense for usage example

    """
    def __init__(self, mean, logsigma, prior_mean=0, prior_logsigma=0,
                 regularizer_scale=1):
        self.regularizer_scale = regularizer_scale
        self.mean = mean
        self.logsigma = logsigma
        self.prior_mean = prior_mean
        self.prior_logsigma = prior_logsigma
        super(GaussianKL, self).__init__()

    def __call__(self, loss):
        # See Variational Auto-Encoding Bayes by Kingma and Welling.
        mean, logsigma = self.mean, self.logsigma
        den = 2 * K.exp(2. * self.prior_logsigma)
        sigma_part = self.prior_logsigma - logsigma + K.exp(2*logsigma) / den
        mean_part = K.pow(mean - self.prior_mean, 2) / den
        kl = sigma_part + mean_part
        loss += K.mean(kl) * self.regularizer_scale
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "prior_mean": self.prior_mean,
                "prior_logsigma": self.prior_logsigma,
                "regularizer_scale": self.regularizer_scale}


def l1(l=0.01):
    return WeightRegularizer(l1=l)


def l2(l=0.01):
    return WeightRegularizer(l2=l)


def l1l2(l1=0.01, l2=0.01):
    return WeightRegularizer(l1=l1, l2=l2)


def activity_l1(l=0.01):
    return ActivityRegularizer(l1=l)


def activity_l2(l=0.01):
    return ActivityRegularizer(l2=l)


def activity_l1l2(l1=0.01, l2=0.01):
    return ActivityRegularizer(l1=l1, l2=l2)

identity = Regularizer

from .utils.generic_utils import get_from_module
def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'regularizer',
                           instantiate=True, kwargs=kwargs)
