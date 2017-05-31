from __future__ import absolute_import
import six
from . import backend as K
from .utils.generic_utils import serialize_keras_object
from .utils.generic_utils import deserialize_keras_object


class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class L1L2(Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(x))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}


# Aliases.


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
    return L1L2(l1=l)


def l2(l=0.01):
    return L1L2(l2=l)


def l1_l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2)


def serialize(regularizer):
    return serialize_keras_object(regularizer)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='regularizer')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret regularizer identifier:',
                         identifier)
