from __future__ import division, print_function

from collections import OrderedDict

import numpy as np

from keras import backend as K
from keras.layers import Dense, concatenate
from keras.activations import softmax


class DistributionBase(object):
    """Defines activation for distribution parameters and (default) loss for
    target given the distribution."""

    def activation(self, x):
        """Activation function to apply to get parameters of distribution."""
        raise NotImplementedError

    def loss(self, y_true, y_pred):
        """Implementation of standard loss for this this distribution, normally
        -log(pdf(y_true)) where pdf is parametrized by y_pred (the parameters).
        """
        raise NotImplementedError

    def pdf(self, y_true, y_pred):
        """Probability density function"""
        raise NotImplementedError

    @property
    def num_params(self):
        """Expected size of x in activation and y_pred in loss"""
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError


class MixtureDistributionBase(DistributionBase):
    """Base class for Mixture Distributions"""

    def __init__(self, num_components):
        self.num_components = num_components

    @property
    def mixture_weight_activation(self):
        return softmax

    def param_type_to_size(self):
        """
        # Returns
            An OrderedDict of param_type (str) to size (int)
        # Example
            return OrderedDict([
                ('mixture_weight', self.n_components),
                ...
            ])
        """
        raise NotImplementedError

    def split_param_types(self, x):
        """Splits input tensor into the different param types. This method is
        useful for applying activation and computing loss.
        # Args
            x : Tensor with shape[-1] == self.n_params
        # Returns
            list of Tensors, one for each param type
        """
        if isinstance(x, np.ndarray):
            last_dim = x.shape[-1]
        else:
            last_dim = x.shape[-1].value  # TODO only works with tf
        if not last_dim == self.num_params:
            raise ValueError(
                'last dimension of x must be equal to the number of parameters'
                ' of distribution, got {}, expected {}'.format(
                    last_dim,
                    self.num_params
                )
            )

        idx = 0
        param_types = []
        for size in self.param_type_to_size.values():
            param_types.append(x[..., idx:idx+size])
            idx += size

        return param_types

    @property
    def num_params(self):
        return sum(self.param_type_to_size.values())

    def get_config(self):
        return dict(num_components=self.num_components)


class ScaledExponential(object):  # TODO move to advanced activations

    def __init__(self, scale=1, epsilon=1e-3):
        self.scale = scale
        self.epsilon = epsilon

    def __call__(self, x):
        return self.scale * K.exp(x) + self.epsilon


class MixtureOfGaussian1D(MixtureDistributionBase):
    """1D Mixture of Gaussian distribution"""

    def __init__(
        self,
        num_components,
        mu_activation=None,
        sigma_activation=None,
    ):
        super(MixtureOfGaussian1D, self).__init__(num_components)
        self.mu_activation = mu_activation or (lambda x: x)
        self.sigma_activation = sigma_activation or ScaledExponential()

    @property
    def param_type_to_size(self):
        return OrderedDict([
            ('mixture_weight', self.num_components),
            ('mu', self.num_components),
            ('sigma', self.num_components)
        ])

    def activation(self, x):
        _mixture_weights, _mu, _sigma = self.split_param_types(x)
        mixture_weights = self.mixture_weight_activation(_mixture_weights)
        mu = self.mu_activation(_mu)
        sigma = self.sigma_activation(_sigma)

        return concatenate([mixture_weights, mu, sigma], axis=-1)

    def loss(self, y_true, y_pred):
        """Negative log pdf. Used logsum trick for numerical stability"""
        mixture_weights, mu, sigma, = self.split_param_types(y_pred)
        norm = 1. / (np.sqrt(2. * np.pi) * sigma)
        exponent = -(
            K.square(y_true - mu) / (2. * K.square(sigma)) -
            K.log(mixture_weights) -
            K.log(norm)
        )
        return -K.logsumexp(exponent, axis=-1)

    def pdf(self, y_true, y_pred):
        raise NotImplementedError  # TODO

    def get_config(self, y_true, y_pred):
        raise NotImplementedError  # TODO


class DistributionOutputLayer(Dense):
    """Wraps Dense layer to output distribution parameters based on passed
    distribution.

    # Arguments
        distribution (DistributionABC): The distribution to output parameters
        for
    """
    def __init__(self, distribution, **kwargs):
        self.distribution = distribution
        if 'units' in kwargs or 'activation' in kwargs:
            raise ValueError(
                '"units" or "activation" should not be passed as kwargs '
                'as this is already specified by the passed distribution'
            )
        super(DistributionOutputLayer, self).__init__(
            units=distribution.num_params,
            activation=distribution.activation
        )
