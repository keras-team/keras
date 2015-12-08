from .. import initializations
from ..layers.core import Layer, MaskedLayer
from .. import backend as K
import numpy as np


class LeakyReLU(MaskedLayer):
    def __init__(self, alpha=0.3, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.alpha = alpha

    def get_output(self, train):
        X = self.get_input(train)
        return K.relu(X, alpha=self.alpha)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha": self.alpha}
        base_config = super(LeakyReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PReLU(MaskedLayer):
    '''
        Reference:
            Delving Deep into Rectifiers: Surpassing Human-Level
            Performance on ImageNet Classification
                http://arxiv.org/pdf/1502.01852v1.pdf
    '''
    def __init__(self, init='zero', weights=None, **kwargs):
        self.init = initializations.get(init)
        self.initial_weights = weights
        super(PReLU, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape[1:]
        self.alphas = self.init(input_shape)
        self.params = [self.alphas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output(self, train):
        X = self.get_input(train)
        pos = K.relu(X)
        neg = self.alphas * (X - abs(X)) * 0.5
        return pos + neg

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "init": self.init.__name__}
        base_config = super(PReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ELU(MaskedLayer):
    def __init__(self, alpha=1.0, **kwargs):
        super(ELU, self).__init__(**kwargs)
        self.alpha = alpha

    def get_output(self, train):
        X = self.get_input(train)
        pos = K.relu(X)
        neg = (X - abs(X)) * 0.5
        return pos + self.alpha * (K.exp(neg) - 1.)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha": self.alpha}
        base_config = super(ELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ParametricSoftplus(MaskedLayer):
    '''
        Parametric Softplus of the form: alpha * log(1 + exp(beta * X))

        Reference:
            Inferring Nonlinear Neuronal Computation
            Based on Physiologically Plausible Inputs
            http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003143
    '''
    def __init__(self, alpha_init=0.2, beta_init=5.0,
                 weights=None, **kwargs):
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.initial_weights = weights
        super(ParametricSoftplus, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape[1:]
        self.alphas = K.variable(self.alpha_init * np.ones(input_shape))
        self.betas = K.variable(self.beta_init * np.ones(input_shape))
        self.params = [self.alphas, self.betas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output(self, train):
        X = self.get_input(train)
        return K.softplus(self.betas * X) * self.alphas

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha_init": self.alpha_init,
                  "beta_init": self.beta_init}
        base_config = super(ParametricSoftplus, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ThresholdedLinear(MaskedLayer):
    '''
        Thresholded Linear Activation

        Reference:
            Zero-Bias Autoencoders and the Benefits of Co-Adapting Features
            http://arxiv.org/pdf/1402.3337.pdf
    '''
    def __init__(self, theta=1.0, **kwargs):
        super(ThresholdedLinear, self).__init__(**kwargs)
        self.theta = theta

    def get_output(self, train):
        X = self.get_input(train)
        return K.switch(K.abs(X) < self.theta, 0, X)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "theta": self.theta}
        base_config = super(ThresholdedLinear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ThresholdedReLU(MaskedLayer):
    '''
        Thresholded Rectified Activation

        Reference:
            Zero-Bias Autoencoders and the Benefits of Co-Adapting Features
            http://arxiv.org/pdf/1402.3337.pdf
    '''
    def __init__(self, theta=1.0, **kwargs):
        super(ThresholdedReLU, self).__init__(**kwargs)
        self.theta = theta

    def get_output(self, train):
        X = self.get_input(train)
        return K.switch(X > self.theta, X, 0)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "theta": self.theta}
        base_config = super(ThresholdedReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
