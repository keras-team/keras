from .. import initializations
from ..layers.core import MaskedLayer
from .. import backend as K
import numpy as np


class LeakyReLU(MaskedLayer):
    '''Special version of a Rectified Linear Unit
    that allows a small gradient when the unit is not active:
    `f(x) = alpha*x for x < 0`.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        alpha: float >= 0. Negative slope coefficient.
    '''
    def __init__(self, alpha=0.3, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.alpha = alpha

    def get_output(self, train):
        X = self.get_input(train)
        return K.relu(X, alpha=self.alpha)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'alpha': self.alpha}
        base_config = super(LeakyReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PReLU(MaskedLayer):
    '''
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments:
        init: initialization function for the weights.
        weights: initial weights, as a list of a single numpy array.

    # References:
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)
    '''
    def __init__(self, init='zero', weights=None, **kwargs):
        self.init = initializations.get(init)
        self.initial_weights = weights
        super(PReLU, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape[1:]
        self.alphas = self.init(input_shape)
        self.trainable_weights = [self.alphas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output(self, train):
        X = self.get_input(train)
        pos = K.relu(X)
        neg = self.alphas * (X - abs(X)) * 0.5
        return pos + neg

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'init': self.init.__name__}
        base_config = super(PReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ELU(MaskedLayer):
    '''
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        alpha: scale for the negative factor.

    # References
        - [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/pdf/1511.07289v1.pdf)
    '''
    def __init__(self, alpha=1.0, **kwargs):
        super(ELU, self).__init__(**kwargs)
        self.alpha = alpha

    def get_output(self, train):
        X = self.get_input(train)
        pos = K.relu(X)
        neg = (X - abs(X)) * 0.5
        return pos + self.alpha * (K.exp(neg) - 1.)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'alpha': self.alpha}
        base_config = super(ELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ParametricSoftplus(MaskedLayer):
    '''Parametric Softplus of the form: alpha * log(1 + exp(beta * X))

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        alpha_init: float. Initial value of the alpha weights.
        beta_init: float. Initial values of the beta weights.
        weights: initial weights, as a list of 2 numpy arrays.

    # References:
        - [Inferring Nonlinear Neuronal Computation Based on Physiologically Plausible Inputs](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003143)
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
        self.trainable_weights = [self.alphas, self.betas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output(self, train):
        X = self.get_input(train)
        return K.softplus(self.betas * X) * self.alphas

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'alpha_init': self.alpha_init,
                  'beta_init': self.beta_init}
        base_config = super(ParametricSoftplus, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ThresholdedLinear(MaskedLayer):
    '''Thresholded Linear Activation.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        theta: float >= 0. Threshold location of activation.

    # References
        [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/pdf/1402.3337.pdf)
    '''
    def __init__(self, theta=1.0, **kwargs):
        super(ThresholdedLinear, self).__init__(**kwargs)
        self.theta = theta

    def get_output(self, train):
        X = self.get_input(train)
        return K.switch(K.abs(X) < self.theta, 0, X)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'theta': self.theta}
        base_config = super(ThresholdedLinear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ThresholdedReLU(MaskedLayer):
    '''Thresholded Rectified Activation.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        theta: float >= 0. Threshold location of activation.

    # References
        [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/pdf/1402.3337.pdf)
    '''
    def __init__(self, theta=1.0, **kwargs):
        super(ThresholdedReLU, self).__init__(**kwargs)
        self.theta = theta

    def get_output(self, train):
        X = self.get_input(train)
        return K.switch(X > self.theta, X, 0)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'theta': self.theta}
        base_config = super(ThresholdedReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SReLU(MaskedLayer):
    '''SReLU

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        t_left_init: initialization function for the left part intercept
        a_left_init: initialization function for the left part slope
        t_right_init: initialization function for the right part intercept
        a_right_init: initialization function for the right part slope

    # References
        [Deep Learning with S-shaped Rectified Linear Activation Units](http://arxiv.org/abs/1512.07030)
    '''
    def __init__(self, t_left_init='zero', a_left_init='glorot_uniform',
                 t_right_init='glorot_uniform', a_right_init='one', **kwargs):
        self.t_left_init = t_left_init
        self.a_left_init = a_left_init
        self.t_right_init = t_right_init
        self.a_right_init = a_right_init
        super(SReLU, self).__init__(**kwargs)

    def build(self):
        params_shape = self.input_shape[1:]
        self.t_left = initializations.get(self.t_left_init)(params_shape)
        self.a_left = initializations.get(self.a_left_init)(params_shape)
        self.t_right = initializations.get(self.t_right_init)(params_shape)
        self.a_right = initializations.get(self.a_right_init)(params_shape)
        # ensure the the right part is always to the right of the left
        self.t_right_actual = self.t_left + abs(self.t_right)
        self.trainable_weights = [self.t_left, self.a_left, self.t_right, self.a_right]

    def get_output(self, train=False):
        X = self.get_input(train)
        Y_left_and_center = self.t_left + K.relu(X - self.t_left, self.a_left, self.t_right_actual - self.t_left)
        Y_right = K.relu(X - self.t_right_actual) * self.a_right
        return Y_left_and_center + Y_right

    def get_config(self):
        return {'name': self.__class__.__name__,
                'input_shape': self.input_shape,
                't_left_init': self.t_left_init,
                'a_left_init': self.a_left_init,
                't_right_init': self.t_right_init,
                'a_right_init': self.a_right_init}
