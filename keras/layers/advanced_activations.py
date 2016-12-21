from .. import initializations
from ..engine import Layer
from .. import backend as K
import numpy as np


class LeakyReLU(Layer):
    '''Special version of a Rectified Linear Unit
    that allows a small gradient when the unit is not active:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`.

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
        self.supports_masking = True
        self.alpha = alpha
        super(LeakyReLU, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.relu(x, alpha=self.alpha)

    def get_config(self):
        config = {'alpha': self.alpha}
        base_config = super(LeakyReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PReLU(Layer):
    '''Parametric Rectified Linear Unit:
    `f(x) = alphas * x for x < 0`,
    `f(x) = x for x >= 0`,
    where `alphas` is a learned array with the same shape as x.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        init: initialization function for the weights.
        weights: initial weights, as a list of a single Numpy array.
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.

    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)
    '''
    def __init__(self, init='zero', weights=None, shared_axes=None, **kwargs):
        self.supports_masking = True
        self.init = initializations.get(init)
        self.initial_weights = weights
        if type(shared_axes) is not list and type(shared_axes) is not tuple:
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)
        super(PReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes[0] is not None:
            for i in self.shared_axes:
                param_shape[i] = 1
                self.param_broadcast[i] = True

        self.alphas = self.init(param_shape,
                                name='{}_alphas'.format(self.name))
        self.trainable_weights = [self.alphas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        pos = K.relu(x)
        if K.backend() == 'theano':
            neg = K.pattern_broadcast(self.alphas, self.param_broadcast) * (x - abs(x)) * 0.5
        else:
            neg = self.alphas * (x - abs(x)) * 0.5
        return pos + neg

    def get_config(self):
        config = {'init': self.init.__name__}
        base_config = super(PReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ELU(Layer):
    '''Exponential Linear Unit:
    `f(x) =  alpha * (exp(x) - 1.) for x < 0`,
    `f(x) = x for x >= 0`.

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
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)
        super(ELU, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.elu(x, self.alpha)

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(ELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ParametricSoftplus(Layer):
    '''Parametric Softplus:
    `alpha * log(1 + exp(beta * x))`

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
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.

    # References
        - [Inferring Nonlinear Neuronal Computation Based on Physiologically Plausible Inputs](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003143)
    '''
    def __init__(self, alpha_init=0.2, beta_init=5.0,
                 weights=None, shared_axes=None, **kwargs):
        self.supports_masking = True
        self.alpha_init = K.cast_to_floatx(alpha_init)
        self.beta_init = K.cast_to_floatx(beta_init)
        self.initial_weights = weights
        if type(shared_axes) is not list and type(shared_axes) is not tuple:
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)
        super(ParametricSoftplus, self).__init__(**kwargs)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes[0] is not None:
            for i in self.shared_axes:
                param_shape[i] = 1
                self.param_broadcast[i] = True

        self.alphas = K.variable(self.alpha_init * np.ones(param_shape),
                                 name='{}_alphas'.format(self.name))
        self.betas = K.variable(self.beta_init * np.ones(param_shape),
                                name='{}_betas'.format(self.name))
        self.trainable_weights = [self.alphas, self.betas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        if K.backend() == 'theano':
            return K.softplus(K.pattern_broadcast(self.betas, self.param_broadcast) * x) * K.pattern_broadcast(self.alphas, self.param_broadcast)
        else:
            return K.softplus(self.betas * x) * self.alphas

    def get_config(self):
        config = {'alpha_init': float(self.alpha_init),
                  'beta_init': float(self.beta_init)}
        base_config = super(ParametricSoftplus, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ThresholdedReLU(Layer):
    '''Thresholded Rectified Linear Unit:
    `f(x) = x for x > theta`
    `f(x) = 0 otherwise`.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        theta: float >= 0. Threshold location of activation.

    # References
        - [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/pdf/1402.3337.pdf)
    '''
    def __init__(self, theta=1.0, **kwargs):
        self.supports_masking = True
        self.theta = K.cast_to_floatx(theta)
        super(ThresholdedReLU, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x * K.cast(x > self.theta, K.floatx())

    def get_config(self):
        config = {'theta': float(self.theta)}
        base_config = super(ThresholdedReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SReLU(Layer):
    '''S-shaped Rectified Linear Unit.

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
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.

    # References
        - [Deep Learning with S-shaped Rectified Linear Activation Units](http://arxiv.org/abs/1512.07030)
    '''
    def __init__(self, t_left_init='zero', a_left_init='glorot_uniform',
                 t_right_init='glorot_uniform', a_right_init='one', shared_axes=None, **kwargs):
        self.supports_masking = True
        self.t_left_init = t_left_init
        self.a_left_init = a_left_init
        self.t_right_init = t_right_init
        self.a_right_init = a_right_init
        if type(shared_axes) is not list and type(shared_axes) is not tuple:
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)
        super(SReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes[0] is not None:
            for i in self.shared_axes:
                param_shape[i] = 1
                self.param_broadcast[i] = True

        t_left_init = initializations.get(self.t_left_init)
        a_left_init = initializations.get(self.a_left_init)
        t_right_init = initializations.get(self.t_right_init)
        a_right_init = initializations.get(self.a_right_init)

        self.t_left = t_left_init(param_shape,
                                  name='{}_t_left'.format(self.name))
        self.a_left = a_left_init(param_shape,
                                  name='{}_a_left'.format(self.name))
        self.t_right = t_right_init(param_shape,
                                    name='{}_t_right'.format(self.name))
        self.a_right = a_right_init(param_shape,
                                    name='{}_a_right'.format(self.name))
        # ensure the the right part is always to the right of the left
        self.t_right_actual = self.t_left + abs(self.t_right)
        self.trainable_weights = [self.t_left, self.a_left,
                                  self.t_right, self.a_right]

    def call(self, x, mask=None):
        if K.backend() == 'theano':
            t_left = K.pattern_broadcast(self.t_left, self.param_broadcast)
            a_left = K.pattern_broadcast(self.a_left, self.param_broadcast)
            a_right = K.pattern_broadcast(self.a_right, self.param_broadcast)
            t_right_actual = K.pattern_broadcast(self.t_right_actual, self.param_broadcast)
        else:
            t_left = self.t_left
            a_left = self.a_left
            a_right = self.a_right
            t_right_actual = self.t_right_actual

        Y_left_and_center = t_left + K.relu(x - t_left,
                                            a_left,
                                            t_right_actual - t_left)
        Y_right = K.relu(x - t_right_actual) * a_right
        return Y_left_and_center + Y_right

    def get_config(self):
        config = {'t_left_init': self.t_left_init,
                  'a_left_init': self.a_left_init,
                  't_right_init': self.t_right_init,
                  'a_right_init': self.a_right_init}
        base_config = super(SReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
