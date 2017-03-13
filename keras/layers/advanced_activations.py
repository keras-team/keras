from .. import initializations
from ..engine import Layer
from .. import backend as K
import numpy as np


class LeakyReLU(Layer):
    """Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active:
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

    # References
        - [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
    """

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
    """Parametric Rectified Linear Unit.

    It follows:
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
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
    """

    def __init__(self, init='zero', weights=None, shared_axes=None, **kwargs):
        self.supports_masking = True
        self.init = initializations.get(init)
        self.initial_weights = weights
        if not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)
        super(PReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes[0] is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True

        self.alphas = self.init(param_shape,
                                name='{}_alphas'.format(self.name))
        self.trainable_weights = [self.alphas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        pos = K.relu(x)
        if K.backend() == 'theano':
            neg = (K.pattern_broadcast(self.alphas, self.param_broadcast) *
                   (x - K.abs(x)) * 0.5)
        else:
            neg = self.alphas * (x - K.abs(x)) * 0.5
        return pos + neg

    def get_config(self):
        config = {'init': self.init.__name__}
        base_config = super(PReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ELU(Layer):
    """Exponential Linear Unit.

    It follows:
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
        - [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289v1)
    """

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
    """Parametric Softplus.

    It follows:
    `f(x) = alpha * log(1 + exp(beta * x))`

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
    """

    def __init__(self, alpha_init=0.2, beta_init=5.0,
                 weights=None, shared_axes=None, **kwargs):
        self.supports_masking = True
        self.alpha_init = K.cast_to_floatx(alpha_init)
        self.beta_init = K.cast_to_floatx(beta_init)
        self.initial_weights = weights
        if not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)
        super(ParametricSoftplus, self).__init__(**kwargs)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes[0] is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True

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
            return (K.softplus(K.pattern_broadcast(self.betas,
                                                   self.param_broadcast) * x) *
                    K.pattern_broadcast(self.alphas, self.param_broadcast))
        else:
            return K.softplus(self.betas * x) * self.alphas

    def get_config(self):
        config = {'alpha_init': float(self.alpha_init),
                  'beta_init': float(self.beta_init)}
        base_config = super(ParametricSoftplus, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ThresholdedReLU(Layer):
    """Thresholded Rectified Linear Unit.

    It follows:
    `f(x) = x for x > theta`,
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
        - [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/abs/1402.3337)
    """

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
    """S-shaped Rectified Linear Unit.

    It follows:
    `f(x) = t^r + a^r(x - t^r) for x >= t^r`,
    `f(x) = x for t^r > x > t^l`,
    `f(x) = t^l + a^l(x - t^l) for x <= t^l`.

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
    """

    def __init__(self, t_left_init='zero', a_left_init='glorot_uniform',
                 t_right_init='glorot_uniform', a_right_init='one',
                 shared_axes=None, **kwargs):
        self.supports_masking = True
        self.t_left_init = t_left_init
        self.a_left_init = a_left_init
        self.t_right_init = t_right_init
        self.a_right_init = a_right_init
        if not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)
        super(SReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes[0] is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True

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
        self.t_right_actual = self.t_left + K.abs(self.t_right)
        self.trainable_weights = [self.t_left, self.a_left,
                                  self.t_right, self.a_right]

    def call(self, x, mask=None):
        if K.backend() == 'theano':
            t_left = K.pattern_broadcast(self.t_left, self.param_broadcast)
            a_left = K.pattern_broadcast(self.a_left, self.param_broadcast)
            a_right = K.pattern_broadcast(self.a_right, self.param_broadcast)
            t_right_actual = K.pattern_broadcast(self.t_right_actual,
                                                 self.param_broadcast)
        else:
            t_left = self.t_left
            a_left = self.a_left
            a_right = self.a_right
            t_right_actual = self.t_right_actual

        y_left_and_center = t_left + K.relu(x - t_left,
                                            a_left,
                                            t_right_actual - t_left)
        y_right = K.relu(x - t_right_actual) * a_right
        return y_left_and_center + y_right

    def get_config(self):
        config = {'t_left_init': self.t_left_init,
                  'a_left_init': self.a_left_init,
                  't_right_init': self.t_right_init,
                  'a_right_init': self.a_right_init}
        base_config = super(SReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class HierarchicalSoftmax(Layer):
    '''Two-layer Hierarchical Softmax layer. Provides an approximate
    softmax that is much faster to compute in cases where there are a
    large number (~10K+) of classes.

    # Input shape
        A list of two tensors:
           - The first tensor should have shape (nb_samples, dim) and represents the input feature vector
           - The second tensor should have shape (nb_samples,), have integer type, and represent the
           labels of each training example.

    # Output shape
        1D Tensor with shape (nb_samples,) representing the negative log probability of the correct class

    # Arguments
        total_outputs: How many outputs the hierarchical softmax is over
        per_class: How many outputs per top level class (should be on the order of the square root of the total number of classes)

    # References
    - [Classes for Fast Maximum Entropy Training](http://arxiv.org/pdf/cs/0108006.pdf)
    - [Hierarchical Probabilistic Neural Network Language Model](http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf)
    - [Strategies for Training Large Vocabulary Neural Language Models](http://arxiv.org/pdf/1512.04906)

    '''
    def __init__(self, total_outputs, per_class = None,
                 top_weights_init = 'uniform', top_bias_init = 'zero',
                 bottom_weights_init = 'uniform', bottom_bias_init = 'zero',
                 **kwargs):
        assert K.backend() == 'theano', "HierarchicalSoftmax only supported by Theano"
        
        if per_class is None:
            per_class = int(np.ceil(np.sqrt(total_outputs)))

        #naming convention:
        #the first layer maps inputs to classes
        #the second layer maps classes to outputs
            
        self.total_outputs = total_outputs
        self.per_class = per_class
        
        self.n_classes = int(np.ceil(self.total_outputs * 1. / self.per_class))
        self.n_outputs_actual = self.n_classes * self.per_class

        self.top_weights_init = initializations.get(top_weights_init)
        self.top_bias_init = initializations.get(top_bias_init)

        self.bottom_weights_init = initializations.get(bottom_weights_init)
        self.bottom_bias_init = initializations.get(bottom_bias_init)
        
        assert self.n_outputs_actual >= self.total_outputs, "The number of actual HSM outputs must be at least the number of outputs you're modeling over."
        super(HierarchicalSoftmax, self).__init__(**kwargs)
        
    def build(self, input_shape):
        input_dim = input_shape[0][1]

        self.top_weights = self.top_weights_init((input_dim, self.n_classes,),
                                                 name='{}_top_weights'.format(self.name))

        self.top_bias = self.top_bias_init((self.n_classes,),
                                           name='{}_top_bias'.format(self.name))

        self.bottom_weights = self.bottom_weights_init((self.n_classes, input_dim, self.per_class,),
                                                       name='{}_bottom_weights'.format(self.name))

        self.bottom_bias = self.top_bias_init((self.n_classes, self.per_class,),
                                              name='{}_top_bias'.format(self.name))
        
        self.trainable_weights = [self.top_weights,
                                  self.top_bias,
                                  self.bottom_weights,
                                  self.bottom_bias]

        
    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception('HierarchicalSoftmax must be called on a list of two tensors, got: ' + str(inputs))
        input_vecs, labels = inputs
        return -K.hierarchical_softmax(input_vecs, input_vecs.shape[0],
                                       self.total_outputs, self.n_classes, self.per_class,
                                       self.top_weights, self.top_bias,
                                       self.bottom_weights, self.bottom_bias,
                                       labels)
                                      
    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0],1)

    def get_config(self):
        config = {'total_class': self.total_class,
                  'per_top': self.per_top}
        base_config = super(HierarchicalSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
