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

    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)
    '''
    def __init__(self, init='zero', weights=None, **kwargs):
        self.supports_masking = True
        self.init = initializations.get(init)
        self.initial_weights = weights
        super(PReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alphas = self.init(input_shape[1:],
                                name='{}_alphas'.format(self.name))
        self.trainable_weights = [self.alphas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        pos = K.relu(x)
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
        pos = K.relu(x)
        neg = (x - abs(x)) * 0.5
        return pos + self.alpha * (K.exp(neg) - 1.)

    def get_config(self):
        config = {'alpha': self.alpha}
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

    # References
        - [Inferring Nonlinear Neuronal Computation Based on Physiologically Plausible Inputs](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003143)
    '''
    def __init__(self, alpha_init=0.2, beta_init=5.0,
                 weights=None, **kwargs):
        self.supports_masking = True
        self.alpha_init = K.cast_to_floatx(alpha_init)
        self.beta_init = K.cast_to_floatx(beta_init)
        self.initial_weights = weights
        super(ParametricSoftplus, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape[1:]
        self.alphas = K.variable(self.alpha_init * np.ones(input_shape),
                                 name='{}_alphas'.format(self.name))
        self.betas = K.variable(self.beta_init * np.ones(input_shape),
                                name='{}_betas'.format(self.name))
        self.trainable_weights = [self.alphas, self.betas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        return K.softplus(self.betas * x) * self.alphas

    def get_config(self):
        config = {'alpha_init': self.alpha_init,
                  'beta_init': self.beta_init}
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
        config = {'theta': self.theta}
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

    # References
        - [Deep Learning with S-shaped Rectified Linear Activation Units](http://arxiv.org/abs/1512.07030)
    '''
    def __init__(self, t_left_init='zero', a_left_init='glorot_uniform',
                 t_right_init='glorot_uniform', a_right_init='one', **kwargs):
        self.supports_masking = True
        self.t_left_init = t_left_init
        self.a_left_init = a_left_init
        self.t_right_init = t_right_init
        self.a_right_init = a_right_init
        super(SReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape[1:]

        t_left_init = initializations.get(self.t_left_init)
        a_left_init = initializations.get(self.a_left_init)
        t_right_init = initializations.get(self.t_right_init)
        a_right_init = initializations.get(self.a_right_init)

        self.t_left = t_left_init(input_shape,
                                  name='{}_t_left'.format(self.name))
        self.a_left = a_left_init(input_shape,
                                  name='{}_a_left'.format(self.name))
        self.t_right = t_right_init(input_shape,
                                    name='{}_t_right'.format(self.name))
        self.a_right = a_right_init(input_shape,
                                    name='{}_a_right'.format(self.name))
        # ensure the the right part is always to the right of the left
        self.t_right_actual = self.t_left + abs(self.t_right)
        self.trainable_weights = [self.t_left, self.a_left,
                                  self.t_right, self.a_right]

    def call(self, x, mask=None):
        Y_left_and_center = self.t_left + K.relu(x - self.t_left,
                                                 self.a_left,
                                                 self.t_right_actual - self.t_left)
        Y_right = K.relu(x - self.t_right_actual) * self.a_right
        return Y_left_and_center + Y_right

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
        total_class: How many classes the hierarchical softmax is over
        per_top: How many classes per top level code (should be on the order of the square root of the total number of classes)

    # References
    - [Classes for Fast Maximum Entropy Training](http://arxiv.org/pdf/cs/0108006.pdf)
    - [Hierarchical Probabilistic Neural Network Language Model](http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf)
    - [Strategies for Training Large Vocabulary Neural Language Models](http://arxiv.org/pdf/1512.04906)

    '''
    def __init__(self, total_class, per_top = None, **kwargs):
        if per_top is None:
            per_top = int(np.ceil(np.sqrt(total_class)))
        self.per_top = per_top
        self.total_cass = total_class

        self.n_top_level = int(np.ceil(self.total_class * 1. / self.per_top))
        self.n_second_level = self.n_top_level * self.per_top
        
        self.class_to_all_top = np.reshape(np.array([item for sublist in
                                                     [[i for _ in range(self.per_top)]
                                                      for i in range(self.n_top_level)]
                                                     for item in sublist]),
                                           (self.n_second_level, 1))

        self.top_to_all_second = np.reshape(np.arange(self.n_second_level),
                                            (self.n_top_level, per_top))
        
        self.class_to_all_second = np.array([item for sublist in
                                             [range(per_top)
                                              for _ in range(self.n_second_level)]
                                             for item in sublist])

        super(HierarchicalSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):

        input_dim = input_shape[0][1]

        #Some matrices that provide required mappings into the weight matrices
        self.class_to_top = K.variable(self.class_to_all_top, dtype = np.int32)
        self.top_to_second = K.variable(self.top_to_all_second, dtype = np.int32)
        self.class_to_second = K.variable(self.class_to_all_second, dtype = np.int32)

        #A few identity matrices we need in memory for indexing purposes
        self.identity_constant_top = K.eye(self.n_top_level)
        self.identity_constant_second = K.eye(self.per_top)

        self.top_level_weights = K.variable(np.random.random((self.n_top_level,input_dim)))
        self.top_level_biases = K.zeros((self.n_top_level,))

        self.second_level_weights = K.variable(np.random.random((self.n_second_level,input_dim)))
        self.second_level_biases = K.zeros((self.n_second_level,))

        
        self.trainable_weights = [self.top_level_weights,
                                  self.top_level_biases,
                                  self.second_level_weights,
                                  self.second_level_biases]

        self.non_trainable_weights = [self.class_to_top,
                                      self.top_to_second,
                                      self.class_to_second,
                                      self.identity_constant_top,
                                      self.identity_constant_second]
        
        
    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception('HierarchicalSoftmax must be called on a list of two tensors, got: ' + str(inputs))
        input_vecs, labels = inputs

        #first: get the prob of the top level class for all inputs
        top_logprobs = K.dot(input_vecs, K.transpose(self.top_level_weights))
        top_logprobs += self.top_level_biases
        top_probs = K.softmax(top_logprobs)

        #second: for each label, record the probability of the correct top class
        batch_tops = K.gather(self.class_to_top, labels)
        top_index_mat = K.squeeze(K.squeeze(K.gather(self.identity_constant_top, batch_tops),1),1)
        batch_top_probs = K.max(K.minimum(top_probs, top_index_mat), 1)

        #third: get the (batch, per_second_level) matrix of indices, so we know which weights to use
        second_level_indices = K.squeeze(K.squeeze(K.gather(self.top_to_second, batch_tops),1),1)

        #fourth: compute the second level probabilities
        second_level_selected_weights = K.permute_dimensions(K.gather(self.second_level_weights,
                                                                      second_level_indices), (0,2,1))
        second_level_selected_biases = K.gather(self.second_level_biases,
                                       second_level_indices)

        second_logprobs = K.squeeze(K.batch_dot(K.expand_dims(input_vecs,1),
                                                second_level_selected_weights),1)
        second_logprobs += second_level_selected_biases
        second_probs = K.softmax(second_logprobs)

        #finally, select the columns of the correct labels
        second_index_mat = K.squeeze(K.gather(self.identity_constant_second,
                                              K.gather(self.class_to_second, labels)),1)
        batch_second_probs = K.max(K.minimum(second_probs, second_index_mat), 1)

        #now we can output the negative log probability of the correct class for each input
        batch_top_probs = -K.log(batch_top_probs)
        batch_second_probs = -K.log(batch_second_probs)
        return batch_top_probs + batch_second_probs
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0],1)

    def get_config(self):
        config = {'total_class' = self.total_class
                  'per_top' = self.per_top}
        base_config = super(HierarchicalSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
