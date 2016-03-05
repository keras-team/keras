from keras import backend as K
from keras.layers.core import Layer
from keras import initializations, activations

from keras.regularizers import GaussianKL


class VariationalDense(Layer):
    """VariationalDense
    Hidden layer for Variational Autoencoding Bayes method - Kigma and
    Welling 2013.
    This layer projects the input twice to calculate the mean and variance
    of a Gaussian distribution. During training, the output is sampled from
    that distribution as mean + random_noise * variance, during testing the
    output is the mean, i.e the expected value of the encoded distribution.

    Parameters:
    -----------
    output_dim: int, output dimension
    batch_size: Tensorflow backend need the batch_size to be defined
        for sampling random numbers. Make sure your batch size is kept
        fixed during training. We infer the batch size for Theano backend.
        You can use any batch size for testing.
    regularizer_scale: By default the regularization is already properly
        scaled if you use binary or categorical crossentropy cost functions.
        In most cases this regularizers should be kept fixed at one.

    Notes:
    ------
    During training we output ONE sample from the reparameterized
    distribution given by "mean + std * noise".
    For testing, the output of this layer is deterministic given by "mean".

    References:
    -----------
    - [Auto-Encoding Variational Bayes](http://arxiv.org/pdf/1312.6114v10.pdf)

    """
    def __init__(self, output_dim, batch_size, init='glorot_uniform',
                 activation='tanh',
                 weights=None, input_dim=None, regularizer_scale=1,
                 prior_mean=0, prior_logsigma=1, **kwargs):
        self.prior_mean = prior_mean
        self.prior_logsigma = prior_logsigma
        self.regularizer_scale = regularizer_scale
        self.batch_size = batch_size
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.initial_weights = weights
        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        self.input = K.placeholder(ndim=2)
        super(VariationalDense, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[-1]

        self.W_mean = self.init((input_dim, self.output_dim))
        self.b_mean = K.zeros((self.output_dim,))
        self.W_logsigma = self.init((input_dim, self.output_dim))
        self.b_logsigma = K.zeros((self.output_dim,))

        self.trainable_weights = [self.W_mean, self.b_mean, self.W_logsigma,
                                  self.b_logsigma]

        self.regularizers = []
        reg = self.get_variational_regularization(self.get_input())
        self.regularizers.append(reg)

    def get_variational_regularization(self, X):
        mean = self.activation(K.dot(X, self.W_mean) + self.b_mean)
        logsigma = self.activation(K.dot(X, self.W_logsigma) + self.b_logsigma)
        return GaussianKL(mean, logsigma,
                          regularizer_scale=self.regularizer_scale,
                          prior_mean=self.prior_mean,
                          prior_logsigma=self.prior_logsigma)

    def get_mean_logsigma(self, X):
        mean = self.activation(K.dot(X, self.W_mean) + self.b_mean)
        logsigma = self.activation(K.dot(X, self.W_logsigma) + self.b_logsigma)
        return mean, logsigma

    def _get_output(self, X, train=False):
        mean, logsigma = self.get_mean_logsigma(X)
        if train:
            # infer batch size for theano backend
            if K._BACKEND == 'theano':
                eps = K.random_normal((X.shape[0], self.output_dim))
            else:
                eps = K.random_normal((self.batch_size, self.output_dim))
            # return sample : mean + std * noise
            return mean + K.exp(logsigma) * eps
        else:
            # for testing, the sample is deterministic
            return mean

    def get_output(self, train=False):
        X = self.get_input()
        return self._get_output(X, train)

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)
