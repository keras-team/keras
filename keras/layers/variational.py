import numpy as np
from keras import backend as K
from keras.layers.core import Layer
from keras import initializations, activations

from keras.regularizers import GaussianKL


class VariationalDense(Layer):
    """VariationalDense
    Hidden layer for Variational Autoencoding Bayes method - Kigma and
    Welling 2013.
    Given and input `X=self.get_input()`, we calculate a mean and a log standard
    deviation as:

    ``` python
    mean = K.dot(X, W_mean) + b_mean
    logsigma = K.dot(X, W_logsigma) + b_logsigma
    ```

    and we sample Gaussian random variable as:

    ```python
    z = mean + K.exp(logsigma) * K.random_normal
    ```

    Thus, we reparameterize the distribution of the output of this layer to be
    Gaussian, this is called `reparametrization trick`. Further, as
    required by the Variational Autoencoder method, we
    impoose a regularizer on that Gaussian distribution to be as close as
    possible to a given prior Gaussian defined by mean_prior and logsigma_prior.
    They are both zero by default, which equals to choosing the prior to be zero
    mean and unit standard deviation.

    Parameters:
    -----------
    output_dim: int, output dimension
    prior_mean: float (default 0), mean of the prior Gaussian distribution
    prior_logsigma: float (default 0), logarithm of the standard deviation of
        the prior Gaussian distributions
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
    def __init__(self, output_dim, init='glorot_uniform',
                 activation='tanh',
                 weights=None, input_dim=None, regularizer_scale=1,
                 prior_mean=0, prior_logsigma=0, **kwargs):
        self.prior_mean = prior_mean
        self.prior_logsigma = prior_logsigma
        self.regularizer_scale = regularizer_scale
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

        self.W_mean = self.init((input_dim, self.output_dim),
                                name='{}_W_mean'.format(self.name))
        self.b_mean = K.zeros((self.output_dim,), name='{}_b_mean'.format(self.name))
        self.W_logsigma = self.init((input_dim, self.output_dim),
                                    name='{}_W_logsigma'.format(self.name))
        self.b_logsigma = K.zeros((self.output_dim,), name='{}_b_logsigma'.format(self.name))

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
                sizes = K.concatenate([K.shape(X)[0:1],
                                       np.asarray([self.output_dim, ])])
                eps = K.random_normal(sizes)
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
