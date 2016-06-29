from __future__ import absolute_import
from ..engine import Layer
from .. import backend as K


class GaussianNoise(Layer):
    '''Apply to the input an additive zero-centered Gaussian noise with
    standard deviation `sigma`. This is useful to mitigate overfitting
    (you could see it as a kind of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.

    As it is a regularization layer, it is only active at training time.

    # Arguments
        sigma: float, standard deviation of the noise distribution.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    '''
    def __init__(self, sigma, **kwargs):
        self.supports_masking = True
        self.sigma = sigma
        self.uses_learning_phase = True
        super(GaussianNoise, self).__init__(**kwargs)

    def call(self, x, mask=None):
        noise_x = x + K.random_normal(shape=K.shape(x),
                                      mean=0.,
                                      std=self.sigma)
        return K.in_train_phase(noise_x, x)

    def get_config(self):
        config = {'sigma': self.sigma}
        base_config = super(GaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GaussianDropout(Layer):
    '''Apply to the input an multiplicative one-centered Gaussian noise
    with standard deviation `sqrt(p/(1-p))`.

    As it is a regularization layer, it is only active at training time.

    # Arguments
        p: float, drop probability (as with `Dropout`).

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    '''
    def __init__(self, p, **kwargs):
        self.supports_masking = True
        self.p = p
        if 0 < p < 1:
            self.uses_learning_phase = True
        super(GaussianDropout, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if 0 < self.p < 1:
            noise_x = x * K.random_normal(shape=K.shape(x), mean=1.0,
                                          std=K.sqrt(self.p / (1.0 - self.p)))
            return K.in_train_phase(noise_x, x)
        return x

    def get_config(self):
        config = {'p': self.p}
        base_config = super(GaussianDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
