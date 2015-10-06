from ..layers.core import Layer
from ..utils.theano_utils import shared_zeros, shared_ones, ndim_tensor, floatX
from .. import initializations

import theano.tensor as T
import theano


class BatchNormalization(Layer):
    '''
        Reference:
            Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
                http://arxiv.org/pdf/1502.03167v3.pdf

            mode: 0 -> featurewise normalization
                  1 -> samplewise normalization (may sometimes outperform featurewise mode)

            momentum: momentum term in the computation of a running estimate of the mean and std of the data
    '''
    def __init__(self, epsilon=1e-6, mode=0, momentum=0.9, weights=None, **kwargs):
        self.init = initializations.get("uniform")
        self.epsilon = epsilon
        self.mode = mode
        self.momentum = momentum
        self.initial_weights = weights
        super(BatchNormalization, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape  # starts with samples axis
        input_shape = input_shape[1:]
        self.input = ndim_tensor(len(input_shape) + 1)

        self.gamma = self.init((input_shape))
        self.beta = shared_zeros(input_shape)

        self.params = [self.gamma, self.beta]
        self.running_mean = shared_zeros(input_shape)
        self.running_std = shared_ones((input_shape))

        # initialize self.updates: batch mean/std computation
        X = self.get_input(train=True)
        m = X.mean(axis=0)
        std = T.mean((X - m) ** 2 + self.epsilon, axis=0) ** 0.5
        mean_update = self.momentum * self.running_mean + (1-self.momentum) * m
        std_update = self.momentum * self.running_std + (1-self.momentum) * std
        self.updates = [(self.running_mean, mean_update), (self.running_std, std_update)]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_weights(self):
        return super(BatchNormalization, self).get_weights() + [self.running_mean.get_value(), self.running_std.get_value()]

    def set_weights(self, weights):
        self.running_mean.set_value(floatX(weights[-2]))
        self.running_std.set_value(floatX(weights[-1]))
        super(BatchNormalization, self).set_weights(weights[:-2])

    def get_output(self, train):
        X = self.get_input(train)

        if self.mode == 0:
            X_normed = (X - self.running_mean) / (self.running_std + self.epsilon)

        elif self.mode == 1:
            m = X.mean(axis=-1, keepdims=True)
            std = X.std(axis=-1, keepdims=True)
            X_normed = (X - m) / (std + self.epsilon)

        out = self.gamma * X_normed + self.beta
        return out

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "epsilon": self.epsilon,
                  "mode": self.mode,
                  "momentum": self.momentum}
        base_config = super(BatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SpatialBatchNormalization(Layer):
    '''
        Reference:
            Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
                http://arxiv.org/pdf/1502.03167v3.pdf

            mode: 0 -> featurewise normalization
                  1 -> samplewise normalization (may sometimes outperform featurewise mode)

            momentum: momentum term in the computation of a running estimate of the mean and std of the data

        Adapted from https://github.com/torch/nn/blob/master/SpatialBatchNormalization.lua
    '''
    def __init__(self, input_shape, epsilon=1e-6, mode=0, momentum=0.9, weights=None):
        super(SpatialBatchNormalization, self).__init__()
        self.init = initializations.get("uniform")
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.mode = mode
        self.momentum = momentum
        self.input = ndim_tensor(len(self.input_shape) + 1)

        self.gamma = self.init((1, self.input_shape[1], 1, 1))
        self.beta = shared_zeros((1, self.input_shape[1], 1, 1))

        self.params = [self.gamma, self.beta]
        self.running_mean = shared_zeros((1, self.input_shape[1], 1, 1))
        self.running_std = shared_ones((1, self.input_shape[1], 1, 1))
        if weights is not None:
            self.set_weights(weights)

    def get_weights(self):
        return super(SpatialBatchNormalization, self).get_weights() + [self.running_mean.get_value(), self.running_std.get_value()]

    def set_weights(self, weights):
        self.running_mean.set_value(floatX(weights[-2]))
        self.running_std.set_value(floatX(weights[-1]))
        super(SpatialBatchNormalization, self).set_weights(weights[:-2])

    def init_updates(self):
        X = self.get_input(train=True)

        mean_per_feature = self.compute_mean(X)
        mean_update = self.momentum * self.running_mean + (1-self.momentum) * mean_per_feature

        _, std_per_feature = self.compute_std(X, mean_per_feature)
        std_update = self.momentum * self.running_std + (1-self.momentum) * std_per_feature
        self.updates = [(self.running_mean, mean_update), (self.running_std, std_update)]

    def get_output(self, train):
        X = self.get_input(train)
        nb_batches, nb_features, x_height, x_width = X.shape

        if train:
            mean_per_feature = self.compute_mean(X)
            X_centered, std_per_feature = self.compute_std(X, mean_per_feature)

            X_normed = X_centered / (std_per_feature + self.epsilon)

            gamma_out = T.alloc(0., nb_batches, nb_features, x_height, x_width)
            gamma_out = T.set_subtensor(gamma_out[:, :, :, :], self.gamma)

            beta_out = T.alloc(0., nb_batches, nb_features, x_height, x_width)
            beta_out = T.set_subtensor(beta_out[:, :, :, :], self.beta)

            out = gamma_out * X_normed + beta_out
        else:
            mean_repeated = T.alloc(0., nb_batches, nb_features, x_height, x_width)
            mean_repeated = T.set_subtensor(mean_repeated[:, :, :, :], self.running_mean)
            X_centered = X - mean_repeated

            out = X_centered / (self.running_std + self.epsilon)
        return out

    def compute_mean(self, X):
        """
        :return mean_per_feature with shape (1, nb_features, 1, 1)
        """
        nb_batches, nb_features, x_height, x_width = T.shape(X)

        # calculate mean over mini-batch, over feature-maps
        X_reshaped = T.reshape(X, (nb_batches, nb_features, x_height * x_width))

        mean_batch = T.mean(X_reshaped, axis=0, keepdims=True)
        mean_per_feature = T.mean(mean_batch, axis=2)
        mean_per_feature = T.reshape(mean_per_feature, (1, nb_features, 1, 1))

        return mean_per_feature

    def compute_std(self, X, mean_per_feature):
        """
        :return X_centered, std

        std with shape (1, nb_features, 1, 1)
        """
        nb_batches, nb_features, x_height, x_width = X.shape

        mean_repeated = T.alloc(0., nb_batches, nb_features, x_height, x_width)
        mean_repeated = T.set_subtensor(mean_repeated[:, :, :, :], mean_per_feature)

        X_centered = X - mean_repeated
        X_centered_reshaped = T.reshape(X_centered, (nb_batches, nb_features, x_height * x_width))

        std = T.mean(T.mean(X_centered_reshaped ** 2, axis=0, keepdims=True) + self.epsilon, axis=2) ** 0.5
        std = T.reshape(std, (1, nb_features, 1, 1))

        return X_centered, std

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_shape": self.input_shape,
                "epsilon": self.epsilon,
                "mode": self.mode}


class LRN2D(Layer):
    """
    This code is adapted from pylearn2.
    License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
    """

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = X.shape
        half_n = self.n // 2
        input_sqr = T.sqr(X)
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :], input_sqr)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
