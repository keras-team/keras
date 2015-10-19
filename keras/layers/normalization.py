from ..layers.core import Layer
from ..utils.theano_utils import shared_zeros, shared_ones, ndim_tensor, floatX, shared_scalar
from .. import initializations

import theano.tensor as T
import theano
import numpy as np


class BatchNormalization(Layer):
    '''
        Reference:
            Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
                http://arxiv.org/pdf/1502.03167v3.pdf

            mode: 0 -> featurewise normalization
                  1 -> samplewise normalization (may sometimes outperform featurewise mode)

            momentum: momentum term in the computation of a running estimate of the mean and std of the data

        Adapted spatial batch normalization from
        https://github.com/takacsg84/Lasagne/blob/d5545988e6484d1db4bb54bcfa541ba62e898829/lasagne/layers/bn2.py
    '''
    def __init__(self, input_shape, beta_init='zero', gamma_init='uniform', epsilon=1e-6, mode=0, momentum=0.9, weights=None):
        super(BatchNormalization, self).__init__()
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.mode = mode
        self.momentum = momentum
        self.input = ndim_tensor(len(self.input_shape) + 1)

        if len(input_shape) == 2:       # in case of dense layer
            self.axis = (0)
            param_shape = (input_shape[-1])
            self.gamma = self.gamma_init(param_shape)
            self.beta = self.beta_init(param_shape)
            ema_shape = (1, input_shape[-1])
            ema_bc = (True, False)
        elif len(input_shape) == 4:     # in case of conv2d layer
            self.axis = (0, 2, 3)
            param_shape = (1, input_shape[1], 1, 1)

            # it has to be made broadcastable on the first axis
            uniform = np.random.uniform(low=-0.05, high=0.05, size=param_shape)
            self.gamma = theano.shared(uniform.astype(theano.config.floatX),
                                       broadcastable=(True, False, True, True),
                                       borrow=True)
            self.beta = theano.shared(np.zeros(param_shape, dtype=theano.config.floatX),
                                      broadcastable=(True, False, True, True),
                                      borrow=True)

            ema_shape = (1, input_shape[1], 1, 1)
            ema_bc = (True, False, True, True)
        else:
            raise NotImplementedError

        self.params = [self.gamma, self.beta]

        self.running_mean = theano.shared(
            np.zeros(ema_shape, dtype=theano.config.floatX),
            borrow=True, broadcastable=ema_bc)

        self.running_std = theano.shared(
            np.ones(ema_shape, dtype=theano.config.floatX),
            borrow=True, broadcastable=ema_bc)

        self.batch_cnt = shared_scalar(0)

        if weights is not None:
            self.set_weights(weights)

    def get_weights(self):
        return super(BatchNormalization, self).get_weights() + \
               [self.running_mean.get_value(), self.running_std.get_value()]

    def set_weights(self, weights):
        self.running_mean.set_value(floatX(weights[-2]))
        self.running_std.set_value(floatX(weights[-1]))
        super(BatchNormalization, self).set_weights(weights[:-2])

    def init_updates(self):
        X = self.get_input(train=True)

        m = T.mean(X, self.axis, keepdims=True)
        std = T.sqrt(T.var(X, self.axis, keepdims=True) + self.epsilon)
        mean_update = self.momentum * self.running_mean + (1-self.momentum) * m
        std_update = self.momentum * self.running_std + (1-self.momentum) * std
        self.updates = [(self.running_mean, mean_update),
                        (self.running_std, std_update),
                        (self.batch_cnt, self.batch_cnt + 1)]

    def get_output(self, train):
        X = self.get_input(train)

        if self.mode == 0:
            m = T.mean(X, self.axis, keepdims=True)
            std = T.sqrt(T.var(X, self.axis, keepdims=True) + self.epsilon)
            X_normed = (X - m) / (std + self.epsilon)
            out = self.gamma * X_normed + self.beta
        elif self.mode == 1:
            out = (X - self.running_mean) / (self.running_std + self.epsilon)
        return out

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

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__()
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
        return {"name": self.__class__.__name__,
                "alpha": self.alpha,
                "k": self.k,
                "beta": self.beta,
                "n": self.n}
