from ..layers.core import Layer
from .. import initializations
from .. import backend as K


class BatchNormalization(Layer):
    '''
        Reference:
            Batch Normalization: Accelerating Deep Network Training
            by Reducing Internal Covariate Shift
                http://arxiv.org/pdf/1502.03167v3.pdf

            mode: 0 -> featurewise normalization
                  1 -> samplewise normalization
                       (may sometimes outperform featurewise mode)

            momentum: momentum term in the computation
            of a running estimate of the mean and std of the data
    '''
    def __init__(self, epsilon=1e-6, mode=0, momentum=0.9,
                 weights=None, **kwargs):
        self.init = initializations.get("uniform")
        self.epsilon = epsilon
        self.mode = mode
        self.momentum = momentum
        self.initial_weights = weights
        super(BatchNormalization, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape  # starts with samples axis
        input_shape = input_shape[1:]

        self.gamma = self.init((input_shape))
        self.beta = K.zeros(input_shape)

        self.params = [self.gamma, self.beta]
        self.running_mean = K.zeros(input_shape)
        self.running_std = K.ones((input_shape))

        # initialize self.updates: batch mean/std computation
        X = self.get_input(train=True)
        m = K.mean(X, axis=0)
        std = K.mean(K.square(X - m) + self.epsilon, axis=0)
        std = K.sqrt(std)
        mean_update = self.momentum * self.running_mean + (1-self.momentum) * m
        std_update = self.momentum * self.running_std + (1-self.momentum) * std
        self.updates = [(self.running_mean, mean_update),
                        (self.running_std, std_update)]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_weights(self):
        super_weights = super(BatchNormalization, self).get_weights()
        return super_weights + [K.get_value(self.running_mean),
                                K.get_value(self.running_std)]

    def set_weights(self, weights):
        K.set_value(self.running_mean, weights[-2])
        K.set_value(self.running_std, weights[-1])
        super(BatchNormalization, self).set_weights(weights[:-2])

    def get_output(self, train):
        X = self.get_input(train)
        if self.mode == 0:
            X_normed = ((X - self.running_mean) /
                        (self.running_std + self.epsilon))
        elif self.mode == 1:
            m = K.mean(X, axis=-1, keepdims=True)
            std = K.std(X, axis=-1, keepdims=True)
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
