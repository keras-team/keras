from ..layers.core import Layer
from .. import initializations
from .. import backend as K


class BatchNormalization(Layer):
    '''Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # Arguments
        epsilon: small float > 0. Fuzz parameter.
        mode: integer, 0 or 1.
            - 0: feature-wise normalization.
                If the input has multiple feature dimensions,
                each will be normalized separately
                (e.g. for an image input with shape
                `(channels, rows, cols)`,
                each combination of a channel, row and column
                will be normalized separately).
            - 1: sample-wise normalization. This mode assumes a 2D input.
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`

    # References
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/pdf/1502.03167v3.pdf)
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

        self.gamma = self.init(input_shape)
        self.beta = K.zeros(input_shape)

        self.params = [self.gamma, self.beta]
        self.running_mean = K.zeros(input_shape)
        self.running_std = K.ones(input_shape)

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
            m = K.mean(X, axis=0)
            std = K.mean(K.square(X - m) + self.epsilon, axis=0)
            std = K.sqrt(std)
            mean_update = self.momentum * self.running_mean + (1-self.momentum) * m
            std_update = self.momentum * self.running_std + (1-self.momentum) * std
            self.updates = [(self.running_mean, mean_update),
                            (self.running_std, std_update)]
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
