from ..layers.core import Layer
from ..utils.theano_utils import shared_zeros
from .. import initializations

import theano.tensor as T

class BatchNormalization(Layer):
    '''
        Reference: 
            Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
                http://arxiv.org/pdf/1502.03167v3.pdf

            mode: 0 -> featurewise normalization
                  1 -> samplewise normalization (may sometimes outperform featurewise mode)
    '''
    def __init__(self, input_shape, epsilon=1e-6, mode=0, weights=None):
        super(BatchNormalization,self).__init__()
        self.init = initializations.get("uniform")
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.mode = mode

        self.gamma = self.init((self.input_shape))
        self.beta = shared_zeros(self.input_shape)

        self.params = [self.gamma, self.beta]
        if weights is not None:
            self.set_weights(weights)

    def output(self, train):
        X = self.get_input(train)

        if self.mode == 0:
            m = X.mean(axis=0)
            # manual computation of std to prevent NaNs
            std = T.mean((X-m)**2 + self.epsilon, axis=0) ** 0.5
            X_normed = (X - m) / (std + self.epsilon)

        elif self.mode == 1:
            m = X.mean(axis=-1, keepdims=True)
            std = X.std(axis=-1, keepdims=True)
            X_normed = (X - m) / (std + self.epsilon)

        out = self.gamma * X_normed + self.beta
        return out

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_shape":self.input_shape,
            "epsilon":self.epsilon,
            "mode":self.mode}
