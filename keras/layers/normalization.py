from ..layers.core import Layer
from ..utils.theano_utils import shared_zeros
from .. import initializations

class BatchNormalization(Layer):
    '''
        Reference: 
            Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
                http://arxiv.org/pdf/1502.03167v3.pdf
    '''
    def __init__(self, input_shape, epsilon=1e-6, weights=None):
        self.init = initializations.get("uniform")
        self.input_shape = input_shape
        self.epsilon = epsilon

        self.gamma = self.init((self.input_shape))
        self.beta = shared_zeros(self.input_shape)

        self.params = [self.gamma, self.beta]
        if weights is not None:
            self.set_weights(weights)

    def output(self, train):
        X = self.get_input(train)
        X_normed = (X - X.mean(keepdims=True)) / (X.std(keepdims=True) + self.epsilon)
        out = self.gamma * X_normed + self.beta
        return out