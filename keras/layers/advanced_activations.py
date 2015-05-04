from ..layers.core import Layer
from ..utils.theano_utils import shared_zeros

class LeakyReLU(Layer):
    def __init__(self, alpha=0.3):
        super(LeakyReLU,self).__init__()
        self.alpha = alpha

    def output(self, train):
        X = self.get_input(train)
        return ((X + abs(X)) / 2.0) + self.alpha * ((X - abs(X)) / 2.0)

    def get_config(self):
        return {"name":self.__class__.__name__,
            "alpha":self.alpha}


class PReLU(Layer):
    '''
        Reference: 
            Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
                http://arxiv.org/pdf/1502.01852v1.pdf
    '''
    def __init__(self, input_shape):
        super(PReLU,self).__init__()
        self.alphas = shared_zeros(input_shape)
        self.params = [self.alphas]
        self.input_shape = input_shape

    def output(self, train):
        X = self.get_input(train)
        pos = ((X + abs(X)) / 2.0)
        neg = self.alphas * ((X - abs(X)) / 2.0)
        return pos + neg

    def get_config(self):
        return {"name":self.__class__.__name__,
        "input_shape":self.input_shape}
