from ..layers.core import Layer
from ..utils.theano_utils import shared_zeros

class LeakyReLU(Layer):
    def __init__(self, alpha=0.3, name=None, prev=None, input_dim=(None,)):
        super(LeakyReLU,self).__init__(name, prev, input_dim)
        self.alpha = alpha

    def get_output(self, train):
        X = self.get_input(train)
        return ((X + abs(X)) / 2.0) + self.alpha * ((X - abs(X)) / 2.0)

    def get_output_dim(self, input_dim):
        return input_dim

    def get_config(self):
        return {"name":self.__class__.__name__,
            "alpha":self.alpha}


class PReLU(Layer):
    '''
        Reference:
            Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
                http://arxiv.org/pdf/1502.01852v1.pdf
    '''
    def __init__(self, input_shape=None, name=None, prev=None, input_dim=(None,)):
        super(PReLU,self).__init__(name, prev, input_dim)
        self.input_shape = input_shape

    def setup(self):
        if self.input_shape is None:
            if self.input_dim == (None, ):
                raise RuntimeError
            else:
                self.input_shape = self.input_dim
        self.alphas = shared_zeros(self.input_shape)
        self.params = [self.alphas]


    def get_output(self, train):
        X = self.get_input(train)
        pos = ((X + abs(X)) / 2.0)
        neg = self.alphas * ((X - abs(X)) / 2.0)
        return pos + neg

    def get_output_dim(self, input_dim):
        return input_dim

    def get_config(self):
        return {"name":self.__class__.__name__,
        "input_dim":self.input_dim}
