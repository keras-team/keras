from __future__ import absolute_import
import theano.tensor as T


def softmax(x):
    return T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape)


def time_distributed_softmax(x):
    import warnings
    warnings.warn("time_distributed_softmax is deprecated. Just use softmax!", DeprecationWarning)
    return softmax(x)


def softplus(x):
    return T.nnet.softplus(x)


def relu(x):
    return T.nnet.relu(x)


def elu(x, alpha=1.):
    '''
    FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)
    http://arxiv.org/pdf/1511.07289v1.pdf
    '''
    return T.switch(T.gt(x, 0.), x, alpha*(T.exp(x)-1))


def tanh(x):
    return T.tanh(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)


def linear(x):
    '''
    The function returns the variable that is passed in, so all types work
    '''
    return x


from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'activation function')
