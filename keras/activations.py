import theano
import theano.tensor as T
import types

def softmax(x):
    return T.nnet.softmax(x)

def softplus(x):
    return T.nnet.softplus(x)

def relu(x):
    return (x + abs(x)) / 2.0

def tanh(x):
    return T.tanh(x)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)

def linear(x):
    return x

from utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'activation function')