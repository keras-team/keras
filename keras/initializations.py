import theano
import theano.tensor as T
import numpy as np

from utils.theano_utils import sharedX

def uniform(shape, scale=0.05):
    return sharedX(np.random.uniform(low=-scale, high=scale, size=shape))

def normal(shape, scale=0.05):
    return sharedX(np.random.randn(*shape) * scale)

def lecun_uniform(shape):
    ''' Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    '''
    m = 1
    for s in shape:
        m *= s
    scale = 1./np.sqrt(m)
    return uniform(shape, scale)

def glorot_normal(shape):
    ''' Reference: Glorot & Bengio, AISTATS 2010
    '''
    s = np.sqrt(2 / (shape[0] + shape[1]))
    return normal(shape, s)

def he_normal(shape):
    ''' Reference:  He et al., http://arxiv.org/abs/1502.01852
    '''
    s = np.sqrt(2 / (shape[1]))
    return normal(shape, s)

def orthogonal(shape, scale=1.1):
    ''' From Lasagne
    '''
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    return sharedX(scale * q[:shape[0], :shape[1]])


from utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'initialization')
