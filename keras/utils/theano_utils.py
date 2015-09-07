from __future__ import absolute_import
import numpy as np
import theano
import theano.tensor as T


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def shared_zeros(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def shared_scalar(val=0., dtype=theano.config.floatX, name=None):
    return theano.shared(np.cast[dtype](val))


def shared_ones(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.ones(shape), dtype=dtype, name=name)


def alloc_zeros_matrix(*dims):
    return T.alloc(np.cast[theano.config.floatX](0.), *dims)


def ndim_tensor(ndim):
    if ndim == 1:
        return T.vector()
    elif ndim == 2:
        return T.matrix()
    elif ndim == 3:
        return T.tensor3()
    elif ndim == 4:
        return T.tensor4()
    return T.matrix()


def on_gpu():
    return theano.config.device[:3] == 'gpu'
