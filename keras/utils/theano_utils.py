from __future__ import absolute_import
import numpy as np
import theano
import theano.tensor as T

default_mask_val = -999

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



def get_mask(X, mask_val, steps_back=0):
    '''
        Given X, a tensor or matrix, return mask tension of same number of dimensions thus:
                    X.shape                     mask.shape
            -------------------------------------------------
            (time, samples, dimensions)  -> (time, samples, 1)
            (samples, dimensions)        -> (samples, 1)
        
        and, if steps_back>0, a padded_mask tensor that is left-padded with `steps_back` zeros
        along the time dimension.
        
        The mask has a 1 for every entry except for those corresponding to a vector in X that has every entry equal to mask_val.
    '''
    mask = T.neq(X, mask_val).sum(axis=-1) > 0 # (time, nb_samples) matrix with a 1 for every unmasked entry
    mask = T.shape_padright(mask)
    mask = T.addbroadcast(mask, -1) # (time, nb_samples, 1) matrix.

    if steps_back > 0:
        # left-pad in time with 0
        pad = alloc_zeros_matrix(steps_back, mask.shape[1], 1).astype('uint8')
        return mask, T.concatenate([pad, mask], axis=0)
    return mask

