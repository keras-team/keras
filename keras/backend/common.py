import numpy as np

from collections import defaultdict

# the type of float to use throughout the session.
_FLOATX = 'float32'
_EPSILON = 10e-8
_UID_PREFIXES = defaultdict(int)
_IMAGE_DIM_ORDERING = 'th'


def epsilon():
    '''Returns the value of the fuzz
    factor used in numeric expressions.
    '''
    return _EPSILON


def set_epsilon(e):
    '''Sets the value of the fuzz
    factor used in numeric expressions.
    '''
    global _EPSILON
    _EPSILON = e


def floatx():
    '''Returns the default float type, as a string
    (e.g. 'float16', 'float32', 'float64').
    '''
    return _FLOATX


def set_floatx(floatx):
    global _FLOATX
    if floatx not in {'float16', 'float32', 'float64'}:
        raise Exception('Unknown floatx type: ' + str(floatx))
    _FLOATX = str(floatx)


def cast_to_floatx(x):
    '''Cast a Numpy array to floatx.
    '''
    return np.asarray(x, dtype=_FLOATX)


def image_dim_ordering():
    '''Returns the image dimension ordering
    convention ('th' or 'tf').
    '''
    return _IMAGE_DIM_ORDERING


def set_image_dim_ordering(dim_ordering):
    '''Sets the value of the image dimension
    ordering convention ('th' or 'tf').
    '''
    global _IMAGE_DIM_ORDERING
    if dim_ordering not in {'tf', 'th'}:
        raise Exception('Unknown dim_ordering:', dim_ordering)
    _IMAGE_DIM_ORDERING = str(dim_ordering)


def get_uid(prefix=''):
    _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]
