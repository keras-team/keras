import numpy as np

# the type of float to use throughout the session.
_FLOATX = 'float32'
_EPSILON = 10e-8
_UID_PREFIXES = {}


def epsilon():
    return _EPSILON


def set_epsilon(e):
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
    floatx = str(floatx)
    _FLOATX = floatx


def cast_to_floatx(x):
    '''Cast a Numpy array to floatx.
    '''
    return np.asarray(x, dtype=_FLOATX)


def get_uid(prefix=''):
    if prefix not in _UID_PREFIXES:
        _UID_PREFIXES[prefix] = 1
        return 1
    else:
        _UID_PREFIXES[prefix] += 1
        return _UID_PREFIXES[prefix]
