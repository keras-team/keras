import numpy as np

# the type of float to use throughout the session.
_FLOATX = 'float32'
_EPSILON = 10e-8
_DEBUG_MODE = 'none'


def epsilon():
    return _EPSILON


def set_epsilon(e):
    global _EPSILON
    _EPSILON = e


def floatx():
    return _FLOATX


def set_floatx(floatx):
    global _FLOATX
    if floatx not in {'float32', 'float64'}:
        raise Exception('Unknown floatx type: ' + str(floatx))
    floatx = str(floatx)
    _FLOATX = floatx


def cast_to_floatx(x):
    '''Cast a Numpy array to floatx.
    '''
    return np.asarray(x, dtype=_FLOATX)


def debug_mode():
    return _DEBUG_MODE


def set_debug_mode(debug_mode):
    global _DEBUG_MODE
    if debug_mode not in {'none', 'detect_nan'}:
        raise Exception('Unknown debug_mode: ' + debug_mode)
    _DEBUG_MODE = debug_mode
