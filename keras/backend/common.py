import numpy as np

from collections import defaultdict

# the type of float to use throughout the session.
_FLOATX = 'float32'
_EPSILON = 10e-8
_UID_PREFIXES = defaultdict(int)
_IMAGE_DIM_ORDERING = 'tf'
_LEGACY_WEIGHT_ORDERING = False


def epsilon():
    """Returns the value of the fuzz
    factor used in numeric expressions.

    # Returns
        A float.

    # Example
    ```python
        >>> keras.backend.epsilon()
        1e-08
    ```
    """
    return _EPSILON


def set_epsilon(e):
    """Sets the value of the fuzz
    factor used in numeric expressions.

    # Arguments
        e: float. New value of epsilon.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.epsilon()
        1e-08
        >>> K.set_epsilon(1e-05)
        >>> K.epsilon()
        1e-05
    ```
    """
    global _EPSILON
    _EPSILON = e


def floatx():
    """Returns the default float type, as a string
    (e.g. 'float16', 'float32', 'float64').

    # Returns
        String, the current default float type.

    # Example
    ```python
        >>> keras.backend.floatx()
        'float32'
    ```
    """
    return _FLOATX


def set_floatx(floatx):
    """Sets the default float type.

    # Arguments
        String: 'float16', 'float32', or 'float64'.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.floatx()
        'float32'
        >>> K.set_floatx('float16')
        >>> K.floatx()
        'float16'
    ```
    """
    global _FLOATX
    if floatx not in {'float16', 'float32', 'float64'}:
        raise ValueError('Unknown floatx type: ' + str(floatx))
    _FLOATX = str(floatx)


def cast_to_floatx(x):
    """Cast a Numpy array to the default Keras float type.

    # Arguments
        x: Numpy array.

    # Returns
        The same Numpy array, cast to its new type.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.floatx()
        'float32'
        >>> arr = numpy.array([1.0, 2.0], dtype='float64')
        >>> arr.dtype
        dtype('float64')
        >>> new_arr = K.cast_to_floatx(arr)
        >>> new_arr
        array([ 1.,  2.], dtype=float32)
        >>> new_arr.dtype
        dtype('float32')
    ```
    """
    return np.asarray(x, dtype=_FLOATX)


def image_dim_ordering():
    """Returns the default image dimension ordering
    convention ('th' or 'tf').

    # Returns
        A string, either `'th'` or `'tf'`

    # Example
    ```python
        >>> keras.backend.image_dim_ordering()
        'th'
    ```
    """
    return _IMAGE_DIM_ORDERING


def set_image_dim_ordering(dim_ordering):
    """Sets the value of the image dimension
    ordering convention ('th' or 'tf').

    # Arguments
        dim_ordering: string. `'th'` or `'tf'`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.image_dim_ordering()
        'th'
        >>> K.set_image_dim_ordering('tf')
        >>> K.image_dim_ordering()
        'tf'
    ```
    """
    global _IMAGE_DIM_ORDERING
    if dim_ordering not in {'tf', 'th'}:
        raise ValueError('Unknown dim_ordering:', dim_ordering)
    _IMAGE_DIM_ORDERING = str(dim_ordering)


def get_uid(prefix=''):
    """Provides a unique UID given a string prefix.

    # Arguments
        prefix: string.

    # Returns
        An integer.

    # Example
    ```
        >>> keras.backend.get_uid('dense')
        >>> 1
        >>> keras.backend.get_uid('dense')
        >>> 2
    ```

    """
    _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]


def reset_uids():
    global _UID_PREFIXES
    _UID_PREFIXES = defaultdict(int)


def is_keras_tensor(x):
    """Returns whether `x` is a Keras tensor.

    # Arguments
        x: a potential tensor.

    # Returns
        A boolean: whether the argument is a Keras tensor.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> np_var = numpy.array([1, 2])
        >>> K.is_keras_tensor(np_var)
        False
        >>> keras_var = K.variable(np_var)
        >>> K.is_keras_tensor(keras_var)  # A variable is not a Tensor.
        False
        >>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
        >>> K.is_keras_tensor(keras_placeholder)  # A placeholder is a Tensor.
        True
    ```
    """
    if hasattr(x, '_keras_shape'):
        return True
    else:
        return False


def set_legacy_weight_ordering(value):
    global _LEGACY_WEIGHT_ORDERING
    assert value in {True, False}
    _LEGACY_WEIGHT_ORDERING = value


def legacy_weight_ordering():
    return _LEGACY_WEIGHT_ORDERING
