import numpy as np

from collections import defaultdict

# the type of float to use throughout the session.
_FLOATX = 'float32'
_EPSILON = 10e-8
_UID_PREFIXES = defaultdict(int)
_IMAGE_DIM_ORDERING = 'tf'
_LEGACY_WEIGHT_ORDERING = False


def epsilon():
    '''Returns the value of the fuzz
    factor used in numeric expressions.

    # Return
        Float (scalar)

    # Example
    ```python
        >>> keras.backend.epsilon()
        1e-07
    ```

    '''
    return _EPSILON


def set_epsilon(e):
    '''Sets the value of the fuzz
    factor used in numeric expressions.

    # Argument
        e: float. New value of epsilon

    # Return
        `None`

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.epsilon()
        1e-07
        >>> K.set_epsilon(1e-05)
        >>> K.epsilon()
        1e-05
    ```
    '''
    global _EPSILON
    _EPSILON = e


def floatx():
    '''Returns the default float type, as a string
    (e.g. 'float16', 'float32', 'float64').

    # Return
        String, the current default float type

    # Example
    ```python
        >>> keras.backend.floatx()
        'float32'
    ```
    '''
    return _FLOATX


def set_floatx(floatx):
    '''Sets the default float type.

    # Argument
        floatx: string. 'float16', 'float32', or 'float64'.

    # Return
        `None`

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.floatx()
        'float32'
        >>> K.set_floatx('float16')
        >>> K.floatx()
        'float16'
    ```
    '''

    global _FLOATX
    if floatx not in {'float16', 'float32', 'float64'}:
        raise Exception('Unknown floatx type: ' + str(floatx))
    _FLOATX = str(floatx)


def cast_to_floatx(x):
    '''Cast a Numpy array to floatx.

    # Argument
        x: Numpy array.

    # Return
        A Numpy array

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
    '''
    return np.asarray(x, dtype=_FLOATX)


def image_dim_ordering():
    '''Returns the image dimension ordering
    convention ('th' or 'tf').

    # Return
        String, either `'th'` or `'tf'`

    # Example
    ```python
        >>> keras.backend.image_dim_ordering()
        'th'
    ```
    '''
    return _IMAGE_DIM_ORDERING


def set_image_dim_ordering(dim_ordering):
    '''Sets the value of the image dimension
    ordering convention ('th' or 'tf').

    # Argument
        dim_ordering: string. 'th' or 'tf'.

    # Return
        `None`

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.image_dim_ordering()
        'th'
        >>> K.set_image_dim_ordering('tf')
        >>> K.image_dim_ordering()
        'tf'
    ```
    '''
    global _IMAGE_DIM_ORDERING
    if dim_ordering not in {'tf', 'th'}:
        raise Exception('Unknown dim_ordering:', dim_ordering)
    _IMAGE_DIM_ORDERING = str(dim_ordering)


def get_uid(prefix=''):
    '''

    # Argument
        prefix: string

    # Return
        integer (scalar)

    # Example
    ```
        >>> a = keras.backend.get_uid()
        >>> type(a)
        <type 'int'>
    ```

    '''
    _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]


def reset_uids():
    '''
    # Return
        `None`
    '''
    global _UID_PREFIXES
    _UID_PREFIXES = defaultdict(int)


def is_keras_tensor(x):
    '''Returns if `x` is a Keras tensor,
    which is equivalent to `tf.placeholder()` or
    `T.matrix()`, `T.tensor3()`.

    # Argument
        x: any type.

    # Return
        Boolean

    # Example
    ```python
        >>> from keras import backend as K
        >>> np_var = numpy.array([1, 2])
        >>> K.is_keras_tensor(np_var)
        False
        >>> shared_var = K.variable(np_var)
        >>> K.is_keras_tensor(shared_var)
        False
        >>> input = K.placeholder(shape=(2, 4, 5))
        >>> K.is_keras_tensor(input)
        True
    ```
    '''
    if hasattr(x, '_keras_shape'):
        return True
    else:
        return False


def set_legacy_weight_ordering(value):
    '''
    # Return
        `None`

    '''
    global _LEGACY_WEIGHT_ORDERING
    assert value in {True, False}
    _LEGACY_WEIGHT_ORDERING = value


def legacy_weight_ordering():
    '''
    # Return
        Boolean

    # Example
    ```python
        >>> keras.backend.legacy_weight_ordering()
        False
    ```
    '''
    return _LEGACY_WEIGHT_ORDERING
