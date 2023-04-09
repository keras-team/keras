from keras_core.initializers.initializer import Initializer
from keras_core.operations import numpy as knp
from keras_core.backend import standardize_dtype


class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.

    Examples:

    >>> # Standalone usage:
    >>> initializer = Zeros()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Zeros()
    >>> layer = Dense(3, kernel_initializer=initializer)
    """

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras_core.backend.floatx()` is
                used, which default to `float32` unless you configured it otherwise
                (via `keras_core.backend.set_floatx(float_dtype)`).
            **kwargs: Additional keyword arguments.
        """
        dtype = standardize_dtype(dtype)
        return knp.zeros(shape, dtype=dtype)


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.

    Also available via the shortcut function `ones`.

    Examples:

    >>> # Standalone usage:
    >>> initializer = Ones()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Ones()
    >>> layer = Dense(3, kernel_initializer=initializer)
    """

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras_core.backend.floatx()` is
                used, which default to `float32` unless you configured it otherwise
                (via `keras_core.backend.set_floatx(float_dtype)`).
            **kwargs: Additional keyword arguments.
        """
        dtype = standardize_dtype(dtype)
        return knp.ones(shape, dtype=dtype)
