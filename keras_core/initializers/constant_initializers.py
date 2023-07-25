from keras_core import ops
from keras_core.api_export import keras_core_export
from keras_core.backend import standardize_dtype
from keras_core.initializers.initializer import Initializer


@keras_core_export(
    ["keras_core.initializers.Constant", "keras_core.initializers.constant"]
)
class Constant(Initializer):
    """Initializer that generates tensors with constant values.

    Only scalar values are allowed.
    The constant value provided must be convertible to the dtype requested
    when calling the initializer.

    Examples:

    >>> # Standalone usage:
    >>> initializer = Constant(10.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Constant(10.)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        value: A Python scalar.
    """

    def __init__(self, value=0.0):
        self.value = float(value)

    def __call__(self, shape, dtype=None):
        dtype = standardize_dtype(dtype)
        return self.value * ops.ones(shape=shape, dtype=dtype)

    def get_config(self):
        return {"value": self.value}


@keras_core_export(
    ["keras_core.initializers.Zeros", "keras_core.initializers.zeros"]
)
class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.

    Examples:

    >>> # Standalone usage:
    >>> initializer = Zeros()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Zeros()
    >>> layer = Dense(units=3, kernel_initializer=initializer)
    """

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras_core.backend.floatx()`
                is used, which default to `float32` unless you configured it
                otherwise (via `keras_core.backend.set_floatx(float_dtype)`).
            **kwargs: Additional keyword arguments.
        """
        dtype = standardize_dtype(dtype)
        return ops.zeros(shape, dtype=dtype)


@keras_core_export(
    ["keras_core.initializers.Ones", "keras_core.initializers.ones"]
)
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

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras_core.backend.floatx()`
                is used, which default to `float32` unless you configured it
                otherwise (via `keras_core.backend.set_floatx(float_dtype)`).
            **kwargs: Additional keyword arguments.
        """
        dtype = standardize_dtype(dtype)
        return ops.ones(shape, dtype=dtype)


@keras_core_export(
    [
        "keras_core.initializers.IdentityInitializer",
        "keras_core.initializers.Identity",
        "keras_core.initializers.identity",
    ]
)
class Identity(Initializer):
    """Initializer that generates the identity matrix.

    Only usable for generating 2D matrices.

    Examples:

    >>> # Standalone usage:
    >>> initializer = Identity()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Identity()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        gain: Multiplicative factor to apply to the identity matrix.
    """

    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras_core.backend.floatx()`
                is used, which default to `float32` unless you configured it
                otherwise (via `keras_core.backend.set_floatx(float_dtype)`).
            **kwargs: Additional keyword arguments.
        """
        if len(shape) != 2:
            raise ValueError(
                "Identity matrix initializer can only be used for 2D matrices. "
                f"Received: shape={shape} of rank {len(shape)}."
            )
        dtype = standardize_dtype(dtype)
        return self.gain * ops.eye(*shape, dtype=dtype)
