# The type of float to use throughout a session.
_FLOATX = "float32"

# Epsilon fuzz factor used throughout the codebase.
_EPSILON = 1e-7

# Default image data format, one of "channels_last", "channels_first".
_IMAGE_DATA_FORMAT = "channels_last"


def epsilon():
    """Return the value of the fuzz factor used in numeric expressions.

    Returns:
        A float.

    Example:
    >>> keras_core.backend.epsilon()
    1e-07
    """
    return _EPSILON


def set_epsilon(value):
    """Set the value of the fuzz factor used in numeric expressions.

    Args:
        value: float. New value of epsilon.

    Example:
    >>> keras_core.backend.epsilon()
    1e-07
    >>> keras_core.backend.set_epsilon(1e-5)
    >>> keras_core.backend.epsilon()
    1e-05
     >>> keras_core.backend.set_epsilon(1e-7)
    """
    global _EPSILON
    _EPSILON = value


def floatx():
    """Return the default float type, as a string.

    E.g. `'float16'`, `'float32'`, `'float64'`.

    Returns:
        String, the current default float type.

    Example:
    >>> keras_core.backend.floatx()
    'float32'
    """
    return _FLOATX


def set_floatx(value):
    """Set the default float type.

    Note: It is not recommended to set this to float16 for training, as this
    will likely cause numeric stability issues. Instead, mixed precision, which
    is using a mix of float16 and float32, can be used by calling
    `keras_core.mixed_precision.set_global_policy('mixed_float16')`. See the
    [mixed precision guide](
      https://www.tensorflow.org/guide/keras/mixed_precision) for details.

    Args:
        value: String; `'float16'`, `'float32'`, or `'float64'`.

    Example:
    >>> keras_core.backend.floatx()
    'float32'
    >>> keras_core.backend.set_floatx('float64')
    >>> keras_core.backend.floatx()
    'float64'
    >>> keras_core.backend.set_floatx('float32')

    Raises:
        ValueError: In case of invalid value.
    """
    global _FLOATX
    accepted_dtypes = {"float16", "float32", "float64"}
    if value not in accepted_dtypes:
        raise ValueError(
            f"Unknown `floatx` value: {value}. " f"Expected one of {accepted_dtypes}"
        )
    _FLOATX = str(value)


def image_data_format():
    """Return the default image data format convention.

    Returns:
        A string, either `'channels_first'` or `'channels_last'`

    Example:
    >>> keras_core.backend.image_data_format()
    'channels_last'
    """
    return _IMAGE_DATA_FORMAT


def set_image_data_format(data_format):
    """Set the value of the image data format convention.

    Args:
        data_format: string. `'channels_first'` or `'channels_last'`.

    Example:
    >>> keras_core.backend.image_data_format()
    'channels_last'
    >>> keras_core.backend.set_image_data_format('channels_first')
    >>> keras_core.backend.image_data_format()
    'channels_first'
    >>> keras_core.backend.set_image_data_format('channels_last')

    Raises:
        ValueError: In case of invalid `data_format` value.
    """
    global _IMAGE_DATA_FORMAT
    accepted_formats = {"channels_last", "channels_first"}
    if data_format not in accepted_formats:
        raise ValueError(
            f"Unknown `data_format`: {data_format}. "
            f"Expected one of {accepted_formats}"
        )
    _IMAGE_DATA_FORMAT = str(data_format)
