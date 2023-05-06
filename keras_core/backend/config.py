import json
import os

from keras_core.api_export import keras_core_export

# The type of float to use throughout a session.
_FLOATX = "float32"

# Epsilon fuzz factor used throughout the codebase.
_EPSILON = 1e-7

# Default image data format, one of "channels_last", "channels_first".
_IMAGE_DATA_FORMAT = "channels_last"

# Default backend: TensorFlow.
_BACKEND = "tensorflow"


@keras_core_export(["keras_core.config.floatx", "keras_core.backend.floatx"])
def floatx():
    """Return the default float type, as a string.

    E.g. `'float16'`, `'float32'`, `'float64'`.

    Returns:
        String, the current default float type.

    Example:
    >>> keras_core.config.floatx()
    'float32'
    """
    return _FLOATX


@keras_core_export(
    ["keras_core.config.set_floatx", "keras_core.backend.set_floatx"]
)
def set_floatx(value):
    """Set the default float dtype.

    Note: It is not recommended to set this to `"float16"` for training,
    as this will likely cause numeric stability issues.
    Instead, mixed precision, which leverages
    a mix of `float16` and `float32`. It can be configured by calling
    `keras_core.mixed_precision.set_global_policy('mixed_float16')`.

    Args:
        value: String; `'float16'`, `'float32'`, or `'float64'`.

    Example:
    >>> keras_core.config.floatx()
    'float32'
    >>> keras_core.config.set_floatx('float64')
    >>> keras_core.config.floatx()
    'float64'
    >>> keras_core.config.set_floatx('float32')

    Raises:
        ValueError: In case of invalid value.
    """
    global _FLOATX
    accepted_dtypes = {"float16", "float32", "float64"}
    if value not in accepted_dtypes:
        raise ValueError(
            f"Unknown `floatx` value: {value}. "
            f"Expected one of {accepted_dtypes}"
        )
    _FLOATX = str(value)


@keras_core_export(["keras_core.config.epsilon", "keras_core.backend.epsilon"])
def epsilon():
    """Return the value of the fuzz factor used in numeric expressions.

    Returns:
        A float.

    Example:
    >>> keras_core.config.epsilon()
    1e-07
    """
    return _EPSILON


@keras_core_export(
    ["keras_core.config.set_epsilon", "keras_core.backend.set_epsilon"]
)
def set_epsilon(value):
    """Set the value of the fuzz factor used in numeric expressions.

    Args:
        value: float. New value of epsilon.

    Example:
    >>> keras_core.config.epsilon()
    1e-07
    >>> keras_core.config.set_epsilon(1e-5)
    >>> keras_core.config.epsilon()
    1e-05
     >>> keras_core.config.set_epsilon(1e-7)
    """
    global _EPSILON
    _EPSILON = value


@keras_core_export(
    [
        "keras_core.config.image_data_format",
        "keras_core.backend.image_data_format",
    ]
)
def image_data_format():
    """Return the default image data format convention.

    Returns:
        A string, either `'channels_first'` or `'channels_last'`.

    Example:
    >>> keras_core.config.image_data_format()
    'channels_last'
    """
    return _IMAGE_DATA_FORMAT


@keras_core_export(
    [
        "keras_core.config.set_image_data_format",
        "keras_core.backend.set_image_data_format",
    ]
)
def set_image_data_format(data_format):
    """Set the value of the image data format convention.

    Args:
        data_format: string. `'channels_first'` or `'channels_last'`.

    Example:
    >>> keras_core.config.image_data_format()
    'channels_last'
    >>> keras_core.config.set_image_data_format('channels_first')
    >>> keras_core.config.image_data_format()
    'channels_first'
    >>> keras_core.config.set_image_data_format('channels_last')
    """
    global _IMAGE_DATA_FORMAT
    accepted_formats = {"channels_last", "channels_first"}
    if data_format not in accepted_formats:
        raise ValueError(
            f"Unknown `data_format`: {data_format}. "
            f"Expected one of {accepted_formats}"
        )
    _IMAGE_DATA_FORMAT = str(data_format)


# Set Keras base dir path given KERAS_HOME env variable, if applicable.
# Otherwise either ~/.keras or /tmp.
if "KERAS_HOME" in os.environ:
    _keras_dir = os.environ.get("KERAS_HOME")
else:
    _keras_base_dir = os.path.expanduser("~")
    if not os.access(_keras_base_dir, os.W_OK):
        _keras_base_dir = "/tmp"
    _keras_dir = os.path.join(_keras_base_dir, ".keras")


# Attempt to read Keras config file.
_config_path = os.path.expanduser(os.path.join(_keras_dir, "keras.json"))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}
    _floatx = _config.get("floatx", floatx())
    assert _floatx in {"float16", "float32", "float64"}
    _epsilon = _config.get("epsilon", epsilon())
    assert isinstance(_epsilon, float)
    _backend = _config.get("backend", _BACKEND)
    _image_data_format = _config.get("image_data_format", image_data_format())
    assert _image_data_format in {"channels_last", "channels_first"}

    set_floatx(_floatx)
    set_epsilon(_epsilon)
    set_image_data_format(_image_data_format)
    _BACKEND = _backend

# Save config file, if possible.
if not os.path.exists(_keras_dir):
    try:
        os.makedirs(_keras_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

if not os.path.exists(_config_path):
    _config = {
        "floatx": floatx(),
        "epsilon": epsilon(),
        "backend": _BACKEND,
        "image_data_format": image_data_format(),
    }
    try:
        with open(_config_path, "w") as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass

# Set backend based on KERAS_BACKEND flag, if applicable.
if "KERAS_BACKEND" in os.environ:
    _backend = os.environ["KERAS_BACKEND"]
    if _backend:
        _BACKEND = _backend


@keras_core_export("keras_core.backend.backend")
def backend():
    """Publicly accessible method for determining the current backend.

    Returns:
        String, the name of the backend Keras is currently using.

    Example:

    >>> keras.backend.backend()
    'tensorflow'
    """
    return _BACKEND
