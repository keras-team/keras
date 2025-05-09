import json
import os

from keras.src.api_export import keras_export

# The type of float to use throughout a session.
_FLOATX = "float32"

# Epsilon fuzz factor used throughout the codebase.
_EPSILON = 1e-7

# Default image data format, one of "channels_last", "channels_first".
_IMAGE_DATA_FORMAT = "channels_last"

# Default backend: TensorFlow.
_BACKEND = "tensorflow"

# Cap run duration for debugging.
_MAX_EPOCHS = None
_MAX_STEPS_PER_EPOCH = None


@keras_export(["keras.config.floatx", "keras.backend.floatx"])
def floatx():
    """Return the default float type, as a string.

    E.g. `'bfloat16'`, `'float16'`, `'float32'`, `'float64'`.

    Returns:
        String, the current default float type.

    Example:

    >>> keras.config.floatx()
    'float32'

    """
    return _FLOATX


@keras_export(["keras.config.set_floatx", "keras.backend.set_floatx"])
def set_floatx(value):
    """Set the default float dtype.

    Note: It is not recommended to set this to `"float16"` for training,
    as this will likely cause numeric stability issues.
    Instead, mixed precision, which leverages
    a mix of `float16` and `float32`. It can be configured by calling
    `keras.mixed_precision.set_dtype_policy('mixed_float16')`.

    Args:
        value: String; `'bfloat16'`, `'float16'`, `'float32'`, or `'float64'`.

    Examples:
    >>> keras.config.floatx()
    'float32'

    >>> keras.config.set_floatx('float64')
    >>> keras.config.floatx()
    'float64'

    >>> # Set it back to float32
    >>> keras.config.set_floatx('float32')

    Raises:
        ValueError: In case of invalid value.
    """
    global _FLOATX
    accepted_dtypes = {"bfloat16", "float16", "float32", "float64"}
    if value not in accepted_dtypes:
        raise ValueError(
            f"Unknown `floatx` value: {value}. "
            f"Expected one of {accepted_dtypes}"
        )
    _FLOATX = str(value)


@keras_export(["keras.config.epsilon", "keras.backend.epsilon"])
def epsilon():
    """Return the value of the fuzz factor used in numeric expressions.

    Returns:
        A float.

    Example:

    >>> keras.config.epsilon()
    1e-07

    """
    return _EPSILON


@keras_export(["keras.config.set_epsilon", "keras.backend.set_epsilon"])
def set_epsilon(value):
    """Set the value of the fuzz factor used in numeric expressions.

    Args:
        value: float. New value of epsilon.

    Examples:
    >>> keras.config.epsilon()
    1e-07

    >>> keras.config.set_epsilon(1e-5)
    >>> keras.config.epsilon()
    1e-05

    >>> # Set it back to the default value.
    >>> keras.config.set_epsilon(1e-7)

    """
    global _EPSILON
    _EPSILON = value


@keras_export(
    [
        "keras.config.image_data_format",
        "keras.backend.image_data_format",
    ]
)
def image_data_format():
    """Return the default image data format convention.

    Returns:
        A string, either `'channels_first'` or `'channels_last'`.

    Example:

    >>> keras.config.image_data_format()
    'channels_last'

    """
    return _IMAGE_DATA_FORMAT


@keras_export(
    [
        "keras.config.set_image_data_format",
        "keras.backend.set_image_data_format",
    ]
)
def set_image_data_format(data_format):
    """Set the value of the image data format convention.

    Args:
        data_format: string. `'channels_first'` or `'channels_last'`.

    Examples:

    >>> keras.config.image_data_format()
    'channels_last'

    >>> keras.config.set_image_data_format('channels_first')
    >>> keras.config.image_data_format()
    'channels_first'

    >>> # Set it back to `'channels_last'`
    >>> keras.config.set_image_data_format('channels_last')

    """
    global _IMAGE_DATA_FORMAT
    data_format = str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            "{'channels_first', 'channels_last'}. "
            f"Received: data_format={data_format}"
        )
    _IMAGE_DATA_FORMAT = data_format


@keras_export("keras.config.enable_flash_attention")
def enable_flash_attention():
    """Enable flash attention.

    Flash attention offers performance optimization for attention layers,
    making it especially useful for large language models (LLMs) that
    benefit from faster and more memory-efficient attention computations.

    Once enabled, supported layers like `MultiHeadAttention` will **attempt** to
    use flash attention for faster computations. By default, this feature is
    enabled.

    Note that enabling flash attention does not guarantee it will always be
    used. Typically, the inputs must be in `float16` or `bfloat16` dtype, and
    input layout requirements may vary depending on the backend.
    """
    from keras.src.backend.common import global_state

    global_state.set_global_attribute("flash_attention", None)


@keras_export("keras.config.disable_flash_attention")
def disable_flash_attention():
    """Disable flash attention.

    Flash attention offers performance optimization for attention layers,
    making it especially useful for large language models (LLMs) that
    benefit from faster and more memory-efficient attention computations.

    Once disabled, supported layers like `MultiHeadAttention` will not
    use flash attention for faster computations.
    """
    from keras.src.backend.common import global_state

    global_state.set_global_attribute("flash_attention", False)


@keras_export("keras.config.is_flash_attention_enabled")
def is_flash_attention_enabled():
    """Checks whether flash attention is globally enabled in Keras.

    Flash attention is a performance-optimized method for computing attention
    in large models, such as transformers, allowing for faster and more
    memory-efficient operations. This function checks the global Keras
    configuration to determine if flash attention is enabled for compatible
    layers (e.g., `MultiHeadAttention`).

    Note that enabling flash attention does not guarantee it will always be
    used. Typically, the inputs must be in `float16` or `bfloat16` dtype, and
    input layout requirements may vary depending on the backend.

    Returns:
        `False` if disabled; otherwise, it indicates that it is enabled.
    """
    from keras.src.backend.common import global_state

    return global_state.get_global_attribute("flash_attention", default=None)


def standardize_data_format(data_format):
    if data_format is None:
        return image_data_format()
    data_format = str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            "{'channels_first', 'channels_last'}. "
            f"Received: data_format={data_format}"
        )
    return data_format


# Set Keras base dir path given KERAS_HOME env variable, if applicable.
# Otherwise either ~/.keras or /tmp.
if "KERAS_HOME" in os.environ:
    _KERAS_DIR = os.environ.get("KERAS_HOME")
else:
    _keras_base_dir = os.path.expanduser("~")
    if not os.access(_keras_base_dir, os.W_OK):
        _keras_base_dir = "/tmp"
    _KERAS_DIR = os.path.join(_keras_base_dir, ".keras")


def keras_home():
    # Private accessor for the keras home location.
    return _KERAS_DIR


# Attempt to read Keras config file.
_config_path = os.path.expanduser(os.path.join(_KERAS_DIR, "keras.json"))
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
if not os.path.exists(_KERAS_DIR):
    try:
        os.makedirs(_KERAS_DIR)
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
if "KERAS_MAX_EPOCHS" in os.environ:
    _MAX_EPOCHS = int(os.environ["KERAS_MAX_EPOCHS"])
if "KERAS_MAX_STEPS_PER_EPOCH" in os.environ:
    _MAX_STEPS_PER_EPOCH = int(os.environ["KERAS_MAX_STEPS_PER_EPOCH"])

if _BACKEND != "tensorflow":
    # If we are not running on the tensorflow backend, we should stop tensorflow
    # from using all available GPU memory. See
    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


@keras_export(
    [
        "keras.config.backend",
        "keras.backend.backend",
    ]
)
def backend():
    """Publicly accessible method for determining the current backend.

    Returns:
        String, the name of the backend Keras is currently using. One of
            `"tensorflow"`, `"torch"`, or `"jax"`.

    Example:

    >>> keras.config.backend()
    'tensorflow'

    """
    return _BACKEND


@keras_export(["keras.config.set_max_epochs"])
def set_max_epochs(max_epochs):
    """Limit the maximum number of epochs for any call to fit.

    This will cap the number of epochs for any training run using `model.fit()`.
    This is purely for debugging, and can also be set via the `KERAS_MAX_EPOCHS`
    environment variable to quickly run a script without modifying its source.

    Args:
        max_epochs: The integer limit on the number of epochs or `None`. If
            `None`, no limit is applied.
    """
    global _MAX_EPOCHS
    _MAX_EPOCHS = max_epochs


@keras_export(["keras.config.set_max_steps_per_epoch"])
def set_max_steps_per_epoch(max_steps_per_epoch):
    """Limit the maximum number of steps for any call to fit/evaluate/predict.

    This will cap the number of steps for single epoch of a call to `fit()`,
    `evaluate()`, or `predict()`. This is purely for debugging, and can also be
    set via the `KERAS_MAX_STEPS_PER_EPOCH` environment variable to quickly run
    a scrip without modifying its source.

    Args:
        max_epochs: The integer limit on the number of epochs or `None`. If
            `None`, no limit is applied.
    """
    global _MAX_STEPS_PER_EPOCH
    _MAX_STEPS_PER_EPOCH = max_steps_per_epoch


@keras_export(["keras.config.max_epochs"])
def max_epochs():
    """Get the maximum number of epochs for any call to fit.

    Retrieves the limit on the number of epochs set by
    `keras.config.set_max_epochs` or the `KERAS_MAX_EPOCHS` environment
    variable.

    Returns:
        The integer limit on the number of epochs or `None`, if no limit has
        been set.
    """
    return _MAX_EPOCHS


@keras_export(["keras.config.max_steps_per_epoch"])
def max_steps_per_epoch():
    """Get the maximum number of steps for any call to fit/evaluate/predict.

    Retrieves the limit on the number of epochs set by
    `keras.config.set_max_steps_per_epoch` or the `KERAS_MAX_STEPS_PER_EPOCH`
    environment variable.

    Args:
        max_epochs: The integer limit on the number of epochs or `None`. If
            `None`, no limit is applied.
    """
    return _MAX_STEPS_PER_EPOCH
