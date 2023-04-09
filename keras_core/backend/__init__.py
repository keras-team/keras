import os
import json
import sys
from keras_core.utils.io_utils import print_msg

from keras_core.backend.keras_tensor import KerasTensor
from keras_core.backend.keras_tensor import is_keras_tensor
from keras_core.backend.keras_tensor import any_symbolic_tensors
from keras_core.backend.config import floatx
from keras_core.backend.config import epsilon
from keras_core.backend.config import image_data_format
from keras_core.backend.config import set_floatx
from keras_core.backend.config import set_epsilon
from keras_core.backend.config import set_image_data_format
from keras_core.backend.common import standardize_shape
from keras_core.backend.common import standardize_dtype
from keras_core.backend.common import StatelessScope


# Set Keras base dir path given KERAS_HOME env variable, if applicable.
# Otherwise either ~/.keras or /tmp.
if "KERAS_HOME" in os.environ:
    _keras_dir = os.environ.get("KERAS_HOME")
else:
    _keras_base_dir = os.path.expanduser("~")
    if not os.access(_keras_base_dir, os.W_OK):
        _keras_base_dir = "/tmp"
    _keras_dir = os.path.join(_keras_base_dir, ".keras")

# Default backend: TensorFlow.
_BACKEND = "tensorflow"

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

# Import backend functions.
if _BACKEND == "tensorflow":
    print_msg("Using TensorFlow backend")
    from keras_core.backend.tensorflow import *
elif _BACKEND == "jax":
    print_msg("Using JAX backend.")
    from keras_core.backend.jax import *
else:
    raise ValueError(f"Unable to import backend : {_BACKEND}")


def backend():
    """Publicly accessible method for determining the current backend.

    Returns:
        String, the name of the backend Keras is currently using.

    Example:

    >>> keras.backend.backend()
    'tensorflow'
    """
    return _BACKEND
