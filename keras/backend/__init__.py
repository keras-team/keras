from __future__ import absolute_import
from __future__ import print_function
import os
import json
import sys
import logging
from .common import epsilon
from .common import floatx
from .common import set_epsilon
from .common import set_floatx
from .common import cast_to_floatx
from .common import image_data_format
from .common import set_image_data_format

# Obtain Keras base dir path: either ~/.keras or /tmp.
_keras_base_dir = os.path.expanduser('~')
if not os.access(_keras_base_dir, os.W_OK):
    _keras_base_dir = '/tmp'
_keras_dir = os.path.join(_keras_base_dir, '.keras')

# Default logger level settings for backend messages
_LOG_LEVEL_NAMES = {'NOTSET': logging.NOTSET,      # 0
                    'DEBUG': logging.DEBUG,        # 10
                    'INFO': logging.INFO,          # 20
                    'WARNING': logging.WARNING,    # 30
                    'ERROR': logging.ERROR,        # 40
                    'CRITICAL': logging.CRITICAL}  # 50
_LOG_LEVEL = logging.WARNING
logging.basicConfig(level=_LOG_LEVEL,
                    format=logging.BASIC_FORMAT,
                    stream=sys.stderr)
_user_logging_level = _LOG_LEVEL
logger = logging.getLogger(__name__)
logger.setLevel(_LOG_LEVEL)

# Default backend: TensorFlow.
_SUPPORTED_BACKENDS = {'theano', 'tensorflow', 'cntk'}
_BACKEND = 'tensorflow'

# Attempt to read Keras config file.
_config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
if os.path.exists(_config_path):
    try:
        _config = json.load(open(_config_path))
    except ValueError:
        _config = {}
    _floatx = _config.get('floatx', floatx())
    assert _floatx in {'float16', 'float32', 'float64'}
    _epsilon = _config.get('epsilon', epsilon())
    assert isinstance(_epsilon, float)
    _backend = _config.get('backend', _BACKEND)
    assert _backend in _SUPPORTED_BACKENDS
    _image_data_format = _config.get('image_data_format',
                                     image_data_format())
    assert _image_data_format in {'channels_last', 'channels_first'}
    _user_logging_level = _config.get('logging_level', _LOG_LEVEL)
    assert (_user_logging_level in _LOG_LEVEL_NAMES or
            isinstance(_user_logging_level, int))

    set_floatx(_floatx)
    set_epsilon(_epsilon)
    set_image_data_format(_image_data_format)
    logger.setLevel(_user_logging_level)
    _BACKEND = _backend

# Save config file, if possible.
if not os.path.exists(_keras_dir):
    try:
        os.makedirs(_keras_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        log.warning("""The following Keras folders could not be created
                       due to either permission settings or another Keras
                       instance also trying to manipulate the same folder.""")
        pass

if not os.path.exists(_config_path):
    _config = {
        'floatx': floatx(),
        'epsilon': epsilon(),
        'backend': _BACKEND,
        'image_data_format': image_data_format(),
        'logging_level': _user_logging_level
    }
    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        logger.warning("""Couldn't create a configuration file for Keras
                          under '{}'. This script probably doesn't have
                          access to this folder. The library defaults
                          settings will be used.""")
        pass

# Set backend based on KERAS_BACKEND flag, if applicable.
if 'KERAS_BACKEND' in os.environ:
    _backend = os.environ['KERAS_BACKEND']
    logger.info('KERAS_BACKEND flag is set in the environment.')
    assert _backend in _SUPPORTED_BACKENDS
    _BACKEND = _backend

# Import backend functions.
if _BACKEND == 'cntk':
    logger.info('Using CNTK backend.')
    from .cntk_backend import *
elif _BACKEND == 'theano':
    logger.info('Using Theano backend.')
    from .theano_backend import *
elif _BACKEND == 'tensorflow':
    logger.info('Using TensorFlow backend.')
    from .tensorflow_backend import *
else:
    raise ValueError('Unknown backend: ' + str(_BACKEND))


def backend():
    """Publicly accessible method
    for determining the current backend.

    # Returns
        String, the name of the backend Keras is currently using.

    # Example
    ```python
        >>> keras.backend.backend()
        'tensorflow'
    ```
    """
    return _BACKEND
