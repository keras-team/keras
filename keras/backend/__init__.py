from __future__ import absolute_import
from __future__ import print_function
import os
import json
import sys
import tempfile
from .common import epsilon
from .common import floatx
from .common import set_epsilon
from .common import set_floatx
from .common import cast_to_floatx
from .common import image_data_format
from .common import set_image_data_format

_keras_dirname = '.keras'
_keras_config_filename = 'keras.json'

_keras_base_dir = None
_keras_dir = None
_config_path = None

if 'KERAS_DIR' in os.environ:
    # Obtain the Keras base dir from a user specified path
    _user_configured_base_dir = os.environ['KERAS_DIR']
    _user_configured_keras_dir = None
    if os.path.basename( _user_configured_base_dir ) == _keras_dirname:
        _user_configured_keras_dir = _user_configured_base_dir
        _user_configured_base_dir = os.path.dirname( _user_configured_base_dir )
    else:
        _user_configured_keras_dir = os.path.join( _user_configured_base_dir, _keras_dirname )
    _user_configured_config_path = os.path.join( _user_configured_keras_dir, _keras_config_filename )
    if os.path.exists( _user_configured_config_path ):
        _keras_base_dir = _user_configured_base_dir
        _keras_dir = _user_configured_keras_dir
        _config_path = _user_configured_config_path
    else:
        raise IOError( 'User specified configuration file ' \
                      + _user_configured_config_path + \
                      ' does not exist' )

if _config_path is None:
    # default to the user's directory
    _default_base_dir = os.path.expanduser('~')
    _default_keras_dir = os.path.join( _default_base_dir, _keras_dirname )
    _default_config_path = os.path.join( _default_keras_dir, _keras_config_filename )
    if os.path.exists( _default_config_path ) or os.access( _default_base_dir, os.W_OK ):
        _keras_base_dir = _default_base_dir
        _keras_dir = _default_keras_dir
        _config_path = _default_config_path

if _config_path is None:
    # backoff to the temp directory
    _keras_base_dir = tempfile.gettempdir()
    _keras_dir = os.path.join( _keras_base_dir, _keras_dirname )
    _config_path = os.path.join( _keras_dir, _keras_config_filename )

print( _keras_base_dir, _keras_dir, _config_path )

# Default backend: TensorFlow.
_BACKEND = 'tensorflow'

# Attempt to read Keras config file.
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
    assert _backend in {'theano', 'tensorflow', 'cntk'}
    _image_data_format = _config.get('image_data_format',
                                     image_data_format())
    assert _image_data_format in {'channels_last', 'channels_first'}

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
        'floatx': floatx(),
        'epsilon': epsilon(),
        'backend': _BACKEND,
        'image_data_format': image_data_format()
    }
    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass

# Set backend based on KERAS_BACKEND flag, if applicable.
if 'KERAS_BACKEND' in os.environ:
    _backend = os.environ['KERAS_BACKEND']
    assert _backend in {'theano', 'tensorflow', 'cntk'}
    _BACKEND = _backend

# Import backend functions.
if _BACKEND == 'cntk':
    sys.stderr.write('Using CNTK backend\n')
    from .cntk_backend import *
elif _BACKEND == 'theano':
    sys.stderr.write('Using Theano backend.\n')
    from .theano_backend import *
elif _BACKEND == 'tensorflow':
    sys.stderr.write('Using TensorFlow backend.\n')
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
