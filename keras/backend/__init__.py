from __future__ import absolute_import
from __future__ import print_function
import os
import json
import sys
from .common import epsilon
from .common import floatx
from .common import set_epsilon
from .common import set_floatx
from .common import get_uid
from .common import cast_to_floatx
from .common import image_dim_ordering
from .common import set_image_dim_ordering
from .common import is_keras_tensor
from .common import legacy_weight_ordering
from .common import set_legacy_weight_ordering

_keras_base_dir = os.path.expanduser('~')
if not os.access(_keras_base_dir, os.W_OK):
    _keras_base_dir = '/tmp'

_keras_dir = os.path.join(_keras_base_dir, '.keras')
if not os.path.exists(_keras_dir):
    os.makedirs(_keras_dir)

# Default backend: TensorFlow.
_BACKEND = 'tensorflow'

_config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
if os.path.exists(_config_path):
    _config = json.load(open(_config_path))
    _floatx = _config.get('floatx', floatx())
    assert _floatx in {'float16', 'float32', 'float64'}
    _epsilon = _config.get('epsilon', epsilon())
    assert isinstance(_epsilon, float)
    _backend = _config.get('backend', _BACKEND)
    assert _backend in {'theano', 'tensorflow'}
    _image_dim_ordering = _config.get('image_dim_ordering',
                                      image_dim_ordering())
    assert _image_dim_ordering in {'tf', 'th'}

    set_floatx(_floatx)
    set_epsilon(_epsilon)
    set_image_dim_ordering(_image_dim_ordering)
    _BACKEND = _backend

# save config file
if not os.path.exists(_config_path):
    _config = {'floatx': floatx(),
               'epsilon': epsilon(),
               'backend': _BACKEND,
               'image_dim_ordering': image_dim_ordering()}
    with open(_config_path, 'w') as f:
        f.write(json.dumps(_config, indent=4))

if 'KERAS_BACKEND' in os.environ:
    _backend = os.environ['KERAS_BACKEND']
    assert _backend in {'theano', 'tensorflow'}
    _BACKEND = _backend

# import backend
if _BACKEND == 'theano':
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
    """
    return _BACKEND
