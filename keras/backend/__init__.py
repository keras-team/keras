from __future__ import absolute_import
from __future__ import print_function
import os
import json
from .common import epsilon, floatx, debug_mode, set_epsilon, set_floatx, set_debug_mode

_keras_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(_keras_dir):
    os.makedirs(_keras_dir)

_BACKEND = 'theano'
_config_path = os.path.expanduser(os.path.join('~', '.keras', 'keras.json'))
if os.path.exists(_config_path):
    _config = json.load(open(_config_path))
    _floatx = _config.get('floatx', floatx())
    assert _floatx in {'float32', 'float64'}
    _epsilon = _config.get('epsilon', epsilon())
    assert type(_epsilon) == float
    _debug_mode = _config.get('debug_mode', debug_mode())
    assert _debug_mode in {'none', 'detect_nan'}
    _backend = _config.get('backend', _BACKEND)
    assert _backend in {'theano', 'tensorflow'}

    set_floatx(_floatx)
    set_epsilon(_epsilon)
    set_debug_mode(_debug_mode)
    _BACKEND = _backend
else:
    # save config file, for easy edition
    _config = {'floatx': floatx(),
               'epsilon': epsilon(),
               'debug_mode': debug_mode(),
               'backend': _BACKEND}
    with open(_config_path, 'w') as f:
        # add new line in order for bash 'cat' display the content correctly
        f.write(json.dumps(_config) + '\n')

if _BACKEND == 'theano':
    print('Using Theano backend.')
    from .theano_backend import *
elif _BACKEND == 'tensorflow':
    print('Using TensorFlow backend.')
    from .tensorflow_backend import *
else:
    raise Exception('Unknown backend: ' + str(_BACKEND))
