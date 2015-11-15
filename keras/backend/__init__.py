from __future__ import absolute_import

_BACKEND = 'theano'

if _BACKEND == 'theano':
    from .theano_backend import *
elif _BACKEND == 'tensorflow':
    from .tensorflow_backend import *
else:
    raise Exception('Unknown backend: ' + str(backend))
