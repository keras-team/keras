from __future__ import absolute_import, print_function

import os

KERAS_DIR = os.environ.get('KERAS_DIR', os.path.expanduser('~/.keras'))

if not os.path.exists(KERAS_DIR):
    os.mkdir(KERAS_DIR)

if not os.access(KERAS_DIR, os.W_OK):
    import tempfile
    KERAS_DIR = tempfile.mkdtemp('','keras')

