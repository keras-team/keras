# This file should NEVER be packaged! This is a hack to make "import keras" from
# the base of the repo just import the source files. We'll keep it for compat
# mostly.


from keras.api import *  # noqa: F403
from keras.api import __version__  # Import * ignores names start with "_".

import os  # isort: skip

# Add everything in /api/ to the module search path.
__path__.append(os.path.join(os.path.dirname(__file__), "api"))  # noqa: F405

# Don't pollute namespace.
del os
