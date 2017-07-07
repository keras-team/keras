from __future__ import absolute_import

from . import utils
from . import activations
from . import applications
from . import backend
from . import datasets
from . import engine
from . import layers
from . import preprocessing
from . import wrappers
from . import callbacks
from . import constraints
from . import initializers
from . import metrics
from . import models
from . import losses
from . import optimizers
from . import regularizers
# Importable from root because it's technically not a layer
from .layers import Input

# Default logger settings for the whole module
import logging
from logging import (NOTSET as _notset, BASIC_FORMAT as _basic_fmt)
logging.basicConfig(format=_basic_fmt, level=_notset)

__version__ = '2.0.5'
