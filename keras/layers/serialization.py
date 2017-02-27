"""Layer serialization/deserialization functions.
"""
# pylint: disable=wildcard-import
# pylint: disable=unused-import
from __future__ import absolute_import

from .advanced_activations import *
from .convolutional import *
from .convolutional_recurrent import *
from .core import *
from .embeddings import *
from ..engine import Input
from ..engine import InputLayer
from .local import *
from .merge import *
from .noise import *
from .normalization import *
from .pooling import *
from .recurrent import *
from ..utils.generic_utils import deserialize_keras_object
from .wrappers import *


def serialize(layer):
    return {'class_name': layer.__class__.__name__,
            'config': layer.get_config()}


def deserialize(config, custom_objects=None):
    """Instantiates a layer from a config dictionary.

    # Arguments
        config: dict of the form {'class_name': str, 'config': dict}
        custom_objects: dict mapping class names (or function names)
            of custom (non-Keras) objects to class/functions

    # Returns
        Layer instance (may be Model, Sequential, Layer...)
    """
    from .. import models  # pylint: disable=g-import-not-at-top
    globs = globals()  # All layers.
    globs['Model'] = models.Model
    globs['Sequential'] = models.Sequential
    return deserialize_keras_object(config,
                                    module_objects=globs,
                                    custom_objects=custom_objects,
                                    printable_module_name='layer')
