from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import xception
from . import keras_modules_injection


@keras_modules_injection
def Xception(*args, **kwargs):
    return xception.Xception(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return xception.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return xception.preprocess_input(*args, **kwargs)
