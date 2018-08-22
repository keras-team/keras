from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import mobilenet
from . import keras_modules_injection


@keras_modules_injection
def MobileNet(*args, **kwargs):
    return mobilenet.MobileNet(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return mobilenet.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return mobilenet.preprocess_input(*args, **kwargs)
