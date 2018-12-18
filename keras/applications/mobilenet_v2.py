from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import mobilenet_v2
from . import keras_modules_injection


@keras_modules_injection
def MobileNetV2(*args, **kwargs):
    return mobilenet_v2.MobileNetV2(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return mobilenet_v2.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return mobilenet_v2.preprocess_input(*args, **kwargs)
