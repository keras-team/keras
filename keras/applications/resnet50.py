from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import resnet50
from . import keras_modules_injection


@keras_modules_injection
def ResNet50(*args, **kwargs):
    return resnet50.ResNet50(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return resnet50.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return resnet50.preprocess_input(*args, **kwargs)
