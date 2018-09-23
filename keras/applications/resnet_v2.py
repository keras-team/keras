from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from keras_applications import resnet_v2
except:
    resnet_v2 = None
from . import keras_modules_injection


@keras_modules_injection
def ResNet50V2(*args, **kwargs):
    return resnet_v2.ResNet50V2(*args, **kwargs)


@keras_modules_injection
def ResNet101V2(*args, **kwargs):
    return resnet_v2.ResNet101V2(*args, **kwargs)


@keras_modules_injection
def ResNet152V2(*args, **kwargs):
    return resnet_v2.ResNet152V2(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return resnet_v2.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return resnet_v2.preprocess_input(*args, **kwargs)
