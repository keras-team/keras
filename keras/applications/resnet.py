from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from keras_applications import resnet
except:
    resnet = None
from . import keras_modules_injection


@keras_modules_injection
def ResNet50(*args, **kwargs):
    return resnet.ResNet50(*args, **kwargs)


@keras_modules_injection
def ResNet101(*args, **kwargs):
    return resnet.ResNet101(*args, **kwargs)


@keras_modules_injection
def ResNet152(*args, **kwargs):
    return resnet.ResNet152(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return resnet.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return resnet.preprocess_input(*args, **kwargs)
