from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from keras_applications import efficientnet
except:
    efficientnet = None
from . import keras_modules_injection


@keras_modules_injection
def EfficientNetSmall(*args, **kwargs):
    return efficientnet.EfficientNetSmall(*args, **kwargs)


@keras_modules_injection
def EfficientNetLarge(*args, **kwargs):
    return efficientnet.EfficientNetLarge(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return efficientnet.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return efficientnet.preprocess_input(*args, **kwargs)
