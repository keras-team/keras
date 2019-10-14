from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import efficientnet
from . import keras_modules_injection


@keras_modules_injection
def EfficientNetB0(*args, **kwargs):
    return efficientnet.EfficientNetB0(*args, **kwargs)


@keras_modules_injection
def EfficientNetB1(*args, **kwargs):
    return efficientnet.EfficientNetB1(*args, **kwargs)


@keras_modules_injection
def EfficientNetB2(*args, **kwargs):
    return efficientnet.EfficientNetB2(*args, **kwargs)


@keras_modules_injection
def EfficientNetB3(*args, **kwargs):
    return efficientnet.EfficientNetB3(*args, **kwargs)


@keras_modules_injection
def EfficientNetB4(*args, **kwargs):
    return efficientnet.EfficientNetB4(*args, **kwargs)


@keras_modules_injection
def EfficientNetB5(*args, **kwargs):
    return efficientnet.EfficientNetB5(*args, **kwargs)


@keras_modules_injection
def EfficientNetB6(*args, **kwargs):
    return efficientnet.EfficientNetB6(*args, **kwargs)


@keras_modules_injection
def EfficientNetB7(*args, **kwargs):
    return efficientnet.EfficientNetB7(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return densenet.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return densenet.preprocess_input(*args, **kwargs)
