from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import densenet
from . import keras_modules_injection


@keras_modules_injection
def DenseNet121(*args, **kwargs):
    return densenet.DenseNet121(*args, **kwargs)


@keras_modules_injection
def DenseNet169(*args, **kwargs):
    return densenet.DenseNet169(*args, **kwargs)


@keras_modules_injection
def DenseNet201(*args, **kwargs):
    return densenet.DenseNet201(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return densenet.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return densenet.preprocess_input(*args, **kwargs)
