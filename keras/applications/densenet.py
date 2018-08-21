from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import keras_applications
from . import keras_modules_injection


@keras_modules_injection
def DenseNet121(*args, **kwargs):
    return keras_applications.densenet.DenseNet121(*args, **kwargs)


@keras_modules_injection
def DenseNet169(*args, **kwargs):
    return keras_applications.densenet.DenseNet169(*args, **kwargs)


@keras_modules_injection
def DenseNet201(*args, **kwargs):
    return keras_applications.densenet.DenseNet201(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return keras_applications.densenet.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return keras_applications.densenet.preprocess_input(*args, **kwargs)
