from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import inception_v3
from . import keras_modules_injection


@keras_modules_injection
def InceptionV3(*args, **kwargs):
    return inception_v3.InceptionV3(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return inception_v3.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return inception_v3.preprocess_input(*args, **kwargs)
