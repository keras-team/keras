from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import vgg19
from . import keras_modules_injection


@keras_modules_injection
def VGG19(*args, **kwargs):
    return vgg19.VGG19(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return vgg19.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return vgg19.preprocess_input(*args, **kwargs)
