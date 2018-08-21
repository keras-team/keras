from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import keras_applications
from . import keras_modules_injection


@keras_modules_injection
def InceptionResNetV2(*args, **kwargs):
    return keras_applications.inception_resnet_v2.InceptionResNetV2(
        *args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return keras_applications.inception_resnet_v2.decode_predictions(
        *args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return keras_applications.inception_resnet_v2.preprocess_input(
        *args, **kwargs)
