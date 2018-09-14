from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import nasnet
from . import keras_modules_injection


@keras_modules_injection
def NASNetMobile(*args, **kwargs):
    return nasnet.NASNetMobile(*args, **kwargs)


@keras_modules_injection
def NASNetLarge(*args, **kwargs):
    return nasnet.NASNetLarge(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return nasnet.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return nasnet.preprocess_input(*args, **kwargs)
