from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import keras_applications
from . import keras_modules_injection


@keras_modules_injection
def NASNetMobile(*args, **kwargs):
    return keras_applications.nasnet.NASNetMobile(*args, **kwargs)


@keras_modules_injection
def NASNetLarge(*args, **kwargs):
    return keras_applications.nasnet.NASNetLarge(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return keras_applications.nasnet.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return keras_applications.nasnet.preprocess_input(*args, **kwargs)
