from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from keras_applications import resnext
except:
    resnext = None
from . import keras_modules_injection


@keras_modules_injection
def ResNeXt50(*args, **kwargs):
    return resnext.ResNeXt50(*args, **kwargs)


@keras_modules_injection
def ResNeXt101(*args, **kwargs):
    return resnext.ResNeXt101(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return resnext.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return resnext.preprocess_input(*args, **kwargs)
