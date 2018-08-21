"""Utilities for ImageNet data preprocessing & prediction decoding.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import imagenet_utils
from . import keras_modules_injection


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return imagenet_utils.decode_predictions(
        *args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return imagenet_utils.preprocess_input(*args, **kwargs)
