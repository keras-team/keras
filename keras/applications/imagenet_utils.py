"""Utilities for ImageNet data preprocessing & prediction decoding.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import imagenet_utils

preprocess_input = imagenet_utils.preprocess_input
decode_predictions = imagenet_utils.decode_predictions
