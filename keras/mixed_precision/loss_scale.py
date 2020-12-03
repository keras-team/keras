# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains keras-specific LossScale functionality.

This functions cannot be in the non-keras loss_scale.py file since they depend
on keras, and files outside of keras should not depend on files inside keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import six

from keras.utils import generic_utils


def serialize(loss_scale):
  return generic_utils.serialize_keras_object(loss_scale)


def deserialize(config, custom_objects=None):
  loss_scale_module_objects = {
      'FixedLossScale': tf.mixed_precision.experimental.FixedLossScale,
      'DynamicLossScale': tf.mixed_precision.experimental.DynamicLossScale,
  }

  return generic_utils.deserialize_keras_object(
      config,
      module_objects=loss_scale_module_objects,
      custom_objects=custom_objects,
      printable_module_name='loss scale'
  )


def get(identifier):
  """Get a loss scale object."""
  if isinstance(identifier, dict):
    return deserialize(identifier)

  if isinstance(identifier, six.integer_types + (float,)):
    return tf.mixed_precision.experimental.FixedLossScale(identifier)
  if identifier == 'dynamic':
    return tf.mixed_precision.experimental.DynamicLossScale()
  if isinstance(identifier, tf.mixed_precision.experimental.LossScale):
    return identifier
  elif identifier is None:
    return None
  else:
    raise ValueError('Could not interpret loss scale identifier: %s' %
                     identifier)
