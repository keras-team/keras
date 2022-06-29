# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Sigmoid activation layer."""

import tensorflow.compat.v2 as tf

from keras import backend
from keras.engine.base_layer import Layer
from keras.utils import tf_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


def _large_compatible_negative(tensor_type):
    """Large negative number as Tensor.

    This function is necessary because the standard value for epsilon
    in this module (-1e9) cannot be represented using tf.float16

    Args:
      tensor_type: a dtype to determine the type.

    Returns:
      a large negative number.
    """
    if tensor_type == tf.float16:
        return tf.float16.min
    return -1e9


@keras_export("keras.layers.Sigmoid")
class Sigmoid(Layer):
    """Sigmoid activation function.

	Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as the input.

    Args:
	  None
    Call arguments:
      inputs: The inputs, or logits to the sigmoid layer.
      mask: A boolean mask of the same shape as `inputs`. Defaults to `None`.
        The mask specifies 1 to keep and 0 to mask.
    Returns:
      Outputs with the same shape as `inputs` with sigmoid applied on every
      element.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            # Since mask is 1.0 for positions we want to keep and 0.0 for masked
            # positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -1e.9 for masked positions.
            adder = (1.0 - tf.cast(mask, inputs.dtype)) * (
                _large_compatible_negative(inputs.dtype))

            # Since we are adding it to the raw scores before the sigmoid, this
            # is effectively the same as removing these entirely.
            inputs += adder
        return backend.sigmoid(inputs)

    def get_config(self):
        return super().get_config()

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape