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
"""Thresholded Rectified Linear Unit activation layer."""


import tensorflow.compat.v2 as tf

from tf_keras.src import backend
from tf_keras.src.engine.base_layer import Layer
from tf_keras.src.utils import tf_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.ThresholdedReLU")
class ThresholdedReLU(Layer):
    """Thresholded Rectified Linear Unit.

    It follows:

    ```
        f(x) = x for x > theta
        f(x) = 0 otherwise`
    ```

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Args:
        theta: Float >= 0. Threshold location of activation.
    """

    def __init__(self, theta=1.0, **kwargs):
        super().__init__(**kwargs)
        if theta is None:
            raise ValueError(
                "Theta of a Thresholded ReLU layer cannot be None, expecting a "
                f"float. Received: {theta}"
            )
        if theta < 0:
            raise ValueError(
                "The theta value of a Thresholded ReLU layer "
                f"should be >=0. Received: {theta}"
            )
        self.supports_masking = True
        self.theta = backend.cast_to_floatx(theta)

    def call(self, inputs):
        dtype = self.compute_dtype
        return inputs * tf.cast(tf.greater(inputs, self.theta), dtype)

    def get_config(self):
        config = {"theta": float(self.theta)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

