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
"""Contains the ActivityRegularization layer."""


from keras import regularizers
from keras.engine.base_layer import Layer

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.ActivityRegularization")
class ActivityRegularization(Layer):
    """Layer that applies an update to the cost function based input activity.

    Args:
      l1: L1 regularization factor (positive float).
      l2: L2 regularization factor (positive float).

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
    """

    def __init__(self, l1=0.0, l2=0.0, **kwargs):
        super().__init__(
            activity_regularizer=regularizers.L1L2(l1=l1, l2=l2), **kwargs
        )
        self.supports_masking = True
        self.l1 = l1
        self.l2 = l2

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"l1": self.l1, "l2": self.l2}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
