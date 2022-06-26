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
"""Leaky version of a Rectified Linear Unit activation layer."""


from keras import backend
from keras.engine.base_layer import Layer
from keras.utils import tf_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.LeakyReLU")
class LeakyReLU(Layer):
    """Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active:

    ```
      f(x) = alpha * x if x < 0
      f(x) = x if x >= 0
    ```

    Usage:

    >>> layer = tf.keras.layers.LeakyReLU()
    >>> output = layer([-3.0, -1.0, 0.0, 2.0])
    >>> list(output.numpy())
    [-0.9, -0.3, 0.0, 2.0]
    >>> layer = tf.keras.layers.LeakyReLU(alpha=0.1)
    >>> output = layer([-3.0, -1.0, 0.0, 2.0])
    >>> list(output.numpy())
    [-0.3, -0.1, 0.0, 2.0]

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the batch axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as the input.

    Args:
      alpha: Float >= 0. Negative slope coefficient. Default to 0.3.

    """

    def __init__(self, alpha=0.3, **kwargs):
        super().__init__(**kwargs)
        if alpha is None:
            raise ValueError(
                "The alpha value of a Leaky ReLU layer cannot be None, "
                f"Expecting a float. Received: {alpha}"
            )
        self.supports_masking = True
        self.alpha = backend.cast_to_floatx(alpha)

    def call(self, inputs):
        return backend.relu(inputs, alpha=self.alpha)

    def get_config(self):
        config = {"alpha": float(self.alpha)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
