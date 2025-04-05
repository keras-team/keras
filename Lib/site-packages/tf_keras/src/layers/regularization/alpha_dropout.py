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
"""Contains the AlphaDropout layer."""


import tensorflow.compat.v2 as tf

from tf_keras.src import backend
from tf_keras.src.engine import base_layer
from tf_keras.src.utils import tf_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.AlphaDropout")
class AlphaDropout(base_layer.BaseRandomLayer):
    """Applies Alpha Dropout to the input.

    Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
    to their original values, in order to ensure the self-normalizing property
    even after this dropout.
    Alpha Dropout fits well to Scaled Exponential Linear Units
    by randomly setting activations to the negative saturation value.

    Args:
      rate: float, drop probability (as with `Dropout`).
        The multiplicative noise will have
        standard deviation `sqrt(rate / (1 - rate))`.
      seed: Integer, optional random seed to enable deterministic behavior.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
    """

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        return self.noise_shape if self.noise_shape else tf.shape(inputs)

    def call(self, inputs, training=None):
        if 0.0 < self.rate < 1.0:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs(inputs=inputs, rate=self.rate):
                alpha = 1.6732632423543772848170429916717
                scale = 1.0507009873554804934193349852946
                alpha_p = -alpha * scale

                kept_idx = tf.greater_equal(
                    self._random_generator.random_uniform(noise_shape), rate
                )
                kept_idx = tf.cast(kept_idx, inputs.dtype)

                # Get affine transformation params
                a = ((1 - rate) * (1 + rate * alpha_p**2)) ** -0.5
                b = -a * alpha_p * rate

                # Apply mask
                x = inputs * kept_idx + alpha_p * (1 - kept_idx)

                # Do affine transformation
                return a * x + b

            return backend.in_train_phase(
                dropped_inputs, inputs, training=training
            )
        return inputs

    def get_config(self):
        config = {"rate": self.rate, "seed": self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

