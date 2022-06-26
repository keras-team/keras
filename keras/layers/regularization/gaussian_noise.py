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
"""Contains the GaussianNoise layer."""


import tensorflow.compat.v2 as tf

from keras import backend
from keras.engine import base_layer
from keras.utils import tf_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.GaussianNoise")
class GaussianNoise(base_layer.BaseRandomLayer):
    """Apply additive zero-centered Gaussian noise.

    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.

    As it is a regularization layer, it is only active at training time.

    Args:
      stddev: Float, standard deviation of the noise distribution.
      seed: Integer, optional random seed to enable deterministic behavior.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding noise) or in inference mode (doing nothing).

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
    """

    def __init__(self, stddev, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.supports_masking = True
        self.stddev = stddev
        self.seed = seed

    def call(self, inputs, training=None):
        def noised():
            return inputs + self._random_generator.random_normal(
                shape=tf.shape(inputs),
                mean=0.0,
                stddev=self.stddev,
                dtype=inputs.dtype,
            )

        return backend.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {"stddev": self.stddev, "seed": self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
