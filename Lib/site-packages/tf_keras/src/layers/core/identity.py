# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Contains the Identity layer."""

import tensorflow.compat.v2 as tf

from tf_keras.src.engine.base_layer import Layer

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Identity")
class Identity(Layer):
    """Identity layer.

    This layer should be used as a placeholder when no operation is to be
    performed. The layer is argument insensitive, and returns its `inputs`
    argument as output.

    Args:
        name: Optional name for the layer instance.
    """

    def call(self, inputs):
        return tf.nest.map_structure(tf.identity, inputs)

