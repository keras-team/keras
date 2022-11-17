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
"""Contains the Identity layer."""

from keras.engine.base_layer import Layer

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Identity")
class Identity(Layer):
    """
    Identity layer.

    A layer which returns inputs irrespective or their shape or datatypes.
    The call function is argument-insensitive.

    Args:
        None
    """

    def __init__(*args, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, *args, **kwargs):
        return inputs
