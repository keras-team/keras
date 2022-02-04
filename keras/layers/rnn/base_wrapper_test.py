# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the Wrapper base class."""

from absl.testing import parameterized
import keras
import tensorflow.compat.v2 as tf


class ExampleWrapper(keras.layers.Wrapper):
  """Simple Wrapper subclass."""

  def call(self, inputs, *args, **kwargs):
    return self.layer(inputs, *args, **kwargs)


class WrapperTest(parameterized.TestCase):

  def test_wrapper_from_config_no_mutation(self):
    wrapper = ExampleWrapper(keras.layers.Dense(1))
    config = wrapper.get_config()
    config_copy = config.copy()
    self.assertEqual(config, config_copy)

    wrapper_from_config = ExampleWrapper.from_config(config)
    new_config = wrapper_from_config.get_config()
    self.assertEqual(new_config, config)
    self.assertEqual(new_config, config_copy)


if __name__ == '__main__':
  tf.test.main()
