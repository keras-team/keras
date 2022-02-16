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
"""Tests for RNN cell wrappers v1 implementation."""

from absl.testing import parameterized
from keras.layers.rnn import legacy_cell_wrappers
from keras.layers.rnn import legacy_cells
from keras.testing_infra import test_combinations
import tensorflow.compat.v2 as tf


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class RNNCellWrapperV1Test(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      legacy_cell_wrappers.DropoutWrapper, legacy_cell_wrappers.ResidualWrapper
  ])
  def testWrapperKerasStyle(self, wrapper):
    """Tests if wrapper cell is instantiated in keras style scope."""
    wrapped_cell = wrapper(legacy_cells.BasicRNNCell(1))
    self.assertFalse(wrapped_cell._keras_style)


if __name__ == "__main__":
  tf.test.main()
