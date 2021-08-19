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
"""RaggedKerasTensor tests."""

import tensorflow.compat.v2 as tf

from absl.testing import parameterized
from keras import keras_parameterized
from keras import layers
from keras.engine import training


class SparseKerasTensorTest(keras_parameterized.TestCase):
  @parameterized.parameters(
      {'property_name': 'values'},
      {'property_name': 'indices'},
      {'property_name': 'dense_shape'},
  )
  def test_instance_property(self, property_name):
    inp = layers.Input(shape=[3], sparse=True)
    out = getattr(inp, property_name)
    model = training.Model(inp, out)

    x = tf.SparseTensor([[0, 0], [0, 1], [1, 1], [1, 2]], [1, 2, 3, 4], [2, 3])
    expected_property = getattr(x, property_name)
    self.assertAllEqual(model(x), expected_property)

    # Test that it works with serialization and deserialization as well
    model_config = model.get_config()
    model2 = training.Model.from_config(model_config)
    self.assertAllEqual(model2(x), expected_property)


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  tf.compat.v1.enable_v2_tensorshape()
  tf.test.main()
