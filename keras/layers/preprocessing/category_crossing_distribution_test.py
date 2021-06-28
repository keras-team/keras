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
"""Distribution tests for keras.layers.preprocessing.category_crossing."""

import tensorflow.compat.v2 as tf

import numpy as np

import keras
from keras import keras_parameterized
from keras.distribute.strategy_combinations import all_strategies
from keras.layers.preprocessing import category_crossing
from keras.layers.preprocessing import preprocessing_test_utils


def batch_wrapper(dataset, batch_size, distribution, repeat=None):
  if repeat:
    dataset = dataset.repeat(repeat)
  # TPUs currently require fully defined input shapes, drop_remainder ensures
  # the input will have fully defined shapes.
  if isinstance(distribution,
                (tf.distribute.experimental.TPUStrategy, tf.compat.v1.distribute.experimental.TPUStrategy)):
    return dataset.batch(batch_size, drop_remainder=True)
  else:
    return dataset.batch(batch_size)


@tf.__internal__.distribute.combinations.generate(
    tf.__internal__.test.combinations.combine(
        # Investigate why crossing is not supported with TPU.
        distribution=all_strategies,
        mode=['eager', 'graph']))
class CategoryCrossingDistributionTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_distribution(self, distribution):
    input_array_1 = np.array([['a', 'b'], ['c', 'd']])
    input_array_2 = np.array([['e', 'f'], ['g', 'h']])
    inp_dataset = tf.data.Dataset.from_tensor_slices(
        {'input_1': input_array_1, 'input_2': input_array_2})
    inp_dataset = batch_wrapper(inp_dataset, 2, distribution)

    # pyformat: disable
    expected_output = [[b'a_X_e', b'a_X_f', b'b_X_e', b'b_X_f'],
                       [b'c_X_g', b'c_X_h', b'd_X_g', b'd_X_h']]
    tf.config.set_soft_device_placement(True)

    with distribution.scope():
      input_data_1 = keras.Input(shape=(2,), dtype=tf.string,
                                 name='input_1')
      input_data_2 = keras.Input(shape=(2,), dtype=tf.string,
                                 name='input_2')
      input_data = [input_data_1, input_data_2]
      layer = category_crossing.CategoryCrossing()
      int_data = layer(input_data)
      model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(inp_dataset)
    self.assertAllEqual(expected_output, output_dataset)


if __name__ == '__main__':
  tf.test.main()
