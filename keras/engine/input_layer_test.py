# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
#,============================================================================
"""Tests for InputLayer construction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras import combinations
from keras import keras_parameterized
from keras import testing_utils
from keras.engine import functional
from keras.engine import input_layer as input_layer_lib


class InputLayerTest(keras_parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testBasicOutputShapeNoBatchSize(self):
    # Create a Keras Input
    x = input_layer_lib.Input(shape=(32,), name='input_a')
    self.assertAllEqual(x.shape.as_list(), [None, 32])

    # Verify you can construct and use a model w/ this input
    model = functional.Functional(x, x * 2.0)
    self.assertAllEqual(model(tf.ones((3, 32))),
                        tf.ones((3, 32)) * 2.0)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testBasicOutputShapeWithBatchSize(self):
    # Create a Keras Input
    x = input_layer_lib.Input(batch_size=6, shape=(32,), name='input_b')
    self.assertAllEqual(x.shape.as_list(), [6, 32])

    # Verify you can construct and use a model w/ this input
    model = functional.Functional(x, x * 2.0)
    self.assertAllEqual(model(tf.ones(x.shape)),
                        tf.ones(x.shape) * 2.0)

  @combinations.generate(combinations.combine(mode=['eager']))
  def testBasicOutputShapeNoBatchSizeInTFFunction(self):
    model = None
    @tf.function
    def run_model(inp):
      nonlocal model
      if not model:
        # Create a Keras Input
        x = input_layer_lib.Input(shape=(8,), name='input_a')
        self.assertAllEqual(x.shape.as_list(), [None, 8])

        # Verify you can construct and use a model w/ this input
        model = functional.Functional(x, x * 2.0)
      return model(inp)

    self.assertAllEqual(run_model(tf.ones((10, 8))),
                        tf.ones((10, 8)) * 2.0)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInputTensorArg(self):
    with testing_utils.use_keras_tensors_scope(True):
      # Create a Keras Input
      x = input_layer_lib.Input(tensor=tf.zeros((7, 32)))
      self.assertAllEqual(x.shape.as_list(), [7, 32])

      # Verify you can construct and use a model w/ this input
      model = functional.Functional(x, x * 2.0)
      self.assertAllEqual(model(tf.ones(x.shape)),
                          tf.ones(x.shape) * 2.0)

  @combinations.generate(combinations.combine(mode=['eager']))
  def testInputTensorArgInTFFunction(self):
    with testing_utils.use_keras_tensors_scope(True):
      # We use a mutable model container instead of a model python variable,
      # because python 2.7 does not have `nonlocal`
      model_container = {}

      @tf.function
      def run_model(inp):
        if not model_container:
          # Create a Keras Input
          x = input_layer_lib.Input(tensor=tf.zeros((10, 16)))
          self.assertAllEqual(x.shape.as_list(), [10, 16])

          # Verify you can construct and use a model w/ this input
          model_container['model'] = functional.Functional(x, x * 3.0)
        return model_container['model'](inp)

      self.assertAllEqual(run_model(tf.ones((10, 16))),
                          tf.ones((10, 16)) * 3.0)

  @combinations.generate(combinations.combine(mode=['eager']))
  def testCompositeInputTensorArg(self):
    with testing_utils.use_keras_tensors_scope(True):
      # Create a Keras Input
      rt = tf.RaggedTensor.from_row_splits(
          values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
      x = input_layer_lib.Input(tensor=rt)

      # Verify you can construct and use a model w/ this input
      model = functional.Functional(x, x * 2)

      # And that the model works
      rt = tf.RaggedTensor.from_row_splits(
          values=[3, 21, 4, 1, 53, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
      self.assertAllEqual(model(rt), rt * 2)

  @combinations.generate(combinations.combine(mode=['eager']))
  def testCompositeInputTensorArgInTFFunction(self):
    with testing_utils.use_keras_tensors_scope(True):
      # We use a mutable model container instead of a model python variable,
      # because python 2.7 does not have `nonlocal`
      model_container = {}

      @tf.function
      def run_model(inp):
        if not model_container:
          # Create a Keras Input
          rt = tf.RaggedTensor.from_row_splits(
              values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
          x = input_layer_lib.Input(tensor=rt)

          # Verify you can construct and use a model w/ this input
          model_container['model'] = functional.Functional(x, x * 3)
        return model_container['model'](inp)

      # And verify the model works
      rt = tf.RaggedTensor.from_row_splits(
          values=[3, 21, 4, 1, 53, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
      self.assertAllEqual(run_model(rt), rt * 3)

if __name__ == '__main__':
  tf.test.main()
