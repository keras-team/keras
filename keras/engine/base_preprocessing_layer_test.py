# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras' base preprocessing layer."""

import os

import keras
from keras import keras_parameterized
from keras import testing_utils
from keras.engine import base_preprocessing_layer
import numpy as np
import tensorflow.compat.v2 as tf


# Define a test-only implementation of BasePreprocessingLayer to validate
# its correctness directly.
class AddingPreprocessingLayer(base_preprocessing_layer.PreprocessingLayer):

  def build(self, input_shape):
    super(AddingPreprocessingLayer, self).build(input_shape)
    self.sum = tf.Variable(0., dtype=tf.float32)

  def update_state(self, data):
    self.sum.assign_add(tf.reduce_sum(tf.cast(data, tf.float32)))

  def reset_state(self):  # pylint: disable=method-hidden
    self.sum.assign(0.)

  def set_total(self, sum_value):
    """This is an example of how a subclass would implement a direct setter.

    Args:
      sum_value: The total to set.
    """
    self.sum.assign(sum_value)

  def call(self, inputs):
    return inputs + self.sum


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class PreprocessingLayerTest(keras_parameterized.TestCase):

  def test_adapt_bad_input_fails(self):
    """Test that non-Dataset/Numpy inputs cause a reasonable error."""
    input_dataset = {"foo": 0}

    layer = AddingPreprocessingLayer()
    if tf.executing_eagerly():
      with self.assertRaisesRegex(ValueError, "Failed to find data adapter"):
        layer.adapt(input_dataset)
    else:
      with self.assertRaisesRegex(ValueError, "requires a"):
        layer.adapt(input_dataset)

  def test_adapt_infinite_dataset_fails(self):
    """Test that preproc layers fail if an infinite dataset is passed."""
    input_dataset = tf.data.Dataset.from_tensor_slices(
        np.array([[1], [2], [3], [4], [5], [0]])).repeat()

    layer = AddingPreprocessingLayer()
    if tf.executing_eagerly():
      with self.assertRaisesRegex(ValueError, "infinite dataset"):
        layer.adapt(input_dataset)
    else:
      with self.assertRaisesRegex(ValueError,
                                  ".*infinite number of elements.*"):
        layer.adapt(input_dataset)

  def test_setter_update(self):
    """Test the prototyped setter method."""
    input_data = keras.Input(shape=(1,))
    layer = AddingPreprocessingLayer()
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()

    layer.set_total(15)

    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

  def test_pre_build_adapt_update_numpy(self):
    """Test that preproc layers can adapt() before build() is called."""
    input_dataset = np.array([1, 2, 3, 4, 5])

    layer = AddingPreprocessingLayer()
    layer.adapt(input_dataset)

    input_data = keras.Input(shape=(1,))
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()

    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

  def test_post_build_adapt_update_numpy(self):
    """Test that preproc layers can adapt() after build() is called."""
    input_dataset = np.array([1, 2, 3, 4, 5])

    input_data = keras.Input(shape=(1,))
    layer = AddingPreprocessingLayer()
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()

    layer.adapt(input_dataset)

    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

  def test_pre_build_adapt_update_dataset(self):
    """Test that preproc layers can adapt() before build() is called."""
    input_dataset = tf.data.Dataset.from_tensor_slices(
        np.array([[1], [2], [3], [4], [5], [0]]))

    layer = AddingPreprocessingLayer()
    layer.adapt(input_dataset)

    input_data = keras.Input(shape=(1,))
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()

    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

  def test_post_build_adapt_update_dataset(self):
    """Test that preproc layers can adapt() after build() is called."""
    input_dataset = tf.data.Dataset.from_tensor_slices(
        np.array([[1], [2], [3], [4], [5], [0]]))

    input_data = keras.Input(shape=(1,))
    layer = AddingPreprocessingLayer()
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()

    layer.adapt(input_dataset)

    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

  def test_weight_based_state_transfer(self):
    """Test that preproc layers can transfer state via get/set weights.."""

    def get_model():
      input_data = keras.Input(shape=(1,))
      layer = AddingPreprocessingLayer()
      output = layer(input_data)
      model = keras.Model(input_data, output)
      model._run_eagerly = testing_utils.should_run_eagerly()
      return (model, layer)

    input_dataset = np.array([1, 2, 3, 4, 5])
    model, layer = get_model()
    layer.adapt(input_dataset)
    self.assertAllEqual([[16], [17], [18]], model.predict([1., 2., 3.]))

    # Create a new model and verify it has no state carryover.
    weights = model.get_weights()
    model_2, _ = get_model()
    self.assertAllEqual([[1], [2], [3]], model_2.predict([1., 2., 3.]))

    # Transfer state from model to model_2 via get/set weights.
    model_2.set_weights(weights)
    self.assertAllEqual([[16], [17], [18]], model_2.predict([1., 2., 3.]))

  def test_loading_without_providing_class_fails(self):
    input_data = keras.Input(shape=(1,))
    layer = AddingPreprocessingLayer()
    output = layer(input_data)
    model = keras.Model(input_data, output)

    if not tf.executing_eagerly():
      self.evaluate(tf.compat.v1.variables_initializer(model.variables))

    output_path = os.path.join(self.get_temp_dir(), "tf_keras_saved_model")
    model.save(output_path, save_format="tf")

    with self.assertRaisesRegex(RuntimeError, "Unable to restore a layer of"):
      _ = keras.models.load_model(output_path)

  def test_adapt_sets_input_shape_rank(self):
    """Check that `.adapt()` sets the `input_shape`'s rank."""
    # Shape: (3,1,2)
    adapt_dataset = np.array([[[1., 2.]], [[3., 4.]], [[5., 6.]]],
                             dtype=np.float32)

    layer = AddingPreprocessingLayer()
    layer.adapt(adapt_dataset)

    input_dataset = np.array([[[1., 2.], [3., 4.]], [[3., 4.], [5., 6.]]],
                             dtype=np.float32)
    layer(input_dataset)

    model = keras.Sequential([layer])
    self.assertTrue(model.built)
    self.assertEqual(model.input_shape, (None, None, None))

  def test_adapt_doesnt_overwrite_input_shape(self):
    """Check that `.adapt()` doesn't change the `input_shape`."""
    # Shape: (3, 1, 2)
    adapt_dataset = np.array([[[1., 2.]], [[3., 4.]], [[5., 6.]]],
                             dtype=np.float32)

    layer = AddingPreprocessingLayer(input_shape=[1, 2])
    layer.adapt(adapt_dataset)

    model = keras.Sequential([layer])
    self.assertTrue(model.built)
    self.assertEqual(model.input_shape, (None, 1, 2))


class PreprocessingLayerV1Test(keras_parameterized.TestCase):

  def test_adapt_fails(self):
    """Test that calling adapt leads to a runtime error."""
    input_dataset = {"foo": 0}

    with tf.Graph().as_default():
      layer = AddingPreprocessingLayer()
      with self.assertRaisesRegex(RuntimeError,
                                  "`adapt` is only supported in tensorflow v2"):
        layer.adapt(input_dataset)


if __name__ == "__main__":
  tf.test.main()
