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
"""Tests for keras.layers.preprocessing.normalization."""

import tensorflow.compat.v2 as tf

import os

from absl.testing import parameterized

import numpy as np

import keras
from keras import keras_parameterized
from keras import testing_utils
from keras.layers.preprocessing import normalization
from keras.layers.preprocessing import preprocessing_test_utils


def _get_layer_computation_test_cases():
  test_cases = ({
      "adapt_data": np.array([[1.], [2.], [3.], [4.], [5.]], dtype=np.float32),
      "axis": -1,
      "test_data": np.array([[1.], [2.], [3.]], np.float32),
      "expected": np.array([[-1.414214], [-.707107], [0]], np.float32),
      "testcase_name": "2d_single_element"
  }, {
      "adapt_data": np.array([[1], [2], [3], [4], [5]], dtype=np.int32),
      "axis": -1,
      "test_data": np.array([[1], [2], [3]], np.int32),
      "expected": np.array([[-1.414214], [-.707107], [0]], np.float32),
      "testcase_name": "2d_int_data"
  }, {
      "adapt_data": np.array([[1.], [2.], [3.], [4.], [5.]], dtype=np.float32),
      "axis": None,
      "test_data": np.array([[1.], [2.], [3.]], np.float32),
      "expected": np.array([[-1.414214], [-.707107], [0]], np.float32),
      "testcase_name": "2d_single_element_none_axis"
  }, {
      "adapt_data": np.array([[1., 2., 3., 4., 5.]], dtype=np.float32),
      "axis": None,
      "test_data": np.array([[1.], [2.], [3.]], np.float32),
      "expected": np.array([[-1.414214], [-.707107], [0]], np.float32),
      "testcase_name": "2d_single_element_none_axis_flat_data"
  }, {
      "adapt_data":
          np.array([[[1., 2., 3.], [2., 3., 4.]], [[3., 4., 5.], [4., 5., 6.]]],
                   np.float32),
      "axis":
          1,
      "test_data":
          np.array([[[1., 2., 3.], [2., 3., 4.]], [[3., 4., 5.], [4., 5., 6.]]],
                   np.float32),
      "expected":
          np.array([[[-1.549193, -0.774597, 0.], [-1.549193, -0.774597, 0.]],
                    [[0., 0.774597, 1.549193], [0., 0.774597, 1.549193]]],
                   np.float32),
      "testcase_name":
          "3d_internal_axis"
  }, {
      "adapt_data":
          np.array(
              [[[1., 0., 3.], [2., 3., 4.]], [[3., -1., 5.], [4., 5., 8.]]],
              np.float32),
      "axis": (1, 2),
      "test_data":
          np.array(
              [[[3., 1., -1.], [2., 5., 4.]], [[3., 0., 5.], [2., 5., 8.]]],
              np.float32),
      "expected":
          np.array(
              [[[1., 3., -5.], [-1., 1., -1.]], [[1., 1., 1.], [-1., 1., 1.]]],
              np.float32),
      "testcase_name":
          "3d_multiple_axis"
  }, {
      "adapt_data":
          np.zeros((3, 4)),
      "axis": -1,
      "test_data":
          np.zeros((3, 4)),
      "expected":
          np.zeros((3, 4)),
      "testcase_name":
          "zero_variance"
  })

  crossed_test_cases = []
  # Cross above test cases with use_dataset in (True, False)
  for use_dataset in (True, False):
    for case in test_cases:
      case = case.copy()
      if use_dataset:
        case["testcase_name"] = case["testcase_name"] + "_with_dataset"
      case["use_dataset"] = use_dataset
      crossed_test_cases.append(case)

  return crossed_test_cases


@keras_parameterized.run_all_keras_modes
class NormalizationTest(keras_parameterized.TestCase,
                        preprocessing_test_utils.PreprocessingLayerTest):

  def test_broadcasting_during_direct_setting(self):
    layer = normalization.Normalization(axis=-1, mean=[1.0], variance=[1.0])
    output = layer(np.array([[1., 2.]]))
    expected_output = [[0., 1.]]
    self.assertAllClose(output, expected_output)
    self.assertAllClose(layer.get_weights(), [])

  def test_broadcasting_during_direct_setting_with_tensors(self):
    if not tf.executing_eagerly():
      self.skipTest("Only supported in TF2.")

    layer = normalization.Normalization(
        axis=-1,
        mean=tf.constant([1.0]),
        variance=tf.constant([1.0]))
    output = layer(np.array([[1., 2.]]))
    expected_output = [[0., 1.]]
    self.assertAllClose(output, expected_output)
    self.assertAllClose(layer.get_weights(), [])

  def test_1d_data(self):
    data = np.array([0., 2., 0., 2.])
    layer = normalization.Normalization(mean=1.0, variance=1.0)
    output = layer(data)
    self.assertListEqual(output.shape.as_list(), [4])
    self.assertAllClose(output, [-1, 1, -1, 1])

  def test_0d_data(self):
    layer = normalization.Normalization(axis=None, mean=1.0, variance=1.0)
    output = layer(0.)
    self.assertListEqual(output.shape.as_list(), [])
    self.assertAllClose(output, -1)

  def test_broadcasting_during_direct_setting_with_variables_fails(self):
    with self.assertRaisesRegex(ValueError, "passing a Variable"):
      _ = normalization.Normalization(
          axis=-1,
          mean=tf.Variable([1.0]),
          variance=tf.Variable([2.0]))

  def test_keeping_an_unknown_axis_fails(self):
    layer = normalization.Normalization(axis=-1)
    with self.assertRaisesRegex(ValueError, "axis.*must have known shape"):
      layer.build([None])

  @parameterized.parameters(
      # Out of bounds
      {"axis": 3},
      {"axis": -4},
      # In a tuple
      {"axis": (1, 3)},
      {"axis": (1, -4)},
  )
  def test_bad_axis_fail_build(self, axis):
    layer = normalization.Normalization(axis=axis)
    with self.assertRaisesRegex(ValueError, "in the range"):
      layer.build([None, 2, 3])

  def test_list_input(self):
    with self.assertRaisesRegex(
        ValueError, ("Normalization only accepts a single input. If you are "
                     "passing a python list or tuple as a single input, "
                     "please convert to a numpy array or `tf.Tensor`.")):
      normalization.Normalization()([1, 2, 3])

  def test_scalar_input(self):
    with self.assertRaisesRegex(ValueError,
                                "axis.*values must be in the range"):
      normalization.Normalization()(1)


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class NormalizationAdaptTest(keras_parameterized.TestCase,
                             preprocessing_test_utils.PreprocessingLayerTest):

  def test_layer_api_compatibility(self):
    cls = normalization.Normalization
    output_data = testing_utils.layer_test(
        cls,
        kwargs={"axis": -1},
        input_shape=(None, 3),
        input_data=np.array([[3, 1, 2], [6, 5, 4]], dtype=np.float32),
        validate_training=False,
        adapt_data=np.array([[1, 2, 1], [2, 3, 4], [1, 2, 1], [2, 3, 4]]))
    expected = np.array([[3., -3., -0.33333333], [9., 5., 1.]])
    self.assertAllClose(expected, output_data)

  @parameterized.named_parameters(*_get_layer_computation_test_cases())
  def test_layer_computation(self, adapt_data, axis, test_data, use_dataset,
                             expected):
    input_shape = tuple([test_data.shape[i] for i in range(1, test_data.ndim)])
    if use_dataset:
      # Keras APIs expect batched datasets
      adapt_data = tf.data.Dataset.from_tensor_slices(adapt_data).batch(
          test_data.shape[0] // 2)
      test_data = tf.data.Dataset.from_tensor_slices(test_data).batch(
          test_data.shape[0] // 2)

    layer = normalization.Normalization(axis=axis)
    layer.adapt(adapt_data)

    input_data = keras.Input(shape=input_shape)
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()
    output_data = model.predict(test_data)
    self.assertAllClose(expected, output_data)

  def test_1d_unbatched_adapt(self):
    ds = tf.data.Dataset.from_tensor_slices([
        [2., 0., 2., 0.],
        [0., 2., 0., 2.],
    ])
    layer = normalization.Normalization(axis=-1)
    layer.adapt(ds)
    output_ds = ds.map(layer)
    self.assertAllClose(
        list(output_ds.as_numpy_iterator()), [
            [1., -1., 1., -1.],
            [-1., 1., -1., 1.],
        ])

  def test_0d_unbatched_adapt(self):
    ds = tf.data.Dataset.from_tensor_slices([2., 0., 2., 0.])
    layer = normalization.Normalization(axis=None)
    layer.adapt(ds)
    output_ds = ds.map(layer)
    self.assertAllClose(list(output_ds.as_numpy_iterator()), [1., -1., 1., -1.])

  @parameterized.parameters(
      # Results should be identical no matter how the axes are specified (3d).
      {"axis": (1, 2)},
      {"axis": (2, 1)},
      {"axis": (1, -1)},
      {"axis": (-1, 1)},
  )
  def test_axis_permutations(self, axis):
    layer = normalization.Normalization(axis=axis)
    # data.shape = [2, 2, 3]
    data = np.array([[[0., 1., 2.], [0., 2., 6.]],
                     [[2., 3., 4.], [3., 6., 10.]]])
    expect = np.array([[[-1., -1., -1.], [-1., -1., -1.]],
                       [[1., 1., 1.], [1., 1., 1.]]])
    layer.adapt(data)
    self.assertAllClose(expect, layer(data))

  def test_model_summary_after_layer_adapt(self):
    data = np.array([[[0., 1., 2.], [0., 2., 6.]],
                     [[2., 3., 4.], [3., 6., 10.]]])
    layer = normalization.Normalization(axis=-1)
    layer.adapt(data)
    model = keras.Sequential(
        [layer,
         keras.layers.Dense(64, activation="relu"),
         keras.layers.Dense(1)])
    model.summary()

  def test_multiple_adapts(self):
    first_adapt = [[0], [2], [0], [2]]
    second_adapt = [[2], [4], [2], [4]]
    predict_input = [[2], [2]]
    expected_first_output = [[1], [1]]
    expected_second_output = [[-1], [-1]]

    inputs = keras.Input(shape=(1,), dtype=tf.int32)
    layer = normalization.Normalization(axis=-1)
    layer.adapt(first_adapt)
    outputs = layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    actual_output = model.predict(predict_input)
    self.assertAllClose(actual_output, expected_first_output)

    # Re-adapt the layer on new inputs.
    layer.adapt(second_adapt)
    # Re-compile the model.
    model.compile()
    # `predict` should now use the new model state.
    actual_output = model.predict(predict_input)
    self.assertAllClose(actual_output, expected_second_output)

  @parameterized.parameters(
      {"adapted": True},
      {"adapted": False},
  )
  def test_saved_model_tf(self, adapted):
    input_data = [[0.], [2.], [0.], [2.]]
    expected_output = [[-1.], [1.], [-1.], [1.]]

    inputs = keras.Input(shape=(1,), dtype=tf.float32)
    if adapted:
      layer = normalization.Normalization(axis=-1)
      layer.adapt(input_data)
    else:
      layer = normalization.Normalization(mean=1., variance=1.)
    outputs = layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    output_data = model.predict(input_data)
    self.assertAllClose(output_data, expected_output)

    # Save the model to disk.
    output_path = os.path.join(self.get_temp_dir(), "tf_saved_model")
    tf.saved_model.save(model, output_path)
    loaded_model = tf.saved_model.load(output_path)
    f = loaded_model.signatures["serving_default"]

    # Ensure that the loaded model is unique (so that the save/load is real)
    self.assertIsNot(model, loaded_model)

    # Validate correctness of the new model.
    new_output_data = f(tf.constant(input_data))["normalization"]
    self.assertAllClose(new_output_data, expected_output)

  @parameterized.parameters(
      {"adapted": True},
      {"adapted": False},
  )
  def test_saved_model_keras(self, adapted):
    input_data = [[0.], [2.], [0.], [2.]]
    expected_output = [[-1.], [1.], [-1.], [1.]]

    cls = normalization.Normalization
    inputs = keras.Input(shape=(1,), dtype=tf.float32)
    if adapted:
      layer = cls(axis=-1)
      layer.adapt(input_data)
    else:
      layer = cls(mean=1., variance=1.)
    outputs = layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    output_data = model.predict(input_data)
    self.assertAllClose(output_data, expected_output)

    # Save the model to disk.
    output_path = os.path.join(self.get_temp_dir(), "tf_keras_saved_model")
    model.save(output_path, save_format="tf")
    loaded_model = keras.models.load_model(
        output_path, custom_objects={"Normalization": cls})

    # Ensure that the loaded model is unique (so that the save/load is real)
    self.assertIsNot(model, loaded_model)

    # Validate correctness of the new model.
    new_output_data = loaded_model.predict(input_data)
    self.assertAllClose(new_output_data, expected_output)

  @parameterized.parameters(
      {"adapted": True},
      {"adapted": False},
  )
  def test_saved_weights_keras(self, adapted):
    input_data = [[0.], [2.], [0.], [2.]]
    expected_output = [[-1.], [1.], [-1.], [1.]]

    cls = normalization.Normalization
    inputs = keras.Input(shape=(1,), dtype=tf.float32)
    if adapted:
      layer = cls(axis=-1)
      layer.adapt(input_data)
    else:
      layer = cls(mean=1., variance=1.)
    outputs = layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    output_data = model.predict(input_data)
    self.assertAllClose(output_data, expected_output)

    # Save the model to disk.
    output_path = os.path.join(self.get_temp_dir(), "tf_keras_saved_weights")
    model.save_weights(output_path, save_format="tf")
    new_model = keras.Model.from_config(
        model.get_config(), custom_objects={"Normalization": cls})
    new_model.load_weights(output_path)

    # Validate correctness of the new model.
    new_output_data = new_model.predict(input_data)
    self.assertAllClose(new_output_data, expected_output)


if __name__ == "__main__":
  tf.test.main()
