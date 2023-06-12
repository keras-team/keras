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
"""Tests for Softmax layer."""

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.run_all_keras_modes
class SoftmaxTest(test_combinations.TestCase):
    def test_softmax(self):
        test_utils.layer_test(
            keras.layers.Softmax,
            kwargs={"axis": 1},
            input_shape=(2, 3, 4),
            supports_masking=True,
        )

    @parameterized.named_parameters(
        dict(
            testcase_name="masked_out_infinite_or_nan_input_is_not_nan",
            inputs=[float("inf"), float("nan"), 0.0],
            mask=[False, False, True],
            expected_activations=[0.0, 0.0, 1.0],
            expected_jacobian=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ),
        dict(
            testcase_name="gradient_preserved_through_mask",
            inputs=[0.0, 0.0, 0.0],
            mask=[False, True, True],
            expected_activations=[0.0, 0.5, 0.5],
            expected_jacobian=[
                [0.0, 0.0, 0.0],
                [0.0, 0.25, -0.25],
                [0.0, -0.25, 0.25],
            ],
        ),
    )
    def test_softmax_masked_activations_and_jacobian(
        self,
        inputs: list[float],
        mask: list[bool],
        expected_activations: list[float],
        expected_jacobian: list[list[float]],
    ):
        softmax = keras.layers.Softmax()
        inputs = tf.constant(inputs, dtype=tf.float32)
        mask = tf.constant(mask, dtype=tf.bool)
        expected_activations = tf.constant(
            expected_activations, dtype=tf.float32
        )
        expected_jacobian = tf.constant(expected_jacobian, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            # Softmax requires at least a 2D input, so we add a 0th axis to the
            # input and immediately strip it off from the output activations
            activations = softmax(inputs[tf.newaxis, :], mask)[0, :]

        jacobian = tape.jacobian(activations, inputs)

        self.assertAllEqual(
            activations,
            expected_activations,
        )
        self.assertAllEqual(
            jacobian,
            expected_jacobian,
        )


if __name__ == "__main__":
    tf.test.main()
