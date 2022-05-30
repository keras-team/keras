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
"""Distribution tests for keras.layers.preprocessing.normalization."""


import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras.distribute import strategy_combinations
from keras.layers.preprocessing import normalization
from keras.layers.preprocessing import preprocessing_test_utils
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


def _get_layer_computation_test_cases():
    test_cases = (
        {
            "adapt_data": np.array(
                [[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32
            ),
            "axis": -1,
            "test_data": np.array([[1.0], [2.0], [3.0]], np.float32),
            "expected": np.array([[-1.414214], [-0.707107], [0]], np.float32),
            "testcase_name": "2d_single_element",
        },
        {
            "adapt_data": np.array(
                [[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32
            ),
            "axis": None,
            "test_data": np.array([[1.0], [2.0], [3.0]], np.float32),
            "expected": np.array([[-1.414214], [-0.707107], [0]], np.float32),
            "testcase_name": "2d_single_element_none_axis",
        },
        {
            "adapt_data": np.array(
                [[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32
            ),
            "axis": None,
            "test_data": np.array([[1.0], [2.0], [3.0]], np.float32),
            "expected": np.array([[-1.414214], [-0.707107], [0]], np.float32),
            "testcase_name": "2d_single_element_none_axis_flat_data",
        },
        {
            "adapt_data": np.array(
                [
                    [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
                    [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
                ],
                np.float32,
            ),
            "axis": 1,
            "test_data": np.array(
                [
                    [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
                    [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
                ],
                np.float32,
            ),
            "expected": np.array(
                [
                    [[-1.549193, -0.774597, 0.0], [-1.549193, -0.774597, 0.0]],
                    [[0.0, 0.774597, 1.549193], [0.0, 0.774597, 1.549193]],
                ],
                np.float32,
            ),
            "testcase_name": "3d_internal_axis",
        },
        {
            "adapt_data": np.array(
                [
                    [[1.0, 0.0, 3.0], [2.0, 3.0, 4.0]],
                    [[3.0, -1.0, 5.0], [4.0, 5.0, 8.0]],
                ],
                np.float32,
            ),
            "axis": (1, 2),
            "test_data": np.array(
                [
                    [[3.0, 1.0, -1.0], [2.0, 5.0, 4.0]],
                    [[3.0, 0.0, 5.0], [2.0, 5.0, 8.0]],
                ],
                np.float32,
            ),
            "expected": np.array(
                [
                    [[1.0, 3.0, -5.0], [-1.0, 1.0, -1.0]],
                    [[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]],
                ],
                np.float32,
            ),
            "testcase_name": "3d_multiple_axis",
        },
    )

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


@test_utils.run_v2_only
@tf.__internal__.distribute.combinations.generate(
    tf.__internal__.test.combinations.times(
        tf.__internal__.test.combinations.combine(
            strategy=strategy_combinations.all_strategies
            + strategy_combinations.multi_worker_mirrored_strategies
            + strategy_combinations.parameter_server_strategies_single_worker
            + strategy_combinations.parameter_server_strategies_multi_worker,
            mode=["eager"],
        ),
        _get_layer_computation_test_cases(),
    )
)
class NormalizationTest(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    def test_layer_computation(
        self, strategy, adapt_data, axis, test_data, use_dataset, expected
    ):
        input_shape = tuple([None for _ in range(test_data.ndim - 1)])
        if use_dataset:
            # Keras APIs expect batched datasets
            adapt_data = tf.data.Dataset.from_tensor_slices(adapt_data).batch(2)
            test_data = tf.data.Dataset.from_tensor_slices(test_data).batch(2)

        with strategy.scope():
            input_data = keras.Input(shape=input_shape)
            layer = normalization.Normalization(axis=axis)
            layer.adapt(adapt_data)
            output = layer(input_data)
            model = keras.Model(input_data, output)
        output_data = model.predict(test_data)
        self.assertAllClose(expected, output_data)


if __name__ == "__main__":
    tf.__internal__.distribute.multi_process_runner.test_main()
