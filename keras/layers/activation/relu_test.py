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
"""Tests for ReLU layer."""

import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.run_all_keras_modes
class ReLUTest(test_combinations.TestCase):
    def test_relu(self):
        test_utils.layer_test(
            keras.layers.ReLU,
            kwargs={"max_value": 10},
            input_shape=(2, 3, 4),
            supports_masking=True,
        )
        x = keras.backend.ones((3, 4))
        if not tf.executing_eagerly():
            # Test that we use `leaky_relu` when appropriate in graph mode.
            self.assertIn(
                "LeakyRelu", keras.layers.ReLU(negative_slope=0.2)(x).name
            )
            # Test that we use `relu` when appropriate in graph mode.
            self.assertIn("Relu", keras.layers.ReLU()(x).name)
            # Test that we use `relu6` when appropriate in graph mode.
            self.assertIn("Relu6", keras.layers.ReLU(max_value=6)(x).name)

    def test_relu_with_invalid_max_value(self):
        with self.assertRaisesRegex(
            ValueError,
            "max_value of a ReLU layer cannot be a negative "
            "value. Received: -10",
        ):
            test_utils.layer_test(
                keras.layers.ReLU,
                kwargs={"max_value": -10},
                input_shape=(2, 3, 4),
                supports_masking=True,
            )

    def test_relu_with_invalid_negative_slope(self):
        with self.assertRaisesRegex(
            ValueError,
            "negative_slope of a ReLU layer cannot be a negative "
            "value. Received: None",
        ):
            test_utils.layer_test(
                keras.layers.ReLU,
                kwargs={"negative_slope": None},
                input_shape=(2, 3, 4),
                supports_masking=True,
            )

        with self.assertRaisesRegex(
            ValueError,
            "negative_slope of a ReLU layer cannot be a negative "
            "value. Received: -10",
        ):
            test_utils.layer_test(
                keras.layers.ReLU,
                kwargs={"negative_slope": -10},
                input_shape=(2, 3, 4),
                supports_masking=True,
            )

    def test_relu_with_invalid_threshold(self):
        with self.assertRaisesRegex(
            ValueError,
            "threshold of a ReLU layer cannot be a negative "
            "value. Received: None",
        ):
            test_utils.layer_test(
                keras.layers.ReLU,
                kwargs={"threshold": None},
                input_shape=(2, 3, 4),
                supports_masking=True,
            )

        with self.assertRaisesRegex(
            ValueError,
            "threshold of a ReLU layer cannot be a negative "
            "value. Received: -10",
        ):
            test_utils.layer_test(
                keras.layers.ReLU,
                kwargs={"threshold": -10},
                input_shape=(2, 3, 4),
                supports_masking=True,
            )

    @test_combinations.run_with_all_model_types
    def test_relu_layer_as_activation(self):
        layer = keras.layers.Dense(1, activation=keras.layers.ReLU())
        model = test_utils.get_model_from_layers([layer], input_shape=(10,))
        model.compile("sgd", "mse", run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.ones((10, 10)), np.ones((10, 1)), batch_size=2)


if __name__ == "__main__":
    tf.test.main()
