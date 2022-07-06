# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Unit Normalization layer."""


import tensorflow.compat.v2 as tf

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


def squared_l2_norm(x):
    return tf.reduce_sum(x**2)


@test_utils.run_v2_only
class UnitNormalizationTest(test_combinations.TestCase):
    @test_combinations.run_all_keras_modes
    def test_basics(self):
        test_utils.layer_test(
            keras.layers.UnitNormalization,
            kwargs={"axis": -1},
            input_shape=(2, 3),
        )
        test_utils.layer_test(
            keras.layers.UnitNormalization,
            kwargs={"axis": (1, 2)},
            input_shape=(1, 3, 3),
        )

    def test_correctness(self):
        layer = keras.layers.UnitNormalization(axis=-1)
        inputs = tf.random.normal(shape=(2, 3))
        outputs = layer(inputs).numpy()
        self.assertAllClose(squared_l2_norm(outputs[0, :]), 1.0)
        self.assertAllClose(squared_l2_norm(outputs[1, :]), 1.0)

        layer = keras.layers.UnitNormalization(axis=(1, 2))
        inputs = tf.random.normal(shape=(2, 3, 3))
        outputs = layer(inputs).numpy()
        self.assertAllClose(squared_l2_norm(outputs[0, :, :]), 1.0)
        self.assertAllClose(squared_l2_norm(outputs[1, :, :]), 1.0)

        layer = keras.layers.UnitNormalization(axis=1)
        inputs = tf.random.normal(shape=(2, 3, 2))
        outputs = layer(inputs).numpy()
        self.assertAllClose(squared_l2_norm(outputs[0, :, 0]), 1.0)
        self.assertAllClose(squared_l2_norm(outputs[1, :, 0]), 1.0)
        self.assertAllClose(squared_l2_norm(outputs[0, :, 1]), 1.0)
        self.assertAllClose(squared_l2_norm(outputs[1, :, 1]), 1.0)

    def testInvalidAxis(self):
        with self.assertRaisesRegex(
            TypeError, r"Invalid value for `axis` argument"
        ):
            layer = keras.layers.UnitNormalization(axis=None)

        with self.assertRaisesRegex(
            ValueError, r"Invalid value for `axis` argument"
        ):
            layer = keras.layers.UnitNormalization(axis=3)
            layer.build(input_shape=(2, 2, 2))


if __name__ == "__main__":
    tf.test.main()
