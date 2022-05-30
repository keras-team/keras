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
# ==============================================================================
"""Tests for keras.layers.preprocessing.hashing."""


import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras import backend
from keras.distribute import strategy_combinations
from keras.layers.preprocessing import hashing
from keras.layers.preprocessing import preprocessing_test_utils
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils

# isort: off
from tensorflow.python.framework import (
    test_util as tf_test_utils,
)


@test_utils.run_v2_only
@tf.__internal__.distribute.combinations.generate(
    tf.__internal__.test.combinations.combine(
        strategy=strategy_combinations.all_strategies
        + strategy_combinations.multi_worker_mirrored_strategies
        + strategy_combinations.parameter_server_strategies_single_worker
        + strategy_combinations.parameter_server_strategies_multi_worker,
        mode=["eager"],
    )
)
class HashingDistributionTest(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    def test_strategy(self, strategy):
        if (
            backend.is_tpu_strategy(strategy)
            and not tf_test_utils.is_mlir_bridge_enabled()
        ):
            self.skipTest("TPU tests require MLIR bridge")

        input_data = np.asarray([["omar"], ["stringer"], ["marlo"], ["wire"]])
        input_dataset = tf.data.Dataset.from_tensor_slices(input_data).batch(
            2, drop_remainder=True
        )
        expected_output = [[0], [0], [1], [0]]

        tf.config.set_soft_device_placement(True)

        with strategy.scope():
            input_data = keras.Input(shape=(None,), dtype=tf.string)
            layer = hashing.Hashing(num_bins=2)
            int_data = layer(input_data)
            model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(input_dataset)
        self.assertAllEqual(expected_output, output_dataset)


if __name__ == "__main__":
    tf.__internal__.distribute.multi_process_runner.test_main()
