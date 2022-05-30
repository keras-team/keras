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
"""Distribution tests for keras.layers.preprocessing.category_encoding."""


import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras import backend
from keras.distribute import strategy_combinations
from keras.layers.preprocessing import category_encoding
from keras.layers.preprocessing import preprocessing_test_utils
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils

# isort: off
from tensorflow.python.framework import (
    test_util as tf_test_utils,
)


def batch_wrapper(dataset, batch_size, strategy, repeat=None):
    if repeat:
        dataset = dataset.repeat(repeat)
    # TPUs currently require fully defined input shapes, drop_remainder ensures
    # the input will have fully defined shapes.
    if backend.is_tpu_strategy(strategy):
        return dataset.batch(batch_size, drop_remainder=True)
    else:
        return dataset.batch(batch_size)


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
class CategoryEncodingDistributionTest(
    test_combinations.TestCase, preprocessing_test_utils.PreprocessingLayerTest
):
    def test_strategy(self, strategy):
        if (
            backend.is_tpu_strategy(strategy)
            and not tf_test_utils.is_mlir_bridge_enabled()
        ):
            self.skipTest("TPU tests require MLIR bridge")

        input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])
        inp_dataset = tf.data.Dataset.from_tensor_slices(input_array)
        inp_dataset = batch_wrapper(inp_dataset, 2, strategy)

        # pyformat: disable
        expected_output = [[0, 1, 1, 1, 0, 0], [1, 1, 0, 1, 0, 0]]
        # pyformat: enable
        num_tokens = 6
        tf.config.set_soft_device_placement(True)

        with strategy.scope():
            input_data = keras.Input(shape=(4,), dtype=tf.int32)
            layer = category_encoding.CategoryEncoding(
                num_tokens=num_tokens, output_mode=category_encoding.MULTI_HOT
            )
            int_data = layer(input_data)
            model = keras.Model(inputs=input_data, outputs=int_data)
        output_dataset = model.predict(inp_dataset)
        self.assertAllEqual(expected_output, output_dataset)


if __name__ == "__main__":
    tf.__internal__.distribute.multi_process_runner.test_main()
