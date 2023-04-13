# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for normalization layers under DTensor context."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras.dtensor import test_util
from keras.dtensor import utils
from keras.layers.normalization import batch_normalization
from keras.testing_infra import test_utils

# isort: off
# Import the MirroredStrategy that is backed by DTensor
# It is not a public API yet, so we do a private symbol import for now.
from tensorflow.python.distribute.experimental import (
    mirrored_strategy as dtensor_mirrored_strategy,
)


@test_utils.run_v2_only
class BatchNormalizationDTensorTest(test_util.DTensorBaseTest):
    def setUp(self):
        super().setUp()

        global_ids = test_util.create_device_ids_array((2,))
        local_device_ids = np.ravel(global_ids).tolist()
        mesh_dict = {
            "CPU": tf.experimental.dtensor.Mesh(
                ["batch"],
                global_ids,
                local_device_ids,
                test_util.create_device_list((2,), "CPU"),
            )
        }
        self.mesh = self.configTestMesh(mesh_dict)

    def test_strategy_backed_by_dtensor(self):
        strategy = dtensor_mirrored_strategy.MirroredStrategy(self.mesh)

        with strategy.scope():
            self.assertTrue(utils.running_with_dtensor_strategy())

        self.assertFalse(utils.running_with_dtensor_strategy())

        normal_mirrored_strategy = tf.distribute.MirroredStrategy(
            ["CPU:0", "CPU:1"]
        )
        self.assertFalse(utils.running_with_dtensor_strategy())
        with normal_mirrored_strategy.scope():
            self.assertFalse(utils.running_with_dtensor_strategy())

    @parameterized.product(
        training=[True, False],
        synchronized=[True, False],
        renorm=[True, False],
    )
    def test_batch_normalization_with_dtensor_strategy(
        self, training, synchronized, renorm
    ):
        num_replica = 2
        local_batch_size = 4
        global_batch_size = num_replica * local_batch_size
        feature_shape = [3, 5]
        global_inputs = tf.random.uniform(
            shape=[global_batch_size, *feature_shape], dtype=tf.float32
        )
        replica_inputs = tf.reshape(
            global_inputs, [num_replica, local_batch_size, *feature_shape]
        )

        def value_fn(value_context):
            return replica_inputs[value_context.replica_id_in_sync_group]

        normal_strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])
        dtensor_strategy = dtensor_mirrored_strategy.MirroredStrategy(
            mesh=self.mesh
        )
        init_kwargs = {"synchronized": synchronized, "renorm": renorm}
        bn_layer_0 = batch_normalization.BatchNormalization(**init_kwargs)
        bn_layer_1 = batch_normalization.BatchNormalization(**init_kwargs)
        run_kwargs = {"training": training}

        normal_strategy_result = self._run_bn_training_with_strategy(
            normal_strategy, value_fn, bn_layer_0, run_kwargs
        )
        if training and not synchronized and renorm:
            # This is an unsupported case at the moment.
            with self.assertRaisesRegexp(NotImplementedError, "not supported"):
                self._run_bn_training_with_strategy(
                    dtensor_strategy, value_fn, bn_layer_1, run_kwargs
                )
            return
        else:
            dtensor_strategy_result = self._run_bn_training_with_strategy(
                dtensor_strategy, value_fn, bn_layer_1, run_kwargs
            )
        self.assertAllClose(
            normal_strategy_result.values, dtensor_strategy_result.values
        )
        self.assertAllClose(bn_layer_0.moving_mean, bn_layer_1.moving_mean)
        self.assertAllClose(
            bn_layer_0.moving_variance, bn_layer_1.moving_variance
        )

    def _run_bn_training_with_strategy(
        self, strategy, value_fn, bn_layer, run_kwargs
    ):
        @tf.function
        def run_fn(inputs):
            return bn_layer(inputs, **run_kwargs)

        distributed_inputs = (
            strategy.experimental_distribute_values_from_function(value_fn)
        )

        return strategy.run(run_fn, args=(distributed_inputs,))


if __name__ == "__main__":
    tf.test.main()
