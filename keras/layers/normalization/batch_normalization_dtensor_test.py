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

from keras.dtensor import test_util
from keras.layers.normalization import batch_normalization

# isort: off
# Import the MirroredStrategy that is backed by DTensor
# It is not a public API yet, so we do a private symbol import for now.
from tensorflow.python.distribute.experimental import (
    mirrored_strategy,
)


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
        strategy = mirrored_strategy.MirroredStrategy(self.mesh)

        with strategy.scope():
            self.assertTrue(
                batch_normalization._running_with_dtensor_strategy()
            )

        self.assertFalse(batch_normalization._running_with_dtensor_strategy())

        normal_mirrored_strategy = tf.distribute.MirroredStrategy(
            ["CPU:0", "CPU:1"]
        )
        self.assertFalse(batch_normalization._running_with_dtensor_strategy())
        with normal_mirrored_strategy.scope():
            self.assertFalse(
                batch_normalization._running_with_dtensor_strategy()
            )


if __name__ == "__main__":
    tf.test.main()
