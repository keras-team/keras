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
"""Tests for tf.train.Checkpoint integration with DTensor using keras."""

import numpy as np
import tensorflow.compat.v2 as tf

from keras import backend
from keras import initializers
from keras import layers
from keras.dtensor import dtensor_api as dtensor
from keras.dtensor import test_util
from keras.utils import tf_utils

# isort: off
from tensorflow.python.eager import test


@test.mock.patch.dict("os.environ", {"DTENSOR_ENABLE_CHECKPOINT_V2": "True"})
class CheckpointTest(test_util.DTensorBaseTest):
    def setUp(self):
        super().setUp()
        backend.enable_tf_random_generator()
        tf_utils.set_random_seed(1337)
        global_ids = test_util.create_device_ids_array((2, 2))
        local_device_ids = np.ravel(global_ids).tolist()
        mesh_dict = {
            "CPU": dtensor.Mesh(
                ["X", "Y"],
                global_ids,
                local_device_ids,
                test_util.create_device_list((2, 2), "CPU"),
            )
        }
        self.mesh = self.configTestMesh(mesh_dict)
        self.layout_2d = dtensor.Layout.replicated(self.mesh, rank=2)
        self.layout_1d = dtensor.Layout.replicated(self.mesh, rank=1)

        self.sharded_2d = dtensor.Layout.batch_sharded(self.mesh, "X", rank=2)
        self.sharded_1d = dtensor.Layout.batch_sharded(self.mesh, "X", rank=1)

    def test_checkpoint_late_variable_creation(self):
        class SimpleLayer(layers.Layer):
            def __init__(self):
                super().__init__()

            def create_weights(self, layout_one, layout_two):
                self.weight_one = self.add_weight(
                    shape=[4, 8],
                    name="one",
                    initializer=initializers.RandomUniform(),
                    layout=layout_one,
                )
                self.weight_two = self.add_weight(
                    shape=[8, 16],
                    name="two",
                    initializer=initializers.RandomUniform(),
                    layout=layout_two,
                )

        layer = SimpleLayer()
        layer.create_weights(self.layout_2d, self.layout_2d)

        ckpt = tf.train.Checkpoint(layer=layer)
        options = tf.train.CheckpointOptions(
            experimental_io_device=dtensor.device_name()
        )

        # Checkpoint the layer.
        with dtensor.run_on(self.mesh.host_mesh()):
            save_path = ckpt.save(self.get_temp_dir(), options=options)

        # Create a new layer
        layer_uncreated_weights = SimpleLayer()

        # Restore on the uncreated weights layer.
        new_ckpt = tf.train.Checkpoint(layer=layer_uncreated_weights)
        with dtensor.run_on(self.mesh.host_mesh()):
            new_ckpt.restore(save_path, options=options)

        # Now create the weights.
        layer_uncreated_weights.create_weights(self.layout_2d, self.sharded_2d)

        restored_one = layer_uncreated_weights.weight_one
        restored_two = layer_uncreated_weights.weight_two

        # Check correctness of the restored weights.
        self.assertEqual(dtensor.fetch_layout(restored_one), self.layout_2d)
        self.assertEqual(dtensor.fetch_layout(restored_two), self.sharded_2d)

        self.assertAllClose(
            dtensor.relayout(restored_one, self.layout_2d), layer.weight_one
        )
        self.assertAllClose(
            dtensor.relayout(restored_two, self.layout_2d), layer.weight_two
        )


if __name__ == "__main__":
    tf.test.main()
