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
"""Tests for keras model save/load."""

import numpy as np
import tensorflow.compat.v2 as tf

from keras import backend
from keras import layers
from keras import models
from keras.dtensor import dtensor_api as dtensor
from keras.dtensor import layout_map as layout_map_lib
from keras.dtensor import test_util
from keras.utils import tf_utils


def _create_test_model():
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            32,
            name="conv2d_1",
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(28, 28, 1),  # channel last gray scale input
        )
    )
    model.add(
        layers.Conv2D(
            64,
            name="conv2d_2",
            kernel_size=(3, 3),
            activation="relu",
        )
    )
    return model


class SaveLoadTest(test_util.DTensorBaseTest):
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

    def test_save_h5_weights_for_dtensor_model(self):
        layout_map = layout_map_lib.LayoutMap(mesh=self.mesh)
        with layout_map_lib.layout_map_scope(layout_map):
            dtensor_model = _create_test_model()

        self.assertNotEmpty(dtensor_model.weights)
        for w in dtensor_model.weights:
            # Make sure the weights are DVariable
            self.assertIsNotNone(w.layout)

        save_file = self.create_tempfile("dtensor_model.h5")
        dtensor_model.save_weights(save_file)

        # Make sure the weights can be load back to a normal keras model.
        normal_model = _create_test_model()
        normal_model.load_weights(save_file)

        for (
            w1,
            w2,
        ) in zip(normal_model.weights, dtensor_model.weights):
            self.assertAllClose(w1.numpy(), w2.numpy())
            self.assertIsNone(getattr(w1, "layout", None))

    def test_load_h5_weights_for_dtensor_model(self):
        normal_model = _create_test_model()

        save_file = self.create_tempfile("normal_model.h5")
        normal_model.save_weights(save_file)

        layout_map = layout_map_lib.LayoutMap(mesh=self.mesh)
        with layout_map_lib.layout_map_scope(layout_map):
            dtensor_model = _create_test_model()

        self.assertNotEmpty(dtensor_model.weights)
        for w in dtensor_model.weights:
            self.assertIsNotNone(w.layout)

        dtensor_model.load_weights(save_file)

        for (
            w1,
            w2,
        ) in zip(normal_model.weights, dtensor_model.weights):
            self.assertAllClose(w1.numpy(), w2.numpy())


if __name__ == "__main__":
    tf.test.main()
