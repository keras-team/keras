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
"""E2E Tests for mnist_model."""

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2.experimental import dtensor

from keras import backend
from keras.dtensor import integration_test_utils
from keras.dtensor import layout_map as layout_map_lib
from keras.dtensor import test_util
from keras.optimizers import adam
from keras.utils import tf_utils


class MnistTest(test_util.DTensorBaseTest):
    def setUp(self):
        super().setUp()
        backend.enable_tf_random_generator()
        tf_utils.set_random_seed(1337)
        global_ids = test_util.create_device_ids_array((2,))
        local_device_ids = np.ravel(global_ids).tolist()
        mesh_dict = {
            device: tf.experimental.dtensor.Mesh(
                ["batch"],
                global_ids,
                local_device_ids,
                test_util.create_device_list((2,), device),
            )
            for device in ("CPU", "GPU", "TPU")
        }
        self.mesh = self.configTestMesh(mesh_dict)

    def test_mnist_training(self):
        layout_map = layout_map_lib.LayoutMap(self.mesh)
        with layout_map.scope():
            model = integration_test_utils.get_model()

        optimizer = adam.Adam(learning_rate=0.001, mesh=self.mesh)
        optimizer.build(model.trainable_variables)

        train_losses = integration_test_utils.train_mnist_model_batch_sharded(
            model,
            optimizer,
            self.mesh,
            num_epochs=3,
            steps_per_epoch=20,
            global_batch_size=64,
        )
        # Make sure the losses are decreasing
        self.assertEqual(train_losses, sorted(train_losses, reverse=True))

    def test_model_fit(self):
        if self.mesh.device_type() == "GPU":
            self.skipTest("TODO(b/292596476)")

        layout_map = layout_map_lib.LayoutMap(self.mesh)
        with layout_map.scope():
            model = integration_test_utils.get_model()

        optimizer = adam.Adam(learning_rate=0.001, mesh=self.mesh)
        optimizer.build(model.trainable_variables)

        global_batch_size = 64
        model.compile(
            loss="CategoricalCrossentropy", optimizer=optimizer, metrics="acc"
        )
        train_ds, eval_ds = integration_test_utils.get_mnist_datasets(
            integration_test_utils.NUM_CLASS, global_batch_size
        )

        def distribute_ds(dataset):
            dataset = dataset.unbatch()

            def _create_batch_layout(tensor_spec):
                rank = len(tensor_spec.shape) + 1
                return dtensor.Layout.batch_sharded(
                    self.mesh, batch_dim="batch", rank=rank
                )

            layouts = tf.nest.map_structure(
                _create_batch_layout, dataset.element_spec
            )

            return dtensor.DTensorDataset(
                dataset=dataset,
                mesh=self.mesh,
                layouts=layouts,
                global_batch_size=global_batch_size,
                dataset_already_batched=False,
                batch_dim="batch",
                prefetch=None,
                tf_data_service_config=None,
            )

        train_ds = distribute_ds(train_ds)
        eval_ds = distribute_ds(eval_ds)
        model.fit(train_ds, steps_per_epoch=10)
        model.evaluate(eval_ds, steps=10)


if __name__ == "__main__":
    tf.test.main()
