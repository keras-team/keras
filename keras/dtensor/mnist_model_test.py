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

import tensorflow.compat.v2 as tf

from keras import backend
from keras.dtensor import dtensor_api as dtensor
from keras.dtensor import integration_test_utils
from keras.dtensor import test_util
from keras.optimizers import adam
from keras.utils import tf_utils


class MnistTest(test_util.DTensorBaseTest):
    def test_mnist_training_cpu(self):
        devices = tf.config.list_physical_devices("CPU")
        tf.config.set_logical_device_configuration(
            devices[0],
            [
                tf.config.LogicalDeviceConfiguration(),
            ]
            * 8,
        )

        mesh = dtensor.create_mesh(
            devices=["CPU:%d" % i for i in range(8)], mesh_dims=[("batch", 8)]
        )

        backend.enable_tf_random_generator()
        # Needed by keras initializers.
        tf_utils.set_random_seed(1337)

        model = integration_test_utils.get_model_with_layout_map(
            integration_test_utils.get_all_replicated_layout_map(mesh)
        )

        optimizer = adam.Adam(learning_rate=0.001, mesh=mesh)
        optimizer.build(model.trainable_variables)

        train_losses = integration_test_utils.train_mnist_model_batch_sharded(
            model,
            optimizer,
            mesh,
            num_epochs=3,
            steps_per_epoch=100,
            global_batch_size=64,
        )
        # Make sure the losses are decreasing
        self.assertEqual(train_losses, sorted(train_losses, reverse=True))

    def test_model_fit(self):
        devices = tf.config.list_physical_devices("CPU")
        tf.config.set_logical_device_configuration(
            devices[0],
            [
                tf.config.LogicalDeviceConfiguration(),
            ]
            * 8,
        )

        mesh = dtensor.create_mesh(
            devices=["CPU:%d" % i for i in range(8)], mesh_dims=[("batch", 8)]
        )

        backend.enable_tf_random_generator()
        # Needed by keras initializers.
        tf_utils.set_random_seed(1337)

        model = integration_test_utils.get_model_with_layout_map(
            integration_test_utils.get_all_replicated_layout_map(mesh)
        )

        optimizer = adam.Adam(learning_rate=0.001, mesh=mesh)
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
                    mesh, batch_dim="batch", rank=rank
                )

            layouts = tf.nest.map_structure(
                _create_batch_layout, dataset.element_spec
            )

            return dtensor.DTensorDataset(
                dataset=dataset,
                mesh=mesh,
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

    def DISABLED_test_mnist_training_tpu(self):
        # TODO(scottzhu): Enable TPU test once the dtensor_test rule is migrated
        # out of learning/brain
        dtensor.initialize_accelerator_system()
        total_tpu_device_count = dtensor.num_global_devices("TPU")
        mesh_shape = [total_tpu_device_count]
        mesh = dtensor.create_tpu_mesh(["batch"], mesh_shape, "tpu_mesh")

        # Needed by keras initializers.
        tf_utils.set_random_seed(1337)

        model = integration_test_utils.get_model_with_layout_map(
            integration_test_utils.get_all_replicated_layout_map(mesh)
        )

        optimizer = adam.Adam(learning_rate=0.001, mesh=mesh)
        optimizer.build(model.trainable_variables)

        train_losses = integration_test_utils.train_mnist_model_batch_sharded(
            model,
            optimizer,
            mesh,
            num_epochs=3,
            steps_per_epoch=100,
            global_batch_size=64,
        )
        # Make sure the losses are decreasing
        self.assertEqual(train_losses, sorted(train_losses, reverse=True))


if __name__ == "__main__":
    tf.test.main()
