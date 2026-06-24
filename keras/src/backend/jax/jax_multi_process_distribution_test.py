import jax
import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend import backend
from keras.src.backend.jax import distribution_lib as jax_distribution_lib
from keras.src.distribution import distribution_lib
from keras.src.trainers.data_adapters import tf_dataset_adapter
from keras.src.utils import rng_utils


class MultiProcessTest:
    pass


@pytest.mark.multi_device
@pytest.mark.skipif(backend() != "jax", reason="JAX only")
class JaxMultiProcessDistributeTest(
    MultiProcessTest, testing.TestCase, parameterized.TestCase
):
    def setUp(self):
        # We need a consistent seed across all processes.
        rng_utils.set_random_seed(1234)

        num_processes = jax.process_count()
        num_devices = jax.device_count()
        local_devices = jax.local_device_count()

        # In the internal multi-process environment, we expect a specific count.
        if num_processes > 1:
            self.assertEqual(num_processes, 4)
            self.assertEqual(num_devices, 8)
            self.assertEqual(local_devices, 2)

    def test_list_device(self):
        devices = distribution_lib.list_devices()
        jax_devices = jax.devices()

        for d, jax_d in zip(devices, jax_devices):
            converted_jax_device = jax_distribution_lib._to_backend_device(d)
            self.assertIsInstance(converted_jax_device, jax.Device)
            self.assertEqual(jax_d, converted_jax_device)

    def test_distribute_variable(self):
        num_processes = jax.process_count()
        num_devices = jax.device_count()
        local_devices = jax.local_device_count()

        global_shape = (8, 4)
        kernel = np.arange(np.prod(global_shape)).reshape(global_shape)

        axis_names = ["batch", "model"]
        model_dim = min(kernel.shape[1], num_devices // num_processes)
        jax_mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(
                num_devices // model_dim, model_dim
            ),
            axis_names,
        )
        jax_sharding = jax.sharding.NamedSharding(
            jax_mesh, jax.sharding.PartitionSpec("batch", "model")
        )

        distributed_kernel = jax_distribution_lib.distribute_tensor(
            kernel, jax_sharding
        )
        # After distribution, the value should be a global shape tensor.
        self.assertEqual(distributed_kernel.shape, global_shape)
        self.assertEqual(
            distributed_kernel.is_fully_addressable, num_processes == 1
        )
        self.assertTrue(
            jax_sharding.is_equivalent_to(distributed_kernel.sharding, ndim=2)
        )

        # Each local process will have `local_devices` addressable data.
        self.assertEqual(
            len(distributed_kernel.addressable_shards), local_devices
        )

        # Also make sure the gathered global value has the same value as the
        # original value.
        local_copy = jax.experimental.multihost_utils.process_allgather(
            distributed_kernel,
            tiled=True,
        )
        self.assertAllClose(local_copy, kernel)

    def test_dataset_distribution_data_parallel(self):
        num_processes = jax.process_count()

        # Create a dataset with range, so that we can verify the numerical
        # correctness.
        global_batch_size = 8
        num_batch = 4
        dataset = tf.data.Dataset.range(global_batch_size * num_batch).batch(
            global_batch_size
        )
        distribution = distribution_lib.DataParallel(
            devices=distribution_lib.list_devices()
        )

        # Since there are `num_processes` worker/processes, we will have
        # `num_processes` shards of the data.
        adapter = tf_dataset_adapter.TFDatasetAdapter(
            dataset, distribution=distribution
        )
        distributed_dataset = adapter.get_tf_dataset()

        process_id = jax.process_index()
        per_process_batch_size = global_batch_size // num_processes
        expected_value = (
            np.arange(per_process_batch_size)
            + process_id * per_process_batch_size
        )
        for d in distributed_dataset:
            self.assertEqual(d.shape, (per_process_batch_size,))
            self.assertAllClose(d, expected_value)
            expected_value += global_batch_size

    @parameterized.named_parameters(
        [
            ("data_only", 1),
            ("data_model", 2),
            ("model_data", 4),
            ("model_only", 8),
        ]
    )
    def test_dataset_distribution_model_parallel(self, model_dim):
        num_processes = jax.process_count()
        num_devices = jax.device_count()
        local_devices = jax.local_device_count()

        # Ensure model_dim doesn't exceed available devices.
        model_dim = min(model_dim, num_devices)
        mesh_shape = (num_devices // model_dim, model_dim)

        global_batch_size = 8
        num_batch = 4
        dataset = tf.data.Dataset.range(global_batch_size * num_batch).batch(
            global_batch_size
        )

        device_mesh = distribution_lib.DeviceMesh(
            shape=mesh_shape,
            axis_names=["batch", "model"],
            devices=distribution_lib.list_devices(),
        )
        layout_map = distribution_lib.LayoutMap(device_mesh)
        distribution = distribution_lib.ModelParallel(layout_map=layout_map)

        adapter = tf_dataset_adapter.TFDatasetAdapter(
            dataset, distribution=distribution
        )
        distributed_dataset = adapter.get_tf_dataset()

        process_id = jax.process_index()
        # Calculate how many replicas this local process is responsible for.
        # num_replicas = mesh_shape[0]
        # num_devices_per_replica = num_devices // num_replicas
        # num_local_replicas = local_devices // num_devices_per_replica
        num_devices_per_replica = max(1, num_devices // mesh_shape[0])
        num_local_replicas = max(1, local_devices // num_devices_per_replica)
        per_worker_batch_size = num_local_replicas * (
            global_batch_size // mesh_shape[0]
        )

        expected_value = np.arange(per_worker_batch_size)
        processes_per_replica = num_processes // mesh_shape[0]
        if processes_per_replica > 1:
            worker_factor = process_id // processes_per_replica
        else:
            worker_factor = process_id
        expected_value += worker_factor * per_worker_batch_size

        for batch_index, batch in enumerate(distributed_dataset):
            self.assertEqual(batch.shape, (per_worker_batch_size,))
            self.assertAllClose(
                batch,
                expected_value,
                msg=f"process {process_id} batch {batch_index}",
            )
            expected_value += global_batch_size

    def test_e2e_data_parallel_model(self):
        distribution = distribution_lib.DataParallel(
            devices=distribution_lib.list_devices(),
        )

        with distribution.scope():
            inputs = layers.Input(shape=[28, 28, 1])
            y = layers.Flatten()(inputs)
            y = layers.Dense(units=200, use_bias=False, activation="relu")(y)
            y = layers.Dropout(0.4)(y)
            y = layers.Dense(units=10, activation="softmax")(y)
            model = models.Model(inputs=inputs, outputs=y)

        # Make sure all the weights are properly sharded.
        for weight in model.weights:
            self.assertTrue(weight.value.sharding.is_fully_replicated)

        inputs = np.random.normal(size=(128, 28, 28, 1))
        labels = np.random.normal(size=(128, 10))
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(16)

        with distribution.scope():
            model.compile(loss="mse")
            model.fit(dataset, epochs=3)

            model.evaluate(dataset)

    @parameterized.named_parameters(
        [
            ("data_only", 1),
            ("data_model", 2),
            ("model_data", 4),
            ("model_only", 8),
        ]
    )
    def test_e2e_data_model_parallel_model(self, model_dim):
        num_devices = jax.device_count()
        # Ensure model_dim doesn't exceed available devices.
        model_dim = min(model_dim, num_devices)
        mesh_shape = (num_devices // model_dim, model_dim)

        device_mesh = distribution_lib.DeviceMesh(
            shape=mesh_shape,
            axis_names=["batch", "model"],
            devices=distribution_lib.list_devices(),
        )
        layout_map = distribution_lib.LayoutMap(device_mesh)
        distribution = distribution_lib.ModelParallel(layout_map=layout_map)

        with distribution.scope():
            inputs = layers.Input(shape=[28, 28, 1])
            y = layers.Flatten()(inputs)
            y = layers.Dense(units=200, use_bias=False, activation="relu")(y)
            y = layers.Dropout(0.4)(y)
            y = layers.Dense(units=10, activation="softmax")(y)
            model = models.Model(inputs=inputs, outputs=y)

        # Since no layout map was specified, all weights are replicated.
        for weight in model.weights:
            self.assertTrue(weight.value.sharding.is_fully_replicated)

        inputs = np.random.normal(size=(128, 28, 28, 1))
        labels = np.random.normal(size=(128, 10))
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(16)

        with distribution.scope():
            model.compile(loss="mse")
            model.fit(dataset, epochs=3)

            model.evaluate(dataset)


if __name__ == "__main__":
    pytest.main([__file__])
