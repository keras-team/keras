import os

import numpy as np
import pytest
import torch
from absl.testing import parameterized

from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend import backend
from keras.src.backend.torch import distribution_lib as torch_distribution_lib
from keras.src.distribution import distribution_lib
from keras.src.trainers.data_adapters import tf_dataset_adapter
from keras.src.utils import rng_utils


class MultiProcessTest:
    pass


@pytest.mark.multi_device
@pytest.mark.skipif(backend() != "torch", reason="Torch only")
class TorchMultiProcessDistributeTest(
    MultiProcessTest, testing.TestCase, parameterized.TestCase
):
    def setUp(self):
        super().setUp()
        # We need a consistent seed across all processes.
        rng_utils.set_random_seed(1234)

        if not torch.distributed.is_initialized():
            # In the multi-process environment, initialize using env vars.
            if "WORLD_SIZE" in os.environ:
                torch_distribution_lib.initialize()

    def test_list_device(self):
        devices = distribution_lib.list_devices()
        for d in devices:
            converted_torch_device = torch_distribution_lib._to_backend_device(
                d
            )
            self.assertIsInstance(converted_torch_device, torch.device)

    def test_distribute_variable(self):
        if not torch.distributed.is_initialized():
            self.skipTest("torch.distributed is not initialized")

        num_processes = torch_distribution_lib.num_processes()

        global_shape = (8, 4)
        kernel = np.arange(np.prod(global_shape)).reshape(global_shape)

        device_mesh = distribution_lib.DeviceMesh(
            shape=(num_processes,),
            axis_names=["batch"],
            devices=distribution_lib.list_devices(),
        )
        layout = distribution_lib.TensorLayout(
            axes=("batch", None), device_mesh=device_mesh
        )

        distributed_kernel = torch_distribution_lib.distribute_tensor(
            kernel, layout
        )

        # After distribution, the value should be a global shape tensor.
        self.assertEqual(tuple(distributed_kernel.shape), global_shape)

        from torch.distributed.tensor import DTensor

        self.assertIsInstance(distributed_kernel, DTensor)

        # Also make sure the gathered global value has the same value as the
        # original value.
        local_copy = distributed_kernel.full_tensor().cpu().numpy()
        self.assertAllClose(local_copy, kernel)

    def test_dataset_distribution_data_parallel(self):
        # We use tensorflow for dataset creation in these tests
        import tensorflow as tf

        num_processes = torch_distribution_lib.num_processes()

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

        process_id = torch_distribution_lib.process_id()
        per_process_batch_size = global_batch_size // num_processes

        expected_value = (
            np.arange(per_process_batch_size)
            + process_id * per_process_batch_size
        )
        for d in distributed_dataset:
            d = d.numpy()
            self.assertEqual(d.shape, (per_process_batch_size,))
            self.assertAllClose(d, expected_value)
            expected_value += global_batch_size

    def test_e2e_data_parallel_model(self):
        distribution = distribution_lib.DataParallel(
            devices=distribution_lib.list_devices(),
        )

        with distribution.scope():
            model = models.Sequential(
                [
                    layers.Input(shape=(28, 28, 1)),
                    layers.Flatten(),
                    layers.Dense(units=200, use_bias=False, activation="relu"),
                    layers.Dropout(0.4),
                    layers.Dense(units=10, activation="softmax"),
                ]
            )

        # For Torch, weight sharding in DataParallel usually means replicated.
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor import Replicate

        for weight in model.weights:
            self.assertIsInstance(weight.value, DTensor)
            for placement in weight.value.placements:
                self.assertIsInstance(placement, Replicate)

        import tensorflow as tf

        inputs = np.random.normal(size=(128, 28, 28, 1)).astype("float32")
        labels = np.random.normal(size=(128, 10)).astype("float32")
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(16)

        with distribution.scope():
            model.compile(loss="mse", optimizer="adam")
            model.fit(dataset, epochs=2)
            model.evaluate(dataset)

    @parameterized.named_parameters(
        [
            ("data_only", 1),
            ("model_only", 2),
        ]
    )
    def test_e2e_model_parallel_model(self, model_dim):
        num_processes = torch_distribution_lib.num_processes()
        # Ensure model_dim doesn't exceed available devices.
        model_dim = min(model_dim, num_processes)
        mesh_shape = (num_processes // model_dim, model_dim)

        device_mesh = distribution_lib.DeviceMesh(
            shape=mesh_shape,
            axis_names=["batch", "model"],
            devices=distribution_lib.list_devices(),
        )
        layout_map = distribution_lib.LayoutMap(device_mesh)
        distribution = distribution_lib.ModelParallel(layout_map=layout_map)

        with distribution.scope():
            model = models.Sequential(
                [
                    layers.Input(shape=(28, 28, 1)),
                    layers.Flatten(),
                    layers.Dense(units=200, use_bias=False, activation="relu"),
                    layers.Dense(units=10, activation="softmax"),
                ]
            )

        import tensorflow as tf

        inputs = np.random.normal(size=(64, 28, 28, 1)).astype("float32")
        labels = np.random.normal(size=(64, 10)).astype("float32")
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(8)

        with distribution.scope():
            model.compile(loss="mse", optimizer="adam")
            model.fit(dataset, epochs=1)
            model.evaluate(dataset)


if __name__ == "__main__":
    pytest.main([__file__])
