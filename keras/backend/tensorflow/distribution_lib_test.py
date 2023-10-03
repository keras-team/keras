"""Test for Tensorflow backend distribution_lib.py."""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.experimental import dtensor
from tensorflow.python.eager import context

from keras import backend
from keras import layers
from keras import models
from keras import testing
from keras.backend import distribution_lib as backend_dlib
from keras.distribution import distribution_lib


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="Backend specific test",
)
class TensorflowDistributionLibTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Config virtual devices for testing.
        cpus = tf.config.list_physical_devices("CPU")
        context._reset_context()
        tf.config.set_logical_device_configuration(
            cpus[0], [tf.config.LogicalDeviceConfiguration()] * 8
        )

        dtensor.initialize_accelerator_system("cpu")

    def tearDown(self) -> None:
        super().tearDown()
        dtensor.shutdown_accelerator_system()

    def test_list_devices(self):
        self.assertEqual(len(distribution_lib.list_devices()), 8)
        self.assertEqual(len(distribution_lib.list_devices("cpu")), 8)
        self.assertEqual(len(distribution_lib.list_devices("cpu")), 8)

    def test_to_dtensor_mesh(self):
        devices = [f"cpu:{i}" for i in range(8)]
        shape = (4, 2)
        axis_names = ["batch", "model"]

        mesh = distribution_lib.DeviceMesh(shape, axis_names, devices)
        dtensor_mesh = backend_dlib._to_dtensor_mesh(mesh)

        self.assertIsInstance(dtensor_mesh, dtensor.Mesh)
        self.assertEqual(dtensor_mesh.shape(), list(shape))
        self.assertEqual(dtensor_mesh.dim_names, axis_names)

    def test_to_dtensor_layout(self):
        axes = ["data", None]
        mesh = distribution_lib.DeviceMesh(
            (4, 2), ["data", "model"], [f"cpu:{i}" for i in range(8)]
        )
        layout = distribution_lib.TensorLayout(axes, mesh)
        dtensor_layout = backend_dlib._to_dtensor_layout(layout)
        dtensor_mesh = backend_dlib._to_dtensor_mesh(mesh)
        self.assertEqual(
            dtensor_layout,
            dtensor.Layout(["data", dtensor.UNSHARDED], dtensor_mesh),
        )

    def test_validation_for_device_mesh(self):
        axes = ["data", None]
        layout = distribution_lib.TensorLayout(axes, device_mesh=None)

        with self.assertRaisesRegex(
            ValueError, "Cannot create sharding when device mesh is not set"
        ):
            backend_dlib._to_dtensor_layout(layout)

    def test_variable_assignment_reuse_layout(self):
        shape = (4, 2)
        axis_names = ["batch", "model"]
        devs = backend_dlib.list_devices()
        device_mesh = distribution_lib.DeviceMesh(shape, axis_names, devs)
        layout_map = distribution_lib.LayoutMap(device_mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
            [None, "model"]
        )
        layout_map[".*dense.*bias"] = distribution_lib.TensorLayout(["model"])

        distribution = distribution_lib.ModelParallel(
            device_mesh, layout_map, batch_dim_name="batch"
        )

        with distribution.scope():
            dense_layer = layers.Dense(8)
            dense_layer.build((16, 16))

        self.assertEqual(
            dtensor.fetch_layout(dense_layer.kernel._value).sharding_specs,
            [dtensor.UNSHARDED, "model"],
        )
        self.assertEqual(
            dtensor.fetch_layout(dense_layer.bias._value).sharding_specs,
            ["model"],
        )

        # Assign a numpy value to dense layer to mimic the model weight loading
        new_kernel = np.random.normal(size=(16, 8))
        new_bias = np.random.normal(size=(8))
        dense_layer.kernel.assign(new_kernel)
        dense_layer.bias.assign(new_bias)

        # Make sure the loaded value still use the layout when it is
        # initialized, even outside of the distribution scope.
        self.assertEqual(
            dtensor.fetch_layout(dense_layer.kernel._value).sharding_specs,
            [dtensor.UNSHARDED, "model"],
        )
        self.assertEqual(
            dtensor.fetch_layout(dense_layer.bias._value).sharding_specs,
            ["model"],
        )
