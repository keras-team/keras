"""Test for distribution_lib.py."""

import os
from unittest import mock

import jax
import numpy as np
import pytest

from keras_core import backend
from keras_core import layers
from keras_core import models
from keras_core import testing
from keras_core.backend import distribution_lib as backend_dlib
from keras_core.distribution import distribution_lib

if backend.backend() == "jax":
    # Due to https://github.com/google/jax/issues/17188, we can't
    # override the XLA flag after the JAX back init. We have to
    # run this at top level to let JAX pick the flag value.
    xla_flags = os.getenv("XLA_FLAGS") or ""
    # Don't override user-specified device count, or other XLA flags.
    if "xla_force_host_platform_device_count" not in xla_flags:
        os.environ["XLA_FLAGS"] = (
            xla_flags + " --xla_force_host_platform_device_count=8"
        )


class DeviceMeshTest(testing.TestCase):
    def test_mesh_creation(self):
        devices = ["CPU:{i}" for i in range(8)]
        shape = (4, 2)
        axis_names = ["batch", "model"]

        mesh = distribution_lib.DeviceMesh(shape, axis_names, devices)
        self.assertEqual(mesh.shape, shape)
        self.assertEqual(mesh.axis_names, axis_names)
        self.assertEqual(mesh.devices.shape, shape)

    def test_input_validation(self):
        devices = ["CPU:{i}" for i in range(4)]
        with self.assertRaisesRegex(
            ValueError, "Shape and axis_names cannot be empty"
        ):
            distribution_lib.DeviceMesh((4,), "", devices)

        with self.assertRaisesRegex(
            ValueError, "Shape and axis_names should have same size"
        ):
            distribution_lib.DeviceMesh((4, 2), ["batch"], devices)

        with self.assertRaisesRegex(
            ValueError, "Shape does not match the number of devices"
        ):
            distribution_lib.DeviceMesh((4, 2), ["batch", "model"], devices)


class TensorLayoutTest(testing.TestCase):
    def setUp(self):
        self.mesh = distribution_lib.DeviceMesh(
            (4, 2), ["data", "model"], [f"CPU:{i}" for i in range(8)]
        )

    def test_tensor_layout_creation(self):
        axes = ["data", None]
        layout = distribution_lib.TensorLayout(axes, self.mesh)

        self.assertEqual(layout.device_mesh, self.mesh)
        self.assertEqual(layout.axes, axes)

    def test_tensor_layout_validation(self):
        axes = ["data", "unknown", None]
        with self.assertRaisesRegex(
            ValueError, "Invalid axis names for Layout"
        ):
            distribution_lib.TensorLayout(axes, self.mesh)

    def test_lazy_device_mesh_injection(self):
        axes = ["data", None]
        layout = distribution_lib.TensorLayout(axes, None)

        self.assertIsNone(layout.device_mesh)
        self.assertEqual(layout.axes, axes)

        layout.device_mesh = self.mesh

        self.assertEqual(layout.device_mesh, self.mesh)
        self.assertEqual(layout.axes, axes)

    def test_lazy_device_mesh_validation(self):
        axes = ["data", "unknown", None]
        layout = distribution_lib.TensorLayout(axes, None)

        self.assertIsNone(layout.device_mesh)
        self.assertEqual(layout.axes, axes)

        with self.assertRaisesRegex(
            ValueError, "Invalid axis names for Layout"
        ):
            layout.device_mesh = self.mesh


class DistributionTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        devices = ["CPU:{i}" for i in range(8)]
        shape = (4, 2)
        axis_names = ["batch", "model"]

        self.device_mesh = distribution_lib.DeviceMesh(
            shape, axis_names, devices
        )

    def test_init_with_device_mesh(self):
        distribution = distribution_lib.Distribution(self.device_mesh)
        self.assertIs(distribution.device_mesh, self.device_mesh)

    def test_scope(self):
        distribution_1 = distribution_lib.Distribution(self.device_mesh)
        distribution_2 = distribution_lib.Distribution(self.device_mesh)

        self.assertIsNone(distribution_lib.distribution())
        with distribution_1.scope():
            self.assertIs(distribution_lib.distribution(), distribution_1)
            with distribution_2.scope():
                self.assertIs(distribution_lib.distribution(), distribution_2)

            self.assertIs(distribution_lib.distribution(), distribution_1)

        self.assertIsNone(distribution_lib.distribution())


class DataParallelDistributionTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.devices = ["CPU:{i}" for i in range(8)]
        shape = (8,)
        axis_names = ["data"]

        self.device_mesh = distribution_lib.DeviceMesh(
            shape, axis_names, self.devices
        )

    def test_create_with_device_mesh(self):
        distribution = distribution_lib.DataParallel(
            device_mesh=self.device_mesh
        )

        device_mesh = distribution.device_mesh
        self.assertEqual(len(device_mesh.devices), 8)
        self.assertEqual(device_mesh.axis_names, ["data"])
        self.assertEqual(distribution._batch_dim_name, "data")

    def test_create_with_devices(self):
        distribution = distribution_lib.DataParallel(devices=self.devices)
        device_mesh = distribution.device_mesh
        self.assertEqual(len(device_mesh.devices), 8)
        self.assertEqual(device_mesh.axis_names, ["batch"])
        self.assertEqual(distribution._batch_dim_name, "batch")

    @mock.patch.object(
        distribution_lib,
        "list_devices",
        return_value=["CPU:{i}" for i in range(8)],
    )
    def test_create_with_list_devices(self, mock_list_devices):
        distribution = distribution_lib.DataParallel()
        mock_list_devices.assert_called_once()

        device_mesh = distribution.device_mesh
        self.assertEqual(len(device_mesh.devices), 8)
        self.assertEqual(device_mesh.axis_names, ["batch"])
        self.assertEqual(distribution._batch_dim_name, "batch")

    def test_get_data_layout(self):
        distribution = distribution_lib.DataParallel(
            device_mesh=self.device_mesh
        )

        data = np.arange(16).reshape((4, 2, 2))
        data_layout = distribution.get_data_layout(data.shape)
        self.assertIs(data_layout.device_mesh, self.device_mesh)
        self.assertEqual(data_layout.axes, ["data", None, None])

    def test_get_variable_layout(self):
        distribution = distribution_lib.DataParallel(
            device_mesh=self.device_mesh
        )

        variable = backend.Variable(initializer=[1, 2, 3])
        variable_layout = distribution.get_variable_layout(variable)
        self.assertIs(variable_layout.device_mesh, self.device_mesh)
        self.assertEqual(variable_layout.axes, [None])


class ModelParallelDistributionTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.devices = ["CPU:{i}" for i in range(8)]
        shape = (2, 4)
        axis_names = ["data", "model"]

        self.device_mesh = distribution_lib.DeviceMesh(
            shape, axis_names, self.devices
        )

    def test_distribute_weights(self):
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map[".*kernel"] = distribution_lib.TensorLayout([None, "model"])
        layout_map[".*bias"] = distribution_lib.TensorLayout(["model"])

        distribution = distribution_lib.ModelParallel(
            self.device_mesh, layout_map, batch_dim_name="data"
        )
        kernel = backend.Variable(initializer=np.arange(8, 4), name="kernel")
        bias = backend.Variable(initializer=np.arange(4), name="bias")
        rng_seed = backend.Variable(initializer=[0, 1], name="seed")

        kernel_layout = distribution.get_variable_layout(kernel)
        self.assertIs(kernel_layout.device_mesh, self.device_mesh)
        self.assertEqual(kernel_layout.axes, [None, "model"])

        bias_layout = distribution.get_variable_layout(bias)
        self.assertIs(bias_layout.device_mesh, self.device_mesh)
        self.assertEqual(bias_layout.axes, ["model"])

        rng_seed_layout = distribution.get_variable_layout(rng_seed)
        self.assertIs(rng_seed_layout.device_mesh, self.device_mesh)
        self.assertEqual(rng_seed_layout.axes, [None])

    def test_distribute_data(self):
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        distribution = distribution_lib.ModelParallel(
            self.device_mesh, layout_map, batch_dim_name="data"
        )

        data = np.arange(16).reshape((4, 2, 2))
        data_layout = distribution.get_data_layout(data.shape)
        self.assertIs(data_layout.device_mesh, self.device_mesh)
        self.assertEqual(data_layout.axes, ["data", None, None])


class LayoutMapTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.devices = ["CPU:{i}" for i in range(8)]
        shape = (4, 2)
        axis_names = ["data", "model"]

        self.device_mesh = distribution_lib.DeviceMesh(
            shape, axis_names, self.devices
        )
        self.sharded_2d = distribution_lib.TensorLayout([None, "model"])
        self.sharded_1d = distribution_lib.TensorLayout(["model"])

        self.replicated_2d = distribution_lib.TensorLayout([None, None])
        self.replicated_1d = distribution_lib.TensorLayout([None])

    def test_add(self):
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map["dense/kernel"] = self.sharded_2d
        layout_map["dense/bias"] = self.sharded_1d

        # Make there are two items in the map, and we access them via the
        # underlying container at layout_map._layout_map
        self.assertLen(layout_map, 2)

        kernel_layout = layout_map["dense/kernel"]
        self.assertEqual(kernel_layout.axes, [None, "model"])
        self.assertIs(kernel_layout.device_mesh, self.device_mesh)

        bias_layout = layout_map["dense/bias"]
        self.assertEqual(bias_layout.axes, ["model"])
        self.assertIs(bias_layout.device_mesh, self.device_mesh)

        with self.assertRaisesRegex(ValueError, "dense/kernel already exist"):
            layout_map["dense/kernel"] = self.sharded_2d

        with self.assertRaisesRegex(ValueError, "should be a TensorLayout"):
            layout_map["conv.kernel"] = [1, 2, 3]

    def test_get(self):
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map["dense/kernel"] = self.sharded_2d
        layout_map["dense/bias"] = self.sharded_1d

        layout_map["dense.*kernel"] = self.replicated_2d
        layout_map["dense.*bias"] = self.replicated_1d

        layout_map[".*bias"] = self.sharded_1d

        self.assertEqual(layout_map["dense/kernel"], self.sharded_2d)
        self.assertEqual(layout_map["dense/bias"], self.sharded_1d)

        # Map against the wildcard bias rule for dense, and based on the order
        # of insertion, it will not use .*bias.
        self.assertEqual(layout_map["dense_2/kernel"], self.replicated_2d)
        self.assertEqual(layout_map["dense_2/bias"], self.replicated_1d)

        self.assertIsNone(layout_map["conv2d/kernel"])
        self.assertEqual(layout_map["conv2d/bias"], self.sharded_1d)

    def test_delete(self):
        layout_map = distribution_lib.LayoutMap(self.device_mesh)

        layout_map["dense/kernel"] = self.sharded_2d
        layout_map["dense/bias"] = self.sharded_1d

        self.assertEqual(layout_map.pop("dense/kernel"), self.sharded_2d)
        # Make sure to match against the exact string, not the regex
        with self.assertRaises(KeyError):
            layout_map.pop(".*bias")

        # Make sure del also works
        del layout_map["dense/bias"]

        self.assertLen(layout_map, 0)

    def test_len(self):
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        self.assertLen(layout_map, 0)

        layout_map["dense/kernel"] = self.sharded_2d
        layout_map["dense/bias"] = self.sharded_1d

        self.assertLen(layout_map, 2)

    def test_iter(self):
        layout_map = distribution_lib.LayoutMap(self.device_mesh)

        layout_map["dense/kernel"] = self.sharded_2d
        layout_map["dense/bias"] = self.sharded_1d

        # Make sure the items are ordered based on the insertion order.
        self.assertEqual(
            list(layout_map.keys()), ["dense/kernel", "dense/bias"]
        )

        keys = []
        values = []
        for k, v in layout_map.items():
            keys.append(k)
            values.append(v)

        self.assertEqual(keys, ["dense/kernel", "dense/bias"])
        self.assertEqual(values, [self.sharded_2d, self.sharded_1d])


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="Backend specific test",
)
class JaxDistributionLibTest(testing.TestCase):
    def test_list_devices(self):
        self.assertEqual(len(distribution_lib.list_devices()), 8)
        self.assertEqual(len(distribution_lib.list_devices("cpu")), 8)
        self.assertEqual(len(distribution_lib.list_devices("CPU")), 8)

    def test_to_jax_mesh(self):
        devices = ["CPU:{i}" for i in range(8)]
        shape = (4, 2)
        axis_names = ["batch", "model"]

        mesh = distribution_lib.DeviceMesh(shape, axis_names, devices)
        jax_mesh = backend_dlib.to_jax_mesh(mesh)

        self.assertIsInstance(jax_mesh, jax.sharding.Mesh)
        self.assertEqual(jax_mesh.devices.shape, shape)
        self.assertEqual(jax_mesh.axis_names, ("batch", "model"))

    def test_to_jax_layout(self):
        axes = ["data", None]
        mesh = distribution_lib.DeviceMesh(
            (4, 2), ["data", "model"], [f"CPU:{i}" for i in range(8)]
        )
        layout = distribution_lib.TensorLayout(axes, mesh)
        jax_sharding = backend_dlib.to_jax_layout(layout)
        jax_mesh = backend_dlib.to_jax_mesh(mesh)
        self.assertEqual(
            jax_sharding,
            jax.sharding.NamedSharding(
                jax_mesh, jax.sharding.PartitionSpec("data", None)
            ),
        )

    def test_validation_for_device_mesh(self):
        axes = ["data", None]
        layout = distribution_lib.TensorLayout(axes, device_mesh=None)

        with self.assertRaisesRegex(
            ValueError, "Cannot create sharding when device mesh is not set"
        ):
            backend_dlib.to_jax_layout(layout)

    def test_e2e_data_parallel_model(self):
        distribution = distribution_lib.DataParallel(
            devices=backend_dlib.list_devices()
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
            self.assertTrue(weight._value.sharding.is_fully_replicated)

        inputs = np.random.normal(size=(32, 28, 28, 1))
        labels = np.random.normal(size=(32, 10))

        with distribution.scope():
            model.compile(loss="mse")
            model.fit(inputs, labels)

    def test_e2e_model_parallel_model(self):
        shape = (4, 2)
        axis_names = ["batch", "model"]
        device_mesh = distribution_lib.DeviceMesh(
            shape, axis_names, backend_dlib.list_devices()
        )

        layout_map = distribution_lib.LayoutMap(device_mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
            [None, "model"]
        )
        layout_map[".*dense.*bias"] = distribution_lib.TensorLayout(["model"])

        distribution = distribution_lib.ModelParallel(
            device_mesh, layout_map, batch_dim_name="batch"
        )
        with distribution.scope():
            inputs = layers.Input(shape=[28, 28, 1])
            y = layers.Flatten()(inputs)
            y = layers.Dense(units=200, use_bias=False, activation="relu")(y)
            y = layers.Dropout(0.4)(y)
            y = layers.Dense(units=10, activation="softmax")(y)
            model = models.Model(inputs=inputs, outputs=y)

        for weight in model.weights:
            if "kernel" in weight.name:
                self.assertEqual(weight._value.sharding.spec, (None, "model"))
            elif "bias" in weight.name:
                self.assertEqual(weight._value.sharding.spec, ("model",))
            else:
                self.assertTrue(weight._value.sharding.is_fully_replicated)

        inputs = np.random.normal(size=(32, 28, 28, 1))
        labels = np.random.normal(size=(32, 10))

        with distribution.scope():
            model.compile(loss="mse")
            model.fit(inputs, labels)
