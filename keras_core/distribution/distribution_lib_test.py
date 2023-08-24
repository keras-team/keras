"""Test for distribution_lib.py."""

import os

import jax
import pytest

from keras_core import backend
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
