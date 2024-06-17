"""Test for distribution_lib.py."""

import functools
import os
from unittest import mock

import jax
import numpy as np
import pytest

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend import distribution_lib as backend_dlib
from keras.src.distribution import distribution_lib

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


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="Backend specific test",
)
class JaxDistributionLibTest(testing.TestCase):
    def test_list_devices(self):
        self.assertEqual(len(distribution_lib.list_devices()), 8)
        self.assertEqual(len(distribution_lib.list_devices("cpu")), 8)
        self.assertEqual(len(distribution_lib.list_devices("cpu")), 8)

    def test_device_conversion(self):
        devices = distribution_lib.list_devices("cpu")
        jax_devices = jax.devices("cpu")

        for d, jax_d in zip(devices, jax_devices):
            converted_jax_device = backend_dlib._to_jax_device(d)
            self.assertIsInstance(converted_jax_device, jax.Device)
            self.assertEqual(jax_d, converted_jax_device)

    @mock.patch.object(jax.distributed, "initialize", return_value=None)
    def test_initialize_with_all_job_addresses(self, mock_jax_initialze):
        backend_dlib.initialize("10.0.0.1:1234,10.0.0.2:2345", 2, 0)
        mock_jax_initialze.assert_called_once_with(
            coordinator_address="10.0.0.1:1234", num_processes=2, process_id=0
        )

    def test_initialize_validate_job_and_process(self):
        with self.assertRaisesRegex(
            ValueError, "has 2 jobs, but num_processes is 3"
        ):
            backend_dlib.initialize("10.0.0.1:1234,10.0.0.2:2345", 3, 0)

    @mock.patch.object(jax.distributed, "initialize", return_value=None)
    def test_initialize_with_coordinater_address(self, mock_jax_initialze):
        backend_dlib.initialize("10.0.0.1:1234", 2, 0)
        mock_jax_initialze.assert_called_once_with(
            coordinator_address="10.0.0.1:1234", num_processes=2, process_id=0
        )

    def test_distribute_tensor(self):
        jax_mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(2, 4), ("batch", "model")
        )

        inputs = jax.numpy.array(np.random.normal(size=(16, 8)))
        target_layout = jax.sharding.NamedSharding(
            jax_mesh, jax.sharding.PartitionSpec("batch", None)
        )

        @functools.partial(jax.jit, static_argnames="target_layout")
        def test_function(inputs, target_layout):
            return distribution_lib.distribute_tensor(inputs, target_layout)

        result = test_function(inputs, target_layout)
        # Note that the returned tensor has a different sharding implementation
        # which is GSPMDSharding, but it should be equivalent as the target
        # layout specified.
        self.assertTrue(result.sharding.is_equivalent_to(target_layout, ndim=2))

        # Test without jit
        result = distribution_lib.distribute_tensor(inputs, target_layout)
        self.assertTrue(result.sharding.is_equivalent_to(target_layout, ndim=2))

    def test_distribute_variable(self):
        # This test only verify the single worker/process behavior.
        # The multi-process test lives in g3.
        jax_mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(2, 4), ("batch", "model")
        )

        variable = jax.numpy.array(np.random.normal(size=(16, 8)))
        target_layout = jax.sharding.NamedSharding(
            jax_mesh, jax.sharding.PartitionSpec("model", None)
        )

        result = backend_dlib.distribute_variable(variable, target_layout)
        # Note that the returned tensor has a different sharding implementation
        # which is GSPMDSharding, but it should be equivalent as the target
        # layout specified.
        self.assertTrue(result.sharding.is_equivalent_to(target_layout, ndim=2))

    def test_distribute_input_data(self):
        # This test only verify the single worker/process behavior.
        # The multi-process test lives in g3.
        jax_mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(2, 4), ("batch", "model")
        )

        input_data = jax.numpy.array(np.random.normal(size=(16, 8)))
        target_layout = jax.sharding.NamedSharding(
            jax_mesh, jax.sharding.PartitionSpec("batch", None)
        )

        result = backend_dlib.distribute_variable(input_data, target_layout)
        # Note that the returned tensor has a different sharding implementation
        # which is GSPMDSharding, but it should be equivalent as the target
        # layout specified.
        self.assertTrue(result.sharding.is_equivalent_to(target_layout, ndim=2))

    def test_processes(self):
        self.assertEqual(backend_dlib.process_id(), 0)
        self.assertEqual(backend_dlib.num_processes(), 1)

    def test_to_jax_mesh(self):
        devices = [f"cpu:{i}" for i in range(8)]
        shape = (4, 2)
        axis_names = ["batch", "model"]

        mesh = distribution_lib.DeviceMesh(shape, axis_names, devices)
        jax_mesh = backend_dlib._to_jax_mesh(mesh)

        self.assertIsInstance(jax_mesh, jax.sharding.Mesh)
        self.assertEqual(jax_mesh.devices.shape, shape)
        self.assertEqual(jax_mesh.axis_names, ("batch", "model"))

    def test_to_jax_layout(self):
        axes = ["data", None]
        mesh = distribution_lib.DeviceMesh(
            (4, 2), ["data", "model"], [f"cpu:{i}" for i in range(8)]
        )
        layout = distribution_lib.TensorLayout(axes, mesh)
        jax_sharding = backend_dlib._to_jax_layout(layout)
        jax_mesh = backend_dlib._to_jax_mesh(mesh)
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
            backend_dlib._to_jax_layout(layout)

    def test_variable_assignment_reuse_layout(self):
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
            layout_map=layout_map, batch_dim_name="batch"
        )

        with distribution.scope():
            dense_layer = layers.Dense(8)
            dense_layer.build((16, 16))

        self.assertEqual(
            dense_layer.kernel._value.sharding.spec, (None, "model")
        )
        self.assertEqual(dense_layer.bias._value.sharding.spec, ("model",))

        # Assign a numpy value to dense layer to mimic the model weight loading
        new_kernel = np.random.normal(size=(16, 8))
        new_bias = np.random.normal(size=(8))
        dense_layer.kernel.assign(new_kernel)
        dense_layer.bias.assign(new_bias)

        # Make sure the loaded value still use the layout when it is
        # initialized, even outside of the distribution scope.
        self.assertEqual(
            dense_layer.kernel._value.sharding.spec, (None, "model")
        )
        self.assertEqual(dense_layer.bias._value.sharding.spec, ("model",))

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
            layout_map=layout_map, batch_dim_name="batch"
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

    def test_e2e_model_parallel_with_output_sharding(self):
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
        # Force the dense layer output to be batch parallel only, and not
        # sharded on model dimension.
        layout_map[".*dense.*output"] = ("batch", None)

        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        sharding_capture = ShardingCaptureLayer()
        with distribution.scope():
            inputs = layers.Input(shape=[28, 28, 1])
            y = layers.Flatten()(inputs)
            y = layers.Dense(units=200, use_bias=False, activation="relu")(y)
            y = sharding_capture(y)
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

        # Note that the intermediate_tensor_layout is only captured during the
        # actual training, and not at the model building time.
        intermediate_tensor_layout = jax.sharding.NamedSharding(
            backend_dlib._to_jax_mesh(distribution.device_mesh),
            jax.sharding.PartitionSpec("batch", None),
        )
        self.assertTrue(
            sharding_capture.captured_input_sharding.is_equivalent_to(
                intermediate_tensor_layout, ndim=2
            )
        )


class ShardingCaptureLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.captured_input_sharding = None
        self.supports_masking = True

    def call(self, inputs):
        jax.debug.inspect_array_sharding(
            inputs, callback=lambda x: self.capture_input_sharding(x)
        )
        return inputs

    def capture_input_sharding(self, sharding):
        self.captured_input_sharding = sharding
