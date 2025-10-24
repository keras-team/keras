"""Test for distribution_lib.py."""

import os

# FILE: keras/src/distribution/distribution_lib_test.py


# --- TOP-LEVEL ENVIRONMENT SETUP ---
# This MUST be at the top of the file, before any Keras/TF imports.
# It configures the environment for all tests in this file.
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

# --- Now continue with the rest of the imports ---
# ... and so on
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

import keras
from keras.src import backend
from keras.src import testing
from keras.src.backend import distribution_lib as backend_dlib
from keras.src.distribution import distribution_lib
from keras.src.distribution.distribution_lib import AutoTPDistribution

try:
    import keras_hub
except ImportError:
    keras_hub = None


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="Only JAX has the backend to mock at the moment",
)
@mock.patch.object(
    backend_dlib,
    "initialize",
    return_value=None,
)
class MultiProcessInitializeTest(testing.TestCase):
    def tearDown(self):
        super().tearDown()
        os.environ.clear()

    def test_initialize_with_explicit_param(self, mock_backend_initialize):
        job_addresses = "10.0.0.1:1234,10.0.0.2:2345"
        num_processes = 2
        current_process_id = 0

        distribution_lib.initialize(
            job_addresses, num_processes, current_process_id
        )

        mock_backend_initialize.assert_called_once_with(
            job_addresses, num_processes, current_process_id
        )

    def test_initialize_with_env_vars(self, mock_backend_initialize):
        job_addresses = "10.0.0.1:1234,10.0.0.2:2345"
        num_processes = 2
        current_process_id = 0
        os.environ["KERAS_DISTRIBUTION_JOB_ADDRESSES"] = job_addresses
        os.environ["KERAS_DISTRIBUTION_NUM_PROCESSES"] = str(num_processes)
        os.environ["KERAS_DISTRIBUTION_PROCESS_ID"] = str(current_process_id)

        distribution_lib.initialize()
        mock_backend_initialize.assert_called_once_with(
            job_addresses, num_processes, current_process_id
        )

    def test_init_with_nones(self, mock_backend_initialize):
        # This is also valid case for Cloud TPU on JAX
        distribution_lib.initialize()
        mock_backend_initialize.assert_called_once_with(None, None, None)


class DeviceMeshTest(testing.TestCase):
    def test_mesh_creation(self):
        devices = [f"cpu:{i}" for i in range(8)]
        shape = (4, 2)
        axis_names = ["batch", "model"]

        mesh = distribution_lib.DeviceMesh(shape, axis_names, devices)
        self.assertEqual(mesh.shape, shape)
        self.assertEqual(mesh.axis_names, axis_names)
        self.assertEqual(mesh.devices.shape, shape)

    def test_input_validation(self):
        devices = [f"cpu:{i}" for i in range(4)]
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
            (4, 2), ["data", "model"], [f"cpu:{i}" for i in range(8)]
        )

    def test_tensor_layout_creation(self):
        axes = ("data", None)
        layout = distribution_lib.TensorLayout(axes, self.mesh)

        self.assertEqual(layout.device_mesh, self.mesh)
        self.assertEqual(layout.axes, axes)

    def test_tensor_layout_validation(self):
        axes = ("data", "unknown", None)
        with self.assertRaisesRegex(
            ValueError, "Invalid axis names for Layout"
        ):
            distribution_lib.TensorLayout(axes, self.mesh)

    def test_lazy_device_mesh_injection(self):
        axes = ("data", None)
        layout = distribution_lib.TensorLayout(axes, None)

        self.assertIsNone(layout.device_mesh)
        self.assertEqual(layout.axes, axes)

        layout.device_mesh = self.mesh

        self.assertEqual(layout.device_mesh, self.mesh)
        self.assertEqual(layout.axes, axes)

    def test_lazy_device_mesh_validation(self):
        axes = ("data", "unknown", None)
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
        devices = [f"cpu:{i}" for i in range(8)]
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


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="Only JAX has the proper backend distribution lib",
)
class DataParallelDistributionTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.devices = [f"cpu:{i}" for i in range(8)]
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
        self.assertEqual(distribution.batch_dim_name, "data")

        self.assertFalse(distribution._is_multi_process)
        self.assertEqual(distribution._process_id, 0)
        self.assertEqual(distribution._num_process, 1)

    def test_create_with_devices(self):
        distribution = distribution_lib.DataParallel(devices=self.devices)
        device_mesh = distribution.device_mesh
        self.assertEqual(len(device_mesh.devices), 8)
        self.assertEqual(device_mesh.axis_names, ["batch"])
        self.assertEqual(distribution.batch_dim_name, "batch")

    @mock.patch.object(
        distribution_lib,
        "list_devices",
        return_value=[f"cpu:{i}" for i in range(8)],
    )
    def test_create_with_list_devices(self, mock_list_devices):
        distribution = distribution_lib.DataParallel()
        mock_list_devices.assert_called_once()

        device_mesh = distribution.device_mesh
        self.assertEqual(len(device_mesh.devices), 8)
        self.assertEqual(device_mesh.axis_names, ["batch"])
        self.assertEqual(distribution.batch_dim_name, "batch")

    def test_get_data_layout(self):
        distribution = distribution_lib.DataParallel(
            device_mesh=self.device_mesh
        )

        data = np.arange(16).reshape((4, 2, 2))
        data_layout = distribution.get_data_layout(data.shape)
        self.assertIs(data_layout.device_mesh, self.device_mesh)
        self.assertEqual(data_layout.axes, ("data", None, None))

    @pytest.mark.skipif(testing.jax_uses_gpu(), reason="CI segfault")
    def test_get_variable_layout(self):
        distribution = distribution_lib.DataParallel(
            device_mesh=self.device_mesh
        )

        variable = backend.Variable(initializer=[1, 2, 3])
        variable_layout = distribution.get_variable_layout(variable)
        self.assertIs(variable_layout.device_mesh, self.device_mesh)
        self.assertEqual(variable_layout.axes, (None,))

    @pytest.mark.skipif(testing.jax_uses_gpu(), reason="CI segfault")
    def test_get_variable_layout_with_explicit_layout(self):
        distribution = distribution_lib.DataParallel(
            device_mesh=self.device_mesh
        )

        explicit_mesh = distribution_lib.DeviceMesh((8,), ["x"], self.devices)
        explicit_layout = distribution_lib.TensorLayout(["x"], explicit_mesh)

        variable = backend.Variable(initializer=[1, 2, 3])
        variable._layout = explicit_layout
        variable_layout = distribution.get_variable_layout(variable)
        self.assertIs(variable_layout.device_mesh, explicit_mesh)
        self.assertEqual(variable_layout.axes, explicit_layout.axes)

    def test_get_tensor_layout(self):
        distribution = distribution_lib.DataParallel(
            device_mesh=self.device_mesh
        )

        path = "path/to/tensor"
        tensor_layout = distribution.get_tensor_layout(path)
        self.assertIsNone(tensor_layout)

    def test_distribute_dataset(self):
        # We can only verify the single worker/process case in OSS for now.
        dataset = tf.data.Dataset.range(8)
        distribution = distribution_lib.DataParallel(
            device_mesh=self.device_mesh
        )
        distributed_dataset = distribution.distribute_dataset(dataset)
        self.assertIs(dataset, distributed_dataset)


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="Only JAX has the proper backend distribution lib",
)
class ModelParallelDistributionTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.devices = [f"cpu:{i}" for i in range(8)]
        shape = (2, 4)
        axis_names = ["data", "model"]

        self.device_mesh = distribution_lib.DeviceMesh(
            shape, axis_names, self.devices
        )

    @pytest.mark.skipif(testing.jax_uses_gpu(), reason="CI segfault")
    def test_distribute_weights(self):
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map[".*kernel"] = distribution_lib.TensorLayout([None, "model"])
        layout_map[".*bias"] = distribution_lib.TensorLayout(["model"])

        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="data"
        )
        kernel = backend.Variable(initializer=np.arange(8, 4), name="kernel")
        bias = backend.Variable(initializer=np.arange(4), name="bias")
        rng_seed = backend.Variable(initializer=[0, 1], name="seed")

        kernel_layout = distribution.get_variable_layout(kernel)
        self.assertIs(kernel_layout.device_mesh, self.device_mesh)
        self.assertEqual(kernel_layout.axes, (None, "model"))

        bias_layout = distribution.get_variable_layout(bias)
        self.assertIs(bias_layout.device_mesh, self.device_mesh)
        self.assertEqual(bias_layout.axes, ("model",))

        rng_seed_layout = distribution.get_variable_layout(rng_seed)
        self.assertIs(rng_seed_layout.device_mesh, self.device_mesh)
        self.assertEqual(rng_seed_layout.axes, (None,))

    def test_distribute_data(self):
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="data"
        )

        data = np.arange(16).reshape((4, 2, 2))
        data_layout = distribution.get_data_layout(data.shape)
        self.assertIs(data_layout.device_mesh, self.device_mesh)
        self.assertEqual(data_layout.axes, ("data", None, None))

    def test_get_tensor_layout(self):
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map[".*kernel"] = distribution_lib.TensorLayout([None, "model"])
        layout_map[".*bias"] = distribution_lib.TensorLayout(["model"])
        layout_map["/model/layer/tensor"] = ("data", None)

        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="data"
        )
        layout = distribution.get_tensor_layout("/model/layer/tensor")
        self.assertIs(layout.device_mesh, self.device_mesh)
        self.assertEqual(layout.axes, ("data", None))

        layout = distribution.get_tensor_layout("/model/layer/other_tensor")
        self.assertIsNone(layout)

    @pytest.mark.skipif(testing.jax_uses_gpu(), reason="CI segfault")
    def test_get_variable_layout_with_explicit_layout(self):
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map[".*kernel"] = distribution_lib.TensorLayout([None, "model"])
        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="data"
        )

        explicit_mesh = distribution_lib.DeviceMesh((8,), ["x"], self.devices)
        explicit_layout = distribution_lib.TensorLayout(["x"], explicit_mesh)
        variable = backend.Variable(initializer=[1, 2, 3], name="kernel")
        variable._layout = explicit_layout
        variable_layout = distribution.get_variable_layout(variable)
        self.assertIs(variable_layout.device_mesh, explicit_mesh)
        self.assertEqual(variable_layout.axes, explicit_layout.axes)

    def test_distribute_dataset(self):
        # We can only verify the single worker/process case in OSS for now.
        dataset = tf.data.Dataset.range(8)
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="data"
        )
        distributed_dataset = distribution.distribute_dataset(dataset)
        self.assertIs(dataset, distributed_dataset)


class LayoutMapTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.devices = [f"cpu:{i}" for i in range(8)]
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
        # Test for adding list/tuple as shortcut for TensorLayout
        layout_map["conv/bias"] = ("model",)

        # Make there are two items in the map, and we access them via the
        # underlying container at layout_map._layout_map
        self.assertLen(layout_map, 3)

        kernel_layout = layout_map["dense/kernel"]
        self.assertEqual(kernel_layout.axes, (None, "model"))
        self.assertIs(kernel_layout.device_mesh, self.device_mesh)

        bias_layout = layout_map["dense/bias"]
        self.assertEqual(bias_layout.axes, ("model",))
        self.assertIs(bias_layout.device_mesh, self.device_mesh)

        conv_bias_layout = layout_map["conv/bias"]
        self.assertEqual(conv_bias_layout.axes, ("model",))
        self.assertIs(bias_layout.device_mesh, self.device_mesh)

        with self.assertRaisesRegex(ValueError, "dense/kernel already exist"):
            layout_map["dense/kernel"] = self.sharded_2d

        with self.assertRaisesRegex(ValueError, "should be a TensorLayout"):
            layout_map["conv.kernel"] = ["a", "b"]

    def test_get(self):
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map["dense/kernel"] = self.sharded_2d
        layout_map["dense/bias"] = self.sharded_1d

        layout_map["dense.*kernel"] = self.replicated_2d
        layout_map["dense.*bias"] = self.replicated_1d

        layout_map["bias"] = self.sharded_1d

        self.assertEqual(layout_map["dense/kernel"], self.sharded_2d)
        self.assertEqual(layout_map["dense/bias"], self.sharded_1d)

        self.assertEqual(layout_map["dense_2/kernel"], self.replicated_2d)
        # Map against the wildcard bias rule for dense. This will cause a
        # ValueError
        with self.assertRaisesRegex(
            ValueError, "Path 'dense_2/bias' matches multiple layout"
        ):
            layout_map["dense_2/bias"]

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


# @pytest.mark.skipif(
#     backend.backend() != "tensorflow",
#     reason="Backend specific test",
# )
# class TensorflowDistributionLibTest(testing.TestCase):
#     def setUp(self):
#         super().setUp()
#         # Config virtual devices for testing.
#         cpus = tf.config.list_physical_devices("cpu")
#         context._reset_context()
#         tf.config.set_logical_device_configuration(
#             cpus[0], [tf.config.LogicalDeviceConfiguration()] * 8
#         )

#         dtensor.initialize_accelerator_system("cpu")

#     def tearDown(self) -> None:
#         super().tearDown()
#         dtensor.shutdown_accelerator_system()

#     def test_list_devices(self):
#         self.assertEqual(len(distribution_lib.list_devices()), 8)
#         self.assertEqual(len(distribution_lib.list_devices("cpu")), 8)
#         self.assertEqual(len(distribution_lib.list_devices("cpu")), 8)

#     def test_to_dtensor_mesh(self):
#         devices = [f"cpu:{i}" for i in range(8)]
#         shape = (4, 2)
#         axis_names = ["batch", "model"]

#         mesh = distribution_lib.DeviceMesh(shape, axis_names, devices)
#         dtensor_mesh = backend_dlib._to_dtensor_mesh(mesh)

#         self.assertIsInstance(dtensor_mesh, dtensor.Mesh)
#         self.assertEqual(dtensor_mesh.shape(), list(shape))
#         self.assertEqual(dtensor_mesh.dim_names, axis_names)

#     def test_to_dtensor_layout(self):
#         axes = ["data", None]
#         mesh = distribution_lib.DeviceMesh(
#             (4, 2), ["data", "model"], [f"cpu:{i}" for i in range(8)]
#         )
#         layout = distribution_lib.TensorLayout(axes, mesh)
#         dtensor_layout = backend_dlib._to_dtensor_layout(layout)
#         dtensor_mesh = backend_dlib._to_dtensor_mesh(mesh)
#         self.assertEqual(
#             dtensor_layout,
#             dtensor.Layout(["data", dtensor.UNSHARDED], dtensor_mesh),
#         )

#     def test_validation_for_device_mesh(self):
#         axes = ["data", None]
#         layout = distribution_lib.TensorLayout(axes, device_mesh=None)

#         with self.assertRaisesRegex(
#             ValueError, "Cannot create sharding when device mesh is not set"
#         ):
#             backend_dlib._to_dtensor_layout(layout)


class AutoTPDistributionTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.devices = distribution_lib.list_devices()
        if len(self.devices) < 2:
            self.skipTest("This test requires at least 2 devices.")
        inputs = keras.Input(shape=(4,), name="input_layer")
        x = keras.layers.Dense(8, name="dense_1")(inputs)
        outputs = keras.layers.Dense(2, name="dense_2")(x)
        self.model = keras.Model(inputs, outputs)

    def test_init_with_explicit_device_mesh(self):
        """Tests initialization with a user-provided DeviceMesh."""
        device_mesh = distribution_lib.DeviceMesh(
            shape=(1, 2), axis_names=["data", "model"], devices=self.devices
        )
        distribution = AutoTPDistribution(self.model, device_mesh=device_mesh)

        self.assertIs(distribution.device_mesh, device_mesh)
        self.assertEqual(distribution.batch_dim_name, "data")
        self.assertIsInstance(
            distribution.model,
            keras.src.distribution.tensor_parallel.tensor_parallel.TensorParallelKeras,
        )
        self.assertEqual(distribution.model.world_size, 2)

    @mock.patch.object(
        distribution_lib,
        "list_devices",
        return_value=[f"cpu:{i}" for i in range(2)],
    )
    def test_init_without_device_mesh_for_auto_creation(
        self, mock_list_devices
    ):
        """Tests the automatic creation of DeviceMesh when none is provided."""
        distribution = AutoTPDistribution(self.model, device_mesh=None)
        mock_list_devices.assert_called_once()

        device_mesh = distribution.device_mesh
        self.assertEqual(device_mesh.shape, (1, 2))
        self.assertEqual(device_mesh.axis_names, ("data", "model"))
        self.assertEqual(distribution.batch_dim_name, "data")
        self.assertEqual(distribution.model.world_size, 2)

    def test_init_raises_error_on_missing_data_axis(self):
        """Ensures an error is raised if the DeviceMesh lacks a 'data' axis."""
        device_mesh = distribution_lib.DeviceMesh(
            shape=(2,), axis_names=["model"], devices=self.devices
        )
        with self.assertRaisesRegex(ValueError, "must have a 'data' axis"):
            AutoTPDistribution(self.model, device_mesh=device_mesh)

    def test_get_data_layout(self):
        """Verifies the layout for input data sharding."""
        distribution = AutoTPDistribution(self.model)
        data_shape = (16, 4)
        layout = distribution.get_data_layout(data_shape)

        self.assertEqual(layout.axes, ("data", None))
        self.assertIs(layout.device_mesh, distribution.device_mesh)

    def test_get_variable_layout_warns_and_returns_replicated(self):
        """Verifies that variable layout is handled internally."""
        distribution = AutoTPDistribution(self.model)
        dummy_variable = backend.Variable(initializer=np.zeros((8, 2)))

        with self.assertWarns(UserWarning) as w:
            layout = distribution.get_variable_layout(dummy_variable)

        self.assertIn(
            "Variable layout is determined automatically",
            str(w.warnings[0].message),
        )

        self.assertEqual(layout.axes, (None, None))

    def test_distribute_dataset_in_single_process_mode(self):
        """Tests dataset distribution in a single-process environment."""
        distribution = AutoTPDistribution(self.model)
        dataset = tf.data.Dataset.from_tensor_slices(
            (np.zeros((16, 4)), np.zeros((16, 1)))
        )

        distributed_dataset = distribution.distribute_dataset(dataset)
        self.assertIs(dataset, distributed_dataset)

    def test_full_compile_and_fit_integration(self):
        """A test to ensure the distributed model can compile and train."""
        distribution = AutoTPDistribution(self.model)

        x_train = np.random.rand(16, 4).astype("float32")
        y_train = np.random.randint(0, 2, size=(16, 1))

        dist_model = distribution.model

        with distribution.scope():
            dist_model.compile(
                optimizer=keras.optimizers.Adam(0.01),
                loss=keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True
                ),
                metrics=["accuracy"],
            )

        self.assertEqual(self.model.count_params(), dist_model.count_params())

        history = dist_model.fit(
            x_train,
            y_train,
            epochs=1,
            batch_size=4,
            verbose=0,
        )
        self.assertIn("loss", history.history)
        self.assertIn("accuracy", history.history)
