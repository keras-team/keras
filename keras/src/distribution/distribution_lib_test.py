"""Test for distribution_lib.py."""

import os
import tempfile
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

import keras
from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.backend import distribution_lib as backend_dlib
from keras.src.distribution import distribution_lib


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

    @pytest.mark.skipif(testing.jax_uses_gpu(), reason="CI segfault")
    def test_model_parallel_sharded_variable_loading(self):
        """
        Test that all layer types can load variables with sharding support.

        This test specifically validates:
        1. Variables are sharded across devices using ModelParallel
        2. Each device receives the correct shard shape
        3. Weight loading preserves sharding and correctness
        """
        import os

        import jax
        from absl import logging

        # Ensure we have JAX devices
        jax_devices = jax.devices()
        logging.debug(f"JAX devices available: {len(jax_devices)}")
        for i, device in enumerate(jax_devices):
            logging.debug(f"  Device {i}: {device}")

        # Use available devices instead of the setUp device mesh
        devices = keras.distribution.list_devices()
        num_devices = min(len(devices), len(jax_devices))

        # Create device mesh for model parallelism across available devices
        device_mesh = distribution_lib.DeviceMesh(
            shape=(num_devices,),
            axis_names=["model"],
            devices=devices[:num_devices],
        )

        # Create layout map to shard Dense layer kernels across devices
        layout_map = distribution_lib.LayoutMap(device_mesh)
        layout_map[".*einsum_dense.*kernel"] = (
            "model",
            None,
        )  # Shard EinsumDense
        layout_map[".*(?<!einsum_)dense.*kernel"] = (
            "model",
            None,
        )  # Shard Dense
        layout_map[".*conv1d.*kernel"] = (None, "model", None)  # Shard conv
        layout_map[".*embedding.*embeddings"] = ("model", None)  # Shard emb
        layout_map[".*bias"] = ("model",)  # Shard all biases
        layout_map[".*batch_normalization.*gamma"] = ("model",)  # Shard BN
        layout_map[".*batch_normalization.*beta"] = ("model",)  # Shard BN

        # Set up ModelParallel distribution
        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map,
            batch_dim_name="batch",
        )

        # Apply distribution globally
        with distribution.scope():
            model = keras.Sequential(
                [
                    layers.Input(shape=(32,)),
                    # Core layers that were modified
                    layers.Dense(128, activation="relu", name="dense_1"),
                    layers.Dense(64, activation="relu", name="dense_2"),
                    # EinsumDense layer (also uses kernel/bias like Dense)
                    layers.EinsumDense(
                        "ab,bc->ac", output_shape=32, name="einsum_dense"
                    ),
                    # Embedding layer (modified in commit)
                    layers.Embedding(
                        input_dim=100, output_dim=32, name="embedding"
                    ),
                    layers.Flatten(),
                    # Convolutional layer (modified in commit)
                    layers.Reshape((64, 16)),  # Reshape for conv: 64*16 = 1024
                    layers.Conv1D(
                        32, kernel_size=3, activation="relu", name="conv1d"
                    ),
                    layers.Flatten(),
                    # Normalization layer (modified in commit)
                    layers.BatchNormalization(name="batch_norm"),
                    # Output
                    layers.Dense(10, name="output"),
                ]
            )

            # Build the model to trigger variable creation and sharding
            model.build((None, 32))

            # Initialize weights with some values
            test_input = np.random.randn(4, 32)
            _ = model(test_input)  # Forward pass to initialize variables

            # Verify that variables are actually sharded
            sharded_vars_info = []
            for var in model.weights:
                if hasattr(var, "_layout") and var._layout is not None:
                    # This variable is sharded
                    layout = var._layout
                    full_shape = (
                        var._full_shape
                        if hasattr(var, "_full_shape")
                        else var.shape
                    )
                    sharded_vars_info.append(
                        {
                            "name": var.name,
                            "full_shape": full_shape,
                            "layout": layout,
                            "shards": (
                                len(var._shard_references)
                                if hasattr(var, "_shard_references")
                                else 0
                            ),
                        }
                    )

            self.assertGreater(
                len(sharded_vars_info),
                0,
                "No variables were sharded - ModelParallel may not be working",
            )

            logging.debug(f"Found {len(sharded_vars_info)} sharded variables:")
            for info in sharded_vars_info:
                logging.debug(
                    f"  {info['name']}: full_shape={info['full_shape']}, "
                    f"layout={info['layout']}"
                )

            # Store original weights for comparison (accessing sharded values)
            original_weights = []
            for var in model.weights:
                if hasattr(var, "_layout") and var._layout is not None:
                    # For sharded variables, get the full distributed value
                    original_weights.append(var.value.copy())
                else:
                    original_weights.append(var.numpy().copy())

            # Save model weights to temporary file
            with tempfile.TemporaryDirectory() as temp_dir:
                weights_path = os.path.join(temp_dir, "model.weights.h5")

                # Save weights
                model.save_weights(weights_path)

                new_model = keras.Sequential(
                    [
                        layers.Input(shape=(32,)),
                        layers.Dense(128, activation="relu", name="dense_1"),
                        layers.Dense(64, activation="relu", name="dense_2"),
                        layers.EinsumDense(
                            "ab,bc->ac", output_shape=32, name="einsum_dense"
                        ),
                        layers.Embedding(
                            input_dim=100, output_dim=32, name="embedding"
                        ),
                        layers.Flatten(),
                        layers.Reshape(
                            (64, 16)
                        ),  # Reshape for conv: 64*16 = 1024
                        layers.Conv1D(
                            32, kernel_size=3, activation="relu", name="conv1d"
                        ),
                        layers.Flatten(),
                        layers.BatchNormalization(name="batch_norm"),
                        layers.Dense(10, name="output"),
                    ]
                )

                # Build the new model (this should trigger sharding)
                new_model.build((None, 32))

                # Load weights - this should use the new sharded loading logic
                new_model.load_weights(weights_path)

                # Verify that loaded variables are also sharded
                loaded_sharded_vars_info = []
                for var in new_model.weights:
                    if hasattr(var, "_layout") and var._layout is not None:
                        layout = var._layout
                        full_shape = (
                            var._full_shape
                            if hasattr(var, "_full_shape")
                            else var.shape
                        )
                        loaded_sharded_vars_info.append(
                            {
                                "name": var.name,
                                "full_shape": full_shape,
                                "layout": layout,
                                "shards": (
                                    len(var._shard_references)
                                    if hasattr(var, "_shard_references")
                                    else 0
                                ),
                            }
                        )

                self.assertEqual(
                    len(sharded_vars_info),
                    len(loaded_sharded_vars_info),
                    "Number of sharded variables changed after loading",
                )

                # Verify weights were loaded correctly
                loaded_weights = []
                for var in new_model.weights:
                    if hasattr(var, "_layout") and var._layout is not None:
                        # For sharded variables, get the full distributed value
                        loaded_weights.append(var.value.copy())
                    else:
                        loaded_weights.append(var.numpy().copy())

                # Compare original and loaded weights
                self.assertEqual(len(original_weights), len(loaded_weights))
                for i, (orig, loaded) in enumerate(
                    zip(original_weights, loaded_weights)
                ):
                    np.testing.assert_array_almost_equal(
                        orig,
                        loaded,
                        decimal=5,
                        err_msg=f"Weight {i} mismatch after loading",
                    )

                # Test that inference works with loaded weights
                test_output_original = model(test_input)
                test_output_loaded = new_model(test_input)

                # Outputs should be identical
                np.testing.assert_array_almost_equal(
                    np.asarray(test_output_original),
                    np.asarray(test_output_loaded),
                    decimal=5,
                    err_msg="Inference output mismatch after weight loading",
                )

                # Validate shard shapes on each device
                for i, (orig_info, loaded_info) in enumerate(
                    zip(sharded_vars_info, loaded_sharded_vars_info)
                ):
                    self.assertEqual(
                        orig_info["full_shape"],
                        loaded_info["full_shape"],
                        f"Full shape mismatch for {orig_info['name']}",
                    )
                    self.assertEqual(
                        orig_info["layout"],
                        loaded_info["layout"],
                        f"Layout mismatch for {orig_info['name']}",
                    )
                    self.assertEqual(
                        orig_info["shards"],
                        loaded_info["shards"],
                        f"Shard count mismatch for {orig_info['name']}",
                    )
                logging.debug("Validating shard shapes and device assignments:")
                for var_name in [info["name"] for info in sharded_vars_info]:
                    orig_var = next(
                        v for v in model.weights if v.name == var_name
                    )
                    loaded_var = next(
                        v for v in new_model.weights if v.name == var_name
                    )

                    logging.debug(f"Variable: {var_name}")
                    logging.debug(f"  Full shape: {orig_var.shape}")

                    # Get expected shard shapes from layout
                    try:
                        expected_shard_shape = orig_var._layout.shard_shape(
                            orig_var.shape
                        )
                        logging.debug(
                            f"  Expected shard shape: {expected_shard_shape}"
                        )
                    except Exception as e:
                        logging.debug(
                            f"  Could not determine expected shard shape: {e}"
                        )
                        expected_shard_shape = None

                    # Basic validation that sharding structure exists
                    has_shard_refs_orig = (
                        hasattr(orig_var, "_shard_references")
                        and orig_var._shard_references
                    )
                    has_shard_refs_loaded = (
                        hasattr(loaded_var, "_shard_references")
                        and loaded_var._shard_references
                    )

                    logging.debug(
                        f"  Original has shard references: "
                        f"{has_shard_refs_orig}"
                    )
                    logging.debug(
                        f"  Loaded has shard references: "
                        f"{has_shard_refs_loaded}"
                    )

                    self.assertTrue(
                        has_shard_refs_orig,
                        f"Original {var_name} should have shard references",
                    )
                    self.assertTrue(
                        has_shard_refs_loaded,
                        f"Loaded {var_name} should have shard references",
                    )

                    self.assertGreater(
                        len(orig_var._shard_references),
                        0,
                        f"Original {var_name} has empty shard references",
                    )
                    self.assertGreater(
                        len(loaded_var._shard_references),
                        0,
                        f"Loaded {var_name} has empty shard references",
                    )

                    if has_shard_refs_orig and expected_shard_shape is not None:
                        first_shard = orig_var._shard_references[0]
                        if (
                            isinstance(first_shard, (list, tuple))
                            and len(first_shard) > 0
                        ):
                            shard_data = first_shard[0]
                            self.assertEqual(
                                shard_data.shape,
                                expected_shard_shape,
                                f"Incorrect shard shape for {var_name}. "
                                f"Expected {expected_shard_shape}, "
                                f"got {shard_data.shape}",
                            )

                    if (
                        has_shard_refs_loaded
                        and expected_shard_shape is not None
                    ):
                        first_shard = loaded_var._shard_references[0]
                        if (
                            isinstance(first_shard, (list, tuple))
                            and len(first_shard) > 0
                        ):
                            shard_data = first_shard[0]
                            self.assertEqual(
                                shard_data.shape,
                                expected_shard_shape,
                                f"Incorrect shard shape for loaded "
                                f"{var_name}. "
                                f"Expected {expected_shard_shape}, "
                                f"got {shard_data.shape}",
                            )

                logging.debug(
                    f"ModelParallel test passed: {len(sharded_vars_info)} "
                    f"variables "
                    f"sharded across {num_devices} devices"
                )


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
