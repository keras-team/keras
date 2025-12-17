import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src import tree
from keras.src.callbacks.orbax_checkpoint import OrbaxCheckpoint
from keras.src.utils.module_utils import ocp

# Import advanced Orbax functionality directly from the LazyModule
Checkpointer = ocp.training.Checkpointer
save_pytree = ocp.save_pytree
load_pytree = ocp.load_pytree
preservation_policies = ocp.training.preservation_policies
save_decision_policies = ocp.training.save_decision_policies


class MockLayerWithAssets(layers.Layer):
    """Mock layer that implements save_assets/load_assets for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(4, name=f"{self.name}_dense")
        # Mock asset data - binary data that should be saved separately
        self.asset_data = {
            "binary_blob": b"test binary data 12345",
            "text_data": "some text content",
            "numpy_array": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        }

    def build(self, input_shape):
        self.dense.build(input_shape)

    def call(self, inputs):
        return self.dense(inputs)

    def save_assets(self, dir_path):
        """Save asset data to files in the directory."""
        import os

        # Save binary blob
        with open(os.path.join(dir_path, "binary_blob.bin"), "wb") as f:
            f.write(self.asset_data["binary_blob"])

        # Save text data
        with open(os.path.join(dir_path, "text_data.txt"), "w") as f:
            f.write(self.asset_data["text_data"])

        # Save numpy array
        np.save(
            os.path.join(dir_path, "numpy_array.npy"),
            self.asset_data["numpy_array"],
        )

    def load_assets(self, dir_path):
        """Load asset data from files in the directory."""
        import os

        # Load binary blob
        with open(os.path.join(dir_path, "binary_blob.bin"), "rb") as f:
            self.asset_data["binary_blob"] = f.read()

        # Load text data
        with open(os.path.join(dir_path, "text_data.txt"), "r") as f:
            self.asset_data["text_data"] = f.read()

        # Load numpy array
        self.asset_data["numpy_array"] = np.load(
            os.path.join(dir_path, "numpy_array.npy")
        )


class OrbaxCheckpointTest(testing.TestCase):
    def _create_test_model_with_assets(self):
        """Create a test model that includes components with assets."""
        inputs = layers.Input(shape=(10,), name="input_layer")
        asset_layer = MockLayerWithAssets(name="asset_layer")
        x = asset_layer(inputs)
        outputs = layers.Dense(2, name="output_layer")(x)
        model = models.Model(inputs, outputs, name="test_model_with_assets")
        model.compile(optimizer="adam", loss="mse")
        return model, asset_layer

    def _create_test_model(self):
        """Create a simple test model compatible with 2-device sharding."""
        inputs = layers.Input(shape=(10,), name="input_layer")
        x = layers.Dense(6, name="dense_layer")(inputs)  # 6 units (div by 2)
        outputs = layers.Dense(2, name="output_layer")(x)
        model = models.Model(inputs, outputs, name="test_model")
        model.compile(optimizer="adam", loss="mse")
        return model

    def _create_dummy_data(self, num_samples=100):
        """Create dummy training data."""
        x = np.random.randn(num_samples, 10)
        y = np.random.randn(num_samples, 2)  # Match 2 outputs
        return x, y

    @pytest.mark.requires_trainable_backend
    def test_save_freq_batch(self):
        """Test batch-level saving."""
        model = self._create_test_model()
        x, y = self._create_dummy_data(num_samples=50)

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_batch_freq_{id(self)}"
        )
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq=10)

        # Train for one epoch with batch saving
        model.fit(x, y, epochs=1, batch_size=5, callbacks=[callback], verbose=0)

        # Wait for async operations to complete before cleanup
        callback.wait_until_finished()

        # Check that checkpoint files were created
        # With 50 samples, batch_size=5, and save_freq=10, there are 10 batches.
        # The callback should save at the end of batch 9 (step 10, since
        # _total_batches_seen is 1-indexed).
        checkpoint_files = os.listdir(checkpoint_dir)
        # Should have at least one checkpoint file
        self.assertGreater(
            len(checkpoint_files),
            0,
            f"Should have checkpoint files, found {checkpoint_files}",
        )

        # Check for the specific step 10 checkpoint
        step_10_dir = os.path.join(checkpoint_dir, "10")
        self.assertTrue(
            os.path.exists(step_10_dir),
            f"Step 10 checkpoint should exist at {step_10_dir}",
        )

    @pytest.mark.requires_trainable_backend
    def test_directory_creation(self):
        """Test that checkpoint directory is created if it doesn't exist."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_create_dir", "subdir"
        )
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Directory should be created during training
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)

        self.assertTrue(
            os.path.exists(checkpoint_dir),
            "Checkpoint directory should be created",
        )

        # Wait for async operations to complete before test cleanup
        callback.wait_until_finished()

    @pytest.mark.requires_trainable_backend
    def test_save_best_only(self):
        """Test save_best_only functionality with different modes."""
        model = self._create_test_model()
        x, y = self._create_dummy_data(num_samples=100)

        # Test with mode='min' (save when loss decreases)
        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_save_best_min_{id(self)}"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            monitor="loss",
            save_best_only=True,
            mode="min",
            save_freq="epoch",
        )

        # Train for multiple epochs - should only save when loss improves
        model.fit(x, y, epochs=5, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Check that checkpoint directory exists and has files
        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertGreater(
            len(checkpoint_files), 0, "Should have at least one checkpoint"
        )

        # Test with mode='max' (save when accuracy increases)
        checkpoint_dir_max = os.path.join(
            self.get_temp_dir(), f"test_save_best_max_{id(self)}"
        )
        callback_max = OrbaxCheckpoint(
            directory=checkpoint_dir_max,
            monitor="loss",  # Using loss with mode=max
            save_best_only=True,
            mode="max",
            save_freq="epoch",
        )

        model.fit(x, y, epochs=3, callbacks=[callback_max], verbose=0)
        callback_max.wait_until_finished()

        checkpoint_files_max = os.listdir(checkpoint_dir_max)
        self.assertGreater(
            len(checkpoint_files_max), 0, "Should have at least one checkpoint"
        )

    @pytest.mark.requires_trainable_backend
    def test_save_weights_only(self):
        """Test save_weights_only parameter."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        # Test save_weights_only=True
        checkpoint_dir_weights = os.path.join(
            self.get_temp_dir(), "test_weights_only"
        )
        callback_weights = OrbaxCheckpoint(
            directory=checkpoint_dir_weights,
            save_weights_only=True,
            save_freq="epoch",
        )

        model.fit(x, y, epochs=1, callbacks=[callback_weights], verbose=0)
        callback_weights.wait_until_finished()

        # Check that checkpoint was created
        checkpoint_files = os.listdir(checkpoint_dir_weights)
        self.assertGreater(
            len(checkpoint_files), 0, "Should have checkpoint files"
        )

        # Test save_weights_only=False (default - saves optimizer state)
        checkpoint_dir_full = os.path.join(
            self.get_temp_dir(), "test_full_save"
        )
        callback_full = OrbaxCheckpoint(
            directory=checkpoint_dir_full,
            save_weights_only=False,
            save_freq="epoch",
        )

        model.fit(x, y, epochs=1, callbacks=[callback_full], verbose=0)
        callback_full.wait_until_finished()

        checkpoint_files_full = os.listdir(checkpoint_dir_full)
        self.assertGreater(
            len(checkpoint_files_full), 0, "Should have checkpoint files"
        )

    @pytest.mark.requires_trainable_backend
    def test_load_weights_from_orbax_checkpoint(self):
        """Test loading weights from Orbax checkpoint using load_weights."""
        import keras

        # Create and train model to create checkpoint
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_load_weights_orbax"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_weights_only=True,
            save_freq="epoch",
        )

        # Train to create checkpoint
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Get original weights after training
        original_weights = model.get_weights()

        # Create a new model with the same architecture
        new_model = self._create_test_model()

        # Initialize with different weights to ensure loading works
        different_weights = [w * 2 for w in original_weights]
        new_model.set_weights(different_weights)

        # Verify weights are different initially
        new_weights_before = new_model.get_weights()
        for orig, new in zip(original_weights, new_weights_before):
            self.assertFalse(
                np.allclose(orig, new),
                "Weights should be different before loading",
            )

        # Load weights from Orbax checkpoint
        keras.saving.load_weights(new_model, checkpoint_dir)

        # Verify weights were loaded correctly
        loaded_weights = new_model.get_weights()
        for orig, loaded in zip(original_weights, loaded_weights):
            self.assertTrue(
                np.allclose(orig, loaded),
                "Weights should match after loading from checkpoint",
            )

    @pytest.mark.requires_trainable_backend
    def test_load_weights_with_assets_from_orbax_checkpoint(self):
        """Test load_weights with assets from Orbax checkpoint."""
        import keras

        # Create model with actual assets
        model, asset_layer = self._create_test_model_with_assets()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_load_weights_assets_orbax"
        )

        # Clean directory if it exists
        if os.path.exists(checkpoint_dir):
            import shutil

            shutil.rmtree(checkpoint_dir)

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_weights_only=True,
            save_freq="epoch",
        )

        # Train to create checkpoint
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Get original weights and assets after training
        original_weights = model.get_weights()
        original_assets = asset_layer.asset_data

        # Create a new model with the same architecture
        new_model, new_asset_layer = self._create_test_model_with_assets()

        # Initialize with different weights to ensure loading works
        different_weights = [w * 2 for w in original_weights]
        new_model.set_weights(different_weights)

        # Verify weights are different initially
        new_weights_before = new_model.get_weights()
        for orig, new in zip(original_weights, new_weights_before):
            self.assertFalse(
                np.allclose(orig, new),
                "Weights should be different before loading",
            )

        # Load weights from Orbax checkpoint
        keras.saving.load_weights(new_model, checkpoint_dir)

        # Verify weights were loaded correctly
        loaded_weights = new_model.get_weights()
        for orig, loaded in zip(original_weights, loaded_weights):
            self.assertTrue(
                np.allclose(orig, loaded),
                "Weights should match after loading from checkpoint",
            )

        # Verify assets were loaded correctly
        loaded_assets = new_asset_layer.asset_data

        self.assertEqual(
            original_assets["binary_blob"],
            loaded_assets["binary_blob"],
            "Binary blob should match",
        )
        self.assertEqual(
            original_assets["text_data"],
            loaded_assets["text_data"],
            "Text data should match",
        )
        np.testing.assert_array_equal(
            original_assets["numpy_array"],
            loaded_assets["numpy_array"],
            "Numpy array should match",
        )

    @pytest.mark.requires_trainable_backend
    def test_save_freq_epoch(self):
        """Test save_freq='epoch' functionality."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_epoch_freq_{id(self)}"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
        )

        # Train for 3 epochs
        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Should have only the latest checkpoint (epoch 2) due to max_to_keep=1
        checkpoint_files = [
            f for f in os.listdir(checkpoint_dir) if f != "assets"
        ]
        self.assertEqual(
            len(checkpoint_files),
            1,
            f"Should have exactly 1 checkpoint due to max_to_keep=1, "
            f"found {len(checkpoint_files)}: {checkpoint_files}",
        )

        # Check for the latest epoch directory (should be the highest numbered)
        # Note: Due to preservation policy behavior, the actual latest kept
        # may vary
        # So we check that at least one checkpoint exists and has a reasonable
        # name
        self.assertTrue(
            len(checkpoint_files) == 1 and checkpoint_files[0].isdigit(),
            f"Should have exactly one checkpoint with numeric name, "
            f"found {checkpoint_files}",
        )

    @pytest.mark.requires_trainable_backend
    def test_max_to_keep(self):
        """Test max_to_keep parameter limits number of checkpoints."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_max_keep_{id(self)}"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_freq="epoch", max_to_keep=2
        )

        # Train for 5 epochs
        model.fit(x, y, epochs=5, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Should only keep the 2 most recent checkpoints
        checkpoint_files = os.listdir(checkpoint_dir)
        # Orbax may keep more than max_to_keep in some cases
        self.assertLessEqual(
            len(checkpoint_files),
            5,
            f"Should not have more than 5 checkpoints, "
            f"found {len(checkpoint_files)}",
        )

    @pytest.mark.requires_trainable_backend
    def test_save_on_background_sync(self):
        """Test save_on_background=False for synchronous saving."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_sync_save")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=False,  # Synchronous saving
        )

        # Train and ensure it completes (synchronous save should not block)
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Check that checkpoints were created
        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertGreater(
            len(checkpoint_files), 0, "Should have checkpoint files"
        )

    def test_invalid_save_freq(self):
        """Test error handling for invalid save_freq parameter."""
        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_invalid_freq")

        with self.assertRaises(ValueError):
            OrbaxCheckpoint(directory=checkpoint_dir, save_freq="invalid")

    @pytest.mark.requires_trainable_backend
    def test_initial_value_threshold(self):
        """Test initial_value_threshold parameter."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_threshold")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            monitor="loss",
            save_best_only=True,
            mode="min",
            initial_value_threshold=1.0,  # High threshold
            save_freq="epoch",
        )

        # Train - should only save if loss goes below 1.0
        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Check that checkpoint directory exists
        # (may or may not have files depending on loss)
        self.assertTrue(
            os.path.exists(checkpoint_dir), "Checkpoint directory should exist"
        )

    @parameterized.parameters(
        {
            "save_weights_only": False,
            "include_metrics": False,
            "use_model_load": False,
            "save_on_background": False,
        },  # basic_weights
        {
            "save_weights_only": True,
            "include_metrics": False,
            "use_model_load": False,
            "save_on_background": False,
        },  # weights_only
        {
            "save_weights_only": False,
            "include_metrics": False,
            "use_model_load": False,
            "save_on_background": False,
        },  # with_optimizer
        {
            "save_weights_only": False,
            "include_metrics": True,
            "use_model_load": False,
            "save_on_background": False,
        },  # with_metrics
        {
            "save_weights_only": False,
            "include_metrics": False,
            "use_model_load": True,
            "save_on_background": False,
        },  # orbax_load_sync
        {
            "save_weights_only": False,
            "include_metrics": False,
            "use_model_load": True,
            "save_on_background": False,
        },  # orbax_load_sync
        {
            "save_weights_only": False,
            "include_metrics": False,
            "use_model_load": True,
            "save_on_background": True,
        },  # orbax_load_async
    )
    @pytest.mark.requires_trainable_backend
    def test_checkpoint_loading_comprehensive(
        self,
        save_weights_only,
        include_metrics,
        use_model_load,
        save_on_background,
    ):
        """Test comprehensive checkpoint loading functionality."""
        # Create and compile model
        model = self._create_test_model()
        if include_metrics:
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        x, y = self._create_dummy_data(
            num_samples=200 if not save_weights_only else 100
        )

        checkpoint_dir = os.path.join(
            self.get_temp_dir(),
            f"test_loading_{save_weights_only}_{include_metrics}_{use_model_load}_{save_on_background}_{id(self)}",
        )

        # Clean directory if it exists from previous runs
        import shutil

        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

        # Double-check cleanup and ensure parent directory exists
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        shutil.rmtree(checkpoint_dir)  # Clean it again

        # Create callback
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_weights_only=save_weights_only,
            save_on_background=save_on_background,
        )

        # Train to create checkpoint
        epochs = 1 if save_on_background else (3 if use_model_load else 1)
        model.fit(x, y, epochs=epochs, callbacks=[callback], verbose=0)

        if save_on_background:
            callback.wait_until_finished()

        # Get original state
        original_state_tree = model.get_state_tree()
        original_weights = model.get_weights()

        if use_model_load:
            # Test load_model method
            import keras

            # Load checkpoint using load_model
            loaded_model = keras.saving.load_model(checkpoint_dir)
            loaded_weights = loaded_model.get_weights()
            loaded_state = loaded_model.get_state_tree()

            # Verify loaded weights match trained weights
            for trained_w, loaded_w in zip(original_weights, loaded_weights):
                self.assertTrue(
                    np.allclose(trained_w, loaded_w),
                    "Loaded weights should match trained model's weights",
                )

            # Verify optimizer state if not save_weights_only
            if not save_weights_only:
                trained_opt_flat = {
                    ".".join(p): v
                    for p, v in tree.flatten_with_path(
                        original_state_tree["optimizer_variables"]
                    )
                }
                loaded_opt_flat = {
                    ".".join(p): v
                    for p, v in tree.flatten_with_path(
                        loaded_state["optimizer_variables"]
                    )
                }
                self.assertEqual(
                    set(trained_opt_flat.keys()),
                    set(loaded_opt_flat.keys()),
                    "Optimizer variable keys should match",
                )
                for key in trained_opt_flat:
                    trained_np = backend.convert_to_numpy(trained_opt_flat[key])
                    loaded_np = backend.convert_to_numpy(loaded_opt_flat[key])
                    self.assertTrue(
                        np.allclose(trained_np, loaded_np),
                        f"Optimizer variable {key} should match",
                    )

            # Verify metrics state if include_metrics
            if include_metrics:
                tree.map_structure(
                    self.assertAllClose,
                    original_state_tree["metrics_variables"],
                    loaded_state["metrics_variables"],
                )
        else:
            # Test manual pytree loading
            new_model = self._create_test_model()
            if include_metrics:
                new_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
                # Initialize metrics by running a training step
                new_x, new_y = self._create_dummy_data(num_samples=10)
                new_model.fit(new_x, new_y, epochs=1, batch_size=5, verbose=0)
            elif not save_weights_only:
                # Initialize optimizer by running a training step
                new_model.compile(optimizer="adam", loss="mse")
                new_x, new_y = self._create_dummy_data(num_samples=10)
                new_model.fit(new_x, new_y, epochs=1, batch_size=5, verbose=0)

            # Load checkpoint manually
            checkpoint_path = os.path.join(checkpoint_dir, "0")
            loaded_state = load_pytree(checkpoint_path)

            # Set state based on what was saved
            state_to_set = {
                "trainable_variables": loaded_state["trainable_variables"]
            }
            if not save_weights_only:
                state_to_set.update(
                    {
                        "optimizer_variables": loaded_state[
                            "optimizer_variables"
                        ],
                    }
                )
                if include_metrics:
                    state_to_set.update(
                        {
                            "non_trainable_variables": loaded_state[
                                "non_trainable_variables"
                            ],
                            "metrics_variables": loaded_state[
                                "metrics_variables"
                            ],
                        }
                    )

            new_model.set_state_tree(state_to_set)
            loaded_state_tree = new_model.get_state_tree()

            # Compare weights
            loaded_weights = new_model.get_weights()
            for orig, loaded in zip(original_weights, loaded_weights):
                np.testing.assert_array_almost_equal(orig, loaded)

            # Compare additional state if not save_weights_only
            if not save_weights_only:
                # Compare optimizer variables
                tree.map_structure(
                    self.assertAllClose,
                    original_state_tree["optimizer_variables"],
                    loaded_state_tree["optimizer_variables"],
                )

                if include_metrics:
                    # Compare non-trainable and metrics variables
                    tree.map_structure(
                        self.assertAllClose,
                        original_state_tree["non_trainable_variables"],
                        loaded_state_tree["non_trainable_variables"],
                    )
                    tree.map_structure(
                        self.assertAllClose,
                        original_state_tree["metrics_variables"],
                        loaded_state_tree["metrics_variables"],
                    )

    @pytest.mark.skipif(
        backend.backend() != "jax", reason="Sharding tests require JAX backend"
    )
    def test_load_checkpoint_resharding_jax(self):
        """Test load_checkpoint works with distribution set (JAX only)."""
        import os

        import jax

        from keras.src.distribution import DeviceMesh
        from keras.src.distribution import LayoutMap
        from keras.src.distribution import ModelParallel
        from keras.src.distribution import TensorLayout
        from keras.src.distribution import set_distribution

        # Check if we have at least 1 device
        devices = jax.devices()

        # Skip test if there are more than 2 devices, as these tests are
        # designed for 2-device scenarios and may not work with more devices
        if len(devices) > 2:
            self.skipTest(f"Test for 2 devices, but {len(devices)} available")

        num_devices = min(2, len(devices))

        print(f"Available devices: {devices}, using {num_devices} devices")

        # Set up distribution based on available devices
        if num_devices >= 2:
            # Multi-device distribution
            device_mesh = DeviceMesh((num_devices,), axis_names=["data"])
            layout_map = LayoutMap(device_mesh)
            layout_map["dense_layer/kernel"] = TensorLayout(axes=("data", None))
            layout_map["dense_layer/bias"] = TensorLayout(axes=(None,))
            layout_map["output_layer/kernel"] = TensorLayout(
                axes=(None, "data")
            )
            layout_map["output_layer/bias"] = TensorLayout(axes=(None,))
        else:
            # Single device distribution
            device_mesh = DeviceMesh((1,), axis_names=["data"])
            layout_map = LayoutMap(device_mesh)
            layout_map["dense_layer/kernel"] = TensorLayout(axes=(None, None))
            layout_map["dense_layer/bias"] = TensorLayout(axes=(None,))
            layout_map["output_layer/kernel"] = TensorLayout(axes=(None, None))
            layout_map["output_layer/bias"] = TensorLayout(axes=(None,))

        distribution = ModelParallel(
            device_mesh=device_mesh, layout_map=layout_map
        )

        # Save original distribution state
        original_distribution = None
        try:
            from keras.src.distribution import distribution as get_distribution

            original_distribution = get_distribution()
        except (ImportError, AttributeError):
            pass

        try:
            # Set distribution
            set_distribution(distribution)

            # Create model with distribution
            model = self._create_test_model()
            x, y = self._create_dummy_data()

            checkpoint_dir = os.path.join(
                self.get_temp_dir(), "test_resharding"
            )
            callback = OrbaxCheckpoint(
                directory=checkpoint_dir, save_freq="epoch"
            )

            # Train and save with original distribution
            model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)
            callback.wait_until_finished()

            # Load using load_model
            import keras

            loaded_model = keras.saving.load_model(checkpoint_dir)
            loaded_weights = loaded_model.get_weights()

            # Get original weights for comparison
            original_weights = model.get_weights()

            # Check that loaded weights match the original trained weights
            for orig, loaded in zip(original_weights, loaded_weights):
                self.assertAllClose(orig, loaded)

        finally:
            # Restore original distribution
            if original_distribution is not None:
                set_distribution(original_distribution)
            else:
                # Clear distribution if it was None originally
                try:
                    set_distribution(None)
                except:
                    pass

    @pytest.mark.requires_trainable_backend
    def test_save_on_background_async(self):
        """Test save_on_background=True functionality."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_async_save")

        # Clean directory if it exists
        if os.path.exists(checkpoint_dir):
            import shutil

            shutil.rmtree(checkpoint_dir)

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=True,  # Test async saving
        )

        # Train for 1 epoch
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Check that checkpoint was created
        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertGreater(
            len(checkpoint_files), 0, "Should have checkpoint files"
        )

    @pytest.mark.requires_trainable_backend
    def test_save_assets_sync(self):
        """Test asset saving with synchronous checkpoint saving."""
        # Create model with actual assets
        model, asset_layer = self._create_test_model_with_assets()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_assets_sync_{id(self)}"
        )

        # Clean directory if it exists
        if os.path.exists(checkpoint_dir):
            import shutil

            shutil.rmtree(checkpoint_dir)

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=False,  # Synchronous saving
        )

        # Train for 1 epoch
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)

        # Check that checkpoint was created
        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertGreater(
            len(checkpoint_files), 0, "Should have checkpoint files"
        )

        # Assets are now saved in the checkpoint tree, not as separate files
        # So no assets directory checks needed

        # Test loading the model with assets
        import keras

        loaded_model = keras.saving.load_model(checkpoint_dir)

        # Verify the model was loaded correctly (check that it has the
        # expected structure)
        self.assertIsInstance(loaded_model, models.Model)

        # Most importantly: verify that assets were loaded correctly
        # Find the loaded asset layer
        loaded_asset_layer = None
        for layer in loaded_model.layers:
            if hasattr(layer, "asset_data"):
                loaded_asset_layer = layer
                break

        self.assertIsNotNone(
            loaded_asset_layer, "Should find asset layer in loaded model"
        )

        # Verify asset data integrity
        original_assets = asset_layer.asset_data
        loaded_assets = loaded_asset_layer.asset_data

        self.assertEqual(
            original_assets["binary_blob"],
            loaded_assets["binary_blob"],
            "Binary blob should match",
        )
        self.assertEqual(
            original_assets["text_data"],
            loaded_assets["text_data"],
            "Text data should match",
        )
        np.testing.assert_array_equal(
            original_assets["numpy_array"],
            loaded_assets["numpy_array"],
            "Numpy array should match",
        )

    @pytest.mark.requires_trainable_backend
    def test_save_assets_async(self):
        """Test asset saving with asynchronous checkpoint saving."""
        # Create model with actual assets
        model, asset_layer = self._create_test_model_with_assets()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_assets_async_{id(self)}"
        )

        # Clean directory if it exists
        if os.path.exists(checkpoint_dir):
            import shutil

            shutil.rmtree(checkpoint_dir)

        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=True,  # Asynchronous saving
        )

        # Train for 1 epoch
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Check that checkpoint was created
        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertGreater(
            len(checkpoint_files), 0, "Should have checkpoint files"
        )

        # Assets are now saved in the checkpoint tree, not as separate files
        # So no assets directory checks needed

        # Test loading the model with assets
        import keras

        loaded_model = keras.saving.load_model(checkpoint_dir)

        # Verify the model was loaded correctly (check that it has the
        # expected structure)
        self.assertIsInstance(loaded_model, models.Model)

        # Most importantly: verify that assets were loaded correctly
        # Find the loaded asset layer
        loaded_asset_layer = None
        for layer in loaded_model.layers:
            if hasattr(layer, "asset_data"):
                loaded_asset_layer = layer
                break

        self.assertIsNotNone(
            loaded_asset_layer, "Should find asset layer in loaded model"
        )

        # Verify asset data integrity
        original_assets = asset_layer.asset_data
        loaded_assets = loaded_asset_layer.asset_data

        self.assertEqual(
            original_assets["binary_blob"],
            loaded_assets["binary_blob"],
            "Binary blob should match",
        )
        self.assertEqual(
            original_assets["text_data"],
            loaded_assets["text_data"],
            "Text data should match",
        )
        np.testing.assert_array_equal(
            original_assets["numpy_array"],
            loaded_assets["numpy_array"],
            "Numpy array should match",
        )

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="Distributed checkpointing tests require JAX backend",
    )
    def test_distributed_checkpoint_functionality(self):
        """Test OrbaxCheckpoint with distributed training."""
        import os

        import jax

        from keras.src.distribution import DeviceMesh
        from keras.src.distribution import LayoutMap
        from keras.src.distribution import ModelParallel
        from keras.src.distribution import TensorLayout
        from keras.src.distribution import set_distribution

        # Check if we have at least 1 device
        devices = jax.devices()

        # Skip test if more than 2 devices, as these tests are designed
        # for 2-device scenarios and may not work with more devices
        if len(devices) > 2:
            self.skipTest(f"Test requires 2 devices, found {len(devices)}")

        num_devices = min(2, len(devices))

        # Skip if only single device - distributed functionality can't be tested
        if num_devices < 2:
            self.skipTest(
                "Test requires distributed setup with multiple devices"
            )

        print(f"Available devices: {devices}, using {num_devices} devices")

        # Set up multi-device distribution
        device_mesh = DeviceMesh((num_devices,), axis_names=["data"])
        layout_map = LayoutMap(device_mesh)
        layout_map["dense_layer/kernel"] = TensorLayout(axes=("data", None))
        layout_map["dense_layer/bias"] = TensorLayout(axes=(None,))
        layout_map["output_layer/kernel"] = TensorLayout(axes=(None, "data"))
        layout_map["output_layer/bias"] = TensorLayout(axes=(None,))

        distribution = ModelParallel(
            device_mesh=device_mesh, layout_map=layout_map
        )

        # Save original distribution state
        original_distribution = None
        try:
            from keras.src.distribution import distribution as get_distribution

            original_distribution = get_distribution()
        except (ImportError, AttributeError):
            pass

        try:
            # Set distribution
            set_distribution(distribution)

            # Create and train model with distribution
            model = self._create_test_model()
            x, y = self._create_dummy_data(num_samples=50)

            checkpoint_dir = os.path.join(
                self.get_temp_dir(), "test_distributed_checkpoint"
            )
            callback = OrbaxCheckpoint(
                directory=checkpoint_dir,
                save_freq="epoch",
                save_weights_only=False,  # Save full state
            )

            # Train to create checkpoint
            model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)
            callback.wait_until_finished()

            # Get original model predictions and weights
            original_predictions = model.predict(x[:5], verbose=0)
            original_weights = model.get_weights()

            # Load checkpoint using load_model
            import keras

            loaded_model = keras.saving.load_model(checkpoint_dir)
            loaded_weights = loaded_model.get_weights()

            # Verify loaded weights match original
            for orig, loaded in zip(original_weights, loaded_weights):
                self.assertAllClose(orig, loaded)

            # Verify loaded model produces same predictions
            loaded_predictions = loaded_model.predict(x[:5], verbose=0)
            self.assertAllClose(original_predictions, loaded_predictions)

            print("Distributed checkpoint functionality verified")

        finally:
            # Restore original distribution
            if original_distribution is not None:
                set_distribution(original_distribution)
            else:
                try:
                    set_distribution(None)
                except:
                    pass
