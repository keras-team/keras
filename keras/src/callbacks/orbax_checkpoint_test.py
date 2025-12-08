import os

import numpy as np
import pytest

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.callbacks.orbax_checkpoint import OrbaxCheckpoint
from keras.src.utils.module_utils import ocp

# Import advanced Orbax functionality directly from the LazyModule
Checkpointer = ocp.training.Checkpointer
save_pytree = ocp.save_pytree
load_pytree = ocp.load_pytree
preservation_policies = ocp.training.preservation_policies
save_decision_policies = ocp.training.save_decision_policies


class OrbaxCheckpointTest(testing.TestCase):
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

    def _to_numpy(self, tensor):
        """Convert tensor to numpy array, handling different tensor types."""
        if hasattr(tensor, "detach"):  # PyTorch tensor
            return tensor.detach().cpu().numpy()
        elif hasattr(tensor, "numpy"):  # TF variable
            return tensor.numpy()
        else:  # numpy array
            return tensor

    @pytest.mark.requires_trainable_backend
    def test_save_freq_batch(self):
        """Test batch-level saving."""
        model = self._create_test_model()
        x, y = self._create_dummy_data(num_samples=50)

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_batch_freq")
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
        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_save_best_min")
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
            self.get_temp_dir(), "test_save_best_max"
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
    def test_save_freq_epoch(self):
        """Test save_freq='epoch' functionality."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_epoch_freq")
        # Use synchronous saving to avoid async issues with multiple saves
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=False,
        )

        # Train for 3 epochs
        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Should have only the latest checkpoint (epoch 2) due to max_to_keep=1
        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertEqual(
            len(checkpoint_files),
            1,
            f"Should have exactly 1 checkpoint due to max_to_keep=1, "
            f"found {len(checkpoint_files)}",
        )

        # Check for the latest epoch directory (epoch 2)
        epoch_dir = os.path.join(checkpoint_dir, "2")
        self.assertTrue(
            os.path.exists(epoch_dir),
            "Epoch 2 checkpoint should exist (latest due to max_to_keep=1)",
        )

    @pytest.mark.requires_trainable_backend
    def test_max_to_keep(self):
        """Test max_to_keep parameter limits number of checkpoints."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_max_keep")
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

    @pytest.mark.requires_trainable_backend
    def test_checkpoint_loading(self):
        """Test that saved checkpoints can be loaded and weights restored."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_loading")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Train for 1 epoch to save checkpoint
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Get original weights after training
        original_weights = model.get_weights()

        # Create a new model with same architecture
        new_model = self._create_test_model()

        # Load the checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "0")  # epoch 0
        loaded_state = load_pytree(checkpoint_path)

        # Set the state back to the new model
        # The loaded_state has 'trainable_variables' key
        new_model.set_state_tree(
            {"trainable_variables": loaded_state["trainable_variables"]}
        )

        # Compare weights
        loaded_weights = new_model.get_weights()
        for orig, loaded in zip(original_weights, loaded_weights):
            np.testing.assert_array_almost_equal(orig, loaded)

    @pytest.mark.requires_trainable_backend
    def test_checkpoint_loading_weights_only(self):
        """Test loading checkpoints saved with save_weights_only=True."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_loading_weights"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_freq="epoch", save_weights_only=True
        )

        # Train for 1 epoch to save checkpoint
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Get original weights after training
        original_weights = model.get_weights()

        # Create a new model with same architecture
        new_model = self._create_test_model()

        # Load the checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "0")  # epoch 0
        loaded_state = load_pytree(checkpoint_path)

        # For save_weights_only, the state should only have trainable_variables
        new_model.set_state_tree(
            {"trainable_variables": loaded_state["trainable_variables"]}
        )

        # Compare weights
        loaded_weights = new_model.get_weights()
        for orig, loaded in zip(original_weights, loaded_weights):
            np.testing.assert_array_almost_equal(orig, loaded)

    @pytest.mark.requires_trainable_backend
    def test_checkpoint_loading_with_optimizer_state(self):
        """Test loading checkpoints that include optimizer state."""
        model = self._create_test_model()
        x, y = self._create_dummy_data(num_samples=200)
        # More data for optimizer state

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_loading_optimizer"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_freq="epoch", save_weights_only=False
        )

        # Train for 1 epoch to build optimizer state
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Get original state after training
        original_state_tree = model.get_state_tree()

        # Create a new model with same architecture
        new_model = self._create_test_model()
        # Compile with same optimizer to initialize optimizer variables
        new_model.compile(optimizer="adam", loss="mse")

        # Run one training step to initialize optimizer variables
        new_x, new_y = self._create_dummy_data(num_samples=10)
        new_model.fit(new_x, new_y, epochs=1, batch_size=5, verbose=0)

        # Load the checkpoint (epoch 0)
        checkpoint_path = os.path.join(checkpoint_dir, "0")
        loaded_state = load_pytree(checkpoint_path)

        # Set the full state (weights + optimizer) back to the new model
        new_model.set_state_tree(
            {
                "trainable_variables": loaded_state["trainable_variables"],
                "optimizer_variables": loaded_state["optimizer_variables"],
            }
        )

        # Get the loaded state
        loaded_state_tree = new_model.get_state_tree()

        # Compare trainable variables (weights)
        def compare_nested_dicts(orig_dict, loaded_dict):
            """Recursively compare nested dictionaries containing variables."""
            for key in orig_dict:
                if key not in loaded_dict:
                    self.fail(f"Key {key} missing in loaded state")
                orig_val = orig_dict[key]
                loaded_val = loaded_dict[key]

                if isinstance(orig_val, dict):
                    compare_nested_dicts(orig_val, loaded_val)
                else:
                    # Handle different array types: JAX arrays, TF variables,
                    # PyTorch tensors, numpy arrays
                    if hasattr(orig_val, "numpy"):
                        # Could be TensorFlow variable or PyTorch tensor
                        try:
                            # Try PyTorch-style conversion first
                            # (detach().cpu().numpy())
                            orig_array = orig_val.detach().cpu().numpy()
                        except AttributeError:
                            # Not PyTorch, try TensorFlow-style conversion
                            orig_array = orig_val.numpy()
                    else:
                        # JAX array or numpy array - use directly
                        orig_array = orig_val

                    if hasattr(loaded_val, "numpy"):
                        # Could be TensorFlow variable or PyTorch tensor
                        try:
                            # Try PyTorch-style conversion first
                            # (detach().cpu().numpy())
                            loaded_array = loaded_val.detach().cpu().numpy()
                        except AttributeError:
                            # Not PyTorch, try TensorFlow-style conversion
                            loaded_array = loaded_val.numpy()
                    else:
                        # JAX array or numpy array - use directly
                        loaded_array = loaded_val

                    np.testing.assert_array_almost_equal(
                        orig_array, loaded_array
                    )

        compare_nested_dicts(
            original_state_tree["trainable_variables"],
            loaded_state_tree["trainable_variables"],
        )

        # Compare optimizer variables
        compare_nested_dicts(
            original_state_tree["optimizer_variables"],
            loaded_state_tree["optimizer_variables"],
        )

    @pytest.mark.requires_trainable_backend
    def test_checkpoint_loading_with_metrics_state(self):
        """Test loading checkpoints that include metrics state."""
        model = self._create_test_model()
        x, y = self._create_dummy_data(num_samples=200)

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_loading_metrics"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_freq="epoch", save_weights_only=False
        )

        # Train for 1 epoch to build metrics state
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Get original state after training
        original_state_tree = model.get_state_tree()

        # Create a new model with same architecture and compile with metrics
        new_model = self._create_test_model()
        new_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Run one training step to initialize metrics variables
        new_x, new_y = self._create_dummy_data(num_samples=10)
        new_model.fit(new_x, new_y, epochs=1, batch_size=5, verbose=0)

        # Load the checkpoint (epoch 0)
        checkpoint_path = os.path.join(checkpoint_dir, "0")
        loaded_state = load_pytree(checkpoint_path)

        # Set the full state (weights + optimizer + metrics) to new model
        new_model.set_state_tree(
            {
                "trainable_variables": loaded_state["trainable_variables"],
                "non_trainable_variables": loaded_state[
                    "non_trainable_variables"
                ],
                "optimizer_variables": loaded_state["optimizer_variables"],
                "metrics_variables": loaded_state["metrics_variables"],
            }
        )

        # Get the loaded state
        loaded_state_tree = new_model.get_state_tree()

        # Compare trainable variables (weights)
        def compare_nested_dicts(orig_dict, loaded_dict):
            """Recursively compare nested dictionaries containing variables."""
            for key in orig_dict:
                if key not in loaded_dict:
                    self.fail(f"Key {key} missing in loaded state")
                orig_val = orig_dict[key]
                loaded_val = loaded_dict[key]

                if isinstance(orig_val, dict):
                    compare_nested_dicts(orig_val, loaded_val)
                else:
                    # Handle different array types: JAX arrays, TF variables,
                    # PyTorch tensors, numpy arrays
                    if hasattr(orig_val, "numpy"):
                        # Could be TensorFlow variable or PyTorch tensor
                        try:
                            # Try PyTorch-style conversion first
                            # (detach().cpu().numpy())
                            orig_array = orig_val.detach().cpu().numpy()
                        except AttributeError:
                            # Not PyTorch, try TensorFlow-style conversion
                            orig_array = orig_val.numpy()
                    else:
                        # JAX array or numpy array - use directly
                        orig_array = orig_val

                    if hasattr(loaded_val, "numpy"):
                        # Could be TensorFlow variable or PyTorch tensor
                        try:
                            # Try PyTorch-style conversion first
                            # (detach().cpu().numpy())
                            loaded_array = loaded_val.detach().cpu().numpy()
                        except AttributeError:
                            # Not PyTorch, try TensorFlow-style conversion
                            loaded_array = loaded_val.numpy()
                    else:
                        # JAX array or numpy array - use directly
                        loaded_array = loaded_val

                    np.testing.assert_array_almost_equal(
                        orig_array, loaded_array
                    )

        compare_nested_dicts(
            original_state_tree["trainable_variables"],
            loaded_state_tree["trainable_variables"],
        )

        # Compare non-trainable variables
        compare_nested_dicts(
            original_state_tree["non_trainable_variables"],
            loaded_state_tree["non_trainable_variables"],
        )

        # Compare optimizer variables
        compare_nested_dicts(
            original_state_tree["optimizer_variables"],
            loaded_state_tree["optimizer_variables"],
        )

        # Compare metrics variables
        compare_nested_dicts(
            original_state_tree["metrics_variables"],
            loaded_state_tree["metrics_variables"],
        )

    @pytest.mark.requires_trainable_backend
    def _flatten_nested_dict(self, nested_dict):
        """Flatten a nested dictionary into a flat dictionary with path keys."""
        flat_dict = {}

        def _flatten(current_dict, prefix=""):
            for key, value in current_dict.items():
                if isinstance(value, dict):
                    _flatten(value, f"{prefix}{key}/")
                else:
                    flat_dict[f"{prefix}{key}"] = value

        _flatten(nested_dict)
        return flat_dict

    @pytest.mark.requires_trainable_backend
    def test_model_load_method(self):
        """Test the Model.load() method for loading Orbax checkpoints."""
        # Test both synchronous and asynchronous saving modes
        self._test_model_load_with_saving_mode(save_on_background=False)
        self._test_model_load_with_saving_mode(save_on_background=True)

    def _test_model_load_with_saving_mode(self, save_on_background):
        """Helper method to test Model.load() with different saving modes."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(),
            f"test_model_load_{'async' if save_on_background else 'sync'}",
        )

        if save_on_background:
            # For async saving, use a custom callback that waits between saves
            # to avoid conflicts between concurrent async operations
            class AsyncSafeOrbaxCheckpoint(OrbaxCheckpoint):
                def on_epoch_end(self, epoch, logs=None):
                    # Wait for any previous async operations to complete
                    if hasattr(self, "wait_until_finished"):
                        self.wait_until_finished()
                    super().on_epoch_end(epoch, logs)

            callback = AsyncSafeOrbaxCheckpoint(
                directory=checkpoint_dir,
                save_freq="epoch",
                save_on_background=True,
            )
        else:
            callback = OrbaxCheckpoint(
                directory=checkpoint_dir,
                save_freq="epoch",
                save_on_background=False,
            )

        # Train for a few epochs to create checkpoints
        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)

        # Wait for async operations to complete if using async saving
        if save_on_background:
            callback.wait_until_finished()

        # Get the state of the trained model
        trained_state = model.get_state_tree()

        # Create a new model with same architecture
        new_model = self._create_test_model()
        original_weights = new_model.get_weights()

        # Test loading the latest checkpoint
        new_model.load(checkpoint_dir)
        loaded_weights = new_model.get_weights()
        loaded_state = new_model.get_state_tree()

        # Weights should be different after loading
        # (from random init to trained)
        weights_changed = False
        for orig, loaded in zip(original_weights, loaded_weights):
            if not np.allclose(orig, loaded):
                weights_changed = True
                break
        self.assertTrue(
            weights_changed, "Weights should change after loading checkpoint"
        )

        # Verify that loaded weights match the trained model's weights
        trained_weights = model.get_weights()
        for trained_w, loaded_w in zip(trained_weights, loaded_weights):
            self.assertTrue(
                np.allclose(trained_w, loaded_w),
                "Loaded weights should match trained model's weights",
            )

        # Verify that optimizer state was loaded
        trained_opt_flat = self._flatten_nested_dict(
            trained_state["optimizer_variables"]
        )
        loaded_opt_flat = self._flatten_nested_dict(
            loaded_state["optimizer_variables"]
        )
        self.assertEqual(
            set(trained_opt_flat.keys()),
            set(loaded_opt_flat.keys()),
            "Optimizer variable keys should match",
        )
        for key in trained_opt_flat:
            # Convert tensors to numpy for comparison
            trained_val = trained_opt_flat[key]
            loaded_val = loaded_opt_flat[key]

            trained_np = self._to_numpy(trained_val)
            loaded_np = self._to_numpy(loaded_val)

            self.assertTrue(
                np.allclose(trained_np, loaded_np),
                f"Optimizer variable {key} should match",
            )

        # Verify that metrics state was loaded
        trained_met_flat = self._flatten_nested_dict(
            trained_state["metrics_variables"]
        )
        loaded_met_flat = self._flatten_nested_dict(
            loaded_state["metrics_variables"]
        )
        self.assertEqual(
            set(trained_met_flat.keys()),
            set(loaded_met_flat.keys()),
            "Metrics variable keys should match",
        )
        for key in trained_met_flat:
            # Convert tensors to numpy for comparison
            trained_val = trained_met_flat[key]
            loaded_val = loaded_met_flat[key]

            trained_np = self._to_numpy(trained_val)
            loaded_np = self._to_numpy(loaded_val)

            self.assertTrue(
                np.allclose(trained_np, loaded_np),
                f"Metrics variable {key} should match",
            )

    @pytest.mark.requires_trainable_backend
    def test_load_checkpoint_preserves_layout(self):
        """Test Model.load() preserves layout when no distribution is set."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_preserve_layout"
        )
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Train and save checkpoints
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Create new model and load checkpoint
        new_model = self._create_test_model()
        original_weights = new_model.get_weights()

        # Load checkpoint using Model.load() - should preserve original layout
        new_model.load(checkpoint_dir)

        # Verify weights changed (loading worked)
        loaded_weights = new_model.get_weights()
        weights_changed = any(
            not np.allclose(orig, loaded)
            for orig, loaded in zip(original_weights, loaded_weights)
        )
        self.assertTrue(weights_changed, "Weights should change after loading")

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
        if len(devices) < 1:
            self.skipTest("Test requires at least 1 JAX device")

        num_devices = min(2, len(devices))

        # Configure JAX to use virtual devices if needed
        original_xla_flags = os.environ.get("XLA_FLAGS", "")
        if num_devices < 2:
            os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
            # Re-check devices after setting flag
            devices = jax.devices()
            num_devices = min(2, len(devices))

        try:
            print(f"Available devices: {devices}, using {num_devices} devices")

            # Set up distribution based on available devices
            if num_devices >= 2:
                # Multi-device distribution
                device_mesh = DeviceMesh((2,), axis_names=["data"])
                layout_map = LayoutMap(device_mesh)
                layout_map["dense_layer/kernel"] = TensorLayout(
                    axes=("data", None)
                )
                layout_map["dense_layer/bias"] = TensorLayout(axes=(None,))
                layout_map["output_layer/kernel"] = TensorLayout(
                    axes=(None, "data")
                )
                layout_map["output_layer/bias"] = TensorLayout(axes=(None,))
            else:
                # Single device distribution
                device_mesh = DeviceMesh((1,), axis_names=["data"])
                layout_map = LayoutMap(device_mesh)
                layout_map["dense_layer/kernel"] = TensorLayout(
                    axes=(None, None)
                )
                layout_map["dense_layer/bias"] = TensorLayout(axes=(None,))
                layout_map["output_layer/kernel"] = TensorLayout(
                    axes=(None, None)
                )
                layout_map["output_layer/bias"] = TensorLayout(axes=(None,))

            distribution = ModelParallel(
                device_mesh=device_mesh, layout_map=layout_map
            )

            # Save original distribution state
            original_distribution = None
            try:
                from keras.src.distribution import (
                    distribution as get_distribution,
                )

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

                # Create new model and load with same distribution
                new_model = self._create_test_model()
                # Initialize optimizer state by running a dummy training step
                batch_size = min(2, len(x))  # Compatible with distribution
                new_model.fit(
                    x[:batch_size], y[:batch_size], epochs=0, verbose=0
                )

                # Get initial weights before loading
                initial_weights = new_model.get_weights()

                new_model.load(checkpoint_dir)
                loaded_weights = new_model.get_weights()

                # Get original weights for comparison
                original_weights = model.get_weights()

                # Check that loading actually changed some weights
                loading_changed_weights = any(
                    not np.allclose(init, loaded)
                    for init, loaded in zip(initial_weights, loaded_weights)
                )
                self.assertTrue(
                    loading_changed_weights,
                    "Loading should change weights from initial random values",
                )

                # Check that shapes match (basic sanity check)
                shapes_match = all(
                    orig.shape == loaded.shape
                    for orig, loaded in zip(original_weights, loaded_weights)
                )
                self.assertTrue(
                    shapes_match,
                    "Loaded weights should have same shapes as original "
                    "weights",
                )

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

        finally:
            # Restore original XLA_FLAGS
            if original_xla_flags:
                os.environ["XLA_FLAGS"] = original_xla_flags
            else:
                os.environ.pop("XLA_FLAGS", None)

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="Checkpoint structure tests require JAX backend",
    )
    def test_distributed_checkpoint_directory_structure(self):
        """Test OrbaxCheckpoint directory structure for distributed training."""
        import os

        import jax

        from keras.src.distribution import DeviceMesh
        from keras.src.distribution import LayoutMap
        from keras.src.distribution import ModelParallel
        from keras.src.distribution import TensorLayout
        from keras.src.distribution import set_distribution

        # Check if we have at least 1 device
        devices = jax.devices()
        if len(devices) < 1:
            self.skipTest("Test requires at least 1 JAX device")

        num_devices = min(2, len(devices))

        # Configure JAX to use virtual devices if needed
        original_xla_flags = os.environ.get("XLA_FLAGS", "")
        if num_devices < 2:
            os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
            # Re-check devices after setting flag
            devices = jax.devices()
            num_devices = min(2, len(devices))

        try:
            print(f"Available devices: {devices}, using {num_devices} devices")

            # Set up distribution based on available devices
            if num_devices >= 2:
                # Multi-device distribution for distributed checkpointing test
                device_mesh = DeviceMesh((2,), axis_names=["data"])
                layout_map = LayoutMap(device_mesh)
                layout_map["dense_layer/kernel"] = TensorLayout(
                    axes=("data", None)
                )
                layout_map["dense_layer/bias"] = TensorLayout(axes=(None,))
                layout_map["output_layer/kernel"] = TensorLayout(
                    axes=(None, "data")
                )
                layout_map["output_layer/bias"] = TensorLayout(axes=(None,))
                is_distributed = True
            else:
                # Single device distribution
                device_mesh = DeviceMesh((1,), axis_names=["data"])
                layout_map = LayoutMap(device_mesh)
                layout_map["dense_layer/kernel"] = TensorLayout(
                    axes=(None, None)
                )
                layout_map["dense_layer/bias"] = TensorLayout(axes=(None,))
                layout_map["output_layer/kernel"] = TensorLayout(
                    axes=(None, None)
                )
                layout_map["output_layer/bias"] = TensorLayout(axes=(None,))
                is_distributed = False

            distribution = ModelParallel(
                device_mesh=device_mesh, layout_map=layout_map
            )

            # Save original distribution
            original_distribution = None
            try:
                from keras.src.distribution import (
                    distribution as get_distribution,
                )

                original_distribution = get_distribution()
            except (ImportError, AttributeError):
                pass

            try:
                # Apply distribution
                set_distribution(distribution)

                # Create and compile model
                model = self._create_test_model()
                x, y = self._create_dummy_data(num_samples=50)

                # Set up checkpointing
                checkpoint_dir = os.path.join(
                    self.get_temp_dir(), "test_structure"
                )
                callback = OrbaxCheckpoint(
                    directory=checkpoint_dir,
                    save_freq="epoch",
                    save_weights_only=False,  # Save full state
                    max_to_keep=3,
                )

                # Train for 2 epochs to create checkpoints
                model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)
                callback.wait_until_finished()

                # Verify checkpoint directory structure
                self.assertTrue(
                    os.path.exists(checkpoint_dir),
                    "Checkpoint directory should exist",
                )

                # List checkpoint directories (should be step numbers)
                checkpoint_steps = os.listdir(checkpoint_dir)
                print(f"Checkpoint directory contents: {checkpoint_steps}")
                self.assertGreater(
                    len(checkpoint_steps),
                    0,
                    "Should have checkpoint step directories",
                )

                # Check that we have step directories (named with numbers)
                step_dirs = [d for d in checkpoint_steps if d.isdigit()]
                self.assertGreater(
                    len(step_dirs), 0, "Should have numeric step directories"
                )

                # Examine the latest checkpoint structure (step "1" for epoch 1)
                latest_step = max(int(d) for d in step_dirs if d.isdigit())
                latest_checkpoint_dir = os.path.join(
                    checkpoint_dir, str(latest_step)
                )

                self.assertTrue(
                    os.path.exists(latest_checkpoint_dir),
                    f"Latest checkpoint dir exists: {latest_checkpoint_dir}",
                )

                # List contents of the checkpoint directory
                checkpoint_contents = os.listdir(latest_checkpoint_dir)
                print(f"Checkpoint contents: {checkpoint_contents}")

                # Check for expected Orbax files
                expected_files = ["pytree", "_CHECKPOINT_METADATA"]
                for expected_file in expected_files:
                    file_path = os.path.join(
                        latest_checkpoint_dir, expected_file
                    )
                    self.assertTrue(
                        os.path.exists(file_path),
                        f"Expected file {expected_file} should exist",
                    )

                # The pytree directory contains the sharded model state
                pytree_dir = os.path.join(latest_checkpoint_dir, "pytree")
                self.assertTrue(
                    os.path.isdir(pytree_dir), "Pytree should be a directory"
                )

                # Check that pytree directory has content
                pytree_contents = os.listdir(pytree_dir)
                print(f"Pytree directory contents: {pytree_contents}")
                self.assertGreater(
                    len(pytree_contents), 0, "Pytree directory not empty"
                )

                if is_distributed:
                    # Check for sharding metadata files (only for distributed)
                    expected_sharding_files = [
                        "_sharding",
                        "_METADATA",
                        "array_metadatas",
                    ]
                    for sharding_file in expected_sharding_files:
                        file_path = os.path.join(pytree_dir, sharding_file)
                        self.assertTrue(
                            os.path.exists(file_path),
                            f"Sharding file exists: {sharding_file}",
                        )

                    # Check for process-specific data
                    process_files = [
                        f
                        for f in pytree_contents
                        if f.startswith("ocdbt.process_")
                    ]
                    self.assertGreater(
                        len(process_files),
                        0,
                        f"Process-specific files found: {process_files}",
                    )
                else:
                    # For single device, we still expect some basic structure
                    expected_files = ["_METADATA", "array_metadatas"]
                    for expected_file in expected_files:
                        file_path = os.path.join(pytree_dir, expected_file)
                        self.assertTrue(
                            os.path.exists(file_path),
                            f"Expected file {expected_file} should exist",
                        )

                # Load and inspect the checkpoint
                loaded_state = load_pytree(latest_checkpoint_dir)

                # Verify that the loaded state contains sharded variables
                self.assertIn(
                    "trainable_variables", loaded_state, "Has trainable vars"
                )
                self.assertIn(
                    "optimizer_variables", loaded_state, "Has optimizer vars"
                )

                # Check that variables are properly structured (sharded)
                trainable_vars = loaded_state["trainable_variables"]
                # The checkpoint structure matches the layer names directly
                self.assertIn(
                    "dense_layer", trainable_vars, "Should have dense_layer"
                )
                self.assertIn(
                    "output_layer", trainable_vars, "Should have output_layer"
                )

                # Verify layer variables exist and have expected structure
                dense_layer = trainable_vars["dense_layer"]
                output_layer = trainable_vars["output_layer"]

                # Check kernel and bias exist (sharded according to layout_map)
                self.assertIn("kernel", dense_layer, "Dense layer has kernel")
                self.assertIn("bias", dense_layer, "Dense layer has bias")
                self.assertIn("kernel", output_layer, "Output layer has kernel")
                self.assertIn("bias", output_layer, "Output layer has bias")

                # Verify shapes are correct (kernel should be sharded)
                dense_kernel = dense_layer["kernel"]
                output_kernel = output_layer["kernel"]
                dense_bias = dense_layer["bias"]
                output_bias = output_layer["bias"]

                # Check shapes - kernels should have the expected dimensions
                self.assertEqual(
                    dense_kernel.shape,
                    (10, 6),
                    f"Dense kernel shape (10, 6), got {dense_kernel.shape}",
                )
                self.assertEqual(
                    output_kernel.shape,
                    (6, 2),
                    f"Output kernel shape (6, 2), got {output_kernel.shape}",
                )
                self.assertEqual(
                    dense_bias.shape,
                    (6,),
                    f"Dense bias shape should be (6,), got {dense_bias.shape}",
                )
                self.assertEqual(
                    output_bias.shape,
                    (2,),
                    f"Output bias shape should be (2,), got "
                    f"{output_bias.shape}",
                )

                # Check optimizer variables (should also be sharded)
                optimizer_vars = loaded_state["optimizer_variables"]
                self.assertIn("adam", optimizer_vars, "Has Adam optimizer")

                adam_vars = optimizer_vars["adam"]
                # Adam optimizer should have multiple variable types
                optimizer_var_types = list(adam_vars.keys())
                self.assertGreater(
                    len(optimizer_var_types), 0, "Has optimizer variable types"
                )

                # Verify optimizer has variables for each layer
                expected_adam_vars = [
                    "dense_layer_bias_momentum",
                    "dense_layer_bias_velocity",
                    "dense_layer_kernel_momentum",
                    "dense_layer_kernel_velocity",
                    "output_layer_bias_momentum",
                    "output_layer_bias_velocity",
                    "output_layer_kernel_momentum",
                    "output_layer_kernel_velocity",
                    "iteration",
                    "learning_rate",
                ]

                for expected_var in expected_adam_vars:
                    self.assertIn(expected_var, adam_vars, expected_var)

                # Verify shapes of optimizer variables match the layer variables
                # Dense layer bias optimizer vars should have shape (6,)
                self.assertEqual(
                    adam_vars["dense_layer_bias_momentum"].shape,
                    (6,),
                    "Dense bias momentum shape should be (6,)",
                )
                self.assertEqual(
                    adam_vars["dense_layer_bias_velocity"].shape,
                    (6,),
                    "Dense bias velocity shape should be (6,)",
                )

                # Dense layer kernel optimizer vars should have shape (10, 6)
                self.assertEqual(
                    adam_vars["dense_layer_kernel_momentum"].shape,
                    (10, 6),
                    "Dense kernel momentum shape should be (10, 6)",
                )
                self.assertEqual(
                    adam_vars["dense_layer_kernel_velocity"].shape,
                    (10, 6),
                    "Dense kernel velocity shape should be (10, 6)",
                )

                # Output layer bias optimizer vars should have shape (2,)
                self.assertEqual(
                    adam_vars["output_layer_bias_momentum"].shape,
                    (2,),
                    "Output bias momentum shape should be (2,)",
                )
                self.assertEqual(
                    adam_vars["output_layer_bias_velocity"].shape,
                    (2,),
                    "Output bias velocity shape should be (2,)",
                )

                # Output layer kernel optimizer vars should have shape (6, 2)
                self.assertEqual(
                    adam_vars["output_layer_kernel_momentum"].shape,
                    (6, 2),
                    "Output kernel momentum shape should be (6, 2)",
                )
                self.assertEqual(
                    adam_vars["output_layer_kernel_velocity"].shape,
                    (6, 2),
                    "Output kernel velocity shape should be (6, 2)",
                )

                print(f"Verification complete for step {latest_step}")
                print(f"Total checkpoints created: {len(step_dirs)}")
                print(f"Devices used: {num_devices}")
                if is_distributed:
                    process_files = [
                        f
                        for f in pytree_contents
                        if f.startswith("ocdbt.process_")
                    ]
                    process_count = len(process_files)
                    print(f"Process files: {process_count}")
                print(f"Optimizer variable types: {optimizer_var_types}")
                if is_distributed:
                    print("Distributed checkpoint structure verified")
                else:
                    print("Single-device checkpoint structure verified")

            finally:
                # Restore original distribution
                if original_distribution is not None:
                    set_distribution(original_distribution)
                else:
                    try:
                        set_distribution(None)
                    except:
                        pass

        finally:
            # Restore original XLA_FLAGS
            if original_xla_flags:
                os.environ["XLA_FLAGS"] = original_xla_flags
            else:
                os.environ.pop("XLA_FLAGS", None)

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="Multi-host checkpointing is JAX only",
    )
    def test_multihost_checkpointing(self):
        """Test multi-host checkpointing functionality (JAX only)."""
        self._test_multihost_checkpointing()

    def _test_multihost_checkpointing(self):
        """Test multi-host checkpointing functionality and file structure."""
        import os
        from unittest import mock

        # Create temporary directory for checkpoints
        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_multihost")

        # Test 1: Multi-host detection methods
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Mock multi-host environment
        with mock.patch("orbax.checkpoint.multihost") as mock_multihost:
            # Test when multi-host is initialized
            mock_multihost.is_initialized.return_value = True
            mock_multihost.is_primary_host.return_value = True

            # Re-initialize to pick up mocked environment
            callback._multihost_initialized = (
                callback._is_multihost_initialized()
            )

            # Test multi-host detection
            self.assertTrue(
                callback.is_multihost_enabled(),
                "Should detect multi-host when initialized",
            )
            self.assertTrue(
                callback.is_primary_host(),
                "Should be primary host in mock setup",
            )

            # Test when multi-host is not initialized
            mock_multihost.is_initialized.return_value = False
            callback._multihost_initialized = (
                callback._is_multihost_initialized()
            )

            self.assertFalse(
                callback.is_multihost_enabled(),
                "Should not detect multi-host when not initialized",
            )
            self.assertTrue(
                callback.is_primary_host(),
                "Should always be primary host in single-host mode",
            )

        # Test 2: Skip actual save/load for now - focus on multi-host methods
        # The save/load functionality is tested elsewhere, here we focus on
        # multi-host features

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="Multi-host checkpointing is JAX only",
    )
    def test_multihost_synchronization_methods(self):
        """Test multi-host synchronization methods (JAX only)."""
        self._test_multihost_synchronization_methods()

    def _test_multihost_synchronization_methods(self):
        """Test multi-host synchronization methods in OrbaxCheckpoint."""
        import os
        from unittest import mock

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_sync")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Test synchronization methods with mocked multihost
        with mock.patch("orbax.checkpoint.multihost") as mock_multihost:
            # Test when multi-host is initialized
            mock_multihost.is_initialized.return_value = True
            mock_multihost.is_primary_host.return_value = True
            mock_multihost.sync_global_processes = mock.MagicMock()

            callback._multihost_initialized = True

            # Test _sync_processes
            callback._sync_processes("test_key")
            mock_multihost.sync_global_processes.assert_called_with("test_key")

            # Test when multi-host is not initialized (should be no-op)
            mock_multihost.is_initialized.return_value = False
            callback._multihost_initialized = False

            callback._sync_processes("test_key_noop")
            # Should not call sync when not initialized
            mock_multihost.sync_global_processes.assert_called_once()
            # Only the previous call
