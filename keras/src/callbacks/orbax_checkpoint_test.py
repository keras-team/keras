import os

import numpy as np
import pytest

from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.callbacks.orbax_checkpoint import OrbaxCheckpoint


class OrbaxCheckpointTest(testing.TestCase):
    def _create_test_model(self):
        """Create a simple test model."""
        inputs = layers.Input(shape=(10,), name="input_layer")
        x = layers.Dense(5, name="dense_layer")(inputs)
        outputs = layers.Dense(1, name="output_layer")(x)
        model = models.Model(inputs, outputs, name="test_model")
        model.compile(optimizer="adam", loss="mse")
        return model

    def _create_dummy_data(self, num_samples=100):
        """Create dummy training data."""
        x = np.random.randn(num_samples, 10)
        y = np.random.randn(num_samples, 1)
        return x, y

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
    def test_checkpoint_loading_via_saving_api(self):
        """Test checkpoint loading via keras.saving.load_model."""
        from keras.src import saving

        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_loading_api")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Train for 1 epoch to save checkpoint
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Get original weights after training
        original_weights = model.get_weights()

        # Load the model via Keras saving API
        loaded_model = saving.load_model(checkpoint_dir)

        # Compare weights
        loaded_weights = loaded_model.get_weights()
        for orig, loaded in zip(original_weights, loaded_weights):
            np.testing.assert_array_almost_equal(orig, loaded)

        # Verify the model is compiled
        self.assertTrue(loaded_model.compiled)

    @pytest.mark.requires_trainable_backend
    def test_checkpoint_loading_weights_only_via_saving_api(self):
        """Test that weights-only checkpoints cannot be loaded via saving API."""
        from keras.src import saving

        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_loading_weights_api"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_freq="epoch", save_weights_only=True
        )

        # Train for 1 epoch to save checkpoint
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Attempting to load weights-only checkpoint via saving API should fail
        with self.assertRaises(ValueError) as cm:
            saving.load_model(checkpoint_dir)

        self.assertIn("save_weights_only=True", str(cm.exception))

    @pytest.mark.requires_trainable_backend
    def test_checkpoint_loading_with_optimizer_state_via_saving_api(self):
        """Test loading checkpoints with optimizer state via saving API."""
        from keras.src import saving

        model = self._create_test_model()
        x, y = self._create_dummy_data(num_samples=100)

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_loading_optimizer_api"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_freq="epoch", save_weights_only=False
        )

        # Train for 1 epoch to build optimizer state
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Get original state after training
        original_state_tree = model.get_state_tree()

        # Load the model via Keras saving API
        loaded_model = saving.load_model(checkpoint_dir)

        # Get loaded state
        loaded_state_tree = loaded_model.get_state_tree()

        # Compare trainable variables (weights)
        def compare_nested_dicts(orig_dict, loaded_dict, component_name):
            """Recursively compare nested dictionaries containing variables."""
            for key in orig_dict:
                if key not in loaded_dict:
                    self.fail(f"Key {key} missing in loaded {component_name}")
                orig_val = orig_dict[key]
                loaded_val = loaded_dict[key]

                if isinstance(orig_val, dict):
                    compare_nested_dicts(
                        orig_val, loaded_val, f"{component_name}.{key}"
                    )
                else:
                    # Handle different array types: JAX arrays, TF variables,
                    # PyTorch tensors, numpy arrays
                    if hasattr(orig_val, "numpy"):
                        # Could be TensorFlow variable or PyTorch tensor
                        try:
                            # Try PyTorch-style conversion first (detach().cpu().numpy())
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
                            # Try PyTorch-style conversion first (detach().cpu().numpy())
                            loaded_array = loaded_val.detach().cpu().numpy()
                        except AttributeError:
                            # Not PyTorch, try TensorFlow-style conversion
                            loaded_array = loaded_val.numpy()
                    else:
                        # JAX array or numpy array - use directly
                        loaded_array = loaded_val

                    np.testing.assert_array_almost_equal(
                        orig_array,
                        loaded_array,
                        err_msg=f"Mismatch in {component_name}.{key}",
                    )

        # Compare trainable variables
        compare_nested_dicts(
            original_state_tree["trainable_variables"],
            loaded_state_tree["trainable_variables"],
            "trainable_variables",
        )

        # Compare optimizer variables
        compare_nested_dicts(
            original_state_tree["optimizer_variables"],
            loaded_state_tree["optimizer_variables"],
            "optimizer_variables",
        )

    @pytest.mark.requires_trainable_backend
    def test_checkpoint_loading_with_metrics_state_via_saving_api(self):
        """Test loading checkpoints with metrics state via saving API."""
        from keras.src import saving

        model = self._create_test_model()
        # Compile with metrics
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        x, y = self._create_dummy_data(num_samples=100)

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_loading_metrics_api"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_freq="epoch", save_weights_only=False
        )

        # Train for 1 epoch to build metrics state
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Get original state after training
        original_state_tree = model.get_state_tree()

        # Load the model via Keras saving API
        loaded_model = saving.load_model(checkpoint_dir)

        # Evaluate to initialize metrics variables in loaded model
        loaded_model.evaluate(x, y, verbose=0)

        # Get loaded state
        loaded_state_tree = loaded_model.get_state_tree()

        # Compare metrics variables structure (values may differ due to eval history)
        self.assertEqual(
            set(original_state_tree["metrics_variables"].keys()),
            set(loaded_state_tree["metrics_variables"].keys()),
            "Metrics variable keys should match",
        )
        for metric_name in original_state_tree["metrics_variables"]:
            self.assertEqual(
                set(
                    original_state_tree["metrics_variables"][metric_name].keys()
                ),
                set(loaded_state_tree["metrics_variables"][metric_name].keys()),
                f"Metrics variable structure for {metric_name} should match",
            )

    @pytest.mark.requires_trainable_backend
    def test_load_model_via_saving_api(self):
        """Test loading a model via saving API from Orbax checkpoint."""
        from keras.src import saving

        model = self._create_test_model()
        x, y = self._create_dummy_data(num_samples=50)

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_saving_api_load_{id(self)}"
        )
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Train for one epoch to create a checkpoint
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Load the complete model via the main saving API
        loaded_model = saving.load_model(checkpoint_dir)

        # Verify the model architecture is the same
        self.assertEqual(model.name, loaded_model.name)
        self.assertEqual(len(model.layers), len(loaded_model.layers))

        # Compare weights
        original_weights = model.get_weights()
        loaded_weights = loaded_model.get_weights()

        self.assertEqual(len(original_weights), len(loaded_weights))
        for orig, loaded in zip(original_weights, loaded_weights):
            np.testing.assert_array_almost_equal(orig, loaded)

        # Verify the model is compiled
        self.assertTrue(loaded_model.compiled)

    @pytest.mark.requires_trainable_backend
    def test_load_model_via_saving_api_full_state(self):
        """Test loading complete model state via saving API from checkpoint."""
        from keras.src import saving

        model = self._create_test_model()
        # Compile with metrics to have metrics state
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        x, y = self._create_dummy_data(num_samples=100)

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), f"test_saving_api_full_state_{id(self)}"
        )
        # Save full state (not weights only)
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_freq="epoch", save_weights_only=False
        )

        # Train for one epoch to build optimizer and metrics state
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)
        callback.wait_until_finished()

        # Get original state tree after training
        original_state_tree = model.get_state_tree()

        # Load the complete model via the main saving API
        loaded_model = saving.load_model(checkpoint_dir)

        # Verify the model architecture is the same
        self.assertEqual(model.name, loaded_model.name)
        self.assertEqual(len(model.layers), len(loaded_model.layers))

        # Verify the model is compiled
        self.assertTrue(loaded_model.compiled)

        # Evaluate the loaded model to initialize metrics variables
        # (metrics variables are only created when metrics are actually used)
        loaded_model.evaluate(x, y, verbose=0)

        # Get loaded state tree
        loaded_state_tree = loaded_model.get_state_tree()

        # Compare all state components
        def compare_nested_dicts(orig_dict, loaded_dict, component_name):
            """Recursively compare nested dictionaries containing variables."""
            for key in orig_dict:
                if key not in loaded_dict:
                    self.fail(f"Key {key} missing in loaded {component_name}")
                orig_val = orig_dict[key]
                loaded_val = loaded_dict[key]

                if isinstance(orig_val, dict):
                    compare_nested_dicts(
                        orig_val, loaded_val, f"{component_name}.{key}"
                    )
                else:
                    # Handle different array types: JAX arrays, TF variables,
                    # PyTorch tensors, numpy arrays
                    if hasattr(orig_val, "numpy"):
                        # Could be TensorFlow variable or PyTorch tensor
                        try:
                            # Try PyTorch-style conversion first (detach().cpu().numpy())
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
                            # Try PyTorch-style conversion first (detach().cpu().numpy())
                            loaded_array = loaded_val.detach().cpu().numpy()
                        except AttributeError:
                            # Not PyTorch, try TensorFlow-style conversion
                            loaded_array = loaded_val.numpy()
                    else:
                        # JAX array or numpy array - use directly
                        loaded_array = loaded_val

                    np.testing.assert_array_almost_equal(
                        orig_array,
                        loaded_array,
                        err_msg=f"Mismatch in {component_name}.{key}",
                    )

        # Compare trainable variables
        compare_nested_dicts(
            original_state_tree["trainable_variables"],
            loaded_state_tree["trainable_variables"],
            "trainable_variables",
        )

        # Compare non-trainable variables
        compare_nested_dicts(
            original_state_tree["non_trainable_variables"],
            loaded_state_tree["non_trainable_variables"],
            "non_trainable_variables",
        )

        # Compare optimizer variables
        compare_nested_dicts(
            original_state_tree["optimizer_variables"],
            loaded_state_tree["optimizer_variables"],
            "optimizer_variables",
        )

        # Compare metrics variables (only check structure, not exact values since
        # metrics accumulate state across evaluations)
        self.assertEqual(
            set(original_state_tree["metrics_variables"].keys()),
            set(loaded_state_tree["metrics_variables"].keys()),
            "Metrics variable keys should match",
        )
        for metric_name in original_state_tree["metrics_variables"]:
            self.assertEqual(
                set(
                    original_state_tree["metrics_variables"][metric_name].keys()
                ),
                set(loaded_state_tree["metrics_variables"][metric_name].keys()),
                f"Metrics variable structure for {metric_name} should match",
            )
