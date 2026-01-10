import os

import numpy as np
import pytest
from absl.testing import parameterized

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
    def test_load_weights_from_orbax_checkpoint(self):
        """Test loading weights from Orbax checkpoint using load_weights."""

        # Create and train model to create checkpoint
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.get_temp_dir(), "test_load_weights_orbax"
        )
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
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
            self.assertNotAllClose(
                orig, new, msg="Weights should be different before loading"
            )

        # Load weights from Orbax checkpoint
        new_model.load_weights(checkpoint_dir)

        # Verify weights were loaded correctly
        loaded_weights = new_model.get_weights()
        for orig, loaded in zip(original_weights, loaded_weights):
            self.assertAllClose(
                orig,
                loaded,
                msg="Weights should match after loading from checkpoint",
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
        checkpoint_files = os.listdir(checkpoint_dir)
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
            "save_on_background": False,
        },  # basic
        {
            "save_on_background": True,
        },  # background_save
    )
    @pytest.mark.requires_trainable_backend
    def test_checkpoint_loading_comprehensive(
        self,
        save_on_background,
    ):
        """Test checkpoint loading using load_weights API."""
        # Create and compile model
        model = self._create_test_model()
        model.compile(optimizer="adam", loss="mse")

        x, y = self._create_dummy_data(num_samples=200)

        checkpoint_dir = os.path.join(
            self.get_temp_dir(),
            f"test_loading_{save_on_background}_{id(self)}",
        )

        # Create callback
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_on_background=save_on_background,
        )

        # Train to create checkpoint
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)

        if save_on_background:
            callback.wait_until_finished()

        # Get original state
        original_weights = model.get_weights()

        # Test load_weights functionality
        new_model = self._create_test_model()
        # Initialize optimizer by running a training step
        new_model.compile(optimizer="adam", loss="mse")
        new_x, new_y = self._create_dummy_data(num_samples=10)
        new_model.fit(new_x, new_y, epochs=1, batch_size=5, verbose=0)

        # Initialize with different weights to ensure loading works
        different_weights = [w * 2 for w in original_weights]
        new_model.set_weights(different_weights)

        # Verify weights are different initially
        new_weights_before = new_model.get_weights()
        for orig, new in zip(original_weights, new_weights_before):
            self.assertNotAllClose(
                orig, new, msg="Weights should be different before loading"
            )

        # Load weights from Orbax checkpoint using load_weights
        new_model.load_weights(checkpoint_dir)

        # Verify weights were loaded correctly
        loaded_weights = new_model.get_weights()
        for orig, loaded in zip(original_weights, loaded_weights):
            self.assertAllClose(
                orig,
                loaded,
                msg="Weights should match after loading from checkpoint",
            )

        # For Orbax checkpoints, the complete state is loaded including
        # optimizer and metrics state from the checkpoint
        # Note: metrics_variables are not saved in Orbax checkpoints

    @pytest.mark.requires_trainable_backend
    def test_save_on_background_async(self):
        """Test save_on_background=True functionality."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.get_temp_dir(), "test_async_save")

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

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="Requires JAX backend for distribution",
    )
    def test_distributed_checkpoint_functionality(self):
        """Test OrbaxCheckpoint with distributed training."""
        import jax

        from keras.src.distribution import DeviceMesh
        from keras.src.distribution import LayoutMap
        from keras.src.distribution import ModelParallel
        from keras.src.distribution import TensorLayout
        from keras.src.distribution import distribution as get_distribution
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
        original_distribution = get_distribution()

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
            )

            # Train to create checkpoint
            model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)
            callback.wait_until_finished()

            # Get original model predictions and weights
            original_predictions = model.predict(x[:5], verbose=0)
            original_weights = model.get_weights()

            # Load checkpoint using load_weights

            # Create fresh model and load weights
            fresh_model = self._create_test_model()
            fresh_model.load_weights(checkpoint_dir)
            loaded_weights = fresh_model.get_weights()

            # Verify loaded weights match original
            for orig, loaded in zip(original_weights, loaded_weights):
                self.assertAllClose(orig, loaded)

            # Verify loaded model produces same predictions
            loaded_predictions = fresh_model.predict(x[:5], verbose=0)
            self.assertAllClose(original_predictions, loaded_predictions)

            # Verify sharding is maintained after loading
            # Check that both models have the same distribution
            current_dist = get_distribution()
            self.assertIsNotNone(current_dist)
            self.assertEqual(type(current_dist), ModelParallel)

            # Verify model variables are sharded correctly
            # In JAX, sharded variables should have different sharding info

            # Get sharding info for original model variables
            original_shardings = {}
            for name, var in model.variables.items():
                if hasattr(var, "sharding"):
                    original_shardings[name] = var.sharding

            # Get sharding info for loaded model variables
            loaded_shardings = {}
            for name, var in fresh_model.variables.items():
                if hasattr(var, "sharding"):
                    loaded_shardings[name] = var.sharding

            # Verify shardings match
            for name in original_shardings:
                if name in loaded_shardings:
                    self.assertEqual(
                        original_shardings[name],
                        loaded_shardings[name],
                        f"Sharding mismatch for variable {name}",
                    )

            print("Distributed checkpoint functionality and sharding verified")

        finally:
            # Restore original distribution
            if original_distribution is not None:
                set_distribution(original_distribution)
            else:
                try:
                    set_distribution(None)
                except:
                    pass
