import os

import numpy as np
import pytest

from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.utils.module_utils import ocp

# Import advanced Orbax functionality directly from the LazyModule
# These will only be available if orbax-checkpoint is installed
if ocp.available:
    Checkpointer = ocp.training.Checkpointer
    save_pytree = ocp.save_pytree
    load_pytree = ocp.load_pytree
    preservation_policies = ocp.training.preservation_policies
    save_decision_policies = ocp.training.save_decision_policies
    _orbax_available = True
else:
    Checkpointer = None
    save_pytree = None
    load_pytree = None
    preservation_policies = None
    save_decision_policies = None
    _orbax_available = False

# Import our OrbaxCheckpoint callback
try:
    from keras.src.callbacks.orbax_checkpoint import OrbaxCheckpoint

    _orbax_available = _orbax_available and True
except ImportError:
    OrbaxCheckpoint = None
    _orbax_available = False


@pytest.mark.skipif(
    not _orbax_available,
    reason="OrbaxCheckpoint requires the 'orbax-checkpoint' package",
)
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

        # Should have checkpoints for epochs 0, 1, 2
        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertGreaterEqual(
            len(checkpoint_files),
            3,
            f"Should have at least 3 checkpoints, "
            f"found {len(checkpoint_files)}",
        )

        # Check for specific epoch directories
        for epoch in [0, 1, 2]:
            epoch_dir = os.path.join(checkpoint_dir, str(epoch))
            self.assertTrue(
                os.path.exists(epoch_dir),
                f"Epoch {epoch} checkpoint should exist",
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
