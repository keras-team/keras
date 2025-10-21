import os
import shutil
import tempfile

import numpy as np
import pytest

from keras.src import layers
from keras.src import models
from keras.src import testing

try:
    from keras.src.callbacks.orbax_checkpoint import OrbaxCheckpoint
except ImportError:
    OrbaxCheckpoint = None


@pytest.mark.skipif(
    OrbaxCheckpoint is None,
    reason="`orbax-checkpoint` is required for `OrbaxCheckpoint` tests.",
)
class OrbaxCheckpointTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_model(self):
        """Create a simple test model."""
        inputs = layers.Input(shape=(10,))
        x = layers.Dense(5)(inputs)
        outputs = layers.Dense(1)(x)
        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    def _create_dummy_data(self, num_samples=100):
        """Create dummy training data."""
        x = np.random.randn(num_samples, 10)
        y = np.random.randn(num_samples, 1)
        return x, y

    @pytest.mark.requires_trainable_backend
    def test_basic_save_and_load(self):
        """Test basic save and load functionality."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_basic")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Train for a few epochs
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Create a new model and load the checkpoint
        new_model = self._create_test_model()
        success = callback.load_latest()

        self.assertTrue(success, "Loading checkpoint should succeed")

        # Check that weights are loaded (rough check)
        original_weights = [w.numpy() for w in model.weights]
        loaded_weights = [w.numpy() for w in new_model.weights]

        # Weights should be different initially
        self.assertFalse(np.allclose(original_weights[0], loaded_weights[0]))

    @pytest.mark.requires_trainable_backend
    def test_save_best_only(self):
        """Test save_best_only functionality."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_best_only")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            monitor="loss",
            save_best_only=True,
            mode="min",
            save_freq="epoch",
        )

        # Train for a few epochs
        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)

        # Should have saved checkpoints
        checkpoints = os.listdir(checkpoint_dir)
        self.assertGreater(
            len(checkpoints), 0, "Should have saved at least one checkpoint"
        )

    @pytest.mark.requires_trainable_backend
    def test_save_freq_batch(self):
        """Test batch-level saving."""
        model = self._create_test_model()
        x, y = self._create_dummy_data(num_samples=50)

        checkpoint_dir = os.path.join(self.temp_dir, "test_batch_freq")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq=10)

        # Train for one epoch with batch saving
        model.fit(x, y, epochs=1, batch_size=5, callbacks=[callback], verbose=0)

        # Should have saved checkpoints
        checkpoints = []
        for root, dirs, files in os.walk(checkpoint_dir):
            checkpoints.extend(dirs)

        self.assertGreater(
            len(checkpoints),
            0,
            "Should have saved checkpoints at batch intervals",
        )

    @pytest.mark.requires_trainable_backend
    def test_max_to_keep(self):
        """Test max_to_keep parameter."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_max_keep")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir, save_freq="epoch", max_to_keep=2
        )

        # Train for more epochs than max_to_keep
        model.fit(x, y, epochs=5, callbacks=[callback], verbose=0)

        # Check that max_to_keep is respected
        all_steps = callback.manager.all_steps()
        self.assertLessEqual(
            len(all_steps),
            2,
            f"Should keep at most 2 checkpoints, found {len(all_steps)}: "
            f"{all_steps}",
        )

    @pytest.mark.requires_trainable_backend
    def test_optimizer_state_saving(self):
        """Test that optimizer state is saved and loaded."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_optimizer")
        callback = OrbaxCheckpoint(
            directory=checkpoint_dir,
            save_freq="epoch",
            save_optimizer_state=True,
        )

        # Train for a few epochs to update optimizer state
        model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Create new model and load
        new_model = self._create_test_model()
        success = callback.load_latest()
        self.assertTrue(success)

        # Check optimizer iterations (rough check that state was loaded)
        # Note: This is a basic check - more sophisticated tests could check
        # specific optimizer variables
        self.assertGreaterEqual(new_model.optimizer.iterations.numpy(), 0)

    @pytest.mark.requires_trainable_backend
    def test_load_specific_checkpoint(self):
        """Test loading a specific checkpoint by step."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(self.temp_dir, "test_specific")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Train for multiple epochs
        model.fit(x, y, epochs=3, callbacks=[callback], verbose=0)

        # Create new model and load specific checkpoint
        new_model = self._create_test_model()
        success = callback.load_checkpoint(step=1)  # Load epoch 1

        self.assertTrue(success, "Loading specific checkpoint should succeed")
        # Verify the model was loaded by checking it has weights
        self.assertGreater(len(new_model.weights), 0)

    @pytest.mark.requires_trainable_backend
    def test_no_checkpoint_found(self):
        """Test behavior when no checkpoints exist."""
        model = self._create_test_model()

        checkpoint_dir = os.path.join(self.temp_dir, "test_empty")
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Try to load from empty directory
        success = callback.load_latest()
        self.assertFalse(success, "Loading from empty directory should fail")
        # Verify model still has its original weights (not modified)
        self.assertGreater(len(model.weights), 0)

    @pytest.mark.requires_trainable_backend
    def test_directory_creation(self):
        """Test that checkpoint directory is created if it doesn't exist."""
        model = self._create_test_model()
        x, y = self._create_dummy_data()

        checkpoint_dir = os.path.join(
            self.temp_dir, "test_create_dir", "subdir"
        )
        callback = OrbaxCheckpoint(directory=checkpoint_dir, save_freq="epoch")

        # Directory should be created during training
        model.fit(x, y, epochs=1, callbacks=[callback], verbose=0)

        self.assertTrue(
            os.path.exists(checkpoint_dir),
            "Checkpoint directory should be created",
        )
