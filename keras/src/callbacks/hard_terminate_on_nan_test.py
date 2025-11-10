"""Tests for HardTerminateOnNaN callback."""

import os
import tempfile

import numpy as np
import pytest

import keras
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.callbacks import BackupAndRestore
from keras.src.callbacks import HardTerminateOnNaN


class HardTerminateOnNaNTest(testing.TestCase):
    """Test suite for HardTerminateOnNaN callback."""

    def test_hard_terminate_on_nan_raises_error(self):
        """Test that HardTerminateOnNaN raises RuntimeError on NaN loss."""
        # Create a simple model
        model = models.Sequential([layers.Dense(1, input_shape=(1,))])
        model.compile(optimizer="sgd", loss="mse")

        # Create data that will cause NaN (extreme values)
        x = np.array([[1.0], [2.0]])
        y = np.array([[np.inf], [np.inf]])  # This should cause NaN

        callback = HardTerminateOnNaN()

        # Training should raise RuntimeError
        with pytest.raises(RuntimeError, match="NaN or Inf loss encountered"):
            model.fit(
                x, y, epochs=1, batch_size=1, callbacks=[callback], verbose=0
            )

    def test_hard_terminate_does_not_trigger_on_train_end(self):
        """Test that on_train_end is NOT called when
        HardTerminateOnNaN raises.
        """

        # Create a custom callback to track if on_train_end was called
        class TrackingCallback(keras.src.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.train_end_called = False

            def on_train_end(self, logs=None):
                self.train_end_called = True

        model = models.Sequential([layers.Dense(1, input_shape=(1,))])
        model.compile(optimizer="sgd", loss="mse")

        x = np.array([[1.0]])
        y = np.array([[np.inf]])

        tracking_callback = TrackingCallback()
        hard_terminate_callback = HardTerminateOnNaN()

        # Should raise RuntimeError
        with pytest.raises(RuntimeError):
            model.fit(
                x,
                y,
                epochs=1,
                callbacks=[tracking_callback, hard_terminate_callback],
                verbose=0,
            )

        # on_train_end should NOT have been called
        self.assertFalse(tracking_callback.train_end_called)

    def test_hard_terminate_preserves_backup(self):
        """Ensure BackupAndRestore directory is preserved when
        HardTerminateOnNaN triggers.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = os.path.join(tmpdir, "backups")
            os.makedirs(backup_dir, exist_ok=True)

            # Create a fake file in the backup folder
            fake_file = os.path.join(backup_dir, "checkpoint.txt")
            open(fake_file, "w").write("dummy checkpoint")

            # Define a simple model
            model = models.Sequential([layers.Dense(1, input_shape=(1,))])
            model.compile(optimizer="sgd", loss="mse")

            # Data that causes NaN
            x_nan = np.array([[1.0]])
            y_nan = np.array([[np.inf]])

            hard_terminate_callback = HardTerminateOnNaN()
            backup_callback = BackupAndRestore(backup_dir=backup_dir)

            # Monkeypatch BackupAndRestore to prevent cleanup on train_end
            backup_callback.on_train_end = lambda logs=None: None

            # Training should raise RuntimeError
            with pytest.raises(RuntimeError):
                model.fit(
                    x_nan,
                    y_nan,
                    epochs=1,
                    callbacks=[backup_callback, hard_terminate_callback],
                    verbose=0,
                )

            # Verify backup directory still exists and file inside is untouched
            assert os.path.exists(backup_dir), (
                f"Backup dir deleted: {backup_dir}"
            )
            assert os.path.exists(fake_file), (
                "Backup file missing unexpectedly."
            )

    def test_normal_training_does_not_raise(self):
        """Test that HardTerminateOnNaN does not raise on normal training."""
        model = models.Sequential([layers.Dense(1, input_shape=(1,))])
        model.compile(optimizer="sgd", loss="mse")

        x = np.array([[1.0], [2.0]])
        y = np.array([[1.0], [2.0]])

        callback = HardTerminateOnNaN()

        # Should complete without raising
        history = model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Should have completed 2 epochs
        self.assertEqual(len(history.history["loss"]), 2)
