import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import callbacks
from keras.src import initializers
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.callbacks import BackupAndRestore
from keras.src.callbacks import TerminateOnNaN
from keras.src.models import Sequential
from keras.src.utils import numerical_utils


@pytest.mark.requires_trainable_backend
class TerminateOnNaNTest(testing.TestCase):
    """Test suite for TerminateOnNaN callback."""

    def test_TerminateOnNaN(self):
        TRAIN_SAMPLES = 10
        TEST_SAMPLES = 10
        INPUT_DIM = 3
        NUM_CLASSES = 2
        BATCH_SIZE = 4

        np.random.seed(1337)
        x_train = np.random.random((TRAIN_SAMPLES, INPUT_DIM))
        y_train = np.random.choice(np.arange(NUM_CLASSES), size=TRAIN_SAMPLES)
        x_test = np.random.random((TEST_SAMPLES, INPUT_DIM))
        y_test = np.random.choice(np.arange(NUM_CLASSES), size=TEST_SAMPLES)

        y_test = numerical_utils.to_categorical(y_test)
        y_train = numerical_utils.to_categorical(y_train)
        model = Sequential()
        initializer = initializers.Constant(value=1e5)
        for _ in range(5):
            model.add(
                layers.Dense(
                    2,
                    activation="relu",
                    kernel_initializer=initializer,
                )
            )
        model.add(layers.Dense(NUM_CLASSES))
        model.compile(loss="mean_squared_error", optimizer="sgd")

        history = model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=[callbacks.TerminateOnNaN()],
            epochs=20,
        )
        loss = history.history["loss"]
        self.assertEqual(len(loss), 1)
        self.assertTrue(np.isnan(loss[0]) or np.isinf(loss[0]))

    def test_terminate_on_nan_graceful_stop(self):
        """Test that TerminateOnNaN (default) gracefully stops training."""
        model = models.Sequential([layers.Dense(1, input_shape=(1,))])
        model.compile(optimizer="sgd", loss="mse")

        x = np.array([[1.0], [2.0]])
        y = np.array([[np.inf], [np.inf]])

        callback = TerminateOnNaN(raise_error=False)

        # Training should complete without raising RuntimeError
        history = model.fit(
            x, y, epochs=2, batch_size=1, callbacks=[callback], verbose=0
        )

        # Training should stop early
        self.assertLess(len(history.history["loss"]), 4)

    def test_terminate_on_nan_raise_error_raises_error(self):
        """Test that TerminateOnNaN(raise_error=True) raises
        RuntimeError on NaN loss.
        """
        model = models.Sequential([layers.Dense(1, input_shape=(1,))])
        model.compile(optimizer="sgd", loss="mse")

        x = np.array([[1.0], [2.0]])
        y = np.array([[np.inf], [np.inf]])

        callback = TerminateOnNaN(raise_error=True)

        # Training should raise RuntimeError
        with self.assertRaisesRegex(
            RuntimeError,
            "NaN or Inf loss encountered",
        ):
            model.fit(
                x, y, epochs=1, batch_size=1, callbacks=[callback], verbose=0
            )

    def test_raise_error_terminate_does_not_trigger_on_train_end(self):
        """Test that on_train_end is NOT called when
        TerminateOnNaN(raise_error=True) raises.
        """

        class TrackingCallback(callbacks.Callback):
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
        raise_error_terminate_callback = TerminateOnNaN(raise_error=True)

        # Should raise RuntimeError
        with self.assertRaises(RuntimeError):
            model.fit(
                x,
                y,
                epochs=1,
                callbacks=[tracking_callback, raise_error_terminate_callback],
                verbose=0,
            )

        # on_train_end should NOT have been called
        self.assertFalse(tracking_callback.train_end_called)

    def test_raise_error_terminate_preserves_backup(self):
        """Ensure BackupAndRestore directory is preserved when
        TerminateOnNaN(raise_error=True) triggers.
        """
        tmpdir = self.get_temp_dir()
        backup_dir = os.path.join(tmpdir, "backups")
        os.makedirs(backup_dir, exist_ok=True)

        fake_file = os.path.join(backup_dir, "checkpoint.txt")
        with open(fake_file, "w") as f:
            f.write("dummy checkpoint")

        model = models.Sequential([layers.Dense(1, input_shape=(1,))])
        model.compile(optimizer="sgd", loss="mse")

        x_nan = np.array([[1.0]])
        y_nan = np.array([[np.inf]])

        raise_error_terminate_callback = TerminateOnNaN(raise_error=True)
        backup_callback = BackupAndRestore(backup_dir=backup_dir)

        # Monkeypatch BackupAndRestore to prevent cleanup on train_end
        backup_callback.on_train_end = lambda logs=None: None

        # Training should raise RuntimeError
        with self.assertRaises(RuntimeError):
            model.fit(
                x_nan,
                y_nan,
                epochs=1,
                callbacks=[backup_callback, raise_error_terminate_callback],
                verbose=0,
            )

        # Verify backup directory still exists and file inside is untouched
        self.assertTrue(
            os.path.exists(backup_dir),
            f"Backup dir deleted: {backup_dir}",
        )
        self.assertTrue(
            os.path.exists(fake_file),
            "Backup file missing unexpectedly.",
        )

    @parameterized.named_parameters(
        ("raise_error_false", False),
        ("raise_error_true", True),
    )
    def test_normal_training_does_not_raise(self, raise_error):
        """Test that TerminateOnNaN does not raise on normal training."""
        model = models.Sequential([layers.Dense(1, input_shape=(1,))])
        model.compile(optimizer="sgd", loss="mse")

        x = np.array([[1.0], [2.0]])
        y = np.array([[1.0], [2.0]])

        callback = TerminateOnNaN(raise_error=raise_error)

        # Should complete without raising RuntimeError
        history = model.fit(x, y, epochs=2, callbacks=[callback], verbose=0)

        # Should have completed 2 epochs
        self.assertEqual(len(history.history["loss"]), 2)

    def test_raise_error_terminate_stops_on_later_batch(self):
        """Ensure TerminateOnNaN(raise_error=True) stops training
        if NaN appears in later batch.
        """
        model = models.Sequential([layers.Dense(1, input_shape=(1,))])
        model.compile(optimizer="sgd", loss="mse")

        # Batch 1: normal loss, Batch 2: NaN loss
        x = np.array([[1.0], [2.0]])
        y = np.array([[1.0], [np.inf]])  # NaN/Inf appears only in 2nd batch

        callback = TerminateOnNaN(raise_error=True)

        with self.assertRaises(RuntimeError) as exc:
            model.fit(
                x, y, epochs=1, batch_size=1, callbacks=[callback], verbose=0
            )

        self.assertTrue(any(f"batch {i}" in str(exc.exception) for i in [0, 1]))
