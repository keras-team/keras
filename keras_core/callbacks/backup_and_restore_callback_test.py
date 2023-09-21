import os

import numpy as np
import pytest

from keras_core import callbacks
from keras_core import layers
from keras_core import testing
from keras_core.models import Sequential
from keras_core.utils import file_utils


class InterruptingCallback(callbacks.Callback):
    """A callback to intentionally introduce interruption to
    training."""

    def __init__(self, steps_int, epoch_int):
        self.batch_count = 0
        self.epoch_count = 0
        self.steps_int = steps_int
        self.epoch_int = epoch_int

    def on_epoch_end(self, epoch, log=None):
        self.epoch_count += 1
        if self.epoch_int is not None and self.epoch_count == self.epoch_int:
            raise RuntimeError("EpochInterruption")

    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.steps_int is not None and self.batch_count == self.steps_int:
            raise RuntimeError("StepsInterruption")


class BackupAndRestoreCallbackTest(testing.TestCase):
    # Checking for invalid backup_dir
    def test_empty_backup_dir(self):
        with self.assertRaisesRegex(
            ValueError, expected_regex="Empty " "`backup_dir`"
        ):
            callbacks.BackupAndRestore(file_path=None)

    # Checking save_freq and save_before_preemption both unset
    def test_save_set_error(self):
        with self.assertRaisesRegex(
            ValueError,
            expected_regex="`save_freq` or "
            "`save_before_preemption` "
            ""
            "must be set",
        ):
            callbacks.BackupAndRestore(
                file_path="backup_dir",
                save_freq=None,
                save_before_preemption=False,
            )

    # Check invalid save_freq, both string and non integer
    def test_save_freq_unknown_error(self):
        with self.assertRaisesRegex(
            ValueError, expected_regex="Unrecognized save_freq"
        ):
            callbacks.BackupAndRestore(
                file_path="backup_dir", save_freq="batch"
            )

        with self.assertRaisesRegex(
            ValueError, expected_regex="Unrecognized save_freq"
        ):
            callbacks.BackupAndRestore(file_path="backup_dir", save_freq=0.15)

    # Checking if after interruption, correct model params and
    # weights are loaded in step-wise backup
    @pytest.mark.requires_trainable_backend
    def test_best_case_step(self):
        def make_model():
            np.random.seed(1337)
            model = Sequential(
                [
                    layers.Dense(2, activation="relu"),
                    layers.Dense(1),
                ]
            )
            model.compile(
                loss="mse",
                optimizer="sgd",
                metrics=["mse"],
            )
            return model

        temp_dir = self.get_temp_dir()
        filepath = os.path.join(temp_dir, "subdir", "checkpoint.weights.h5")
        file_utils.rmtree(filepath)
        self.assertFalse(os.path.exists(filepath))

        model = make_model()
        cbk = callbacks.BackupAndRestore(file_path=filepath, save_freq=1)

        x_train = np.random.random((10, 3))
        y_train = np.random.random((10, 1))

        try:
            model.fit(
                x_train,
                y_train,
                batch_size=4,
                callbacks=[
                    cbk,
                    InterruptingCallback(steps_int=2, epoch_int=None),
                ],
                epochs=2,
                verbose=0,
            )
        except RuntimeError:
            self.assertTrue(os.path.exists(filepath))
            self.assertEqual(cbk._current_epoch, 0)
            self.assertEqual(cbk._last_batch_seen, 1)

            hist = model.fit(
                x_train, y_train, batch_size=4, callbacks=[cbk], epochs=5
            )

            self.assertEqual(cbk._current_epoch, 4)
            self.assertEqual(hist.epoch[-1], 4)

    # Checking if after interruption, correct model params and
    # weights are loaded in epoch-wise backup
    @pytest.mark.requires_trainable_backend
    def test_best_case_epoch(self):
        def make_model():
            np.random.seed(1337)
            model = Sequential(
                [
                    layers.Dense(2, activation="relu"),
                    layers.Dense(1),
                ]
            )
            model.compile(
                loss="mse",
                optimizer="sgd",
                metrics=["mse"],
            )
            return model

        temp_dir = self.get_temp_dir()
        filepath = os.path.join(temp_dir, "subdir", "checkpoint.weights.h5")
        file_utils.rmtree(filepath)
        self.assertFalse(os.path.exists(filepath))

        model = make_model()
        cbk = callbacks.BackupAndRestore(file_path=filepath, save_freq="epoch")

        x_train = np.random.random((10, 3))
        y_train = np.random.random((10, 1))

        try:
            model.fit(
                x_train,
                y_train,
                batch_size=4,
                callbacks=[
                    cbk,
                    InterruptingCallback(steps_int=None, epoch_int=2),
                ],
                epochs=6,
                verbose=0,
            )
        except RuntimeError:
            self.assertEqual(cbk._current_epoch, 1)
            self.assertTrue(os.path.exists(filepath))

            hist = model.fit(
                x_train, y_train, batch_size=4, callbacks=[cbk], epochs=5
            )
            self.assertEqual(cbk._current_epoch, 4)
            self.assertEqual(hist.epoch[-1], 4)

    # Checking if after interruption, when model is deleted
    @pytest.mark.requires_trainable_backend
    def test_model_deleted_case_epoch(self):
        def make_model():
            np.random.seed(1337)
            model = Sequential(
                [
                    layers.Dense(2, activation="relu"),
                    layers.Dense(1),
                ]
            )
            model.compile(
                loss="mse",
                optimizer="sgd",
                metrics=["mse"],
            )
            return model

        temp_dir = self.get_temp_dir()
        filepath = os.path.join(temp_dir, "subdir", "checkpoint.weights.h5")
        file_utils.rmtree(filepath)
        self.assertFalse(os.path.exists(filepath))

        model = make_model()
        cbk = callbacks.BackupAndRestore(file_path=filepath, save_freq="epoch")

        x_train = np.random.random((10, 3))
        y_train = np.random.random((10, 1))

        try:
            model.fit(
                x_train,
                y_train,
                batch_size=4,
                callbacks=[
                    cbk,
                    InterruptingCallback(steps_int=None, epoch_int=2),
                ],
                epochs=6,
                verbose=0,
            )
        except RuntimeError:
            self.assertEqual(cbk._current_epoch, 1)
            self.assertTrue(os.path.exists(filepath))
            file_utils.rmtree(filepath)

            model.fit(x_train, y_train, batch_size=4, callbacks=[cbk], epochs=5)
            self.assertEqual(cbk._current_epoch, 4)
