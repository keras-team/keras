import numpy as np
import pytest

from keras.src import callbacks
from keras.src import layers
from keras.src import metrics
from keras.src import models
from keras.src import ops
from keras.src import testing


class MonitorCallbackTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_monitor_op_logic(self):
        x_train = np.random.random((10, 5))
        y_train = np.random.random((10, 1))
        x_test = np.random.random((10, 5))
        y_test = np.random.random((10, 1))
        model = models.Sequential(
            (
                layers.Dense(1, activation="relu"),
                layers.Dense(1, activation="relu"),
            )
        )
        model.compile(
            loss="mae",
            optimizer="adam",
            metrics=[
                "mse",
                "acc",
                "accuracy",
                "hinge",
                metrics.F1Score(name="f1_score"),
            ],
        )

        cases = [
            ("max", "val_mse", "max"),
            ("min", "val_loss", "min"),
            ("auto", "val_mse", "min"),
            ("auto", "loss", "min"),
            ("auto", "acc", "max"),
            ("auto", "val_accuracy", "max"),
            ("auto", "hinge", "min"),
            ("auto", "f1_score", "max"),
        ]
        for mode, monitor, expected_mode in cases:
            monitor_callback = callbacks.MonitorCallback(monitor, mode)
            monitor_callback.set_model(model)
            model.fit(
                x_train,
                y_train,
                batch_size=5,
                validation_data=(x_test, y_test),
                epochs=2,
                verbose=0,
            )
            monitor_callback._set_monitor_op()
            if expected_mode == "max":
                monitor_op = ops.greater
            else:
                monitor_op = ops.less
            self.assertEqual(monitor_callback.monitor_op, monitor_op)

        with self.assertRaises(ValueError):
            monitor = "unknown"
            monitor_callback = callbacks.MonitorCallback(monitor)
            monitor_callback.set_model(model)
            model.fit(
                x_train,
                y_train,
                batch_size=5,
                validation_data=(x_test, y_test),
                epochs=2,
                verbose=0,
            )
            monitor_callback._set_monitor_op()

    @pytest.mark.requires_trainable_backend
    def test_min_delta(self):
        monitor_callback = callbacks.MonitorCallback(mode="max", min_delta=0.5)
        monitor_callback._set_monitor_op()
        self.assertTrue(monitor_callback._is_improvement(0.75, 0))
        self.assertTrue(monitor_callback._is_improvement(0.5, None))
        self.assertFalse(monitor_callback._is_improvement(0.5, 0))
        self.assertFalse(monitor_callback._is_improvement(0.2, 0.5))
