import warnings
from unittest import mock

import numpy as np

from keras import backend
from keras import callbacks
from keras import layers
from keras import testing
from keras.models import Sequential
from keras.utils import numerical_utils

try:
    import requests
except ImportError:
    requests = None


class TerminateOnNaNTest(testing.TestCase):
    def test_RemoteMonitor(self):
        if requests is None:
            self.skipTest("`requests` required to run this test")

        monitor = callbacks.RemoteMonitor()
        # This will raise a warning since the default address in unreachable:
        warning_msg = "Could not reach RemoteMonitor root server"
        with warnings.catch_warnings(record=True) as warning_logs:
            warnings.simplefilter("always")
            monitor.on_epoch_end(0, logs={"loss": 0.0})
            self.assertIn(warning_msg, str(warning_logs[-1].message))

    def test_RemoteMonitor_np_array(self):
        if requests is None:
            self.skipTest("`requests` required to run this test")

        with mock.patch("requests.post") as requests_post:
            monitor = callbacks.RemoteMonitor(send_as_json=True)
            a = np.arange(1)  # a 1 by 1 array
            logs = {"loss": 0.0, "val": a}
            monitor.on_epoch_end(0, logs=logs)
            send = {"loss": 0.0, "epoch": 0, "val": 0}
            requests_post.assert_called_once_with(
                monitor.root + monitor.path, json=send, headers=monitor.headers
            )

    def test_RemoteMonitor_np_float32(self):
        if requests is None:
            self.skipTest("`requests` required to run this test")

        with mock.patch("requests.post") as requests_post:
            monitor = callbacks.RemoteMonitor(send_as_json=True)
            a = np.float32(1.0)  # a float32 generic type
            logs = {"loss": 0.0, "val": a}
            monitor.on_epoch_end(0, logs=logs)
            send = {"loss": 0.0, "epoch": 0, "val": 1.0}
            requests_post.assert_called_once_with(
                monitor.root + monitor.path, json=send, headers=monitor.headers
            )

    def test_RemoteMonitorWithJsonPayload(self):
        if requests is None:
            self.skipTest("`requests` required to run this test")

        if backend.backend() == "numpy":
            self.skipTest("Trainer not implemented from NumPy backend.")
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

        model = Sequential([layers.Dense(NUM_CLASSES)])
        model.compile(loss="mean_squared_error", optimizer="sgd")

        with mock.patch("requests.post") as requests_post:
            monitor = callbacks.RemoteMonitor(send_as_json=True)
            hist = model.fit(
                x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=[monitor],
                epochs=1,
            )
            send = {
                "epoch": 0,
                "loss": hist.history["loss"][0],
                "val_loss": hist.history["val_loss"][0],
            }
            requests_post.assert_called_once_with(
                monitor.root + monitor.path, json=send, headers=monitor.headers
            )
