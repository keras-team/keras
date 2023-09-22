import numpy as np
import pytest
from absl import logging

from keras import callbacks
from keras import layers
from keras import losses
from keras import optimizers
from keras import testing
from keras.models.sequential import Sequential


class LambdaCallbackTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_lambda_callback(self):
        """Test standard LambdaCallback functionalities with training."""
        batch_size = 4
        model = Sequential(
            [layers.Input(shape=(2,), batch_size=batch_size), layers.Dense(1)]
        )
        model.compile(
            optimizer=optimizers.SGD(), loss=losses.MeanSquaredError()
        )
        x = np.random.randn(16, 2)
        y = np.random.randn(16, 1)
        lambda_log_callback = callbacks.LambdaCallback(
            on_train_begin=lambda logs: logging.warning("on_train_begin"),
            on_epoch_begin=lambda epoch, logs: logging.warning(
                "on_epoch_begin"
            ),
            on_epoch_end=lambda epoch, logs: logging.warning("on_epoch_end"),
            on_train_end=lambda logs: logging.warning("on_train_end"),
        )
        with self.assertLogs(level="WARNING") as logs:
            model.fit(
                x,
                y,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[lambda_log_callback],
                epochs=5,
                verbose=0,
            )
            self.assertTrue(any("on_train_begin" in log for log in logs.output))
            self.assertTrue(any("on_epoch_begin" in log for log in logs.output))
            self.assertTrue(any("on_epoch_end" in log for log in logs.output))
            self.assertTrue(any("on_train_end" in log for log in logs.output))

    @pytest.mark.requires_trainable_backend
    def test_lambda_callback_with_batches(self):
        """Test LambdaCallback's behavior with batch-level callbacks."""
        batch_size = 4
        model = Sequential(
            [layers.Input(shape=(2,), batch_size=batch_size), layers.Dense(1)]
        )
        model.compile(
            optimizer=optimizers.SGD(), loss=losses.MeanSquaredError()
        )
        x = np.random.randn(16, 2)
        y = np.random.randn(16, 1)
        lambda_log_callback = callbacks.LambdaCallback(
            on_train_batch_begin=lambda batch, logs: logging.warning(
                "on_train_batch_begin"
            ),
            on_train_batch_end=lambda batch, logs: logging.warning(
                "on_train_batch_end"
            ),
        )
        with self.assertLogs(level="WARNING") as logs:
            model.fit(
                x,
                y,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[lambda_log_callback],
                epochs=5,
                verbose=0,
            )
            self.assertTrue(
                any("on_train_batch_begin" in log for log in logs.output)
            )
            self.assertTrue(
                any("on_train_batch_end" in log for log in logs.output)
            )

    @pytest.mark.requires_trainable_backend
    def test_lambda_callback_with_kwargs(self):
        """Test LambdaCallback's behavior with custom defined callback."""
        batch_size = 4
        model = Sequential(
            [layers.Input(shape=(2,), batch_size=batch_size), layers.Dense(1)]
        )
        model.compile(
            optimizer=optimizers.SGD(), loss=losses.MeanSquaredError()
        )
        x = np.random.randn(16, 2)
        y = np.random.randn(16, 1)
        model.fit(
            x, y, batch_size=batch_size, epochs=1, verbose=0
        )  # Train briefly for evaluation to work.

        def custom_on_test_begin(logs):
            logging.warning("custom_on_test_begin_executed")

        lambda_log_callback = callbacks.LambdaCallback(
            on_test_begin=custom_on_test_begin
        )
        with self.assertLogs(level="WARNING") as logs:
            model.evaluate(
                x,
                y,
                batch_size=batch_size,
                callbacks=[lambda_log_callback],
                verbose=0,
            )
            self.assertTrue(
                any(
                    "custom_on_test_begin_executed" in log
                    for log in logs.output
                )
            )

    @pytest.mark.requires_trainable_backend
    def test_lambda_callback_no_args(self):
        """Test initializing LambdaCallback without any arguments."""
        lambda_callback = callbacks.LambdaCallback()
        self.assertIsInstance(lambda_callback, callbacks.LambdaCallback)

    @pytest.mark.requires_trainable_backend
    def test_lambda_callback_with_additional_kwargs(self):
        """Test initializing LambdaCallback with non-predefined kwargs."""

        def custom_callback(logs):
            pass

        lambda_callback = callbacks.LambdaCallback(
            custom_method=custom_callback
        )
        self.assertTrue(hasattr(lambda_callback, "custom_method"))

    @pytest.mark.requires_trainable_backend
    def test_lambda_callback_during_prediction(self):
        """Test LambdaCallback's functionality during model prediction."""
        batch_size = 4
        model = Sequential(
            [layers.Input(shape=(2,), batch_size=batch_size), layers.Dense(1)]
        )
        model.compile(
            optimizer=optimizers.SGD(), loss=losses.MeanSquaredError()
        )
        x = np.random.randn(16, 2)

        def custom_on_predict_begin(logs):
            logging.warning("on_predict_begin_executed")

        lambda_callback = callbacks.LambdaCallback(
            on_predict_begin=custom_on_predict_begin
        )
        with self.assertLogs(level="WARNING") as logs:
            model.predict(
                x, batch_size=batch_size, callbacks=[lambda_callback], verbose=0
            )
            self.assertTrue(
                any("on_predict_begin_executed" in log for log in logs.output)
            )
