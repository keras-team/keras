from absl import logging
import numpy as np

from keras_core import testing
from keras_core import optimizers
from keras_core import layers
from keras_core import losses
from keras_core import callbacks
from keras_core.models.sequential import Sequential


class LambdaCallbackTest(testing.TestCase):
    def test_LambdaCallback(self):
        BATCH_SIZE = 4
        model = Sequential(
            [layers.Input(shape=(2,), batch_size=BATCH_SIZE), layers.Dense(1)]
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
                batch_size=BATCH_SIZE,
                validation_split=0.2,
                callbacks=[lambda_log_callback],
                epochs=5,
                verbose=0,
            )
            self.assertTrue(any("on_train_begin" in log for log in logs.output))
            self.assertTrue(any("on_epoch_begin" in log for log in logs.output))
            self.assertTrue(any("on_epoch_end" in log for log in logs.output))
            self.assertTrue(any("on_train_end" in log for log in logs.output))
