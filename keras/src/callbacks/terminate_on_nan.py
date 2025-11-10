import numpy as np

from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.utils import io_utils


@keras_export("keras.callbacks.TerminateOnNaN")
class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered."""

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                io_utils.print_msg(
                    f"Batch {batch}: Invalid loss, terminating training"
                )
                self.model.stop_training = True


@keras_export("keras.callbacks.HardTerminateOnNaN")
class HardTerminateOnNaN(Callback):
    """Callback that terminates training immediately
    when NaN/Inf loss is detected.

    This callback raises a RuntimeError when a NaN or Inf loss is encountered,
    which immediately stops training without triggering `on_train_end()` hooks
    for other callbacks. This is useful when you want to preserve backup states
    or prevent early stopping from restoring weights after a NaN failure.

    Unlike `TerminateOnNaN`, which gracefully stops training using
    `model.stop_training = True` and triggers all callback cleanup methods,
    `HardTerminateOnNaN` crashes the training loop immediately.

    Example:

    ```
    callback = keras.callbacks.HardTerminateOnNaN()
    model.fit(x, y, callbacks=[callback])
    ```
    """

    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_batch_end(self, batch, logs=None):
        """Check for NaN/Inf loss at the end of each batch.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict, contains the return value of `model.train_step()`.

        Raises:
            RuntimeError: If loss is NaN or Inf.
        """
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                raise RuntimeError(
                    f"NaN or Inf loss encountered at batch {batch}. "
                    f"Loss value: {loss}. Terminating training immediately."
                )
