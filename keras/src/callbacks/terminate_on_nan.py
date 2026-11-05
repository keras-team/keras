import numpy as np

from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.utils import io_utils


@keras_export("keras.callbacks.TerminateOnNaN")
class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered.

    This callback monitors the loss value during training
    and terminates training when a NaN or Inf loss is detected.
    By default, training is stopped gracefully
    by setting `model.stop_training = True`, which triggers all callback cleanup
    methods including `on_train_end()`.

    Alternatively, you can use `raise_error=True` to immediately raise a
    RuntimeError when NaN/Inf is detected. This raise_error termination
    prevents `on_train_end()` from being called on other callbacks, which
    is useful for preserving backup states or preventing unintended cleanup
    when training fails.

    Args:
        raise_error: Boolean, default False. If False, uses graceful stop via
            `model.stop_training = True`. If True, immediately raises
            RuntimeError on NaN/Inf loss, bypassing callback cleanup methods.

    Example:

    ```
    # Graceful termination (default)
    callback = keras.callbacks.TerminateOnNaN()
    model.fit(x, y, callbacks=[callback])

    # raise_error termination (strict failure)
    callback = keras.callbacks.TerminateOnNaN(raise_error=True)
    model.fit(x, y, callbacks=[callback])
    ```
    """

    def __init__(self, raise_error: bool = False):
        super().__init__()
        self.raise_error = raise_error

    def on_batch_end(self, batch, logs=None):
        """Check for NaN/Inf loss at the end of each batch.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict, contains the return value of `model.train_step()`.

        Raises:
            RuntimeError: If loss is NaN/Inf and raise_error=True.
        """
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                if self.raise_error:
                    raise RuntimeError(
                        f"NaN or Inf loss encountered at batch {batch}. "
                        f"Loss value: {loss}. Terminating training immediately."
                    )
                else:
                    io_utils.print_msg(
                        f"Batch {batch}: Invalid loss, terminating training"
                    )
                    self.model.stop_training = True
