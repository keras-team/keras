import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.utils import io_utils


@keras_export("keras.callbacks.LearningRateScheduler")
class LearningRateScheduler(Callback):
    """Learning rate scheduler.

    At the beginning of every epoch, this callback gets the updated learning
    rate value from `schedule` function provided at `__init__`, with the current
    epoch and current learning rate, and applies the updated learning rate on
    the optimizer.

    Args:
        schedule: A function that takes an epoch index (integer, indexed from 0)
            and current learning rate (float) as inputs and returns a new
            learning rate as output (float).
        verbose: Integer. 0: quiet, 1: log update messages.

    Example:

    >>> # This function keeps the initial learning rate for the first ten epochs
    >>> # and decreases it exponentially after that.
    >>> def scheduler(epoch, lr):
    ...     if epoch < 10:
    ...         return lr
    ...     else:
    ...         return lr * ops.exp(-0.1)
    >>>
    >>> model = keras.models.Sequential([keras.layers.Dense(10)])
    >>> model.compile(keras.optimizers.SGD(), loss='mse')
    >>> round(model.optimizer.learning_rate, 5)
    0.01

    >>> callback = keras.callbacks.LearningRateScheduler(scheduler)
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=15, callbacks=[callback], verbose=0)
    >>> round(model.optimizer.learning_rate, 5)
    0.00607

    """

    def __init__(self, schedule, verbose=0):
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')

        try:  # new API
            learning_rate = float(
                backend.convert_to_numpy(self.model.optimizer.learning_rate)
            )
            learning_rate = self.schedule(epoch, learning_rate)
        except TypeError:  # Support for old API for backward compatibility
            learning_rate = self.schedule(epoch)

        if not isinstance(learning_rate, (float, np.float32, np.float64)):
            raise ValueError(
                "The output of the `schedule` function should be a float. "
                f"Got: {learning_rate}"
            )

        self.model.optimizer.learning_rate = learning_rate
        if self.verbose > 0:
            io_utils.print_msg(
                f"\nEpoch {epoch + 1}: LearningRateScheduler setting learning "
                f"rate to {learning_rate}."
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["learning_rate"] = float(
            backend.convert_to_numpy(self.model.optimizer.learning_rate)
        )
