import warnings

import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.callbacks.monitor_callback import MonitorCallback
from keras.src.utils import io_utils


@keras_export("keras.callbacks.ReduceLROnPlateau")
class ReduceLROnPlateau(MonitorCallback):
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Example:

    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(x_train, y_train, callbacks=[reduce_lr])
    ```

    Args:
        monitor: String. Quantity to be monitored.
        factor: Float. Factor by which the learning rate will be reduced.
            `new_lr = lr * factor`.
        patience: Integer. Number of epochs with no improvement after which
            learning rate will be reduced.
        verbose: Integer. 0: quiet, 1: update messages.
        mode: String. One of `{'auto', 'min', 'max'}`. In `'min'` mode,
            the learning rate will be reduced when the
            quantity monitored has stopped decreasing; in `'max'` mode it will
            be reduced when the quantity monitored has stopped increasing; in
            `'auto'` mode, the direction is automatically inferred from the name
            of the monitored quantity.
        min_delta: Float. Threshold for measuring the new optimum, to only focus
            on significant changes.
        cooldown: Integer. Number of epochs to wait before resuming normal
            operation after the learning rate has been reduced.
        min_lr: Float. Lower bound on the learning rate.
    """

    def __init__(
        self,
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=1e-4,
        cooldown=0,
        min_lr=0.0,
        **kwargs,
    ):
        super().__init__(monitor, mode, min_delta=min_delta)
        if factor >= 1.0:
            raise ValueError(
                "ReduceLROnPlateau does not support a factor >= 1.0. "
                f"Received factor={factor}"
            )

        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0

    def _reset(self):
        """Resets wait counter and cooldown counter."""
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        if self.monitor_op is None:
            # Delay setup until the model's metrics are all built
            self._set_monitor_op()
        logs = logs or {}
        logs["learning_rate"] = float(
            backend.convert_to_numpy(self.model.optimizer.learning_rate)
        )
        current = logs.get(self.monitor)

        if current is None:
            warnings.warn(
                "Learning rate reduction is conditioned on metric "
                f"`{self.monitor}` which is not available. Available metrics "
                f"are: {','.join(list(logs.keys()))}.",
                stacklevel=2,
            )
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self._is_improvement(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(
                        backend.convert_to_numpy(
                            self.model.optimizer.learning_rate
                        )
                    )
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        self.model.optimizer.learning_rate = new_lr
                        if self.verbose > 0:
                            io_utils.print_msg(
                                f"\nEpoch {epoch + 1}: "
                                "ReduceLROnPlateau reducing "
                                f"learning rate to {new_lr}."
                            )
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0
