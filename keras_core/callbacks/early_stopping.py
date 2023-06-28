import warnings

from keras_core import ops
from keras_core.api_export import keras_core_export
from keras_core.callbacks.callback import Callback
from keras_core.utils import io_utils


@keras_core_export("keras_core.callbacks.EarlyStopping")
class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.

    Assuming the goal of a training is to minimize the loss. With this, the
    metric to be monitored would be `'loss'`, and mode would be `'min'`. A
    `model.fit()` training loop will check at end of every epoch whether
    the loss is no longer decreasing, considering the `min_delta` and
    `patience` if applicable. Once it's found no longer decreasing,
    `model.stop_training` is marked True and the training terminates.

    The quantity to be monitored needs to be available in `logs` dict.
    To make it so, pass the loss or metrics at `model.compile()`.

    Args:
        monitor: Quantity to be monitored. Defaults to `"val_loss"`.
        min_delta: Minimum change in the monitored quantity to qualify as an
            improvement, i.e. an absolute change of less than min_delta, will
            count as no improvement. Defaults to `0`.
        patience: Number of epochs with no improvement after which training will
            be stopped. Defaults to `0`.
        verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays
            messages when the callback takes an action. Defaults to `0`.
        mode: One of `{"auto", "min", "max"}`. In `min` mode, training will stop
            when the quantity monitored has stopped decreasing; in `"max"` mode
            it will stop when the quantity monitored has stopped increasing; in
            `"auto"` mode, the direction is automatically inferred from the name
            of the monitored quantity. Defaults to `"auto"`.
        baseline: Baseline value for the monitored quantity. If not `None`,
            training will stop if the model doesn't show improvement over the
            baseline. Defaults to `None`.
        restore_best_weights: Whether to restore model weights from the epoch
            with the best value of the monitored quantity. If `False`, the model
            weights obtained at the last step of training are used. An epoch
            will be restored regardless of the performance relative to the
            `baseline`. If no epoch improves on `baseline`, training will run
            for `patience` epochs and restore weights from the best epoch in
            that set. Defaults to `False`.
        start_from_epoch: Number of epochs to wait before starting to monitor
            improvement. This allows for a warm-up period in which no
            improvement is expected and thus training will not be stopped.
            Defaults to `0`.


    Example:

    >>> callback = keras_core.callbacks.EarlyStopping(monitor='loss',
    ...                                               patience=3)
    >>> # This callback will stop the training when there is no improvement in
    >>> # the loss for three consecutive epochs.
    >>> model = keras_core.models.Sequential([keras_core.layers.Dense(10)])
    >>> model.compile(keras_core.optimizers.SGD(), loss='mse')
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=10, batch_size=1, callbacks=[callback],
    ...                     verbose=0)
    >>> len(history.history['loss'])  # Only 4 epochs are run.
    4
    """

    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
    ):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.start_from_epoch = start_from_epoch

        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                f"EarlyStopping mode {mode} is unknown, fallback to auto mode.",
                stacklevel=2,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = ops.less
        elif mode == "max":
            self.monitor_op = ops.greater
        else:
            if (
                self.monitor.endswith("acc")
                or self.monitor.endswith("accuracy")
                or self.monitor.endswith("auc")
            ):
                self.monitor_op = ops.greater
            else:
                self.monitor_op = ops.less

        if self.monitor_op == ops.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = (
            float("inf") if self.monitor_op == ops.less else -float("inf")
        )
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous
            # best.
            if self.baseline is None or self._is_improvement(
                current, self.baseline
            ):
                self.wait = 0
            return

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    io_utils.print_msg(
                        "Restoring model weights from "
                        "the end of the best epoch: "
                        f"{self.best_epoch + 1}."
                    )
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            io_utils.print_msg(
                f"Epoch {self.stopped_epoch + 1}: early stopping"
            )

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                (
                    f"Early stopping conditioned on metric `{self.monitor}` "
                    "which is not available. "
                    f"Available metrics are: {','.join(list(logs.keys()))}"
                ),
                stacklevel=2,
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)
