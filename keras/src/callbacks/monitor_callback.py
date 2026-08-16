import warnings

from keras.src import ops
from keras.src.callbacks.callback import Callback
from keras.src.trainers import compile_utils


class MonitorCallback(Callback):
    """Base class for callbacks that monitor a quantity and evaluates
    improvements.

    This class provides common functionality for callbacks that monitor a
    metric during training to determine whether a condition has been met,
    such as improvement over time. It encapsulates logic for selecting
    the comparison operation based on a `monitor` value and `mode`, and
    computing whether a new value is an improvement.

    It is intended to be subclassed by other callbacks like `ModelCheckpoint`,
    `EarlyStopping`, or `ReduceLROnPlateau`, and is not meant to be used
    directly.

    Arguments:
        monitor: Quantity to be monitored. Defaults to `"val_loss"`.
        mode: One of `{"auto", "min", "max"}`. In `min` mode, training will aim
            to minimize the monitored quantity; in `'max'` mode it will aim to
            maximize it.; in `"auto"` mode, the direction is automatically
            inferred from the name of the monitored quantity. Defaults to
            `"auto"`.
        baseline: Floating point initial "best" value of the metric to be
            monitored. If `None` (default), the first monitored value will be
            used.
        min_delta: Minimum change in the monitored quantity to qualify as an
            improvement, i.e. an absolute change of less than min_delta, will
            count as no improvement. Defaults to `0`.

    Raises:
        ValueError: If `mode='auto'` is selected and the direction of the metric
        cannot be inferred.
    """

    def __init__(
        self,
        monitor="val_loss",
        mode="auto",
        baseline=None,
        min_delta=0,
    ):
        super().__init__()
        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                f"{self.__class__.__name__} mode '{mode}' is unknown, fallback "
                "to auto mode.",
                stacklevel=2,
            )
            mode = "auto"
        self.monitor = monitor
        self.mode = mode
        self.best = baseline
        self.min_delta = abs(min_delta)
        self.monitor_op = None

    def _set_monitor_op(self):
        if self.mode == "min":
            self.monitor_op = ops.less
        elif self.mode == "max":
            self.monitor_op = ops.greater
        else:
            metric_name = self.monitor.removeprefix("val_")
            if metric_name == "loss":
                self.monitor_op = ops.less
            if hasattr(self.model, "metrics"):
                all_metrics = []
                for m in self.model.metrics:
                    if isinstance(
                        m,
                        (
                            compile_utils.CompileMetrics,
                            compile_utils.MetricsList,
                        ),
                    ):
                        all_metrics.extend(m.metrics)
                for m in all_metrics:
                    if m.name == metric_name:
                        if hasattr(m, "_direction"):
                            if m._direction == "up":
                                self.monitor_op = ops.greater
                            else:
                                self.monitor_op = ops.less
            if self.monitor_op is None:
                raise ValueError(
                    f"{self.__class__.__name__} callback received "
                    f"monitor={self.monitor}, but Keras isn't able to "
                    "automatically determine whether that metric should be "
                    "maximized or minimized. Pass `mode='max'` in order to "
                    "monitor based on the highest metric value, or pass "
                    "`mode='min'` in order to use the lowest value."
                )
        if self.monitor_op == ops.less:
            self.min_delta *= -1

    def _is_improvement(self, monitor_value, reference_value):
        if reference_value is None:
            return True
        return self.monitor_op(monitor_value - self.min_delta, reference_value)
