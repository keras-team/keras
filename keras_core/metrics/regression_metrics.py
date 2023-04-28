from keras_core.api_export import keras_core_export
from keras_core.losses.losses import mean_squared_error
from keras_core.metrics import reduction_metrics


@keras_core_export("keras_core.metrics.MeanSquaredError")
class MeanSquaredError(reduction_metrics.MeanMetricWrapper):
    """Computes the mean squared error between `y_true` and `y_pred`.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.MeanSquaredError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    0.25
    """

    def __init__(self, name="mean_squared_error", dtype=None):
        super().__init__(fn=mean_squared_error, name=name, dtype=dtype)

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}
