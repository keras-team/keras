from keras.src.api_export import keras_export
from keras.src.losses.losses import categorical_hinge
from keras.src.losses.losses import hinge
from keras.src.losses.losses import squared_hinge
from keras.src.metrics import reduction_metrics


@keras_export("keras.metrics.Hinge")
class Hinge(reduction_metrics.MeanMetricWrapper):
    """Computes the hinge metric between `y_true` and `y_pred`.

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:

    >>> m = keras.metrics.Hinge()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result()
    1.3
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    1.1
    """

    def __init__(self, name="hinge", dtype=None):
        super().__init__(fn=hinge, name=name, dtype=dtype)
        # Metric should be minimized during optimization.
        self._direction = "down"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_export("keras.metrics.SquaredHinge")
class SquaredHinge(reduction_metrics.MeanMetricWrapper):
    """Computes the hinge metric between `y_true` and `y_pred`.

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

    >>> m = keras.metrics.SquaredHinge()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result()
    1.86
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    1.46
    """

    def __init__(self, name="squared_hinge", dtype=None):
        super().__init__(fn=squared_hinge, name=name, dtype=dtype)
        # Metric should be minimized during optimization.
        self._direction = "down"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_export("keras.metrics.CategoricalHinge")
class CategoricalHinge(reduction_metrics.MeanMetricWrapper):
    """Computes the categorical hinge metric between `y_true` and `y_pred`.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.CategoricalHinge()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result().numpy()
    1.4000001
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    1.2
    """

    def __init__(self, name="categorical_hinge", dtype=None):
        super().__init__(fn=categorical_hinge, name=name, dtype=dtype)
        # Metric should be minimized during optimization.
        self._direction = "down"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}
