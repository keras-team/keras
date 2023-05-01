from keras_core.api_export import keras_core_export
from keras_core.losses.losses import kl_divergence
from keras_core.losses.losses import poisson
from keras_core.metrics import reduction_metrics


@keras_core_export("keras_core.metrics.KLDivergence")
class KLDivergence(reduction_metrics.MeanMetricWrapper):
    """Computes Kullback-Leibler divergence metric between `y_true` and
    `y_pred`.

    Formula:

    ```python
    metric = y_true * log(y_true / y_pred)
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.KLDivergence()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result()
    0.45814306

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    0.9162892

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[keras_core.metrics.KLDivergence()])
    ```
    """

    def __init__(self, name="kl_divergence", dtype=None):
        super().__init__(fn=kl_divergence, name=name, dtype=dtype)

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_core_export("keras_core.metrics.Poisson")
class Poisson(reduction_metrics.MeanMetricWrapper):
    """Computes the Poisson metric between `y_true` and `y_pred`.

    Formula:

    ```python
    metric = y_pred - y_true * log(y_pred)
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.Poisson()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    0.49999997

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    0.99999994

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[keras_core.metrics.Poisson()])
    ```
    """

    def __init__(self, name="poisson", dtype=None):
        super().__init__(fn=poisson, name=name, dtype=dtype)

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}
