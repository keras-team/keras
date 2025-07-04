from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.metrics import reduction_metrics


@keras_export("keras.metrics.pearson_correlation")
def pearson_correlation(y_true, y_pred, axis=-1):
    """Computes the Pearson coefficient between labels and predictions.

    Formula:

    ```python
    loss = mean(l2norm(y_true - mean(y_true) * l2norm(y_pred - mean(y_pred)))
    ```

    Args:
        y_true: Tensor of true targets.
        y_pred: Tensor of predicted targets.
        axis: Axis along which to determine similarity. Defaults to `-1`.

    Returns:
        Pearson Correlation Coefficient tensor.

    Example:

    >>> y_true = [[0, 1, 0.5], [1, 1, 0.2]]
    >>> y_pred = [[0.1, 0.9, 0.5], [1, 0.9, 0.2]]
    >>> loss = keras.losses.concordance_correlation(
    ...     y_true, y_pred, axis=-1
    ... ).numpy()
    [1.         0.99339927]
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)

    y_true_norm = y_true - ops.mean(y_true, axis=axis, keepdims=True)
    y_pred_norm = y_pred - ops.mean(y_pred, axis=axis, keepdims=True)

    y_true_norm = y_true_norm / ops.std(y_true_norm, axis=axis, keepdims=True)
    y_pred_norm = y_pred_norm / ops.std(y_pred_norm, axis=axis, keepdims=True)

    return ops.mean(y_true_norm * y_pred_norm, axis=axis)


@keras_export("keras.metrics.concordance_correlation")
def concordance_correlation(y_true, y_pred, axis=-1):
    """Computes the Concordance coefficient between labels and predictions.

    Formula:

    ```python
    loss = mean(
        2 * (y_true - mean(y_true) * (y_pred - mean(y_pred)) / (
            var(y_true) + var(y_pred) + square(mean(y_true) - mean(y_pred))
        )
    )
    ```

    Args:
        y_true: Tensor of true targets.
        y_pred: Tensor of predicted targets.
        axis: Axis along which to determine similarity. Defaults to `-1`.

    Returns:
        Concordance Correlation Coefficient tensor.

    Example:

    >>> y_true = [[0, 1, 0.5], [1, 1, 0.2]]
    >>> y_pred = [[0.1, 0.9, 0.5], [1, 0.9, 0.2]]
    >>> loss = keras.losses.concordance_correlation(
    ...     y_true, y_pred, axis=-1
    ... ).numpy()
    [0.97560976 0.98765432]
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)

    y_true_mean = ops.mean(y_true, axis=axis, keepdims=True)
    y_pred_mean = ops.mean(y_pred, axis=axis, keepdims=True)

    y_true_var = ops.var(y_true - y_true_mean, axis=axis, keepdims=True)
    y_pred_var = ops.var(y_pred - y_pred_mean, axis=axis, keepdims=True)

    covar = (y_true - y_pred_mean) * (y_pred - y_pred_mean)
    norm = y_true_var + y_pred_var + ops.square(y_true_mean - y_pred_mean)

    return ops.mean(2 * covar / (norm + backend.epsilon()), axis=axis)


@keras_export("keras.metrics.PearsonCorrelation")
class PearsonCorrelation(reduction_metrics.MeanMetricWrapper):
    """Calculates the Pearson Correlation Coefficient (PCC).

    PCC measures the linear relationship between the true values (`y_true`) and
    the predicted values (`y_pred`). The coefficient ranges from -1 to 1, where
    a value of 1 implies a perfect positive linear correlation, 0 indicates no
    linear correlation, and -1 indicates a perfect negative linear correlation.

    This metric is widely used in regression tasks where the strength of the
    linear relationship between predictions and true labels is an
    important evaluation criterion.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        axis: (Optional) integer or tuple of integers of the axis/axes along
            which to compute the metric. Defaults to `-1`.

    Example:

    >>> pcc = keras.metrics.PearsonCorrelation(axis=-1)
    >>> y_true = [[0, 1, 0.5], [1, 1, 0.2]]
    >>> y_pred = [[0.1, 0.9, 0.5], [1, 0.9, 0.2]]
    >>> pcc.update_state(y_true, y_pred)
    >>> pcc.result()
    0.9966996338993913

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=[keras.metrics.PearsonCorrelation()])
    ```
    """

    def __init__(
        self,
        name="pearson_correlation",
        dtype=None,
        axis=-1,
    ):
        super().__init__(
            fn=pearson_correlation,
            name=name,
            dtype=dtype,
            axis=axis,
        )
        self.axis = axis
        # Metric should be maximized during optimization.
        self._direction = "up"

    def get_config(self):
        return {
            "name": self.name,
            "dtype": self.dtype,
            "axis": self.axis,
        }


@keras_export("keras.metrics.ConcordanceCorrelation")
class ConcordanceCorrelation(reduction_metrics.MeanMetricWrapper):
    """Calculates the Concordance Correlation Coefficient (CCC).

    CCC evaluates the agreement between true values (`y_true`) and predicted
    values (`y_pred`) by considering both precision and accuracy. The
    coefficient ranges from -1 to 1, where a value of 1 indicates perfect
    agreement.

    This metric is useful in regression tasks where it is important to assess
    how well the predictions match the true values, taking into account both
    their correlation and proximity to the 45-degree line of perfect
    concordance.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        axis: (Optional) integer or tuple of integers of the axis/axes along
            which to compute the metric. Defaults to `-1`.

    Example:

    >>> ccc = keras.metrics.ConcordanceCorrelation(axis=-1)
    >>> y_true = [[0, 1, 0.5], [1, 1, 0.2]]
    >>> y_pred = [[0.1, 0.9, 0.5], [1, 0.9, 0.2]]
    >>> ccc.update_state(y_true, y_pred)
    >>> ccc.result()
    0.9816320385426076

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=[keras.metrics.ConcordanceCorrelation()])
    ```
    """

    def __init__(
        self,
        name="concordance_correlation",
        dtype=None,
        axis=-1,
    ):
        super().__init__(
            fn=concordance_correlation,
            name=name,
            dtype=dtype,
            axis=axis,
        )
        self.axis = axis
        # Metric should be maximized during optimization.
        self._direction = "up"

    def get_config(self):
        return {
            "name": self.name,
            "dtype": self.dtype,
            "axis": self.axis,
        }
