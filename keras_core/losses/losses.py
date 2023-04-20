from keras_core import backend
from keras_core import operations as ops
from keras_core.losses.loss import Loss
from keras_core.losses.loss import squeeze_to_same_rank


class LossFunctionWrapper(Loss):
    def __init__(
        self, fn, reduction="sum_over_batch_size", name=None, **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(clf, config):
        raise NotImplementedError


class MeanSquaredError(LossFunctionWrapper):
    """Computes the mean of squares of errors between labels and predictions.

    Formula:

    ```python
    loss = mean(square(y_true - y_pred))
    ```

    Args:
        reduction: Type of reduction to apply to loss. For almost all cases
            this defaults to `"sum_over_batch_size"`. Options are `"sum"`,
            `"sum_over_batch_size"` or None.
        name: Optional name for the instance.
    """

    def __init__(
        self, reduction="sum_over_batch_size", name="mean_squared_error"
    ):
        super().__init__(mean_squared_error, reduction=reduction, name=name)


class MeanAbsoluteError(LossFunctionWrapper):
    """Computes the mean of absolute difference between labels and predictions.

    Formula:

    ```python
    loss = mean(abs(y_true - y_pred))
    ```

    Args:
        reduction: Type of reduction to apply to loss. For almost all cases
            this defaults to `"sum_over_batch_size"`. Options are `"sum"`,
            `"sum_over_batch_size"` or None.
        name: Optional name for the instance.
    """

    def __init__(
        self, reduction="sum_over_batch_size", name="mean_absolute_error"
    ):
        super().__init__(mean_absolute_error, reduction=reduction, name=name)


class MeanAbsolutePercentageError(LossFunctionWrapper):
    """Computes the mean absolute percentage error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = 100 * mean(abs((y_true - y_pred) / y_true))
    ```

    Args:
        reduction: Type of reduction to apply to loss. For almost all cases
            this defaults to `"sum_over_batch_size"`. Options are `"sum"`,
            `"sum_over_batch_size"` or None.
        name: Optional name for the instance.
    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="mean_absolute_percentage_error",
    ):
        super().__init__(
            mean_absolute_percentage_error, reduction=reduction, name=name
        )


class MeanSquaredLogarithmicError(LossFunctionWrapper):
    """Computes the mean squared logarithmic error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
    ```

    Args:
        reduction: Type of reduction to apply to loss. For almost all cases
            this defaults to `"sum_over_batch_size"`. Options are `"sum"`,
            `"sum_over_batch_size"` or None.
        name: Optional name for the instance.
    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="mean_squared_logarithmic_error",
    ):
        super().__init__(
            mean_squared_logarithmic_error, reduction=reduction, name=name
        )


def mean_squared_error(y_true, y_pred):
    """Computes the mean squared error between labels and predictions.

    Formula:

    ```python
    loss = mean(square(y_true - y_pred), axis=-1)
    ```

    Standalone usage:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.mean_squared_error(y_true, y_pred)

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean squared error values with shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
    return ops.mean(ops.square(y_true - y_pred), axis=-1)


def mean_absolute_error(y_true, y_pred):
    """Computes the mean absolute error between labels and predictions.

    ```python
    loss = mean(abs(y_true - y_pred), axis=-1)
    ```

    Standalone usage:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.mean_absolute_error(y_true, y_pred)

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean absolute error values with shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
    return ops.mean(ops.abs(y_true - y_pred), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    """Computes the mean absolute percentage error between `y_true` & `y_pred`.

    `loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)`

    Division by zero is prevented by dividing by `maximum(y_true, epsilon)`
    where `epsilon = keras_core.backend.epsilon()`
    (default to `1e-7`).

    Standalone usage:

    >>> y_true = np.random.random(size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.mean_absolute_percentage_error(y_true, y_pred)

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean absolute percentage error values with shape = `[batch_size, d0, ..
        dN-1]`.
    """
    epsilon = ops.convert_to_tensor(backend.epsilon())
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
    diff = ops.abs((y_true - y_pred) / ops.maximum(ops.abs(y_true), epsilon))
    return 100.0 * ops.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    """Computes the mean squared logarithmic error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)
    ```

    Note that `y_pred` and `y_true` cannot be less or equal to 0. Negative
    values and 0 values will be replaced with `keras_core.backend.epsilon()`
    (default to `1e-7`).

    Standalone usage:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.mean_squared_logarithmic_error(y_true, y_pred)

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean squared logarithmic error values. shape = `[batch_size, d0, ..
        dN-1]`.
    """
    epsilon = ops.convert_to_tensor(backend.epsilon())
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
    first_log = ops.log(ops.maximum(y_pred, epsilon) + 1.0)
    second_log = ops.log(ops.maximum(y_true, epsilon) + 1.0)
    return ops.mean(ops.square(first_log - second_log), axis=-1)
