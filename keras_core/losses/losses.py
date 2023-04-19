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

    `loss = mean(square(y_true - y_pred))`

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

    `loss = mean(abs(y_true - y_pred))`

    Args:
        reduction: Type of reduction to apply to loss. For almost all cases
            this defaults to `"sum_over_batch_size"`. Options are `"sum"`,
            `"sum_over_batch_size"` or None.
        name: Optional name for the instance.
    """

    def __init__(
        self, reduction="sum_over_batch_size", name="mean_squared_error"
    ):
        super().__init__(mean_absolute_error, reduction=reduction, name=name)


def mean_squared_error(y_true, y_pred):
    """Computes the mean squared error between labels and predictions.

    After computing the squared distance between the inputs, the mean value over
    the last dimension is returned.

    `loss = mean(square(y_true - y_pred), axis=-1)`

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

    After computing the absolute distance between the inputs, the mean value
    over the last dimension is returned.

    `loss = mean(abs(y_true - y_pred), axis=-1)`

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
