from keras_core import backend
from keras_core import operations as ops
from keras_core.api_export import keras_core_export
from keras_core.losses.loss import Loss
from keras_core.losses.loss import squeeze_to_same_rank
from keras_core.saving import serialization_lib
from keras_core.utils.numerical_utils import normalize


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
        base_config = super().get_config()
        config = {"fn": serialization_lib.serialize_keras_object(self.fn)}
        config.update(serialization_lib.serialize_keras_object(self._fn_kwargs))
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        if "fn" in config:
            config = serialization_lib.deserialize_keras_object(config)
        return cls(**config)


@keras_core_export("keras_core.losses.MeanSquaredError")
class MeanSquaredError(LossFunctionWrapper):
    """Computes the mean of squares of errors between labels and predictions.

    Formula:

    ```python
    loss = mean(square(y_true - y_pred))
    ```

    Args:
        reduction: Type of reduction to apply to loss. For almost all cases
            this defaults to `"sum_over_batch_size"`. Options are `"sum"`,
            `"sum_over_batch_size"` or `None`.
        name: Optional name for the instance.
    """

    def __init__(
        self, reduction="sum_over_batch_size", name="mean_squared_error"
    ):
        super().__init__(mean_squared_error, reduction=reduction, name=name)

    def get_config(self):
        return Loss.get_config(self)


@keras_core_export("keras_core.losses.MeanAbsoluteError")
class MeanAbsoluteError(LossFunctionWrapper):
    """Computes the mean of absolute difference between labels and predictions.

    Formula:

    ```python
    loss = mean(abs(y_true - y_pred))
    ```

    Args:
        reduction: Type of reduction to apply to loss. For almost all cases
            this defaults to `"sum_over_batch_size"`. Options are `"sum"`,
            `"sum_over_batch_size"` or `None`.
        name: Optional name for the instance.
    """

    def __init__(
        self, reduction="sum_over_batch_size", name="mean_absolute_error"
    ):
        super().__init__(mean_absolute_error, reduction=reduction, name=name)

    def get_config(self):
        return Loss.get_config(self)


@keras_core_export("keras_core.losses.MeanAbsolutePercentageError")
class MeanAbsolutePercentageError(LossFunctionWrapper):
    """Computes the mean absolute percentage error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = 100 * mean(abs((y_true - y_pred) / y_true))
    ```

    Args:
        reduction: Type of reduction to apply to loss. For almost all cases
            this defaults to `"sum_over_batch_size"`. Options are `"sum"`,
            `"sum_over_batch_size"` or `None`.
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

    def get_config(self):
        return Loss.get_config(self)


@keras_core_export("keras_core.losses.MeanSquaredLogarithmicError")
class MeanSquaredLogarithmicError(LossFunctionWrapper):
    """Computes the mean squared logarithmic error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
    ```

    Args:
        reduction: Type of reduction to apply to loss. For almost all cases
            this defaults to `"sum_over_batch_size"`. Options are `"sum"`,
            `"sum_over_batch_size"` or `None`.
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

    def get_config(self):
        return Loss.get_config(self)


@keras_core_export("keras_core.losses.CosineSimilarity")
class CosineSimilarity(LossFunctionWrapper):
    """Computes the cosine similarity between `y_true` & `y_pred`.

    Note that it is a number between -1 and 1. When it is a negative number
    between -1 and 0, 0 indicates orthogonality and values closer to -1
    indicate greater similarity. This makes it usable as a loss function in a
    setting where you try to maximize the proximity between predictions and
    targets. If either `y_true` or `y_pred` is a zero vector, cosine similarity
    will be 0 regardless of the proximity between predictions and targets.

    Formula:

    ```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
    ```

    Args:
        axis: The axis along which the cosine similarity is computed
            (the features axis). Defaults to -1.
        reduction: Type of reduction to apply to loss. Options are `"sum"`,
            `"sum_over_batch_size"` or `None`. Defaults to
            `"sum_over_batch_size"`.
        name: Optional name for the instance.
    """

    def __init__(
        self,
        axis=-1,
        reduction="sum_over_batch_size",
        name="cosine_similarity",
    ):
        super().__init__(
            cosine_similarity, reduction=reduction, name=name, axis=axis
        )


@keras_core_export("keras_core.losses.Hinge")
class Hinge(LossFunctionWrapper):
    """Computes the hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = maximum(1 - y_true * y_pred, 0)
    ```

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Args:
        reduction: Type of reduction to apply to loss. For almost all cases
            this defaults to `"sum_over_batch_size"`. Options are `"sum"`,
            `"sum_over_batch_size"` or `None`.
        name: Optional name for the instance. Defaults to `"hinge"`
    """

    def __init__(self, reduction="sum_over_batch_size", name="hinge"):
        super().__init__(hinge, reduction=reduction, name=name)

    def get_config(self):
        return Loss.get_config(self)


@keras_core_export("keras_core.losses.SquaredHinge")
class SquaredHinge(LossFunctionWrapper):
    """Computes the squared hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = square(maximum(1 - y_true * y_pred, 0))
    ```

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Args:
        reduction: Type of reduction to apply to loss. For almost all cases
            this defaults to `"sum_over_batch_size"`. Options are `"sum"`,
            `"sum_over_batch_size"` or `None`.
        name: Optional name for the instance. Defaults to `"squared_hinge"`
    """

    def __init__(self, reduction="sum_over_batch_size", name="squared_hinge"):
        super().__init__(squared_hinge, reduction=reduction, name=name)

    def get_config(self):
        return Loss.get_config(self)


@keras_core_export("keras_core.losses.CategoricalHinge")
class CategoricalHinge(LossFunctionWrapper):
    """Computes the categorical hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = maximum(neg - pos + 1, 0)
    ```

    where `neg=maximum((1-y_true)*y_pred)` and `pos=sum(y_true*y_pred)`

    Args:
        reduction: Type of reduction to apply to loss. For almost all cases
            this defaults to `"sum_over_batch_size"`. Options are `"sum"`,
            `"sum_over_batch_size"` or `None`.
        name: Optional name for the instance. Defaults to
            `"categorical_hinge"`
    """

    def __init__(
        self, reduction="sum_over_batch_size", name="categorical_hinge"
    ):
        super().__init__(categorical_hinge, reduction=reduction, name=name)

    def get_config(self):
        return Loss.get_config(self)


def convert_binary_labels_to_hinge(y_true):
    """Converts binary labels into -1/1 for hinge loss/metric calculation."""
    are_zeros = ops.equal(y_true, 0)
    are_ones = ops.equal(y_true, 1)
    is_binary = ops.all((ops.logical_or(are_zeros, are_ones)))

    def _convert_binary_labels():
        # Convert the binary labels to -1 or 1.
        return 2.0 * y_true - 1.0

    def _return_labels_unconverted():
        # Returns the labels unchanged if they are non-binary
        return y_true

    updated_y_true = ops.cond(
        is_binary, _convert_binary_labels, _return_labels_unconverted
    )
    return updated_y_true


@keras_core_export(
    [
        "keras_core.metrics.hinge",
        "keras_core.losses.hinge",
    ]
)
def hinge(y_true, y_pred):
    """Computes the hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(maximum(1 - y_true * y_pred, 0), axis=-1)
    ```

    Standalone usage:

    >>> y_true = np.random.choice([-1, 1], size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.hinge(y_true, y_pred)

    Args:
        y_true: The ground truth values. `y_true` values are expected to be -1
            or 1. If binary (0 or 1) labels are provided they will be converted
            to -1 or 1 with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Hinge loss values with shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, dtype=y_pred.dtype)
    y_true = ops.convert_to_tensor(y_true)
    y_true = convert_binary_labels_to_hinge(y_true)
    return ops.mean(ops.maximum(1.0 - y_true * y_pred, 0.0), axis=-1)


@keras_core_export(
    [
        "keras_core.metrics.squared_hinge",
        "keras_core.losses.squared_hinge",
    ]
)
def squared_hinge(y_true, y_pred):
    """Computes the squared hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)
    ```

    Standalone usage:

    >>> y_true = np.random.choice([-1, 1], size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.squared_hinge(y_true, y_pred)

    Args:
        y_true: The ground truth values. `y_true` values are expected to be -1
            or 1. If binary (0 or 1) labels are provided we will convert them
            to -1 or 1 with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Squared hinge loss values with shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)
    y_true = convert_binary_labels_to_hinge(y_true)
    return ops.mean(
        ops.square(ops.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1
    )


@keras_core_export(
    [
        "keras_core.metrics.categorical_hinge",
        "keras_core.losses.categorical_hinge",
    ]
)
def categorical_hinge(y_true, y_pred):
    """Computes the categorical hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = maximum(neg - pos + 1, 0)
    ```

    where `neg=maximum((1-y_true)*y_pred)` and `pos=sum(y_true*y_pred)`

    Standalone usage:

    >>> y_true = np.random.randint(0, 3, size=(2,))
    >>> y_true = np.eye(np.max(y_true) + 1)[y_true]
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras_core.losses.categorical_hinge(y_true, y_pred)

    Args:
        y_true: The ground truth values. `y_true` values are expected to be
            either `{-1, +1}` or `{0, 1}` (i.e. a one-hot-encoded tensor) with
            shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Categorical hinge loss values with shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)
    pos = ops.sum(y_true * y_pred, axis=-1)
    neg = ops.max((1.0 - y_true) * y_pred, axis=-1)
    zero = ops.cast(0.0, y_pred.dtype)
    return ops.maximum(neg - pos + 1.0, zero)


@keras_core_export(
    [
        "keras_core.metrics.mean_squared_error",
        "keras_core.losses.mean_squared_error",
    ]
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


@keras_core_export(
    [
        "keras_core.metrics.mean_absolute_error",
        "keras_core.losses.mean_absolute_error",
    ]
)
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


@keras_core_export(
    [
        "keras_core.metrics.mean_absolute_percentage_error",
        "keras_core.losses.mean_absolute_percentage_error",
    ]
)
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


@keras_core_export(
    [
        "keras_core.metrics.mean_squared_logarithmic_error",
        "keras_core.losses.mean_squared_logarithmic_error",
    ]
)
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


@keras_core_export("keras_core.losses.cosine_similarity")
def cosine_similarity(y_true, y_pred, axis=-1):
    """Computes the cosine similarity between labels and predictions.

    Formula:
    ```python
    loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
    ```

    Note that it is a number between -1 and 1. When it is a negative number
    between -1 and 0, 0 indicates orthogonality and values closer to -1
    indicate greater similarity. This makes it usable as a loss function in a
    setting where you try to maximize the proximity between predictions and
    targets. If either `y_true` or `y_pred` is a zero vector, cosine
    similarity will be 0 regardless of the proximity between predictions
    and targets.

    Standalone usage:
    >>> y_true = [[0., 1.], [1., 1.], [1., 1.]]
    >>> y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
    >>> loss = keras_core.losses.cosine_similarity(y_true, y_pred, axis=-1)
    [-0., -0.99999994, 0.99999994]

    Args:
        y_true: Tensor of true targets.
        y_pred: Tensor of predicted targets.
        axis: Axis along which to determine similarity. Defaults to -1.

    Returns:
        Cosine similarity tensor.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
    y_pred = normalize(y_pred, axis=axis)
    y_true = normalize(y_true, axis=axis)
    return -ops.sum(y_true * y_pred, axis=axis)
