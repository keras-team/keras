from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.metrics import reduction_metrics


def accuracy(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    return ops.cast(ops.equal(y_true, y_pred), dtype=backend.floatx())


@keras_export("keras.metrics.Accuracy")
class Accuracy(reduction_metrics.MeanMetricWrapper):
    """Calculates how often predictions equal labels.

    This metric creates two local variables, `total` and `count` that are used
    to compute the frequency with which `y_pred` matches `y_true`. This
    frequency is ultimately returned as `binary accuracy`: an idempotent
    operation that simply divides `total` by `count`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:

    >>> m = keras.metrics.Accuracy()
    >>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]])
    >>> m.result()
    0.75

    >>> m.reset_state()
    >>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]],
    ...                sample_weight=[1, 1, 0, 0])
    >>> m.result()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.Accuracy()])
    ```
    """

    def __init__(self, name="accuracy", dtype=None):
        super().__init__(fn=accuracy, name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_export("keras.metrics.binary_accuracy")
def binary_accuracy(y_true, y_pred, threshold=0.5):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    threshold = ops.cast(threshold, y_pred.dtype)
    y_pred = ops.cast(y_pred > threshold, y_true.dtype)
    return ops.cast(ops.equal(y_true, y_pred), dtype=backend.floatx())


@keras_export("keras.metrics.BinaryAccuracy")
class BinaryAccuracy(reduction_metrics.MeanMetricWrapper):
    """Calculates how often predictions match binary labels.

    This metric creates two local variables, `total` and `count` that are used
    to compute the frequency with which `y_pred` matches `y_true`. This
    frequency is ultimately returned as `binary accuracy`: an idempotent
    operation that simply divides `total` by `count`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        threshold: (Optional) Float representing the threshold for deciding
        whether prediction values are 1 or 0.

    Example:

    >>> m = keras.metrics.BinaryAccuracy()
    >>> m.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]])
    >>> m.result()
    0.75

    >>> m.reset_state()
    >>> m.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]],
    ...                sample_weight=[1, 0, 0, 1])
    >>> m.result()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.BinaryAccuracy()])
    ```
    """

    def __init__(self, name="binary_accuracy", dtype=None, threshold=0.5):
        if threshold is not None and (threshold <= 0 or threshold >= 1):
            raise ValueError(
                "Invalid value for argument `threshold`. "
                "Expected a value in interval (0, 1). "
                f"Received: threshold={threshold}"
            )
        super().__init__(
            fn=binary_accuracy, name=name, dtype=dtype, threshold=threshold
        )
        self.threshold = threshold
        # Metric should be maximized during optimization.
        self._direction = "up"

    def get_config(self):
        return {
            "name": self.name,
            "dtype": self.dtype,
            "threshold": self.threshold,
        }


@keras_export("keras.metrics.categorical_accuracy")
def categorical_accuracy(y_true, y_pred):
    y_true = ops.argmax(y_true, axis=-1)

    reshape_matches = False
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)

    y_true_org_shape = ops.shape(y_true)
    y_pred_rank = len(y_pred.shape)
    y_true_rank = len(y_true.shape)

    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (
        (y_true_rank is not None)
        and (y_pred_rank is not None)
        and (len(y_true.shape) == len(y_pred.shape))
    ):
        y_true = ops.squeeze(y_true, -1)
        reshape_matches = True
    y_pred = ops.argmax(y_pred, axis=-1)

    # If the predicted output and actual output types don't match, force cast
    # them to match.
    if y_pred.dtype != y_true.dtype:
        y_pred = ops.cast(y_pred, dtype=y_true.dtype)
    matches = ops.cast(ops.equal(y_true, y_pred), backend.floatx())
    if reshape_matches:
        matches = ops.reshape(matches, y_true_org_shape)
    return matches


@keras_export("keras.metrics.CategoricalAccuracy")
class CategoricalAccuracy(reduction_metrics.MeanMetricWrapper):
    """Calculates how often predictions match one-hot labels.

    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    This metric creates two local variables, `total` and `count` that are used
    to compute the frequency with which `y_pred` matches `y_true`. This
    frequency is ultimately returned as `categorical accuracy`: an idempotent
    operation that simply divides `total` by `count`.

    `y_pred` and `y_true` should be passed in as vectors of probabilities,
    rather than as labels. If necessary, use `ops.one_hot` to expand `y_true` as
    a vector.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

    >>> m = keras.metrics.CategoricalAccuracy()
    >>> m.update_state([[0, 0, 1], [0, 1, 0]], [[0.1, 0.9, 0.8],
    ...                 [0.05, 0.95, 0]])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([[0, 0, 1], [0, 1, 0]], [[0.1, 0.9, 0.8],
    ...                 [0.05, 0.95, 0]],
    ...                sample_weight=[0.7, 0.3])
    >>> m.result()
    0.3

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=[keras.metrics.CategoricalAccuracy()])
    ```
    """

    def __init__(self, name="categorical_accuracy", dtype=None):
        super().__init__(fn=categorical_accuracy, name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_export("keras.metrics.sparse_categorical_accuracy")
def sparse_categorical_accuracy(y_true, y_pred):
    reshape_matches = False
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true_org_shape = ops.shape(y_true)
    y_pred_rank = len(y_pred.shape)
    y_true_rank = len(y_true.shape)

    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (
        (y_true_rank is not None)
        and (y_pred_rank is not None)
        and (len(y_true.shape) == len(y_pred.shape))
        and ops.shape(y_true)[-1] == 1
    ):
        y_true = ops.squeeze(y_true, -1)
        reshape_matches = True
    y_pred = ops.argmax(y_pred, axis=-1)

    # If the predicted output and actual output types don't match, force cast
    # them to match.
    if y_pred.dtype != y_true.dtype:
        y_pred = ops.cast(y_pred, y_true.dtype)
    matches = ops.cast(ops.equal(y_true, y_pred), backend.floatx())
    if reshape_matches:
        matches = ops.reshape(matches, y_true_org_shape)
    # if shape is (num_samples, 1) squeeze
    if len(matches.shape) > 1 and matches.shape[-1] == 1:
        matches = ops.squeeze(matches, -1)
    return matches


@keras_export("keras.metrics.SparseCategoricalAccuracy")
class SparseCategoricalAccuracy(reduction_metrics.MeanMetricWrapper):
    """Calculates how often predictions match integer labels.

    ```python
    acc = np.dot(sample_weight, np.equal(y_true, np.argmax(y_pred, axis=1))
    ```

    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    This metric creates two local variables, `total` and `count` that are used
    to compute the frequency with which `y_pred` matches `y_true`. This
    frequency is ultimately returned as `sparse categorical accuracy`: an
    idempotent operation that simply divides `total` by `count`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

    >>> m = keras.metrics.SparseCategoricalAccuracy()
    >>> m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]],
    ...                sample_weight=[0.7, 0.3])
    >>> m.result()
    0.3

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    ```
    """

    def __init__(self, name="sparse_categorical_accuracy", dtype=None):
        super().__init__(fn=sparse_categorical_accuracy, name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_export("keras.metrics.top_k_categorical_accuracy")
def top_k_categorical_accuracy(y_true, y_pred, k=5):
    reshape_matches = False
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true = ops.argmax(y_true, axis=-1)
    y_true_rank = len(y_true.shape)
    y_pred_rank = len(y_pred.shape)
    y_true_org_shape = ops.shape(y_true)

    # Flatten y_pred to (batch_size, num_samples) and y_true to (num_samples,)
    if (y_true_rank is not None) and (y_pred_rank is not None):
        if y_pred_rank > 2:
            y_pred = ops.reshape(y_pred, [-1, y_pred.shape[-1]])
        if y_true_rank > 1:
            reshape_matches = True
            y_true = ops.reshape(y_true, [-1])

    matches = ops.cast(
        ops.in_top_k(ops.cast(y_true, "int32"), y_pred, k=k),
        dtype=backend.floatx(),
    )

    # returned matches is expected to have same shape as y_true input
    if reshape_matches:
        matches = ops.reshape(matches, y_true_org_shape)

    return matches


@keras_export("keras.metrics.TopKCategoricalAccuracy")
class TopKCategoricalAccuracy(reduction_metrics.MeanMetricWrapper):
    """Computes how often targets are in the top `K` predictions.

    Args:
        k: (Optional) Number of top elements to look at for computing accuracy.
            Defaults to `5`.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

    >>> m = keras.metrics.TopKCategoricalAccuracy(k=1)
    >>> m.update_state([[0, 0, 1], [0, 1, 0]],
    ...                [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([[0, 0, 1], [0, 1, 0]],
    ...                [[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
    ...                sample_weight=[0.7, 0.3])
    >>> m.result()
    0.3

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=[keras.metrics.TopKCategoricalAccuracy()])
    ```
    """

    def __init__(self, k=5, name="top_k_categorical_accuracy", dtype=None):
        super().__init__(
            fn=top_k_categorical_accuracy,
            name=name,
            dtype=dtype,
            k=k,
        )
        self.k = k
        # Metric should be maximized during optimization.
        self._direction = "up"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype, "k": self.k}


@keras_export("keras.metrics.sparse_top_k_categorical_accuracy")
def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    reshape_matches = False
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true_rank = len(y_true.shape)
    y_pred_rank = len(y_pred.shape)
    y_true_org_shape = ops.shape(y_true)

    # Flatten y_pred to (batch_size, num_samples) and y_true to (num_samples,)
    if (y_true_rank is not None) and (y_pred_rank is not None):
        if y_pred_rank > 2:
            y_pred = ops.reshape(y_pred, [-1, y_pred.shape[-1]])
        if y_true_rank > 1:
            reshape_matches = True
            y_true = ops.reshape(y_true, [-1])

    matches = ops.cast(
        ops.in_top_k(ops.cast(y_true, "int32"), y_pred, k=k),
        dtype=backend.floatx(),
    )

    # returned matches is expected to have same shape as y_true input
    if reshape_matches:
        matches = ops.reshape(matches, y_true_org_shape)

    return matches


@keras_export("keras.metrics.SparseTopKCategoricalAccuracy")
class SparseTopKCategoricalAccuracy(reduction_metrics.MeanMetricWrapper):
    """Computes how often integer targets are in the top `K` predictions.

    Args:
        k: (Optional) Number of top elements to look at for computing accuracy.
            Defaults to `5`.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

    >>> m = keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    >>> m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
    ...                sample_weight=[0.7, 0.3])
    >>> m.result()
    0.3

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=[keras.metrics.SparseTopKCategoricalAccuracy()])
    ```
    """

    def __init__(
        self, k=5, name="sparse_top_k_categorical_accuracy", dtype=None
    ):
        super().__init__(
            fn=sparse_top_k_categorical_accuracy,
            name=name,
            dtype=dtype,
            k=k,
        )
        self.k = k
        # Metric should be maximized during optimization.
        self._direction = "up"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype, "k": self.k}
