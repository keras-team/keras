from keras.src.api_export import keras_export
from keras.src.losses.losses import binary_crossentropy
from keras.src.losses.losses import categorical_crossentropy
from keras.src.losses.losses import kl_divergence
from keras.src.losses.losses import poisson
from keras.src.losses.losses import sparse_categorical_crossentropy
from keras.src.metrics import reduction_metrics


@keras_export("keras.metrics.KLDivergence")
class KLDivergence(reduction_metrics.MeanMetricWrapper):
    """Computes Kullback-Leibler divergence metric between `y_true` and
    `y_pred`.

    Formula:

    ```python
    metric = y_true * log(y_true / y_pred)
    ```

    `y_true` and `y_pred` are expected to be probability
    distributions, with values between 0 and 1. They will get
    clipped to the `[0, 1]` range.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:

    >>> m = keras.metrics.KLDivergence()
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
                  metrics=[keras.metrics.KLDivergence()])
    ```
    """

    def __init__(self, name="kl_divergence", dtype=None):
        super().__init__(fn=kl_divergence, name=name, dtype=dtype)

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_export("keras.metrics.Poisson")
class Poisson(reduction_metrics.MeanMetricWrapper):
    """Computes the Poisson metric between `y_true` and `y_pred`.

    Formula:

    ```python
    metric = y_pred - y_true * log(y_pred)
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:

    >>> m = keras.metrics.Poisson()
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
                  metrics=[keras.metrics.Poisson()])
    ```
    """

    def __init__(self, name="poisson", dtype=None):
        super().__init__(fn=poisson, name=name, dtype=dtype)

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_export("keras.metrics.BinaryCrossentropy")
class BinaryCrossentropy(reduction_metrics.MeanMetricWrapper):
    """Computes the crossentropy metric between the labels and predictions.

    This is the crossentropy metric class to be used when there are only two
    label classes (0 and 1).

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        from_logits: (Optional) Whether output is expected
            to be a logits tensor. By default, we consider
            that output encodes a probability distribution.
        label_smoothing: (Optional) Float in `[0, 1]`.
            When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed.
            e.g. `label_smoothing=0.2` means that we will use
            a value of 0.1 for label "0" and 0.9 for label "1".

    Examples:

    >>> m = keras.metrics.BinaryCrossentropy()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result()
    0.81492424

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    0.9162905

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.BinaryCrossentropy()])
    ```
    """

    def __init__(
        self,
        name="binary_crossentropy",
        dtype=None,
        from_logits=False,
        label_smoothing=0,
    ):
        super().__init__(
            binary_crossentropy,
            name,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
        )
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        # Metric should be minimized during optimization.
        self._direction = "down"

    def get_config(self):
        return {
            "name": self.name,
            "dtype": self.dtype,
            "from_logits": self.from_logits,
            "label_smoothing": self.label_smoothing,
        }


@keras_export("keras.metrics.CategoricalCrossentropy")
class CategoricalCrossentropy(reduction_metrics.MeanMetricWrapper):
    """Computes the crossentropy metric between the labels and predictions.

    This is the crossentropy metric class to be used when there are multiple
    label classes (2 or more). It assumes that labels are one-hot encoded,
    e.g., when labels values are `[2, 0, 1]`, then
    `y_true` is `[[0, 0, 1], [1, 0, 0], [0, 1, 0]]`.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        from_logits: (Optional) Whether output is expected to be
            a logits tensor. By default, we consider that output
            encodes a probability distribution.
        label_smoothing: (Optional) Float in `[0, 1]`.
            When > 0, label values are smoothed, meaning the confidence
            on label values are relaxed. e.g. `label_smoothing=0.2` means
            that we will use a value of 0.1 for label
            "0" and 0.9 for label "1".
        axis: (Optional) Defaults to `-1`.
            The dimension along which entropy is computed.

    Examples:

    >>> # EPSILON = 1e-7, y = y_true, y` = y_pred
    >>> # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    >>> # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
    >>> # xent = -sum(y * log(y'), axis = -1)
    >>> #      = -((log 0.95), (log 0.1))
    >>> #      = [0.051, 2.302]
    >>> # Reduced xent = (0.051 + 2.302) / 2
    >>> m = keras.metrics.CategoricalCrossentropy()
    >>> m.update_state([[0, 1, 0], [0, 0, 1]],
    ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    >>> m.result()
    1.1769392

    >>> m.reset_state()
    >>> m.update_state([[0, 1, 0], [0, 0, 1]],
    ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
    ...                sample_weight=np.array([0.3, 0.7]))
    >>> m.result()
    1.6271976

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.CategoricalCrossentropy()])
    ```
    """

    def __init__(
        self,
        name="categorical_crossentropy",
        dtype=None,
        from_logits=False,
        label_smoothing=0,
        axis=-1,
    ):
        super().__init__(
            categorical_crossentropy,
            name,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis
        # Metric should be minimized during optimization.
        self._direction = "down"

    def get_config(self):
        return {
            "name": self.name,
            "dtype": self.dtype,
            "from_logits": self.from_logits,
            "label_smoothing": self.label_smoothing,
            "axis": self.axis,
        }


@keras_export("keras.metrics.SparseCategoricalCrossentropy")
class SparseCategoricalCrossentropy(reduction_metrics.MeanMetricWrapper):
    """Computes the crossentropy metric between the labels and predictions.

    Use this crossentropy metric when there are two or more label classes.
    It expects labels to be provided as integers. If you want to provide labels
    that are one-hot encoded, please use the `CategoricalCrossentropy`
    metric instead.

    There should be `num_classes` floating point values per feature for `y_pred`
    and a single floating point value per feature for `y_true`.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        from_logits: (Optional) Whether output is expected
            to be a logits tensor. By default, we consider that output
            encodes a probability distribution.
        axis: (Optional) Defaults to `-1`.
            The dimension along which entropy is computed.

    Examples:

    >>> # y_true = one_hot(y_true) = [[0, 1, 0], [0, 0, 1]]
    >>> # logits = log(y_pred)
    >>> # softmax = exp(logits) / sum(exp(logits), axis=-1)
    >>> # softmax = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
    >>> # xent = -sum(y * log(softmax), 1)
    >>> # log(softmax) = [[-2.9957, -0.0513, -16.1181],
    >>> #                [-2.3026, -0.2231, -2.3026]]
    >>> # y_true * log(softmax) = [[0, -0.0513, 0], [0, 0, -2.3026]]
    >>> # xent = [0.0513, 2.3026]
    >>> # Reduced xent = (0.0513 + 2.3026) / 2
    >>> m = keras.metrics.SparseCategoricalCrossentropy()
    >>> m.update_state([1, 2],
    ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    >>> m.result()
    1.1769392

    >>> m.reset_state()
    >>> m.update_state([1, 2],
    ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
    ...                sample_weight=np.array([0.3, 0.7]))
    >>> m.result()
    1.6271976

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.SparseCategoricalCrossentropy()])
    ```
    """

    def __init__(
        self,
        name="sparse_categorical_crossentropy",
        dtype=None,
        from_logits=False,
        axis=-1,
    ):
        super().__init__(
            sparse_categorical_crossentropy,
            name=name,
            dtype=dtype,
            from_logits=from_logits,
            axis=axis,
        )
        self.from_logits = from_logits
        self.axis = axis
        # Metric should be minimized during optimization.
        self._direction = "down"

    def get_config(self):
        return {
            "name": self.name,
            "dtype": self.dtype,
            "from_logits": self.from_logits,
            "axis": self.axis,
        }
