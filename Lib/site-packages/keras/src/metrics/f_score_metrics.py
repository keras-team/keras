from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.metrics.metric import Metric


@keras_export("keras.metrics.FBetaScore")
class FBetaScore(Metric):
    """Computes F-Beta score.

    Formula:

    ```python
    b2 = beta ** 2
    f_beta_score = (1 + b2) * (precision * recall) / (precision * b2 + recall)
    ```
    This is the weighted harmonic mean of precision and recall.
    Its output range is `[0, 1]`. It works for both multi-class
    and multi-label classification.

    Args:
        average: Type of averaging to be performed across per-class results
            in the multi-class case.
            Acceptable values are `None`, `"micro"`, `"macro"` and
            `"weighted"`. Defaults to `None`.
            If `None`, no averaging is performed and `result()` will return
            the score for each class.
            If `"micro"`, compute metrics globally by counting the total
            true positives, false negatives and false positives.
            If `"macro"`, compute metrics for each label,
            and return their unweighted mean.
            This does not take label imbalance into account.
            If `"weighted"`, compute metrics for each label,
            and return their average weighted by support
            (the number of true instances for each label).
            This alters `"macro"` to account for label imbalance.
            It can result in an score that is not between precision and recall.
        beta: Determines the weight of given to recall
            in the harmonic mean between precision and recall (see pseudocode
            equation above). Defaults to `1`.
        threshold: Elements of `y_pred` greater than `threshold` are
            converted to be 1, and the rest 0. If `threshold` is
            `None`, the argmax of `y_pred` is converted to 1, and the rest to 0.
        name: Optional. String name of the metric instance.
        dtype: Optional. Data type of the metric result.

    Returns:
        F-Beta Score: float.

    Example:

    >>> metric = keras.metrics.FBetaScore(beta=2.0, threshold=0.5)
    >>> y_true = np.array([[1, 1, 1],
    ...                    [1, 0, 0],
    ...                    [1, 1, 0]], np.int32)
    >>> y_pred = np.array([[0.2, 0.6, 0.7],
    ...                    [0.2, 0.6, 0.6],
    ...                    [0.6, 0.8, 0.0]], np.float32)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result
    [0.3846154 , 0.90909094, 0.8333334 ]
    """

    def __init__(
        self,
        average=None,
        beta=1.0,
        threshold=None,
        name="fbeta_score",
        dtype=None,
    ):
        super().__init__(name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(
                "Invalid `average` argument value. Expected one of: "
                "{None, 'micro', 'macro', 'weighted'}. "
                f"Received: average={average}"
            )

        if not isinstance(beta, float):
            raise ValueError(
                "Invalid `beta` argument value. "
                "It should be a Python float. "
                f"Received: beta={beta} of type '{type(beta)}'"
            )
        if beta <= 0.0:
            raise ValueError(
                "Invalid `beta` argument value. "
                "It should be > 0. "
                f"Received: beta={beta}"
            )

        if threshold is not None:
            if not isinstance(threshold, float):
                raise ValueError(
                    "Invalid `threshold` argument value. "
                    "It should be a Python float. "
                    f"Received: threshold={threshold} "
                    f"of type '{type(threshold)}'"
                )
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError(
                    "Invalid `threshold` argument value. "
                    "It should verify 0 < threshold <= 1. "
                    f"Received: threshold={threshold}"
                )

        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.axis = None
        self._built = False

        if self.average != "micro":
            self.axis = 0

    def _build(self, y_true_shape, y_pred_shape):
        if len(y_pred_shape) != 2 or len(y_true_shape) != 2:
            raise ValueError(
                "FBetaScore expects 2D inputs with shape "
                "(batch_size, output_dim). Received input "
                f"shapes: y_pred.shape={y_pred_shape} and "
                f"y_true.shape={y_true_shape}."
            )
        if y_pred_shape[-1] is None or y_true_shape[-1] is None:
            raise ValueError(
                "FBetaScore expects 2D inputs with shape "
                "(batch_size, output_dim), with output_dim fully "
                "defined (not None). Received input "
                f"shapes: y_pred.shape={y_pred_shape} and "
                f"y_true.shape={y_true_shape}."
            )
        num_classes = y_pred_shape[-1]
        if self.average != "micro":
            init_shape = (num_classes,)
        else:
            init_shape = ()

        def _add_zeros_variable(name):
            return self.add_variable(
                name=name,
                shape=init_shape,
                initializer=initializers.Zeros(),
                dtype=self.dtype,
            )

        self.true_positives = _add_zeros_variable("true_positives")
        self.false_positives = _add_zeros_variable("false_positives")
        self.false_negatives = _add_zeros_variable("false_negatives")
        self.intermediate_weights = _add_zeros_variable("intermediate_weights")
        self._built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = ops.convert_to_tensor(y_pred, dtype=self.dtype)
        if not self._built:
            self._build(y_true.shape, y_pred.shape)

        if self.threshold is None:
            threshold = ops.max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = ops.logical_and(
                y_pred >= threshold, ops.abs(y_pred) > 1e-9
            )
        else:
            y_pred = y_pred > self.threshold

        y_pred = ops.cast(y_pred, dtype=self.dtype)
        y_true = ops.cast(y_true, dtype=self.dtype)
        if sample_weight is not None:
            sample_weight = ops.convert_to_tensor(
                sample_weight, dtype=self.dtype
            )

        def _weighted_sum(val, sample_weight):
            if sample_weight is not None:
                val = ops.multiply(val, ops.expand_dims(sample_weight, 1))
            return ops.sum(val, axis=self.axis)

        self.true_positives.assign(
            self.true_positives + _weighted_sum(y_pred * y_true, sample_weight)
        )
        self.false_positives.assign(
            self.false_positives
            + _weighted_sum(y_pred * (1 - y_true), sample_weight)
        )
        self.false_negatives.assign(
            self.false_negatives
            + _weighted_sum((1 - y_pred) * y_true, sample_weight)
        )
        self.intermediate_weights.assign(
            self.intermediate_weights + _weighted_sum(y_true, sample_weight)
        )

    def result(self):
        precision = ops.divide(
            self.true_positives,
            self.true_positives + self.false_positives + backend.epsilon(),
        )
        recall = ops.divide(
            self.true_positives,
            self.true_positives + self.false_negatives + backend.epsilon(),
        )

        precision = ops.convert_to_tensor(precision, dtype=self.dtype)
        recall = ops.convert_to_tensor(recall, dtype=self.dtype)

        mul_value = precision * recall
        add_value = ((self.beta**2) * precision) + recall
        mean = ops.divide(mul_value, add_value + backend.epsilon())
        f1_score = mean * (1 + (self.beta**2))

        if self.average == "weighted":
            weights = ops.divide(
                self.intermediate_weights,
                ops.sum(self.intermediate_weights) + backend.epsilon(),
            )
            f1_score = ops.sum(f1_score * weights)

        elif self.average is not None:  # [micro, macro]
            f1_score = ops.mean(f1_score)

        return f1_score

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "name": self.name,
            "dtype": self.dtype,
            "average": self.average,
            "beta": self.beta,
            "threshold": self.threshold,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        for v in self.variables:
            v.assign(ops.zeros(v.shape, dtype=v.dtype))


@keras_export("keras.metrics.F1Score")
class F1Score(FBetaScore):
    r"""Computes F-1 Score.

    Formula:

    ```python
    f1_score = 2 * (precision * recall) / (precision + recall)
    ```
    This is the harmonic mean of precision and recall.
    Its output range is `[0, 1]`. It works for both multi-class
    and multi-label classification.

    Args:
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `"micro"`, `"macro"`
            and `"weighted"`. Defaults to `None`.
            If `None`, no averaging is performed and `result()` will return
            the score for each class.
            If `"micro"`, compute metrics globally by counting the total
            true positives, false negatives and false positives.
            If `"macro"`, compute metrics for each label,
            and return their unweighted mean.
            This does not take label imbalance into account.
            If `"weighted"`, compute metrics for each label,
            and return their average weighted by support
            (the number of true instances for each label).
            This alters `"macro"` to account for label imbalance.
            It can result in an score that is not between precision and recall.
        threshold: Elements of `y_pred` greater than `threshold` are
            converted to be 1, and the rest 0. If `threshold` is
            `None`, the argmax of `y_pred` is converted to 1, and the rest to 0.
        name: Optional. String name of the metric instance.
        dtype: Optional. Data type of the metric result.

    Returns:
        F-1 Score: float.

    Example:

    >>> metric = keras.metrics.F1Score(threshold=0.5)
    >>> y_true = np.array([[1, 1, 1],
    ...                    [1, 0, 0],
    ...                    [1, 1, 0]], np.int32)
    >>> y_pred = np.array([[0.2, 0.6, 0.7],
    ...                    [0.2, 0.6, 0.6],
    ...                    [0.6, 0.8, 0.0]], np.float32)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    array([0.5      , 0.8      , 0.6666667], dtype=float32)
    """

    def __init__(
        self,
        average=None,
        threshold=None,
        name="f1_score",
        dtype=None,
    ):
        super().__init__(
            average=average,
            beta=1.0,
            threshold=threshold,
            name=name,
            dtype=dtype,
        )

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config
