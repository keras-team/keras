from keras_core import backend
from keras_core import initializers
from keras_core import operations as ops
from keras_core.api_export import keras_core_export
from keras_core.metrics import metrics_utils
from keras_core.metrics.metric import Metric
from keras_core.utils.python_utils import to_list


class _ConfusionMatrixConditionCount(Metric):
    """Calculates the number of the given confusion matrix condition.

    Args:
        confusion_matrix_cond: One of `metrics_utils.ConfusionMatrix`
            conditions.
        thresholds: (Optional) Defaults to 0.5. A float value or a python list /
            tuple of float threshold values in `[0, 1]`. A threshold is compared
            with prediction values to determine the truth value of predictions
            (i.e., above the threshold is `True`, below is `False`). One metric
            value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(
        self, confusion_matrix_cond, thresholds=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        self._confusion_matrix_cond = confusion_matrix_cond
        self.init_thresholds = thresholds
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=0.5
        )
        self._thresholds_distributed_evenly = (
            metrics_utils.is_evenly_distributed_thresholds(self.thresholds)
        )
        self.accumulator = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=initializers.Zeros(),
            name="accumulator",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the metric statistics.

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Defaults to 1.
                Can be a tensor whose rank is either 0, or the same rank as
                `y_true`, and must be broadcastable to `y_true`.
        """
        return metrics_utils.update_confusion_matrix_variables(
            {self._confusion_matrix_cond: self.accumulator},
            y_true,
            y_pred,
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            sample_weight=sample_weight,
        )

    def result(self):
        if len(self.thresholds) == 1:
            result = self.accumulator[0]
        else:
            result = self.accumulator
        return backend.convert_to_tensor(result)

    def get_config(self):
        config = {"thresholds": self.init_thresholds}
        base_config = super().get_config()
        return {**base_config, **config}


@keras_core_export("keras_core.metrics.FalsePositives")
class FalsePositives(_ConfusionMatrixConditionCount):
    """Calculates the number of false positives.

    If `sample_weight` is given, calculates the sum of the weights of
    false positives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of false positives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        thresholds: (Optional) Defaults to 0.5. A float value, or a Python
            list/tuple of float threshold values in `[0, 1]`. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `True`, below is `False`).
            If used with a loss function that sets `from_logits=True` (i.e. no
            sigmoid applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.FalsePositives()
    >>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1])
    >>> m.result()
    2.0

    >>> m.reset_state()
    >>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0
    """

    def __init__(self, thresholds=None, name=None, dtype=None):
        super().__init__(
            confusion_matrix_cond=metrics_utils.ConfusionMatrix.FALSE_POSITIVES,
            thresholds=thresholds,
            name=name,
            dtype=dtype,
        )


@keras_core_export("keras_core.metrics.FalseNegatives")
class FalseNegatives(_ConfusionMatrixConditionCount):
    """Calculates the number of false negatives.

    If `sample_weight` is given, calculates the sum of the weights of
    false negatives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of false negatives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        thresholds: (Optional) Defaults to 0.5. A float value, or a Python
            list/tuple of float threshold values in `[0, 1]`. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `True`, below is `False`).
            If used with a loss function that sets `from_logits=True` (i.e. no
            sigmoid applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.FalseNegatives()
    >>> m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
    >>> m.result()
    2.0

    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [0, 1, 0, 0], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0
    """

    def __init__(self, thresholds=None, name=None, dtype=None):
        super().__init__(
            confusion_matrix_cond=metrics_utils.ConfusionMatrix.FALSE_NEGATIVES,
            thresholds=thresholds,
            name=name,
            dtype=dtype,
        )


@keras_core_export("keras_core.metrics.TrueNegatives")
class TrueNegatives(_ConfusionMatrixConditionCount):
    """Calculates the number of true negatives.

    If `sample_weight` is given, calculates the sum of the weights of
    true negatives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of true negatives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        thresholds: (Optional) Defaults to 0.5. A float value, or a Python
            list/tuple of float threshold values in `[0, 1]`. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `True`, below is `False`).
            If used with a loss function that sets `from_logits=True` (i.e. no
            sigmoid applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.TrueNegatives()
    >>> m.update_state([0, 1, 0, 0], [1, 1, 0, 0])
    >>> m.result()
    2.0

    >>> m.reset_state()
    >>> m.update_state([0, 1, 0, 0], [1, 1, 0, 0], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0
    """

    def __init__(self, thresholds=None, name=None, dtype=None):
        super().__init__(
            confusion_matrix_cond=metrics_utils.ConfusionMatrix.TRUE_NEGATIVES,
            thresholds=thresholds,
            name=name,
            dtype=dtype,
        )


@keras_core_export("keras_core.metrics.TruePositives")
class TruePositives(_ConfusionMatrixConditionCount):
    """Calculates the number of true positives.

    If `sample_weight` is given, calculates the sum of the weights of
    true positives. This metric creates one local variable, `true_positives`
    that is used to keep track of the number of true positives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        thresholds: (Optional) Defaults to 0.5. A float value, or a Python
            list/tuple of float threshold values in `[0, 1]`. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `True`, below is `False`).
            If used with a loss function that sets `from_logits=True` (i.e. no
            sigmoid applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.TruePositives()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
    >>> m.result()
    2.0

    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0
    """

    def __init__(self, thresholds=None, name=None, dtype=None):
        super().__init__(
            confusion_matrix_cond=metrics_utils.ConfusionMatrix.TRUE_POSITIVES,
            thresholds=thresholds,
            name=name,
            dtype=dtype,
        )


@keras_core_export("keras_core.metrics.Precision")
class Precision(Metric):
    """Computes the precision of the predictions with respect to the labels.

    The metric creates two local variables, `true_positives` and
    `false_positives` that are used to compute the precision. This value is
    ultimately returned as `precision`, an idempotent operation that simply
    divides `true_positives` by the sum of `true_positives` and
    `false_positives`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `top_k` is set, we'll calculate precision as how often on average a class
    among the top-k classes with the highest predicted values of a batch entry
    is correct and can be found in the label for that entry.

    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold and/or in
    the top-k highest predictions, and computing the fraction of them for which
    `class_id` is indeed a correct label.

    Args:
        thresholds: (Optional) A float value, or a Python list/tuple of float
            threshold values in `[0, 1]`. A threshold is compared with
            prediction values to determine the truth value of predictions (i.e.,
            above the threshold is `True`, below is `False`). If used with a
            loss function that sets `from_logits=True` (i.e. no sigmoid applied
            to predictions), `thresholds` should be set to 0. One metric value
            is generated for each threshold value. If neither `thresholds` nor
            `top_k` are set, the default is to calculate precision with
            `thresholds=0.5`.
        top_k: (Optional) Unset by default. An int value specifying the top-k
            predictions to consider when calculating precision.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.Precision()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
    >>> m.result()
    0.6666667

    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0

    >>> # With top_k=2, it will calculate precision over y_true[:2]
    >>> # and y_pred[:2]
    >>> m = keras_core.metrics.Precision(top_k=2)
    >>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
    >>> m.result()
    0.0

    >>> # With top_k=4, it will calculate precision over y_true[:4]
    >>> # and y_pred[:4]
    >>> m = keras_core.metrics.Precision(top_k=4)
    >>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
    >>> m.result()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[keras_core.metrics.Precision()])
    ```

    Usage with a loss with `from_logits=True`:

    ```python
    model.compile(optimizer='adam',
                  loss=keras_core.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras_core.metrics.Precision(thresholds=0)])
    ```
    """

    def __init__(
        self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold
        )
        self._thresholds_distributed_evenly = (
            metrics_utils.is_evenly_distributed_thresholds(self.thresholds)
        )
        self.true_positives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=initializers.Zeros(),
            name="true_positives",
        )
        self.false_positives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=initializers.Zeros(),
            name="false_positives",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive and false positive statistics.

        Args:
            y_true: The ground truth values, with the same dimensions as
                `y_pred`. Will be cast to `bool`.
            y_pred: The predicted values. Each element must be in the range
                `[0, 1]`.
            sample_weight: Optional weighting of each example. Defaults to 1.
                Can be a `Tensor` whose rank is either 0, or the same rank as
                `y_true`, and must be broadcastable to `y_true`.
        """
        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,  # noqa: E501
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight,
        )

    def result(self):
        result = ops.divide(
            self.true_positives,
            self.true_positives + self.false_positives + backend.epsilon(),
        )
        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        self.true_positives.assign(ops.zeros((num_thresholds,)))
        self.false_positives.assign(ops.zeros((num_thresholds,)))

    def get_config(self):
        config = {
            "thresholds": self.init_thresholds,
            "top_k": self.top_k,
            "class_id": self.class_id,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@keras_core_export("keras_core.metrics.Recall")
class Recall(Metric):
    """Computes the recall of the predictions with respect to the labels.

    This metric creates two local variables, `true_positives` and
    `false_negatives`, that are used to compute the recall. This value is
    ultimately returned as `recall`, an idempotent operation that simply divides
    `true_positives` by the sum of `true_positives` and `false_negatives`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `top_k` is set, recall will be computed as how often on average a class
    among the labels of a batch entry is in the top-k predictions.

    If `class_id` is specified, we calculate recall by considering only the
    entries in the batch for which `class_id` is in the label, and computing the
    fraction of them for which `class_id` is above the threshold and/or in the
    top-k predictions.

    Args:
        thresholds: (Optional) A float value, or a Python list/tuple of float
            threshold values in `[0, 1]`. A threshold is compared with
            prediction values to determine the truth value of predictions (i.e.,
            above the threshold is `True`, below is `False`). If used with a
            loss function that sets `from_logits=True` (i.e. no sigmoid
            applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
            If neither `thresholds` nor `top_k` are set,
            the default is to calculate recall with `thresholds=0.5`.
        top_k: (Optional) Unset by default. An int value specifying the top-k
            predictions to consider when calculating recall.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.Recall()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
    >>> m.result()
    0.6666667

    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[keras_core.metrics.Recall()])
    ```

    Usage with a loss with `from_logits=True`:

    ```python
    model.compile(optimizer='adam',
                  loss=keras_core.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras_core.metrics.Recall(thresholds=0)])
    ```
    """

    def __init__(
        self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold
        )
        self._thresholds_distributed_evenly = (
            metrics_utils.is_evenly_distributed_thresholds(self.thresholds)
        )
        self.true_positives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=initializers.Zeros(),
            name="true_positives",
        )
        self.false_negatives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=initializers.Zeros(),
            name="false_negatives",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive and false negative statistics.

        Args:
            y_true: The ground truth values, with the same dimensions as
                `y_pred`. Will be cast to `bool`.
            y_pred: The predicted values. Each element must be in the range
                `[0, 1]`.
            sample_weight: Optional weighting of each example. Defaults to 1.
                Can be a `Tensor` whose rank is either 0, or the same rank as
                `y_true`, and must be broadcastable to `y_true`.
        """
        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,  # noqa: E501
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight,
        )

    def result(self):
        result = ops.divide(
            self.true_positives,
            self.true_positives + self.false_negatives + backend.epsilon(),
        )
        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        self.true_positives.assign(ops.zeros((num_thresholds,)))
        self.false_negatives.assign(ops.zeros((num_thresholds,)))

    def get_config(self):
        config = {
            "thresholds": self.init_thresholds,
            "top_k": self.top_k,
            "class_id": self.class_id,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class SensitivitySpecificityBase(Metric):
    """Abstract base class for computing sensitivity and specificity.

    For additional information about specificity and sensitivity, see
    [the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).
    """

    def __init__(
        self, value, num_thresholds=200, class_id=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        if num_thresholds <= 0:
            raise ValueError(
                "Argument `num_thresholds` must be an integer > 0. "
                f"Received: num_thresholds={num_thresholds}"
            )
        self.value = value
        self.class_id = class_id

        # Compute `num_thresholds` thresholds in [0, 1]
        if num_thresholds == 1:
            self.thresholds = [0.5]
            self._thresholds_distributed_evenly = False
        else:
            thresholds = [
                (i + 1) * 1.0 / (num_thresholds - 1)
                for i in range(num_thresholds - 2)
            ]
            self.thresholds = [0.0] + thresholds + [1.0]
            self._thresholds_distributed_evenly = True

        self.true_positives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=initializers.Zeros(),
            name="true_positives",
        )
        self.false_positives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=initializers.Zeros(),
            name="false_positives",
        )
        self.true_negatives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=initializers.Zeros(),
            name="true_negatives",
        )
        self.false_negatives = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=initializers.Zeros(),
            name="false_negatives",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates confusion matrix statistics.

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Defaults to 1.
                Can be a `Tensor` whose rank is either 0, or the same rank as
                `y_true`, and must be broadcastable to `y_true`.
        """
        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,  # noqa: E501
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            class_id=self.class_id,
            sample_weight=sample_weight,
        )

    def reset_state(self):
        num_thresholds = len(self.thresholds)
        self.true_positives.assign(ops.zeros((num_thresholds,)))
        self.false_positives.assign(ops.zeros((num_thresholds,)))
        self.true_negatives.assign(ops.zeros((num_thresholds,)))
        self.false_negatives.assign(ops.zeros((num_thresholds,)))

    def get_config(self):
        config = {"class_id": self.class_id}
        base_config = super().get_config()
        return {**base_config, **config}

    def _find_max_under_constraint(self, constrained, dependent, predicate):
        """Returns the maximum of dependent_statistic that satisfies the
        constraint.

        Args:
            constrained: Over these values the constraint is specified. A rank-1
                tensor.
            dependent: From these values the maximum that satiesfies the
                constraint is selected. Values in this tensor and in
                `constrained` are linked by having the same threshold at each
                position, hence this tensor must have the same shape.
            predicate: A binary boolean functor to be applied to arguments
                `constrained` and `self.value`, e.g. `ops.greater`.

        Returns:
            maximal dependent value, if no value satisfies the constraint 0.0.
        """
        feasible = ops.array(ops.nonzero(predicate(constrained, self.value)))

        print(feasible)
        feasible_exists = ops.greater(ops.size(feasible), 0)
        max_dependent = ops.max(ops.take(dependent, feasible), initial=0)

        return ops.where(feasible_exists, max_dependent, 0.0)


@keras_core_export("keras_core.metrics.SensitivityAtSpecificity")
class SensitivityAtSpecificity(SensitivitySpecificityBase):
    """Computes best sensitivity where specificity is >= specified value.

    `Sensitivity` measures the proportion of actual positives that are correctly
    identified as such `(tp / (tp + fn))`.
    `Specificity` measures the proportion of actual negatives that are correctly
    identified as such `(tn / (tn + fp))`.

    This metric creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the sensitivity at the given specificity. The threshold for the
    given specificity value is computed and used to evaluate the corresponding
    sensitivity.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold
    predictions, and computing the fraction of them for which `class_id` is
    indeed a correct label.

    For additional information about specificity and sensitivity, see
    [the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).

    Args:
        specificity: A scalar value in range `[0, 1]`.
        num_thresholds: (Optional) Defaults to 200. The number of thresholds to
            use for matching the given specificity.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.SensitivityAtSpecificity(0.5)
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
    ...                sample_weight=[1, 1, 2, 2, 1])
    >>> m.result()
    0.333333

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras_core.metrics.SensitivityAtSpecificity()])
    ```
    """

    def __init__(
        self,
        specificity,
        num_thresholds=200,
        class_id=None,
        name=None,
        dtype=None,
    ):
        if specificity < 0 or specificity > 1:
            raise ValueError(
                "Argument `specificity` must be in the range [0, 1]. "
                f"Received: specificity={specificity}"
            )
        self.specificity = specificity
        self.num_thresholds = num_thresholds
        super().__init__(
            specificity,
            num_thresholds=num_thresholds,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )

    def result(self):
        sensitivities = ops.divide(
            self.true_positives,
            self.true_positives + self.false_negatives + backend.epsilon(),
        )
        specificities = ops.divide(
            self.true_negatives,
            self.true_negatives + self.false_positives + backend.epsilon(),
        )
        return self._find_max_under_constraint(
            specificities, sensitivities, ops.greater_equal
        )

    def get_config(self):
        config = {
            "num_thresholds": self.num_thresholds,
            "specificity": self.specificity,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@keras_core_export("keras_core.metrics.SpecificityAtSensitivity")
class SpecificityAtSensitivity(SensitivitySpecificityBase):
    """Computes best specificity where sensitivity is >= specified value.

    `Sensitivity` measures the proportion of actual positives that are correctly
    identified as such `(tp / (tp + fn))`.
    `Specificity` measures the proportion of actual negatives that are correctly
    identified as such `(tn / (tn + fp))`.

    This metric creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the specificity at the given sensitivity. The threshold for the
    given sensitivity value is computed and used to evaluate the corresponding
    specificity.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold
    predictions, and computing the fraction of them for which `class_id` is
    indeed a correct label.

    For additional information about specificity and sensitivity, see
    [the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).

    Args:
        sensitivity: A scalar value in range `[0, 1]`.
        num_thresholds: (Optional) Defaults to 200. The number of thresholds to
            use for matching the given sensitivity.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.SpecificityAtSensitivity(0.5)
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
    >>> m.result()
    0.66666667

    >>> m.reset_state()
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
    ...                sample_weight=[1, 1, 2, 2, 2])
    >>> m.result()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras_core.metrics.SpecificityAtSensitivity()])
    ```
    """

    def __init__(
        self,
        sensitivity,
        num_thresholds=200,
        class_id=None,
        name=None,
        dtype=None,
    ):
        if sensitivity < 0 or sensitivity > 1:
            raise ValueError(
                "Argument `sensitivity` must be in the range [0, 1]. "
                f"Received: sensitivity={sensitivity}"
            )
        self.sensitivity = sensitivity
        self.num_thresholds = num_thresholds
        super().__init__(
            sensitivity,
            num_thresholds=num_thresholds,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )

    def result(self):
        sensitivities = ops.divide(
            self.true_positives,
            self.true_positives + self.false_negatives + backend.epsilon(),
        )
        specificities = ops.divide(
            self.true_negatives,
            self.true_negatives + self.false_positives + backend.epsilon(),
        )
        return self._find_max_under_constraint(
            sensitivities, specificities, ops.greater_equal
        )

    def get_config(self):
        config = {
            "num_thresholds": self.num_thresholds,
            "sensitivity": self.sensitivity,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@keras_core_export("keras_core.metrics.PrecisionAtRecall")
class PrecisionAtRecall(SensitivitySpecificityBase):
    """Computes best precision where recall is >= specified value.

    This metric creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the precision at the given recall. The threshold for the given
    recall value is computed and used to evaluate the corresponding precision.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold
    predictions, and computing the fraction of them for which `class_id` is
    indeed a correct label.

    Args:
        recall: A scalar value in range `[0, 1]`.
        num_thresholds: (Optional) Defaults to 200. The number of thresholds to
            use for matching the given recall.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.PrecisionAtRecall(0.5)
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
    ...                sample_weight=[2, 2, 2, 1, 1])
    >>> m.result()
    0.33333333

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras_core.metrics.PrecisionAtRecall(recall=0.8)])
    ```
    """

    def __init__(
        self, recall, num_thresholds=200, class_id=None, name=None, dtype=None
    ):
        if recall < 0 or recall > 1:
            raise ValueError(
                "Argument `recall` must be in the range [0, 1]. "
                f"Received: recall={recall}"
            )
        self.recall = recall
        self.num_thresholds = num_thresholds
        super().__init__(
            value=recall,
            num_thresholds=num_thresholds,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )

    def result(self):
        recalls = ops.divide(
            self.true_positives,
            self.true_positives + self.false_negatives + backend.epsilon(),
        )
        precisions = ops.divide(
            self.true_positives,
            self.true_positives + self.false_positives + backend.epsilon(),
        )
        return self._find_max_under_constraint(
            recalls, precisions, ops.greater_equal
        )

    def get_config(self):
        config = {"num_thresholds": self.num_thresholds, "recall": self.recall}
        base_config = super().get_config()
        return {**base_config, **config}


@keras_core_export("keras_core.metrics.RecallAtPrecision")
class RecallAtPrecision(SensitivitySpecificityBase):
    """Computes best recall where precision is >= specified value.

    For a given score-label-distribution the required precision might not
    be achievable, in this case 0.0 is returned as recall.

    This metric creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the recall at the given precision. The threshold for the given
    precision value is computed and used to evaluate the corresponding recall.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold
    predictions, and computing the fraction of them for which `class_id` is
    indeed a correct label.

    Args:
        precision: A scalar value in range `[0, 1]`.
            num_thresholds: (Optional) Defaults to 200. The number of thresholds
            to use for matching the given precision.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.RecallAtPrecision(0.8)
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
    ...                sample_weight=[1, 0, 0, 1])
    >>> m.result()
    1.0

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras_core.metrics.RecallAtPrecision(precision=0.8)])
    ```
    """

    def __init__(
        self,
        precision,
        num_thresholds=200,
        class_id=None,
        name=None,
        dtype=None,
    ):
        if precision < 0 or precision > 1:
            raise ValueError(
                "Argument `precision` must be in the range [0, 1]. "
                f"Received: precision={precision}"
            )
        self.precision = precision
        self.num_thresholds = num_thresholds
        super().__init__(
            value=precision,
            num_thresholds=num_thresholds,
            class_id=class_id,
            name=name,
            dtype=dtype,
        )

    def result(self):
        recalls = ops.divide(
            self.true_positives,
            self.true_positives + self.false_negatives + backend.epsilon(),
        )
        precisions = ops.divide(
            self.true_positives,
            self.true_positives + self.false_positives + backend.epsilon(),
        )
        return self._find_max_under_constraint(
            precisions, recalls, ops.greater_equal
        )

    def get_config(self):
        config = {
            "num_thresholds": self.num_thresholds,
            "precision": self.precision,
        }
        base_config = super().get_config()
        return {**base_config, **config}
