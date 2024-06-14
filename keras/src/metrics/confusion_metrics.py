import numpy as np

from keras.src import activations
from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.metrics import metrics_utils
from keras.src.metrics.metric import Metric
from keras.src.utils.python_utils import to_list


class _ConfusionMatrixConditionCount(Metric):
    """Calculates the number of the given confusion matrix condition.

    Args:
        confusion_matrix_cond: One of `metrics_utils.ConfusionMatrix`
            conditions.
        thresholds: (Optional) Defaults to `0.5`. A float value or a python list
            / tuple of float threshold values in `[0, 1]`. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `True`, below is `False`).
            One metric value is generated for each threshold value.
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
            sample_weight: Optional weighting of each example. Defaults to `1`.
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


@keras_export("keras.metrics.FalsePositives")
class FalsePositives(_ConfusionMatrixConditionCount):
    """Calculates the number of false positives.

    If `sample_weight` is given, calculates the sum of the weights of
    false positives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of false positives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        thresholds: (Optional) Defaults to `0.5`. A float value, or a Python
            list/tuple of float threshold values in `[0, 1]`. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `True`, below is `False`).
            If used with a loss function that sets `from_logits=True` (i.e. no
            sigmoid applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:

    >>> m = keras.metrics.FalsePositives()
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


@keras_export("keras.metrics.FalseNegatives")
class FalseNegatives(_ConfusionMatrixConditionCount):
    """Calculates the number of false negatives.

    If `sample_weight` is given, calculates the sum of the weights of
    false negatives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of false negatives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        thresholds: (Optional) Defaults to `0.5`. A float value, or a Python
            list/tuple of float threshold values in `[0, 1]`. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `True`, below is `False`).
            If used with a loss function that sets `from_logits=True` (i.e. no
            sigmoid applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

    >>> m = keras.metrics.FalseNegatives()
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


@keras_export("keras.metrics.TrueNegatives")
class TrueNegatives(_ConfusionMatrixConditionCount):
    """Calculates the number of true negatives.

    If `sample_weight` is given, calculates the sum of the weights of
    true negatives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of true negatives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        thresholds: (Optional) Defaults to `0.5`. A float value, or a Python
            list/tuple of float threshold values in `[0, 1]`. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `True`, below is `False`).
            If used with a loss function that sets `from_logits=True` (i.e. no
            sigmoid applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

    >>> m = keras.metrics.TrueNegatives()
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


@keras_export("keras.metrics.TruePositives")
class TruePositives(_ConfusionMatrixConditionCount):
    """Calculates the number of true positives.

    If `sample_weight` is given, calculates the sum of the weights of
    true positives. This metric creates one local variable, `true_positives`
    that is used to keep track of the number of true positives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        thresholds: (Optional) Defaults to `0.5`. A float value, or a Python
            list/tuple of float threshold values in `[0, 1]`. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `True`, below is `False`).
            If used with a loss function that sets `from_logits=True` (i.e. no
            sigmoid applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

    >>> m = keras.metrics.TruePositives()
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


@keras_export("keras.metrics.Precision")
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

    Example:

    >>> m = keras.metrics.Precision()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
    >>> m.result()
    0.6666667

    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0

    >>> # With top_k=2, it will calculate precision over y_true[:2]
    >>> # and y_pred[:2]
    >>> m = keras.metrics.Precision(top_k=2)
    >>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
    >>> m.result()
    0.0

    >>> # With top_k=4, it will calculate precision over y_true[:4]
    >>> # and y_pred[:4]
    >>> m = keras.metrics.Precision(top_k=4)
    >>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
    >>> m.result()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.Precision()])
    ```

    Usage with a loss with `from_logits=True`:

    ```python
    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.Precision(thresholds=0)])
    ```
    """

    def __init__(
        self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

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
            sample_weight: Optional weighting of each example. Defaults to `1`.
                Can be a tensor whose rank is either 0, or the same rank as
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
        result = ops.divide_no_nan(
            self.true_positives,
            ops.add(self.true_positives, self.false_positives),
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


@keras_export("keras.metrics.Recall")
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

    Example:

    >>> m = keras.metrics.Recall()
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
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.Recall()])
    ```

    Usage with a loss with `from_logits=True`:

    ```python
    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.Recall(thresholds=0)])
    ```
    """

    def __init__(
        self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

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
            sample_weight: Optional weighting of each example. Defaults to `1`.
                Can be a tensor whose rank is either 0, or the same rank as
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
        result = ops.divide_no_nan(
            self.true_positives,
            ops.add(self.true_positives, self.false_negatives),
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
        # Metric should be maximized during optimization.
        self._direction = "up"

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
            sample_weight: Optional weighting of each example. Defaults to `1`.
                Can be a tensor whose rank is either 0, or the same rank as
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
        feasible = ops.nonzero(predicate(constrained, self.value))
        feasible_exists = ops.greater(ops.size(feasible), 0)
        max_dependent = ops.max(ops.take(dependent, feasible), initial=0)

        return ops.where(feasible_exists, max_dependent, 0.0)


@keras_export("keras.metrics.SensitivityAtSpecificity")
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

    Example:

    >>> m = keras.metrics.SensitivityAtSpecificity(0.5)
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
        loss='binary_crossentropy',
        metrics=[keras.metrics.SensitivityAtSpecificity()])
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
        sensitivities = ops.divide_no_nan(
            self.true_positives,
            ops.add(self.true_positives, self.false_negatives),
        )
        specificities = ops.divide_no_nan(
            self.true_negatives,
            ops.add(self.true_negatives, self.false_positives),
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


@keras_export("keras.metrics.SpecificityAtSensitivity")
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

    Example:

    >>> m = keras.metrics.SpecificityAtSensitivity(0.5)
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
        loss='binary_crossentropy',
        metrics=[keras.metrics.SpecificityAtSensitivity()])
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
        sensitivities = ops.divide_no_nan(
            self.true_positives,
            ops.add(self.true_positives, self.false_negatives),
        )
        specificities = ops.divide_no_nan(
            self.true_negatives,
            ops.add(self.true_negatives, self.false_positives),
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


@keras_export("keras.metrics.PrecisionAtRecall")
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

    Example:

    >>> m = keras.metrics.PrecisionAtRecall(0.5)
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
        loss='binary_crossentropy',
        metrics=[keras.metrics.PrecisionAtRecall(recall=0.8)])
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
        recalls = ops.divide_no_nan(
            self.true_positives,
            ops.add(self.true_positives, self.false_negatives),
        )
        precisions = ops.divide_no_nan(
            self.true_positives,
            ops.add(self.true_positives, self.false_positives),
        )
        return self._find_max_under_constraint(
            recalls, precisions, ops.greater_equal
        )

    def get_config(self):
        config = {"num_thresholds": self.num_thresholds, "recall": self.recall}
        base_config = super().get_config()
        return {**base_config, **config}


@keras_export("keras.metrics.RecallAtPrecision")
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

    Example:

    >>> m = keras.metrics.RecallAtPrecision(0.8)
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
        loss='binary_crossentropy',
        metrics=[keras.metrics.RecallAtPrecision(precision=0.8)])
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
        recalls = ops.divide_no_nan(
            self.true_positives,
            ops.add(self.true_positives, self.false_negatives),
        )
        precisions = ops.divide_no_nan(
            self.true_positives,
            ops.add(self.true_positives, self.false_positives),
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


@keras_export("keras.metrics.AUC")
class AUC(Metric):
    """Approximates the AUC (Area under the curve) of the ROC or PR curves.

    The AUC (Area under the curve) of the ROC (Receiver operating
    characteristic; default) or PR (Precision Recall) curves are quality
    measures of binary classifiers. Unlike the accuracy, and like cross-entropy
    losses, ROC-AUC and PR-AUC evaluate all the operational points of a model.

    This class approximates AUCs using a Riemann sum. During the metric
    accumulation phrase, predictions are accumulated within predefined buckets
    by value. The AUC is then computed by interpolating per-bucket averages.
    These buckets define the evaluated operational points.

    This metric creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the AUC.  To discretize the AUC curve, a linearly spaced set of
    thresholds is used to compute pairs of recall and precision values. The area
    under the ROC-curve is therefore computed using the height of the recall
    values by the false positive rate, while the area under the PR-curve is the
    computed using the height of the precision values by the recall.

    This value is ultimately returned as `auc`, an idempotent operation that
    computes the area under a discretized curve of precision versus recall
    values (computed using the aforementioned variables). The `num_thresholds`
    variable controls the degree of discretization with larger numbers of
    thresholds more closely approximating the true AUC. The quality of the
    approximation may vary dramatically depending on `num_thresholds`. The
    `thresholds` parameter can be used to manually specify thresholds which
    split the predictions more evenly.

    For a best approximation of the real AUC, `predictions` should be
    distributed approximately uniformly in the range `[0, 1]` (if
    `from_logits=False`). The quality of the AUC approximation may be poor if
    this is not the case. Setting `summation_method` to 'minoring' or 'majoring'
    can help quantify the error in the approximation by providing lower or upper
    bound estimate of the AUC.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        num_thresholds: (Optional) The number of thresholds to
            use when discretizing the roc curve. Values must be > 1.
            Defaults to `200`.
        curve: (Optional) Specifies the name of the curve to be computed,
            `'ROC'` (default) or `'PR'` for the Precision-Recall-curve.
        summation_method: (Optional) Specifies the [Riemann summation method](
              https://en.wikipedia.org/wiki/Riemann_sum) used.
              'interpolation' (default) applies mid-point summation scheme for
              `ROC`.  For PR-AUC, interpolates (true/false) positives but not
              the ratio that is precision (see Davis & Goadrich 2006 for
              details); 'minoring' applies left summation for increasing
              intervals and right summation for decreasing intervals; 'majoring'
              does the opposite.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        thresholds: (Optional) A list of floating point values to use as the
            thresholds for discretizing the curve. If set, the `num_thresholds`
            parameter is ignored. Values should be in `[0, 1]`. Endpoint
            thresholds equal to {`-epsilon`, `1+epsilon`} for a small positive
            epsilon value will be automatically included with these to correctly
            handle predictions equal to exactly 0 or 1.
        multi_label: boolean indicating whether multilabel data should be
            treated as such, wherein AUC is computed separately for each label
            and then averaged across labels, or (when `False`) if the data
            should be flattened into a single label before AUC computation. In
            the latter case, when multilabel data is passed to AUC, each
            label-prediction pair is treated as an individual data point. Should
            be set to `False` for multi-class data.
        num_labels: (Optional) The number of labels, used when `multi_label` is
            True. If `num_labels` is not specified, then state variables get
            created on the first call to `update_state`.
        label_weights: (Optional) list, array, or tensor of non-negative weights
            used to compute AUCs for multilabel data. When `multi_label` is
            True, the weights are applied to the individual label AUCs when they
            are averaged to produce the multi-label AUC. When it's False, they
            are used to weight the individual label predictions in computing the
            confusion matrix on the flattened data. Note that this is unlike
            `class_weights` in that `class_weights` weights the example
            depending on the value of its label, whereas `label_weights` depends
            only on the index of that label before flattening; therefore
            `label_weights` should not be used for multi-class data.
        from_logits: boolean indicating whether the predictions (`y_pred` in
        `update_state`) are probabilities or sigmoid logits. As a rule of thumb,
        when using a keras loss, the `from_logits` constructor argument of the
        loss should match the AUC `from_logits` constructor argument.

    Example:

    >>> m = keras.metrics.AUC(num_thresholds=3)
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
    >>> # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
    >>> # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
    >>> # tp_rate = recall = [1, 0.5, 0], fp_rate = [1, 0, 0]
    >>> # auc = ((((1 + 0.5) / 2) * (1 - 0)) + (((0.5 + 0) / 2) * (0 - 0)))
    >>> #     = 0.75
    >>> m.result()
    0.75

    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
    ...                sample_weight=[1, 0, 0, 1])
    >>> m.result()
    1.0

    Usage with `compile()` API:

    ```python
    # Reports the AUC of a model outputting a probability.
    model.compile(optimizer='sgd',
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.AUC()])

    # Reports the AUC of a model outputting a logit.
    model.compile(optimizer='sgd',
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.AUC(from_logits=True)])
    ```
    """

    def __init__(
        self,
        num_thresholds=200,
        curve="ROC",
        summation_method="interpolation",
        name=None,
        dtype=None,
        thresholds=None,
        multi_label=False,
        num_labels=None,
        label_weights=None,
        from_logits=False,
    ):
        # Metric should be maximized during optimization.
        self._direction = "up"

        # Validate configurations.
        if isinstance(curve, metrics_utils.AUCCurve) and curve not in list(
            metrics_utils.AUCCurve
        ):
            raise ValueError(
                f'Invalid `curve` argument value "{curve}". '
                f"Expected one of: {list(metrics_utils.AUCCurve)}"
            )
        if isinstance(
            summation_method, metrics_utils.AUCSummationMethod
        ) and summation_method not in list(metrics_utils.AUCSummationMethod):
            raise ValueError(
                "Invalid `summation_method` "
                f'argument value "{summation_method}". '
                f"Expected one of: {list(metrics_utils.AUCSummationMethod)}"
            )

        # Update properties.
        self._init_from_thresholds = thresholds is not None
        if thresholds is not None:
            # If specified, use the supplied thresholds.
            self.num_thresholds = len(thresholds) + 2
            thresholds = sorted(thresholds)
            self._thresholds_distributed_evenly = (
                metrics_utils.is_evenly_distributed_thresholds(
                    np.array([0.0] + thresholds + [1.0])
                )
            )
        else:
            if num_thresholds <= 1:
                raise ValueError(
                    "Argument `num_thresholds` must be an integer > 1. "
                    f"Received: num_thresholds={num_thresholds}"
                )

            # Otherwise, linearly interpolate (num_thresholds - 2) thresholds in
            # (0, 1).
            self.num_thresholds = num_thresholds
            thresholds = [
                (i + 1) * 1.0 / (num_thresholds - 1)
                for i in range(num_thresholds - 2)
            ]
            self._thresholds_distributed_evenly = True

        # Add an endpoint "threshold" below zero and above one for either
        # threshold method to account for floating point imprecisions.
        self._thresholds = np.array(
            [0.0 - backend.epsilon()] + thresholds + [1.0 + backend.epsilon()]
        )

        if isinstance(curve, metrics_utils.AUCCurve):
            self.curve = curve
        else:
            self.curve = metrics_utils.AUCCurve.from_str(curve)
        if isinstance(summation_method, metrics_utils.AUCSummationMethod):
            self.summation_method = summation_method
        else:
            self.summation_method = metrics_utils.AUCSummationMethod.from_str(
                summation_method
            )
        super().__init__(name=name, dtype=dtype)

        # Handle multilabel arguments.
        self.multi_label = multi_label
        self.num_labels = num_labels
        if label_weights is not None:
            label_weights = ops.array(label_weights, dtype=self.dtype)
            self.label_weights = label_weights

        else:
            self.label_weights = None

        self._from_logits = from_logits

        self._built = False
        if self.multi_label:
            if num_labels:
                shape = [None, num_labels]
                self._build(shape)
        else:
            if num_labels:
                raise ValueError(
                    "`num_labels` is needed only when `multi_label` is True."
                )
            self._build(None)

    @property
    def thresholds(self):
        """The thresholds used for evaluating AUC."""
        return list(self._thresholds)

    def _build(self, shape):
        """Initialize TP, FP, TN, and FN tensors, given the shape of the
        data."""
        if self.multi_label:
            if len(shape) != 2:
                raise ValueError(
                    "`y_pred` must have rank 2 when `multi_label=True`. "
                    f"Found rank {len(shape)}. "
                    f"Full shape received for `y_pred`: {shape}"
                )
            self._num_labels = shape[1]
            variable_shape = [self.num_thresholds, self._num_labels]
        else:
            variable_shape = [self.num_thresholds]

        self._build_input_shape = shape
        # Create metric variables
        self.true_positives = self.add_variable(
            shape=variable_shape,
            initializer=initializers.Zeros(),
            name="true_positives",
        )
        self.false_positives = self.add_variable(
            shape=variable_shape,
            initializer=initializers.Zeros(),
            name="false_positives",
        )
        self.true_negatives = self.add_variable(
            shape=variable_shape,
            initializer=initializers.Zeros(),
            name="true_negatives",
        )
        self.false_negatives = self.add_variable(
            shape=variable_shape,
            initializer=initializers.Zeros(),
            name="false_negatives",
        )

        self._built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates confusion matrix statistics.

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Can
                be a tensor whose rank is either 0, or the same rank as
                `y_true`, and must be broadcastable to `y_true`. Defaults to
                `1`.
        """
        if not self._built:
            self._build(y_pred.shape)

        if self.multi_label or (self.label_weights is not None):
            # y_true should have shape (number of examples, number of labels).
            shapes = [(y_true, ("N", "L"))]
            if self.multi_label:
                # TP, TN, FP, and FN should all have shape
                # (number of thresholds, number of labels).
                shapes.extend(
                    [
                        (self.true_positives, ("T", "L")),
                        (self.true_negatives, ("T", "L")),
                        (self.false_positives, ("T", "L")),
                        (self.false_negatives, ("T", "L")),
                    ]
                )
            if self.label_weights is not None:
                # label_weights should be of length equal to the number of
                # labels.
                shapes.append((self.label_weights, ("L",)))

        # Only forward label_weights to update_confusion_matrix_variables when
        # multi_label is False. Otherwise the averaging of individual label AUCs
        # is handled in AUC.result
        label_weights = None if self.multi_label else self.label_weights

        if self._from_logits:
            y_pred = activations.sigmoid(y_pred)

        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,  # noqa: E501
            },
            y_true,
            y_pred,
            self._thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            sample_weight=sample_weight,
            multi_label=self.multi_label,
            label_weights=label_weights,
        )

    def interpolate_pr_auc(self):
        """Interpolation formula inspired by section 4 of Davis & Goadrich 2006.

        https://www.biostat.wisc.edu/~page/rocpr.pdf

        Note here we derive & use a closed formula not present in the paper
        as follows:

            Precision = TP / (TP + FP) = TP / P

        Modeling all of TP (true positive), FP (false positive) and their sum
        P = TP + FP (predicted positive) as varying linearly within each
        interval [A, B] between successive thresholds, we get

            Precision slope = dTP / dP
                            = (TP_B - TP_A) / (P_B - P_A)
                            = (TP - TP_A) / (P - P_A)
            Precision = (TP_A + slope * (P - P_A)) / P

        The area within the interval is (slope / total_pos_weight) times

            int_A^B{Precision.dP} = int_A^B{(TP_A + slope * (P - P_A)) * dP / P}
            int_A^B{Precision.dP} = int_A^B{slope * dP + intercept * dP / P}

        where intercept = TP_A - slope * P_A = TP_B - slope * P_B, resulting in

            int_A^B{Precision.dP} = TP_B - TP_A + intercept * log(P_B / P_A)

        Bringing back the factor (slope / total_pos_weight) we'd put aside, we
        get

            slope * [dTP + intercept *  log(P_B / P_A)] / total_pos_weight

        where dTP == TP_B - TP_A.

        Note that when P_A == 0 the above calculation simplifies into

            int_A^B{Precision.dTP} = int_A^B{slope * dTP}
                                   = slope * (TP_B - TP_A)

        which is really equivalent to imputing constant precision throughout the
        first bucket having >0 true positives.

        Returns:
            pr_auc: an approximation of the area under the P-R curve.
        """

        dtp = ops.subtract(
            self.true_positives[: self.num_thresholds - 1],
            self.true_positives[1:],
        )
        p = ops.add(self.true_positives, self.false_positives)
        dp = ops.subtract(p[: self.num_thresholds - 1], p[1:])
        prec_slope = ops.divide_no_nan(dtp, ops.maximum(dp, 0))
        intercept = ops.subtract(
            self.true_positives[1:], ops.multiply(prec_slope, p[1:])
        )

        safe_p_ratio = ops.where(
            ops.logical_and(p[: self.num_thresholds - 1] > 0, p[1:] > 0),
            ops.divide_no_nan(
                p[: self.num_thresholds - 1], ops.maximum(p[1:], 0)
            ),
            ops.ones_like(p[1:]),
        )

        pr_auc_increment = ops.divide_no_nan(
            ops.multiply(
                prec_slope,
                (ops.add(dtp, ops.multiply(intercept, ops.log(safe_p_ratio)))),
            ),
            ops.maximum(
                ops.add(self.true_positives[1:], self.false_negatives[1:]), 0
            ),
        )

        if self.multi_label:
            by_label_auc = ops.sum(pr_auc_increment, axis=0)
            if self.label_weights is None:
                # Evenly weighted average of the label AUCs.
                return ops.mean(by_label_auc)
            else:
                # Weighted average of the label AUCs.
                return ops.divide_no_nan(
                    ops.sum(ops.multiply(by_label_auc, self.label_weights)),
                    ops.sum(self.label_weights),
                )
        else:
            return ops.sum(pr_auc_increment)

    def result(self):
        if (
            self.curve == metrics_utils.AUCCurve.PR
            and self.summation_method
            == metrics_utils.AUCSummationMethod.INTERPOLATION
        ):
            # This use case is different and is handled separately.
            return self.interpolate_pr_auc()

        # Set `x` and `y` values for the curves based on `curve` config.
        recall = ops.divide_no_nan(
            self.true_positives,
            ops.add(self.true_positives, self.false_negatives),
        )
        if self.curve == metrics_utils.AUCCurve.ROC:
            fp_rate = ops.divide_no_nan(
                self.false_positives,
                ops.add(self.false_positives, self.true_negatives),
            )
            x = fp_rate
            y = recall
        else:  # curve == 'PR'.
            precision = ops.divide_no_nan(
                self.true_positives,
                ops.add(self.true_positives, self.false_positives),
            )
            x = recall
            y = precision

        # Find the rectangle heights based on `summation_method`.
        if (
            self.summation_method
            == metrics_utils.AUCSummationMethod.INTERPOLATION
        ):
            # Note: the case ('PR', 'interpolation') has been handled above.
            heights = ops.divide(
                ops.add(y[: self.num_thresholds - 1], y[1:]), 2.0
            )
        elif self.summation_method == metrics_utils.AUCSummationMethod.MINORING:
            heights = ops.minimum(y[: self.num_thresholds - 1], y[1:])
        # self.summation_method = metrics_utils.AUCSummationMethod.MAJORING:
        else:
            heights = ops.maximum(y[: self.num_thresholds - 1], y[1:])

        # Sum up the areas of all the rectangles.
        riemann_terms = ops.multiply(
            ops.subtract(x[: self.num_thresholds - 1], x[1:]), heights
        )
        if self.multi_label:
            by_label_auc = ops.sum(riemann_terms, axis=0)

            if self.label_weights is None:
                # Unweighted average of the label AUCs.
                return ops.mean(by_label_auc)
            else:
                # Weighted average of the label AUCs.
                return ops.divide_no_nan(
                    ops.sum(ops.multiply(by_label_auc, self.label_weights)),
                    ops.sum(self.label_weights),
                )
        else:
            return ops.sum(riemann_terms)

    def reset_state(self):
        if self._built:
            if self.multi_label:
                variable_shape = (self.num_thresholds, self._num_labels)
            else:
                variable_shape = (self.num_thresholds,)

            self.true_positives.assign(ops.zeros(variable_shape))
            self.false_positives.assign(ops.zeros(variable_shape))
            self.true_negatives.assign(ops.zeros(variable_shape))
            self.false_negatives.assign(ops.zeros(variable_shape))

    def get_config(self):
        label_weights = self.label_weights
        config = {
            "num_thresholds": self.num_thresholds,
            "curve": self.curve.value,
            "summation_method": self.summation_method.value,
            "multi_label": self.multi_label,
            "num_labels": self.num_labels,
            "label_weights": label_weights,
            "from_logits": self._from_logits,
        }
        # optimization to avoid serializing a large number of generated
        # thresholds
        if self._init_from_thresholds:
            # We remove the endpoint thresholds as an inverse of how the
            # thresholds were initialized. This ensures that a metric
            # initialized from this config has the same thresholds.
            config["thresholds"] = self.thresholds[1:-1]
        base_config = super().get_config()
        return {**base_config, **config}
