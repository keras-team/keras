# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Built-in metrics."""

import abc
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow.compat.v2 as tf

from keras import activations
from keras import backend
from keras.dtensor import utils as dtensor_utils
from keras.losses import binary_crossentropy
from keras.losses import categorical_crossentropy
from keras.losses import categorical_hinge
from keras.losses import hinge
from keras.losses import kullback_leibler_divergence
from keras.losses import logcosh
from keras.losses import mean_absolute_error
from keras.losses import mean_absolute_percentage_error
from keras.losses import mean_squared_error
from keras.losses import mean_squared_logarithmic_error
from keras.losses import poisson
from keras.losses import sparse_categorical_crossentropy
from keras.losses import squared_hinge
from keras.metrics import base_metric
from keras.utils import losses_utils
from keras.utils import metrics_utils
from keras.utils.generic_utils import to_list
from keras.utils.tf_utils import is_tensor_or_variable

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.metrics.MeanRelativeError")
class MeanRelativeError(base_metric.Mean):
    """Computes the mean relative error by normalizing with the given values.

    This metric creates two local variables, `total` and `count` that are used
    to compute the mean relative error. This is weighted by `sample_weight`, and
    it is ultimately returned as `mean_relative_error`: an idempotent operation
    that simply divides `total` by `count`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
      normalizer: The normalizer values with same shape as predictions.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.MeanRelativeError(normalizer=[1, 3, 2, 3])
    >>> m.update_state([1, 3, 2, 3], [2, 4, 6, 8])

    >>> # metric = mean(|y_pred - y_true| / normalizer)
    >>> #        = mean([1, 1, 4, 5] / [1, 3, 2, 3]) = mean([1, 1/3, 2, 5/3])
    >>> #        = 5/4 = 1.25
    >>> m.result().numpy()
    1.25

    Usage with `compile()` API:

    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.MeanRelativeError(normalizer=[1, 3])])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, normalizer, name=None, dtype=None):
        super().__init__(name=name, dtype=dtype)
        normalizer = tf.cast(normalizer, self._dtype)
        self.normalizer = normalizer

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates metric statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        [
            y_pred,
            y_true,
        ], sample_weight = metrics_utils.ragged_assert_compatible_and_get_flat_values(  # noqa: E501
            [y_pred, y_true], sample_weight
        )
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
            y_pred, y_true
        )

        y_pred, self.normalizer = losses_utils.remove_squeezable_dimensions(
            y_pred, self.normalizer
        )
        y_pred.shape.assert_is_compatible_with(y_true.shape)
        relative_errors = tf.math.divide_no_nan(
            tf.abs(y_true - y_pred), self.normalizer
        )

        return super().update_state(
            relative_errors, sample_weight=sample_weight
        )

    def get_config(self):
        n = self.normalizer
        config = {
            "normalizer": backend.eval(n) if is_tensor_or_variable(n) else n
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export("keras.metrics.Accuracy")
class Accuracy(base_metric.MeanMetricWrapper):
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

    Standalone usage:

    >>> m = tf.keras.metrics.Accuracy()
    >>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]])
    >>> m.result().numpy()
    0.75

    >>> m.reset_state()
    >>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]],
    ...                sample_weight=[1, 1, 0, 0])
    >>> m.result().numpy()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.Accuracy()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="accuracy", dtype=None):
        super().__init__(accuracy, name, dtype=dtype)


@keras_export("keras.metrics.BinaryAccuracy")
class BinaryAccuracy(base_metric.MeanMetricWrapper):
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

    Standalone usage:

    >>> m = tf.keras.metrics.BinaryAccuracy()
    >>> m.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]])
    >>> m.result().numpy()
    0.75

    >>> m.reset_state()
    >>> m.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]],
    ...                sample_weight=[1, 0, 0, 1])
    >>> m.result().numpy()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="binary_accuracy", dtype=None, threshold=0.5):
        super().__init__(
            metrics_utils.binary_matches, name, dtype=dtype, threshold=threshold
        )


@keras_export("keras.metrics.CategoricalAccuracy")
class CategoricalAccuracy(base_metric.MeanMetricWrapper):
    """Calculates how often predictions match one-hot labels.

    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    This metric creates two local variables, `total` and `count` that are used
    to compute the frequency with which `y_pred` matches `y_true`. This
    frequency is ultimately returned as `categorical accuracy`: an idempotent
    operation that simply divides `total` by `count`.

    `y_pred` and `y_true` should be passed in as vectors of probabilities,
    rather than as labels. If necessary, use `tf.one_hot` to expand `y_true` as
    a vector.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.CategoricalAccuracy()
    >>> m.update_state([[0, 0, 1], [0, 1, 0]], [[0.1, 0.9, 0.8],
    ...                 [0.05, 0.95, 0]])
    >>> m.result().numpy()
    0.5

    >>> m.reset_state()
    >>> m.update_state([[0, 0, 1], [0, 1, 0]], [[0.1, 0.9, 0.8],
    ...                 [0.05, 0.95, 0]],
    ...                sample_weight=[0.7, 0.3])
    >>> m.result().numpy()
    0.3

    Usage with `compile()` API:

    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.CategoricalAccuracy()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="categorical_accuracy", dtype=None):
        super().__init__(
            lambda y_true, y_pred: metrics_utils.sparse_categorical_matches(
                tf.math.argmax(y_true, axis=-1), y_pred
            ),
            name,
            dtype=dtype,
        )


@keras_export("keras.metrics.SparseCategoricalAccuracy")
class SparseCategoricalAccuracy(base_metric.MeanMetricWrapper):
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

    Standalone usage:

    >>> m = tf.keras.metrics.SparseCategoricalAccuracy()
    >>> m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]])
    >>> m.result().numpy()
    0.5

    >>> m.reset_state()
    >>> m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]],
    ...                sample_weight=[0.7, 0.3])
    >>> m.result().numpy()
    0.3

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="sparse_categorical_accuracy", dtype=None):
        super().__init__(
            metrics_utils.sparse_categorical_matches, name, dtype=dtype
        )


_SPARSE_CATEGORICAL_UPDATE_STATE_DOCSTRING = """Accumulates metric statistics.

For sparse categorical metrics, the shapes of `y_true` and `y_pred` are
different.

Args:
  y_true: Ground truth label values. shape = `[batch_size, d0, .. dN-1]` or
    shape = `[batch_size, d0, .. dN-1, 1]`.
  y_pred: The predicted probability values. shape = `[batch_size, d0, .. dN]`.
  sample_weight: Optional `sample_weight` acts as a
    coefficient for the metric. If a scalar is provided, then the metric is
    simply scaled by the given value. If `sample_weight` is a tensor of size
    `[batch_size]`, then the metric for each sample of the batch is rescaled
    by the corresponding element in the `sample_weight` vector. If the shape
    of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted
    to this shape), then each metric element of `y_pred` is scaled by the
    corresponding value of `sample_weight`. (Note on `dN-1`: all metric
    functions reduce by 1 dimension, usually the last axis (-1)).

Returns:
  Update op.
"""

SparseCategoricalAccuracy.update_state.__doc__ = (
    _SPARSE_CATEGORICAL_UPDATE_STATE_DOCSTRING
)


@keras_export("keras.metrics.TopKCategoricalAccuracy")
class TopKCategoricalAccuracy(base_metric.MeanMetricWrapper):
    """Computes how often targets are in the top `K` predictions.

    Args:
      k: (Optional) Number of top elements to look at for computing accuracy.
        Defaults to 5.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
    >>> m.update_state([[0, 0, 1], [0, 1, 0]],
    ...                [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
    >>> m.result().numpy()
    0.5

    >>> m.reset_state()
    >>> m.update_state([[0, 0, 1], [0, 1, 0]],
    ...                [[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
    ...                sample_weight=[0.7, 0.3])
    >>> m.result().numpy()
    0.3

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.TopKCategoricalAccuracy()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, k=5, name="top_k_categorical_accuracy", dtype=None):
        super().__init__(
            lambda yt, yp, k: metrics_utils.sparse_top_k_categorical_matches(
                tf.math.argmax(yt, axis=-1), yp, k
            ),
            name,
            dtype=dtype,
            k=k,
        )


@keras_export("keras.metrics.SparseTopKCategoricalAccuracy")
class SparseTopKCategoricalAccuracy(base_metric.MeanMetricWrapper):
    """Computes how often integer targets are in the top `K` predictions.

    Args:
      k: (Optional) Number of top elements to look at for computing accuracy.
        Defaults to 5.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    >>> m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
    >>> m.result().numpy()
    0.5

    >>> m.reset_state()
    >>> m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
    ...                sample_weight=[0.7, 0.3])
    >>> m.result().numpy()
    0.3

    Usage with `compile()` API:

    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(
        self, k=5, name="sparse_top_k_categorical_accuracy", dtype=None
    ):
        super().__init__(
            metrics_utils.sparse_top_k_categorical_matches,
            name,
            dtype=dtype,
            k=k,
        )


SparseTopKCategoricalAccuracy.update_state.__doc__ = (
    _SPARSE_CATEGORICAL_UPDATE_STATE_DOCSTRING
)


class _ConfusionMatrixConditionCount(base_metric.Metric):
    """Calculates the number of the given confusion matrix condition.

    Args:
      confusion_matrix_cond: One of `metrics_utils.ConfusionMatrix` conditions.
      thresholds: (Optional) Defaults to 0.5. A float value or a python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). One metric
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
        self.accumulator = self.add_weight(
            "accumulator", shape=(len(self.thresholds),), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the metric statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns:
          Update op.
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
        return tf.convert_to_tensor(result)

    def reset_state(self):
        backend.batch_set_value(
            [(v, np.zeros(v.shape.as_list())) for v in self.variables]
        )

    def get_config(self):
        config = {"thresholds": self.init_thresholds}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export("keras.metrics.FalsePositives")
class FalsePositives(_ConfusionMatrixConditionCount):
    """Calculates the number of false positives.

    If `sample_weight` is given, calculates the sum of the weights of
    false positives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of false positives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
      thresholds: (Optional) Defaults to 0.5. A float value, or a Python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). If used with a
        loss function that sets `from_logits=True` (i.e. no sigmoid applied to
        predictions), `thresholds` should be set to 0. One metric value is
        generated for each threshold value.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.FalsePositives()
    >>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1])
    >>> m.result().numpy()
    2.0

    >>> m.reset_state()
    >>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result().numpy()
    1.0

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.FalsePositives()])
    ```

    Usage with a loss with `from_logits=True`:

    ```python
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.FalsePositives(thresholds=0)])
    ```
    """

    @dtensor_utils.inject_mesh
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
      thresholds: (Optional) Defaults to 0.5. A float value, or a Python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). If used with a
        loss function that sets `from_logits=True` (i.e. no sigmoid applied to
        predictions), `thresholds` should be set to 0. One metric value is
        generated for each threshold value.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.FalseNegatives()
    >>> m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
    >>> m.result().numpy()
    2.0

    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [0, 1, 0, 0], sample_weight=[0, 0, 1, 0])
    >>> m.result().numpy()
    1.0

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.FalseNegatives()])
    ```

    Usage with a loss with `from_logits=True`:

    ```python
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.FalseNegatives(thresholds=0)])
    ```
    """

    @dtensor_utils.inject_mesh
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
      thresholds: (Optional) Defaults to 0.5. A float value, or a Python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). If used with a
        loss function that sets `from_logits=True` (i.e. no sigmoid applied to
        predictions), `thresholds` should be set to 0. One metric value is
        generated for each threshold value.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.TrueNegatives()
    >>> m.update_state([0, 1, 0, 0], [1, 1, 0, 0])
    >>> m.result().numpy()
    2.0

    >>> m.reset_state()
    >>> m.update_state([0, 1, 0, 0], [1, 1, 0, 0], sample_weight=[0, 0, 1, 0])
    >>> m.result().numpy()
    1.0

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.TrueNegatives()])
    ```

    Usage with a loss with `from_logits=True`:

    ```python
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.TrueNegatives(thresholds=0)])
    ```
    """

    @dtensor_utils.inject_mesh
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
      thresholds: (Optional) Defaults to 0.5. A float value, or a Python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). If used with a
        loss function that sets `from_logits=True` (i.e. no sigmoid applied to
        predictions), `thresholds` should be set to 0. One metric value is
        generated for each threshold value.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.TruePositives()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
    >>> m.result().numpy()
    2.0

    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result().numpy()
    1.0

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.TruePositives()])
    ```

    Usage with a loss with `from_logits=True`:

    ```python
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.TruePositives(thresholds=0)])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, thresholds=None, name=None, dtype=None):
        super().__init__(
            confusion_matrix_cond=metrics_utils.ConfusionMatrix.TRUE_POSITIVES,
            thresholds=thresholds,
            name=name,
            dtype=dtype,
        )


@keras_export("keras.metrics.Precision")
class Precision(base_metric.Metric):
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
        threshold values in [0, 1]. A threshold is compared with prediction
        values to determine the truth value of predictions (i.e., above the
        threshold is `true`, below is `false`). If used with a loss function
        that sets `from_logits=True` (i.e. no sigmoid applied to predictions),
        `thresholds` should be set to 0. One metric value is generated for each
        threshold value. If neither thresholds nor top_k are set, the default is
        to calculate precision with `thresholds=0.5`.
      top_k: (Optional) Unset by default. An int value specifying the top-k
        predictions to consider when calculating precision.
      class_id: (Optional) Integer class ID for which we want binary metrics.
        This must be in the half-open interval `[0, num_classes)`, where
        `num_classes` is the last dimension of predictions.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.Precision()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
    >>> m.result().numpy()
    0.6666667

    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result().numpy()
    1.0

    >>> # With top_k=2, it will calculate precision over y_true[:2]
    >>> # and y_pred[:2]
    >>> m = tf.keras.metrics.Precision(top_k=2)
    >>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
    >>> m.result().numpy()
    0.0

    >>> # With top_k=4, it will calculate precision over y_true[:4]
    >>> # and y_pred[:4]
    >>> m = tf.keras.metrics.Precision(top_k=4)
    >>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
    >>> m.result().numpy()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.Precision()])
    ```

    Usage with a loss with `from_logits=True`:

    ```python
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.Precision(thresholds=0)])
    ```
    """

    @dtensor_utils.inject_mesh
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
        self.true_positives = self.add_weight(
            "true_positives", shape=(len(self.thresholds),), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            "false_positives",
            shape=(len(self.thresholds),),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive and false positive statistics.

        Args:
          y_true: The ground truth values, with the same dimensions as `y_pred`.
            Will be cast to `bool`.
          y_pred: The predicted values. Each element must be in the range
            `[0, 1]`.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        return metrics_utils.update_confusion_matrix_variables(
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
        result = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_positives),
        )
        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        backend.batch_set_value(
            [
                (v, np.zeros((num_thresholds,)))
                for v in (self.true_positives, self.false_positives)
            ]
        )

    def get_config(self):
        config = {
            "thresholds": self.init_thresholds,
            "top_k": self.top_k,
            "class_id": self.class_id,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export("keras.metrics.Recall")
class Recall(base_metric.Metric):
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
        threshold values in [0, 1]. A threshold is compared with prediction
        values to determine the truth value of predictions (i.e., above the
        threshold is `true`, below is `false`). If used with a loss function
        that sets `from_logits=True` (i.e. no sigmoid applied to predictions),
        `thresholds` should be set to 0. One metric value is generated for each
        threshold value. If neither thresholds nor top_k are set, the default is
        to calculate recall with `thresholds=0.5`.
      top_k: (Optional) Unset by default. An int value specifying the top-k
        predictions to consider when calculating recall.
      class_id: (Optional) Integer class ID for which we want binary metrics.
        This must be in the half-open interval `[0, num_classes)`, where
        `num_classes` is the last dimension of predictions.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.Recall()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
    >>> m.result().numpy()
    0.6666667

    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result().numpy()
    1.0

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.Recall()])
    ```

    Usage with a loss with `from_logits=True`:

    ```python
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.Recall(thresholds=0)])
    ```
    """

    @dtensor_utils.inject_mesh
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
        self.true_positives = self.add_weight(
            "true_positives", shape=(len(self.thresholds),), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            "false_negatives",
            shape=(len(self.thresholds),),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive and false negative statistics.

        Args:
          y_true: The ground truth values, with the same dimensions as `y_pred`.
            Will be cast to `bool`.
          y_pred: The predicted values. Each element must be in the range
            `[0, 1]`.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        return metrics_utils.update_confusion_matrix_variables(
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
        result = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_negatives),
        )
        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        backend.batch_set_value(
            [
                (v, np.zeros((num_thresholds,)))
                for v in (self.true_positives, self.false_negatives)
            ]
        )

    def get_config(self):
        config = {
            "thresholds": self.init_thresholds,
            "top_k": self.top_k,
            "class_id": self.class_id,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SensitivitySpecificityBase(base_metric.Metric, metaclass=abc.ABCMeta):
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
        self.true_positives = self.add_weight(
            "true_positives", shape=(num_thresholds,), initializer="zeros"
        )
        self.true_negatives = self.add_weight(
            "true_negatives", shape=(num_thresholds,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            "false_positives", shape=(num_thresholds,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            "false_negatives", shape=(num_thresholds,), initializer="zeros"
        )

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

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        return metrics_utils.update_confusion_matrix_variables(
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
        confusion_matrix_variables = (
            self.true_positives,
            self.true_negatives,
            self.false_positives,
            self.false_negatives,
        )
        backend.batch_set_value(
            [
                (v, np.zeros((num_thresholds,)))
                for v in confusion_matrix_variables
            ]
        )

    def get_config(self):
        config = {"class_id": self.class_id}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _find_max_under_constraint(self, constrained, dependent, predicate):
        """Returns the maximum of dependent_statistic that satisfies the
        constraint.

        Args:
          constrained: Over these values the constraint
            is specified. A rank-1 tensor.
          dependent: From these values the maximum that satiesfies the
            constraint is selected. Values in this tensor and in
            `constrained` are linked by having the same threshold at each
            position, hence this tensor must have the same shape.
          predicate: A binary boolean functor to be applied to arguments
          `constrained` and `self.value`, e.g. `tf.greater`.

        Returns:
          maximal dependent value, if no value satiesfies the constraint 0.0.
        """
        feasible = tf.where(predicate(constrained, self.value))
        feasible_exists = tf.greater(tf.size(feasible), 0)
        max_dependent = tf.reduce_max(tf.gather(dependent, feasible))

        return tf.where(feasible_exists, max_dependent, 0.0)


@keras_export("keras.metrics.SensitivityAtSpecificity")
class SensitivityAtSpecificity(SensitivitySpecificityBase):
    """Computes best sensitivity where specificity is >= specified value.

    the sensitivity at a given specificity.

    `Sensitivity` measures the proportion of actual positives that are correctly
    identified as such (tp / (tp + fn)).
    `Specificity` measures the proportion of actual negatives that are correctly
    identified as such (tn / (tn + fp)).

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

    >>> m = tf.keras.metrics.SensitivityAtSpecificity(0.5)
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
    >>> m.result().numpy()
    0.5

    >>> m.reset_state()
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
    ...                sample_weight=[1, 1, 2, 2, 1])
    >>> m.result().numpy()
    0.333333

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.SensitivityAtSpecificity()])
    ```
    """

    @dtensor_utils.inject_mesh
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
        specificities = tf.math.divide_no_nan(
            self.true_negatives,
            tf.math.add(self.true_negatives, self.false_positives),
        )
        sensitivities = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_negatives),
        )
        return self._find_max_under_constraint(
            specificities, sensitivities, tf.greater_equal
        )

    def get_config(self):
        config = {
            "num_thresholds": self.num_thresholds,
            "specificity": self.specificity,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export("keras.metrics.SpecificityAtSensitivity")
class SpecificityAtSensitivity(SensitivitySpecificityBase):
    """Computes best specificity where sensitivity is >= specified value.

    `Sensitivity` measures the proportion of actual positives that are correctly
    identified as such (tp / (tp + fn)).
    `Specificity` measures the proportion of actual negatives that are correctly
    identified as such (tn / (tn + fp)).

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

    >>> m = tf.keras.metrics.SpecificityAtSensitivity(0.5)
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
    >>> m.result().numpy()
    0.66666667

    >>> m.reset_state()
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
    ...                sample_weight=[1, 1, 2, 2, 2])
    >>> m.result().numpy()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.SpecificityAtSensitivity()])
    ```
    """

    @dtensor_utils.inject_mesh
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
        sensitivities = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_negatives),
        )
        specificities = tf.math.divide_no_nan(
            self.true_negatives,
            tf.math.add(self.true_negatives, self.false_positives),
        )
        return self._find_max_under_constraint(
            sensitivities, specificities, tf.greater_equal
        )

    def get_config(self):
        config = {
            "num_thresholds": self.num_thresholds,
            "sensitivity": self.sensitivity,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

    Standalone usage:

    >>> m = tf.keras.metrics.PrecisionAtRecall(0.5)
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
    >>> m.result().numpy()
    0.5

    >>> m.reset_state()
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
    ...                sample_weight=[2, 2, 2, 1, 1])
    >>> m.result().numpy()
    0.33333333

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.PrecisionAtRecall(recall=0.8)])
    ```
    """

    @dtensor_utils.inject_mesh
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
        recalls = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_negatives),
        )
        precisions = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_positives),
        )
        return self._find_max_under_constraint(
            recalls, precisions, tf.greater_equal
        )

    def get_config(self):
        config = {"num_thresholds": self.num_thresholds, "recall": self.recall}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
      num_thresholds: (Optional) Defaults to 200. The number of thresholds to
        use for matching the given precision.
      class_id: (Optional) Integer class ID for which we want binary metrics.
        This must be in the half-open interval `[0, num_classes)`, where
        `num_classes` is the last dimension of predictions.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.RecallAtPrecision(0.8)
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
    >>> m.result().numpy()
    0.5

    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
    ...                sample_weight=[1, 0, 0, 1])
    >>> m.result().numpy()
    1.0

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.RecallAtPrecision(precision=0.8)])
    ```
    """

    @dtensor_utils.inject_mesh
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
        precisions = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_positives),
        )
        recalls = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_negatives),
        )
        return self._find_max_under_constraint(
            precisions, recalls, tf.greater_equal
        )

    def get_config(self):
        config = {
            "num_thresholds": self.num_thresholds,
            "precision": self.precision,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export("keras.metrics.AUC")
class AUC(base_metric.Metric):
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
    distributed approximately uniformly in the range [0, 1] (if
    `from_logits=False`). The quality of the AUC approximation may be poor if
    this is not the case. Setting `summation_method` to 'minoring' or 'majoring'
    can help quantify the error in the approximation by providing lower or upper
    bound estimate of the AUC.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
      num_thresholds: (Optional) Defaults to 200. The number of thresholds to
        use when discretizing the roc curve. Values must be > 1.
      curve: (Optional) Specifies the name of the curve to be computed, 'ROC'
        [default] or 'PR' for the Precision-Recall-curve.
      summation_method: (Optional) Specifies the [Riemann summation method](
          https://en.wikipedia.org/wiki/Riemann_sum) used.
          'interpolation' (default) applies mid-point summation scheme for
          `ROC`.  For PR-AUC, interpolates (true/false) positives but not the
          ratio that is precision (see Davis & Goadrich 2006 for details);
          'minoring' applies left summation for increasing intervals and right
          summation for decreasing intervals; 'majoring' does the opposite.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      thresholds: (Optional) A list of floating point values to use as the
        thresholds for discretizing the curve. If set, the `num_thresholds`
        parameter is ignored. Values should be in [0, 1]. Endpoint thresholds
        equal to {-epsilon, 1+epsilon} for a small positive epsilon value will
        be automatically included with these to correctly handle predictions
        equal to exactly 0 or 1.
      multi_label: boolean indicating whether multilabel data should be
        treated as such, wherein AUC is computed separately for each label and
        then averaged across labels, or (when False) if the data should be
        flattened into a single label before AUC computation. In the latter
        case, when multilabel data is passed to AUC, each label-prediction pair
        is treated as an individual data point. Should be set to False for
        multi-class data.
      num_labels: (Optional) The number of labels, used when `multi_label` is
        True. If `num_labels` is not specified, then state variables get created
        on the first call to `update_state`.
      label_weights: (Optional) list, array, or tensor of non-negative weights
        used to compute AUCs for multilabel data. When `multi_label` is True,
        the weights are applied to the individual label AUCs when they are
        averaged to produce the multi-label AUC. When it's False, they are used
        to weight the individual label predictions in computing the confusion
        matrix on the flattened data. Note that this is unlike class_weights in
        that class_weights weights the example depending on the value of its
        label, whereas label_weights depends only on the index of that label
        before flattening; therefore `label_weights` should not be used for
        multi-class data.
      from_logits: boolean indicating whether the predictions (`y_pred` in
        `update_state`) are probabilities or sigmoid logits. As a rule of thumb,
        when using a keras loss, the `from_logits` constructor argument of the
        loss should match the AUC `from_logits` constructor argument.

    Standalone usage:

    >>> m = tf.keras.metrics.AUC(num_thresholds=3)
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
    >>> # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
    >>> # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
    >>> # tp_rate = recall = [1, 0.5, 0], fp_rate = [1, 0, 0]
    >>> # auc = ((((1+0.5)/2)*(1-0)) + (((0.5+0)/2)*(0-0))) = 0.75
    >>> m.result().numpy()
    0.75

    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
    ...                sample_weight=[1, 0, 0, 1])
    >>> m.result().numpy()
    1.0

    Usage with `compile()` API:

    ```python
    # Reports the AUC of a model outputting a probability.
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC()])

    # Reports the AUC of a model outputting a logit.
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.AUC(from_logits=True)])
    ```
    """

    @dtensor_utils.inject_mesh
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
            label_weights = tf.constant(label_weights, dtype=self.dtype)
            tf.debugging.assert_non_negative(
                label_weights,
                message="All values of `label_weights` must be non-negative.",
            )
            self.label_weights = label_weights

        else:
            self.label_weights = None

        self._from_logits = from_logits

        self._built = False
        if self.multi_label:
            if num_labels:
                shape = tf.TensorShape([None, num_labels])
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
            if shape.ndims != 2:
                raise ValueError(
                    "`y_true` must have rank 2 when `multi_label=True`. "
                    f"Found rank {shape.ndims}. "
                    f"Full shape received for `y_true`: {shape}"
                )
            self._num_labels = shape[1]
            variable_shape = tf.TensorShape(
                [self.num_thresholds, self._num_labels]
            )
        else:
            variable_shape = tf.TensorShape([self.num_thresholds])

        self._build_input_shape = shape
        # Create metric variables
        self.true_positives = self.add_weight(
            "true_positives", shape=variable_shape, initializer="zeros"
        )
        self.true_negatives = self.add_weight(
            "true_negatives", shape=variable_shape, initializer="zeros"
        )
        self.false_positives = self.add_weight(
            "false_positives", shape=variable_shape, initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            "false_negatives", shape=variable_shape, initializer="zeros"
        )

        if self.multi_label:
            with tf.init_scope():
                # This should only be necessary for handling v1 behavior. In v2,
                # AUC should be initialized outside of any tf.functions, and
                # therefore in eager mode.
                if not tf.executing_eagerly():
                    backend._initialize_variables(backend._get_session())

        self._built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        if not self._built:
            self._build(tf.TensorShape(y_pred.shape))

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
                tf.debugging.assert_shapes(
                    shapes, message="Number of labels is not consistent."
                )

        # Only forward label_weights to update_confusion_matrix_variables when
        # multi_label is False. Otherwise the averaging of individual label AUCs
        # is handled in AUC.result
        label_weights = None if self.multi_label else self.label_weights

        if self._from_logits:
            y_pred = activations.sigmoid(y_pred)

        return metrics_utils.update_confusion_matrix_variables(
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

          int_A^B{Precision.dTP} = int_A^B{slope * dTP} = slope * (TP_B - TP_A)

        which is really equivalent to imputing constant precision throughout the
        first bucket having >0 true positives.

        Returns:
          pr_auc: an approximation of the area under the P-R curve.
        """
        dtp = (
            self.true_positives[: self.num_thresholds - 1]
            - self.true_positives[1:]
        )
        p = tf.math.add(self.true_positives, self.false_positives)
        dp = p[: self.num_thresholds - 1] - p[1:]
        prec_slope = tf.math.divide_no_nan(
            dtp, tf.maximum(dp, 0), name="prec_slope"
        )
        intercept = self.true_positives[1:] - tf.multiply(prec_slope, p[1:])

        safe_p_ratio = tf.where(
            tf.logical_and(p[: self.num_thresholds - 1] > 0, p[1:] > 0),
            tf.math.divide_no_nan(
                p[: self.num_thresholds - 1],
                tf.maximum(p[1:], 0),
                name="recall_relative_ratio",
            ),
            tf.ones_like(p[1:]),
        )

        pr_auc_increment = tf.math.divide_no_nan(
            prec_slope * (dtp + intercept * tf.math.log(safe_p_ratio)),
            tf.maximum(self.true_positives[1:] + self.false_negatives[1:], 0),
            name="pr_auc_increment",
        )

        if self.multi_label:
            by_label_auc = tf.reduce_sum(
                pr_auc_increment, name=self.name + "_by_label", axis=0
            )
            if self.label_weights is None:
                # Evenly weighted average of the label AUCs.
                return tf.reduce_mean(by_label_auc, name=self.name)
            else:
                # Weighted average of the label AUCs.
                return tf.math.divide_no_nan(
                    tf.reduce_sum(
                        tf.multiply(by_label_auc, self.label_weights)
                    ),
                    tf.reduce_sum(self.label_weights),
                    name=self.name,
                )
        else:
            return tf.reduce_sum(pr_auc_increment, name="interpolate_pr_auc")

    def result(self):
        if (
            self.curve == metrics_utils.AUCCurve.PR
            and self.summation_method
            == metrics_utils.AUCSummationMethod.INTERPOLATION
        ):
            # This use case is different and is handled separately.
            return self.interpolate_pr_auc()

        # Set `x` and `y` values for the curves based on `curve` config.
        recall = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_negatives),
        )
        if self.curve == metrics_utils.AUCCurve.ROC:
            fp_rate = tf.math.divide_no_nan(
                self.false_positives,
                tf.math.add(self.false_positives, self.true_negatives),
            )
            x = fp_rate
            y = recall
        else:  # curve == 'PR'.
            precision = tf.math.divide_no_nan(
                self.true_positives,
                tf.math.add(self.true_positives, self.false_positives),
            )
            x = recall
            y = precision

        # Find the rectangle heights based on `summation_method`.
        if (
            self.summation_method
            == metrics_utils.AUCSummationMethod.INTERPOLATION
        ):
            # Note: the case ('PR', 'interpolation') has been handled above.
            heights = (y[: self.num_thresholds - 1] + y[1:]) / 2.0
        elif self.summation_method == metrics_utils.AUCSummationMethod.MINORING:
            heights = tf.minimum(y[: self.num_thresholds - 1], y[1:])
        # self.summation_method = metrics_utils.AUCSummationMethod.MAJORING:
        else:
            heights = tf.maximum(y[: self.num_thresholds - 1], y[1:])

        # Sum up the areas of all the rectangles.
        if self.multi_label:
            riemann_terms = tf.multiply(
                x[: self.num_thresholds - 1] - x[1:], heights
            )
            by_label_auc = tf.reduce_sum(
                riemann_terms, name=self.name + "_by_label", axis=0
            )

            if self.label_weights is None:
                # Unweighted average of the label AUCs.
                return tf.reduce_mean(by_label_auc, name=self.name)
            else:
                # Weighted average of the label AUCs.
                return tf.math.divide_no_nan(
                    tf.reduce_sum(
                        tf.multiply(by_label_auc, self.label_weights)
                    ),
                    tf.reduce_sum(self.label_weights),
                    name=self.name,
                )
        else:
            return tf.reduce_sum(
                tf.multiply(x[: self.num_thresholds - 1] - x[1:], heights),
                name=self.name,
            )

    def reset_state(self):
        if self._built:
            confusion_matrix_variables = (
                self.true_positives,
                self.true_negatives,
                self.false_positives,
                self.false_negatives,
            )
            if self.multi_label:
                backend.batch_set_value(
                    [
                        (v, np.zeros((self.num_thresholds, self._num_labels)))
                        for v in confusion_matrix_variables
                    ]
                )
            else:
                backend.batch_set_value(
                    [
                        (v, np.zeros((self.num_thresholds,)))
                        for v in confusion_matrix_variables
                    ]
                )

    def get_config(self):
        if is_tensor_or_variable(self.label_weights):
            label_weights = backend.eval(self.label_weights)
        else:
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
        return dict(list(base_config.items()) + list(config.items()))


@keras_export("keras.metrics.CosineSimilarity")
class CosineSimilarity(base_metric.MeanMetricWrapper):
    """Computes the cosine similarity between the labels and predictions.

    `cosine similarity = (a . b) / ||a|| ||b||`

    See: [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).

    This metric keeps the average cosine similarity between `predictions` and
    `labels` over a stream of data.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      axis: (Optional) Defaults to -1. The dimension along which the cosine
        similarity is computed.

    Standalone usage:

    >>> # l2_norm(y_true) = [[0., 1.], [1./1.414, 1./1.414]]
    >>> # l2_norm(y_pred) = [[1., 0.], [1./1.414, 1./1.414]]
    >>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
    >>> # result = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
    >>> #        = ((0. + 0.) +  (0.5 + 0.5)) / 2
    >>> m = tf.keras.metrics.CosineSimilarity(axis=1)
    >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]])
    >>> m.result().numpy()
    0.49999997

    >>> m.reset_state()
    >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]],
    ...                sample_weight=[0.3, 0.7])
    >>> m.result().numpy()
    0.6999999

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="cosine_similarity", dtype=None, axis=-1):
        super().__init__(cosine_similarity, name, dtype=dtype, axis=axis)


@keras_export("keras.metrics.MeanAbsoluteError")
class MeanAbsoluteError(base_metric.MeanMetricWrapper):
    """Computes the mean absolute error between the labels and predictions.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.MeanAbsoluteError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    0.25

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="mean_absolute_error", dtype=None):
        super().__init__(mean_absolute_error, name, dtype=dtype)


@keras_export("keras.metrics.MeanAbsolutePercentageError")
class MeanAbsolutePercentageError(base_metric.MeanMetricWrapper):
    """Computes the mean absolute percentage error between `y_true` and
    `y_pred`.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.MeanAbsolutePercentageError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    250000000.0

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    500000000.0

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="mean_absolute_percentage_error", dtype=None):
        super().__init__(mean_absolute_percentage_error, name, dtype=dtype)


@keras_export("keras.metrics.MeanSquaredError")
class MeanSquaredError(base_metric.MeanMetricWrapper):
    """Computes the mean squared error between `y_true` and `y_pred`.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.MeanSquaredError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    0.25

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.MeanSquaredError()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="mean_squared_error", dtype=None):
        super().__init__(mean_squared_error, name, dtype=dtype)


@keras_export("keras.metrics.MeanSquaredLogarithmicError")
class MeanSquaredLogarithmicError(base_metric.MeanMetricWrapper):
    """Computes the mean squared logarithmic error between `y_true` and
    `y_pred`.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.MeanSquaredLogarithmicError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    0.12011322

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.24022643

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.MeanSquaredLogarithmicError()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="mean_squared_logarithmic_error", dtype=None):
        super().__init__(mean_squared_logarithmic_error, name, dtype=dtype)


@keras_export("keras.metrics.Hinge")
class Hinge(base_metric.MeanMetricWrapper):
    """Computes the hinge metric between `y_true` and `y_pred`.

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.Hinge()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result().numpy()
    1.3

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    1.1

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd', loss='mse', metrics=[tf.keras.metrics.Hinge()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="hinge", dtype=None):
        super().__init__(hinge, name, dtype=dtype)


@keras_export("keras.metrics.SquaredHinge")
class SquaredHinge(base_metric.MeanMetricWrapper):
    """Computes the squared hinge metric between `y_true` and `y_pred`.

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.SquaredHinge()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result().numpy()
    1.86

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    1.46

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.SquaredHinge()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="squared_hinge", dtype=None):
        super().__init__(squared_hinge, name, dtype=dtype)


@keras_export("keras.metrics.CategoricalHinge")
class CategoricalHinge(base_metric.MeanMetricWrapper):
    """Computes the categorical hinge metric between `y_true` and `y_pred`.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.CategoricalHinge()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result().numpy()
    1.4000001

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    1.2

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.CategoricalHinge()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="categorical_hinge", dtype=None):
        super().__init__(categorical_hinge, name, dtype=dtype)


@keras_export("keras.metrics.RootMeanSquaredError")
class RootMeanSquaredError(base_metric.Mean):
    """Computes root mean squared error metric between `y_true` and `y_pred`.

    Standalone usage:

    >>> m = tf.keras.metrics.RootMeanSquaredError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    0.5

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.70710677

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="root_mean_squared_error", dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates root mean squared error statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
            y_pred, y_true
        )
        error_sq = tf.math.squared_difference(y_pred, y_true)
        return super().update_state(error_sq, sample_weight=sample_weight)

    def result(self):
        return tf.sqrt(tf.math.divide_no_nan(self.total, self.count))


@keras_export("keras.metrics.LogCoshError")
class LogCoshError(base_metric.MeanMetricWrapper):
    """Computes the logarithm of the hyperbolic cosine of the prediction error.

    `logcosh = log((exp(x) + exp(-x))/2)`, where x is the error (y_pred -
    y_true)

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.LogCoshError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    0.10844523

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.21689045

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.LogCoshError()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="logcosh", dtype=None):
        super().__init__(logcosh, name, dtype=dtype)


@keras_export("keras.metrics.Poisson")
class Poisson(base_metric.MeanMetricWrapper):
    """Computes the Poisson metric between `y_true` and `y_pred`.

    `metric = y_pred - y_true * log(y_pred)`

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.Poisson()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    0.49999997

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.99999994

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.Poisson()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="poisson", dtype=None):
        super().__init__(poisson, name, dtype=dtype)


@keras_export("keras.metrics.KLDivergence")
class KLDivergence(base_metric.MeanMetricWrapper):
    """Computes Kullback-Leibler divergence metric between `y_true` and
    `y_pred`.

    `metric = y_true * log(y_true / y_pred)`

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.KLDivergence()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result().numpy()
    0.45814306

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.9162892

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.KLDivergence()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="kullback_leibler_divergence", dtype=None):
        super().__init__(kullback_leibler_divergence, name, dtype=dtype)


class _IoUBase(base_metric.Metric):
    """Computes the confusion matrix for Intersection-Over-Union metrics.

    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    For an individual class, the IoU metric is defined as follows:

    ```
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```

    From IoUs of individual classes, the MeanIoU can be computed as the mean of
    the individual IoUs.

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
      num_classes: The possible number of labels the prediction task can have.
        This value must be provided, since a confusion matrix of size
        `(num_classes, num_classes)` will be allocated.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      ignore_class: Optional integer. The ID of a class to be ignored during
        metric computation. This is useful, for example, in segmentation
        problems featuring a "void" class (commonly -1 or 255) in segmentation
        maps. By default (`ignore_class=None`), all classes are considered.
      sparse_y_true: Whether labels are encoded using integers or
        dense floating point vectors. If `False`, the `tf.argmax` function
        will be used to determine each sample's most likely associated label.
      sparse_y_pred: Whether predictions are encoded using integers or
        dense floating point vectors. If `False`, the `tf.argmax` function
        will be used to determine each sample's most likely associated label.
      axis: (Optional) Defaults to -1. The dimension containing the logits.
    """

    def __init__(
        self,
        num_classes: int,
        name: Optional[str] = None,
        dtype: Optional[Union[str, tf.dtypes.DType]] = None,
        ignore_class: Optional[int] = None,
        sparse_y_true: bool = True,
        sparse_y_pred: bool = True,
        axis: int = -1,
    ):
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.sparse_y_true = sparse_y_true
        self.sparse_y_pred = sparse_y_pred
        self.axis = axis

        # Variable to accumulate the predictions in the confusion matrix.
        self.total_cm = self.add_weight(
            "total_confusion_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns:
          Update op.
        """

        if not self.sparse_y_true:
            y_true = tf.argmax(y_true, axis=self.axis)
        if not self.sparse_y_pred:
            y_pred = tf.argmax(y_pred, axis=self.axis)

        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = tf.reshape(sample_weight, [-1])

        if self.ignore_class is not None:
            ignore_class = tf.cast(self.ignore_class, y_true.dtype)
            valid_mask = tf.not_equal(y_true, ignore_class)
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            if sample_weight is not None:
                sample_weight = sample_weight[valid_mask]

        # Accumulate the prediction to current confusion matrix.
        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=self._dtype,
        )
        return self.total_cm.assign_add(current_cm)

    def reset_state(self):
        backend.set_value(
            self.total_cm, np.zeros((self.num_classes, self.num_classes))
        )


@keras_export("keras.metrics.IoU")
class IoU(_IoUBase):
    """Computes the Intersection-Over-Union metric for specific target classes.

    General definition and computation:

    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    For an individual class, the IoU metric is defined as follows:

    ```
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Note, this class first computes IoUs for all individual classes, then
    returns the mean of IoUs for the classes that are specified by
    `target_class_ids`. If `target_class_ids` has only one id value, the IoU of
    that specific class is returned.

    Args:
      num_classes: The possible number of labels the prediction task can have.
        A confusion matrix of dimension = [num_classes, num_classes] will be
        allocated to accumulate predictions from which the metric is calculated.
      target_class_ids: A tuple or list of target class ids for which the metric
        is returned. To compute IoU for a specific class, a list (or tuple) of a
        single id value should be provided.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      ignore_class: Optional integer. The ID of a class to be ignored during
        metric computation. This is useful, for example, in segmentation
        problems featuring a "void" class (commonly -1 or 255) in segmentation
        maps. By default (`ignore_class=None`), all classes are considered.
      sparse_y_true: Whether labels are encoded using integers or
        dense floating point vectors. If `False`, the `tf.argmax` function
        will be used to determine each sample's most likely associated label.
      sparse_y_pred: Whether predictions are encoded using integers or
        dense floating point vectors. If `False`, the `tf.argmax` function
        will be used to determine each sample's most likely associated label.
      axis: (Optional) Defaults to -1. The dimension containing the logits.

    Standalone usage:

    >>> # cm = [[1, 1],
    >>> #        [1, 1]]
    >>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
    >>> # iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # iou = [0.33, 0.33]
    >>> m = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
    >>> m.result().numpy()
    0.33333334

    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
    ...                sample_weight=[0.3, 0.3, 0.3, 0.1])
    >>> # cm = [[0.3, 0.3],
    >>> #        [0.3, 0.1]]
    >>> # sum_row = [0.6, 0.4], sum_col = [0.6, 0.4],
    >>> # true_positives = [0.3, 0.1]
    >>> # iou = [0.33, 0.14]
    >>> m.result().numpy()
    0.33333334

    Usage with `compile()` API:

    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(
        self,
        num_classes: int,
        target_class_ids: Union[List[int], Tuple[int, ...]],
        name: Optional[str] = None,
        dtype: Optional[Union[str, tf.dtypes.DType]] = None,
        ignore_class: Optional[int] = None,
        sparse_y_true: bool = True,
        sparse_y_pred: bool = True,
        axis: int = -1,
    ):
        super().__init__(
            name=name,
            num_classes=num_classes,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            dtype=dtype,
        )
        if max(target_class_ids) >= num_classes:
            raise ValueError(
                f"Target class id {max(target_class_ids)} "
                "is out of range, which is "
                f"[{0}, {num_classes})."
            )
        self.target_class_ids = list(target_class_ids)

    def result(self):
        """Compute the intersection-over-union via the confusion matrix."""
        sum_over_row = tf.cast(
            tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype
        )
        sum_over_col = tf.cast(
            tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype
        )
        true_positives = tf.cast(
            tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype
        )

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        # Only keep the target classes
        true_positives = tf.gather(true_positives, self.target_class_ids)
        denominator = tf.gather(denominator, self.target_class_ids)

        # If the denominator is 0, we need to ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype)
        )

        iou = tf.math.divide_no_nan(true_positives, denominator)

        return tf.math.divide_no_nan(
            tf.reduce_sum(iou, name="mean_iou"), num_valid_entries
        )

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "target_class_ids": self.target_class_ids,
            "ignore_class": self.ignore_class,
            "sparse_y_true": self.sparse_y_true,
            "sparse_y_pred": self.sparse_y_pred,
            "axis": self.axis,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export("keras.metrics.BinaryIoU")
class BinaryIoU(IoU):
    """Computes the Intersection-Over-Union metric for class 0 and/or 1.

    General definition and computation:

    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    For an individual class, the IoU metric is defined as follows:

    ```
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    This class can be used to compute IoUs for a binary classification task
    where the predictions are provided as logits. First a `threshold` is applied
    to the predicted values such that those that are below the `threshold` are
    converted to class 0 and those that are above the `threshold` are converted
    to class 1.

    IoUs for classes 0 and 1 are then computed, the mean of IoUs for the classes
    that are specified by `target_class_ids` is returned.

    Note: with `threshold=0`, this metric has the same behavior as `IoU`.

    Args:
      target_class_ids: A tuple or list of target class ids for which the metric
        is returned. Options are `[0]`, `[1]`, or `[0, 1]`. With `[0]` (or
        `[1]`), the IoU metric for class 0 (or class 1, respectively) is
        returned. With `[0, 1]`, the mean of IoUs for the two classes is
        returned.
      threshold: A threshold that applies to the prediction logits to convert
        them to either predicted class 0 if the logit is below `threshold` or
        predicted class 1 if the logit is above `threshold`.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.3)
    >>> m.update_state([0, 1, 0, 1], [0.1, 0.2, 0.4, 0.7])
    >>> m.result().numpy()
    0.33333334

    >>> m.reset_state()
    >>> m.update_state([0, 1, 0, 1], [0.1, 0.2, 0.4, 0.7],
    ...                sample_weight=[0.2, 0.3, 0.4, 0.1])
    >>> # cm = [[0.2, 0.4],
    >>> #        [0.3, 0.1]]
    >>> # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5],
    >>> # true_positives = [0.2, 0.1]
    >>> # iou = [0.222, 0.125]
    >>> m.result().numpy()
    0.17361112

    Usage with `compile()` API:

    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[0], threshold=0.5)])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(
        self,
        target_class_ids: Union[List[int], Tuple[int, ...]] = (0, 1),
        threshold=0.5,
        name=None,
        dtype=None,
    ):

        super().__init__(
            num_classes=2,
            target_class_ids=target_class_ids,
            name=name,
            dtype=dtype,
        )
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Before the confusion matrix is updated, the predicted values are
        thresholded to be:
          0 for values that are smaller than the `threshold`
          1 for values that are larger or equal to the `threshold`

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        y_pred = tf.cast(y_pred, self._dtype)
        y_pred = tf.cast(y_pred >= self.threshold, self._dtype)
        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        return {
            "target_class_ids": self.target_class_ids,
            "threshold": self.threshold,
            "name": self.name,
            "dtype": self._dtype,
        }


@keras_export("keras.metrics.MeanIoU")
class MeanIoU(IoU):
    """Computes the mean Intersection-Over-Union metric.

    General definition and computation:

    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    For an individual class, the IoU metric is defined as follows:

    ```
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Note that this class first computes IoUs for all individual classes, then
    returns the mean of these values.

    Args:
      num_classes: The possible number of labels the prediction task can have.
        This value must be provided, since a confusion matrix of dimension =
        [num_classes, num_classes] will be allocated.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      ignore_class: Optional integer. The ID of a class to be ignored during
        metric computation. This is useful, for example, in segmentation
        problems featuring a "void" class (commonly -1 or 255) in segmentation
        maps. By default (`ignore_class=None`), all classes are considered.
      sparse_y_true: Whether labels are encoded using integers or
        dense floating point vectors. If `False`, the `tf.argmax` function
        will be used to determine each sample's most likely associated label.
      sparse_y_pred: Whether predictions are encoded using integers or
        dense floating point vectors. If `False`, the `tf.argmax` function
        will be used to determine each sample's most likely associated label.
      axis: (Optional) Defaults to -1. The dimension containing the logits.

    Standalone usage:

    >>> # cm = [[1, 1],
    >>> #        [1, 1]]
    >>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
    >>> # iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2 = 0.33
    >>> m = tf.keras.metrics.MeanIoU(num_classes=2)
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
    >>> m.result().numpy()
    0.33333334

    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
    ...                sample_weight=[0.3, 0.3, 0.3, 0.1])
    >>> m.result().numpy()
    0.23809525

    Usage with `compile()` API:

    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(
        self,
        num_classes: int,
        name: Optional[str] = None,
        dtype: Optional[Union[str, tf.dtypes.DType]] = None,
        ignore_class: Optional[int] = None,
        sparse_y_true: bool = True,
        sparse_y_pred: bool = True,
        axis: int = -1,
    ):
        target_class_ids = list(range(num_classes))
        super().__init__(
            name=name,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            axis=axis,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
        )

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "name": self.name,
            "dtype": self._dtype,
            "ignore_class": self.ignore_class,
            "sparse_y_true": self.sparse_y_true,
            "sparse_y_pred": self.sparse_y_pred,
            "axis": self.axis,
        }


@keras_export("keras.metrics.OneHotIoU")
class OneHotIoU(IoU):
    """Computes the Intersection-Over-Union metric for one-hot encoded labels.

    General definition and computation:

    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    For an individual class, the IoU metric is defined as follows:

    ```
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    This class can be used to compute IoU for multi-class classification tasks
    where the labels are one-hot encoded (the last axis should have one
    dimension per class). Note that the predictions should also have the same
    shape. To compute the IoU, first the labels and predictions are converted
    back into integer format by taking the argmax over the class axis. Then the
    same computation steps as for the base `IoU` class apply.

    Note, if there is only one channel in the labels and predictions, this class
    is the same as class `IoU`. In this case, use `IoU` instead.

    Also, make sure that `num_classes` is equal to the number of classes in the
    data, to avoid a "labels out of bound" error when the confusion matrix is
    computed.

    Args:
      num_classes: The possible number of labels the prediction task can have.
        A confusion matrix of shape `(num_classes, num_classes)` will be
        allocated to accumulate predictions from which the metric is calculated.
      target_class_ids: A tuple or list of target class ids for which the metric
        is returned. To compute IoU for a specific class, a list (or tuple) of a
        single id value should be provided.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      ignore_class: Optional integer. The ID of a class to be ignored during
        metric computation. This is useful, for example, in segmentation
        problems featuring a "void" class (commonly -1 or 255) in segmentation
        maps. By default (`ignore_class=None`), all classes are considered.
      sparse_y_pred: Whether predictions are encoded using natural numbers or
        probability distribution vectors. If `False`, the `tf.argmax` function
        will be used to determine each sample's most likely associated label.
      axis: (Optional) Defaults to -1. The dimension containing the logits.

    Standalone usage:

    >>> y_true = tf.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> y_pred = tf.constant([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1],
    ...                       [0.1, 0.4, 0.5]])
    >>> sample_weight = [0.1, 0.2, 0.3, 0.4]
    >>> m = tf.keras.metrics.OneHotIoU(num_classes=3, target_class_ids=[0, 2])
    >>> m.update_state(
    ...     y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    >>> # cm = [[0, 0, 0.2+0.4],
    >>> #       [0.3, 0, 0],
    >>> #       [0, 0, 0.1]]
    >>> # sum_row = [0.3, 0, 0.7], sum_col = [0.6, 0.3, 0.1]
    >>> # true_positives = [0, 0, 0.1]
    >>> # single_iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # mean_iou = (0 / (0.3 + 0.6 - 0) + 0.1 / (0.7 + 0.1 - 0.1)) / 2
    >>> m.result().numpy()
    0.071

    Usage with `compile()` API:

    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.OneHotIoU(num_classes=3, target_class_id=[1])])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(
        self,
        num_classes: int,
        target_class_ids: Union[List[int], Tuple[int, ...]],
        name=None,
        dtype=None,
        ignore_class: Optional[int] = None,
        sparse_y_pred: bool = False,
        axis: int = -1,
    ):
        super().__init__(
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=False,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
        )

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "target_class_ids": self.target_class_ids,
            "name": self.name,
            "dtype": self._dtype,
            "ignore_class": self.ignore_class,
            "sparse_y_pred": self.sparse_y_pred,
            "axis": self.axis,
        }


@keras_export("keras.metrics.OneHotMeanIoU")
class OneHotMeanIoU(MeanIoU):
    """Computes mean Intersection-Over-Union metric for one-hot encoded labels.

    General definition and computation:

    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    For an individual class, the IoU metric is defined as follows:

    ```
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    This class can be used to compute the mean IoU for multi-class
    classification tasks where the labels are one-hot encoded (the last axis
    should have one dimension per class). Note that the predictions should also
    have the same shape. To compute the mean IoU, first the labels and
    predictions are converted back into integer format by taking the argmax over
    the class axis. Then the same computation steps as for the base `MeanIoU`
    class apply.

    Note, if there is only one channel in the labels and predictions, this class
    is the same as class `MeanIoU`. In this case, use `MeanIoU` instead.

    Also, make sure that `num_classes` is equal to the number of classes in the
    data, to avoid a "labels out of bound" error when the confusion matrix is
    computed.

    Args:
      num_classes: The possible number of labels the prediction task can have.
        A confusion matrix of shape `(num_classes, num_classes)` will be
        allocated to accumulate predictions from which the metric is calculated.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      ignore_class: Optional integer. The ID of a class to be ignored during
        metric computation. This is useful, for example, in segmentation
        problems featuring a "void" class (commonly -1 or 255) in segmentation
        maps. By default (`ignore_class=None`), all classes are considered.
      sparse_y_pred: Whether predictions are encoded using natural numbers or
        probability distribution vectors. If `False`, the `tf.argmax` function
        will be used to determine each sample's most likely associated label.
      axis: (Optional) Defaults to -1. The dimension containing the logits.

    Standalone usage:

    >>> y_true = tf.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> y_pred = tf.constant([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1],
    ...                       [0.1, 0.4, 0.5]])
    >>> sample_weight = [0.1, 0.2, 0.3, 0.4]
    >>> m = tf.keras.metrics.OneHotMeanIoU(num_classes=3)
    >>> m.update_state(
    ...     y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    >>> # cm = [[0, 0, 0.2+0.4],
    >>> #       [0.3, 0, 0],
    >>> #       [0, 0, 0.1]]
    >>> # sum_row = [0.3, 0, 0.7], sum_col = [0.6, 0.3, 0.1]
    >>> # true_positives = [0, 0, 0.1]
    >>> # single_iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # mean_iou = (0 + 0 + 0.1 / (0.7 + 0.1 - 0.1)) / 3
    >>> m.result().numpy()
    0.048

    Usage with `compile()` API:

    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.OneHotMeanIoU(num_classes=3)])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(
        self,
        num_classes: int,
        name: str = None,
        dtype: Optional[Union[str, tf.dtypes.DType]] = None,
        ignore_class: Optional[int] = None,
        sparse_y_pred: bool = False,
        axis: int = -1,
    ):
        super().__init__(
            num_classes=num_classes,
            axis=axis,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=False,
            sparse_y_pred=sparse_y_pred,
        )

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "name": self.name,
            "dtype": self._dtype,
            "ignore_class": self.ignore_class,
            "sparse_y_pred": self.sparse_y_pred,
            "axis": self.axis,
        }


@keras_export("keras.metrics.BinaryCrossentropy")
class BinaryCrossentropy(base_metric.MeanMetricWrapper):
    """Computes the crossentropy metric between the labels and predictions.

    This is the crossentropy metric class to be used when there are only two
    label classes (0 and 1).

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      from_logits: (Optional )Whether output is expected to be a logits tensor.
        By default, we consider that output encodes a probability distribution.
      label_smoothing: (Optional) Float in [0, 1]. When > 0, label values are
        smoothed, meaning the confidence on label values are relaxed.
        e.g. `label_smoothing=0.2` means that we will use a value of `0.1` for
        label `0` and `0.9` for label `1`".

    Standalone usage:

    >>> m = tf.keras.metrics.BinaryCrossentropy()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result().numpy()
    0.81492424

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.9162905

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.BinaryCrossentropy()])
    ```
    """

    @dtensor_utils.inject_mesh
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


@keras_export("keras.metrics.CategoricalCrossentropy")
class CategoricalCrossentropy(base_metric.MeanMetricWrapper):
    """Computes the crossentropy metric between the labels and predictions.

    This is the crossentropy metric class to be used when there are multiple
    label classes (2 or more). Here we assume that labels are given as a
    `one_hot` representation. eg., When labels values are [2, 0, 1],
     `y_true` = [[0, 0, 1], [1, 0, 0], [0, 1, 0]].

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      from_logits: (Optional) Whether output is expected to be a logits tensor.
        By default, we consider that output encodes a probability distribution.
      label_smoothing: (Optional) Float in [0, 1]. When > 0, label values are
        smoothed, meaning the confidence on label values are relaxed. e.g.
        `label_smoothing=0.2` means that we will use a value of `0.1` for label
        `0` and `0.9` for label `1`"
      axis: (Optional) Defaults to -1. The dimension along which entropy is
        computed.

    Standalone usage:

    >>> # EPSILON = 1e-7, y = y_true, y` = y_pred
    >>> # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    >>> # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
    >>> # xent = -sum(y * log(y'), axis = -1)
    >>> #      = -((log 0.95), (log 0.1))
    >>> #      = [0.051, 2.302]
    >>> # Reduced xent = (0.051 + 2.302) / 2
    >>> m = tf.keras.metrics.CategoricalCrossentropy()
    >>> m.update_state([[0, 1, 0], [0, 0, 1]],
    ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    >>> m.result().numpy()
    1.1769392

    >>> m.reset_state()
    >>> m.update_state([[0, 1, 0], [0, 0, 1]],
    ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
    ...                sample_weight=tf.constant([0.3, 0.7]))
    >>> m.result().numpy()
    1.6271976

    Usage with `compile()` API:

    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.CategoricalCrossentropy()])
    ```
    """

    @dtensor_utils.inject_mesh
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


@keras_export("keras.metrics.SparseCategoricalCrossentropy")
class SparseCategoricalCrossentropy(base_metric.MeanMetricWrapper):
    """Computes the crossentropy metric between the labels and predictions.

    Use this crossentropy metric when there are two or more label classes.
    We expect labels to be provided as integers. If you want to provide labels
    using `one-hot` representation, please use `CategoricalCrossentropy` metric.
    There should be `# classes` floating point values per feature for `y_pred`
    and a single floating point value per feature for `y_true`.

    In the snippet below, there is a single floating point value per example for
    `y_true` and `# classes` floating pointing values per example for `y_pred`.
    The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
    `[batch_size, num_classes]`.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      from_logits: (Optional) Whether output is expected to be a logits tensor.
        By default, we consider that output encodes a probability distribution.
      ignore_class: Optional integer. The ID of a class to be ignored during
        metric computation. This is useful, for example, in segmentation
        problems featuring a "void" class (commonly -1 or 255) in segmentation
        maps. By default (`ignore_class=None`), all classes are considered.
      axis: (Optional) Defaults to -1. The dimension along which entropy is
        computed.

    Standalone usage:

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
    >>> m = tf.keras.metrics.SparseCategoricalCrossentropy()
    >>> m.update_state([1, 2],
    ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    >>> m.result().numpy()
    1.1769392

    >>> m.reset_state()
    >>> m.update_state([1, 2],
    ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
    ...                sample_weight=tf.constant([0.3, 0.7]))
    >>> m.result().numpy()
    1.6271976

    Usage with `compile()` API:

    ```python
    model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()])
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(
        self,
        name: str = "sparse_categorical_crossentropy",
        dtype: Optional[Union[str, tf.dtypes.DType]] = None,
        from_logits: bool = False,
        ignore_class: Optional[int] = None,
        axis: int = -1,
    ):
        super().__init__(
            sparse_categorical_crossentropy,
            name,
            dtype=dtype,
            from_logits=from_logits,
            ignore_class=ignore_class,
            axis=axis,
        )


SparseCategoricalCrossentropy.update_state.__doc__ = (
    _SPARSE_CATEGORICAL_UPDATE_STATE_DOCSTRING
)


def accuracy(y_true, y_pred):
    [
        y_pred,
        y_true,
    ], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
        [y_pred, y_true]
    )
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    if y_true.dtype != y_pred.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)
    return tf.cast(tf.equal(y_true, y_pred), backend.floatx())


@keras_export("keras.metrics.binary_accuracy")
@tf.__internal__.dispatch.add_dispatch_support
def binary_accuracy(y_true, y_pred, threshold=0.5):
    """Calculates how often predictions match binary labels.

    Standalone usage:
    >>> y_true = [[1], [1], [0], [0]]
    >>> y_pred = [[1], [1], [0], [0]]
    >>> m = tf.keras.metrics.binary_accuracy(y_true, y_pred)
    >>> assert m.shape == (4,)
    >>> m.numpy()
    array([1., 1., 1., 1.], dtype=float32)

    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
      threshold: (Optional) Float representing the threshold for deciding
        whether prediction values are 1 or 0.

    Returns:
      Binary accuracy values. shape = `[batch_size, d0, .. dN-1]`
    """
    # Note: calls metrics_utils.binary_matches with mean reduction. This
    # maintains public facing binary_accuracy behavior and seperates it from the
    # vital behavior of the binary_matches method needed in backend
    # dependencies.

    return tf.reduce_mean(
        metrics_utils.binary_matches(y_true, y_pred, threshold), axis=-1
    )


@keras_export("keras.metrics.categorical_accuracy")
@tf.__internal__.dispatch.add_dispatch_support
def categorical_accuracy(y_true, y_pred):
    """Calculates how often predictions match one-hot labels.

    Standalone usage:
    >>> y_true = [[0, 0, 1], [0, 1, 0]]
    >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    >>> m = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    >>> assert m.shape == (2,)
    >>> m.numpy()
    array([0., 1.], dtype=float32)

    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    Args:
      y_true: One-hot ground truth values.
      y_pred: The prediction values.

    Returns:
      Categorical accuracy values.
    """
    # Note: wraps metrics_utils.categorical_matches. This seperates public
    # facing categorical_accuracy behavior from the vital behavior of the
    # categorical_matches method needed in backend dependencies.

    return metrics_utils.sparse_categorical_matches(
        tf.math.argmax(y_true, axis=-1), y_pred
    )


@keras_export("keras.metrics.sparse_categorical_accuracy")
@tf.__internal__.dispatch.add_dispatch_support
def sparse_categorical_accuracy(y_true, y_pred):
    """Calculates how often predictions match integer labels.

    Standalone usage:
    >>> y_true = [2, 1]
    >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    >>> m = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    >>> assert m.shape == (2,)
    >>> m.numpy()
    array([0., 1.], dtype=float32)

    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    Args:
      y_true: Integer ground truth values.
      y_pred: The prediction values.

    Returns:
      Sparse categorical accuracy values.
    """
    # Note: wraps metrics_utils.sparse_categorical_matches method and checks for
    # squeezing to align with expected public facing behavior. This seperates
    # public facing sparse_categorical_accuracy behavior from the vital behavior
    # of the sparse_categorical_matches method needed in backend dependencies.

    matches = metrics_utils.sparse_categorical_matches(y_true, y_pred)

    # if shape is (num_samples, 1) squeeze
    if matches.shape.ndims > 1 and matches.shape[-1] == 1:
        matches = tf.squeeze(matches, [-1])

    return matches


@keras_export("keras.metrics.top_k_categorical_accuracy")
@tf.__internal__.dispatch.add_dispatch_support
def top_k_categorical_accuracy(y_true, y_pred, k=5):
    """Computes how often targets are in the top `K` predictions.

    Standalone usage:
    >>> y_true = [[0, 0, 1], [0, 1, 0]]
    >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    >>> m = tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
    >>> assert m.shape == (2,)
    >>> m.numpy()
    array([1., 1.], dtype=float32)

    Args:
      y_true: The ground truth values.
      y_pred: The prediction values.
      k: (Optional) Number of top elements to look at for computing accuracy.
        Defaults to 5.

    Returns:
      Top K categorical accuracy value.
    """
    # Note: wraps metrics_utils.top_k_categorical_matches. This seperates
    # public facing top_k_categorical_accuracy behavior from the vital behavior
    # of the top_k_categorical_matches method needed in backend dependencies.

    return metrics_utils.sparse_top_k_categorical_matches(
        tf.math.argmax(y_true, axis=-1), y_pred, k
    )


@keras_export("keras.metrics.sparse_top_k_categorical_accuracy")
@tf.__internal__.dispatch.add_dispatch_support
def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    """Computes how often integer targets are in the top `K` predictions.

    Standalone usage:
    >>> y_true = [2, 1]
    >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    >>> m = tf.keras.metrics.sparse_top_k_categorical_accuracy(
    ...     y_true, y_pred, k=3)
    >>> assert m.shape == (2,)
    >>> m.numpy()
    array([1., 1.], dtype=float32)

    Args:
      y_true: tensor of true targets.
      y_pred: tensor of predicted targets.
      k: (Optional) Number of top elements to look at for computing accuracy.
        Defaults to 5.

    Returns:
      Sparse top K categorical accuracy value.
    """
    # Note: wraps metrics_utils.sparse_top_k_categorical_matches. This seperates
    # public facing sparse_top_k_categorical_accuracy behavior from the vital
    # behavior of the sparse_top_k_categorical_matches method needed in backend
    # dependencies.

    return metrics_utils.sparse_top_k_categorical_matches(y_true, y_pred, k)


def cosine_similarity(y_true, y_pred, axis=-1):
    """Computes the cosine similarity between labels and predictions.

    Args:
      y_true: The ground truth values.
      y_pred: The prediction values.
      axis: (Optional) Defaults to -1. The dimension along which the cosine
        similarity is computed.

    Returns:
      Cosine similarity value.
    """
    y_true = tf.linalg.l2_normalize(y_true, axis=axis)
    y_pred = tf.linalg.l2_normalize(y_pred, axis=axis)
    return tf.reduce_sum(y_true * y_pred, axis=axis)
