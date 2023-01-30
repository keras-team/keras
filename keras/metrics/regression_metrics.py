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
"""Regression metrics, e.g. MAE/MSE/etc."""

import tensorflow.compat.v2 as tf

from keras import backend
from keras.dtensor import utils as dtensor_utils
from keras.losses import logcosh
from keras.losses import mean_absolute_error
from keras.losses import mean_absolute_percentage_error
from keras.losses import mean_squared_error
from keras.losses import mean_squared_logarithmic_error
from keras.metrics import base_metric
from keras.utils import losses_utils
from keras.utils import metrics_utils
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
