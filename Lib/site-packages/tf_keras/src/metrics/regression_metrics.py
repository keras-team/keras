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

import warnings

import tensorflow.compat.v2 as tf

from tf_keras.src import backend
from tf_keras.src.dtensor import utils as dtensor_utils
from tf_keras.src.losses import logcosh
from tf_keras.src.losses import mean_absolute_error
from tf_keras.src.losses import mean_absolute_percentage_error
from tf_keras.src.losses import mean_squared_error
from tf_keras.src.losses import mean_squared_logarithmic_error
from tf_keras.src.metrics import base_metric
from tf_keras.src.utils import losses_utils
from tf_keras.src.utils import metrics_utils
from tf_keras.src.utils.tf_utils import is_tensor_or_variable

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
          sample_weight: Optional weighting of each example. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`. Defaults to `1`.

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
      axis: (Optional) The dimension along which the cosine
        similarity is computed. Defaults to `-1`.

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
          sample_weight: Optional weighting of each example. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`. Defaults to `1`.

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


# Adapted from TF-Addons implementation (RSquare class).
@keras_export("keras.metrics.R2Score")
class R2Score(base_metric.Metric):
    """Computes R2 score.

    This is also called the
    [coefficient of
    determination](https://en.wikipedia.org/wiki/Coefficient_of_determination).

    It indicates how close the fitted regression line
    is to ground-truth data.

    - The highest score possible is 1.0. It indicates that the predictors
        perfectly accounts for variation in the target.
    - A score of 0.0 indicates that the predictors do not
        account for variation in the target.
    - It can also be negative if the model is worse than random.

    This metric can also compute the "Adjusted R2" score.

    Args:
        class_aggregation: Specifies how to aggregate scores corresponding to
            different output classes (or target dimensions),
            i.e. different dimensions on the last axis of the predictions.
            Equivalent to `multioutput` argument in Scikit-Learn.
            Should be one of
            `None` (no aggregation), `"uniform_average"`,
            `"variance_weighted_average"`.
        num_regressors: Number of independent regressors used
            ("Adjusted R2" score). 0 is the standard R2 score.
            Defaults to `0`.
        name: Optional. string name of the metric instance.
        dtype: Optional. data type of the metric result.

    Example:

    >>> y_true = np.array([[1], [4], [3]], dtype=np.float32)
    >>> y_pred = np.array([[2], [4], [4]], dtype=np.float32)
    >>> metric = tf.keras.metrics.R2Score()
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy()
    0.57142854
    """

    @dtensor_utils.inject_mesh
    def __init__(
        self,
        class_aggregation="uniform_average",
        num_regressors=0,
        name="r2_score",
        dtype=None,
    ):
        super().__init__(name=name, dtype=dtype)

        valid_class_aggregation_values = (
            None,
            "uniform_average",
            "variance_weighted_average",
        )
        if class_aggregation not in valid_class_aggregation_values:
            raise ValueError(
                "Invalid value for argument `class_aggregation`. Expected "
                f"one of {valid_class_aggregation_values}. "
                f"Received: class_aggregation={class_aggregation}"
            )
        if num_regressors < 0:
            raise ValueError(
                "Invalid value for argument `num_regressors`. "
                "Expected a value >= 0. "
                f"Received: num_regressors={num_regressors}"
            )
        self.class_aggregation = class_aggregation
        self.num_regressors = num_regressors
        self.num_samples = self.add_weight(name="num_samples", dtype="int32")
        self.built = False

    def build(self, y_true_shape, y_pred_shape):
        if len(y_pred_shape) != 2 or len(y_true_shape) != 2:
            raise ValueError(
                "R2Score expects 2D inputs with shape "
                "(batch_size, output_dim). Received input "
                f"shapes: y_pred.shape={y_pred_shape} and "
                f"y_true.shape={y_true_shape}."
            )
        if y_pred_shape[-1] is None or y_true_shape[-1] is None:
            raise ValueError(
                "R2Score expects 2D inputs with shape "
                "(batch_size, output_dim), with output_dim fully "
                "defined (not None). Received input "
                f"shapes: y_pred.shape={y_pred_shape} and "
                f"y_true.shape={y_true_shape}."
            )
        num_classes = y_pred_shape[-1]
        self.squared_sum = self.add_weight(
            name="squared_sum",
            shape=[num_classes],
            initializer="zeros",
        )
        self.sum = self.add_weight(
            name="sum",
            shape=[num_classes],
            initializer="zeros",
        )
        self.total_mse = self.add_weight(
            name="residual",
            shape=[num_classes],
            initializer="zeros",
        )
        self.count = self.add_weight(
            name="count",
            shape=[num_classes],
            initializer="zeros",
        )
        self.built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)
        if not self.built:
            self.build(y_true.shape, y_pred.shape)

        if sample_weight is None:
            sample_weight = 1

        sample_weight = tf.convert_to_tensor(sample_weight, dtype=self.dtype)
        if sample_weight.shape.rank == 1:
            # Make sure there's a features dimension
            sample_weight = tf.expand_dims(sample_weight, axis=1)
        sample_weight = tf.__internal__.ops.broadcast_weights(
            weights=sample_weight, values=y_true
        )

        weighted_y_true = y_true * sample_weight
        self.sum.assign_add(tf.reduce_sum(weighted_y_true, axis=0))
        self.squared_sum.assign_add(
            tf.reduce_sum(y_true * weighted_y_true, axis=0)
        )
        self.total_mse.assign_add(
            tf.reduce_sum((y_true - y_pred) ** 2 * sample_weight, axis=0)
        )
        self.count.assign_add(tf.reduce_sum(sample_weight, axis=0))
        self.num_samples.assign_add(tf.size(y_true))

    def result(self):
        mean = self.sum / self.count
        total = self.squared_sum - self.sum * mean
        raw_scores = 1 - (self.total_mse / total)
        raw_scores = tf.where(tf.math.is_inf(raw_scores), 0.0, raw_scores)

        if self.class_aggregation == "uniform_average":
            r2_score = tf.reduce_mean(raw_scores)
        elif self.class_aggregation == "variance_weighted_average":
            weighted_sum = tf.reduce_sum(total * raw_scores)
            sum_of_weights = tf.reduce_sum(total)
            r2_score = weighted_sum / sum_of_weights
        else:
            r2_score = raw_scores

        if self.num_regressors != 0:
            if self.num_regressors > self.num_samples - 1:
                warnings.warn(
                    "More independent predictors than datapoints "
                    "in adjusted R2 score. Falling back to standard R2 score.",
                    stacklevel=2,
                )
            elif self.num_regressors == self.num_samples - 1:
                warnings.warn(
                    "Division by zero in Adjusted R2 score. "
                    "Falling back to standard R2 score.",
                    stacklevel=2,
                )
            else:
                n = tf.cast(self.num_samples, dtype=tf.float32)
                p = tf.cast(self.num_regressors, dtype=tf.float32)
                num = tf.multiply(
                    tf.subtract(1.0, r2_score), tf.subtract(n, 1.0)
                )
                den = tf.subtract(tf.subtract(n, p), 1.0)
                r2_score = tf.subtract(1.0, tf.divide(num, den))
        return r2_score

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros(v.shape, dtype=v.dtype))

    def get_config(self):
        config = {
            "class_aggregation": self.class_aggregation,
            "num_regressors": self.num_regressors,
        }
        base_config = super().get_config()
        return {**base_config, **config}


def cosine_similarity(y_true, y_pred, axis=-1):
    """Computes the cosine similarity between labels and predictions.

    Args:
      y_true: The ground truth values.
      y_pred: The prediction values.
      axis: (Optional) -1 is the dimension along which the cosine
        similarity is computed. Defaults to `-1`.

    Returns:
      Cosine similarity value.
    """
    y_true = tf.linalg.l2_normalize(y_true, axis=axis)
    y_pred = tf.linalg.l2_normalize(y_pred, axis=axis)
    return tf.reduce_sum(y_true * y_pred, axis=axis)

