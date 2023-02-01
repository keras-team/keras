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
"""Accuracy metrics."""

import tensorflow.compat.v2 as tf

from keras import backend
from keras.dtensor import utils as dtensor_utils
from keras.metrics import base_metric
from keras.utils import metrics_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


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
