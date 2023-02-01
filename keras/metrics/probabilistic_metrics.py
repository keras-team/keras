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
"""Probabilistic metrics (based on Entropy)."""

from typing import Optional
from typing import Union

import tensorflow.compat.v2 as tf

from keras.dtensor import utils as dtensor_utils
from keras.losses import binary_crossentropy
from keras.losses import categorical_crossentropy
from keras.losses import kullback_leibler_divergence
from keras.losses import poisson
from keras.losses import sparse_categorical_crossentropy
from keras.metrics import base_metric

# isort: off
from tensorflow.python.util.tf_export import keras_export


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

SparseCategoricalCrossentropy.update_state.__doc__ = (
    _SPARSE_CATEGORICAL_UPDATE_STATE_DOCSTRING
)
