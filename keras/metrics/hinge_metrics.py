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
"""Hinge metrics."""

from keras.dtensor import utils as dtensor_utils
from keras.losses import categorical_hinge
from keras.losses import hinge
from keras.losses import squared_hinge
from keras.metrics import base_metric

# isort: off
from tensorflow.python.util.tf_export import keras_export


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
