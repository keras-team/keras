# Copyright 2023 The Keras Authors. All Rights Reserved.
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
"""Base class for Python-based metrics"""

import types

import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export

from keras.metrics import base_metric


@keras_export("keras.metrics.experimental.PyMetric", v1=[])
class PyMetric(base_metric.Metric):
    """Metric which runs in Python, compiled outside of the TensorFlow graph.

    Args:
      name: (Optional) string name of the PyMetric instance.
      dtype: (Optional) data type of the PyMetric result.
      **kwargs: Additional layer keywords arguments.

    Usage of `PyMetric` is generally identical to `keras.metrics.Metric`.
    It can be used in isolation, or in tandem with the `compile()` API. For more
    information about the usage of `PyMetric`, see `keras.metrics.Metric`.

    Unlike regular metrics, `PyMetric` instances are outside-compiled
    with respect to the TensorFlow graph during training or evaluation.
    They have access to the same
    inputs of a standard in-graph metric, but they run in a Python interpreter
    on the host CPU. Any data stored in a `PyMetric` is located on the main
    memory of the host CPU, and any TensorFlow ops used in a PyMetric are
    run eagerly on the host CPU.

    As a result, `PyMetric` instances are generally not as performant
    as in-graph metrics, and should only be used in cases where computing
    the metric inside of the TensorFlow graph is either impossible
    or prohibitively expensive.

    **Note:** Due to the use of `tf.py_function`, PyMetrics
    are incompatible with XLA and therefore TPUs.

    Methods to be implemented by subclasses:

    * `update_state()`: Handles updates to internal state variables
    * `result()`: Computes and returns a scalar value or a dict of scalar values
      for the metric from the state variables.
    * `reset_state()`: Computes and returns a scalar value for the metric from
      the state variables.

    This subclass implementation is similar to that of `keras.metrics.Metric`,
    with two notable differences:

    * Inputs to `update_state()` in a `PyMetric` are eager tensors, and both
    `update_state()` and `result()` run outside of the TensorFlow graph,
    executing any TensorFlow ops eagerly.
    * `reset_state()` is also called at initialization time to initialize the
    Python state of the metric.
    * `result()` can only return a single scalar. It does not support returning
    a dictionary of results like `keras.metrics.Metric`.

    Example subclass implementation using sklearn's Jaccard Score:

    ```python
    from sklearn.metrics import jaccard_score
    import tensorflow as tf

    class JaccardScore(tf.keras.metrics.experimental.PyMetric):

      def __init__(self, name='jaccard_score', **kwargs):
        super().__init__(name=name, **kwargs)

      def update_state(self, y_true, y_pred, sample_weight=None):
        self.jaccard_sum += jaccard_score(y_pred, y_true, average="macro")
        self.count += 1

      def reset_state(self):
        self.jaccard_sum = 0.
        self.count = 0.

      def result(self):
        return self.jaccard_sum / self.count
    ```
    """

    def __init__(self, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.reset_state()

    def __new__(cls, *args, **kwargs):
        obj = super(base_metric.Metric, cls).__new__(cls)

        # Wrap the update_state function in a py_function and scope it to /cpu:0
        obj_update_state = obj.update_state

        def update_state_on_cpu(y_true, y_pred, sample_weight=None):
            with tf.device("/cpu:0"):
                return obj_update_state(y_true, y_pred, sample_weight)

        obj.update_state_on_cpu = update_state_on_cpu

        def update_state_fn(self, y_true, y_pred, sample_weight=None):
            eager_inputs = [y_true, y_pred]
            if sample_weight is not None:
                eager_inputs.append(sample_weight)
            return tf.py_function(
                func=self.update_state_on_cpu, inp=eager_inputs, Tout=[]
            )

        obj.update_state = types.MethodType(update_state_fn, obj)

        # Wrap the result function in a py_function and scope it to /cpu:0
        obj_result = obj.result

        def result_on_host_cpu():
            with tf.device("/cpu:0"):
                return obj_result()

        obj.result_on_host_cpu = result_on_host_cpu

        def result_fn(self):
            return tf.py_function(
                self.result_on_host_cpu, inp=[], Tout=obj.dtype
            )

        obj.result = types.MethodType(result_fn, obj)

        return obj

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates statistics for the metric.

        **Note:** This function is executed outside of the TensorFlow graph
        on the CPU host.

        This means:

        a) Inputs are eager tensors.
        b) Any TensorFlow ops run in this method are run eagerly.
        c) Any Tensors created are allocated to the CPU's main memory.

        Args:
          y_true: Target output
          y_pred: Predicted output
          sample_weight: (Optional) weights for the individual samples in
            `y_true` and `y_pred`
        """
        raise NotImplementedError("Subclasses should implement `update_state`")

    def merge_state(self, metrics):
        """Merges the state from one or more metrics.

        `PyMetric` instances that intend to support merging state must override
         this method, as the default implementation
        in `keras.metrics.Metric` does not apply to `PyMetric`.
        """
        raise NotImplementedError("Subclasses should implement `merge_state`")

    def reset_state(self):
        """Resets all of the metric state variables.

        This function is called between epochs when a metric is evaluated during
        training. It's also called when the metric is initialized.
        """
        raise NotImplementedError("Subclasses should implement `reset_state`")

    def result(self):
        """Computes and returns the scalar metric value.

        **Note:** This function is executed outside of the TensorFlow graph
         on the CPU host. This means any TensorFlow ops run in this method
         are run eagerly.

        Result computation is an idempotent operation that simply calculates the
        metric value using the state variables.

        Returns:
            A Python scalar.
        """
        raise NotImplementedError("Subclasses should implement `result`")
