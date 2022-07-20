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


"""Base Metric classes."""

import abc
import copy
import types
import warnings

import numpy as np
import tensorflow.compat.v2 as tf

from keras import backend
from keras.dtensor import dtensor_api as dtensor
from keras.dtensor import utils as dtensor_utils
from keras.engine import base_layer
from keras.engine import base_layer_utils
from keras.engine import keras_tensor
from keras.saving.saved_model import metric_serialization
from keras.utils import generic_utils
from keras.utils import losses_utils
from keras.utils import metrics_utils
from keras.utils import tf_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls


@keras_export("keras.metrics.Metric")
class Metric(base_layer.Layer, metaclass=abc.ABCMeta):
    """Encapsulates metric logic and state.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      **kwargs: Additional layer keywords arguments.

    Standalone usage:

    ```python
    m = SomeMetric(...)
    for input in ...:
      m.update_state(input)
    print('Final result: ', m.result().numpy())
    ```

    Usage with `compile()` API:

    ```python
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)

    model.fit(dataset, epochs=10)
    ```

    To be implemented by subclasses:
    * `__init__()`: All state variables should be created in this method by
      calling `self.add_weight()` like: `self.var = self.add_weight(...)`
    * `update_state()`: Has all updates to the state variables like:
      self.var.assign_add(...).
    * `result()`: Computes and returns a scalar value or a dict of scalar values
      for the metric from the state variables.

    Example subclass implementation:

    ```python
    class BinaryTruePositives(tf.keras.metrics.Metric):

      def __init__(self, name='binary_true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

      def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
          sample_weight = tf.cast(sample_weight, self.dtype)
          sample_weight = tf.broadcast_to(sample_weight, values.shape)
          values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

      def result(self):
        return self.true_positives
    ```
    """

    def __init__(self, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.stateful = True  # All metric layers are stateful.
        self.built = True
        if not base_layer_utils.v2_dtype_behavior_enabled():
            # We only do this when the V2 behavior is not enabled, as when it is
            # enabled, the dtype already defaults to floatx.
            self._dtype = (
                backend.floatx() if dtype is None else tf.as_dtype(dtype).name
            )

    def __new__(cls, *args, **kwargs):
        obj = super(Metric, cls).__new__(cls)

        # If `update_state` is not in eager/tf.function and it is not from a
        # built-in metric, wrap it in `tf.function`. This is so that users
        # writing custom metrics in v1 need not worry about control dependencies
        # and return ops.
        if base_layer_utils.is_in_eager_or_tf_function() or is_built_in(cls):
            obj_update_state = obj.update_state

            def update_state_fn(*args, **kwargs):
                control_status = tf.__internal__.autograph.control_status_ctx()
                ag_update_state = tf.__internal__.autograph.tf_convert(
                    obj_update_state, control_status
                )
                return ag_update_state(*args, **kwargs)

        else:
            if isinstance(obj.update_state, tf.__internal__.function.Function):
                update_state_fn = obj.update_state
            else:
                update_state_fn = tf.function(obj.update_state)

        obj.update_state = types.MethodType(
            metrics_utils.update_state_wrapper(update_state_fn), obj
        )

        obj_result = obj.result

        def result_fn(*args, **kwargs):
            control_status = tf.__internal__.autograph.control_status_ctx()
            ag_result = tf.__internal__.autograph.tf_convert(
                obj_result, control_status
            )
            return ag_result(*args, **kwargs)

        obj.result = types.MethodType(
            metrics_utils.result_wrapper(result_fn), obj
        )

        return obj

    def __call__(self, *args, **kwargs):
        """Accumulates statistics and then computes metric result value.

        Args:
          *args:
          **kwargs: A mini-batch of inputs to the Metric,
            passed on to `update_state()`.

        Returns:
          The metric value tensor.
        """

        def replica_local_fn(*args, **kwargs):
            """Updates the state of the metric in a replica-local context."""
            if any(
                isinstance(arg, keras_tensor.KerasTensor)
                for arg in tf.nest.flatten((args, kwargs))
            ):
                update_op = None
            else:
                update_op = self.update_state(*args, **kwargs)
            update_ops = []
            if update_op is not None:
                update_ops.append(update_op)
            with tf.control_dependencies(update_ops):
                result_t = self.result()

                # We are adding the metric object as metadata on the result
                # tensor.  This is required when we want to use a metric with
                # `add_metric` API on a Model/Layer in graph mode. This metric
                # instance will later be used to reset variable state after each
                # epoch of training.
                # Example:
                #   model = Model()
                #   mean = Mean()
                #   model.add_metric(mean(values), name='mean')
                result_t._metric_obj = self
                return result_t

        from keras.distribute import (
            distributed_training_utils,
        )

        return distributed_training_utils.call_replica_local_fn(
            replica_local_fn, *args, **kwargs
        )

    def __str__(self):
        args = ",".join(f"{k}={v}" for k, v in self.get_config().items())
        return f"{self.__class__.__name__}({args})"

    def __deepcopy__(self, memo):
        result = type(self)(name=self.name, dtype=self.dtype)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k in ["update_state", "result"]:
                # `update_state` keeps a closure of `update_state_fn`, and deep
                # copying it would result in copying that old reference. Avoid
                # that.  Likewise for `result`.
                continue
            if k in ["_obj_reference_counts_dict"]:
                # `Layer.__setattr__` attempts to flatten the
                # `ObjectIdentityDictionary`, which can't be done since it
                # stores heterogeneous instances.
                tf.Module.__setattr__(result, k, copy.deepcopy(v, memo))
            elif k in ["_thread_local", "_metrics_lock"]:
                # Can't pickle _thread.lock objects.
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))

        return result

    @property
    def dtype(self):
        return self._dtype

    def get_config(self):
        """Returns the serializable config of the metric."""
        return {"name": self.name, "dtype": self.dtype}

    def reset_state(self):
        """Resets all of the metric state variables.

        This function is called between epochs/steps,
        when a metric is evaluated during training.
        """
        if not generic_utils.is_default(self.reset_states):
            warnings.warn(
                "Metric %s implements a `reset_states()` method; rename it "
                'to `reset_state()` (without the final "s"). The name '
                "`reset_states()` has been deprecated to improve API "
                "consistency." % (self.__class__.__name__,),
                stacklevel=2,
            )
            return self.reset_states()
        else:
            backend.batch_set_value([(v, 0) for v in self.variables])

    @abc.abstractmethod
    def update_state(self, *args, **kwargs):
        """Accumulates statistics for the metric.

        Note: This function is executed as a graph function in graph mode.
        This means:
          a) Operations on the same resource are executed in textual order.
             This should make it easier to do things like add the updated
             value of a variable to another, for example.
          b) You don't need to worry about collecting the update ops to execute.
             All update ops added to the graph by this function will be
             executed.
          As a result, code should generally work the same way with graph or
          eager execution.

        Args:
          *args:
          **kwargs: A mini-batch of inputs to the Metric.
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    def merge_state(self, metrics):
        """Merges the state from one or more metrics.

        This method can be used by distributed systems to merge the state
        computed by different metric instances. Typically the state will be
        stored in the form of the metric's weights. For example, a
        tf.keras.metrics.Mean metric contains a list of two weight values: a
        total and a count. If there were two instances of a
        tf.keras.metrics.Accuracy that each independently aggregated partial
        state for an overall accuracy calculation, these two metric's states
        could be combined as follows:

        >>> m1 = tf.keras.metrics.Accuracy()
        >>> _ = m1.update_state([[1], [2]], [[0], [2]])

        >>> m2 = tf.keras.metrics.Accuracy()
        >>> _ = m2.update_state([[3], [4]], [[3], [4]])

        >>> m2.merge_state([m1])
        >>> m2.result().numpy()
        0.75

        Args:
          metrics: an iterable of metrics. The metrics must have compatible
            state.

        Raises:
          ValueError: If the provided iterable does not contain metrics matching
            the metric's required specifications.
        """
        assign_add_ops = []
        for metric in metrics:
            if len(self.weights) != len(metric.weights):
                raise ValueError(
                    f"Metric {metric} is not compatible with {self}"
                )
            for weight, weight_to_add in zip(self.weights, metric.weights):
                assign_add_ops.append(weight.assign_add(weight_to_add))
        return assign_add_ops

    @abc.abstractmethod
    def result(self):
        """Computes and returns the scalar metric value tensor or a dict of
        scalars.

        Result computation is an idempotent operation that simply calculates the
        metric value using the state variables.

        Returns:
          A scalar tensor, or a dictionary of scalar tensors.
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    ### For use by subclasses ###
    @doc_controls.for_subclass_implementers
    def add_weight(
        self,
        name,
        shape=(),
        aggregation=tf.VariableAggregation.SUM,
        synchronization=tf.VariableSynchronization.ON_READ,
        initializer=None,
        dtype=None,
    ):
        """Adds state variable. Only for use by subclasses."""
        if tf.distribute.has_strategy():
            strategy = tf.distribute.get_strategy()
        else:
            strategy = None

        # TODO(b/120571621): Make `ON_READ` work with Keras metrics on TPU.
        if backend.is_tpu_strategy(strategy):
            synchronization = tf.VariableSynchronization.ON_WRITE
        if getattr(self, "_mesh", None) is not None:
            # When self._mesh is set, it means this metric is used for DTensor.
            additional_kwargs = {
                "layout": dtensor.Layout.replicated(
                    self._mesh, tf.TensorShape(shape).rank
                )
            }
        else:
            additional_kwargs = {}

        with tf_utils.maybe_init_scope(layer=self):
            return super().add_weight(
                name=name,
                shape=shape,
                dtype=self._dtype if dtype is None else dtype,
                trainable=False,
                initializer=initializer,
                collections=[],
                synchronization=synchronization,
                aggregation=aggregation,
                **additional_kwargs,
            )

    ### End: For use by subclasses ###

    @property
    def trainable_weights(self):
        # Overridden from Layer class to track submetric weights.
        if self.trainable:
            trainable_weights = self._trainable_weights
            for m in self._metrics:
                trainable_weights += m.trainable_weights
            return self._dedup_weights(trainable_weights)
        else:
            return []

    @property
    def non_trainable_weights(self):
        # Overridden from Layer class to track submetric weights.
        if self.trainable:
            non_trainable_weights = self._non_trainable_weights
            for m in self._metrics:
                non_trainable_weights += m.non_trainable_weights
        else:
            non_trainable_weights = (
                self._non_trainable_weights + self._trainable_weights
            )
            for m in self._metrics:
                non_trainable_weights += m.weights
        return self._dedup_weights(non_trainable_weights)

    @property
    def _trackable_saved_model_saver(self):
        return metric_serialization.MetricSavedModelSaver(self)

    @generic_utils.default
    @doc_controls.do_not_generate_docs
    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        return self.reset_state()


class Reduce(Metric):
    """Encapsulates metrics that perform a reduce operation on the values.

    Args:
      reduction: a `tf.keras.metrics.Reduction` enum value.
      name: string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """

    def __init__(self, reduction, name, dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.reduction = reduction
        self.total = self.add_weight("total", initializer="zeros")
        if reduction in [
            metrics_utils.Reduction.SUM_OVER_BATCH_SIZE,
            metrics_utils.Reduction.WEIGHTED_MEAN,
        ]:
            self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, values, sample_weight=None):
        """Accumulates statistics for computing the metric.

        Args:
          values: Per-example value.
          sample_weight: Optional weighting of each example. Defaults to 1.

        Returns:
          Update op.
        """
        [
            values
        ], sample_weight = metrics_utils.ragged_assert_compatible_and_get_flat_values(  # noqa: E501
            [values], sample_weight
        )
        try:
            values = tf.cast(values, self._dtype)
        except (ValueError, TypeError):
            msg = (
                "The output of a metric function can only be a single Tensor. "
                f"Received: {values}. "
            )
            if isinstance(values, dict):
                msg += (
                    "To return a dict of values, implement a custom Metric "
                    "subclass."
                )
            raise RuntimeError(msg)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)
            # Update dimensions of weights to match with values if possible.
            (
                values,
                _,
                sample_weight,
            ) = losses_utils.squeeze_or_expand_dimensions(
                values, sample_weight=sample_weight
            )
            try:
                # Broadcast weights if possible.
                sample_weight = tf.__internal__.ops.broadcast_weights(
                    sample_weight, values
                )
            except ValueError:
                # Reduce values to same ndim as weight array
                ndim = backend.ndim(values)
                weight_ndim = backend.ndim(sample_weight)
                if self.reduction == metrics_utils.Reduction.SUM:
                    values = tf.reduce_sum(
                        values, axis=list(range(weight_ndim, ndim))
                    )
                else:
                    values = tf.reduce_mean(
                        values, axis=list(range(weight_ndim, ndim))
                    )
            values = tf.multiply(values, sample_weight)

        value_sum = tf.reduce_sum(values)
        with tf.control_dependencies([value_sum]):
            update_total_op = self.total.assign_add(value_sum)

        # Exit early if the reduction doesn't have a denominator.
        if self.reduction == metrics_utils.Reduction.SUM:
            return update_total_op

        # Update `count` for reductions that require a denominator.
        if self.reduction == metrics_utils.Reduction.SUM_OVER_BATCH_SIZE:
            num_values = tf.cast(tf.size(values), self._dtype)
        elif self.reduction == metrics_utils.Reduction.WEIGHTED_MEAN:
            if sample_weight is None:
                num_values = tf.cast(tf.size(values), self._dtype)
            else:
                num_values = tf.reduce_sum(sample_weight)
        else:
            raise NotImplementedError(
                f'Reduction "{self.reduction}" not implemented. Expected '
                '"sum", "weighted_mean", or "sum_over_batch_size".'
            )

        with tf.control_dependencies([update_total_op]):
            return self.count.assign_add(num_values)

    def result(self):
        if self.reduction == metrics_utils.Reduction.SUM:
            return tf.identity(self.total)
        elif self.reduction in [
            metrics_utils.Reduction.WEIGHTED_MEAN,
            metrics_utils.Reduction.SUM_OVER_BATCH_SIZE,
        ]:
            return tf.math.divide_no_nan(self.total, self.count)
        else:
            raise NotImplementedError(
                f'Reduction "{self.reduction}" not implemented. Expected '
                '"sum", "weighted_mean", or "sum_over_batch_size".'
            )


@keras_export("keras.metrics.Sum")
class Sum(Reduce):
    """Computes the (weighted) sum of the given values.

    For example, if values is [1, 3, 5, 7] then the sum is 16.
    If the weights were specified as [1, 1, 0, 0] then the sum would be 4.

    This metric creates one variable, `total`, that is used to compute the sum
    of `values`. This is ultimately returned as `sum`.

    If `sample_weight` is `None`, weights default to 1.  Use `sample_weight` of
    0 to mask values.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.Sum()
    >>> m.update_state([1, 3, 5, 7])
    >>> m.result().numpy()
    16.0

    Usage with `compile()` API:

    ```python
    model.add_metric(tf.keras.metrics.Sum(name='sum_1')(outputs))
    model.compile(optimizer='sgd', loss='mse')
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="sum", dtype=None):
        super().__init__(
            reduction=metrics_utils.Reduction.SUM, name=name, dtype=dtype
        )


@keras_export("keras.metrics.Mean")
class Mean(Reduce):
    """Computes the (weighted) mean of the given values.

    For example, if values is [1, 3, 5, 7] then the mean is 4.
    If the weights were specified as [1, 1, 0, 0] then the mean would be 2.

    This metric creates two variables, `total` and `count` that are used to
    compute the average of `values`. This average is ultimately returned as
    `mean` which is an idempotent operation that simply divides `total` by
    `count`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = tf.keras.metrics.Mean()
    >>> m.update_state([1, 3, 5, 7])
    >>> m.result().numpy()
    4.0
    >>> m.reset_state()
    >>> m.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0])
    >>> m.result().numpy()
    2.0

    Usage with `compile()` API:

    ```python
    model.add_metric(tf.keras.metrics.Mean(name='mean_1')(outputs))
    model.compile(optimizer='sgd', loss='mse')
    ```
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="mean", dtype=None):
        super().__init__(
            reduction=metrics_utils.Reduction.WEIGHTED_MEAN,
            name=name,
            dtype=dtype,
        )


@keras_export("keras.metrics.MeanMetricWrapper")
class MeanMetricWrapper(Mean):
    """Wraps a stateless metric function with the Mean metric.

    You could use this class to quickly build a mean metric from a function. The
    function needs to have the signature `fn(y_true, y_pred)` and return a
    per-sample loss array. `MeanMetricWrapper.result()` will return
    the average metric value across all samples seen so far.

    For example:

    ```python
    def accuracy(y_true, y_pred):
      return tf.cast(tf.math.equal(y_true, y_pred), tf.float32)

    accuracy_metric = tf.keras.metrics.MeanMetricWrapper(fn=accuracy)

    keras_model.compile(..., metrics=accuracy_metric)
    ```

    Args:
      fn: The metric function to wrap, with signature `fn(y_true, y_pred,
        **kwargs)`.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      **kwargs: Keyword arguments to pass on to `fn`.
    """

    @dtensor_utils.inject_mesh
    def __init__(self, fn, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype)
        self._fn = fn
        self._fn_kwargs = kwargs

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates metric statistics.

        `y_true` and `y_pred` should have the same shape.

        Args:
          y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
          y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
          sample_weight: Optional `sample_weight` acts as a
            coefficient for the metric. If a scalar is provided, then the metric
            is simply scaled by the given value. If `sample_weight` is a tensor
            of size `[batch_size]`, then the metric for each sample of the batch
            is rescaled by the corresponding element in the `sample_weight`
            vector. If the shape of `sample_weight` is `[batch_size, d0, ..
            dN-1]` (or can be broadcasted to this shape), then each metric
            element of `y_pred` is scaled by the corresponding value of
            `sample_weight`. (Note on `dN-1`: all metric functions reduce by 1
            dimension, usually the last axis (-1)).

        Returns:
          Update op.
        """
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        [
            y_true,
            y_pred,
        ], sample_weight = metrics_utils.ragged_assert_compatible_and_get_flat_values(  # noqa: E501
            [y_true, y_pred], sample_weight
        )
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
            y_pred, y_true
        )

        ag_fn = tf.__internal__.autograph.tf_convert(
            self._fn, tf.__internal__.autograph.control_status_ctx()
        )
        matches = ag_fn(y_true, y_pred, **self._fn_kwargs)
        mask = losses_utils.get_mask(matches)
        sample_weight = losses_utils.apply_valid_mask(
            matches, sample_weight, mask, self.reduction
        )
        return super().update_state(matches, sample_weight=sample_weight)

    def get_config(self):
        config = {
            k: backend.eval(v) if tf_utils.is_tensor_or_variable(v) else v
            for k, v in self._fn_kwargs.items()
        }

        if type(self) is MeanMetricWrapper:
            # Only include function argument when the object is a
            # MeanMetricWrapper and not a subclass.
            config["fn"] = self._fn

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        from keras.metrics import get

        # Note that while MeanMetricWrapper itself isn't public, objects of this
        # class may be created and added to the model by calling model.compile.
        fn = config.pop("fn", None)
        if cls is MeanMetricWrapper:
            return cls(get(fn), **config)
        return super(MeanMetricWrapper, cls).from_config(config)


@keras_export("keras.metrics.MeanTensor")
class MeanTensor(Metric):
    """Computes the element-wise (weighted) mean of the given tensors.

    `MeanTensor` returns a tensor with the same shape of the input tensors. The
    mean value is updated by keeping local variables `total` and `count`. The
    `total` tracks the sum of the weighted values, and `count` stores the sum of
    the weighted counts.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      shape: (Optional) A list of integers, a tuple of integers, or a 1-D Tensor
        of type int32. If not specified, the shape is inferred from the values
        at the first call of update_state.

    Standalone usage:

    >>> m = tf.keras.metrics.MeanTensor()
    >>> m.update_state([0, 1, 2, 3])
    >>> m.update_state([4, 5, 6, 7])
    >>> m.result().numpy()
    array([2., 3., 4., 5.], dtype=float32)

    >>> m.update_state([12, 10, 8, 6], sample_weight= [0, 0.2, 0.5, 1])
    >>> m.result().numpy()
    array([2.       , 3.6363635, 4.8      , 5.3333335], dtype=float32)

    >>> m = tf.keras.metrics.MeanTensor(dtype=tf.float64, shape=(1, 4))
    >>> m.result().numpy()
    array([[0., 0., 0., 0.]])
    >>> m.update_state([[0, 1, 2, 3]])
    >>> m.update_state([[4, 5, 6, 7]])
    >>> m.result().numpy()
    array([[2., 3., 4., 5.]])
    """

    @dtensor_utils.inject_mesh
    def __init__(self, name="mean_tensor", dtype=None, shape=None):
        super().__init__(name=name, dtype=dtype)
        self._shape = None
        self._total = None
        self._count = None
        self._built = False
        if shape is not None:
            self._build(shape)

    def _build(self, shape):
        self._shape = tf.TensorShape(shape)
        self._build_input_shape = self._shape
        # Create new state variables
        self._total = self.add_weight(
            name="total", shape=shape, initializer="zeros"
        )
        self._count = self.add_weight(
            name="count", shape=shape, initializer="zeros"
        )
        with tf.init_scope():
            if not tf.executing_eagerly():
                backend._initialize_variables(backend._get_session())
        self._built = True

    @property
    def total(self):
        return self._total if self._built else None

    @property
    def count(self):
        return self._count if self._built else None

    def update_state(self, values, sample_weight=None):
        """Accumulates statistics for computing the element-wise mean.

        Args:
          values: Per-example value.
          sample_weight: Optional weighting of each example. Defaults to 1.

        Returns:
          Update op.
        """
        values = tf.cast(values, self._dtype)
        if not self._built:
            self._build(values.shape)
        elif values.shape != self._shape:
            raise ValueError(
                "MeanTensor input values must always have the same "
                f"shape. Expected shape (set during the first call): "
                f"{self._shape}. "
                f"Got: {values.shape}."
            )

        num_values = tf.ones_like(values)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)

            # Update dimensions of weights to match with values if possible.
            (
                values,
                _,
                sample_weight,
            ) = losses_utils.squeeze_or_expand_dimensions(
                values, sample_weight=sample_weight
            )
            try:
                # Broadcast weights if possible.
                sample_weight = tf.__internal__.ops.broadcast_weights(
                    sample_weight, values
                )
            except ValueError:
                # Reduce values to same ndim as weight array
                ndim = backend.ndim(values)
                weight_ndim = backend.ndim(sample_weight)
                values = tf.reduce_mean(
                    values, axis=list(range(weight_ndim, ndim))
                )

            num_values = tf.multiply(num_values, sample_weight)
            values = tf.multiply(values, sample_weight)

        update_total_op = self._total.assign_add(values)
        with tf.control_dependencies([update_total_op]):
            return self._count.assign_add(num_values)

    def result(self):
        if not self._built:
            raise ValueError(
                "MeanTensor does not have any value yet. Please call the "
                "MeanTensor instance or use `.update_state(value)` "
                "before retrieving the result."
            )
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        if self._built:
            backend.batch_set_value(
                [(v, np.zeros(v.shape.as_list())) for v in self.variables]
            )


class SumOverBatchSize(Reduce):
    """Computes the weighted sum over batch size of the given values.

    For example, if values is [1, 3, 5, 7] then the metric value is 4.
    If the weights were specified as [1, 1, 0, 0] then the value would be 1.

    This metric creates two variables, `total` and `count` that are used to
    compute the average of `values`. This average is ultimately returned as sum
    over batch size which is an idempotent operation that simply divides `total`
    by `count`.

    If `sample_weight` is `None`, weights default to 1.  Use `sample_weight` of
    0 to mask values.
    """

    def __init__(self, name="sum_over_batch_size", dtype=None):
        super().__init__(
            reduction=metrics_utils.Reduction.SUM_OVER_BATCH_SIZE,
            name=name,
            dtype=dtype,
        )


class SumOverBatchSizeMetricWrapper(SumOverBatchSize):
    """Wraps a function with the `SumOverBatchSizeMetricWrapper` metric."""

    def __init__(self, fn, name=None, dtype=None, **kwargs):
        """Creates a `SumOverBatchSizeMetricWrapper` instance.

        Args:
          fn: The metric function to wrap, with signature `fn(y_true, y_pred,
            **kwargs)`.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
          **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super().__init__(name=name, dtype=dtype)
        self._fn = fn
        self._fn_kwargs = kwargs

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
            y_pred, y_true
        )

        ag_fn = tf.__internal__.autograph.tf_convert(
            self._fn, tf.__internal__.autograph.control_status_ctx()
        )
        matches = ag_fn(y_true, y_pred, **self._fn_kwargs)
        mask = losses_utils.get_mask(matches)
        sample_weight = losses_utils.apply_valid_mask(
            matches, sample_weight, mask, self.reduction
        )
        return super().update_state(matches, sample_weight=sample_weight)

    def get_config(self):
        config = {
            k: backend.eval(v) if tf_utils.is_tensor_or_variable(v) else v
            for k, v in self._fn_kwargs.items()
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def clone_metric(metric):
    """Returns a clone of the metric if stateful, otherwise returns it as is."""
    if isinstance(metric, Metric):
        with tf.init_scope():
            return metric.__class__.from_config(metric.get_config())
    return metric


def clone_metrics(metrics):
    """Clones the given metric list/dict."""
    return tf.nest.map_structure(clone_metric, metrics)


def is_built_in(cls):
    return cls.__module__.startswith(
        ".".join(Metric.__module__.split(".")[:-1])
    )
