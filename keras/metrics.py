"""Built-in metrics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import six
import types

from . import backend as K
from .layers import Layer
from .losses import mean_squared_error
from .losses import mean_absolute_error
from .losses import mean_absolute_percentage_error
from .losses import mean_squared_logarithmic_error
from .losses import hinge
from .losses import logcosh
from .losses import squared_hinge
from .losses import categorical_hinge
from .losses import categorical_crossentropy
from .losses import sparse_categorical_crossentropy
from .losses import binary_crossentropy
from .losses import kullback_leibler_divergence
from .losses import poisson
from .utils import losses_utils
from .utils import metrics_utils
from .utils.generic_utils import deserialize_keras_object
from .utils.generic_utils import serialize_keras_object


@six.add_metaclass(abc.ABCMeta)
class Metric(Layer):
    """Encapsulates metric logic and state.

    Standalone usage:
    ```python
    m = SomeMetric(...)
    for input in ...:
        m.update_state(input)
    m.result()
    ```

    Usage with the `compile` API:
    ```python
    model.compile(optimizer='rmsprop',
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.CategoricalAccuracy()])
    ```

    To be implemented by subclasses:
    * `__init__()`: All state variables should be created in this method by
        calling `self.add_weight()` like: `self.var = self.add_weight(...)`
    * `update_state()`: Has all updates to the state variables like:
        self.var.assign_add(...).
    * `result()`: Computes and returns a value for the metric
        from the state variables.
    """

    def __init__(self, name=None, dtype=None, **kwargs):
        super(Metric, self).__init__(name=name, dtype=dtype, **kwargs)
        self.stateful = True  # All metric layers are stateful.
        self.built = True
        self.dtype = dtype or K.floatx()

    def __new__(cls, *args, **kwargs):
        obj = super(Metric, cls).__new__(cls)

        obj.update_state = types.MethodType(
            metrics_utils.update_state_wrapper(obj.update_state), obj)

        obj.result = types.MethodType(
            metrics_utils.result_wrapper(obj.result), obj)
        return obj

    @K.symbolic
    def __call__(self, *args, **kwargs):
        """Accumulates statistics and then computes metric result value."""
        update_op = self.update_state(*args, **kwargs)
        with K.control_dependencies(update_op):  # For TF
            result_t = self.result()

            # We are adding the metric object as metadata on the result tensor.
            # This is required when we want to use a metric with `add_metric` API on
            # a Model/Layer in graph mode. This metric instance will later be used
            # to reset variable state after each epoch of training.
            # Example:
            #   model = Model()
            #   mean = Mean()
            #   model.add_metric(mean(values), name='mean')
            result_t._metric_obj = self
            return result_t

    def get_config(self):
        """Returns the serializable config of the metric."""
        return {'name': self.name, 'dtype': self.dtype}

    def reset_states(self):
        """Resets all of the metric state variables.

        This function is called between epochs/steps,
        when a metric is evaluated during training.
        """
        K.batch_set_value([(v, 0) for v in self.weights])

    @abc.abstractmethod
    def update_state(self, *args, **kwargs):
        """Accumulates statistics for the metric. """
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def result(self):
        """Computes and returns the metric value tensor.

        Result computation is an idempotent operation that simply calculates the
        metric value using the state variables.
        """
        raise NotImplementedError('Must be implemented in subclasses.')

    # For use by subclasses #
    def add_weight(self,
                   name,
                   shape=(),
                   initializer=None,
                   dtype=None):
        """Adds state variable. Only for use by subclasses."""
        return super(Metric, self).add_weight(
            name=name,
            shape=shape,
            dtype=self.dtype if dtype is None else dtype,
            trainable=False,
            initializer=initializer)

    # End: For use by subclasses ###


class Reduce(Metric):
    """Encapsulates metrics that perform a reduce operation on the values."""

    def __init__(self, reduction, name, dtype=None):
        """Creates a `Reduce` instance.

        # Arguments
            reduction: a metrics `Reduction` enum value.
            name: string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super(Reduce, self).__init__(name=name, dtype=dtype)
        self.reduction = reduction
        self.total = self.add_weight(
            'total', initializer='zeros')
        if reduction in [metrics_utils.Reduction.SUM_OVER_BATCH_SIZE,
                         metrics_utils.Reduction.WEIGHTED_MEAN]:
            self.count = self.add_weight(
                'count', initializer='zeros')

    def update_state(self, values, sample_weight=None):
        """Accumulates statistics for computing the reduction metric.

        For example, if `values` is [1, 3, 5, 7] and reduction=SUM_OVER_BATCH_SIZE,
        then the value of `result()` is 4. If the `sample_weight` is specified as
        [1, 1, 0, 0] then value of `result()` would be 2.

        # Arguments
            values: Per-example value.
            sample_weight: Optional weighting of each example. Defaults to 1.

        # Returns
            List of update ops.
        """
        values = K.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = K.cast(sample_weight, self.dtype)

            # Update dimensions of weights to match with values if possible.
            values, _, sample_weight = losses_utils.squeeze_or_expand_dimensions(
                values, sample_weight=sample_weight)

            # Broadcast weights if possible.
            sample_weight = losses_utils.broadcast_weights(values, sample_weight)

            values = values * sample_weight

        value_sum = K.sum(values)
        update_total_op = K.update_add(self.total, value_sum)

        # Exit early if the reduction doesn't have a denominator.
        if self.reduction == metrics_utils.Reduction.SUM:
            return [update_total_op]

        # Update `count` for reductions that require a denominator.
        if self.reduction == metrics_utils.Reduction.SUM_OVER_BATCH_SIZE:
            num_values = K.cast(K.size(values), self.dtype)
        elif self.reduction == metrics_utils.Reduction.WEIGHTED_MEAN:
            if sample_weight is None:
                num_values = K.cast(K.size(values), self.dtype)
            else:
                num_values = K.sum(sample_weight)
        else:
            raise NotImplementedError(
                'reduction [%s] not implemented' % self.reduction)

        return [update_total_op, K.update_add(self.count, num_values)]

    def result(self):
        if self.reduction == metrics_utils.Reduction.SUM:
            return self.total
        elif self.reduction in [
            metrics_utils.Reduction.WEIGHTED_MEAN,
            metrics_utils.Reduction.SUM_OVER_BATCH_SIZE
        ]:
            return self.total / self.count
        else:
            raise NotImplementedError(
                'reduction [%s] not implemented' % self.reduction)


class Sum(Reduce):
    """Computes the (weighted) sum of the given values.

    For example, if values is [1, 3, 5, 7] then the sum is 16.
    If the weights were specified as [1, 1, 0, 0] then the sum would be 4.

    This metric creates one variable, `total`, that is used to compute the sum of
    `values`. This is ultimately returned as `sum`.
    If `sample_weight` is `None`, weights default to 1.  Use `sample_weight` of 0
    to mask values.

    Standalone usage:
    ```python
    m = keras.metrics.Sum()
    m.update_state([1, 3, 5, 7])
    m.result()
    ```
    """

    def __init__(self, name='sum', dtype=None):
        """Creates a `Sum` instance.

        # Arguments
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super(Sum, self).__init__(reduction=metrics_utils.Reduction.SUM,
                                  name=name, dtype=dtype)


class Mean(Reduce):
    """Computes the (weighted) mean of the given values.

    For example, if values is [1, 3, 5, 7] then the mean is 4.
    If the weights were specified as [1, 1, 0, 0] then the mean would be 2.

    This metric creates two variables, `total` and `count` that are used to
    compute the average of `values`. This average is ultimately returned as `mean`
    which is an idempotent operation that simply divides `total` by `count`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage:

    ```python
    m = keras.metrics.Mean()
    m.update_state([1, 3, 5, 7])
    m.result()
    ```
    """

    def __init__(self, name='mean', dtype=None):
        """Creates a `Mean` instance.

        #Arguments
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super(Mean, self).__init__(
            reduction=metrics_utils.Reduction.WEIGHTED_MEAN, name=name, dtype=dtype)


class MeanMetricWrapper(Mean):
    """Wraps a stateless metric function with the Mean metric."""

    def __init__(self, fn, name=None, dtype=None, **kwargs):
        """Creates a `MeanMetricWrapper` instance.

        # Arguments
            fn: The metric function to wrap, with signature
                `fn(y_true, y_pred, **kwargs)`.
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
            **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)
        self._fn = fn
        self._fn_kwargs = kwargs

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates metric statistics.

        `y_true` and `y_pred` should have the same shape.

        # Arguments
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Defaults to 1. Can be
                a `Tensor` whose rank is either 0, or the same rank as `y_true`,
                and must be broadcastable to `y_true`.

        # Returns
            Update op.
        """
        y_true = K.cast(y_true, self.dtype)
        y_pred = K.cast(y_pred, self.dtype)
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)

        matches = self._fn(y_true, y_pred, **self._fn_kwargs)
        return super(MeanMetricWrapper, self).update_state(
            matches, sample_weight=sample_weight)

    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = K.eval(v) if K.is_tensor(v) else v
        base_config = super(MeanMetricWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MeanSquaredError(MeanMetricWrapper):
    """Computes the mean squared error between `y_true` and `y_pred`.

    Standalone usage:

    ```python
    m = keras.metrics.MeanSquaredError()
    m.update_state([0., 0., 1., 1.], [1., 1., 1., 0.])
    m.result()
    ```

    Usage with compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', metrics=[keras.metrics.MeanSquaredError()])
    ```
    """

    def __init__(self, name='mean_squared_error', dtype=None):
        super(MeanSquaredError, self).__init__(
            mean_squared_error, name, dtype=dtype)


class Hinge(MeanMetricWrapper):
    """Computes the hinge metric between `y_true` and `y_pred`.

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.
    For example, if `y_true` is [-1., 1., 1.], and `y_pred` is [0.6, -0.7, -0.5]
    the hinge metric value is 1.6.

    Usage:

    ```python
    m = keras.metrics.Hinge()
    m.update_state([-1., 1., 1.], [0.6, -0.7, -0.5])
    # result = max(0, 1-y_true * y_pred) = [1.6 + 1.7 + 1.5] / 3
    # Final result: 1.6
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', metrics=[keras.metrics.Hinge()])
    ```
    """

    def __init__(self, name='hinge', dtype=None):
        super(Hinge, self).__init__(hinge, name, dtype=dtype)


class SquaredHinge(MeanMetricWrapper):
    """Computes the squared hinge metric between `y_true` and `y_pred`.

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.
    For example, if `y_true` is [-1., 1., 1.], and `y_pred` is [0.6, -0.7, -0.5]
    the squared hinge metric value is 2.6.

    Usage:

    ```python
    m = keras.metrics.SquaredHinge()
    m.update_state([-1., 1., 1.], [0.6, -0.7, -0.5])
    # result = max(0, 1-y_true * y_pred) = [1.6^2 + 1.7^2 + 1.5^2] / 3
    # Final result: 2.6
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', metrics=[keras.metrics.SquaredHinge()])
    ```
    """

    def __init__(self, name='squared_hinge', dtype=None):
        super(SquaredHinge, self).__init__(squared_hinge, name, dtype=dtype)


class CategoricalHinge(MeanMetricWrapper):
    """Computes the categorical hinge metric between `y_true` and `y_pred`.

    For example, if `y_true` is [0., 1., 1.], and `y_pred` is [1., 0., 1.]
    the categorical hinge metric value is 1.0.

    Usage:

    ```python
    m = keras.metrics.CategoricalHinge()
    m.update_state([0., 1., 1.], [1., 0., 1.])
    # Final result: 1.0
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', metrics=[keras.metrics.CategoricalHinge()])
    ```
    """

    def __init__(self, name='categorical_hinge', dtype=None):
        super(CategoricalHinge, self).__init__(
            categorical_hinge, name, dtype=dtype)


class Accuracy(MeanMetricWrapper):
    """Calculates how often predictions matches labels.

    For example, if `y_true` is [1, 2, 3, 4] and `y_pred` is [0, 2, 3, 4]
    then the accuracy is 3/4 or .75.  If the weights were specified as
    [1, 1, 0, 0] then the accuracy would be 1/2 or .5.

    This metric creates two local variables, `total` and `count` that are used to
    compute the frequency with which `y_pred` matches `y_true`. This frequency is
    ultimately returned as `binary accuracy`: an idempotent operation that simply
    divides `total` by `count`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    ```

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss='mse', metrics=[keras.metrics.Accuracy()])
    ```
    """

    def __init__(self, name='accuracy', dtype=None):
        super(Accuracy, self).__init__(accuracy, name, dtype=dtype)


class BinaryAccuracy(MeanMetricWrapper):
    """Calculates how often predictions matches labels.

    For example, if `y_true` is [1, 1, 0, 0] and `y_pred` is [0.98, 1, 0, 0.6]
    then the binary accuracy is 3/4 or .75.  If the weights were specified as
    [1, 0, 0, 1] then the binary accuracy would be 1/2 or .5.

    This metric creates two local variables, `total` and `count` that are used to
    compute the frequency with which `y_pred` matches `y_true`. This frequency is
    ultimately returned as `binary accuracy`: an idempotent operation that simply
    divides `total` by `count`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss='mse', metrics=[keras.metrics.BinaryAccuracy()])
    ```

    # Arguments
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        threshold: (Optional) Float representing the threshold for deciding
            whether prediction values are 1 or 0.
    """

    def __init__(self, name='binary_accuracy', dtype=None, threshold=0.5):
        super(BinaryAccuracy, self).__init__(
            binary_accuracy, name, dtype=dtype, threshold=threshold)


class CategoricalAccuracy(MeanMetricWrapper):
    """Calculates how often predictions matches labels.

    For example, if `y_true` is [[0, 0, 1], [0, 1, 0]] and `y_pred` is
    [[0.1, 0.9, 0.8], [0.05, 0.95, 0]] then the categorical accuracy is 1/2 or .5.
    If the weights were specified as [0.7, 0.3] then the categorical accuracy
    would be .3. You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    This metric creates two local variables, `total` and `count` that are used to
    compute the frequency with which `y_pred` matches `y_true`. This frequency is
    ultimately returned as `categorical accuracy`: an idempotent operation that
    simply divides `total` by `count`.

    `y_pred` and `y_true` should be passed in as vectors of probabilities, rather
    than as labels. If necessary, use `K.one_hot` to expand `y_true` as a vector.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile(
        'sgd',
        loss='mse',
        metrics=[keras.metrics.CategoricalAccuracy()])
    ```

    # Arguments
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(self, name='categorical_accuracy', dtype=None):
        super(CategoricalAccuracy, self).__init__(
            categorical_accuracy, name, dtype=dtype)


class SparseCategoricalAccuracy(MeanMetricWrapper):
    """Calculates how often predictions matches integer labels.

    For example, if `y_true` is [[2], [1]] and `y_pred` is
    [[0.1, 0.9, 0.8], [0.05, 0.95, 0]] then the categorical accuracy is 1/2 or .5.
    If the weights were specified as [0.7, 0.3] then the categorical accuracy
    would be .3. You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    This metric creates two local variables, `total` and `count` that are used to
    compute the frequency with which `y_pred` matches `y_true`. This frequency is
    ultimately returned as `sparse categorical accuracy`: an idempotent operation
    that simply divides `total` by `count`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile(
        'sgd',
        loss='mse',
        metrics=[keras.metrics.SparseCategoricalAccuracy()])
    ```
    """

    def __init__(self, name='sparse_categorical_accuracy', dtype=None):
        super(SparseCategoricalAccuracy, self).__init__(
            sparse_categorical_accuracy, name, dtype=dtype)


class TopKCategoricalAccuracy(MeanMetricWrapper):
    """Computes how often targets are in the top `K` predictions.

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', metrics=[keras.metrics.TopKCategoricalAccuracy()])
    ```

    # Arguments
        k: (Optional) Number of top elements to look at for computing accuracy.
            Defaults to 5.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(self, k=5, name='top_k_categorical_accuracy', dtype=None):
        super(TopKCategoricalAccuracy, self).__init__(
            top_k_categorical_accuracy, name, dtype=dtype, k=k)


class SparseTopKCategoricalAccuracy(MeanMetricWrapper):
    """Computes how often integer targets are in the top `K` predictions.

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile(
        'sgd',
        metrics=[keras.metrics.SparseTopKCategoricalAccuracy()])
    ```

    # Arguments
        k: (Optional) Number of top elements to look at for computing accuracy.
            Defaults to 5.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(self, k=5, name='sparse_top_k_categorical_accuracy', dtype=None):
        super(SparseTopKCategoricalAccuracy, self).__init__(
            sparse_top_k_categorical_accuracy, name, dtype=dtype, k=k)


class LogCoshError(MeanMetricWrapper):
    """Computes the logarithm of the hyperbolic cosine of the prediction error.

    `metric = log((exp(x) + exp(-x))/2)`, where x is the error (y_pred - y_true)

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', metrics=[keras.metrics.LogCoshError()])
    ```
    """

    def __init__(self, name='logcosh', dtype=None):
        super(LogCoshError, self).__init__(logcosh, name, dtype=dtype)


class Poisson(MeanMetricWrapper):
    """Computes the Poisson metric between `y_true` and `y_pred`.

    `metric = y_pred - y_true * log(y_pred)`

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', metrics=[keras.metrics.Poisson()])
    ```
    """

    def __init__(self, name='poisson', dtype=None):
        super(Poisson, self).__init__(poisson, name, dtype=dtype)


class KLDivergence(MeanMetricWrapper):
    """Computes Kullback-Leibler divergence metric between `y_true` and `y_pred`.

    `metric = y_true * log(y_true / y_pred)`

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', metrics=[keras.metrics.KLDivergence()])
    ```
    """

    def __init__(self, name='kullback_leibler_divergence', dtype=None):
        super(KLDivergence, self).__init__(
            kullback_leibler_divergence, name, dtype=dtype)


class CosineSimilarity(MeanMetricWrapper):
    """Computes the cosine similarity between the labels and predictions.

    cosine similarity = (a . b) / ||a|| ||b||
    [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
    For example, if `y_true` is [0, 1, 1], and `y_pred` is [1, 0, 1], the cosine
    similarity is 0.5.

    This metric keeps the average cosine similarity between `predictions` and
    `labels` over a stream of data.

    # Arguments
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        axis: (Optional) Defaults to -1. The dimension along which the cosine
        similarity is computed.

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile(
      'sgd',
      loss='mse',
      metrics=[keras.metrics.CosineSimilarity(axis=1)])
    ```
    """

    def __init__(self, name='cosine_similarity', dtype=None, axis=-1):
        super(CosineSimilarity, self).__init__(
            cosine_similarity, name, dtype=dtype, axis=axis)


class MeanAbsoluteError(MeanMetricWrapper):
    """Computes the mean absolute error between the labels and predictions.

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', metrics=[keras.metrics.MeanAbsoluteError()])
    ```
    """

    def __init__(self, name='mean_absolute_error', dtype=None):
        super(MeanAbsoluteError, self).__init__(
            mean_absolute_error, name, dtype=dtype)


class MeanAbsolutePercentageError(MeanMetricWrapper):
    """Computes the mean absolute percentage error between `y_true` and `y_pred`.

    For example, if `y_true` is [0., 0., 1., 1.], and `y_pred` is [1., 1., 1., 0.]
    the mean absolute percentage error is 5e+08.

    Usage:

    ```python
    m = keras.metrics.MeanAbsolutePercentageError()
    m.update_state([0., 0., 1., 1.], [1., 1., 1., 0.])
    # Final result: 5e+08
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', metrics=[keras.metrics.MeanAbsolutePercentageError()])
    ```
    """

    def __init__(self, name='mean_absolute_percentage_error', dtype=None):
        super(MeanAbsolutePercentageError, self).__init__(
            mean_absolute_percentage_error, name, dtype=dtype)


class MeanSquaredError(MeanMetricWrapper):
    """Computes the mean squared error between `y_true` and `y_pred`.

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', metrics=[keras.metrics.MeanSquaredError()])
    ```
    """

    def __init__(self, name='mean_squared_error', dtype=None):
        super(MeanSquaredError, self).__init__(
            mean_squared_error, name, dtype=dtype)


class MeanSquaredLogarithmicError(MeanMetricWrapper):
    """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', metrics=[keras.metrics.MeanSquaredLogarithmicError()])
    ```
    """

    def __init__(self, name='mean_squared_logarithmic_error', dtype=None):
        super(MeanSquaredLogarithmicError, self).__init__(
            mean_squared_logarithmic_error, name, dtype=dtype)


class RootMeanSquaredError(Mean):
    """Computes root mean squared error metric between `y_true` and `y_pred`.

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', metrics=[keras.metrics.RootMeanSquaredError()])
    ```
    """

    def __init__(self, name='root_mean_squared_error', dtype=None):
        super(RootMeanSquaredError, self).__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates root mean squared error statistics.

        # Arguments
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Defaults to 1.
                Can be a `Tensor` whose rank is either 0,
                or the same rank as `y_true`,
                and must be broadcastable to `y_true`.

        # Returns
            List of update ops.
        """
        error_sq = K.square(y_pred - y_true)
        return super(RootMeanSquaredError, self).update_state(
            error_sq, sample_weight=sample_weight)

    def result(self):
        return K.sqrt(self.total / self.count)


class BinaryCrossentropy(MeanMetricWrapper):
    """Computes the crossentropy metric between the labels and predictions.

    This is the crossentropy metric class to be used when there are only two
    label classes (0 and 1).

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile(
      'sgd',
      loss='mse',
      metrics=[keras.metrics.BinaryCrossentropy()])
    ```

    # Arguments
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        from_logits: (Optional )Whether output is expected to be a logits tensor.
            By default, we consider that output encodes a probability distribution.
        label_smoothing: (Optional) Float in [0, 1]. When > 0, label values are
            smoothed, meaning the confidence on label values are relaxed.
            e.g. `label_smoothing=0.2` means that we will use a value of `0.1` for
            label `0` and `0.9` for label `1`"
    """

    def __init__(self,
                 name='binary_crossentropy',
                 dtype=None,
                 from_logits=False,
                 label_smoothing=0):
        super(BinaryCrossentropy, self).__init__(
            binary_crossentropy,
            name,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing)


class CategoricalCrossentropy(MeanMetricWrapper):
    """Computes the crossentropy metric between the labels and predictions.

    This is the crossentropy metric class to be used when there are multiple
    label classes (2 or more). Here we assume that labels are given as a `one_hot`
    representation. eg., When labels values are [2, 0, 1],
    `y_true` = [[0, 0, 1], [1, 0, 0], [0, 1, 0]].

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile(
    'sgd',
    loss='mse',
    metrics=[keras.metrics.CategoricalCrossentropy()])
    ```

    # Arguments
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        from_logits: (Optional ) Whether `y_pred` is expected to be a logits tensor.
            By default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed. e.g.
            `label_smoothing=0.2` means that we will use a value of `0.1` for label
            `0` and `0.9` for label `1`"
    """

    def __init__(self,
                 name='categorical_crossentropy',
                 dtype=None,
                 from_logits=False,
                 label_smoothing=0):
        super(CategoricalCrossentropy, self).__init__(
            categorical_crossentropy,
            name,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing)


class SparseCategoricalCrossentropy(MeanMetricWrapper):
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

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile(
    'sgd',
    loss='mse',
    metrics=[keras.metrics.SparseCategoricalCrossentropy()])
    ```

    # Arguments
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        from_logits: (Optional ) Whether `y_pred` is expected to be a logits tensor.
            By default, we assume that `y_pred` encodes a probability distribution.
        axis: (Optional) Defaults to -1. The dimension along which the metric is
            computed.
    """

    def __init__(self,
                 name='sparse_categorical_crossentropy',
                 dtype=None,
                 from_logits=False,
                 axis=-1):
        super(SparseCategoricalCrossentropy, self).__init__(
            sparse_categorical_crossentropy,
            name,
            dtype=dtype,
            from_logits=from_logits,
            axis=axis)


class _ConfusionMatrixConditionCount(Metric):
    """Calculates the number of the given confusion matrix condition.

    # Arguments
        confusion_matrix_cond: One of `metrics_utils.ConfusionMatrix` conditions.
        thresholds: (Optional) Defaults to 0.5. A float value or a python
            list/tuple of float threshold values in [0, 1]. A threshold is compared
            with prediction values to determine the truth value of predictions
            (i.e., above the threshold is `true`, below is `false`). One metric
            value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(self,
                 confusion_matrix_cond,
                 thresholds=None,
                 name=None,
                 dtype=None):
        super(_ConfusionMatrixConditionCount, self).__init__(name=name, dtype=dtype)
        self._confusion_matrix_cond = confusion_matrix_cond
        self.init_thresholds = thresholds
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=0.5)
        self.accumulator = self.add_weight(
            'accumulator',
            shape=(len(self.thresholds),),
            initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {self._confusion_matrix_cond: self.accumulator},
            y_true,
            y_pred,
            thresholds=self.thresholds,
            sample_weight=sample_weight)

    def result(self):
        if len(self.thresholds) == 1:
            return self.accumulator[0]
        return self.accumulator

    def reset_states(self):
        num_thresholds = len(metrics_utils.to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.weights])

    def get_config(self):
        config = {'thresholds': self.init_thresholds}
        base_config = super(_ConfusionMatrixConditionCount, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FalsePositives(_ConfusionMatrixConditionCount):
    """Calculates the number of false positives.

    For example, if `y_true` is [0, 1, 0, 0] and `y_pred` is [0, 0, 1, 1]
    then the false positives value is 2.  If the weights were specified as
    [0, 0, 1, 0] then the false positives value would be 1.

    If `sample_weight` is given, calculates the sum of the weights of
    false positives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of false positives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss='mse', metrics=[keras.metrics.FalsePositives()])
    ```

    # Arguments
        thresholds: (Optional) Defaults to 0.5. A float value or a python
            list/tuple of float threshold values in [0, 1]. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `true`, below is `false`).
            One metric value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(self, thresholds=None, name=None, dtype=None):
        super(FalsePositives, self).__init__(
            confusion_matrix_cond=metrics_utils.ConfusionMatrix.FALSE_POSITIVES,
            thresholds=thresholds,
            name=name,
            dtype=dtype)


class TruePositives(_ConfusionMatrixConditionCount):
    """Calculates the number of true positives.

    For example, if `y_true` is [0, 1, 1, 1] and `y_pred` is [1, 0, 1, 1]
    then the true positives value is 2.  If the weights were specified as
    [0, 0, 1, 0] then the true positives value would be 1.

    If `sample_weight` is given, calculates the sum of the weights of
    true positives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of true positives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss='mse', metrics=[keras.metrics.TruePositives()])
    ```

    # Arguments
        thresholds: (Optional) Defaults to 0.5. A float value or a python
            list/tuple of float threshold values in [0, 1]. A threshold is compared
            with prediction values to determine the truth value of predictions
            (i.e., above the threshold is `true`, below is `false`). One metric
            value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(self, thresholds=None, name=None, dtype=None):
        super(TruePositives, self).__init__(
            confusion_matrix_cond=metrics_utils.ConfusionMatrix.TRUE_POSITIVES,
            thresholds=thresholds,
            name=name,
            dtype=dtype)


class TrueNegatives(_ConfusionMatrixConditionCount):
    """Calculates the number of true negatives.

    For example, if `y_true` is [0, 1, 0, 0] and `y_pred` is [1, 1, 0, 0]
    then the true negatives value is 2.  If the weights were specified as
    [0, 0, 1, 0] then the true negatives value would be 1.

    If `sample_weight` is given, calculates the sum of the weights of
    true negatives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of true negatives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss='mse', metrics=[keras.metrics.TrueNegatives()])
    ```

    # Arguments
        thresholds: (Optional) Defaults to 0.5. A float value or a python
            list/tuple of float threshold values in [0, 1]. A threshold is compared
            with prediction values to determine the truth value of predictions
            (i.e., above the threshold is `true`, below is `false`). One metric
            value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(self, thresholds=None, name=None, dtype=None):
        super(TrueNegatives, self).__init__(
            confusion_matrix_cond=metrics_utils.ConfusionMatrix.TRUE_NEGATIVES,
            thresholds=thresholds,
            name=name,
            dtype=dtype)


class FalseNegatives(_ConfusionMatrixConditionCount):
    """Calculates the number of false negatives.

    For example, if `y_true` is [0, 1, 1, 1] and `y_pred` is [0, 1, 0, 0]
    then the false negatives value is 2.  If the weights were specified as
    [0, 0, 1, 0] then the false negatives value would be 1.

    If `sample_weight` is given, calculates the sum of the weights of
    false negatives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of false negatives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss='mse', metrics=[keras.metrics.FalseNegatives()])
    ```

    # Arguments
        thresholds: (Optional) Defaults to 0.5. A float value or a python
            list/tuple of float threshold values in [0, 1]. A threshold is compared
            with prediction values to determine the truth value of predictions
            (i.e., above the threshold is `true`, below is `false`). One metric
            value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(self, thresholds=None, name=None, dtype=None):
        super(FalseNegatives, self).__init__(
            confusion_matrix_cond=metrics_utils.ConfusionMatrix.FALSE_NEGATIVES,
            thresholds=thresholds,
            name=name,
            dtype=dtype)


class SensitivitySpecificityBase(Metric):
    """Abstract base class for computing sensitivity and specificity.

    For additional information about specificity and sensitivity, see the
    following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    """

    def __init__(self, value, num_thresholds=200, name=None, dtype=None):
        super(SensitivitySpecificityBase, self).__init__(name=name, dtype=dtype)
        if num_thresholds <= 0:
            raise ValueError('`num_thresholds` must be > 0.')
        self.value = value
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(num_thresholds,),
            initializer='zeros')
        self.true_negatives = self.add_weight(
            'true_negatives',
            shape=(num_thresholds,),
            initializer='zeros')
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(num_thresholds,),
            initializer='zeros')
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(num_thresholds,),
            initializer='zeros')

        # Compute `num_thresholds` thresholds in [0, 1]
        if num_thresholds == 1:
            self.thresholds = [0.5]
        else:
            thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                          for i in range(num_thresholds - 2)]
            self.thresholds = [0.0] + thresholds + [1.0]

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            sample_weight=sample_weight)

    def reset_states(self):
        num_thresholds = len(self.thresholds)
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.weights])


class SensitivityAtSpecificity(SensitivitySpecificityBase):
    """Computes the sensitivity at a given specificity.

    `Sensitivity` measures the proportion of actual positives that are correctly
    identified as such (tp / (tp + fn)).
    `Specificity` measures the proportion of actual negatives that are correctly
    identified as such (tn / (tn + fp)).

    This metric creates four local variables, `true_positives`, `true_negatives`,
    `false_positives` and `false_negatives` that are used to compute the
    sensitivity at the given specificity. The threshold for the given specificity
    value is computed and used to evaluate the corresponding sensitivity.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    For additional information about specificity and sensitivity, see the
    following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile(
        'sgd',
        loss='mse',
        metrics=[keras.metrics.SensitivityAtSpecificity()])
    ```

    # Arguments
        specificity: A scalar value in range `[0, 1]`.
        num_thresholds: (Optional) Defaults to 200. The number of thresholds to
            use for matching the given specificity.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(self, specificity, num_thresholds=200, name=None, dtype=None):
        if specificity < 0 or specificity > 1:
            raise ValueError('`specificity` must be in the range [0, 1].')
        self.specificity = specificity
        self.num_thresholds = num_thresholds
        super(SensitivityAtSpecificity, self).__init__(
            specificity, num_thresholds=num_thresholds, name=name, dtype=dtype)

    def result(self):
        # Calculate specificities at all the thresholds.
        specificities = K.switch(
            K.greater(self.true_negatives + self.false_positives, 0),
            (self.true_negatives / (self.true_negatives + self.false_positives)),
            K.zeros_like(self.thresholds))

        # Find the index of the threshold where the specificity is closest to the
        # given specificity.
        min_index = K.argmin(
            K.abs(specificities - self.value), axis=0)
        min_index = K.cast(min_index, 'int32')

        # Compute sensitivity at that index.
        denom = self.true_positives[min_index] + self.false_negatives[min_index]
        return K.switch(
            K.greater(denom, 0),
            self.true_positives[min_index] / denom,
            K.zeros_like(self.true_positives[min_index]))

    def get_config(self):
        config = {
            'num_thresholds': self.num_thresholds,
            'specificity': self.specificity
        }
        base_config = super(SensitivityAtSpecificity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SpecificityAtSensitivity(SensitivitySpecificityBase):
    """Computes the specificity at a given sensitivity.

    `Sensitivity` measures the proportion of actual positives that are correctly
    identified as such (tp / (tp + fn)).
    `Specificity` measures the proportion of actual negatives that are correctly
    identified as such (tn / (tn + fp)).

    This metric creates four local variables, `true_positives`, `true_negatives`,
    `false_positives` and `false_negatives` that are used to compute the
    specificity at the given sensitivity. The threshold for the given sensitivity
    value is computed and used to evaluate the corresponding specificity.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    For additional information about specificity and sensitivity, see the
    following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity

    Usage with the compile API:
    ```python
    model = keras.Model(inputs, outputs)
    model.compile(
        'sgd',
        loss='mse',
        metrics=[keras.metrics.SpecificityAtSensitivity()])
    ```

    # Arguments
        sensitivity: A scalar value in range `[0, 1]`.
        num_thresholds: (Optional) Defaults to 200. The number of thresholds to
            use for matching the given specificity.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(self, sensitivity, num_thresholds=200, name=None, dtype=None):
        if sensitivity < 0 or sensitivity > 1:
            raise ValueError('`sensitivity` must be in the range [0, 1].')
        self.sensitivity = sensitivity
        self.num_thresholds = num_thresholds
        super(SpecificityAtSensitivity, self).__init__(
            sensitivity, num_thresholds=num_thresholds, name=name, dtype=dtype)

    def result(self):
        # Calculate sensitivities at all the thresholds.
        sensitivities = K.switch(
            K.greater(self.true_positives + self.false_negatives, 0),
            (self.true_positives / (self.true_positives + self.false_negatives)),
            K.zeros_like(self.thresholds))

        # Find the index of the threshold where the sensitivity is closest to the
        # given specificity.
        min_index = K.argmin(
            K.abs(sensitivities - self.value), axis=0)
        min_index = K.cast(min_index, 'int32')

        # Compute specificity at that index.
        denom = (self.true_negatives[min_index] + self.false_positives[min_index])
        return K.switch(
            K.greater(denom, 0),
            self.true_negatives[min_index] / denom,
            K.zeros_like(self.true_negatives[min_index]))

    def get_config(self):
        config = {
            'num_thresholds': self.num_thresholds,
            'sensitivity': self.sensitivity
        }
        base_config = super(SpecificityAtSensitivity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Precision(Metric):
    """Computes the precision of the predictions with respect to the labels.

    For example, if `y_true` is [0, 1, 1, 1] and `y_pred` is [1, 0, 1, 1]
    then the precision value is 2/(2+1) ie. 0.66. If the weights were specified as
    [0, 0, 1, 0] then the precision value would be 1.

    The metric creates two local variables, `true_positives` and `false_positives`
    that are used to compute the precision. This value is ultimately returned as
    `precision`, an idempotent operation that simply divides `true_positives`
    by the sum of `true_positives` and `false_positives`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `top_k` is set, we'll calculate precision as how often on average a class
    among the top-k classes with the highest predicted values of a batch entry is
    correct and can be found in the label for that entry.

    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold and/or in the
    top-k highest predictions, and computing the fraction of them for which
    `class_id` is indeed a correct label.

    Usage with the compile API:
    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss='mse', metrics=[keras.metrics.Precision()])
    ```

    # Arguments
        thresholds: (Optional) A float value or a python list/tuple of float
            threshold values in [0, 1]. A threshold is compared with prediction
            values to determine the truth value of predictions (i.e., above the
            threshold is `true`, below is `false`). One metric value is generated
            for each threshold value. If neither thresholds nor top_k are set, the
            default is to calculate precision with `thresholds=0.5`.
        top_k: (Optional) Unset by default. An int value specifying the top-k
            predictions to consider when calculating precision.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name=None,
                 dtype=None):
        super(Precision, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        if top_k is not None and K.backend() != 'tensorflow':
            raise RuntimeError(
                '`top_k` argument for `Precision` metric is currently supported '
                'only with TensorFlow backend.')

        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer='zeros')
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        denom = (self.true_positives + self.false_positives)
        result = K.switch(
            K.greater(denom, 0),
            self.true_positives / denom,
            K.zeros_like(self.true_positives))

        return result[0] if len(self.thresholds) == 1 else result

    def reset_states(self):
        num_thresholds = len(metrics_utils.to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.weights])

    def get_config(self):
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super(Precision, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Recall(Metric):
    """Computes the recall of the predictions with respect to the labels.

    For example, if `y_true` is [0, 1, 1, 1] and `y_pred` is [1, 0, 1, 1]
    then the recall value is 2/(2+1) ie. 0.66. If the weights were specified as
    [0, 0, 1, 0] then the recall value would be 1.

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

    Usage with the compile API:
    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss='mse', metrics=[keras.metrics.Recall()])
    ```

    # Arguments
        thresholds: (Optional) A float value or a python list/tuple of float
            threshold values in [0, 1]. A threshold is compared with prediction
            values to determine the truth value of predictions (i.e., above the
            threshold is `true`, below is `false`). One metric value is generated
            for each threshold value. If neither thresholds nor top_k are set, the
            default is to calculate recall with `thresholds=0.5`.
        top_k: (Optional) Unset by default. An int value specifying the top-k
            predictions to consider when calculating recall.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name=None,
                 dtype=None):
        super(Recall, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        if top_k is not None and K.backend() != 'tensorflow':
            raise RuntimeError(
                '`top_k` argument for `Recall` metric is currently supported only '
                'with TensorFlow backend.')

        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer='zeros')
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(len(self.thresholds),),
            initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        denom = (self.true_positives + self.false_negatives)
        result = K.switch(
            K.greater(denom, 0),
            self.true_positives / denom,
            K.zeros_like(self.true_positives))
        return result[0] if len(self.thresholds) == 1 else result

    def reset_states(self):
        num_thresholds = len(metrics_utils.to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.weights])

    def get_config(self):
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super(Recall, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AUC(Metric):
    """Computes the approximate AUC (Area under the curve) via a Riemann sum.

    This metric creates four local variables, `true_positives`, `true_negatives`,
    `false_positives` and `false_negatives` that are used to compute the AUC.
    To discretize the AUC curve, a linearly spaced set of thresholds is used to
    compute pairs of recall and precision values. The area under the ROC-curve is
    therefore computed using the height of the recall values by the false positive
    rate, while the area under the PR-curve is the computed using the height of
    the precision values by the recall.

    This value is ultimately returned as `auc`, an idempotent operation that
    computes the area under a discretized curve of precision versus recall values
    (computed using the aforementioned variables). The `num_thresholds` variable
    controls the degree of discretization with larger numbers of thresholds more
    closely approximating the true AUC. The quality of the approximation may vary
    dramatically depending on `num_thresholds`. The `thresholds` parameter can be
    used to manually specify thresholds which split the predictions more evenly.

    For best results, `predictions` should be distributed approximately uniformly
    in the range [0, 1] and not peaked around 0 or 1. The quality of the AUC
    approximation may be poor if this is not the case. Setting `summation_method`
    to 'minoring' or 'majoring' can help quantify the error in the approximation
    by providing lower or upper bound estimate of the AUC.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss='mse', metrics=[keras.metrics.AUC()])
    ```

    # Arguments
        num_thresholds: (Optional) Defaults to 200. The number of thresholds to
            use when discretizing the roc curve. Values must be > 1.
            curve: (Optional) Specifies the name of the curve to be computed, 'ROC'
            [default] or 'PR' for the Precision-Recall-curve.
        summation_method: (Optional) Specifies the Riemann summation method used
            (https://en.wikipedia.org/wiki/Riemann_sum): 'interpolation' [default],
              applies mid-point summation scheme for `ROC`. For PR-AUC, interpolates
              (true/false) positives but not the ratio that is precision (see Davis
              & Goadrich 2006 for details); 'minoring' that applies left summation
              for increasing intervals and right summation for decreasing intervals;
              'majoring' that does the opposite.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        thresholds: (Optional) A list of floating point values to use as the
            thresholds for discretizing the curve. If set, the `num_thresholds`
            parameter is ignored. Values should be in [0, 1]. Endpoint thresholds
            equal to {-epsilon, 1+epsilon} for a small positive epsilon value will
            be automatically included with these to correctly handle predictions
            equal to exactly 0 or 1.
    """

    def __init__(self,
                 num_thresholds=200,
                 curve='ROC',
                 summation_method='interpolation',
                 name=None,
                 dtype=None,
                 thresholds=None):
        # Validate configurations.
        if (isinstance(curve, metrics_utils.AUCCurve) and
                curve not in list(metrics_utils.AUCCurve)):
            raise ValueError('Invalid curve: "{}". Valid options are: "{}"'.format(
                curve, list(metrics_utils.AUCCurve)))
        if isinstance(
            summation_method,
            metrics_utils.AUCSummationMethod) and summation_method not in list(
                metrics_utils.AUCSummationMethod):
            raise ValueError(
                'Invalid summation method: "{}". Valid options are: "{}"'.format(
                    summation_method, list(metrics_utils.AUCSummationMethod)))

        # Update properties.
        if thresholds is not None:
            # If specified, use the supplied thresholds.
            self.num_thresholds = len(thresholds) + 2
            thresholds = sorted(thresholds)
        else:
            if num_thresholds <= 1:
                raise ValueError('`num_thresholds` must be > 1.')

            # Otherwise, linearly interpolate (num_thresholds - 2) thresholds in
            # (0, 1).
            self.num_thresholds = num_thresholds
            thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                          for i in range(num_thresholds - 2)]

        # Add an endpoint "threshold" below zero and above one for either
        # threshold method to account for floating point imprecisions.
        self.thresholds = [0.0 - K.epsilon()] + thresholds + [1.0 + K.epsilon()]

        if isinstance(curve, metrics_utils.AUCCurve):
            self.curve = curve
        else:
            self.curve = metrics_utils.AUCCurve.from_str(curve)
        if isinstance(summation_method, metrics_utils.AUCSummationMethod):
            self.summation_method = summation_method
        else:
            self.summation_method = metrics_utils.AUCSummationMethod.from_str(
                summation_method)
        super(AUC, self).__init__(name=name, dtype=dtype)

        # Create metric variables
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(self.num_thresholds,),
            initializer='zeros')
        self.true_negatives = self.add_weight(
            'true_negatives',
            shape=(self.num_thresholds,),
            initializer='zeros')
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(self.num_thresholds,),
            initializer='zeros')
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(self.num_thresholds,),
            initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables({
            metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
            metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
            metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
            metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
        }, y_true, y_pred, self.thresholds, sample_weight=sample_weight)

    def interpolate_pr_auc(self):
        """Interpolation formula inspired by section 4 of Davis & Goadrich 2006.

        https://www.biostat.wisc.edu/~page/rocpr.pdf

        Note here we derive & use a closed formula not present in the paper
        as follows:

          Precision = TP / (TP + FP) = TP / P

        Modeling all of TP (true positive), FP (false positive) and their sum
        P = TP + FP (predicted positive) as varying linearly within each interval
        [A, B] between successive thresholds, we get

          Precision slope = dTP / dP
                          = (TP_B - TP_A) / (P_B - P_A)
                          = (TP - TP_A) / (P - P_A)
          Precision = (TP_A + slope * (P - P_A)) / P

        The area within the interval is (slope / total_pos_weight) times

          int_A^B{Precision.dP} = int_A^B{(TP_A + slope * (P - P_A)) * dP / P}
          int_A^B{Precision.dP} = int_A^B{slope * dP + intercept * dP / P}

        where intercept = TP_A - slope * P_A = TP_B - slope * P_B, resulting in

          int_A^B{Precision.dP} = TP_B - TP_A + intercept * log(P_B / P_A)

        Bringing back the factor (slope / total_pos_weight) we'd put aside, we get

          slope * [dTP + intercept *  log(P_B / P_A)] / total_pos_weight

        where dTP == TP_B - TP_A.

        Note that when P_A == 0 the above calculation simplifies into

          int_A^B{Precision.dTP} = int_A^B{slope * dTP} = slope * (TP_B - TP_A)

        which is really equivalent to imputing constant precision throughout the
        first bucket having >0 true positives.

        # Returns
            pr_auc: an approximation of the area under the P-R curve.
        """
        dtp = self.true_positives[:self.num_thresholds -
                                  1] - self.true_positives[1:]
        p = self.true_positives + self.false_positives
        dp = p[:self.num_thresholds - 1] - p[1:]

        dp = K.maximum(dp, 0)
        prec_slope = K.switch(
            K.greater(dp, 0),
            dtp / dp,
            K.zeros_like(dtp))
        intercept = self.true_positives[1:] - (prec_slope * p[1:])

        # Logical and
        pMin = K.expand_dims(p[:self.num_thresholds - 1] > 0, 0)
        pMax = K.expand_dims(p[1:] > 0, 0)
        are_different = K.concatenate([pMin, pMax], axis=0)
        switch_condition = K.all(are_different, axis=0)

        safe_p_ratio = K.switch(
            switch_condition,
            K.switch(
                K.greater(p[1:], 0),
                p[:self.num_thresholds - 1] / p[1:],
                K.zeros_like(p[:self.num_thresholds - 1])),
            K.ones_like(p[1:]))

        numer = prec_slope * (dtp + intercept * K.log(safe_p_ratio))
        denom = K.maximum(self.true_positives[1:] + self.false_negatives[1:], 0)
        return K.sum(K.switch(
            K.greater(denom, 0),
            numer / denom,
            K.zeros_like(numer)))

    def result(self):
        if (self.curve == metrics_utils.AUCCurve.PR and
                (self.summation_method ==
                 metrics_utils.AUCSummationMethod.INTERPOLATION)):
            # This use case is different and is handled separately.
            return self.interpolate_pr_auc()

        # Set `x` and `y` values for the curves based on `curve` config.
        recall = K.switch(
            K.greater((self.true_positives), 0),
            (self.true_positives /
                (self.true_positives + self.false_negatives)),
            K.zeros_like(self.true_positives))
        if self.curve == metrics_utils.AUCCurve.ROC:
            fp_rate = K.switch(
                K.greater((self.false_positives), 0),
                (self.false_positives /
                    (self.false_positives + self.true_negatives)),
                K.zeros_like(self.false_positives))
            x = fp_rate
            y = recall
        else:  # curve == 'PR'.
            precision = K.switch(
                K.greater((self.true_positives), 0),
                (self.true_positives / (self.true_positives + self.false_positives)),
                K.zeros_like(self.true_positives))
            x = recall
            y = precision

        # Find the rectangle heights based on `summation_method`.
        if self.summation_method == metrics_utils.AUCSummationMethod.INTERPOLATION:
            # Note: the case ('PR', 'interpolation') has been handled above.
            heights = (y[:self.num_thresholds - 1] + y[1:]) / 2.
        elif self.summation_method == metrics_utils.AUCSummationMethod.MINORING:
            heights = K.minimum(y[:self.num_thresholds - 1], y[1:])
        else:  # self.summation_method = metrics_utils.AUCSummationMethod.MAJORING:
            heights = K.maximum(y[:self.num_thresholds - 1], y[1:])

        # Sum up the areas of all the rectangles.
        return K.sum((x[:self.num_thresholds - 1] - x[1:]) * heights)

    def reset_states(self):
        K.batch_set_value(
            [(v, np.zeros((self.num_thresholds,))) for v in self.weights])

    def get_config(self):
        config = {
            'num_thresholds': self.num_thresholds,
            'curve': self.curve.value,
            'summation_method': self.summation_method.value,
            # We remove the endpoint thresholds as an inverse of how the thresholds
            # were initialized. This ensures that a metric initialized from this
            # config has the same thresholds.
            'thresholds': self.thresholds[1:-1],
        }
        base_config = super(AUC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


BaseMeanIoU = object
if K.backend() == 'tensorflow':
    import tensorflow as tf
    if tf.__version__ >= '2.0.0':
        BaseMeanIoU = tf.keras.metrics.MeanIoU


class MeanIoU(BaseMeanIoU):
    """Computes the mean Intersection-Over-Union metric.

    Mean Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation, which first computes the IOU for each semantic class and then
    computes the average over classes. IOU is defined as follows:
    IOU = true_positive / (true_positive + false_positive + false_negative).
    The predictions are accumulated in a confusion matrix, weighted by
    `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage with the compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile(
        'sgd',
        loss='mse',
        metrics=[keras.metrics.MeanIoU(num_classes=2)])
    ```

    # Arguments
        num_classes: The possible number of labels the prediction task can have.
            This value must be provided, since a confusion matrix of dimension =
            [num_classes, num_classes] will be allocated.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """
    def __init__(self, num_classes, name=None, dtype=None):
        if K.backend() != 'tensorflow' or BaseMeanIoU is object:
            raise RuntimeError(
                '`MeanIoU` metric is currently supported only '
                'with TensorFlow backend and TF version >= 2.0.0.')
        super(MeanIoU, self).__init__(num_classes, name=name, dtype=dtype)


def accuracy(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.cast(K.equal(y_true, y_pred), K.floatx())


def binary_accuracy(y_true, y_pred, threshold=0.5):
    if threshold != 0.5:
        threshold = K.cast(threshold, y_pred.dtype)
        y_pred = K.cast(y_pred > threshold, y_pred.dtype)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def sparse_categorical_accuracy(y_true, y_pred):
    # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
    if K.ndim(y_true) == K.ndim(y_pred):
        y_true = K.squeeze(y_true, -1)
    # convert dense predictions to labels
    y_pred_labels = K.argmax(y_pred, axis=-1)
    y_pred_labels = K.cast(y_pred_labels, K.floatx())
    return K.cast(K.equal(y_true, y_pred_labels), K.floatx())


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    return K.cast(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), K.floatx())


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    # If the shape of y_true is (num_samples, 1), flatten to (num_samples,)
    return K.cast(K.in_top_k(y_pred, K.cast(K.flatten(y_true), 'int32'), k),
                  K.floatx())


def cosine_proximity(y_true, y_pred, axis=-1):
    y_true = K.l2_normalize(y_true, axis=axis)
    y_pred = K.l2_normalize(y_pred, axis=axis)
    return K.sum(y_true * y_pred, axis=axis)


def clone_metric(metric):
    """Returns a clone of the metric if stateful, otherwise returns it as is."""
    if isinstance(metric, Metric):
        return metric.__class__.from_config(metric.get_config())
    return metric


def clone_metrics(metrics):
    """Clones the given metric list/dict."""
    if metrics is None:
        return None
    if isinstance(metrics, dict):
        return {key: clone_metric(value) for key, value in metrics.items()}
    return [clone_metric(metric) for metric in metrics]


# Aliases

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_similarity = cosine_proximity


def serialize(metric):
    return serialize_keras_object(metric)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='metric function')


def get(identifier):
    if isinstance(identifier, dict):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif isinstance(identifier, six.string_types):
        return deserialize(str(identifier))
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'metric function identifier:', identifier)
