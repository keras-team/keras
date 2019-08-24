"""Built-in metrics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
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
from .losses import categorical_crossentropy
from .losses import sparse_categorical_crossentropy
from .losses import binary_crossentropy
from .losses import kullback_leibler_divergence
from .losses import poisson
from .losses import cosine_proximity
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

    def __call__(self, *args, **kwargs):
        """Accumulates statistics and then computes metric result value."""
        if K.backend() != 'tensorflow':
            raise RuntimeError(
                'Metric calling only supported with TensorFlow backend.')
        update_op = self.update_state(*args, **kwargs)
        with K.control_dependencies([update_op]):  # For TF
            return self.result()

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
    m = tf.keras.metrics.Mean()
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
            config[k] = K.eval(v) if is_tensor_or_variable(v) else v
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


def binary_accuracy(y_true, y_pred):
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
cosine = cosine_proximity


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
