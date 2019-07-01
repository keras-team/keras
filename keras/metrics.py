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
        self.dtype = K.floatx() if dtype is None else dtype

    def __new__(cls, *args, **kwargs):
        obj = super(Metric, cls).__new__(cls)
        update_state_fn = obj.update_state

        obj.update_state = types.MethodType(
            metrics_utils.update_state_wrapper(update_state_fn), obj)
        return obj

    def __call__(self, *args, **kwargs):
        """Accumulates statistics and then computes metric result value."""
        update_op = self.update_state(*args, **kwargs)
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
        self.total = self.add_weight('total', initializer='zeros')
        if reduction in [metrics_utils.Reduction.SUM_OVER_BATCH_SIZE,
                         metrics_utils.Reduction.WEIGHTED_MEAN]:
            self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, values, sample_weight=None):
        """Accumulates statistics for computing the reduction metric.
        For example, if `values` is [1, 3, 5, 7] and reduction=SUM_OVER_BATCH_SIZE,
        then the value of `result()` is 4. If the `sample_weight` is specified as
        [1, 1, 0, 0] then value of `result()` would be 2.
        # Arguments
            values: Per-example value.
            sample_weight: Optional weighting of each example. Defaults to 1.
        """
        values = K.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = K.cast(sample_weight, self.dtype)
            # Update dimensions of weights to match with values if possible.
            values, _, sample_weight = losses_utils.squeeze_or_expand_dimensions(
                values, sample_weight=sample_weight)

            # Broadcast weights if possible.
            sample_weight = losses_utils.broadcast_weights(sample_weight, values)
            values = values * sample_weight

        value_sum = K.sum(values)
        update_total_op = K.update_add(self.total, value_sum)

        # Exit early if the reduction doesn't have a denominator.
        if self.reduction == metrics_utils.Reduction.SUM:
            return update_total_op

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

        with K.control_dependencies([update_total_op]):
            return K.update_add(self.count, num_values)

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
