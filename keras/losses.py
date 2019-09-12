"""Built-in loss functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

from . import backend as K
from .utils import losses_utils
from .utils.generic_utils import deserialize_keras_object
from .utils.generic_utils import serialize_keras_object


@six.add_metaclass(abc.ABCMeta)
class Loss(object):
    """Loss base class.

    To be implemented by subclasses:
        * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.

    Example subclass implementation:
    ```python
    class MeanSquaredError(Loss):
        def call(self, y_true, y_pred):
            y_pred = ops.convert_to_tensor(y_pred)
            y_true = math_ops.cast(y_true, y_pred.dtype)
            return K.mean(math_ops.square(y_pred - y_true), axis=-1)
    ```

    # Arguments
        reduction: (Optional) Type of loss Reduction to apply to loss.
          Default value is `SUM_OVER_BATCH_SIZE`.
        name: Optional name for the object.
    """

    def __init__(self,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name=None):
        self.reduction = reduction
        self.name = name

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Invokes the `Loss` instance.

        # Arguments
            y_true: Ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional `Tensor` whose rank is either 0, or the same rank
            as `y_true`, or is broadcastable to `y_true`. `sample_weight` acts as a
            coefficient for the loss. If a scalar is provided, then the loss is
            simply scaled by the given value. If `sample_weight` is a tensor of size
            `[batch_size]`, then the total loss for each sample of the batch is
            rescaled by the corresponding element in the `sample_weight` vector. If
            the shape of `sample_weight` matches the shape of `y_pred`, then the
            loss of each measurable element of `y_pred` is scaled by the
            corresponding value of `sample_weight`.

        # Returns
            Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
                shape as `y_true`; otherwise, it is scalar.

        # Raises
            ValueError: If the shape of `sample_weight` is invalid.
        """
        # If we are wrapping a lambda function strip '<>' from the name as it is not
        # accepted in scope name.
        scope_name = 'lambda' if self.name == '<lambda>' else self.name
        with K.name_scope(scope_name):
            losses = self.call(y_true, y_pred)
            return losses_utils.compute_weighted_loss(
                losses, sample_weight, reduction=self.reduction)

    @classmethod
    def from_config(cls, config):
        """Instantiates a `Loss` from its config (output of `get_config()`).

        # Arguments
            config: Output of `get_config()`.

        # Returns
            A `Loss` instance.
        """
        return cls(**config)

    def get_config(self):
        return {'reduction': self.reduction, 'name': self.name}

    @abc.abstractmethod
    def call(self, y_true, y_pred):
        """Invokes the `Loss` instance.

        # Arguments
            y_true: Ground truth values, with the same shape as 'y_pred'.
            y_pred: The predicted values.
        """
        raise NotImplementedError('Must be implemented in subclasses.')


class LossFunctionWrapper(Loss):
    """Wraps a loss function in the `Loss` class.

    # Arguments
        fn: The loss function to wrap, with signature `fn(y_true, y_pred,
            **kwargs)`.
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) name for the loss.
        **kwargs: The keyword arguments that are passed on to `fn`.
    """

    def __init__(self,
                 fn,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name=None,
                 **kwargs):
        super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        """Invokes the `LossFunctionWrapper` instance.

        # Arguments
            y_true: Ground truth values.
            y_pred: The predicted values.

        # Returns
            Loss values per sample.
        """
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = K.eval(v) if K.is_tensor(v) or K.is_variable(v) else v
        base_config = super(LossFunctionWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MeanSquaredError(LossFunctionWrapper):
    """Computes the mean of squares of errors between labels and predictions.

    Standalone usage:

    ```python
    mse = keras.losses.MeanSquaredError()
    loss = mse([0., 0., 1., 1.], [1., 1., 1., 0.])
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.MeanSquaredError())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) name for the loss.
    """

    def __init__(self,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name='mean_squared_error'):
        super(MeanSquaredError, self).__init__(
            mean_squared_error, name=name, reduction=reduction)


class MeanAbsoluteError(LossFunctionWrapper):
    """Computes the mean of absolute difference between labels and predictions.

    Standalone usage:

    ```python
    mae = keras.losses.MeanAbsoluteError()
    loss = mae([0., 0., 1., 1.], [1., 1., 1., 0.])
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.MeanAbsoluteError())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) name for the loss.
    """

    def __init__(self,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name='mean_absolute_error'):
        super(MeanAbsoluteError, self).__init__(
            mean_absolute_error, name=name, reduction=reduction)


class MeanAbsolutePercentageError(LossFunctionWrapper):
    """Computes the mean absolute percentage error between `y_true` and `y_pred`.

    Standalone usage:

    ```python
    mape = keras.losses.MeanAbsolutePercentageError()
    loss = mape([0., 0., 1., 1.], [1., 1., 1., 0.])
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.MeanAbsolutePercentageError())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) name for the loss.
    """

    def __init__(self,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name='mean_absolute_percentage_error'):
        super(MeanAbsolutePercentageError, self).__init__(
            mean_absolute_percentage_error, name=name, reduction=reduction)


class MeanSquaredLogarithmicError(LossFunctionWrapper):
    """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

    Standalone usage:

    ```python
    msle = keras.losses.MeanSquaredLogarithmicError()
    loss = msle([0., 0., 1., 1.], [1., 1., 1., 0.])
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.MeanSquaredLogarithmicError())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) name for the loss.
    """

    def __init__(self,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name='mean_squared_logarithmic_error'):
        super(MeanSquaredLogarithmicError, self).__init__(
            mean_squared_logarithmic_error, name=name, reduction=reduction)


class BinaryCrossentropy(LossFunctionWrapper):
    """Computes the cross-entropy loss between true labels and predicted labels.

    Use this cross-entropy loss when there are only two label classes (assumed to
    be 0 and 1). For each example, there should be a single floating-point value
    per prediction.

    In the snippet below, each of the four examples has only a single
    floating-pointing value, and both `y_pred` and `y_true` have the shape
    `[batch_size]`.

    Standalone usage:

    ```python
    bce = keras.losses.BinaryCrossentropy()
    loss = bce([0., 0., 1., 1.], [1., 1., 1., 0.])
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.BinaryCrossentropy())
    ```

    # Arguments
        from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default,
            we assume that `y_pred` contains probabilities
            (i.e., values in [0, 1]).
        label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When > 0, we
            compute the loss between the predicted labels and a smoothed version of
            the true labels, where the smoothing squeezes the labels towards 0.5.
            Larger values of `label_smoothing` correspond to heavier smoothing.
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """

    def __init__(self,
                 from_logits=False,
                 label_smoothing=0,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name='binary_crossentropy'):
        super(BinaryCrossentropy, self).__init__(
            binary_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing)
        self.from_logits = from_logits


class CategoricalCrossentropy(LossFunctionWrapper):
    """Computes the crossentropy loss between the labels and predictions.

    Use this crossentropy loss function when there are two or more label classes.
    We expect labels to be provided in a `one_hot` representation. If you want to
    provide labels as integers, please use `SparseCategoricalCrossentropy` loss.
    There should be `# classes` floating point values per feature.

    In the snippet below, there is `# classes` floating pointing values per
    example. The shape of both `y_pred` and `y_true` are
    `[batch_size, num_classes]`.

    Standalone usage:

    ```python
    cce = keras.losses.CategoricalCrossentropy()
    loss = cce(
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.CategoricalCrossentropy())
    ```

    # Arguments
        from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default,
            we assume that `y_pred` contains probabilities
            (i.e., values in [0, 1]).
        label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When > 0, we
            compute the loss between the predicted labels and a smoothed version of
            the true labels, where the smoothing squeezes the labels towards 0.5.
            Larger values of `label_smoothing` correspond to heavier smoothing.
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """

    def __init__(self,
                 from_logits=False,
                 label_smoothing=0,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name='categorical_crossentropy'):
        super(CategoricalCrossentropy, self).__init__(
            categorical_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing)


class SparseCategoricalCrossentropy(LossFunctionWrapper):
    """Computes the crossentropy loss between the labels and predictions.

    Use this crossentropy loss function when there are two or more label classes.
    We expect labels to be provided as integers. If you want to provide labels
    using `one-hot` representation, please use `CategoricalCrossentropy` loss.
    There should be `# classes` floating point values per feature for `y_pred`
    and a single floating point value per feature for `y_true`.

    In the snippet below, there is a single floating point value per example for
    `y_true` and `# classes` floating pointing values per example for `y_pred`.
    The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
    `[batch_size, num_classes]`.

    Standalone usage:

    ```python
    cce = keras.losses.SparseCategoricalCrossentropy()
    loss = cce(
        [0, 1, 2],
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.SparseCategoricalCrossentropy())
    ```

    # Arguments
        from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default,
            we assume that `y_pred` contains probabilities
            (i.e., values in [0, 1]).
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """

    def __init__(self,
                 from_logits=False,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name='sparse_categorical_crossentropy'):
        super(SparseCategoricalCrossentropy, self).__init__(
            sparse_categorical_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits)


class Hinge(LossFunctionWrapper):
    """Computes the hinge loss between `y_true` and `y_pred`.

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.Hinge())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """

    def __init__(self,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name='hinge'):
        super(Hinge, self).__init__(hinge, name=name, reduction=reduction)


class SquaredHinge(LossFunctionWrapper):
    """Computes the squared hinge loss between `y_true` and `y_pred`.

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.SquaredHinge())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """

    def __init__(self,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name='squared_hinge'):
        super(SquaredHinge, self).__init__(
            squared_hinge, name=name, reduction=reduction)


class CategoricalHinge(LossFunctionWrapper):
    """Computes the categorical hinge loss between `y_true` and `y_pred`.

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.CategoricalHinge())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """

    def __init__(self,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name='categorical_hinge'):
        super(CategoricalHinge, self).__init__(
            categorical_hinge, name=name, reduction=reduction)


class Poisson(LossFunctionWrapper):
    """Computes the Poisson loss between `y_true` and `y_pred`.

    `loss = y_pred - y_true * log(y_pred)`

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.Poisson())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """

    def __init__(self,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name='poisson'):
        super(Poisson, self).__init__(poisson, name=name, reduction=reduction)


class LogCosh(LossFunctionWrapper):
    """Computes the logarithm of the hyperbolic cosine of the prediction error.

    `logcosh = log((exp(x) + exp(-x))/2)`,
    where x is the error (y_pred - y_true)

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.LogCosh())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """

    def __init__(self,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name='logcosh'):
        super(LogCosh, self).__init__(logcosh, name=name, reduction=reduction)


class KLDivergence(LossFunctionWrapper):
    """Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

    `loss = y_true * log(y_true / y_pred)`

    See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.KLDivergence())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """

    def __init__(self,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name='kullback_leibler_divergence'):
        super(KLDivergence, self).__init__(
            kullback_leibler_divergence, name=name, reduction=reduction)


class Huber(LossFunctionWrapper):
    """Computes the Huber loss between `y_true` and `y_pred`.

    Given `x = y_true - y_pred`:
    ```
    loss = 0.5 * x^2                  if |x| <= d
    loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
    ```
    where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.Huber())
    ```

    # Arguments
        delta: A float, the point where the Huber loss function changes from a
            quadratic to linear.
        reduction: (Optional) Type of reduction to apply to loss.
        name: Optional name for the object.
    """
    def __init__(self,
                 delta=1.0,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name='huber_loss'):
        super(Huber, self).__init__(
            huber_loss, name=name, reduction=reduction, delta=delta)


def mean_squared_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mean_absolute_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 100. * K.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def squared_hinge(y_true, y_pred):
    y_true = _maybe_convert_labels(y_true)
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
    y_true = _maybe_convert_labels(y_true)
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(0., neg - pos + 1.)


def logcosh(y_true, y_pred):
    """Logarithm of the hyperbolic cosine of the prediction error.

    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
    to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
    like the mean squared error, but will not be so strongly affected by the
    occasional wildly incorrect prediction.

    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.

    # Returns
        Tensor with one scalar loss entry per sample.
    """
    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)
    return K.mean(_logcosh(y_pred - y_true), axis=-1)


def huber_loss(y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    abs_error = K.abs(error)
    quadratic = K.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * K.square(quadratic) + delta * linear


def categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
    return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)


def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
    return K.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=from_logits, axis=axis)


def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)
    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)
        y_true = K.switch(K.greater(smoothing, 0),
                          lambda: y_true * (1.0 - smoothing) + 0.5 * smoothing,
                          lambda: y_true)
    return K.mean(
        K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)


def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


def cosine_proximity(y_true, y_pred, axis=-1):
    y_true = K.l2_normalize(y_true, axis=axis)
    y_pred = K.l2_normalize(y_pred, axis=axis)
    return - K.sum(y_true * y_pred, axis=axis)


def _maybe_convert_labels(y_true):
    """Converts binary labels into -1/1."""
    are_zeros = K.equal(y_true, 0)
    are_ones = K.equal(y_true, 1)

    are_zeros = K.expand_dims(are_zeros, 0)
    are_ones = K.expand_dims(are_ones, 0)

    are_different = K.concatenate([are_zeros, are_ones], axis=0)
    are_different = K.any(are_different, axis=0)
    is_binary = K.all(are_different)

    def _convert_binary_labels():
        # Convert the binary labels to -1 or 1.
        return 2. * y_true - 1.

    updated_y_true = K.switch(is_binary,
                              _convert_binary_labels,
                              lambda: y_true)
    return updated_y_true


# Aliases.

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_similarity = cosine_proximity


def is_categorical_crossentropy(loss):
    return (isinstance(loss, CategoricalCrossentropy) or
            (isinstance(loss, LossFunctionWrapper) and
                loss.fn == categorical_crossentropy) or
            (hasattr(loss, '__name__') and
                loss.__name__ == 'categorical_crossentropy') or
            loss == 'categorical_crossentropy')


def serialize(loss):
    return serialize_keras_object(loss)


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='loss function')


def get(identifier):
    """Get the `identifier` loss function.

    # Arguments
        identifier: None or str, name of the function.

    # Returns
        The loss function or None if `identifier` is None.

    # Raises
        ValueError if unknown identifier.
    """
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'loss function identifier:', identifier)
