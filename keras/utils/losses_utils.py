"""Utilities related to losses."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .. import backend as K


class Reduction(object):
    """Types of loss reduction.

    Contains the following values:

    * `NONE`: Un-reduced weighted losses with the same shape as input. When this
        reduction type used with built-in Keras training loops like
        `fit`/`evaluate`, the unreduced vector loss is passed to the optimizer but
        the reported loss will be a scalar value.
    * `SUM`: Scalar sum of weighted losses.
    * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
    """

    NONE = 'none'
    SUM = 'sum'
    SUM_OVER_BATCH_SIZE = 'sum_over_batch_size'

    @classmethod
    def all(cls):
        return (cls.NONE, cls.SUM, cls.SUM_OVER_BATCH_SIZE)

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError('Invalid Reduction Key %s.' % key)


def squeeze_or_expand_dimensions(y_pred, y_true=None, sample_weight=None):
    """Squeeze or expand last dimension if needed.

    1. Squeezes last dim of `y_pred` or `y_true` if their rank differs by 1.
    2. Squeezes or expands last dim of `sample_weight` if its rank differs by 1
    from the new rank of `y_pred`.
    If `sample_weight` is scalar, it is kept scalar.

    # Arguments
        y_pred: Predicted values, a `Tensor` of arbitrary dimensions.
        y_true: Optional label `Tensor` whose dimensions match `y_pred`.
        sample_weight: Optional weight scalar or `Tensor` whose dimensions match
            `y_pred`.

    # Returns
        Tuple of `y_pred`, `y_true` and `sample_weight`. Each of them possibly has
        the last dimension squeezed, `sample_weight` could be extended by one
        dimension.
    """
    if y_true is not None:
        y_pred_rank = K.ndim(y_pred)
        y_pred_shape = K.int_shape(y_pred)
        y_true_rank = K.ndim(y_true)
        y_true_shape = K.int_shape(y_true)

        if (y_pred_rank - y_true_rank == 1) and (y_pred_shape[-1] == 1):
            y_pred = K.squeeze(y_pred, -1)
        elif (y_true_rank - y_pred_rank == 1) and (y_true_shape[-1] == 1):
            y_true = K.squeeze(y_true, -1)

    if sample_weight is None:
        return y_pred, y_true

    y_pred_rank = K.ndim(y_pred)
    weights_rank = K.ndim(sample_weight)
    if weights_rank != 0:
        if y_pred_rank == 0 and weights_rank == 1:
            y_pred = K.expand_dims(y_pred, -1)
        elif weights_rank - y_pred_rank == 1:
            sample_weight = K.squeeze(sample_weight, -1)
        elif y_pred_rank - weights_rank == 1:
            sample_weight = K.expand_dims(sample_weight, -1)
    return y_pred, y_true, sample_weight


def _num_elements(losses):
    """Computes the number of elements in `losses` tensor."""
    with K.name_scope('num_elements') as scope:
        return K.cast(K.size(losses, name=scope), losses.dtype)


def reduce_weighted_loss(weighted_losses, reduction=Reduction.SUM_OVER_BATCH_SIZE):
    """Reduces the individual weighted loss measurements."""
    if reduction == Reduction.NONE:
        loss = weighted_losses
    else:
        loss = K.sum(weighted_losses)
        if reduction == Reduction.SUM_OVER_BATCH_SIZE:
            loss = loss / _num_elements(weighted_losses)
    return loss


def broadcast_weights(values, sample_weight):
    # Broadcast weights if possible.
    weights_shape = K.int_shape(sample_weight)
    values_shape = K.int_shape(values)

    if values_shape != weights_shape:
        weights_rank = K.ndim(sample_weight)
        values_rank = K.ndim(values)

        # Raise error if ndim of weights is > values.
        if weights_rank > values_rank:
            raise ValueError(
                'Incompatible shapes: `values` {} vs `sample_weight` {}'.format(
                    values_shape, weights_shape))

        # Expand dim of weights to match ndim of values, if required.
        for i in range(weights_rank, values_rank):
            sample_weight = K.expand_dims(sample_weight, axis=i)

        if weights_shape is not None and values_shape is not None:
            for i in range(weights_rank):
                if (weights_shape[i] is not None and
                    values_shape[i] is not None and
                        weights_shape[i] != values_shape[i]):
                    # Cannot be broadcasted.
                    if weights_shape[i] != 1:
                        raise ValueError(
                            'Incompatible shapes: `values` {} vs '
                            '`sample_weight` {}'.format(
                                values_shape, weights_shape))
                    sample_weight = K.repeat_elements(
                        sample_weight, values_shape[i], axis=i)
    return sample_weight


def compute_weighted_loss(losses,
                          sample_weight=None,
                          reduction=Reduction.SUM_OVER_BATCH_SIZE,
                          name=None):
    """Computes the weighted loss.

    # Arguments
        losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
        sample_weight: Optional `Tensor` whose rank is either 0, or the same rank as
        `   losses`, or be broadcastable to `losses`.
        reduction: (Optional) Type of Reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: Optional name for the op.

    # Raises
        ValueError: If the shape of `sample_weight` is not compatible with `losses`.

    # Returns
        Weighted loss `Tensor` of the same type as `losses`. If `reduction` is
            `NONE`, this has the same shape as `losses`; otherwise, it is scalar.
    """
    Reduction.validate(reduction)
    if sample_weight is None:
        sample_weight = 1.0
    with K.name_scope(name or 'weighted_loss'):
        input_dtype = K.dtype(losses)
        losses = K.cast(losses, K.floatx())
        sample_weight = K.cast(sample_weight, K.floatx())

        # Update dimensions of `sample_weight` to match with `losses` if possible.
        losses, _, sample_weight = squeeze_or_expand_dimensions(
            losses, None, sample_weight)

        # Broadcast weights if possible.
        sample_weight = broadcast_weights(losses, sample_weight)

        # Apply weights to losses.
        weighted_losses = sample_weight * losses

        # Apply reduction function to the individual weighted losses.
        loss = reduce_weighted_loss(weighted_losses, reduction)
        # Convert the result back to the input type.
        loss = K.cast(loss, input_dtype)
        return loss
