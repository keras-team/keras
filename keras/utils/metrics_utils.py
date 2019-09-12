"""Utilities related to metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum

from .. import backend as K
from . import losses_utils


NEG_INF = -1e10


class Reduction(object):
    """Types of metrics reduction.

    Contains the following values:
    * `SUM`: Scalar sum of weighted values.
    * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` of weighted values divided by
        number of elements in values.
    * `WEIGHTED_MEAN`: Scalar sum of weighted values divided by sum of weights.
    """

    SUM = 'sum'
    SUM_OVER_BATCH_SIZE = 'sum_over_batch_size'
    WEIGHTED_MEAN = 'weighted_mean'


def update_state_wrapper(update_state_fn):
    """Decorator to wrap metric `update_state()` with `add_update()`.

    # Arguments
        update_state_fn: function that accumulates metric statistics.

    # Returns
        Decorated function that wraps `update_state_fn()` with `add_update()`.
    """
    def decorated(metric_obj, *args, **kwargs):
        """Decorated function with `add_update()`."""

        update_op = update_state_fn(*args, **kwargs)
        metric_obj.add_update(update_op)
        return update_op

    return decorated


def result_wrapper(result_fn):
    """Decorator to wrap metric `result()` with identity op.

    Wrapping result in identity so that control dependency between
    update_op from `update_state` and result works in case result returns
    a tensor.

    # Arguments
        result_fn: function that computes the metric result.

    # Returns
        Decorated function that wraps `result()` with identity op.
    """
    def decorated(metric_obj, *args, **kwargs):
        result_t = K.identity(result_fn(*args, **kwargs))
        metric_obj._call_result = result_t
        result_t._is_metric = True
        return result_t

    return decorated


def filter_top_k(x, k):
    """Filters top-k values in the last dim of x and set the rest to NEG_INF.
    Used for computing top-k prediction values in dense labels (which has the same
    shape as predictions) for recall and precision top-k metrics.

    # Arguments
        x: tensor with any dimensions.
        k: the number of values to keep.

    # Returns
        tensor with same shape and dtype as x.
    """
    import tensorflow as tf
    _, top_k_idx = tf.nn.top_k(x, k, sorted=False)
    top_k_mask = K.sum(
        K.one_hot(top_k_idx, x.shape[-1]), axis=-2)
    return x * top_k_mask + NEG_INF * (1 - top_k_mask)


def to_list(x):
    if isinstance(x, list):
        return x
    return [x]


def assert_thresholds_range(thresholds):
    if thresholds is not None:
        invalid_thresholds = [t for t in thresholds if t is None or t < 0 or t > 1]
    if invalid_thresholds:
        raise ValueError(
            'Threshold values must be in [0, 1]. Invalid values: {}'.format(
                invalid_thresholds))


def parse_init_thresholds(thresholds, default_threshold=0.5):
    if thresholds is not None:
        assert_thresholds_range(to_list(thresholds))
    thresholds = to_list(default_threshold if thresholds is None else thresholds)
    return thresholds


class ConfusionMatrix(Enum):
    TRUE_POSITIVES = 'tp'
    FALSE_POSITIVES = 'fp'
    TRUE_NEGATIVES = 'tn'
    FALSE_NEGATIVES = 'fn'


class AUCCurve(Enum):
    """Type of AUC Curve (ROC or PR)."""
    ROC = 'ROC'
    PR = 'PR'

    @staticmethod
    def from_str(key):
        if key in ('pr', 'PR'):
            return AUCCurve.PR
        elif key in ('roc', 'ROC'):
            return AUCCurve.ROC
        else:
            raise ValueError('Invalid AUC curve value "%s".' % key)


class AUCSummationMethod(Enum):
    """Type of AUC summation method.

    https://en.wikipedia.org/wiki/Riemann_sum)

    Contains the following values:
    * 'interpolation': Applies mid-point summation scheme for `ROC` curve. For
    `PR` curve, interpolates (true/false) positives but not the ratio that is
    precision (see Davis & Goadrich 2006 for details).
    * 'minoring': Applies left summation for increasing intervals and right
    summation for decreasing intervals.
    * 'majoring': Applies right summation for increasing intervals and left
    summation for decreasing intervals.
    """
    INTERPOLATION = 'interpolation'
    MAJORING = 'majoring'
    MINORING = 'minoring'

    @staticmethod
    def from_str(key):
        if key in ('interpolation', 'Interpolation'):
            return AUCSummationMethod.INTERPOLATION
        elif key in ('majoring', 'Majoring'):
            return AUCSummationMethod.MAJORING
        elif key in ('minoring', 'Minoring'):
            return AUCSummationMethod.MINORING
        else:
            raise ValueError('Invalid AUC summation method value "%s".' % key)


def weighted_assign_add(label, pred, weights, var):
    # Logical and
    label = K.expand_dims(label, 0)
    pred = K.expand_dims(pred, 0)
    are_different = K.concatenate([label, pred], axis=0)
    label_and_pred = K.all(are_different, axis=0)

    label_and_pred = K.cast(label_and_pred, dtype=K.floatx())
    if weights is not None:
        label_and_pred *= weights
    return K.update_add(var, K.sum(label_and_pred, 1))


def update_confusion_matrix_variables(variables_to_update,
                                      y_true,
                                      y_pred,
                                      thresholds=0.5,
                                      top_k=None,
                                      class_id=None,
                                      sample_weight=None):
    """Returns op to update the given confusion matrix variables.

    For every pair of values in y_true and y_pred:

    true_positive: y_true == True and y_pred > thresholds
    false_negatives: y_true == True and y_pred <= thresholds
    true_negatives: y_true == False and y_pred <= thresholds
    false_positive: y_true == False and y_pred > thresholds

    The results will be weighted and added together. When multiple thresholds are
    provided, we will repeat the same for every threshold.

    For estimation of these metrics over a stream of data, the function creates an
    `update_op` operation that updates the given variables.

    If `sample_weight` is `None`, weights default to 1.
    Use weights of 0 to mask values.

    # Arguments
    variables_to_update: Dictionary with 'tp', 'fn', 'tn', 'fp' as valid keys
      and corresponding variables to update as values.
    y_true: A `Tensor` whose shape matches `y_pred`. Will be cast to `bool`.
    y_pred: A floating point `Tensor` of arbitrary shape and whose values are in
      the range `[0, 1]`.
    thresholds: A float value or a python list or tuple of float thresholds in
      `[0, 1]`, or NEG_INF (used when top_k is set).
    top_k: Optional int, indicates that the positive labels should be limited to
      the top k predictions.
    class_id: Optional int, limits the prediction and labels to the class
      specified by this argument.
    sample_weight: Optional `Tensor` whose rank is either 0, or the same rank as
      `y_true`, and must be broadcastable to `y_true` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `y_true` dimension).

    # Returns
        Update ops.

    # Raises
        ValueError: If `y_pred` and `y_true` have mismatched shapes, or if
            `sample_weight` is not `None` and its shape doesn't match `y_pred`, or if
            `variables_to_update` contains invalid keys.
    """
    if variables_to_update is None:
        return
    y_true = K.cast(y_true, dtype=K.floatx())
    y_pred = K.cast(y_pred, dtype=K.floatx())
    if sample_weight is not None:
        sample_weight = K.cast(sample_weight, dtype=K.floatx())

    if not any(key
               for key in variables_to_update
               if key in list(ConfusionMatrix)):
        raise ValueError(
            'Please provide at least one valid confusion matrix '
            'variable to update. Valid variable key options are: "{}". '
            'Received: "{}"'.format(
                list(ConfusionMatrix), variables_to_update.keys()))

    invalid_keys = [
        key for key in variables_to_update if key not in list(ConfusionMatrix)
    ]
    if invalid_keys:
        raise ValueError(
            'Invalid keys: {}. Valid variable key options are: "{}"'.format(
                invalid_keys, list(ConfusionMatrix)))

    if sample_weight is None:
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
            y_pred, y_true=y_true)
    else:
        y_pred, y_true, sample_weight = (
            losses_utils.squeeze_or_expand_dimensions(
                y_pred, y_true=y_true, sample_weight=sample_weight))

    if top_k is not None:
        y_pred = filter_top_k(y_pred, top_k)
    if class_id is not None:
        y_true = y_true[..., class_id]
        y_pred = y_pred[..., class_id]

    thresholds = to_list(thresholds)
    num_thresholds = len(thresholds)
    num_predictions = K.size(y_pred)

    # Reshape predictions and labels.
    predictions_2d = K.reshape(y_pred, [1, -1])
    labels_2d = K.reshape(
        K.cast(y_true, dtype='bool'), [1, -1])

    # Tile the thresholds for every prediction.
    thresh_tiled = K.tile(
        K.expand_dims(K.constant(thresholds), 1),
        K.cast(
            K.stack([1, num_predictions]),
            dtype='int32',
        )
    )

    # Tile the predictions for every threshold.
    preds_tiled = K.tile(predictions_2d, [num_thresholds, 1])

    # Compare predictions and threshold.
    pred_is_pos = K.greater(preds_tiled, thresh_tiled)

    # Tile labels by number of thresholds
    label_is_pos = K.tile(labels_2d, [num_thresholds, 1])

    if sample_weight is not None:
        weights = losses_utils.broadcast_weights(
            y_pred, K.cast(sample_weight, dtype=K.floatx()))
        weights_tiled = K.tile(
            K.reshape(weights, [1, -1]), [num_thresholds, 1])
    else:
        weights_tiled = None

    update_ops = []
    loop_vars = {
        ConfusionMatrix.TRUE_POSITIVES: (label_is_pos, pred_is_pos),
    }
    update_tn = ConfusionMatrix.TRUE_NEGATIVES in variables_to_update
    update_fp = ConfusionMatrix.FALSE_POSITIVES in variables_to_update
    update_fn = ConfusionMatrix.FALSE_NEGATIVES in variables_to_update

    if update_fn or update_tn:
        pred_is_neg = K.equal(
            pred_is_pos, K.zeros_like(pred_is_pos, dtype=pred_is_pos.dtype))
        loop_vars[ConfusionMatrix.FALSE_NEGATIVES] = (label_is_pos, pred_is_neg)

    if update_fp or update_tn:
        label_is_neg = K.equal(
            label_is_pos, K.zeros_like(label_is_pos, dtype=label_is_pos.dtype))
        loop_vars[ConfusionMatrix.FALSE_POSITIVES] = (label_is_neg, pred_is_pos)
        if update_tn:
            loop_vars[ConfusionMatrix.TRUE_NEGATIVES] = (label_is_neg, pred_is_neg)

    for matrix_cond, (label, pred) in loop_vars.items():
        if matrix_cond in variables_to_update:
            update_ops.append(
                weighted_assign_add(label, pred, weights_tiled,
                                    variables_to_update[matrix_cond]))
    return update_ops
