from enum import Enum

import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.utils.python_utils import to_list

NEG_INF = -1e10


def assert_thresholds_range(thresholds):
    if thresholds is not None:
        invalid_thresholds = [
            t for t in thresholds if t is None or t < 0 or t > 1
        ]
        if invalid_thresholds:
            raise ValueError(
                "Threshold values must be in [0, 1]. "
                f"Received: {invalid_thresholds}"
            )


def parse_init_thresholds(thresholds, default_threshold=0.5):
    if thresholds is not None:
        assert_thresholds_range(to_list(thresholds))
    thresholds = to_list(
        default_threshold if thresholds is None else thresholds
    )
    return thresholds


class ConfusionMatrix(Enum):
    TRUE_POSITIVES = "tp"
    FALSE_POSITIVES = "fp"
    TRUE_NEGATIVES = "tn"
    FALSE_NEGATIVES = "fn"


class AUCCurve(Enum):
    """Type of AUC Curve (ROC or PR)."""

    ROC = "ROC"
    PR = "PR"
    PRGAIN = "PRGAIN"

    @staticmethod
    def from_str(key):
        if key in ("pr", "PR"):
            return AUCCurve.PR
        elif key in ("roc", "ROC"):
            return AUCCurve.ROC
        elif key in ("prgain", "PRGAIN"):
            return AUCCurve.PRGAIN
        else:
            raise ValueError(
                f'Invalid AUC curve value: "{key}". '
                'Expected values are ["PR", "ROC", "PRGAIN"]'
            )


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

    INTERPOLATION = "interpolation"
    MAJORING = "majoring"
    MINORING = "minoring"

    @staticmethod
    def from_str(key):
        if key in ("interpolation", "Interpolation"):
            return AUCSummationMethod.INTERPOLATION
        elif key in ("majoring", "Majoring"):
            return AUCSummationMethod.MAJORING
        elif key in ("minoring", "Minoring"):
            return AUCSummationMethod.MINORING
        else:
            raise ValueError(
                f'Invalid AUC summation method value: "{key}". '
                'Expected values are ["interpolation", "majoring", "minoring"]'
            )


def _update_confusion_matrix_variables_optimized(
    variables_to_update,
    y_true,
    y_pred,
    thresholds,
    multi_label=False,
    sample_weights=None,
    label_weights=None,
    thresholds_with_epsilon=False,
):
    """Update confusion matrix variables with memory efficient alternative.

    Note that the thresholds need to be evenly distributed within the list, eg,
    the diff between consecutive elements are the same.

    To compute TP/FP/TN/FN, we are measuring a binary classifier
      C(t) = (predictions >= t)
    at each threshold 't'. So we have
      TP(t) = sum( C(t) * true_labels )
      FP(t) = sum( C(t) * false_labels )

    But, computing C(t) requires computation for each t. To make it fast,
    observe that C(t) is a cumulative integral, and so if we have
      thresholds = [t_0, ..., t_{n-1}];  t_0 < ... < t_{n-1}
    where n = num_thresholds, and if we can compute the bucket function
      B(i) = Sum( (predictions == t), t_i <= t < t{i+1} )
    then we get
      C(t_i) = sum( B(j), j >= i )
    which is the reversed cumulative sum in ops.cumsum().

    We can compute B(i) efficiently by taking advantage of the fact that
    our thresholds are evenly distributed, in that
      width = 1.0 / (num_thresholds - 1)
      thresholds = [0.0, 1*width, 2*width, 3*width, ..., 1.0]
    Given a prediction value p, we can map it to its bucket by
      bucket_index(p) = floor( p * (num_thresholds - 1) )
    so we can use ops.segment_sum() to update the buckets in one pass.

    Consider following example:
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.5, 0.3, 0.9]
    thresholds = [0.0, 0.5, 1.0]
    num_buckets = 2   # [0.0, 1.0], (1.0, 2.0]
    bucket_index(y_pred) = ops.floor(y_pred * num_buckets)
                         = ops.floor([0.2, 1.0, 0.6, 1.8])
                         = [0, 0, 0, 1]
    # The meaning of this bucket is that if any of the label is true,
    # then 1 will be added to the corresponding bucket with the index.
    # Eg, if the label for 0.2 is true, then 1 will be added to bucket 0. If the
    # label for 1.8 is true, then 1 will be added to bucket 1.
    #
    # Note the second item "1.0" is floored to 0, since the value need to be
    # strictly larger than the bucket lower bound.
    # In the implementation, we use ops.ceil() - 1 to achieve this.
    tp_bucket_value = ops.segment_sum(true_labels, bucket_indices,
                                                   num_segments=num_thresholds)
                    = [1, 1, 0]
    # For [1, 1, 0] here, it means there is 1 true value contributed by bucket
    # 0, and 1 value contributed by bucket 1. When we aggregate them to
    # together, the result become [a + b + c, b + c, c], since large thresholds
    # will always contribute to the value for smaller thresholds.
    true_positive = ops.cumsum(tp_bucket_value, reverse=True)
                  = [2, 1, 0]

    This implementation exhibits a run time and space complexity of O(T + N),
    where T is the number of thresholds and N is the size of predictions.
    Metrics that rely on standard implementation instead exhibit a complexity of
    O(T * N).

    Args:
        variables_to_update: Dictionary with 'tp', 'fn', 'tn', 'fp' as valid
            keys and corresponding variables to update as values.
        y_true: A floating point `Tensor` whose shape matches `y_pred`. Will be
            cast to `bool`.
        y_pred: A floating point `Tensor` of arbitrary shape and whose values
            are in the range `[0, 1]`.
        thresholds: A sorted floating point `Tensor` with value in `[0, 1]`.
            It need to be evenly distributed (the diff between each element need
            to be the same).
        multi_label: Optional boolean indicating whether multidimensional
            prediction/labels should be treated as multilabel responses, or
            flattened into a single label. When True, the values of
            `variables_to_update` must have a second dimension equal to the
            number of labels in y_true and y_pred, and those tensors must not be
            RaggedTensors.
        sample_weights: Optional `Tensor` whose rank is either 0, or the same
            rank as `y_true`, and must be broadcastable to `y_true` (i.e., all
            dimensions must be either `1`, or the same as the corresponding
            `y_true` dimension).
        label_weights: Optional tensor of non-negative weights for multilabel
            data. The weights are applied when calculating TP, FP, FN, and TN
            without explicit multilabel handling (i.e. when the data is to be
            flattened).
        thresholds_with_epsilon: Optional boolean indicating whether the leading
            and tailing thresholds has any epsilon added for floating point
            imprecisions.  It will change how we handle the leading and tailing
            bucket.
    """
    num_thresholds = ops.shape(thresholds)[0]

    if sample_weights is None:
        sample_weights = 1.0
    else:
        sample_weights = ops.broadcast_to(
            ops.cast(sample_weights, dtype=y_pred.dtype), ops.shape(y_pred)
        )
        if not multi_label:
            sample_weights = ops.reshape(sample_weights, [-1])
    if label_weights is None:
        label_weights = 1.0
    else:
        label_weights = ops.expand_dims(label_weights, 0)
        label_weights = ops.broadcast_to(label_weights, ops.shape(y_pred))
        if not multi_label:
            label_weights = ops.reshape(label_weights, [-1])
    weights = ops.cast(
        ops.multiply(sample_weights, label_weights), y_true.dtype
    )

    # We shouldn't need this, but in case there are predict value that is out of
    # the range of [0.0, 1.0]
    y_pred = ops.clip(y_pred, x_min=0.0, x_max=1.0)

    y_true = ops.cast(ops.cast(y_true, "bool"), y_true.dtype)
    if not multi_label:
        y_true = ops.reshape(y_true, [-1])
        y_pred = ops.reshape(y_pred, [-1])

    true_labels = ops.multiply(y_true, weights)
    false_labels = ops.multiply((1.0 - y_true), weights)

    # Compute the bucket indices for each prediction value.
    # Since the predict value has to be strictly greater than the thresholds,
    # eg, buckets like [0, 0.5], (0.5, 1], and 0.5 belongs to first bucket.
    # We have to use math.ceil(val) - 1 for the bucket.
    bucket_indices = (
        ops.ceil(y_pred * (ops.cast(num_thresholds, dtype=y_pred.dtype) - 1))
        - 1
    )

    if thresholds_with_epsilon:
        # In this case, the first bucket should actually take into account since
        # the any prediction between [0.0, 1.0] should be larger than the first
        # threshold. We change the bucket value from -1 to 0.
        bucket_indices = ops.relu(bucket_indices)

    bucket_indices = ops.cast(bucket_indices, "int32")

    if multi_label:
        # We need to run bucket segment sum for each of the label class. In the
        # multi_label case, the rank of the label is 2. We first transpose it so
        # that the label dim becomes the first and we can parallel run though
        # them.
        true_labels = ops.transpose(true_labels)
        false_labels = ops.transpose(false_labels)
        bucket_indices = ops.transpose(bucket_indices)

        def gather_bucket(label_and_bucket_index):
            label, bucket_index = (
                label_and_bucket_index[0],
                label_and_bucket_index[1],
            )
            return ops.segment_sum(
                data=label,
                segment_ids=bucket_index,
                num_segments=num_thresholds,
            )

        tp_bucket_v = backend.vectorized_map(
            gather_bucket,
            (true_labels, bucket_indices),
        )
        fp_bucket_v = backend.vectorized_map(
            gather_bucket, (false_labels, bucket_indices)
        )
        tp = ops.transpose(ops.flip(ops.cumsum(ops.flip(tp_bucket_v), axis=1)))
        fp = ops.transpose(ops.flip(ops.cumsum(ops.flip(fp_bucket_v), axis=1)))
    else:
        tp_bucket_v = ops.segment_sum(
            data=true_labels,
            segment_ids=bucket_indices,
            num_segments=num_thresholds,
        )
        fp_bucket_v = ops.segment_sum(
            data=false_labels,
            segment_ids=bucket_indices,
            num_segments=num_thresholds,
        )
        tp = ops.flip(ops.cumsum(ops.flip(tp_bucket_v)))
        fp = ops.flip(ops.cumsum(ops.flip(fp_bucket_v)))

    # fn = sum(true_labels) - tp
    # tn = sum(false_labels) - fp
    if (
        ConfusionMatrix.TRUE_NEGATIVES in variables_to_update
        or ConfusionMatrix.FALSE_NEGATIVES in variables_to_update
    ):
        if multi_label:
            total_true_labels = ops.sum(true_labels, axis=1)
            total_false_labels = ops.sum(false_labels, axis=1)
        else:
            total_true_labels = ops.sum(true_labels)
            total_false_labels = ops.sum(false_labels)

    if ConfusionMatrix.TRUE_POSITIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.TRUE_POSITIVES]
        variable.assign(variable + tp)
    if ConfusionMatrix.FALSE_POSITIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.FALSE_POSITIVES]
        variable.assign(variable + fp)
    if ConfusionMatrix.TRUE_NEGATIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.TRUE_NEGATIVES]
        tn = total_false_labels - fp
        variable.assign(variable + tn)
    if ConfusionMatrix.FALSE_NEGATIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.FALSE_NEGATIVES]
        fn = total_true_labels - tp
        variable.assign(variable + fn)


def is_evenly_distributed_thresholds(thresholds):
    """Check if the thresholds list is evenly distributed.

    We could leverage evenly distributed thresholds to use less memory when
    calculate metrics like AUC where each individual threshold need to be
    evaluated.

    Args:
      thresholds: A python list or tuple, or 1D numpy array whose value is
        ranged in [0, 1].

    Returns:
      boolean, whether the values in the inputs are evenly distributed.
    """
    # Check the list value and see if it is evenly distributed.
    num_thresholds = len(thresholds)
    if num_thresholds < 3:
        return False
    even_thresholds = np.arange(num_thresholds, dtype=np.float32) / (
        num_thresholds - 1
    )
    return np.allclose(thresholds, even_thresholds, atol=backend.epsilon())


def update_confusion_matrix_variables(
    variables_to_update,
    y_true,
    y_pred,
    thresholds,
    top_k=None,
    class_id=None,
    sample_weight=None,
    multi_label=False,
    label_weights=None,
    thresholds_distributed_evenly=False,
):
    """Updates the given confusion matrix variables.

    For every pair of values in y_true and y_pred:

    true_positive: y_true == True and y_pred > thresholds
    false_negatives: y_true == True and y_pred <= thresholds
    true_negatives: y_true == False and y_pred <= thresholds
    false_positive: y_true == False and y_pred > thresholds

    The results will be weighted and added together. When multiple thresholds
    are provided, we will repeat the same for every threshold.

    For estimation of these metrics over a stream of data, the function creates
    an `update_op` operation that updates the given variables.

    If `sample_weight` is `None`, weights default to 1.
    Use weights of 0 to mask values.

    Args:
      variables_to_update: Dictionary with 'tp', 'fn', 'tn', 'fp' as valid keys
        and corresponding variables to update as values.
      y_true: A `Tensor` whose shape matches `y_pred`. Will be cast to `bool`.
      y_pred: A floating point `Tensor` of arbitrary shape and whose values are
        in the range `[0, 1]`.
      thresholds: A float value, float tensor, python list, or tuple of float
        thresholds in `[0, 1]`, or NEG_INF (used when top_k is set).
      top_k: Optional int, indicates that the positive labels should be limited
        to the top k predictions.
      class_id: Optional int, limits the prediction and labels to the class
        specified by this argument.
      sample_weight: Optional `Tensor` whose rank is either 0, or the same rank
        as `y_true`, and must be broadcastable to `y_true` (i.e., all dimensions
        must be either `1`, or the same as the corresponding `y_true`
        dimension).
      multi_label: Optional boolean indicating whether multidimensional
        prediction/labels should be treated as multilabel responses, or
        flattened into a single label. When True, the values of
        `variables_to_update` must have a second dimension equal to the number
        of labels in y_true and y_pred, and those tensors must not be
        RaggedTensors.
      label_weights: (optional) tensor of non-negative weights for multilabel
        data. The weights are applied when calculating TP, FP, FN, and TN
        without explicit multilabel handling (i.e. when the data is to be
        flattened).
      thresholds_distributed_evenly: Boolean, whether the thresholds are evenly
        distributed within the list. An optimized method will be used if this is
        the case. See _update_confusion_matrix_variables_optimized() for more
        details.

    Raises:
      ValueError: If `y_pred` and `y_true` have mismatched shapes, or if
        `sample_weight` is not `None` and its shape doesn't match `y_pred`, or
        if `variables_to_update` contains invalid keys.
    """
    if multi_label and label_weights is not None:
        raise ValueError(
            "`label_weights` for multilabel data should be handled "
            "outside of `update_confusion_matrix_variables` when "
            "`multi_label` is True."
        )
    if variables_to_update is None:
        return
    if not any(
        key for key in variables_to_update if key in list(ConfusionMatrix)
    ):
        raise ValueError(
            "Please provide at least one valid confusion matrix "
            "variable to update. Valid variable key options are: "
            f'"{list(ConfusionMatrix)}". '
            f'Received: "{variables_to_update.keys()}"'
        )

    variable_dtype = list(variables_to_update.values())[0].dtype

    y_true = ops.cast(y_true, dtype=variable_dtype)
    y_pred = ops.cast(y_pred, dtype=variable_dtype)

    if thresholds_distributed_evenly:
        # Check whether the thresholds has any leading or tailing epsilon added
        # for floating point imprecision. The leading and tailing threshold will
        # be handled bit differently as the corner case.  At this point,
        # thresholds should be a list/array with more than 2 items, and ranged
        # between [0, 1]. See is_evenly_distributed_thresholds() for more
        # details.
        thresholds_with_epsilon = thresholds[0] < 0.0 or thresholds[-1] > 1.0

    thresholds = ops.convert_to_tensor(thresholds, dtype=variable_dtype)
    num_thresholds = ops.shape(thresholds)[0]

    if multi_label:
        one_thresh = ops.equal(
            np.array(1, dtype="int32"),
            len(thresholds.shape),
        )
    else:
        one_thresh = np.array(True, dtype="bool")

    invalid_keys = [
        key for key in variables_to_update if key not in list(ConfusionMatrix)
    ]
    if invalid_keys:
        raise ValueError(
            f'Invalid keys: "{invalid_keys}". '
            f'Valid variable key options are: "{list(ConfusionMatrix)}"'
        )

    y_pred, y_true = squeeze_or_expand_to_same_rank(y_pred, y_true)
    if sample_weight is not None:
        sample_weight = ops.expand_dims(
            ops.cast(sample_weight, dtype=variable_dtype), axis=-1
        )
        _, sample_weight = squeeze_or_expand_to_same_rank(
            y_true, sample_weight, expand_rank_1=False
        )

    if top_k is not None:
        y_pred = _filter_top_k(y_pred, top_k)

    if class_id is not None:
        if len(y_pred.shape) == 1:
            raise ValueError(
                "When class_id is provided, y_pred must be a 2D array "
                "with shape (num_samples, num_classes), found shape: "
                f"{y_pred.shape}"
            )

        # Preserve dimension to match with sample_weight
        y_true = y_true[..., class_id, None]
        y_pred = y_pred[..., class_id, None]

    if thresholds_distributed_evenly:
        return _update_confusion_matrix_variables_optimized(
            variables_to_update,
            y_true,
            y_pred,
            thresholds,
            multi_label=multi_label,
            sample_weights=sample_weight,
            label_weights=label_weights,
            thresholds_with_epsilon=thresholds_with_epsilon,
        )

    if None in y_pred.shape:
        pred_shape = ops.shape(y_pred)
        num_predictions = pred_shape[0]
        if len(y_pred.shape) == 1:
            num_labels = 1
        else:
            num_labels = ops.cast(
                ops.prod(ops.array(pred_shape[1:]), axis=0), "int32"
            )
        thresh_label_tile = ops.where(one_thresh, num_labels, 1)
    else:
        pred_shape = ops.shape(y_pred)
        num_predictions = pred_shape[0]
        if len(y_pred.shape) == 1:
            num_labels = 1
        else:
            num_labels = np.prod(pred_shape[1:], axis=0).astype("int32")
        thresh_label_tile = np.where(one_thresh, num_labels, 1)

    # Reshape predictions and labels, adding a dim for thresholding.
    if multi_label:
        predictions_extra_dim = ops.expand_dims(y_pred, 0)
        labels_extra_dim = ops.expand_dims(ops.cast(y_true, dtype="bool"), 0)
    else:
        # Flatten predictions and labels when not multilabel.
        predictions_extra_dim = ops.reshape(y_pred, [1, -1])
        labels_extra_dim = ops.reshape(ops.cast(y_true, dtype="bool"), [1, -1])

    # Tile the thresholds for every prediction.
    if multi_label:
        thresh_pretile_shape = [num_thresholds, 1, -1]
        thresh_tiles = [1, num_predictions, thresh_label_tile]
        data_tiles = [num_thresholds, 1, 1]
    else:
        thresh_pretile_shape = [num_thresholds, -1]
        thresh_tiles = [1, num_predictions * num_labels]
        data_tiles = [num_thresholds, 1]

    thresh_tiled = ops.tile(
        ops.reshape(thresholds, thresh_pretile_shape), thresh_tiles
    )

    # Tile the predictions for every threshold.
    preds_tiled = ops.tile(predictions_extra_dim, data_tiles)

    # Compare predictions and threshold.
    pred_is_pos = ops.greater(preds_tiled, thresh_tiled)

    # Tile labels by number of thresholds
    label_is_pos = ops.tile(labels_extra_dim, data_tiles)

    if sample_weight is not None:
        sample_weight = ops.broadcast_to(
            ops.cast(sample_weight, dtype=y_pred.dtype), ops.shape(y_pred)
        )
        weights_tiled = ops.tile(
            ops.reshape(sample_weight, thresh_tiles), data_tiles
        )
    else:
        weights_tiled = None

    if label_weights is not None and not multi_label:
        label_weights = ops.expand_dims(label_weights, 0)
        label_weights = ops.broadcast_to(label_weights, ops.shape(y_pred))
        label_weights_tiled = ops.tile(
            ops.reshape(label_weights, thresh_tiles), data_tiles
        )
        if weights_tiled is None:
            weights_tiled = label_weights_tiled
        else:
            weights_tiled = ops.multiply(weights_tiled, label_weights_tiled)

    def weighted_assign_add(label, pred, weights, var):
        label_and_pred = ops.cast(ops.logical_and(label, pred), dtype=var.dtype)
        if weights is not None:
            label_and_pred *= ops.cast(weights, dtype=var.dtype)
        var.assign(var + ops.sum(label_and_pred, 1))

    loop_vars = {
        ConfusionMatrix.TRUE_POSITIVES: (label_is_pos, pred_is_pos),
    }
    update_tn = ConfusionMatrix.TRUE_NEGATIVES in variables_to_update
    update_fp = ConfusionMatrix.FALSE_POSITIVES in variables_to_update
    update_fn = ConfusionMatrix.FALSE_NEGATIVES in variables_to_update

    if update_fn or update_tn:
        pred_is_neg = ops.logical_not(pred_is_pos)
        loop_vars[ConfusionMatrix.FALSE_NEGATIVES] = (label_is_pos, pred_is_neg)

    if update_fp or update_tn:
        label_is_neg = ops.logical_not(label_is_pos)
        loop_vars[ConfusionMatrix.FALSE_POSITIVES] = (label_is_neg, pred_is_pos)
        if update_tn:
            loop_vars[ConfusionMatrix.TRUE_NEGATIVES] = (
                label_is_neg,
                pred_is_neg,
            )

    for matrix_cond, (label, pred) in loop_vars.items():
        if matrix_cond in variables_to_update:
            weighted_assign_add(
                label, pred, weights_tiled, variables_to_update[matrix_cond]
            )


def _filter_top_k(x, k):
    """Filters top-k values in the last dim of x and set the rest to NEG_INF.

    Used for computing top-k prediction values in dense labels (which has the
    same shape as predictions) for recall and precision top-k metrics.

    Args:
      x: tensor with any dimensions.
      k: the number of values to keep.

    Returns:
      tensor with same shape and dtype as x.
    """
    _, top_k_idx = ops.top_k(x, k)
    top_k_mask = ops.sum(
        ops.one_hot(top_k_idx, ops.shape(x)[-1], axis=-1), axis=-2
    )
    return x * top_k_mask + NEG_INF * (1 - top_k_mask)


def confusion_matrix(
    labels,
    predictions,
    num_classes,
    weights=None,
    dtype="int32",
):
    """Computes the confusion matrix from predictions and labels.

    The matrix columns represent the prediction labels and the rows represent
    the real labels. The confusion matrix is always a 2-D array of shape
    `(n, n)`, where `n` is the number of valid labels for a given classification
    task. Both prediction and labels must be 1-D arrays of the same shape in
    order for this function to work.

    If `num_classes` is `None`, then `num_classes` will be set to one plus the
    maximum value in either predictions or labels. Class labels are expected to
    start at 0. For example, if `num_classes` is 3, then the possible labels
    would be `[0, 1, 2]`.

    If `weights` is not `None`, then each prediction contributes its
    corresponding weight to the total value of the confusion matrix cell.

    For example:

    ```python
    keras.metrics.metrics_utils.confusion_matrix([1, 2, 4], [2, 2, 4]) ==>
        [[0 0 0 0 0]
        [0 0 1 0 0]
        [0 0 1 0 0]
        [0 0 0 0 0]
        [0 0 0 0 1]]
    ```

    Note that the possible labels are assumed to be `[0, 1, 2, 3, 4]`,
    resulting in a 5x5 confusion matrix.

    Args:
        labels: 1-D tensor of real labels for the classification task.
        predictions: 1-D tensor of predictions for a given classification.
        num_classes: The possible number of labels the classification
            task can have.
        weights: An optional tensor whose shape matches `predictions`.
        dtype: Data type of the confusion matrix.

    Returns:
        A tensor of type `dtype` with shape `(n, n)` representing the confusion
        matrix, where `n` is the number of possible labels in the classification
        task.
    """
    labels = ops.convert_to_tensor(labels, dtype)
    predictions = ops.convert_to_tensor(predictions, dtype)
    labels, predictions = squeeze_or_expand_to_same_rank(labels, predictions)

    predictions = ops.cast(predictions, dtype)
    labels = ops.cast(labels, dtype)

    if weights is not None:
        weights = ops.convert_to_tensor(weights, dtype)

    indices = ops.stack([labels, predictions], axis=1)
    values = ops.ones_like(predictions, dtype) if weights is None else weights
    indices = ops.cast(indices, dtype="int64")
    values = ops.cast(values, dtype=dtype)
    num_classes = int(num_classes)
    confusion_matrix = ops.scatter(indices, values, (num_classes, num_classes))
    return confusion_matrix
