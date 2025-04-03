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

"""Utils related to keras metrics."""

import functools
import weakref
from enum import Enum

import numpy as np
import tensorflow.compat.v2 as tf

from tf_keras.src import backend
from tf_keras.src.utils import losses_utils
from tf_keras.src.utils import tf_utils
from tf_keras.src.utils.generic_utils import to_list

NEG_INF = -1e10


class Reduction(Enum):
    """Types of metrics reduction.

    Contains the following values:

    * `SUM`: Scalar sum of weighted values.
    * `SUM_OVER_BATCH_SIZE`: Scalar sum of weighted values divided by
          number of elements.
    * `WEIGHTED_MEAN`: Scalar sum of weighted values divided by sum of weights.
    """

    SUM = "sum"
    SUM_OVER_BATCH_SIZE = "sum_over_batch_size"
    WEIGHTED_MEAN = "weighted_mean"


def update_state_wrapper(update_state_fn):
    """Decorator to wrap metric `update_state()` with `add_update()`.

    Args:
      update_state_fn: function that accumulates metric statistics.

    Returns:
      Decorated function that wraps `update_state_fn()` with `add_update()`.
    """

    def decorated(metric_obj, *args, **kwargs):
        """Decorated function with `add_update()`."""
        strategy = tf.distribute.get_strategy()

        for weight in metric_obj.weights:
            if (
                backend.is_tpu_strategy(strategy)
                and not strategy.extended.variable_created_in_scope(weight)
                and not tf.distribute.in_cross_replica_context()
            ):
                raise ValueError(
                    "Trying to run metric.update_state in replica context when "
                    "the metric was not created in TPUStrategy scope. "
                    "Make sure the keras Metric is created in TPUstrategy "
                    "scope. "
                )

        with tf_utils.graph_context_for_symbolic_tensors(*args, **kwargs):
            result = update_state_fn(*args, **kwargs)
        if not tf.executing_eagerly():
            result = tf.compat.v1.get_default_graph().get_operations()[-1]
            metric_obj.add_update(result)
        return result

    return tf.__internal__.decorator.make_decorator(update_state_fn, decorated)


def result_wrapper(result_fn):
    """Decorator to wrap metric `result()` function in `merge_call()`.

    Result computation is an idempotent operation that simply calculates the
    metric value using the state variables.

    If metric state variables are distributed across replicas/devices and
    `result()` is requested from the context of one device - This function wraps
    `result()` in a distribution strategy `merge_call()`. With this,
    the metric state variables will be aggregated across devices.

    Args:
      result_fn: function that computes the metric result.

    Returns:
      Decorated function that wraps `result_fn()` in distribution strategy
      `merge_call()`.
    """

    def decorated(metric_obj, *args):
        """Decorated function with merge_call."""
        replica_context = tf.distribute.get_replica_context()

        # The purpose of using `merge_call` to call `result()` is to trigger
        # cross replica aggregation of metric state variables
        # (SyncOnReadVariable). After we introduced
        # `variable_sync_on_read_context`, in principle there is no need to use
        # `merge_call` here. However the branch still exists because:
        #
        # 1. TF-Keras V1 training code sometimes assumes `result_t` is the same
        #    tensor across replicas (achieved by `merge_call`). With
        #    `variable_sync_on_read_context` each replica gets their own tensors
        #    residing on replica's device, thus breaking the assumption.
        # 2. TF-Keras c/fit creates a tf.function (a.k.a, train_function) that
        #    returns the metric values of the first replica. With
        #    `variable_sync_on_read_context` since each replica gets their own
        #    tensors, the metric result tensors on the non-first replicas are
        #    not in the return value of train_function, making TF graph
        #    optimizer prune the branch that computes and aggregates those
        #    metric results. As a result, if NCCL is used to do the aggregation,
        #    the program will hang because NCCL ops are only launched on the
        #    non-pruned first replica.
        #
        # We condition on strategy_supports_no_merge_call() since we know if it
        # is True, the program uses `jit_compile` to compile replica fn, meaning
        # it is not V1 training (hence #1 is okay), and no pruning will happen
        # as compiled functions are not inlined (hence #2 is okay).
        if (
            replica_context is None
            or tf.__internal__.distribute.strategy_supports_no_merge_call()
        ):
            with tf.__internal__.distribute.variable_sync_on_read_context():
                raw_result = result_fn(*args)
                # Results need to be wrapped in a `tf.identity` op to ensure
                # correct execution order.
                if isinstance(raw_result, (tf.Tensor, tf.Variable, float, int)):
                    result_t = tf.identity(raw_result)
                elif isinstance(raw_result, dict):
                    result_t = tf.nest.map_structure(tf.identity, raw_result)
                else:
                    try:
                        result_t = tf.identity(raw_result)
                    except (ValueError, TypeError):
                        raise RuntimeError(
                            "The output of `metric.result()` can only be a "
                            "single Tensor/Variable, or a dict of "
                            "Tensors/Variables. "
                            f"For metric {metric_obj.name}, "
                            f"got result {raw_result}."
                        )
        else:
            # TODO(psv): Test distribution of metrics using different
            # distribution strategies.

            # Creating a wrapper for merge_fn. merge_call invokes the given
            # merge_fn with distribution object as the first parameter. We
            # create a wrapper here so that the result function need not have
            # that parameter.
            def merge_fn_wrapper(distribution, merge_fn, *args):
                # We will get `PerReplica` merge function. Taking the first one
                # as all are identical copies of the function that we had passed
                # below.
                result = distribution.experimental_local_results(merge_fn)[0](
                    *args
                )

                # Wrapping result in identity so that control dependency between
                # update_op from `update_state` and result works in case result
                # returns a tensor.
                return tf.nest.map_structure(tf.identity, result)

            # Wrapping result in merge_call. merge_call is used when we want to
            # leave replica mode and compute a value in cross replica mode.
            result_t = replica_context.merge_call(
                merge_fn_wrapper, args=(result_fn,) + args
            )

        # We are saving the result op here to be used in train/test execution
        # functions. This basically gives the result op that was generated with
        # a control dep to the updates for these workflows.
        metric_obj._call_result = result_t
        return result_t

    return tf.__internal__.decorator.make_decorator(result_fn, decorated)


def weakmethod(method):
    """Creates a weak reference to the bound method."""

    cls = method.im_class
    func = method.im_func
    instance_ref = weakref.ref(method.im_self)

    @functools.wraps(method)
    def inner(*args, **kwargs):
        return func.__get__(instance_ref(), cls)(*args, **kwargs)

    del method
    return inner


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

    @staticmethod
    def from_str(key):
        if key in ("pr", "PR"):
            return AUCCurve.PR
        elif key in ("roc", "ROC"):
            return AUCCurve.ROC
        else:
            raise ValueError(
                f'Invalid AUC curve value: "{key}". '
                'Expected values are ["PR", "ROC"]'
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
    which is the reversed cumulative sum in tf.cumsum().

    We can compute B(i) efficiently by taking advantage of the fact that
    our thresholds are evenly distributed, in that
      width = 1.0 / (num_thresholds - 1)
      thresholds = [0.0, 1*width, 2*width, 3*width, ..., 1.0]
    Given a prediction value p, we can map it to its bucket by
      bucket_index(p) = floor( p * (num_thresholds - 1) )
    so we can use tf.math.unsorted_segment_sum() to update the buckets in one
    pass.

    Consider following example:
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.5, 0.3, 0.9]
    thresholds = [0.0, 0.5, 1.0]
    num_buckets = 2   # [0.0, 1.0], (1.0, 2.0]
    bucket_index(y_pred) = tf.math.floor(y_pred * num_buckets)
                         = tf.math.floor([0.2, 1.0, 0.6, 1.8])
                         = [0, 0, 0, 1]
    # The meaning of this bucket is that if any of the label is true,
    # then 1 will be added to the corresponding bucket with the index.
    # Eg, if the label for 0.2 is true, then 1 will be added to bucket 0. If the
    # label for 1.8 is true, then 1 will be added to bucket 1.
    #
    # Note the second item "1.0" is floored to 0, since the value need to be
    # strictly larger than the bucket lower bound.
    # In the implementation, we use tf.math.ceil() - 1 to achieve this.
    tp_bucket_value = tf.math.unsorted_segment_sum(true_labels, bucket_indices,
                                                   num_segments=num_thresholds)
                    = [1, 1, 0]
    # For [1, 1, 0] here, it means there is 1 true value contributed by bucket
    # 0, and 1 value contributed by bucket 1. When we aggregate them to
    # together, the result become [a + b + c, b + c, c], since large thresholds
    # will always contribute to the value for smaller thresholds.
    true_positive = tf.math.cumsum(tp_bucket_value, reverse=True)
                  = [2, 1, 0]

    This implementation exhibits a run time and space complexity of O(T + N),
    where T is the number of thresholds and N is the size of predictions.
    Metrics that rely on standard implementation instead exhibit a complexity of
    O(T * N).

    Args:
      variables_to_update: Dictionary with 'tp', 'fn', 'tn', 'fp' as valid keys
        and corresponding variables to update as values.
      y_true: A floating point `Tensor` whose shape matches `y_pred`. Will be
        cast to `bool`.
      y_pred: A floating point `Tensor` of arbitrary shape and whose values are
        in the range `[0, 1]`.
      thresholds: A sorted floating point `Tensor` with value in `[0, 1]`.
        It need to be evenly distributed (the diff between each element need to
        be the same).
      multi_label: Optional boolean indicating whether multidimensional
        prediction/labels should be treated as multilabel responses, or
        flattened into a single label. When True, the valus of
        `variables_to_update` must have a second dimension equal to the number
        of labels in y_true and y_pred, and those tensors must not be
        RaggedTensors.
      sample_weights: Optional `Tensor` whose rank is either 0, or the same rank
        as `y_true`, and must be broadcastable to `y_true` (i.e., all dimensions
        must be either `1`, or the same as the corresponding `y_true`
        dimension).
      label_weights: Optional tensor of non-negative weights for multilabel
        data. The weights are applied when calculating TP, FP, FN, and TN
        without explicit multilabel handling (i.e. when the data is to be
        flattened).
      thresholds_with_epsilon: Optional boolean indicating whether the leading
        and tailing thresholds has any epsilon added for floating point
        imprecisions.  It will change how we handle the leading and tailing
        bucket.

    Returns:
      Update op.
    """
    num_thresholds = thresholds.shape.as_list()[0]

    if sample_weights is None:
        sample_weights = 1.0
    else:
        sample_weights = tf.__internal__.ops.broadcast_weights(
            tf.cast(sample_weights, dtype=y_pred.dtype), y_pred
        )
        if not multi_label:
            sample_weights = tf.reshape(sample_weights, [-1])
    if label_weights is None:
        label_weights = 1.0
    else:
        label_weights = tf.expand_dims(label_weights, 0)
        label_weights = tf.__internal__.ops.broadcast_weights(
            label_weights, y_pred
        )
        if not multi_label:
            label_weights = tf.reshape(label_weights, [-1])
    weights = tf.cast(tf.multiply(sample_weights, label_weights), y_true.dtype)

    # We shouldn't need this, but in case there are predict value that is out of
    # the range of [0.0, 1.0]
    y_pred = tf.clip_by_value(y_pred, clip_value_min=0.0, clip_value_max=1.0)

    y_true = tf.cast(tf.cast(y_true, tf.bool), y_true.dtype)
    if not multi_label:
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

    true_labels = tf.multiply(y_true, weights)
    false_labels = tf.multiply((1.0 - y_true), weights)

    # Compute the bucket indices for each prediction value.
    # Since the predict value has to be strictly greater than the thresholds,
    # eg, buckets like [0, 0.5], (0.5, 1], and 0.5 belongs to first bucket.
    # We have to use math.ceil(val) - 1 for the bucket.
    bucket_indices = tf.math.ceil(y_pred * (num_thresholds - 1)) - 1

    if thresholds_with_epsilon:
        # In this case, the first bucket should actually take into account since
        # the any prediction between [0.0, 1.0] should be larger than the first
        # threshold. We change the bucket value from -1 to 0.
        bucket_indices = tf.nn.relu(bucket_indices)

    bucket_indices = tf.cast(bucket_indices, tf.int32)

    if multi_label:
        # We need to run bucket segment sum for each of the label class. In the
        # multi_label case, the rank of the label is 2. We first transpose it so
        # that the label dim becomes the first and we can parallel run though
        # them.
        true_labels = tf.transpose(true_labels)
        false_labels = tf.transpose(false_labels)
        bucket_indices = tf.transpose(bucket_indices)

        def gather_bucket(label_and_bucket_index):
            label, bucket_index = (
                label_and_bucket_index[0],
                label_and_bucket_index[1],
            )
            return tf.math.unsorted_segment_sum(
                data=label,
                segment_ids=bucket_index,
                num_segments=num_thresholds,
            )

        tp_bucket_v = tf.vectorized_map(
            gather_bucket, (true_labels, bucket_indices), warn=False
        )
        fp_bucket_v = tf.vectorized_map(
            gather_bucket, (false_labels, bucket_indices), warn=False
        )
        tp = tf.transpose(tf.cumsum(tp_bucket_v, reverse=True, axis=1))
        fp = tf.transpose(tf.cumsum(fp_bucket_v, reverse=True, axis=1))
    else:
        tp_bucket_v = tf.math.unsorted_segment_sum(
            data=true_labels,
            segment_ids=bucket_indices,
            num_segments=num_thresholds,
        )
        fp_bucket_v = tf.math.unsorted_segment_sum(
            data=false_labels,
            segment_ids=bucket_indices,
            num_segments=num_thresholds,
        )
        tp = tf.cumsum(tp_bucket_v, reverse=True)
        fp = tf.cumsum(fp_bucket_v, reverse=True)

    # fn = sum(true_labels) - tp
    # tn = sum(false_labels) - fp
    if (
        ConfusionMatrix.TRUE_NEGATIVES in variables_to_update
        or ConfusionMatrix.FALSE_NEGATIVES in variables_to_update
    ):
        if multi_label:
            total_true_labels = tf.reduce_sum(true_labels, axis=1)
            total_false_labels = tf.reduce_sum(false_labels, axis=1)
        else:
            total_true_labels = tf.reduce_sum(true_labels)
            total_false_labels = tf.reduce_sum(false_labels)

    update_ops = []
    if ConfusionMatrix.TRUE_POSITIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.TRUE_POSITIVES]
        update_ops.append(variable.assign_add(tp))
    if ConfusionMatrix.FALSE_POSITIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.FALSE_POSITIVES]
        update_ops.append(variable.assign_add(fp))
    if ConfusionMatrix.TRUE_NEGATIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.TRUE_NEGATIVES]
        tn = total_false_labels - fp
        update_ops.append(variable.assign_add(tn))
    if ConfusionMatrix.FALSE_NEGATIVES in variables_to_update:
        variable = variables_to_update[ConfusionMatrix.FALSE_NEGATIVES]
        fn = total_true_labels - tp
        update_ops.append(variable.assign_add(fn))
    return tf.group(update_ops)


def is_evenly_distributed_thresholds(thresholds):
    """Check if the thresholds list is evenly distributed.

    We could leverage evenly distributed thresholds to use less memory when
    calculate metrcis like AUC where each individual threshold need to be
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
    """Returns op to update the given confusion matrix variables.

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
        flattened into a single label. When True, the valus of
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

    Returns:
      Update op.

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

    y_true = tf.cast(y_true, dtype=variable_dtype)
    y_pred = tf.cast(y_pred, dtype=variable_dtype)

    if thresholds_distributed_evenly:
        # Check whether the thresholds has any leading or tailing epsilon added
        # for floating point imprecision. The leading and tailing threshold will
        # be handled bit differently as the corner case.  At this point,
        # thresholds should be a list/array with more than 2 items, and ranged
        # between [0, 1]. See is_evenly_distributed_thresholds() for more
        # details.
        thresholds_with_epsilon = thresholds[0] < 0.0 or thresholds[-1] > 1.0

    thresholds = tf.convert_to_tensor(thresholds, dtype=variable_dtype)
    num_thresholds = thresholds.shape.as_list()[0]

    if multi_label:
        one_thresh = tf.equal(
            tf.cast(1, dtype=tf.int32),
            tf.rank(thresholds),
            name="one_set_of_thresholds_cond",
        )
    else:
        [y_pred, y_true], _ = ragged_assert_compatible_and_get_flat_values(
            [y_pred, y_true], sample_weight
        )
        one_thresh = tf.cast(True, dtype=tf.bool)

    invalid_keys = [
        key for key in variables_to_update if key not in list(ConfusionMatrix)
    ]
    if invalid_keys:
        raise ValueError(
            f'Invalid keys: "{invalid_keys}". '
            f'Valid variable key options are: "{list(ConfusionMatrix)}"'
        )

    if sample_weight is None:
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
            y_pred, y_true
        )
    else:
        sample_weight = tf.cast(sample_weight, dtype=variable_dtype)
        (
            y_pred,
            y_true,
            sample_weight,
        ) = losses_utils.squeeze_or_expand_dimensions(
            y_pred, y_true, sample_weight=sample_weight
        )
    y_pred.shape.assert_is_compatible_with(y_true.shape)

    if top_k is not None:
        y_pred = _filter_top_k(y_pred, top_k)
    if class_id is not None:
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

    pred_shape = tf.shape(y_pred)
    num_predictions = pred_shape[0]
    if y_pred.shape.ndims == 1:
        num_labels = 1
    else:
        num_labels = tf.math.reduce_prod(pred_shape[1:], axis=0)
    thresh_label_tile = tf.where(
        one_thresh, num_labels, tf.ones([], dtype=tf.int32)
    )

    # Reshape predictions and labels, adding a dim for thresholding.
    if multi_label:
        predictions_extra_dim = tf.expand_dims(y_pred, 0)
        labels_extra_dim = tf.expand_dims(tf.cast(y_true, dtype=tf.bool), 0)
    else:
        # Flatten predictions and labels when not multilabel.
        predictions_extra_dim = tf.reshape(y_pred, [1, -1])
        labels_extra_dim = tf.reshape(tf.cast(y_true, dtype=tf.bool), [1, -1])

    # Tile the thresholds for every prediction.
    if multi_label:
        thresh_pretile_shape = [num_thresholds, 1, -1]
        thresh_tiles = [1, num_predictions, thresh_label_tile]
        data_tiles = [num_thresholds, 1, 1]
    else:
        thresh_pretile_shape = [num_thresholds, -1]
        thresh_tiles = [1, num_predictions * num_labels]
        data_tiles = [num_thresholds, 1]

    thresh_tiled = tf.tile(
        tf.reshape(thresholds, thresh_pretile_shape), tf.stack(thresh_tiles)
    )

    # Tile the predictions for every threshold.
    preds_tiled = tf.tile(predictions_extra_dim, data_tiles)

    # Compare predictions and threshold.
    pred_is_pos = tf.greater(preds_tiled, thresh_tiled)

    # Tile labels by number of thresholds
    label_is_pos = tf.tile(labels_extra_dim, data_tiles)

    if sample_weight is not None:
        sample_weight = tf.__internal__.ops.broadcast_weights(
            tf.cast(sample_weight, dtype=variable_dtype), y_pred
        )
        weights_tiled = tf.tile(
            tf.reshape(sample_weight, thresh_tiles), data_tiles
        )
    else:
        weights_tiled = None

    if label_weights is not None and not multi_label:
        label_weights = tf.expand_dims(label_weights, 0)
        label_weights = tf.__internal__.ops.broadcast_weights(
            label_weights, y_pred
        )
        label_weights_tiled = tf.tile(
            tf.reshape(label_weights, thresh_tiles), data_tiles
        )
        if weights_tiled is None:
            weights_tiled = label_weights_tiled
        else:
            weights_tiled = tf.multiply(weights_tiled, label_weights_tiled)

    update_ops = []

    def weighted_assign_add(label, pred, weights, var):
        label_and_pred = tf.cast(tf.logical_and(label, pred), dtype=var.dtype)
        if weights is not None:
            label_and_pred *= tf.cast(weights, dtype=var.dtype)
        return var.assign_add(tf.reduce_sum(label_and_pred, 1))

    loop_vars = {
        ConfusionMatrix.TRUE_POSITIVES: (label_is_pos, pred_is_pos),
    }
    update_tn = ConfusionMatrix.TRUE_NEGATIVES in variables_to_update
    update_fp = ConfusionMatrix.FALSE_POSITIVES in variables_to_update
    update_fn = ConfusionMatrix.FALSE_NEGATIVES in variables_to_update

    if update_fn or update_tn:
        pred_is_neg = tf.logical_not(pred_is_pos)
        loop_vars[ConfusionMatrix.FALSE_NEGATIVES] = (label_is_pos, pred_is_neg)

    if update_fp or update_tn:
        label_is_neg = tf.logical_not(label_is_pos)
        loop_vars[ConfusionMatrix.FALSE_POSITIVES] = (label_is_neg, pred_is_pos)
        if update_tn:
            loop_vars[ConfusionMatrix.TRUE_NEGATIVES] = (
                label_is_neg,
                pred_is_neg,
            )

    for matrix_cond, (label, pred) in loop_vars.items():
        if matrix_cond in variables_to_update:
            update_ops.append(
                weighted_assign_add(
                    label, pred, weights_tiled, variables_to_update[matrix_cond]
                )
            )

    return tf.group(update_ops)


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
    _, top_k_idx = tf.math.top_k(x, k, sorted=False)
    top_k_mask = tf.reduce_sum(
        tf.one_hot(top_k_idx, tf.shape(x)[-1], axis=-1), axis=-2
    )
    return x * top_k_mask + NEG_INF * (1 - top_k_mask)


def ragged_assert_compatible_and_get_flat_values(values, mask=None):
    """If ragged, it checks the compatibility and then returns the flat_values.

       Note: If two tensors are dense, it does not check their compatibility.
       Note: Although two ragged tensors with different ragged ranks could have
             identical overall rank and dimension sizes and hence be compatible,
             we do not support those cases.
    Args:
       values: A list of potentially ragged tensor of the same ragged_rank.
       mask: A potentially ragged tensor of the same ragged_rank as elements in
         Values.

    Returns:
       A tuple in which the first element is the list of tensors and the second
       is the mask tensor. ([Values], mask). Mask and the element in Values
       are equal to the flat_values of the input arguments (if they were
       ragged).
    """
    if isinstance(values, list):
        is_all_ragged = all(isinstance(rt, tf.RaggedTensor) for rt in values)
        is_any_ragged = any(isinstance(rt, tf.RaggedTensor) for rt in values)
    else:
        is_all_ragged = isinstance(values, tf.RaggedTensor)
        is_any_ragged = is_all_ragged
    if is_all_ragged and ((mask is None) or isinstance(mask, tf.RaggedTensor)):
        to_be_stripped = False
        if not isinstance(values, list):
            values = [values]
            to_be_stripped = True

        # NOTE: we leave the flat_values compatibility to
        # tf.TensorShape `assert_is_compatible_with` check if both dynamic
        # dimensions are equal and then use the flat_values.
        nested_row_split_list = [rt.nested_row_splits for rt in values]
        assertion_list = _assert_splits_match(nested_row_split_list)

        # if both are ragged sample_weights also should be ragged with same
        # dims.
        if isinstance(mask, tf.RaggedTensor):
            assertion_list_for_mask = _assert_splits_match(
                [nested_row_split_list[0], mask.nested_row_splits]
            )
            with tf.control_dependencies(assertion_list_for_mask):
                mask = tf.expand_dims(mask.flat_values, -1)

        # values has at least 1 element.
        flat_values = []
        for value in values:
            with tf.control_dependencies(assertion_list):
                flat_values.append(tf.expand_dims(value.flat_values, -1))

        values = flat_values[0] if to_be_stripped else flat_values

    elif is_any_ragged:
        raise TypeError(
            "Some of the inputs are not tf.RaggedTensor. "
            f"Input received: {values}"
        )
    # values are empty or value are not ragged and mask is ragged.
    elif isinstance(mask, tf.RaggedTensor):
        raise TypeError(
            "Ragged mask is not allowed with non-ragged inputs. "
            f"Input received: {values}, mask received: {mask}"
        )

    return values, mask


def _assert_splits_match(nested_splits_lists):
    """Checks that the given splits lists are identical.

    Performs static tests to ensure that the given splits lists are identical,
    and returns a list of control dependency op tensors that check that they are
    fully identical.

    Args:
      nested_splits_lists: A list of nested_splits_lists, where each split_list
        is a list of `splits` tensors from a `RaggedTensor`, ordered from
        outermost ragged dimension to innermost ragged dimension.

    Returns:
      A list of control dependency op tensors.
    Raises:
      ValueError: If the splits are not identical.
    """
    error_msg = (
        "Inputs must have identical ragged splits. "
        f"Input received: {nested_splits_lists}"
    )
    for splits_list in nested_splits_lists:
        if len(splits_list) != len(nested_splits_lists[0]):
            raise ValueError(error_msg)
    return [
        tf.debugging.assert_equal(s1, s2, message=error_msg)
        for splits_list in nested_splits_lists[1:]
        for (s1, s2) in zip(nested_splits_lists[0], splits_list)
    ]


def binary_matches(y_true, y_pred, threshold=0.5):
    """Creates int Tensor, 1 for label-prediction match, 0 for mismatch.

    Args:
      y_true: Ground truth values, of shape (batch_size, d0, .. dN).
      y_pred: The predicted values, of shape (batch_size, d0, .. dN).
      threshold: (Optional) Float representing the threshold for deciding
        whether prediction values are 1 or 0.

    Returns:
      Binary matches, of shape (batch_size, d0, .. dN).
    """
    y_pred = tf.convert_to_tensor(y_pred)
    threshold = tf.cast(threshold, y_pred.dtype)
    y_pred = tf.cast(y_pred > threshold, y_pred.dtype)
    return tf.cast(tf.equal(y_true, y_pred), backend.floatx())


def sparse_categorical_matches(y_true, y_pred):
    """Creates float Tensor, 1.0 for label-prediction match, 0.0 for mismatch.

    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    Args:
      y_true: Integer ground truth values.
      y_pred: The prediction values.

    Returns:
      Match tensor: 1.0 for label-prediction match, 0.0 for mismatch.
    """
    reshape_matches = False
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true)
    y_true_org_shape = tf.shape(y_true)
    y_pred_rank = y_pred.shape.ndims
    y_true_rank = y_true.shape.ndims

    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (
        (y_true_rank is not None)
        and (y_pred_rank is not None)
        and (len(backend.int_shape(y_true)) == len(backend.int_shape(y_pred)))
    ):
        y_true = tf.squeeze(y_true, [-1])
        reshape_matches = True
    y_pred = tf.math.argmax(y_pred, axis=-1)

    # If the predicted output and actual output types don't match, force cast
    # them to match.
    if backend.dtype(y_pred) != backend.dtype(y_true):
        y_pred = tf.cast(y_pred, backend.dtype(y_true))
    matches = tf.cast(tf.equal(y_true, y_pred), backend.floatx())
    if reshape_matches:
        matches = tf.reshape(matches, shape=y_true_org_shape)
    return matches


def sparse_top_k_categorical_matches(y_true, y_pred, k=5):
    """Creates float Tensor, 1.0 for label-TopK_prediction match, 0.0 for
    mismatch.

    Args:
      y_true: tensor of true targets.
      y_pred: tensor of predicted targets.
      k: (Optional) Number of top elements to look at for computing accuracy.
        Defaults to `5`.

    Returns:
      Match tensor: 1.0 for label-prediction match, 0.0 for mismatch.
    """
    reshape_matches = False
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true_rank = y_true.shape.ndims
    y_pred_rank = y_pred.shape.ndims
    y_true_org_shape = tf.shape(y_true)

    # Flatten y_pred to (batch_size, num_samples) and y_true to (num_samples,)
    if (y_true_rank is not None) and (y_pred_rank is not None):
        if y_pred_rank > 2:
            y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        if y_true_rank > 1:
            reshape_matches = True
            y_true = tf.reshape(y_true, [-1])

    matches = tf.cast(
        tf.math.in_top_k(
            predictions=y_pred, targets=tf.cast(y_true, "int32"), k=k
        ),
        dtype=backend.floatx(),
    )

    # returned matches is expected to have same shape as y_true input
    if reshape_matches:
        return tf.reshape(matches, shape=y_true_org_shape)

    return matches

