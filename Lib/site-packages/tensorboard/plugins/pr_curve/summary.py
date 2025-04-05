# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Precision--recall curves and TensorFlow operations to create them.

NOTE: This module is in beta, and its API is subject to change, but the
data that it stores to disk will be supported forever.
"""


import numpy as np

from tensorboard.plugins.pr_curve import metadata


# A value that we use as the minimum value during division of counts to prevent
# division by 0. 1.0 does not work: Certain weights could cause counts below 1.
_MINIMUM_COUNT = 1e-7

# The default number of thresholds.
_DEFAULT_NUM_THRESHOLDS = 201


def op(
    name,
    labels,
    predictions,
    num_thresholds=None,
    weights=None,
    display_name=None,
    description=None,
    collections=None,
):
    """Create a PR curve summary op for a single binary classifier.

    Computes true/false positive/negative values for the given `predictions`
    against the ground truth `labels`, against a list of evenly distributed
    threshold values in `[0, 1]` of length `num_thresholds`.

    Each number in `predictions`, a float in `[0, 1]`, is compared with its
    corresponding boolean label in `labels`, and counts as a single tp/fp/tn/fn
    value at each threshold. This is then multiplied with `weights` which can be
    used to reweight certain values, or more commonly used for masking values.

    Args:
      name: A tag attached to the summary. Used by TensorBoard for organization.
      labels: The ground truth values. A Tensor of `bool` values with arbitrary
          shape.
      predictions: A float32 `Tensor` whose values are in the range `[0, 1]`.
          Dimensions must match those of `labels`.
      num_thresholds: Number of thresholds, evenly distributed in `[0, 1]`, to
          compute PR metrics for. Should be `>= 2`. This value should be a
          constant integer value, not a Tensor that stores an integer.
      weights: Optional float32 `Tensor`. Individual counts are multiplied by this
          value. This tensor must be either the same shape as or broadcastable to
          the `labels` tensor.
      display_name: Optional name for this summary in TensorBoard, as a
          constant `str`. Defaults to `name`.
      description: Optional long-form description for this summary, as a
          constant `str`. Markdown is supported. Defaults to empty.
      collections: Optional list of graph collections keys. The new
          summary op is added to these collections. Defaults to
          `[Graph Keys.SUMMARIES]`.

    Returns:
      A summary operation for use in a TensorFlow graph. The float32 tensor
      produced by the summary operation is of dimension (6, num_thresholds). The
      first dimension (of length 6) is of the order: true positives,
      false positives, true negatives, false negatives, precision, recall.
    """
    # TODO(nickfelt): remove on-demand imports once dep situation is fixed.
    import tensorflow.compat.v1 as tf

    if num_thresholds is None:
        num_thresholds = _DEFAULT_NUM_THRESHOLDS

    if weights is None:
        weights = 1.0

    dtype = predictions.dtype

    with tf.name_scope(name, values=[labels, predictions, weights]):
        tf.assert_type(labels, tf.bool)
        # We cast to float to ensure we have 0.0 or 1.0.
        f_labels = tf.cast(labels, dtype)
        # Ensure predictions are all in range [0.0, 1.0].
        predictions = tf.minimum(1.0, tf.maximum(0.0, predictions))
        # Get weighted true/false labels.
        true_labels = f_labels * weights
        false_labels = (1.0 - f_labels) * weights

        # Before we begin, flatten predictions.
        predictions = tf.reshape(predictions, [-1])

        # Shape the labels so they are broadcast-able for later multiplication.
        true_labels = tf.reshape(true_labels, [-1, 1])
        false_labels = tf.reshape(false_labels, [-1, 1])

        # To compute TP/FP/TN/FN, we are measuring a binary classifier
        #   C(t) = (predictions >= t)
        # at each threshold 't'. So we have
        #   TP(t) = sum( C(t) * true_labels )
        #   FP(t) = sum( C(t) * false_labels )
        #
        # But, computing C(t) requires computation for each t. To make it fast,
        # observe that C(t) is a cumulative integral, and so if we have
        #   thresholds = [t_0, ..., t_{n-1}];  t_0 < ... < t_{n-1}
        # where n = num_thresholds, and if we can compute the bucket function
        #   B(i) = Sum( (predictions == t), t_i <= t < t{i+1} )
        # then we get
        #   C(t_i) = sum( B(j), j >= i )
        # which is the reversed cumulative sum in tf.cumsum().
        #
        # We can compute B(i) efficiently by taking advantage of the fact that
        # our thresholds are evenly distributed, in that
        #   width = 1.0 / (num_thresholds - 1)
        #   thresholds = [0.0, 1*width, 2*width, 3*width, ..., 1.0]
        # Given a prediction value p, we can map it to its bucket by
        #   bucket_index(p) = floor( p * (num_thresholds - 1) )
        # so we can use tf.scatter_add() to update the buckets in one pass.

        # Compute the bucket indices for each prediction value.
        bucket_indices = tf.cast(
            tf.floor(predictions * (num_thresholds - 1)), tf.int32
        )

        # Bucket predictions.
        tp_buckets = tf.reduce_sum(
            input_tensor=tf.one_hot(bucket_indices, depth=num_thresholds)
            * true_labels,
            axis=0,
        )
        fp_buckets = tf.reduce_sum(
            input_tensor=tf.one_hot(bucket_indices, depth=num_thresholds)
            * false_labels,
            axis=0,
        )

        # Set up the cumulative sums to compute the actual metrics.
        tp = tf.cumsum(tp_buckets, reverse=True, name="tp")
        fp = tf.cumsum(fp_buckets, reverse=True, name="fp")
        # fn = sum(true_labels) - tp
        #    = sum(tp_buckets) - tp
        #    = tp[0] - tp
        # Similarly,
        # tn = fp[0] - fp
        tn = fp[0] - fp
        fn = tp[0] - tp

        precision = tp / tf.maximum(_MINIMUM_COUNT, tp + fp)
        recall = tp / tf.maximum(_MINIMUM_COUNT, tp + fn)

        return _create_tensor_summary(
            name,
            tp,
            fp,
            tn,
            fn,
            precision,
            recall,
            num_thresholds,
            display_name,
            description,
            collections,
        )


def pb(
    name,
    labels,
    predictions,
    num_thresholds=None,
    weights=None,
    display_name=None,
    description=None,
):
    """Create a PR curves summary protobuf.

    Arguments:
      name: A name for the generated node. Will also serve as a series name in
          TensorBoard.
      labels: The ground truth values. A bool numpy array.
      predictions: A float32 numpy array whose values are in the range `[0, 1]`.
          Dimensions must match those of `labels`.
      num_thresholds: Optional number of thresholds, evenly distributed in
          `[0, 1]`, to compute PR metrics for. When provided, should be an int of
          value at least 2. Defaults to 201.
      weights: Optional float or float32 numpy array. Individual counts are
          multiplied by this value. This tensor must be either the same shape as
          or broadcastable to the `labels` numpy array.
      display_name: Optional name for this summary in TensorBoard, as a `str`.
          Defaults to `name`.
      description: Optional long-form description for this summary, as a `str`.
          Markdown is supported. Defaults to empty.
    """
    # TODO(nickfelt): remove on-demand imports once dep situation is fixed.
    import tensorflow.compat.v1 as tf  # noqa: F401

    if num_thresholds is None:
        num_thresholds = _DEFAULT_NUM_THRESHOLDS

    if weights is None:
        weights = 1.0

    # Compute bins of true positives and false positives.
    bucket_indices = np.int32(np.floor(predictions * (num_thresholds - 1)))
    float_labels = labels.astype(float)
    histogram_range = (0, num_thresholds - 1)
    tp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=float_labels * weights,
    )
    fp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=(1.0 - float_labels) * weights,
    )

    # Obtain the reverse cumulative sum.
    tp = np.cumsum(tp_buckets[::-1])[::-1]
    fp = np.cumsum(fp_buckets[::-1])[::-1]
    tn = fp[0] - fp
    fn = tp[0] - tp
    precision = tp / np.maximum(_MINIMUM_COUNT, tp + fp)
    recall = tp / np.maximum(_MINIMUM_COUNT, tp + fn)

    return raw_data_pb(
        name,
        true_positive_counts=tp,
        false_positive_counts=fp,
        true_negative_counts=tn,
        false_negative_counts=fn,
        precision=precision,
        recall=recall,
        num_thresholds=num_thresholds,
        display_name=display_name,
        description=description,
    )


def streaming_op(
    name,
    labels,
    predictions,
    num_thresholds=None,
    weights=None,
    metrics_collections=None,
    updates_collections=None,
    display_name=None,
    description=None,
):
    """Computes a precision-recall curve summary across batches of data.

    This function is similar to op() above, but can be used to compute the PR
    curve across multiple batches of labels and predictions, in the same style
    as the metrics found in tf.metrics.

    This function creates multiple local variables for storing true positives,
    true negative, etc. accumulated over each batch of data, and uses these local
    variables for computing the final PR curve summary. These variables can be
    updated with the returned update_op.

    Args:
      name: A tag attached to the summary. Used by TensorBoard for organization.
      labels: The ground truth values, a `Tensor` whose dimensions must match
        `predictions`. Will be cast to `bool`.
      predictions: A floating point `Tensor` of arbitrary shape and whose values
        are in the range `[0, 1]`.
      num_thresholds: The number of evenly spaced thresholds to generate for
        computing the PR curve. Defaults to 201.
      weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `labels` dimension).
      metrics_collections: An optional list of collections that `auc` should be
        added to.
      updates_collections: An optional list of collections that `update_op` should
        be added to.
      display_name: Optional name for this summary in TensorBoard, as a
          constant `str`. Defaults to `name`.
      description: Optional long-form description for this summary, as a
          constant `str`. Markdown is supported. Defaults to empty.

    Returns:
      pr_curve: A string `Tensor` containing a single value: the
        serialized PR curve Tensor summary. The summary contains a
        float32 `Tensor` of dimension (6, num_thresholds). The first
        dimension (of length 6) is of the order: true positives, false
        positives, true negatives, false negatives, precision, recall.
      update_op: An operation that updates the summary with the latest data.
    """
    # TODO(nickfelt): remove on-demand imports once dep situation is fixed.
    import tensorflow.compat.v1 as tf

    if num_thresholds is None:
        num_thresholds = _DEFAULT_NUM_THRESHOLDS

    thresholds = [i / float(num_thresholds - 1) for i in range(num_thresholds)]

    with tf.name_scope(name, values=[labels, predictions, weights]):
        tp, update_tp = tf.metrics.true_positives_at_thresholds(
            labels=labels,
            predictions=predictions,
            thresholds=thresholds,
            weights=weights,
        )
        fp, update_fp = tf.metrics.false_positives_at_thresholds(
            labels=labels,
            predictions=predictions,
            thresholds=thresholds,
            weights=weights,
        )
        tn, update_tn = tf.metrics.true_negatives_at_thresholds(
            labels=labels,
            predictions=predictions,
            thresholds=thresholds,
            weights=weights,
        )
        fn, update_fn = tf.metrics.false_negatives_at_thresholds(
            labels=labels,
            predictions=predictions,
            thresholds=thresholds,
            weights=weights,
        )

        def compute_summary(tp, fp, tn, fn, collections):
            precision = tp / tf.maximum(_MINIMUM_COUNT, tp + fp)
            recall = tp / tf.maximum(_MINIMUM_COUNT, tp + fn)

            return _create_tensor_summary(
                name,
                tp,
                fp,
                tn,
                fn,
                precision,
                recall,
                num_thresholds,
                display_name,
                description,
                collections,
            )

        pr_curve = compute_summary(tp, fp, tn, fn, metrics_collections)
        update_op = tf.group(update_tp, update_fp, update_tn, update_fn)
        if updates_collections:
            for collection in updates_collections:
                tf.add_to_collection(collection, update_op)

        return pr_curve, update_op


def raw_data_op(
    name,
    true_positive_counts,
    false_positive_counts,
    true_negative_counts,
    false_negative_counts,
    precision,
    recall,
    num_thresholds=None,
    display_name=None,
    description=None,
    collections=None,
):
    """Create an op that collects data for visualizing PR curves.

    Unlike the op above, this one avoids computing precision, recall, and the
    intermediate counts. Instead, it accepts those tensors as arguments and
    relies on the caller to ensure that the calculations are correct (and the
    counts yield the provided precision and recall values).

    This op is useful when a caller seeks to compute precision and recall
    differently but still use the PR curves plugin.

    Args:
      name: A tag attached to the summary. Used by TensorBoard for organization.
      true_positive_counts: A rank-1 tensor of true positive counts. Must contain
          `num_thresholds` elements and be castable to float32. Values correspond
          to thresholds that increase from left to right (from 0 to 1).
      false_positive_counts: A rank-1 tensor of false positive counts. Must
          contain `num_thresholds` elements and be castable to float32. Values
          correspond to thresholds that increase from left to right (from 0 to 1).
      true_negative_counts: A rank-1 tensor of true negative counts. Must contain
          `num_thresholds` elements and be castable to float32. Values
          correspond to thresholds that increase from left to right (from 0 to 1).
      false_negative_counts: A rank-1 tensor of false negative counts. Must
          contain `num_thresholds` elements and be castable to float32. Values
          correspond to thresholds that increase from left to right (from 0 to 1).
      precision: A rank-1 tensor of precision values. Must contain
          `num_thresholds` elements and be castable to float32. Values correspond
          to thresholds that increase from left to right (from 0 to 1).
      recall: A rank-1 tensor of recall values. Must contain `num_thresholds`
          elements and be castable to float32. Values correspond to thresholds
          that increase from left to right (from 0 to 1).
      num_thresholds: Number of thresholds, evenly distributed in `[0, 1]`, to
          compute PR metrics for. Should be `>= 2`. This value should be a
          constant integer value, not a Tensor that stores an integer.
      display_name: Optional name for this summary in TensorBoard, as a
          constant `str`. Defaults to `name`.
      description: Optional long-form description for this summary, as a
          constant `str`. Markdown is supported. Defaults to empty.
      collections: Optional list of graph collections keys. The new
          summary op is added to these collections. Defaults to
          `[Graph Keys.SUMMARIES]`.

    Returns:
      A summary operation for use in a TensorFlow graph. See docs for the `op`
      method for details on the float32 tensor produced by this summary.
    """
    # TODO(nickfelt): remove on-demand imports once dep situation is fixed.
    import tensorflow.compat.v1 as tf

    with tf.name_scope(
        name,
        values=[
            true_positive_counts,
            false_positive_counts,
            true_negative_counts,
            false_negative_counts,
            precision,
            recall,
        ],
    ):
        return _create_tensor_summary(
            name,
            true_positive_counts,
            false_positive_counts,
            true_negative_counts,
            false_negative_counts,
            precision,
            recall,
            num_thresholds,
            display_name,
            description,
            collections,
        )


def raw_data_pb(
    name,
    true_positive_counts,
    false_positive_counts,
    true_negative_counts,
    false_negative_counts,
    precision,
    recall,
    num_thresholds=None,
    display_name=None,
    description=None,
):
    """Create a PR curves summary protobuf from raw data values.

    Args:
      name: A tag attached to the summary. Used by TensorBoard for organization.
      true_positive_counts: A rank-1 numpy array of true positive counts. Must
          contain `num_thresholds` elements and be castable to float32.
      false_positive_counts: A rank-1 numpy array of false positive counts. Must
          contain `num_thresholds` elements and be castable to float32.
      true_negative_counts: A rank-1 numpy array of true negative counts. Must
          contain `num_thresholds` elements and be castable to float32.
      false_negative_counts: A rank-1 numpy array of false negative counts. Must
          contain `num_thresholds` elements and be castable to float32.
      precision: A rank-1 numpy array of precision values. Must contain
          `num_thresholds` elements and be castable to float32.
      recall: A rank-1 numpy array of recall values. Must contain `num_thresholds`
          elements and be castable to float32.
      num_thresholds: Number of thresholds, evenly distributed in `[0, 1]`, to
          compute PR metrics for. Should be an int `>= 2`.
      display_name: Optional name for this summary in TensorBoard, as a `str`.
          Defaults to `name`.
      description: Optional long-form description for this summary, as a `str`.
          Markdown is supported. Defaults to empty.

    Returns:
      A summary operation for use in a TensorFlow graph. See docs for the `op`
      method for details on the float32 tensor produced by this summary.
    """
    # TODO(nickfelt): remove on-demand imports once dep situation is fixed.
    import tensorflow.compat.v1 as tf

    if display_name is None:
        display_name = name
    summary_metadata = metadata.create_summary_metadata(
        display_name=display_name if display_name is not None else name,
        description=description or "",
        num_thresholds=num_thresholds,
    )
    tf_summary_metadata = tf.SummaryMetadata.FromString(
        summary_metadata.SerializeToString()
    )
    summary = tf.Summary()
    data = np.stack(
        (
            true_positive_counts,
            false_positive_counts,
            true_negative_counts,
            false_negative_counts,
            precision,
            recall,
        )
    )
    tensor = tf.make_tensor_proto(np.float32(data), dtype=tf.float32)
    summary.value.add(
        tag="%s/pr_curves" % name, metadata=tf_summary_metadata, tensor=tensor
    )
    return summary


def _create_tensor_summary(
    name,
    true_positive_counts,
    false_positive_counts,
    true_negative_counts,
    false_negative_counts,
    precision,
    recall,
    num_thresholds=None,
    display_name=None,
    description=None,
    collections=None,
):
    """A private helper method for generating a tensor summary.

    We use a helper method instead of having `op` directly call `raw_data_op`
    to prevent the scope of `raw_data_op` from being embedded within `op`.

    Arguments are the same as for raw_data_op.

    Returns:
      A tensor summary that collects data for PR curves.
    """
    # TODO(nickfelt): remove on-demand imports once dep situation is fixed.
    import tensorflow.compat.v1 as tf

    # Store the number of thresholds within the summary metadata because
    # that value is constant for all pr curve summaries with the same tag.
    summary_metadata = metadata.create_summary_metadata(
        display_name=display_name if display_name is not None else name,
        description=description or "",
        num_thresholds=num_thresholds,
    )

    # Store values within a tensor. We store them in the order:
    # true positives, false positives, true negatives, false
    # negatives, precision, and recall.
    combined_data = tf.stack(
        [
            tf.cast(true_positive_counts, tf.float32),
            tf.cast(false_positive_counts, tf.float32),
            tf.cast(true_negative_counts, tf.float32),
            tf.cast(false_negative_counts, tf.float32),
            tf.cast(precision, tf.float32),
            tf.cast(recall, tf.float32),
        ]
    )

    return tf.summary.tensor_summary(
        name="pr_curves",
        tensor=combined_data,
        collections=collections,
        summary_metadata=summary_metadata,
    )
