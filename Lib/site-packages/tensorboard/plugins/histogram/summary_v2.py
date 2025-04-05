# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Histogram summaries and TensorFlow operations to create them, V2 versions.

A histogram summary stores a list of buckets. Each bucket is encoded as a triple
`[left_edge, right_edge, count]`. Thus, a full histogram is encoded as a tensor
of dimension `[k, 3]`, where the first `k - 1` buckets are closed-open and the
last bucket is closed-closed.

In general, the shape of the output histogram is always constant (`[k, 3]`).
In the case of empty data, the output will be an all-zero histogram of shape
`[k, 3]`, where all edges and counts are zeros. If there is data but all points
have the same value, then all buckets' left and right edges are the same and only
the last bucket has nonzero count.
"""

import numpy as np

from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.histogram import metadata
from tensorboard.util import lazy_tensor_creator
from tensorboard.util import tensor_util


DEFAULT_BUCKET_COUNT = 30


def histogram_pb(tag, data, buckets=None, description=None):
    """Create a histogram summary protobuf.

    Arguments:
      tag: String tag for the summary.
      data: A `np.array` or array-like form of any shape. Must have type
        castable to `float`.
      buckets: Optional positive `int`. The output shape will always be
        [buckets, 3]. If there is no data, then an all-zero array of shape
        [buckets, 3] will be returned. If there is data but all points have
        the same value, then all buckets' left and right endpoints are the
        same and only the last bucket has nonzero count. Defaults to 30 if
        not specified.
      description: Optional long-form description for this summary, as a
        `str`. Markdown is supported. Defaults to empty.

    Returns:
      A `summary_pb2.Summary` protobuf object.
    """
    bucket_count = DEFAULT_BUCKET_COUNT if buckets is None else buckets
    data = np.array(data).flatten().astype(float)
    if bucket_count == 0 or data.size == 0:
        histogram_buckets = np.zeros((bucket_count, 3))
    else:
        min_ = np.min(data)
        max_ = np.max(data)
        range_ = max_ - min_
        if range_ == 0:
            left_edges = right_edges = np.array([min_] * bucket_count)
            bucket_counts = np.array([0] * (bucket_count - 1) + [data.size])
            histogram_buckets = np.array(
                [left_edges, right_edges, bucket_counts]
            ).transpose()
        else:
            bucket_width = range_ / bucket_count
            offsets = data - min_
            bucket_indices = np.floor(offsets / bucket_width).astype(int)
            clamped_indices = np.minimum(bucket_indices, bucket_count - 1)
            one_hots = np.array([clamped_indices]).transpose() == np.arange(
                0, bucket_count
            )  # broadcast
            assert one_hots.shape == (data.size, bucket_count), (
                one_hots.shape,
                (data.size, bucket_count),
            )
            bucket_counts = np.sum(one_hots, axis=0)
            edges = np.linspace(min_, max_, bucket_count + 1)
            left_edges = edges[:-1]
            right_edges = edges[1:]
            histogram_buckets = np.array(
                [left_edges, right_edges, bucket_counts]
            ).transpose()
    tensor = tensor_util.make_tensor_proto(histogram_buckets, dtype=np.float64)

    summary_metadata = metadata.create_summary_metadata(
        display_name=None, description=description
    )
    summary = summary_pb2.Summary()
    summary.value.add(tag=tag, metadata=summary_metadata, tensor=tensor)
    return summary


# This is the TPU compatible V3 histogram implementation as of 2021-12-01.
def histogram(name, data, step=None, buckets=None, description=None):
    """Write a histogram summary.

    See also `tf.summary.scalar`, `tf.summary.SummaryWriter`.

    Writes a histogram to the current default summary writer, for later analysis
    in TensorBoard's 'Histograms' and 'Distributions' dashboards (data written
    using this API will appear in both places). Like `tf.summary.scalar` points,
    each histogram is associated with a `step` and a `name`. All the histograms
    with the same `name` constitute a time series of histograms.

    The histogram is calculated over all the elements of the given `Tensor`
    without regard to its shape or rank.

    This example writes 2 histograms:

    ```python
    w = tf.summary.create_file_writer('test/logs')
    with w.as_default():
        tf.summary.histogram("activations", tf.random.uniform([100, 50]), step=0)
        tf.summary.histogram("initial_weights", tf.random.normal([1000]), step=0)
    ```

    A common use case is to examine the changing activation patterns (or lack
    thereof) at specific layers in a neural network, over time.

    ```python
    w = tf.summary.create_file_writer('test/logs')
    with w.as_default():
    for step in range(100):
        # Generate fake "activations".
        activations = [
            tf.random.normal([1000], mean=step, stddev=1),
            tf.random.normal([1000], mean=step, stddev=10),
            tf.random.normal([1000], mean=step, stddev=100),
        ]

        tf.summary.histogram("layer1/activate", activations[0], step=step)
        tf.summary.histogram("layer2/activate", activations[1], step=step)
        tf.summary.histogram("layer3/activate", activations[2], step=step)
    ```

    Arguments:
      name: A name for this summary. The summary tag used for TensorBoard will
        be this name prefixed by any active name scopes.
      data: A `Tensor` of any shape. The histogram is computed over its elements,
        which must be castable to `float64`.
      step: Explicit `int64`-castable monotonic step value for this summary. If
        omitted, this defaults to `tf.summary.experimental.get_step()`, which must
        not be None.
      buckets: Optional positive `int`. The output will have this
        many buckets, except in two edge cases. If there is no data, then
        there are no buckets. If there is data but all points have the
        same value, then all buckets' left and right endpoints are the same
        and only the last bucket has nonzero count. Defaults to 30 if not
        specified.
      description: Optional long-form description for this summary, as a
        constant `str`. Markdown is supported. Defaults to empty.

    Returns:
      True on success, or false if no summary was emitted because no default
      summary writer was available.

    Raises:
      ValueError: if a default writer exists, but no step was provided and
        `tf.summary.experimental.get_step()` is None.
    """
    # Avoid building unused gradient graphs for conds below. This works around
    # an error building second-order gradient graphs when XlaDynamicUpdateSlice
    # is used, and will generally speed up graph building slightly.
    data = tf.stop_gradient(data)
    summary_metadata = metadata.create_summary_metadata(
        display_name=None, description=description
    )
    # TODO(https://github.com/tensorflow/tensorboard/issues/2109): remove fallback
    summary_scope = (
        getattr(tf.summary.experimental, "summary_scope", None)
        or tf.summary.summary_scope
    )

    # TODO(ytjing): add special case handling.
    with summary_scope(
        name, "histogram_summary", values=[data, buckets, step]
    ) as (tag, _):
        # Defer histogram bucketing logic by passing it as a callable to
        # write(), wrapped in a LazyTensorCreator for backwards
        # compatibility, so that we only do this work when summaries are
        # actually written.
        @lazy_tensor_creator.LazyTensorCreator
        def lazy_tensor():
            return _buckets(data, buckets)

        return tf.summary.write(
            tag=tag,
            tensor=lazy_tensor,
            step=step,
            metadata=summary_metadata,
        )


def _buckets(data, bucket_count=None):
    """Create a TensorFlow op to group data into histogram buckets.

    Arguments:
      data: A `Tensor` of any shape. Must be castable to `float64`.
      bucket_count: Optional non-negative `int` or scalar `int32` `Tensor`,
        defaults to 30.
    Returns:
      A `Tensor` of shape `[k, 3]` and type `float64`. The `i`th row is
      a triple `[left_edge, right_edge, count]` for a single bucket.
      The value of `k` is either `bucket_count` or `0` (when input data
      is empty).
    """
    if bucket_count is None:
        bucket_count = DEFAULT_BUCKET_COUNT
    with tf.name_scope("buckets"):
        tf.debugging.assert_scalar(bucket_count)
        tf.debugging.assert_type(bucket_count, tf.int32)
        # Treat a negative bucket count as zero.
        bucket_count = tf.math.maximum(0, bucket_count)
        data = tf.reshape(data, shape=[-1])  # flatten
        data = tf.cast(data, tf.float64)
        data_size = tf.size(input=data)
        is_empty = tf.logical_or(
            tf.equal(data_size, 0), tf.less_equal(bucket_count, 0)
        )

        def when_empty():
            """When input data is empty or bucket_count is zero.

            1. If bucket_count is specified as zero, an empty tensor of shape
              (0, 3) will be returned.
            2. If the input data is empty, a tensor of shape (bucket_count, 3)
              of all zero values will be returned.
            """
            return tf.zeros((bucket_count, 3), dtype=tf.float64)

        def when_nonempty():
            min_ = tf.reduce_min(input_tensor=data)
            max_ = tf.reduce_max(input_tensor=data)
            range_ = max_ - min_
            has_single_value = tf.equal(range_, 0)

            def when_multiple_values():
                """When input data contains multiple values."""
                bucket_width = range_ / tf.cast(bucket_count, tf.float64)
                offsets = data - min_
                bucket_indices = tf.cast(
                    tf.floor(offsets / bucket_width), dtype=tf.int32
                )
                clamped_indices = tf.minimum(bucket_indices, bucket_count - 1)
                # Use float64 instead of float32 to avoid accumulating floating point error
                # later in tf.reduce_sum when summing more than 2^24 individual `1.0` values.
                # See https://github.com/tensorflow/tensorflow/issues/51419 for details.
                one_hots = tf.one_hot(
                    clamped_indices, depth=bucket_count, dtype=tf.float64
                )
                bucket_counts = tf.cast(
                    tf.reduce_sum(input_tensor=one_hots, axis=0),
                    dtype=tf.float64,
                )
                edges = tf.linspace(min_, max_, bucket_count + 1)
                # Ensure edges[-1] == max_, which TF's linspace implementation does not
                # do, leaving it subject to the whim of floating point rounding error.
                edges = tf.concat([edges[:-1], [max_]], 0)
                left_edges = edges[:-1]
                right_edges = edges[1:]
                return tf.transpose(
                    a=tf.stack([left_edges, right_edges, bucket_counts])
                )

            def when_single_value():
                """When input data contains a single unique value."""
                # Left and right edges are the same for single value input.
                edges = tf.fill([bucket_count], max_)
                # Bucket counts are 0 except the last bucket (if bucket_count > 0),
                # which is `data_size`. Ensure that the resulting counts vector has
                # length `bucket_count` always, including the bucket_count==0 case.
                zeroes = tf.fill([bucket_count], 0)
                bucket_counts = tf.cast(
                    tf.concat([zeroes[:-1], [data_size]], 0)[:bucket_count],
                    dtype=tf.float64,
                )
                return tf.transpose(a=tf.stack([edges, edges, bucket_counts]))

            return tf.cond(
                has_single_value, when_single_value, when_multiple_values
            )

        return tf.cond(is_empty, when_empty, when_nonempty)
