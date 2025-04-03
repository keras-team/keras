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
"""Utilities to migrate legacy protos to their modern equivalents."""


import numpy as np

from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.audio import metadata as audio_metadata
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
from tensorboard.util import tensor_util


def migrate_event(event):
    if not event.HasField("summary"):
        return event
    old_values = event.summary.value
    new_values = [migrate_value(value) for value in old_values]
    # Optimization: Don't create a new event if there were no changes.
    if len(old_values) == len(new_values) and all(
        x is y for (x, y) in zip(old_values, new_values)
    ):
        return event
    result = event_pb2.Event()
    result.CopyFrom(event)
    del result.summary.value[:]
    result.summary.value.extend(new_values)
    return result


def migrate_value(value):
    """Convert `value` to a new-style value, if necessary and possible.

    An "old-style" value is a value that uses any `value` field other than
    the `tensor` field. A "new-style" value is a value that uses the
    `tensor` field. TensorBoard continues to support old-style values on
    disk; this method converts them to new-style values so that further
    code need only deal with one data format.

    Arguments:
      value: A `Summary.Value` object. This argument is not modified.

    Returns:
      If the `value` is an old-style value for which there is a new-style
      equivalent, the result is the new-style value. Otherwise---if the
      value is already new-style or does not yet have a new-style
      equivalent---the value will be returned unchanged.

    :type value: Summary.Value
    :rtype: Summary.Value
    """
    handler = {
        "histo": _migrate_histogram_value,
        "image": _migrate_image_value,
        "audio": _migrate_audio_value,
        "simple_value": _migrate_scalar_value,
    }.get(value.WhichOneof("value"))
    return handler(value) if handler else value


def make_summary(tag, metadata, data):
    tensor_proto = tensor_util.make_tensor_proto(data)
    return summary_pb2.Summary.Value(
        tag=tag, metadata=metadata, tensor=tensor_proto
    )


def _migrate_histogram_value(value):
    """Convert `old-style` histogram value to `new-style`.

    The "old-style" format can have outermost bucket limits of -DBL_MAX and
    DBL_MAX, which are problematic for visualization. We replace those here
    with the actual min and max values seen in the input data, but then in
    order to avoid introducing "backwards" buckets (where left edge > right
    edge), we first must drop all empty buckets on the left and right ends.
    """
    histogram_value = value.histo
    bucket_counts = histogram_value.bucket
    # Find the indices of the leftmost and rightmost non-empty buckets.
    n = len(bucket_counts)
    start = next((i for i in range(n) if bucket_counts[i] > 0), n)
    end = next((i for i in reversed(range(n)) if bucket_counts[i] > 0), -1)
    if start > end:
        # If all input buckets were empty, treat it as a zero-bucket
        # new-style histogram.
        buckets = np.zeros([0, 3], dtype=np.float32)
    else:
        # Discard empty buckets on both ends, and keep only the "inner"
        # edges from the remaining buckets. Note that bucket indices range
        # from `start` to `end` inclusive, but bucket_limit indices are
        # exclusive of `end` - this is because bucket_limit[i] is the
        # right-hand edge for bucket[i].
        bucket_counts = bucket_counts[start : end + 1]
        inner_edges = histogram_value.bucket_limit[start:end]
        # Use min as the left-hand limit for the first non-empty bucket.
        bucket_lefts = [histogram_value.min] + inner_edges
        # Use max as the right-hand limit for the last non-empty bucket.
        bucket_rights = inner_edges + [histogram_value.max]
        buckets = np.array(
            [bucket_lefts, bucket_rights, bucket_counts], dtype=np.float32
        ).transpose()

    summary_metadata = histogram_metadata.create_summary_metadata(
        display_name=value.metadata.display_name or value.tag,
        description=value.metadata.summary_description,
    )

    return make_summary(value.tag, summary_metadata, buckets)


def _migrate_image_value(value):
    image_value = value.image
    data = [
        str(image_value.width).encode("ascii"),
        str(image_value.height).encode("ascii"),
        image_value.encoded_image_string,
    ]

    summary_metadata = image_metadata.create_summary_metadata(
        display_name=value.metadata.display_name or value.tag,
        description=value.metadata.summary_description,
        converted_to_tensor=True,
    )
    return make_summary(value.tag, summary_metadata, data)


def _migrate_audio_value(value):
    audio_value = value.audio
    data = [[audio_value.encoded_audio_string, b""]]  # empty label
    summary_metadata = audio_metadata.create_summary_metadata(
        display_name=value.metadata.display_name or value.tag,
        description=value.metadata.summary_description,
        encoding=audio_metadata.Encoding.Value("WAV"),
        converted_to_tensor=True,
    )
    return make_summary(value.tag, summary_metadata, data)


def _migrate_scalar_value(value):
    scalar_value = value.simple_value
    summary_metadata = scalar_metadata.create_summary_metadata(
        display_name=value.metadata.display_name or value.tag,
        description=value.metadata.summary_description,
    )
    return make_summary(value.tag, summary_metadata, scalar_value)
