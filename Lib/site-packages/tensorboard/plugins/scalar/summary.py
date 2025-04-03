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
"""Scalar summaries and TensorFlow operations to create them.

A scalar summary stores a single floating-point value, as a rank-0
tensor.
"""


import numpy as np

from tensorboard.plugins.scalar import metadata
from tensorboard.plugins.scalar import summary_v2


# Export V2 versions.
scalar = summary_v2.scalar
scalar_pb = summary_v2.scalar_pb


def op(name, data, display_name=None, description=None, collections=None):
    """Create a legacy scalar summary op.

    Arguments:
      name: A unique name for the generated summary node.
      data: A real numeric rank-0 `Tensor`. Must have `dtype` castable
        to `float32`.
      display_name: Optional name for this summary in TensorBoard, as a
        constant `str`. Defaults to `name`.
      description: Optional long-form description for this summary, as a
        constant `str`. Markdown is supported. Defaults to empty.
      collections: Optional list of graph collections keys. The new
        summary op is added to these collections. Defaults to
        `[Graph Keys.SUMMARIES]`.

    Returns:
      A TensorFlow summary op.
    """
    # TODO(nickfelt): remove on-demand imports once dep situation is fixed.
    import tensorflow.compat.v1 as tf

    if display_name is None:
        display_name = name
    summary_metadata = metadata.create_summary_metadata(
        display_name=display_name, description=description
    )
    with tf.name_scope(name):
        with tf.control_dependencies([tf.assert_scalar(data)]):
            return tf.summary.tensor_summary(
                name="scalar_summary",
                tensor=tf.cast(data, tf.float32),
                collections=collections,
                summary_metadata=summary_metadata,
            )


def pb(name, data, display_name=None, description=None):
    """Create a legacy scalar summary protobuf.

    Arguments:
      name: A unique name for the generated summary, including any desired
        name scopes.
      data: A rank-0 `np.array` or array-like form (so raw `int`s and
        `float`s are fine, too).
      display_name: Optional name for this summary in TensorBoard, as a
        `str`. Defaults to `name`.
      description: Optional long-form description for this summary, as a
        `str`. Markdown is supported. Defaults to empty.

    Returns:
      A `tf.Summary` protobuf object.
    """
    # TODO(nickfelt): remove on-demand imports once dep situation is fixed.
    import tensorflow.compat.v1 as tf

    data = np.array(data)
    if data.shape != ():
        raise ValueError(
            "Expected scalar shape for data, saw shape: %s." % data.shape
        )
    if data.dtype.kind not in ("b", "i", "u", "f"):  # bool, int, uint, float
        raise ValueError("Cast %s to float is not supported" % data.dtype.name)
    tensor = tf.make_tensor_proto(data.astype(np.float32))

    if display_name is None:
        display_name = name
    summary_metadata = metadata.create_summary_metadata(
        display_name=display_name, description=description
    )
    tf_summary_metadata = tf.SummaryMetadata.FromString(
        summary_metadata.SerializeToString()
    )
    summary = tf.Summary()
    summary.value.add(
        tag="%s/scalar_summary" % name,
        metadata=tf_summary_metadata,
        tensor=tensor,
    )
    return summary
