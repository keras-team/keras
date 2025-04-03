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
"""Text summaries and TensorFlow operations to create them."""


from tensorboard.plugins.text import metadata
from tensorboard.plugins.text import summary_v2


# Export V2 versions.
text = summary_v2.text
text_pb = summary_v2.text_pb


def op(name, data, display_name=None, description=None, collections=None):
    """Create a legacy text summary op.

    Text data summarized via this plugin will be visible in the Text Dashboard
    in TensorBoard. The standard TensorBoard Text Dashboard will render markdown
    in the strings, and will automatically organize 1D and 2D tensors into tables.
    If a tensor with more than 2 dimensions is provided, a 2D subarray will be
    displayed along with a warning message. (Note that this behavior is not
    intrinsic to the text summary API, but rather to the default TensorBoard text
    plugin.)

    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      data: A string-type Tensor to summarize. The text must be encoded in UTF-8.
      display_name: Optional name for this summary in TensorBoard, as a
        constant `str`. Defaults to `name`.
      description: Optional long-form description for this summary, as a
        constant `str`. Markdown is supported. Defaults to empty.
      collections: Optional list of ops.GraphKeys. The collections to which to add
        the summary. Defaults to [Graph Keys.SUMMARIES].

    Returns:
      A TensorSummary op that is configured so that TensorBoard will recognize
      that it contains textual data. The TensorSummary is a scalar `Tensor` of
      type `string` which contains `Summary` protobufs.

    Raises:
      ValueError: If tensor has the wrong type.
    """
    # TODO(nickfelt): remove on-demand imports once dep situation is fixed.
    import tensorflow.compat.v1 as tf

    if display_name is None:
        display_name = name
    summary_metadata = metadata.create_summary_metadata(
        display_name=display_name, description=description
    )
    with tf.name_scope(name):
        with tf.control_dependencies([tf.assert_type(data, tf.string)]):
            return tf.summary.tensor_summary(
                name="text_summary",
                tensor=data,
                collections=collections,
                summary_metadata=summary_metadata,
            )


def pb(name, data, display_name=None, description=None):
    """Create a legacy text summary protobuf.

    Arguments:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      data: A Python bytestring (of type bytes), or Unicode string. Or a numpy
        data array of those types.
      display_name: Optional name for this summary in TensorBoard, as a
        `str`. Defaults to `name`.
      description: Optional long-form description for this summary, as a
        `str`. Markdown is supported. Defaults to empty.

    Raises:
      ValueError: If the type of the data is unsupported.

    Returns:
      A `tf.Summary` protobuf object.
    """
    # TODO(nickfelt): remove on-demand imports once dep situation is fixed.
    import tensorflow.compat.v1 as tf

    try:
        tensor = tf.make_tensor_proto(data, dtype=tf.string)
    except TypeError as e:
        raise ValueError(e)

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
        tag="%s/text_summary" % name,
        metadata=tf_summary_metadata,
        tensor=tensor,
    )
    return summary
