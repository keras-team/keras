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
"""Text summaries and TensorFlow operations to create them, V2 versions."""


import numpy as np

from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.text import metadata
from tensorboard.util import tensor_util


def text(name, data, step=None, description=None):
    r"""Write a text summary.

    See also `tf.summary.scalar`, `tf.summary.SummaryWriter`, `tf.summary.image`.

    Writes text Tensor values for later visualization and analysis in TensorBoard.
    Writes go to the current default summary writer.  Like `tf.summary.scalar`
    points, text points are each associated with a `step` and a `name`.
    All the points with the same `name` constitute a time series of text values.

    For Example:
    ```python
    test_summary_writer = tf.summary.create_file_writer('test/logdir')
    with test_summary_writer.as_default():
        tf.summary.text('first_text', 'hello world!', step=0)
        tf.summary.text('first_text', 'nice to meet you!', step=1)
    ```

    The text summary can also contain Markdown, and TensorBoard will render the text
    as such.

    ```python
    with test_summary_writer.as_default():
        text_data = '''
              | *hello* | *there* |
              |---------|---------|
              | this    | is      |
              | a       | table   |
        '''
        text_data = '\n'.join(l.strip() for l in text_data.splitlines())
        tf.summary.text('markdown_text', text_data, step=0)
    ```

    Since text is Tensor valued, each text point may be a Tensor of string values.
    rank-1 and rank-2 Tensors are rendered as tables in TensorBoard.  For higher ranked
    Tensors, you'll see just a 2D slice of the data.  To avoid this, reshape the Tensor
    to at most rank-2 prior to passing it to this function.

    Demo notebook at
    ["Displaying text data in TensorBoard"](https://www.tensorflow.org/tensorboard/text_summaries).

    Arguments:
      name: A name for this summary. The summary tag used for TensorBoard will
        be this name prefixed by any active name scopes.
      data: A UTF-8 string Tensor value.
      step: Explicit `int64`-castable monotonic step value for this summary. If
        omitted, this defaults to `tf.summary.experimental.get_step()`, which must
        not be None.
      description: Optional long-form description for this summary, as a
        constant `str`. Markdown is supported. Defaults to empty.

    Returns:
      True on success, or false if no summary was emitted because no default
      summary writer was available.

    Raises:
      ValueError: if a default writer exists, but no step was provided and
        `tf.summary.experimental.get_step()` is None.
    """
    summary_metadata = metadata.create_summary_metadata(
        display_name=None, description=description
    )
    # TODO(https://github.com/tensorflow/tensorboard/issues/2109): remove fallback
    summary_scope = (
        getattr(tf.summary.experimental, "summary_scope", None)
        or tf.summary.summary_scope
    )
    with summary_scope(name, "text_summary", values=[data, step]) as (tag, _):
        tf.debugging.assert_type(data, tf.string)
        return tf.summary.write(
            tag=tag, tensor=data, step=step, metadata=summary_metadata
        )


def text_pb(tag, data, description=None):
    """Create a text tf.Summary protobuf.

    Arguments:
      tag: String tag for the summary.
      data: A Python bytestring (of type bytes), a Unicode string, or a numpy data
        array of those types.
      description: Optional long-form description for this summary, as a `str`.
        Markdown is supported. Defaults to empty.

    Raises:
      TypeError: If the type of the data is unsupported.

    Returns:
      A `tf.Summary` protobuf object.
    """
    try:
        tensor = tensor_util.make_tensor_proto(data, dtype=np.object_)
    except TypeError as e:
        raise TypeError("tensor must be of type string", e)
    summary_metadata = metadata.create_summary_metadata(
        display_name=None, description=description
    )
    summary = summary_pb2.Summary()
    summary.value.add(tag=tag, metadata=summary_metadata, tensor=tensor)
    return summary
