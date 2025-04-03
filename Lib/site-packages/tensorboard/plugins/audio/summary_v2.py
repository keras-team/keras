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
"""Audio summaries and TensorFlow operations to create them, V2 versions.

An audio summary stores a rank-2 string tensor of shape `[k, 2]`, where
`k` is the number of audio clips recorded in the summary. Each row of
the tensor is a pair `[encoded_audio, label]`, where `encoded_audio` is
a binary string whose encoding is specified in the summary metadata, and
`label` is a UTF-8 encoded Markdown string describing the audio clip.
"""


import functools

from tensorboard.compat import tf2 as tf
from tensorboard.plugins.audio import metadata
from tensorboard.util import lazy_tensor_creator


def audio(
    name,
    data,
    sample_rate,
    step=None,
    max_outputs=3,
    encoding=None,
    description=None,
):
    """Write an audio summary.

    Arguments:
      name: A name for this summary. The summary tag used for TensorBoard will
        be this name prefixed by any active name scopes.
      data: A `Tensor` representing audio data with shape `[k, t, c]`,
        where `k` is the number of audio clips, `t` is the number of
        frames, and `c` is the number of channels. Elements should be
        floating-point values in `[-1.0, 1.0]`. Any of the dimensions may
        be statically unknown (i.e., `None`).
      sample_rate: An `int` or rank-0 `int32` `Tensor` that represents the
        sample rate, in Hz. Must be positive.
      step: Explicit `int64`-castable monotonic step value for this summary. If
        omitted, this defaults to `tf.summary.experimental.get_step()`, which must
        not be None.
      max_outputs: Optional `int` or rank-0 integer `Tensor`. At most this
        many audio clips will be emitted at each step. When more than
        `max_outputs` many clips are provided, the first `max_outputs`
        many clips will be used and the rest silently discarded.
      encoding: Optional constant `str` for the desired encoding. Only "wav"
        is currently supported, but this is not guaranteed to remain the
        default, so if you want "wav" in particular, set this explicitly.
      description: Optional long-form description for this summary, as a
        constant `str`. Markdown is supported. Defaults to empty.

    Returns:
      True on success, or false if no summary was emitted because no default
      summary writer was available.

    Raises:
      ValueError: if a default writer exists, but no step was provided and
        `tf.summary.experimental.get_step()` is None.
    """
    audio_ops = getattr(tf, "audio", None)
    if audio_ops is None:
        # Fallback for older versions of TF without tf.audio.
        from tensorflow.python.ops import gen_audio_ops as audio_ops

    if encoding is None:
        encoding = "wav"
    if encoding != "wav":
        raise ValueError("Unknown encoding: %r" % encoding)
    summary_metadata = metadata.create_summary_metadata(
        display_name=None,
        description=description,
        encoding=metadata.Encoding.Value("WAV"),
    )
    inputs = [data, sample_rate, max_outputs, step]
    # TODO(https://github.com/tensorflow/tensorboard/issues/2109): remove fallback
    summary_scope = (
        getattr(tf.summary.experimental, "summary_scope", None)
        or tf.summary.summary_scope
    )
    with summary_scope(name, "audio_summary", values=inputs) as (tag, _):
        # Defer audio encoding preprocessing by passing it as a callable to write(),
        # wrapped in a LazyTensorCreator for backwards compatibility, so that we
        # only do this work when summaries are actually written.
        @lazy_tensor_creator.LazyTensorCreator
        def lazy_tensor():
            tf.debugging.assert_rank(data, 3)
            tf.debugging.assert_non_negative(max_outputs)
            limited_audio = data[:max_outputs]
            encode_fn = functools.partial(
                audio_ops.encode_wav, sample_rate=sample_rate
            )
            encoded_audio = tf.map_fn(
                encode_fn,
                limited_audio,
                dtype=tf.string,
                name="encode_each_audio",
            )
            # Workaround for map_fn returning float dtype for an empty elems input.
            encoded_audio = tf.cond(
                tf.shape(input=encoded_audio)[0] > 0,
                lambda: encoded_audio,
                lambda: tf.constant([], tf.string),
            )
            limited_labels = tf.tile([""], tf.shape(input=limited_audio)[:1])
            return tf.transpose(a=tf.stack([encoded_audio, limited_labels]))

        # To ensure that audio encoding logic is only executed when summaries
        # are written, we pass callable to `tensor` parameter.
        return tf.summary.write(
            tag=tag, tensor=lazy_tensor, step=step, metadata=summary_metadata
        )
