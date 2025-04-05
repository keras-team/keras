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

"""TensorBoard encoder helper module.

Encoder depends on TensorFlow.
"""


import numpy as np

from tensorboard.util import op_evaluator


class _TensorFlowPngEncoder(op_evaluator.PersistentOpEvaluator):
    """Encode an image to PNG.

    This function is thread-safe, and has high performance when run in
    parallel. See `encode_png_benchmark.py` for details.

    Arguments:
      image: A numpy array of shape `[height, width, channels]`, where
        `channels` is 1, 3, or 4, and of dtype uint8.

    Returns:
      A bytestring with PNG-encoded data.
    """

    def __init__(self):
        super().__init__()
        self._image_placeholder = None
        self._encode_op = None

    def initialize_graph(self):
        # TODO(nickfelt): remove on-demand imports once dep situation is fixed.
        import tensorflow.compat.v1 as tf

        self._image_placeholder = tf.placeholder(
            dtype=tf.uint8, name="image_to_encode"
        )
        self._encode_op = tf.image.encode_png(self._image_placeholder)

    def run(self, image):  # pylint: disable=arguments-differ
        if not isinstance(image, np.ndarray):
            raise ValueError("'image' must be a numpy array: %r" % image)
        if image.dtype != np.uint8:
            raise ValueError(
                "'image' dtype must be uint8, but is %r" % image.dtype
            )
        return self._encode_op.eval(feed_dict={self._image_placeholder: image})


encode_png = _TensorFlowPngEncoder()


class _TensorFlowWavEncoder(op_evaluator.PersistentOpEvaluator):
    """Encode an audio clip to WAV.

    This function is thread-safe and exhibits good parallel performance.

    Arguments:
      audio: A numpy array of shape `[samples, channels]`.
      samples_per_second: A positive `int`, in Hz.

    Returns:
      A bytestring with WAV-encoded data.
    """

    def __init__(self):
        super().__init__()
        self._audio_placeholder = None
        self._samples_per_second_placeholder = None
        self._encode_op = None

    def initialize_graph(self):
        # TODO(nickfelt): remove on-demand imports once dep situation is fixed.
        import tensorflow.compat.v1 as tf

        self._audio_placeholder = tf.placeholder(
            dtype=tf.float32, name="image_to_encode"
        )
        self._samples_per_second_placeholder = tf.placeholder(
            dtype=tf.int32, name="samples_per_second"
        )
        self._encode_op = tf.audio.encode_wav(
            self._audio_placeholder,
            sample_rate=self._samples_per_second_placeholder,
        )

    def run(
        self, audio, samples_per_second
    ):  # pylint: disable=arguments-differ
        if not isinstance(audio, np.ndarray):
            raise ValueError("'audio' must be a numpy array: %r" % audio)
        if not isinstance(samples_per_second, int):
            raise ValueError(
                "'samples_per_second' must be an int: %r" % samples_per_second
            )
        feed_dict = {
            self._audio_placeholder: audio,
            self._samples_per_second_placeholder: samples_per_second,
        }
        return self._encode_op.eval(feed_dict=feed_dict)


encode_wav = _TensorFlowWavEncoder()
