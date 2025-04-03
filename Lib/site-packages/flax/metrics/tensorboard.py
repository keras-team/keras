# Copyright 2024 The Flax Authors.
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

"""Write Summaries from JAX for use with Tensorboard."""

import contextlib
import functools
import os

import numpy as np
import tensorflow as tf  # pytype: disable=import-error
from tensorboard.plugins.hparams import api as hparams_api

# pylint: disable=g-import-not-at-top
from flax import io


def _flatten_dict(input_dict, parent_key='', sep='.'):
  """Flattens and simplifies dict such that it can be used by hparams.

  Args:
    input_dict: Input dict, e.g., from ConfigDict.
    parent_key: String used in recursion.
    sep: String used to separate parent and child keys.

  Returns:
   Flattened dict.
  """
  items = []
  for k, v in input_dict.items():
    new_key = parent_key + sep + k if parent_key else k

    # Valid types according to https://github.com/tensorflow/tensorboard/blob/1204566da5437af55109f7a4af18f9f8b7c4f864/tensorboard/plugins/hparams/summary_v2.py
    valid_types = (
      bool,
      int,
      float,
      str,
      np.bool_,
      np.integer,
      np.floating,
      np.character,
    )

    if isinstance(v, dict):
      # Recursively flatten the dict.
      items.extend(_flatten_dict(v, new_key, sep=sep).items())
      continue
    elif not isinstance(v, valid_types):
      # Cast any incompatible values as strings such that they can be handled by hparams
      v = str(v)
    items.append((new_key, v))
  return dict(items)


@contextlib.contextmanager
def _as_default(summary_writer: tf.summary.SummaryWriter, auto_flush: bool):
  """No-flush variation of summary_writer.as_default()."""
  context_manager = summary_writer.as_default()
  try:
    context_manager.__enter__()
    yield summary_writer
  finally:
    old_flush = summary_writer.flush
    new_flush = old_flush if auto_flush else lambda: None
    summary_writer.flush = new_flush
    context_manager.__exit__(None, None, None)
    summary_writer.flush = old_flush


class SummaryWriter:
  """Saves data in event and summary protos for tensorboard."""

  def __init__(self, log_dir, auto_flush=True):
    """Create a new SummaryWriter.

    Args:
      log_dir: path to record tfevents files in.
      auto_flush: if true, flush after every reported metric.
    """
    log_dir = os.fspath(log_dir)

    # If needed, create log_dir directory as well as missing parent directories.
    if not io.isdir(log_dir):
      io.makedirs(log_dir)

    self._event_writer = tf.summary.create_file_writer(log_dir)
    self._as_default = functools.partial(_as_default, auto_flush=auto_flush)
    self._closed = False

  def close(self):
    """Close SummaryWriter. Final!"""
    if not self._closed:
      self._event_writer.close()
      self._closed = True
      del self._event_writer

  def flush(self):
    self._event_writer.flush()

  def scalar(self, tag, value, step):
    """Saves scalar value.

    Args:
      tag: str: label for this data
      value: int/float: number to log
      step: int: training step
    """
    value = float(np.array(value))
    with self._as_default(self._event_writer):
      tf.summary.scalar(name=tag, data=value, step=step)

  def image(self, tag, image, step, max_outputs=3):
    """Saves RGB image summary from np.ndarray [H,W], [H,W,1], or [H,W,3].

    Args:
      tag: str: label for this data
      image: ndarray: [H,W], [H,W,1], [H,W,3], [K,H,W], [K,H,W,1], [K,H,W,3]
        Save image in greyscale or colors.
        Pixel values could be either uint8 or float.
        Floating point values should be in range [0, 1).
      step: int: training step
      max_outputs: At most this many images will be emitted at each step.
        Defaults to 3.
    """
    image = np.array(image)
    # tf.summary.image expects image to have shape [k, h, w, c] where,
    # k = number of samples, h = height, w = width, c = number of channels.
    if len(np.shape(image)) == 2:
      image = image[np.newaxis, :, :, np.newaxis]
    elif len(np.shape(image)) == 3:
      # this could be either [k, h, w] or [h, w, c]
      if np.shape(image)[-1] in (1, 3):
        image = image[np.newaxis, :, :, :]
      else:
        image = image[:, :, :, np.newaxis]
    if np.shape(image)[-1] == 1:
      image = np.repeat(image, 3, axis=-1)

    # Convert to tensor value as tf.summary.image expects data to be a tensor.
    image = tf.convert_to_tensor(image)
    with self._as_default(self._event_writer):
      tf.summary.image(name=tag, data=image, step=step, max_outputs=max_outputs)

  def audio(self, tag, audiodata, step, sample_rate=44100, max_outputs=3):
    """Saves audio as wave.

    NB: single channel only right now.

    Args:
      tag: str: label for this data
      audiodata: ndarray [Nsamples, Nframes, Nchannels]: audio data to
        be saved as wave. The data will be clipped to [-1.0, 1.0].
      step: int: training step
      sample_rate: sample rate of passed in audio buffer
      max_outputs: At most this many audio clips will be emitted at each
        step. Defaults to 3.
    """
    # tf.summary.audio expects the audio data to have floating values in
    # [-1.0, 1.0].
    audiodata = np.clip(np.array(audiodata), -1, 1)

    # Convert to tensor value as tf.summary.audio expects data to be a tensor.
    audio = tf.convert_to_tensor(audiodata, dtype=tf.float32)
    with self._as_default(self._event_writer):
      tf.summary.audio(
        name=tag,
        data=audio,
        sample_rate=sample_rate,
        step=step,
        max_outputs=max_outputs,
        encoding='wav',
      )

  def histogram(self, tag, values, step, bins=None):
    """Saves histogram of values.

    Args:
      tag: str: label for this data
      values: ndarray: will be flattened by this routine
      step: int: training step
      bins: number of bins in histogram
    """
    values = np.array(values)
    values = np.reshape(values, -1)
    with self._as_default(self._event_writer):
      tf.summary.histogram(name=tag, data=values, step=step, buckets=bins)

  def text(self, tag, textdata, step):
    """Saves a text summary.

    Args:
      tag: str: label for this data
      textdata: string
      step: int: training step
    Note: markdown formatting is rendered by tensorboard.
    """
    if not isinstance(textdata, (str, bytes)):
      raise ValueError('`textdata` should be of the type `str` or `bytes`.')
    with self._as_default(self._event_writer):
      tf.summary.text(name=tag, data=tf.constant(textdata), step=step)

  def write(self, tag, tensor, step, metadata=None):
    """Saves an arbitrary tensor summary.

    Useful when working with custom plugins or constructing a summary directly.

    Args:
      tag: str: label for this data
      tensor: ndarray: tensor data to save.
      step: int: training step
      metadata: Optional SummaryMetadata, as a proto or serialized bytes.
    Note: markdown formatting is rendered by tensorboard.
    """
    with self._as_default(self._event_writer):
      tf.summary.write(tag=tag, tensor=tensor, step=step, metadata=metadata)

  def hparams(self, hparams):
    """Saves hyper parameters.

    Args:
      hparams: Flat mapping from hyper parameter name to value.
    """

    with self._as_default(self._event_writer):
      hparams_api.hparams(hparams=_flatten_dict(hparams))
