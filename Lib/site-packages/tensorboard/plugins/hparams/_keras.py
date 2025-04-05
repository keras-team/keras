# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Keras integration for TensorBoard hparams.

Use `tensorboard.plugins.hparams.api` to access this module's contents.
"""


import tensorflow as tf

from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary
from tensorboard.plugins.hparams import summary_v2


class Callback(tf.keras.callbacks.Callback):
    """Callback for logging hyperparameters to TensorBoard.

    NOTE: This callback only works in TensorFlow eager mode.
    """

    def __init__(self, writer, hparams, trial_id=None):
        """Create a callback for logging hyperparameters to TensorBoard.

        As with the standard `tf.keras.callbacks.TensorBoard` class, each
        callback object is valid for only one call to `model.fit`.

        Args:
          writer: The `SummaryWriter` object to which hparams should be
            written, or a logdir (as a `str`) to be passed to
            `tf.summary.create_file_writer` to create such a writer.
          hparams: A `dict` mapping hyperparameters to the values used in
            this session. Keys should be the names of `HParam` objects used
            in an experiment, or the `HParam` objects themselves. Values
            should be Python `bool`, `int`, `float`, or `string` values,
            depending on the type of the hyperparameter.
          trial_id: An optional `str` ID for the set of hyperparameter
            values used in this trial. Defaults to a hash of the
            hyperparameters.

        Raises:
          ValueError: If two entries in `hparams` share the same
            hyperparameter name.
        """
        # Defer creating the actual summary until we write it, so that the
        # timestamp is correct. But create a "dry-run" first to fail fast in
        # case the `hparams` are invalid.
        self._hparams = dict(hparams)
        self._trial_id = trial_id
        summary_v2.hparams_pb(self._hparams, trial_id=self._trial_id)
        if writer is None:
            raise TypeError(
                "writer must be a `SummaryWriter` or `str`, not None"
            )
        elif isinstance(writer, str):
            self._writer = tf.compat.v2.summary.create_file_writer(writer)
        else:
            self._writer = writer

    def _get_writer(self):
        if self._writer is None:
            raise RuntimeError(
                "hparams Keras callback cannot be reused across training sessions"
            )
        if not tf.executing_eagerly():
            raise RuntimeError(
                "hparams Keras callback only supported in TensorFlow eager mode"
            )
        return self._writer

    def on_train_begin(self, logs=None):
        del logs  # unused
        with self._get_writer().as_default():
            summary_v2.hparams(self._hparams, trial_id=self._trial_id)

    def on_train_end(self, logs=None):
        del logs  # unused
        with self._get_writer().as_default():
            pb = summary.session_end_pb(api_pb2.STATUS_SUCCESS)
            raw_pb = pb.SerializeToString()
            tf.compat.v2.summary.experimental.write_raw_pb(raw_pb, step=0)
        self._writer = None
