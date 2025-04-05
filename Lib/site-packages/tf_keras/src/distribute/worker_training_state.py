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
"""Training state management."""

import os

import tensorflow.compat.v2 as tf

from tf_keras.src import backend
from tf_keras.src.distribute import distributed_file_utils
from tf_keras.src.utils import mode_keys

# isort: off
from tf_keras.src.distribute.distributed_file_utils import (
    support_on_demand_checkpoint_callback,
)  # noqa: E501


MAX_CHECKPOINT_TO_KEEP = 1


class WorkerTrainingState:
    """Training state management class.

    This class provides apis for backing up and restoring the training state.
    This allows model and epoch and batch information to be saved periodically
    and restore for fault-tolerance, also known as preemption-recovery purpose.
    """

    # Constant for `tf.keras.Model` attribute to store the epoch and batch
    # at which the most recently saved checkpoint was saved.
    CKPT_SAVED_EPOCH_UNUSED_VALUE = -1

    CKPT_SAVED_BATCH_UNUSED_VALUE = -1

    def __init__(
        self,
        model,
        checkpoint_dir,
        save_freq="epoch",
        save_before_preemption_arg=None,
    ):
        self._enable_save_before_preemption = save_before_preemption_arg and (
            support_on_demand_checkpoint_callback(model.distribute_strategy)
        )
        self._model = model

        self._save_freq = save_freq
        # The batch and epoch at which the checkpoint is saved. Used for
        # fault-tolerance. GPU device only has int64 dtype registered
        # VarHandleOp.
        self._ckpt_saved_epoch = tf.Variable(
            initial_value=tf.constant(
                self.CKPT_SAVED_EPOCH_UNUSED_VALUE, dtype=tf.int64
            ),
            name="ckpt_saved_epoch",
        )
        self._ckpt_saved_batch = tf.Variable(
            initial_value=tf.constant(
                self.CKPT_SAVED_BATCH_UNUSED_VALUE, dtype=tf.int64
            ),
            name="ckpt_saved_batch",
        )
        # Variable initialization.
        backend.set_value(
            self._ckpt_saved_epoch, self.CKPT_SAVED_EPOCH_UNUSED_VALUE
        )
        backend.set_value(
            self._ckpt_saved_batch, self.CKPT_SAVED_BATCH_UNUSED_VALUE
        )
        # _ckpt_saved_epoch and _ckpt_saved_batch gets tracked and is included
        # in the checkpoint file when backing up.
        checkpoint = tf.train.Checkpoint(
            model=self._model,
            ckpt_saved_epoch=self._ckpt_saved_epoch,
            ckpt_saved_batch=self._ckpt_saved_batch,
            train_counter=self._model._train_counter,
        )

        # If this is single-worker training, checkpoint_dir are the same for
        # write_checkpoint_manager and read_checkpoint_manager.
        #
        # If this is multi-worker training, and this worker should not save
        # checkpoint, we replace the write_checkpoint_manager's checkpoint_dir
        # with a temp filepath, so it writes to a file that will be removed at
        # the end of back_up() call. This is necessary because the
        # SyncOnReadVariable needs to be synced across all the workers in order
        # to be read, and all workers need to perform `save()`.  But all workers
        # should restore from the same checkpoint_dir as passed in
        # read_checkpoint_manager.
        self.read_checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=os.path.join(checkpoint_dir, "chief"),
            max_to_keep=MAX_CHECKPOINT_TO_KEEP,
        )
        write_checkpoint_dir = distributed_file_utils.write_dirpath(
            checkpoint_dir, self._model.distribute_strategy
        )
        if self._model.distribute_strategy.extended.should_checkpoint:
            self.write_checkpoint_manager = self.read_checkpoint_manager
        else:
            self.write_checkpoint_manager = tf.train.CheckpointManager(
                checkpoint,
                directory=write_checkpoint_dir,
                max_to_keep=MAX_CHECKPOINT_TO_KEEP,
            )

        if self._enable_save_before_preemption:
            self.preemption_handler = (
                tf.distribute.experimental.PreemptionCheckpointHandler(
                    self._model.distribute_strategy.cluster_resolver,
                    self.write_checkpoint_manager,
                )
            )
            self.preemption_handler._read_checkpoint_manager = (
                self.read_checkpoint_manager
            )
            self._model._preemption_handler = self.preemption_handler

    def back_up(self, epoch, batch=0):
        """Back up the current state of training into a checkpoint file.

        Args:
          epoch: The current epoch information to be saved.
          batch: The current batch(step) information to be saved.
        """
        # Save the model plus CKPT_SAVED_EPOCH and CKPT_SAVED_BATCH variable.
        if self.write_checkpoint_manager.save():
            distributed_file_utils.remove_temp_dirpath(
                self.write_checkpoint_manager.directory,
                self._model.distribute_strategy,
            )

    def backup_if_preempted(self):
        if self._enable_save_before_preemption:
            self.preemption_handler._run_counter += 1
            self.preemption_handler._check_preemption_and_maybe_checkpoint()

    def restore(self):
        """Restore the training state from the backed up checkpoint file.

        Returns:
          True if the training state is successfully restored. False if the
          training state doesn't need to be restored, or error occurred so it
          can't.
        """
        # When creating the PreemptionCheckpointHandler object, we have already
        # restored the checkpoint.
        if not self._enable_save_before_preemption:
            self.read_checkpoint_manager.restore_or_initialize()

    def delete_backup(self):
        """Delete the backup directories.

        Delete the backup directories which should not exist after `fit()`
        successfully finishes.
        """
        if self.write_checkpoint_manager is self.read_checkpoint_manager:
            try:
                tf.io.gfile.rmtree(self.write_checkpoint_manager.directory)
            except tf.errors.NotFoundError:
                pass

    def maybe_load_initial_counters_from_ckpt(
        self, steps_per_epoch, initial_epoch, mode
    ):
        """Maybe load 1st epoch from checkpoint, considering worker recovery.

        When `_ckpt_saved_epoch` attribute exists and is not
        `CKPT_SAVED_EPOCH_UNUSED_VALUE`, this is under multi-worker training
        setting and indicates the worker is recovering from previous failure. In
        this case, infer `initial_epoch` from `self._ckpt_saved_epoch` to
        continue previous unfinished training from certain epoch.

        Args:
          steps_per_epoch: The number of steps per epoch value.
          initial_epoch: The original initial_epoch user passes in in `fit()`.
          mode: The mode for running `model.fit()`.

        Returns:
          If the training is recovering from previous failure under multi-worker
          training setting, return the (epoch, step) the training is supposed to
          continue at. Otherwise, return the `initial_epoch, initial_step` the
          user passes in.
        """

        initial_step = 0
        epoch = backend.eval(self._ckpt_saved_epoch)
        batch = backend.eval(self._ckpt_saved_batch)
        if mode == mode_keys.ModeKeys.TRAIN:
            # For batch-level saving
            if self._enable_save_before_preemption or isinstance(
                self._save_freq, int
            ):
                if batch >= 0:
                    # If the checkpoint was last saved at last batch of the
                    # epoch, return the next epoch number and batch=0
                    if batch == steps_per_epoch - 1:
                        initial_epoch = epoch + 1
                        initial_step = 0
                    else:
                        # If the checkpoint was not last saved at last batch of
                        # the epoch, return the same epoch and next batch number
                        initial_epoch = epoch
                        initial_step = batch + 1
            else:
                if epoch >= 0:
                    # The most recently saved epoch is one epoch prior to the
                    # epoch it failed at, so return the value of
                    # 'self._ckpt_saved_epoch' plus one.
                    initial_epoch = epoch + 1

        return (initial_epoch, initial_step)

