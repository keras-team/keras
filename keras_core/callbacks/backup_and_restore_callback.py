import os
import warnings

from keras_core.api_export import keras_core_export
from keras_core.callbacks.callback import Callback
from keras_core.utils import file_utils


@keras_core_export("keras_core.callbacks.BackupAndRestore")
class BackupAndRestore(Callback):
    """Callback to back up and restore the training state.

    `BackupAndRestore` callback is intended to recover training from an
    interruption that has happened in the middle of a `Model.fit` execution, by
    backing up the training states in a temporary checkpoint file, at the end of
    each epoch. Each backup overwrites the previously written checkpoint file,
    so at any given time there is at most one such checkpoint file for
    backup/restoring purpose.

    If training restarts before completion, the training state (which includes
    the `Model` weights and epoch number) is restored to the most recently saved
    state at the beginning of a new `Model.fit` run. At the completion of a
    `Model.fit` run, the temporary checkpoint file is deleted.

    Note that the user is responsible to bring jobs back after the interruption.
    This callback is important for the backup and restore mechanism for fault
    tolerance purpose, and the model to be restored from a previous checkpoint
    is expected to be the same as the one used to back up. If user changes
    arguments passed to compile or fit, the checkpoint saved for fault tolerance
    can become invalid.

    Example:

    >>> class InterruptingCallback(keras.callbacks.Callback):
    ...   def on_epoch_begin(self, epoch, logs=None):
    ...     if epoch == 4:
    ...       raise RuntimeError('Interrupting!')
    >>> callback = keras.callbacks.BackupAndRestore(backup_dir="/tmp/backup")
    >>> model = keras.models.Sequential([keras.layers.Dense(10)])
    >>> model.compile(keras.optimizers.SGD(), loss='mse')
    >>> try:
    ...   model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
    ...             batch_size=1, callbacks=[callback, InterruptingCallback()],
    ...             verbose=0)
    ... except:
    ...   pass
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=10, batch_size=1, callbacks=[callback],
    ...                     verbose=0)
    >>> # Only 6 more epochs are run, since first training got interrupted at
    >>> # zero-indexed epoch 4, second training will continue from 4 to 9.
    >>> len(history.history['loss'])
    >>> 6

    Args:
        file_path: String, path to store the checkpoint.
          e.g. `backup_dir = os.path.join(working_dir, "backup")`.
          This is the directory in which the system stores temporary files to
          recover the model from jobs terminated unexpectedly. The directory
          cannot be reused elsewhere to store other files, e.g. by the
          `BackupAndRestore` callback of another training run,
          or by another callback
          (e.g. `ModelCheckpoint`) of the same training.
        save_freq: `"epoch"`, integer, or `False`. When set to `"epoch"`
          the callback saves the checkpoint at the end of each epoch.
          When set to an integer, the callback saves the checkpoint every
          `save_freq` batches. Set `save_freq` to `False` if only using
          preemption checkpointing (with `save_before_preemption=True`).
        delete_checkpoint: Boolean, default to True. This `BackupAndRestore`
          callback works by saving a checkpoint to back up the training state.
          If `delete_checkpoint=True`, the checkpoint will be deleted after
          training is finished. Use `False` if you'd like to keep the checkpoint
          for future usage.
        save_before_preemption: A boolean value instructing whether to turn on
          the automatic checkpoint saving for preemption/maintenance events.
    """

    def __init__(
        self,
        file_path,
        save_freq="epoch",
        delete_checkpoint=True,
        save_before_preemption=False,
    ):
        super().__init__()
        self._current_epoch = 0
        self.save_freq = save_freq
        self.delete_checkpoint = delete_checkpoint
        self.save_before_preemption = save_before_preemption
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0

        if not file_path:
            raise ValueError("Empty `backup_dir` argument passed")
        self.file_path = file_path

        if not save_freq and not save_before_preemption:
            raise ValueError(
                "Either `save_freq` or `save_before_preemption` " "must be set."
            )

        if self.save_freq != "epoch" and not isinstance(self.save_freq, int):
            raise ValueError(
                f"Unrecognized save_freq: {self.save_freq}. "
                "Expected save_freq are 'epoch' or integer values"
            )
        if self.save_before_preemption:
            warnings.warn("`save_before_preemption` not yet implemented")

    def on_train_begin(self, logs=None):
        """
        Get training state from temporary file and restore it
        """
        if self._check_checkpoints_exists(self.file_path):
            self._model.load_weights(filepath=self.file_path)

    def on_train_end(self, logs=None):
        if self.delete_checkpoint and self._check_checkpoints_exists(
            self.file_path
        ):
            self._cleanup_checkpoint()

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if self.save_freq == "epoch":
            self._save_model(epoch=epoch, batch=None, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self._save_model(epoch=self._current_epoch, batch=batch, logs=logs)

    def _save_model(self, epoch, batch, logs):
        """Saves the model.

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
                is set to `"epoch"`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        filepath = self._get_file_path(epoch, batch, logs)
        # Create host directory if it doesn't exist.
        dirname = os.path.dirname(filepath)
        if dirname and not file_utils.exists(dirname):
            file_utils.makedirs(dirname)

        try:
            self._model.save_weights(filepath=filepath, overwrite=True)
        except IsADirectoryError:  # h5py 3.x
            raise IOError(
                "Please specify a non-directory filepath for "
                "ModelCheckpoint. Filepath used is an existing "
                f"directory: {filepath}"
            )
        except IOError as e:  # h5py 2.x
            # `e.errno` appears to be `None` so checking the content of
            # `e.args[0]`.
            if "is a directory" in str(e.args[0]).lower():
                raise IOError(
                    "Please specify a non-directory filepath for "
                    "ModelCheckpoint. Filepath used is an existing "
                    f"directory: f{filepath}"
                )
            # Re-throw the error for any other causes.
            raise e

    def _get_file_path(self, epoch, batch, logs):
        """Returns the file path for checkpoint."""

        try:
            # `filepath` may contain placeholders such as
            # `{epoch:02d}`,`{batch:02d}` and `{mape:.2f}`. A mismatch between
            # logged metrics and the path's placeholders can cause formatting to
            # fail.
            if batch is None or "batch" in logs:
                file_path = self.file_path.format(epoch=epoch + 1, **logs)
            else:
                file_path = self.file_path.format(
                    epoch=epoch + 1, batch=batch + 1, **logs
                )
        except KeyError as e:
            raise KeyError(
                f'Failed to format this callback filepath: "{self.file_path}". '
                f"Reason: {e}"
            )
        return file_path

    def _should_save_on_batch(self, batch):
        """Handles batch-level saving logic, supports steps_per_execution."""
        if self.save_freq == "epoch":
            return False
        if batch <= self._last_batch_seen:  # New epoch.
            add_batches = batch + 1  # batches are zero-indexed.
        else:
            add_batches = batch - self._last_batch_seen
        self._batches_seen_since_last_saving += add_batches
        self._last_batch_seen = batch

        if self._batches_seen_since_last_saving >= self.save_freq:
            self._batches_seen_since_last_saving = 0
            return True
        return False

    def _cleanup_checkpoint(self):
        """
        Delete other checkpoint files (if present) in the directory
        """
        if self._check_checkpoints_exists(filepath=self.file_path):
            file_utils.rmtree(self.file_path)

    def _check_checkpoints_exists(self, filepath):
        return file_utils.exists(filepath)
