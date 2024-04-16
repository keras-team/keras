import json

from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.utils import file_utils


@keras_export("keras.callbacks.BackupAndRestore")
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
        backup_dir: String, path of directory where to store the data
            needed to restore the model. The directory
            cannot be reused elsewhere to store other files, e.g. by the
            `BackupAndRestore` callback of another training run,
            or by another callback (e.g. `ModelCheckpoint`)
            of the same training run.
        save_freq: `"epoch"`, integer, or `False`. When set to `"epoch"`
          the callback saves the checkpoint at the end of each epoch.
          When set to an integer, the callback saves the checkpoint every
          `save_freq` batches. Set `save_freq=False` only if using
          preemption checkpointing (i.e. with `save_before_preemption=True`).
        delete_checkpoint: Boolean, defaults to `True`. This `BackupAndRestore`
          callback works by saving a checkpoint to back up the training state.
          If `delete_checkpoint=True`, the checkpoint will be deleted after
          training is finished. Use `False` if you'd like to keep the checkpoint
          for future usage.
    """

    def __init__(
        self,
        backup_dir,
        save_freq="epoch",
        delete_checkpoint=True,
    ):
        super().__init__()
        self.save_freq = save_freq
        self.delete_checkpoint = delete_checkpoint
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self._current_epoch = 0

        if not backup_dir:
            raise ValueError("Empty `backup_dir` argument passed")
        self.backup_dir = backup_dir
        self._weights_path = file_utils.join(backup_dir, "latest.weights.h5")
        self._training_metadata_path = file_utils.join(
            backup_dir, "training_metadata.json"
        )
        if save_freq != "epoch" and not isinstance(save_freq, int):
            raise ValueError(
                "Invalid value for argument `save_freq`. "
                f"Received: save_freq={save_freq}. "
                "Expected either 'epoch' or an integer value."
            )

    def on_train_begin(self, logs=None):
        """Get training state from temporary file and restore it."""
        if not self.model.built:
            raise ValueError(
                "To use the BackupAndRestore callback, "
                "you model must be built before you call `fit()`. "
                f"Model {self.model} is unbuilt. You can build it "
                "beforehand by calling it on a batch of data."
            )
        if file_utils.exists(self._weights_path):
            if (
                self.model.optimizer is not None
                and not self.model.optimizer.built
            ):
                # Make sure optimizer weights exist before loading.
                self.model.optimizer.build(self.model.trainable_variables)
            self.model.load_weights(self._weights_path)

        if file_utils.exists(self._training_metadata_path):
            with file_utils.File(self._training_metadata_path, "r") as f:
                training_metadata = json.loads(f.read())
            epoch = training_metadata["epoch"]
            self.model._initial_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self._current_epoch = epoch + 1
        self._last_batch_seen = 0
        if self.save_freq == "epoch":
            self._save_model()

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self._save_model()

    def _save_model(self):
        """Saves the model.

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
                is set to `"epoch"`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        # Create host directory if it doesn't exist.
        if not file_utils.exists(self.backup_dir):
            file_utils.makedirs(self.backup_dir)
        self.model.save_weights(filepath=self._weights_path, overwrite=True)
        with file_utils.File(self._training_metadata_path, "w") as f:
            training_metadata = {
                "epoch": self._current_epoch,
                "batch": self._last_batch_seen,
            }
            f.write(json.dumps(training_metadata))

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

    def on_train_end(self, logs=None):
        if self.delete_checkpoint and file_utils.exists(self.backup_dir):
            file_utils.rmtree(self.backup_dir)
