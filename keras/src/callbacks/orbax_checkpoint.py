import warnings

import numpy as np

from keras.src import backend
from keras.src import tree
from keras.src.callbacks.monitor_callback import (
    MonitorCallback,  # For metric monitoring logic
)
from keras.src.utils.io_utils import print_msg
from keras.src.utils.module_utils import ocp

# Context and AsyncOptions are accessed through the lazy-loaded ocp module

# JAX monitoring compatibility: ensure record_scalar exists
# to prevent AttributeError in older JAX versions
try:
    import jax

    if not hasattr(jax.monitoring, "record_scalar"):
        jax.monitoring.record_scalar = lambda *args, **kwargs: None
except ImportError:
    pass


def _get_state_tree(model):
    """Get the complete model state as a nested tree structure."""
    # For JAX backend, preserve native arrays for performance
    # For other backends, convert to numpy arrays
    if backend.backend() == "jax":
        state_tree = model.get_state_tree()
        did_numpy_conversion = False
    else:
        state_tree = model.get_state_tree(value_format="numpy_array")
        did_numpy_conversion = True

    # Convert numpy scalar types to Python types for Orbax compatibility
    # Only needed when we did numpy conversion
    if did_numpy_conversion:

        def convert_scalars(obj):
            if isinstance(obj, np.ndarray) and obj.ndim == 0:
                # Convert 0-dimensional numpy arrays (scalars) to Python types
                return obj.item()
            elif isinstance(obj, np.generic):
                # Convert numpy scalar types (like np.float32) to Python types
                return obj.item()
            else:
                return obj

        return tree.map_structure(convert_scalars, state_tree)
    else:
        return state_tree


class OrbaxCheckpoint(MonitorCallback):
    """Callback to save and load model state using Orbax with a similar API to
    ModelCheckpoint.

    This callback saves the model's weights and optimizer state asynchronously
    using Orbax, allowing training to continue without blocking for I/O.

    Example:

    ```python
    model.compile(loss=..., optimizer=..., metrics=['accuracy'])

    EPOCHS = 10
    checkpoint_dir = '/tmp/ckpt'
    orbax_checkpoint_callback = keras.callbacks.OrbaxCheckpoint(
        directory=checkpoint_dir,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Model is saved at the end of every epoch, if it's the best seen so far.
    model.fit(epochs=EPOCHS, callbacks=[orbax_checkpoint_callback])

    # Alternatively, save checkpoints every N batches -
    orbax_checkpoint_callback = keras.callbacks.OrbaxCheckpoint(
        directory=checkpoint_dir,
        save_freq=100)  # Save every 100 batches

    model.fit(epochs=EPOCHS, callbacks=[orbax_checkpoint_callback])
    ```

    Args:
        directory: path to the directory where to save the checkpoints.
        monitor: The metric name to monitor (e.g., 'val_loss').
        verbose: Verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`, it only saves when the model
            is considered the "best" based on the monitored quantity.
        save_weights_only: if `save_weights_only=True`, only the model's
            weights will be saved. Otherwise, the full model state
            (weights, non-trainable variables, optimizer state, and
            metrics state) will be saved. Defaults to False.
        mode: one of {'auto', 'min', 'max'}. Used with `save_best_only`.
        save_freq: `'epoch'` or integer. Frequency to save checkpoints.
        max_to_keep: Integer, maximum number of recent checkpoints to keep.
            If None, keeps all. Defaults to 1.
        save_on_background: Boolean, whether to save asynchronously in the
            background. Defaults to True.
        initial_value_threshold: Floating point initial "best" value for the
            monitor, used with `save_best_only`.
    """

    def __init__(
        self,
        directory,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
        max_to_keep=1,
        save_on_background=True,
    ):
        # Ensure orbax is available
        ocp.initialize()

        # Initialize MonitorCallback for handling 'monitor', 'mode', 'best'
        # logic
        super().__init__(monitor, mode, initial_value_threshold)

        self.directory = directory
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.max_to_keep = max_to_keep
        self.save_on_background = save_on_background
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self._current_epoch = 0  # Keep track of epoch
        self._total_batches_seen = 0  # Global batch counter for step tracking

        if self.save_freq != "epoch" and not isinstance(self.save_freq, int):
            raise ValueError(
                f"Unrecognized save_freq: {self.save_freq}. "
                "Expected save_freq are 'epoch' or integer values"
            )

        # --- Orbax Checkpointer Setup (V1 API) ---
        policies = []
        if max_to_keep is not None:
            policies.append(
                ocp.training.preservation_policies.LatestN(max_to_keep)
            )

        # Use AnyPreservationPolicy to combine them.
        preservation_policy = None
        if policies:
            preservation_policy = (
                ocp.training.preservation_policies.AnyPreservationPolicy(
                    policies
                )
            )

        # Create the V1 Checkpointer with direct parameter passing
        # Orbax will handle directory creation on all processes as needed
        self.checkpointer = ocp.training.Checkpointer(
            directory=directory,
            preservation_policy=preservation_policy,
        )

    def _should_save_on_batch(self, batch):
        """Check if we should save on this batch."""
        if self.save_freq == "epoch":
            return False

        if batch <= self._last_batch_seen:  # New epoch.
            add_batches = batch + 1
        else:
            add_batches = batch - self._last_batch_seen
        self._batches_seen_since_last_saving += add_batches
        self._last_batch_seen = batch
        self._total_batches_seen += add_batches

        if self._batches_seen_since_last_saving >= self.save_freq:
            self._batches_seen_since_last_saving = 0
            return True
        return False

    def _save_checkpoint(self, step, logs=None):
        """Save a checkpoint at the given step."""

        # --- Prepare Composite State (Backend-Agnostic) ---
        state_tree = _get_state_tree(self.model)

        # Save the nested state structures directly (preserving layer
        # names and structure)
        if self.save_weights_only:
            composite_state = {
                "trainable_variables": state_tree["trainable_variables"],
            }
            if "non_trainable_variables" in state_tree:
                composite_state["non_trainable_variables"] = state_tree[
                    "non_trainable_variables"
                ]
        else:
            composite_state = state_tree

        # --- Save Logic (V1 API) ---
        # All processes participate in distributed checkpointing
        # Checkpointer is configured to save unconditionally when
        # save_pytree is called
        if self.verbose > 0:
            print_msg(
                f"OrbaxCheckpoint: Triggering async save for step {step}..."
            )

        # Use a single with statement. If context_options is empty,
        # Context() uses defaults.
        with ocp.Context():
            if self.save_on_background:
                self.checkpointer.save_pytree_async(step, composite_state)
            else:
                self.checkpointer.save_pytree(step, composite_state)

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            # Handle save_best_only logic for batch-level saving
            should_save = True
            if self.save_best_only:
                current = logs.get(self.monitor) if logs else None
                if current is None:
                    warnings.warn(
                        f"Can save best model only with {self.monitor} "
                        f"available, skipping save at batch {batch}.",
                        stacklevel=2,
                    )
                    should_save = False
                elif not self._is_improvement(current, self.best):
                    should_save = False
                else:
                    # Update best value when there's improvement
                    self.best = current

            if should_save:
                # Use global batch count for Orbax save step
                step = self._total_batches_seen
                self._save_checkpoint(step=step, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self._current_epoch = epoch
        if self.monitor_op is None:
            self._set_monitor_op()  # From MonitorCallback

        # For save_freq="epoch", save at every epoch
        should_save = self.save_freq == "epoch"

        # Handle save_best_only logic
        if should_save and self.save_best_only:
            current = logs.get(self.monitor) if logs else None
            if current is None:
                warnings.warn(
                    f"Can save best model only with {self.monitor} available, "
                    f"skipping save at epoch {epoch}.",
                    stacklevel=2,
                )
                should_save = False
            elif not self._is_improvement(current, self.best):
                should_save = False
            else:
                # Update best value when there's improvement
                self.best = current

        if should_save:
            # Use epoch number as the step for Orbax save
            # Keras has already made the save decision - Checkpointer will
            # save unconditionally
            self._save_checkpoint(step=epoch, logs=logs)

    def on_train_end(self, logs=None):
        # Close the Checkpointer to ensure all pending saves complete
        try:
            self.checkpointer.close()
        except Exception:
            pass  # Ignore errors during cleanup

    def wait_until_finished(self):
        """Wait for any in-progress checkpoint operations to complete.
        This method blocks until all asynchronous checkpoint save operations
        have completed. It should be called before attempting to load
        checkpoints if there might be pending save operations.
        """
        # Wait for any async operations to complete
        if hasattr(self.checkpointer, "wait"):
            self.checkpointer.wait()
        else:
            # Fallback for older Orbax versions that don't have wait() method
            while self.checkpointer.is_saving_in_progress():
                import time

                time.sleep(0.1)
