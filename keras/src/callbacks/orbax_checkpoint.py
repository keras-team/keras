import os
import warnings

import keras  # Import Keras itself
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.callbacks.monitor_callback import (
    MonitorCallback,  # For metric monitoring logic
)

try:
    import orbax.checkpoint as ocp
except ImportError:
    ocp = None


def _get_state_as_numpy(model):
    # Explicitly convert Keras weights/variables to NumPy arrays
    try:
        model_weights_np = [
            keras.ops.convert_to_numpy(w) for w in model.weights
        ]
        optimizer_vars_np = [
            keras.ops.convert_to_numpy(v) for v in model.optimizer.variables
        ]
        return model_weights_np, optimizer_vars_np
    except Exception as e:
        warnings.warn(f"Could not convert state to NumPy: {e}")
        return None, None


# Conditional export decorator
def _conditional_export(cls):
    if ocp is not None:
        return keras_export("keras.callbacks.OrbaxCheckpoint")(cls)
    return cls


@_conditional_export
class OrbaxCheckpoint(MonitorCallback):
    """Callback to save and load model state using Orbax with a similar API to
    ModelCheckpoint.

    This callback saves the model's weights and optimizer state asynchronously
    using Orbax, allowing training to continue without blocking for I/O.
    It also provides methods to load checkpoints for resuming training or
    inference.
    It supports policies for keeping checkpoints and deciding when to save.

    Args:
        directory: string, path to the directory where to save the checkpoints.
        monitor: The metric name to monitor (e.g., 'val_loss').
        verbose: Verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`, it only saves when the model
            is considered the "best" based on the monitored quantity.
        mode: one of {'auto', 'min', 'max'}. Used with `save_best_only`.
        save_freq: `'epoch'` or integer. Frequency to save checkpoints.
        max_to_keep: Integer, maximum number of recent checkpoints to keep.
            If None, keeps all. Defaults to 5.
        keep_period: Integer, keep one checkpoint every `keep_period` saves.
            Useful for keeping checkpoints less frequently over long runs.
        initial_value_threshold: Floating point initial "best" value for the
             monitor, used with `save_best_only`.
        save_optimizer_state: Boolean, whether to include optimizer variables
            in the checkpoint. Defaults to True.
        save_on_background: Boolean, whether to save asynchronously in the
            background. Defaults to True.
        save_metadata: Dict or callable, additional metadata to save with each
            checkpoint. If callable, it will be called with (epoch, logs) and
            should return a dict. Defaults to None.
    """

    def __init__(
        self,
        directory,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        mode="auto",
        save_freq="epoch",
        max_to_keep=5,
        keep_period=None,
        initial_value_threshold=None,
        save_optimizer_state=True,
        save_on_background=True,
        save_metadata=None,
    ):
        if ocp is None:
            raise ImportError(
                "OrbaxCheckpoint requires the 'orbax-checkpoint' package. "
                "Install it with: pip install orbax-checkpoint"
            )

        # Initialize MonitorCallback for handling 'monitor', 'mode', 'best'
        # logic
        super().__init__(monitor, mode, initial_value_threshold)

        self.directory = directory
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.save_optimizer_state = save_optimizer_state
        self.save_metadata = save_metadata
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self._current_epoch = 0  # Keep track of epoch

        if self.save_freq != "epoch" and not isinstance(self.save_freq, int):
            raise ValueError("Unrecognized save_freq")

        # --- Orbax CheckpointManager Setup ---
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            keep_period=keep_period,
            enable_async_checkpointing=save_on_background,  # Correct parameter
            # name
            # Add more options here if exposing them (e.g., custom handlers)
        )
        # Ensure directory exists (only needed on one process in multi-host)
        if backend.get_process_index() == 0:
            os.makedirs(directory, exist_ok=True)

        # Create the CheckpointManager
        self.manager = ocp.CheckpointManager(
            directory=directory,
            options=options,
        )

    def set_model(self, model):
        self._model = model

    def _should_save_on_batch(self, batch):
        """Check if we should save on this batch."""
        if self.save_freq == "epoch":
            return False

        self._batches_seen_since_last_saving += 1
        if self._batches_seen_since_last_saving >= self.save_freq:
            self._batches_seen_since_last_saving = 0
            return True
        return False

    def _get_current_step(self):
        # A reliable way to get a global step count
        # Using optimizer iterations is common
        if hasattr(self.model, "optimizer") and hasattr(
            self.model.optimizer, "iterations"
        ):
            # Convert potential backend tensor to int
            return int(
                backend.convert_to_numpy(self.model.optimizer.iterations)
            )
        else:
            # Fallback: use batch count
            return self._last_batch_seen

    def _save_checkpoint(self, step, logs=None):
        """Save a checkpoint at the given step."""
        if self.model is None:
            return

        # --- Prepare Composite State (Backend-Agnostic) ---
        model_weights_np, optimizer_vars_np = _get_state_as_numpy(self.model)

        if model_weights_np is None:
            if self.verbose > 0:
                print("OrbaxCheckpoint: Skipping save due to conversion error")
            return

        composite_state = {"model_weights": model_weights_np}
        if self.save_optimizer_state and optimizer_vars_np is not None:
            composite_state["optimizer_state"] = optimizer_vars_np

        # Add metadata if specified
        if self.save_metadata is not None:
            if callable(self.save_metadata):
                metadata = self.save_metadata(self._current_epoch, logs)
            else:
                metadata = self.save_metadata
            if metadata:
                composite_state["metadata"] = metadata

        # --- Save Logic ---
        # Assuming single host or JAX backend with jax.distributed initialized
        # for now.
        # A robust implementation would need a backend-aware way to check
        # process_index.
        is_primary_host = backend.get_process_index() == 0

        if is_primary_host:
            if self.verbose > 0:
                print(
                    f"OrbaxCheckpoint: Triggering async save for step {step}..."
                )

            # Save the checkpoint
            save_args = ocp.args.StandardSave(composite_state)
            self.manager.save(step, args=save_args)

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            # Use step number (e.g., optimizer iterations) for Orbax save step
            step = self._get_current_step()
            self._save_checkpoint(step=step, logs=logs)
            # Ensure all processes sync after save operation
            self.manager.wait_until_finished()

    def on_epoch_end(self, epoch, logs=None):
        self._current_epoch = epoch
        if self.monitor_op is None:
            self._set_monitor_op()  # From MonitorCallback

        if self.save_freq == "epoch":
            # Use epoch number as the step for Orbax save
            self._save_checkpoint(step=epoch, logs=logs)
            # Ensure all processes sync after save operation
            self.manager.wait_until_finished()

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print("OrbaxCheckpoint: Waiting for final saves to complete...")
        self.manager.wait_until_finished()
        if self.verbose > 0:
            print("OrbaxCheckpoint: All saves finalized.")

    def load_checkpoint(self, step):
        """Load model and optimizer state from a specific checkpoint step.

        Args:
            step: The checkpoint step to load from.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        # In distributed training, only load on primary process
        if backend.get_process_index() != 0:
            return True  # Return True to indicate no error, but no loading
            # performed

        try:
            if self.verbose > 0:
                print(
                    f"OrbaxCheckpoint: Loading checkpoint from step {step}..."
                )

            # Prepare restore arguments - Orbax can restore without explicit
            # template
            restore_args = ocp.args.StandardRestore()

            # Load the checkpoint
            checkpoint_data = self.manager.restore(step, args=restore_args)

            # Restore the model state
            return self._restore_model_state(checkpoint_data)

        except Exception as e:
            if self.verbose > 0:
                print(
                    f"OrbaxCheckpoint: Failed to load checkpoint from step "
                    f"{step}: {e}"
                )
            return False

    def load_latest(self):
        """Load the most recent checkpoint.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            # Get the latest step
            latest_step = self.manager.latest_step()
            if latest_step is None:
                if self.verbose > 0:
                    print("OrbaxCheckpoint: No checkpoints found")
                return False

            return self.load_checkpoint(latest_step)

        except Exception as e:
            if self.verbose > 0:
                print(f"OrbaxCheckpoint: Failed to load latest checkpoint: {e}")
            return False

    def _restore_model_state(self, checkpoint_data):
        """Restore model and optimizer state from checkpoint data."""
        try:
            # Restore model weights
            if "model_weights" in checkpoint_data:
                model_weights_np = checkpoint_data["model_weights"]
                # Convert NumPy arrays back to backend tensors and assign to
                # model
                for i, weight_np in enumerate(model_weights_np):
                    # Convert numpy array back to appropriate backend tensor
                    weight_tensor = keras.ops.convert_to_tensor(weight_np)
                    self.model.weights[i].assign(weight_tensor)

            # Restore optimizer state if available
            if (
                "optimizer_state" in checkpoint_data
                and self.save_optimizer_state
            ):
                optimizer_vars_np = checkpoint_data["optimizer_state"]
                # Convert NumPy arrays back to backend tensors and assign to
                # optimizer
                for i, var_np in enumerate(optimizer_vars_np):
                    var_tensor = keras.ops.convert_to_tensor(var_np)
                    self.model.optimizer.variables[i].assign(var_tensor)

            if self.verbose > 0:
                print("OrbaxCheckpoint: Successfully restored model state")
            return True

        except Exception as e:
            if self.verbose > 0:
                print(f"OrbaxCheckpoint: Failed to restore model state: {e}")
            return False
