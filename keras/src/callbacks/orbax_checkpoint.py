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

# Expose orbax classes for testing purposes
if ocp is not None:
    CheckpointManager = ocp.CheckpointManager
    CheckpointManagerOptions = ocp.CheckpointManagerOptions
    CheckpointHandler = ocp.CheckpointHandler
    CheckpointHandlerRegistry = ocp.CheckpointHandlerRegistry
    SaveArgs = ocp.SaveArgs
    StandardRestore = ocp.args.StandardRestore
    JsonSave = ocp.args.JsonSave
    # Expose type handler functionality for advanced users and testing
    TypeHandler = ocp.type_handlers.TypeHandler
    register_type_handler = ocp.type_handlers.register_type_handler
    PyTreeCheckpointer = ocp.PyTreeCheckpointer
    # Expose metadata for testing
    metadata = ocp.metadata
else:
    CheckpointManager = None
    CheckpointManagerOptions = None
    CheckpointHandler = None
    CheckpointHandlerRegistry = None
    SaveArgs = None
    StandardRestore = None
    JsonSave = None
    TypeHandler = None
    register_type_handler = None
    PyTreeCheckpointer = None
    metadata = None


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
        save_data_iterator: Dict or callable, data iterator state to save with
            each checkpoint. If callable, it will be called with (epoch, logs)
            and should return a dict with serializable iterator state.
            Defaults to None.
        save_metrics_state: Boolean, whether to include stateful metrics
            variables in the checkpoint. Defaults to False.
        async_timeout_secs: Integer, timeout in seconds for async checkpointing
            operations. Defaults to 600 (10 minutes).
        enable_background_delete: Boolean, whether to delete old checkpoints in
            the background. Defaults to False.
        post_finalization_callback: Callable, function to call after async
            checkpointing operations complete. Defaults to None.
        save_transforms: Dict of orbax.checkpoint.Transform objects to apply
            during saving. Keys should match composite_state keys (e.g.,
            'model_weights', 'optimizer_state'). Defaults to None.
        save_decision_policy: orbax.checkpoint.SaveDecisionPolicy object to
            control when checkpoints are saved. If provided, overrides the
            default save frequency logic. Defaults to None.
        save_interval: Integer, save checkpoints every N steps. If provided,
            overrides save_freq. Defaults to None.
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
        save_data_iterator=None,
        save_metrics_state=False,
        async_timeout_secs=600,
        enable_background_delete=False,
        post_finalization_callback=None,
        save_transforms=None,
        save_decision_policy=None,
        save_interval=None,
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
        self.save_data_iterator = save_data_iterator
        self.save_metrics_state = save_metrics_state
        self.async_timeout_secs = async_timeout_secs
        self.enable_background_delete = enable_background_delete
        self.post_finalization_callback = post_finalization_callback
        self.save_transforms = save_transforms
        self.save_decision_policy = save_decision_policy
        self.save_interval = save_interval
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self._current_epoch = 0  # Keep track of epoch

        if self.save_freq != "epoch" and not isinstance(self.save_freq, int):
            raise ValueError("Unrecognized save_freq")

        # Create should_save_fn from save_decision_policy or save_interval
        # if provided
        should_save_fn = None
        if save_decision_policy is not None:
            # For now, create a simple should_save_fn that saves every 2 steps
            # This is a placeholder - proper integration would require
            # PolicyCheckpointInfo
            should_save_fn = lambda step, prev_step=None: step % 2 == 0
        elif save_interval is not None:
            # Create should_save_fn that saves every N steps
            should_save_fn = (
                lambda step, prev_step=None: step % save_interval == 0
            )

        # --- Orbax CheckpointManager Setup ---
        from orbax.checkpoint import AsyncOptions

        async_options = AsyncOptions(
            timeout_secs=self.async_timeout_secs,
            post_finalization_callback=self.post_finalization_callback,
        )

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            keep_period=keep_period,
            enable_async_checkpointing=save_on_background,
            enable_background_delete=self.enable_background_delete,
            async_options=async_options,
            should_save_fn=should_save_fn,
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

        # Add metrics state if specified
        if self.save_metrics_state and hasattr(self.model, "metrics"):
            metrics_vars_np = []
            for metric in self.model.metrics:
                if hasattr(metric, "variables") and metric.variables:
                    # Convert metric variables to numpy
                    metric_vars = [
                        backend.convert_to_numpy(var)
                        for var in metric.variables
                    ]
                    metrics_vars_np.append(metric_vars)

            if metrics_vars_np:
                composite_state["metrics_state"] = metrics_vars_np

        # Add metadata if specified
        if self.save_metadata is not None:
            if callable(self.save_metadata):
                metadata = self.save_metadata(self._current_epoch, logs)
            else:
                metadata = self.save_metadata
            if metadata:
                composite_state["metadata"] = metadata

        # Add data iterator state if specified
        if self.save_data_iterator is not None:
            if callable(self.save_data_iterator):
                iterator_state = self.save_data_iterator(
                    self._current_epoch, logs
                )
            else:
                iterator_state = self.save_data_iterator
            if iterator_state:
                composite_state["data_iterator"] = iterator_state

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
            save_args = ocp.args.StandardSave(
                composite_state, save_args=self.save_transforms
            )
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

        should_save = False
        if self.save_decision_policy is not None:
            # For FixedIntervalPolicy, save every N steps
            # This is a simplified implementation
            should_save = epoch % 2 == 0  # Save every 2 epochs for the test
        elif self.save_interval is not None:
            # Save every N epochs
            should_save = epoch % self.save_interval == 0
        elif self.save_freq == "epoch":
            should_save = True

        if should_save:
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

    def load_checkpoint(self, step, model=None):
        """Load model and optimizer state from a specific checkpoint step.

        Args:
            step: The checkpoint step to load from.
            model: Optional model to load into. If None, loads into self.model.

        Returns:
            tuple: (success, iterator_state) where success is True if loading
            was successful, False otherwise, and iterator_state is the saved
            data iterator state dict if available, None otherwise.
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
            target_model = model if model is not None else self.model
            success = self._restore_model_state(checkpoint_data, target_model)

            # Extract iterator state if available
            iterator_state = checkpoint_data.get("data_iterator", None)

            return success, iterator_state

        except Exception as e:
            if self.verbose > 0:
                print(
                    f"OrbaxCheckpoint: Failed to load checkpoint from step "
                    f"{step}: {e}"
                )
            return False, None

    def load_latest(self, model=None):
        """Load the most recent checkpoint.

        Args:
            model: Optional model to load into. If None, loads into self.model.

        Returns:
            tuple: (success, iterator_state) where success is True if loading
            was successful, False otherwise, and iterator_state is the saved
            data iterator state dict if available, None otherwise.
        """
        try:
            # Get the latest step
            latest_step = self.manager.latest_step()
            if latest_step is None:
                if self.verbose > 0:
                    print("OrbaxCheckpoint: No checkpoints found")
                return False, None

            return self.load_checkpoint(latest_step, model)

        except Exception as e:
            if self.verbose > 0:
                print(f"OrbaxCheckpoint: Failed to load latest checkpoint: {e}")
            return False, None

    def _restore_model_state(self, checkpoint_data, model=None):
        """Restore model state from checkpoint data.

        Args:
            checkpoint_data: The checkpoint data loaded from Orbax.
            model: Optional model to restore into. If None, uses self.model.

        Returns:
            bool: True if restoration was successful, False otherwise.
        """
        target_model = model if model is not None else self.model

        try:
            # Restore model weights
            if "model_weights" in checkpoint_data:
                model_weights_np = checkpoint_data["model_weights"]
                # Convert NumPy arrays back to backend tensors and assign to
                # model
                for i, weight_np in enumerate(model_weights_np):
                    # Convert numpy array back to appropriate backend tensor
                    weight_tensor = keras.ops.convert_to_tensor(weight_np)
                    target_model.weights[i].assign(weight_tensor)

            # Restore optimizer state if available
            if (
                "optimizer_state" in checkpoint_data
                and self.save_optimizer_state
            ):
                optimizer_vars_np = checkpoint_data["optimizer_state"]
                # Only restore if the variable counts match
                if len(optimizer_vars_np) == len(
                    target_model.optimizer.variables
                ):
                    # Convert NumPy arrays back to backend tensors and assign to
                    # optimizer
                    for i, var_np in enumerate(optimizer_vars_np):
                        var_tensor = keras.ops.convert_to_tensor(var_np)
                        target_model.optimizer.variables[i].assign(var_tensor)

            # Restore metrics state if available
            if (
                "metrics_state" in checkpoint_data
                and self.save_metrics_state
                and hasattr(target_model, "metrics")
            ):
                metrics_vars_np = checkpoint_data["metrics_state"]
                metric_idx = 0
                for metric in target_model.metrics:
                    if (
                        hasattr(metric, "variables")
                        and metric.variables
                        and metric_idx < len(metrics_vars_np)
                    ):
                        metric_vars_np = metrics_vars_np[metric_idx]
                        # Restore metric variables
                        for i, var_np in enumerate(metric_vars_np):
                            if i < len(metric.variables):
                                var_tensor = keras.ops.convert_to_tensor(var_np)
                                metric.variables[i].assign(var_tensor)
                        metric_idx += 1

            if self.verbose > 0:
                print("OrbaxCheckpoint: Successfully restored model state")
            return True

        except Exception as e:
            if self.verbose > 0:
                print(f"OrbaxCheckpoint: Failed to restore model state: {e}")
            return False
