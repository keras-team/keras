import os
import warnings

import numpy as np

from keras.src import backend
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.callbacks.monitor_callback import (
    MonitorCallback,  # For metric monitoring logic
)
from keras.src.distribution.distribution_lib import process_id
from keras.src.utils.io_utils import print_msg
from keras.src.utils.module_utils import ocp


def _get_state_tree(model):
    """Get the complete model state as a nested tree structure."""
    state_tree = model.get_state_tree(value_format="numpy_array")

    # Convert numpy scalar types to Python types for Orbax compatibility
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


def _reconstruct_state_tree_with_values(structure, values):
    """Reconstruct state tree structure with provided values."""
    value_iter = iter(values)

    def _reconstruct_value(obj):
        value = next(value_iter)
        # Handle different cases for value conversion
        if isinstance(obj, np.generic):
            # obj is a numpy scalar (0-dimensional)
            if isinstance(value, (int, float)):
                # Convert Python scalar to numpy scalar
                return np.array(value, dtype=obj.dtype)
            elif isinstance(value, np.ndarray):
                # value is a numpy array, convert to scalar if needed
                if value.ndim == 0:
                    return np.array(value.item(), dtype=obj.dtype)
                elif value.ndim == 1 and value.size == 1:
                    return np.array(value.item(), dtype=obj.dtype)
                else:
                    return value.astype(obj.dtype).reshape(obj.shape)
            else:
                return np.array(value, dtype=obj.dtype)
        elif isinstance(obj, np.ndarray):
            # obj is a numpy array
            # Use backend-specific conversion that handles JAX arrays properly
            return backend.convert_checkpoint_value(value, obj.dtype, obj.shape)
        else:
            return value

    return tree.map_structure(_reconstruct_value, structure)


@keras_export("keras.callbacks.OrbaxCheckpoint")
class OrbaxCheckpoint(MonitorCallback):
    """Callback to save and load model state using Orbax with a similar API to
    ModelCheckpoint.

    This callback saves the model's weights and optimizer state asynchronously
    using Orbax, allowing training to continue without blocking for I/O.
    It also provides methods to load checkpoints for resuming training or
    inference.
    It supports policies for keeping checkpoints and deciding when to save.

    Example:

    ```python
    model.compile(loss=..., optimizer=...,
                  metrics=['accuracy'])

    EPOCHS = 10
    checkpoint_dir = '/tmp/ckpt'
    orbax_checkpoint_callback = keras.callbacks.OrbaxCheckpoint(
        directory=checkpoint_dir,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Model is saved at the end of every epoch, if it's the best seen so far.
    model.fit(epochs=EPOCHS, callbacks=[orbax_checkpoint_callback])

    # The model can be loaded from a specific checkpoint step as -
    checkpoint = keras.callbacks.OrbaxCheckpoint(directory=checkpoint_dir)
    checkpoint.load_checkpoint(step=5, model=model)  # Load from step 5

    # Alternatively, save checkpoints every N batches -
    orbax_checkpoint_callback = keras.callbacks.OrbaxCheckpoint(
        directory=checkpoint_dir,
        save_freq=100)  # Save every 100 batches

    model.fit(epochs=EPOCHS, callbacks=[orbax_checkpoint_callback])

    # Or use a SaveDecisionPolicy for more control -
    from orbax.checkpoint import checkpoint_managers
    policy = checkpoint_managers.FixedIntervalPolicy(interval=5)
    orbax_checkpoint_callback = keras.callbacks.OrbaxCheckpoint(
        directory=checkpoint_dir,
        save_decision_policy=policy)  # Save every 5 epochs

    model.fit(epochs=EPOCHS, callbacks=[orbax_checkpoint_callback])
    ```

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
            control when checkpoints are saved. Currently supports
            FixedIntervalPolicy for saving at regular intervals. If provided,
            overrides the default save frequency logic. Defaults to None.
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
        # Ensure orbax is available
        ocp.initialize()

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
        self._total_batches_seen = 0  # Global batch counter for step tracking

        if self.save_freq != "epoch" and not isinstance(self.save_freq, int):
            raise ValueError("Unrecognized save_freq")

        # Create should_save_fn from save_decision_policy or save_interval
        # if provided
        should_save_fn = None
        if save_decision_policy is not None:
            # When using save_decision_policy, let Orbax handle
            # should_save_fn internally
            # Don't override should_save_fn
            pass
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
            save_decision_policy=save_decision_policy,
        )
        # Ensure directory exists (only needed on one process in multi-host)
        if process_id() == 0:
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
            # Fallback: use global batch count
            return self._total_batches_seen

    def _save_checkpoint(self, step, logs=None):
        """Save a checkpoint at the given step."""
        if self.model is None:
            return

        # --- Prepare Composite State (Backend-Agnostic) ---
        state_tree = _get_state_tree(self.model)

        if state_tree is None:
            raise RuntimeError(
                "OrbaxCheckpoint: Failed to get model state tree. "
                "The model may not be properly built or may have no "
                "savable state."
            )

        # Save the nested state structures directly (preserving layer
        # names and structure)
        composite_state = {
            "trainable_variables": state_tree["trainable_variables"],
        }

        if self.save_optimizer_state and "optimizer_variables" in state_tree:
            composite_state["optimizer_variables"] = state_tree[
                "optimizer_variables"
            ]

        if self.save_metrics_state and "metrics_variables" in state_tree:
            composite_state["metrics_variables"] = state_tree[
                "metrics_variables"
            ]

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
        # Only save on the primary process (rank 0) in distributed setups
        is_primary_host = process_id() == 0

        if is_primary_host:
            if self.verbose > 0:
                print_msg(
                    f"OrbaxCheckpoint: Triggering async save for step {step}..."
                )

            # Save the checkpoint
            save_args = ocp.args.StandardSave(
                composite_state, save_args=self.save_transforms
            )
            self.manager.save(step, args=save_args)

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
                # Use step number (e.g., optimizer iterations) for Orbax save
                # step
                step = self._get_current_step()
                self._save_checkpoint(step=step, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self._current_epoch = epoch
        if self.monitor_op is None:
            self._set_monitor_op()  # From MonitorCallback

        should_save = False
        if self.save_decision_policy is not None:
            # Handle FixedIntervalPolicy by extracting its interval
            from orbax.checkpoint import checkpoint_managers

            if isinstance(
                self.save_decision_policy,
                checkpoint_managers.FixedIntervalPolicy,
            ):
                should_save = epoch % self.save_decision_policy.interval == 0
            else:
                # For other policies, fall back to saving every epoch
                # TODO: Implement full support for other SaveDecisionPolicy
                # types
                should_save = True
        elif self.save_interval is not None:
            # Save every N epochs
            should_save = epoch % self.save_interval == 0
        elif self.save_freq == "epoch":
            should_save = True

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
            self._save_checkpoint(step=epoch, logs=logs)

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print_msg("OrbaxCheckpoint: Waiting for final saves to complete...")
        self.manager.wait_until_finished()
        if self.verbose > 0:
            print_msg("OrbaxCheckpoint: All saves finalized.")

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
        if process_id() != 0:
            return True  # Return True to indicate no error, but no loading

        if self.verbose > 0:
            print_msg(
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

    def load_latest(self, model=None):
        """Load the most recent checkpoint.

        Args:
            model: Optional model to load into. If None, loads into self.model.

        Returns:
            tuple: (success, iterator_state) where success is True if loading
            was successful, False otherwise, and iterator_state is the saved
            data iterator state dict if available, None otherwise.
        """
        # Get the latest step
        latest_step = self.manager.latest_step()
        if latest_step is None:
            raise FileNotFoundError("OrbaxCheckpoint: No checkpoints found")

        return self.load_checkpoint(latest_step, model)

    def _restore_model_state(self, checkpoint_data, model=None):
        """Restore model state from checkpoint data.

        Args:
            checkpoint_data: The checkpoint data loaded from Orbax.
            model: Optional model to restore into. If None, uses self.model.

        Returns:
            bool: True if restoration was successful.
        """
        target_model = model if model is not None else self.model

        # Check if this is the new nested structure format
        if "trainable_variables" in checkpoint_data and isinstance(
            checkpoint_data["trainable_variables"], dict
        ):
            # New format: nested structures
            return self._restore_from_nested_structures(
                checkpoint_data, target_model
            )
        elif "model_weights" in checkpoint_data and isinstance(
            checkpoint_data["model_weights"], list
        ):
            # Old format: flattened values (for backward compatibility)
            return self._restore_from_flattened_values(
                checkpoint_data, target_model
            )
        elif "model_state" in checkpoint_data:
            # Old format: full state tree (for backward compatibility)
            return self._restore_from_state_tree(
                checkpoint_data["model_state"], target_model
            )
        else:
            # Unsupported checkpoint format
            return False

    def _restore_from_nested_structures(self, checkpoint_data, target_model):
        """Restore from the new nested structures format."""
        # Ensure the target model is built so it has variables
        if len(target_model.trainable_variables) == 0:
            try:
                # Try to build the model by doing a dummy forward pass
                if (
                    hasattr(target_model, "input_shape")
                    and target_model.input_shape is not None
                ):
                    dummy_input_shape = target_model.input_shape
                    if dummy_input_shape[0] is None:  # Batch dimension is None
                        dummy_input = np.zeros((1,) + dummy_input_shape[1:])
                    else:
                        dummy_input = np.zeros(dummy_input_shape)
                    target_model(dummy_input)
            except Exception:
                # If dummy forward pass fails, try build
                try:
                    if (
                        hasattr(target_model, "input_shape")
                        and target_model.input_shape is not None
                    ):
                        build_shape = target_model.input_shape
                        if (
                            isinstance(build_shape, (list, tuple))
                            and len(build_shape) > 1
                            and build_shape[0] is None
                        ):
                            build_shape = build_shape[1:]
                        target_model.build(build_shape)
                except Exception:
                    # If building fails, continue anyway
                    pass

        # Prepare the state tree to restore
        reconstructed_state = {}

        # Restore trainable variables
        if "trainable_variables" in checkpoint_data:
            reconstructed_state["trainable_variables"] = checkpoint_data[
                "trainable_variables"
            ]

        # Restore optimizer variables if available and model has optimizer
        if (
            "optimizer_variables" in checkpoint_data
            and self.save_optimizer_state
            and hasattr(target_model, "optimizer")
            and target_model.optimizer is not None
        ):
            reconstructed_state["optimizer_variables"] = checkpoint_data[
                "optimizer_variables"
            ]

        # Restore metrics variables if available
        if "metrics_variables" in checkpoint_data and self.save_metrics_state:
            reconstructed_state["metrics_variables"] = checkpoint_data[
                "metrics_variables"
            ]

        # Use set_state_tree to restore the state
        target_model.set_state_tree(reconstructed_state)

        if self.verbose > 0:
            print_msg("OrbaxCheckpoint: Successfully restored model state")
        return True

    def _restore_from_flattened_values(self, checkpoint_data, target_model):
        """Restore from the new flattened values format."""
        # Ensure the target model is built so it has variables
        if len(target_model.trainable_variables) == 0:
            try:
                # Try to build the model by doing a dummy forward pass
                if (
                    hasattr(target_model, "input_shape")
                    and target_model.input_shape is not None
                ):
                    dummy_input_shape = target_model.input_shape
                    if dummy_input_shape[0] is None:  # Batch dimension is None
                        dummy_input = np.zeros((1,) + dummy_input_shape[1:])
                    else:
                        dummy_input = np.zeros(dummy_input_shape)
                    target_model(dummy_input)
            except Exception:
                # If dummy forward pass fails, try build
                try:
                    if (
                        hasattr(target_model, "input_shape")
                        and target_model.input_shape is not None
                    ):
                        build_shape = target_model.input_shape
                        if (
                            isinstance(build_shape, (list, tuple))
                            and len(build_shape) > 1
                            and build_shape[0] is None
                        ):
                            build_shape = build_shape[1:]
                        target_model.build(build_shape)
                except Exception:
                    # If building fails, continue anyway
                    pass

        # Get the target model's state tree structure (without convert_scalars)
        target_state_tree = target_model.get_state_tree(
            value_format="numpy_array"
        )
        if target_state_tree is None:
            if self.verbose > 0:
                print_msg(
                    "OrbaxCheckpoint: Could not get target model state tree"
                )
            return False

        # Reconstruct state tree with saved values
        reconstructed_state = {}

        # Restore trainable variables
        if "model_weights" in checkpoint_data:
            saved_trainable_values = checkpoint_data["model_weights"]
            target_trainable_structure = target_state_tree[
                "trainable_variables"
            ]
            reconstructed_state["trainable_variables"] = (
                _reconstruct_state_tree_with_values(
                    target_trainable_structure, saved_trainable_values
                )
            )

        # Restore optimizer variables if available
        if (
            "optimizer_state" in checkpoint_data
            and self.save_optimizer_state
            and "optimizer_variables" in target_state_tree
        ):
            saved_optimizer_values = checkpoint_data["optimizer_state"]
            target_optimizer_structure = target_state_tree[
                "optimizer_variables"
            ]
            reconstructed_state["optimizer_variables"] = (
                _reconstruct_state_tree_with_values(
                    target_optimizer_structure, saved_optimizer_values
                )
            )

        # Restore metrics variables if available
        if (
            "metrics_variables" in checkpoint_data
            and self.save_metrics_state
            and "metrics_variables" in target_state_tree
        ):
            saved_metrics_values = checkpoint_data["metrics_variables"]
            target_metrics_structure = target_state_tree["metrics_variables"]
            reconstructed_state["metrics_variables"] = (
                _reconstruct_state_tree_with_values(
                    target_metrics_structure, saved_metrics_values
                )
            )

        # Use set_state_tree to restore the reconstructed state
        target_model.set_state_tree(reconstructed_state)

        if self.verbose > 0:
            print_msg("OrbaxCheckpoint: Successfully restored model state")
        return True

    def _restore_from_state_tree(self, state_tree, target_model):
        """Restore from the old full state tree format
        (for backward compatibility)."""
        target_model.set_state_tree(state_tree)
        if self.verbose > 0:
            print_msg("OrbaxCheckpoint: Successfully restored model state")
        return True


# Export additional Orbax functionality for advanced users (only if available)
if ocp.available:
    CheckpointManager = ocp.CheckpointManager
    PyTreeCheckpointer = ocp.PyTreeCheckpointer
    SaveArgs = ocp.SaveArgs
    StandardRestore = ocp.args.StandardRestore
    TypeHandler = ocp.type_handlers.TypeHandler
    metadata = ocp.metadata
    register_type_handler = ocp.type_handlers.register_type_handler
