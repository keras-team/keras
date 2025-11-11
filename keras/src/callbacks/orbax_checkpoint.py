import warnings

import numpy as np

from keras.src import backend
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.callbacks.monitor_callback import (
    MonitorCallback,  # For metric monitoring logic
)
from keras.src.utils.io_utils import print_msg
from keras.src.utils.module_utils import ocp

# Context and AsyncOptions are accessed through the lazy-loaded ocp module


def _get_state_tree(model):
    """Get the complete model state as a nested tree structure."""
    # For JAX backend, preserve native arrays if JAX monitoring available
    # to avoid unnecessary conversions. Otherwise convert to numpy.
    if backend.backend() == "jax":
        try:
            import jax

            # Check if jax.monitoring.record_scalar exists (JAX 0.7.0+)
            jax.monitoring.record_scalar
            state_tree = model.get_state_tree()
        except (ImportError, AttributeError):
            # Fallback to numpy conversion for older JAX versions
            state_tree = model.get_state_tree(value_format="numpy_array")
    else:
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
        post_finalization_callback: Callable, function to call after async
            checkpointing operations complete. Defaults to None.
        save_decision_policy: orbax.checkpoint.SaveDecisionPolicy object to
            control when checkpoints are saved. If None, defaults to saving
            every epoch for save_freq="epoch" or every save_freq batches.
            Defaults to None.
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
        post_finalization_callback=None,
        save_decision_policy=None,
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
        self.save_on_background = save_on_background
        self.save_metadata = save_metadata
        self.save_data_iterator = save_data_iterator
        self.save_metrics_state = save_metrics_state
        self.post_finalization_callback = post_finalization_callback
        self.save_decision_policy = save_decision_policy
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self._current_epoch = 0  # Keep track of epoch
        self._total_batches_seen = 0  # Global batch counter for step tracking

        if self.save_freq != "epoch" and not isinstance(self.save_freq, int):
            raise ValueError("Unrecognized save_freq")

        # Set up save_decision_policy if not provided
        if save_decision_policy is None:
            # Let Keras handle all save decisions - configure Checkpointer
            # to save unconditionally when save_pytree/save_pytree_async
            # is called
            class _AlwaysSavePolicy(
                ocp.training.save_decision_policies.SaveDecisionPolicy
            ):
                def should_save(
                    self, current_step_info, previous_steps=None, context=None
                ):
                    return True

            save_decision_policy = _AlwaysSavePolicy()

        # --- Orbax Checkpointer Setup (V1 API) ---
        # Map V0 options to V1 parameters
        policies = []
        if keep_period is not None:
            policies.append(
                ocp.training.preservation_policies.EveryNSteps(keep_period)
            )
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
            save_decision_policy=save_decision_policy,
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

        # --- Save Logic (V1 API) ---
        # All processes participate in distributed checkpointing
        # Checkpointer is configured to save unconditionally when
        # save_pytree is called
        if self.verbose > 0:
            print_msg(
                f"OrbaxCheckpoint: Triggering async save for step {step}..."
            )

        # Configure context if a callback is provided
        context_options = {}
        async_options = {}

        if self.post_finalization_callback is not None:
            async_options["post_finalization_callback"] = (
                self.post_finalization_callback
            )

        if async_options:
            context_options["async_options"] = ocp.options.AsyncOptions(
                **async_options
            )

        # Use a single with statement. If context_options is empty,
        # Context() uses defaults.
        with ocp.Context(**context_options):
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
                # Use step number (e.g., optimizer iterations) for Orbax save
                # step
                step = self._get_current_step()
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
        if self.verbose > 0:
            print_msg("OrbaxCheckpoint: Training completed.")

        # Close the Checkpointer to ensure all pending saves complete
        try:
            self.checkpointer.close()
        except Exception:
            pass  # Ignore errors during cleanup

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
        # All processes participate in distributed checkpoint loading
        if self.verbose > 0:
            print_msg(
                f"OrbaxCheckpoint: Loading checkpoint from step {step}..."
            )

        # Load the checkpoint using V1 API
        checkpoint_data = self.checkpointer.load_pytree(step)

        # Extract model state (exclude metadata and data_iterator)
        model_state = {}
        iterator_state = None

        for key, value in checkpoint_data.items():
            if key == "data_iterator":
                iterator_state = value
            elif key == "metadata":
                pass  # Metadata is not used in loading
            else:
                # This is model state (trainable_variables, optimizer_variables,
                # etc.)
                model_state[key] = value

        # Restore the model state
        target_model = model if model is not None else self.model
        success = self._restore_model_state_from_full_tree(
            model_state, target_model
        )

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
        # Wait for any in-progress saves to complete
        self.wait_until_finished()

        # Get the latest step using V1 API
        latest_metadata = self.checkpointer.latest
        if latest_metadata is None:
            raise FileNotFoundError("OrbaxCheckpoint: No checkpoints found")

        return self.load_checkpoint(latest_metadata.step, model)

    def all_steps(self):
        """Get all available checkpoint steps.

        Returns:
            list: List of available checkpoint step numbers, sorted.
        """
        return sorted([int(cp.step) for cp in self.checkpointer.checkpoints])

    def wait_until_finished(self):
        """Wait for any in-progress checkpoint operations to complete.

        This method blocks until all asynchronous checkpoint save operations
        have completed. It should be called before attempting to load
        checkpoints if there might be pending save operations.
        """
        # Wait for any async operations to complete
        while self.checkpointer.is_saving_in_progress():
            import time

            time.sleep(0.1)

    def _restore_model_state_from_full_tree(self, state_tree, model=None):
        """Restore model state from full state tree (V1 format)."""
        target_model = model if model is not None else self.model
        target_model.set_state_tree(state_tree)
        if self.verbose > 0:
            print_msg("OrbaxCheckpoint: Successfully restored model state")
        return True


# Export additional Orbax functionality for advanced users (only if available)
if ocp.available:
    Checkpointer = ocp.training.Checkpointer
    save_pytree = ocp.save_pytree
    load_pytree = ocp.load_pytree
    save_pytree_async = ocp.save_pytree_async
    load_pytree_async = ocp.load_pytree_async
    preservation_policies = ocp.training.preservation_policies
    save_decision_policies = ocp.training.save_decision_policies
