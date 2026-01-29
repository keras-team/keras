import dataclasses
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from keras.src import backend
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.callbacks.monitor_callback import (
    MonitorCallback,  # For metric monitoring logic
)
from keras.src.saving.saving_lib import _walk_saveable
from keras.src.utils.module_utils import ocp

# JAX monitoring compatibility: ensure record_scalar exists
# to prevent AttributeError in older JAX versions
try:
    import jax

    if not hasattr(jax.monitoring, "record_scalar"):
        jax.monitoring.record_scalar = lambda *args, **kwargs: None
except ImportError:
    pass


# ============================================================================
# Asset Handler Implementation
# ============================================================================


class KerasAssetHandler:
    """Custom handler for Keras model assets."""

    def __init__(self, primary_host_only: bool = True):
        self._primary_host_only = primary_host_only
        self._asset_metadata = {}

    @classmethod
    def typestr(cls):
        return "KerasAssetHandler"

    def _is_primary_host(self):
        """Check if this is the primary host in distributed training."""
        if not self._primary_host_only:
            return True
        try:
            if backend.backend() == "jax":
                import jax

                return jax.process_index() == 0
            elif backend.backend() == "tensorflow":
                import tensorflow as tf

                try:
                    strategy = tf.distribute.get_strategy()
                    return (
                        not hasattr(strategy.extended, "_in_multi_worker_mode")
                        or not strategy.extended._in_multi_worker_mode()
                        or strategy.extended._task_id == 0
                    )
                except Exception:
                    return True
            elif backend.backend() == "torch":
                import torch

                return (
                    not torch.distributed.is_initialized()
                    or torch.distributed.get_rank() == 0
                )
        except Exception:
            pass
        return True

    def _collect_layers_with_assets(self, model):
        """Recursively collect all layers with assets."""
        from keras.src.saving.keras_saveable import KerasSaveable
        from keras.src.utils import naming

        layers_with_assets = []
        visited_saveables = set()

        def _collect_from_container(container, path):
            if isinstance(container, dict):
                container = list(container.values())

            used_names = {}
            for item in container:
                if isinstance(item, KerasSaveable):
                    name = (
                        item.name
                        if hasattr(item, "name") and item.name
                        else naming.to_snake_case(item.__class__.__name__)
                    )
                    if name in used_names:
                        used_names[name] += 1
                        name = f"{name}_{used_names[name]}"
                    else:
                        used_names[name] = 0
                    collect_from_layer(item, f"{path}/{name}")

        def collect_from_layer(layer, path):
            if id(layer) in visited_saveables:
                return
            visited_saveables.add(id(layer))

            if hasattr(layer, "assets") and callable(layer.assets):
                assets_list = layer.assets()
                if assets_list:
                    layers_with_assets.append((path, layer))

            for name, sublayer in _walk_saveable(layer):
                sublayer_path = f"{path}/{name}"
                if isinstance(sublayer, KerasSaveable):
                    collect_from_layer(sublayer, sublayer_path)
                elif isinstance(sublayer, (list, dict, tuple, set)):
                    _collect_from_container(sublayer, sublayer_path)

        collect_from_layer(model, model.name)
        return layers_with_assets

    def save(self, directory, args):
        """Save model assets to the directory."""
        if not self._is_primary_host():
            return []

        model = args.model
        if model is None:
            return []

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if args.save_model_config:
            from keras.src.saving import saving_lib

            config_json, _ = saving_lib._serialize_model_as_json(model)
            (directory / "model_config.json").write_text(config_json)

        layers_with_assets = self._collect_layers_with_assets(model)

        for layer_path, layer in layers_with_assets:
            layer_assets_dir = directory / layer_path
            layer_assets_dir.mkdir(parents=True, exist_ok=True)

            for i, asset in enumerate(layer.assets()):
                asset_data = (
                    np.array(asset)
                    if hasattr(asset, "numpy")
                    or not isinstance(asset, np.ndarray)
                    else asset
                )
                np.save(
                    layer_assets_dir / f"asset_{i}.npy",
                    asset_data,
                    allow_pickle=False,
                )

        self._asset_metadata[str(directory)] = {
            "layers_with_assets": [path for path, _ in layers_with_assets]
        }

        return []

    def restore(self, directory, args):
        """Restore model assets from the directory."""
        model = args.model
        if model is None or not (directory := Path(directory)).exists():
            return model

        layers_with_assets = self._collect_layers_with_assets(model)

        for layer_path, layer in layers_with_assets:
            layer_assets_dir = directory / layer_path
            if layer_assets_dir.exists():
                asset_files = sorted(layer_assets_dir.glob("asset_*.npy"))
                if asset_files and hasattr(layer, "_set_assets"):
                    layer._set_assets([np.load(f) for f in asset_files])

        return model

    def metadata(self, directory):
        return self._asset_metadata.get(str(directory), {})

    def finalize(self, directory):
        pass

    def close(self):
        pass


# Placeholder for AssetArgs - will be properly initialized
# when OrbaxCheckpoint is instantiated
AssetArgs = None


def _get_state_tree(model):
    """Get the complete model state as a nested tree structure."""
    if backend.backend() == "jax":
        return model.get_state_tree()

    state_tree = model.get_state_tree(value_format="numpy_array")

    def convert_scalars(obj):
        if isinstance(obj, (np.ndarray, np.generic)) and (
            not isinstance(obj, np.ndarray) or obj.ndim == 0
        ):
            return obj.item()
        return obj

    return tree.map_structure(convert_scalars, state_tree)


@keras_export("keras.callbacks.OrbaxCheckpoint")
class OrbaxCheckpoint(MonitorCallback):
    """Callback to save and load model state using Orbax with a similar API to
    ModelCheckpoint.

    This callback saves the model's weights and optimizer state asynchronously
    using Orbax, allowing training to continue without blocking for I/O.

    **Multi-host Support**: When running in a multi-host distributed training
    environment with JAX backend, this callback automatically coordinates
    checkpointing across all hosts to ensure consistency and proper
    synchronization. Multi-host checkpointing is only supported on JAX.

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
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
        max_to_keep=1,
        save_on_background=True,
        save_weights_only=False,
    ):
        # Ensure orbax is available
        ocp.initialize()

        # Initialize AssetArgs on first use
        global AssetArgs
        if AssetArgs is None:

            @ocp.register_with_handler(
                KerasAssetHandler, for_save=True, for_restore=True
            )
            @dataclasses.dataclass
            class AssetArgs(ocp.CheckpointArgs):
                """Arguments for asset checkpointing.

                Attributes:
                    model: The Keras model to save/restore assets for.
                    save_model_config: Whether to save the model configuration.
                """

                model: Any = None
                save_model_config: bool = True

        # Initialize MonitorCallback for metric monitoring
        super().__init__(monitor, mode, initial_value_threshold)

        self.directory = directory
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.max_to_keep = max_to_keep
        self.save_on_background = save_on_background
        self.save_weights_only = save_weights_only

        # Tracking for batch-level saving
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self._current_epoch = 0
        self._total_batches_seen = 0

        # Validate save_freq
        if self.save_freq != "epoch" and not isinstance(self.save_freq, int):
            raise ValueError(
                f"Unrecognized save_freq: {self.save_freq}. "
                "Expected 'epoch' or integer."
            )

        # Multi-host detection
        self._multihost_initialized = self._is_multihost_initialized()

        # Create checkpoint handlers
        state_handler = ocp.StandardCheckpointHandler()
        asset_handler = KerasAssetHandler()

        asset_checkpointer = ocp.Checkpointer(asset_handler)
        checkpointer_class = (
            ocp.AsyncCheckpointer if save_on_background else ocp.Checkpointer
        )
        state_checkpointer = checkpointer_class(state_handler)

        # Use CheckpointManager with named checkpointers dict
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True,
            save_interval_steps=1,
        )

        self.manager = ocp.CheckpointManager(
            directory,
            {"state": state_checkpointer, "assets": asset_checkpointer},
            options,
        )

    def _is_multihost_initialized(self):
        """Check if multi-host environment is initialized."""
        # Multi-host checkpointing is only supported on JAX backend
        if backend.backend() != "jax":
            return False

        multihost = ocp.multihost
        # Check if JAX distributed client is initialized
        # (indicates multihost setup)
        return multihost.is_jax_distributed_client_initialized()

    def _sync_processes(self, key=None):
        """Synchronize all processes across hosts."""
        if not self._multihost_initialized:
            return  # No-op for single host

        multihost = ocp.multihost
        sync_key = key or "orbax_checkpoint_sync"
        multihost.sync_global_processes(sync_key)

    def is_multihost_enabled(self):
        """Return True if multi-host checkpointing is enabled and initialized.

        This method can be used to check if the callback is operating in
        a multi-host distributed training environment. Multi-host checkpointing
        is only supported on JAX backend.

        Returns:
            bool: True if multi-host support is active, False otherwise.
        """
        return self._multihost_initialized

    def is_primary_host(self):
        """Return True if this process is the primary host in multi-host setup.

        In multi-host environments, only the primary host typically handles
        logging and coordination tasks. Multi-host checkpointing is only
        supported on JAX backend.

        Returns:
            bool: True if this is the primary host, False otherwise.
            Always returns True in single-host environments.
        """
        if not self._multihost_initialized:
            return True  # Single host is always primary
        multihost = ocp.multihost
        return multihost.is_primary_host()

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
        """Save checkpoint with state and assets."""
        # Access checkpoint args through ocp LazyModule
        CompositeArgs = ocp.CompositeArgs
        StandardSaveArgs = ocp.StandardSaveArgs

        # Get model state tree
        state_tree = _get_state_tree(self.model)

        # Prepare state dictionary
        if self.save_weights_only:
            composite_state = {
                "trainable_variables": state_tree["trainable_variables"],
                "non_trainable_variables": (
                    state_tree["non_trainable_variables"]
                ),
            }
        else:
            composite_state = state_tree

        # Save using CheckpointManager with Composite args
        composite_args = CompositeArgs(
            state=StandardSaveArgs(composite_state),
            assets=AssetArgs(
                model=self.model, save_model_config=not self.save_weights_only
            ),
        )

        self.manager.save(step, args=composite_args)

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
            self._save_checkpoint(step=epoch, logs=logs)

    def on_train_end(self, logs=None):
        # Close the Checkpointer - this waits for any pending async saves
        # to complete before closing
        try:
            self.manager.close()
        except Exception:
            pass  # Ignore errors during cleanup

        # Multi-host synchronization: ensure all hosts complete cleanup
        self._sync_processes("checkpoint_cleanup")

    def wait_until_finished(self):
        """Wait for any in-progress checkpoint operations to complete.
        This method blocks until all asynchronous checkpoint save
        operations have completed across all hosts in a multi-host
        setup.
        """
        # Wait for any remaining async operations to complete
        self.manager.wait_until_finished()

        # Multi-host synchronization: ensure all hosts complete
        self._sync_processes("checkpoint_wait_complete")
