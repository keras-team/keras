"""Orbax checkpoint loading functionality."""

import os

import numpy as np

from keras.src.utils.module_utils import ocp


def _load_model_from_orbax_checkpoint(
    filepath, custom_objects=None, compile=True, safe_mode=True
):
    """Load a model from an Orbax checkpoint."""
    from keras.src import backend
    from keras.src.models import model as model_module

    filepath = str(filepath)

    # Determine if this is a root directory or a step directory
    items = os.listdir(filepath)
    has_step_subdirs = any(
        os.path.isdir(os.path.join(filepath, item)) and item.isdigit()
        for item in items
    )

    if has_step_subdirs:
        # It's a root directory, find the latest checkpoint
        checkpoint_path = _find_latest_orbax_checkpoint(filepath)
    else:
        # It's a step directory, use it directly
        checkpoint_path = filepath

    # Load checkpoint
    loaded_state = ocp.load_pytree(checkpoint_path)

    if "model_config" not in loaded_state:
        raise ValueError(
            f"Orbax checkpoint at {filepath} does not contain model "
            "configuration. Cannot recreate model from checkpoint. This "
            "may happen when saving weights only."
        )

    # Recreate model from config
    model_config = loaded_state["model_config"]

    # Determine model type from config
    if "layers" in model_config:
        # Sequential model
        from keras.src.models import sequential as sequential_module

        model = sequential_module.Sequential.from_config(
            model_config, custom_objects=custom_objects
        )
    else:
        # Functional model
        model = model_module.Model.from_config(
            model_config, custom_objects=custom_objects
        )

    # Compile if requested and if the original model was compiled
    # (we can infer this from the presence of optimizer_variables)
    if compile and "optimizer_variables" in loaded_state:
        # Try to compile with default settings
        # This may not work if the model was compiled with custom settings
        try:
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        except Exception:
            # If compilation fails, leave the model uncompiled
            pass

    # Set the state in the model, but only for components that exist
    state_to_set = {}

    # Always load trainable and non-trainable variables
    if "trainable_variables" in loaded_state:
        state_to_set["trainable_variables"] = loaded_state[
            "trainable_variables"
        ]
    if "non_trainable_variables" in loaded_state:
        state_to_set["non_trainable_variables"] = loaded_state[
            "non_trainable_variables"
        ]

    # Only load optimizer state if the model has an optimizer
    if (
        "optimizer_variables" in loaded_state
        and hasattr(model, "optimizer")
        and model.optimizer is not None
    ):
        # Ensure optimizer variables are created by doing a dummy
        # apply_gradients. This creates the momentum/velocity
        # variables that are needed
        import numpy as np

        # Create zero gradients for all trainable variables
        zero_grads = [
            backend.convert_to_tensor(np.zeros_like(v.numpy()))
            for v in model.trainable_variables
        ]
        # Apply gradients to create optimizer slots
        model.optimizer.apply_gradients(
            zip(zero_grads, model.trainable_variables)
        )
        state_to_set["optimizer_variables"] = loaded_state[
            "optimizer_variables"
        ]

    # Only load metrics state if the model has metrics variables
    if (
        "metrics_variables" in loaded_state
        and hasattr(model, "metrics_variables")
        and model.metrics_variables
    ):
        state_to_set["metrics_variables"] = loaded_state["metrics_variables"]

    model.set_state_tree(state_to_set)

    # Load assets from state if present (new format)
    if "assets" in loaded_state:
        _load_assets_from_tree(model, loaded_state["assets"])

    # Load assets if they exist (fallback to old format)
    _load_orbax_assets(model, filepath)

    return model


def _load_assets_from_tree(model, assets_tree):
    """Load assets from a nested assets tree structure."""
    import base64
    import tempfile

    from keras.src.saving.keras_saveable import KerasSaveable
    from keras.src.saving.saving_lib import _walk_saveable

    def _get_nested_asset(tree, path):
        """Get asset dict from nested tree at the given path."""
        if not path:
            return None
        parts = path.split("/")
        current = tree
        for part in parts:
            if part in current:
                current = current[part]
            else:
                return None
        return (
            current
            if isinstance(current, dict)
            and not any(isinstance(v, dict) for v in current.values())
            else None
        )

    def _load_assets_recursive(saveable, current_tree, path=""):
        # Check if this saveable has assets at the current path
        if hasattr(saveable, "load_assets"):
            asset_dict = _get_nested_asset(current_tree, path)
            if asset_dict:
                # Create temporary directory and write files for load_assets
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Write asset files from base64-encoded strings
                    for rel_path, content in asset_dict.items():
                        file_path = os.path.join(temp_dir, rel_path)
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)

                        if isinstance(content, str):
                            # Try to decode as base64, if it fails, treat as
                            # raw content
                            try:
                                file_content = base64.b64decode(content)
                            except:
                                # Not base64, treat as raw string content
                                file_content = content.encode("utf-8")
                        elif isinstance(content, np.ndarray):
                            # For numpy arrays, save them as .npy files
                            np.save(file_path, content)
                            continue  # Skip the write below
                        else:
                            # Other types, convert to bytes
                            file_content = str(content).encode("utf-8")

                        with open(file_path, "wb") as f:
                            f.write(file_content)

                    # Call load_assets
                    saveable.load_assets(temp_dir)

        # Handle lists of KerasSaveable objects
        if isinstance(saveable, list):
            for i, item in enumerate(saveable):
                if isinstance(item, KerasSaveable):
                    item_path = f"{path}/layers/{i}" if path else f"layers/{i}"
                    _load_assets_recursive(item, current_tree, item_path)
            return

        # Only process KerasSaveable objects
        if not isinstance(saveable, KerasSaveable):
            return

        # Recursively walk through all child KerasSaveable objects
        for attr_name, child in _walk_saveable(saveable):
            child_path = f"{path}/{attr_name}" if path else attr_name
            if isinstance(child, KerasSaveable):
                _load_assets_recursive(child, current_tree, child_path)
            elif isinstance(child, list):
                # Handle lists of KerasSaveable objects
                for i, item in enumerate(child):
                    if isinstance(item, KerasSaveable):
                        item_path_full = f"{child_path}/{i}"
                        _load_assets_recursive(
                            item, current_tree, item_path_full
                        )

    _load_assets_recursive(model, assets_tree)


def _load_orbax_assets(model, checkpoint_dir):
    """Load assets from an Orbax checkpoint directory."""
    from keras.src.saving import saving_lib
    from keras.src.saving.saving_lib import _walk_saveable

    # For load_model, checkpoint_dir is the root directory
    # For load_weights, it might be a step directory
    assets_dir = None

    # Check for new format: checkpoint_dir/assets/step/
    assets_root = os.path.join(checkpoint_dir, "assets")
    if os.path.exists(assets_root):
        # Find the latest step in assets directory
        items = os.listdir(assets_root)
        step_dirs = [
            item
            for item in items
            if os.path.isdir(os.path.join(assets_root, item)) and item.isdigit()
        ]
        if step_dirs:
            latest_step = max(step_dirs, key=int)
            assets_dir = os.path.join(assets_root, latest_step)

    # Fallback to old format: checkpoint_dir/step/assets/
    if not assets_dir:
        items = os.listdir(checkpoint_dir)
        for item in items:
            step_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(step_path) and os.path.exists(
                os.path.join(step_path, "assets")
            ):
                assets_dir = os.path.join(step_path, "assets")
                break

    if assets_dir:
        assets_store = saving_lib.DiskIOStore(assets_dir, mode="r")
        try:
            visited = set()
            for child_attr, child_obj in _walk_saveable(model):
                if hasattr(child_obj, "load_assets"):
                    inner_path = child_attr.replace("\\", "/")
                    try:
                        child_obj.load_assets(assets_store.get(inner_path))
                    except KeyError:
                        # Asset not found, skip
                        pass
                    visited.add(id(child_obj))
        finally:
            assets_store.close()


def _is_orbax_checkpoint(filepath):
    """Check if the given path is an Orbax checkpoint directory."""
    if not os.path.exists(filepath):
        return False

    try:
        return ocp.is_orbax_checkpoint(filepath)
    except (ImportError, AttributeError):
        # Fallback to check for orbax.checkpoint file if Orbax API not available
        return os.path.isfile(os.path.join(filepath, "orbax.checkpoint"))


def _find_latest_orbax_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in an Orbax checkpoint directory."""
    checkpointer = ocp.training.Checkpointer(directory=checkpoint_dir)
    latest = checkpointer.latest
    if latest is None:
        raise ValueError(f"No valid checkpoints found in {checkpoint_dir}")
    return os.path.join(checkpoint_dir, str(latest.step))
