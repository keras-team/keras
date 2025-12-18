"""Orbax checkpoint loading functionality."""

import os

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

    return model


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
