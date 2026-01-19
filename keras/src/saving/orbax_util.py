"""Orbax checkpoint loading functionality."""

import os

from keras.src.utils.module_utils import ocp


def is_orbax_checkpoint(filepath):
    """Check if the given path is an Orbax checkpoint directory."""
    if not os.path.exists(filepath):
        return False

    # Check for orbax.checkpoint file or step subdirectories
    if os.path.isfile(os.path.join(filepath, "orbax.checkpoint")):
        return True

    # Check if it has step subdirectories (digit-named directories)
    try:
        items = os.listdir(filepath)
        has_step_subdirs = any(
            os.path.isdir(os.path.join(filepath, item)) and item.isdigit()
            for item in items
        )
        if has_step_subdirs:
            return True
    except OSError:
        pass

    return False


def find_latest_orbax_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in an Orbax checkpoint directory."""
    checkpointer = ocp.training.Checkpointer(directory=checkpoint_dir)
    latest = checkpointer.latest
    if latest is None:
        raise ValueError(f"No valid checkpoints found in {checkpoint_dir}")
    return os.path.join(checkpoint_dir, str(latest.step))
