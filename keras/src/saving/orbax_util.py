"""Orbax checkpoint loading functionality."""

import os

from keras.src.utils.module_utils import ocp


def is_orbax_checkpoint(filepath):
    """Check if the given path is an Orbax checkpoint directory."""
    if not os.path.exists(filepath):
        return False

    try:
        return ocp.is_orbax_checkpoint(filepath)
    except (ImportError, AttributeError):
        # Fallback to check for orbax.checkpoint file if Orbax API not available
        return os.path.isfile(os.path.join(filepath, "orbax.checkpoint"))


def find_latest_orbax_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in an Orbax checkpoint directory."""
    checkpointer = ocp.training.Checkpointer(directory=checkpoint_dir)
    latest = checkpointer.latest
    if latest is None:
        raise ValueError(f"No valid checkpoints found in {checkpoint_dir}")
    return os.path.join(checkpoint_dir, str(latest.step))
