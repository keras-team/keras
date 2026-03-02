"""Orbax checkpoint loading functionality."""

import os

from keras.src.utils import file_utils
from keras.src.utils.module_utils import ocp


def is_orbax_checkpoint(filepath):
    """Check if the given path is an Orbax checkpoint directory.

    This function implements custom detection logic instead of relying on
    Orbax APIs which may be unreliable in some environments.
    """
    if not file_utils.exists(filepath) or not file_utils.isdir(filepath):
        return False

    try:
        # List directory contents
        contents = file_utils.listdir(filepath)

        # A set is more efficient for membership testing
        orbax_indicators = {
            "orbax.checkpoint",
            "pytree.orbax-checkpoint",
            "checkpoint_metadata",
        }

        # Fast check for standard files
        if not orbax_indicators.isdisjoint(contents):
            return True

        # Check for step directories or temporary files in a single pass
        return any(
            ".orbax-checkpoint-tmp" in item
            or (
                item.isdigit()
                and file_utils.isdir(file_utils.join(filepath, item))
            )
            for item in contents
        )

    except (OSError, PermissionError):
        # If we can't read the directory, assume it's not a checkpoint
        return False


def find_latest_orbax_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in an Orbax checkpoint directory."""
    checkpointer = ocp.training.Checkpointer(directory=checkpoint_dir)
    latest = checkpointer.latest
    if latest is None:
        raise ValueError(f"No valid checkpoints found in {checkpoint_dir}")
    return os.path.join(checkpoint_dir, str(latest.step))
