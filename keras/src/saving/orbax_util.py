"""Orbax checkpoint loading functionality."""

import os

from keras.src.utils.module_utils import ocp


def is_orbax_checkpoint(filepath):
    """Check if the given path is an Orbax checkpoint directory.

    This function implements custom detection logic instead of relying on
    Orbax APIs which may be unreliable in some environments.
    """
    if not os.path.exists(filepath) or not os.path.isdir(filepath):
        return False

    try:
        # List directory contents
        contents = os.listdir(filepath)

        # Check for common Orbax checkpoint indicators
        orbax_indicators = [
            # Standard Orbax checkpoint files
            "orbax.checkpoint",
            "pytree.orbax-checkpoint",
            "checkpoint_metadata",
            # Checkpoint step directories (numeric names)
        ]

        has_orbax_files = any(
            indicator in contents for indicator in orbax_indicators
        )

        # Check for numeric step directories (common in Orbax)
        has_step_dirs = any(
            os.path.isdir(os.path.join(filepath, item)) and item.isdigit()
            for item in contents
        )

        # Check for temporary checkpoint directories
        has_tmp_dirs = any(".orbax-checkpoint-tmp" in item for item in contents)

        # A directory is considered an Orbax checkpoint if it has:
        # 1. Orbax-specific files, OR
        # 2. Numeric step directories, OR
        # 3. Temporary checkpoint directories
        return has_orbax_files or has_step_dirs or has_tmp_dirs

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
