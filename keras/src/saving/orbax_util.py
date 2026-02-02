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
            or (item.isdigit() and os.path.isdir(os.path.join(filepath, item)))
            for item in contents
        )

    except (OSError, PermissionError):
        # If we can't read the directory, assume it's not a checkpoint
        return False


def find_latest_orbax_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in an Orbax checkpoint directory.

    Args:
        checkpoint_dir: Path to either a root checkpoint directory (containing
            numbered step subdirectories) or a specific step directory.

    Returns:
        Path to the step directory. If checkpoint_dir is already a step
        directory, returns it as-is. If it's a root directory, returns
        the path to the latest step.
    """
    # Check if this is already a step directory or a root directory
    items = os.listdir(checkpoint_dir)
    has_step_subdirs = any(
        os.path.isdir(os.path.join(checkpoint_dir, item)) and item.isdigit()
        for item in items
    )

    if not has_step_subdirs:
        # It's already a step directory, return it as-is
        return checkpoint_dir

    # It's a root directory, find the latest checkpoint
    checkpointer = ocp.training.Checkpointer(directory=checkpoint_dir)
    latest = checkpointer.latest
    if latest is None:
        raise ValueError(f"No valid checkpoints found in {checkpoint_dir}")
    return os.path.join(checkpoint_dir, str(latest.step))
