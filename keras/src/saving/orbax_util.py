"""Orbax checkpoint loading functionality."""

import os

from keras.src import backend
from keras.src.distribution import distribution as get_distribution
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


def build_orbax_abstract_pytree(checkpoint_path, ref_state=None):
    """Build an abstract pytree for Orbax loading with target shardings.

    On JAX with an active distribution, returns a pytree of
    ``jax.ShapeDtypeStruct`` so that Orbax reshards arrays onto the
    current distribution layout instead of restoring saved shardings.
    On all other backends, or when no distribution is active, returns
    ``None`` (Orbax will use saved shardings — fine when the topology
    hasn't changed).

    Args:
        checkpoint_path: Path to a specific Orbax checkpoint step
            directory (e.g. ``<root>/2``).
        ref_state: Optional reference state tree (from
            ``model.get_state_tree()``) whose variables carry the
            target shardings. If ``None``, shardings default to
            ``None`` per leaf (Orbax uses saved shardings).

    Returns:
        A pytree of ``jax.ShapeDtypeStruct`` matching the checkpoint
        structure, or ``None`` when resharding is not needed.
    """
    if backend.backend() != "jax":
        return None

    if get_distribution() is None:
        return None

    import jax

    pytree_meta = ocp.pytree_metadata(checkpoint_path).metadata

    # ---- collect variable → sharding from reference state ----
    def _collect_shardings(tree, path=()):
        """Flatten a nested dict tree into {keypath: sharding}."""
        if hasattr(tree, "sharding"):
            return {path: tree.sharding}
        if isinstance(tree, dict):
            out = {}
            for k, v in tree.items():
                out.update(_collect_shardings(v, path + (k,)))
            return out
        return {}

    ref_shardings = {}
    if ref_state:
        for group in (
            "trainable_variables",
            "non_trainable_variables",
        ):
            ref_shardings.update(_collect_shardings(ref_state.get(group, {})))

    # ---- recursively build the abstract tree ----
    def _to_abstract(meta, ref=None, path=()):
        """Convert metadata leaf → jax.ShapeDtypeStruct with sharding."""
        if hasattr(meta, "shape") and hasattr(meta, "dtype"):
            # Array leaf: use ref sharding when available, otherwise
            # try matching an optimizer slot path to a model variable
            # path (strip the optimizer-name prefix and slot-name
            # suffix: path[1:-1]).
            sharding = getattr(ref, "sharding", None)
            if sharding is None and len(path) >= 3:
                sharding = ref_shardings.get(path[1:-1])
            return jax.ShapeDtypeStruct(
                meta.shape, meta.dtype, sharding=sharding
            )
        if isinstance(meta, dict):
            r = ref if isinstance(ref, dict) else {}
            return {
                k: _to_abstract(v, r.get(k), path + (k,))
                for k, v in meta.items()
            }
        return None

    return {
        key: _to_abstract(val, (ref_state or {}).get(key), (key,))
        for key, val in pytree_meta.items()
    }
