"""Subprocess worker for multi-host checkpoint tests.

This script is spawned by ``orbax_checkpoint_multihost_test.py`` as a
separate OS process.  Each worker initialises JAX distributed, builds
a sharded model, trains, saves a **full** checkpoint (weights +
optimizer state + model config), reloads it, and writes a JSON result
file with diffs and config for the test harness to verify.

Usage (called automatically by the test harness)::

    python _multihost_worker.py <coordinator> <num_processes> \\
        <process_id> <checkpoint_dir> --result_file <path>

Environment variables consumed:
    LOCAL_DEVICE_COUNT  Number of virtual CPU devices (default: 2).
    KERAS_BACKEND       Must be ``jax`` (set by the harness).
"""

import argparse
import json
import os

# ── JAX virtual devices (must precede any JAX import) ───────────────────────
os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count="
    + os.environ.get("LOCAL_DEVICE_COUNT", "2")
)
os.environ["KERAS_BACKEND"] = "jax"

import jax  # noqa: E402
import numpy as np  # noqa: E402

from keras.src import layers  # noqa: E402
from keras.src import models  # noqa: E402
from keras.src import saving  # noqa: E402
from keras.src import utils as keras_utils  # noqa: E402
from keras.src.callbacks.orbax_checkpoint import OrbaxCheckpoint  # noqa: E402
from keras.src.distribution import DeviceMesh  # noqa: E402
from keras.src.distribution import LayoutMap  # noqa: E402
from keras.src.distribution import ModelParallel  # noqa: E402
from keras.src.distribution import TensorLayout  # noqa: E402
from keras.src.distribution import set_distribution  # noqa: E402

# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args():
    """Parse command-line arguments passed by the test harness."""
    parser = argparse.ArgumentParser(
        description="Multi-host checkpoint worker",
    )
    parser.add_argument(
        "coordinator",
        help="coordinator_address (host:port)",
    )
    parser.add_argument(
        "num_processes",
        type=int,
        help="Total number of JAX processes",
    )
    parser.add_argument(
        "process_id",
        type=int,
        help="This worker's process ID",
    )
    parser.add_argument(
        "checkpoint_dir",
        help="Shared checkpoint directory",
    )
    parser.add_argument(
        "--result_file",
        required=True,
        help="Path to write the JSON result file",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  Shard-safe helpers                                                          #
# --------------------------------------------------------------------------- #
#  In multi-host, each process only owns local shards of global
#  arrays.  Calling ``np.array(var.value)`` on a global array that
#  spans non-addressable devices raises RuntimeError.  These helpers
#  operate on locally-addressable shards only.


def _local_shard_arrays(var):
    """Return a list of numpy arrays for locally addressable shards."""
    jax_arr = var.value
    if hasattr(jax_arr, "addressable_shards"):
        return [np.array(s.data) for s in jax_arr.addressable_shards]
    return [np.array(jax_arr)]


def _var_diffs(vars_a, vars_b):
    """Max abs diff per variable between two variable lists."""
    diffs = {}
    for va, vb in zip(vars_a, vars_b):
        sa = _local_shard_arrays(va)
        sb = _local_shard_arrays(vb)
        diffs[va.path] = max(
            float(np.max(np.abs(a - b))) for a, b in zip(sa, sb)
        )
    return diffs


# --------------------------------------------------------------------------- #
#  Setup                                                                       #
# --------------------------------------------------------------------------- #


def initialize_distributed(coordinator, num_processes, process_id):
    """Call ``jax.distributed.initialize`` for this worker."""
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=num_processes,
        process_id=process_id,
    )


def setup_distribution():
    """Create a global device mesh and configure ``ModelParallel``."""
    all_devices = jax.devices()
    mesh = DeviceMesh(
        (len(all_devices),),
        axis_names=["model"],
        devices=all_devices,
    )
    layout_map = LayoutMap(mesh)
    layout_map["dense_layer/kernel"] = TensorLayout(axes=("model", None))
    set_distribution(
        ModelParallel(
            device_mesh=mesh,
            layout_map=layout_map,
            auto_shard_dataset=False,
        )
    )


# --------------------------------------------------------------------------- #
#  Model                                                                       #
# --------------------------------------------------------------------------- #


def build_model():
    """Build and compile the small test model.

    Architecture::

        Input(8) → Dense(12, name='dense_layer')
                 → Dense(4, name='output_layer')
    """
    # All processes must share the same seed for consistent weight
    # initialisation across the global mesh.
    keras_utils.set_random_seed(42)

    inputs = layers.Input(shape=(8,), name="input_layer")
    x = layers.Dense(12, name="dense_layer")(inputs)
    outputs = layers.Dense(4, name="output_layer")(x)
    model = models.Model(inputs, outputs, name="multihost_model")
    model.compile(optimizer="adam", loss="mse", jit_compile=False)
    return model


def make_training_data():
    """Return deterministic ``(x_train, y_train)`` numpy arrays."""
    rng = np.random.RandomState(0)
    x_train = rng.randn(64, 8).astype(np.float32)
    y_train = rng.randn(64, 4).astype(np.float32)
    return x_train, y_train


# --------------------------------------------------------------------------- #
#  Train, save, reload                                                         #
# --------------------------------------------------------------------------- #


def train_and_save(model, x_train, y_train, checkpoint_dir):
    """Train for two epochs and save a full checkpoint.

    Saves the complete model: weights + optimizer state + model config.

    Returns:
        The ``OrbaxCheckpoint`` callback (for flag queries).
    """
    cb = OrbaxCheckpoint(
        directory=checkpoint_dir,
        save_freq="epoch",
        save_weights_only=False,  # full model: weights + optimizer + config
    )
    model.fit(
        x_train,
        y_train,
        epochs=2,
        batch_size=16,
        callbacks=[cb],
        verbose=0,
    )
    cb.wait_until_finished()
    cb.checkpointer.close()
    return cb


def reload_model(checkpoint_dir):
    """Load the full model from checkpoint (config + weights + optimizer)."""
    return saving.load_model(checkpoint_dir)


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #


def main():
    args = parse_args()

    # 1. Distributed init
    initialize_distributed(
        args.coordinator,
        args.num_processes,
        args.process_id,
    )

    # 2. Distribution / mesh
    setup_distribution()

    # 3. Build, train (2 epochs), and save full checkpoint
    model = build_model()
    x_train, y_train = make_training_data()
    cb = train_and_save(model, x_train, y_train, args.checkpoint_dir)

    # 4. Load full model from checkpoint
    loaded = reload_model(args.checkpoint_dir)

    # 5. Compare weights (element-wise max diff per variable)
    weight_diffs = _var_diffs(
        model.trainable_variables,
        loaded.trainable_variables,
    )

    # 6. Compare optimizer state
    optimizer_diffs = _var_diffs(
        model.optimizer.variables,
        loaded.optimizer.variables,
    )

    # 7. Model config / architecture
    model_config = {
        "name": model.name,
        "num_layers": len(model.layers),
        "compiled": model.compiled,
        "optimizer_class": type(model.optimizer).__name__,
        "loss": model.loss,
    }
    loaded_config = {
        "name": loaded.name,
        "num_layers": len(loaded.layers),
        "compiled": loaded.compiled,
        "optimizer_class": type(loaded.optimizer).__name__,
        "loss": loaded.loss,
    }

    # 8. Write results to file for the test harness to read.
    result = {
        "multihost_enabled": cb.is_multihost_enabled(),
        "primary_host": cb.is_primary_host(),
        "global_devices": len(jax.devices()),
        "local_devices": len(jax.local_devices()),
        "weight_diffs": weight_diffs,
        "optimizer_diffs": optimizer_diffs,
        "model_config": model_config,
        "loaded_config": loaded_config,
    }
    with open(args.result_file, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
