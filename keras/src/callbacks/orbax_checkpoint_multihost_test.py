"""Tests for OrbaxCheckpoint multi-host distributed checkpointing.

These tests verify that OrbaxCheckpoint correctly saves and restores a
**full** model checkpoint (weights + optimizer state + model config)
across multiple JAX processes coordinated via
``jax.distributed.initialize()`` on localhost.

Each test reads from a shared ``multihost_results`` fixture that spawns
2 worker subprocesses (``_multihost_worker.py``) once per module.  The
workers train a small sharded model for 2 epochs, save the checkpoint,
reload via ``saving.load_model``, and report back diffs and config so
the tests can verify the roundtrip.

Prerequisites
-------------
The system hostname must resolve via ``getaddrinfo()`` (Gloo's UV
transport requires it).  On macOS with ``.roam.internal`` suffixes::

    sudo bash -c 'echo "127.0.0.1 $(hostname)" >> /etc/hosts'

Run
---
::

    KERAS_BACKEND=jax pytest \\
        keras/src/callbacks/orbax_checkpoint_multihost_test.py -xvs
"""

import json
import os
import socket
import subprocess
import sys

import jax
import pytest

from keras.src import backend

# All tests in this file require the JAX backend.
pytestmark = pytest.mark.skipif(
    backend.backend() != "jax",
    reason="Multi-host tests require JAX backend",
)

# Path to the standalone worker script (lives next to this file).
_WORKER_SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "_multihost_worker.py",
)


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _free_port():
    """Return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _hostname_resolves():
    """Return True if the system hostname resolves via getaddrinfo."""
    try:
        socket.getaddrinfo(socket.gethostname(), None)
        return True
    except socket.gaierror:
        return False


def _latest_step(ckpt_dir):
    """Return the highest numbered step directory name as a string."""
    return str(
        max(
            int(d)
            for d in os.listdir(ckpt_dir)
            if os.path.isdir(os.path.join(ckpt_dir, d)) and d.isdigit()
        )
    )


def _spawn_workers(
    ckpt_dir,
    num_processes=2,
    devices_per_process=2,
    timeout=120,
):
    """Launch *num_processes* workers and return parsed results.

    Each worker runs ``_multihost_worker.py`` with its own virtual
    CPU devices, trains a small sharded model, saves a full checkpoint,
    reloads it, and writes a JSON result file.

    Returns
    -------
    dict[int, dict]
        Mapping from ``process_id`` to the worker's JSON result.
    """
    port = _free_port()
    coordinator = f"localhost:{port}"

    # Each worker writes its result to a dedicated JSON file.
    result_files = {
        pid: os.path.join(ckpt_dir, f"_result_{pid}.json")
        for pid in range(num_processes)
    }
    os.makedirs(ckpt_dir, exist_ok=True)

    procs = []
    for pid in range(num_processes):
        env = os.environ.copy()
        env["LOCAL_DEVICE_COUNT"] = str(devices_per_process)
        env["KERAS_BACKEND"] = "jax"
        env.pop("XLA_FLAGS", None)  # worker sets its own
        p = subprocess.Popen(
            [
                sys.executable,
                _WORKER_SCRIPT,
                coordinator,
                str(num_processes),
                str(pid),
                ckpt_dir,
                "--result_file",
                result_files[pid],
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(p)

    for i, p in enumerate(procs):
        try:
            _, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            raise AssertionError(f"Worker {i} timed out after {timeout}s")
        if p.returncode != 0:
            raise AssertionError(
                f"Worker {i} exited with rc={p.returncode}:\n"
                + err.decode()[-2000:]
            )

    # Read results from JSON files.
    results = {}
    for pid in range(num_processes):
        path = result_files[pid]
        assert os.path.isfile(path), (
            f"Worker {pid} did not write result file {path}"
        )
        with open(path) as f:
            results[pid] = json.load(f)

    return results


# --------------------------------------------------------------------------- #
#  Fixture — runs workers once and shares results across all tests            #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def multihost_results(tmp_path_factory):
    """Run the two-process multi-host simulation once per module."""
    if not hasattr(jax, "distributed"):
        pytest.skip("jax.distributed not available")
    if not _hostname_resolves():
        pytest.skip(
            f"System hostname {socket.gethostname()!r} does not "
            "resolve; add it to /etc/hosts to enable this test"
        )

    ckpt_dir = str(tmp_path_factory.mktemp("multihost_ckpt"))
    results = _spawn_workers(ckpt_dir)
    results["_ckpt_dir"] = ckpt_dir
    return results


# --------------------------------------------------------------------------- #
#  1. Distributed flags                                                        #
# --------------------------------------------------------------------------- #


class TestMultihostFlags:
    """Verify JAX distributed flags reported by each worker."""

    def test_multihost_enabled(self, multihost_results):
        """Both processes must report multihost as enabled."""
        for pid in (0, 1):
            assert multihost_results[pid]["multihost_enabled"], (
                f"Process {pid}: expected is_multihost_enabled=True"
            )

    def test_device_counts(self, multihost_results):
        """Each process sees 4 global devices, 2 local."""
        for pid in (0, 1):
            r = multihost_results[pid]
            assert r["global_devices"] == 4
            assert r["local_devices"] == 2

    def test_primary_host(self, multihost_results):
        """Process 0 is primary, process 1 is not."""
        assert multihost_results[0]["primary_host"] is True
        assert multihost_results[1]["primary_host"] is False


# --------------------------------------------------------------------------- #
#  2. Checkpoint directory structure                                           #
# --------------------------------------------------------------------------- #


class TestMultihostCheckpointStructure:
    """Verify on-disk checkpoint layout produced by multi-host save."""

    def test_step_directory_exists(self, multihost_results):
        """At least one numbered step directory must be created."""
        ckpt_dir = multihost_results["_ckpt_dir"]
        step_dirs = [
            d
            for d in os.listdir(ckpt_dir)
            if os.path.isdir(os.path.join(ckpt_dir, d)) and d.isdigit()
        ]
        assert len(step_dirs) > 0

    def test_per_process_ocdbt_shards(self, multihost_results):
        """Each process writes its own ``ocdbt.process_N`` directory."""
        ckpt_dir = multihost_results["_ckpt_dir"]
        pytree = os.path.join(ckpt_dir, _latest_step(ckpt_dir), "pytree")
        assert os.path.isdir(pytree)
        assert os.path.isdir(os.path.join(pytree, "ocdbt.process_0"))
        assert os.path.isdir(os.path.join(pytree, "ocdbt.process_1"))

    def test_metadata_file(self, multihost_results):
        """``_METADATA`` file must exist and be non-empty."""
        ckpt_dir = multihost_results["_ckpt_dir"]
        metadata = os.path.join(
            ckpt_dir,
            _latest_step(ckpt_dir),
            "pytree",
            "_METADATA",
        )
        assert os.path.isfile(metadata)
        assert os.path.getsize(metadata) > 0


# --------------------------------------------------------------------------- #
#  3. Weights                                                                  #
# --------------------------------------------------------------------------- #


class TestMultihostWeights:
    """Verify that weights are preserved through save → load."""

    def test_weight_diffs_near_zero(self, multihost_results):
        """Loaded weights must match trained weights (max diff ≤ 1e-5)."""
        for pid in (0, 1):
            for path, diff in multihost_results[pid]["weight_diffs"].items():
                assert diff <= 1e-5, (
                    f"Process {pid}: weight {path} "
                    f"differs by {diff} after reload"
                )


# --------------------------------------------------------------------------- #
#  4. Optimizer state                                                          #
# --------------------------------------------------------------------------- #


class TestMultihostOptimizer:
    """Verify that optimizer state is preserved through save → load."""

    def test_optimizer_diffs_near_zero(self, multihost_results):
        """Loaded optimizer state must match trained state."""
        for pid in (0, 1):
            for path, diff in multihost_results[pid]["optimizer_diffs"].items():
                assert diff <= 1e-5, (
                    f"Process {pid}: optimizer var {path} "
                    f"differs by {diff} after reload"
                )


# --------------------------------------------------------------------------- #
#  5. Model config / architecture                                              #
# --------------------------------------------------------------------------- #


class TestMultihostModelConfig:
    """Verify that model architecture and compilation state survive."""

    def test_model_name_preserved(self, multihost_results):
        """Loaded model must have the same name."""
        for pid in (0, 1):
            original = multihost_results[pid]["model_config"]["name"]
            loaded = multihost_results[pid]["loaded_config"]["name"]
            assert original == loaded, (
                f"Process {pid}: model name changed from "
                f"{original!r} to {loaded!r}"
            )

    def test_layer_count_preserved(self, multihost_results):
        """Loaded model must have the same number of layers."""
        for pid in (0, 1):
            original = multihost_results[pid]["model_config"]["num_layers"]
            loaded = multihost_results[pid]["loaded_config"]["num_layers"]
            assert original == loaded, (
                f"Process {pid}: layer count changed from "
                f"{original} to {loaded}"
            )

    def test_compiled_state_preserved(self, multihost_results):
        """Loaded model must still be in compiled state."""
        for pid in (0, 1):
            assert multihost_results[pid]["loaded_config"]["compiled"], (
                f"Process {pid}: loaded model is not compiled"
            )

    def test_optimizer_class_preserved(self, multihost_results):
        """Loaded model must have the same optimizer class."""
        for pid in (0, 1):
            original = multihost_results[pid]["model_config"]["optimizer_class"]
            loaded = multihost_results[pid]["loaded_config"]["optimizer_class"]
            assert original == loaded, (
                f"Process {pid}: optimizer class changed from "
                f"{original!r} to {loaded!r}"
            )

    def test_loss_preserved(self, multihost_results):
        """Loaded model must have the same loss function."""
        for pid in (0, 1):
            original = multihost_results[pid]["model_config"]["loss"]
            loaded = multihost_results[pid]["loaded_config"]["loss"]
            assert original == loaded, (
                f"Process {pid}: loss changed from {original!r} to {loaded!r}"
            )
