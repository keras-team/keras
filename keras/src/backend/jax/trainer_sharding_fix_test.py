"""Tests for mixed-sharding warning in JAX trainer.

Verifies that _get_state_sharding_spec() emits a warning when it detects
a mix of SingleDeviceSharding (local) and mesh-aware (NamedSharding)
variables, and stays silent when shardings are consistent.

Run with:
    KERAS_BACKEND=jax python -m pytest <this_file> -v
"""

import os
import warnings

import jax
import numpy as np
import pytest

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src.backend import distribution_lib as backend_dlib
from keras.src.distribution import distribution_lib

if backend.backend() == "jax":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    xla_flags = os.getenv("XLA_FLAGS") or ""
    if "xla_force_host_platform_device_count" not in xla_flags:
        os.environ["XLA_FLAGS"] = (
            f"{xla_flags} --xla_force_host_platform_device_count=8"
        )

_skip = pytest.mark.skipif(
    backend.backend() != "jax" or len(jax.devices()) != 8,
    reason="JAX backend with 8 simulated CPU devices required",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_data_parallel():
    return distribution_lib.DataParallel(devices=backend_dlib.list_devices())


def _make_model_parallel():
    shape = (4, 2)
    axis_names = ["batch", "model"]
    device_mesh = distribution_lib.DeviceMesh(
        shape, axis_names, backend_dlib.list_devices()
    )
    layout_map = distribution_lib.LayoutMap(device_mesh)
    layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
        [None, "model"]
    )
    layout_map[".*dense.*bias"] = distribution_lib.TensorLayout(["model"])
    return distribution_lib.ModelParallel(
        layout_map=layout_map, batch_dim_name="batch"
    )


def _build_model_outside_scope():
    """Functional model built without any distribution scope."""
    inputs = layers.Input(shape=[16])
    y = layers.Dense(units=32, activation="relu")(inputs)
    y = layers.Dense(units=10, activation="softmax")(y)
    return models.Model(inputs=inputs, outputs=y)


def _build_model_inside_scope(dist):
    """Build the same model entirely inside the distribution scope."""
    with dist.scope():
        inputs = layers.Input(shape=[16])
        y = layers.Dense(units=32, activation="relu")(inputs)
        y = layers.Dense(units=10, activation="softmax")(y)
        return models.Model(inputs=inputs, outputs=y)


# ---------------------------------------------------------------------------
# Tests — 4 total (2 parameterised functions x 2 distribution strategies)
# ---------------------------------------------------------------------------
@_skip
@pytest.mark.parametrize(
    "dist_factory,dist_name",
    [
        (_make_data_parallel, "DataParallel"),
        (_make_model_parallel, "ModelParallel"),
    ],
)
def test_warns_when_model_built_outside_scope(dist_factory, dist_name):
    """Model built outside scope + compiled inside -> mixed warning."""
    dist = dist_factory()
    model = _build_model_outside_scope()

    # All model weights should be local (SingleDeviceSharding)
    for w in model.weights:
        assert isinstance(
            w.value.sharding, jax.sharding.SingleDeviceSharding
        ), f"Expected SingleDeviceSharding before scope, got {w.value.sharding}"

    inputs = np.random.normal(size=(8, 16)).astype("float32")
    labels = np.random.normal(size=(8, 10)).astype("float32")

    with dist.scope():
        model.compile(loss="mse", optimizer="adam")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model._symbolic_build(data_batch=(inputs[:2], labels[:2]))
            model._get_state_sharding_spec()

        mixed = [w for w in caught if "mix of local" in str(w.message)]
        assert len(mixed) > 0, (
            f"[{dist_name}] Expected a mixed-sharding warning "
            f"but none was raised"
        )
        msg = str(mixed[0].message)
        assert "SingleDeviceSharding" in msg
        assert "distribution.scope()" in msg


@_skip
@pytest.mark.parametrize(
    "dist_factory,dist_name",
    [
        (_make_data_parallel, "DataParallel"),
        (_make_model_parallel, "ModelParallel"),
    ],
)
def test_no_warning_when_model_built_inside_scope(dist_factory, dist_name):
    """Model built inside scope -> all shardings consistent -> silent."""
    dist = dist_factory()
    model = _build_model_inside_scope(dist)

    inputs = np.random.normal(size=(8, 16)).astype("float32")
    labels = np.random.normal(size=(8, 10)).astype("float32")

    with dist.scope():
        model.compile(loss="mse", optimizer="adam")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model._symbolic_build(data_batch=(inputs[:2], labels[:2]))
            model._get_state_sharding_spec()

        mixed = [w for w in caught if "mix of local" in str(w.message)]
        assert len(mixed) == 0, (
            f"[{dist_name}] Unexpected mixed-sharding warning "
            f"when model is built inside scope"
        )
