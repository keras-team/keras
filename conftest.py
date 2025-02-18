import os

# When using jax.experimental.enable_x64 in unit test, we want to keep the
# default dtype with 32 bits, aligning it with Keras's default.
os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"

try:
    # When using torch and tensorflow, torch needs to be imported first,
    # otherwise it will segfault upon import. This should force the torch
    # import to happen first for all tests.
    import torch  # noqa: F401
except ImportError:
    pass

import pytest  # noqa: E402

from keras.src.backend import backend  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_trainable_backend: mark test for trainable backend only",
    )


def pytest_collection_modifyitems(config, items):
    openvino_skipped_tests = []
    if backend() == "openvino":
        with open(
            "keras/src/backend/openvino/excluded_concrete_tests.txt", "r"
        ) as file:
            openvino_skipped_tests = file.readlines()
            # it is necessary to check if stripped line is not empty
            # and exclude such lines
            openvino_skipped_tests = [
                line.strip() for line in openvino_skipped_tests if line.strip()
            ]

    requires_trainable_backend = pytest.mark.skipif(
        backend() == "numpy" or backend() == "openvino",
        reason="Trainer not implemented for NumPy and OpenVINO backend.",
    )
    for item in items:
        if "requires_trainable_backend" in item.keywords:
            item.add_marker(requires_trainable_backend)
        # also, skip concrete tests for openvino, listed in the special file
        # this is more granular mechanism to exclude tests rather
        # than using --ignore option
        for skipped_test in openvino_skipped_tests:
            if skipped_test in item.nodeid:
                item.add_marker(
                    skip_if_backend(
                        "openvino",
                        "Not supported operation by openvino backend",
                    )
                )


def skip_if_backend(given_backend, reason):
    return pytest.mark.skipif(backend() == given_backend, reason=reason)
