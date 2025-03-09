import os

# Ensure JAX uses 32-bit dtype to align with Keras defaults
os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"

# Ensuring proper import order for torch and TensorFlow
try:
    import torch  # noqa: F401
except ImportError:
    torch = None  # Explicitly set torch to None if not installed

import pytest  # noqa: E402
from keras.src.backend import backend  # noqa: E402


def pytest_configure(config):
    """Registers custom markers for pytest."""
    config.addinivalue_line(
        "markers",
        "requires_trainable_backend: mark test for trainable backend only",
    )


def skip_if_backend(given_backend, reason):
    """Helper function to skip tests based on backend."""
    return pytest.mark.skipif(backend() == given_backend, reason=reason)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip unsupported tests."""
    openvino_skipped_tests = []

    # Safely handle missing file by checking its existence
    excluded_tests_path = "keras/src/backend/openvino/excluded_concrete_tests.txt"
    if backend() == "openvino" and os.path.exists(excluded_tests_path):
        with open(excluded_tests_path, "r") as file:
            openvino_skipped_tests = [
                line.strip() for line in file if line.strip()
            ]

    # Skip trainable backend tests for NumPy and OpenVINO backends
    requires_trainable_backend = pytest.mark.skipif(
        backend() in ["numpy", "openvino"],
        reason="Trainer not implemented for NumPy and OpenVINO backend.",
    )

    for item in items:
        if "requires_trainable_backend" in item.keywords:
            item.add_marker(requires_trainable_backend)

        # Skip specific OpenVINO tests listed in the exclusion file
        if backend() == "openvino":
            for skipped_test in openvino_skipped_tests:
                if skipped_test in item.nodeid:
                    item.add_marker(
                        skip_if_backend(
                            "openvino",
                            "Not supported operation by OpenVINO backend.",
                        )
                    )
