try:
    # When using torch and tensorflow, torch needs to be imported first,
    # otherwise it will segfault upon import. This should force the torch
    # import to happen first for all tests.
    import torch  # noqa: F401
except ImportError:
    torch = None

import pytest  # noqa: E402

from keras.src.backend import backend  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_trainable_backend: mark test for trainable backend only",
    )
    config.addinivalue_line(
        "markers",
        "multi_device: mark test for running with multiple devices only",
    )

    # Disable CUDA TF32 to get higher numerical accuracy for correctness tests.
    if backend() == "jax":
        import jax

        if jax.default_backend() == "gpu":
            jax.config.update("jax_default_matmul_precision", "float32")
    elif backend() == "tensorflow":
        import tensorflow as tf

        if tf.config.list_physical_devices("GPU"):
            tf.config.experimental.enable_tensor_float_32_execution(False)
    elif backend() == "torch":
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = False


def pytest_collection_modifyitems(config, items):
    has_multiple_devices = False

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

    if backend() == "jax":
        import jax

        has_multiple_devices = jax.device_count() > 1

    requires_trainable_backend = pytest.mark.skipif(
        backend() in ["numpy", "openvino"],
        reason="Trainer not implemented for NumPy and OpenVINO backend.",
    )
    requires_multiple_devices = (
        None
        if has_multiple_devices
        else pytest.mark.skip(reason="Requires multiple devices")
    )

    for item in items:
        if "requires_trainable_backend" in item.keywords:
            item.add_marker(requires_trainable_backend)
        if requires_multiple_devices and "multi_device" in item.keywords:
            item.add_marker(requires_multiple_devices)

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
