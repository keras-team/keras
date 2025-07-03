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
    torch = None

import pytest  # noqa: E402

from keras.src.backend import backend  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_trainable_backend: mark test for trainable backend only",
    )
    config.addinivalue_line(
        "markers", "requires_tpu: mark test to run only on TPU"
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
        backend() in ["numpy", "openvino"],
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




def _cleanup_tpu_state():
    import tensorflow as tf

    try:
        tf.config.experimental_disconnect_from_cluster()
    except:
        pass

    try:
        tf.config.experimental_reset_memory_stats("TPU_SYSTEM")
    except:
        pass


@pytest.fixture(scope="session")
def tpu_strategy_fixture():
    import tensorflow as tf
    import time

    os.environ["TPU_NAME"] = "harshith-tf-4"
    os.environ["JAX_PLATFORMS"] = ""
    max_retries = int(os.environ.get("TPU_MAX_RETRIES", "3"))
    base_delay = float(os.environ.get("TPU_BASE_DELAY", "2.0"))
    tpu_available = False
    strategy = None

    for attempt in range(max_retries):
        try:
            print(f"TPU initialization attempt {attempt + 1}/{max_retries}")
            _cleanup_tpu_state()
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            tpu_available = True
            print("✓ TPU initialization successful!")
            break
        except (ValueError, RuntimeError, Exception) as e:
            print(f"✗ TPU initialization attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt) + (attempt * 0.5)
                print(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
                _cleanup_tpu_state()
            else:
                print("All TPU initialization attempts failed.")

    if not tpu_available:
        pytest.skip("TPU not available")

    yield strategy

    # Teardown
    _cleanup_tpu_state()


@pytest.fixture(autouse=True)
def tpu(request):
    marker = request.node.get_closest_marker("requires_tpu")
    if marker:
        strategy = request.getfixturevalue("tpu_strategy_fixture")
        request.node.cls.tpu_strategy = strategy