# Description:
#  Contains the Keras locally-connected layers.

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "locally_connected",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":locally_connected1d",
        ":locally_connected2d",
    ],
)

py_library(
    name = "locally_connected_utils",
    srcs = ["locally_connected_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "locally_connected1d",
    srcs = ["locally_connected1d.py"],
    srcs_version = "PY3",
    deps = [
        ":locally_connected_utils",
        "//keras:activations",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/initializers",
        "//keras/utils:engine_utils",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "locally_connected2d",
    srcs = ["locally_connected2d.py"],
    srcs_version = "PY3",
    deps = [
        ":locally_connected_utils",
        "//keras:activations",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/initializers",
        "//keras/utils:engine_utils",
        "//keras/utils:tf_utils",
    ],
)

tf_py_test(
    name = "locally_connected_test",
    size = "medium",
    srcs = ["locally_connected_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = ["no_windows"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/optimizers/legacy:optimizers",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)
