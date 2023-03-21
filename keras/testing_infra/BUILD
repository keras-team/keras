# Description:
#   Contains the Keras testing infrastructure.

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = ["//keras:friends"],
    licenses = ["notice"],
)

py_library(
    name = "test_combinations",
    srcs = [
        "test_combinations.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":test_utils",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

py_library(
    name = "test_utils",
    srcs = [
        "test_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer_utils",
        "//keras/layers",
        "//keras/models",
        "//keras/optimizers/legacy:optimizers",
        "//keras/utils:tf_contextlib",
        "//keras/utils:tf_inspect",
    ],
)

# TODO(mattdangerw): For now, Keras will maintain its own doc checker.
# If TensorFlow exposes one, we could consider depending on that directly.
py_library(
    name = "keras_doctest_lib",
    srcs = ["keras_doctest_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
    ],
)

py_test(
    name = "keras_doctest_lib_test",
    srcs = ["keras_doctest_lib_test.py"],
    python_version = "PY3",
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        ":keras_doctest_lib",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "test_combinations_test",
    size = "small",
    srcs = ["test_combinations_test.py"],
    python_version = "PY3",
    tags = ["notsan"],
    deps = [
        ":test_combinations",
        ":test_utils",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)
