# Description:
#   Contains the Keras Premade Models (internal TensorFlow version).
load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "premade_models",
    srcs = [
        "__init__.py",
        "linear.py",
        "wide_deep.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend_config",
        "//keras:regularizers",
    ],
)

tf_py_test(
    name = "linear_test",
    size = "medium",
    srcs = ["linear_test.py"],
    python_version = "PY3",
    shard_count = 2,
    deps = [
        ":premade_models",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "wide_deep_test",
    size = "medium",
    srcs = ["wide_deep_test.py"],
    python_version = "PY3",
    shard_count = 2,
    srcs_version = "PY3",
    deps = [
        ":premade_models",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)
