# Description:
#   Contains the Keras Optimizer API.

load("@org_keras//keras:keras.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:friends",
        "//third_party/tensorflow/python:__pkg__",
        "//third_party/tensorflow/python/distribute:__pkg__",
        "//third_party/tensorflow/python/training/tracking:__pkg__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "optimizers",
    srcs = [
        "__init__.py",
        "optimizer_v1.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//keras:backend",
        "//keras/optimizers/legacy:optimizer",
        "//keras/optimizers/optimizer_experimental:optimizer",
        "//keras/optimizers/optimizer_v2",
        "//keras/optimizers/schedules:learning_rate_schedule",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "legacy_learning_rate_decay",
    srcs = ["legacy_learning_rate_decay.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/optimizers/schedules:learning_rate_schedule",
    ],
)

tf_py_test(
    name = "optimizers_test",
    size = "medium",
    srcs = ["optimizers_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = ["notsan"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

cuda_py_test(
    name = "legacy_learning_rate_decay_test",
    size = "medium",
    srcs = ["legacy_learning_rate_decay_test.py"],
    deps = [
        ":legacy_learning_rate_decay",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)
