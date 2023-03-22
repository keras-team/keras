# Description:
#   Contains the Keras Optimizer API.

load("@org_keras//keras:keras.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("@org_keras//keras:keras.bzl", "tf_py_test")
load("@org_keras//keras:keras.bzl", "distribute_py_test")

package(
    default_visibility = [
        "//keras:friends",
        "//third_party/tensorflow/cc/saved_model:__pkg__",  # For unit tests.
        "//third_party/tensorflow/python:__pkg__",
        "//third_party/tensorflow/python/distribute:__pkg__",
        "//third_party/tensorflow/python/saved_model:__pkg__",  # For unit tests.
        "//third_party/tensorflow/python/training/tracking:__pkg__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "optimizers",
    srcs = [
        "__init__.py",
        "adadelta.py",
        "adafactor.py",
        "adagrad.py",
        "adam.py",
        "adamax.py",
        "adamw.py",
        "ftrl.py",
        "lion.py",
        "nadam.py",
        "optimizer.py",
        "optimizer_v1.py",
        "rmsprop.py",
        "sgd.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/dtensor:utils",
        "//keras/optimizers/legacy:optimizers",
        "//keras/optimizers/schedules:learning_rate_schedule",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
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
    name = "optimizer_v1_test",
    size = "medium",
    srcs = ["optimizer_v1_test.py"],
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

# TODO(b/228209527): Combine this test with optimizer_test after
# fixing the NCCL issue.
distribute_py_test(
    name = "optimizer_pss_test",
    size = "medium",
    srcs = ["optimizer_pss_test.py"],
    shard_count = 32,
    tags = [
        "multi_gpu",
        "no_oss",
        "no_windows",
    ],
    deps = [
        ":optimizers",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

distribute_py_test(
    name = "optimizer_test",
    size = "medium",
    srcs = ["optimizer_test.py"],
    shard_count = 16,
    tags = [
        "multi_gpu",
        "no_windows",
        "nomultivm",  # TODO(b/203558991): Re-enable.
    ],
    deps = [
        ":optimizers",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

cuda_py_test(
    name = "lion_test",
    size = "medium",
    srcs = ["lion_test.py"],
    shard_count = 4,
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)
