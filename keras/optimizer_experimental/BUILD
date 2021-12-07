# Reworked keras optimizer. For more context, please refer to go/new-keras-optimizer.

load("@org_keras//keras:keras.bzl", "distribute_py_test")

package(
    default_visibility = [
        "//keras:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "optimizer",
    srcs = [
        "adadelta.py",
        "adagrad.py",
        "adam.py",
        "optimizer.py",
        "rmsprop.py",
        "sgd.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":optimizer_lib",
        "//:expect_tensorflow_installed",
        "//keras/optimizer_v2:learning_rate_schedule",
    ],
)

py_library(
    name = "optimizer_lib",
    srcs = [
        "optimizer_lib.py",
    ],
    srcs_version = "PY3",
    deps = [],
)

py_test(
    name = "optimizer_lib_test",
    srcs = ["optimizer_lib_test.py"],
    deps = [
        ":optimizer_lib",
        "//:expect_tensorflow_installed",
    ],
)

distribute_py_test(
    name = "optimizer_test",
    size = "medium",
    srcs = ["optimizer_test.py"],
    shard_count = 1,
    tags = [
        "multi_gpu",
        "no_windows",
        "nomultivm",  # TODO(b/203558991): Re-enable.
    ],
    deps = [
        ":optimizer",
        ":optimizer_lib",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)
