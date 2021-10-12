# Reworked keras optimizer. For more context, please refer to go/new-keras-optimizer.

load("@org_keras//keras:keras.bzl", "distribute_py_test")

package(
    default_visibility = [
        "//keras:__subpackages__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "optimizer",
    srcs = [
        "adam.py",
        "optimizer.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/optimizer_v2:learning_rate_schedule",
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
    ],
    deps = [
        ":optimizer",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)
