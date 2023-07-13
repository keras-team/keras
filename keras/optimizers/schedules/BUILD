# Description:
#   Contains the learning rate schedule API,

# Placeholder: load unaliased py_library
load("@org_keras//keras:keras.bzl", "cuda_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//keras:license"],
    default_visibility = [
        "//keras:friends",
        "//third_party/tensorflow/python:__pkg__",
        "//third_party/tensorflow/python/distribute:__pkg__",
        "//third_party/tensorflow/python/trackable:__pkg__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "learning_rate_schedule",
    srcs = [
        "learning_rate_schedule.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/utils:generic_utils",
    ],
)

cuda_py_test(
    name = "learning_rate_schedule_test",
    size = "medium",
    srcs = ["learning_rate_schedule_test.py"],
    shard_count = 4,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/optimizers/legacy:optimizers",
        "//keras/testing_infra:test_combinations",
    ],
)
