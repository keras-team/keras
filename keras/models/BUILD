# Keras models

load("@org_keras//keras:keras.bzl", "distribute_py_test")
load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "sharpness_aware_minimization",
    srcs = ["sharpness_aware_minimization.py"],
    srcs_version = "PY3",
    deps = [
        ":cloning",
        "//:expect_tensorflow_installed",
        "//keras/engine:data_adapter",
    ],
)

py_library(
    name = "models",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":cloning",
        ":sharpness_aware_minimization",
    ],
)

py_library(
    name = "cloning",
    srcs = [
        "cloning.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine",
        "//keras/engine:base_layer",
        "//keras/metrics",
        "//keras/optimizers",
        "//keras/saving",
        "//keras/utils:generic_utils",
        "//keras/utils:version_utils",
    ],
)

tf_py_test(
    name = "cloning_test",
    size = "medium",
    srcs = ["cloning_test.py"],
    main = "cloning_test.py",
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "notsan",  # b/67509773
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

distribute_py_test(
    name = "sharpness_aware_minimization_test",
    size = "medium",
    srcs = ["sharpness_aware_minimization_test.py"],
    shard_count = 8,
    tags = [
        "multi_gpu",
        "nomultivm",
        "requires-net:ipv4",
    ],
    deps = [
        ":sharpness_aware_minimization",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/optimizers",
        "//keras/testing_infra:test_combinations",
    ],
)
