# Description:
#   Contains the Keras OptimizerV2 API (internal TensorFlow version).

load("@org_keras//keras:keras.bzl", "cuda_py_test")

package(
    # TODO(scottzhu): Remove non-keras deps from TF.
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
        "adadelta.py",
        "adagrad.py",
        "adam.py",
        "adamax.py",
        "ftrl.py",
        "gradient_descent.py",
        "nadam.py",
        "optimizer_v2.py",
        "rmsprop.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:backend_config",
        "//keras/engine:base_layer_utils",
        "//keras/initializers",
        "//keras/optimizers:utils",
        "//keras/optimizers/schedules:learning_rate_schedule",
        "//keras/utils:layer_utils",
        "//keras/utils:tf_utils",
    ],
)

cuda_py_test(
    name = "adagrad_test",
    size = "medium",
    srcs = ["adagrad_test.py"],
    shard_count = 4,
    deps = [
        ":optimizers",
        "//:expect_tensorflow_installed",
        "//keras/testing_infra:test_combinations",
    ],
)

cuda_py_test(
    name = "adam_test",
    size = "medium",
    srcs = ["adam_test.py"],
    shard_count = 4,
    tags = [
        "no_rocm",
        "no_windows",  # TODO(b/171384138)
    ],
    deps = [
        ":optimizers",
        "//:expect_tensorflow_installed",
        "//keras/testing_infra:test_combinations",
    ],
)

cuda_py_test(
    name = "adamax_test",
    size = "medium",
    srcs = ["adamax_test.py"],
    shard_count = 4,
    # TODO(b/168527439): invalid resource variable reference on GPU for TFRT.
    tags = ["no_rocm"],
    deps = [
        ":optimizers",
        "//:expect_tensorflow_installed",
        "//keras/testing_infra:test_combinations",
    ],
)

cuda_py_test(
    name = "adadelta_test",
    size = "medium",
    srcs = ["adadelta_test.py"],
    shard_count = 4,
    # TODO(b/168527439): invalid resource variable reference on GPU for TFRT.
    deps = [
        ":optimizers",
        "//:expect_tensorflow_installed",
        "//keras/testing_infra:test_combinations",
    ],
)

cuda_py_test(
    name = "ftrl_test",
    size = "medium",
    srcs = ["ftrl_test.py"],
    shard_count = 4,
    deps = [
        ":optimizers",
        "//:expect_tensorflow_installed",
    ],
)

cuda_py_test(
    name = "gradient_descent_test",
    size = "medium",
    srcs = ["gradient_descent_test.py"],
    shard_count = 4,
    deps = [
        ":optimizers",
        "//:expect_tensorflow_installed",
        "//keras/testing_infra:test_combinations",
    ],
)

cuda_py_test(
    name = "nadam_test",
    size = "medium",
    srcs = ["nadam_test.py"],
    shard_count = 4,
    deps = [
        ":optimizers",
        "//:expect_tensorflow_installed",
    ],
)

cuda_py_test(
    name = "optimizer_v2_test",
    size = "medium",
    srcs = ["optimizer_v2_test.py"],
    shard_count = 8,
    tags = [
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

cuda_py_test(
    name = "rmsprop_test",
    size = "medium",
    srcs = ["rmsprop_test.py"],
    shard_count = 2,
    # TODO(b/168527439): invalid resource variable reference on GPU for TFRT.
    deps = [
        ":optimizers",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras/testing_infra:test_combinations",
    ],
)
