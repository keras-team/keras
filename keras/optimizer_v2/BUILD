# Description:
#   Contains the Keras OptimizerV2 API (internal TensorFlow version).

load("@org_keras//keras:keras.bzl", "cuda_py_test")

package(
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//keras:__subpackages__",
        "//third_party/tensorflow/python:__pkg__",
        "//third_party/tensorflow/python/distribute:__pkg__",
        "//third_party/tensorflow/python/training/tracking:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "optimizer_v2",
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
        "utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":learning_rate_schedule",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:backend_config",
        "//keras/engine:base_layer_utils",
        "//keras/initializers",
        "//keras/utils:layer_utils",
        "//keras/utils:tf_utils",
    ],
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

py_library(
    name = "legacy_learning_rate_decay",
    srcs = ["legacy_learning_rate_decay.py"],
    srcs_version = "PY3",
    deps = [
        ":learning_rate_schedule",
        "//:expect_tensorflow_installed",
    ],
)

cuda_py_test(
    name = "adagrad_test",
    size = "medium",
    srcs = ["adagrad_test.py"],
    shard_count = 4,
    deps = [
        ":optimizer_v2",
        "//:expect_tensorflow_installed",
        "//keras:combinations",
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
        ":optimizer_v2",
        "//:expect_tensorflow_installed",
        "//keras:combinations",
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
        ":optimizer_v2",
        "//:expect_tensorflow_installed",
        "//keras:combinations",
    ],
)

cuda_py_test(
    name = "adadelta_test",
    size = "medium",
    srcs = ["adadelta_test.py"],
    shard_count = 4,
    # TODO(b/168527439): invalid resource variable reference on GPU for TFRT.
    deps = [
        ":optimizer_v2",
        "//:expect_tensorflow_installed",
        "//keras:combinations",
    ],
)

cuda_py_test(
    name = "ftrl_test",
    size = "medium",
    srcs = ["ftrl_test.py"],
    shard_count = 4,
    deps = [
        ":optimizer_v2",
        "//:expect_tensorflow_installed",
    ],
)

cuda_py_test(
    name = "gradient_descent_test",
    size = "medium",
    srcs = ["gradient_descent_test.py"],
    shard_count = 4,
    deps = [
        ":optimizer_v2",
        "//:expect_tensorflow_installed",
        "//keras:combinations",
    ],
)

cuda_py_test(
    name = "nadam_test",
    size = "medium",
    srcs = ["nadam_test.py"],
    shard_count = 4,
    deps = [
        ":optimizer_v2",
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
        ":optimizer_v2",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)

cuda_py_test(
    name = "learning_rate_schedule_test",
    size = "medium",
    srcs = ["learning_rate_schedule_test.py"],
    shard_count = 4,
    deps = [
        ":optimizer_v2",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
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
        "//keras:combinations",
    ],
)

cuda_py_test(
    name = "rmsprop_test",
    size = "medium",
    srcs = ["rmsprop_test.py"],
    shard_count = 2,
    # TODO(b/168527439): invalid resource variable reference on GPU for TFRT.
    deps = [
        ":optimizer_v2",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras:combinations",
    ],
)
