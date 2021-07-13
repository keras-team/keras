# Description:
#   Contains Keras test utils and integration tests.

# buildifier: disable=same-origin-load
load("@org_keras//keras:keras.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("@org_keras//keras:keras.bzl", "tf_py_test")
load("@org_keras//keras:keras.bzl", "tpu_py_test")

package(
    default_visibility = [
        "//keras:__subpackages__",
        "//third_party/tensorflow/tools/pip_package:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//keras/google/private_tf_api_test:__pkg__"],
)

tf_py_test(
    name = "get_config_test",
    srcs = ["get_config_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_pip",
    ],
    deps = [
        ":get_config_samples",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "add_loss_correctness_test",
    srcs = ["add_loss_correctness_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/api:keras_api",
    ],
)

tpu_py_test(
    name = "automatic_outside_compilation_test",
    srcs = [
        "automatic_outside_compilation_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    tags = ["no_oss"],
    deps = [
        "//:expect_tensorboard_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/api:keras_api",
        "//keras/distribute:distribute_strategy_test_lib",
    ],
)

tf_py_test(
    name = "convert_to_constants_test",
    srcs = ["convert_to_constants_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:testing_utils",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "custom_training_loop_test",
    srcs = ["custom_training_loop_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = ["notsan"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "graph_util_test",
    srcs = ["graph_util_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "integration_test",
    size = "medium",
    srcs = ["integration_test.py"],
    python_version = "PY3",
    shard_count = 16,
    tags = [
        "notsan",
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/api:keras_api",
        "//keras/layers/legacy_rnn:rnn_cell_impl",
        "//keras/legacy_tf_layers:layers_base",
    ],
)

py_library(
    name = "model_architectures",
    srcs = [
        "model_architectures.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//keras",
    ],
)

tf_py_test(
    name = "model_architectures_test",
    srcs = ["model_architectures_test.py"],
    python_version = "PY3",
    shard_count = 16,
    deps = [
        ":model_architectures",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//keras",
        "//keras/api:keras_api",
    ],
)

py_library(
    name = "model_subclassing_test_util",
    srcs = ["model_subclassing_test_util.py"],
    srcs_version = "PY3",
    deps = [
        "//keras",
    ],
)

tf_py_test(
    name = "model_subclassing_test",
    srcs = ["model_subclassing_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_windows",
        "notsan",
    ],
    deps = [
        ":model_subclassing_test_util",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "model_subclassing_compiled_test",
    srcs = ["model_subclassing_compiled_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_windows",
        "notsan",
    ],
    deps = [
        ":model_subclassing_test_util",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/api:keras_api",
    ],
)

cuda_py_test(
    name = "memory_test",
    size = "medium",
    srcs = ["memory_test.py"],
    tags = [
        "manual",
        "no_oss",
        "notap",  #TODO(b/140640597): this test is flaky at the moment
        "optonly",  # The test is too slow in non-opt mode
    ],
    # TODO(b/140065350): Re-enable
    xla_enable_strict_auto_jit = False,
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "memory_checker_test",
    size = "medium",
    srcs = ["memory_checker_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
        "noasan",  # TODO(b/149948895): Re-enable.
        "nomsan",  # TODO(b/149948895): Re-enable.
        "notsan",  # TODO(b/149948895): Re-enable.
    ],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "saved_model_test",
    size = "small",
    srcs = ["saved_model_test.py"],
    tags = [
        "no_oss",  # TODO(b/170766453)
        "notap",  # TODO(b/170766453)
    ],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:metrics",
        "//keras/api:keras_api",
        "//keras/layers/core",
        "//keras/optimizer_v2",
    ],
)

cuda_py_test(
    name = "saver_test",
    size = "medium",
    srcs = ["saver_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/api:keras_api",
        "//keras/engine",
        "//keras/layers/core",
    ],
)

tf_py_test(
    name = "serialization_util_test",
    size = "small",
    srcs = ["serialization_util_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
        "//keras/engine",
        "//keras/layers/core",
    ],
)

tf_py_test(
    name = "temporal_sample_weights_correctness_test",
    srcs = ["temporal_sample_weights_correctness_test.py"],
    python_version = "PY3",
    shard_count = 20,
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "tracking_test",
    srcs = ["tracking_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",
        "nomac",
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
        "//keras/engine",
        "//keras/layers/core",
        "//keras/layers/normalization",
    ],
)

tf_py_test(
    name = "tracking_util_test",
    srcs = ["tracking_util_test.py"],
    python_version = "PY3",
    tags = ["notsan"],  # b/74395663
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
        "//keras/engine",
        "//keras/layers/core",
        "//keras/optimizer_v2",
    ],
)

tf_py_test(
    name = "tracking_util_with_v1_optimizers_test",
    srcs = ["tracking_util_with_v1_optimizers_test.py"],
    tags = [
        "no_windows",  # TODO(b/184424727): Re-enable this.
        "notsan",  # b/74395663
    ],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
        "//keras/engine",
        "//keras/layers/core",
    ],
)

py_library(
    name = "get_config_samples",
    srcs = ["get_config_samples.py"],
    srcs_version = "PY3",
    deps = [],
)
