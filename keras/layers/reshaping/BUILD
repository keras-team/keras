# Description:
#  Contains the Keras reshaping layers.

load("@org_keras//keras:keras.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:__subpackages__",
        "//third_party/tensorflow/python/distribute:__pkg__",
        "//third_party/tensorflow/python/feature_column:__pkg__",
        "//third_party/tensorflow/python/keras:__subpackages__",
        "//third_party/tensorflow/python/training/tracking:__pkg__",
        "//third_party/tensorflow/tools/pip_package:__pkg__",
        "//third_party/tensorflow_models/official/projects/residual_mobilenet/modeling/backbones:__pkg__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "reshaping",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":cropping1d",
        ":cropping2d",
        ":cropping3d",
        ":flatten",
        ":permute",
        ":repeat_vector",
        ":reshape",
        ":up_sampling1d",
        ":up_sampling2d",
        ":up_sampling3d",
        ":zero_padding1d",
        ":zero_padding2d",
        ":zero_padding3d",
    ],
)

py_library(
    name = "cropping1d",
    srcs = ["cropping1d.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "cropping2d",
    srcs = ["cropping2d.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "cropping3d",
    srcs = ["cropping3d.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "flatten",
    srcs = ["flatten.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "permute",
    srcs = ["permute.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
    ],
)

py_library(
    name = "repeat_vector",
    srcs = ["repeat_vector.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
    ],
)

py_library(
    name = "reshape",
    srcs = ["reshape.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
    ],
)

py_library(
    name = "up_sampling1d",
    srcs = ["up_sampling1d.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
    ],
)

py_library(
    name = "up_sampling2d",
    srcs = ["up_sampling2d.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "up_sampling3d",
    srcs = ["up_sampling3d.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "zero_padding1d",
    srcs = ["zero_padding1d.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "zero_padding2d",
    srcs = ["zero_padding2d.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "zero_padding3d",
    srcs = ["zero_padding3d.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/utils:engine_utils",
    ],
)

cuda_py_test(
    name = "cropping_test",
    size = "medium",
    srcs = ["cropping_test.py"],
    python_version = "PY3",
    shard_count = 8,
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "flatten_test",
    size = "medium",
    srcs = ["flatten_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "permute_test",
    size = "medium",
    srcs = ["permute_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "repeat_vector_test",
    size = "medium",
    srcs = ["repeat_vector_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "reshape_test",
    size = "medium",
    srcs = ["reshape_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

cuda_py_test(
    name = "up_sampling_test",
    size = "medium",
    srcs = ["up_sampling_test.py"],
    python_version = "PY3",
    shard_count = 8,
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

cuda_py_test(
    name = "zero_padding_test",
    size = "medium",
    srcs = ["zero_padding_test.py"],
    python_version = "PY3",
    shard_count = 8,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)
