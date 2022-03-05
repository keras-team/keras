# Description:
#  Contains the Keras convolution layers.

load("@org_keras//keras:keras.bzl", "cuda_py_test")

package(
    default_visibility = [
        "//keras:__subpackages__",
        "//third_party/tensorflow/python/distribute:__pkg__",
        "//third_party/tensorflow/python/feature_column:__pkg__",
        "//third_party/tensorflow/python/keras:__subpackages__",
        "//third_party/tensorflow/python/training/tracking:__pkg__",
        "//third_party/tensorflow/tools/pip_package:__pkg__",
        "//third_party/tensorflow_models/official/vision/beta/projects/residual_mobilenet/modeling/backbones:__pkg__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "convolutional",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":conv1d",
        ":conv1d_transpose",
        ":conv2d",
        ":conv2d_transpose",
        ":conv3d",
        ":conv3d_transpose",
        ":depthwise_conv1d",
        ":depthwise_conv2d",
        ":separable_conv1d",
        ":separable_conv2d",
        "//keras/layers/pooling:average_pooling1d",
        "//keras/layers/pooling:average_pooling2d",
        "//keras/layers/pooling:average_pooling3d",
        "//keras/layers/pooling:max_pooling1d",
        "//keras/layers/pooling:max_pooling2d",
        "//keras/layers/pooling:max_pooling3d",
        "//keras/layers/reshaping:cropping1d",
        "//keras/layers/reshaping:cropping2d",
        "//keras/layers/reshaping:cropping3d",
        "//keras/layers/reshaping:up_sampling1d",
        "//keras/layers/reshaping:up_sampling2d",
        "//keras/layers/reshaping:up_sampling3d",
        "//keras/layers/reshaping:zero_padding1d",
        "//keras/layers/reshaping:zero_padding2d",
        "//keras/layers/reshaping:zero_padding3d",
    ],
)

py_library(
    name = "base_conv",
    srcs = ["base_conv.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:activations",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/initializers",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "conv1d",
    srcs = ["conv1d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_conv",
        "//keras:activations",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/initializers",
    ],
)

py_library(
    name = "conv2d",
    srcs = ["conv2d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_conv",
        "//keras:activations",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/dtensor:utils",
        "//keras/initializers",
    ],
)

py_library(
    name = "conv3d",
    srcs = ["conv3d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_conv",
        "//keras:activations",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/initializers",
    ],
)

py_library(
    name = "conv1d_transpose",
    srcs = ["conv1d_transpose.py"],
    srcs_version = "PY3",
    deps = [
        ":conv1d",
        "//:expect_tensorflow_installed",
        "//keras:activations",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:input_spec",
        "//keras/initializers",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "conv2d_transpose",
    srcs = ["conv2d_transpose.py"],
    srcs_version = "PY3",
    deps = [
        ":conv2d",
        "//:expect_tensorflow_installed",
        "//keras:activations",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:input_spec",
        "//keras/initializers",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "conv3d_transpose",
    srcs = ["conv3d_transpose.py"],
    srcs_version = "PY3",
    deps = [
        ":conv3d",
        "//:expect_tensorflow_installed",
        "//keras:activations",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:input_spec",
        "//keras/initializers",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "base_separable_conv",
    srcs = ["base_separable_conv.py"],
    srcs_version = "PY3",
    deps = [
        ":base_conv",
        "//:expect_tensorflow_installed",
        "//keras:activations",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:input_spec",
        "//keras/initializers",
    ],
)

py_library(
    name = "separable_conv1d",
    srcs = ["separable_conv1d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_separable_conv",
        "//:expect_tensorflow_installed",
        "//keras:activations",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/initializers",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "separable_conv2d",
    srcs = ["separable_conv2d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_separable_conv",
        "//:expect_tensorflow_installed",
        "//keras:activations",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/initializers",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "base_depthwise_conv",
    srcs = ["base_depthwise_conv.py"],
    srcs_version = "PY3",
    deps = [
        ":base_conv",
        "//:expect_tensorflow_installed",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:input_spec",
        "//keras/initializers",
    ],
)

py_library(
    name = "depthwise_conv1d",
    srcs = ["depthwise_conv1d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_depthwise_conv",
        "//:expect_tensorflow_installed",
        "//keras/utils:engine_utils",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "depthwise_conv2d",
    srcs = ["depthwise_conv2d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_depthwise_conv",
        "//keras:backend",
        "//keras/utils:engine_utils",
        "//keras/utils:tf_utils",
    ],
)

cuda_py_test(
    name = "conv_test",
    size = "medium",
    srcs = ["conv_test.py"],
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

cuda_py_test(
    name = "conv_transpose_test",
    size = "medium",
    srcs = ["conv_transpose_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

cuda_py_test(
    name = "depthwise_conv_test",
    size = "medium",
    srcs = ["depthwise_conv_test.py"],
    python_version = "PY3",
    shard_count = 8,
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

cuda_py_test(
    name = "separable_conv_test",
    size = "medium",
    srcs = ["separable_conv_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)
