# Description:
#  Contains the Keras attention layers.

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:friends",
        "//third_party/py/tensorflow_gnn:__subpackages__",
        "//third_party/tensorflow/python/distribute:__pkg__",
        "//third_party/tensorflow/python/feature_column:__pkg__",
        "//third_party/tensorflow/python/training/tracking:__pkg__",
        "//third_party/tensorflow/tools/pip_package:__pkg__",
        "//third_party/tensorflow_models/official/projects/residual_mobilenet/modeling/backbones:__pkg__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "attention",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":additive_attention",
        ":attention_layer",
        ":multi_head_attention",
    ],
)

py_library(
    name = "multi_head_attention",
    srcs = ["multi_head_attention.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/initializers",
        "//keras/layers/activation",
        "//keras/layers/core",
        "//keras/layers/regularization",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "base_dense_attention",
    srcs = ["base_dense_attention.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:base_layer",
        "//keras/utils:control_flow_util",
    ],
)

py_library(
    name = "attention_layer",
    srcs = ["attention.py"],
    srcs_version = "PY3",
    deps = [
        ":base_dense_attention",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "additive_attention",
    srcs = ["additive_attention.py"],
    srcs_version = "PY3",
    deps = [
        ":base_dense_attention",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "multi_head_attention_test",
    srcs = ["multi_head_attention_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "base_dense_attention_test",
    size = "medium",
    srcs = ["base_dense_attention_test.py"],
    python_version = "PY3",
    deps = [
        ":base_dense_attention",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "attention_test",
    size = "medium",
    srcs = ["attention_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/layers/core",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "additive_attention_test",
    size = "medium",
    srcs = ["additive_attention_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/mixed_precision:policy",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)
