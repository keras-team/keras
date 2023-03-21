# Description:
#  Contains the Keras core layers.
load("@org_keras//keras:keras.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:friends",
        "//third_party/py/tensorflow_gnn:__subpackages__",
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
    name = "core",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":activation",
        ":dense",
        ":einsum_dense",
        ":embedding",
        ":identity",
        ":lambda",
        ":masking",
        ":tf_op_layer",
        "//keras/layers/regularization:activity_regularization",
        "//keras/layers/regularization:dropout",
        "//keras/layers/regularization:spatial_dropout1d",
        "//keras/layers/regularization:spatial_dropout2d",
        "//keras/layers/regularization:spatial_dropout3d",
        "//keras/layers/reshaping:flatten",
        "//keras/layers/reshaping:permute",
        "//keras/layers/reshaping:repeat_vector",
        "//keras/layers/reshaping:reshape",
    ],
)

py_library(
    name = "activation",
    srcs = ["activation.py"],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "dense",
    srcs = ["dense.py"],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/dtensor:utils",
    ],
)

py_library(
    name = "einsum_dense",
    srcs = ["einsum_dense.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:activations",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/initializers",
    ],
)

py_library(
    name = "embedding",
    srcs = ["embedding.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/dtensor:utils",
        "//keras/engine:base_layer",
        "//keras/engine:base_layer_utils",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "lambda",
    srcs = ["lambda_layer.py"],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "masking",
    srcs = ["masking.py"],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "tf_op_layer",
    srcs = ["tf_op_layer.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "identity",
    srcs = ["identity.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
    ],
)

tf_py_test(
    name = "core_test",
    size = "medium",
    srcs = ["core_test.py"],
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
    name = "einsum_dense_test",
    srcs = ["einsum_dense_test.py"],
    python_version = "PY3",
    deps = [
        ":einsum_dense",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

cuda_py_test(
    name = "embedding_test",
    size = "medium",
    srcs = ["embedding_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/mixed_precision:policy",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)
