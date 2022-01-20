# Description:
#  Contains the Keras core layers.
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
        ":activity_regularization",
        ":dense",
        ":dropout",
        ":lambda",
        ":masking",
        ":spatial_dropout",
        ":tf_op_layer",
        "//keras/layers/reshaping:flatten",
        "//keras/layers/reshaping:permute",
        "//keras/layers/reshaping:repeat_vector",
        "//keras/layers/reshaping:reshape",
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
    name = "activity_regularization",
    srcs = ["activity_regularization.py"],
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
    ],
)

py_library(
    name = "dropout",
    srcs = ["dropout.py"],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
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
    name = "spatial_dropout",
    srcs = ["spatial_dropout.py"],
    srcs_version = "PY3",
    deps = [
        ":dropout",
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
