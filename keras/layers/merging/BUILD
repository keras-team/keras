# Description:
#  Contains the Keras merging layers.

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
    name = "merging",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":add",
        ":average",
        ":concatenate",
        ":dot",
        ":maximum",
        ":minimum",
        ":multiply",
        ":subtract",
    ],
)

py_library(
    name = "base_merge",
    srcs = ["base_merge.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "add",
    srcs = ["add.py"],
    srcs_version = "PY3",
    deps = [
        ":base_merge",
    ],
)

py_library(
    name = "subtract",
    srcs = ["subtract.py"],
    srcs_version = "PY3",
    deps = [
        ":base_merge",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "multiply",
    srcs = ["multiply.py"],
    srcs_version = "PY3",
    deps = [
        ":base_merge",
    ],
)

py_library(
    name = "average",
    srcs = ["average.py"],
    srcs_version = "PY3",
    deps = [
        ":base_merge",
    ],
)

py_library(
    name = "maximum",
    srcs = ["maximum.py"],
    srcs_version = "PY3",
    deps = [
        ":base_merge",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "minimum",
    srcs = ["minimum.py"],
    srcs_version = "PY3",
    deps = [
        ":base_merge",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "concatenate",
    srcs = ["concatenate.py"],
    srcs_version = "PY3",
    deps = [
        ":base_merge",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "dot",
    srcs = ["dot.py"],
    srcs_version = "PY3",
    deps = [
        ":base_merge",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer_utils",
        "//keras/utils:tf_utils",
    ],
)

tf_py_test(
    name = "merging_test",
    size = "medium",
    srcs = ["merging_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)
