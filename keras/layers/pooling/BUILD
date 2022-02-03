# Description:
#  Contains the Keras pooling layers.

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
    name = "pooling",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":average_pooling1d",
        ":average_pooling2d",
        ":average_pooling3d",
        ":global_average_pooling1d",
        ":global_average_pooling2d",
        ":global_average_pooling3d",
        ":global_max_pooling1d",
        ":global_max_pooling2d",
        ":global_max_pooling3d",
        ":max_pooling1d",
        ":max_pooling2d",
        ":max_pooling3d",
    ],
)

py_library(
    name = "base_pooling1d",
    srcs = ["base_pooling1d.py"],
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
    name = "base_pooling2d",
    srcs = ["base_pooling2d.py"],
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
    name = "base_pooling3d",
    srcs = ["base_pooling3d.py"],
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
    name = "base_global_pooling1d",
    srcs = ["base_global_pooling1d.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "base_global_pooling2d",
    srcs = ["base_global_pooling2d.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "base_global_pooling3d",
    srcs = ["base_global_pooling3d.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "max_pooling1d",
    srcs = ["max_pooling1d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_pooling1d",
        "//keras:backend",
    ],
)

py_library(
    name = "max_pooling2d",
    srcs = ["max_pooling2d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_pooling2d",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "max_pooling3d",
    srcs = ["max_pooling3d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_pooling3d",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "average_pooling1d",
    srcs = ["average_pooling1d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_pooling1d",
        "//keras:backend",
    ],
)

py_library(
    name = "average_pooling2d",
    srcs = ["average_pooling2d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_pooling2d",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "average_pooling3d",
    srcs = ["average_pooling3d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_pooling3d",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "global_max_pooling1d",
    srcs = ["global_max_pooling1d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_global_pooling1d",
        "//keras:backend",
    ],
)

py_library(
    name = "global_max_pooling2d",
    srcs = ["global_max_pooling2d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_global_pooling2d",
        "//keras:backend",
    ],
)

py_library(
    name = "global_max_pooling3d",
    srcs = ["global_max_pooling3d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_global_pooling3d",
        "//keras:backend",
    ],
)

py_library(
    name = "global_average_pooling1d",
    srcs = ["global_average_pooling1d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_global_pooling1d",
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "global_average_pooling2d",
    srcs = ["global_average_pooling2d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_global_pooling2d",
        "//keras:backend",
    ],
)

py_library(
    name = "global_average_pooling3d",
    srcs = ["global_average_pooling3d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_global_pooling3d",
        "//keras:backend",
    ],
)

tf_py_test(
    name = "average_pooling_test",
    size = "medium",
    srcs = ["average_pooling_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "notsan",  # TODO(b/183962355)
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "max_pooling_test",
    size = "medium",
    srcs = ["max_pooling_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "notsan",  # TODO(b/183962355)
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "global_average_pooling_test",
    size = "medium",
    srcs = ["global_average_pooling_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "notsan",  # TODO(b/183962355)
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "global_max_pooling_test",
    size = "medium",
    srcs = ["global_max_pooling_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "notsan",  # TODO(b/183962355)
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)
