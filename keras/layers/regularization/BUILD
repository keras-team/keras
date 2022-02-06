# Description:
#  Contains the Keras regularization layers.

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
    name = "regularization",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":activity_regularization",
        ":alpha_dropout",
        ":dropout",
        ":gaussian_dropout",
        ":gaussian_noise",
        ":spatial_dropout1d",
        ":spatial_dropout2d",
        ":spatial_dropout3d",
    ],
)

py_library(
    name = "dropout",
    srcs = ["dropout.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/utils:control_flow_util",
    ],
)

py_library(
    name = "spatial_dropout1d",
    srcs = ["spatial_dropout1d.py"],
    srcs_version = "PY3",
    deps = [
        ":dropout",
        "//:expect_tensorflow_installed",
        "//keras/engine:input_spec",
    ],
)

py_library(
    name = "spatial_dropout2d",
    srcs = ["spatial_dropout2d.py"],
    srcs_version = "PY3",
    deps = [
        ":dropout",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:input_spec",
    ],
)

py_library(
    name = "spatial_dropout3d",
    srcs = ["spatial_dropout3d.py"],
    srcs_version = "PY3",
    deps = [
        ":dropout",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:input_spec",
    ],
)

py_library(
    name = "gaussian_dropout",
    srcs = ["gaussian_dropout.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "gaussian_noise",
    srcs = ["gaussian_noise.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "activity_regularization",
    srcs = ["activity_regularization.py"],
    srcs_version = "PY3",
    deps = [
        "//keras:regularizers",
        "//keras/engine:base_layer",
    ],
)

py_library(
    name = "alpha_dropout",
    srcs = ["alpha_dropout.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/utils:tf_utils",
    ],
)

tf_py_test(
    name = "dropout_test",
    size = "medium",
    srcs = ["dropout_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "spatial_dropout_test",
    size = "medium",
    srcs = ["spatial_dropout_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "gaussian_dropout_test",
    size = "medium",
    srcs = ["gaussian_dropout_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "gaussian_noise_test",
    size = "medium",
    srcs = ["gaussian_noise_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "activity_regularization_test",
    size = "medium",
    srcs = ["activity_regularization_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "alpha_dropout_test",
    size = "medium",
    srcs = ["alpha_dropout_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)
