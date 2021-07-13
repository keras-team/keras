load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:__subpackages__",
        "//third_party/tensorflow/python/feature_column:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "feature_column",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":base_feature_layer",
        ":dense_features",
        ":dense_features_v2",
        ":sequence_feature_column",
    ],
)

py_library(
    name = "base_feature_layer",
    srcs = ["base_feature_layer.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
        "//keras/utils:generic_utils",
    ],
)

py_library(
    name = "dense_features",
    srcs = [
        "dense_features.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":base_feature_layer",
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "dense_features_v2",
    srcs = [
        "dense_features_v2.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":base_feature_layer",
        ":dense_features",
        "//:expect_tensorflow_installed",
        "//keras/utils:tf_contextlib",
    ],
)

tf_py_test(
    name = "dense_features_test",
    srcs = ["dense_features_test.py"],
    tags = ["no_pip"],
    deps = [
        ":dense_features",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)

tf_py_test(
    name = "dense_features_v2_test",
    srcs = ["dense_features_v2_test.py"],
    tags = ["no_pip"],
    deps = [
        ":dense_features_v2",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)

py_library(
    name = "sequence_feature_column",
    srcs = ["sequence_feature_column.py"],
    srcs_version = "PY3",
    deps = [
        ":base_feature_layer",
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

tf_py_test(
    name = "sequence_feature_column_test",
    srcs = ["sequence_feature_column_test.py"],
    deps = [
        ":sequence_feature_column",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)

tf_py_test(
    name = "sequence_feature_column_integration_test",
    srcs = ["sequence_feature_column_integration_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":dense_features",
        ":sequence_feature_column",
        "//:expect_tensorflow_installed",
        "//keras:metrics",  # Import it here since base_layer didn't import it due to circular dependency.
        "//keras/layers:merge",
        "//keras/layers:recurrent",
        "//keras/layers/core",
    ],
)
