# Description:
#   Contains the Keras save model API (internal TensorFlow version).

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//keras:__subpackages__",
        "//third_party/tensorflow/python/distribute:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob([
        "*.py",
        "**/*.py",
    ]),
    visibility = ["//keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "saving",
    srcs = [
        "__init__.py",
        "hdf5_format.py",
        "model_config.py",
        "pickle_utils.py",
        "save.py",
        "saved_model_experimental.py",
        "saving_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_h5py_installed",
        "//:expect_tensorflow_installed",
        "//:expect_yaml_installed",
        "//keras:backend",
        "//keras:losses",
        "//keras:optimizers",
        "//keras:regularizers",
        "//keras/engine:input_spec",
        "//keras/mixed_precision:autocast_variable",
        "//keras/protobuf:saved_metadata_proto_py_pb2",
        "//keras/saving/saved_model",
        "//keras/saving/utils_v1",
        "//keras/utils:engine_utils",
        "//keras/utils:metrics_utils",
        "//keras/utils:mode_keys",
    ],
)

tf_py_test(
    name = "metrics_serialization_test",
    size = "medium",
    srcs = ["metrics_serialization_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "notsan",  # TODO(b/170870790)
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "losses_serialization_test",
    size = "medium",
    srcs = ["losses_serialization_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "pickle_utils_test",
    size = "medium",
    srcs = ["pickle_utils_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "save_weights_test",
    size = "medium",
    srcs = ["save_weights_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_oss_py35",  # b/147011479
        "no_windows",
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)

tf_py_test(
    name = "save_test",
    size = "medium",
    srcs = ["save_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)

tf_py_test(
    name = "saved_model_experimental_test",
    size = "medium",
    srcs = ["saved_model_experimental_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_oss",  # TODO(b/119349471): Re-enable
        "no_windows",
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "saving_utils_test",
    size = "medium",
    srcs = ["saving_utils_test.py"],
    python_version = "PY3",
    tags = ["notsan"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)
