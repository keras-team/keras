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
        "save.py",
        "saved_model/base_serialization.py",
        "saved_model/constants.py",
        "saved_model/json_utils.py",
        "saved_model/layer_serialization.py",
        "saved_model/load.py",
        "saved_model/metric_serialization.py",
        "saved_model/model_serialization.py",
        "saved_model/network_serialization.py",
        "saved_model/save.py",
        "saved_model/save_impl.py",
        "saved_model/serialized_attributes.py",
        "saved_model/utils.py",
        "saved_model_experimental.py",
        "saving_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":load_context",
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
        "//keras/saving/utils_v1",
        "//keras/utils:engine_utils",
        "//keras/utils:metrics_utils",
        "//keras/utils:mode_keys",
    ],
)

py_library(
    name = "load_context",
    srcs = [
        "saved_model/load_context.py",
    ],
    srcs_version = "PY3",
    deps = [],
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
    name = "saved_model_test",
    size = "medium",
    srcs = ["saved_model/saved_model_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_rocm",
        "no_windows",
        "notsan",  #TODO(b/181771982): it is flaky
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

tf_py_test(
    name = "revive_test",
    size = "medium",
    srcs = ["saved_model/revive_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_windows",  # b/158005583
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "json_utils_test",
    size = "small",
    srcs = ["saved_model/json_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":saving",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)
