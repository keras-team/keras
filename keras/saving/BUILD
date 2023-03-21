# Description:
#   Contains the Keras save model API (internal TensorFlow version).

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//keras:friends",
        "//third_party/tensorflow/python/distribute:__pkg__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "saving",
    srcs = [
        "__init__.py",
        "legacy/hdf5_format.py",
        "legacy/model_config.py",
        "legacy/save.py",
        "legacy/saving_utils.py",
        "pickle_utils.py",
        "saving_api.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":object_registration",
        ":serialization",
        ":serialization_lib",
        "//:expect_h5py_installed",
        "//:expect_tensorflow_installed",
        "//:expect_yaml_installed",
        "//keras:backend",
        "//keras:losses",
        "//keras:regularizers",
        "//keras/engine:input_spec",
        "//keras/mixed_precision:autocast_variable",
        "//keras/optimizers",
        "//keras/protobuf:saved_metadata_proto_py_pb2",
        "//keras/saving/legacy/saved_model",
        "//keras/utils:engine_utils",
        "//keras/utils:metrics_utils",
        "//keras/utils:mode_keys",
    ],
)

py_library(
    name = "saving_lib",
    srcs = [
        "saving_lib.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":serialization_lib",
        "//:expect_tensorflow_installed",
        "//keras/utils:generic_utils",
        "//keras/utils:io_utils",
    ],
)

tf_py_test(
    name = "saving_lib_test",
    size = "medium",
    srcs = ["saving_lib_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/utils:generic_utils",
    ],
)

py_library(
    name = "object_registration",
    srcs = [
        "object_registration.py",
    ],
    srcs_version = "PY3",
)

py_library(
    name = "serialization_lib",
    srcs = [
        "serialization_lib.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":object_registration",
        ":serialization",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/saving/legacy/saved_model:utils",
    ],
)

py_library(
    name = "serialization",
    srcs = [
        "legacy/serialization.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":object_registration",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/utils:tf_contextlib",
        "//keras/utils:tf_inspect",
    ],
)

tf_py_test(
    name = "object_registration_test",
    size = "small",
    srcs = ["object_registration_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "metrics_serialization_test",
    size = "medium",
    srcs = ["legacy/metrics_serialization_test.py"],
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
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "losses_serialization_test",
    size = "medium",
    srcs = ["legacy/losses_serialization_test.py"],
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
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "save_weights_test",
    size = "medium",
    srcs = ["legacy/save_weights_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_oss_py35",  # b/147011479
        "no_pip",  # TODO(b/202022379)
        "no_windows",
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "save_test",
    size = "medium",
    srcs = ["legacy/save_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_pip",  # TODO(b/202022379)
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "saving_utils_test",
    size = "medium",
    srcs = ["legacy/saving_utils_test.py"],
    python_version = "PY3",
    tags = ["notsan"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "serialization_lib_test",
    size = "small",
    srcs = ["serialization_lib_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/saving:serialization",
        "//keras/testing_infra:test_combinations",
    ],
)
