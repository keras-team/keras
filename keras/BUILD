# Description:
#   Contains the Keras API (internal TensorFlow version).

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        ":friends",
        "//third_party/py/tensorflow:__subpackages__",
        "//third_party/tensorflow/python/feature_column:__subpackages__",  # For unit test
        "//third_party/tensorflow/python/tpu:__subpackages__",  # For unit test
        "//third_party/tensorflow_estimator:__subpackages__",
    ],
    licenses = ["notice"],
)

# Keras code that doesn't live in core Keras directory, but still
# need to directly access the keras code.
# We shouldn't add any client side package to this list.
package_group(
    name = "friends",
    packages = ["//keras/..."],
)

exports_files(["LICENSE"])

config_setting(
    name = "no_keras_py_deps",
    define_values = {"no_keras_py_deps": "true"},
    visibility = ["//visibility:public"],
)

py_library(
    name = "keras",
    srcs = [
        "__init__.py",
        "keras_parameterized.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        ":engine",
        ":testing_utils",  # TODO(scottzhu): Stop exporting the test_utils after removing all the callers.
        "//:expect_h5py_installed",
        "//:expect_numpy_installed",
        "//:expect_pydot_installed",
        "//:expect_scipy_installed",
        "//:expect_tensorflow_installed",
        "//:expect_yaml_installed",
        "//keras/applications",
        "//keras/datasets",
        "//keras/distribute",
        "//keras/estimator",
        "//keras/feature_column",
        "//keras/layers",
        "//keras/layers/legacy_rnn:rnn_cell_impl",
        "//keras/legacy_tf_layers:layers",
        "//keras/mixed_precision:mixed_precision_experimental",
        "//keras/optimizer_v2",
        "//keras/premade_models",
        "//keras/preprocessing",
        "//keras/saving",
        "//keras/utils",
        "//keras/wrappers",
    ],
)

py_library(
    name = "backend",
    srcs = ["backend.py"],
    srcs_version = "PY3",
    deps = [
        ":backend_config",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/distribute:distribute_coordinator_utils",
        "//keras/engine:keras_tensor",
        "//keras/utils:control_flow_util",
        "//keras/utils:object_identity",
        "//keras/utils:tf_contextlib",
        "//keras/utils:tf_inspect",
    ],
)

py_library(
    name = "backend_config",
    srcs = ["backend_config.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

# TODO(scottzhu): Cleanup this target and point all the user to keras/engine.
py_library(
    name = "engine",
    srcs = [
        ":models",
        "//keras/metrics",
    ],
    srcs_version = "PY3",
    deps = [
        "//keras/engine",
    ],
)

py_library(
    name = "activations",
    srcs = [
        "activations.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//keras/layers/activation",
        "//keras/utils:engine_utils",
    ],
)

# TODO(scottzhu): Cleanup this target and point all the user to keras/engine.
py_library(
    name = "base_layer",
    srcs = [],
    srcs_version = "PY3",
    deps = [
        "//keras/engine:base_layer",
    ],
)

py_library(
    name = "callbacks",
    srcs = [
        "callbacks.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//:expect_tensorboard_installed",
        "//:expect_tensorflow_installed",
        "//keras/distribute:distributed_file_utils",
        "//keras/distribute:worker_training_state",
        "//keras/protobuf:projector_config_proto_py_pb2",
        "//keras/utils:engine_utils",
        "//keras/utils:mode_keys",
    ],
)

py_library(
    name = "combinations",
    srcs = [
        "combinations.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":testing_utils",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "callbacks_v1",
    srcs = [
        "callbacks_v1.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//:expect_tensorboard_installed",
        "//:expect_tensorflow_installed",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "constraints",
    srcs = [
        "constraints.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "losses",
    srcs = [
        "losses.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "models",
    srcs = [
        "models.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        ":optimizers",
        "//:expect_tensorflow_installed",
        "//keras/engine",
        "//keras/engine:base_layer",
        "//keras/metrics",
        "//keras/saving",
        "//keras/utils:generic_utils",
        "//keras/utils:version_utils",
    ],
)

py_library(
    name = "optimizers",
    srcs = [
        "optimizer_v1.py",
        "optimizers.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//keras/optimizer_experimental:optimizer",
        "//keras/optimizer_v2",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "regularizers",
    srcs = [
        "regularizers.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//keras/utils:engine_utils",
    ],
)

# Internally urllib.request.urlretrieve library requires Google
# SSL context to be provided to work in python 3. This isn't needed in OSS.
# copybara:uncomment_begin(google-only)
# py_library(
#     name = "url_utils",
#     srcs = ["google/url_utils.py"],
#     srcs_version = "PY3",
#     deps = ["//pyglib/contrib/google_ssl"],
# )
# copybara:uncomment_end

py_library(
    name = "testing_utils",
    srcs = [
        "testing_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        ":models",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer_utils",
        "//keras/layers",
        "//keras/optimizer_v2",
    ],
)

tf_py_test(
    name = "activations_test",
    size = "small",
    srcs = ["activations_test.py"],
    python_version = "PY3",
    deps = [
        ":activations",
        ":backend",
        ":combinations",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_scipy_installed",
        "//:expect_tensorflow_installed",
        "//keras/layers",
        "//keras/layers/activation",
        "//keras/layers/core",
    ],
)

tf_py_test(
    name = "combinations_test",
    size = "small",
    srcs = ["combinations_test.py"],
    python_version = "PY3",
    deps = [
        ":combinations",
        ":testing_utils",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "constraints_test",
    size = "small",
    srcs = ["constraints_test.py"],
    python_version = "PY3",
    deps = [
        ":backend",
        ":combinations",
        ":constraints",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "regularizers_test",
    size = "medium",
    srcs = ["regularizers_test.py"],
    python_version = "PY3",
    deps = [
        ":keras",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "optimizers_test",
    size = "medium",
    srcs = ["optimizers_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = ["notsan"],
    deps = [
        ":keras",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "losses_test",
    size = "small",
    srcs = ["losses_test.py"],
    python_version = "PY3",
    tags = [
        "noasan",  # b/186128525
    ],
    deps = [
        ":backend",
        ":combinations",
        ":losses",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/utils:engine_utils",
    ],
)

tf_py_test(
    name = "callbacks_test",
    size = "medium",
    srcs = ["callbacks_test.py"],
    python_version = "PY3",
    shard_count = 6,
    tags = [
        "no_tfrt",  # TODO(b/179690526)
        "notsan",
    ],
    deps = [
        ":keras",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "callbacks_v1_test",
    size = "medium",
    srcs = ["callbacks_v1_test.py"],
    python_version = "PY3",
    tags = ["notsan"],
    deps = [
        ":callbacks",
        ":callbacks_v1",
        ":combinations",
        ":testing_utils",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/engine",
        "//keras/layers",
        "//keras/utils:np_utils",
    ],
)

tf_py_test(
    name = "models_test",
    size = "medium",
    srcs = ["models_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "notsan",  # b/67509773
    ],
    deps = [
        ":keras",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "backend_test",
    size = "medium",
    srcs = ["backend_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":backend",
        ":combinations",
        ":engine",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_scipy_installed",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "backend_config_test",
    size = "medium",
    srcs = ["backend_config_test.py"],
    python_version = "PY3",
    deps = [
        ":backend",
        ":backend_config",
        ":combinations",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "keras_parameterized_test",
    size = "small",
    srcs = ["keras_parameterized_test.py"],
    python_version = "PY3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

# copybara:uncomment_begin(google-only)
# tf_py_test(
#     name = "url_utils_test",
#     srcs = ["google/url_utils_test.py"],
#     python_version = "PY3",
#     deps = [
#         ":url_utils",
#         "//:expect_tensorflow_installed",
#         "//testing/pymocks:matchers",
#     ],
# )
# copybara:uncomment_end
