# Description:
#   Contains the Keras engine API (internal TensorFlow version).

# buildifier: disable=same-origin-load
load("@org_keras//keras:keras.bzl", "tf_py_test")

# buildifier: disable=same-origin-load
load("@org_keras//keras:keras.bzl", "cuda_py_test")

package(
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//keras:__subpackages__",
        "//third_party/tensorflow/python:__pkg__",
        "//third_party/tensorflow/python/feature_column:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "engine",
    srcs = [
        "__init__.py",
        "compile_utils.py",
        "functional.py",
        "partial_batch_padding_handler.py",
        "saving.py",
        "sequential.py",
        "training.py",
        "training_arrays_v1.py",
        "training_distributed_v1.py",
        "training_eager_v1.py",
        "training_generator_v1.py",
        "training_utils.py",
        "training_utils_v1.py",
        "training_v1.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":base_preprocessing_layer",
        ":data_adapter",
        ":functional_utils",
        ":input_layer",
        ":input_spec",
        ":keras_tensor",
        ":node",
        "//:expect_h5py_installed",
        "//:expect_tensorboard_installed",
        "//:expect_tensorflow_installed",
        "//:expect_yaml_installed",
        "//keras:activations",
        "//keras:backend",
        "//keras:callbacks",
        "//keras:callbacks_v1",
        "//keras:constraints",
        "//keras:losses",
        "//keras:metrics",
        "//keras:optimizers",
        "//keras:regularizers",
        "//keras/distribute",
        "//keras/distribute:distribute_coordinator_utils",
        "//keras/initializers",
        "//keras/mixed_precision:autocast_variable",
        "//keras/mixed_precision:loss_scale_optimizer",
        "//keras/mixed_precision:policy",
        "//keras/saving",
        "//keras/utils:engine_utils",
        "//keras/utils:metrics_utils",
        "//keras/utils:mode_keys",
        "//keras/utils:tf_utils",
        "//keras/utils:version_utils",
    ],
)

py_library(
    name = "base_layer_utils",
    srcs = ["base_layer_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/utils:tf_inspect",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "base_layer",
    srcs = [
        "base_layer.py",
        "base_layer_v1.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":base_layer_utils",
        ":input_spec",
        ":node",
        "//:expect_numpy_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras/initializers",
        # TODO(keras-team): Fix the cyclar deps between layer and metrics.
        # "//keras:metrics",
        "//keras:regularizers",
        "//keras/mixed_precision:autocast_variable",
        "//keras/mixed_precision:loss_scale_optimizer",
        "//keras/mixed_precision:policy",
        "//keras/saving",
        "//keras/utils:generic_utils",
        "//keras/utils:layer_utils",
        "//keras/utils:object_identity",
        "//keras/utils:tf_utils",
        "//keras/utils:traceback_utils",
        "//keras/utils:version_utils",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "input_layer",
    srcs = ["input_layer.py"],
    deps = [
        ":base_layer",
        ":keras_tensor",
        ":node",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/distribute",
        "//keras/saving",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "functional_utils",
    srcs = ["functional_utils.py"],
    deps = [
        ":input_layer",
        ":keras_tensor",
        ":node",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "data_adapter",
    srcs = ["data_adapter.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/utils:dataset_creator",
        "//keras/utils:engine_utils",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "input_spec",
    srcs = ["input_spec.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "keras_tensor",
    srcs = ["keras_tensor.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/utils:object_identity",
    ],
)

py_library(
    name = "base_preprocessing_layer",
    srcs = [
        "base_preprocessing_layer.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "node",
    srcs = ["node.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer_utils",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/utils:tf_utils",
    ],
)

tf_py_test(
    name = "base_layer_utils_test",
    srcs = ["base_layer_utils_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":base_layer_utils",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:backend",
        "//keras:combinations",
    ],
)

tf_py_test(
    name = "data_adapter_test",
    size = "medium",
    srcs = ["data_adapter_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_oss_py38",  # TODO(b/150615192)
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":data_adapter",
        "//:expect_numpy_installed",
        "//:expect_pandas_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "base_preprocessing_layer_test",
    size = "medium",
    srcs = ["base_preprocessing_layer_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":base_preprocessing_layer",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "functional_utils_test",
    size = "medium",
    srcs = ["functional_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":functional_utils",
        ":input_layer",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:models",
        "//keras/layers",
    ],
)

cuda_py_test(
    name = "training_gpu_test",
    size = "small",
    srcs = ["training_gpu_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":engine",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:combinations",
        "//keras/layers:convolutional",
    ],
)

tf_py_test(
    name = "correctness_test",
    size = "medium",
    srcs = ["correctness_test.py"],
    python_version = "PY3",
    shard_count = 2,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "keras_tensor_test",
    size = "small",
    srcs = ["keras_tensor_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "ragged_keras_tensor_test",
    size = "small",
    srcs = ["ragged_keras_tensor_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "input_spec_test",
    size = "small",
    srcs = ["input_spec_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "training_test",
    size = "medium",
    srcs = ["training_test.py"],
    python_version = "PY3",
    shard_count = 20,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
    deps = [
        ":engine",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:backend",
        "//keras:callbacks",
        "//keras:combinations",
        "//keras:losses",
        "//keras:metrics",
        "//keras:testing_utils",
        "//keras/layers",
        "//keras/utils:data_utils",
        "//keras/utils:np_utils",
    ],
)

tf_py_test(
    name = "compile_utils_test",
    size = "medium",
    srcs = ["compile_utils_test.py"],
    tags = [
        "nomac",  # TODO(b/146226927)
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "training_dataset_test",
    size = "medium",
    srcs = ["training_dataset_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "training_arrays_test",
    size = "medium",
    srcs = ["training_arrays_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/layers",
    ],
)

tf_py_test(
    name = "training_generator_test",
    size = "medium",
    srcs = ["training_generator_test.py"],
    python_version = "PY3",
    shard_count = 6,
    tags = [
        "noasan",  # TODO(b/132183295): Re-enable this.
        "nomac",  # TODO(b/140193633): Re-enable this.
        "notsan",
    ],
    deps = [
        ":engine",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras:losses",
        "//keras:metrics",
        "//keras:testing_utils",
        "//keras/layers",
        "//keras/optimizer_v2",
        "//keras/utils:data_utils",
    ],
)

tf_py_test(
    name = "training_integration_test",
    size = "medium",
    srcs = ["training_integration_test.py"],
    python_version = "PY3",
    shard_count = 30,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "feature_columns_integration_test",
    size = "medium",
    srcs = ["feature_columns_integration_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "training_eager_test",
    size = "medium",
    srcs = ["training_eager_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "training_utils_v1_test",
    size = "medium",
    srcs = ["training_utils_v1_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",  # TODO(b/135021748) reenable
        "notsan",
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "input_layer_test",
    size = "medium",
    srcs = ["input_layer_test.py"],
    python_version = "PY3",
    shard_count = 3,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":base_layer",
        ":engine",
        "//keras",
        "//keras:combinations",
        "//keras:testing_utils",
        "//keras/utils:layer_utils",
    ],
)

tf_py_test(
    name = "functional_test",
    size = "medium",
    srcs = ["functional_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no-internal-py3",
        "no_rocm",
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":base_layer",
        ":engine",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:backend",
        "//keras:combinations",
        "//keras:models",
        "//keras:testing_utils",
        "//keras/initializers",
        "//keras/layers",
        "//keras/utils:layer_utils",
        "//keras/utils:tf_utils",
    ],
)

tf_py_test(
    name = "node_test",
    size = "medium",
    srcs = ["node_test.py"],
    python_version = "PY3",
    shard_count = 3,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":base_layer",
        ":engine",
        "//keras",
        "//keras:testing_utils",
        "//keras/utils:layer_utils",
    ],
)

tf_py_test(
    name = "base_layer_test",
    size = "medium",
    srcs = ["base_layer_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":base_layer",
        ":engine",
        "//:expect_numpy_installed",
        "//:expect_tensorboard_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:backend",
        "//keras:combinations",
        "//keras:regularizers",
        "//keras:testing_utils",
        "//keras/layers",
        "//keras/legacy_tf_layers:core",
        "//keras/mixed_precision:policy",
        "//keras/optimizer_v2",
        "//keras/utils:tf_utils",
    ],
)

tf_py_test(
    name = "control_flow_test",
    size = "medium",
    srcs = ["control_flow_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "sequential_test",
    size = "medium",
    srcs = ["sequential_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "deferred_sequential_test",
    size = "medium",
    srcs = ["deferred_sequential_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)
