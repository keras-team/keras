# Description:
#  Contains the Keras recurrent layers.

load("@org_keras//keras:keras.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:friends",
        "//third_party/tensorflow_models/official/projects/residual_mobilenet/modeling/backbones:__pkg__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "rnn",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":abstract_rnn_cell",
        ":base_rnn",
        ":base_wrapper",
        ":bidirectional",
        ":cell_wrappers",
        ":conv_lstm1d",
        ":conv_lstm2d",
        ":conv_lstm3d",
        ":cudnn_gru",
        ":cudnn_lstm",
        ":gru",
        ":gru_v1",
        ":lstm",
        ":lstm_v1",
        ":simple_rnn",
        ":stacked_rnn_cells",
        ":time_distributed",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "rnn_utils",
    srcs = ["rnn_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/utils:control_flow_util",
    ],
)

py_library(
    name = "abstract_rnn_cell",
    srcs = ["abstract_rnn_cell.py"],
    srcs_version = "PY3",
    deps = [
        ":rnn_utils",
        "//keras/engine:base_layer",
    ],
)

py_library(
    name = "dropout_rnn_cell_mixin",
    srcs = ["dropout_rnn_cell_mixin.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "gru_lstm_utils",
    srcs = ["gru_lstm_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "gru",
    srcs = ["gru.py"],
    srcs_version = "PY3",
    deps = [
        ":base_rnn",
        ":dropout_rnn_cell_mixin",
        ":gru_lstm_utils",
        ":rnn_utils",
        "//:expect_tensorflow_installed",
        "//keras:activations",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "gru_v1",
    srcs = ["gru_v1.py"],
    srcs_version = "PY3",
    deps = [
        ":base_rnn",
        ":gru",
        ":rnn_utils",
        "//keras:activations",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:input_spec",
        "//keras/initializers",
    ],
)

py_library(
    name = "lstm",
    srcs = ["lstm.py"],
    srcs_version = "PY3",
    deps = [
        ":base_rnn",
        ":dropout_rnn_cell_mixin",
        ":gru_lstm_utils",
        ":rnn_utils",
        "//:expect_tensorflow_installed",
        "//keras:activations",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "lstm_v1",
    srcs = ["lstm_v1.py"],
    srcs_version = "PY3",
    deps = [
        ":base_rnn",
        ":lstm",
        ":rnn_utils",
        "//keras:activations",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:input_spec",
        "//keras/initializers",
    ],
)

py_library(
    name = "stacked_rnn_cells",
    srcs = ["stacked_rnn_cells.py"],
    srcs_version = "PY3",
    deps = [
        ":rnn_utils",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/utils:generic_utils",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "base_rnn",
    srcs = ["base_rnn.py"],
    srcs_version = "PY3",
    deps = [
        ":dropout_rnn_cell_mixin",
        ":rnn_utils",
        ":stacked_rnn_cells",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/saving/saved_model",
        "//keras/utils:generic_utils",
    ],
)

py_library(
    name = "simple_rnn",
    srcs = ["simple_rnn.py"],
    srcs_version = "PY3",
    deps = [
        ":base_rnn",
        ":dropout_rnn_cell_mixin",
        ":rnn_utils",
        "//:expect_tensorflow_installed",
        "//keras:activations",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "base_conv_rnn",
    srcs = ["base_conv_rnn.py"],
    srcs_version = "PY3",
    deps = [
        ":base_rnn",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:base_layer",
        "//keras/engine:input_spec",
        "//keras/utils:engine_utils",
        "//keras/utils:generic_utils",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "base_conv_lstm",
    srcs = ["base_conv_lstm.py"],
    srcs_version = "PY3",
    deps = [
        ":base_conv_rnn",
        ":dropout_rnn_cell_mixin",
        "//:expect_tensorflow_installed",
        "//keras:activations",
        "//keras:backend",
        "//keras:base_layer",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/initializers",
        "//keras/utils:engine_utils",
    ],
)

py_library(
    name = "conv_lstm1d",
    srcs = ["conv_lstm1d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_conv_lstm",
    ],
)

py_library(
    name = "conv_lstm2d",
    srcs = ["conv_lstm2d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_conv_lstm",
    ],
)

py_library(
    name = "conv_lstm3d",
    srcs = ["conv_lstm3d.py"],
    srcs_version = "PY3",
    deps = [
        ":base_conv_lstm",
    ],
)

py_library(
    name = "base_cudnn_rnn",
    srcs = ["base_cudnn_rnn.py"],
    srcs_version = "PY3",
    deps = [
        ":base_rnn",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:input_spec",
    ],
)

py_library(
    name = "cudnn_lstm",
    srcs = ["cudnn_lstm.py"],
    srcs_version = "PY3",
    deps = [
        ":base_cudnn_rnn",
        ":gru_lstm_utils",
        "//:expect_tensorflow_installed",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/initializers",
    ],
)

py_library(
    name = "cudnn_gru",
    srcs = ["cudnn_gru.py"],
    srcs_version = "PY3",
    deps = [
        ":base_cudnn_rnn",
        ":gru_lstm_utils",
        "//:expect_tensorflow_installed",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/initializers",
    ],
)

py_library(
    name = "cell_wrappers",
    srcs = ["cell_wrappers.py"],
    srcs_version = "PY3",
    deps = [
        ":abstract_rnn_cell",
        ":lstm",
        "//:expect_tensorflow_installed",
        "//keras/utils:generic_utils",
        "//keras/utils:tf_inspect",
    ],
)

py_library(
    name = "legacy_cell_wrappers",
    srcs = ["legacy_cell_wrappers.py"],
    srcs_version = "PY3",
    deps = [
        ":cell_wrappers",
        ":legacy_cells",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "legacy_cells",
    srcs = ["legacy_cells.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:activations",
        "//keras:backend",
        "//keras/engine:base_layer_utils",
        "//keras/engine:input_spec",
        "//keras/initializers",
        "//keras/legacy_tf_layers:layers_base",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "base_wrapper",
    srcs = ["base_wrapper.py"],
    srcs_version = "PY3",
    deps = [
        "//keras/engine:base_layer",
        "//keras/utils:generic_utils",
    ],
)

py_library(
    name = "bidirectional",
    srcs = ["bidirectional.py"],
    srcs_version = "PY3",
    deps = [
        ":base_wrapper",
        ":rnn_utils",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/utils:generic_utils",
        "//keras/utils:tf_inspect",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "time_distributed",
    srcs = ["time_distributed.py"],
    srcs_version = "PY3",
    deps = [
        ":base_wrapper",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/utils:generic_utils",
        "//keras/utils:layer_utils",
        "//keras/utils:tf_utils",
    ],
)

cuda_py_test(
    name = "gru_lstm_test",
    size = "medium",
    srcs = ["gru_lstm_test.py"],
    python_version = "PY3",
    shard_count = 2,
    deps = [
        ":gru",
        ":lstm",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/layers:embeddings",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

cuda_py_test(
    name = "gru_test",
    size = "medium",
    srcs = ["gru_test.py"],
    python_version = "PY3",
    shard_count = 12,
    tags = ["no_rocm"],
    deps = [
        ":gru_lstm_utils",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
        "//keras/utils:np_utils",
    ],
)

tf_py_test(
    name = "gru_v1_test",
    size = "medium",
    srcs = ["gru_v1_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "notsan",  # http://b/62136390
    ],
    deps = [
        ":gru",
        ":gru_v1",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
        "//keras/utils:np_utils",
    ],
)

cuda_py_test(
    name = "lstm_test",
    size = "medium",
    srcs = ["lstm_test.py"],
    python_version = "PY3",
    shard_count = 12,
    tags = [
        "no_oss",
        "notsan",  # TODO(b/170954246)
    ],
    deps = [
        ":gru_lstm_utils",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
        "//keras/utils:np_utils",
    ],
)

tf_py_test(
    name = "lstm_v1_test",
    size = "medium",
    srcs = ["lstm_v1_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "noasan",  # times out b/63678675
        "notsan",  # http://b/62189182
    ],
    deps = [
        ":lstm",
        ":lstm_v1",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
        "//keras/utils:np_utils",
    ],
)

tf_py_test(
    name = "base_rnn_test",
    size = "medium",
    srcs = ["base_rnn_test.py"],
    python_version = "PY3",
    shard_count = 12,
    tags = [
        "no_rocm",
        "notsan",  # TODO(b/170870794)
    ],
    deps = [
        ":gru",
        ":gru_v1",
        ":legacy_cells",
        ":lstm",
        ":lstm_v1",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/engine:base_layer_utils",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
        "//keras/utils:generic_utils",
    ],
)

tf_py_test(
    name = "simple_rnn_test",
    size = "medium",
    srcs = ["simple_rnn_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = ["notsan"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "conv_lstm_test",
    size = "medium",
    srcs = ["conv_lstm_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = ["no_rocm"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

cuda_py_test(
    name = "cudnn_test",
    size = "medium",
    srcs = ["cudnn_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_windows_gpu",
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/optimizers/optimizer_v2",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "cell_wrappers_test",
    size = "medium",
    srcs = ["cell_wrappers_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "notsan",
    ],
    deps = [
        ":cell_wrappers",
        ":legacy_cells",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/layers",
        "//keras/legacy_tf_layers:layers_base",
        "//keras/testing_infra:test_combinations",
        "//keras/utils:generic_utils",
    ],
)

tf_py_test(
    name = "legacy_cell_wrappers_test",
    size = "small",
    srcs = ["legacy_cell_wrappers_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":legacy_cell_wrappers",
        ":legacy_cells",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "base_wrapper_test",
    size = "small",
    srcs = ["base_wrapper_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "bidirectional_test",
    size = "medium",
    srcs = ["bidirectional_test.py"],
    python_version = "PY3",
    shard_count = 12,
    tags = [
        "noasan",  # http://b/78599823
        "notsan",
    ],
    deps = [
        ":cell_wrappers",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/engine:base_layer_utils",
        "//keras/layers/core",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
        "//keras/utils:generic_utils",
    ],
)

tf_py_test(
    name = "time_distributed_test",
    size = "medium",
    srcs = ["time_distributed_test.py"],
    python_version = "PY3",
    shard_count = 12,
    tags = [
        "noasan",  # http://b/78599823
        "notsan",
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)
