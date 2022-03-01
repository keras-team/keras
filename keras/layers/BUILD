# Description:
#   Contains the Keras layers (internal TensorFlow version).

load("@org_keras//keras:keras.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//keras:friends",
        "//third_party/tensorflow/python/distribute:__pkg__",
        "//third_party/tensorflow/python/feature_column:__pkg__",
        "//third_party/tensorflow/python/training/tracking:__pkg__",
        "//third_party/tensorflow/tools/pip_package:__pkg__",
    ],
    licenses = ["notice"],
)

# A separate build for layers without serialization to avoid circular deps
# with feature column.
py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "serialization.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":einsum_dense",
        ":embeddings",
        ":kernelized",
        ":noise",
        "//keras/feature_column",
        "//keras/layers/activation",
        "//keras/layers/attention",
        "//keras/layers/convolutional",
        "//keras/layers/core",
        "//keras/layers/locally_connected",
        "//keras/layers/merging",
        "//keras/layers/normalization",
        "//keras/layers/pooling",
        "//keras/layers/preprocessing",
        "//keras/layers/regularization",
        "//keras/layers/reshaping",
        "//keras/layers/rnn",
        "//keras/premade_models",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "embeddings",
    srcs = ["embeddings.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/engine:base_layer",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

cuda_py_test(
    name = "embeddings_test",
    size = "medium",
    srcs = ["embeddings_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/layers",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

py_library(
    name = "einsum_dense",
    srcs = ["einsum_dense.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:activations",
        "//keras:base_layer",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/initializers",
    ],
)

py_library(
    name = "kernelized",
    srcs = ["kernelized.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:base_layer",
        "//keras/engine:input_spec",
        "//keras/initializers",
    ],
)

py_library(
    name = "noise",
    srcs = ["noise.py"],
    srcs_version = "PY3",
    deps = [
        "//keras/layers/regularization:alpha_dropout",
        "//keras/layers/regularization:gaussian_dropout",
        "//keras/layers/regularization:gaussian_noise",
    ],
)

tf_py_test(
    name = "tensorflow_op_layer_test",
    size = "medium",
    srcs = ["tensorflow_op_layer_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/saving",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "subclassed_layers_test",
    size = "medium",
    srcs = ["subclassed_layers_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "einsum_dense_test",
    srcs = ["einsum_dense_test.py"],
    python_version = "PY3",
    deps = [
        ":einsum_dense",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "serialization_test",
    size = "small",
    srcs = ["serialization_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "kernelized_test",
    size = "small",
    srcs = ["kernelized_test.py"],
    python_version = "PY3",
    deps = [
        ":layers",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:backend",
        "//keras/initializers",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "layers_test",
    size = "small",
    srcs = ["layers_test.py"],
    python_version = "PY3",
    deps = [
        ":layers",
        "//:expect_tensorflow_installed",
    ],
)
