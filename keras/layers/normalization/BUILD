# Description:
#   Contains the Keras normalization layers (internal TensorFlow version).

# buildifier: disable=same-origin-load
load("@org_keras//keras:keras.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = ["//keras:friends"],
    licenses = ["notice"],
)

py_library(
    name = "normalization",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":batch_normalization",
        ":batch_normalization_v1",
        ":group_normalization",
        ":layer_normalization",
        ":spectral_normalization",
        ":unit_normalization",
    ],
)

py_library(
    name = "batch_normalization",
    srcs = ["batch_normalization.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/dtensor:utils",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/initializers",
        "//keras/utils:control_flow_util",
    ],
)

py_library(
    name = "batch_normalization_v1",
    srcs = ["batch_normalization_v1.py"],
    srcs_version = "PY3",
    deps = [
        ":batch_normalization",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "group_normalization",
    srcs = ["group_normalization.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/dtensor:utils",
        "//keras/engine:base_layer",
        "//keras/initializers",
    ],
)

py_library(
    name = "layer_normalization",
    srcs = ["layer_normalization.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:constraints",
        "//keras:regularizers",
        "//keras/dtensor:utils",
        "//keras/engine:base_layer",
        "//keras/initializers",
    ],
)

py_library(
    name = "unit_normalization",
    srcs = ["unit_normalization.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
    ],
)

py_library(
    name = "spectral_normalization",
    srcs = ["spectral_normalization.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
    ],
)

cuda_py_test(
    name = "group_normalization_test",
    size = "medium",
    srcs = ["group_normalization_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "notsan",
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":group_normalization",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/layers",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

cuda_py_test(
    name = "batch_normalization_test",
    size = "medium",
    srcs = ["batch_normalization_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "notsan",
    ],
    deps = [
        ":batch_normalization",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/layers",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "batch_normalization_dtensor_test",
    srcs = ["batch_normalization_dtensor_test.py"],
    tags = ["no_oss"],
    deps = [
        ":batch_normalization",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/dtensor:test_util",
        "//keras/testing_infra:test_utils",
        "//third_party/tensorflow/python/distribute/experimental:mirrored_strategy",
    ],
)

cuda_py_test(
    name = "layer_normalization_test",
    size = "medium",
    srcs = ["layer_normalization_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "notsan",
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

cuda_py_test(
    name = "unit_normalization_test",
    size = "small",
    srcs = ["unit_normalization_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

cuda_py_test(
    name = "spectral_normalization_test",
    size = "small",
    srcs = ["spectral_normalization_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)
