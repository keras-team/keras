# Description:
#   Contains the legacy TF layers (internal TensorFlow version).

# buildifier: disable=same-origin-load
load("@org_keras//keras:keras.bzl", "tf_py_test")
load("@org_keras//keras:keras.bzl", "cuda_py_test")

package(
    default_visibility = [
        "//keras:friends",
        "//learning/brain/contrib:__subpackages__",
        "//third_party/tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "layers",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":convolutional",
        ":core",
        ":layers_base",
        ":normalization",
        ":pooling",
    ],
)

py_library(
    name = "layers_base",
    srcs = [
        "__init__.py",
        "base.py",
        "migration_utils.py",
        "variable_scope_shim.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/mixed_precision:policy",
    ],
)

py_library(
    name = "convolutional",
    srcs = ["convolutional.py"],
    srcs_version = "PY3",
    deps = [
        ":layers_base",
        "//:expect_tensorflow_installed",
        "//keras/layers",
    ],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    deps = [
        ":layers_base",
        "//:expect_tensorflow_installed",
        "//keras/layers",
    ],
)

py_library(
    name = "normalization",
    srcs = ["normalization.py"],
    srcs_version = "PY3",
    deps = [
        ":layers_base",
        "//:expect_tensorflow_installed",
        "//keras/layers/normalization",
    ],
)

py_library(
    name = "pooling",
    srcs = ["pooling.py"],
    srcs_version = "PY3",
    deps = [
        ":layers_base",
        "//:expect_tensorflow_installed",
        "//keras/layers",
    ],
)

tf_py_test(
    name = "base_test",
    size = "small",
    srcs = ["base_test.py"],
    main = "base_test.py",
    python_version = "PY3",
    deps = [
        ":core",
        ":layers_base",
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "core_test",
    size = "small",
    srcs = ["core_test.py"],
    main = "core_test.py",
    python_version = "PY3",
    deps = [
        ":core",
        "//:expect_tensorflow_installed",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "convolutional_test",
    size = "small",
    srcs = ["convolutional_test.py"],
    main = "convolutional_test.py",
    python_version = "PY3",
    deps = [
        ":convolutional",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "pooling_test",
    size = "small",
    srcs = ["pooling_test.py"],
    main = "pooling_test.py",
    python_version = "PY3",
    tags = ["no_rocm"],
    deps = [
        ":pooling",
        "//:expect_tensorflow_installed",
    ],
)

cuda_py_test(
    name = "normalization_test",
    size = "medium",
    srcs = ["normalization_test.py"],
    main = "normalization_test.py",
    python_version = "PY3",
    shard_count = 10,
    deps = [
        ":convolutional",
        ":layers_base",
        ":normalization",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "variable_scope_shim_test",
    size = "small",
    srcs = ["variable_scope_shim_test.py"],
    main = "variable_scope_shim_test.py",
    python_version = "PY3",
    tags = ["no_windows"],
    deps = [
        ":core",
        ":layers_base",
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "migration_utils_test",
    size = "small",
    srcs = ["migration_utils_test.py"],
    main = "migration_utils_test.py",
    python_version = "PY3",
    deps = [
        ":layers",
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
        "//keras/engine:input_spec",
        "//keras/testing_infra:test_combinations",
    ],
)
