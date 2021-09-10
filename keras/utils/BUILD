# Description:
#   Contains the Keras Utilities (internal TensorFlow version).

load("@org_keras//keras:keras.bzl", "cuda_py_test")
load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//keras:__subpackages__",
        "//third_party/tensorflow/python/feature_column:__pkg__",
        "//third_party/tensorflow/tools/pip_package:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "utils",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":all_utils",
    ],
)

py_library(
    name = "all_utils",
    srcs = [
        "all_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":control_flow_util",
        ":engine_utils",
        ":generic_utils",
        ":layer_utils",
        ":multi_gpu_utils",
        ":np_utils",
        ":traceback_utils",
        ":vis_utils",
    ],
)

py_library(
    name = "control_flow_util",
    srcs = ["control_flow_util.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "kpl_test_utils",
    srcs = ["kpl_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//keras",
        "//keras/layers/preprocessing:string_lookup",
    ],
)

py_library(
    name = "data_utils",
    srcs = ["data_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":generic_utils",
        ":io_utils",
        ":tf_inspect",
    ],
)

py_library(
    name = "engine_utils",
    srcs = [
        "conv_utils.py",
        "losses_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":data_utils",
        ":io_utils",
        "//keras:backend",
    ],
)

py_library(
    name = "io_utils",
    srcs = ["io_utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "tf_utils",
    srcs = ["tf_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":object_identity",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "traceback_utils",
    srcs = ["traceback_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "generic_utils",
    srcs = [
        "generic_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":tf_contextlib",
        ":tf_inspect",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "mode_keys",
    srcs = [
        "mode_keys.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "layer_utils",
    srcs = [
        "kernelized_utils.py",
        "layer_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":engine_utils",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
    ],
)

py_library(
    name = "metrics_utils",
    srcs = [
        "metrics_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":generic_utils",
        ":tf_utils",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "version_utils",
    srcs = [
        "version_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "multi_gpu_utils",
    srcs = [
        "multi_gpu_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/layers",
    ],
)

py_library(
    name = "np_utils",
    srcs = [
        "np_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "object_identity",
    srcs = ["object_identity.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "tf_contextlib",
    srcs = ["tf_contextlib.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "tf_inspect",
    srcs = ["tf_inspect.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "vis_utils",
    srcs = [
        "vis_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "dataset_creator",
    srcs = [
        "dataset_creator.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "dataset_creator_test",
    srcs = ["dataset_creator_test.py"],
    python_version = "PY3",
    deps = [
        ":dataset_creator",
        "//:expect_portpicker_installed",
        "//:expect_tensorflow_installed",
        "//keras:combinations",
        "//keras/distribute:multi_worker_testing_utils",
        "//keras/engine",
        "//keras/layers/core",
    ],
)

tf_py_test(
    name = "data_utils_test",
    size = "medium",
    srcs = ["data_utils_test.py"],
    python_version = "PY3",
    shard_count = 6,
    tags = [
        "noasan",  # times out
        "notsan",
        "optonly",  # times out
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "generic_utils_test",
    size = "small",
    srcs = ["generic_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":generic_utils",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "version_utils_test",
    size = "small",
    srcs = ["version_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":version_utils",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "tf_utils_test",
    size = "small",
    srcs = ["tf_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":tf_utils",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)

tf_py_test(
    name = "composite_tensor_support_test",
    size = "medium",
    srcs = ["composite_tensor_support_test.py"],
    python_version = "PY3",
    shard_count = 8,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:engine",
        "//keras/layers",
    ],
)

tf_py_test(
    name = "io_utils_test",
    size = "small",
    srcs = ["io_utils_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",  # TODO: needs investigation on Windows
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
    name = "layer_utils_test",
    size = "small",
    srcs = ["layer_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":layer_utils",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "np_utils_test",
    size = "small",
    srcs = ["np_utils_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "kernelized_utils_test",
    size = "small",
    srcs = ["kernelized_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":layer_utils",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
    ],
)

cuda_py_test(
    name = "multi_gpu_utils_test",
    srcs = ["multi_gpu_utils_test.py"],
    python_version = "PY3",
    tags = [
        "guitar",
        "multi_gpu",
    ],
    xla_enable_strict_auto_jit = True,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "vis_utils_test",
    size = "small",
    srcs = ["vis_utils_test.py"],
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
    name = "conv_utils_test",
    size = "small",
    srcs = ["conv_utils_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "metrics_utils_test",
    size = "small",
    srcs = ["metrics_utils_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)

tf_py_test(
    name = "losses_utils_test",
    size = "small",
    srcs = ["losses_utils_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)

tf_py_test(
    name = "traceback_utils_test",
    size = "small",
    srcs = ["traceback_utils_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)
