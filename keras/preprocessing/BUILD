# Description:
#   Contains the Keras preprocessing layers (internal TensorFlow version).

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "preprocessing",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":image",
        ":sequence",
        ":text",
        ":timeseries",
        "//keras/utils:all_utils",
    ],
)

py_library(
    name = "image",
    srcs = [
        "dataset_utils.py",
        "image.py",
        "image_dataset.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_keras_preprocessing_installed",
        "//:expect_pillow_installed",
        "//:expect_scipy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/utils:data_utils",
    ],
)

py_library(
    name = "sequence",
    srcs = [
        "sequence.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_keras_preprocessing_installed",
        "//:expect_pillow_installed",
        "//:expect_tensorflow_installed",
        "//keras/utils:data_utils",
    ],
)

py_library(
    name = "timeseries",
    srcs = [
        "timeseries.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "text",
    srcs = [
        "dataset_utils.py",
        "text.py",
        "text_dataset.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_keras_preprocessing_installed",
        "//:expect_pillow_installed",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "image_test",
    size = "medium",
    srcs = ["image_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",  # TODO(scottzhu): Fix for multiple export issue.
    ],
    deps = [
        ":image",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "image_dataset_test",
    size = "small",
    srcs = ["image_dataset_test.py"],
    python_version = "PY3",
    deps = [
        ":image",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "sequence_test",
    size = "small",
    srcs = ["sequence_test.py"],
    python_version = "PY3",
    deps = [
        ":sequence",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "text_test",
    size = "small",
    srcs = ["text_test.py"],
    python_version = "PY3",
    deps = [
        ":text",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "text_dataset_test",
    size = "small",
    srcs = ["text_dataset_test.py"],
    python_version = "PY3",
    deps = [
        ":text",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "timeseries_test",
    size = "small",
    srcs = ["timeseries_test.py"],
    python_version = "PY3",
    deps = [
        ":timeseries",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)
