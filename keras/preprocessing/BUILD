# Description:
#   Contains the Keras preprocessing layers (internal TensorFlow version).

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:friends",
    ],
    licenses = ["notice"],
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
        "//keras/utils",
    ],
)

py_library(
    name = "image",
    srcs = [
        "image.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_pandas_installed",
        "//:expect_pillow_installed",
        "//:expect_scipy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/utils:data_utils",
        "//keras/utils:image_utils",
    ],
)

py_library(
    name = "sequence",
    srcs = [
        "sequence.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/utils:data_utils",
    ],
)

py_library(
    name = "text",
    srcs = [
        "text.py",
    ],
    srcs_version = "PY3",
    deps = ["//:expect_tensorflow_installed"],
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
        "//keras/testing_infra:test_combinations",
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
        "//keras/testing_infra:test_combinations",
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
        "//keras/testing_infra:test_combinations",
    ],
)
