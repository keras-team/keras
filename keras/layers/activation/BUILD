# Description:
#  Contains the Keras activation layers.

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "activation",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":elu",
        ":leaky_relu",
        ":prelu",
        ":relu",
        ":softmax",
        ":thresholded_relu",
    ],
)

py_library(
    name = "relu",
    srcs = ["relu.py"],
    srcs_version = "PY3",
    deps = [
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "softmax",
    srcs = ["softmax.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "leaky_relu",
    srcs = ["leaky_relu.py"],
    srcs_version = "PY3",
    deps = [
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "prelu",
    srcs = ["prelu.py"],
    srcs_version = "PY3",
    deps = [
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
    name = "elu",
    srcs = ["elu.py"],
    srcs_version = "PY3",
    deps = [
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "thresholded_relu",
    srcs = ["thresholded_relu.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/utils:tf_utils",
    ],
)

tf_py_test(
    name = "relu_test",
    size = "medium",
    srcs = ["relu_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "softmax_test",
    size = "medium",
    srcs = ["softmax_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "leaky_relu_test",
    size = "medium",
    srcs = ["leaky_relu_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "prelu_test",
    size = "medium",
    srcs = ["prelu_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "elu_test",
    size = "medium",
    srcs = ["elu_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "thresholded_relu_test",
    size = "medium",
    srcs = ["thresholded_relu_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)
