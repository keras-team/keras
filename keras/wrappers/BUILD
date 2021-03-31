# Description:
#   Contains the Keras wrapper API (internal TensorFlow version).

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = ["//keras:__subpackages__"],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "wrappers",
    srcs = [
        "__init__.py",
        "scikit_learn.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:engine",
        "//keras:losses",
        "//keras/utils:generic_utils",
    ],
)

tf_py_test(
    name = "scikit_learn_test",
    size = "small",
    srcs = ["scikit_learn_test.py"],
    python_version = "PY3",
    tags = ["notsan"],
    deps = [
        ":wrappers",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:testing_utils",
        "//keras/utils:np_utils",
    ],
)
