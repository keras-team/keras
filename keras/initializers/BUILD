# Description:
#   Contains the Keras initializer API (internal TensorFlow version).

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
    visibility = [
        "//keras/google/private_tf_api_test:__pkg__",
    ],
)

py_library(
    name = "initializers",
    srcs = [
        "__init__.py",
        "initializers_v1.py",
        "initializers_v2.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/utils:generic_utils",
        "//keras/utils:tf_inspect",
    ],
)

tf_py_test(
    name = "initializers_test",
    size = "small",
    srcs = ["initializers_test.py"],
    python_version = "PY3",
    deps = [
        ":initializers",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:combinations",
        "//keras:models",
        "//keras/engine",
    ],
)
