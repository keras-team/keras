# Description:
#   Contains the Keras initializer API (internal TensorFlow version).

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "initializers",
    srcs = [
        "__init__.py",
        "initializers.py",
        "initializers_v1.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/dtensor:utils",
        "//keras/saving:serialization_lib",
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
        "//keras/engine",
        "//keras/models",
        "//keras/testing_infra:test_combinations",
    ],
)
