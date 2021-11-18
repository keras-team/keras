# Description:
#   Contains the Keras datasets package (internal TensorFlow version).

package(
    default_visibility = [
        "//keras:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "datasets",
    srcs = [
        "__init__.py",
        "boston_housing.py",
        "cifar.py",
        "cifar10.py",
        "cifar100.py",
        "fashion_mnist.py",
        "imdb.py",
        "mnist.py",
        "reuters.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/utils:engine_utils",
    ],
)
