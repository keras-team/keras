# This package contains all the DTensor related Keras components.
# Since DTensor is not a public API yet, all the DTensor related change
# can't be exposed to public yet.

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:friends",
        "//learning/brain/experimental/dtensor/models:__subpackages__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "dtensor",
    srcs = ["__init__.py"],
)

tf_py_test(
    name = "initializers_test",
    srcs = ["initializers_test.py"],
    shard_count = 4,
    deps = [
        ":dtensor",
        ":test_util",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/initializers",
        "//keras/utils:tf_utils",
    ],
)

tf_py_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    shard_count = 4,
    tags = ["no_oss"],
    deps = [
        ":dtensor",
        ":test_util",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/layers",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "layout_map",
    srcs = ["layout_map.py"],
    deps = [
        ":dtensor",
        ":lazy_variable",
        ":utils",
        "//keras/engine:base_layer",
    ],
)

tf_py_test(
    name = "layout_map_test",
    srcs = ["layout_map_test.py"],
    tags = ["no_oss"],
    deps = [
        ":dtensor",
        ":layout_map",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/layers",
        "//keras/utils:tf_utils",
        "//learning/brain/experimental/dtensor/tests:test_util",
    ],
)

py_library(
    name = "integration_test_utils",
    srcs = ["integration_test_utils.py"],
    deps = [
        ":dtensor",
        ":layout_map",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:losses",
        "//keras/datasets",
        "//keras/layers",
        "//keras/models",
        "//keras/utils:np_utils",
    ],
)

tf_py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    shard_count = 4,
    deps = [
        ":dtensor",
        ":test_util",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/metrics",
        "//keras/utils:tf_utils",
    ],
)

tf_py_test(
    name = "mnist_model_test",
    srcs = ["mnist_model_test.py"],
    tags = [
        "requires-net:external",
    ],
    deps = [
        ":integration_test_utils",
        ":optimizers",
        ":test_util",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "optimizers",
    srcs = ["optimizers.py"],
    deps = [
        ":dtensor",
        "//:expect_tensorflow_installed",
        "//keras/optimizers/optimizer_experimental:optimizer",
        "//keras/optimizers/schedules:learning_rate_schedule",
    ],
)

tf_py_test(
    name = "optimizers_test",
    srcs = ["optimizers_test.py"],
    deps = [
        ":dtensor",
        ":optimizers",
        ":test_util",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "lazy_variable",
    srcs = ["lazy_variable.py"],
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":dtensor",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":dtensor",
        ":test_util",
        ":utils",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/layers",
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)
