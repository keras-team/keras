# Description:
#   Contains the Keras experimental idempotent saving API.

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//keras:friends",
        "//third_party/tensorflow/python/distribute:__pkg__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "experimental",
    srcs = [
        "saving_lib.py",
        "serialization_lib.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/saving/legacy/saved_model",
        "//keras/utils:generic_utils",
    ],
)

tf_py_test(
    name = "saving_lib_test",
    size = "medium",
    srcs = ["saving_lib_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/utils:generic_utils",
    ],
)

tf_py_test(
    name = "serialization_lib_test",
    size = "small",
    srcs = ["serialization_lib_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/saving:serialization",
        "//keras/testing_infra:test_combinations",
    ],
)
