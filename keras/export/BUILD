# Description:
#   Contains the Keras save model API (internal TensorFlow version).

# Placeholder: load unaliased py_library
load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//keras:license"],
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//keras:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "export_lib",
    srcs = [
        "export_lib.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "export_lib_test",
    size = "medium",
    srcs = ["export_lib_test.py"],
    python_version = "PY3",
    deps = [
        ":export_lib",
        "//:expect_absl_installed",  # absl/testing:parameterized
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)
