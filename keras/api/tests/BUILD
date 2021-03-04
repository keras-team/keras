# TensorFlow API backwards compatibility tests.

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = ["//keras/api:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "README.txt",
    "API_UPDATE_WARNING.txt",
])

tf_py_test(
    name = "api_compatibility_test",
    srcs = ["api_compatibility_test.py"],
    data = [
        "//keras/api/golden:api_golden_v1",
        "//keras/api/golden:api_golden_v2",
        "//keras/api/tests:API_UPDATE_WARNING.txt",
        "//keras/api/tests:README.txt",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_oss",  # TODO(scottzhu): Fix this in OSS test.
        "no_pip",
        "no_rocm",
        "no_windows",  # Bugs due to some paths.
    ],
    deps = [
        "//:expect_six_installed",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python:lib",
        "//third_party/tensorflow/python:platform",
        "//third_party/tensorflow/tools/api/lib:python_object_to_proto_visitor",
        "//third_party/tensorflow/tools/common:public_api",
        "//third_party/tensorflow/tools/common:traverse",
    ],
)
