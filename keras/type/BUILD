package(
    default_visibility = ["//keras:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_six_installed",
    ],
)
