package(default_visibility = ["//keras:__subpackages__"])

# Description:
#  Tools for building the TensorFlow pip package.

COMMON_PIP_DEPS = [
    "//keras",
]

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = COMMON_PIP_DEPS,
)
