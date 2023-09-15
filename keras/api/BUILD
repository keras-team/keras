# Description:
# Package for Keras.

# Placeholder: load unaliased py_binary
# Placeholder: load unaliased py_library
load("//keras/api:api_gen.bzl", "generate_apis")
load("//keras/api:api_init_files.bzl", "KERAS_API_INIT_FILES", "KERAS_API_INIT_FILES_V1")

package(
    # copybara:uncomment default_applicable_licenses = ["//keras:license"],
    default_visibility = [
        "//keras:friends",
        "//third_party/py/tensorflow:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0 License
)

# The target used by PIP package which need to generate API init files during OSS build.
py_library(
    name = "keras_api",
    srcs = [
        ":keras_python_api_gen",
        ":keras_python_api_gen_compat_v1",
        ":keras_python_api_gen_compat_v2",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

py_binary(
    name = "extractor_wrapper",
    srcs = ["extractor_wrapper.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//:expect_absl_installed",  # absl:app
        # "//third_party/tensorflow/python/tools/api/generator2/extractor",
    ],
)

py_binary(
    name = "generator_wrapper",
    srcs = ["generator_wrapper.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//:expect_absl_installed",  # absl:app
        # "//third_party/tensorflow/python/tools/api/generator2/generator",
    ],
)

generate_apis(
    name = "keras_python_api_gen",
    api_version = 1,
    output_files = KERAS_API_INIT_FILES_V1,
    output_package = "keras.api",
    deps = [
        "//keras",
        "//:expect_tensorflow_installed",
        # "//third_party/tensorflow/lite/python:analyzer",
        # "//third_party/tensorflow/lite/python:lite",
        # "//third_party/tensorflow/lite/python/authoring",
    ],
)

generate_apis(
    name = "keras_python_api_gen_compat_v1",
    api_version = 1,
    output_dir = "_v1/",
    output_files = KERAS_API_INIT_FILES_V1,
    output_package = "keras.api._v1",
    deps = [
        "//keras",
        "//:expect_tensorflow_installed",
        # "//third_party/tensorflow/lite/python:analyzer",
        # "//third_party/tensorflow/lite/python:lite",
        # "//third_party/tensorflow/lite/python/authoring",
    ],
)

generate_apis(
    name = "keras_python_api_gen_compat_v2",
    api_version = 2,
    output_dir = "_v2/",
    output_files = KERAS_API_INIT_FILES,
    output_package = "keras.api._v2",
    deps = [
        "//keras",
        "//:expect_tensorflow_installed",
        # "//third_party/tensorflow/lite/python:analyzer",
        # "//third_party/tensorflow/lite/python:lite",
        # "//third_party/tensorflow/lite/python/authoring",
    ],
)
