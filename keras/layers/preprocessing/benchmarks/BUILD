# Benchmarks for Keras preprocessing layers.
load("@org_keras//keras:keras.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//keras/google/private_tf_api_test:__pkg__"],
)

tf_py_test(
    name = "category_encoding_benchmark",
    srcs = ["category_encoding_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/api:keras_api",
        "//keras/layers/preprocessing:category_encoding",
    ],
)

tf_py_test(
    name = "category_crossing_benchmark",
    srcs = ["category_crossing_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/api:keras_api",
        "//keras/layers/preprocessing:category_crossing",
    ],
)

tf_py_test(
    name = "hashing_benchmark",
    srcs = ["hashing_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/api:keras_api",
        "//keras/layers/preprocessing:hashing",
    ],
)

tf_py_test(
    name = "index_lookup_adapt_benchmark",
    srcs = ["index_lookup_adapt_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/api:keras_api",
        "//keras/layers/preprocessing:index_lookup",
    ],
)

tf_py_test(
    name = "index_lookup_forward_benchmark",
    srcs = ["index_lookup_forward_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/api:keras_api",
        "//keras/layers/preprocessing:index_lookup",
    ],
)

tf_py_test(
    name = "normalization_adapt_benchmark",
    srcs = ["normalization_adapt_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/api:keras_api",
        "//keras/layers/preprocessing:normalization",
    ],
)

tf_py_test(
    name = "discretization_adapt_benchmark",
    srcs = ["discretization_adapt_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/api:keras_api",
        "//keras/layers/preprocessing:discretization",
    ],
)

cuda_py_test(
    name = "image_preproc_benchmark",
    srcs = ["image_preproc_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/api:keras_api",
        "//keras/layers/preprocessing:image_preprocessing",
    ],
)
