# Description:
# DynamicEmbeddingLayer allows for the continuous updating of the vocabulary and embeddings during
# the training process.

# Placeholder: load unaliased py_library
load("@org_keras//keras:keras.bzl", "tf_py_test")
load("@org_keras//keras:keras.bzl", "distribute_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//keras:license"],
    default_visibility = [
        "//keras:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "dynamic_lookup",
    srcs = ["dynamic_lookup.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/layers",
        "//third_party/tensorflow/python/util:core",
    ],
)

tf_py_test(
    name = "dynamic_lookup_test",
    size = "small",
    srcs = ["dynamic_lookup_test.py"],
    python_version = "PY3",
    shard_count = 6,
    deps = [
        ":dynamic_lookup",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
        "//testing/pybase",
    ],
)

py_library(
    name = "dynamic_embedding",
    srcs = ["dynamic_embedding.py"],
    deps = [
        ":dynamic_lookup",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/layers",
        "//keras/utils",
        "//third_party/py/absl/logging",
        "//third_party/tensorflow/python/util:core",
    ],
)

tf_py_test(
    name = "dynamic_embedding_test",
    srcs = ["dynamic_embedding_test.py"],
    deps = [
        ":dynamic_embedding",
        "//:expect_tensorflow_installed",
        "//keras:callbacks",
        "//keras/layers",
        "//keras/models",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
        "//testing/pybase",
    ],
)

distribute_py_test(
    name = "dynamic_embedding_distributed_test",
    srcs = ["dynamic_embedding_distributed_test.py"],
    tags = [
        "no_oss",
        "no_windows",
        "nomultivm",
    ],
    deps = [
        ":dynamic_embedding",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:callbacks",
    ],
)

distribute_py_test(
    name = "dynamic_lookup_distributed_test",
    srcs = ["dynamic_lookup_distributed_test.py"],
    tags = ["nomultivm"],
    deps = [
        ":dynamic_lookup",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)
