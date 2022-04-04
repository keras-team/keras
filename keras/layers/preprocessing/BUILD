# Description:
#   Contains the Keras preprocess layers (internal TensorFlow version).

load("@org_keras//keras:keras.bzl", "tf_py_test")

# buildifier: disable=same-origin-load
load("@org_keras//keras:keras.bzl", "cuda_py_test")
load("@org_keras//keras:keras.bzl", "distribute_py_test")

package(
    default_visibility = [
        "//keras:friends",
        "//third_party/tensorflow/tools/pip_package:__pkg__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "preprocessing",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":discretization",
        ":hashed_crossing",
        ":hashing",
        ":image_preprocessing",
        ":integer_lookup",
        ":normalization",
        ":preprocessing_stage",
        ":preprocessing_test_utils",
        ":string_lookup",
        ":text_vectorization",
    ],
)

py_library(
    name = "discretization",
    srcs = [
        "discretization.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":preprocessing_utils",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/engine",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "hashing",
    srcs = [
        "hashing.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/engine",
    ],
)

py_library(
    name = "hashed_crossing",
    srcs = [
        "hashed_crossing.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":preprocessing_utils",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/engine:base_preprocessing_layer",
        "//keras/utils:layer_utils",
    ],
)

py_library(
    name = "image_preprocessing",
    srcs = [
        "image_preprocessing.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine",
        "//keras/engine:input_spec",
        "//keras/preprocessing:image",
        "//keras/utils:image_utils",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "index_lookup",
    srcs = [
        "index_lookup.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":category_encoding",
        ":preprocessing_utils",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine",
    ],
)

py_library(
    name = "normalization",
    srcs = [
        "normalization.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":preprocessing_utils",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine",
    ],
)

py_library(
    name = "integer_lookup",
    srcs = [
        "integer_lookup.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":index_lookup",
        "//:expect_tensorflow_installed",
        "//keras/engine",
    ],
)

py_library(
    name = "text_vectorization",
    srcs = [
        "text_vectorization.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":category_encoding",
        ":preprocessing_utils",
        ":string_lookup",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine",
        "//keras/utils:layer_utils",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "category_encoding",
    srcs = [
        "category_encoding.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine",
        "//keras/engine:input_spec",
        "//keras/utils:layer_utils",
    ],
)

py_library(
    name = "string_lookup",
    srcs = [
        "string_lookup.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":index_lookup",
        "//:expect_tensorflow_installed",
        "//keras/engine",
    ],
)

py_library(
    name = "preprocessing_stage",
    srcs = [
        "preprocessing_stage.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/engine",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "preprocessing_test_utils",
    srcs = ["preprocessing_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "preprocessing_utils",
    srcs = ["preprocessing_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "preprocessing_utils_test",
    srcs = ["preprocessing_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":preprocessing_utils",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/utils:generic_utils",
    ],
)

tf_py_test(
    name = "category_encoding_test",
    srcs = ["category_encoding_test.py"],
    python_version = "PY3",
    deps = [
        ":category_encoding",
        ":preprocessing_test_utils",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/utils:generic_utils",
    ],
)

distribute_py_test(
    name = "category_encoding_distribution_test",
    srcs = ["category_encoding_distribution_test.py"],
    disable_mlir_bridge = False,
    main = "category_encoding_distribution_test.py",
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # b/189866692
        "noguitar",  # b/190034522
        "nomultivm",  # TODO(b/170502145)
    ],
    tpu_tags = [
        "no_oss",  # b/155502591
    ],
    deps = [
        ":category_encoding",
        ":preprocessing_test_utils",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:backend",
        "//keras/distribute:strategy_combinations",
        "//keras/testing_infra:test_combinations",
    ],
)

distribute_py_test(
    name = "image_preprocessing_distribution_test",
    srcs = ["image_preprocessing_distribution_test.py"],
    main = "image_preprocessing_distribution_test.py",
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "nomultivm",  # TODO(b/170502145)
        "notpu",  # TODO(b/210148622)
    ],
    tpu_tags = [
        "no_oss",
        "noguitar",  # TODO(b/183957207)
    ],
    deps = [
        ":image_preprocessing",
        ":preprocessing_test_utils",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/distribute:strategy_combinations",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "discretization_test",
    srcs = ["discretization_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = ["no_rocm"],
    deps = [
        ":discretization",
        ":preprocessing_test_utils",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

distribute_py_test(
    name = "discretization_distribution_test",
    srcs = ["discretization_distribution_test.py"],
    main = "discretization_distribution_test.py",
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/189956080)
        "noguitar",  # b/190034522
        "nomultivm",  # TODO(b/170502145)
    ],
    deps = [
        ":discretization",
        ":preprocessing_test_utils",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/distribute:strategy_combinations",
        "//keras/testing_infra:test_combinations",
    ],
)

cuda_py_test(
    name = "hashing_test",
    srcs = ["hashing_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":hashing",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/engine",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

distribute_py_test(
    name = "hashing_distribution_test",
    srcs = ["hashing_distribution_test.py"],
    disable_mlir_bridge = False,
    main = "hashing_distribution_test.py",
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "nomultivm",  # TODO(b/170502145)
    ],
    deps = [
        ":hashing",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/distribute:strategy_combinations",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "hashed_crossing_test",
    srcs = ["hashed_crossing_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":hashing",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/engine",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
    ],
)

tf_py_test(
    name = "index_lookup_test",
    srcs = ["index_lookup_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = ["noasan"],  # TODO(b/183961255)
    deps = [
        ":index_lookup",
        ":preprocessing_test_utils",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/utils:generic_utils",
    ],
)

distribute_py_test(
    name = "index_lookup_distribution_test",
    srcs = ["index_lookup_distribution_test.py"],
    disable_mlir_bridge = False,
    main = "index_lookup_distribution_test.py",
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "nomultivm",  # TODO(b/170502145)
    ],
    tpu_tags = ["no_oss"],
    deps = [
        ":index_lookup",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/distribute:strategy_combinations",
        "//keras/testing_infra:test_combinations",
    ],
)

cuda_py_test(
    name = "image_preprocessing_test",
    srcs = ["image_preprocessing_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_windows",  # TODO(b/184424727): Re-enable this.
    ],
    deps = [
        ":image_preprocessing",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/engine",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
        "//keras/utils:generic_utils",
    ],
)

tf_py_test(
    name = "normalization_test",
    srcs = ["normalization_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "noasan",  # TODO(b/337374867) fails with -fsanitize=null
    ],
    deps = [
        ":normalization",
        ":preprocessing_test_utils",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "integer_lookup_test",
    srcs = ["integer_lookup_test.py"],
    python_version = "PY3",
    tags = ["noasan"],  # TODO(b/183961255)
    deps = [
        ":integer_lookup",
        ":preprocessing_test_utils",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/utils:generic_utils",
    ],
)

distribute_py_test(
    name = "normalization_distribution_test",
    srcs = ["normalization_distribution_test.py"],
    main = "normalization_distribution_test.py",
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_oss",
        "nomultivm",  # TODO(b/170502145)
    ],
    deps = [
        ":normalization",
        ":preprocessing_test_utils",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/distribute:strategy_combinations",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "text_vectorization_test",
    srcs = ["text_vectorization_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":preprocessing_test_utils",
        ":text_vectorization",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/utils:generic_utils",
    ],
)

distribute_py_test(
    name = "text_vectorization_distribution_test",
    srcs = ["text_vectorization_distribution_test.py"],
    disable_mlir_bridge = False,
    main = "text_vectorization_distribution_test.py",
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "nomultivm",  # TODO(b/170502145)
    ],
    tpu_tags = [
        "no_oss",  # b/155502591
    ],
    deps = [
        ":preprocessing_test_utils",
        ":text_vectorization",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/distribute:strategy_combinations",
        "//keras/testing_infra:test_combinations",
    ],
)

tf_py_test(
    name = "string_lookup_test",
    srcs = ["string_lookup_test.py"],
    python_version = "PY3",
    tags = [
        "notsan",  #b/168758821
    ],
    deps = [
        ":preprocessing_test_utils",
        ":string_lookup",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
        "//keras/utils:generic_utils",
    ],
)

tf_py_test(
    name = "preprocessing_stage_test",
    srcs = ["preprocessing_stage_test.py"],
    python_version = "PY3",
    tags = ["no_windows"],  # TODO(b/152991402)
    deps = [
        ":preprocessing_stage",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/testing_infra:test_combinations",
    ],
)
