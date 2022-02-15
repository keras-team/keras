package(default_visibility = ["//keras:__subpackages__"])

# Description:
#  Tools for building the TensorFlow pip package.

COMMON_PIP_DEPS = [
    "//keras/api:keras_api",
    # The following targets are not included by //keras:keras,
    # eg to avoid circular dependency with TF, but they should still be included
    # in the PIP package.
    "//keras/legacy_tf_layers:convolutional",
    "//keras/legacy_tf_layers:core",
    "//keras/legacy_tf_layers:layers_base",
    "//keras/legacy_tf_layers:normalization",
    "//keras/legacy_tf_layers:pooling",
    "//keras/layers/rnn:legacy_cell_wrappers",
    "//keras/layers/rnn:legacy_cells",
    "//keras/optimizers:legacy_learning_rate_decay",
    # Need to include testing libraries in pip package so our pip
    # release tests can run. (see py_test rule in keras.bzl for more context).
    # Essentially, everything needed to run the test (except the test file itself)
    # must be contained in the pip package since we strip away all deps.
    "//keras/testing_infra:test_combinations",
    "//keras/testing_infra:test_utils",
    "//keras/benchmarks:keras_benchmark_lib_pip",
    "//keras/distribute:distribute_test_lib_pip",
    "//keras/integration_test:preprocessing_test_utils",
    "//keras/layers/preprocessing:preprocessing_test_utils",
    "//keras/layers/preprocessing/benchmarks:feature_column_benchmark",
    "//keras/mixed_precision:test_util",
    "//keras/tests:model_architectures",
    "//keras/tests:model_subclassing_test_util",
    "//keras/utils:dataset_creator",
    "//keras/utils:kpl_test_utils",
]

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = COMMON_PIP_DEPS,
)
