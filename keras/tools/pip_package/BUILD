package(default_visibility = ["//keras:__subpackages__"])

# Description:
#  Tools for building the TensorFlow pip package.

COMMON_PIP_DEPS = [
    "//keras",
    # The following targets are not included by //keras:keras,
    # eg to avoid circular dependency with TF, but they should still be included
    # in the PIP package.
    "//keras/legacy_tf_layers:convolutional",
    "//keras/legacy_tf_layers:core",
    "//keras/legacy_tf_layers:layers_base",
    "//keras/legacy_tf_layers:normalization",
    "//keras/legacy_tf_layers:pooling",
    "//keras/layers/legacy_rnn:rnn_cell_impl",
    "//keras/layers/legacy_rnn:rnn_cell_wrapper_impl",
    "//keras/optimizer_v2:legacy_learning_rate_decay",
    "//keras/type:types",
]

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = COMMON_PIP_DEPS,
)
