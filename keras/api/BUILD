# Description:
# Package for Keras.

load("//keras/api:api_gen.bzl", "gen_api_init_files")
load("//keras/api:api_init_files.bzl", "KERAS_API_INIT_FILES", "KERAS_API_INIT_FILES_V1")

package(
    default_visibility = [
        "//keras:friends",
        "//third_party/py/tensorflow:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0 License
)

exports_files(
    [
        "create_python_api_wrapper.py",
    ],
)

keras_packages = [
    "keras",
    "keras.activations",
    "keras.applications.densenet",
    "keras.applications.efficientnet",
    "keras.applications.efficientnet_v2",
    "keras.applications.imagenet_utils",
    "keras.applications.inception_resnet_v2",
    "keras.applications.inception_v3",
    "keras.applications.mobilenet",
    "keras.applications.mobilenet_v2",
    "keras.applications.mobilenet_v3",
    "keras.applications.nasnet",
    "keras.applications.regnet",
    "keras.applications.resnet",
    "keras.applications.resnet_v2",
    "keras.applications.vgg16",
    "keras.applications.vgg19",
    "keras.applications.xception",
    "keras.backend",
    "keras.backend_config",
    "keras.callbacks",
    "keras.callbacks_v1",
    "keras.constraints",
    "keras.datasets.boston_housing",
    "keras.datasets.cifar10",
    "keras.datasets.cifar100",
    "keras.datasets.fashion_mnist",
    "keras.datasets.imdb",
    "keras.datasets.mnist",
    "keras.datasets.reuters",
    "keras.engine.base_layer",
    "keras.engine.data_adapter",
    "keras.engine.input_layer",
    "keras.engine.input_spec",
    "keras.engine.sequential",
    "keras.engine.training",
    "keras.estimator",
    "keras.feature_column.dense_features",
    "keras.feature_column.dense_features_v2",
    "keras.feature_column.sequence_feature_column",
    # Placeholder for internal API
    "keras.initializers",
    "keras.initializers.initializers_v1",
    "keras.initializers.initializers_v2",
    "keras.layers.activation",
    "keras.layers.attention",
    "keras.layers.convolutional",
    "keras.layers.core",
    "keras.layers.embeddings",
    "keras.layers.locally_connected",
    "keras.layers.merging",
    "keras.layers.normalization.batch_normalization",
    "keras.layers.normalization.batch_normalization_v1",
    "keras.layers.normalization.layer_normalization",
    "keras.layers.preprocessing",
    "keras.layers.pooling",
    "keras.layers.regularization",
    "keras.layers.rnn",
    "keras.layers.rnn.legacy_cell_wrappers",
    "keras.layers.rnn.legacy_cells",
    "keras.layers.serialization",
    "keras.legacy_tf_layers.base",
    "keras.legacy_tf_layers.convolutional",
    "keras.legacy_tf_layers.core",
    "keras.legacy_tf_layers.normalization",
    "keras.legacy_tf_layers.pooling",
    "keras.losses",
    "keras.metrics",
    "keras.mixed_precision.get_layer_policy",
    "keras.mixed_precision.loss_scale_optimizer",
    "keras.mixed_precision.policy",
    "keras.models",
    "keras.optimizers.learning_rate_schedule",
    "keras.optimizers.optimizer_v2.adadelta",
    "keras.optimizers.optimizer_v2.adagrad",
    "keras.optimizers.optimizer_v2.adam",
    "keras.optimizers.optimizer_v2.adamax",
    "keras.optimizers.optimizer_v2.ftrl",
    "keras.optimizers.optimizer_v2.gradient_descent",
    "keras.optimizers.optimizer_v2.nadam",
    "keras.optimizers.optimizer_v2.optimizer_v2",
    "keras.optimizers.optimizer_v2.rmsprop",
    "keras.optimizers",
    "keras.premade_models.linear",
    "keras.premade_models.wide_deep",
    "keras.preprocessing.image",
    "keras.preprocessing.sequence",
    "keras.preprocessing.text",
    "keras.regularizers",
    "keras.saving.model_config",
    "keras.saving.save",
    "keras.saving.saved_model_experimental",
    "keras.testing_infra.test_utils",
    "keras.utils.data_utils",
    "keras.utils.generic_utils",
    "keras.utils.io_utils",
    "keras.utils.layer_utils",
    "keras.utils.losses_utils",
    "keras.utils.np_utils",
    "keras.utils.tf_utils",
    "keras.utils.vis_utils",
    "keras.wrappers.scikit_learn",
]

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

gen_api_init_files(
    name = "keras_python_api_gen",
    api_name = "keras",
    api_version = 1,
    output_files = KERAS_API_INIT_FILES_V1,
    output_package = "keras.api",
    package_deps = [
        "//keras",
        "//:expect_tensorflow_installed",
    ],
    packages = keras_packages,
)

gen_api_init_files(
    name = "keras_python_api_gen_compat_v1",
    api_name = "keras",
    api_version = 1,
    output_dir = "_v1/",
    output_files = KERAS_API_INIT_FILES_V1,
    output_package = "keras.api._v1",
    package_deps = [
        "//keras",
        "//:expect_tensorflow_installed",
    ],
    packages = keras_packages,
)

gen_api_init_files(
    name = "keras_python_api_gen_compat_v2",
    api_name = "keras",
    api_version = 2,
    output_dir = "_v2/",
    output_files = KERAS_API_INIT_FILES,
    output_package = "keras.api._v2",
    package_deps = [
        "//keras",
        "//:expect_tensorflow_installed",
    ],
    packages = keras_packages,
)
