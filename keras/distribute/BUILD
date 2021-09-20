# Description:
#   keras/distribute package is intended to serve as the centralized place for things
#   related to dist-strat used by Keras..

load("@org_keras//keras:keras.bzl", "distribute_py_test")
load("@org_keras//keras:keras.bzl", "cuda_py_test")
load("@org_keras//keras:keras.bzl", "tf_py_test")  # buildifier: disable=same-origin-load

package(
    # TODO(scottzhu): Remove this deps when distribute test are converted to integration test.
    default_visibility = [
        "//keras:__subpackages__",
        "//third_party/tensorflow/python/distribute:__pkg__",
        "//third_party/tensorflow/tools/pip_package:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "distribute",
    srcs = [
        "__init__.py",
        "distributed_training_utils.py",
        "distributed_training_utils_v1.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":distribute_coordinator_utils",
        ":sidecar_evaluator",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras:callbacks",
        "//keras:callbacks_v1",
        "//keras:constraints",
        "//keras:losses",
        "//keras:optimizers",
        "//keras:regularizers",
        "//keras/initializers",
        "//keras/mixed_precision:policy",
        "//keras/utils:engine_utils",
        "//keras/utils:mode_keys",
    ],
)

py_library(
    name = "distribute_test_lib_pip",
    srcs_version = "PY3",
    deps = [
        ":dataset_creator_model_fit_test_base",
        ":distribute_strategy_test_lib",
        ":keras_correctness_test_lib",
        ":keras_test_lib",
        ":model_combinations",
        ":multi_worker_testing_utils",
        ":saved_model_test_base",
        ":test_example",
    ],
)

py_library(
    name = "optimizer_combinations",
    srcs = ["optimizer_combinations.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/optimizer_v2",
    ],
)

py_library(
    name = "worker_training_state",
    srcs = [
        "worker_training_state.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":distributed_file_utils",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/utils:mode_keys",
    ],
)

py_library(
    name = "model_collection_base",
    srcs = ["model_collection_base.py"],
    srcs_version = "PY3",
)

py_library(
    name = "model_combinations",
    srcs = ["model_combinations.py"],
    srcs_version = "PY3",
    deps = [
        ":simple_models",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "simple_models",
    srcs = ["simple_models.py"],
    srcs_version = "PY3",
    deps = [
        ":model_collection_base",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

py_library(
    name = "saved_model_test_base",
    srcs = ["saved_model_test_base.py"],
    srcs_version = "PY3",
    deps = [
        ":model_combinations",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

cuda_py_test(
    name = "worker_training_state_test",
    srcs = ["worker_training_state_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":multi_worker_testing_utils",
        ":worker_training_state",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

distribute_py_test(
    name = "checkpointing_test",
    srcs = ["checkpointing_test.py"],
    main = "checkpointing_test.py",
    tags = [
        "multi_and_single_gpu",
        "nomultivm",  # TODO(b/170502145)
    ],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/optimizer_v2",
    ],
)

cuda_py_test(
    name = "collective_all_reduce_strategy_test",
    srcs = ["collective_all_reduce_strategy_test.py"],
    python_version = "PY3",
    tags = [
        "multi_and_single_gpu",
        "nomsan",  # TODO(b/162894966)
        "notsan",  # TODO(b/171040408): data race
    ],
    # b/155301154 broken with XLA:GPU
    xla_enable_strict_auto_jit = True,
    deps = [
        "//:expect_absl_installed",
        "//:expect_portpicker_installed",
        "//:expect_tensorflow_installed",
        "//keras:testing_utils",
        "//keras/engine",
        "//keras/layers",
        "//keras/mixed_precision:policy",
        "//keras/mixed_precision:test_util",
    ],
)

distribute_py_test(
    name = "ctl_correctness_test",
    srcs = ["ctl_correctness_test.py"],
    main = "ctl_correctness_test.py",
    shard_count = 10,
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
        "nomultivm",  # TODO(b/170502145)
    ],
    deps = [
        ":optimizer_combinations",
        ":strategy_combinations",
        "//:expect_portpicker_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

distribute_py_test(
    name = "custom_training_loop_metrics_test",
    srcs = ["custom_training_loop_metrics_test.py"],
    disable_mlir_bridge = False,
    main = "custom_training_loop_metrics_test.py",
    tags = [
        "multi_and_single_gpu",
        "nomultivm",  # TODO(b/170502145)
    ],
    deps = [
        ":strategy_combinations",
        "//:expect_absl_installed",
        "//:expect_portpicker_installed",
        "//:expect_tensorflow_installed",
        "//keras:metrics",
    ],
)

distribute_py_test(
    name = "custom_training_loop_models_test",
    srcs = ["custom_training_loop_models_test.py"],
    main = "custom_training_loop_models_test.py",
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
        "nomultivm",  # TODO(b/170502145)
        "notsan",  # TODO(b/170954243)
    ],
    tpu_tags = [
        "no_oss",  # b/153615544.
        "notsan",  # TODO(b/170869466)
    ],
    deps = [
        ":strategy_combinations",
        "//:expect_absl_installed",
        "//:expect_portpicker_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

distribute_py_test(
    name = "custom_training_loop_optimizer_test",
    srcs = ["custom_training_loop_optimizer_test.py"],
    disable_mlir_bridge = False,
    main = "custom_training_loop_optimizer_test.py",
    tags = [
        "multi_and_single_gpu",
        "nomultivm",  # TODO(b/170502145)
    ],
    deps = [
        ":strategy_combinations",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras/optimizer_v2",
    ],
)

py_library(
    name = "distribute_strategy_test_lib",
    srcs = [
        "distribute_strategy_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":multi_worker_testing_utils",
        ":optimizer_combinations",
        ":strategy_combinations",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

distribute_py_test(
    name = "keras_premade_models_test",
    size = "medium",
    srcs = ["keras_premade_models_test.py"],
    disable_mlir_bridge = False,
    full_precision = True,
    main = "keras_premade_models_test.py",
    shard_count = 8,
    tags = [
        "multi_and_single_gpu",
        "nomultivm",  # TODO(b/170502145)
    ],
    deps = [
        ":distribute_strategy_test_lib",
        ":keras_correctness_test_lib",
        "//:expect_portpicker_installed",
    ],
)

distribute_py_test(
    name = "distribute_strategy_test",
    srcs = ["distribute_strategy_test.py"],
    disable_mlir_bridge = True,  # TODO(b/170352626)
    full_precision = True,
    main = "distribute_strategy_test.py",
    python_version = "PY3",
    shard_count = 20,
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # TODO(b/182391774)
        "no_oss",  # TODO(b/191770103)
        "no_rocm",  # times out on ROCm
        "no_windows_gpu",
        "noguitar",  # TODO(b/172354344)
        "nomultivm",  # TODO(b/170502145)
        "notpu",  # b/188061768
        "notsan",
    ],
    tpu_tags = [
        "no_oss",  # b/155502591
    ],
    deps = [
        ":distribute_strategy_test_lib",
        ":optimizer_combinations",
        "//:expect_portpicker_installed",
    ],
)

distribute_py_test(
    name = "distributed_training_utils_test",
    srcs = ["distributed_training_utils_test.py"],
    disable_mlir_bridge = False,
    full_precision = True,
    main = "distributed_training_utils_test.py",
    tags = [
        "nomultivm",  # TODO(b/170502145)
    ],
    deps = [
        ":distribute",
        ":distribute_strategy_test_lib",
        "//:expect_tensorflow_installed",
        "//keras:callbacks",
    ],
)

py_library(
    name = "keras_correctness_test_lib",
    srcs = [
        "keras_correctness_test_base.py",
        "keras_dnn_correctness_test.py",
        "keras_embedding_model_correctness_test.py",
        "keras_image_model_correctness_test.py",
        "keras_rnn_model_correctness_test.py",
        "keras_stateful_lstm_model_correctness_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":strategy_combinations",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:backend",
    ],
)

distribute_py_test(
    name = "keras_dnn_correctness_test",
    size = "medium",
    srcs = ["keras_dnn_correctness_test.py"],
    disable_mlir_bridge = True,  # TODO(b/170352626)
    full_precision = True,
    main = "keras_dnn_correctness_test.py",
    # Shard count is set to an odd number to distribute tasks across
    # shards more evenly.
    shard_count = 19,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/173021094)
        "no_rocm",  # times out on ROCm
        "no_windows_gpu",
        "nogpu",  # TODO(b/170905292)
        "nomultivm",  # TODO(b/170502145)
        "notap",  # TODO(b/178803051): flaky
        "notsan",
    ],
    deps = [
        ":keras_correctness_test_lib",
        "//:expect_portpicker_installed",
    ],
)

distribute_py_test(
    name = "keras_embedding_model_correctness_test",
    size = "medium",
    srcs = ["keras_embedding_model_correctness_test.py"],
    disable_mlir_bridge = True,  # TODO(b/170352626)
    full_precision = True,
    main = "keras_embedding_model_correctness_test.py",
    shard_count = 8,
    tags = [
        "broken",  # b/170975619
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
        "no_rocm",
        "no_windows_gpu",
        "nomultivm",  # TODO(b/170502145)
        "notsan",
    ],
    deps = [
        ":keras_correctness_test_lib",
        "//:expect_portpicker_installed",
    ],
)

distribute_py_test(
    name = "keras_image_model_correctness_test",
    size = "medium",
    srcs = ["keras_image_model_correctness_test.py"],
    disable_mlir_bridge = True,  # TODO(b/170352626)
    full_precision = True,
    main = "keras_image_model_correctness_test.py",
    shard_count = 16,
    tags = [
        "multi_and_single_gpu",
        "no_rocm",  # times out on ROCm
        "no_windows_gpu",
        "noasan",  # TODO(b/337374867) fails with -fsanitize=null
        "nomultivm",  # TODO(b/170502145)
        "notsan",
    ],
    xla_enable_strict_auto_jit = False,  # Tensorflow also fails.
    deps = [
        ":keras_correctness_test_lib",
        "//:expect_portpicker_installed",
    ],
)

distribute_py_test(
    name = "keras_metrics_test",
    srcs = ["keras_metrics_test.py"],
    disable_mlir_bridge = False,
    main = "keras_metrics_test.py",
    shard_count = 8,
    tags = [
        "multi_and_single_gpu",
        "nomultivm",  # TODO(b/170502145)
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras:metrics",
    ],
)

distribute_py_test(
    name = "keras_models_test",
    srcs = ["keras_models_test.py"],
    main = "keras_models_test.py",
    tags = [
        "multi_and_single_gpu",
        "nomultivm",  # TODO(b/170502145)
    ],
    deps = [
        ":strategy_combinations",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

distribute_py_test(
    name = "keras_rnn_model_correctness_test",
    size = "medium",
    srcs = ["keras_rnn_model_correctness_test.py"],
    full_precision = True,
    main = "keras_rnn_model_correctness_test.py",
    # Shard count is set to an odd number to distribute tasks across
    # shards more evenly.
    shard_count = 31,
    tags = [
        "multi_and_single_gpu",
        "no_rocm",  # Would require size large, but that effectively disables the test for presubmits.
        "no_windows_gpu",
        "noasan",  # TODO(b/337374867) fails with -fsanitize=null
        "nomultivm",  # TODO(b/170502145)
        "notpu",  # TODO(b/153672562)
        "notsan",
    ],
    deps = [
        ":keras_correctness_test_lib",
        "//:expect_portpicker_installed",
    ],
)

distribute_py_test(
    name = "keras_save_load_test",
    size = "medium",
    srcs = ["keras_save_load_test.py"],
    full_precision = True,
    main = "keras_save_load_test.py",
    shard_count = 7,
    tags = [
        "multi_and_single_gpu",
        "no_rocm",
        "nomultivm",  # TODO(b/170502145)
    ],
    deps = [
        ":saved_model_test_base",
        "//keras/saving",
    ],
)

distribute_py_test(
    name = "keras_stateful_lstm_model_correctness_test",
    size = "medium",
    srcs = ["keras_stateful_lstm_model_correctness_test.py"],
    disable_mlir_bridge = False,
    full_precision = True,
    main = "keras_stateful_lstm_model_correctness_test.py",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "no_pip",
        "no_windows_gpu",
        "nomultivm",  # TODO(b/170502145)
        "notsan",
    ],
    deps = [
        ":keras_correctness_test_lib",
    ],
)

distribute_py_test(
    name = "keras_utils_test",
    srcs = ["keras_utils_test.py"],
    disable_mlir_bridge = True,  # TODO(b/170352626)
    full_precision = True,
    main = "keras_utils_test.py",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
        "no_windows_gpu",
        "nomultivm",  # TODO(b/170502145)
        "notsan",
    ],
    deps = [
        ":distribute_strategy_test_lib",
        ":keras_test_lib",
        ":optimizer_combinations",
        "//:expect_portpicker_installed",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "keras_test_lib",
    srcs = [
        "keras_utils_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

cuda_py_test(
    name = "keras_optimizer_v2_test",
    srcs = ["keras_optimizer_v2_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "tf_integration_test",
    ],
    deps = [
        ":keras_test_lib",
    ],
)

distribute_py_test(
    name = "minimize_loss_test",
    srcs = ["minimize_loss_test.py"],
    main = "minimize_loss_test.py",
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/139815303): enable after this is fixed.
        "noguitar",  # TODO(b/140755528): enable after this is fixed.
        "nomultivm",  # TODO(b/170502145)
        "notap",  # TODO(b/139815303): enable after this is fixed.
    ],
    deps = [
        ":optimizer_combinations",
        ":test_example",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras/layers",
    ],
)

cuda_py_test(
    name = "mirrored_strategy_test",
    srcs = ["mirrored_strategy_test.py"],
    python_version = "PY3",
    tags = [
        "multi_and_single_gpu",
        "no_windows_gpu",  # TODO(b/130551176)
    ],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/engine",
        "//keras/layers/core",
        "//keras/utils:kpl_test_utils",
    ],
)

cuda_py_test(
    name = "mirrored_variable_test",
    srcs = ["mirrored_variable_test.py"],
    python_version = "PY3",
    tags = [
        "guitar",
        "multi_and_single_gpu",
    ],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:metrics",
        "//keras/layers/core",
    ],
)

cuda_py_test(
    name = "multi_worker_test",
    srcs = ["multi_worker_test.py"],
    python_version = "PY3",
    shard_count = 2,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/130369494): Investigate why it times out on OSS.
    ],
    deps = [
        ":multi_worker_testing_utils",
        "//:expect_absl_installed",
        "//:expect_portpicker_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:backend",
        "//keras:callbacks",
        "//keras:engine",
        "//keras:optimizers",
        "//keras/optimizer_v2",
        "//keras/utils:kpl_test_utils",
    ],
)

tf_py_test(
    name = "multi_worker_callback_tf2_test",
    srcs = ["multi_worker_callback_tf2_test.py"],
    python_version = "PY3",
    shard_count = 5,
    tags = [
        "no_windows",  # TODO(b/184424727): Re-enable this.
    ],
    deps = [
        ":distributed_file_utils",
        ":multi_worker_testing_utils",
        "//:expect_portpicker_installed",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "multi_worker_testing_utils",
    srcs = [
        "multi_worker_testing_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/optimizer_v2",
    ],
)

py_library(
    name = "tpu_strategy_test_utils",
    srcs = ["tpu_strategy_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

distribute_py_test(
    name = "saved_model_save_load_test",
    size = "medium",
    srcs = ["saved_model_save_load_test.py"],
    full_precision = True,
    main = "saved_model_save_load_test.py",
    shard_count = 7,
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # times out
        "no_rocm",
        "nomultivm",  # TODO(b/170502145)
    ],
    deps = [
        ":saved_model_test_base",
        "//:expect_tensorflow_installed",
    ],
)

distribute_py_test(
    name = "saved_model_mixed_api_test",
    size = "medium",
    srcs = ["saved_model_mixed_api_test.py"],
    full_precision = True,
    main = "saved_model_mixed_api_test.py",
    shard_count = 7,
    tags = [
        "multi_and_single_gpu",
        "no_rocm",
        "nomultivm",  # TODO(b/170502145)
    ],
    deps = [
        ":saved_model_test_base",
        "//:expect_tensorflow_installed",
        "//keras/saving",
    ],
)

distribute_py_test(
    name = "sharded_variable_test",
    srcs = ["sharded_variable_test.py"],
    python_version = "PY3",
    shard_count = 2,
    tags = [
        "multi_and_single_gpu",
        "no_rocm",
        "nomultivm",  # TODO(b/170502145)
    ],
    deps = [
        ":multi_worker_testing_utils",
        "//:expect_tensorflow_installed",
        "//keras/engine:base_layer",
    ],
)

distribute_py_test(
    name = "parameter_server_evaluation_test",
    srcs = ["parameter_server_evaluation_test.py"],
    python_version = "PY3",
    shard_count = 1,
    tags = [
        "multi_and_single_gpu",
        "no_cuda_asan",  # TODO(b/186361027)
        "no_oss",  # TODO(b/186248973)
        "no_tfrt",
        "nomultivm",  # TODO(b/170502145)
        "notpu",
    ],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/utils:dataset_creator",
    ],
)

distribute_py_test(
    name = "dataset_creator_model_fit_test",
    srcs = ["dataset_creator_model_fit_test.py"],
    disable_mlir_bridge = True,  # TODO(b/170352626)
    disable_tpu_use_tfrt = True,  # TODO(b/195081590)
    full_precision = True,
    main = "dataset_creator_model_fit_test.py",
    shard_count = 50,
    tags = [
        "multi_gpu",
        "no_oss",  # TODO(b/183640564): Reenable
        "no_rocm",
        "nomultivm",  # TODO(b/170502145)
        "notsan",  # TODO(b/184542721)
    ],
    deps = [
        ":dataset_creator_model_fit_test_base",
        ":strategy_combinations",
        "//:expect_portpicker_installed",
        "//:expect_tensorflow_installed",
        "//keras:callbacks",
    ],
)

distribute_py_test(
    name = "dataset_creator_model_fit_ps_only_test",
    size = "medium",
    srcs = ["dataset_creator_model_fit_ps_only_test.py"],
    disable_mlir_bridge = True,  # TODO(b/170352626)
    full_precision = True,
    main = "dataset_creator_model_fit_ps_only_test.py",
    shard_count = 21,
    tags = [
        "multi_gpu",
        "no_oss",  # TODO(b/183640564): Reenable
        "no_rocm",
        "nomultivm",  # TODO(b/170502145)
        "notsan",  # TODO(b/184542721)
    ],
    deps = [
        ":dataset_creator_model_fit_test_base",
        ":strategy_combinations",
        "//:expect_tensorflow_installed",
        "//keras:callbacks",
    ],
)

py_library(
    name = "distributed_file_utils",
    srcs = [
        "distributed_file_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "distributed_file_utils_test",
    srcs = ["distributed_file_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":distributed_file_utils",
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "sidecar_evaluator",
    srcs = ["sidecar_evaluator.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorboard_installed",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "sidecar_evaluator_test",
    size = "medium",
    srcs = ["sidecar_evaluator_test.py"],
    python_version = "PY3",
    deps = [
        ":sidecar_evaluator",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

py_library(
    name = "strategy_combinations",
    srcs = ["strategy_combinations.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "test_example",
    srcs = ["test_example.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

py_library(
    name = "distribute_coordinator_utils",
    srcs = [
        "distribute_coordinator_utils.py",
    ],
    srcs_version = "PY3",
    deps = ["//:expect_tensorflow_installed"],
)

py_library(
    name = "dataset_creator_model_fit_test_base",
    srcs = [
        "dataset_creator_model_fit_test_base.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":multi_worker_testing_utils",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:callbacks",
        "//keras/engine",
        "//keras/layers/core",
        "//keras/layers/preprocessing:string_lookup",
        "//keras/optimizer_v2",
        "//keras/utils:dataset_creator",
    ],
)
