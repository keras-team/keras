# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Description:
#   Contains the Keras Mixed Precision API (TensorFlow version).

load("@org_keras//keras:keras.bzl", "cuda_py_test")
load("@org_keras//keras:keras.bzl", "tf_py_test")  # buildifier: disable=same-origin-load

package(
    default_visibility = [
        # TODO(scottzhu): Remove these two deps and convert the test to integration test.
        "//third_party/tensorflow/python/distribute:__pkg__",  # For collective_all_reduce_strategy_test
        "//keras:__subpackages__",
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
    name = "mixed_precision_experimental",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":get_layer_policy",
        ":loss_scale_optimizer",
        ":policy",
    ],
)

py_library(
    name = "policy",
    srcs = [
        "policy.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":device_compatibility_check",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "policy_test",
    size = "medium",
    srcs = [
        "policy_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_rocm"],
    deps = [
        ":policy",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/mixed_precision:loss_scale_optimizer",
        "//keras/optimizer_v2",
    ],
)

py_library(
    name = "device_compatibility_check",
    srcs = ["device_compatibility_check.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)

cuda_py_test(
    name = "device_compatibility_check_test",
    srcs = ["device_compatibility_check_test.py"],
    srcs_version = "PY3",
    tfrt_enabled = True,
    deps = [
        ":device_compatibility_check",
        "//:expect_tensorflow_installed",
        "//keras:combinations",
    ],
)

py_library(
    name = "get_layer_policy",
    srcs = ["get_layer_policy.py"],
    srcs_version = "PY3",
    deps = [
        "//keras/engine:base_layer",
    ],
)

tf_py_test(
    name = "get_layer_policy_test",
    srcs = ["get_layer_policy_test.py"],
    srcs_version = "PY3",
    deps = [
        ":get_layer_policy",
        ":policy",
        "//:expect_tensorflow_installed",
        "//keras/layers",
    ],
)

py_library(
    name = "autocast_variable",
    srcs = [
        "autocast_variable.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/distribute",
    ],
)

tf_py_test(
    name = "autocast_variable_test",
    size = "medium",
    srcs = ["autocast_variable_test.py"],
    python_version = "PY3",
    deps = [
        ":autocast_variable",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras/optimizer_v2",
    ],
)

py_library(
    name = "loss_scale",
    srcs = ["loss_scale.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras/utils:generic_utils",
    ],
)

py_library(
    name = "loss_scale_optimizer",
    srcs = ["loss_scale_optimizer.py"],
    srcs_version = "PY3",
    deps = [
        ":loss_scale",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras/optimizer_v2",
    ],
)

cuda_py_test(
    name = "loss_scale_optimizer_test",
    size = "medium",
    srcs = ["loss_scale_optimizer_test.py"],
    python_version = "PY3",
    deps = [
        ":loss_scale_optimizer",
        ":test_util",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)

cuda_py_test(
    name = "loss_scale_benchmark",
    size = "medium",
    srcs = ["loss_scale_benchmark.py"],
    tfrt_enabled = True,
    deps = [
        ":loss_scale_optimizer",
        ":test_util",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

cuda_py_test(
    name = "mixed_precision_graph_rewrite_test",
    size = "small",
    srcs = ["mixed_precision_graph_rewrite_test.py"],
    python_version = "PY3",
    tfrt_enabled = True,
    deps = [
        ":loss_scale_optimizer",
        ":policy",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras:testing_utils",
        "//keras/optimizer_v2",
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

cuda_py_test(
    name = "layer_test",
    size = "medium",
    srcs = ["layer_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",
        "no_windows",  # b/139083295: bfloat16 tests fail on Windows
    ],
    deps = [
        ":test_util",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)

cuda_py_test(
    name = "model_test",
    size = "medium",
    srcs = ["model_test.py"],
    data = [
        "//keras/mixed_precision/testdata:lso_ckpt_tf2.2",
        "//keras/mixed_precision/testdata:lso_savedmodel_tf2.2",
    ],
    python_version = "PY3",
    shard_count = 5,
    tags = [
        "no_pip",
        "no_windows",  # b/139083295: bfloat16 tests fail on Windows
    ],
    deps = [
        ":test_util",
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)

cuda_py_test(
    name = "layer_correctness_test",
    size = "medium",
    srcs = ["layer_correctness_test.py"],
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "no_rocm",
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)
