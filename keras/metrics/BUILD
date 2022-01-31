# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
#   Contains the Keras metrics submodule.

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras:friends",
        "//third_party/tensorflow/python/feature_column:__subpackages__",
        "//third_party/tensorflow/python/tpu:__subpackages__",
        "//third_party/tensorflow_estimator:__subpackages__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "metrics",
    srcs = [
        "__init__.py",
        "base_metric.py",
        "metrics.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:activations",
        "//keras:backend",
        "//keras:losses",
        "//keras/distribute",
        "//keras/engine:base_layer",
        "//keras/engine:base_layer_utils",
        "//keras/utils:generic_utils",
        "//keras/utils:metrics_utils",
        "//keras/utils:tf_utils",
    ],
)

tf_py_test(
    name = "metrics_functional_test",
    size = "small",
    srcs = ["metrics_functional_test.py"],
    python_version = "PY3",
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)

tf_py_test(
    name = "metrics_test",
    size = "medium",
    srcs = ["metrics_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":metrics",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras:testing_utils",
        "//keras/layers",
    ],
)

tf_py_test(
    name = "base_metric_test",
    size = "medium",
    srcs = ["base_metric_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":metrics",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras:testing_utils",
        "//keras/layers",
    ],
)

tf_py_test(
    name = "confusion_matrix_test",
    size = "medium",
    srcs = ["confusion_matrix_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":metrics",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_scipy_installed",
        "//:expect_tensorflow_installed",
        "//keras:combinations",
        "//keras:models",
        "//keras/layers",
        "//keras/utils:metrics_utils",
    ],
)

tf_py_test(
    name = "metrics_correctness_test",
    size = "medium",
    srcs = ["metrics_correctness_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)
