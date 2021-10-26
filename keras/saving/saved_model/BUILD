# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
#   Keras saving and loading files for SavedModel.

# buildifier: disable=same-origin-load

load("@org_keras//keras:keras.bzl", "tf_py_test")

package(
    default_visibility = [
        "//keras/saving:__subpackages__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "order_preserving_set",
    srcs = ["order_preserving_set.py"],
)

py_library(
    name = "load_context",
    srcs = ["load_context.py"],
    visibility = ["//third_party/tensorflow/python/distribute:__pkg__"],
)

py_library(
    name = "saved_model",
    srcs = [
        "base_serialization.py",
        "constants.py",
        "json_utils.py",
        "layer_serialization.py",
        "load.py",
        "load_context.py",
        "metric_serialization.py",
        "model_serialization.py",
        "network_serialization.py",
        "save.py",
        "save_impl.py",
        "serialized_attributes.py",
        "utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":order_preserving_set",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "revive_test",
    size = "medium",
    srcs = ["revive_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_windows",  # b/158005583
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
    ],
)

tf_py_test(
    name = "json_utils_test",
    size = "small",
    srcs = ["json_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":saved_model",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
    ],
)

tf_py_test(
    name = "saved_model_test",
    size = "medium",
    srcs = ["saved_model_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_pip",  # TODO(b/202022379)
        "no_rocm",
        "no_windows",
        "notsan",  #TODO(b/181771982): it is flaky
    ],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
    ],
)

py_binary(
    name = "create_test_saved_model",
    srcs = ["create_test_saved_model.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_oss"],
    deps = [
        "//:expect_tensorflow_installed",
        "//keras:regularizers",
        "//keras:testing_utils",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
    ],
)

tf_py_test(
    name = "determinism_test",
    srcs = ["determinism_test.py"],
    data = [":create_test_saved_model.par"],
    python_version = "PY3",
    tags = ["no_oss"],
    deps = [
        "//:expect_tensorflow_installed",
    ],
)
