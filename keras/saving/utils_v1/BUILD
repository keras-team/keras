# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#   Keras saving and loading libraries.

# buildifier: disable=same-origin-load

package(
    default_visibility = [
        "//keras:__subpackages__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "utils_v1",
    srcs = [
        "__init__.py",
        "export_output.py",
        "export_utils.py",
        "mode_keys.py",
        "signature_def_utils.py",
        "unexported_constants.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//:expect_tensorflow_installed",
    ],
)
