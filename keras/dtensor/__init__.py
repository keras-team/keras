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
"""Keras' DTensor library."""

_DTENSOR_API_ENABLED = False


# Conditional import the dtensor API, since it is currently broken in OSS.
if _DTENSOR_API_ENABLED:
  try:
    # pylint: disable=g-direct-tensorflow-import, g-import-not-at-top
    from tensorflow.dtensor import python as dtensor_api
  except ImportError:
    # TODO(b/222341036): Remove this conditional import after dtensor have a
    # trimmed target that can be used by Keras.
    dtensor_api = None
else:
  # Leave it with a placeholder, so that the import line from other python file
  # will not break.
  dtensor_api = None


