# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Python bindings to an experimental TensorBoard data server."""

import os

# Version of this Python package. This may differ from the versions of
# both TensorBoard and the data server.
__version__ = "0.7.2"


def server_binary():
    """Return the path to a TensorBoard data server binary, if possible.

    Returns:
      A string path on disk, or `None` if there is no binary bundled
      with this package.
    """
    path = os.path.join(os.path.dirname(__file__), "bin", "server")
    if not os.path.exists(path):
        return None
    return path
