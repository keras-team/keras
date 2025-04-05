# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import pathlib
import importlib.util

__all__ = ["Path"]

logger = logging.getLogger(__name__)

# If etils.epath (aka etils[epath] to pip) is present, we prefer it because it
# can read and write to, e.g., GCS buckets. Otherwise we use the builtin
# pathlib and can only read/write to the local filesystem.
epath_installed = bool(
    importlib.util.find_spec("etils") and
    importlib.util.find_spec("etils.epath")
)
if epath_installed:
  logger.debug("etils.epath found. Using etils.epath for file I/O.")

  def __dir__():
    return ["Path"]

  def __getattr__(name):
    if name != "Path":
      raise AttributeError(f"module '{__name__}' has no attribute '{name}")

    global Path
    from etils import epath
    Path = epath.Path
    return Path
else:
  logger.debug("etils.epath was not found. Using pathlib for file I/O.")
  Path = pathlib.Path
