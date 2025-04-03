# Copyright 2024 The etils Authors.
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

"""Multiprocessing utilities."""

import atexit
import os
import sys


def is_ipython_subprocess() -> bool:
  """Check if we are in a sub-process launched from within a `ipython` terminal.

  Returns:
    `True` only if we are in ipython terminal (e.g. `ml_python`) and inside
    a sub-process.
  """
  return False


def register_adhoc_init() -> None:
  """Registers a callback to activate adhoc imports in sub-processes."""
