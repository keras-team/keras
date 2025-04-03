# Copyright 2024 The Orbax Authors.
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

"""Utils for path constructs."""

import os
import time
from typing import Iterable, Optional

from etils import epath



class Timer(object):
  """A simple timer to measure the time it takes to run a function."""

  def __init__(self):
    self._start = time.time()

  def get_duration(self):
    return time.time() - self._start




def recursively_copy_files(src: str, dst: str):
  """Recursively copies files from src to dst."""
  src_path = epath.Path(src)
  dst_path = epath.Path(dst)
  for root, _, files in os.walk(src_path):
    relative_path = str(root)[len(str(src_path)) :].lstrip(os.sep)
    dst_root = dst_path / relative_path
    dst_root.mkdir(parents=True, exist_ok=True)
    for file in files:
      src_file = epath.Path(root) / file
      dst_file = epath.Path(dst_root) / file
      src_file.copy(dst_file)
