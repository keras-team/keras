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

"""Container for stat objects."""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class StatResult:
  """Results of `path.stat()`.

  Attributes:
    is_directory: Whether the file is a directory
    length: Size in bytes
    mtime: Time since last modification (in sec since the epoch)
    owner: the owner of the path.
    group: the group of the path.
    mode: the st_mode of the path.
  """

  is_directory: bool
  length: int
  mtime: int
  owner: Optional[str] = None
  group: Optional[str] = None
  mode: Optional[int] = None
