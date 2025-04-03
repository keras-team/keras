# Copyright 2021 The JAX Authors.
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

from __future__ import annotations

import abc

from jax._src import path as pathlib
from jax._src import util


class CacheInterface(util.StrictABC):
  _path: pathlib.Path

  @abc.abstractmethod
  def get(self, key: str):
    pass

  @abc.abstractmethod
  def put(self, key: str, value: bytes):
    pass
