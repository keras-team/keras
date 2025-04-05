# Copyright 2024 The JAX Authors.
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
"""Backend for colocated_python.func."""

from __future__ import annotations

import threading
from typing import Sequence

import jax


class _ResultStore:
  """Temporarily stores results from synchronous execution of functions."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._storage: dict[int, Sequence[jax.Array]] = {}

  def push(self, uid: int, out: Sequence[jax.Array]) -> None:
    with self._lock:
      if uid in self._storage:
        raise ValueError(f"uid {uid} already exists")
      self._storage[uid] = out

  def pop(self, uid: int) -> Sequence[jax.Array]:
    with self._lock:
      if uid not in self._storage:
        raise ValueError(f"uid {uid} does not exist")
      return self._storage.pop(uid)


SINGLETON_RESULT_STORE = _ResultStore()
