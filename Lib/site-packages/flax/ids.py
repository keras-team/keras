# Copyright 2024 The Flax Authors.
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

"""UUIDs for Flax internals."""

import threading


class UUIDManager:
  """Globally unique counter-based id manager.

  We need globally unique key ids for Module and Variable object instances
  to preserve and recreate sharing-by-reference relationship when lifting
  transforms and adopting outside Modules.
  - Use of id() is unacceptable because these identifiers are literally
    pointers which can be recycled, so we rely on a globally unique counter id
    instead.
  - We need to handle copy/deepcopy uniqueness via a wrapped type.
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._id = 0

  def __call__(self):
    with self._lock:
      self._id += 1
      return FlaxId(self._id)


uuid = UUIDManager()


class FlaxId:
  """Hashable wrapper for ids that handles uniqueness of copies."""

  def __init__(self, rawid):
    self.id = rawid

  def __eq__(self, other):
    return isinstance(other, FlaxId) and other.id == self.id

  def __hash__(self):
    return hash(self.id)

  def __repr__(self):
    return f'FlaxId({self.id})'

  def __deepcopy__(self, memo):
    del memo
    return uuid()

  def __copy__(self):
    return uuid()
