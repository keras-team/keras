# Copyright 2023 The JAX Authors.
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

"""Python definitions for third_party.jax.jaxlib.mosaic.python.tpu

These definitions are needed internally by the tpu module.
TODO(tlongeri): Migrate definitions to tpu module
"""
import collections
import enum
from typing import Literal

TargetTuple = collections.namedtuple("TargetTuple", ["sublanes", "lanes"])

@enum.unique
class Direction(enum.Enum):
  SUBLANES = "sublanes"
  LANES = "lanes"
  SUBELEMENTS = "subelements"

  def __repr__(self):
    return self.name.lower()
SUBLANES = Direction.SUBLANES
LANES = Direction.LANES
SUBELEMENTS = Direction.SUBELEMENTS


class Replicated(enum.Enum):
  REPLICATED = "*"

  def __repr__(self):
    return "*"
  __str__ = __repr__

  def __bool__(self):
    return False  # Useful because we can then say `offset or 0`
REPLICATED = Replicated.REPLICATED
Offset = int | Literal[REPLICATED]


class ImplicitDim(enum.IntEnum):
  MINOR = -1
  SECOND_MINOR = -2

  def __repr__(self) -> str:
    return str(int(self))
