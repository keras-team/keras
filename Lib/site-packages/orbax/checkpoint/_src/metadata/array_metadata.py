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

"""Metadata describing Arrays. Meant to be used internally."""

import dataclasses
from typing import Any, Dict, TypeAlias

from orbax.checkpoint._src.arrays import types


ExtMetadata: TypeAlias = Dict[str, Any]

# Keys in the `ext_metadata` dict.
RANDOM_KEY_IMPL = 'random_key_impl'


@dataclasses.dataclass(frozen=True)
class ArrayMetadata:
  """TensorStore metadata for a single array in a checkpoint."""

  param_name: str  # Unique full name of the parameter.
  shape: types.Shape
  dtype: types.DType
  write_shape: types.Shape
  chunk_shape: types.Shape
  use_ocdbt: bool
  use_zarr3: bool
  ext_metadata: ExtMetadata | None = (
      None  # to contain any extension metadata for an array.
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SerializedArrayMetadata:
  """Serialized version of `ArrayMetadata`.

  Not all fields of `ArrayMetadata` are serialized.

  Used in subchunking based checkpointing context.
  """

  param_name: str  # Unique full name of the parameter.
  write_shape: types.Shape
  chunk_shape: types.Shape
  ext_metadata: ExtMetadata | None = None
