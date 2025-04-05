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

"""Metadata for `training.Checkpointer`."""

import dataclasses
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


CheckpointableMetadataT = metadata_types.CheckpointableMetadataT


@dataclasses.dataclass(frozen=True, kw_only=True)
class CheckpointMetadata(
    metadata_types.CheckpointMetadata[CheckpointableMetadataT]
):
  """Represents metadata for a single checkpoint (corresponding to a step).

  Like its parent, the class has a `metadata` attribute that is a generic type.

  See superclass documentation for more information, and for a list of base
  attributes.

  Attributes:
    metrics: User-provided metrics for the step (e.g. loss, accuracy, etc.)
  """
  metrics: tree_types.JsonType | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class RootMetadata:
  """Metadata of a checkpoint at root level (contains all steps).

  Attributes:
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.
  """

  custom_metadata: tree_types.JsonType | None = None
