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

"""Metadata describing checkpoints."""

import dataclasses
from typing import Any, Generic, TypeVar
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


CheckpointableMetadataT = TypeVar('CheckpointableMetadataT')


@dataclasses.dataclass(frozen=True, kw_only=True)
class PyTreeMetadata:
  """Represents the metadata for a PyTree.

  See `CheckpointMetadata` to further understand how this is used.

  Attributes:
    pytree: A serialized PyTree structure with the same structure as the
      checkpointed PyTree. By "serialized", we mean that the PyTree has been
      converted to a standardized representation, with all container nodes
      represented as standard types (e.g., tuple, list, dict, etc.). The leaves


@dataclasses.dataclass(frozen=True, kw_only=True)
class CheckpointMetadata(Generic[CheckpointableMetadataT]):
  """Represents complete metadata describing a checkpoint.

  Note that this class has a generic type `CheckpointableMetadataT`. This
  will typically be either `PyTreeMetadata` (see above), or `dict[str, Any]`.

  `CheckpointMetadata` can be accessed via one of two metadata methods. Please
  see `ocp.pytree_metadata` and `ocp.checkpointables_metadata` for more
  information and usage instructions.

  If the checkpoint contains a PyTree, this metadata can be acccessed via::

    metadata = ocp.pytree_metadata(path)

    # Inspect various properties
    metadata.init_timestamp_nsecs

    # Inspect the tree structure
    metadata.metadata.pytree
    metadata.metadata.pytree['layer0']['bias'].shape
    metadata.metadata.pytree['layer0']['bias'].dtype

  The checkpoint metadata can also be accessed more generically via::

    metadata = ocp.checkpointables_metadata(path)

    metadata.metadata.keys()  # == ['pytree', 'dataset', etc.]
    metadata.metadata['pytree']  # instance of PyTreeMetadata

  Attributes:
    metadata: Metadata for the checkpointable.
    init_timestamp_nsecs: timestamp when uncommitted checkpoint was initialized.
      Specified as nano seconds since epoch. default=None.
    commit_timestamp_nsecs: commit timestamp of a checkpoint, specified as nano
      seconds since epoch. default=None.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.
  """
  metadata: CheckpointableMetadataT | None = None
  init_timestamp_nsecs: int | None = None
  commit_timestamp_nsecs: int | None = None
  custom_metadata: tree_types.JsonType | None = None
