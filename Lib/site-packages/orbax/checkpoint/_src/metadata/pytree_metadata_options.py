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

"""Basic constructs for PyTree Metadata handling."""

import dataclasses


@dataclasses.dataclass(kw_only=True)
class PyTreeMetadataOptions:
  """Options for managing PyTree metadata.

  Attributes:
    support_rich_types: [Experimental feature: subject to change without
      notice.] If True, supports NamedTuple and Tuple node types in the
      metadata. Otherwise, a NamedTuple node is converted to dict and Tuple node
      to list.
  """

  # TODO: b/365169723 - Support different namedtuple ser/deser strategies.

  support_rich_types: bool = False


# Global default options.
PYTREE_METADATA_OPTIONS = PyTreeMetadataOptions(support_rich_types=False)
