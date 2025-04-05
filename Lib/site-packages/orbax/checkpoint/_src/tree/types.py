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

"""Public common types to work with pytrees."""

from __future__ import annotations

from typing import Any, TypeVar, Union

from jax import tree_util as jtu


PyTree = Any

# This trick doesn't allow us to achieve full type checking but we at least can
# use concrete types to document things like `PyTreeOf[jax.Array]`.
T = TypeVar('T')
PyTreeOf = Union[PyTree, T]

PyTreeKey = Union[
    jtu.SequenceKey, jtu.DictKey, jtu.GetAttrKey, jtu.FlattenedIndexKey
]
PyTreePath = tuple[PyTreeKey, ...]

JsonType = list['JsonValue'] | dict[str, 'JsonValue']
JsonValue = str | int | float | bool | None | JsonType
