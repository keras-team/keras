# Copyright 2022-2025 MetaOPT Team. All Rights Reserved.
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
# ==============================================================================
r"""Utilities for working with ``PyTree``\s.

The :mod:`optree.pytree` namespace contains aliases of ``optree.tree_*`` utilities.

>>> import optree.pytree as pytree
>>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
>>> leaves, treespec = pytree.flatten(tree)
>>> leaves, treespec  # doctest: +IGNORE_WHITESPACE
(
    [1, 2, 3, 4, 5],
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
)
>>> tree == pytree.unflatten(treespec, leaves)
True

.. versionadded:: 0.14.1
"""

from __future__ import annotations

from optree.ops import tree_accessors as accessors
from optree.ops import tree_all as all  # pylint: disable=redefined-builtin
from optree.ops import tree_any as any  # pylint: disable=redefined-builtin
from optree.ops import tree_broadcast_common as broadcast_common
from optree.ops import tree_broadcast_map as broadcast_map
from optree.ops import tree_broadcast_map_with_accessor as broadcast_map_with_accessor
from optree.ops import tree_broadcast_map_with_path as broadcast_map_with_path
from optree.ops import tree_broadcast_prefix as broadcast_prefix
from optree.ops import tree_flatten as flatten
from optree.ops import tree_flatten_one_level as flatten_one_level
from optree.ops import tree_flatten_with_accessor as flatten_with_accessor
from optree.ops import tree_flatten_with_path as flatten_with_path
from optree.ops import tree_is_leaf as is_leaf
from optree.ops import tree_iter as iter  # pylint: disable=redefined-builtin
from optree.ops import tree_leaves as leaves
from optree.ops import tree_map as map  # pylint: disable=redefined-builtin
from optree.ops import tree_map_ as map_
from optree.ops import tree_map_with_accessor as map_with_accessor
from optree.ops import tree_map_with_accessor_ as map_with_accessor_
from optree.ops import tree_map_with_path as map_with_path
from optree.ops import tree_map_with_path_ as map_with_path_
from optree.ops import tree_max as max  # pylint: disable=redefined-builtin
from optree.ops import tree_min as min  # pylint: disable=redefined-builtin
from optree.ops import tree_paths as paths
from optree.ops import tree_reduce as reduce
from optree.ops import tree_replace_nones as replace_nones
from optree.ops import tree_structure as structure
from optree.ops import tree_sum as sum  # pylint: disable=redefined-builtin
from optree.ops import tree_transpose as transpose
from optree.ops import tree_transpose_map as transpose_map
from optree.ops import tree_transpose_map_with_accessor as transpose_map_with_accessor
from optree.ops import tree_transpose_map_with_path as transpose_map_with_path
from optree.ops import tree_unflatten as unflatten
from optree.registry import dict_insertion_ordered
from optree.registry import register_pytree_node as register_node
from optree.registry import register_pytree_node_class as register_node_class
from optree.registry import unregister_pytree_node as unregister_node


__all__ = [
    'flatten',
    'flatten_with_path',
    'flatten_with_accessor',
    'unflatten',
    'iter',
    'leaves',
    'structure',
    'paths',
    'accessors',
    'is_leaf',
    'map',
    'map_',
    'map_with_path',
    'map_with_path_',
    'map_with_accessor',
    'map_with_accessor_',
    'replace_nones',
    'transpose',
    'transpose_map',
    'transpose_map_with_path',
    'transpose_map_with_accessor',
    'broadcast_prefix',
    'broadcast_common',
    'broadcast_map',
    'broadcast_map_with_path',
    'broadcast_map_with_accessor',
    'reduce',
    'sum',
    'max',
    'min',
    'all',
    'any',
    'flatten_one_level',
    'register_node',
    'register_node_class',
    'unregister_node',
    'dict_insertion_ordered',
]
