# Copyright 2018 The JAX Authors.
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

"""Utilities for working with tree-like container data structures.

This module provides a small set of utility functions for working with tree-like
data structures, such as nested tuples, lists, and dicts. We call these
structures pytrees. They are trees in that they are defined recursively (any
non-pytree is a pytree, i.e. a leaf, and any pytree of pytrees is a pytree) and
can be operated on recursively (object identity equivalence is not preserved by
mapping operations, and the structures cannot contain reference cycles).

The set of Python types that are considered pytree nodes (e.g. that can be
mapped over, rather than treated as leaves) is extensible. There is a single
module-level registry of types, and class hierarchy is ignored. By registering a
new pytree node type, that type in effect becomes transparent to the utility
functions in this file.

The primary purpose of this module is to enable the interoperability between
user defined data structures and JAX transformations (e.g. `jit`). This is not
meant to be a general purpose tree-like data structure handling library.

See the `JAX pytrees note <pytrees.html>`_
for examples.
"""

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.tree_util import (
    DictKey as DictKey,
    FlattenedIndexKey as FlattenedIndexKey,
    GetAttrKey as GetAttrKey,
    KeyEntry as KeyEntry,
    KeyPath as KeyPath,
    Partial as Partial,
    PyTreeDef as PyTreeDef,
    SequenceKey as SequenceKey,
    all_leaves as all_leaves,
    build_tree as build_tree,
    default_registry as default_registry,
    keystr as keystr,
    register_pytree_node_class as register_pytree_node_class,
    register_pytree_node as register_pytree_node,
    register_pytree_with_keys_class as register_pytree_with_keys_class,
    register_dataclass as register_dataclass,
    register_pytree_with_keys as register_pytree_with_keys,
    register_static as register_static,
    tree_all as tree_all,
    tree_flatten_with_path as tree_flatten_with_path,
    tree_flatten as tree_flatten,
    tree_leaves_with_path as tree_leaves_with_path,
    tree_leaves as tree_leaves,
    tree_map_with_path as tree_map_with_path,
    tree_map as tree_map,
    tree_reduce as tree_reduce,
    tree_structure as tree_structure,
    tree_transpose as tree_transpose,
    tree_unflatten as tree_unflatten,
    treedef_children as treedef_children,
    treedef_is_leaf as treedef_is_leaf,
    treedef_tuple as treedef_tuple,
)
