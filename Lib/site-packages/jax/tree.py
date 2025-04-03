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

"""Utilities for working with tree-like container data structures.

The :mod:`jax.tree` namespace contains aliases of utilities from :mod:`jax.tree_util`.
"""

from jax._src.tree import (
    all as all,
    flatten_with_path as flatten_with_path,
    flatten as flatten,
    leaves_with_path as leaves_with_path,
    leaves as leaves,
    map_with_path as map_with_path,
    map as map,
    reduce as reduce,
    structure as structure,
    transpose as transpose,
    unflatten as unflatten,
)
