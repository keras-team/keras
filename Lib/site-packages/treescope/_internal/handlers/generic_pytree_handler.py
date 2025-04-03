# Copyright 2024 The Treescope Authors.
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

"""Pretty-print handlers for generic pytrees."""

import sys
from typing import Any

from treescope import handlers
from treescope import renderers
from treescope import rendering_parts


def handle_arbitrary_pytrees(
    node: Any,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Generic foldable fallback for an unrecognized pytree type."""
  if "jax" not in sys.modules:
    # JAX isn't imported, so we can't check the JAX pytree registry.
    return NotImplemented

  jax = sys.modules["jax"]

  # Is this a pytree?
  paths_and_subtrees, treedef = jax.tree_util.tree_flatten_with_path(
      node, is_leaf=lambda subtree: subtree is not node
  )
  leaf_treedef = jax.tree_util.tree_structure(1)

  if treedef == leaf_treedef:
    # Not a pytree.
    return NotImplemented

  # First, render the object with repr.
  repr_rendering = handlers.handle_anything_with_repr(
      node=node,
      path=path,
      subtree_renderer=subtree_renderer,
  )

  # Then add an extra block that pretty-prints its children.
  list_items = []
  for (key,), child in paths_and_subtrees:
    child_path = None if path is None else path + str(key)
    list_items.append(
        rendering_parts.siblings_with_annotations(
            subtree_renderer(key, path=None),
            ": ",
            subtree_renderer(child, path=child_path),
            ", ",
        )
    )

  boxed_pytree_children = rendering_parts.indented_children([
      rendering_parts.in_outlined_box(
          rendering_parts.build_full_line_with_annotations(
              rendering_parts.build_custom_foldable_tree_node(
                  label=rendering_parts.comment_color(
                      rendering_parts.text("# PyTree children: ")
                  ),
                  contents=rendering_parts.indented_children(list_items),
              )
          ),
      )
  ])
  return rendering_parts.siblings_with_annotations(
      repr_rendering,
      rendering_parts.fold_condition(
          expanded=rendering_parts.roundtrip_condition(
              not_roundtrip=boxed_pytree_children
          )
      ),
  )
