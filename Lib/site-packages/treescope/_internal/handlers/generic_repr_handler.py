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

"""Handler that falls back to ordinary `repr`."""

from __future__ import annotations

from typing import Any

from treescope import renderers
from treescope import rendering_parts


def handle_anything_with_repr(
    node: Any,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Builds a foldable from its repr."""
  del subtree_renderer
  node_repr = repr(node)
  basic_repr = object.__repr__(node)
  if "\n" in node_repr:
    # Multiline repr. Use a simpler version for the one-line summary.
    if node_repr.startswith("<") and node_repr.endswith(">"):
      # This node has an idiomatic non-roundtrippable repr.
      # Replace its newlines with markers, and add the basic summary as a
      # comment
      lines = node_repr.split("\n")
      lines_with_markers = [
          rendering_parts.siblings(
              rendering_parts.text(line),
              rendering_parts.fold_condition(
                  expanded=rendering_parts.text("\n"),
                  collapsed=rendering_parts.comment_color(
                      rendering_parts.text("↩")
                  ),
              ),
          )
          for line in lines[:-1]
      ]
      lines_with_markers.append(rendering_parts.text(lines[-1]))
      result = rendering_parts.siblings_with_annotations(
          rendering_parts.build_custom_foldable_tree_node(
              contents=rendering_parts.abbreviation_color(
                  rendering_parts.siblings(*lines_with_markers)
              ),
              path=path,
          ),
          extra_annotations=[
              rendering_parts.comment_color(
                  rendering_parts.text("  # " + basic_repr)
              )
          ],
      )
    else:
      # Use basic repr as the summary.
      result = rendering_parts.build_custom_foldable_tree_node(
          label=rendering_parts.abbreviation_color(
              rendering_parts.comment_color_when_expanded(
                  rendering_parts.siblings(
                      rendering_parts.fold_condition(
                          expanded=rendering_parts.text("# ")
                      ),
                      basic_repr,
                  )
              )
          ),
          contents=rendering_parts.fold_condition(
              expanded=rendering_parts.indented_children([
                  rendering_parts.abbreviation_color(
                      rendering_parts.text(node_repr)
                  )
              ])
          ),
          path=path,
      )
  elif node_repr == basic_repr:
    # Just use the basic repr as the summary, since we don't have anything else.
    result = rendering_parts.build_one_line_tree_node(
        line=rendering_parts.abbreviation_color(
            rendering_parts.text(node_repr)
        ),
        path=path,
    )
  else:
    # Use the custom repr as a one-line summary, but float the basic repr to
    # the right to tell the user what the type is in case the custom repr
    # doesn't include that info.
    result = rendering_parts.siblings_with_annotations(
        rendering_parts.build_one_line_tree_node(
            line=rendering_parts.abbreviation_color(
                rendering_parts.text(node_repr)
            ),
            path=path,
        ),
        extra_annotations=[
            rendering_parts.comment_color(
                rendering_parts.text("  # " + basic_repr)
            )
        ],
    )

  # Possibly apply abbreviation.
  if len(node_repr) > 20:
    result = rendering_parts.abbreviatable_with_annotations(
        result,
        rendering_parts.abbreviation_color(
            rendering_parts.text(f"<{type(node).__name__}...>")
        ),
    )

  return result
