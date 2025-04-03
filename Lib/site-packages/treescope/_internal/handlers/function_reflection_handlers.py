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

"""Reflection-based handlers for functions, closures, and classes.

These handlers provide:
- deep-linking to jump to the place a particular object was defined,
- visualization of closure variables for closures
"""

import functools
import inspect
import re
from typing import Any

from treescope import renderers
from treescope import rendering_parts


@functools.cache
def _get_filepath_and_lineno(value) -> tuple[str, int] | tuple[None, None]:
  """Retrieves a filepath and line number for an inspectable value."""
  try:
    filepath = inspect.getsourcefile(value)
    if filepath is None:
      return None, None
    # This also returns the source lines, but we don't need those.
    _, lineno = inspect.findsource(value)
    # Add 1, since findsource is zero-indexed.
    return filepath, lineno + 1
  except (TypeError, OSError):
    return None, None


def format_source_location(
    filepath: str, lineno: int
) -> rendering_parts.RenderableTreePart:
  """Formats a reference to a given filepath and line number."""

  # Try to match it as an IPython file
  ipython_output_path = re.fullmatch(
      r"<ipython-input-(?P<cell_number>\d+)-.*>", filepath
  )
  if ipython_output_path:
    cell_number = ipython_output_path.group("cell_number")
    return rendering_parts.text(f"line {lineno} of output cell {cell_number}")

  return rendering_parts.text(f"line {lineno} of {filepath}")


def handle_code_objects_with_reflection(
    node: Any,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
    show_closure_vars: bool = False,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders code objects using reflection and closure inspection."""
  if inspect.isclass(node):
    # Render class.
    closure_vars = None
  elif inspect.isfunction(node):
    # Render function.
    if show_closure_vars:
      closure_vars = inspect.getclosurevars(node).nonlocals
    else:
      closure_vars = None
  else:
    # Not a supported object.
    return NotImplemented

  annotations = []
  filepath, lineno = _get_filepath_and_lineno(node)
  if filepath is not None:
    annotations.append(
        rendering_parts.comment_color(
            rendering_parts.siblings(
                rendering_parts.text("  # Defined at "),
                format_source_location(filepath, lineno),
            )
        )
    )

  if closure_vars:
    boxed_closure_var_rendering = rendering_parts.in_outlined_box(
        rendering_parts.on_separate_lines([
            rendering_parts.comment_color(
                rendering_parts.text("# Closure variables:")
            ),
            subtree_renderer(closure_vars),
        ])
    )
    return rendering_parts.siblings_with_annotations(
        rendering_parts.build_custom_foldable_tree_node(
            label=rendering_parts.abbreviation_color(
                rendering_parts.text(repr(node))
            ),
            contents=rendering_parts.fold_condition(
                expanded=rendering_parts.indented_children(
                    [boxed_closure_var_rendering]
                )
            ),
            path=path,
        ),
        extra_annotations=annotations,
    )

  else:
    return rendering_parts.siblings_with_annotations(
        rendering_parts.build_one_line_tree_node(
            line=rendering_parts.abbreviation_color(
                rendering_parts.text(repr(node))
            ),
            path=path,
        ),
        extra_annotations=annotations,
    )
