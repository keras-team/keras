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

"""Handler for any object with a _repr_html_.

`_repr_html_` is a method standardized by IPython, which is used to register
a pretty HTML representation of an object. This means we can support rendering
any rich object with a custom notebook rendering strategy.

See
https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.display
for details on the purpose of the `_repr_html_` method.
"""

from __future__ import annotations

from typing import Any

from treescope import context
from treescope import lowering
from treescope import renderers
from treescope import rendering_parts
from treescope._internal import object_inspection


_already_processing_repr_html: context.ContextualValue[bool] = (
    context.ContextualValue(
        module=__name__,
        qualname="_already_processing_repr_html",
        initial_value=False,
    )
)
"""Tracks whether we are already rendering this subtree in some parent.

This is used to prevent situations where a node defines ``_repr_html_`` but
also includes children that define ``_repr_html_``. We only want to use the
rich HTML representation of the outermost element.
"""


def append_repr_html_when_present(
    node: Any,
    path: str | None,
    node_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Appends rich HTML representations of objects that have them."""
  if _already_processing_repr_html.get():
    # We've processed the repr_html for a parent of this node already.
    return NotImplemented

  if not isinstance(node, object_inspection.HasReprHtml):
    return NotImplemented

  # Make sure we don't try to call _repr_html_ on the children of this node,
  # since this node's _repr_html_ probably handles that already.
  with _already_processing_repr_html.set_scoped(True):
    # In "auto" mode, collapse the plaintext repr of objects with a _repr_html_,
    # and expand the HTML view instead.
    node_rendering = node_renderer(node, path=path)

  def _thunk(_):
    html_rendering = object_inspection.to_html(node)
    if html_rendering:
      return rendering_parts.embedded_iframe(
          embedded_html=html_rendering,
          fallback_in_text_mode=rendering_parts.abbreviation_color(
              rendering_parts.text("# (not shown in text mode)")
          ),
      )
    else:
      return rendering_parts.error_color(
          rendering_parts.text("# (couldn't compute HTML representation)")
      )

  iframe_rendering = lowering.maybe_defer_rendering(
      main_thunk=_thunk,
      placeholder_thunk=lambda: rendering_parts.deferred_placeholder_style(
          rendering_parts.text("...")
      ),
  )

  boxed_html_repr = rendering_parts.indented_children([
      rendering_parts.floating_annotation_with_separate_focus(
          rendering_parts.in_outlined_box(
              rendering_parts.build_full_line_with_annotations(
                  rendering_parts.build_custom_foldable_tree_node(
                      label=rendering_parts.comment_color(
                          rendering_parts.text("# Rich HTML representation")
                      ),
                      contents=rendering_parts.fold_condition(
                          expanded=iframe_rendering
                      ),
                  )
              ),
          )
      )
  ])
  return rendering_parts.siblings_with_annotations(
      node_rendering,
      rendering_parts.fold_condition(
          expanded=rendering_parts.roundtrip_condition(
              not_roundtrip=boxed_html_repr
          )
      ),
  )
