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

"""Handler for built-in collection types (lists, sets, dicts, etc)."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional, Sequence

from treescope import dataclass_util
from treescope import layout_algorithms
from treescope import renderers
from treescope._internal.parts import basic_parts
from treescope._internal.parts import common_structures
from treescope._internal.parts import common_styles
from treescope._internal.parts import part_interface


CSSStyleRule = part_interface.CSSStyleRule
HtmlContextForSetup = part_interface.HtmlContextForSetup


def build_field_children(
    node: Any,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
    fields_or_attribute_names: Sequence[dataclasses.Field[Any] | str],
    attr_style_fn: (
        Callable[[str], part_interface.RenderableTreePart] | None
    ) = None,
) -> list[part_interface.RenderableTreePart]:
  """Renders a set of fields/attributes into a list of comma-separated children.

  This is a helper function used for rendering dataclasses, namedtuples, and
  similar objects, of the form ::

    ClassName(
        field_name_one=value1,
        field_name_two=value2,
    )

  If `fields_or_attribute_names` includes dataclass fields:

  * Metadata for the fields will be visible on hover,

  * Fields with ``repr=False`` will be hidden unless roundtrip mode is enabled.

  Args:
    node: Node to render.
    path: Path to this node.
    subtree_renderer: How to render subtrees (see `TreescopeSubtreeRenderer`)
    fields_or_attribute_names: Sequence of fields or attribute names to render.
      Any field with the metadata key "treescope_always_collapse" set to True
      will always render collapsed.
    attr_style_fn: Optional function which makes attributes to a part that
      should render them. If not provided, all parts are rendered as plain text.

  Returns:
    A list of child objects. This can be passed to
    `common_structures.build_foldable_tree_node_from_children` (with
    ``comma_separated=False``)
  """
  if attr_style_fn is None:
    attr_style_fn = basic_parts.Text

  field_names = []
  fields: list[Optional[dataclasses.Field[Any]]] = []
  for field_or_name in fields_or_attribute_names:
    if isinstance(field_or_name, str):
      field_names.append(field_or_name)
      fields.append(None)
    else:
      field_names.append(field_or_name.name)
      fields.append(field_or_name)

  children = []
  for i, (field_name, maybe_field) in enumerate(zip(field_names, fields)):
    child_path = None if path is None else f"{path}.{field_name}"

    if i < len(fields) - 1:
      # Not the last child. Always show a comma, and add a space when
      # collapsed.
      comma_after = basic_parts.siblings(
          ",", basic_parts.FoldCondition(collapsed=basic_parts.Text(" "))
      )
    else:
      # Last child: only show the comma when the node is expanded.
      comma_after = basic_parts.FoldCondition(expanded=basic_parts.Text(","))

    if maybe_field is not None:
      hide_except_in_roundtrip = not maybe_field.repr
      force_collapsed = maybe_field.metadata.get(
          "treescope_always_collapse", False
      )
    else:
      hide_except_in_roundtrip = False
      force_collapsed = False

    field_name_rendering = attr_style_fn(field_name)

    try:
      field_value = getattr(node, field_name)
    except AttributeError:
      child = basic_parts.FoldCondition(
          expanded=common_styles.CommentColor(
              basic_parts.siblings("# ", field_name_rendering, " is missing")
          )
      )
    else:
      child = basic_parts.siblings_with_annotations(
          field_name_rendering,
          "=",
          subtree_renderer(field_value, path=child_path),
      )

    child_line = basic_parts.build_full_line_with_annotations(
        child, comma_after
    )
    if force_collapsed:
      layout_algorithms.expand_to_depth(child_line, 0)
    if hide_except_in_roundtrip:
      child_line = basic_parts.RoundtripCondition(roundtrip=child_line)

    children.append(child_line)

  return children


def render_dataclass_constructor(
    node: Any,
) -> part_interface.RenderableTreePart:
  """Renders the constructor for a dataclass, including the open parenthesis."""
  assert dataclasses.is_dataclass(node) and not isinstance(node, type)
  if not dataclass_util.init_takes_fields(type(node)):
    constructor_open = basic_parts.siblings(
        basic_parts.RoundtripCondition(
            roundtrip=basic_parts.Text(
                "treescope.dataclass_util.dataclass_from_attributes("
            )
        ),
        common_structures.maybe_qualified_type_name(type(node)),
        basic_parts.RoundtripCondition(
            roundtrip=basic_parts.Text(", "),
            not_roundtrip=basic_parts.Text("("),
        ),
    )
  else:
    constructor_open = basic_parts.siblings(
        common_structures.maybe_qualified_type_name(type(node)), "("
    )
  return constructor_open
