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

"""Handlers for basic Python types, along with namedtuples and dataclasses."""


import ast
import collections
import dataclasses
import enum
import types
from typing import Any

from treescope import formatting_util
from treescope import renderers
from treescope import rendering_parts
from treescope._internal import html_escaping
from treescope._internal.parts import basic_parts
from treescope._internal.parts import common_styles
from treescope._internal.parts import part_interface
from treescope._internal.parts import foldable_impl

CSSStyleRule = part_interface.CSSStyleRule
HtmlContextForSetup = part_interface.HtmlContextForSetup


class KeywordColor(basic_parts.BaseSpanGroup):
  """Renders its child in a color for keywords."""

  def _span_css_class(self) -> str:
    return "color_keyword"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
      .color_keyword
      {
        color: #0000ff;
      }
    """))


class NumberColor(basic_parts.BaseSpanGroup):
  """Renders its child in a color for numbers."""

  def _span_css_class(self) -> str:
    return "color_number"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
      .color_number
      {
        color: #098156;
      }
    """))


class StringLiteralColor(basic_parts.BaseSpanGroup):
  """Renders its child in a color for string literals."""

  def _span_css_class(self) -> str:
    return "color_string"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
      .color_string
      {
        color: #a31515;
      }
    """))


def render_string_or_bytes(
    node: str | bytes,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a string or bytes literal."""
  del subtree_renderer
  lines = node.splitlines(keepends=True)
  if len(lines) > 1:
    # For multiline strings, we use two renderings:
    # - When collapsed, they render with ordinary `repr`,
    # - When expanded, they render as the implicit concatenation of per-line
    #   string literals.
    # Note that the `repr` for a string sometimes switches delimiters
    # depending on whether the string contains quotes or not, so we can't do
    # much manipulation of the strings themselves. This means that the safest
    # thing to do is to just embed two copies of the string into the IR,
    # one for the full string and the other for each line.
    result = rendering_parts.build_custom_foldable_tree_node(
        contents=StringLiteralColor(
            rendering_parts.fold_condition(
                collapsed=rendering_parts.text(repr(node)),
                expanded=rendering_parts.indented_children(
                    children=[
                        rendering_parts.text(repr(line)) for line in lines
                    ],
                    comma_separated=False,
                ),
            )
        ),
        path=path,
    )
  else:
    # No newlines, so render it on a single line.
    result = rendering_parts.build_one_line_tree_node(
        StringLiteralColor(rendering_parts.text(repr(node))), path
    )
  # Abbreviate long strings.
  if len(node) > 20:
    remaining = len(node) - 10
    result = foldable_impl.abbreviatable_with_annotations(
        result,
        abbreviation=basic_parts.siblings(
            StringLiteralColor(rendering_parts.text(repr(node[:5]))),
            common_styles.abbreviation_color(
                rendering_parts.text(f"<{remaining} chars...>")
            ),
            StringLiteralColor(rendering_parts.text(repr(node[-5:]))),
        ),
    )
  return result


def render_numeric_literal(
    node: int | float,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a numeric literal."""
  del subtree_renderer
  return rendering_parts.build_one_line_tree_node(
      NumberColor(rendering_parts.text(repr(node))), path
  )


def render_keyword(
    node: bool | None | type(Ellipsis) | type(NotImplemented),
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a builtin constant (None, False, True, ..., NotImplemented)."""
  del subtree_renderer
  return rendering_parts.build_one_line_tree_node(
      KeywordColor(rendering_parts.text(repr(node))), path
  )


def render_enum(
    node: enum.Enum,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders an enum (roundtrippably, unlike the normal enum `repr`)."""
  del subtree_renderer
  cls = type(node)
  if node is getattr(cls, node.name):
    return rendering_parts.build_one_line_tree_node(
        rendering_parts.siblings_with_annotations(
            rendering_parts.maybe_qualified_type_name(cls),
            "." + node.name,
            extra_annotations=[
                rendering_parts.comment_color(
                    rendering_parts.text(f"  # value: {repr(node.value)}")
                )
            ],
        ),
        path,
    )
  else:
    return NotImplemented


def render_dict(
    node: dict[Any, Any],
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> rendering_parts.RenderableAndLineAnnotations:
  """Renders a dictionary."""

  children = []
  for i, (key, child) in enumerate(node.items()):
    if i < len(node) - 1:
      # Not the last child. Always show a comma, and add a space when
      # collapsed.
      comma_after = rendering_parts.siblings(
          ",",
          rendering_parts.fold_condition(collapsed=rendering_parts.text(" ")),
      )
    else:
      # Last child: only show the comma when the node is expanded.
      comma_after = rendering_parts.fold_condition(
          expanded=rendering_parts.text(",")
      )

    child_path = None if path is None else f"{path}[{repr(key)}]"
    # Figure out whether this key is simple enough to render inline with
    # its value.
    key_rendering = subtree_renderer(key)
    value_rendering = subtree_renderer(child, path=child_path)

    if (
        key_rendering.renderable.collapsed_width < 40
        and not key_rendering.renderable.foldables_in_this_part()
        and key_rendering.annotations.collapsed_width == 0
    ):
      # Simple enough to render on one line.
      children.append(
          rendering_parts.siblings_with_annotations(
              key_rendering, ": ", value_rendering, comma_after
          )
      )
    else:
      # Should render on multiple lines.
      children.append(
          rendering_parts.siblings(
              rendering_parts.build_full_line_with_annotations(
                  key_rendering,
                  ":",
                  rendering_parts.fold_condition(
                      collapsed=rendering_parts.text(" ")
                  ),
              ),
              rendering_parts.indented_children([
                  rendering_parts.siblings_with_annotations(
                      value_rendering, comma_after
                  ),
                  rendering_parts.fold_condition(
                      expanded=rendering_parts.vertical_space("0.5em")
                  ),
              ]),
          )
      )

  if type(node) is dict:  # pylint: disable=unidiomatic-typecheck
    start = "{"
    end = "}"
  else:
    start = rendering_parts.siblings(
        rendering_parts.maybe_qualified_type_name(type(node)), "({"
    )
    end = "})"

  if not children:
    return rendering_parts.build_one_line_tree_node(
        line=rendering_parts.siblings(start, end), path=path
    )
  else:
    return rendering_parts.build_foldable_tree_node_from_children(
        prefix=start,
        children=children,
        suffix=end,
        path=path,
        child_type_single_and_plural=("item", "items"),
    )


def render_sequence_or_set(
    sequence: dict[Any, Any],
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> rendering_parts.RenderableAndLineAnnotations:
  """Renders a sequence or set to a foldable."""
  if (
      isinstance(sequence, tuple)
      and type(sequence) is not tuple  # pylint: disable=unidiomatic-typecheck
      and hasattr(type(sequence), "_fields")
  ):
    # This is actually a namedtuple, which renders with keyword arguments.
    return render_namedtuple_or_ast(sequence, path, subtree_renderer)

  children = []
  for i, child in enumerate(sequence):
    child_path = None if path is None else f"{path}[{repr(i)}]"
    children.append(subtree_renderer(child, path=child_path))

  force_trailing_comma = False
  if isinstance(sequence, tuple):
    before = "("
    after = ")"
    if type(sequence) is not tuple:  # pylint: disable=unidiomatic-typecheck
      # Subclass of `tuple`.
      assert not hasattr(type(sequence), "_fields"), "impossible: checked above"
      # Unusual situation: this is a subclass of `tuple`, but it isn't a
      # namedtuple. Assume we can call it with a single ordinary tuple as an
      # argument.
      before = rendering_parts.siblings(
          rendering_parts.maybe_qualified_type_name(type(sequence)),
          "(" + before,
      )
      after = after + ")"
    force_trailing_comma = len(sequence) == 1
  elif isinstance(sequence, list):
    before = "["
    after = "]"
    if type(sequence) is not list:  # pylint: disable=unidiomatic-typecheck
      before = rendering_parts.siblings(
          rendering_parts.maybe_qualified_type_name(type(sequence)),
          "(" + before,
      )
      after = after + ")"
  elif isinstance(sequence, set):
    if not sequence:
      before = "set("
      after = ")"
    else:  # pylint: disable=unidiomatic-typecheck
      before = "{"
      after = "}"

    if type(sequence) is not set:  # pylint: disable=unidiomatic-typecheck
      before = rendering_parts.siblings(
          rendering_parts.maybe_qualified_type_name(type(sequence)),
          "(" + before,
      )
      after = after + ")"
  elif isinstance(sequence, frozenset):
    before = "frozenset({"
    after = "})"
    if type(sequence) is not frozenset:  # pylint: disable=unidiomatic-typecheck
      before = rendering_parts.siblings(
          rendering_parts.maybe_qualified_type_name(type(sequence)),
          "(" + before,
      )
      after = after + ")"
  else:
    raise ValueError(f"Unrecognized sequence {sequence}")

  if not children:
    return rendering_parts.build_one_line_tree_node(
        line=rendering_parts.siblings(before, after), path=path
    )
  else:
    return rendering_parts.build_foldable_tree_node_from_children(
        prefix=before,
        children=children,
        suffix=after,
        path=path,
        comma_separated=True,
        force_trailing_comma=force_trailing_comma,
        child_type_single_and_plural=("element", "elements"),
    )


def render_simplenamespace(
    node: types.SimpleNamespace,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> rendering_parts.RenderableAndLineAnnotations:
  """Renders a SimpleNamespace."""
  return rendering_parts.build_foldable_tree_node_from_children(
      prefix=rendering_parts.siblings(
          rendering_parts.maybe_qualified_type_name(type(node)), "("
      ),
      children=rendering_parts.build_field_children(
          node,
          path,
          subtree_renderer,
          fields_or_attribute_names=tuple(node.__dict__.keys()),
      ),
      suffix=")",
      path=path,
      child_type_single_and_plural=("attribute", "attributes"),
  )


def render_namedtuple_or_ast(
    node: tuple[Any, ...] | ast.AST,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> rendering_parts.RenderableAndLineAnnotations:
  """Renders a namedtuple or AST class."""
  ty = type(node)
  assert hasattr(ty, "_fields")
  return rendering_parts.build_foldable_tree_node_from_children(
      prefix=rendering_parts.siblings(
          rendering_parts.maybe_qualified_type_name(ty), "("
      ),
      children=rendering_parts.build_field_children(
          node, path, subtree_renderer, fields_or_attribute_names=ty._fields
      ),
      suffix=")",
      path=path,
      child_type_single_and_plural=("attribute", "attributes"),
  )


BUILTINS_REGISTRY = {
    # Builtin atomic types.
    str: render_string_or_bytes,
    bytes: render_string_or_bytes,
    int: render_numeric_literal,
    float: render_numeric_literal,
    bool: render_keyword,
    type(None): render_keyword,
    type(NotImplemented): render_keyword,
    type(Ellipsis): render_keyword,
    enum.Enum: render_enum,
    # Builtin basic structures.
    dict: render_dict,
    tuple: render_sequence_or_set,
    list: render_sequence_or_set,
    set: render_sequence_or_set,
    frozenset: render_sequence_or_set,
    types.SimpleNamespace: render_simplenamespace,
    ast.AST: render_namedtuple_or_ast,
    collections.UserDict: render_dict,
}


def handle_basic_types(
    node: Any,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders basic builtin Python types."""
  candidate_type = type(node)
  for supertype in candidate_type.__mro__:
    if supertype in BUILTINS_REGISTRY:
      return BUILTINS_REGISTRY[supertype](node, path, subtree_renderer)

  if dataclasses.is_dataclass(node) and not isinstance(node, type):
    constructor_open = rendering_parts.render_dataclass_constructor(node)
    if hasattr(node, "__treescope_color__") and callable(
        node.__treescope_color__
    ):
      background_color, background_pattern = (
          formatting_util.parse_simple_color_and_pattern_spec(
              node.__treescope_color__(), type(node).__name__
          )
      )
    else:
      background_color = None
      background_pattern = None

    return rendering_parts.build_foldable_tree_node_from_children(
        prefix=constructor_open,
        children=rendering_parts.build_field_children(
            node,
            path,
            subtree_renderer,
            fields_or_attribute_names=dataclasses.fields(node),
        ),
        suffix=")",
        path=path,
        background_color=background_color,
        background_pattern=background_pattern,
        child_type_single_and_plural=("field", "fields"),
    )

  return NotImplemented
