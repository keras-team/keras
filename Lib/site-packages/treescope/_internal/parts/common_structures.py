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

"""Helpers to build frequently-used configurations of rendered parts."""

from __future__ import annotations

from typing import Any, Sequence

from treescope import canonical_aliases
from treescope._internal.parts import basic_parts
from treescope._internal.parts import common_styles
from treescope._internal.parts import foldable_impl
from treescope._internal.parts import part_interface


CSSStyleRule = part_interface.CSSStyleRule
JavaScriptDefn = part_interface.JavaScriptDefn
HtmlContextForSetup = part_interface.HtmlContextForSetup
RenderableTreePart = part_interface.RenderableTreePart
ExpandState = part_interface.ExpandState
FoldableTreeNode = part_interface.FoldableTreeNode

RenderableAndLineAnnotations = basic_parts.RenderableAndLineAnnotations


def build_copy_button(path: str | None) -> RenderableTreePart:
  """Builds a copy-path button, if `path` is provided and not empty."""
  if not path:
    return basic_parts.EmptyPart()
  else:
    return foldable_impl.StringCopyButton(
        annotation="Copy path: ", copy_string=path
    )


def build_custom_foldable_tree_node(
    contents: RenderableTreePart,
    path: str | None = None,
    label: RenderableTreePart = basic_parts.EmptyPart(),
    expand_state: part_interface.ExpandState = (
        part_interface.ExpandState.WEAKLY_COLLAPSED
    ),
) -> RenderableAndLineAnnotations:
  """Builds a custom foldable tree node with path buttons.

  Args:
    contents: Contents of this foldable that should not open/close the custom
      foldable when clicked.
    path: Keypath to this node from the root. If provided, a copy-path button
      will be added at the end of the node.
    label: The beginning of the first line, which should allow opening/closing
      the custom foldable when clicked. Should not contain any other foldables.
    expand_state: Initial expand state for the foldable.

  Returns:
    A new renderable part, possibly with a copy button annotation, for use
    in part of a rendered treescope tree.
  """
  maybe_copy_button = build_copy_button(path)

  return RenderableAndLineAnnotations(
      renderable=foldable_impl.FoldableTreeNodeImpl(
          label=label, contents=contents, expand_state=expand_state
      ),
      annotations=maybe_copy_button,
  )


def build_one_line_tree_node(
    line: RenderableAndLineAnnotations | RenderableTreePart | str,
    path: str | None = None,
    background_color: str | None = None,
    background_pattern: str | None = None,
) -> RenderableAndLineAnnotations:
  """Builds a single-line tree node with path buttons.

  Args:
    line: Contents of the line.
    path: Keypath to this node from the root. If provided, copy-path buttons
      will be added.
    background_color: Optional background and border color for this node.
    background_pattern: Optional background pattern as a CSS "image". If
      provided, `background_color` must also be provided, and will be used as
      the border for the pattern.

  Returns:
    A new renderable part, possibly with a copy button annotation, for use
    in part of a rendered treescope tree.
  """
  maybe_copy_button = build_copy_button(path)

  if isinstance(line, RenderableAndLineAnnotations):
    line_primary = line.renderable
    annotations = basic_parts.siblings(maybe_copy_button, line.annotations)
  elif isinstance(line, str):
    line_primary = basic_parts.Text(line)
    annotations = maybe_copy_button
  else:
    line_primary = line
    annotations = maybe_copy_button

  if background_pattern is not None:
    if background_color is None:
      raise ValueError(
          "background_color must be provided if background_pattern is"
      )
    line_primary = common_styles.WithBlockPattern(
        common_styles.PatternedSingleLineSpanGroup(line_primary),
        color=background_color,
        pattern=background_pattern,
    )
  elif background_color is not None and background_color != "transparent":
    line_primary = common_styles.WithBlockColor(
        common_styles.ColoredSingleLineSpanGroup(line_primary),
        color=background_color,
    )

  return RenderableAndLineAnnotations(
      renderable=line_primary,
      annotations=annotations,
  )


def build_foldable_tree_node_from_children(
    prefix: RenderableTreePart | str,
    children: Sequence[RenderableAndLineAnnotations | RenderableTreePart | str],
    suffix: RenderableTreePart | str,
    comma_separated: bool = False,
    force_trailing_comma: bool = False,
    path: str | None = None,
    background_color: str | None = None,
    background_pattern: str | None = None,
    first_line_annotation: RenderableTreePart | None = None,
    expand_state: part_interface.ExpandState = (
        part_interface.ExpandState.WEAKLY_COLLAPSED
    ),
    *,
    child_type_single_and_plural: tuple[str, str] | None = None,
) -> RenderableAndLineAnnotations:
  """Builds a foldable tree node with path buttons.

  Args:
    prefix: Contents of the first line, before the children. Should not contain
      any other foldables. Usually ends with an opening paren/bracket, e.g.
      "SomeClass("
    children: Sequence of children of this node, which should each be rendered
      on their own line.
    suffix: Contents of the last line, after the children. Usually a closing
      paren/bracket for `prefix`.
    comma_separated: Whether to insert commas between children.
    force_trailing_comma: Whether to always insert a trailing comma after the
      last child.
    path: Keypath to this node from the root. If provided, copy-path buttons
      will be added.
    background_color: Optional background and border color for this node.
    background_pattern: Optional background pattern as a CSS "image". If
      provided, `background_color` must also be provided, and will be used as
      the border for the pattern.
    first_line_annotation: An annotation for the first line of the node when it
      is expanded.
    expand_state: Initial expand state for the foldable.
    child_type_single_and_plural: If provided, this will be used as the
      single and plural forms of the child type in the abbreviation.

  Returns:
    A new renderable part, possibly with a copy button annotation, for use
    in part of a rendered treescope tree.
  """
  if not children:
    return build_one_line_tree_node(
        line=basic_parts.siblings(prefix, suffix),
        path=path,
        background_color=background_color,
    )

  maybe_copy_button = build_copy_button(path)

  if isinstance(prefix, str):
    prefix = basic_parts.Text(prefix)

  if isinstance(suffix, str):
    suffix = basic_parts.Text(suffix)

  if background_pattern is not None:
    if background_color is None:
      raise ValueError(
          "background_color must be provided if background_pattern is"
      )

    def wrap_block(block):
      return common_styles.WithBlockPattern(
          block, color=background_color, pattern=background_pattern
      )

    wrap_topline = common_styles.PatternedTopLineSpanGroup
    wrap_bottomline = common_styles.PatternedBottomLineSpanGroup
    indented_child_class = common_styles.ColoredBorderIndentedChildren

  elif background_color is not None and background_color != "transparent":

    def wrap_block(block):
      return common_styles.WithBlockColor(block, color=background_color)

    wrap_topline = common_styles.ColoredTopLineSpanGroup
    wrap_bottomline = common_styles.ColoredBottomLineSpanGroup
    indented_child_class = common_styles.ColoredBorderIndentedChildren

  else:
    wrap_block = lambda rendering: rendering
    wrap_topline = lambda rendering: rendering
    wrap_bottomline = lambda rendering: rendering
    indented_child_class = basic_parts.IndentedChildren

  if first_line_annotation is not None:
    maybe_first_line_annotation = basic_parts.FoldCondition(
        expanded=first_line_annotation
    )
  else:
    maybe_first_line_annotation = basic_parts.EmptyPart()

  if child_type_single_and_plural:
    single, plural = child_type_single_and_plural
    if len(children) == 1:
      middle = f"1 {single}..."
    else:
      middle = f"{len(children)} {plural}..."
    abbreviation = basic_parts.siblings(
        common_styles.comment_color(basic_parts.text("<")),
        common_styles.abbreviation_color(basic_parts.text(middle)),
        common_styles.comment_color(basic_parts.text(">")),
    )
  else:
    abbreviation = None

  return RenderableAndLineAnnotations(
      renderable=wrap_block(
          foldable_impl.FoldableTreeNodeImpl(
              label=wrap_topline(prefix),
              contents=basic_parts.siblings(
                  maybe_copy_button,
                  maybe_first_line_annotation,
                  foldable_impl.abbreviatable(
                      foldable_impl.abbreviation_level(
                          indented_child_class.build(
                              children,
                              comma_separated=comma_separated,
                              force_trailing_comma=force_trailing_comma,
                          )
                      ),
                      abbreviation=abbreviation,
                  ),
                  wrap_bottomline(suffix),
              ),
              expand_state=expand_state,
          )
      ),
      annotations=maybe_copy_button,
  )


def maybe_qualified_type_name(ty: type[Any]) -> RenderableTreePart:
  """Formats the name of a type so that it is qualified in roundtrip mode.

  Args:
    ty: A type object to render.

  Returns:
    A tree part that renders as a fully-qualified name in roundtrip mode, or as
    the simple class name otherwise. Module names will be either inferred from
    the type's definition or looked up using `canonical_aliases`.
  """
  class_name = ty.__name__

  alias = canonical_aliases.lookup_alias(
      ty, allow_outdated=True, allow_relative=True
  )
  if alias:
    access_path = str(alias)
  else:
    access_path = f"<unknown>.{class_name}"

  if access_path.endswith(class_name):
    return basic_parts.siblings(
        basic_parts.RoundtripCondition(
            roundtrip=common_styles.QualifiedTypeNameSpanGroup(
                basic_parts.Text(access_path.removesuffix(class_name))
            )
        ),
        basic_parts.Text(class_name),
    )
  else:
    return basic_parts.RoundtripCondition(
        roundtrip=basic_parts.Text(access_path),
        not_roundtrip=basic_parts.Text(class_name),
    )
