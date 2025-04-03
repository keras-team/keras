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

"""Common building blocks for the treescope intermediate representation."""

from __future__ import annotations

import abc
from collections.abc import Hashable
import dataclasses
import functools
import html
import io
import itertools
import operator
import typing
from typing import Any, Sequence

from treescope._internal import html_escaping
from treescope._internal.parts import part_interface


CSSStyleRule = part_interface.CSSStyleRule
JavaScriptDefn = part_interface.JavaScriptDefn
HtmlContextForSetup = part_interface.HtmlContextForSetup
RenderableTreePart = part_interface.RenderableTreePart
ExpandState = part_interface.ExpandState
FoldableTreeNode = part_interface.FoldableTreeNode
RenderableAndLineAnnotations = part_interface.RenderableAndLineAnnotations

################################################################################
# Basic combinators
################################################################################


@dataclasses.dataclass(frozen=True)
class BaseContentlessLeaf(RenderableTreePart):
  """A part whose defaults indicate no renderable content, for subclassing."""

  def _compute_collapsed_width(self) -> int:
    return 0

  def _compute_newlines_in_expanded_parent(self) -> int:
    return 0

  def foldables_in_this_part(self) -> Sequence[FoldableTreeNode]:
    return ()

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return frozenset()

  def html_setup_parts(
      self, context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    return set()

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    pass

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    pass


@dataclasses.dataclass(frozen=True)
@typing.final
class EmptyPart(BaseContentlessLeaf):
  """A definitely-empty part, which can be detected and special-cased."""


def empty_part() -> EmptyPart:
  """Returns an empty part."""
  return EmptyPart()


@dataclasses.dataclass(frozen=True)
class Text(RenderableTreePart):
  """A raw text literal."""

  text: str

  def _compute_collapsed_width(self) -> int:
    if "\n" in self.text:
      raise ValueError(
          "Cannot compute collapsed length for a text literal with a newline!"
          " Text literals with newlines need to be wrapped to only show in"
          " expanded mode."
      )
    return len(self.text)

  def _compute_newlines_in_expanded_parent(self) -> int:
    return self.text.count("\n")

  def foldables_in_this_part(self) -> Sequence[FoldableTreeNode]:
    return ()

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return frozenset()

  def html_setup_parts(
      self, context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    return set()

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    stream.write(html.escape(self.text))

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    if "\n" in self.text and not expanded_parent:
      raise ValueError(
          "Cannot render a text literal with a newline in collapsed mode!"
          " Text literals with newlines need to be wrapped to only show in"
          " expanded mode."
      )
    stream.write(("\n" + " " * indent).join(self.text.split("\n")))


def text(text_content: str) -> RenderableTreePart:
  """Builds a one-line text part."""
  return Text(text_content)


@dataclasses.dataclass(frozen=True)
class Siblings(RenderableTreePart):
  """A sequence of children parts, rendered inline."""

  children: Sequence[RenderableTreePart]

  def _compute_collapsed_width(self) -> int:
    return sum(part.collapsed_width for part in self.children)

  def _compute_newlines_in_expanded_parent(self) -> int:
    return sum(part.newlines_in_expanded_parent for part in self.children)

  def foldables_in_this_part(self) -> Sequence[FoldableTreeNode]:
    return list(
        itertools.chain(
            *(part.foldables_in_this_part() for part in self.children)
        )
    )

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return functools.reduce(
        operator.or_,
        (part.layout_marks_in_this_part for part in self.children),
        frozenset(),
    )

  def html_setup_parts(
      self, context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    return functools.reduce(
        operator.or_,
        (part.html_setup_parts(context) for part in self.children),
        set(),
    )

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    for part in self.children:
      part.render_to_html(
          stream,
          at_beginning_of_line=at_beginning_of_line,
          render_context=render_context,
      )
      # Only the first part gets beginning-of-line privileges.
      at_beginning_of_line = False

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    for part in self.children:
      part.render_to_text(
          stream,
          expanded_parent=expanded_parent,
          indent=indent,
          roundtrip_mode=roundtrip_mode,
          render_context=render_context,
      )


def siblings(*args: RenderableTreePart | str) -> RenderableTreePart:
  """Builds a Siblings part from inline arguments.

  Args:
    *args: Sequence of renderables or strings (which will be wrapped in Text).

  Returns:
    A new Siblings part containing these concatenated together.
  """
  parts = []
  for arg in args:
    if isinstance(arg, str):
      parts.append(Text(arg))
    elif isinstance(arg, Siblings):
      parts.extend(arg.children)
    elif isinstance(arg, EmptyPart):
      pass
    elif isinstance(arg, RenderableTreePart):
      parts.append(arg)
    else:
      raise ValueError(f"Invalid argument type {type(arg)}")
  return Siblings(tuple(parts))


class DeferringToChild(RenderableTreePart):
  """Helper base class that defers methods to a child attribute.

  Subclasses can derive from this to get default implementations for all
  required methods, then override the ones where they have custom behavior.
  """

  # note: not a dataclass, just here for type annotations
  child: RenderableTreePart

  def _compute_collapsed_width(self) -> int:
    return self.child.collapsed_width

  def _compute_newlines_in_expanded_parent(self) -> int:
    return self.child.newlines_in_expanded_parent

  def foldables_in_this_part(self) -> Sequence[FoldableTreeNode]:
    return self.child.foldables_in_this_part()

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return self.child.layout_marks_in_this_part

  def html_setup_parts(
      self, context: HtmlContextForSetup, /
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    return self.child.html_setup_parts(context)

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    self.child.render_to_text(
        stream,
        expanded_parent=expanded_parent,
        indent=indent,
        roundtrip_mode=roundtrip_mode,
        render_context=render_context,
    )


@dataclasses.dataclass(frozen=True)
class BaseSpanGroup(DeferringToChild, abc.ABC):
  """A group that wraps its child in a span, and may apply CSS styles.

  Subclasses can define a particular CSS style to apply.

  Attributes:
    child: Contents of the group.
  """

  child: RenderableTreePart

  # Overridable methods defined here.

  @abc.abstractmethod
  def _span_css_class(self) -> str:
    """Returns a CSS class for the span's style. Intended to be overridden."""
    raise NotImplementedError("_span_css_class should be overridden")

  @abc.abstractmethod
  def _span_css_rule(
      self, context: HtmlContextForSetup, /
  ) -> CSSStyleRule | set[CSSStyleRule]:
    """Returns a CSS style rule or rules for the class in `_span_css_class`.

    Intended to be overridden.

    Args:
      context: Context for setting up.
    """
    raise NotImplementedError("_span_css_rule should be overridden")

  # Implementations of parent class abstract methods.

  def html_setup_parts(
      self, context: HtmlContextForSetup, /
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    my_rule = self._span_css_rule(context)
    if isinstance(my_rule, CSSStyleRule):
      my_rule = {my_rule}
    return my_rule | self.child.html_setup_parts(context)

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    class_string = self._span_css_class()
    stream.write(f'<span class="{class_string}">')
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")


@dataclasses.dataclass(frozen=True)
class WithLayoutMark(DeferringToChild):
  """A group that marks its child for layout purposes.

  Attributes:
    child: Contents of the group.
    mark: A layout mark to apply to the child. This can later be used by layout
      algorithms to show this node.
  """

  child: RenderableTreePart
  mark: Hashable

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return frozenset((self.mark,)) | self.child.layout_marks_in_this_part


def with_layout_mark(
    child: RenderableTreePart, mark: Hashable
) -> RenderableTreePart:
  """Returns a part that marks its child for layout purposes.

  Args:
    child: Contents of the group.
    mark: A layout mark to apply to the child. This can later be used by layout
      algorithms to show this node.

  Returns:
    A new part that marks its child for layout purposes.
  """
  if not isinstance(child, RenderableTreePart):
    raise ValueError(f"child must be a renderable part, got {type(child)}")
  if not isinstance(mark, Hashable):
    raise ValueError(f"Layout marks must be hashable, got {mark}")
  return WithLayoutMark(child=child, mark=mark)


@dataclasses.dataclass(frozen=True)
class VerticalSpace(RenderableTreePart):
  """A vertical space in HTML mode."""

  height: str

  def _compute_collapsed_width(self) -> int:
    return 0

  def _compute_newlines_in_expanded_parent(self) -> int:
    return 0

  def foldables_in_this_part(self) -> Sequence[FoldableTreeNode]:
    return ()

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return frozenset()

  def html_setup_parts(
      self, context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    return set()

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    pass

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    stream.write(
        "<div"
        f' style="height:{html_escaping.escape_html_attribute(self.height)};'
        ' width:0;"></div>'
    )


def vertical_space(css_height: str) -> RenderableTreePart:
  """Returns a vertical space with the given height in HTML mode.

  Args:
    css_height: The height of the space, as a CSS length string.

  Returns:
    A renderable part that renders as a vertical space in HTML mode, and does
    not render in text mode.
  """
  if not isinstance(css_height, str):
    raise ValueError(f"css_height must be a string, got {css_height}")
  return VerticalSpace(height=css_height)


################################################################################
# Conditional rendering
################################################################################


@dataclasses.dataclass(frozen=True)
class FoldCondition(RenderableTreePart):
  """Renders conditionally depending on whether it's collapsed or expanded.

  Attributes:
    collapsed: Contents to render when parent is collapsed.
    expanded: Contents to render when parent is expanded.
  """

  collapsed: RenderableTreePart = EmptyPart()
  expanded: RenderableTreePart = EmptyPart()

  def html_setup_parts(
      self, context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    rule = html_escaping.without_repeated_whitespace(f"""
        .when_collapsed {{
            display: none;
        }}
        {context.collapsed_selector} .when_collapsed {{
            display: inline;
        }}
        .when_expanded {{
            display: inline;
        }}
        {context.collapsed_selector} .when_expanded {{
            display: none;
        }}
        """)
    return (
        {CSSStyleRule(rule)}
        | self.collapsed.html_setup_parts(context)
        | self.expanded.html_setup_parts(context)
    )

  def _compute_collapsed_width(self) -> int:
    return self.collapsed.collapsed_width

  def _compute_newlines_in_expanded_parent(self) -> int:
    # Disappears when expanded.
    return self.expanded.newlines_in_expanded_parent

  def foldables_in_this_part(self) -> Sequence[FoldableTreeNode]:
    # Only the part that's visible when expanded will have foldables that are
    # possible to fold/unfold.
    return self.expanded.foldables_in_this_part()

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return (
        self.collapsed.layout_marks_in_this_part
        | self.expanded.layout_marks_in_this_part
    )

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    if not isinstance(self.collapsed, EmptyPart):
      stream.write('<span class="when_collapsed">')
      self.collapsed.render_to_html(
          stream,
          at_beginning_of_line=at_beginning_of_line,
          render_context=render_context,
      )
      stream.write("</span>")
    if not isinstance(self.expanded, EmptyPart):
      stream.write('<span class="when_expanded">')
      self.expanded.render_to_html(
          stream,
          at_beginning_of_line=at_beginning_of_line,
          render_context=render_context,
      )
      stream.write("</span>")

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    if expanded_parent:
      self.expanded.render_to_text(
          stream,
          expanded_parent=expanded_parent,
          indent=indent,
          roundtrip_mode=roundtrip_mode,
          render_context=render_context,
      )
    else:
      self.collapsed.render_to_text(
          stream,
          expanded_parent=expanded_parent,
          indent=indent,
          roundtrip_mode=roundtrip_mode,
          render_context=render_context,
      )


def fold_condition(
    collapsed: RenderableTreePart | None = None,
    expanded: RenderableTreePart | None = None,
) -> RenderableTreePart:
  """Builds a part that renders differently when collapsed or expanded.

  Args:
    collapsed: Contents to render when parent is collapsed.
    expanded: Contents to render when parent is expanded.

  Returns:
    A renderable part that renders as ``collapsed`` when the parent is collapsed
    and as ``expanded`` when the parent is expanded.
  """
  if collapsed is None:
    collapsed = EmptyPart()
  if expanded is None:
    expanded = EmptyPart()
  if not isinstance(collapsed, RenderableTreePart):
    raise ValueError(
        "`collapsed` must be a renderable part or None. Got"
        f" {type(collapsed).__name__}"
    )
  if not isinstance(expanded, RenderableTreePart):
    raise ValueError(
        "`expanded` must be a renderable part or None. Got"
        f" {type(expanded).__name__}"
    )
  return FoldCondition(collapsed=collapsed, expanded=expanded)


@dataclasses.dataclass(frozen=True)
class RoundtripCondition(RenderableTreePart):
  """Renders conditionally depending on whether it's in roundtrip mode.

  Attributes:
    roundtrip: Contents to render when rendering in round trip mode.
    not_roundtrip: Contents to render when rendering in ordinary mode.
  """

  roundtrip: RenderableTreePart = EmptyPart()
  not_roundtrip: RenderableTreePart = EmptyPart()

  def html_setup_parts(
      self, context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    rule = html_escaping.without_repeated_whitespace(f"""
        .when_roundtrip {{
            display: none;
        }}
        {context.roundtrip_selector} .when_roundtrip {{
            display: inline;
        }}
        .when_not_roundtrip {{
            display: inline;
        }}
        {context.roundtrip_selector} .when_not_roundtrip {{
            display: none;
        }}
        """)
    return (
        {CSSStyleRule(rule)}
        | self.roundtrip.html_setup_parts(context)
        | self.not_roundtrip.html_setup_parts(context)
    )

  def _compute_collapsed_width(self) -> int:
    # Defer to the not_roundtrip version for expansion decisions.
    return self.not_roundtrip.collapsed_width

  def _compute_newlines_in_expanded_parent(self) -> int:
    # Defer to the not_roundtrip version for expansion decisions.
    return self.not_roundtrip.newlines_in_expanded_parent

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return (
        self.roundtrip.layout_marks_in_this_part
        | self.not_roundtrip.layout_marks_in_this_part
    )

  def foldables_in_this_part(self) -> Sequence[FoldableTreeNode]:
    # Defer to the not_roundtrip version for expansion decisions.
    return self.not_roundtrip.foldables_in_this_part()

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    if not isinstance(self.roundtrip, EmptyPart):
      stream.write('<span class="when_roundtrip">')
      self.roundtrip.render_to_html(
          stream,
          at_beginning_of_line=at_beginning_of_line,
          render_context=render_context,
      )
      stream.write("</span>")

    if not isinstance(self.not_roundtrip, EmptyPart):
      stream.write('<span class="when_not_roundtrip">')
      self.not_roundtrip.render_to_html(
          stream,
          at_beginning_of_line=at_beginning_of_line,
          render_context=render_context,
      )
      stream.write("</span>")

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    if roundtrip_mode:
      self.roundtrip.render_to_text(
          stream,
          expanded_parent=expanded_parent,
          indent=indent,
          roundtrip_mode=roundtrip_mode,
          render_context=render_context,
      )
    else:
      self.not_roundtrip.render_to_text(
          stream,
          expanded_parent=expanded_parent,
          indent=indent,
          roundtrip_mode=roundtrip_mode,
          render_context=render_context,
      )


def roundtrip_condition(
    roundtrip: RenderableTreePart | None = None,
    not_roundtrip: RenderableTreePart | None = None,
) -> RenderableTreePart:
  """Builds a part that renders differently in roundtrip mode.

  Args:
    roundtrip: Contents to render when rendering in round trip mode.
    not_roundtrip: Contents to render when rendering in ordinary mode.

  Returns:
    A renderable part that renders as ``roundtrip`` in roundtrip mode
    and as ``not_roundtrip`` in ordinary mode.
  """
  if roundtrip is None:
    roundtrip = EmptyPart()
  if not_roundtrip is None:
    not_roundtrip = EmptyPart()
  if not isinstance(roundtrip, RenderableTreePart):
    raise ValueError(
        "`roundtrip` must be a renderable part or None. Got"
        f" {type(roundtrip).__name__}"
    )
  if not isinstance(not_roundtrip, RenderableTreePart):
    raise ValueError(
        "`not_roundtrip` must be a renderable part or None. Got"
        f" {type(not_roundtrip).__name__}"
    )
  return RoundtripCondition(roundtrip=roundtrip, not_roundtrip=not_roundtrip)


@dataclasses.dataclass(frozen=True)
class SummarizableCondition(RenderableTreePart):
  """Renders conditionally depending on combination of roundtrip/collapsed.

  The idea is that, when collapsed and not in roundtrip mode, it's sometimes
  convenient to summarize a compound node with a simpler non-roundtrippable
  representation.

  Attributes:
    summary: Contents to render when collapsed and not in roundtrip mode.
    detail: Contents to render when either expanded or in roundtrip mode.
  """

  summary: RenderableTreePart = EmptyPart()
  detail: RenderableTreePart = EmptyPart()

  def html_setup_parts(
      self, context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    rule = html_escaping.without_repeated_whitespace(f"""
        .when_expanded_or_roundtrip {{
            display: none;
        }}
        .when_expanded_or_roundtrip:not({context.collapsed_selector} *),
        {context.roundtrip_selector} .when_expanded_or_roundtrip {{
            display: inline;
        }}
        .when_collapsed_and_not_roundtrip {{
            display: inline;
        }}
        .when_collapsed_and_not_roundtrip:not({context.collapsed_selector} *),
        {context.roundtrip_selector} .when_collapsed_and_not_roundtrip {{
            display: none;
        }}
        """)
    return (
        {CSSStyleRule(rule)}
        | self.summary.html_setup_parts(context)
        | self.detail.html_setup_parts(context)
    )

  def _compute_collapsed_width(self) -> int:
    # Defer to the summary version for expansion decisions.
    return self.summary.collapsed_width

  def _compute_newlines_in_expanded_parent(self) -> int:
    # Defer to the detail version for expansion decisions.
    return self.detail.newlines_in_expanded_parent

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return (
        self.summary.layout_marks_in_this_part
        | self.detail.layout_marks_in_this_part
    )

  def foldables_in_this_part(self) -> Sequence[FoldableTreeNode]:
    # Defer to the detail version for expansion decisions.
    return self.detail.foldables_in_this_part()

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    if not isinstance(self.summary, EmptyPart):
      stream.write('<span class="when_collapsed_and_not_roundtrip">')
      self.summary.render_to_html(
          stream,
          at_beginning_of_line=at_beginning_of_line,
          render_context=render_context,
      )
      stream.write("</span>")

    if not isinstance(self.detail, EmptyPart):
      stream.write('<span class="when_expanded_or_roundtrip">')
      self.detail.render_to_html(
          stream,
          at_beginning_of_line=at_beginning_of_line,
          render_context=render_context,
      )
      stream.write("</span>")

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    if not expanded_parent and not roundtrip_mode:
      self.summary.render_to_text(
          stream,
          expanded_parent=expanded_parent,
          indent=indent,
          roundtrip_mode=roundtrip_mode,
          render_context=render_context,
      )
    else:
      self.detail.render_to_text(
          stream,
          expanded_parent=expanded_parent,
          indent=indent,
          roundtrip_mode=roundtrip_mode,
          render_context=render_context,
      )


def summarizable_condition(
    summary: RenderableTreePart | None = None,
    detail: RenderableTreePart | None = None,
) -> RenderableTreePart:
  """Builds a part that renders depending on combination of roundtrip/collapsed.

  The idea is that, when collapsed and not in roundtrip mode, it's sometimes
  convenient to summarize a compound node with a simpler non-roundtrippable
  representation.

  Args:
    summary: Contents to render when collapsed and not in roundtrip mode.
    detail: Contents to render when either expanded or in roundtrip mode.

  Returns:
    A renderable part that renders as ``summary`` when both collapsed and not
    in roundtrip mode, and as ``detail`` otherwise.
  """
  if summary is None:
    summary = EmptyPart()
  if detail is None:
    detail = EmptyPart()
  if not isinstance(summary, RenderableTreePart):
    raise ValueError(
        "`summary` must be a renderable part or None. Got"
        f" {type(summary).__name__}"
    )
  if not isinstance(detail, RenderableTreePart):
    raise ValueError(
        "`detail` must be a renderable part or None. Got"
        f" {type(detail).__name__}"
    )
  return SummarizableCondition(summary=summary, detail=detail)


################################################################################
# Line comments
################################################################################


def siblings_with_annotations(
    *args: str | RenderableTreePart | RenderableAndLineAnnotations,
    extra_annotations: Sequence[RenderableTreePart] = (),
) -> RenderableAndLineAnnotations:
  """Combines siblings that may have annotations, aggregating separately.

  This can be used to lay out multiple objects on the same line, when some
  may have annotations.

  Args:
    *args: Sequence of strings, renderable tree parts, or commented tree parts
      to render.
    extra_annotations: Additional annotations to add.

  Returns:
    A new pair of renderable and annotations, with main renderables and
    annotations combined separately.
  """
  parts = []
  annotations = []
  for arg in args:
    if isinstance(arg, RenderableTreePart):
      parts.append(arg)
    elif isinstance(arg, str):
      parts.append(Text(arg))
    elif isinstance(arg, RenderableAndLineAnnotations):
      parts.append(arg.renderable)
      if arg.annotations is not None:
        annotations.append(arg.annotations)
    else:
      raise ValueError(
          "Expected a renderable tree part (possibly with line annotations) or"
          f" a string, but got: {type(arg)}"
      )

  for annotation in extra_annotations:
    annotations.append(annotation)

  return RenderableAndLineAnnotations(siblings(*parts), siblings(*annotations))


def build_full_line_with_annotations(
    *args: str | RenderableTreePart | RenderableAndLineAnnotations,
) -> RenderableTreePart:
  """Concatenates tree parts and appends their line comments at the end.

  To ensure that the output is formatted correctly, the output of this function
  should always appear at the end of a line when its parent is expanded. This
  is not checked. (Failing to do this may cause delimiters to be erroneously
  commented out.)

  Args:
    *args: Sequence of strings, renderable tree parts, or commented tree parts
      to render.

  Returns:
    A renderable tree part that combines all of the main renderables, and also
    shows the line annotations on the right when expanded.
  """
  combined = siblings_with_annotations(*args)
  if (
      combined.annotations is None
      or isinstance(combined.annotations, EmptyPart)
      or (
          isinstance(combined.annotations, Siblings)
          and not combined.annotations.children
      )
  ):
    return combined.renderable
  return siblings(
      combined.renderable,
      FoldCondition(expanded=combined.annotations),
  )


################################################################################
# Structure combinators
################################################################################


@dataclasses.dataclass(frozen=True)
class OnSeparateLines(RenderableTreePart):
  """A sequence of children, one per line, not indented.

  Attributes:
    children: Children to render on separate lines when expanded.
  """

  children: Sequence[RenderableTreePart]

  @classmethod
  def build(
      cls,
      children: Sequence[RenderableAndLineAnnotations | RenderableTreePart],
  ) -> OnSeparateLines:
    """Builds a OnSeparateLines instance, supporting annotations."""
    return cls([build_full_line_with_annotations(line) for line in children])

  def _compute_collapsed_width(self) -> int:
    # When collapsed, all children appear inline.
    return sum(part.collapsed_width for part in self.children)

  def _compute_newlines_in_expanded_parent(self) -> int:
    # When expanded, we format every child on its own line. This introduces
    # len(children) + 1 newlines total, under the assumption that there was
    # some inline content before and after this IndentedChildren node.
    return (
        sum(part.newlines_in_expanded_parent + 1 for part in self.children) + 1
    )

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return functools.reduce(
        operator.or_,
        (part.layout_marks_in_this_part for part in self.children),
        frozenset(),
    )

  def foldables_in_this_part(self) -> Sequence[FoldableTreeNode]:
    return list(
        itertools.chain(
            *(part.foldables_in_this_part() for part in self.children)
        )
    )

  def html_setup_parts(
      self, context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    rule = html_escaping.without_repeated_whitespace(f"""
        .separate_lines_child:not({context.collapsed_selector} *)
        {{
            display: block;
        }}
        .separate_lines_children:not({context.collapsed_selector} *)
        {{
            display: block;
        }}
        """)
    return functools.reduce(
        operator.or_,
        (part.html_setup_parts(context) for part in self.children),
        {CSSStyleRule(rule)},
    )

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    stream.write('<span class="separate_lines_children">')
    for part in self.children:
      # All children render at the beginning of their line.
      stream.write('<span class="separate_lines_child">')
      part.render_to_html(
          stream,
          at_beginning_of_line=True,
          render_context=render_context,
      )
      stream.write("</span>")

    stream.write("</span>")

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    # Insert separators before and after each child if in expanded mode.
    if expanded_parent:
      separator = "\n" + " " * indent
    else:
      separator = ""

    for line_part in self.children:
      stream.write(separator)
      line_part.render_to_text(
          stream,
          expanded_parent=expanded_parent,
          indent=indent,
          roundtrip_mode=roundtrip_mode,
          render_context=render_context,
      )

    if expanded_parent:
      stream.write("\n" + " " * indent)


def on_separate_lines(
    children: Sequence[RenderableAndLineAnnotations | RenderableTreePart],
) -> RenderableTreePart:
  """Builds a part that renders its children on separate lines.

  The resulting part stacks the children together, moving any comments to the
  end of their lines.

  Args:
    children: Children to render.

  Returns:
    A renderable part that renders the children on separate lines when expanded.
    When collapsed, it instead concatenates them.
  """
  return OnSeparateLines.build(children)


@dataclasses.dataclass(frozen=True)
class IndentedChildren(RenderableTreePart):
  """A sequence of children, one per line, and indented.

  IndentedChildren is the primary way to lay out the content of ordinary
  containers across multiple lines.

  Attributes:
    children: Children to render on separate lines when expanded.
  """

  children: Sequence[RenderableTreePart]

  @classmethod
  def build(
      cls,
      children: Sequence[RenderableAndLineAnnotations | RenderableTreePart],
      comma_separated: bool = False,
      force_trailing_comma: bool = False,
  ) -> IndentedChildren:
    """Builds a IndentedChildren instance."""
    lines = []
    for i, child in enumerate(children):
      if comma_separated:
        if i < len(children) - 1:
          # Not the last child. Always show a comma, and add a space when
          # collapsed.
          delimiter = siblings(",", FoldCondition(collapsed=Text(" ")))
        elif force_trailing_comma:
          # Last child, forced comma.
          delimiter = Text(",")
        else:
          # Last child, but only show the comma when the node is expanded.
          delimiter = FoldCondition(expanded=Text(","))
        line = build_full_line_with_annotations(child, delimiter)
      else:
        line = build_full_line_with_annotations(child)

      lines.append(line)

    return cls(lines)

  def _compute_collapsed_width(self) -> int:
    # When collapsed, all children appear inline.
    return sum(part.collapsed_width for part in self.children)

  def _compute_newlines_in_expanded_parent(self) -> int:
    # When expanded, we format every child on its own line. This introduces
    # len(children) + 1 newlines total, under the assumption that there was
    # some inline content before and after this IndentedChildren node.
    return (
        sum(part.newlines_in_expanded_parent + 1 for part in self.children) + 1
    )

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return functools.reduce(
        operator.or_,
        (part.layout_marks_in_this_part for part in self.children),
        frozenset(),
    )

  def foldables_in_this_part(self) -> Sequence[FoldableTreeNode]:
    return list(
        itertools.chain(
            *(part.foldables_in_this_part() for part in self.children)
        )
    )

  def html_setup_parts(
      self, context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    rule = html_escaping.without_repeated_whitespace(f"""
        .indented_children {{
            contain: content;
        }}
        .indented_child:not({context.collapsed_selector} *)
        {{
            display: block;
            margin-left: calc(2ch - 1px);
        }}
        .stacked_children:not({context.collapsed_selector} *)
        {{
            display: block;
            border-left: dotted 1px #e0e0e0;
        }}
        """)
    return functools.reduce(
        operator.or_,
        (part.html_setup_parts(context) for part in self.children),
        {CSSStyleRule(rule)},
    )

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    stream.write('<span class="indented_children">')
    for part in self.children:
      # All children render at the beginning of their line.
      stream.write('<span class="indented_child">')
      part.render_to_html(
          stream,
          at_beginning_of_line=True,
          render_context=render_context,
      )
      stream.write("</span>")

    stream.write("</span>")

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    # Insert separators before and after each child if in expanded mode.
    if expanded_parent:
      next_indent = indent + 2
      separator = "\n" + " " * next_indent
    else:
      next_indent = indent
      separator = ""

    for line_part in self.children:
      stream.write(separator)
      line_part.render_to_text(
          stream,
          expanded_parent=expanded_parent,
          indent=next_indent,
          roundtrip_mode=roundtrip_mode,
          render_context=render_context,
      )

    if expanded_parent:
      stream.write("\n" + " " * indent)


def indented_children(
    children: Sequence[RenderableAndLineAnnotations | RenderableTreePart],
    comma_separated: bool = False,
    force_trailing_comma: bool = False,
) -> RenderableTreePart:
  """Builds a IndentedChildren instance, supporting annotations and delimiters.

  This method stacks the children together, optionally inserting delimiters,
  and moving any comments to the end of their lines.

  Args:
    children: Children to render.
    comma_separated: Whether to automatically insert commas between children. If
      False, delimiters can be manually inserted into `children` first instead.
    force_trailing_comma: Whether to render a trailing comma in collapsed mode.

  Returns:
    A renderable part that renders the children on separate lines with an
    indent. When collapsed, it instead concatenates them.
  """
  return IndentedChildren.build(
      children=children,
      comma_separated=comma_separated,
      force_trailing_comma=force_trailing_comma,
  )


@dataclasses.dataclass(frozen=True)
class StyledBoxWithOutline(RenderableTreePart, abc.ABC):
  """An outlined box, which displays in "block" mode when rendered to HTML.

  Outlined boxes ensure that their child appears as a contiguous chunk, instead
  of having its first line indented, so that it can be fully encapsulated in
  a box.

  When rendered to HTML, this class may or may not insert extra newlines before
  and after the child, depending on whether this child was already alone on its
  line. When rendered to text, we always insert extra comments above and below
  the line.

  To allow the box to be collapsed separately, consider wrapping it in a
  foldable node.

  Attributes:
    child: The child to render.
    css_style: The style for the inner box.
  """

  child: RenderableTreePart
  css_style: str

  # Implementations of parent class abstract methods.
  def _compute_collapsed_width(self) -> int:
    return self.child.collapsed_width

  def _compute_newlines_in_expanded_parent(self) -> int:
    # Conservatively assume we will insert newlines when expanded
    # (may or may not be true).
    count = 2 + self.child.newlines_in_expanded_parent
    return count

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return self.child.layout_marks_in_this_part

  def foldables_in_this_part(self) -> Sequence[FoldableTreeNode]:
    return self.child.foldables_in_this_part()

  def html_setup_parts(
      self, context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    base_rule = html_escaping.without_repeated_whitespace(f"""
        .outerbox_for_outline
        {{
            display: inline-block;
            padding: 0.25em;
            box-sizing: border-box;
        }}
        .box_with_outline
        {{
            display: inline-block;
            padding: 0.25em;
            box-sizing: border-box;
        }}
        .box_with_outline:not({context.collapsed_selector} *),
        .outerbox_for_outline:not({context.collapsed_selector} *)
        {{
            display: block;
            width: max-content;
        }}
        """)
    return {CSSStyleRule(base_rule)} | self.child.html_setup_parts(context)

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    style = html_escaping.escape_html_attribute(self.css_style)
    stream.write(
        '<span class="outerbox_for_outline"><span class="box_with_outline" '
        f'style="{style}">'
    )
    # Always treat the child as if it is at the beginning of the line, since it
    # will be once the box is expanded.
    self.child.render_to_html(
        stream,
        at_beginning_of_line=True,
        render_context=render_context,
    )
    stream.write("</span></span>")

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    # Insert separators before and after each child if in expanded mode.
    boxwidth = 40
    if expanded_parent:
      stream.write("\n" + " " * indent + "#╭" + "┄" * boxwidth + "╮\n")
      stream.write("\n" + " " * indent)

    self.child.render_to_text(
        stream,
        expanded_parent=expanded_parent,
        indent=indent,
        roundtrip_mode=roundtrip_mode,
        render_context=render_context,
    )

    if expanded_parent:
      stream.write("\n" + " " * indent + "#╰" + "┄" * boxwidth + "╯")
      stream.write("\n" + " " * indent)


def in_outlined_box(
    child: part_interface.RenderableTreePart,
    css_style: str = "outline: 1px dashed #aaaaaa;",
) -> RenderableTreePart:
  """Wraps a child into an outlined box.

  Outlined boxes ensure that their child appears as a contiguous chunk, instead
  of having its first line indented, so that it can be fully encapsulated in
  a box.

  When rendered to HTML, this class may or may not insert extra newlines before
  and after the child, depending on whether this child was already alone on its
  line. When rendered to text, we always insert extra comments above and below
  the line.

  To allow the box to be collapsed separately, consider wrapping it in a
  foldable node.

  Args:
    child: The child to render.
    css_style: The CSS style for the box element. By default, renders with a
      dashed grey outline.

  Returns:
    A renderable part that renders the child inside a standalone box.
  """
  if not isinstance(child, RenderableTreePart):
    raise ValueError(
        f"`child` must be a renderable part, but got {type(child).__name__}"
    )
  if not isinstance(css_style, str):
    raise ValueError(
        f"`css_style` must be a string, but got {type(css_style).__name__}"
    )
  return StyledBoxWithOutline(child, css_style)


@dataclasses.dataclass(frozen=True)
class WithHoverTooltip(DeferringToChild):
  """Customized renderer that shows extra info on hover.

  Attributes:
    child: Child to render.
    tooltip: Text to show when hovered over with the mouse.
  """

  child: RenderableTreePart
  tooltip: str

  def html_setup_parts(
      self, context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    rule = CSSStyleRule(html_escaping.without_repeated_whitespace("""
      .has_hover_tooltip {
        text-decoration: underline dotted;
        position: relative;
      }
      .has_hover_tooltip:hover::after  {
        display: block;
        position: absolute;
        top: calc(100% + 0.2ch);
        left: 0.2ch;
        content: attr(data-tooltip);
        outline: 1px dashed oklch(30% 0 0);
        background-color: oklch(90% 0 0);
        color: oklch(30% 0 0);
        z-index: 100;
        padding: 0.2ch;
        white-space: pre;
      }
    """))
    return {rule} | self.child.html_setup_parts(context)

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    tooltip_attr = html_escaping.escape_html_attribute(self.tooltip)
    stream.write(
        f'<span class="has_hover_tooltip" data-tooltip="{tooltip_attr}">'
    )
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")


@dataclasses.dataclass(frozen=True)
class ScopedSelectableAnnotation(DeferringToChild):
  """Modifies its child so that selections outside it don't include it.

  It is sometimes useful to add additional annotations to an object that aren't
  pretty-printed parts of that object. This causes some problems for ordinary
  roundtrip mode, since we want it to be possible to exactly round-trip an
  object based on its printed representation.

  This object marks its child so that it doesn't get selected by the mouse
  when the selection starts outside the node. This means that if you copy the
  object normally, you don't copy the annotation, so that what you copied
  stays roundtrippable.

  In text mode, selections can't be manipulated. We fake the same thing by
  rendering it as a "comment" (by adding comment markers before every line)
  and hiding it if collapsed.

  Attributes:
    child: Child to render.
  """

  child: part_interface.RenderableTreePart

  def html_setup_parts(
      self, context: part_interface.HtmlContextForSetup
  ) -> set[part_interface.CSSStyleRule | part_interface.JavaScriptDefn]:
    # We want to ensure that the non-roundtrippable child does not get copied
    # if the user is trying to copy the roundtrippable version of the tree.
    # However, if the user wants to copy part of the non-roundtrippable child
    # itself, we don't want to get in the way.
    # We accomplish this by using the HTML/CSS "focus" mechanism, which the
    # browser automatically handles for anything with a "tabindex" attribute.
    # Specifically, when the user clicks on the rendered object, that object
    # gains focus, and when they click outside it (e.g. on a parent of this
    # object), the object looses focus. We can then disable text selection
    # when the object doesn't have focus, which means that if a text selection
    # starts outside the object, it won't include anything inside the object.
    rule = html_escaping.without_repeated_whitespace("""
        .scoped_unselectable:not(:focus)
        {
            user-select: none;
        }
    """)

    return {part_interface.CSSStyleRule(rule)} | self.child.html_setup_parts(
        context
    )

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    # We need to set the "tabindex" attribute to make the text selection
    # adjustment work. -1 means that this should be selectable by the mouse
    # but should not be selectable by pressing "tab" on the keyboard or using
    # assistive tools.
    stream.write('<span tabindex="-1" class="scoped_unselectable">')
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    if roundtrip_mode:
      if not expanded_parent:
        # Don't render anything when collapsed and in roundtrip mode.
        return
      local_stream = io.StringIO()
      self.child.render_to_text(
          local_stream,
          expanded_parent=expanded_parent,
          indent=0,
          roundtrip_mode=roundtrip_mode,
          render_context=render_context,
      )
      for line in local_stream.getvalue().splitlines():
        if line:
          stream.write("\n" + " " * indent + "# ")
          stream.write(line)
        else:
          stream.write("\n")
      stream.write("\n" + " " * indent)
    else:
      self.child.render_to_text(
          stream,
          expanded_parent=expanded_parent,
          indent=indent,
          roundtrip_mode=roundtrip_mode,
          render_context=render_context,
      )


def floating_annotation_with_separate_focus(
    child: part_interface.RenderableTreePart,
) -> part_interface.RenderableTreePart:
  """Wraps a child so that selections outside it don't include it.

  It is sometimes useful to add additional annotations to an object that aren't
  pretty-printed parts of that object. This causes some problems for ordinary
  roundtrip mode, since we want it to be possible to exactly round-trip an
  object based on its printed representation.

  This object marks its child so that it doesn't get selected by the mouse
  when the selection starts outside the node. This means that if you copy the
  object normally, you don't copy the annotation, so that what you copied
  stays roundtrippable.

  In text mode, selections can't be manipulated. We fake the same thing by
  rendering it as a "comment" (by adding comment markers before every line)
  and hiding it if collapsed.

  Args:
    child: Child to render.

  Returns:
    A wrapped version of ``child`` that does not participate in text selection
    in HTML mode, and adds comments in text mode.
  """
  if not isinstance(child, RenderableTreePart):
    raise ValueError(
        f"`child` must be a renderable part, but got {type(child).__name__}"
    )
  return ScopedSelectableAnnotation(child)
