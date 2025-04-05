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

"""Common styles for parts of renderings."""

from __future__ import annotations

import dataclasses
import io
from typing import Any

from treescope._internal import html_escaping
from treescope._internal.parts import basic_parts
from treescope._internal.parts import part_interface


CSSStyleRule = part_interface.CSSStyleRule
JavaScriptDefn = part_interface.JavaScriptDefn
RenderableTreePart = part_interface.RenderableTreePart
HtmlContextForSetup = part_interface.HtmlContextForSetup


class AbbreviationColor(basic_parts.BaseSpanGroup):
  """Renders its child in a color for non-roundtrippable abbreviations."""

  def _span_css_class(self) -> str:
    return "color_abbrev"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
      .color_abbrev
      {
        color: #682a00;
      }
    """))


def abbreviation_color(
    child: part_interface.RenderableTreePart,
) -> part_interface.RenderableTreePart:
  """Returns a wrapped child in a color for non-roundtrippable abbreviations."""
  if not isinstance(child, RenderableTreePart):
    raise ValueError(
        f"`child` must be a renderable part, but got {type(child).__name__}"
    )
  return AbbreviationColor(child)


class CommentColor(basic_parts.BaseSpanGroup):
  """Renders its child in a color for comments."""

  def _span_css_class(self) -> str:
    return "color_comment"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
      .color_comment {
          color: #aaaaaa;
      }
      .color_comment a:not(:hover){
          color: #aaaaaa;
      }
    """))


def comment_color(
    child: part_interface.RenderableTreePart,
) -> part_interface.RenderableTreePart:
  """Returns a wrapped child in a color for comments."""
  if not isinstance(child, RenderableTreePart):
    raise ValueError(
        f"`child` must be a renderable part, but got {type(child).__name__}"
    )
  return CommentColor(child)


class ErrorColor(basic_parts.BaseSpanGroup):
  """Renders its child in red to indicate errors / problems during rendering."""

  def _span_css_class(self) -> str:
    return "color_error"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
      .color_error {
          color: red;
      }
    """))


def error_color(
    child: part_interface.RenderableTreePart,
) -> part_interface.RenderableTreePart:
  """Returns a wrapped child in red to indicate errors during rendering."""
  if not isinstance(child, RenderableTreePart):
    raise ValueError(
        f"`child` must be a renderable part, but got {type(child).__name__}"
    )
  return ErrorColor(child)


class DeferredPlaceholderStyle(basic_parts.BaseSpanGroup):
  """Renders its child in italics to indicate a deferred placeholder."""

  def _span_css_class(self) -> str:
    return "deferred_placeholder"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
      .deferred_placeholder {
          color: #a7a7a7;
          font-style: italic;
      }
    """))


def deferred_placeholder_style(
    child: part_interface.RenderableTreePart,
) -> part_interface.RenderableTreePart:
  """Returns a wrapped child in italics to indicate a deferred placeholder."""
  if not isinstance(child, RenderableTreePart):
    raise ValueError(
        f"`child` must be a renderable part, but got {type(child).__name__}"
    )
  return DeferredPlaceholderStyle(child)


class CommentColorWhenExpanded(basic_parts.BaseSpanGroup):
  """Renders its child in a color for comments, but only when expanded.

  This can be used in combination with another color (usually AbbreviationColor)
  to show a summary and then transform it into a comment when clicked, e.g. with

  AbbreviationColor(CommentColorWhenExpanded(...))
  """

  def _span_css_class(self) -> str:
    return "color_comment_when_expanded"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace(f"""
      .color_comment_when_expanded:not({context.collapsed_selector} *) {{
          color: #aaaaaa;
      }}
    """))


def comment_color_when_expanded(
    child: part_interface.RenderableTreePart,
) -> part_interface.RenderableTreePart:
  """Returns a wrapped child in a color for comments, but only when expanded."""
  if not isinstance(child, RenderableTreePart):
    raise ValueError(
        f"`child` must be a renderable part, but got {type(child).__name__}"
    )
  return CommentColorWhenExpanded(child)


@dataclasses.dataclass(frozen=True)
class CustomTextColor(basic_parts.DeferringToChild):
  """A group that wraps its child in a span with a custom text color.

  Attributes:
    child: Contents of the group.
    color: CSS string defining the color.
  """

  child: RenderableTreePart
  color: str

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    colorstr = html_escaping.escape_html_attribute(self.color)
    stream.write(f'<span style="color:{colorstr};">')
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")


def custom_text_color(
    child: part_interface.RenderableTreePart, css_color: str
) -> part_interface.RenderableTreePart:
  """Returns a wrapped child that renders in a particular CSS color."""
  if not isinstance(child, RenderableTreePart):
    raise ValueError(
        f"`child` must be a renderable part, but got {type(child).__name__}"
    )
  if not isinstance(css_color, str):
    raise ValueError(
        f"`css_color` must be a string, but got {type(css_color).__name__}"
    )
  return CustomTextColor(child, css_color)


@dataclasses.dataclass(frozen=True)
class ColoredBorderIndentedChildren(basic_parts.IndentedChildren):
  """A sequence of children that also draws a colored line on the left.

  Assumes that the CSS variables --block-color and --block-border-scale are
  set in some container object. These are set by WithBlockColor.
  """

  def html_setup_parts(
      self, context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    rule = html_escaping.without_repeated_whitespace(f"""
        .stacked_children {{
          contain: content;
        }}
        .stacked_children.colored_border
          > .stacked_child:not({context.collapsed_selector} *)
        {{
          display: block;
          margin-left: calc(2ch - 0.4ch);
        }}
        .stacked_children.colored_border:not({context.collapsed_selector} *)
        {{
          display: block;
          border-left: solid 0.4ch var(--block-color);
          padding-top: calc(0.5lh * (1 - var(--block-border-scale)));
          margin-top: calc(-0.5lh * (1 - var(--block-border-scale)));
          padding-bottom: 0.5lh;
          margin-bottom: -0.5lh;
        }}
        .stacked_children.colored_border:not({context.collapsed_selector} *)
        {{
          border-top-left-radius: calc(1em * var(--block-border-scale));
        }}
        {context.collapsed_selector} .stacked_children.colored_border
        {{
          border-top: solid 1px var(--block-color);
          border-bottom: solid 1px var(--block-color);
        }}
        """)
    return {CSSStyleRule(rule)} | super().html_setup_parts(context)

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    stream.write('<span class="stacked_children colored_border">')
    for part in self.children:
      # All children render at the beginning of their line.
      stream.write('<span class="stacked_child colored_border">')
      part.render_to_html(
          stream,
          at_beginning_of_line=True,
          render_context=render_context,
      )
      stream.write("</span>")

    stream.write("</span>")


def _common_block_rules(
    context: HtmlContextForSetup,
) -> dict[str, CSSStyleRule]:
  return {
      "colored_line": CSSStyleRule(html_escaping.without_repeated_whitespace("""
            .colored_line
            {
              color: black;
              background-color: var(--block-color);
              border-top: 1px solid var(--block-color);
              border-bottom: 1px solid var(--block-color);
            }
          """)),
      "patterned_line": CSSStyleRule(
          html_escaping.without_repeated_whitespace("""
            .patterned_line
            {
              color: black;
              border: 1px solid var(--block-color);
              background-image: var(--block-background-image);
              background-clip: padding-box;
            }
          """)
      ),
      "topline": CSSStyleRule(html_escaping.without_repeated_whitespace(f"""
            .topline:not({context.collapsed_selector} *)
            {{
              padding-bottom: 0.15em;
              line-height: 1.5em;
              z-index: 1;
              position: relative;
            }}
          """)),
      "bottomline": CSSStyleRule(html_escaping.without_repeated_whitespace(f"""
            .bottomline:not({context.collapsed_selector} *)
            {{
              padding-top: 0.15em;
              line-height: 1.5em;
              z-index: 1;
              position: relative;
            }}
          """)),
      "hch_space_left": CSSStyleRule(
          html_escaping.without_repeated_whitespace("""
            .hch_space_left
            {
              padding-left: 0.5ch;
            }
          """)
      ),
      "hch_space_right": CSSStyleRule(
          html_escaping.without_repeated_whitespace("""
            .hch_space_right
            {
              padding-right: 0.5ch;
            }
          """)
      ),
  }


class ColoredTopLineSpanGroup(basic_parts.BaseSpanGroup):
  """Customized span group that colors the prefix of a block."""

  def _span_css_class(self) -> str:
    return "colored_line topline hch_space_left"

  def _span_css_rule(self, context: HtmlContextForSetup) -> set[CSSStyleRule]:
    rules = _common_block_rules(context)
    return {rules["colored_line"], rules["topline"], rules["hch_space_left"]}


class ColoredSingleLineSpanGroup(basic_parts.BaseSpanGroup):
  """Customized span group that colors a single line."""

  def _span_css_class(self) -> str:
    return "colored_line hch_space_left hch_space_right"

  def _span_css_rule(self, context: HtmlContextForSetup) -> set[CSSStyleRule]:
    rules = _common_block_rules(context)
    return {
        rules["colored_line"],
        rules["hch_space_left"],
        rules["hch_space_right"],
    }


class ColoredBottomLineSpanGroup(basic_parts.BaseSpanGroup):
  """Customized span group that colors the suffix of a block."""

  def _span_css_class(self) -> str:
    return "colored_line bottomline hch_space_right"

  def _span_css_rule(self, context: HtmlContextForSetup) -> set[CSSStyleRule]:
    rules = _common_block_rules(context)
    return {
        rules["colored_line"],
        rules["bottomline"],
        rules["hch_space_right"],
    }


@dataclasses.dataclass
class WithBlockColor(basic_parts.DeferringToChild):
  """A group that configures block color CSS variables.

  This provides the necessary colors for ColoredBorderStackedChildren,
  ColoredTopLineSpanGroup, or ColoredBottomLineSpanGroup.

  Attributes:
    child: Contents of the group.
    color: CSS color to render.
  """

  child: RenderableTreePart
  color: str

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    style_string = f"--block-color: {self.color};"
    if at_beginning_of_line:
      style_string += "--block-border-scale: 0;"
    else:
      style_string += "--block-border-scale: 1;"
    stream.write(f'<span style="{style_string}">')
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")


class PatternedTopLineSpanGroup(basic_parts.BaseSpanGroup):
  """Customized span group that adds a pattern to the prefix of a block."""

  def _span_css_class(self) -> str:
    return "patterned_line topline hch_space_left"

  def _span_css_rule(self, context: HtmlContextForSetup) -> set[CSSStyleRule]:
    rules = _common_block_rules(context)
    return {rules["patterned_line"], rules["topline"], rules["hch_space_left"]}


class PatternedBottomLineSpanGroup(basic_parts.BaseSpanGroup):
  """Customized span group that adds a pattern to the suffix of a block."""

  def _span_css_class(self) -> str:
    return "patterned_line bottomline hch_space_left hch_space_right"

  def _span_css_rule(self, context: HtmlContextForSetup) -> set[CSSStyleRule]:
    rules = _common_block_rules(context)
    return {
        rules["patterned_line"],
        rules["bottomline"],
        rules["hch_space_left"],
        rules["hch_space_right"],
    }


class PatternedSingleLineSpanGroup(basic_parts.BaseSpanGroup):
  """Customized span group that adds a pattern to a single line."""

  def _span_css_class(self) -> str:
    return "patterned_line expand_right"

  def _span_css_rule(self, context: HtmlContextForSetup) -> set[CSSStyleRule]:
    rules = _common_block_rules(context)
    return {rules["patterned_line"], rules["expand_right"]}


@dataclasses.dataclass
class WithBlockPattern(basic_parts.DeferringToChild):
  """A group that configures block color CSS variables.

  This should be used around uses of PatternedTopLineSpanGroup,
  PatternedBottomLineSpanGroup, or PatternedSingleLineSpanGroup.

  Attributes:
    child: Contents of the group.
    color: CSS color for borders.
    pattern: CSS value for a background image pattern (e.g. using
      `repeating-linear-gradient`)
  """

  child: RenderableTreePart
  color: str
  pattern: str

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    style_string = (
        f"--block-color: {self.color}; "
        f"--block-background-image: {self.pattern};"
    )
    if at_beginning_of_line:
      style_string += "--block-border-scale: 0;"
    else:
      style_string += "--block-border-scale: 1;"
    stream.write(f'<span style="{style_string}">')
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")


class QualifiedTypeNameSpanGroup(basic_parts.BaseSpanGroup):
  """Customized span group that displays qualified names in a small font."""

  def _span_css_class(self) -> str:
    return "qualname_prefix"

  def _span_css_rule(self, context: HtmlContextForSetup) -> CSSStyleRule:
    return CSSStyleRule(html_escaping.without_repeated_whitespace("""
        .qualname_prefix {
          font-size: 0.8em;
        }
    """))


def qualified_type_name_style(
    child: part_interface.RenderableTreePart,
) -> part_interface.RenderableTreePart:
  """Returns a wrapped child in a small font to indicate a qualified name."""
  if not isinstance(child, RenderableTreePart):
    raise ValueError(
        f"`child` must be a renderable part, but got {type(child).__name__}"
    )
  return QualifiedTypeNameSpanGroup(child)


@dataclasses.dataclass(frozen=True)
class CSSStyled(basic_parts.DeferringToChild):
  """Adjusts the CSS style of its child.

  Attributes:
    child: Child to render.
    css: A CSS style string.
  """

  child: part_interface.RenderableTreePart
  style: str

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    style = html_escaping.escape_html_attribute(self.style)
    stream.write(f'<span style="{style}">')
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")


def custom_style(
    child: part_interface.RenderableTreePart,
    css_style: str,
) -> part_interface.RenderableTreePart:
  """Returns a wrapped child with a custom CSS style.

  Args:
    child: Child to render.
    css_style: A CSS style string.

  Returns:
    A wrapped child with a custom CSS style applied. Intended for an inline
    text component.
  """
  if not isinstance(child, RenderableTreePart):
    raise ValueError(
        f"`child` must be a renderable part, but got {type(child).__name__}"
    )
  if not isinstance(css_style, str):
    raise ValueError(
        f"`css_style` must be a string, but got {type(css_style).__name__}"
    )
  return CSSStyled(child, css_style)
