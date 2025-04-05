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

"""Low-level implementation details of the foldable system.

This module contains low-level wrappers that handle folding and unfolding,
path copy button rendering, deferred rendering, and abbreviation.
"""

from __future__ import annotations

import dataclasses
import io
from typing import Any, Callable, Sequence

from treescope._internal import html_escaping
from treescope._internal.parts import basic_parts
from treescope._internal.parts import common_styles
from treescope._internal.parts import part_interface


CSSStyleRule = part_interface.CSSStyleRule
JavaScriptDefn = part_interface.JavaScriptDefn
HtmlContextForSetup = part_interface.HtmlContextForSetup
RenderableTreePart = part_interface.RenderableTreePart
ExpandState = part_interface.ExpandState
FoldableTreeNode = part_interface.FoldableTreeNode


COLLAPSED_SELECTOR = (
    ".foldable_node:has(>label>.foldable_node_toggle:not(:checked))"
)
ROUNDTRIP_SELECTOR = ".treescope_root.roundtrip_mode"
NOT_ROUNDTRIP_SELECTOR = ".treescope_root:not(.roundtrip_mode)"

ABBREVIATION_LEVEL_KEY = "abbreviation_level"
ABBREVIATION_THRESHOLD_KEY = "abbreviation_threshold"
ABBREVIATION_LEVEL_CLASS = "abbr_lvl"


################################################################################
# Foldable node implementation
################################################################################


@dataclasses.dataclass(frozen=False)
class FoldableTreeNodeImpl(FoldableTreeNode):
  """Concrete implementation of a node that can be expanded or collapsed.

  This is kept separate from the abstract definition of a FoldableTreeNode to
  avoid strong dependencies on its implementation as much as possible.

  Attributes:
    contents: Contents of the foldable node.
    label: Optional label for the foldable node. This appears in front of the
      contents, and clicking it expands or collapses the foldable node. This
      should not contain any other foldables and should generally be a single
      line.
    expand_state: Current expand state for the node.
  """

  contents: RenderableTreePart
  label: RenderableTreePart = basic_parts.EmptyPart()
  expand_state: ExpandState = ExpandState.WEAKLY_COLLAPSED

  def get_expand_state(self) -> ExpandState:
    """Returns the node's expand state."""
    return self.expand_state

  def set_expand_state(self, expand_state: ExpandState):
    """Sets the node's expand state."""
    self.expand_state = expand_state

  def as_expanded_part(self) -> RenderableTreePart:
    """Returns the contents of this foldable when expanded."""
    return basic_parts.siblings(self.label, self.contents)

  def html_setup_parts(
      self, setup_context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    if setup_context.collapsed_selector != COLLAPSED_SELECTOR:
      raise ValueError(
          "FoldableTreeNodeImpl only works properly when the tree is"
          " configured using foldable_impl.COLLAPSED_SELECTOR"
      )
    # These CSS rules ensure that:
    # - Fold markers appear relative to the "foldable_node" HTML element
    # - The checkbox we are using to track collapsed state is hidden.
    # - Before the label, if this node isn't inside a collapsed parent node,
    #   we insert a floating expand/collapse marker depending on whether the
    #   checkbox is checked, and style it appropriately.
    # - If this node is in a collapsed parent node, don't toggle its state when
    #   clicked, since that won't do anything. If it is not, then show a pointer
    #   cursor to indicate that clicking will do something.
    # - When this is the first object on its line, shift the triangle marker
    #   left into the margin. Otherwise, shift the contents to the right.
    rule = html_escaping.without_repeated_whitespace(f"""
        .foldable_node
        {{
            position: relative;
        }}

        .foldable_node_toggle
        {{
            display: none;
        }}

        .foldable_node > label::before
        {{
            color: #cccccc;
            position: relative;
            left: -1ch;
            width: 0;
            display: inline-block;
        }}
        .foldable_node > label:hover::before
        {{
            color: darkseagreen;
        }}

        .foldable_node:not({setup_context.collapsed_selector} *)
          > label:has(>.foldable_node_toggle:checked)::before
        {{
            content: '\\25bc';
        }}
        .foldable_node:not({setup_context.collapsed_selector} *)
          > label:has(>.foldable_node_toggle:not(:checked))::before
        {{
            content: '\\25b6';
        }}

        .foldable_node:not({setup_context.collapsed_selector} *) > label:hover
        {{
            cursor: pointer;
        }}
        .foldable_node:is({setup_context.collapsed_selector} *) > label
        {{
            pointer-events: none;
        }}
        .foldable_node:is({setup_context.collapsed_selector} *)
        {{
            cursor: text;
        }}

        .foldable_node.is_first_on_line > label::before
        {{
            position: relative;
            left: -1.25ch;
        }}
        .foldable_node:not(
            {setup_context.collapsed_selector} *):not(.is_first_on_line)
        {{
            margin-left: 1ch;
        }}
        """)
    return (
        {CSSStyleRule(rule)}
        | self.contents.html_setup_parts(setup_context)
        | self.label.html_setup_parts(setup_context)
    )

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    if at_beginning_of_line:
      classname = "foldable_node is_first_on_line"
    else:
      classname = "foldable_node"

    stream.write(f'<span class="{classname}"><label>')

    stream.write('<input type="checkbox" class="foldable_node_toggle"')
    if (
        self.expand_state == ExpandState.WEAKLY_EXPANDED
        or self.expand_state == ExpandState.EXPANDED
    ):
      stream.write(" checked")
    stream.write("></input>")

    self.label.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</label>")
    self.contents.render_to_html(
        stream,
        at_beginning_of_line=isinstance(self.label, basic_parts.EmptyPart),
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
    # In text mode, we just render the label and contents with no additional
    # wrapping. Note that we only expand if this node is expanded AND the
    # parents are expanded, since collapsed parents override children.
    expanded_here = expanded_parent and (
        self.expand_state == ExpandState.WEAKLY_EXPANDED
        or self.expand_state == ExpandState.EXPANDED
    )
    self.label.render_to_text(
        stream,
        expanded_parent=expanded_here,
        indent=indent,
        roundtrip_mode=roundtrip_mode,
        render_context=render_context,
    )
    self.contents.render_to_text(
        stream,
        expanded_parent=expanded_here,
        indent=indent,
        roundtrip_mode=roundtrip_mode,
        render_context=render_context,
    )


################################################################################
# Path copy button implementation
################################################################################


@dataclasses.dataclass
class StringCopyButton(RenderableTreePart):
  """Builds a button that, when clicked, copies the given path.

  This does nothing in text rendering mode, and only appears when its parent
  is expanded.

  Attributes:
    copy_string: String to copy when clicked
    annotation: Annotation to show when the button is hovered over.
  """

  copy_string: str
  annotation: str = "Copy: "

  def _compute_collapsed_width(self) -> int:
    return 0

  def _compute_newlines_in_expanded_parent(self) -> int:
    return 0

  def foldables_in_this_part(self) -> Sequence[FoldableTreeNode]:
    return ()

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return frozenset()

  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    # Doesn't render at all in text mode.
    pass

  def html_setup_parts(
      self, setup_context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    # pylint: disable=line-too-long
    # https://github.com/google/material-design-icons/blob/master/symbols/web/content_copy/materialsymbolsoutlined/content_copy_grad200_20px.svg
    font_data_url = "data:font/woff2;base64,d09GMgABAAAAAAIQAAoAAAAABRwAAAHFAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAABmAAgkoKdIEBCwYAATYCJAMIBCAFgxIHLhtsBMieg3FDX2LI6YvSpuiPM5T1JXIRVMvWMztPyFFC+WgkTiBD0hiDQuJEdGj0Hb/fvIdpqK6hqWiiQuMnGhHfUtAU6BNr4AFInkE6cUuun+R5qcskwvfFl/qxgEo8gbJwG81HA/nAR5LrrJ1R+gz0Rd0AJf1gN7CwGj2g0oyuR77mE16wHX9OggpeTky4eIbz5cbrOGtaAgQINwDasysQuIIXWEFwAPQpIYdU//+g7T7X3t0fKPqAv52g0LAN7AMwAmgzRS+uZSeEXx2f6czN4RHy5uBAKzBjpFp3iHQCE0ZuP4S7nfBLEHFMmAi+8vE2hn1h7+bVwXjwHrvDGUCnjfEEgt+OcZll759CJwB8h94MMGS3GZAgmI5jBQ9tTGeH9EBBIG3Dg4R/YcybAGEAAVK/AQGaAeMClAHzEOgZtg6BPgOOIDBkiQ5eFBXCBFci0phropnQAApZED1z1kSfCfthyKnHdaFsHf0NmGEN6BdAqVVpatsSZmddai92fz94Uijq6pmr6OoYCSirGmvJG3SWS3FE2cBQfT+HlopG4Fsw5agq68iZeSNlpWnBHIedMreuWqGCm1WFrkSSx526WWswAQAA"
    # pylint: enable=line-too-long

    rules = {
        JavaScriptDefn(html_escaping.without_repeated_whitespace("""
          this.getRootNode().host.defns.handle_copy_click = async (button) => {
            const dataToCopy = button.dataset.copy;
            try {
              await navigator.clipboard.writeText(dataToCopy);
              button.classList.add("was_clicked");
              setTimeout(() => {
                button.classList.remove("was_clicked");
              }, 2000);
            } catch (e) {
              button.classList.add("broken_copy");
              button.textContent = (
                  "  # Failed to copy! Copy manually instead: " + dataToCopy
              );
            }
          };
        """)),
        # Font-face definitions can't live inside a shadow
        # DOM node, so we need to inject them into the root document.
        JavaScriptDefn(html_escaping.without_repeated_whitespace("""
        if (
            !Array.from(document.fonts.values()).some(
              font => font.family == 'Material Symbols Outlined Content Copy"'
            )
        ) {
          const sheet = new CSSStyleSheet();
          sheet.replaceSync(`@font-face {
              font-family: 'Material Symbols Outlined Content Copy';
              font-style: normal;
              font-weight: 400;
              src: url({__FONT_DATA_URL__}) format('woff2');
          }`);
          document.adoptedStyleSheets = [...document.adoptedStyleSheets, sheet];
        }
        """.replace("{__FONT_DATA_URL__}", font_data_url))),
        CSSStyleRule(html_escaping.without_repeated_whitespace(f"""
          {setup_context.collapsed_selector} .copybutton {{
              display: none;
          }}
          .copybutton::before {{
              content: " ";
          }}
          .copybutton > span::before {{
              content: "\\e14d";
              font-family: 'Material Symbols Outlined Content Copy';
              -webkit-font-smoothing: antialiased;
              color: #e0e0e0;
              cursor: pointer;
              font-size: 0.9em;
          }}
          .copybutton:hover > span::before {{
              color: darkseagreen;
          }}
          .copybutton.was_clicked > span::after {{
              color: #cccccc;
          }}
          .copybutton:hover > span::after {{
              color: darkseagreen;
              content: " " var(--data-annotation) var(--data-copy);
              transition: color 1s ease-in-out;
          }}
          .copybutton.was_clicked > span::after {{
              content: " Copied! " var(--data-copy) !important;
          }}
          .copybutton.broken_copy:hover {{
              color: darkseagreen;
          }}
          .copybutton.broken_copy {{
              color: #e0e0e0;
          }}
        """)),
    }
    return rules

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    if self.copy_string is not None:
      copy_string_attr = html_escaping.escape_html_attribute(self.copy_string)
      copy_string_property = html_escaping.escape_html_attribute(
          repr(self.copy_string)
      )
      annotation_property = html_escaping.escape_html_attribute(
          repr(self.annotation)
      )
      attributes = html_escaping.without_repeated_whitespace(
          "class='copybutton'"
          " onClick='this.getRootNode().host.defns.handle_copy_click(this)'"
          f' data-copy="{copy_string_attr}" style="--data-copy:'
          f" {copy_string_property}; --data-annotation:"
          f' {annotation_property}" '
      )
      stream.write(f"<span {attributes}><span></span></span>")


################################################################################
# Deferrables
################################################################################


@dataclasses.dataclass(frozen=False)
class DeferredPlaceholder(basic_parts.DeferringToChild):
  """A deferred part. Renders as a placeholder which will later be replaced."""

  child: RenderableTreePart
  replacement_id: str
  saved_at_beginning_of_line: bool | None = None
  saved_render_context: dict[Any, Any] | None = None
  needs_layout_decision: bool = False

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    self.saved_at_beginning_of_line = at_beginning_of_line
    self.saved_render_context = render_context
    stream.write(f'<span id="{self.replacement_id}">')
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")


@dataclasses.dataclass(frozen=True)
class DeferredWithThunk:
  """Stores a deferred placeholder along with its thunk."""

  placeholder: DeferredPlaceholder
  thunk: Callable[[ExpandState | None], RenderableTreePart]


@dataclasses.dataclass(frozen=True)
class EmptyWithHeightGuess(basic_parts.BaseContentlessLeaf):
  """Helper class that reports a guess of its height."""

  fake_newlines: int

  def _compute_newlines_in_expanded_parent(self) -> int:
    return self.fake_newlines


def fake_placeholder_foldable(
    placeholder_content: RenderableTreePart, extra_newlines_guess: int
) -> FoldableTreeNode:
  """Builds a fake placeholder for deferred renderings.

  The main use for this is is to return as the placeholder for a deferred
  rendering, so that it can participate in layout decisions. The
  `get_expand_state` method can be used to infer whether the part was
  collapsed while deferred.

  Args:
    placeholder_content: The content of the placeholder to render.
    extra_newlines_guess: The number of fake newlines to pretend this object
      has.

  Returns:
    A foldable node that does not have any actual content, but pretends to
    contain the given number of newlines for layout decisions.
  """
  return FoldableTreeNodeImpl(
      contents=basic_parts.siblings(
          placeholder_content,
          EmptyWithHeightGuess(fake_newlines=extra_newlines_guess),
      ),
      expand_state=ExpandState.WEAKLY_EXPANDED,
  )


################################################################################
# Abbreviation
################################################################################


@dataclasses.dataclass
class HasAbbreviation(basic_parts.DeferringToChild):
  """A part with an abbreviation."""

  child: RenderableTreePart
  abbreviation: RenderableTreePart

  def html_setup_parts(
      self, setup_context: HtmlContextForSetup
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    # Show or collapse based on the abbreviation selector.
    rule = html_escaping.without_repeated_whitespace(f"""
        .when_abbr {{
            display: none;
        }}
        {setup_context.abbreviate_selector} .when_abbr {{
            display: inline;
        }}
        {setup_context.abbreviate_selector} .not_abbr {{
            display: none;
        }}
        """)
    return (
        {CSSStyleRule(rule)}
        | self.child.html_setup_parts(setup_context)
        | self.abbreviation.html_setup_parts(setup_context)
    )

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return (
        self.child.layout_marks_in_this_part
        | self.abbreviation.layout_marks_in_this_part
    )

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    stream.write("<span class='not_abbr'>")
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )
    stream.write("</span>")
    stream.write("<span class='when_abbr'>")
    self.abbreviation.render_to_html(
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
    level = render_context.get(ABBREVIATION_LEVEL_KEY, 0)
    threshold = render_context.get(ABBREVIATION_THRESHOLD_KEY, None)
    if threshold is not None and level >= threshold:
      self.abbreviation.render_to_text(
          stream,
          expanded_parent=expanded_parent,
          indent=indent,
          roundtrip_mode=roundtrip_mode,
          render_context=render_context,
      )
    else:
      self.child.render_to_text(
          stream,
          expanded_parent=expanded_parent,
          indent=indent,
          roundtrip_mode=roundtrip_mode,
          render_context=render_context,
      )


@dataclasses.dataclass
class AbbreviationLevel(basic_parts.DeferringToChild):
  """Marks an abbreviation level, indicating that children may be abbreviated.

  Abbreviation levels are only active when their parent is collapsed. Once we
  pass a collapsed parent and the threshold number of further abbreviation
  levels, children with abbreviations will be abbreviated.
  """

  child: RenderableTreePart

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    stream.write(f"<span class='{ABBREVIATION_LEVEL_CLASS}'>")
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
    if expanded_parent:
      new_render_context = render_context
    else:
      # We need to increment the abbreviation level.
      old_level = render_context.get(ABBREVIATION_LEVEL_KEY, 0)
      new_render_context = {
          **render_context,
          ABBREVIATION_LEVEL_KEY: old_level + 1,
      }

    self.child.render_to_text(
        stream,
        expanded_parent=expanded_parent,
        indent=indent,
        roundtrip_mode=roundtrip_mode,
        render_context=new_render_context,
    )


def abbreviatable(
    child: RenderableTreePart, abbreviation: RenderableTreePart | None = None
) -> RenderableTreePart:
  """Marks an object as being able to be abbreviated.

  Args:
    object: The object to mark as abbreviatable. This will be replaced by the
        fallback if the object is past the current abbreviation depth.
    abbreviation: The fallback to use if the object is abbreviated.
  """
  if abbreviation is None:
    abbreviation = basic_parts.siblings(
        common_styles.comment_color(basic_parts.text("<")),
        common_styles.abbreviation_color(basic_parts.text("...")),
        common_styles.comment_color(basic_parts.text(">")),
    )
  return HasAbbreviation(child, abbreviation)


def abbreviatable_with_annotations(
    child: part_interface.RenderableAndLineAnnotations,
    abbreviation: RenderableTreePart | None = None,
) -> part_interface.RenderableAndLineAnnotations:
  """Marks an object with annotations as being able to be abbreviated.

  Args:
    object: The object (with annotations) to mark as abbreviatable. This will
        be replaced by the fallback if the object is past the current
        abbreviation depth.
    abbreviation: The fallback to use if the object is abbreviated.
  """
  return part_interface.RenderableAndLineAnnotations(
      HasAbbreviation(child.renderable, abbreviation), child.annotations
  )


def abbreviation_level(child: RenderableTreePart) -> RenderableTreePart:
  """Marks an abbreviation level, indicating children may be abbreviated."""
  return AbbreviationLevel(child)


def make_abbreviate_selector(
    threshold: int | None, roundtrip_threshold: int | None
) -> str:
  """Builds a CSS selector that matches nodes that should be abbreviated.

  Args:
    threshold: The threshold for normal mode.
    roundtrip_threshold: The threshold for roundtrip mode.

  Returns:
    A CSS selector that matches an ancestor of any node that should be
    abbreviated.
  """
  options = []
  if threshold is not None:
    levels = f" .{ABBREVIATION_LEVEL_CLASS}" * threshold
    options.append(f"{NOT_ROUNDTRIP_SELECTOR} {COLLAPSED_SELECTOR}{levels}")
  if roundtrip_threshold is not None:
    levels = f" .{ABBREVIATION_LEVEL_CLASS}" * roundtrip_threshold
    options.append(f"{ROUNDTRIP_SELECTOR} {COLLAPSED_SELECTOR}{levels}")
  return ":is(" + ", ".join(options) + ")"
