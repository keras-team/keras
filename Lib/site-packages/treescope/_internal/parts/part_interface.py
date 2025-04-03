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

"""Base interface for the intermediate representation for renderings."""
from __future__ import annotations

import abc
import dataclasses
import enum
import functools
import io
from typing import Any, Sequence, TypeAlias

from treescope._internal import docs_util

################################################################################
# Common structure types
################################################################################


@dataclasses.dataclass(frozen=True, order=True)
class CSSStyleRule:
  """A CSS rule (or set of rules).

  The contents of `rule` will be inserted into a <style> block in the rendered
  tree.

  Attributes:
    rule: The rule to insert.
  """

  rule: str


@dataclasses.dataclass(frozen=True, order=True)
class JavaScriptDefn:
  """A JavaScript function definition (or set of definitions).

  The contents of `source` will be inserted into a <script> block in the
  rendered tree. Functions that need to be referenced by event handlers can
  be assigned as a property to `this.getRootNode().defns`, which will resolve
  to a scoped "definitions" property on the treescope rendering container.

  Attributes:
    source: The JS source code to insert.
  """

  source: str


@dataclasses.dataclass(frozen=True)
class HtmlContextForSetup:
  """Configuration variables necessary for setting up rendering.

  Attributes:
    collapsed_selector: A CSS selector that will match any collapsed ancestor of
      this node. A node can apply styles when in a collapsed parent by using
      this as a prefix in the CSS selector.
    roundtrip_selector: A CSS selector that will match an ancestor of this node
      whenever rendering in roundtrip mode.
    abbreviate_selector: A CSS selector that will match an ancestor of this
      node whenever we are requesting that abbreviation should be applied to
      this node (or its children).
  """

  collapsed_selector: str
  roundtrip_selector: str
  abbreviate_selector: str


################################################################################
# Abstract interface
################################################################################


class RenderableTreePart(abc.ABC):
  """Abstract base class for a formatted part of a foldable tree.

  WARNING: The details of this interface are an implementation detail, and are
  subject to change.

  Formatted objects are produced by treescope handlers from the original Python
  objects, and know how to convert themselves to a concrete renderable
  representation.

  Renderable tree parts can appear either expanded or collapsed, depending
  on the status of their parent foldable tree nodes. They are responsible for:

  - computing the relevant sizing metrics for each of these display modes,
  - tracking any children that are foldable tree nodes themselves,
  - and rendering themselves to HTML or text form.
  """

  @docs_util.skip_automatic_documentation
  @functools.cached_property
  def collapsed_width(self) -> int:
    """The length of this rendering if collapsed in one line, in characters.

    This should include the (collapsed) lengths of all children of this tree
    part as well. Cached to avoid O(n^2) computations while iteratively
    expanding children.
    """
    return self._compute_collapsed_width()

  @docs_util.skip_automatic_documentation
  @abc.abstractmethod
  def _compute_collapsed_width(self) -> int:
    """Computes a value for `collapsed_width`."""
    raise NotImplementedError("_compute_collapsed_width must be overridden")

  @docs_util.skip_automatic_documentation
  @functools.cached_property
  def newlines_in_expanded_parent(self) -> int:
    """The number of newlines in this rendering if in an expanded parent.

    For instance, if this will always render as a single line, this should
    return zero. If this renders as three lines, it contains two newlines, so
    this should return 2.

    Being in an expanded parent means that all parent FoldableTreeNodes
    containing this part are in an expanded state, and this tree part is not
    constrained to lie on a single line.

    However, any foldable tree nodes *contained within* this node itself should
    be considered collapsed for the purposes of computing this value; this
    should represent the *smallest* height this part could have if all of its
    children were collapsed. As a special case, if this node is itself a
    subclass of FoldableTreeNode, this should usually return 0 regardless of
    whether the node is currently set to expand or not.
    """
    return self._compute_newlines_in_expanded_parent()

  @docs_util.skip_automatic_documentation
  @abc.abstractmethod
  def _compute_newlines_in_expanded_parent(self) -> int:
    """Computes a value for `newlines_in_expanded_parent`."""
    raise NotImplementedError(
        "_compute_newlines_in_expanded_parent must be overridden"
    )

  @docs_util.skip_automatic_documentation
  @functools.cached_property
  def layout_marks_in_this_part(self) -> frozenset[Any]:
    """Returns a set of "layout mark" objects contained in this part.

    Layout marks allow surfacing of internal information to guide layout
    decisions. For instance, if nodes of a particular type need to be made
    visible, they can add a layout mark and propagate it upward through the
    tree. Then, foldables can be expanded whenever they contain a mark of the
    given type. (This is how expansion of selections is implemented.)

    Cached to avoid O(n^2) computations while iteratively expanding children.
    """
    return self._compute_layout_marks_in_this_part()

  @docs_util.skip_automatic_documentation
  @abc.abstractmethod
  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    """Computes a value for `layout_marks_in_this_part`."""
    raise NotImplementedError(
        "_compute_layout_marks_in_this_part must be overridden"
    )

  @docs_util.skip_automatic_documentation
  @abc.abstractmethod
  def foldables_in_this_part(self) -> Sequence[FoldableTreeNode]:
    """Returns a collection of foldables contained in this node.

    This should return the outermost foldables that are contained in this node,
    e.g. any foldable that is contained in this node but not contained in
    another foldable contained in this node. If this node is already a
    FoldableTreeNode, the result should just be `self`.

    Returns:
      A sequence of foldables contained in this node.
    """
    raise NotImplementedError("foldables_in_this_part must be overridden")

  @docs_util.skip_automatic_documentation
  @abc.abstractmethod
  def html_setup_parts(
      self,
      context: HtmlContextForSetup,
      /,
  ) -> set[CSSStyleRule | JavaScriptDefn]:
    """Returns a set of setup strings for this part.

    Setup strings are strings that should appear in the HTML at least once but
    not more than once if possible, such as definitions of javascript functions
    or CSS styles.

    Args:
      context: Context for the setup.

    Returns:
      A set of setup strings, which can use the given selectors to configure
      themselves.
    """
    raise NotImplementedError("html_setup_parts must be overridden")

  @docs_util.skip_automatic_documentation
  @abc.abstractmethod
  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    """Renders this tree part to an HTML source string.

    Nodes will be automatically indented as needed using parent margins and
    padding, so they do not need to handle this.

    Handling of collapsed parents and roundtrip node should be done in
    `html_setup_parts` and then configured using CSS classes.

    Args:
      stream: Output stream to write to.
      at_beginning_of_line: Whether this node is at the beginning of its line,
        meaning that it only has whitespace to its left and is allowed to extend
        slightly past its ordinary bounds on the left side. For instance, if
        this is True, it's OK to render an "unfold" marker inside the whitespace
        to the left of this node.
      render_context: Dictionary of optional additional context. Specific types
        of renderable part that need to set context should ensure they use a
        unique key, and any unrecognized keys should be passed down to any
        children being rendered.
    """
    raise NotImplementedError("render_to_html must be overridden")

  @docs_util.skip_automatic_documentation
  @abc.abstractmethod
  def render_to_text(
      self,
      stream: io.TextIOBase,
      *,
      expanded_parent: bool,
      indent: int,
      roundtrip_mode: bool,
      render_context: dict[Any, Any],
  ):
    """Renders this tree part to a plain-text string.

    Args:
      stream: Output stream to write to.
      expanded_parent: Whether this node's parent is expanded, and it's thus OK
        to render across multiple lines.
      indent: Number of spaces to indent this part if it breaks across multiple
        lines.
      roundtrip_mode: Whether to render in round-trip mode (with fully qualified
        names).
      render_context: Dictionary of optional additional context. Specific types
        of renderable part that need to set context should ensure they use a
        unique key, and any unrecognized keys should be passed down to any
        children being rendered.

    Returns:
      A plain-text rendering.
    """
    raise NotImplementedError("render_to_text must be overridden")


class ExpandState(enum.Enum):
  """Enum for expand states.

  Members:
    EXPANDED: Indicates that a foldable node should start expanded. This should
      not be modified by automatic formatting logic.
    COLLAPSED: Indicates that a foldable node should start collapsed. This
      should not be modified by automatic formatting logic.
    WEAKLY_EXPANDED: Indicates that a foldable node could start expanded as a
      best guess, but that automatic formatting logic is free to collapse it
      to improve the overall layout.
    WEAKLY_COLLAPSED: Indicates that a foldable node could start collapsed as a
      best guess, but that automatic formatting logic is free to expand it
      to improve the overall layout.
  """

  EXPANDED = "expanded"
  COLLAPSED = "collapsed"
  WEAKLY_EXPANDED = "weakly_expanded"
  WEAKLY_COLLAPSED = "weakly_collapsed"

  def is_weak(self) -> bool:
    """Returns True if this state is weak (e.g. can be changed)."""
    return self in (ExpandState.WEAKLY_EXPANDED, ExpandState.WEAKLY_COLLAPSED)


class FoldableTreeNode(RenderableTreePart):
  """Abstract base class for a node that can be expanded or collapsed.

  Foldable tree nodes can either be expanded or collapsed interactively for
  display. They are responsible for maintaining their own expand/collapse state,
  and for reporting metrics about themselves.

  FoldableTreeNodes are always RenderableTreeParts, but they report normal
  metrics for themselves in their collapsed state, and support additional
  methods for computing their expanded state. This is so that their parents
  can freely call newlines_in_expanded_parent and foldables_in_this_part without
  checking the types of their children.

  Note: Currently there is only one concrete subclass of FoldableTreeNode,
  called FoldableTreeNodeImpl. This is kept separate so that the abstract
  definition of RenderableTreePart doesn't have to depend on implementation
  details of FoldableTreeNodeImpl.
  """

  @abc.abstractmethod
  def get_expand_state(self) -> ExpandState:
    """Returns True if this node is currently set to be expand_state."""
    raise NotImplementedError("expand_state must be overridden")

  @abc.abstractmethod
  def set_expand_state(self, expand_state: ExpandState):
    """Sets whether this node should start expand_state."""
    raise NotImplementedError("set_expand_state must be overridden")

  @abc.abstractmethod
  def as_expanded_part(self) -> RenderableTreePart:
    """Returns a part that is equivalent to this node but always expanded.

    This should be used to extract the height of this foldable when expanded
    and also to identify its child foldables.
    """
    raise NotImplementedError("as_expanded_part must be overridden")

  # Base case definitions for RenderableTreePart shared by FoldableTreeNode
  # implementations:

  def _compute_newlines_in_expanded_parent(self) -> int:
    """Computes a value for `newlines_in_expanded_parent`."""
    # When this node itself is collapsed, it has no newlines, even if its
    # parent is expanded.
    return 0

  def _compute_collapsed_width(self) -> int:
    """Computes a value for `collapsed_width`."""
    return self.as_expanded_part().collapsed_width

  def foldables_in_this_part(self) -> Sequence[FoldableTreeNode]:
    """Returns this node itself, since it is a foldable node."""
    return (self,)

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    """Computes a value for `layout_marks_in_this_part`."""
    return self.as_expanded_part().layout_marks_in_this_part


@dataclasses.dataclass(frozen=True)
class RenderableAndLineAnnotations:
  """Groups a main renderable with some line comments.

  This can be produced by handlers that want to "float" some comments to the
  right, past any delimiters that may follow the object of interest.

  This is not itself a renderable tree part; it should be expanded by
  `basic_parts.build_full_line_with_annotations` or a similar function before
  being displayed.

  Attributes:
    renderable: The main object to render.
    annotations: Annotations for the main object, which should be a single line
      and will be floated to the right past any delimiters.
  """

  renderable: RenderableTreePart
  annotations: RenderableTreePart


NodePath: TypeAlias = str
Rendering: TypeAlias = RenderableTreePart | RenderableAndLineAnnotations
