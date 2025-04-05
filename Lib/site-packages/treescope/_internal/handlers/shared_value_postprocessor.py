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

"""Postprocessor to annotate repetitions of mutable values.

This can be useful for figuring out when multiple mutable objects in a tree are
all references to the same object.
"""

import contextlib
import dataclasses
import io
from typing import Any, Optional, Sequence

from treescope import context
from treescope import renderers
from treescope import rendering_parts
from treescope import type_registries
from treescope._internal import html_escaping
from treescope._internal.parts import basic_parts
from treescope._internal.parts import part_interface


@dataclasses.dataclass
class _SharedObjectTracker:
  """Helper object to track IDs we've seen before.

  This object keeps track of nodes we've encountered while rendering the root
  object, so that we can detect when the same object is encountered in multiple
  places. To ensure that we don't have false positives, we also store the nodes
  themselves until the rendering finishes. (Usually, rendered Python objects
  will be kept alive due to being part of the rendered tree, but some handlers
  may render temporary objects that are discarded, at which point their object
  IDs are sometimes re-used by other temporary objects.)

  Attributes:
    seen_at_least_once: Map with node IDs as keys and nodes as values,
      containing nodes we've seen at least once.
    seen_more_than_once: Set of node IDs we've seen more than once.
  """

  seen_at_least_once: dict[int, Any]
  seen_more_than_once: set[int]


_shared_object_ids_seen: context.ContextualValue[
    Optional[_SharedObjectTracker]
] = context.ContextualValue(
    module=__name__, qualname="_shared_object_ids_seen", initial_value=None
)
"""A collection of IDs of objects that we have already rendered.

This is used to detect "unsafe" sharing of objects.

Normally references to the same object are ignored by JAX, and doing any
PyTree manipulation destroys the sharing. However, leaves of JAX PyTrees
may still involve mutable-object sharing, and we want to be able to
use Treescope to visualize this kind of structure as well. We also
use this to avoid infinite recursion if a node in the tree contains
itself.

This context manager should only be used by code in this module.
"""


class SharedWarningLabel(basic_parts.BaseSpanGroup):
  """A comment identifying this shared Python object."""

  def _span_css_class(self) -> str:
    return "shared_warning_label"

  def _span_css_rule(
      self, setup_context: part_interface.HtmlContextForSetup
  ) -> part_interface.CSSStyleRule:
    del setup_context
    return part_interface.CSSStyleRule(
        html_escaping.without_repeated_whitespace("""
            .shared_warning_label
            {
                color: #ffac13;
            }
        """)
    )


@dataclasses.dataclass(frozen=False)
class DynamicSharedCheck(rendering_parts.RenderableTreePart):
  """Dynamic group that renders its child only if a node is shared.

  This node is used to apply special rendering to nodes that are encountered in
  multiple places in the same input. It works by holding a particular node ID
  as well as a reference to a set of node IDs that have been seen more than
  once. When rendered, it checks if its node ID has been seen more than once
  and renders differently if so.

  Note that we might have only seen this object once at the time the shared
  warning object is constructed, even if eventually we will see it again. This
  means the `seen_more_than_once` attribute has to be (a reference to) a
  *mutable set* modified elsewhere.

  Shared markers always act like empty parts for layout decisions.

  Attributes:
    if_shared: Child to render only if the node is shared.
    node_id: Node ID of the node we are rendering. Used for the warning and also
      looked up in `seen_more_than_once`.
    seen_more_than_once: Reference to an externally-maintained set of node IDs
      we've seen more than once. Usually, this will be the same set from the
      active `_shared_object_ids_seen` context.
  """

  if_shared: rendering_parts.RenderableTreePart
  node_id: int
  seen_more_than_once: set[int]

  def _compute_collapsed_width(self) -> int:
    return 0

  def _compute_newlines_in_expanded_parent(self) -> int:
    return 0

  def foldables_in_this_part(self) -> Sequence[part_interface.FoldableTreeNode]:
    return []

  def _compute_layout_marks_in_this_part(self) -> frozenset[Any]:
    return frozenset()

  def html_setup_parts(
      self, setup_context: part_interface.HtmlContextForSetup
  ) -> set[part_interface.CSSStyleRule | part_interface.JavaScriptDefn]:
    return self.if_shared.html_setup_parts(setup_context)

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    if self.node_id in self.seen_more_than_once:
      self.if_shared.render_to_html(
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
    if self.node_id in self.seen_more_than_once:
      self.if_shared.render_to_text(
          stream,
          expanded_parent=expanded_parent,
          indent=indent,
          roundtrip_mode=roundtrip_mode,
          render_context=render_context,
      )


@dataclasses.dataclass(frozen=False)
class WithDynamicSharedPip(basic_parts.DeferringToChild):
  """Dynamic group that adds an orange marker to its child if it is shared.

  This node is used to apply special rendering to nodes that are encountered in
  multiple places in the same input. It works by holding a particular node ID
  as well as a reference to a set of node IDs that have been seen more than
  once. When rendered, it checks if its node ID has been seen more than once
  and renders differently if so.

  Note that we might have only seen this object once at the time the shared
  warning object is constructed, even if eventually we will see it again. This
  means the `seen_more_than_once` attribute has to be (a reference to) a
  *mutable set* modified elsewhere.

  Shared markers always act like their child. They also do not disrupt children
  drawing themselves first.

  Attributes:
    child: Child to render.
    node_id: Node ID of the node we are rendering. Used for the warning and also
      looked up in `seen_more_than_once`.
    seen_more_than_once: Reference to an externally-maintained set of node IDs
      we've seen more than once. Usually, this will be the same set from the
      active `_shared_object_ids_seen` context.
  """

  child: rendering_parts.RenderableTreePart
  node_id: int
  seen_more_than_once: set[int]

  def html_setup_parts(
      self, setup_context: part_interface.HtmlContextForSetup
  ) -> set[part_interface.CSSStyleRule | part_interface.JavaScriptDefn]:
    return self.child.html_setup_parts(setup_context) | {
        part_interface.CSSStyleRule(
            html_escaping.without_repeated_whitespace(f"""
                .shared_warning_pip
                {{
                    padding-right: 1ch;
                    margin-right: -0.5ch;
                    background: linear-gradient(
                        135deg, orange 0 0.6ch, transparent 0.6ch
                    );
                }}
                .shared_warning_pip.is_first_on_line:not(
                    {setup_context.collapsed_selector} *
                )
                {{
                    margin-left: -0.5ch;
                }}
            """)
        )
    }

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    if self.node_id in self.seen_more_than_once:
      if at_beginning_of_line:
        stream.write(
            "<span class='shared_warning_pip is_first_on_line'></span>"
        )
      else:
        stream.write("<span class='shared_warning_pip'></span>")
    self.child.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )


def setup_shared_value_context() -> contextlib.AbstractContextManager[None]:
  """Returns a context manager for the shared value context.

  Within this context manager, `_shared_object_ids_seen` will refer
  to a consistent tracker. This tracker can then be used to check for repeated
  appearances of the same mutable object.

  This should be included in the `context_builders` argument to any renderer
  that checks for shared values.
  """
  return _shared_object_ids_seen.set_scoped(_SharedObjectTracker({}, set()))


def _is_safe_to_share(node: Any) -> bool:
  """Returns whether the given node is immutable."""
  # According to the Python data model, "If a class defines mutable objects and
  # implements an __eq__() method, it should not implement __hash__()". So, if
  # we find an object that implements __eq__ and __hash__, we can generally
  # assume it is immutable.
  return (
      type(node).__hash__ is not None
      and type(node).__hash__ is not object.__hash__
      and type(node).__eq__ is not object.__eq__
  ) or type_registries.lookup_immutability_for_type(type(node))


def check_for_shared_values(
    node: Any,
    path: str | None,
    node_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  # pylint: disable=g-doc-args,g-doc-return-or-yield
  """Wrapper hook to check for and annotate shared values.

  This wrapper should only be used by renderers that also include
  `build_styles_for_shared_values` in their HTML configuration and
  `setup_shared_value_context` in their context builders.

  Args:
    node: The node that has been rendered
    path: Optionally, a path to this node as a string.
    node_renderer: The inner renderer for this node. This should be used to
      render `node` itself into HTML tags.

  Returns:
    A possibly-modified representation of this object.

  Raises:
    RuntimeError: If called outside of the context constructed via
      setup_shared_value_context.
  """
  # pylint: enable=g-doc-args,g-doc-return-or-yield
  shared_object_tracker = _shared_object_ids_seen.get()
  if shared_object_tracker is None:
    raise RuntimeError(
        "`check_for_shared_values` should only be called in a shared value"
        " context! Make sure the current treescope renderer has"
        " `setup_shared_value_context` in its `context_builders`."
    )

  # Use object identity to track shared references and loops.
  node_id = id(node)

  # For types that we know are immutable, it's not necessary to render shared
  # references in a special way.
  safe_to_share = _is_safe_to_share(node)

  # Render the node normally.
  rendering = node_renderer(node, path)

  if not safe_to_share:
    # Mark this as possibly shared.
    if node_id in shared_object_tracker.seen_at_least_once:
      shared_object_tracker.seen_more_than_once.add(node_id)
    else:
      shared_object_tracker.seen_at_least_once[node_id] = node

    # Wrap it in a shared value wrapper; this will check to see if the same
    # node was seen more than once, and add an annotation if so.
    return rendering_parts.RenderableAndLineAnnotations(
        renderable=WithDynamicSharedPip(
            rendering.renderable,
            node_id=node_id,
            seen_more_than_once=shared_object_tracker.seen_more_than_once,
        ),
        annotations=rendering_parts.siblings(
            DynamicSharedCheck(
                if_shared=SharedWarningLabel(
                    rendering_parts.text(
                        f" # Repeated python obj at 0x{node_id:x}"
                    )
                ),
                node_id=node_id,
                seen_more_than_once=shared_object_tracker.seen_more_than_once,
            ),
            rendering.annotations,
        ),
    )

  return rendering
