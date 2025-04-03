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

"""Renderer objects and handler definitions."""
from __future__ import annotations

import contextlib
import dataclasses
import functools
import traceback
import typing
from typing import Any, Callable, Iterable
import warnings

from treescope import layout_algorithms
from treescope import lowering
from treescope import rendering_parts


class TreescopeSubtreeRenderer(typing.Protocol):
  """Protocol for a (recursive) subtree renderer.

  Implementations of this protocol render full subtrees to HTML tags. Handlers
  take one of these as an argument and use it to render their children.
  """

  def __call__(
      self,
      node: Any,
      path: str | None = None,
  ) -> rendering_parts.RenderableAndLineAnnotations:
    """Signature for a (recursive) subtree renderer.

    Args:
      node: The node to render.
      path: Optionally, a path to this node, represented as a string that can be
        used to reach this node from the root (e.g. ".foo.bar['baz']")

    Returns:
      A representation of the object as a renderable tree part in the
      intermediate representation.
    """


class TreescopeNodeHandler(typing.Protocol):
  """Protocol for a rendering handler for a particular node type.

  Rendering handlers run in order (from most specific to least specific),
  until one of them is able to handle the given node. Handlers should return
  NotImplemented for types of node that they are unable to handle.
  """

  def __call__(
      self,
      node: Any,
      path: str | None,
      subtree_renderer: TreescopeSubtreeRenderer,
  ) -> (
      rendering_parts.RenderableTreePart
      | rendering_parts.RenderableAndLineAnnotations
      | type(NotImplemented)
  ):
    """Signature for a rendering handler for a particular node type.

    Args:
      node: The node to render.
      path: Optionally, a path to this node, represented as a string that can be
        used to reach this node from the root (e.g. ".foo.bar['baz']"). Handlers
        should extend this path and pass it to `subtree_renderer` when rendering
        each of its children.
      subtree_renderer: A recursive renderer for subtrees of this node. This
        should be used to render the children of this node into HTML tags to
        include in the rendering of this node. Should only be called on nodes
        that are children of `node`, to avoid an infinite loop.

    Returns:
      A representation of the object as a tree part, or NotImplemented
      if this handler doesn't support rendering this object (indicating that
      we should try the next handler). Tree parts may have optional end-of-line
      annotations.
    """


class TreescopeCustomWrapperHook(typing.Protocol):
  """Protocol for a custom wrapper hook.

  Custom wrapper hooks are more powerful versions of node handlers. They run
  on every node, and are given control over how that node is rendered. In
  particular, they can modify the arguments that are passed to subsequent
  wrappers and ordinary handlers by calling ``node_renderer`` with modified
  values, and can also insert additional HTML tags around the rendered result.

  Generally, custom wrapper hooks should call ``node_renderer`` exactly once
  with the same ``node`` argument and the same ``path``.
  """

  def __call__(
      self,
      node: Any,
      path: str | None,
      node_renderer: TreescopeSubtreeRenderer,
  ) -> (
      rendering_parts.RenderableTreePart
      | rendering_parts.RenderableAndLineAnnotations
      | type(NotImplemented)
  ):
    """Signature for a custom wrapper hook.

    Args:
      node: The node that we are rendering.
      path: Optionally, a path to this node as a string; see
        `TreescopeNodeHandler`.
      node_renderer: The inner renderer for this node. This can be used to
        render `node` itself according to the ordinary logic. Usually the
        wrapper hook should call `node_renderer` once with `node`, possibly
        customizing its arguments, and then postprocess the return value.

    Returns:
      A modified rendering of the object, or NotImplemented if this
      hook does not want to modify this rendering at all. (Returning
      NotImplemented)
    """


@dataclasses.dataclass
class TreescopeRenderer:
  """A renderer object with a configurable set of handlers.

  Attributes:
    handlers: List of handlers, ordered from most specific to least specific.
      Handlers will be tried in order until one returns something other than
      NotImplemented.
    wrapper_hooks: List of wrapper hooks, ordered from outermost to innermost.
      All wrapper hooks will run in order on every node.
    context_builders: List of functions that build rendering context managers.
      Each of these context managers will be entered before rendering the tree,
      and will be exited after converting the returned tags to an HTML string.
      The main use for this is to set up shared state needed by the handlers or
      wrapper hooks (e.g. using a `treescope.context.ContextualValue`). Context
      builders can also return None instead of a context manager, if they just
      need to do processing at the start of rendering and don't need to clean
      up.
  """

  handlers: list[TreescopeNodeHandler]
  wrapper_hooks: list[TreescopeCustomWrapperHook]
  context_builders: list[
      Callable[[], contextlib.AbstractContextManager[Any] | None]
  ]

  def extended_with(
      self,
      *,
      handlers: Iterable[TreescopeNodeHandler] = (),
      wrapper_hooks: Iterable[TreescopeCustomWrapperHook] = (),
      context_builders: Iterable[
          Callable[[], contextlib.AbstractContextManager[Any] | None]
      ] = (),
  ) -> TreescopeRenderer:
    """Extends a renderer with additional steps, returning a new renderer.

    Args:
      handlers: New handlers to insert. These will be inserted at the beginning
        of the handler list, to allow them to override existing handlers.
      wrapper_hooks: New wrapper hooks. These will be inserted before the
        existing wrapper hooks, to allow them to override existing hooks.
      context_builders: New context builders. These will be inserted after the
        existing context builders.

    Returns:
      A copy of this renderer extended with the new parts.
    """
    return TreescopeRenderer(
        handlers=list(handlers) + self.handlers,
        wrapper_hooks=list(wrapper_hooks) + self.wrapper_hooks,
        context_builders=self.context_builders + list(context_builders),
    )

  def _render_subtree(
      self,
      ignore_exceptions: bool,
      rendering_stack: list[Any],
      already_executed_wrapper_count: int,
      node: Any,
      path: str | None = None,
  ) -> rendering_parts.RenderableAndLineAnnotations:
    """Renders a specific subtree using the renderer.

    Args:
      ignore_exceptions: Whether to catch and ignore errors in handlers or
        postprocessors.
      rendering_stack: A list of the ancestors of this node, used to check for
        and break cyclic references.
      already_executed_wrapper_count: The number of wrappers we have already
        called for this node.
      node: See `TreescopeSubtreeRenderer.__call__`.
      path: See `TreescopeSubtreeRenderer.__call__`.

    Returns:
      A renderable representation of the object.
    """
    while already_executed_wrapper_count < len(self.wrapper_hooks):
      # We have hooks we haven't processed. Defer to the hook.
      # We pass a reference to this method back to the wrapper to allow it to
      # resume execution at the next wrapper.
      hook = self.wrapper_hooks[already_executed_wrapper_count]
      render_without_this_hook = functools.partial(
          self._render_subtree,
          ignore_exceptions,
          rendering_stack,
          already_executed_wrapper_count + 1,
      )
      try:
        postprocessed_result = hook(
            node=node, path=path, node_renderer=render_without_this_hook
        )
        if postprocessed_result is NotImplemented:
          # Skip this hook and defer to the next one (or to a handler)
          already_executed_wrapper_count += 1
          continue

        elif isinstance(
            postprocessed_result, rendering_parts.RenderableAndLineAnnotations
        ):
          return postprocessed_result
        elif isinstance(
            postprocessed_result, rendering_parts.RenderableTreePart
        ):
          return rendering_parts.RenderableAndLineAnnotations(
              renderable=postprocessed_result,
              annotations=rendering_parts.empty_part(),
          )
        else:
          raise ValueError(
              f"Wrapper hook {hook} returned a value of an unexpected type:"
              f" {postprocessed_result}"
          )
      except Exception:  # pylint: disable=broad-exception-caught
        if ignore_exceptions:
          warnings.warn(
              f"Ignoring error inside wrapper hook {hook}"
              f":\n{traceback.format_exc(limit=10)}"
          )
          already_executed_wrapper_count += 1
          continue
        else:
          raise

    # If we get here, all of the wrappers have already processed this object.

    # Use object identity to track cyclic objects.
    node_id = id(node)

    if node_id in rendering_stack:
      # Cycle! This object contains itself.
      return rendering_parts.RenderableAndLineAnnotations(
          renderable=rendering_parts.error_color(
              rendering_parts.text(
                  f"<cyclic reference to {type(node).__name__} at"
                  f" 0x{node_id:x}>"
              )
          ),
          annotations=rendering_parts.empty_part(),
      )
    else:
      # Track cyclic references. We use `try: ... finally: ...` to ensure we
      # clean up after ourselves even if we return a value early. (There's also
      # separate internal try/catch logic for error handling.)
      rendering_stack.append(node_id)
      try:
        # Give each of our handlers a chance to render this node.
        # We pass this method as the renderer for any children of the node, and
        # reenable all of the wrapper hooks.
        rec = functools.partial(
            self._render_subtree, ignore_exceptions, rendering_stack, 0
        )
        for handler in self.handlers:
          try:
            maybe_result = handler(node=node, path=path, subtree_renderer=rec)

            if maybe_result is NotImplemented:
              # Try the next handler.
              continue
            elif isinstance(
                maybe_result, rendering_parts.RenderableAndLineAnnotations
            ):
              # Found a result!
              return maybe_result
            elif isinstance(maybe_result, rendering_parts.RenderableTreePart):
              # Wrap it with empty annotations.
              return rendering_parts.RenderableAndLineAnnotations(
                  renderable=maybe_result,
                  annotations=rendering_parts.empty_part(),
              )
            else:
              raise ValueError(
                  f"Handler {handler} returned a value of an unexpected type:"
                  f" {maybe_result}"
              )

          except Exception:  # pylint: disable=broad-exception-caught
            if ignore_exceptions:
              warnings.warn(
                  "Ignoring error while formatting value of type"
                  f" {type(node)} with"
                  f" {handler}:\n{traceback.format_exc(limit=10)}"
              )
              continue
            else:
              raise

        # If we get here, it means no handler was found.
        if ignore_exceptions:
          warnings.warn(
              f"No handler registered for a node of type {type(node)}."
          )
          # Fall back to a basic `repr` so that we still render something even
          # without a handler for it. We use the object repr because a custom
          # repr may still raise an exception if the object is in an invalid
          # state.
          return rendering_parts.RenderableAndLineAnnotations(
              renderable=rendering_parts.error_color(
                  rendering_parts.text(object.__repr__(node))
              ),
              annotations=rendering_parts.comment_color(
                  rendering_parts.text(
                      "  # Error occurred while formatting this object."
                  )
              ),
          )
        else:
          raise ValueError(
              f"No handler registered for a node of type {type(node)}."
          )

      finally:
        # Remove it from the rendering stack.
        popped_val = rendering_stack.pop()
        assert popped_val == node_id

  def to_foldable_representation(
      self,
      value: Any,
      ignore_exceptions: bool = False,
      root_keypath: str | None = "",
  ) -> rendering_parts.RenderableAndLineAnnotations:
    """Renders an object to the foldable intermediate representation.

    Args:
      value: Value to render.
      ignore_exceptions: Whether to catch errors during rendering of subtrees
        and show a fallback for those subtrees, instead of failing the entire
        renderer. Best used in contexts where `to_foldable_representation` is
        not being called directly by the user, e.g. when registering this as a
        default pretty-printer.
      root_keypath: Keypath to the value being rendered. Can be None to disable
        all keypaths during rendering.

    Returns:
      Renderable representation of the object.
    """
    with contextlib.ExitStack() as stack:
      # Set up any scoped state contexts that handlers need.
      for context_builder in self.context_builders:
        new_context = context_builder()
        if new_context is not None:
          stack.enter_context(new_context)

      # Build the foldable HTML tag recursively.
      return self._render_subtree(
          ignore_exceptions=ignore_exceptions,
          rendering_stack=[],
          already_executed_wrapper_count=0,
          node=value,
          path=root_keypath,
      )

  def to_text(self, value: Any, roundtrip_mode: bool = False) -> str:
    """Convenience method to render an object to text.

    Args:
      value: Value to render.
      roundtrip_mode: Whether to render in roundtrip mode.

    Returns:
      A text representation of the object.
    """
    foldable_ir = rendering_parts.build_full_line_with_annotations(
        self.to_foldable_representation(value)
    )
    layout_algorithms.expand_for_balanced_layout(foldable_ir)
    return lowering.render_to_text_as_root(foldable_ir, roundtrip_mode)

  def to_html(self, value: Any, roundtrip_mode: bool = False) -> str:
    """Convenience method to render an object to HTML.

    Args:
      value: Value to render.
      roundtrip_mode: Whether to render in roundtrip mode.

    Returns:
      HTML source code for the foldable representation of the object.
    """
    foldable_ir = rendering_parts.build_full_line_with_annotations(
        self.to_foldable_representation(value)
    )
    layout_algorithms.expand_for_balanced_layout(foldable_ir)
    return lowering.render_to_html_as_root(foldable_ir, roundtrip_mode)
