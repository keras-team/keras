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

"""Lowering of Treescope's renderable parts to text or HTML.

This module defines the final logic that converts the output of node handlers
and figure parts to the final Treescope representation.
"""

from __future__ import annotations

import contextlib
import io
import json
import traceback
from typing import Any, Callable, Iterator, Sequence
import uuid

from treescope import context
from treescope import rendering_parts
from treescope._internal import html_encapsulation
from treescope._internal import html_escaping
from treescope._internal.parts import foldable_impl
from treescope._internal.parts import part_interface
from treescope._internal.api import abbreviation

_deferrables: context.ContextualValue[
    list[foldable_impl.DeferredWithThunk] | None
] = context.ContextualValue(
    module=__name__, qualname="_deferrables", initial_value=None
)
"""An optional list of accumulated deferrables, for use by this module."""


def maybe_defer_rendering(
    main_thunk: Callable[
        [rendering_parts.ExpandState | None],
        rendering_parts.RenderableTreePart,
    ],
    placeholder_thunk: Callable[[], rendering_parts.RenderableTreePart],
    expanded_newlines_for_layout: int | None = None,
) -> rendering_parts.RenderableTreePart:
  """Possibly defers rendering of a part in interactive contexts.

  This function can be used by advanced handlers and autovisualizers to delay
  the rendering of "expensive" leaves such as `jax.Array` until after the tree
  structure is drawn. If run in a non-interactive context, this just calls the
  main thunk. If run in an interactive context, it instead calls the placeholder
  thunk, and enqueues the placeholder thunk to be called later.

  Rendering can be performed in a deferred context by running the handlers under
  the `collecting_deferred_renderings` context manager, and then rendered to
  a sequence of streaming HTML updates using the `display_streaming_as_root`
  function.

  Note that handlers who call this are responsible for ensuring that the
  logic in `main_thunk` is safe to run at a later point in time. In particular,
  any rendering context managers may have been exited by the time this main
  thunk is called. As a best practice, handlers should control all of the logic
  in `main_thunk` and shouldn't recursively call the subtree renderer inside it;
  subtrees should be rendered before calling `maybe_defer_rendering`.

  Args:
    main_thunk: A callable producing the main part to render. If deferred and if
      ``expanded_newlines_for_layout`` is not None, this will be called with the
      inferred final expand state of placeholder after layout decisions.
      Otherwise, will be called with None.
    placeholder_thunk: A callable producing a placeholder object, which will be
      rendered if we are deferring rendering.
    expanded_newlines_for_layout: A guess at the extra height of this node after
      it has been expanded. If not None, this node will participate in the
      layout algorithm as if it had this height, and the final inferred
      expansion state will be passed to ``main_thunk``.

  Returns:
    Either the rendered main part or a wrapped placeholder that will later be
    replaced with the main part.
  """
  deferral_list = _deferrables.get()
  if deferral_list is None:
    return main_thunk(None)
  else:
    if expanded_newlines_for_layout is None:
      needs_layout_decision = False
      child = placeholder_thunk()
    else:
      needs_layout_decision = True
      child = foldable_impl.fake_placeholder_foldable(
          placeholder_thunk(), extra_newlines_guess=expanded_newlines_for_layout
      )
    placeholder = foldable_impl.DeferredPlaceholder(
        child=child,
        replacement_id="deferred_" + uuid.uuid4().hex,
        needs_layout_decision=needs_layout_decision,
    )
    deferral_list.append(
        foldable_impl.DeferredWithThunk(placeholder, main_thunk)
    )
    return placeholder


@contextlib.contextmanager
def collecting_deferred_renderings() -> (
    Iterator[list[foldable_impl.DeferredWithThunk]]
):
  # pylint: disable=g-doc-return-or-yield
  """Context manager that defers and collects `maybe_defer_rendering` calls.

  This context manager can be used by renderers that wish to render deferred
  objects in a streaming fashion. When used in a
  `with collecting_deferred_renderings() as deferreds:`
  expression, `deferreds` will be a list that is populated by calls to
  `maybe_defer_rendering`. This can later be passed to
  `display_streaming_as_root` to render the deferred object in a streaming
  fashion.

  Returns:
    A context manager in which `maybe_defer_rendering` calls will be deferred
    and collected into the result list.
  """
  # pylint: enable=g-doc-return-or-yield
  try:
    target = []
    with _deferrables.set_scoped(target):
      yield target
  finally:
    pass


################################################################################
# Top-level rendering and roundtrip mode implementation
################################################################################


def render_to_text_as_root(
    root_node: rendering_parts.RenderableTreePart,
    roundtrip: bool = False,
    strip_trailing_whitespace: bool = True,
    strip_whitespace_lines: bool = True,
) -> str:
  """Renders a root node to text.

  Args:
    root_node: The root node to render.
    roundtrip: Whether to render in roundtrip mode.
    strip_trailing_whitespace: Whether to remove trailing whitespace from lines.
    strip_whitespace_lines: Whether to remove lines that are entirely
      whitespace. These lines can sometimes be generated by layout code being
      conservative about line breaks. Should only be True if
      `strip_trailing_whitespace` is True.

  Returns:
    Text for the rendered node.
  """
  if strip_whitespace_lines and not strip_trailing_whitespace:
    raise ValueError(
        "strip_whitespace_lines must be False if "
        "strip_trailing_whitespace is False."
    )

  render_context = {}
  if roundtrip:
    render_context[foldable_impl.ABBREVIATION_THRESHOLD_KEY] = (
        abbreviation.roundtrip_abbreviation_threshold.get()
    )
  else:
    render_context[foldable_impl.ABBREVIATION_THRESHOLD_KEY] = (
        abbreviation.abbreviation_threshold.get()
    )

  stream = io.StringIO()
  root_node.render_to_text(
      stream,
      expanded_parent=True,
      indent=0,
      roundtrip_mode=roundtrip,
      render_context=render_context,
  )
  result = stream.getvalue()

  if strip_trailing_whitespace:
    trimmed_lines = []
    for line in result.split("\n"):
      line = line.rstrip()
      if line or not strip_whitespace_lines:
        trimmed_lines.append(line)
    result = "\n".join(trimmed_lines)
  return result


_TREESCOPE_PREAMBLE_SCRIPT = """(()=> {
  const defns = this.getRootNode().host.defns;
  let _pendingActions = [];
  let _pendingActionHandle = null;
  defns.runSoon = (work) => {
      const doWork = () => {
          const tick = performance.now();
          while (performance.now() - tick < 32) {
            if (_pendingActions.length == 0) {
                _pendingActionHandle = null;
                return;
            } else {
                const thunk = _pendingActions.shift();
                thunk();
            }
          }
          _pendingActionHandle = (
              window.requestAnimationFrame(doWork));
      };
      _pendingActions.push(work);
      if (_pendingActionHandle === null) {
          _pendingActionHandle = (
              window.requestAnimationFrame(doWork));
      }
  };
  defns.toggle_root_roundtrip = (rootelt, event) => {
      if (event.key == "r") {
          rootelt.classList.toggle("roundtrip_mode");
      }
  };
})();
"""


def _render_to_html_as_root_streaming(
    root_node: rendering_parts.RenderableTreePart,
    roundtrip: bool,
    deferreds: Sequence[foldable_impl.DeferredWithThunk],
    ignore_exceptions: bool = False,
) -> Iterator[str]:
  """Helper function: renders a root node to HTML one step at a time.

  Args:
    root_node: The root node to render.
    roundtrip: Whether to render in roundtrip mode.
    deferreds: Sequence of deferred objects to render and splice in.
    ignore_exceptions: Whether to ignore exceptions during deferred rendering,
      replacing them with error markers.

  Yields:
    HTML source for the rendered node, followed by logic to substitute each
    deferred object.
  """
  all_css_styles = set()
  all_js_defns = set()

  def _render_one(
      node,
      at_beginning_of_line: bool,
      render_context: dict[Any, Any],
      stream: io.StringIO,
  ):
    # Extract setup rules.
    html_context_for_setup = part_interface.HtmlContextForSetup(
        collapsed_selector=foldable_impl.COLLAPSED_SELECTOR,
        roundtrip_selector=foldable_impl.ROUNDTRIP_SELECTOR,
        abbreviate_selector=foldable_impl.make_abbreviate_selector(
            threshold=abbreviation.abbreviation_threshold.get(),
            roundtrip_threshold=(
                abbreviation.roundtrip_abbreviation_threshold.get()
            ),
        ),
    )
    setup_parts = node.html_setup_parts(html_context_for_setup)
    current_styles = []
    current_js_defns = []
    for part in setup_parts:
      if isinstance(part, part_interface.CSSStyleRule):
        if part not in all_css_styles:
          current_styles.append(part)
          all_css_styles.add(part)
      elif isinstance(part, part_interface.JavaScriptDefn):
        if part not in all_js_defns:
          current_js_defns.append(part)
          all_js_defns.add(part)
      else:
        raise ValueError(f"Invalid setup object: {part}")

    if current_styles:
      stream.write("<style>")
      for css_style in sorted(current_styles):
        stream.write(css_style.rule)
      stream.write("</style>")

    if current_js_defns:
      stream.write(
          "<treescope-run-here><script type='application/octet-stream'>"
      )
      for js_defn in sorted(current_js_defns):
        stream.write(js_defn.source)
      stream.write("</script></treescope-run-here>")

    # Render the node itself.
    node.render_to_html(
        stream,
        at_beginning_of_line=at_beginning_of_line,
        render_context=render_context,
    )

  # Set up the styles and scripts for the root object.
  stream = io.StringIO()
  stream.write("<style>")
  stream.write(html_escaping.without_repeated_whitespace("""
    .treescope_root {
      position: relative;
      font-family: monospace;
      white-space: pre;
      list-style-type: none;
      background-color: white;
      color: black;
      width: fit-content;
      min-width: 100%;
      box-sizing: border-box;
      padding-left: 2ch;
      line-height: 1.5;
      contain: content;
      content-visibility: auto;
      contain-intrinsic-size: auto none;
    }
  """))
  stream.write("</style>")
  # These scripts allow us to defer execution of javascript blocks until after
  # the content is loaded, avoiding locking up the browser rendering process.
  stream.write("<treescope-run-here><script type='application/octet-stream'>")
  stream.write(
      html_escaping.without_repeated_whitespace(_TREESCOPE_PREAMBLE_SCRIPT)
  )
  stream.write("</script></treescope-run-here>")

  # Render the root node.
  classnames = "treescope_root"
  if roundtrip:
    classnames += " roundtrip_mode"
  stream.write(
      f'<div class="{classnames}" tabindex="0" '
      'part="treescope_root" '
      'onkeydown="this.getRootNode().host.defns'
      '.toggle_root_roundtrip(this, event)">'
  )
  _render_one(root_node, True, {}, stream)
  stream.write("</div>")

  yield stream.getvalue()

  # Render any deferred parts. We insert each part into a hidden element, then
  # move them all out to their appropriate positions.
  if deferreds:
    stream = io.StringIO()
    for deferred in deferreds:
      stream.write(
          '<div style="display: none"'
          f' id="for_{deferred.placeholder.replacement_id}"><span>'
      )
      if (
          deferred.placeholder.saved_at_beginning_of_line is None
          or deferred.placeholder.saved_render_context is None
      ):
        replacement_part = rendering_parts.error_color(
            rendering_parts.text("<deferred rendering error>")
        )
      else:
        if deferred.placeholder.needs_layout_decision:
          assert isinstance(
              deferred.placeholder.child, part_interface.FoldableTreeNode
          )
          layout_decision = deferred.placeholder.child.get_expand_state()
        else:
          layout_decision = None
        try:
          replacement_part = deferred.thunk(layout_decision)
        except Exception as e:  # pylint: disable=broad-except
          if not ignore_exceptions:
            raise
          exc_child = rendering_parts.fold_condition(
              expanded=rendering_parts.indented_children(
                  [rendering_parts.text(traceback.format_exc())]
              ),
          )
          replacement_part = rendering_parts.error_color(
              rendering_parts.build_custom_foldable_tree_node(
                  label=rendering_parts.text(
                      f"<{type(e).__name__} during deferred rendering"
                  ),
                  contents=rendering_parts.siblings(
                      exc_child, rendering_parts.text(">")
                  ),
              ).renderable
          )
      _render_one(
          replacement_part,
          deferred.placeholder.saved_at_beginning_of_line,
          deferred.placeholder.saved_render_context,
          stream,
      )
      stream.write("</span></div>")

    all_ids = [deferred.placeholder.replacement_id for deferred in deferreds]
    # It's sometimes important to preserve node identity when inserting
    # deferred objects, for instance if we've already registered event listeners
    # on some nodes. However, editing the DOM in place can be slow because it
    # requires re-rendering the tree on every edit. To avoid this, we swap out
    # the tree with a clone, edit the original tree, then swap the original
    # tree back in.
    inner_script = (
        f"const targetIds = {json.dumps(all_ids)};"
        + html_escaping.without_repeated_whitespace("""
        const docroot = this.getRootNode();
        const treeroot = docroot.querySelector(".treescope_root");
        const treerootClone = treeroot.cloneNode(true);
        treeroot.replaceWith(treerootClone);
        const fragment = document.createDocumentFragment();
        fragment.appendChild(treeroot);
        for (let i = 0; i < targetIds.length; i++) {
            let target = fragment.getElementById(targetIds[i]);
            let sourceDiv = docroot.querySelector("#for_" + targetIds[i]);
            target.replaceWith(sourceDiv.firstElementChild);
            sourceDiv.remove();
        }
        treerootClone.replaceWith(treeroot);
        """)
    )
    stream.write(
        '<treescope-run-here><script type="application/octet-stream">'
        f"{inner_script}</script></treescope-run-here>"
    )
    yield stream.getvalue()


def render_to_html_as_root(
    root_node: rendering_parts.RenderableTreePart,
    roundtrip: bool = False,
    compressed: bool = False,
) -> str:
  """Renders a root node to HTML.

  This handles collecting styles and JS definitions and inserting the root
  HTML element.

  Args:
    root_node: The root node to render.
    roundtrip: Whether to render in roundtrip mode.
    compressed: Whether to compress the HTML for display.

  Returns:
    HTML source for the rendered node.
  """
  render_iterator = _render_to_html_as_root_streaming(root_node, roundtrip, [])
  html_src = "".join(render_iterator)
  return html_encapsulation.encapsulate_html(html_src, compress=compressed)


def display_streaming_as_root(
    root_node: rendering_parts.RenderableTreePart,
    deferreds: Sequence[foldable_impl.DeferredWithThunk],
    roundtrip: bool = False,
    compressed: bool = True,
    stealable: bool = False,
    ignore_exceptions: bool = False,
) -> str | None:
  """Displays a root node in an IPython notebook in a streaming fashion.

  Args:
    root_node: The root node to render.
    deferreds: Deferred objects to render and splice in.
    roundtrip: Whether to render in roundtrip mode.
    compressed: Whether to compress the HTML for display.
    stealable: Whether to return an extra HTML snippet that allows the streaming
      rendering to be relocated after it is shown.
    ignore_exceptions: Whether to ignore exceptions during deferred rendering,
      replacing them with error markers.

  Returns:
    If ``stealable`` is True, a final HTML snippet which, if inserted into a
    document, will "steal" the root node rendering, moving the DOM nodes for it
    into itself. In particular, using this as the HTML rendering of the root
    node during pretty printing will correctly associate the rendering with the
    IPython "cell output", which is visible in some IPython backends (e.g.
    JupyterLab). If ``stealable`` is False, returns None.
  """
  import IPython.display  # pylint: disable=import-outside-toplevel

  render_iterator = _render_to_html_as_root_streaming(
      root_node, roundtrip, deferreds, ignore_exceptions=ignore_exceptions
  )
  encapsulated_iterator = html_encapsulation.encapsulate_streaming_html(
      render_iterator, compress=compressed, stealable=stealable
  )

  for step in encapsulated_iterator:
    if step.segment_type == html_encapsulation.SegmentType.FINAL_OUTPUT_STEALER:
      return step.html_src
    else:
      IPython.display.display(IPython.display.HTML(step.html_src))
