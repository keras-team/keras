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

"""Internal implementation of array visualizer."""


from __future__ import annotations

import base64
import dataclasses
import io
import json
import os
from typing import Any, Literal, Sequence

import numpy as np
from treescope import ndarray_adapters
from treescope._internal import html_escaping
from treescope._internal.parts import basic_parts
from treescope._internal.parts import part_interface


AxisName = Any

PositionalAxisInfo = ndarray_adapters.PositionalAxisInfo
NamedPositionlessAxisInfo = ndarray_adapters.NamedPositionlessAxisInfo
NamedPositionalAxisInfo = ndarray_adapters.NamedPositionalAxisInfo
AxisInfo = ndarray_adapters.AxisInfo

ArrayInRegistry = Any


def load_arrayvis_javascript() -> str:
  """Loads the contents of `arrayvis.js` from the Python package.

  Returns:
    Source code for arrayviz.
  """
  filepath = __file__
  if filepath is None:
    raise ValueError("Could not find the path to arrayviz.js!")

  # Look for the resource relative to the current module's filesystem path.
  base = filepath.removesuffix("arrayviz_impl.py")
  load_path = os.path.join(base, "js", "arrayviz.js")

  with open(load_path, "r", encoding="utf-8") as f:
    return f.read()


def html_setup() -> (
    set[part_interface.CSSStyleRule | part_interface.JavaScriptDefn]
):
  """Builds the setup HTML that should be included in any arrayviz output."""
  arrayviz_src = html_escaping.heuristic_strip_javascript_comments(
      load_arrayvis_javascript()
  )
  return {
      part_interface.CSSStyleRule(html_escaping.without_repeated_whitespace("""
        .arrayviz_container {
            white-space: normal;
        }
        .arrayviz_container .info {
            font-family: monospace;
            color: #aaaaaa;
            margin-bottom: 0.25em;
            white-space: pre;
        }
        .arrayviz_container .info input[type="range"] {
            vertical-align: middle;
            filter: grayscale(1) opacity(0.5);
        }
        .arrayviz_container .info input[type="range"]:hover {
            filter: grayscale(0.5);
        }
        .arrayviz_container .info input[type="number"]:not(:focus) {
            border-radius: 3px;
        }
        .arrayviz_container .info input[type="number"]:not(:focus):not(:hover) {
            color: #777777;
            border: 1px solid #777777;
        }
        .arrayviz_container .info.sliders {
            white-space: pre;
        }
        .arrayviz_container .hovertip {
            display: none;
            position: absolute;
            background-color: white;
            border: 1px solid black;
            padding: 0.25ch;
            pointer-events: none;
            width: fit-content;
            overflow: visible;
            white-space: pre;
            z-index: 1000;
        }
        .arrayviz_container .hoverbox {
            display: none;
            position: absolute;
            box-shadow: 0 0 0 1px black, 0 0 0 2px white;
            pointer-events: none;
            z-index: 900;
        }
        .arrayviz_container .clickdata {
            white-space: pre;
        }
        .arrayviz_container .loading_message {
            color: #aaaaaa;
        }
      """)),
      part_interface.JavaScriptDefn(
          arrayviz_src + " this.getRootNode().host.defns.arrayviz = arrayviz;"
      ),
  }


def render_array_data_to_html(
    array_data: np.ndarray,
    valid_mask: np.ndarray,
    column_axes: Sequence[int],
    row_axes: Sequence[int],
    slider_axes: Sequence[int],
    axis_labels: list[str],
    vmin: float,
    vmax: float,
    cmap_type: Literal["continuous", "palette_index", "digitbox"],
    cmap_data: list[tuple[int, int, int]],
    info: str = "",
    formatting_instructions: list[dict[str, Any]] | None = None,
    dynamic_continuous_cmap: bool = False,
    raw_min_abs: float | None = None,
    raw_max_abs: float | None = None,
    pixels_per_cell: int | float = 7,
) -> str:
  """Helper to render an array to HTML by passing arguments to javascript.

  Args:
    array_data: Array data to render.
    valid_mask: Mask array, of same shape as array_data, that is True for items
      we should render.
    column_axes: Axes (by index into `array_data`) to arrange as columns,
      ordered from outermost group to innermost group.
    row_axes: Axes (by index into `array_data`) to arrange as rows, ordered from
      outermost group to innermost group.
    slider_axes: Axes to bind to sliders.
    axis_labels: Labels for each axis.
    vmin: Minimum for the colormap.
    vmax: Maximum for the colormap.
    cmap_type: Type of colormap (see `render_array`)
    cmap_data: Data for the colormap, as a sequence of RGB triples.
    info: Info for the plot.
    formatting_instructions: Formatting instructions for values on mouse hover
      or click. These will be interpreted by `formatValueAndIndices` on the
      JavaScript side. Can assume each axis is named "a0", "a1", etc. when
      running in JavaScript.
    dynamic_continuous_cmap: Whether to dynamically adjust the colormap during
      rendering.
    raw_min_abs: Minimum absolute value of the array, for dynamic remapping.
    raw_max_abs: Maximum absolute value of the array, for dynamic remapping.
    pixels_per_cell: The initial number of pixels per cell when rendering.

  Returns:
    HTML source for an arrayviz rendering.
  """
  assert len(array_data.shape) == len(axis_labels)
  assert len(valid_mask.shape) == len(axis_labels)

  if formatting_instructions is None:
    formatting_instructions = [{"type": "value"}]

  # Compute strides for each axis. We refer to each axis as "a0", "a1", etc
  # across the JavaScript boundary.
  stride = 1
  strides = {}
  for i, axis_size in reversed(list(enumerate(array_data.shape))):
    strides[f"a{i}"] = stride
    stride *= axis_size

  if cmap_type == "continuous":
    converted_array_data = array_data.astype(np.float32)
    array_dtype = "float32"
  else:
    converted_array_data = array_data.astype(np.int32)
    array_dtype = "int32"

  def axis_spec_arg(i):
    return {
        "name": f"a{i}",
        "label": axis_labels[i],
        "start": 0,
        "end": array_data.shape[i],
    }

  x_axis_specs_arg = []
  for axis in column_axes:
    x_axis_specs_arg.append(axis_spec_arg(axis))

  y_axis_specs_arg = []
  for axis in row_axes:
    y_axis_specs_arg.append(axis_spec_arg(axis))

  sliced_axis_specs_arg = []
  for axis in slider_axes:
    sliced_axis_specs_arg.append(axis_spec_arg(axis))

  args_json = json.dumps({
      "info": info,
      "arrayBase64": base64.b64encode(converted_array_data.tobytes()).decode(
          "ascii"
      ),
      "arrayDtype": array_dtype,
      "validMaskBase64": base64.b64encode(
          valid_mask.astype(np.uint8).tobytes()
      ).decode("ascii"),
      "dataStrides": strides,
      "xAxisSpecs": x_axis_specs_arg,
      "yAxisSpecs": y_axis_specs_arg,
      "slicedAxisSpecs": sliced_axis_specs_arg,
      "colormapConfig": {
          "type": cmap_type,
          "min": vmin,
          "max": vmax,
          "dynamic": dynamic_continuous_cmap,
          "rawMinAbs": raw_min_abs,
          "rawMaxAbs": raw_max_abs,
          "cmapData": cmap_data,
      },
      "pixelsPerCell": pixels_per_cell,
      "valueFormattingInstructions": formatting_instructions,
  })
  # Note: We need to save the parent of the treescope-run-here element first,
  # because it will be removed before the runSoon callback executes.
  inner_fn = html_escaping.without_repeated_whitespace("""
    const parent = this.parentNode;
    const defns = this.getRootNode().host.defns;
    defns.runSoon(() => {
        const tpl = parent.querySelector('template.deferred_args');
        const config = JSON.parse(
            tpl.content.querySelector('script').textContent
        );
        tpl.remove();
        defns.arrayviz.buildArrayvizFigure(parent, config);
    });
  """)
  src = (
      '<div class="arrayviz_container">'
      '<span class="loading_message">Rendering array...</span>'
      f'<treescope-run-here><script type="application/octet-stream">{inner_fn}'
      "</script></treescope-run-here>"
      '<template class="deferred_args">'
      f'<script type="application/json">{args_json}</script></template></div>'
  )
  return src


def infer_rows_and_columns(
    all_axes: Sequence[AxisInfo],
    known_rows: Sequence[AxisInfo] = (),
    known_columns: Sequence[AxisInfo] = (),
    edge_items_per_axis: tuple[int | None, ...] | None = None,
) -> tuple[list[AxisInfo], list[AxisInfo]]:
  """Infers an ordered assignment of axis indices or names to rows and columns.

  The unassigned axes are sorted by size and then assigned to rows and columns
  to try to balance the total number of elements along the row and column axes.
  This currently uses a greedy algorithm with an adjustment to try to keep
  columns longer than rows, except when there are exactly two axes and both are
  positional, in which case it lays out axis 0 as the rows and axis 1 as the
  columns.

  Axes with logical positions are sorted before axes with only names
  (in reverse order, so that later axes are rendered on the inside). Axes with
  names only appear afterward, with explicitly-assigned ones before unassigned
  ones.

  Args:
    all_axes: Sequence of axis infos in the array that should be assigned.
    known_rows: Sequence of axis indices or names that must map to rows.
    known_columns: Sequence of axis indices or names that must map to columns.
    edge_items_per_axis: Optional edge items specification, determining
      truncated size of each axis. Must match the ordering of `all_axes`.

  Returns:
    Tuple (rows, columns) of assignments, which consist of `known_rows` and
    `known_columns` followed by the remaining unassigned axes in a balanced
    order.
  """
  if edge_items_per_axis is None:
    edge_items_per_axis = (None,) * len(all_axes)

  if not known_rows and not known_columns and len(all_axes) == 2:
    ax_a, ax_b = all_axes
    if (
        isinstance(ax_a, PositionalAxisInfo)
        and isinstance(ax_b, PositionalAxisInfo)
        and {ax_a.axis_logical_index, ax_b.axis_logical_index} == {0, 1}
    ):
      # Two-dimensional positional array. Always do rows then columns.
      if ax_a.axis_logical_index == 0:
        return ([ax_a], [ax_b])
      else:
        return ([ax_b], [ax_a])

  truncated_sizes = {
      ax: ax.size if edge_items is None else 2 * edge_items + 1
      for ax, edge_items in zip(all_axes, edge_items_per_axis)
  }
  unassigned = [
      ax for ax in all_axes if ax not in known_rows and ax not in known_columns
  ]

  # Sort by size descending, so that we make the most important layout decisions
  # first.
  unassigned = sorted(
      unassigned, key=lambda ax: (truncated_sizes[ax], ax.size), reverse=True
  )

  # Compute the total size every axis would have if we assigned them to the
  # same axis.
  unassigned_size = np.prod([truncated_sizes[ax] for ax in unassigned])

  rows = list(known_rows)
  row_size = np.prod([truncated_sizes[ax] for ax in rows])
  columns = list(known_columns)
  column_size = np.prod([truncated_sizes[ax] for ax in columns])

  for ax in unassigned:
    axis_size = truncated_sizes[ax]
    unassigned_size = unassigned_size // axis_size
    if row_size * axis_size > column_size * unassigned_size:
      # If we assign this to the row axis, we'll end up with a visualization
      # with more rows than columns regardless of what we do later, which can
      # waste screen space. Assign to columns instead.
      columns.append(ax)
      column_size *= axis_size
    else:
      # Assign to the row axis. We'll assign columns later.
      rows.append(ax)
      row_size *= axis_size

  # The specific ordering of axes along the rows and the columns is somewhat
  # arbitrary. Re-order each so that explicitly requested axes are first, then
  # unassigned positional axes in reverse position order, then unassigned named
  # axes.
  def ax_sort_key(ax: AxisInfo):
    if ax not in unassigned:
      return (0,)
    elif isinstance(ax, PositionalAxisInfo | NamedPositionalAxisInfo):
      return (1, -ax.axis_logical_index)
    else:
      return (2,)

  return sorted(rows, key=ax_sort_key), sorted(columns, key=ax_sort_key)


def infer_vmin_vmax(
    array: np.ndarray,
    mask: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    around_zero: bool,
    trim_outliers: bool,
) -> tuple[float, float]:
  """Infer reasonable lower and upper colormap bounds from an array."""
  inferring_both_bounds = vmax is None and vmin is None
  finite_mask = np.logical_and(np.isfinite(array), mask)
  if vmax is None:
    if around_zero:
      if vmin is not None:
        vmax = -vmin  # pylint: disable=invalid-unary-operand-type
      else:
        vmax = np.max(np.where(finite_mask, np.abs(array), 0))
    else:
      vmax = np.max(np.where(finite_mask, array, -np.inf))

  assert vmax is not None

  if vmin is None:
    if around_zero:
      vmin = -vmax  # pylint: disable=invalid-unary-operand-type
    else:
      vmin = np.min(np.where(finite_mask, array, np.inf))

  if inferring_both_bounds and trim_outliers:
    if around_zero:
      center = 0
    else:
      center = np.nanmean(np.where(finite_mask, array, np.nan))
      center = np.where(np.isfinite(center), center, 0.0)

    second_moment = np.nanmean(
        np.where(finite_mask, np.square(array - center), np.nan)
    )
    sigma = np.where(
        np.isfinite(second_moment), np.sqrt(second_moment), vmax - vmin
    )

    vmin_limit = center - 3 * sigma
    vmin = np.maximum(vmin, vmin_limit)
    vmax_limit = center + 3 * sigma
    vmax = np.minimum(vmax, vmax_limit)

  return vmin, vmax


def infer_abs_min_max(
    array: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
  """Infer smallest and largest absolute values in array."""
  finite_mask = np.logical_and(np.isfinite(array), mask)
  absmin = np.min(
      np.where(np.logical_and(finite_mask, array != 0), np.abs(array), np.inf)
  )
  absmin = np.where(np.isinf(absmin), 0.0, absmin)
  absmax = np.max(np.where(finite_mask, np.abs(array), -np.inf))
  absmax = np.where(np.isinf(absmax), 0.0, absmax)
  return absmin, absmax


def infer_balanced_truncation(
    shape: Sequence[int],
    maximum_size: int,
    cutoff_size_per_axis: int,
    minimum_edge_items: int,
    doubling_bonus: float = 10.0,
) -> tuple[int | None, ...]:
  """Infers a balanced truncation from a shape.

  This function computes a set of truncation sizes for each axis of the array
  such that it obeys the constraints about array and axis sizes, while also
  keeping the relative proportions of the array consistent (e.g. we keep more
  elements along axes that were originally longer). This means that the aspect
  ratio of the truncated array will still resemble the aspect ratio of the
  original array.

  To avoid very-unbalanced renderings and truncate longer axes more than short
  ones, this function truncates based on the square-root of the axis size by
  default.

  Args:
    shape: The shape of the array we are truncating.
    maximum_size: Maximum number of elements of an array to show. Arrays larger
      than this will be truncated along one or more axes.
    cutoff_size_per_axis: Maximum number of elements of each individual axis to
      show without truncation. Any axis longer than this will be truncated, with
      their visual size increasing logarithmically with the true axis size
      beyond this point.
    minimum_edge_items: How many values to keep along each axis for truncated
      arrays. We may keep more than this up to the budget of maximum_size.
    doubling_bonus: Number of elements to add to each axis each time it doubles
      beyond `cutoff_size_per_axis`. Used to make longer axes appear visually
      longer while still keeping them a reasonable size.

  Returns:
    A tuple of edge sizes. Each element corresponds to an axis in `shape`,
    and is either `None` (for no truncation) or an integer (corresponding to
    the number of elements to keep at the beginning and at the end).
  """
  shape_arr = np.array(list(shape))
  remaining_elements_to_divide = maximum_size
  edge_items_per_axis = {}
  # Order our shape from smallest to largest, since the smallest axes will
  # require the least amount of truncation and will have the most stringent
  # constraints.
  sorted_axes = np.argsort(shape_arr)
  sorted_shape = shape_arr[sorted_axes]

  # Figure out maximum sizes based on the cutoff
  cutoff_adjusted_maximum_sizes = np.where(
      sorted_shape <= cutoff_size_per_axis,
      sorted_shape,
      cutoff_size_per_axis
      + doubling_bonus * np.log2(sorted_shape / cutoff_size_per_axis),
  )

  # Suppose we want to make a scaled version of the array with relative
  # axis sizes
  #   s0, s1, s2, ...
  # The total size is then
  #   size = (c * s0) * (c * s1) * (c * s2) * ...
  #   log(size) = ndim * log(c) + [ log s0 + log s1 + log s2 + ... ]
  # If we have a known final size we want to reach, we can solve for c as
  #   c = exp( (log size - [ log s0 + log s1 + log s2 + ... ]) / ndim )
  axis_proportions = np.sqrt(sorted_shape)
  log_axis_proportions = np.log(axis_proportions)
  for i in range(len(sorted_axes)):
    original_axis = sorted_axes[i]
    size = shape_arr[original_axis]
    # If we truncated this axis and every axis after it proportional to
    # their weights, how small of an axis size would we need for this
    # axis?
    log_c = (
        np.log(remaining_elements_to_divide) - np.sum(log_axis_proportions[i:])
    ) / (len(shape) - i)
    soft_limit_for_this_axis = np.exp(log_c + log_axis_proportions[i])
    cutoff_limit_for_this_axis = np.floor(
        np.minimum(
            soft_limit_for_this_axis,
            cutoff_adjusted_maximum_sizes[i],
        )
    )
    if size <= 2 * minimum_edge_items + 1 or size <= cutoff_limit_for_this_axis:
      # If this axis is already smaller than the minimum size it would have
      # after truncation, there's no reason to truncate it.
      # But pretend we did, so that other axes still grow monotonically if
      # their axis sizes increase.
      remaining_elements_to_divide = (
          remaining_elements_to_divide / soft_limit_for_this_axis
      )
      edge_items_per_axis[original_axis] = None
    elif cutoff_limit_for_this_axis < 2 * minimum_edge_items + 1:
      # If this axis is big enough to truncate, but our naive target size is
      # smaller than the minimum allowed truncation, we should truncate it
      # to the minimum size allowed instead.
      edge_items_per_axis[original_axis] = minimum_edge_items
      remaining_elements_to_divide = remaining_elements_to_divide / (
          2 * minimum_edge_items + 1
      )
    else:
      # Otherwise, truncate it and all remaining axes based on our target
      # truncations.
      for j in range(i, len(sorted_axes)):
        visual_size = np.floor(
            np.minimum(
                np.exp(log_c + log_axis_proportions[j]),
                cutoff_adjusted_maximum_sizes[j],
            )
        )
        edge_items_per_axis[sorted_axes[j]] = int(visual_size // 2)
      break

  return tuple(
      edge_items_per_axis[orig_axis] for orig_axis in range(len(shape))
  )


def compute_truncated_shape(
    shape: tuple[int, ...],
    edge_items: tuple[int | None, ...],
) -> tuple[int, ...]:
  """Computes the shape of a truncated array.

  This can be used to estimate the size of an array visualization after it has
  been truncated by `infer_balanced_truncation`.

  Args:
    shape: The original array shape.
    edge_items: Number of edge items to keep along each axis.

  Returns:
    The shape of the truncated array.
  """
  return tuple(
      orig if edge is None else 2 * edge + 1
      for orig, edge in zip(shape, edge_items)
  )


@dataclasses.dataclass(frozen=True)
class ArrayvizRendering(part_interface.RenderableTreePart):
  """A rendering of an array with Arrayviz.

  Attributes:
    html_src: HTML source for the rendering.
  """

  html_src: str

  def _compute_collapsed_width(self) -> int:
    return 80

  def _compute_newlines_in_expanded_parent(self) -> int:
    return 10

  def foldables_in_this_part(self) -> Sequence[part_interface.FoldableTreeNode]:
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
    stream.write("<Arrayviz rendering>")

  def html_setup_parts(
      self, setup_context: part_interface.HtmlContextForSetup
  ) -> set[part_interface.CSSStyleRule | part_interface.JavaScriptDefn]:
    del setup_context
    return html_setup()

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    stream.write(self.html_src)


@dataclasses.dataclass(frozen=True)
class ArrayvizDigitboxRendering(ArrayvizRendering):
  """A rendering of a single digitbox with Arrayviz."""

  def _compute_collapsed_width(self) -> int:
    return 2

  def _compute_newlines_in_expanded_parent(self) -> int:
    return 1


@dataclasses.dataclass(frozen=True)
class ValueColoredTextbox(basic_parts.DeferringToChild):
  """A rendering of text with a colored background.

  Attributes:
    child: Child part to render.
    text_color: Color for the text.
    background_color: Color for the background, usually from a colormap.
    out_of_bounds: Whether this value was out of bounds of the colormap.
    value: Underlying float value that is being visualized. Rendered on hover.
  """

  child: part_interface.RenderableTreePart
  text_color: str
  background_color: str
  out_of_bounds: bool
  value: float

  def html_setup_parts(
      self, setup_context: part_interface.HtmlContextForSetup
  ) -> set[part_interface.CSSStyleRule | part_interface.JavaScriptDefn]:
    return (
        {
            part_interface.CSSStyleRule(
                html_escaping.without_repeated_whitespace("""
                    .arrayviz_textbox {
                        padding-left: 0.5ch;
                        padding-right: 0.5ch;
                        outline: 1px solid black;
                        position: relative;
                        display: inline-block;
                        font-family: monospace;
                        white-space: pre;
                        margin-top: 1px;
                        box-sizing: border-box;
                    }
                    .arrayviz_textbox.out_of_bounds {
                        outline: 3px double darkorange;
                    }
                    .arrayviz_textbox .value {
                        display: none;
                        position: absolute;
                        bottom: 110%;
                        left: 0;
                        overflow: visible;
                        color: black;
                        background-color: white;
                        font-size: 0.7em;
                    }
                    .arrayviz_textbox:hover .value {
                        display: block;
                    }
                    """)
            )
        }
        | self.child.html_setup_parts(setup_context)
    )

  def render_to_html(
      self,
      stream: io.TextIOBase,
      *,
      at_beginning_of_line: bool = False,
      render_context: dict[Any, Any],
  ):
    class_string = "arrayviz_textbox"
    if self.out_of_bounds:
      class_string += " out_of_bounds"
    bg_color = html_escaping.escape_html_attribute(self.background_color)
    text_color = html_escaping.escape_html_attribute(self.text_color)
    stream.write(
        f'<span class="{class_string}" style="background-color:{bg_color};'
        f' color:{text_color}">'
        f'<span class="value">{float(self.value):.4g}</span>'
    )
    self.child.render_to_html(
        stream,
        at_beginning_of_line=False,
        render_context=render_context,
    )
    stream.write("</span>")
