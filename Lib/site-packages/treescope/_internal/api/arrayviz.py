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

"""Single-purpose ndarray visualizer for Python in vanilla JavaScript.

Designed to quickly visualize the contents of arbitrarily-high-dimensional
NDArrays, to allow them to be visualized by default instead of requiring lots
of manual effort.
"""

from __future__ import annotations

import collections
import itertools
import json
from typing import Any, Literal, Sequence

import numpy as np
from treescope import context
from treescope import dtype_util
from treescope import ndarray_adapters
from treescope import type_registries
from treescope._internal import arrayviz_impl
from treescope._internal import figures_impl
from treescope._internal import html_escaping
from treescope._internal.parts import basic_parts
from treescope._internal.parts import common_styles

AxisName = Any

PositionalAxisInfo = ndarray_adapters.PositionalAxisInfo
NamedPositionlessAxisInfo = ndarray_adapters.NamedPositionlessAxisInfo
NamedPositionalAxisInfo = ndarray_adapters.NamedPositionalAxisInfo
AxisInfo = ndarray_adapters.AxisInfo

ArrayInRegistry = Any


default_sequential_colormap: context.ContextualValue[
    list[tuple[int, int, int]]
] = context.ContextualValue(
    module=__name__,
    qualname="default_sequential_colormap",
    # Matplotlib Viridis_20
    initial_value=[
        (68, 1, 84),
        (72, 20, 103),
        (72, 38, 119),
        (69, 55, 129),
        (63, 71, 136),
        (57, 85, 140),
        (50, 100, 142),
        (45, 113, 142),
        (40, 125, 142),
        (35, 138, 141),
        (31, 150, 139),
        (32, 163, 134),
        (41, 175, 127),
        (59, 187, 117),
        (86, 198, 103),
        (115, 208, 86),
        (149, 216, 64),
        (184, 222, 41),
        (221, 227, 24),
        (253, 231, 37),
    ],
)

default_diverging_colormap: context.ContextualValue[
    list[tuple[int, int, int]]
] = context.ContextualValue(
    module=__name__,
    qualname="default_diverging_colormap",
    # cmocean Balance_19_r[1:-1]
    initial_value=[
        (96, 14, 34),
        (134, 14, 41),
        (167, 36, 36),
        (186, 72, 46),
        (198, 107, 77),
        (208, 139, 115),
        (218, 171, 155),
        (228, 203, 196),
        (241, 236, 235),
        (202, 212, 216),
        (161, 190, 200),
        (117, 170, 190),
        (75, 148, 186),
        (38, 123, 186),
        (12, 94, 190),
        (41, 66, 162),
        (37, 47, 111),
    ],
)


def render_array(
    array: ArrayInRegistry,
    *,
    columns: Sequence[AxisName | int] = (),
    rows: Sequence[AxisName | int] = (),
    sliders: Sequence[AxisName | int] = (),
    valid_mask: Any | None = None,
    continuous: bool | Literal["auto"] = "auto",
    around_zero: bool | Literal["auto"] = "auto",
    vmax: float | None = None,
    vmin: float | None = None,
    trim_outliers: bool = True,
    dynamic_colormap: bool | Literal["auto"] = "auto",
    colormap: list[tuple[int, int, int]] | None = None,
    truncate: bool = False,
    maximum_size: int = 10_000,
    cutoff_size_per_axis: int = 512,
    minimum_edge_items: int = 5,
    axis_item_labels: dict[AxisName | int, list[str]] | None = None,
    value_item_labels: dict[int, str] | None = None,
    axis_labels: dict[AxisName | int, str] | None = None,
    pixels_per_cell: int | float = 7,
) -> figures_impl.TreescopeFigure:
  """Renders an array (positional or named) to a displayable HTML object.

  Each element of the array is rendered to a fixed-size square, with its
  position determined based on its index, and with each level of x and y axis
  represented by a "faceted" plot.

  Out-of-bounds or otherwise unusual data is rendered with an annotation:

  * "X" means a value was NaN (for continuous data) or went out-of-bounds for
    the integer palette (for discrete data).

  * "I" or "-I" means a value was infinity or negative infinity.

  * "+" or "-" means a value was finite but went outside the bounds of the
    colormap (e.g. it was larger than ``vmax`` or smaller than ``vmin``). By
    default this applies to values more than 3 standard deviations outside the
    mean.

  * Four light dots on grey means a value was masked out by ``valid_mask``, or
    truncated due to the maximum size or axis cutoff thresholds.

  By default, this method automatically chooses a color rendering strategy based
  on the arguments:

  * If an explicit colormap is provided:

    * If ``continuous`` is True, the provided colors are interpreted as color
      stops and interpolated between.

    * If ``continuous`` is False, the provided colors are interpreted as an
      indexed color palette, and each index of the palette is used to render
      the corresponding integer, starting from zero.

  * Otherwise:

    * If ``continuous`` is True:

      * If ``around_zero`` is True, uses the diverging colormap
        `default_diverging_colormap`. The initial value of this is a truncated
        version of the perceptually-uniform "Balance" colormap from cmocean,
        with blue for positive numbers and red for negative ones.

      * If ``around_zero`` is False, uses the sequential colormap
        `default_sequential_colormap`.The initial value of this is the
        perceptually-uniform "Viridis" colormap from matplotlib.

    * If ``continuous`` is False, uses a pattern-based "digitbox" rendering
      strategy to render integers up to 9,999,999 as nested squares, with one
      square per integer digit and digit colors drawn from the D3 Category20
      colormap.

  Args:
      array: The array to render. The type of this array must be registered in
        the `type_registries.NDARRAY_ADAPTER_REGISTRY`.
      columns: Sequence of axis names or positional axis indices that should be
        placed on the x axis, from innermost to outermost. If not provided,
        inferred automatically.
      rows: Sequence of axis names or positional axis indices that should be
        placed on the y axis, from innermost to outermost. If not provided,
        inferred automatically.
      sliders: Sequence of axis names or positional axis indices for which we
        should show only a single slice at a time, with the index determined
        with a slider.
      valid_mask: Optionally, a boolean array with the same shape (and, if
        applicable, axis names) as `array`, which is True for the locations that
        we should actually render, and False for locations that do not have
        valid array data.
      continuous: Whether to interpret this array as numbers along the real
        line, and visualize using an interpolated colormap. If "auto", inferred
        from the dtype of `array`.
      around_zero: Whether the array data should be rendered symmetrically
        around zero using a diverging colormap, scaled based on the absolute
        magnitude of the inputs, instead of rescaled to be between the min and
        max of the data. If "auto", treated as True unless both `vmin` and
        `vmax` are set to incompatible values.
      vmax: Largest value represented in the colormap. If omitted and
        around_zero is True, inferred as ``max(abs(array))`` or as ``-vmin``. If
        omitted and around_zero is False, inferred as ``max(array)``.
      vmin: Smallest value represented in the colormap. If omitted and
        around_zero is True, inferred as ``-max(abs(array))`` or as ``-vmax``.
        If omitted and around_zero is False, inferred as ``min(array)``.
      trim_outliers: Whether to try to trim outliers when inferring ``vmin`` and
        ``vmax``. If True, clips them to 3 standard deviations away from the
        mean (or 3 sqrt-second-moments around zero) if they would otherwise
        exceed it.
      dynamic_colormap: Whether to dynamically adjust the colormap based on
        mouse hover. Requires a continuous colormap, and ``around_zero=True``.
        If "auto", will be enabled for continuous arrays if ``around_zero`` is
        True and neither ``vmin`` nor ``vmax`` are provided.
      colormap: An optional explicit colormap to use, represented as a list of
        ``(r,g,b)`` tuples, where each channel is between 0 and 255. A good
        place to get colormaps is the ``palettable`` package, e.g. you can pass
        something like ``palettable.matplotlib.Inferno_20.colors``.
      truncate: Whether or not to truncate the array to a smaller size before
        rendering.
      maximum_size: Maximum number of elements of an array to show. Arrays
        larger than this will be truncated along one or more axes. Ignored
        unless ``truncate`` is True.
      cutoff_size_per_axis: Maximum number of elements of each individual axis
        to show without truncation. Any axis longer than this will be truncated,
        with their visual size increasing logarithmically with the true axis
        size beyond this point. Ignored unless ``truncate`` is True.
      minimum_edge_items: How many values to keep along each axis for truncated
        arrays. We may keep more than this up to the budget of maximum_size.
        Ignored unless ``truncate`` is True.
      axis_item_labels: An optional mapping from axis names/positions to a list
        of strings, of the same length as the axis length, giving a label to
        each item along that axis. For instance, this could be the token string
        corresponding to each position along a sequence axis, or the class label
        corresponding to each category across a classifier's output axis. This
        is shown in the tooltip when hovering over a pixel, and shown below the
        array when a pixel is clicked on. For convenience, names in this
        dictionary that don't match any axes in the input are simply ignored, so
        that you can pass the same labels while rendering arrays that may not
        have the same axis names.
      value_item_labels: For categorical data, an optional mapping from each
        value to a string. For instance, this could be the token value
        corresponding to each token ID in a sequence of tokens.
      axis_labels: Optional mapping from axis names / indices to the labels we
        should use for that axis. If not provided, we label the named axes with
        their names and the positional axes with "axis {i}", and also add th
        axis size.
      pixels_per_cell: Size of each rendered array element in pixels, between 1
        and 21 inclusive. This controls the zoom level of the rendering. Array
        elements are always drawn at 7 pixels per cell and then rescaled, so
        out-of-bounds annotations and "digitbox" integer value patterns may not
        display correctly at fewer than 7 pixels per cell.

  Returns:
    An object which can be rendered in an IPython notebook, containing the
    HTML source of an arrayviz rendering.
  """
  # Retrieve the adapter for this array, which we will use to construct
  # the rendering.
  type_registries.update_registries_for_imports()
  adapter = type_registries.lookup_ndarray_adapter(array)
  if adapter is None:
    raise TypeError(
        f"Cannot render array with unrecognized type {type(array)} (not found"
        " in array adapter registry)"
    )
  if not 1 <= pixels_per_cell <= 21:
    raise ValueError(
        "pixels_per_cell must be between 1 and 21 inclusive, got"
        f" {pixels_per_cell}"
    )

  # Extract information about axis names, indices, and sizes.
  array_axis_info = adapter.get_axis_info_for_array_data(array)

  data_axis_from_axis_info = {
      info: axis for axis, info in enumerate(array_axis_info)
  }
  assert len(data_axis_from_axis_info) == len(array_axis_info)

  info_by_name_or_position = {}
  for info in array_axis_info:
    if isinstance(info, NamedPositionalAxisInfo):
      info_by_name_or_position[info.axis_name] = info
      info_by_name_or_position[info.axis_logical_index] = info
    elif isinstance(info, PositionalAxisInfo):
      info_by_name_or_position[info.axis_logical_index] = info
    elif isinstance(info, NamedPositionlessAxisInfo):
      info_by_name_or_position[info.axis_name] = info
    else:
      raise ValueError(f"Unrecognized axis info {type(info)}")

  row_infos = [info_by_name_or_position[spec] for spec in rows]
  column_infos = [info_by_name_or_position[spec] for spec in columns]
  slider_infos = [info_by_name_or_position[spec] for spec in sliders]

  unassigned_axes = set(array_axis_info)
  seen_axes = set()
  for axis_info in itertools.chain(row_infos, column_infos, slider_infos):
    if axis_info in seen_axes:
      raise ValueError(
          f"Axis {axis_info} appeared multiple times in rows/columns/sliders"
          " specifications. Each axis must be assigned to at most one"
          " location."
      )
    seen_axes.add(axis_info)
    unassigned_axes.remove(axis_info)

  if truncate:
    # Infer a good truncated shape for this array.
    edge_items_per_axis = arrayviz_impl.infer_balanced_truncation(
        tuple(info.size for info in array_axis_info),
        maximum_size=maximum_size,
        cutoff_size_per_axis=cutoff_size_per_axis,
        minimum_edge_items=minimum_edge_items,
    )
  else:
    edge_items_per_axis = (None,) * len(array_axis_info)

  # Obtain truncated array and mask data from the adapter.
  truncated_array_data, truncated_mask_data = (
      adapter.get_array_data_with_truncation(
          array=array,
          mask=valid_mask,
          edge_items_per_axis=edge_items_per_axis,
      )
  )

  # Step 5: Figure out which axes to render as rows, columns, and sliders and
  # in which order. We start with the explicitly-requested axes, then add more
  # axes to the rows and columns until we've assigned all of them, trying to
  # balance rows and columns.

  row_infos, column_infos = arrayviz_impl.infer_rows_and_columns(
      all_axes=[ax for ax in array_axis_info if ax not in slider_infos],
      known_rows=row_infos,
      known_columns=column_infos,
      edge_items_per_axis=edge_items_per_axis,
  )

  return figures_impl.TreescopeFigure(
      _render_pretruncated(
          array_axis_info=array_axis_info,
          row_infos=row_infos,
          column_infos=column_infos,
          slider_infos=slider_infos,
          truncated_array_data=truncated_array_data,
          truncated_mask_data=truncated_mask_data,
          edge_items_per_axis=edge_items_per_axis,
          continuous=continuous,
          around_zero=around_zero,
          vmax=vmax,
          vmin=vmin,
          trim_outliers=trim_outliers,
          dynamic_colormap=dynamic_colormap,
          colormap=colormap,
          axis_item_labels=axis_item_labels,
          value_item_labels=value_item_labels,
          axis_labels=axis_labels,
          pixels_per_cell=pixels_per_cell,
      )
  )


def _render_pretruncated(
    *,
    array_axis_info: Sequence[AxisInfo],
    row_infos: Sequence[AxisInfo],
    column_infos: Sequence[AxisInfo],
    slider_infos: Sequence[AxisInfo],
    truncated_array_data: np.ndarray,
    truncated_mask_data: np.ndarray,
    edge_items_per_axis: Sequence[int | None],
    continuous: bool | Literal["auto"],
    around_zero: bool | Literal["auto"],
    vmax: float | None,
    vmin: float | None,
    trim_outliers: bool,
    dynamic_colormap: bool | Literal["auto"],
    colormap: list[tuple[int, int, int]] | None,
    axis_item_labels: dict[AxisName | int, list[str]] | None,
    value_item_labels: dict[int, str] | None,
    axis_labels: dict[AxisName | int, str] | None,
    pixels_per_cell: int | float = 7,
) -> arrayviz_impl.ArrayvizRendering:
  """Internal helper to render an array that has already been truncated."""
  if axis_item_labels is None:
    axis_item_labels = {}

  if value_item_labels is None:
    value_item_labels = {}

  if axis_labels is None:
    axis_labels = {}

  data_axis_from_axis_info = {
      info: axis for axis, info in enumerate(array_axis_info)
  }
  assert len(data_axis_from_axis_info) == len(array_axis_info)

  has_name_only = False
  positional_count = 0

  info_by_name_or_position = {}
  for info in array_axis_info:
    if isinstance(info, NamedPositionalAxisInfo):
      info_by_name_or_position[info.axis_name] = info
      info_by_name_or_position[info.axis_logical_index] = info
      positional_count += 1
    elif isinstance(info, PositionalAxisInfo):
      info_by_name_or_position[info.axis_logical_index] = info
      positional_count += 1
    elif isinstance(info, NamedPositionlessAxisInfo):
      info_by_name_or_position[info.axis_name] = info
      has_name_only = True
    else:
      raise ValueError(f"Unrecognized axis info {type(info)}")

  axis_labels_by_info = {
      info_by_name_or_position[orig_key]: value
      for orig_key, value in axis_labels.items()
  }
  axis_item_labels_by_info = {
      info_by_name_or_position[orig_key]: value
      for orig_key, value in axis_item_labels.items()
  }

  skip_start_indices = [
      edge_items if edge_items is not None else axis_info.size
      for edge_items, axis_info in zip(edge_items_per_axis, array_axis_info)
  ]
  skip_end_indices = [
      axis_info.size - edge_items if edge_items is not None else axis_info.size
      for edge_items, axis_info in zip(edge_items_per_axis, array_axis_info)
  ]

  # Convert the axis names into indices into our data array.
  column_data_axes = [
      data_axis_from_axis_info[orig_axis] for orig_axis in column_infos
  ]
  row_data_axes = [
      data_axis_from_axis_info[orig_axis] for orig_axis in row_infos
  ]
  slider_data_axes = [
      data_axis_from_axis_info[orig_axis] for orig_axis in slider_infos
  ]

  # Step 6: Figure out how to render the labels and indices of each axis.
  # We render indices using a small interpreted format language that can be
  # serialized to JSON and interpreted in JavaScript.
  data_axis_labels = {}
  formatting_instructions = []
  formatting_instructions.append({"type": "literal", "value": "array"})

  axis_label_instructions = []

  if has_name_only:
    formatting_instructions.append({"type": "literal", "value": "[{"})

    first = True
    for data_axis, axis_info in enumerate(array_axis_info):
      if not isinstance(axis_info, NamedPositionlessAxisInfo):
        continue

      if first:
        formatting_instructions.append(
            {"type": "literal", "value": f"{repr(axis_info.axis_name)}:"}
        )
        first = False
      else:
        formatting_instructions.append(
            {"type": "literal", "value": f", {repr(axis_info.axis_name)}:"}
        )

      formatting_instructions.append({
          "type": "index",
          "axis": f"a{data_axis}",
          "skip_start": skip_start_indices[data_axis],
          "skip_end": skip_end_indices[data_axis],
      })

      if axis_info in axis_labels_by_info:
        data_axis_labels[data_axis] = axis_labels_by_info[axis_info]
        label_name = f"{axis_labels_by_info[axis_info]} ({axis_info.axis_name})"
      elif axis_info in slider_infos:
        label_name = f"{str(axis_info.axis_name)}"
        data_axis_labels[data_axis] = label_name
      else:
        label_name = f"{str(axis_info.axis_name)}"
        data_axis_labels[data_axis] = f"{label_name}: {axis_info.size}"

      if axis_info in axis_item_labels_by_info:
        axis_label_instructions.extend([
            {"type": "literal", "value": f"\n{label_name} @ "},
            {
                "type": "index",
                "axis": f"a{data_axis}",
                "skip_start": skip_start_indices[data_axis],
                "skip_end": skip_end_indices[data_axis],
            },
            {"type": "literal", "value": ": "},
            {
                "type": "axis_lookup",
                "axis": f"a{data_axis}",
                "skip_start": skip_start_indices[data_axis],
                "skip_end": skip_end_indices[data_axis],
                "lookup_table": axis_item_labels_by_info[axis_info],
            },
        ])

    formatting_instructions.append({"type": "literal", "value": "}]"})

  if positional_count:
    formatting_instructions.append({"type": "literal", "value": "["})
    for logical_index in range(positional_count):
      axis_info = info_by_name_or_position[logical_index]
      assert isinstance(axis_info, PositionalAxisInfo | NamedPositionalAxisInfo)
      assert axis_info.axis_logical_index == logical_index
      data_axis = data_axis_from_axis_info[axis_info]
      if logical_index > 0:
        formatting_instructions.append({"type": "literal", "value": ", "})
      formatting_instructions.append({
          "type": "index",
          "axis": f"a{data_axis}",
          "skip_start": skip_start_indices[data_axis],
          "skip_end": skip_end_indices[data_axis],
      })

      if axis_info in axis_labels_by_info:
        data_axis_labels[data_axis] = axis_labels_by_info[axis_info]
        label_name = f"{axis_labels_by_info[axis_info]} (axis {logical_index})"
      else:
        if isinstance(axis_info, NamedPositionalAxisInfo):
          label_name = f"{axis_info.axis_name} (axis {logical_index})"
        else:
          label_name = f"axis {logical_index}"
        if axis_info in slider_infos:
          data_axis_labels[data_axis] = label_name
        else:
          data_axis_labels[data_axis] = f"{label_name}: {axis_info.size}"

      if axis_info in axis_item_labels_by_info:
        axis_label_instructions.extend([
            {"type": "literal", "value": f"\n{label_name} @ "},
            {
                "type": "index",
                "axis": f"a{data_axis}",
                "skip_start": skip_start_indices[data_axis],
                "skip_end": skip_end_indices[data_axis],
            },
            {"type": "literal", "value": ": "},
            {
                "type": "axis_lookup",
                "axis": f"a{data_axis}",
                "skip_start": skip_start_indices[data_axis],
                "skip_end": skip_end_indices[data_axis],
                "lookup_table": axis_item_labels_by_info[axis_info],
            },
        ])

    formatting_instructions.append({"type": "literal", "value": "]"})

  formatting_instructions.append({"type": "literal", "value": "\n  = "})
  formatting_instructions.append({"type": "value"})

  # Step 7: Infer the colormap and rendering strategy.

  # Figure out whether the array is continuous.
  inferred_continuous = dtype_util.is_floating_dtype(truncated_array_data.dtype)
  if continuous == "auto":
    continuous = inferred_continuous
  elif not continuous and inferred_continuous:
    raise ValueError(
        "Cannot use continuous=False when rendering a float array; explicitly"
        " cast it to an integer array first."
    )

  if inferred_continuous:
    # Cast to float32 to ensure we can easily manipulate the truncated data.
    truncated_array_data = truncated_array_data.astype(np.float32)

  if value_item_labels and not continuous:
    formatting_instructions.append({"type": "literal", "value": "  # "})
    formatting_instructions.append(
        {"type": "value_lookup", "lookup_table": value_item_labels}
    )

  formatting_instructions.extend(axis_label_instructions)

  # Figure out centering.
  definitely_not_around_zero = (
      vmin is not None and vmax is not None and vmin != -vmax  # pylint: disable=invalid-unary-operand-type
  )
  if around_zero == "auto":
    around_zero = not definitely_not_around_zero
  elif around_zero and definitely_not_around_zero:
    raise ValueError(
        "Cannot use around_zero=True while also specifying both vmin and vmax"
    )

  # Check whether we should dynamically adjust the colormap.
  if dynamic_colormap == "auto":
    dynamic_colormap = (
        continuous and around_zero and vmin is None and vmax is None
    )

  if dynamic_colormap:
    if not continuous:
      raise ValueError(
          "Cannot use dynamic_colormap with a non-continuous colormap."
      )
    if not around_zero:
      raise ValueError("Cannot use dynamic_colormap without around_zero.")

    raw_min_abs, raw_max_abs = arrayviz_impl.infer_abs_min_max(
        truncated_array_data, truncated_mask_data
    )
    raw_min_abs = float(raw_min_abs)
    raw_max_abs = float(raw_max_abs)
  else:
    raw_min_abs = None
    raw_max_abs = None

  # Infer concrete `vmin` and `vmax`.
  if continuous and (vmin is None or vmax is None):
    vmin, vmax = arrayviz_impl.infer_vmin_vmax(
        array=truncated_array_data,
        mask=truncated_mask_data,
        vmin=vmin,
        vmax=vmax,
        around_zero=around_zero,
        trim_outliers=trim_outliers,
    )
    vmin = float(vmin)
    vmax = float(vmax)

  # Figure out which colormap and rendering strategy to use.
  if colormap is None:
    if continuous:
      colormap_type = "continuous"
      if around_zero:
        colormap_data = default_diverging_colormap.get()
      else:
        colormap_data = default_sequential_colormap.get()

    else:
      colormap_type = "digitbox"
      colormap_data = []

  elif continuous:
    colormap_data = colormap
    colormap_type = "continuous"

  else:
    colormap_data = colormap
    colormap_type = "palette_index"

  # Make a title for it
  info_parts = []
  if dynamic_colormap:
    info_parts.append("Dynamic colormap (click to adjust).")
  elif continuous:
    info_parts.append(f"Linear colormap from {vmin:.6g} to {vmax:.6g}.")
  elif colormap is not None:
    info_parts.append("Indexed colors from a color list.")
  else:
    info_parts.append("Showing integer digits as nested squares.")

  info_parts.append(" Hover/click for array data.")

  # Step 8: Render it!
  html_src = arrayviz_impl.render_array_data_to_html(
      array_data=truncated_array_data,
      valid_mask=truncated_mask_data,
      column_axes=column_data_axes,
      row_axes=row_data_axes,
      slider_axes=slider_data_axes,
      axis_labels=[
          data_axis_labels[i] for i in range(truncated_array_data.ndim)
      ],
      vmin=vmin,
      vmax=vmax,
      cmap_type=colormap_type,
      cmap_data=colormap_data,
      info="".join(info_parts),
      formatting_instructions=formatting_instructions,
      dynamic_continuous_cmap=dynamic_colormap,
      raw_min_abs=raw_min_abs,
      raw_max_abs=raw_max_abs,
      pixels_per_cell=pixels_per_cell,
  )
  return arrayviz_impl.ArrayvizRendering(html_src)


def render_sharding_info(
    array_axis_info: Sequence[AxisInfo],
    sharding_info: ndarray_adapters.ShardingInfo,
    rows: Sequence[int | AxisName] = (),
    columns: Sequence[int | AxisName] = (),
) -> figures_impl.TreescopeFigure:
  """Renders the sharding of an array.

  This is a helper function for rendering array shardings. It can be used either
  to render the sharding of an actual array or of a hypothetical array of a
  given shape and sharding.

  Args:
    array_axis_info: Axis info for each axis of the array data.
    sharding_info: Sharding info for the array, as produced by a NDArrayAdapter.
    rows: Optional explicit ordering of rows in the visualization.
    columns: Optional explicit ordering of columns in the visualization.

  Returns:
    A rendering of the sharding, which re-uses the digitbox rendering mode to
    render sets of devices.
  """
  data_axis_from_axis_info = {
      info: axis for axis, info in enumerate(array_axis_info)
  }

  info_by_name_or_position = {}
  has_name_only = False
  positional_count = 0
  for info in array_axis_info:
    if isinstance(info, NamedPositionalAxisInfo):
      info_by_name_or_position[info.axis_name] = info
      info_by_name_or_position[info.axis_logical_index] = info
      positional_count += 1
    elif isinstance(info, PositionalAxisInfo):
      info_by_name_or_position[info.axis_logical_index] = info
      positional_count += 1
    elif isinstance(info, NamedPositionlessAxisInfo):
      info_by_name_or_position[info.axis_name] = info
      has_name_only = True
    else:
      raise ValueError(f"Unrecognized axis info {type(info)}")

  array_shape = [info.size for info in array_axis_info]
  orig_shard_shape = sharding_info.shard_shape
  num_shards = np.prod(array_shape) // np.prod(orig_shard_shape)
  orig_device_indices_map = sharding_info.device_index_to_shard_slices
  # Possibly adjust the shard shape so that its length is the same as the array
  # shape, and so that all items in device_indices_map are slices.
  device_indices_map = {}
  shard_shape = []
  orig_shard_shape_index = 0
  first = True
  for key, ints_or_slices in orig_device_indices_map.items():
    new_slices = []
    for i, int_or_slc in enumerate(ints_or_slices):
      if isinstance(int_or_slc, int):
        new_slices.append(slice(int_or_slc, int_or_slc + 1))
        if first:
          shard_shape.append(1)
      elif isinstance(int_or_slc, slice):
        new_slices.append(int_or_slc)
        if first:
          shard_shape.append(orig_shard_shape[orig_shard_shape_index])
          orig_shard_shape_index += 1
      else:
        raise ValueError(
            f"Unrecognized axis slice in sharding info: {int_or_slc} at index"
            f" {i} for device {key}"
        )
    device_indices_map[key] = tuple(new_slices)
    first = False

  assert len(shard_shape) == len(array_shape)
  assert orig_shard_shape_index == len(orig_shard_shape)
  # Compute a truncation for visualizing a single shard. Each shard will be
  # shown as a shrunken version of the actual shard dimensions, roughly
  # proportional to the shard sizes.
  mini_trunc = arrayviz_impl.infer_balanced_truncation(
      shape=array_shape,
      maximum_size=1000,
      cutoff_size_per_axis=10,
      minimum_edge_items=2,
      doubling_bonus=5,
  )
  # Infer an axis ordering.
  known_row_infos = [info_by_name_or_position[spec] for spec in rows]
  known_column_infos = [info_by_name_or_position[spec] for spec in columns]
  row_infos, column_infos = arrayviz_impl.infer_rows_and_columns(
      all_axes=array_axis_info,
      known_rows=known_row_infos,
      known_columns=known_column_infos,
      edge_items_per_axis=mini_trunc,
  )
  # Build an actual matrix to represent each shard, with a size determined by
  # the inferred truncation.
  shard_mask = np.ones((), dtype=np.bool_)
  for t, sh_s, arr_s in zip(mini_trunc, shard_shape, array_shape):
    if t is None or sh_s <= 5:
      vec = np.ones((sh_s,), dtype=np.bool_)
    else:
      candidate = t // (arr_s // sh_s)
      if candidate <= 2:
        vec = np.array([True] * 2 + [False] + [True] * 2)
      else:
        vec = np.array([True] * candidate + [False] + [True] * candidate)
    shard_mask = shard_mask[..., None] * vec
  # Figure out which device is responsible for each shard.
  device_to_shard_offsets = {}
  shard_offsets_to_devices = collections.defaultdict(list)
  for device_index, slices in device_indices_map.items():
    shard_offsets = []
    for i, slc in enumerate(slices):
      assert slc.step is None
      if slc.start is None:
        assert slc.stop is None
        shard_offsets.append(0)
      else:
        assert slc.stop == slc.start + shard_shape[i]
        shard_offsets.append(slc.start // shard_shape[i])

    shard_offsets = tuple(shard_offsets)
    device_to_shard_offsets[device_index] = shard_offsets
    shard_offsets_to_devices[shard_offsets].append(device_index)
  # Figure out what value to show for each shard. This determines the
  # visualization color.
  shard_offset_values = {}
  shard_value_descriptions = {}
  if len(device_indices_map) <= 10 and all(
      device_index < 10 for device_index in device_indices_map.keys()
  ):
    # Map each device to an integer digit 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, and
    # then draw replicas as collections of base-10 digits.
    for shard_offsets, shard_devices in shard_offsets_to_devices.items():
      if len(shard_devices) >= 7:
        # All devices are in the same shard! Arrayviz only supports 7 digits at
        # a time, so draw as much as we can.
        assert num_shards == 1
        vis_value = 1234567
      else:
        acc = 0
        for i, device_index in enumerate(shard_devices):
          acc += 10 ** (len(shard_devices) - i - 1) * (device_index + 1)
        vis_value = acc
      shard_offset_values[shard_offsets] = vis_value
      assert vis_value not in shard_value_descriptions
      shard_value_descriptions[vis_value] = (
          sharding_info.device_type
          + " "
          + ",".join(f"{d}" for d in shard_devices)
      )
    render_info_message = "Colored by device index."
  elif num_shards < 10:
    # More than ten devices, less than ten shards. Give each shard its own
    # index but start at 1.
    shard_offset_values = {
        shard_offsets: i + 1
        for i, shard_offsets in enumerate(shard_offsets_to_devices.keys())
    }
    render_info_message = "With a distinct pattern for each shard."
  else:
    # A large number of devices and shards. Start at 0.
    shard_offset_values = {
        shard_offsets: i
        for i, shard_offsets in enumerate(shard_offsets_to_devices.keys())
    }
    render_info_message = "With a distinct pattern for each shard."
  # Build the sharding visualization array.
  viz_shape = tuple(
      shard_mask.shape[i] * array_shape[i] // shard_shape[i]
      for i in range(len(array_shape))
  )
  dest = np.zeros(viz_shape, dtype=np.int32)
  destmask = np.empty(viz_shape, dtype=np.int32)
  shard_labels_by_vis_pos = [
      ["????" for _ in range(viz_shape[i])] for i in range(len(viz_shape))
  ]
  for shard_offsets, value in shard_offset_values.items():
    indexers = []
    for i, offset in enumerate(shard_offsets):
      vizslc = slice(
          offset * shard_mask.shape[i],
          (offset + 1) * shard_mask.shape[i],
          None,
      )
      indexers.append(vizslc)
      label = f"{offset * shard_shape[i]}:{(offset + 1) * shard_shape[i]}"
      for j in range(viz_shape[i])[vizslc]:
        shard_labels_by_vis_pos[i][j] = label
    dest[tuple(indexers)] = np.full_like(shard_mask, value, dtype=np.int32)
    destmask[tuple(indexers)] = shard_mask
  # Create formatting instructions to show what devices are in each shard.
  axis_lookups = [
      {
          "type": "axis_lookup",
          "axis": f"a{data_axis}",
          "skip_start": viz_shape[data_axis],
          "skip_end": viz_shape[data_axis],
          "lookup_table": {
              j: str(v)
              for j, v in enumerate(shard_labels_by_vis_pos[data_axis])
          },
      }
      for data_axis in range(len(array_shape))
  ]
  data_axis_labels = {}
  formatting_instructions = []
  formatting_instructions.append({"type": "literal", "value": "array"})

  if has_name_only:
    formatting_instructions.append({"type": "literal", "value": "[{"})

    first = True
    for data_axis, axis_info in enumerate(array_axis_info):
      if not isinstance(axis_info, NamedPositionlessAxisInfo):
        continue

      if first:
        formatting_instructions.append(
            {"type": "literal", "value": f"{repr(axis_info.axis_name)}:["}
        )
        first = False
      else:
        formatting_instructions.append(
            {"type": "literal", "value": f", {repr(axis_info.axis_name)}:["}
        )

      formatting_instructions.append(axis_lookups[data_axis])
      formatting_instructions.append({"type": "literal", "value": "]"})
      axshards = array_shape[data_axis] // shard_shape[data_axis]
      data_axis_labels[data_axis] = (
          f"{axis_info.axis_name}: {array_shape[data_axis]}/{axshards}"
      )
    formatting_instructions.append({"type": "literal", "value": "}]"})

  if positional_count:
    formatting_instructions.append({"type": "literal", "value": "["})
    for logical_index in range(positional_count):
      axis_info = info_by_name_or_position[logical_index]
      data_axis = data_axis_from_axis_info[axis_info]
      if logical_index:
        formatting_instructions.append({"type": "literal", "value": ", "})
      formatting_instructions.append(axis_lookups[data_axis])
      axshards = array_shape[data_axis] // shard_shape[data_axis]
      data_axis_labels[data_axis] = (
          f"axis {logical_index}: {array_shape[data_axis]}/{axshards}"
      )
    formatting_instructions.append({"type": "literal", "value": "]"})

  formatting_instructions.append({"type": "literal", "value": ":\n  "})
  formatting_instructions.append({
      "type": "value_lookup",
      "lookup_table": shard_value_descriptions,
      "ignore_invalid": True,
  })
  # Build the rendering.
  html_srcs = []
  html_srcs.append(
      arrayviz_impl.render_array_data_to_html(
          array_data=dest,
          valid_mask=destmask,
          column_axes=[data_axis_from_axis_info[c] for c in column_infos],
          row_axes=[data_axis_from_axis_info[r] for r in row_infos],
          slider_axes=(),
          axis_labels=[data_axis_labels[i] for i in range(len(array_shape))],
          vmin=0,
          vmax=0,
          cmap_type="digitbox",
          cmap_data=[],
          info=render_info_message,
          formatting_instructions=formatting_instructions,
          dynamic_continuous_cmap=False,
          raw_min_abs=0.0,
          raw_max_abs=0.0,
      )
  )
  html_srcs.append('<span style="font-family: monospace; white-space: pre">')
  for i, (shard_offsets, shard_devices) in enumerate(
      shard_offsets_to_devices.items()
  ):
    if i == 0:
      html_srcs.append(f"{sharding_info.device_type}")
    label = ",".join(f"{d}" for d in shard_devices)
    part = integer_digitbox(shard_offset_values[shard_offsets]).treescope_part
    assert isinstance(part, arrayviz_impl.ArrayvizDigitboxRendering)
    html_srcs.append(f"  {part.html_src} {label}")
  html_srcs.append("</span>")
  return figures_impl.TreescopeFigure(
      arrayviz_impl.ArrayvizRendering("".join(html_srcs))
  )


def render_array_sharding(
    array: ArrayInRegistry,
    rows: Sequence[int | AxisName] = (),
    columns: Sequence[int | AxisName] = (),
) -> figures_impl.TreescopeFigure:
  """Renders the sharding of an array.

  Args:
    array: The array whose sharding we should render.
    rows: Optional explicit ordering of axes for the visualization rows.
    columns: Optional explicit ordering of axes for the visualization columns.

  Returns:
    A rendering of that array's sharding.
  """
  # Retrieve the adapter for this array, which we will use to construct
  # the rendering.
  type_registries.update_registries_for_imports()
  adapter = type_registries.lookup_ndarray_adapter(array)
  if adapter is None:
    raise TypeError(
        "Cannot render sharding for array with unrecognized type"
        f" {type(array)} (not found in array adapter registry)"
    )

  # Extract information about axis names, indices, and sizes, along with the
  # sharding info.
  array_axis_info = adapter.get_axis_info_for_array_data(array)
  sharding_info = adapter.get_sharding_info_for_array_data(array)
  if sharding_info is None:
    raise ValueError(
        "Cannot render sharding for array without sharding info (not provided"
        f" by array adapter for {type(array)})."
    )

  return render_sharding_info(
      array_axis_info=array_axis_info,
      sharding_info=sharding_info,
      rows=rows,
      columns=columns,
  )


def integer_digitbox(
    value: int, *, label: str | None = None, size: str = "1em"
) -> figures_impl.TreescopeFigure:
  """Returns a "digitbox" rendering of a single integer.

  Args:
    value: Integer value to render.
    label: Optional label to draw next to the digitbox.
    size: Size for the rendering as a CSS length. "1em" means render it at the
      current font size.

  Returns:
    A renderable object showing the digitbox rendering for this integer.
  """
  value = int(value)

  render_args = json.dumps({"value": value})
  size_attr = html_escaping.escape_html_attribute(size)
  # Note: We need to save the parent of the treescope-run-here element first,
  # because it will be removed before the runSoon callback executes.
  src = (
      f'<span class="inline_digitbox" style="font-size: {size_attr}">'
      '<treescope-run-here><script type="application/octet-stream">'
      "const parent = this.parentNode;"
      "const defns = this.getRootNode().host.defns;"
      "defns.runSoon(() => {"
      f"defns.arrayviz.renderOneDigitbox(parent, {render_args});"
      "});"
      "</script></treescope-run-here>"
      "</span>"
  )
  rendering = arrayviz_impl.ArrayvizDigitboxRendering(src)
  if label:
    return figures_impl.TreescopeFigure(
        basic_parts.siblings(
            rendering,
            common_styles.custom_style(
                basic_parts.text(f" {label}"), "color:gray; font-size: 0.5em"
            ),
        )
    )
  else:
    return figures_impl.TreescopeFigure(rendering)
