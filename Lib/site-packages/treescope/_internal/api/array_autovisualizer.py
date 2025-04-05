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

"""An automatic NDArray visualizer using arrayviz."""
from __future__ import annotations

import dataclasses
import sys
from typing import Any, Callable, Collection

import numpy as np
from treescope import dtype_util
from treescope import lowering
from treescope import ndarray_adapters
from treescope import rendering_parts
from treescope import type_registries
from treescope._internal import arrayviz_impl
from treescope._internal.api import arrayviz
from treescope._internal.api import autovisualize


PositionalAxisInfo = ndarray_adapters.PositionalAxisInfo
NamedPositionlessAxisInfo = ndarray_adapters.NamedPositionlessAxisInfo
NamedPositionalAxisInfo = ndarray_adapters.NamedPositionalAxisInfo
AxisInfo = ndarray_adapters.AxisInfo

ArrayInRegistry = Any


def _supported_dtype(dtype: np.dtype | None):
  return dtype is not None and (
      dtype_util.is_integer_dtype(dtype)
      or dtype_util.is_floating_dtype(dtype)
      or np.issubdtype(dtype, np.bool_)
  )


@dataclasses.dataclass
class ArrayAutovisualizer:
  """An automatic visualizer for arrays.

  ArrayAutovisualizer supports any array type registered with an NDArrayAdapter
  in the global type registry.

  Attributes:
    maximum_size: Maximum number of elements of an array to show. Arrays larger
      than this will be truncated along one or more axes.
    cutoff_size_per_axis: Maximum number of elements of each individual axis to
      show without truncation. Any axis longer than this will be truncated, with
      their visual size increasing logarithmically with the true axis size
      beyond this point.
    edge_items: How many values to keep along each axis for truncated arrays.
    prefers_column: Names that should always be assigned to columns.
    prefers_row: Names that should always be assigned to rows.
    around_zero: Whether to center continuous data around zero.
    force_continuous: Whether to always render integer arrays as continuous.
    include_repr_line_threshold: A threshold such that, if the `repr` of the
      array has fewer than that many lines, we will include that repr in the
      visualization. Useful for seeing small array values.
    token_lookup_fn: Optional function that looks up token IDs and adds them to
      the visualization on hover.
    pixels_per_cell: Size of each rendered array element in pixels, between 1
      and 21 inclusive. This controls the zoom level of the rendering. Array
      elements are always drawn at 7 pixels per cell and then rescaled, so
      out-of-bounds annotations and "digitbox" integer value patterns may not
      display correctly at fewer than 7 pixels per cell.
  """

  maximum_size: int = 4_000
  cutoff_size_per_axis: int = 128
  edge_items: int = 5
  prefers_column: Collection[str] = ()
  prefers_row: Collection[str] = ()
  around_zero: bool = True
  force_continuous: bool = False
  include_repr_line_threshold: int = 5
  token_lookup_fn: Callable[[int], str] | None = None
  pixels_per_cell: int | float = 7

  def _autovisualize_array(
      self,
      array: ArrayInRegistry,
      adapter: ndarray_adapters.NDArrayAdapter,
      path: str | None,
      label: rendering_parts.RenderableTreePart,
      expand_state: rendering_parts.ExpandState,
  ) -> rendering_parts.RenderableTreePart:
    """Helper to visualize an array."""
    # Extract information about axis names, indices, and sizes.
    array_axis_info = adapter.get_axis_info_for_array_data(array)

    # Assign axes, using preferred axes if possible.
    row_axes = []
    column_axes = []
    for info in array_axis_info:
      if isinstance(info, NamedPositionalAxisInfo | NamedPositionlessAxisInfo):
        if info.axis_name in self.prefers_column:
          column_axes.append(info)
        elif info.axis_name in self.prefers_row:
          row_axes.append(info)

    # Infer a good truncated shape for this array.
    edge_items_per_axis = arrayviz_impl.infer_balanced_truncation(
        tuple(info.size for info in array_axis_info),
        maximum_size=self.maximum_size,
        cutoff_size_per_axis=self.cutoff_size_per_axis,
        minimum_edge_items=self.edge_items,
    )

    row_axes, column_axes = arrayviz_impl.infer_rows_and_columns(
        all_axes=array_axis_info,
        known_rows=row_axes,
        known_columns=column_axes,
        edge_items_per_axis=edge_items_per_axis,
    )

    # Obtain truncated array and mask data from the adapter.
    truncated_array_data, truncated_mask_data = (
        adapter.get_array_data_with_truncation(
            array=array,
            mask=None,
            edge_items_per_axis=edge_items_per_axis,
        )
    )

    # Maybe infer value labels from a tokenizer.
    if (
        self.token_lookup_fn
        and not self.force_continuous
        and dtype_util.is_integer_dtype(truncated_array_data.dtype)
    ):
      tokens = np.unique(truncated_array_data.flatten()).tolist()
      value_item_labels = {
          token: self.token_lookup_fn(token) for token in tokens
      }
    else:
      value_item_labels = None

    array_rendering = arrayviz._render_pretruncated(  # pylint: disable=protected-access
        array_axis_info=array_axis_info,
        row_infos=row_axes,
        column_infos=column_axes,
        slider_infos=(),
        truncated_array_data=truncated_array_data,
        truncated_mask_data=truncated_mask_data,
        edge_items_per_axis=edge_items_per_axis,
        continuous="auto",
        around_zero=self.around_zero,
        vmax=None,
        vmin=None,
        trim_outliers=True,
        dynamic_colormap="auto",
        colormap=None,
        axis_item_labels=None,
        value_item_labels=value_item_labels,
        axis_labels=None,
        pixels_per_cell=self.pixels_per_cell,
    )
    outputs = [array_rendering]
    last_line_parts = []

    # Render the sharding as well.
    sharding_info = adapter.get_sharding_info_for_array_data(array)
    if sharding_info:
      num_devices = len(sharding_info.device_index_to_shard_slices)
      if num_devices == 1:
        [device_id] = sharding_info.device_index_to_shard_slices.keys()
        if device_id == -1:
          last_line_parts.append(f"| Device: {sharding_info.device_type}")
        else:
          last_line_parts.append(
              f"| Device: {sharding_info.device_type} {device_id}"
          )
      else:
        if sharding_info.fully_replicated:
          sharding_summary_str = (
              "Replicated across"
              f" {num_devices} {sharding_info.device_type} devices"
          )
        else:
          sharding_summary_str = (
              "Sharded across"
              f" {num_devices} {sharding_info.device_type} devices"
          )
        sharding_rendering = arrayviz.render_array_sharding(
            array,
            columns=[c.logical_key() for c in column_axes],
            rows=[r.logical_key() for r in row_axes],
        ).treescope_part
        outputs.append(
            rendering_parts.build_custom_foldable_tree_node(
                label=rendering_parts.abbreviation_color(
                    rendering_parts.siblings(
                        rendering_parts.text(sharding_summary_str),
                        rendering_parts.fold_condition(
                            expanded=rendering_parts.text(":"),
                            collapsed=rendering_parts.text(
                                " (click to expand)"
                            ),
                        ),
                    )
                ),
                contents=rendering_parts.fold_condition(
                    expanded=rendering_parts.indented_children(
                        [sharding_rendering]
                    ),
                ),
            )
        )

    # We render it with a path, but remove the copy path button. This will be
    # added back by the caller.
    if last_line_parts:
      last_line = rendering_parts.siblings(
          rendering_parts.fold_condition(
              expanded=rendering_parts.text("".join(last_line_parts)),
          ),
          rendering_parts.text(">"),
      )
    else:
      last_line = rendering_parts.text(">")
    custom_rendering = rendering_parts.build_custom_foldable_tree_node(
        label=rendering_parts.abbreviation_color(label),
        contents=rendering_parts.siblings(
            rendering_parts.fold_condition(
                expanded=rendering_parts.indented_children(outputs)
            ),
            rendering_parts.abbreviation_color(last_line),
        ),
        path=path,
        expand_state=expand_state,
    )
    return custom_rendering.renderable

  def __call__(
      self, value: Any, path: str | None
  ) -> autovisualize.VisualizationFromTreescopePart | None:
    """Implementation of an autovisualizer, visualizing arrays."""
    # Retrieve the adapter for this array, which we will use to construct
    # the rendering.
    adapter = type_registries.lookup_ndarray_adapter(value)
    if adapter is not None:
      if not adapter.should_autovisualize(value):
        return None

      # This is an array we can visualize!
      # Extract information about axis names, indices, and sizes.
      array_axis_info = adapter.get_axis_info_for_array_data(value)
      total_size = np.prod([ax.size for ax in array_axis_info])
      if total_size == 1:
        # Don't visualize scalars.
        return None

      np_dtype = adapter.get_numpy_dtype(value)
      if not _supported_dtype(np_dtype):
        return None

      def _placeholder() -> rendering_parts.RenderableTreePart:
        summary = adapter.get_array_summary(value, fast=True)
        return rendering_parts.deferred_placeholder_style(
            rendering_parts.siblings(
                rendering_parts.text("<"), summary, rendering_parts.text(">")
            )
        )

      def _thunk(
          expand_state: rendering_parts.ExpandState | None,
      ) -> rendering_parts.RenderableTreePart:
        # Full rendering of the array.
        if expand_state is None:
          expand_state = rendering_parts.ExpandState.WEAKLY_EXPANDED
        summary = adapter.get_array_summary(value, fast=False)
        label = rendering_parts.siblings(rendering_parts.text("<"), summary)

        return self._autovisualize_array(
            value, adapter, path, label, expand_state
        )

      return autovisualize.VisualizationFromTreescopePart(
          rendering_parts.RenderableAndLineAnnotations(
              renderable=lowering.maybe_defer_rendering(
                  _thunk, _placeholder, expanded_newlines_for_layout=8
              ),
              annotations=rendering_parts.build_copy_button(path),
          )
      )

    # Not an array in the registry. But it might be a JAX sharding that we can
    # visualize (if JAX is imported).
    if "jax" in sys.modules:
      import jax  # pylint: disable=import-outside-toplevel

      if isinstance(
          value,
          jax.sharding.PositionalSharding
          | jax.sharding.NamedSharding
          | jax.sharding.Mesh,
      ):
        raw_repr = repr(value)
        repr_oneline = " ".join(line.strip() for line in raw_repr.split("\n"))

        if isinstance(value, jax.sharding.PositionalSharding):
          sharding = value
          fake_axis_info = [
              PositionalAxisInfo(i, size)
              for i, size in enumerate(sharding.shape)
          ]
        elif isinstance(value, jax.sharding.NamedSharding):
          sharding = value
          # Named shardings still act on positional arrays, so show them for
          # the positional shape they require.
          fake_sizes = []
          for part in value.spec:
            if part is None:
              fake_sizes.append(1)
            elif isinstance(part, str):
              fake_sizes.append(value.mesh.shape[part])
            else:
              size = int(np.prod([value.mesh.shape[a] for a in part]))
              fake_sizes.append(size)
          fake_axis_info = [
              PositionalAxisInfo(i, size) for i, size in enumerate(fake_sizes)
          ]
        else:
          # Meshes are based on named axes. We build a temporary positional
          # sharding for visualization, but keep track of name order.
          assert isinstance(value, jax.sharding.Mesh)
          mesh = value
          sharding = jax.sharding.NamedSharding(
              value, jax.sharding.PartitionSpec(*mesh.axis_names)
          )
          fake_axis_info = [
              NamedPositionlessAxisInfo(name, mesh.shape[name])
              for name in mesh.axis_names
          ]

        fake_shape = tuple(ax.size for ax in fake_axis_info)
        some_device = next(iter(sharding.device_set))
        device_index_map = sharding.devices_indices_map(fake_shape)
        sharding_info = ndarray_adapters.ShardingInfo(
            shard_shape=sharding.shard_shape(fake_shape),
            device_index_to_shard_slices={
                d.id: v for d, v in device_index_map.items()
            },
            device_type=some_device.platform.upper(),
            fully_replicated=sharding.is_fully_replicated,
        )
        shardvis = arrayviz.render_sharding_info(
            array_axis_info=fake_axis_info,
            sharding_info=sharding_info,
        ).treescope_part
        custom_rendering = rendering_parts.build_custom_foldable_tree_node(
            label=rendering_parts.abbreviation_color(
                rendering_parts.text("<" + repr_oneline)
            ),
            contents=rendering_parts.siblings(
                rendering_parts.fold_condition(
                    expanded=rendering_parts.indented_children([shardvis])
                ),
                rendering_parts.abbreviation_color(rendering_parts.text(">")),
            ),
            path=path,
            expand_state=rendering_parts.ExpandState.EXPANDED,
        )
        return autovisualize.VisualizationFromTreescopePart(custom_rendering)
      else:
        return None

  @classmethod
  def for_tokenizer(cls, tokenizer: Any):
    """Builds an autovisualizer for a tokenizer.

    This method constructs an ArrayAutovisualizer that annotates integer array
    elements with their token strings. This can then be used to autovisualize
    tokenized arrays.

    Args:
      tokenizer: A tokenizer to use. Either a callable mapping token IDs to
        strings, or a SentencePieceProcessor.

    Returns:
      An ArrayAutovisualizer that annotates integer array elements with their
      token strings.
    """
    if callable(tokenizer):
      return cls(token_lookup_fn=lambda x: repr(tokenizer(x)))
    elif hasattr(tokenizer, "IdToPiece") and hasattr(tokenizer, "GetPieceSize"):

      def lookup(x):
        if x >= 0 and x < tokenizer.GetPieceSize():
          return repr(tokenizer.IdToPiece(x))
        else:
          return f"<out of bounds: {x}>"

      return cls(token_lookup_fn=lookup)
    else:
      raise ValueError(f"Unknown tokenizer type: {tokenizer}")
