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

"""Lazy setup logic for adding Numpy support to treescope."""
from __future__ import annotations

import re
from typing import Any
import warnings

import numpy as np
from treescope import canonical_aliases
from treescope import dtype_util
from treescope import lowering
from treescope import ndarray_adapters
from treescope import renderers
from treescope import rendering_parts
from treescope import type_registries


def _truncate_and_copy(
    array_source: np.ndarray,
    array_dest: np.ndarray,
    prefix_slices: tuple[slice, ...],
    remaining_edge_items_per_axis: tuple[int | None, ...],
) -> None:
  """Recursively copy values along the edges of a source into a destination.

  This function mutates the destination array in place, copying parts of input
  array into them, so that it contains a truncated versions of the original
  array.

  Args:
    array_source: Source array, which we will truncate.
    array_dest: Destination array, whose axis sizes will be either the same as
      `array_source` or of size `2 * edge_items + 1` depending on the
      truncation.
    prefix_slices: Prefix of slices for the source and destination.
    remaining_edge_items_per_axis: Number of edge items to keep for each axis,
      ignoring any axes whose slices are already computed in `source_slices`.
  """
  if not remaining_edge_items_per_axis:
    # Perform the base case slice.
    assert (
        len(prefix_slices) == len(array_source.shape) == len(array_dest.shape)
    )
    array_dest[prefix_slices] = array_source[prefix_slices]
  else:
    # Recursive step.
    axis = len(prefix_slices)
    edge_items = remaining_edge_items_per_axis[0]
    if edge_items is None:
      # Don't need to slice.
      _truncate_and_copy(
          array_source=array_source,
          array_dest=array_dest,
          prefix_slices=prefix_slices + (slice(None),),
          remaining_edge_items_per_axis=remaining_edge_items_per_axis[1:],
      )
    else:
      assert array_source.shape[axis] > 2 * edge_items
      _truncate_and_copy(
          array_source=array_source,
          array_dest=array_dest,
          prefix_slices=prefix_slices + (slice(None, edge_items),),
          remaining_edge_items_per_axis=remaining_edge_items_per_axis[1:],
      )
      _truncate_and_copy(
          array_source=array_source,
          array_dest=array_dest,
          prefix_slices=prefix_slices + (slice(-edge_items, None),),
          remaining_edge_items_per_axis=remaining_edge_items_per_axis[1:],
      )


class NumpyArrayAdapter(ndarray_adapters.NDArrayAdapter[np.ndarray]):
  """NDArray adapter for numpy arrays."""

  def get_axis_info_for_array_data(
      self, array: np.ndarray
  ) -> tuple[ndarray_adapters.AxisInfo, ...]:
    return tuple(
        ndarray_adapters.PositionalAxisInfo(i, size)
        for i, size in enumerate(array.shape)
    )

  def get_array_data_with_truncation(
      self,
      array: np.ndarray,
      mask: np.ndarray | None,
      edge_items_per_axis: tuple[int | None, ...],
  ) -> tuple[np.ndarray, np.ndarray]:

    if mask is None:
      mask = np.ones((1,) * array.ndim, dtype=bool)

    # Broadcast mask. (Note: Broadcasting a Numpy array does not copy data.)
    mask = np.broadcast_to(mask, array.shape)

    if edge_items_per_axis == (None,) * array.ndim:
      # No truncation.
      return array, mask

    dest_shape = [
        size if edge_items is None else 2 * edge_items + 1
        for size, edge_items in zip(array.shape, edge_items_per_axis)
    ]
    array_dest = np.zeros(dest_shape, array.dtype)
    mask_dest = np.zeros(dest_shape, bool)
    _truncate_and_copy(
        array_source=array,
        array_dest=array_dest,
        prefix_slices=(),
        remaining_edge_items_per_axis=edge_items_per_axis,
    )
    _truncate_and_copy(
        array_source=mask,
        array_dest=mask_dest,
        prefix_slices=(),
        remaining_edge_items_per_axis=edge_items_per_axis,
    )
    return array_dest, mask_dest

  def get_array_summary(
      self, array: np.ndarray, fast: bool
  ) -> rendering_parts.RenderableTreePart:
    main_parts = [
        rendering_parts.abbreviatable(
            rendering_parts.text("np.ndarray "),
            rendering_parts.text("np "),
        )
    ]
    main_parts.append(dtype_util.get_dtype_name(array.dtype))
    main_parts.append(repr(array.shape))

    summary_parts = []

    if array.size > 0 and array.size < 100_000 and not fast:
      is_floating = dtype_util.is_floating_dtype(array.dtype)
      is_integer = dtype_util.is_integer_dtype(array.dtype)
      is_bool = np.issubdtype(array.dtype, np.bool_)

      if is_floating:
        isfinite = np.isfinite(array)
        any_finite = np.any(isfinite)
        inf_to_nan = np.where(
            isfinite, array, np.array(np.nan, dtype=array.dtype)
        )
        mean = np.nanmean(inf_to_nan)
        std = np.nanstd(inf_to_nan)

        if any_finite:
          summary_parts.append(f" ≈{float(mean):.2} ±{float(std):.2}")
          summary_parts.append(
              f" [≥{float(np.nanmin(array)):.2}, ≤{float(np.nanmax(array)):.2}]"
          )

      if is_integer:
        summary_parts.append(f" [≥{np.min(array):_d}, ≤{np.max(array):_d}]")

      if is_floating or is_integer:
        ct_zero = np.count_nonzero(array == 0)
        if ct_zero:
          summary_parts.append(f" zero:{ct_zero:_d}")

        ct_nonzero = np.count_nonzero(array)
        if ct_nonzero:
          summary_parts.append(f" nonzero:{ct_nonzero:_d}")

      if is_floating:
        ct_nan = np.count_nonzero(np.isnan(array))
        if ct_nan:
          summary_parts.append(f" nan:{ct_nan:_d}")

        ct_inf = np.count_nonzero(np.isposinf(array))
        if ct_inf:
          summary_parts.append(f" inf:{ct_inf:_d}")

        ct_neginf = np.count_nonzero(np.isneginf(array))
        if ct_neginf:
          summary_parts.append(f" -inf:{ct_neginf:_d}")

      if is_bool:
        ct_true = np.count_nonzero(array)
        if ct_true:
          summary_parts.append(f" true:{ct_true:_d}")

        ct_false = np.count_nonzero(np.logical_not(array))
        if ct_false:
          summary_parts.append(f" false:{ct_false:_d}")

    return rendering_parts.siblings(
        *main_parts,
        rendering_parts.abbreviatable(
            rendering_parts.siblings(*summary_parts),
            rendering_parts.empty_part(),
        ),
    )

  def get_numpy_dtype(self, array: np.ndarray) -> np.dtype:
    return array.dtype


def render_ndarrays(
    node: np.ndarray,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a numpy array."""
  del subtree_renderer
  assert isinstance(node, np.ndarray)
  adapter = NumpyArrayAdapter()

  def _placeholder() -> rendering_parts.RenderableTreePart:
    return rendering_parts.deferred_placeholder_style(
        adapter.get_array_summary(node, fast=True)
    )

  def _thunk(placeholder_expand_state: rendering_parts.ExpandState | None):
    # Is this array simple enough to render without a summary?
    node_repr = repr(node)
    if "\n" not in node_repr and "..." not in node_repr:
      rendering = rendering_parts.text(f"np.{node_repr}")
    else:
      if node_repr.count("\n") <= 15:
        if placeholder_expand_state is None:
          default_expand_state = rendering_parts.ExpandState.WEAKLY_EXPANDED
        else:
          default_expand_state = placeholder_expand_state
      else:
        # Always start big NDArrays in collapsed mode to hide irrelevant detail.
        default_expand_state = rendering_parts.ExpandState.COLLAPSED

      # Render it with a summary.
      summarized = adapter.get_array_summary(node, fast=False)
      rendering = rendering_parts.build_custom_foldable_tree_node(
          label=rendering_parts.abbreviation_color(
              rendering_parts.comment_color_when_expanded(
                  rendering_parts.siblings(
                      rendering_parts.fold_condition(
                          expanded=rendering_parts.text("# "),
                          collapsed=rendering_parts.text("<"),
                      ),
                      summarized,
                      rendering_parts.fold_condition(
                          collapsed=rendering_parts.text(">")
                      ),
                  )
              )
          ),
          contents=rendering_parts.fold_condition(
              expanded=rendering_parts.indented_children(
                  [rendering_parts.text(node_repr)]
              )
          ),
          path=path,
          expand_state=default_expand_state,
      ).renderable

    return rendering

  return rendering_parts.RenderableAndLineAnnotations(
      renderable=lowering.maybe_defer_rendering(
          main_thunk=_thunk, placeholder_thunk=_placeholder
      ),
      annotations=rendering_parts.build_copy_button(path),
  )


def render_dtype_instances(
    node: Any,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a np.dtype, adding the `np.` qualifier."""
  del subtree_renderer
  if not isinstance(node, np.dtype):
    return NotImplemented

  dtype_name = node.name
  if dtype_name in np.sctypeDict and node is np.dtype(
      np.sctypeDict[dtype_name]
  ):
    # Use the named type. (Sometimes extended dtypes don't print in a
    # roundtrippable way otherwise.)
    dtype_string = f"dtype({repr(dtype_name)})"
  else:
    # Hope that `repr` is already round-trippable (true for builtin numpy types)
    # and add the "numpy." prefix as needed.
    dtype_string = repr(node)

  return rendering_parts.build_one_line_tree_node(
      line=rendering_parts.siblings(
          rendering_parts.roundtrip_condition(
              roundtrip=rendering_parts.text("np.")
          ),
          dtype_string,
      ),
      path=path,
  )


def set_up_treescope():
  """Sets up treescope to render Numpy objects."""
  type_registries.NDARRAY_ADAPTER_REGISTRY[np.ndarray] = NumpyArrayAdapter()
  type_registries.TREESCOPE_HANDLER_REGISTRY[np.ndarray] = render_ndarrays
  type_registries.TREESCOPE_HANDLER_REGISTRY[np.dtype] = render_dtype_instances

  with warnings.catch_warnings():
    # This warning is triggered by walking the numpy API, but we are not
    # actually accessing anything under numpy.core while building aliases, so it
    # is safe to ignore temporarily.
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="treescope.canonical_aliases",
        message=re.escape(
            "numpy.core is deprecated and has been renamed to numpy._core."
        ),
    )
    canonical_aliases.populate_from_public_api(
        np, canonical_aliases.prefix_filter("numpy", excludes=("numpy.core",))
    )
