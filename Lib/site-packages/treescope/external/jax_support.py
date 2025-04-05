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

"""Lazy setup logic for adding JAX support to treescope."""

from __future__ import annotations

import functools
import typing
from typing import Any, Mapping, Sequence

import numpy as np
from treescope import canonical_aliases
from treescope import context
from treescope import dtype_util
from treescope import lowering
from treescope import ndarray_adapters
from treescope import renderers
from treescope import rendering_parts
from treescope import repr_lib
from treescope import type_registries

# pylint: disable=import-outside-toplevel
try:
  import jax
except ImportError:
  assert not typing.TYPE_CHECKING
  jax = None
# pylint: enable=import-outside-toplevel


def _is_subdtype(dtype, base) -> bool:
  """Safely checks for dtype subtyping."""
  assert jax is not None
  jnp = jax.numpy
  try:
    return jnp.issubdtype(dtype, base)
  except TypeError:
    return False


summarization_threshold: context.ContextualValue[Mapping[str, int | None]] = (
    context.ContextualValue(
        module=__name__,
        qualname="summarization_threshold",
        initial_value={
            "tpu": 1_000_000_000,
            "gpu": 10_000_000,
            "default": 100_000,
        },
    )
)
"""Threshold for summarization of NDArrays for each backend.

This threshold determines the largest number of elements we will
summarize with summary statistics (e.g. mean, standard deviation)
when rendering in treescope. Larger values may make it slower to
display large NDArrays.

Each key should be the name of a JAX array platform, e.g. "cpu" or
"tpu". It can also be "numpy" to refer to Numpy arrays, or "default"
to refer to any other accelerator. The value is the size of the
array at which point we avoid showing summary statistics. `None`
means no limit.

This configuration argument is intended to be set at the top level
by the user, e.g. in IPython.
"""


SUMMARIZE_USING_NUMPY_THRESHOLD = 10_000_000
"""Threshold for using NumPy to summarize and render JAX Arrays.

Moving arrays to main memory and using numpy is significantly faster because
it avoid jitting the summary and rendering functions.
"""


def _is_locally_available(array: jax.Array) -> bool:
  """Checks if the array is available locally."""
  return getattr(array, "is_fully_addressable", False) or getattr(
      array, "is_fully_replicated", False
  )


def safe_to_summarize(array: jax.Array) -> bool:
  """Checks if the array is safe to summarize (not a tracer, not replicated)."""
  assert jax is not None, "JAX is not available."
  if isinstance(array, jax.core.Tracer):
    return False
  if array.is_deleted():
    return False
  if not _is_locally_available(array):
    return False
  thresh_dict = summarization_threshold.get()
  [platform] = set(device.platform for device in array.devices())
  thresh = thresh_dict.get(platform)
  if thresh is None:
    thresh = thresh_dict["default"]
  return thresh is None or array.size < thresh


def _truncate_part_with_slices(
    array: jax.Array,
    mask: jax.Array,
    prefix_slices: tuple[slice, ...],
    remaining_edge_items_per_axis: tuple[int | None, ...],
    xnp=None,
) -> tuple[jax.Array, jax.Array]:
  """Helper to truncate names of an array.

  Args:
    array: An array to truncate.
    mask: Mask array, which must be broadcastable to `array`.
    prefix_slices: Slices to apply to each axis of `array` and `mask`, starting
      at axis 0, which we have already computed.
    remaining_edge_items_per_axis: Number of edge items to keep for each axis,
      ignoring any axes whose slices are already computed in `prefix_slices`.
    xnp: backend to use (numpy or jax.numpy).

  Returns:
    Truncated array and mask, which will both be the same shape.
  """
  if xnp is None:
    assert jax is not None, "JAX is not available."
    xnp = jax.numpy

  array = xnp.array(array)
  mask = xnp.array(mask)
  mask = xnp.broadcast_to(mask, array.shape)
  if not remaining_edge_items_per_axis:
    # Perform the base case slice.
    assert len(prefix_slices) == len(array.shape)
    truncated_array = array[prefix_slices]

    valid_mask_slices = tuple(
        slice(None) if mask.shape[i] == 1 else array_slice
        for i, array_slice in enumerate(prefix_slices)
    )
    truncated_mask = xnp.broadcast_to(
        xnp.array(mask[valid_mask_slices]), truncated_array.shape
    )
    return truncated_array, truncated_mask

  # Recursive step: extract one name, run the function on each side, and
  # concatenate.
  axis = len(prefix_slices)
  edge_items = remaining_edge_items_per_axis[0]
  if edge_items is None:
    # Don't need to slice.
    return _truncate_part_with_slices(
        array,
        mask,
        prefix_slices=prefix_slices + (slice(None),),
        remaining_edge_items_per_axis=remaining_edge_items_per_axis[1:],
        xnp=xnp,
    )
  else:
    assert array.shape[axis] > 2 * edge_items
    result_a, valid_a = _truncate_part_with_slices(
        array,
        mask,
        prefix_slices=prefix_slices + (slice(None, edge_items),),
        remaining_edge_items_per_axis=remaining_edge_items_per_axis[1:],
        xnp=xnp,
    )
    result_b, valid_b = _truncate_part_with_slices(
        array,
        mask,
        prefix_slices=prefix_slices + (slice(-edge_items, None),),
        remaining_edge_items_per_axis=remaining_edge_items_per_axis[1:],
        xnp=xnp,
    )
    padding_shape = list(result_a.shape)
    padding_shape[axis] = 1
    result = xnp.concatenate(
        [result_a, xnp.zeros(padding_shape, result_a.dtype), result_b],
        axis=axis,
    )
    valid = xnp.concatenate(
        [valid_a, xnp.zeros(padding_shape, valid_a.dtype), valid_b], axis=axis
    )
    return result, valid


def truncate_array_and_mask(
    array: jax.Array,
    mask: jax.Array,
    edge_items_per_axis: tuple[int | None, ...],
) -> tuple[jax.Array, jax.Array]:
  """Truncates an array along the given axis names.

  Args:
    array: Array to truncate.
    mask: Mask array, which must have the same number of dimensions as `array`,
      and whose axis sizes must be either 1 or the same as that axis of `array`
      (e.g. they are broadcast compatible).
    edge_items_per_axis: Number of edge items to keep for each axis, ignoring
      any axes whose slices are already computed in `prefix_slices`.

  Returns:
    A tuple containing a truncated version of the array along with a valid mask.
    Values taken from the original array have the valid mask as True, and there
    is one extra element in the middle with valid as False (standing in for the
    omitted elements). The return value is always fully replicated, because
    we cannot guarantee that it is evenly sharded across devices, and this
    function is usually used immediately before copying to the host.
  """
  assert jax is not None, "JAX is not available."
  sharding_kwargs = {}
  if hasattr(array, "sharding") and hasattr(
      array.sharding, "_device_assignment"
  ):
    # _truncate_part_with_slices usually returns slices that have odd
    # dimensions, which aren't divisible by most shardings. Unfortunately,
    # the XLA GSPMD partitioner sometimes still infers a sharding over one of
    # these axes, which then leads to partitioning errors in JAX whenever we
    # try to `device_get` the resulting array or call any additional operations
    # on it. To avoid this, we'd like to tell JAX to always produce an output
    # that is not sharded over any axis. Unfortunately, this is difficult
    # because JAX requires the in_shardings and out_shardings to have the same
    # devices in the same internal order, and at the time of writing JAX does
    # not provide any public API to look up the order of the devices in a
    # sharding (it allows looking up the device *set*, but not their order).
    # Whether or not this error happens seems to be somewhat nondeterministic.
    # To avoid this, we use the private property `_device_assignment` of
    # each sharding in order to figure out what device order it has, and then
    # explicitly request a fully-replicated output that is definitely safe to
    # retrieve.
    sharding_kwargs["out_shardings"] = (
        jax.sharding.GSPMDSharding.get_replicated(
            array.sharding._device_assignment  # pylint: disable=protected-access
        )
    )
  if array.size < SUMMARIZE_USING_NUMPY_THRESHOLD and safe_to_summarize(array):
    fn = functools.partial(_truncate_part_with_slices, xnp=np)
  else:
    fn = jax.jit(
        _truncate_part_with_slices, static_argnums=(2, 3), **sharding_kwargs
    )
  return fn(array, mask, (), edge_items_per_axis)


def faster_array_repr(array: jax.Array) -> str:
  """Computes ``repr(array)``, only copying the rendered array elements.

  ``repr(array)`` on a very large jax Array can be slow, because it copies the
  entire array to host memory even when only a few elements are actually needed.
  We can avoid this by truncating the array on device before fetching it.

  Args:
    array: The array to summarize.

  Returns:
    A string representation of the array. May differ slightly from the ordinary
    ``repr``, but should contain the same elements.
  """
  assert jax is not None, "JAX is not available."
  jnp = jax.numpy
  if array.size < np.get_printoptions()["threshold"]:
    return repr(array)

  if array.aval is not None and array.aval.weak_type:
    dtype_str = f"dtype={array.dtype.name}, weak_type=True)"
  else:
    dtype_str = f"dtype={array.dtype.name})"

  edgeitems = np.get_printoptions()["edgeitems"]
  edge_items_per_axis = []
  for size in array.shape:
    if size > 2 * edgeitems + 1:
      edge_items_per_axis.append(edgeitems)
    else:
      edge_items_per_axis.append(None)
  array_edges, _ = truncate_array_and_mask(
      array,
      np.ones((1,) * array.ndim, dtype=jnp.bool_),
      edge_items_per_axis=tuple(edge_items_per_axis),
  )
  prefix = "Array("
  datastring = np.array2string(
      np.array(array_edges),
      prefix=prefix,
      suffix=",",
      separator=", ",
      threshold=0,
      edgeitems=edgeitems,
  )
  return f"{prefix}{datastring}, {dtype_str}"


def make_checked_dataclasslike_renderer(
    cls: type[Any],
    fields: Sequence[str],
    fields_with_none_default: Sequence[str] = (),
) -> renderers.TreescopeNodeHandler:
  """Builds a roundtrippable renderer for a dataclass-like class.

  This function can be used to safely render classes that behave like Python
  dataclasses (i.e. they can be roundtripped by calling the constructor with
  attributes as keyword arguments). It is robust to potential new attributes
  being added by checking that it is possible to rebuild the instance correctly.
  This can be ued to render JAX builtin classes.

  Args:
    cls: The class to render.
    fields: A sequence of attribute names to render as keyword args.
    fields_with_none_default: A sequence of attribute names to render as keyword
      args only if they exist and their value is not None.

  Returns:
    A node handler for nodes of this type, which returns a simple rendering
    whenever the object is correctly described by these attributes.
  """

  def render_it(
      node: Any,
      path: str | None,
      subtree_renderer: renderers.TreescopeSubtreeRenderer,
  ) -> (
      rendering_parts.RenderableTreePart
      | rendering_parts.RenderableAndLineAnnotations
      | type(NotImplemented)
  ):
    if type(node) is not cls:  # pylint: disable=unidiomatic-typecheck
      return NotImplemented
    try:
      attributes = {k: getattr(node, k) for k in fields}
    except AttributeError:
      return NotImplemented
    for k in fields_with_none_default:
      if hasattr(node, k) and getattr(node, k) is not None:
        attributes[k] = getattr(node, k)

    # Make sure we can correctly round-trip it.
    rebuilt = cls(**attributes)
    if rebuilt != node:
      return NotImplemented
    else:
      return repr_lib.render_object_constructor(
          object_type=cls,
          attributes=attributes,
          path=path,
          subtree_renderer=subtree_renderer,
          roundtrippable=True,
      )

  return render_it


def render_precision(
    node: jax.lax.Precision,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders jax.lax.Precision."""
  assert jax is not None, "JAX is not available."
  if type(node) is not jax.lax.Precision:  # pylint: disable=unidiomatic-typecheck
    return NotImplemented
  return repr_lib.render_enumlike_item(
      object_type=jax.lax.Precision,
      item_name=node.name,
      item_value=node.value,
      path=path,
      subtree_renderer=subtree_renderer,
  )


def _compute_summary(
    x: jax.Array, is_floating: bool, is_integer: bool, is_bool: bool, xnp=None
) -> dict[str, jax.Array]:
  """Computes a summary of the given array."""
  if xnp is None:
    assert jax is not None, "JAX is not available."
    xnp = jax.numpy
  x = xnp.array(x)
  result = {}
  if is_floating:
    isfinite = xnp.isfinite(x)
    inf_to_nan = xnp.where(isfinite, x, xnp.array(xnp.nan, dtype=x.dtype))
    nanmean = functools.partial(xnp.nanmean, dtype=xnp.float32)
    nanstd = functools.partial(xnp.nanstd, dtype=xnp.float32)
    nanmin = lambda x: xnp.nanmin(x).astype(xnp.float32)
    nanmax = lambda x: xnp.nanmax(x).astype(xnp.float32)
    result.update(mean=nanmean(inf_to_nan), std=nanstd(inf_to_nan))
    result.update(nanmin=nanmin(x), nanmax=nanmax(x))
    result.update(
        nan=xnp.count_nonzero(xnp.isnan(x)),
        inf=xnp.count_nonzero(xnp.isposinf(x)),
    )
    result["any_finite"] = xnp.any(isfinite)
    result["-inf"] = xnp.count_nonzero(xnp.isneginf(x))
  if is_integer:
    result.update(min=xnp.min(x), max=xnp.max(x))
  if is_floating or is_integer:
    result.update(zero=xnp.count_nonzero(x == 0), nonzero=xnp.count_nonzero(x))
  if is_bool:
    result.update(
        true=xnp.count_nonzero(x), false=xnp.count_nonzero(xnp.logical_not(x))
    )
  return result


def _summarize_array_data_unconditionally(array: jax.Array) -> list[str]:
  """Summarized the data of a JAX array."""
  assert jax is not None, "JAX is not available."
  jnp = jax.numpy
  output_parts = []
  # This is required if treescope is invoked inside jitted function.
  with jax.core.ensure_compile_time_eval():
    is_floating = _is_subdtype(array.dtype, jnp.floating)
    is_integer = _is_subdtype(array.dtype, jnp.integer)
    is_bool = _is_subdtype(array.dtype, jnp.bool_)
    if not (is_floating or is_integer or is_bool):
      # Non-numeric non-bool data type (perhaps JAX PRNG key dtype). Can't
      # summarize values.
      return []

    if array.size < SUMMARIZE_USING_NUMPY_THRESHOLD:
      stat = _compute_summary(array, is_floating, is_integer, is_bool, xnp=np)
    else:
      compute_summary = jax.jit(_compute_summary, static_argnums=(1, 2, 3))
      stat = compute_summary(array, is_floating, is_integer, is_bool)
      # Get values in parallel.
      stat = jax.device_get(stat)

    # pylint: disable=inconsistent-quotes
    if is_floating and stat["any_finite"]:
      output_parts.append(f" ≈{stat['mean']:.2} ±{stat['std']:.2}")
      output_parts.append(f" [≥{stat['nanmin']:.2}, ≤{stat['nanmax']:.2}]")

    if is_integer:
      output_parts.append(f" [≥{stat['min']:_d}, ≤{stat['max']:_d}]")
    # pylint: enable=inconsistent-quotes

    def append_if_present(output_parts, *names):
      for name in names:
        if stat[name]:
          output_parts.append(f" {name}:{stat[name]:_d}")

    if is_floating or is_integer:
      append_if_present(output_parts, "zero", "nonzero")
    if is_floating:
      append_if_present(output_parts, "nan", "inf", "-inf")

    if is_bool:
      append_if_present(output_parts, "true", "false")
    return output_parts


def summarize_array_data(array: jax.Array) -> str:
  """Summarized the data of a JAX array.

  Args:
    array: The array to summarize.

  Returns:
    A string summarizing the data of the array.
  """

  output_parts = []

  if isinstance(array, jax.core.Tracer):
    output_parts.append(" - tracer.")
  elif array.is_deleted():
    output_parts.append(" - deleted!")
  elif not _is_locally_available(array):
    output_parts.append(" - multi-host array!")
  elif safe_to_summarize(array):
    output_parts.extend(_summarize_array_data_unconditionally(array))
  else:
    output_parts.append("- too large to summarize.")

  return "".join(output_parts)


class JAXArrayAdapter(ndarray_adapters.NDArrayAdapter[jax.Array]):
  """Array adapter for JAX arrays."""

  def get_axis_info_for_array_data(
      self, array: jax.Array
  ) -> tuple[ndarray_adapters.AxisInfo, ...]:
    assert jax is not None, "JAX is not available."
    return tuple(
        ndarray_adapters.PositionalAxisInfo(i, size)
        for i, size in enumerate(array.shape)
    )

  def get_array_data_with_truncation(
      self,
      array: jax.Array,
      mask: jax.Array | None,
      edge_items_per_axis: tuple[int | None, ...],
  ) -> tuple[np.ndarray, np.ndarray]:
    assert jax is not None, "JAX is not available."
    assert not isinstance(array, jax.core.Tracer)
    assert not array.is_deleted()
    if mask is None:
      mask = np.array(True)

    if edge_items_per_axis == (None,) * array.ndim:
      # No truncation.
      return np.array(array), np.broadcast_to(mask, array.shape)

    array, mask = truncate_array_and_mask(array, mask, edge_items_per_axis)
    return jax.device_get((array, mask))

  def get_array_summary(
      self, array: jax.Array, fast: bool
  ) -> rendering_parts.RenderableTreePart:
    output_parts = [
        rendering_parts.abbreviatable(
            rendering_parts.text("jax.Array "),
            rendering_parts.text("jax "),
        )
    ]

    output_parts.append(dtype_util.get_dtype_name(array.dtype))
    output_parts.append(repr(array.shape))
    if array.is_deleted():
      output_parts.append(" - deleted!")
    elif not fast:
      output_parts.append(
          rendering_parts.abbreviatable(
              rendering_parts.text(summarize_array_data(array)),
              rendering_parts.empty_part(),
          )
      )

    return rendering_parts.siblings(*output_parts)

  def get_numpy_dtype(self, array: jax.Array) -> np.dtype | None:
    if isinstance(array.dtype, np.dtype):
      return array.dtype
    else:
      return None

  def get_sharding_info_for_array_data(
      self, array: jax.Array
  ) -> ndarray_adapters.ShardingInfo | None:
    assert jax is not None, "JAX is not available."
    if isinstance(array, jax.core.Tracer) or array.is_deleted():
      return None

    [platform] = set(device.platform for device in array.sharding.device_set)
    device_map = array.sharding.devices_indices_map(array.shape)
    return ndarray_adapters.ShardingInfo(
        shard_shape=array.sharding.shard_shape(array.shape),
        device_index_to_shard_slices={
            device.id: slices for device, slices in device_map.items()
        },
        device_type=platform.upper(),
        fully_replicated=array.is_fully_replicated,
    )

  def should_autovisualize(self, array: jax.Array) -> bool:
    assert jax is not None, "JAX is not available."
    return not isinstance(array, jax.core.Tracer) and not array.is_deleted()


def render_jax_arrays(
    node: jax.Array,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a JAX array."""
  assert jax is not None, "JAX is not available."
  del subtree_renderer
  assert isinstance(node, jax.Array)
  if isinstance(node, jax.core.Tracer):
    return NotImplemented

  adapter = JAXArrayAdapter()

  if node.is_deleted():
    return rendering_parts.error_color(
        rendering_parts.siblings(
            rendering_parts.text("<"),
            adapter.get_array_summary(node, fast=True),
            rendering_parts.text(">"),
        )
    )

  def _placeholder() -> rendering_parts.RenderableTreePart:
    return rendering_parts.deferred_placeholder_style(
        rendering_parts.siblings(
            rendering_parts.text("<"),
            adapter.get_array_summary(node, fast=True),
            rendering_parts.text(">"),
        )
    )

  def _thunk(placeholder_expand_state: rendering_parts.ExpandState | None):
    # Is this array simple enough to render without a summary?
    node_repr = faster_array_repr(node)
    if "\n" not in node_repr and "..." not in node_repr:
      rendering = rendering_parts.abbreviation_color(
          rendering_parts.text(f"<jax.{node_repr}>")
      )
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


def set_up_treescope():
  """Sets up treescope to render JAX objects."""
  if jax is None:
    raise RuntimeError(
        "Cannot set up JAX support in treescope: JAX cannot be imported."
    )
  type_registries.TREESCOPE_HANDLER_REGISTRY[jax.ShapeDtypeStruct] = (
      make_checked_dataclasslike_renderer(
          jax.ShapeDtypeStruct,
          fields=("shape", "dtype"),
          fields_with_none_default=("sharding",),
      )
  )
  type_registries.TREESCOPE_HANDLER_REGISTRY[jax.tree_util.SequenceKey] = (
      make_checked_dataclasslike_renderer(
          jax.tree_util.SequenceKey, fields=("idx",)
      )
  )
  type_registries.TREESCOPE_HANDLER_REGISTRY[jax.tree_util.DictKey] = (
      make_checked_dataclasslike_renderer(
          jax.tree_util.DictKey, fields=("key",)
      )
  )
  type_registries.TREESCOPE_HANDLER_REGISTRY[jax.tree_util.GetAttrKey] = (
      make_checked_dataclasslike_renderer(
          jax.tree_util.GetAttrKey, fields=("name",)
      )
  )
  type_registries.TREESCOPE_HANDLER_REGISTRY[
      jax.tree_util.FlattenedIndexKey
  ] = make_checked_dataclasslike_renderer(
      jax.tree_util.FlattenedIndexKey, fields=("key",)
  )
  type_registries.TREESCOPE_HANDLER_REGISTRY[jax.lax.Precision] = (
      render_precision
  )

  # The concrete type of a JAX array is a private type that is dynamically
  # registered as a jax.Array subclass, so we need to add it to the list of
  # dynamically-checked virtual base classes.
  type_registries.VIRTUAL_BASE_CLASSES.append(jax.Array)
  type_registries.IMMUTABLE_TYPES_REGISTRY[jax.Array] = True
  type_registries.NDARRAY_ADAPTER_REGISTRY[jax.Array] = JAXArrayAdapter()
  type_registries.TREESCOPE_HANDLER_REGISTRY[jax.Array] = render_jax_arrays

  for jax_api_module in [
      jax.lax,
      jax.numpy,
      jax.scipy,
      jax.random,
      jax.nn,
      jax.custom_derivatives,
      jax,
  ]:
    canonical_aliases.populate_from_public_api(
        jax_api_module, canonical_aliases.prefix_filter("jax")
    )

  for key_cls_name in [
      "SequenceKey",
      "DictKey",
      "GetAttrKey",
      "FlattenedIndexKey",
  ]:
    canonical_aliases.add_alias(
        getattr(jax.tree_util, key_cls_name),
        canonical_aliases.ModuleAttributePath("jax.tree_util", (key_cls_name,)),
        on_conflict="ignore",
    )
