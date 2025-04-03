# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable, Sequence
import enum
import operator
from functools import partial
import math
from typing import NamedTuple
import weakref

import numpy as np

from jax._src import ad_util
from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import source_info_util
from jax._src import util
from jax._src import mesh as mesh_lib
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax
from jax._src.lax.utils import (
    _argnum_weak_type,
    _input_dtype,
    standard_primitive,
)
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.typing import Array, ArrayLike, Shape
from jax._src.util import safe_map, safe_zip

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

_dtype = partial(dtypes.dtype, canonicalize=True)


def slice(operand: ArrayLike, start_indices: Sequence[int],
          limit_indices: Sequence[int],
          strides: Sequence[int] | None = None) -> Array:
  """Wraps XLA's `Slice
  <https://www.tensorflow.org/xla/operation_semantics#slice>`_
  operator.

  Args:
    operand: an array to slice
    start_indices: a sequence of ``operand.ndim`` start indices.
    limit_indices: a sequence of ``operand.ndim`` limit indices.
    strides: an optional sequence of ``operand.ndim`` strides.

  Returns:
    The sliced array

  Examples:
    Here are some examples of simple two-dimensional slices:

    >>> x = jnp.arange(12).reshape(3, 4)
    >>> x
    Array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=int32)

    >>> lax.slice(x, (1, 0), (3, 2))
    Array([[4, 5],
           [8, 9]], dtype=int32)

    >>> lax.slice(x, (0, 0), (3, 4), (1, 2))
    Array([[ 0,  2],
           [ 4,  6],
           [ 8, 10]], dtype=int32)

    These two examples are equivalent to the following Python slicing syntax:

    >>> x[1:3, 0:2]
    Array([[4, 5],
           [8, 9]], dtype=int32)

    >>> x[0:3, 0:4:2]
    Array([[ 0,  2],
           [ 4,  6],
           [ 8, 10]], dtype=int32)

  See Also:
    - :attr:`jax.numpy.ndarray.at`
    - :func:`jax.lax.slice_in_dim`
    - :func:`jax.lax.index_in_dim`
    - :func:`jax.lax.dynamic_slice`
  """
  return slice_p.bind(operand, start_indices=tuple(start_indices),
                      limit_indices=tuple(limit_indices),
                      strides=None if strides is None else tuple(strides))


def dynamic_slice(
    operand: Array | np.ndarray,
    start_indices: Array | np.ndarray | Sequence[ArrayLike],
    slice_sizes: Shape,
) -> Array:
  """Wraps XLA's `DynamicSlice
  <https://www.tensorflow.org/xla/operation_semantics#dynamicslice>`_
  operator.

  Args:
    operand: an array to slice.
    start_indices: a list of scalar indices, one per dimension. These values
      may be dynamic.
    slice_sizes: the size of the slice. Must be a sequence of non-negative
      integers with length equal to `ndim(operand)`. Inside a JIT compiled
      function, only static values are supported (all JAX arrays inside JIT
      must have statically known size).

  Returns:
    An array containing the slice.

  Examples:
    Here is a simple two-dimensional dynamic slice:

    >>> x = jnp.arange(12).reshape(3, 4)
    >>> x
    Array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=int32)

    >>> dynamic_slice(x, (1, 1), (2, 3))
    Array([[ 5,  6,  7],
           [ 9, 10, 11]], dtype=int32)

    Note the potentially surprising behavior for the case where the requested slice
    overruns the bounds of the array; in this case the start index is adjusted to
    return a slice of the requested size:

    >>> dynamic_slice(x, (1, 1), (2, 4))
    Array([[ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=int32)

  See Also:
    - :attr:`jax.numpy.ndarray.at`
    - :func:`jax.lax.slice`
    - :func:`jax.lax.dynamic_slice_in_dim`
    - :func:`jax.lax.dynamic_index_in_dim`
  """
  start_indices = _dynamic_slice_indices(operand, start_indices)
  if config.dynamic_shapes.value:
    dynamic_sizes, static_sizes = lax._extract_tracers_dyn_shape(slice_sizes)
  else:
    dynamic_sizes = []
    static_sizes = core.canonicalize_shape(slice_sizes)  # type: ignore
  return dynamic_slice_p.bind(operand, *start_indices, *dynamic_sizes,
                              slice_sizes=tuple(static_sizes))


def dynamic_update_slice(operand: Array | np.ndarray, update: ArrayLike,
                         start_indices: Array | Sequence[ArrayLike]) -> Array:
  """Wraps XLA's `DynamicUpdateSlice
  <https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice>`_
  operator.

  Args:
    operand: an array to slice.
    update: an array containing the new values to write onto `operand`.
    start_indices: a list of scalar indices, one per dimension.

  Returns:
    An array containing the slice.

  Examples:
    Here is an example of updating a one-dimensional slice update:

    >>> x = jnp.zeros(6)
    >>> y = jnp.ones(3)
    >>> dynamic_update_slice(x, y, (2,))
    Array([0., 0., 1., 1., 1., 0.], dtype=float32)

    If the update slice is too large to fit in the array, the start
    index will be adjusted to make it fit

    >>> dynamic_update_slice(x, y, (3,))
    Array([0., 0., 0., 1., 1., 1.], dtype=float32)
    >>> dynamic_update_slice(x, y, (5,))
    Array([0., 0., 0., 1., 1., 1.], dtype=float32)

    Here is an example of a two-dimensional slice update:

    >>> x = jnp.zeros((4, 4))
    >>> y = jnp.ones((2, 2))
    >>> dynamic_update_slice(x, y, (1, 2))
    Array([[0., 0., 0., 0.],
           [0., 0., 1., 1.],
           [0., 0., 1., 1.],
           [0., 0., 0., 0.]], dtype=float32)

  See Also:
    - :attr:`jax.numpy.ndarray.at`
    - :attr:`lax.dynamic_update_index_in_dim`
    - :attr:`lax.dynamic_update_slice_in_dim`
  """
  start_indices = _dynamic_slice_indices(operand, start_indices)
  return dynamic_update_slice_p.bind(operand, update, *start_indices)


class GatherDimensionNumbers(NamedTuple):
  """
  Describes the dimension number arguments to an `XLA's Gather operator
  <https://www.tensorflow.org/xla/operation_semantics#gather>`_. See the XLA
  documentation for more details of what the dimension numbers mean.

  Args:
    offset_dims: the set of dimensions in the `gather` output that offset into
      an array sliced from `operand`. Must be a tuple of integers in ascending
      order, each representing a dimension number of the output.
    collapsed_slice_dims: the set of dimensions `i` in `operand` that have
      `slice_sizes[i] == 1` and that should not have a corresponding dimension
      in the output of the gather. Must be a tuple of integers in ascending
      order.
    start_index_map: for each dimension in `start_indices`, gives the
      corresponding dimension in the `operand` that is to be sliced. Must be a
      tuple of integers with size equal to `start_indices.shape[-1]`.
    operand_batching_dims: the set of batching dimensions `i` in `operand` that
      have `slice_sizes[i] == 1` and that should have a corresponding dimension
      in both the `start_indices` (at the same index in
      `start_indices_batching_dims`) and output of the gather. Must be a tuple
      of integers in ascending order.
    start_indices_batching_dims: the set of batching dimensions `i` in
      `start_indices` that should have a corresponding dimension in both the
      `operand` (at the same index in `operand_batching_dims`) and output of the
      gather. Must be a tuple of integers (order is fixed based on
      correspondence with `operand_batching_dims`).

  Unlike XLA's `GatherDimensionNumbers` structure, `index_vector_dim` is
  implicit; there is always an index vector dimension and it must always be the
  last dimension. To gather scalar indices, add a trailing dimension of size 1.
  """
  offset_dims: tuple[int, ...]
  collapsed_slice_dims: tuple[int, ...]
  start_index_map: tuple[int, ...]
  operand_batching_dims: tuple[int, ...] = ()
  start_indices_batching_dims: tuple[int, ...] = ()


class GatherScatterMode(enum.Enum):
  """
  Describes how to handle out-of-bounds indices in a gather or scatter.

  Possible values are:

  CLIP:
    Indices will be clamped to the nearest in-range value, i.e., such that the
    entire window to be gathered is in-range.
  FILL_OR_DROP:
    If any part of a gathered window is out of bounds, the entire window
    that is returned, even those elements that were otherwise in-bounds, will be
    filled with a constant.
    If any part of a scattered window is out of bounds, the entire window
    will be discarded.
  PROMISE_IN_BOUNDS:
    The user promises that indices are in bounds. No additional checking will be
    performed. In practice, with the current XLA implementation this means
    that out-of-bounds gathers will be clamped but out-of-bounds scatters will
    be discarded. Gradients will not be correct if indices are out-of-bounds.
  """
  CLIP = enum.auto()
  FILL_OR_DROP = enum.auto()
  PROMISE_IN_BOUNDS = enum.auto()
  ONE_HOT = enum.auto()

  @staticmethod
  def from_any(s: str | GatherScatterMode | None):
    if isinstance(s, GatherScatterMode):
      return s
    if s == "clip":
      return GatherScatterMode.CLIP
    if s is None or s == "fill" or s == "drop":
      return GatherScatterMode.FILL_OR_DROP
    if s == "promise_in_bounds":
      return GatherScatterMode.PROMISE_IN_BOUNDS
    if s == "one_hot":
      return GatherScatterMode.ONE_HOT
    else:
      raise ValueError(f'Unknown gather mode "{s}"')


def gather(operand: ArrayLike, start_indices: ArrayLike,
           dimension_numbers: GatherDimensionNumbers,
           slice_sizes: Shape,
           *,
           unique_indices: bool = False,
           indices_are_sorted: bool = False,
           mode: str | GatherScatterMode | None = None,
           fill_value = None) -> Array:
  """Gather operator.

  Wraps `XLA's Gather operator
  <https://www.tensorflow.org/xla/operation_semantics#gather>`_.

  :func:`gather` is a low-level operator with complicated semantics, and most JAX
  users will never need to call it directly. Instead, you should prefer using
  `Numpy-style indexing`_, and/or :func:`jax.numpy.ndarray.at`, perhaps in combination
  with :func:`jax.vmap`.

  Args:
    operand: an array from which slices should be taken
    start_indices: the indices at which slices should be taken
    dimension_numbers: a `lax.GatherDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices` and the output relate.
    slice_sizes: the size of each slice. Must be a sequence of non-negative
      integers with length equal to `ndim(operand)`.
    indices_are_sorted: whether `indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the elements gathered from ``operand`` are
      guaranteed not to overlap with each other. If ``True``, this may improve
      performance on some backends. JAX does not check this promise: if
      the elements overlap the behavior is undefined.
    mode: how to handle indices that are out of bounds: when set to ``'clip'``,
      indices are clamped so that the slice is within bounds, and when
      set to ``'fill'`` or ``'drop'`` gather returns a slice full of
      ``fill_value`` for the affected slice. The behavior for out-of-bounds
      indices when set to ``'promise_in_bounds'`` is implementation-defined.
    fill_value: the fill value to return for out-of-bounds slices when `mode`
      is ``'fill'``. Ignored otherwise. Defaults to ``NaN`` for inexact types,
      the largest negative value for signed types, the largest positive value
      for unsigned types, and ``True`` for booleans.

  Returns:
    An array containing the gather output.

  Examples:
    As mentioned above, you should basically never use :func:`gather` directly,
    and instead use NumPy-style indexing expressions to gather values from
    arrays.

    For example, here is how you can extract values at particular indices using
    straightforward indexing semantics, which will lower to XLA's Gather operator:

    >>> import jax.numpy as jnp
    >>> x = jnp.array([10, 11, 12])
    >>> indices = jnp.array([0, 1, 1, 2, 2, 2])

    >>> x[indices]
    Array([10, 11, 11, 12, 12, 12], dtype=int32)

    For control over settings like ``indices_are_sorted``, ``unique_indices``, ``mode``,
    and ``fill_value``, you can use the :attr:`jax.numpy.ndarray.at` syntax:

    >>> x.at[indices].get(indices_are_sorted=True, mode="promise_in_bounds")
    Array([10, 11, 11, 12, 12, 12], dtype=int32)

    By comparison, here is the equivalent function call using :func:`gather` directly,
    which is not something typical users should ever need to do:

    >>> from jax import lax
    >>> lax.gather(x, indices[:, None], slice_sizes=(1,),
    ...            dimension_numbers=lax.GatherDimensionNumbers(
    ...                offset_dims=(),
    ...                collapsed_slice_dims=(0,),
    ...                start_index_map=(0,)),
    ...            indices_are_sorted=True,
    ...            mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS)
    Array([10, 11, 11, 12, 12, 12], dtype=int32)

  .. _Numpy-style indexing: https://numpy.org/doc/stable/reference/arrays.indexing.html
  """
  if mode is None:
    mode = GatherScatterMode.PROMISE_IN_BOUNDS
  parsed_mode = GatherScatterMode.from_any(mode)
  if parsed_mode == GatherScatterMode.FILL_OR_DROP:
    if fill_value is None:
      dtype = _dtype(operand)
      if dtypes.issubdtype(dtype, np.inexact):
        fill_value = np.nan
      elif dtypes.issubdtype(dtype, np.signedinteger):
        fill_value = dtypes.iinfo(dtype).min
      elif dtypes.issubdtype(dtype, np.unsignedinteger):
        fill_value = dtypes.iinfo(dtype).max
      elif dtype == dtypes.bool_:
        fill_value = True
      else:
        raise ValueError(f"Unsupported dtype for gather fill_value {dtype}")
  else:
    fill_value = None
  return gather_p.bind(
      operand, start_indices, dimension_numbers=dimension_numbers,
      slice_sizes=core.canonicalize_shape(slice_sizes),
      unique_indices=bool(unique_indices),
      indices_are_sorted=bool(indices_are_sorted),
      mode=parsed_mode,
      fill_value=fill_value)


class ScatterDimensionNumbers(NamedTuple):
  """
  Describes the dimension number arguments to an `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_. See the XLA
  documentation for more details of what the dimension numbers mean.

  Args:
    update_window_dims: the set of dimensions in the `updates` that are window
      dimensions. Must be a tuple of integers in ascending
      order, each representing a dimension number.
    inserted_window_dims: the set of size 1 window dimensions that must be
      inserted into the shape of `updates`. Must be a tuple of integers in
      ascending order, each representing a dimension number of the output. These
      are the mirror image of `collapsed_slice_dims` in the case of `gather`.
    scatter_dims_to_operand_dims: for each dimension in `scatter_indices`, gives
      the corresponding dimension in `operand`. Must be a sequence of integers
      with size equal to `scatter_indices.shape[-1]`.
    operand_batching_dims: the set of batching dimensions `i` in `operand` that
      should have a corresponding dimension in both the `scatter_indices` (at
      the same index in `scatter_indices_batching_dims`) and `updates`. Must be
      a tuple of integers in ascending order. These are the mirror image of
      `operand_batching_dims` in the case of `gather`.
    scatter_indices_batching_dims: the set of batching dimensions `i` in
      `scatter_indices` that should have a corresponding dimension in both the
      `operand` (at the same index in `operand_batching_dims`) and output of the
      gather. Must be a tuple of integers (order is fixed based on
      correspondence with `input_batching_dims`). These are the mirror image of
      `start_indices_batching_dims` in the case of `gather`.

  Unlike XLA's `ScatterDimensionNumbers` structure, `index_vector_dim` is
  implicit; there is always an index vector dimension and it must always be the
  last dimension. To scatter scalar indices, add a trailing dimension of size 1.
  """
  update_window_dims: Sequence[int]
  inserted_window_dims: Sequence[int]
  scatter_dims_to_operand_dims: Sequence[int]
  operand_batching_dims: Sequence[int] = ()
  scatter_indices_batching_dims: Sequence[int] = ()

def scatter_add(
  operand: ArrayLike, scatter_indices: ArrayLike, updates: ArrayLike,
  dimension_numbers: ScatterDimensionNumbers, *,
  indices_are_sorted: bool = False, unique_indices: bool = False,
  mode: str | GatherScatterMode | None = None) -> Array:
  """Scatter-add operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
  addition is used to combine updates and values from `operand`.

  The semantics of scatter are complicated, and its API might change in the
  future. For most use cases, you should prefer the
  :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
  the familiar NumPy indexing syntax.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `scatter_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the elements to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends. JAX does not check this promise: if the updated elements
      overlap when ``unique_indices`` is ``True`` the behavior is undefined.
    mode: how to handle indices that are out of bounds: when set to 'clip',
      indices are clamped so that the slice is within bounds, and when
      set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
      for out-of-bounds indices when set to 'promise_in_bounds' is
      implementation-defined.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = lax._reduction_jaxpr(lax.add,
                                       core.get_aval(lax._const(operand, 0)))
  return scatter_add_p.bind(
      operand, scatter_indices, updates, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=GatherScatterMode.from_any(mode))


def scatter_sub(
    operand: ArrayLike,
    scatter_indices: ArrayLike,
    updates: ArrayLike,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array:
  """Scatter-sub operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
  subtraction is used to combine updates and values from `operand`.

  The semantics of scatter are complicated, and its API might change in the
  future. For most use cases, you should prefer the
  :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
  the familiar NumPy indexing syntax.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes how
      dimensions of `operand`, `start_indices`, `updates` and the output relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the elements to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve
      performance on some backends. JAX does not check this promise: if the
      updated elements overlap when ``unique_indices`` is ``True`` the behavior
      is undefined.
    mode: how to handle indices that are out of bounds: when set to 'clip',
      indices are clamped so that the slice is within bounds, and when set to
      'fill' or 'drop' out-of-bounds updates are dropped. The behavior for
      out-of-bounds indices when set to 'promise_in_bounds' is
      implementation-defined.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = lax._reduction_jaxpr(
      lax.sub, core.get_aval(lax._const(operand, 0))
  )
  return scatter_sub_p.bind(
      operand,
      scatter_indices,
      updates,
      update_jaxpr=jaxpr,
      update_consts=consts,
      dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices,
      mode=GatherScatterMode.from_any(mode),
  )


def scatter_mul(
  operand: ArrayLike, scatter_indices: ArrayLike, updates: ArrayLike,
  dimension_numbers: ScatterDimensionNumbers, *,
  indices_are_sorted: bool = False, unique_indices: bool = False,
  mode: str | GatherScatterMode | None = None) -> Array:
  """Scatter-multiply operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
  multiplication is used to combine updates and values from `operand`.

  The semantics of scatter are complicated, and its API might change in the
  future. For most use cases, you should prefer the
  :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
  the familiar NumPy indexing syntax.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the elements to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends. JAX does not check this promise: if the updated elements
      overlap when ``unique_indices`` is ``True`` the behavior is undefined.
    mode: how to handle indices that are out of bounds: when set to 'clip',
      indices are clamped so that the slice is within bounds, and when
      set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
      for out-of-bounds indices when set to 'promise_in_bounds' is
      implementation-defined.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = lax._reduction_jaxpr(lax.mul,
                                       core.get_aval(lax._const(operand, 1)))
  return scatter_mul_p.bind(
      operand, scatter_indices, updates, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=GatherScatterMode.from_any(mode))

def scatter_min(
  operand: ArrayLike, scatter_indices: ArrayLike, updates: ArrayLike,
  dimension_numbers: ScatterDimensionNumbers, *,
  indices_are_sorted: bool = False, unique_indices: bool = False,
  mode: str | GatherScatterMode | None = None) -> Array:
  """Scatter-min operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
  the `min` function is used to combine updates and values from `operand`.

  The semantics of scatter are complicated, and its API might change in the
  future. For most use cases, you should prefer the
  :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
  the familiar NumPy indexing syntax.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the elements to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends. JAX does not check this promise: if the updated elements
      overlap when ``unique_indices`` is ``True`` the behavior is undefined.
    mode: how to handle indices that are out of bounds: when set to 'clip',
      indices are clamped so that the slice is within bounds, and when
      set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
      for out-of-bounds indices when set to 'promise_in_bounds' is
      implementation-defined.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = lax._reduction_jaxpr(lax.min,
                                       core.get_aval(lax._const(operand, 0)))
  return scatter_min_p.bind(
      operand, scatter_indices, updates, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=GatherScatterMode.from_any(mode))

def scatter_max(
  operand: ArrayLike, scatter_indices: ArrayLike, updates: ArrayLike,
  dimension_numbers: ScatterDimensionNumbers, *,
  indices_are_sorted: bool = False, unique_indices: bool = False,
  mode: str | GatherScatterMode | None = None) -> Array:
  """Scatter-max operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
  the `max` function is used to combine updates and values from `operand`.

  The semantics of scatter are complicated, and its API might change in the
  future. For most use cases, you should prefer the
  :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
  the familiar NumPy indexing syntax.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the elements to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends. JAX does not check this promise: if the updated elements
      overlap when ``unique_indices`` is ``True`` the behavior is undefined.
    mode: how to handle indices that are out of bounds: when set to 'clip',
      indices are clamped so that the slice is within bounds, and when
      set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
      for out-of-bounds indices when set to 'promise_in_bounds' is
      implementation-defined.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = lax._reduction_jaxpr(lax.max,
                                       core.get_aval(lax._const(operand, 0)))
  return scatter_max_p.bind(
      operand, scatter_indices, updates, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=GatherScatterMode.from_any(mode))

# To avoid recompilation, we store a dict of weak references to funcs.
_scatter_apply_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

def scatter_apply(
  operand: Array, scatter_indices: Array,
  func: Callable[[Array], Array],
  dimension_numbers: ScatterDimensionNumbers, *,
  update_shape: Shape = (),
  indices_are_sorted: bool = False, unique_indices: bool = False,
  mode: str | GatherScatterMode | None = None) -> Array:
  """Scatter-apply operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where values
  from ``operand`` are replaced with ``func(operand)``, with duplicate indices
  resulting in multiple applications of ``func``.

  The semantics of scatter are complicated, and its API might change in the
  future. For most use cases, you should prefer the
  :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
  the familiar NumPy indexing syntax.

  Note that in the current implementation, ``scatter_apply`` is not compatible
  with automatic differentiation.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    func: unary function that will be applied at each index.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    update_shape: the shape of the updates at the given indices.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the elements to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends. JAX does not check this promise: if the updated elements
      overlap when ``unique_indices`` is ``True`` the behavior is undefined.
    mode: how to handle indices that are out of bounds: when set to 'clip',
      indices are clamped so that the slice is within bounds, and when
      set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
      for out-of-bounds indices when set to 'promise_in_bounds' is
      implementation-defined.

  Returns:
    An array containing the result of applying `func` to `operand` at the given indices.
  """
  # TODO: can we implement this without a placeholder?
  unused = lax.full(update_shape, 0, operand.dtype)
  _apply = lambda x, _: func(x)
  try:
    _apply = _scatter_apply_cache.setdefault(func, _apply)
  except TypeError:  # func is not weak referenceable
    pass
  jaxpr, consts = lax._reduction_jaxpr(_apply, core.get_aval(lax._zero(operand)))
  # TODO: implement this via its own primitive so we can define appropriate autodiff rules.
  return scatter_p.bind(
      operand, scatter_indices, unused, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=GatherScatterMode.from_any(mode))

# Define this outside of scatter to ensure cache hits.
_scatter_reduction_computation = lambda x, y: y

def scatter(
  operand: ArrayLike, scatter_indices: ArrayLike, updates: ArrayLike,
  dimension_numbers: ScatterDimensionNumbers, *,
  indices_are_sorted: bool = False, unique_indices: bool = False,
  mode: str | GatherScatterMode | None = None) -> Array:
  """Scatter-update operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where updates
  replace values from `operand`.

  If multiple updates are performed to the same index of operand, they may be
  applied in any order.

  :func:`scatter` is a low-level operator with complicated semantics, and most
  JAX users will never need to call it directly. Instead, you should prefer using
  :func:`jax.numpy.ndarray.at` for more familiary NumPy-style indexing syntax.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the elements to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends. JAX does not check this promise: if the updated elements
      overlap when ``unique_indices`` is ``True`` the behavior is undefined.
    mode: how to handle indices that are out of bounds: when set to 'clip',
      indices are clamped so that the slice is within bounds, and when
      set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
      for out-of-bounds indices when set to 'promise_in_bounds' is
      implementation-defined.

  Returns:
    An array containing the sum of `operand` and the scattered updates.

  Examples:
    As mentioned above, you should basically never use :func:`scatter` directly,
    and instead perform scatter-style operations using NumPy-style indexing
    expressions via :attr:`jax.numpy.ndarray.at`.

    Here is and example of updating entries in an array using :attr:`jax.numpy.ndarray.at`,
    which lowers to an XLA Scatter operation:

    >>> x = jnp.zeros(5)
    >>> indices = jnp.array([1, 2, 4])
    >>> values = jnp.array([2.0, 3.0, 4.0])

    >>> x.at[indices].set(values)
    Array([0., 2., 3., 0., 4.], dtype=float32)

    This syntax also supports several of the optional arguments to :func:`scatter`,
    for example:

    >>> x.at[indices].set(values, indices_are_sorted=True, mode='promise_in_bounds')
    Array([0., 2., 3., 0., 4.], dtype=float32)

    By comparison, here is the equivalent function call using :func:`scatter` directly,
    which is not something typical users should ever need to do:

    >>> lax.scatter(x, indices[:, None], values,
    ...             dimension_numbers=lax.ScatterDimensionNumbers(
    ...                 update_window_dims=(),
    ...                 inserted_window_dims=(0,),
    ...                 scatter_dims_to_operand_dims=(0,)),
    ...             indices_are_sorted=True,
    ...             mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS)
    Array([0., 2., 3., 0., 4.], dtype=float32)
  """
  return scatter_p.bind(
      operand, scatter_indices, updates, update_jaxpr=None,
      update_consts=(), dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=GatherScatterMode.from_any(mode))

def index_take(src: Array, idxs: Array, axes: Sequence[int]) -> Array:
  indices = lax.concatenate([lax.expand_dims(i, (1,)) for i in idxs], 1)
  max_idx = lax.expand_dims(np.array([src.shape[ax] for ax in axes]),
                            tuple(range(indices.ndim - 1)))
  indices = indices % max_idx
  slice_sizes = list(src.shape)
  for ax in axes:
    slice_sizes[ax] = 1
  offset_dims = tuple(range(1, src.ndim - indices.shape[1] + 1))
  dnums = GatherDimensionNumbers(
      offset_dims=offset_dims,
      collapsed_slice_dims=tuple(axes),
      start_index_map=tuple(axes),
  )
  return gather(src, indices, dimension_numbers=dnums,
                slice_sizes=tuple(slice_sizes))


### convenience wrappers around traceables

def slice_in_dim(operand: Array | np.ndarray, start_index: int | None,
                 limit_index: int | None,
                 stride: int = 1, axis: int = 0) -> Array:
  """Convenience wrapper around :func:`lax.slice` applying to only one dimension.

  This is effectively equivalent to ``operand[..., start_index:limit_index:stride]``
  with the indexing applied on the specified axis.

  Args:
    operand: an array to slice.
    start_index: an optional start index (defaults to zero)
    limit_index: an optional end index (defaults to operand.shape[axis])
    stride: an optional stride (defaults to 1)
    axis: the axis along which to apply the slice (defaults to 0)

  Returns:
    An array containing the slice.

  Examples:
    Here is a one-dimensional example:

    >>> x = jnp.arange(4)
    >>> lax.slice_in_dim(x, 1, 3)
    Array([1, 2], dtype=int32)

    Here are some two-dimensional examples:

    >>> x = jnp.arange(12).reshape(4, 3)
    >>> x
    Array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11]], dtype=int32)

    >>> lax.slice_in_dim(x, 1, 3)
    Array([[3, 4, 5],
           [6, 7, 8]], dtype=int32)

    >>> lax.slice_in_dim(x, 1, 3, axis=1)
    Array([[ 1,  2],
           [ 4,  5],
           [ 7,  8],
           [10, 11]], dtype=int32)

  See Also:
    - :attr:`jax.numpy.ndarray.at`
    - :func:`jax.lax.slice`
    - :func:`jax.lax.index_in_dim`
    - :func:`jax.lax.dynamic_slice_in_dim`
  """
  start_indices = [0] * operand.ndim
  limit_indices = list(operand.shape)
  strides = [1] * operand.ndim

  # translate `None`
  len_axis = operand.shape[axis]
  start_index_int = (core._canonicalize_dimension(start_index)
                     if start_index is not None else 0)
  limit_index_int = (core._canonicalize_dimension(limit_index)
                     if limit_index is not None else len_axis)

  # translate negative indices
  if start_index_int < 0:
    start_index_int = start_index_int + len_axis
  if limit_index_int < 0:
    limit_index_int = limit_index_int + len_axis

  axis = int(axis)
  start_indices[axis] = start_index_int
  limit_indices[axis] = limit_index_int
  strides[axis] = core._canonicalize_dimension(stride)

  return slice(operand, start_indices, limit_indices, strides)


def index_in_dim(operand: Array | np.ndarray, index: int, axis: int = 0,
                 keepdims: bool = True) -> Array:
  """Convenience wrapper around :func:`lax.slice` to perform int indexing.

  This is effectively equivalent to ``operand[..., start_index:limit_index:stride]``
  with the indexing applied on the specified axis.

  Args:
    operand: an array to index.
    index: integer index
    axis: the axis along which to apply the index (defaults to 0)
    keepdims: boolean specifying whether the output array should preserve the
      rank of the input (default=True)

  Returns:
    The subarray at the specified index.

  Examples:
    Here is a one-dimensional example:

    >>> x = jnp.arange(4)
    >>> lax.index_in_dim(x, 2)
    Array([2], dtype=int32)

    >>> lax.index_in_dim(x, 2, keepdims=False)
    Array(2, dtype=int32)

    Here are some two-dimensional examples:

    >>> x = jnp.arange(12).reshape(3, 4)
    >>> x
    Array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=int32)

    >>> lax.index_in_dim(x, 1)
    Array([[4, 5, 6, 7]], dtype=int32)

    >>> lax.index_in_dim(x, 1, axis=1, keepdims=False)
    Array([1, 5, 9], dtype=int32)

  See Also:
    - :attr:`jax.numpy.ndarray.at`
    - :func:`jax.lax.slice`
    - :func:`jax.lax.slice_in_dim`
    - :func:`jax.lax.dynamic_index_in_dim`
  """
  index, axis = core._canonicalize_dimension(index), int(axis)
  axis_size = operand.shape[axis]
  wrapped_index = index + axis_size if index < 0 else index
  if not 0 <= wrapped_index < axis_size:
    msg = 'index {} is out of bounds for axis {} with size {}'
    raise IndexError(msg.format(index, axis, axis_size))
  result = slice_in_dim(operand, wrapped_index, wrapped_index + 1, 1, axis)
  if keepdims:
    return result
  else:
    return lax.squeeze(result, (axis,))


def dynamic_slice_in_dim(operand: Array | np.ndarray,
                         start_index: ArrayLike,
                         slice_size: int, axis: int = 0) -> Array:
  """Convenience wrapper around :func:`lax.dynamic_slice` applied to one dimension.

  This is roughly equivalent to the following Python indexing syntax applied
  along the specified axis: ``operand[..., start_index:start_index + slice_size]``.

  Args:
    operand: an array to slice.
    start_index: the (possibly dynamic) start index
    slice_size: the static slice size
    axis: the axis along which to apply the slice (defaults to 0)

  Returns:
    An array containing the slice.

  Examples:
    Here is a one-dimensional example:

    >>> x = jnp.arange(5)
    >>> dynamic_slice_in_dim(x, 1, 3)
    Array([1, 2, 3], dtype=int32)

    Like `jax.lax.dynamic_slice`, out-of-bound slices will be clipped to the
    valid range:

    >>> dynamic_slice_in_dim(x, 4, 3)
    Array([2, 3, 4], dtype=int32)

    Here is a two-dimensional example:

    >>> x = jnp.arange(12).reshape(3, 4)
    >>> x
    Array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=int32)

    >>> dynamic_slice_in_dim(x, 1, 2, axis=1)
    Array([[ 1,  2],
           [ 5,  6],
           [ 9, 10]], dtype=int32)

  See Also:
    - :attr:`jax.numpy.ndarray.at`
    - :func:`jax.lax.slice_in_dim`
    - :func:`jax.lax.dynamic_slice`
    - :func:`jax.lax.dynamic_index_in_dim`
  """
  start_indices: list[ArrayLike] = [lax._const(start_index, 0)] * operand.ndim
  slice_sizes = list(operand.shape)

  axis = int(axis)
  start_indices[axis] = start_index
  slice_sizes[axis] = core._canonicalize_dimension(slice_size)
  return dynamic_slice(operand, start_indices, slice_sizes)


def dynamic_index_in_dim(operand: Array | np.ndarray,
                         index: int | Array,
                         axis: int = 0, keepdims: bool = True) -> Array:
  """Convenience wrapper around dynamic_slice to perform int indexing.

  This is roughly equivalent to the following Python indexing syntax applied
  along the specified axis: ``operand[..., index]``.

  Args:
    operand: an array to slice.
    index: the (possibly dynamic) start index
    axis: the axis along which to apply the slice (defaults to 0)
    keepdims: boolean specifying whether the output should have the same rank as
      the input (default = True)

  Returns:
    An array containing the slice.

  Examples:
    Here is a one-dimensional example:

    >>> x = jnp.arange(5)
    >>> dynamic_index_in_dim(x, 1)
    Array([1], dtype=int32)

    >>> dynamic_index_in_dim(x, 1, keepdims=False)
    Array(1, dtype=int32)

    Here is a two-dimensional example:

    >>> x = jnp.arange(12).reshape(3, 4)
    >>> x
    Array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=int32)

    >>> dynamic_index_in_dim(x, 1, axis=1, keepdims=False)
    Array([1, 5, 9], dtype=int32)

  See Also:
    - :attr:`jax.numpy.ndarray.at`
    - :func:`jax.lax.index_in_dim`
    - :func:`jax.lax.dynamic_slice`
    - :func:`jax.lax.dynamic_slice_in_dim`
  """
  result = dynamic_slice_in_dim(operand, index, 1, axis)
  if keepdims:
    return result
  else:
    return lax.squeeze(result, (axis,))


def dynamic_update_slice_in_dim(operand: Array | np.ndarray,
                                update: ArrayLike,
                                start_index: ArrayLike, axis: int) -> Array:
  """Convenience wrapper around :func:`dynamic_update_slice` to update
  a slice in a single ``axis``.

  Args:
    operand: an array to slice.
    update: an array containing the new values to write onto `operand`.
    start_index: a single scalar index
    axis: the axis of the update.

  Returns:
    The updated array

  Examples:

    >>> x = jnp.zeros(6)
    >>> y = jnp.ones(3)
    >>> dynamic_update_slice_in_dim(x, y, 2, axis=0)
    Array([0., 0., 1., 1., 1., 0.], dtype=float32)

    If the update slice is too large to fit in the array, the start
    index will be adjusted to make it fit:

    >>> dynamic_update_slice_in_dim(x, y, 3, axis=0)
    Array([0., 0., 0., 1., 1., 1.], dtype=float32)
    >>> dynamic_update_slice_in_dim(x, y, 5, axis=0)
    Array([0., 0., 0., 1., 1., 1.], dtype=float32)

    Here is an example of a two-dimensional slice update:

    >>> x = jnp.zeros((4, 4))
    >>> y = jnp.ones((2, 4))
    >>> dynamic_update_slice_in_dim(x, y, 1, axis=0)
    Array([[0., 0., 0., 0.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [0., 0., 0., 0.]], dtype=float32)

    Note that the shape of the additional axes in ``update`` need not
    match the associated dimensions of the ``operand``:

    >>> y = jnp.ones((2, 3))
    >>> dynamic_update_slice_in_dim(x, y, 1, axis=0)
    Array([[0., 0., 0., 0.],
           [1., 1., 1., 0.],
           [1., 1., 1., 0.],
           [0., 0., 0., 0.]], dtype=float32)

  See Also:
    - :attr:`jax.numpy.ndarray.at`
    - :func:`jax.lax.dynamic_update_slice`
    - :func:`jax.lax.dynamic_update_index_in_dim`
    - :func:`jax.lax.dynamic_slice_in_dim`
  """
  axis = int(axis)
  start_indices: list[ArrayLike] = [lax._const(start_index, 0)] * lax._ndim(operand)
  start_indices[axis] = start_index
  return dynamic_update_slice(operand, update, start_indices)


def dynamic_update_index_in_dim(operand: Array | np.ndarray,
                                update: ArrayLike, index: ArrayLike,
                                axis: int) -> Array:
  """Convenience wrapper around :func:`dynamic_update_slice` to update a slice
  of size 1 in a single ``axis``.

  Args:
    operand: an array to slice.
    update: an array containing the new values to write onto `operand`.
    index: a single scalar index
    axis: the axis of the update.

  Returns:
    The updated array

  Examples:

    >>> x = jnp.zeros(6)
    >>> y = 1.0
    >>> dynamic_update_index_in_dim(x, y, 2, axis=0)
    Array([0., 0., 1., 0., 0., 0.], dtype=float32)

    >>> y = jnp.array([1.0])
    >>> dynamic_update_index_in_dim(x, y, 2, axis=0)
    Array([0., 0., 1., 0., 0., 0.], dtype=float32)

    If the specified index is out of bounds, the index will be clipped to the
    valid range:

    >>> dynamic_update_index_in_dim(x, y, 10, axis=0)
    Array([0., 0., 0., 0., 0., 1.], dtype=float32)

    Here is an example of a two-dimensional dynamic index update:

    >>> x = jnp.zeros((4, 4))
    >>> y = jnp.ones(4)
    >>> dynamic_update_index_in_dim(x, y, 1, axis=0)
    Array([[0., 0., 0., 0.],
          [1., 1., 1., 1.],
          [0., 0., 0., 0.],
          [0., 0., 0., 0.]], dtype=float32)

    Note that the shape of the additional axes in ``update`` need not
    match the associated dimensions of the ``operand``:

    >>> y = jnp.ones((1, 3))
    >>> dynamic_update_index_in_dim(x, y, 1, 0)
    Array([[0., 0., 0., 0.],
           [1., 1., 1., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]], dtype=float32)

  See Also:
    - :attr:`jax.numpy.ndarray.at`
    - :func:`jax.lax.dynamic_update_slice`
    - :func:`jax.lax.dynamic_update_index_in_dim`
    - :func:`jax.lax.dynamic_index_in_dim`
  """
  axis = int(axis)
  if lax._ndim(update) != lax._ndim(operand):
    assert lax._ndim(update) + 1 == lax._ndim(operand)
    update = lax.expand_dims(update, (axis,))
  return dynamic_update_slice_in_dim(operand, update, index, axis)


def _slice_shape_rule(operand, *, start_indices, limit_indices, strides):
  lax._check_shapelike("slice", "start_indices", start_indices)
  lax._check_shapelike("slice", "limit_indices", limit_indices)
  if operand.ndim != len(start_indices):
    msg = ("slice start_indices must have length equal to the number of "
           "dimensions of the operand, got indices {} for operand shape {}.")
    raise TypeError(msg.format(start_indices, operand.shape))
  if len(start_indices) != len(limit_indices):
    msg = ("slice limit_indices must have the same length as start_indices, "
           "got start_indices {} and limit_indices {}.")
    raise TypeError(msg.format(start_indices, limit_indices))
  if not all(map(operator.ge, operand.shape, limit_indices)):
    msg = ("slice limit_indices must be less than or equal to operand shape, "
           "got limit_indices {} for operand shape {}.")
    raise TypeError(msg.format(limit_indices, operand.shape))
  if not all(si >= 0 for si in start_indices):
    msg = ("slice start_indices must be greater than or equal to zero, "
           "got start_indices of {}.")
    raise TypeError(msg.format(start_indices))
  if not config.dynamic_shapes.value:
    if not all(map(operator.ge, limit_indices, start_indices)):
      msg = ("slice limit_indices must be greater than or equal to start_indices,"
            " got start_indices {} and limit_indices {}.")
      raise TypeError(msg.format(start_indices, limit_indices))
  diff = tuple(map(operator.sub, limit_indices, start_indices))
  if strides is None or tuple(strides) == (1,) * len(operand.shape):
    return diff

  lax._check_shapelike("slice", "strides", strides)
  if len(strides) != operand.ndim:
    msg = ("slice strides must have length equal to the number of dimensions "
            "of the operand, got strides {} for operand shape {}.")
    raise TypeError(msg.format(strides, operand.shape))
  if not all(s >= 0 for s in strides):
    msg = "slice strides must be positive, got {}"
    raise TypeError(msg.format(strides))
  return tuple(core.stride_dim(d, window_size=1, window_stride=s)
               for d, s in zip(diff, strides))

def _get_sub_spec_size(mesh, sub_spec):
  if isinstance(sub_spec, tuple):
    return math.prod(mesh.shape[s] for s in sub_spec)
  return mesh.shape[sub_spec]

def _get_sharding_for_varying_out_shape(out_shape, operand, name):
  """Returns a sharding when out_shape may not be the same as operand shape"""
  mesh = operand.sharding.mesh
  for op_sh, out_sh, op_spec in safe_zip(
      operand.shape, out_shape, operand.sharding.spec):
    if (op_sh != out_sh and op_spec is not None and
        out_sh % _get_sub_spec_size(mesh, op_spec) != 0):
      raise NotImplementedError(
          f"{name} on sharded dims where out dim ({out_sh}) is not divisble by"
          f" mesh axes ({_get_sub_spec_size(mesh, op_spec)}) with spec"
          f" ({op_spec}) is not implemented.")
  # TODO(yashkatariya): Returning operand.sharding as is may or may not move
  # data. So think about how to avoid it which might include creating a new
  # mesh? For example:
  # mesh = {'x': 4}
  # x = jax.device_put(jnp.arange(8), NamedSharding(mesh, P('x')))`
  # ys = lax.split(x, [4, 4])  # This will create outputs of shape (4,)
  # According to the current logic, ys[0].sharding.spec == P('x')
  # which involves data movement.
  return operand.sharding

def _slice_sharding_rule(operand, *, start_indices, limit_indices, strides):
  # TODO(yashkatariya): Once JAX supports uneven sharding at the top level,
  # change this logic to `return operand.sharding` directly.
  out_shape = _slice_shape_rule(operand, start_indices=start_indices,
                                limit_indices=limit_indices, strides=strides)
  return _get_sharding_for_varying_out_shape(out_shape, operand, 'slicing')

def _slice_transpose_rule(t, operand, *, start_indices, limit_indices, strides):
  assert ad.is_undefined_primal(operand)
  operand_shape = operand.aval.shape
  if strides is None or np.all(np.equal(strides, 1)):
    pads = zip(start_indices, np.subtract(operand_shape, limit_indices),
               (0,) * len(start_indices))
  else:
    real_limits = np.add(
      start_indices,
      np.where(np.array(t.shape) == 0, 0,
               np.add(1, np.multiply(np.subtract(t.shape, 1), strides))))
    pads = zip(start_indices, np.subtract(operand_shape, real_limits),
               np.subtract(strides, 1))
  result = lax.pad(t, lax._const(t, 0), pads)
  assert result.shape == operand_shape, f"{result.shape=} {operand_shape=}"
  return [result]


def _slice_batching_rule(batched_args, batch_dims, *, start_indices,
                         limit_indices, strides):
  operand, = batched_args
  bdim, = batch_dims

  new_start_indices = list(start_indices)
  new_start_indices.insert(bdim, 0)

  new_limit_indices = list(limit_indices)
  new_limit_indices.insert(bdim, operand.shape[bdim])

  if strides is None:
    new_strides = None
  else:
    new_strides = list(strides)
    new_strides.insert(bdim, 1)

  out = slice(operand, new_start_indices, new_limit_indices, new_strides)
  return out, bdim

slice_p = standard_primitive(_slice_shape_rule, _input_dtype, 'slice',
                             sharding_rule=_slice_sharding_rule)
ad.deflinear2(slice_p, _slice_transpose_rule)
batching.primitive_batchers[slice_p] = _slice_batching_rule
# TODO(mvoz): A better slice rule for ragged prop, enforcing boundaries
# or supporting nested jumbles. NYI.
batching.ragged_prop_rules[slice_p] = batching.ragged_mask_no_op_rule

# Override the standard impl to defer to dynamic_slice whenever possible.
# This lets us reuse the same program for many applications of slicing for as
# long as they have the same output shape. Note that this only applies in
# op-by-op mode and inside larger computations we still lower to static slices.
@slice_p.def_impl
def _slice_impl(x, start_indices, limit_indices, strides):
  if strides is not None:
    return dispatch.apply_primitive(
      slice_p, x, start_indices=start_indices,
      limit_indices=limit_indices, strides=strides)
  slice_sizes = tuple(np.array(limit_indices) - np.array(start_indices))
  return dispatch.apply_primitive(dynamic_slice_p, x, *start_indices,
                                  slice_sizes=slice_sizes)


def _slice_lower(ctx, x, *, start_indices, limit_indices, strides):
  strides = strides or [1] * len(start_indices)
  aval_out, = ctx.avals_out
  out = mlir.slice_op(ctx, x, aval_out, start_indices=start_indices,
                      limit_indices=limit_indices, strides=strides)
  if config.sharding_in_types.value:
    return [mlir.lower_sharding_under_shit(ctx, out, aval_out)]
  return [out]

mlir.register_lowering(slice_p, _slice_lower)


def _dynamic_slice_shape_rule(operand, *starts_and_dyn_sizes, slice_sizes):
  start_indices, dyn = util.split_list(starts_and_dyn_sizes, [operand.ndim])
  if operand.ndim != len(start_indices):
    msg = ("dynamic_slice start_indices must have length equal to the number "
           "of dimensions of the operand, got indices {} for operand shape {}.")
    raise TypeError(msg.format(start_indices, operand.shape))
  if len(start_indices) != len(slice_sizes):
    msg = ("dynamic_slice slice_sizes must have the same length as "
           "start_indices, got start_indices length {} and slice_sizes {}.")
    raise TypeError(msg.format(len(start_indices), slice_sizes))
  if not dyn and not all(map(operator.ge, operand.shape, slice_sizes)):
    msg = ("slice slice_sizes must be less than or equal to operand shape, "
           "got slice_sizes {} for operand shape {}.")
    raise TypeError(msg.format(slice_sizes, operand.shape))
  if not dyn and not all(ssz >= 0 for ssz in slice_sizes):
    msg = ("slice slice_sizes must be greater than or equal to zero, "
           "got slice_sizes of {}.")
    raise TypeError(msg.format(slice_sizes))
  if any(idx.ndim != 0 for idx in start_indices):
    raise TypeError("start_indices arguments to dynamic_slice must be scalars, "
                    f" got indices {start_indices}")
  return tuple(lax._merge_dyn_shape(slice_sizes, dyn))

def _dynamic_slice_sharding_rule(operand, *starts_and_dyn_sizes, slice_sizes):
  out_shape = _dynamic_slice_shape_rule(
      operand, *starts_and_dyn_sizes, slice_sizes=slice_sizes)
  return _get_sharding_for_varying_out_shape(out_shape, operand, 'dynamic_slice')


def _dynamic_slice_dtype_rule(operand, *starts_and_dyn_sizes, slice_sizes):
  start_indices, dyn = util.split_list(starts_and_dyn_sizes, [operand.ndim])
  if any(i.dtype != start_indices[0].dtype or
         not dtypes.issubdtype(i.dtype, np.integer) for i in start_indices):
    msg = ("index arguments to dynamic_slice must be integers of the same "
           "type, got: {}")
    raise TypeError(msg.format(", ".join(i.dtype.name for i in start_indices)))
  return operand.dtype

def _dynamic_slice_jvp(primals, tangents, *, slice_sizes):
  tangent_out = tangents[0]
  if type(tangent_out) is not ad_util.Zero:
    tangent_out = dynamic_slice_p.bind(tangent_out, *primals[1:], slice_sizes=slice_sizes)
  return dynamic_slice_p.bind(primals[0], *primals[1:], slice_sizes=slice_sizes), tangent_out

def _dynamic_slice_transpose_rule(t, operand, *start_indices, slice_sizes):
  assert ad.is_undefined_primal(operand)
  assert all(not ad.is_undefined_primal(s) for s in start_indices)
  operand_shape, operand_dtype = operand.aval.shape, operand.aval.dtype
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(operand.aval)] + [None] * len(start_indices)
  else:
    zeros = lax.full(operand_shape, 0, operand_dtype)
    return ([dynamic_update_slice_p.bind(zeros, t, *start_indices)] +
            [None] * len(start_indices))

def _batch_dynamic_slice_indices(indices, bdims):
  if len(indices) == 0:
    return np.array([], 'int32'), None
  empty_marker = object()
  size = next((x.shape[i] for x, i in zip(indices, bdims) if i is not None),
              empty_marker)
  if size is empty_marker:
    return lax.concatenate([lax.broadcast(i, (1,)) for i in indices], 0), None
  indices = lax.concatenate(
    [lax.broadcast_in_dim(x, (size, 1),
                          broadcast_dimensions=((0,) if i is not None else ()))
     for x, i in zip(indices, bdims)],
    dimension=1)
  return indices, 0

def _dynamic_slice_batching_rule(batched_args, batch_dims, *, slice_sizes):
  # A dynamic slice is a special case of gather; we can delegate to the gather
  # batching rule.
  # TODO(phawkins): consider removing dynamic_slice entirely and using gather
  # always.
  # TODO(mattjj): Alternately, we could add jnp.unstack and an unstack_p,
  # since it should have easier rules (especially compared to gather).
  operand, *start_indices_and_dyn = batched_args
  operand_bd, *start_idx_and_dyn_bds = batch_dims
  ndims = operand.ndim - (0 if operand_bd is batching.not_mapped else 1)
  dims = tuple(range(ndims))
  start_indices, dyn_slice_sizes = util.split_list(start_indices_and_dyn, [ndims])
  start_idx_bds, dyn_slice_size_bds = util.split_list(start_idx_and_dyn_bds, [ndims])
  dnums = GatherDimensionNumbers(
      offset_dims=dims,
      collapsed_slice_dims=(),
      start_index_map=dims,
  )
  index, index_bdim = _batch_dynamic_slice_indices(start_indices, start_idx_bds)
  return _gather_batching_rule(
    [operand, index, *dyn_slice_sizes],
    [operand_bd, index_bdim, *dyn_slice_size_bds], dimension_numbers=dnums,
    slice_sizes=slice_sizes, unique_indices=True, indices_are_sorted=True,
    mode=GatherScatterMode.PROMISE_IN_BOUNDS, fill_value=None)

def _dynamic_slice_staging_rule(trace, x, *starts_and_dyn_sizes, slice_sizes):
  start_indices, dyn = util.split_list(starts_and_dyn_sizes, [x.ndim])
  if not dyn:
    return trace.default_process_primitive(dynamic_slice_p, (x, *start_indices),
                                           dict(slice_sizes=slice_sizes))
  shape = lax._merge_dyn_shape(slice_sizes, dyn)
  aval = core.DShapedArray(shape, x.dtype, False)
  return lax._dyn_shape_staging_rule(trace, dynamic_slice_p, aval, x,
                                     *starts_and_dyn_sizes,
                                     slice_sizes=slice_sizes)

def _dynamic_slice_typecheck_rule(_, x, *starts_and_dyn_sizes, slice_sizes):
  start_indices, dyn = util.split_list(starts_and_dyn_sizes, [x.aval.ndim])
  if not dyn:
    out_aval, effects = dynamic_slice_p.abstract_eval(
        x.aval, *(d.aval for d in start_indices), slice_sizes=slice_sizes)
    return [out_aval], effects
  else:
    # TODO(mattjj): perform more checks
    out_shape = lax._merge_dyn_shape(slice_sizes, dyn)
    out_shape = [d.val if type(d) is core.Literal else d for d in out_shape]
    out_aval = core.DShapedArray(tuple(out_shape), x.aval.dtype,
                                 x.aval.weak_type)
    return [out_aval], core.no_effects

def _dynamic_slice_padding_rule(in_avals, out_avals, x, *starts_and_dyn,
                                slice_sizes):
  x_aval, start_indices_avals, dyn_avals = util.split_list(in_avals, [1, x.ndim])
  start_indices, dyn = util.split_list(starts_and_dyn, [x.ndim])
  dyn_ = [a.dtype.bound if type(a.dtype) is core.bint else d
          for a, d in zip(dyn_avals, dyn)]
  slice_sizes_ = lax._merge_dyn_shape(slice_sizes, dyn_)
  start_idx = [d.val if type(d) is core.DArray else d for d in start_indices]
  return [dynamic_slice(x, start_idx, slice_sizes_)]

dynamic_slice_p = standard_primitive(
    _dynamic_slice_shape_rule, _dynamic_slice_dtype_rule, 'dynamic_slice',
    weak_type_rule=_argnum_weak_type(0),
    sharding_rule=_dynamic_slice_sharding_rule)
ad.primitive_jvps[dynamic_slice_p] = _dynamic_slice_jvp
ad.primitive_transposes[dynamic_slice_p] = _dynamic_slice_transpose_rule
batching.primitive_batchers[dynamic_slice_p] = _dynamic_slice_batching_rule
pe.custom_staging_rules[dynamic_slice_p] = _dynamic_slice_staging_rule
core.custom_typechecks[dynamic_slice_p] = _dynamic_slice_typecheck_rule
pe.padding_rules[dynamic_slice_p] = _dynamic_slice_padding_rule

def _dynamic_slice_lower(ctx, x, *starts_and_dyn_sizes, slice_sizes):
  x_aval, *_ = ctx.avals_in
  start_indices, dyn = util.split_list(starts_and_dyn_sizes, [x_aval.ndim])
  aval_out, = ctx.avals_out
  if dyn:
    aval_out = aval_out.update(shape=lax._merge_dyn_shape(slice_sizes, dyn))
  out = mlir.dynamic_slice(ctx, aval_out, x, start_indices=start_indices)
  if config.sharding_in_types.value:
    return [mlir.lower_sharding_under_shit(ctx, out, aval_out)]
  return [out]

mlir.register_lowering(dynamic_slice_p, _dynamic_slice_lower)

# def _getslice_lower(ctx, x, lo, hi):
#   aval_out, = ctx.avals_out
#   return hlo.RealDynamicSliceOp(
#       mlir.aval_to_ir_type(aval_out), x,
#       mlir.shape_tensor([lo]), mlir.shape_tensor([hi]), mlir.shape_tensor([1])
#   ).results
# mlir.register_lowering(getslice_p, _getslice_lower)


def _dynamic_update_slice_shape_rule(operand, update, *start_indices):
  if operand.ndim != update.ndim:
    msg = ("dynamic_update_slice update must have the same rank as operand, "
           "got update shape {} for operand shape {}.")
    raise TypeError(msg.format(update.shape, operand.shape))
  if operand.ndim != len(start_indices):
    msg = ("dynamic_update_slice start_indices must have length equal to the "
           "rank of operand, got indices {} for operand shape {}.")
    raise TypeError(msg.format(start_indices, operand.shape))
  if not all(map(operator.ge, operand.shape, update.shape)):
    msg = ("dynamic_update_slice update shape must be smaller than operand "
           "shape, got update shape {} for operand shape {}.")
    raise TypeError(msg.format(update.shape, operand.shape))
  if any(idx.ndim != 0 for idx in start_indices):
    raise TypeError("start_indices arguments to dynamic_update_slice must be "
                    f"scalars, got indices {start_indices}")
  return operand.shape

def _dynamic_update_slice_sharding_rule(operand, update, *start_indices):
  if operand.sharding != update.sharding:
    raise TypeError(
        "dynamic_update_slice update sharding must be equal to operand"
        f" sharding, got update sharding {update.sharding} for operand sharding"
        f" {operand.sharding}.")
  return operand.sharding

def _dynamic_update_slice_dtype_rule(operand, update, *start_indices):
  lax.check_same_dtypes("dynamic_update_slice", operand, update)
  if any(i.dtype != start_indices[0].dtype or
         not dtypes.issubdtype(i.dtype, np.integer) for i in start_indices):
    msg = ("index arguments to dynamic_update_slice must be integers of the "
           "same type, got {}")
    raise TypeError(msg.format(", ".join(i.dtype.name for i in start_indices)))
  return operand.dtype

def _dynamic_update_slice_jvp(primals, tangents):
  operand, update = primals[:2]
  start_indices = primals[2:]
  g_operand, g_update = tangents[:2]
  val_out = dynamic_update_slice_p.bind(operand, update, *start_indices)
  if type(g_operand) is ad_util.Zero and type(g_update) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_primal_value(val_out)
  else:
    g_operand = ad.instantiate_zeros(g_operand)
    g_update = ad.instantiate_zeros(g_update)
    tangent_out = dynamic_update_slice_p.bind(g_operand, g_update, *start_indices)
  return val_out, tangent_out

def _dynamic_update_slice_transpose_rule(t, operand, update, *start_indices):
  assert all(not ad.is_undefined_primal(x) for x in start_indices)
  if ad.is_undefined_primal(update):
    update_shape = update.aval.shape
  else:
    update_shape = update.shape
  if type(t) is ad_util.Zero:
    operand_t = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    update_t = ad_util.Zero(update.aval) if ad.is_undefined_primal(update) else None
  else:
    dus = dynamic_update_slice_p.bind
    ds = dynamic_slice_p.bind
    zeros = lax._zeros(t, shape=update_shape)
    operand_t = dus(t, zeros, *start_indices) if ad.is_undefined_primal(operand) else None
    update_t = ds(t, *start_indices, slice_sizes=update_shape) if ad.is_undefined_primal(update) else None
  return [operand_t, update_t] + [None] * len(start_indices)

def _dynamic_update_slice_batching_rule(batched_args, batch_dims):
  # A dynamic update slice is a special case of scatter; we can delegate to the
  # scatter batching rule.
  # TODO(phawkins): consider removing dynamic_update_slice entirely and using
  # scatter always.
  operand, update, *start_idx = batched_args
  operand_bd, update_bd, *start_idx_bd = batch_dims
  update_shape = (np.shape(update) if update_bd is batching.not_mapped
                  else tuple(np.delete(np.shape(update), update_bd)))
  dims = tuple(range(len(update_shape)))
  dnums = ScatterDimensionNumbers(
      update_window_dims=dims,
      inserted_window_dims=(),
      scatter_dims_to_operand_dims=dims,
  )
  index, index_bdim = _batch_dynamic_slice_indices(start_idx, start_idx_bd)
  return api.vmap(
    partial(scatter, dimension_numbers=dnums,
            indices_are_sorted=True, unique_indices=True,
            mode=GatherScatterMode.CLIP),
    in_axes=(operand_bd, index_bdim, update_bd),
    out_axes=0)(operand, index, update), 0


dynamic_update_slice_p = standard_primitive(
    _dynamic_update_slice_shape_rule, _dynamic_update_slice_dtype_rule,
    'dynamic_update_slice', sharding_rule=_dynamic_update_slice_sharding_rule)
ad.primitive_jvps[dynamic_update_slice_p] = _dynamic_update_slice_jvp
ad.primitive_transposes[dynamic_update_slice_p] = \
    _dynamic_update_slice_transpose_rule
batching.primitive_batchers[dynamic_update_slice_p] = \
    _dynamic_update_slice_batching_rule

def _dynamic_update_slice_lower(ctx, x, update, *start_indices):
  aval_out, = ctx.avals_out
  out = mlir.dynamic_update_slice(ctx, aval_out, x, update,
                                  start_indices=start_indices)
  if config.sharding_in_types.value:
    return [mlir.lower_sharding_under_shit(ctx, out, aval_out)]
  return [out]

mlir.register_lowering(dynamic_update_slice_p, _dynamic_update_slice_lower)


def _gather_dtype_rule(operand, indices, *, fill_value, **kwargs):
  if not dtypes.issubdtype(indices.dtype, np.integer):
    raise ValueError("indices must have an integer type")
  return dtypes.canonicalize_dtype(operand.dtype, allow_extended_dtype=True)

_rank = lambda arr: len(arr.shape)

def _is_sorted(dims, op_name, name):
  for i in range(1, len(dims)):
    if dims[i] < dims[i - 1]:
      raise TypeError(f"{name} in {op_name} op must be sorted; got {dims}")

def _dims_in_range(dims, rank, op_name, name):
  for dim in dims:
    if dim < 0 or dim >= rank:
      raise TypeError(f"Invalid {name} set in {op_name} op; valid range is "
                      f"[0, {rank}); got: {dim}.")

def _sorted_dims_in_range(dims, rank, op_name, name):
  if len(dims) == 0:
    return
  invalid_dim = None
  if dims[0] < 0:
    invalid_dim = dims[0]
  elif dims[-1] >= rank:
    invalid_dim = dims[-1]
  if invalid_dim:
    raise TypeError(f"Invalid {name} set in {op_name} op; valid range is "
                    f"[0, {rank}); got: {invalid_dim}.")

def _no_duplicate_dims(dims, op_name, name):
  if len(set(dims)) != len(dims):
    raise TypeError(f"{name} in {op_name} op must not repeat; got: {dims}.")

def _disjoint_dims(dims1, dims2, op_name, name1, name2):
  if not set(dims1).isdisjoint(set(dims2)):
    raise TypeError(f"{name1} and {name2} in {op_name} op must be disjoint; "
                    f"got: {dims1} and {dims2}.")

def _gather_shape_rule(operand, indices, *, dimension_numbers,
                       slice_sizes, unique_indices, indices_are_sorted,
                       mode, fill_value):
  """Validates the well-formedness of the arguments to Gather.

  The code implements the checks based on the detailed operation semantics of
  XLA's `Gather <https://www.tensorflow.org/xla/operation_semantics#gather>`_
  operator and following the outline of the implementation of
  ShapeInference::InferGatherShape in TensorFlow.
  """

  offset_dims = dimension_numbers.offset_dims
  collapsed_slice_dims = dimension_numbers.collapsed_slice_dims
  operand_batching_dims = dimension_numbers.operand_batching_dims
  start_indices_batching_dims = dimension_numbers.start_indices_batching_dims
  start_index_map = dimension_numbers.start_index_map

  # Note: in JAX, index_vector_dim is always computed as below, cf. the
  # documentation of the GatherDimensionNumbers class.
  index_vector_dim = _rank(indices) - 1

  # This case should never happen in JAX, due to the implicit construction of
  # index_vector_dim, but is included for completeness.
  if _rank(indices) < index_vector_dim or index_vector_dim < 0:
    raise TypeError(f"Gather index leaf dimension must be within [0, rank("
                    f"indices) + 1). rank(indices) is {_rank(indices)} and "
                    f"gather index leaf dimension is {index_vector_dim}.")

  # Start ValidateGatherDimensions
  # In the error messages output by XLA, "offset_dims" is called "Output window
  # dimensions" in error messages. For consistency's sake, our error messages
  # stick to "offset_dims".
  _is_sorted(offset_dims, "gather", "offset_dims")
  _no_duplicate_dims(offset_dims, "gather", "offset_dims")

  output_offset_dim_count = len(offset_dims)
  output_shape_rank = len(offset_dims) + _rank(indices) - 1

  for i in range(output_offset_dim_count):
    offset_dim = offset_dims[i]
    if offset_dim < 0 or offset_dim >= output_shape_rank:
      raise TypeError(f"Offset dimension {i} in gather op is out of bounds; "
                      f"got {offset_dim}, but should have been in "
                      f"[0, {output_shape_rank})")

  if len(start_index_map) != indices.shape[index_vector_dim]:
    raise TypeError(f"Gather op has {len(start_index_map)} elements in "
                    f"start_index_map and the bound of dimension "
                    f"{index_vector_dim=} of indices is "
                    f"{indices.shape[index_vector_dim]}. These two "
                    f"numbers must be equal.")

  for i in range(len(start_index_map)):
    operand_dim_for_start_index_i = start_index_map[i]
    if (operand_dim_for_start_index_i < 0 or
        operand_dim_for_start_index_i >= _rank(operand)):
      raise TypeError(f"Invalid start_index_map; domain is "
                      f"[0, {_rank(operand)}), got: "
                      f"{i}->{operand_dim_for_start_index_i}.")

  _no_duplicate_dims(start_index_map, "gather", "start_index_map")

  # _is_sorted and _sorted_dims_in_range are checked in the opposite order
  # compared to the XLA implementation. In cases when the input is not sorted
  # AND there are problematic collapsed_slice_dims, the error message will thus
  # be different.
  _is_sorted(collapsed_slice_dims, "gather", "collapsed_slice_dims")
  _sorted_dims_in_range(collapsed_slice_dims, _rank(operand), "gather",
                        "collapsed_slice_dims")
  _no_duplicate_dims(collapsed_slice_dims, "gather", "collapsed_slice_dims")

  _no_duplicate_dims(operand_batching_dims, "gather", "operand_batching_dims")
  _is_sorted(operand_batching_dims, "gather", "operand_batching_dims")
  _sorted_dims_in_range(
      operand_batching_dims, _rank(operand), "gather", "operand_batching_dims"
  )

  _disjoint_dims(collapsed_slice_dims, operand_batching_dims, "gather",
                 "collapsed_slice_dims", "operand_batching_dims")
  _disjoint_dims(start_index_map, operand_batching_dims, "gather",
                 "start_index_map", "operand_batching_dims")

  _no_duplicate_dims(
      start_indices_batching_dims, "gather", "start_indices_batching_dims"
  )
  _dims_in_range(
      start_indices_batching_dims,
      _rank(indices),
      "gather",
      "start_indices_batching_dims",
  )
  if index_vector_dim in start_indices_batching_dims:
    raise TypeError(
        "Gather op cannot have the index vector dimension as a batching "
        f"dimension; got {start_indices_batching_dims}."
    )

  if len(operand_batching_dims) != len(start_indices_batching_dims):
    raise TypeError(
        "Gather op requires equal numbers of operand_batching_dims and "
        f"start_indices_batching_dims, got {operand_batching_dims} and"
        f"{start_indices_batching_dims}."
    )

  operand_batch_shape = tuple(operand.shape[i] for i in operand_batching_dims)
  indices_batch_shape = tuple(
      indices.shape[i] for i in start_indices_batching_dims
  )
  if not core.definitely_equal_shape(operand_batch_shape, indices_batch_shape):
    raise TypeError(
        "Gather op requires operand batching dimensions and indices batching "
        f"dimensions to have the same shape, got {operand_batch_shape} and "
        f"{indices_batch_shape}."
    )
  # End ValidateGatherDimensions

  if _rank(operand) != len(slice_sizes):
    raise TypeError(f"Gather op must have one slice size for every input "
                    f"dimension; got: len(slice_sizes)={len(slice_sizes)}, "
                    f"input_shape.rank={_rank(operand)}")

  if len(slice_sizes) != len(offset_dims) + len(collapsed_slice_dims) + len(
      operand_batching_dims
  ):
    raise TypeError(
        "All components of the offset index in a gather op must "
        "either be a offset dimension or explicitly collapsed/batching; "
        f"got len(slice_sizes)={len(slice_sizes)}, "
        f"output_slice_sizes={offset_dims}, collapsed_slice_dims="
        f"{collapsed_slice_dims}, operand_batching_dims="
        f"{operand_batching_dims}."
    )

  for i in range(len(slice_sizes)):
    slice_size = slice_sizes[i]
    corresponding_input_size = operand.shape[i]

    if not (slice_size >= 0 and
            corresponding_input_size >= slice_size):
      raise TypeError(f"Slice size at index {i} in gather op is out of range, "
                      f"must be within [0, {corresponding_input_size} + 1), "
                      f"got {slice_size}.")

  for i in range(len(collapsed_slice_dims)):
    bound = slice_sizes[collapsed_slice_dims[i]]
    if bound != 1:
      raise TypeError(f"Gather op can only collapse slice dims with bound 1, "
                      f"but bound is {bound} for index "
                      f"{collapsed_slice_dims[i]} at position {i}.")

  for i in range(len(operand_batching_dims)):
    bound = slice_sizes[operand_batching_dims[i]]
    if bound > 1:
      raise TypeError(f"Gather op can only have operand batching dims with "
                      f"bound 0/1, but bound is {bound} for index "
                      f"{operand_batching_dims[i]} at position {i}."
      )

  return _gather_shape_computation(indices, dimension_numbers, slice_sizes)


def _gather_shape_computation(indices, dimension_numbers, slice_sizes):
  offset_dims = dimension_numbers.offset_dims
  collapsed_slice_dims = dimension_numbers.collapsed_slice_dims
  operand_batching_dims = dimension_numbers.operand_batching_dims
  output_shape_rank = len(offset_dims) + _rank(indices) - 1

  index_vector_dim = _rank(indices) - 1
  expanded_indices_shape = list(indices.shape)

  # This case should never happen in JAX, due to the implicit construction of
  # index_vector_dim, but is included for completeness.
  if len(expanded_indices_shape) == index_vector_dim:
    expanded_indices_shape.append(1)

  expanded_indices_shape.pop(index_vector_dim)

  indices_shape_gen = iter(expanded_indices_shape)

  slice_sizes_gen = (
      s
      for i, s in enumerate(slice_sizes)
      if i not in collapsed_slice_dims and i not in operand_batching_dims
  )
  ans = tuple(next(slice_sizes_gen) if i in offset_dims
              else next(indices_shape_gen) for i in range(output_shape_rank))
  return ans

class GatherShardingError(Exception):
  pass

def _gather_sharding_rule(operand, indices, *, dimension_numbers,
                          slice_sizes, unique_indices, indices_are_sorted,
                          mode, fill_value):
  # TODO(yashkatariya): Write a proper gather sharding rule.
  if mesh_lib.get_abstract_mesh()._are_all_axes_hidden:  # type: ignore
    return None
  raise GatherShardingError(
      "Use `.at[...].get(out_sharding=)` to provide output PartitionSpec for"
      " the gather indexing.")

def _gather_fill(operand, indices, *, dimension_numbers, slice_sizes,
                 unique_indices, indices_are_sorted, fill_value,
                 output_shape):
  """Lowers a FILL_OR_DROP gather as a PROMISE_IN_BOUNDS gather with masking."""
  dnums = dimension_numbers
  intarray = partial(np.array, dtype=np.int64)
  operand_dims = lax.shape_as_value(operand.shape)
  indices = lax.convert_element_type(indices, np.int64)
  num_batch_dims = len(indices.shape) - 1

  upper_bound = (
      operand_dims[intarray(dnums.start_index_map)] -
      lax.shape_as_value(slice_sizes)[intarray(dnums.start_index_map)])
  mask = lax.bitwise_and(
      lax.ge(indices, np.int64(0)),
      lax.le(indices, lax.expand_dims(upper_bound, tuple(range(num_batch_dims)))))
  mask = lax._reduce_and(mask, [num_batch_dims])

  # Computes the output shape and the positions of the batch dimensions in the
  # output
  output_ndims = num_batch_dims + len(dnums.offset_dims)
  batch_dims_in_output = np.delete(np.arange(output_ndims),
                                   dnums.offset_dims)

  # We don't consume unique_indices directly in gather(), only in its transpose
  # (scatter).
  gather_out = gather(operand, indices, dnums, slice_sizes,
                      indices_are_sorted=indices_are_sorted,
                      mode=GatherScatterMode.PROMISE_IN_BOUNDS)
  return lax.select(
    lax.broadcast_in_dim(mask, output_shape, batch_dims_in_output),
    gather_out, lax.full_like(gather_out, fill_value=fill_value))


def _gather_jvp_rule(g, operand, indices, *, dimension_numbers,
                     slice_sizes, unique_indices, indices_are_sorted, mode,
                     fill_value):
  return gather(g, indices, dimension_numbers, slice_sizes,
                unique_indices=unique_indices,
                indices_are_sorted=indices_are_sorted, mode=mode,
                fill_value=0)

def _gather_transpose_rule(t, operand, indices, *, dimension_numbers,
                           slice_sizes, unique_indices, indices_are_sorted,
                           mode, fill_value):
  assert ad.is_undefined_primal(operand)
  operand_shape = operand.aval.shape
  if type(t) is ad_util.Zero:
    out = ad_util.Zero(operand.aval)
  else:
    zeros = lax.full(operand_shape, lax._zero(t))
    scatter_dnums = ScatterDimensionNumbers(
        update_window_dims=dimension_numbers.offset_dims,
        inserted_window_dims=dimension_numbers.collapsed_slice_dims,
        scatter_dims_to_operand_dims=dimension_numbers.start_index_map,
        operand_batching_dims=dimension_numbers.operand_batching_dims,
        scatter_indices_batching_dims=dimension_numbers.start_indices_batching_dims,
    )
    out = scatter_add(zeros, indices, t, scatter_dnums,
                      unique_indices=unique_indices,
                      indices_are_sorted=indices_are_sorted,
                      mode=mode)
  return [out, None]

def _gather_batching_rule(batched_args, batch_dims, *, dimension_numbers,
                          slice_sizes, unique_indices, indices_are_sorted,
                          mode, fill_value):
  operand, indices, *dyn_slice_sizes = batched_args
  operand_bdim, indices_bdim, *dyn_slice_size_bds = batch_dims
  dyn_slice_size_bounds = [b.dtype.bound for b in dyn_slice_sizes]

  if operand_bdim is not None and indices_bdim is None:
    operand, operand_bdim = batching.move_stacked_axis(operand, operand_bdim, 0)
    slice_sizes = (operand.shape[0],) + slice_sizes
    offset_dims = (0,) + tuple(np.add(1, dimension_numbers.offset_dims))
    collapsed_slice_dims = tuple(np.add(1, dimension_numbers.collapsed_slice_dims))
    operand_batching_dims = tuple(
        np.add(1, dimension_numbers.operand_batching_dims)
    )
    start_index_map = tuple(np.add(1, dimension_numbers.start_index_map))
    dnums = GatherDimensionNumbers(
        offset_dims=offset_dims,
        collapsed_slice_dims=collapsed_slice_dims,
        start_index_map=start_index_map,
        operand_batching_dims=operand_batching_dims,
        start_indices_batching_dims=dimension_numbers.start_indices_batching_dims,
    )
    if isinstance(operand_bdim, batching.RaggedAxis):
      ragged_slice_sizes = batching.bdim_as_shape(operand_bdim, slice_sizes)
      for orig, fabricated in zip(
          lax._merge_dyn_shape(slice_sizes, dyn_slice_sizes),
          ragged_slice_sizes):
        if isinstance(fabricated, batching.IndexedAxisSize):
          if not core.same_referent(orig, fabricated.lengths):
            # Don't know what to do when slicing a ragged dimension with a
            # different size.  To wit, if the client tries to index outside the
            # ragged size, the resulting element should be determined by the
            # out of bounds `mode`, but the underlying gather will only do that
            # if the client tries to index outside the _padded_ array.  I guess
            # we should read the mode and apply a mask that writes the correct
            # fill element into all out-of-bounds locations?
            raise NotImplementedError
      bdim_out = batching.shape_as_bdim(
          operand_bdim.stacked_axis,
          _gather_shape_computation(indices, dnums, ragged_slice_sizes))
    else:
      bdim_out = operand_bdim
    return gather(
        operand, indices, dimension_numbers=dnums,
        slice_sizes=lax._merge_dyn_shape(slice_sizes, dyn_slice_size_bounds),
        unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted, mode=mode,
        fill_value=fill_value), bdim_out

  elif operand_bdim is None and indices_bdim is not None:
    indices = batching.moveaxis(indices, indices_bdim, 0)
    offset_dims = tuple(1 + d for d in dimension_numbers.offset_dims)
    start_indices_batching_dims = tuple(
        np.add(1, dimension_numbers.start_indices_batching_dims)
    )
    dnums = GatherDimensionNumbers(
        offset_dims=offset_dims,
        collapsed_slice_dims=dimension_numbers.collapsed_slice_dims,
        start_index_map=dimension_numbers.start_index_map,
        operand_batching_dims=dimension_numbers.operand_batching_dims,
        start_indices_batching_dims=start_indices_batching_dims,
    )
    # If batching indexed accesses into the same array, the batched gather may
    # no longer have sorted or unique indices.
    return gather(operand, indices, dimension_numbers=dnums,
                  slice_sizes=slice_sizes, unique_indices=False,
                  indices_are_sorted=False, mode=mode, fill_value=fill_value), 0

  else:
    # move batch dimensions to the front to simplify logic
    operand = batching.moveaxis(operand, operand_bdim, 0)
    indices = batching.moveaxis(indices, indices_bdim, 0)

    if core.definitely_equal(operand.shape[0], 0):
      slice_sizes = (0,) + slice_sizes
    else:
      slice_sizes = (1,) + slice_sizes
    collapsed_slice_dims = tuple(
        np.add(1, dimension_numbers.collapsed_slice_dims)
    )
    operand_batching_dims = (0,) + tuple(
        np.add(1, dimension_numbers.operand_batching_dims)
    )
    start_indices_batching_dims = (0,) + tuple(
        np.add(1, dimension_numbers.start_indices_batching_dims)
    )
    offset_dims = tuple(np.add(1, dimension_numbers.offset_dims))
    start_index_map = tuple(np.add(1, dimension_numbers.start_index_map))

    dnums = GatherDimensionNumbers(
        offset_dims=offset_dims,
        collapsed_slice_dims=collapsed_slice_dims,
        start_index_map=start_index_map,
        operand_batching_dims=operand_batching_dims,
        start_indices_batching_dims=start_indices_batching_dims,
    )
    return gather(operand, indices, dimension_numbers=dnums,
                  slice_sizes=slice_sizes, unique_indices=unique_indices,
                  indices_are_sorted=indices_are_sorted, mode=mode,
                  fill_value=fill_value), 0

def _gather_pad_rule(in_avals, out_avals, operand, indices, *,
                     dimension_numbers, slice_sizes, unique_indices,
                     indices_are_sorted, mode, fill_value):
  operand_aval, indices_aval = in_avals
  if any(isinstance(d, pe.BoundedAxisSize) for d in operand_aval.shape):
    raise NotImplementedError
  if mode != GatherScatterMode.PROMISE_IN_BOUNDS:
    # with fill, jnp.where on operand; with clip, jnp.where on indices
    raise NotImplementedError
  return [gather(operand, indices, dimension_numbers=dimension_numbers,
                 slice_sizes=slice_sizes, mode=mode, fill_value=fill_value)]

gather_p = standard_primitive(
    _gather_shape_rule, _gather_dtype_rule, 'gather',
    weak_type_rule=_argnum_weak_type(0), sharding_rule=_gather_sharding_rule)
ad.defjvp(gather_p, _gather_jvp_rule, None)
ad.primitive_transposes[gather_p] = _gather_transpose_rule
batching.primitive_batchers[gather_p] = _gather_batching_rule
pe.padding_rules[gather_p] = _gather_pad_rule


def _gather_lower_opaque(ctx, operand, indices, *,
                         dimension_numbers, slice_sizes, unique_indices,
                         indices_are_sorted, mode, fill_value) -> ir.Value:
  aval_x, aval_indices = ctx.avals_in
  aval_y, = ctx.avals_out
  elt_shape = core.physical_element_aval(aval_x.dtype).shape
  trailing_offset_dims = [aval_y.ndim + i for i in range(len(elt_shape))]
  dimension_numbers = dimension_numbers._replace(
      offset_dims=(*dimension_numbers.offset_dims, *trailing_offset_dims))
  slice_sizes = (*slice_sizes, *elt_shape)
  gather_lower = partial(
      _gather_lower, dimension_numbers=dimension_numbers,
      slice_sizes=slice_sizes, unique_indices=unique_indices,
      indices_are_sorted=indices_are_sorted, mode=mode, fill_value=fill_value)
  res, = mlir.delegate_lowering(
      ctx, gather_lower, operand, indices,
      avals_in=[core.physical_aval(aval_x), aval_indices],
      avals_out=[core.physical_aval(aval_y)])
  return res

def _gather_lower(ctx, operand, indices, *,
                  dimension_numbers, slice_sizes, unique_indices,
                  indices_are_sorted, mode, fill_value):
  aval_out, = ctx.avals_out
  if dtypes.issubdtype(aval_out.dtype, dtypes.extended):
    return [_gather_lower_opaque(
        ctx, operand, indices, dimension_numbers=dimension_numbers,
        slice_sizes=slice_sizes, unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted, mode=mode,
        fill_value=fill_value)]

  if mode == GatherScatterMode.FILL_OR_DROP:
    gather_fill_fn = mlir.lower_fun(_gather_fill, multiple_results=False)
    return gather_fill_fn(
        ctx, operand, indices,
        dimension_numbers=dimension_numbers, slice_sizes=slice_sizes,
        unique_indices=unique_indices, indices_are_sorted=indices_are_sorted,
        fill_value=fill_value, output_shape=aval_out.shape)

  assert mode in (GatherScatterMode.PROMISE_IN_BOUNDS,
                  GatherScatterMode.CLIP), mode
  dnums = hlo.GatherDimensionNumbers.get(
      collapsed_slice_dims=list(dimension_numbers.collapsed_slice_dims),
      operand_batching_dims=list(dimension_numbers.operand_batching_dims),
      start_indices_batching_dims=list(
          dimension_numbers.start_indices_batching_dims
      ),
      index_vector_dim=len(ctx.avals_in[1].shape) - 1,
      offset_dims=list(dimension_numbers.offset_dims),
      start_index_map=list(dimension_numbers.start_index_map),
  )
  if not core.is_constant_shape(slice_sizes):
    slice_sizes = mlir.eval_dynamic_shape_as_tensor(ctx, slice_sizes)
    # TODO(burmako): Fix overly conservative type inference of DynamicGatherOp.
    # For now use the build_generic so that we can specify the result type.
    # return hlo.DynamicGatherOp(
    #     operand, indices, mlir.shape_tensor(slice_sizes),
    #     dnums, indices_are_sorted=ir.BoolAttr.get(indices_are_sorted)).results
    results = [mlir.aval_to_ir_type(aval_out)]
    operands = [operand, indices, slice_sizes]
    attributes = {
        "dimension_numbers": dnums,
        "indices_are_sorted": ir.BoolAttr.get(indices_are_sorted)
    }
    return hlo.DynamicGatherOp.build_generic(
        results=results, operands=operands, attributes=attributes).results
  else:
    return [hlo.gather(
        operand,
        indices,
        dnums,
        mlir.dense_int_array(slice_sizes),
        indices_are_sorted=ir.BoolAttr.get(indices_are_sorted))]

mlir.register_lowering(gather_p, _gather_lower)

def _scatter_dtype_rule(operand, indices, updates, **kwargs):
  if not dtypes.issubdtype(indices.dtype, np.integer):
    raise ValueError("indices must have an integer type")
  lax.check_same_dtypes("scatter", operand, updates)
  return dtypes.canonicalize_dtype(operand.dtype, allow_extended_dtype=True)

def _scatter_shape_rule(operand, indices, updates, *, update_jaxpr,
                        update_consts, dimension_numbers, indices_are_sorted,
                        unique_indices, mode):
  """Validates the well-formedness of the ``dimension_numbers`` argument to
  Scatter.

  The code implements the checks based on the detailed operation semantics of
  XLA's `Scatter <https://www.tensorflow.org/xla/operation_semantics#scatter>`_
  operator and following the outline of the implementation of
  ShapeInference::InferScatterShape in TensorFlow.
  """

  update_window_dims = dimension_numbers.update_window_dims
  inserted_window_dims = dimension_numbers.inserted_window_dims
  operand_batching_dims = dimension_numbers.operand_batching_dims
  scatter_indices_batching_dims = dimension_numbers.scatter_indices_batching_dims
  scatter_dims_to_operand_dims = dimension_numbers.scatter_dims_to_operand_dims
  # Note: in JAX, index_vector_dim is always computed as below, cf. the
  # documentation of the ScatterDimensionNumbers class.
  index_vector_dim = _rank(indices) - 1

  # This case should never happen in JAX, due to the implicit construction of
  # index_vector_dim, but is included for completeness.
  if _rank(indices) < index_vector_dim or index_vector_dim < 0:
    raise TypeError(f"Scatter index leaf dimension must be within [0, "
                    f"rank(indices) + 1). rank(indices) is {_rank(indices)} "
                    f"and scatter index leaf dimension is {index_vector_dim}.")

  expanded_indices_shape = list(indices.shape)
  # This case should never happen in JAX, due to the implicit construction of
  # index_vector_dim, but is included for completeness.
  if len(expanded_indices_shape) == index_vector_dim:
    expanded_indices_shape.append(1)

  expected_updates_rank = (len(expanded_indices_shape) - 1 +
                           len(update_window_dims))

  if _rank(updates) != expected_updates_rank:
    raise TypeError(f"Updates tensor must be of rank {expected_updates_rank}; "
                    f"got {_rank(updates)}.")

  # Validate update_window_dims
  _is_sorted(update_window_dims, "scatter", "update_window_dims")
  _no_duplicate_dims(update_window_dims, "scatter", "update_window_dims")
  _sorted_dims_in_range(update_window_dims, _rank(updates), "scatter",
                        "update_window_dims")

  # Validate inserted_window_dims
  _is_sorted(inserted_window_dims, "scatter", "inserted_window_dims")
  _no_duplicate_dims(inserted_window_dims, "scatter", "inserted_window_dims")
  _sorted_dims_in_range(inserted_window_dims, _rank(operand), "scatter",
                        "inserted_window_dims")

  # Validate operand_batching_dims and scatter_indices_batching_dims
  _is_sorted(operand_batching_dims, "scatter", "operand_batching_dims")
  _no_duplicate_dims(operand_batching_dims, "scatter", "operand_batching_dims")
  _sorted_dims_in_range(
      operand_batching_dims, _rank(operand), "scatter", "operand_batching_dims"
  )
  _disjoint_dims(inserted_window_dims, operand_batching_dims, "scatter",
                 "inserted_window_dims", "operand_batching_dims")
  _disjoint_dims(scatter_dims_to_operand_dims, operand_batching_dims, "scatter",
                 "scatter_dims_to_operand_dims", "operand_batching_dims")

  _no_duplicate_dims(
      scatter_indices_batching_dims, "scatter", "scatter_indices_batching_dims"
  )
  _dims_in_range(
      scatter_indices_batching_dims,
      _rank(indices),
      "scatter",
      "scatter_indices_batching_dims",
  )
  if index_vector_dim in scatter_indices_batching_dims:
    raise TypeError(
        "Scatter op cannot have the index vector dimension as a batching "
        f"dimension; got {scatter_indices_batching_dims}.")

  if len(operand_batching_dims) != len(scatter_indices_batching_dims):
    raise TypeError(
        "Scatter op requires equal numbers of operand_batching_dims and "
        f"scatter_indices_batching_dims, got {operand_batching_dims} and "
        f"{scatter_indices_batching_dims}."
    )

  operand_batch_shape = tuple(operand.shape[i] for i in operand_batching_dims)
  indices_batch_shape = tuple(
      indices.shape[i] for i in scatter_indices_batching_dims
  )
  if not core.definitely_equal_shape(operand_batch_shape, indices_batch_shape):
    raise TypeError(
        "Scatter op requires operand batching dimensions and indices batching "
        f"dimensions to have the same shape, got {operand_batch_shape} and "
        f"{indices_batch_shape}."
    )

  # Validate window_size
  window_size = (
      len(update_window_dims) +
      len(inserted_window_dims) +
      len(operand_batching_dims)
  )
  if _rank(operand) != window_size:
    raise TypeError(f"Scatter op has window of size {window_size}; doesn't "
                    f"match operand of rank {_rank(operand)}.")

  # Validate scatter_dims_to_operand_dims
  if (len(scatter_dims_to_operand_dims) !=
      indices.shape[index_vector_dim]):
    raise TypeError(f"Scatter op has {len(scatter_dims_to_operand_dims)} "
                    f"elements in scatter_dims_to_operand_dims and the bound "
                    f"of dimension {index_vector_dim=} of "
                    f"indices is {indices.shape[index_vector_dim]}. These two "
                    f"numbers must be equal")

  for i in range(len(scatter_dims_to_operand_dims)):
    dim = scatter_dims_to_operand_dims[i]
    if dim < 0 or dim >= _rank(operand):
      raise TypeError(f"Invalid scatter_dims_to_operand_dims mapping; domain "
                      f"is [0, {_rank(operand)}), got: {i}->{dim}.")

  _no_duplicate_dims(scatter_dims_to_operand_dims, "scatter",
                     "scatter_dims_to_operand_dims")

  max_update_slice_sizes = [
      operand.shape[i]
      for i in range(len(operand.shape))
      if (
          i not in set(inserted_window_dims)
          and i not in set(operand_batching_dims)
      )
  ]

  for i in range(len(update_window_dims)):
    update_window_dim = update_window_dims[i]
    if max_update_slice_sizes[i] < updates.shape[update_window_dim]:
      raise TypeError(f"Bounds of the window dimensions of updates must not "
                      f"exceed the bounds of the corresponding dimensions of "
                      f"operand. For dimension {update_window_dim}, updates "
                      f"bound is {updates.shape[update_window_dim]}, operand "
                      f"bound is {max_update_slice_sizes[i]}.")

  update_scatter_dims = [dim for dim in range(_rank(updates)) if dim not in
                         set(update_window_dims)]

  scatter_dims_seen = 0
  for i in update_scatter_dims:
    if scatter_dims_seen == index_vector_dim:
      scatter_dims_seen += 1
    if not core.definitely_equal(updates.shape[i], expanded_indices_shape[scatter_dims_seen]):
      raise TypeError(f"Bounds of the scatter dimensions of updates must be "
                      f"the same as the bounds of the corresponding dimensions "
                      f"of scatter indices. For scatter dimension {i}, updates "
                      f"bound is {updates.shape[i]}, indices bound is "
                      f"{expanded_indices_shape[scatter_dims_seen]}.")
    scatter_dims_seen += 1

  return operand.shape


def _clamp_scatter_indices(operand, indices, updates, *, dnums):
  """Clamps `indices` to be in-range for a scatter."""
  slice_sizes = []
  pos = 0
  for i in range(len(operand.shape)):
    if i in dnums.inserted_window_dims or i in dnums.operand_batching_dims:
      slice_sizes.append(1)
    else:
      slice_sizes.append(updates.shape[dnums.update_window_dims[pos]])
      pos += 1

  upper_bounds: core.Shape = tuple(operand.shape[i] - slice_sizes[i]
                                   for i in dnums.scatter_dims_to_operand_dims)
  # Stack upper_bounds into a Array[n]
  upper_bound = lax.shape_as_value(upper_bounds)
  # This fix fails lax_test_no_jax_array
  upper_bound = lax.min(upper_bound,
                        lax.convert_element_type(np.uint64(np.iinfo(indices.dtype).max),
                                                  np.int64))

  upper_bound = lax.broadcast_in_dim(upper_bound, indices.shape,
                                     (len(indices.shape) - 1,))
  return lax.clamp(np.int64(0), lax.convert_element_type(indices, np.int64),
                   upper_bound)


def _scatter_addsub_jvp(
    prim,
    primals,
    tangents,
    *,
    update_jaxpr,
    update_consts,
    dimension_numbers,
    indices_are_sorted,
    unique_indices,
    mode,
):
  operand, indices, updates = primals
  g_operand, g_indices, g_updates = tangents
  del g_indices  # ignored
  val_out = prim.bind(
      operand,
      indices,
      updates,
      update_jaxpr=update_jaxpr,
      update_consts=update_consts,
      dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices,
      mode=mode,
  )
  if type(g_operand) is ad_util.Zero and type(g_updates) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_primal_value(val_out)
  else:
    g_operand = ad.instantiate_zeros(g_operand)
    g_updates = ad.instantiate_zeros(g_updates)
    tangent_out = prim.bind(
        g_operand,
        indices,
        g_updates,
        update_jaxpr=update_jaxpr,
        update_consts=update_consts,
        dimension_numbers=dimension_numbers,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=mode,
    )
  return val_out, tangent_out


def _scatter_addsub_transpose_rule(
    prim,
    t,
    operand,
    indices,
    updates,
    *,
    update_jaxpr,
    update_consts,
    dimension_numbers,
    indices_are_sorted,
    unique_indices,
    mode,
):
  assert not ad.is_undefined_primal(indices)
  if ad.is_undefined_primal(updates):
    updates_shape = updates.aval.shape
  else:
    updates_shape = updates.shape
  if type(t) is ad_util.Zero:
    operand_t = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    update_t = ad_util.Zero(updates.aval) if ad.is_undefined_primal(updates) else None
  else:
    operand_t = update_t = None
    if ad.is_undefined_primal(operand):
      operand_t = t

    if ad.is_undefined_primal(updates):
      gather_dnums = GatherDimensionNumbers(
          offset_dims=dimension_numbers.update_window_dims,
          collapsed_slice_dims=dimension_numbers.inserted_window_dims,
          start_index_map=dimension_numbers.scatter_dims_to_operand_dims,
          operand_batching_dims=dimension_numbers.operand_batching_dims,
          start_indices_batching_dims=dimension_numbers.scatter_indices_batching_dims,
      )
      slice_sizes = []
      pos = 0
      for i in range(len(t.shape)):
        if (
            i in dimension_numbers.inserted_window_dims
            or i in dimension_numbers.operand_batching_dims
        ):
          slice_sizes.append(1)
        else:
          slice_sizes.append(updates_shape[dimension_numbers.update_window_dims[pos]])
          pos += 1
      update_t = gather(t, indices, dimension_numbers=gather_dnums,
                        slice_sizes=slice_sizes, mode=mode, fill_value=0)
      if prim is scatter_sub_p:
        update_t = lax.neg(update_t)
  return [operand_t, None, update_t]

def _scatter_mul_transpose_rule(t, operand, indices, updates, *,
                                update_jaxpr, update_consts, dimension_numbers,
                                indices_are_sorted, unique_indices, mode):
  assert not ad.is_undefined_primal(indices)
  if ad.is_undefined_primal(updates):
    updates_shape = updates.aval.shape
  else:
    updates_shape = updates.shape
  if type(t) is ad_util.Zero:
    operand_t = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    update_t = ad_util.Zero(updates.aval) if ad.is_undefined_primal(updates) else None
  else:
    operand_t = update_t = None
    if ad.is_undefined_primal(operand):
      operand_t = scatter_mul(
          t, indices, updates, dimension_numbers=dimension_numbers,
          indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
          mode=mode)
    if ad.is_undefined_primal(updates):
      if not unique_indices:
        raise NotImplementedError(
          "scatter_mul gradients are only implemented if `unique_indices=True`")
      gather_dnums = GatherDimensionNumbers(
          offset_dims=dimension_numbers.update_window_dims,
          collapsed_slice_dims=dimension_numbers.inserted_window_dims,
          start_index_map=dimension_numbers.scatter_dims_to_operand_dims,
          operand_batching_dims=dimension_numbers.operand_batching_dims,
          start_indices_batching_dims=dimension_numbers.scatter_indices_batching_dims,
      )
      slice_sizes = []
      pos = 0
      for i in range(len(t.shape)):
        if (
            i in dimension_numbers.inserted_window_dims
            or i in dimension_numbers.operand_batching_dims
        ):
          slice_sizes.append(1)
        else:
          slice_sizes.append(updates_shape[dimension_numbers.update_window_dims[pos]])
          pos += 1
      update_t = gather(lax.mul(t, operand), indices,
                        dimension_numbers=gather_dnums, slice_sizes=slice_sizes,
                        mode=mode, fill_value=0)
  return [operand_t, None, update_t]


def _scatter_batching_rule(scatter_op, batched_args, batch_dims, *,
                           update_jaxpr, update_consts, dimension_numbers,
                           indices_are_sorted, unique_indices, mode):
  operand, indices, updates = batched_args
  operand_bdim, indices_bdim, updates_bdim = batch_dims

  # move the operand batch dim to the front if it is not None, otherwise create
  # it at the front (so that we can scatter into it)
  size = next(x.shape[ax] for x, ax in zip(batched_args, batch_dims)
              if ax is not None)
  operand = batching.bdim_at_front(operand, operand_bdim, size)

  updates = batching.bdim_at_front(updates, updates_bdim, size)

  if indices_bdim is None:
    inserted_window_dims = tuple(np.add(1, dimension_numbers.inserted_window_dims))
    update_window_dims = (0,) + tuple(np.add(1, dimension_numbers.update_window_dims))
    operand_batching_dims = tuple(
        np.add(1, dimension_numbers.operand_batching_dims)
    )
    scatter_dims_to_operand_dims = tuple(np.add(1, dimension_numbers.scatter_dims_to_operand_dims))
    dnums = ScatterDimensionNumbers(
        update_window_dims=update_window_dims,
        inserted_window_dims=inserted_window_dims,
        scatter_dims_to_operand_dims=scatter_dims_to_operand_dims,
        operand_batching_dims=operand_batching_dims,
        scatter_indices_batching_dims=dimension_numbers.scatter_indices_batching_dims,
    )
    return scatter_op.bind(
      operand, indices, updates, dimension_numbers=dnums,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=mode, update_jaxpr=update_jaxpr, update_consts=update_consts), 0

  # see the third case in _gather_batching_rule for comparison and comments
  indices = batching.bdim_at_front(indices, indices_bdim, size)

  update_window_dims = tuple(np.add(1, dimension_numbers.update_window_dims))
  inserted_window_dims = tuple(
      np.add(1, dimension_numbers.inserted_window_dims)
  )
  operand_batching_dims = (0,) + tuple(
      np.add(1, dimension_numbers.operand_batching_dims)
  )
  scatter_indices_batching_dims = (0,) + tuple(
      np.add(1, dimension_numbers.scatter_indices_batching_dims)
  )
  scatter_dims_to_operand_dims = tuple(
      np.add(1, dimension_numbers.scatter_dims_to_operand_dims)
  )

  dnums = ScatterDimensionNumbers(
      update_window_dims=update_window_dims,
      inserted_window_dims=inserted_window_dims,
      scatter_dims_to_operand_dims=scatter_dims_to_operand_dims,
      operand_batching_dims=operand_batching_dims,
      scatter_indices_batching_dims=scatter_indices_batching_dims,
  )
  return scatter_op.bind(
      operand, indices, updates, dimension_numbers=dnums,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=mode, update_jaxpr=update_jaxpr, update_consts=update_consts), 0

scatter_add_p = standard_primitive(
    _scatter_shape_rule, _scatter_dtype_rule, 'scatter-add',
    weak_type_rule=_argnum_weak_type(0))
ad.primitive_jvps[scatter_add_p] = partial(_scatter_addsub_jvp, scatter_add_p)
ad.primitive_transposes[scatter_add_p] = partial(_scatter_addsub_transpose_rule, scatter_add_p)
batching.primitive_batchers[scatter_add_p] = (
  partial(_scatter_batching_rule, scatter_add_p))

scatter_sub_p = standard_primitive(
    _scatter_shape_rule,
    _scatter_dtype_rule,
    "scatter-sub",
    weak_type_rule=_argnum_weak_type(0),
)
ad.primitive_jvps[scatter_sub_p] = partial(_scatter_addsub_jvp, scatter_sub_p)
ad.primitive_transposes[scatter_sub_p] = partial(_scatter_addsub_transpose_rule, scatter_sub_p)
batching.primitive_batchers[scatter_sub_p] = partial(
    _scatter_batching_rule, scatter_sub_p
)

scatter_mul_p = standard_primitive(
    _scatter_shape_rule, _scatter_dtype_rule, 'scatter-mul',
    weak_type_rule=_argnum_weak_type(0))

def _scatter_mul_jvp_rhs(g, x, i, y, *, dimension_numbers,
                         indices_are_sorted, unique_indices, mode, **kw):
  if not unique_indices:
    raise NotImplementedError(
      "scatter_mul gradients are only implemented if `unique_indices=True`")
  return lax.mul(x, scatter_add(
      ad_util.zeros_like_jaxval(x), i, g, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=mode))

ad.defjvp(scatter_mul_p,
          lambda g, x, i, y, **kw: scatter_mul_p.bind(g, i, y, **kw),
          None,
          _scatter_mul_jvp_rhs)
ad.primitive_transposes[scatter_mul_p] = _scatter_mul_transpose_rule
batching.primitive_batchers[scatter_mul_p] = (
  partial(_scatter_batching_rule, scatter_mul_p))

def _scatter_extremal_jvp(scatter_op, primals, tangents, update_jaxpr,
                          update_consts, dimension_numbers,
                          indices_are_sorted, unique_indices, mode):
  operand, indices, updates = primals
  g_operand, g_indices, g_updates = tangents

  scatter_dnums = dimension_numbers
  updates_shape = updates.shape

  val_out = scatter_op.bind(
      operand, indices, updates, update_jaxpr=update_jaxpr,
      update_consts=update_consts, dimension_numbers=scatter_dnums,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices, mode=mode)

  if type(g_operand) is ad_util.Zero and type(g_updates) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_primal_value(val_out)
  else:
    g_operand = ad.instantiate_zeros(g_operand)
    g_updates = ad.instantiate_zeros(g_updates)

    # gather_dnums and slice_sizes define the gather op that is the inverse of
    # the scatter op specified by scatter_dnums
    gather_dnums = GatherDimensionNumbers(
        offset_dims=scatter_dnums.update_window_dims,
        collapsed_slice_dims=scatter_dnums.inserted_window_dims,
        start_index_map=scatter_dnums.scatter_dims_to_operand_dims,
        operand_batching_dims=scatter_dnums.operand_batching_dims,
        start_indices_batching_dims=scatter_dnums.scatter_indices_batching_dims,
    )

    slice_sizes = []
    pos = 0
    for i in range(len(operand.shape)):
      if (
          i in scatter_dnums.inserted_window_dims
          or i in scatter_dnums.operand_batching_dims
      ):
        slice_sizes.append(1)
      else:
        slice_sizes.append(updates_shape[scatter_dnums.update_window_dims[pos]])
        pos += 1

    # For consistency with other max operations, if there are two or more values
    # in updates that are contending to replace the same index location, the
    # resulting tangent at that location will be the average of the associated
    # tangents for the values in updates.

    initial_vals = gather(
        operand, indices, gather_dnums, np.array(slice_sizes))

    target_vals = gather(
        val_out, indices, gather_dnums, np.array(slice_sizes))

    successful_updates = (updates == target_vals)
    retained_values = (initial_vals == target_vals)

    num_updates = gather(
        scatter_add(
            lax._zeros(operand), indices,
            lax.select(successful_updates, lax._ones(updates),
                       lax._zeros(updates)),
            scatter_dnums),
        indices,
        gather_dnums,
        np.array(slice_sizes))

    num_refs = gather(
        scatter_add(lax._zeros(operand),
                    indices,
                    lax._ones(updates),
                    scatter_dnums),
        indices,
        gather_dnums,
        np.array(slice_sizes))

    updates_normalizer = lax.select(retained_values,
                                    1.0 / (num_updates + 1),
                                    1.0 / num_updates)

    updates_coef = lax.select(successful_updates,
                              updates_normalizer,
                              lax._zeros(updates))

    operand_normalizer = lax.select(retained_values,
                                    1.0 / (num_updates + 1),
                                    lax._zeros(num_updates))

    operand_coef = (-1.0 + operand_normalizer) / num_refs

    # This can be simplified once scatter has transpose implemented
    target_tangents = gather(
        g_operand, indices, gather_dnums, np.array(slice_sizes))

    tangent_updates = (target_tangents * operand_coef +
                       g_updates * updates_coef)

    tangent_out = scatter_add(g_operand,
                              indices,
                              tangent_updates,
                              scatter_dnums,
                              indices_are_sorted=indices_are_sorted,
                              unique_indices=unique_indices,
                              mode=mode)

  return val_out, tangent_out

scatter_min_p = standard_primitive(
    _scatter_shape_rule, _scatter_dtype_rule, 'scatter-min',
    weak_type_rule=_argnum_weak_type(0))
batching.primitive_batchers[scatter_min_p] = (
  partial(_scatter_batching_rule, scatter_min_p))
ad.primitive_jvps[scatter_min_p] = partial(_scatter_extremal_jvp, scatter_min_p)

scatter_max_p = standard_primitive(
    _scatter_shape_rule, _scatter_dtype_rule, 'scatter-max',
    weak_type_rule=_argnum_weak_type(0))
batching.primitive_batchers[scatter_max_p] = (
  partial(_scatter_batching_rule, scatter_max_p))
ad.primitive_jvps[scatter_max_p] = partial(_scatter_extremal_jvp, scatter_max_p)

def _scatter_jvp(primals, tangents, *, update_jaxpr, update_consts,
                 dimension_numbers, indices_are_sorted, unique_indices,
                 mode):
  if update_jaxpr is not None:
    raise NotImplementedError("scatter_apply JVP not implemented")
  operand, indices, updates = primals
  g_operand, g_indices, g_updates = tangents
  dnums = dimension_numbers

  if type(g_operand) is ad_util.Zero and type(g_updates) is ad_util.Zero:
    val_out = scatter_p.bind(
      operand, indices, updates, update_jaxpr=update_jaxpr,
      update_consts=update_consts, dimension_numbers=dnums,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=mode)
    return val_out, ad_util.Zero.from_primal_value(val_out)

  g_operand = ad.instantiate_zeros(g_operand)
  g_updates = ad.instantiate_zeros(g_updates)

  if unique_indices:
    # If the user has promised that the updates don't overlap, we can use a much
    # simpler JVP.
    val_out = scatter_p.bind(
      operand, indices, updates, update_jaxpr=update_jaxpr,
      update_consts=update_consts, dimension_numbers=dnums,
      indices_are_sorted=indices_are_sorted, unique_indices=True, mode=mode)
    tangent_out = scatter_p.bind(
      g_operand, indices, g_updates, update_jaxpr=update_jaxpr,
      update_consts=update_consts, dimension_numbers=dnums,
      indices_are_sorted=indices_are_sorted, unique_indices=True, mode=mode)
    return val_out, tangent_out

  # If there are overlapping indices in the scatter, it is unspecified which
  # update "wins". So we use the following perhaps surprising scheme:
  # a) attach a positive ID to each update in updates, and perform the scatter
  #    on the IDs
  # b) perform the inverse gather on the scattered IDs (similar to
  #    _scatter_add_transpose).
  # c) use the gathered IDs to mask the primal and tangent values.
  # d) perform a scatter-add on the masked primal and tangent values. A benefit
  #    of using scatter-add here is that we don't need a `scatter` transpose
  #    rule.

  # a) attach a positive ID to each update in `updates`, and perform a scatter
  #    on the IDs.
  ids_shape = list(updates.shape)
  for update_dim in dnums.update_window_dims:
    ids_shape[update_dim] = 1
  num_ids = math.prod(ids_shape)
  if core.is_constant_dim(num_ids):
    id_dtype = np.uint32 if (num_ids + 1) < np.iinfo(np.uint32).max else np.uint64
  else:
    id_dtype = np.uint64
  update_ids = lax.add(lax.reshape(lax.iota(id_dtype, num_ids), ids_shape),
                       lax._ones(updates, dtype=id_dtype))

  scattered_ids = scatter(lax.full(operand.shape, 0, id_dtype),
                          indices, update_ids, dnums,
                          indices_are_sorted=indices_are_sorted,
                          unique_indices=unique_indices, mode=mode)

  # b) compute the inverse gather that "undoes" the scatter on the id values.
  gather_dnums = GatherDimensionNumbers(
      offset_dims=dnums.update_window_dims,
      collapsed_slice_dims=dnums.inserted_window_dims,
      start_index_map=dnums.scatter_dims_to_operand_dims,
      operand_batching_dims=dnums.operand_batching_dims,
      start_indices_batching_dims=dnums.scatter_indices_batching_dims,
  )
  slice_sizes = []
  pos = 0
  for i in range(len(scattered_ids.shape)):
    if i in dnums.inserted_window_dims or i in dnums.operand_batching_dims:
      slice_sizes.append(1)
    else:
      slice_sizes.append(updates.shape[dnums.update_window_dims[pos]])
      pos += 1
  gathered_update_ids = gather(scattered_ids, indices,
                               dimension_numbers=gather_dnums,
                               slice_sizes=slice_sizes)

  # c) mask off input elements that do not correspond to a primal output.
  masked_operand = lax.select(lax.eq(scattered_ids, lax._zeros(scattered_ids)),
                              operand, lax._zeros(operand))
  masked_updates = lax.select(lax.eq(update_ids,  gathered_update_ids),
                              updates, lax._zeros(updates))
  masked_g_operand = lax.select(lax.eq(scattered_ids, lax._zeros(scattered_ids)),
                                g_operand, lax._zeros(g_operand))
  masked_g_updates = lax.select(lax.eq(update_ids, gathered_update_ids),
                                g_updates, lax._zeros(g_updates))

  # d) perform scatter-adds to compute the primal and tangent outputs.
  val_out = scatter_add(masked_operand, indices, masked_updates,
                        dimension_numbers=dnums,
                        indices_are_sorted=indices_are_sorted,
                        unique_indices=unique_indices, mode=mode)
  tangent_out = scatter_add(masked_g_operand, indices, masked_g_updates,
                            dimension_numbers=dnums,
                            indices_are_sorted=indices_are_sorted,
                            unique_indices=unique_indices, mode=mode)
  return val_out, tangent_out

def _scatter_transpose_rule(t, operand, indices, updates, *,
                            update_jaxpr, update_consts, dimension_numbers,
                            indices_are_sorted, unique_indices, mode):
  if not unique_indices:
    raise NotImplementedError("scatter transpose is only implemented where "
                              "unique_indices=True")
  assert not ad.is_undefined_primal(indices)
  if ad.is_undefined_primal(updates):
    updates_shape = updates.aval.shape
  else:
    updates_shape = updates.shape
  if type(t) is ad_util.Zero:
    operand_t = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    update_t = ad_util.Zero(updates.aval) if ad.is_undefined_primal(updates) else None
  else:
    operand_t = update_t = None
    if ad.is_undefined_primal(operand):
      # Zero out gradient entries that correspond to updated indices.
      operand_t = scatter(t, indices, lax.full(updates_shape, 0, dtype=t.dtype),
                          dimension_numbers=dimension_numbers,
                          indices_are_sorted=indices_are_sorted,
                          unique_indices=True, mode=mode)

    if ad.is_undefined_primal(updates):
      gather_dnums = GatherDimensionNumbers(
          offset_dims=dimension_numbers.update_window_dims,
          collapsed_slice_dims=dimension_numbers.inserted_window_dims,
          start_index_map=dimension_numbers.scatter_dims_to_operand_dims,
          operand_batching_dims=dimension_numbers.operand_batching_dims,
          start_indices_batching_dims=dimension_numbers.scatter_indices_batching_dims,
      )
      slice_sizes = []
      pos = 0
      for i in range(len(t.shape)):
        if (
            i in dimension_numbers.inserted_window_dims
            or i in dimension_numbers.operand_batching_dims
        ):
          slice_sizes.append(1)
        else:
          slice_sizes.append(updates_shape[dimension_numbers.update_window_dims[pos]])
          pos += 1
      update_t = gather(t, indices, dimension_numbers=gather_dnums,
                        slice_sizes=slice_sizes, mode=mode,
                        fill_value=0)

  return [operand_t, None, update_t]

scatter_p = standard_primitive(
    _scatter_shape_rule, _scatter_dtype_rule, 'scatter',
    weak_type_rule=_argnum_weak_type(0))
ad.primitive_jvps[scatter_p] = _scatter_jvp
ad.primitive_transposes[scatter_p] = _scatter_transpose_rule
batching.primitive_batchers[scatter_p] = (
  partial(_scatter_batching_rule, scatter_p))


def _scatter_lower_opaque(ctx, operand, indices, updates, *,
                          update_jaxpr, update_consts, dimension_numbers,
                          unique_indices, indices_are_sorted, mode):
  aval_x, aval_indices, aval_updates = ctx.avals_in
  aval_y, = ctx.avals_out
  elt_shape = core.physical_element_aval(aval_x.dtype).shape
  trailing_window_dims = [aval_updates.ndim + i for i in range(len(elt_shape))]
  dimension_numbers = dimension_numbers._replace(
      update_window_dims=(*dimension_numbers.update_window_dims,
                          *trailing_window_dims))
  scatter_lower = partial(
      _scatter_lower, update_jaxpr=update_jaxpr, update_consts=update_consts,
      dimension_numbers=dimension_numbers, unique_indices=unique_indices,
      indices_are_sorted=indices_are_sorted, mode=mode)
  res, = mlir.delegate_lowering(
      ctx, scatter_lower, operand, indices, updates,
      avals_in=[core.physical_aval(aval_x), aval_indices,
                core.physical_aval(aval_updates)],
      avals_out=[core.physical_aval(aval_y)])
  return res


def _scatter_lower(ctx, operand, indices, updates, *,
                   update_jaxpr, update_consts, dimension_numbers,
                   indices_are_sorted, unique_indices, mode):
  if update_jaxpr is None:
    assert not update_consts
    operand_dtype = ctx.avals_in[0].dtype
    update_jaxpr, update_consts = lax._reduction_jaxpr(
        _scatter_reduction_computation, core.ShapedArray((), operand_dtype))

  aval_out, = ctx.avals_out
  if dtypes.issubdtype(aval_out.dtype, dtypes.extended):
    return [_scatter_lower_opaque(
        ctx, operand, indices, updates,
        update_jaxpr=update_jaxpr, update_consts=update_consts,
        dimension_numbers=dimension_numbers, unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted, mode=mode)]

  if mode == GatherScatterMode.CLIP:
    clip_fn = mlir.lower_fun(_clamp_scatter_indices, multiple_results=False)
    (indices,) = clip_fn(ctx.replace(avals_out=None), operand, indices,
                          updates, dnums=dimension_numbers)

  dnums = dimension_numbers
  scatter_dnums = hlo.ScatterDimensionNumbers.get(
      update_window_dims=list(dnums.update_window_dims),
      inserted_window_dims=list(dnums.inserted_window_dims),
      input_batching_dims=list(dnums.operand_batching_dims),
      scatter_indices_batching_dims=list(dnums.scatter_indices_batching_dims),
      scattered_dims_to_operand_dims=list(dnums.scatter_dims_to_operand_dims),
      index_vector_dim=len(ctx.avals_in[1].shape) - 1,
  )
  result = mlir.aval_to_ir_type(aval_out)
  operand = [operand]
  updates = [updates]
  op = hlo.ScatterOp(
      (result,),
      operand,
      indices,
      updates,
      scatter_dnums,
      indices_are_sorted=ir.BoolAttr.get(indices_are_sorted),
      unique_indices=ir.BoolAttr.get(unique_indices))
  scalar_type = mlir.aval_to_ir_type(core.ShapedArray((), aval_out.dtype))
  update = op.update_computation.blocks.append(scalar_type, scalar_type)
  with ir.InsertionPoint(update):
    name_stack = source_info_util.new_name_stack()
    if update_jaxpr.effects:
      raise NotImplementedError('Cannot lower effectful `scatter`.')
    out_nodes, _ = mlir.jaxpr_subcomp(
        ctx.module_context, update_jaxpr, name_stack, mlir.TokenSet(),
        update_consts, update.arguments[0], update.arguments[1],
        dim_var_values=ctx.dim_var_values)
    hlo.return_(mlir.flatten_ir_values(out_nodes))
  return op.results

mlir.register_lowering(scatter_p, _scatter_lower)
mlir.register_lowering(scatter_add_p, _scatter_lower)
mlir.register_lowering(scatter_sub_p, _scatter_lower)
mlir.register_lowering(scatter_mul_p, _scatter_lower)
mlir.register_lowering(scatter_min_p, _scatter_lower)
mlir.register_lowering(scatter_max_p, _scatter_lower)


def _real_dtype(dtype): return np.finfo(dtype).dtype


def _scatter_addsub_lower_gpu(
    ctx,
    operand,
    indices,
    updates,
    *,
    update_jaxpr,
    update_consts,
    dimension_numbers,
    indices_are_sorted,
    unique_indices,
    mode,
    reduce_op,
):
  operand_aval_in, _, updates_aval_in = ctx.avals_in
  if operand_aval_in.dtype != np.complex128:
    return _scatter_lower(ctx, operand, indices, updates,
                          update_jaxpr=update_jaxpr,
                          update_consts=update_consts,
                          dimension_numbers=dimension_numbers,
                          indices_are_sorted=indices_are_sorted,
                          unique_indices=unique_indices, mode=mode)

  if mode == GatherScatterMode.CLIP:
    clip_fn = mlir.lower_fun(_clamp_scatter_indices, multiple_results=False)
    indices, = clip_fn(ctx.replace(avals_out=None), operand, indices, updates,
                       dnums=dimension_numbers)

  aval_out, = ctx.avals_out
  dnums = dimension_numbers
  scatter_dnums = hlo.ScatterDimensionNumbers.get(
      update_window_dims=list(dnums.update_window_dims),
      inserted_window_dims=list(dnums.inserted_window_dims),
      input_batching_dims=list(dnums.operand_batching_dims),
      scatter_indices_batching_dims=list(dnums.scatter_indices_batching_dims),
      scattered_dims_to_operand_dims=list(dnums.scatter_dims_to_operand_dims),
      index_vector_dim=len(ctx.avals_in[1].shape) - 1,
  )
  real_dtype = _real_dtype(aval_out.dtype)
  operand_type_part = mlir.aval_to_ir_type(
      core.ShapedArray(aval_out.shape, real_dtype))

  def _scatter(operand_part, updates_part):
    operand_part = [operand_part]
    updates_part = [updates_part]

    scatter = hlo.ScatterOp(
        (operand_type_part,),
        operand_part,
        indices,
        updates_part,
        scatter_dnums,
        indices_are_sorted=ir.BoolAttr.get(indices_are_sorted),
        unique_indices=ir.BoolAttr.get(unique_indices))
    scalar_type = mlir.aval_to_ir_type(core.ShapedArray((), real_dtype))
    reducer = scatter.regions[0].blocks.append(scalar_type, scalar_type)
    with ir.InsertionPoint(reducer):
      hlo.return_([reduce_op(*reducer.arguments).result])
    return scatter.result

  real = _scatter(hlo.real(operand), hlo.real(updates))
  imag = _scatter(hlo.imag(operand), hlo.imag(updates))
  return [hlo.complex(real, imag)]


mlir.register_lowering(
    scatter_add_p,
    partial(_scatter_addsub_lower_gpu, reduce_op=hlo.AddOp),
    platform="gpu",
)
mlir.register_lowering(
    scatter_sub_p,
    partial(_scatter_addsub_lower_gpu, reduce_op=hlo.SubtractOp),
    platform="gpu",
)


def _dynamic_slice_indices(
    operand: Array | np.ndarray,
    start_indices: Array | np.ndarray | Sequence[ArrayLike]
  ) -> list[ArrayLike]:
  # Normalize the start_indices w.r.t. operand.shape
  if len(start_indices) != operand.ndim:
    msg = ("Length of slice indices must match number of operand dimensions ({} "
          "vs {})")
    raise ValueError(msg.format(len(start_indices), operand.shape))
  if not isinstance(start_indices, (tuple, list)):
    if start_indices.ndim != 1:  # type: ignore[union-attr]
      raise ValueError("Slice indices must be a 1D sequence, got {}"
                       .format(start_indices.shape))  # type: ignore[union-attr]
    start_indices = list(start_indices)
  result: list[ArrayLike] = []
  # Loop to correct for negative indices.
  for i, d in zip(start_indices, operand.shape):
    # If i is unsigned, then it cannot be negative.
    if dtypes.issubdtype(_dtype(i), np.unsignedinteger):
      result.append(i)
      continue
    # Test whether i and d are static to avoid unnecessary staging.
    if isinstance(i, (int, np.integer)) and core.is_constant_dim(d):
      result.append(lax.convert_element_type(i + d if i < 0 else i, _dtype(i)))
      continue
    d = core.dimension_as_value(d)
    if isinstance(i, (int, np.integer)):
      result.append(i + lax.convert_element_type(d, _dtype(i)) if i < 0 else i)
      continue
    d_arr = lax.convert_element_type(d, _dtype(i))
    result.append(lax.select(i < 0, i + d_arr, i))
  return result
