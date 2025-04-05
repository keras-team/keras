# Copyright 2023 The JAX Authors.
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

"""Contains shared logic and abstractions for Pallas indexing ops."""

from __future__ import annotations

import dataclasses
from typing import Any, Union

from jax._src import core
from jax._src import tree_util
from jax._src.typing import Array
from jax._src.util import merge_lists
from jax._src.util import partition_list
import numpy as np


@tree_util.register_pytree_node_class
@dataclasses.dataclass
class Slice:
  """A slice with a start index and a size.

  Both start index and size can either be static, i.e. known at tracing
  and compilation time, or dynamic.
  """

  start: int | Array
  size: int | Array
  stride: int = 1

  def __post_init__(self):
    if self.stride < 1:
      raise ValueError("`stride` must be >= 1.")

  @property
  def is_dynamic_start(self):
    return not core.is_dim(self.start)

  @property
  def is_dynamic_size(self):
    return not core.is_dim(self.size)

  def tree_flatten(self):
    # If `start` is statically known, we treat it as static information
    xs = ()
    data = ()
    xs += (self.start,) if self.is_dynamic_start else (None,)
    data += (None,) if self.is_dynamic_start else (self.start,)
    xs += (self.size,) if self.is_dynamic_size else (None,)
    data += (None,) if self.is_dynamic_size else (self.size,)
    data += (self.stride,)
    return xs, data

  @classmethod
  def tree_unflatten(cls, aux_data, children) -> Slice:
    start, size = (
        a if a is not None else b for a, b in zip(children, aux_data[:2])
    )
    return cls(start, size, aux_data[2])

  @classmethod
  def from_slice(cls, slc: slice, size: int) -> Slice:
    start, step, size = core.canonicalize_slice(slc, size)
    if step < 1:
      raise ValueError(f"slice must have a step >= 1 (found: {step})")
    return cls(start, size, step)


def dslice(
    start: int | Array | None,
    size: int | Array | None = None,
    stride: int | None = None,
) -> slice | Slice:
  """Constructs a ``Slice`` from a start index and a size.

  The semantics of ``dslice`` mirror those of the builtin ``slice`` type:

  * ``dslice(None)`` is ``:``
  * ``dslice(j)`` is ``:j``
  * ``dslice(i, j)`` is ``i:i+j``
  * ``dslice(i, j, stride)`` is ``i:i+j:stride``
  """
  if start is None:
    return slice(None)
  if stride is None:
    stride = 1
  if not isinstance(stride, int):
    raise ValueError("Non-static stride in `dslice`")
  if size is None:
    if not isinstance(start, int):
      raise ValueError("Non-static `dslice`")
    return Slice(0, start, stride)
  return Slice(start, size, stride)


ds = dslice  # Handy alias


IntIndexer = Union[int, Array]
DimIndexer = Union[IntIndexer, Slice]

def unpack_ndindexer(indexer: NDIndexer) -> tuple[tuple[bool, ...],
                                                  tuple[Slice, ...],
                                                  tuple[IntIndexer, ...]]:
  is_int_indexing = [not isinstance(i, Slice) for i in indexer.indices]
  slice_indexers, int_indexers = partition_list(
      is_int_indexing, indexer.indices)
  return tuple(is_int_indexing), tuple(slice_indexers), tuple(int_indexers)  # type: ignore

def _maybe_concretize(x: Any):
  # This is roughly the same logic as core.concrete_or_error, but we avoid
  # calling that because constructing the ConcretizationTypeError can be
  # expensive as the size of the tracing context (i.e. the jaxpr) grows.
  return core.to_concrete_value(x)

@tree_util.register_pytree_node_class
@dataclasses.dataclass
class NDIndexer:
  indices: tuple[DimIndexer, ...]
  shape: tuple[int, ...]
  int_indexer_shape: tuple[int, ...]
  # Off by default to avoid doing validation during pytree operations.
  validate: bool = False

  def __post_init__(self):
    if not self.validate:
      return
    if len(self.indices) != len(self.shape):
      raise ValueError(
          f"`indices` must be the same length as `Ref` shape.: {self}."
      )
    # We validate integer indexing shapes here
    for idx, s in zip(self.indices, self.shape):
      if isinstance(idx, Slice):
        start = idx.start
        if value := _maybe_concretize(start):
          if value >= s:
            raise ValueError(f"Out of bound slice: start={value}, dim={s}.")
          if size := _maybe_concretize(idx.size):
            if value + (size - 1) * idx.stride >= s:
              raise ValueError(
                  f"Out of bound slice: start={value}, size={size},"
                  f" stride={idx.stride}, dim={s}."
              )
        continue
      # The shape of indexer integers should be broadcastable up to the
      # int_indexer_shape of the whole NDIndexer
      if not np.shape(idx):
        if (value := _maybe_concretize(idx)) and value >= s:
          raise ValueError(f"Out of bound indexer: idx={value}, dim={s}.")
        # For ()-shaped indexers, we can broadcast no problm.
        continue
      # If we don't have a ()-shaped indexer, the rank must match
      # int_indexer_shape
      if np.ndim(idx) != len(self.int_indexer_shape):
        raise ValueError(
            f"Indexer must have rank {np.ndim(idx)}: {idx=} vs."
            f" {self.int_indexer_shape=}"
        )
      # Here we check that the shapes broadcast.
      try:
        np.broadcast_shapes(np.shape(idx), self.int_indexer_shape)
      except ValueError as e:
        raise ValueError(
            f"Could not broadcast integer indexer: {idx=} vs."
            f" {self.int_indexer_shape=}"
        ) from e

  @property
  def is_dynamic_size(self):
    return any(isinstance(i, Slice) and i.is_dynamic_size for i in self.indices)

  def tree_flatten(self):
    flat_idx, idx_tree = tree_util.tree_flatten(self.indices)
    return flat_idx, (idx_tree, self.shape, self.int_indexer_shape)

  @classmethod
  def tree_unflatten(cls, data, flat_idx):
    idx_tree, shape, int_indexer_shape = data
    indices = tree_util.tree_unflatten(idx_tree, flat_idx)
    return cls(tuple(indices), shape, int_indexer_shape)

  @classmethod
  def from_indices_shape(cls, indices, shape) -> NDIndexer:
    if not isinstance(indices, tuple):
      # TODO(slebedev): Consider requiring `indices` to be a Sequence.
      indices = (indices,)

    indices = list(indices)
    if num_ellipsis := sum(idx is ... for idx in indices):
      if num_ellipsis > 1:
        raise ValueError("Only one ellipsis is supported.")
      # Expand ... so that `indices` has the same length as `shape`.
      ip = indices.index(...)
      indices[ip:ip+1] = [slice(None)] * (len(shape) - len(indices) + 1)
    if len(indices) > len(shape):
      indices = tuple(indices)
      raise ValueError("`indices` must not be longer than `shape`: "
                       f"{indices=}, {shape=}")
    elif len(indices) < len(shape):
      # Pad `indices` to have the same length as `shape`.
      indices.extend([slice(None)] * (len(shape) - len(indices)))

    # Promote all builtin `slice`s to `Slice`.
    indices = tuple(
        Slice.from_slice(i, s) if isinstance(i, slice) else i
        for i, s in zip(indices, shape))

    is_int_indexing = [not isinstance(i, Slice) for i in indices]
    if any(is_int_indexing):
      other_indexers, int_indexers = partition_list(is_int_indexing, indices)
      indexer_shapes = tuple(core.get_aval(i).shape for i in int_indexers)
      try:
        int_indexer_shape = np.broadcast_shapes(*indexer_shapes)
      except ValueError as e:
        # Raise a nicer error than the NumPy one.
        raise ValueError(
            f"Cannot broadcast shapes for indexing: {indexer_shapes}") from e

      # Here we use the `broadcast_to` primitive instead of composing lax
      # primitives together because it is easier to lower in targets like
      # Triton/Mosaic.
      #
      # The local import avoids a circular dependency between primitives
      # and this module.
      from jax._src.state import primitives as sp  # pytype: disable=import-error
      int_indexers = [
          sp.broadcast_to(i, int_indexer_shape) for i in int_indexers
      ]
      indices = tuple(merge_lists(is_int_indexing, other_indexers, int_indexers))
    else:
      int_indexer_shape = ()

    return cls(indices, shape, int_indexer_shape, validate=True)

  def get_indexer_shape(self) -> tuple[int | Array, ...]:
    _, slice_indexers, _ = unpack_ndindexer(self)
    slice_shape = [s.size for s in slice_indexers]
    # In NDIndexers, the int_indexer_shape is *always* at the front of the
    # result.
    return (*self.int_indexer_shape, *slice_shape)

  def transform_shape(self, shape: None | tuple[int | Array, ...]) -> None | tuple[int | Array, ...]:
    del shape  # Unused
    return self.get_indexer_shape()

  def transform_dtype(self, dtype):
    return dtype
