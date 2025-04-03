# Copyright 2024 The Orbax Authors.
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

"""Various utilities for working with NumPy arrays and fragments."""

from typing import Sequence

from orbax.checkpoint._src.arrays import types

Shape = types.Shape
NdSlice = types.NdSlice
Index = types.Index

HashableSlice = types.HashableSlice
HashableIndex = types.HashableIndex


def int_tuple_from_slice(s: slice) -> tuple[int, ...]:
  """Represents a slice as a tuple of integers."""
  start, stop, step = s.start, s.stop, s.step
  step = step or 1
  try:
    return (int(start), int(stop), int(step))
  except:
    raise ValueError(f'Slice {s} is not concrete.') from None


def resolve_slice(xs: NdSlice, shape: Shape) -> NdSlice:
  """Turns an N-dimensional slice into an equivalent one with no `None` in it.

  Invariant: `a[dissolve_slice(xs, a.shape)] == a[xs]`.

  Args:
    xs: The slice to make explicit.
    shape: The shape against which to evaluate the slice's effect.

  Returns:
    An N-dimensional slice that, when applied to an array of shape `shape`,
    has the same effect as `xs`, but that has no `None` in any of its
    constituent slices.
  """
  return tuple(
      slice(x.start or 0, x.stop if x.stop is not None else n, x.step or 1)
      if isinstance(x, slice) else slice(x, x+1, 1)
      for x, n in zip(() if xs is Ellipsis else xs, shape))


def to_hashable_index(
    idx: Index, *, shape: Shape | None = None
) -> HashableIndex:
  """Converts an Index into a hashable form.

  Optionally resolves the slices to a concrete index if the shape is provided.
  If not, conversion may fail if the slices are not concrete.

  Args:
    idx: The index to convert.
    shape: Global array shape.

  Returns:
    A hashable index.
  """
  idx = resolve_slice(idx, shape) if shape else idx

  return tuple([int_tuple_from_slice(s) for s in idx])


def from_hashable_index(idx: HashableIndex) -> Index:
  return tuple([slice(s[0], s[1], s[2]) for s in idx])


def dissolve_slice(
    xs: NdSlice,
    shape: Shape,
    preserve_rank: bool = True,
) -> NdSlice:
  """Turns an N-dimensional slice into an equivalent one with `None`s in it.

  Invariant: `a[dissolve_slice(xs, a.shape)] == a[xs]`.

  This is the inverse of `resolve_slice()`.

  Args:
    xs: The slice to simplify.
    shape: The shape against which to simplify.
    preserve_rank: If false, remove any redundant `slice(None)` elements from
      the tail of the result.

  Returns:
    An N-dimensional slice that, when applied to an array of shape `shape`,
    has the same effect as `xs`, but that has `None` wherever possible in its
    constituent slices.
  """
  ys = tuple(
      slice(x.start or None,
            x.stop if x.stop != dim else None,
            x.step if x.step != 1 else None) for x, dim in zip(xs, shape))
  if not preserve_rank:
    while ys and ys[-1] == slice(None):
      ys = ys[:-1]
  return ys if ys else Ellipsis


def normalize_slice(resolved_slice: Index, shape: Shape) -> Index:
  """Ensures that all slice start and stop values are positive.

  Precondition: it is assumed that the slice start and stop values for a
  dimension of length `dim` are already on the interval `[-dim, dim)`. In
  other words, no truncation would occur, so  the result of slicing an array
  of shape `shape` with the given slice would have the same shape as the
  apparent shape of the slice itself:
    `np.empty(shape)[resolve_slice].shape == slice_shape(resolved_slice)`

  Postcondition: the slice start and stop values for a dimension of length `dim`
  are on the interval `[0, dim)`.

  Args:
    resolved_slice: An N-dimensional slice with no `None` values.
    shape: An array shape.

  Returns:
    An equivalent N-dimensional slice, with no `None` values.
  """
  return tuple(
      slice(
          s.start if s.start >= 0 else dim + s.start,  # pytype:disable=unsupported-operands
          s.stop if s.stop >= 0 else dim + s.stop,  # pytype:disable=unsupported-operands
          s.step,
      )
      for s, dim in zip(resolved_slice, shape)
  )


def slice_shape(xs: NdSlice) -> Shape:
  """Calculates the shape of the given slice."""
  return tuple((s.stop - s.start + (s.step - 1)) // s.step for s in xs)


def _pretty_slice(s: slice) -> str:
  start = s.start if s.start is not None else ''
  stop = s.stop if s.stop is not None else ''
  step = f':{s.step}' if s.step is not None else ''
  return f'{start}:{stop}{step}'


def pretty_nd_slice(idx: Sequence[slice] | type(Ellipsis)) -> str:
  """Returns a pretty-printed string representation of a NdSlice."""
  idx_str = (
      '...'
      if not idx or idx is Ellipsis
      else ', '.join(_pretty_slice(s) for s in idx)
  )
  return f'np.s_[{idx_str}]'
