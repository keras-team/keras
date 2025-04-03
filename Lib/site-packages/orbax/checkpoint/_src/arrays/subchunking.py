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

"""Utilities for managing storage layout of arrays in checkpoints."""

import dataclasses
import itertools
import math

from absl import logging
import jax
from jax import numpy as jnp
import numpy as np
from orbax.checkpoint._src.arrays import fragments as fragments_lib
from orbax.checkpoint._src.arrays import types


Fragment = fragments_lib.Fragment
Fragments = fragments_lib.Fragments
Shape = types.Shape


_MIB = 1024**2  # 1 MiB


def _find_divisors(size: int):
  """Fast-ish method for finding divisors of a number."""
  sqrt_divs = [
      i for i in range(1, math.ceil(math.sqrt(size + 1))) if size % i == 0
  ]
  return sorted(set(sqrt_divs + [size // div for div in sqrt_divs][::-1]))


def choose_chunk_shape(
    global_shape: Shape,
    write_shape: Shape,
    dtype: jnp.dtype | np.dtype,
    target_byte_size: int | None,
    *,
    shard_axes: tuple[int, ...] = (),
) -> Shape:
  """Chooses a chunk shape that divides the `write_shape`.

  The chunk shape is chosen such that the resulting byte size is less than
  or equal to `target_byte_size`, but is otherwise as large as possible.

  This uses a greedy algorithm that attempts to split the largest and sharded
  dimensions first, unless the `shard_axes` optional parameter is also provided.
  In the latter case, the algorithm will prioritize these explicitly specified
  axes and ensure that array's storage representation is sharded at least once
  on as many of these axes as possible.

  Args:
    global_shape: The global shape of the array.
    write_shape: The local shape being written.
    dtype: The dtype of the array.
    target_byte_size: Desired chunk byte size. Must be >= dtype.itemsize.
    shard_axes: [optional] A list of axes that should be prioritized for
      storage sharding. The implementation will try to shard at least once on as
      many of these axes as possible.

  Returns:
    List of length `len(write_shape)` specifying the chosen chunk shape.
  """
  # TensorStore Zarr metadata doesn't support 0-sized dimensions.
  write_shape = tuple(max(1, d) for d in write_shape)

  if target_byte_size is None:
    return write_shape

  # TODO: b/354139177 - This check is too generous; the minimally viable chunk
  # size should be set to something within the range of [4 KiB; 1 MiB] (from
  # TensorStore and storage performance considerations).
  if target_byte_size < dtype.itemsize:
    raise ValueError(
        f'target_byte_size={target_byte_size} must be >= {dtype.itemsize}'
    )

  if len(global_shape) != len(write_shape):
    raise ValueError(
        f'global_shape={global_shape} and write_shape={write_shape} must have'
        ' the same length.'
    )
  if target_byte_size < 1 * _MIB:
    logging.warning(
        'Setting the target_byte_size too small could reduce performance.'
    )

  sharded_dimensions = np.array(global_shape) != np.array(write_shape)
  dtype_size = dtype.itemsize
  target_elements = target_byte_size // dtype_size

  rank = len(write_shape)

  # `dim_factors[i]` is the list of divisors of `write_shape[i]`
  dim_factors = [_find_divisors(size) for size in write_shape]

  # The current chunk shape is:
  # [dim_factors[i][-1] for i in range(rank)]

  total_elements = math.prod(write_shape)

  def reduce_dim(dim_to_reduce: int) -> None:
    """Reduces the given dimension in the current chunk shape."""
    nonlocal total_elements
    current_dim = dim_factors[dim_to_reduce].pop()
    new_dim = dim_factors[dim_to_reduce][-1]
    total_elements = (total_elements // current_dim) * new_dim
    sharded_dimensions[dim_to_reduce] = True

  # First, try to reduce the size of the chunk shape on the `shard_axes`.
  # If some of these specified axes are already sharded, we will skip them on
  # the first iteration which ensures that we shard at least once on each of the
  # `shard_axes`. It might also be the case that the given target_byte_size is
  # too big to shard on all of the requested axes, in which case we will
  # maximize the number of the number of axes that are sharded.
  (shard_axes := list(shard_axes)).sort()
  could_shard = bool(shard_axes)
  first_sharding_iteration = True
  while could_shard and total_elements > target_elements:
    could_shard = False
    # For the first pass, exclude dimensions that are already sharded.
    # We do our best to shard at least once of each of the `shard_axes`.
    if first_sharding_iteration:
      must_shard_dims = list(i for i in shard_axes if not sharded_dimensions[i])
      first_sharding_iteration = False
    else:
      must_shard_dims = shard_axes
    # Exclude dimensions that can no longer be sharded.
    must_shard_dims = list(
        i for i in must_shard_dims if len(dim_factors[i]) > 1
    )
    # Shard once on each of the remaining dimensions in a round-robin fashion,
    # while we can.
    while must_shard_dims and total_elements > target_elements:
      could_shard = True
      # Find the minimum available divisor among the remaining dimensions.
      dim_idx = min(
          must_shard_dims,
          key=lambda i: dim_factors[i][-1] // dim_factors[i][-2],
      )
      reduce_dim(dim_idx)
      must_shard_dims.remove(dim_idx)

  if shard_axes:
    current_shape = tuple(dim_factors[i][-1] for i in range(rank))
    if current_shape != write_shape:
      logging.vlog(
          1,
          'Reduced write shape using shard_axes=%s: global_shape=%s,'
          ' write_shape=%s, dtype=%s, target_byte_size=%d; reduced shape: %s',
          shard_axes,
          global_shape,
          write_shape,
          dtype,
          target_byte_size,
          current_shape,
      )

  # If we are not within target_byte_size yet, continue to reduce the current
  # chunk shape until the desired number of elements is reached.
  while total_elements > target_elements:
    # Greedily reduce the largest dimension.  This is not guaranteed to bring us
    # the closest to `target_elements`, but is simple to implement and should
    # work well enough.
    dim_to_reduce = -1
    dim_to_reduce_size = 1
    for i in range(rank):
      size = dim_factors[i][-1]
      if sharded_dimensions[i] and size > dim_to_reduce_size:
        dim_to_reduce_size = size
        dim_to_reduce = i

    if dim_to_reduce_size > 1:
      reduce_dim(dim_to_reduce)
    else:
      # We need to start splitting on unsharded dimensions.
      sharded_dimensions = np.ones(len(write_shape))

  chosen_shape = tuple(dim_factors[i][-1] for i in range(rank))

  # TODO: b/363218206 - Consider info logging the storage shape in saving and
  # loading code.
  logging.vlog(
      1,
      'Reduced write shape: global_shape=%s, write_shape=%s, dtype=%s,'
      ' target_byte_size=%d, shard_axes=%s; chosen shape: %s',
      global_shape,
      write_shape,
      dtype,
      target_byte_size,
      shard_axes,
      chosen_shape,
  )

  return chosen_shape


def validate_divisible_shapes(
    divided_shape: Shape,
    dividing_shape: Shape,
) -> bool:
  """Returns True only if dividing_shape is a divisor of divided_shape."""
  try:
    return not np.mod(divided_shape, dividing_shape).any()
  except ValueError:
    # eg. imcompatible shape
    return False


# TODO(b/363218206): double-check if we can have scalar fragments.


def chunk_fragment(fragment: Fragment, target_shape: Shape) -> list[Fragment]:
  """Chunks the given fragment into the given target shape.

  Args:
    fragment: The fragment to chunk. Fragment's shape must be divisible by the
      target shape. If the fragment is concrete (has a value), the operation is
      zero copy and the returned fragments will have the same underlying buffer.
    target_shape: The shape to chunk the fragment into.

  Returns:
    A list of fragments that cover the entire original fragment, each of
    target_shape shape.

    The order of the returned fragments is deterministic: the index of the
    innermost axis changes fastest. For example: a fragment with index
    [0:4:2, 0:2:1] and target_shape [2, 1] will be split into fragments
    with indices ordered as follows:
      [[0:2:1, 0:1:1], [0:2:1, 1:2:1], [2:4:1, 0:1:1], [2:4:1, 1:2:1]]

  Raises:
    ValueError: If the fragment's shape is not divisible by the target shape.
  """
  if fragment.shape == target_shape:
    return [fragment]

  if len(target_shape) != len(fragment.shape):
    raise ValueError(
        f'target_shape={target_shape} must have the same length as'
        f' fragment.shape={fragment.shape}'
    )
  if not validate_divisible_shapes(fragment.shape, target_shape):
    raise ValueError(
        f'fragment.shape={fragment.shape} is not divisible by'
        f' target_shape={target_shape}'
    )

  start_indices_per_dim = (
      range(start, stop, new_dim)
      for start, stop, new_dim in jax.util.safe_zip(
          fragment.start, fragment.stop, target_shape
      )
  )
  start_indices = itertools.product(*start_indices_per_dim)
  new_fragments = []
  for start_index in start_indices:
    new_index = tuple(
        slice(start_index[i], start_index[i] + target_shape[i], 1)
        for i in range(len(target_shape))
    )
    if fragment.value is None:
      value = None
    else:
      value_index = Fragment(index=new_index).offset_by(-fragment.start).index
      value = fragment.value[value_index]
    new_fragments.append(Fragment(index=new_index, value=value))
  return new_fragments


def chunk_fragments(
    fragments: Fragments,
    target_shape: Shape,
) -> Fragments:
  """Chunks (with zero-copy) the given fragments into the given target shape."""
  new_fragments = []
  for fragment in fragments.fragments:
    new_fragments.extend(chunk_fragment(fragment, target_shape))
  return dataclasses.replace(fragments, fragments=new_fragments)
