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

"""Mini-mask creation library."""
from __future__ import annotations

import collections
from collections.abc import Callable
import functools
from typing import NamedTuple
from jax import util as jax_util
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
import numpy as np

# mypy: ignore-errors

# Logic for processing NumPy masks for kernels
class MaskInfo(NamedTuple):
  """Contains runtime masking information for the Splash attention kernel.

  The arrays data_next, mask_next and block_mask are placed in TPU
  scalar-memory. This is a scarse resource so the mask creation logic attempts
  to shrink the data-type of these arrays to the smallest possible one.
  This can be: np.int32, np.int16 or np.int8.

  For the arrays data_next, mask_next and block_mask the size of the first
  dimension can be one of the two following values: num_head or
  num_head_shards.
  The first dimension has size:
  * num_head_shards when there is only one unique mask for each head in a shard.
  In this case the three arrays are broadcasted to all the heads in the shard.
  * num_heads when there is more than one unique mask for each head in the
  shard.

  Attributes:
    data_next: An integer[num_heads_or_shards, num_q_blocks, num_kv_blocks]
      NumPy array where each entry contains the next `kv` block index to
      prefetch.
    mask_next: An integer[num_heads_or_shards, num_q_blocks, num_kv_blocks]
      NumPy array where each entry contains the next mask block index in
      `partial_mask_blocks` to prefetch.
    block_mask: An integer[num_heads_or_shards, num_q_blocks, num_kv_blocks]
      NumPy array whose entries can be 0, 1 or 2. An entry of 0 indicates that
      the corresponding block in the full mask was all zeros. An entry of 1
      indicates that the corresponding block in the full mask contained both
      zeros and ones. An entry of 2 indicates the corresponding block was
      entirely ones.
    partial_mask_blocks: A bool[num_partial_blocks, block_q, block_kv] NumPy
      array that contains the blocks of the original mask that contained both
      zeros and ones. The entries in `mask_next` point to indices in the first
      axis of this array.
    q_sequence: A i32[q_sequence_length] NumPy array. When using causal masking,
      this contains the list of indices that correspond to q tokens. For plain
      causal this is just np.arange(q_sequence_length).
  """

  data_next: np.ndarray | None
  mask_next: np.ndarray | None
  block_mask: np.ndarray | None
  partial_mask_blocks: np.ndarray | None
  q_sequence: np.ndarray | None


def _downcast_to_small_type(array: np.ndarray) -> np.ndarray:
  """Downcast numpy array.

  If possible, downcast the data-type of the input array to the smallest numpy
  type (among np.int16 and np.int8) that fits the content of the array.

  Args:
    array: the array to downcast

  Returns:
    The downcasted array.

  Raises:
    ValueError: if the input array is not np.int32 or if its elements are not
    all positive.
  """

  if array.dtype != np.int32:
    raise ValueError('Expected int32 input.')

  if not np.all(array >= 0):
    raise ValueError('Expected non-negative array.')

  if array.size == 0:
    return array

  max_value = np.max(array)

  if max_value <= np.iinfo(np.int8).max:
    return array.astype(np.int8)
  elif max_value <= np.iinfo(np.int16).max:
    return array.astype(np.int16)
  else:
    return array.astype(np.int32)


def _check_mask(mask: mask_lib.Mask) -> None:
  """Check that the given mask is valid.

  A row of all zeros along the kv dimension would result in a division by zero
  when computing the softmax. This function is meant to protect against that
  case.

  Args:
    mask: the mask to check.

  Raises:
    ValueError: the mask is invalid.
  """

  assert len(mask.shape) == 2

  exception_message = (
      'Some rows of the mask (along the kv dimension) are all zeros.\nThis is'
      ' would result in a division by zero when computing the attention'
      ' softmax.'
  )

  is_row_non_zero = np.zeros(mask.shape[0], dtype=np.bool_)
  for col in range(mask.shape[1]):
    # Mask only supports slice indices.
    is_row_non_zero = np.logical_or(
        is_row_non_zero,
        mask[(slice(0, mask.shape[0]), slice(col, col + 1))][:, 0],
    )
  if not is_row_non_zero.all():
    raise ValueError(exception_message)


class _HashableNDArray:
  """Helper to make a numpy array hashable: can be added associative containers.

  Attributes:
    array: The underlying numpy array.
  """
  array: np.ndarray

  def __init__(self, array: np.ndarray):
    self.array = array

  def __hash__(self):
    return hash(self.array.tobytes())

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, _HashableNDArray):
      return NotImplemented
    return np.array_equal(self.array, other.array, equal_nan=True)


def _get_mask_info_for_shard(
    output_shape: tuple[int, int, int],
    has_mask_next: bool,
    mask: mask_lib.MultiHeadMask,
    block_shape: tuple[int, int],
    coords_to_partial_mask_block_index: dict[tuple[int, int, int], int],
    masks_per_head_shard: int,
    head_start: int,
    num_heads: int,
    q_seq_start: int,
    q_seq_shard_size: int,
    blocked_q_seq_start: int,
    is_dkv: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
  """Process a slice of the mask to compute data_next and mask_next.

  Args:
    output_shape: The shape of the data_next and mask_next to return
    has_mask_next: Whether mask_next should be constructed. If False None is
      returned for mask_next.
    mask: The full mask to be sliced according to the head and sequence ranges
    block_shape: Shape of the Pallas grid block.
    coords_to_partial_mask_block_index: Mapping between the pallas launch grid
      coordinates and the index of the corresponding block in partial mask block
      list.
    masks_per_head_shard: Number of masks per head shards
    head_start: First head of the current shard.
    num_heads: Number of heads in the shard.
    q_seq_start: Start index along the Q sequence for the current shard (in
      number of tokens).
    q_seq_shard_size: Number of tokens along the Q sequence for the current
      shard.
    blocked_q_seq_start: Start index along the Q sequence for the current shard
      (in number of grid blocks)
    is_dkv: True if we are processing the dKV mask

  Returns:
    Slice of data_next and mask_next (if required) that correspond to the
    current mask slice.
  """
  _, _, kv_seq_len = mask.shape

  q_block_size, kv_block_size = block_shape
  q_block_count, q_mod = divmod(q_seq_shard_size, q_block_size)
  kv_block_count, kv_mod = divmod(kv_seq_len, kv_block_size)

  assert q_mod == 0
  assert kv_mod == 0

  blocked_shape = (
      (kv_block_count, num_heads, q_block_count)
      if is_dkv
      else (num_heads, q_block_count, kv_block_count)
  )

  data_coords = []
  mask_coords = []
  for idx in np.ndindex(blocked_shape):
    if is_dkv:
      kv_index, h_index, q_index = idx
    else:
      h_index, q_index, kv_index = idx

    h_index = h_index if masks_per_head_shard == 1 else head_start + h_index

    chunk = mask[(
        h_index,
        slice(
            q_seq_start + q_index * q_block_size,
            q_seq_start + (q_index + 1) * q_block_size,
        ),
        slice(kv_index * kv_block_size, (kv_index + 1) * kv_block_size),
    )]
    if chunk.any():
      data_coords.append(idx)
      if not chunk.all():
        mask_coords.append(idx)

  # Initialize the output arrays.
  mask_next = None
  if has_mask_next:
    mask_next = np.zeros(output_shape, dtype=np.int32)
  data_next = np.zeros(output_shape, dtype=np.int32)

  # If the mask is completelly zero'd out return freshly initialized outputs.
  if not data_coords:
    return data_next, mask_next

  data_coords_iter = iter(data_coords)
  first_j = coord_j = next(data_coords_iter)
  if mask_next is not None and mask_coords:
    mask_coords_iter = iter(mask_coords)
    first_m = coord_m = next(mask_coords_iter)
  else:
    first_m, coord_m, mask_coords_iter = None, None, None

  for idx in np.ndindex(blocked_shape):
    if is_dkv:
      kv_index, h_index, q_index = idx
      chunk_idx: tuple[int, ...] = (h_index, q_index, kv_index)
      data_dim = 2
    else:
      chunk_idx = idx
      data_dim = 2

    is_next = idx > coord_j

    if is_next:
      try:
        coord_j = next(data_coords_iter)
      except StopIteration:
        coord_j = first_j
    data_next[chunk_idx] = coord_j[data_dim]

    if mask_next is not None and mask_coords:
      assert coord_m is not None
      is_next_mask = idx > coord_m
      if is_next_mask:
        try:
          coord_m = next(mask_coords_iter)  # type: ignore
        except StopIteration:
          coord_m = first_m

      if is_dkv:
        assert coord_m is not None
        coord_m_global = (
            coord_m[1] + head_start,
            coord_m[2] + blocked_q_seq_start,
            coord_m[0],
        )
      else:
        assert coord_m is not None
        coord_m_global = (
            coord_m[0] + head_start,
            coord_m[1] + blocked_q_seq_start,
            coord_m[2],
        )

      mask_next[chunk_idx] = coords_to_partial_mask_block_index[coord_m_global]

  return data_next, mask_next


# When used in a transformer network with multiple layers, the SplashAttention
# kernel is created several times with the same mask. Cache MaskInfo to avoid
# blowing up compile times. Ideally the size of the cache should be determined
# by the client.
@functools.lru_cache(maxsize=12)
def _process_mask(
    mask: mask_lib.MultiHeadMask,  # [num_heads, q_seq_len, kv_seq_len]
    block_shape: tuple[int, int],
    is_dkv: bool,
    *,
    downcast_smem_data: bool = True,
    head_shards: int = 1,
    q_seq_shards: int = 1,
    shrink_grid: bool = True,
) -> tuple[MaskInfo, jax_util.HashableFunction | None]:
  """Transform a dense mask into a sparse representation.

  The number of head and Q sequence shards are needed to create a MaskInfo
  object that is partitionable (with shmap or PartIR) along these two dimension.
  In particular for dKV MaskInfo, for each shard the indices of in the data_next
  array are relative to the current shard.
  The fwd and dQ MaskInfo objects do not change when sharding along the head or
  Q dimensions, they would be different if we were to shard along the KV
  dimension, but the kernel does not support that.

  Args:
    mask: Dense mask to process.
    block_shape: Shape of the Pallas grid block.
    is_dkv: True if we are processing the dKV mask
    downcast_smem_data: If True, downcast the scalar-memory data of MaskInfo to
      a data type smaller than np.int32 (if possible).
    head_shards: Number of head shards of the mesh in which the kernel is
      launched.
    q_seq_shards: Number of Q sequence shards of the mesh in which the kernel is
      launched.
    shrink_grid: Whether or not we should apply the grid shrinking optimization.

  Returns:
    `MaskInfo`, a sparse representation of the dense mask.
    `MaskCallable`: a callable that, given in input Q and KV indices, returns
      the value of the mask at those coordinates.

  Raises:
    ValueError: if the input mask is invalid or the block sizes are not
    compatible with the mask sizes.
  """

  if len(mask.shape) != 3:
    raise ValueError(f'Expected a 3-dim mask, instead got: {mask.shape=}')

  head_count, q_seq_len, kv_seq_len = mask.shape
  q_block_size, kv_block_size = block_shape
  q_blocks_count, q_mod = divmod(q_seq_len, q_block_size)
  kv_blocks_count, kv_mod = divmod(kv_seq_len, kv_block_size)

  if q_mod != 0:
    raise ValueError(f'{q_block_size=} should divide {q_seq_len=}.')
  if kv_mod != 0:
    raise ValueError(f'{kv_block_size=} should divide {kv_seq_len=}.')

  q_seq_len_per_shard, mod = divmod(q_seq_len, q_seq_shards)
  if mod != 0:
    raise ValueError(f'{q_seq_shards=} should divide {q_seq_len=}.')

  q_blocks_per_shard, mod = divmod(q_seq_len_per_shard, q_block_size)
  if mod != 0:
    raise ValueError(f'{q_block_size=} should divide {q_seq_len_per_shard=}.')

  heads_per_shard, mod = divmod(head_count, head_shards)
  if mod != 0:
    raise ValueError(f'{head_shards=} should divide {head_count=}.')


  # Uniquify the masks.
  # Create a collection of the unique head masks in the input multi-head mask.
  # This avoids processing the same mask multiple times and it enables
  # deduplicating the partial mask blocks information.
  # Assign a unique ID to every unique mask.

  def assign_unique_ids(objects):
    id_map = collections.defaultdict(lambda: len(id_map))
    return {obj: id_map[obj] for obj in objects}

  unique_masks_dict: dict[mask_lib.Mask, int] = assign_unique_ids(
      head_mask for head_mask in mask.masks
  )

  # Build a mapping of heads to unique masks and masks to unique masks.

  head_to_mask_id: list[int] = [0] * head_count
  head_shard_to_mask_ids: list[set[int]] = [set() for _ in range(head_shards)]
  mask_id_to_heads: list[list[int]] = [
      [] for _ in range(len(unique_masks_dict))
  ]
  mask_id_to_head_shards: list[set[int]] = [
      set() for _ in range(len(unique_masks_dict))
  ]

  for head in range(head_count):
    mask_id = unique_masks_dict[mask.masks[head]]
    head_to_mask_id[head] = mask_id
    head_shard = head // heads_per_shard
    head_shard_to_mask_ids[head_shard].add(mask_id)
    mask_id_to_heads[mask_id].append(head)
    mask_id_to_head_shards[mask_id].add(head_shard)

  # If we have at most one unique mask per each head shard, then we can brodcast
  # the mask to all the heads in the shard. This is the common case.
  # If we have more than one mask in each head shard, then the optimization
  # cannot kick in and we use one mask for each head.
  # TODO(sharadmv,amagni): In the future we could think of a dynamic mapping of
  # heads to masks. This will likely introduce an additional field in the
  # MaskInfo class and runtime overhead to perform an indirect lookup. Since
  # having multiple masks per head-shard is not a common case we leave this for
  # future work.
  max_masks_per_head_shard = max(len(x) for x in head_shard_to_mask_ids)
  masks_per_head_shard = 1 if max_masks_per_head_shard == 1 else heads_per_shard

  unique_masks = [
      pair[0] for pair in sorted(unique_masks_dict.items(), key=lambda x: x[1])
  ]

  # TODO(amagni): checking the validity of the masks is slow for large masks.
  # Disable it for now, reevalute in the future.

  partial_mask_block_ids: dict[_HashableNDArray, int] = collections.defaultdict(
      lambda: len(partial_mask_block_ids)
  )
  block_id_to_block_coords: dict[int, list[tuple[int, ...]]] = (
      collections.defaultdict(list)
  )

  # Shape of the block_mask, mask_next and data_next fields of MaskInfo.
  block_mask_shape = (
      head_shards if masks_per_head_shard == 1 else head_count,
      q_blocks_count,
      kv_blocks_count,
  )

  block_mask = np.zeros(block_mask_shape, dtype=np.int32)

  def set_block_mask(mask_id: int, q_index: int, kv_index: int, value: int):
    if masks_per_head_shard == 1:
      for shard_index in mask_id_to_head_shards[mask_id]:
        block_mask[shard_index, q_index, kv_index] = value
    else:
      for head_index in mask_id_to_heads[mask_id]:
        block_mask[head_index, q_index, kv_index] = value

  q_sequence = None
  mask_function = None
  if len(unique_masks) == 1:
    unique_mask = unique_masks[0]
    # The mask object either define q_sequence and mask_function or none of
    # them.
    assert hasattr(unique_mask, 'q_sequence') == hasattr(
        unique_mask, 'mask_function'
    )

    # If the mask object defines a q_sequence and a mask_function, then make use
    # of these in the kernel rather. This is preferable over loading the mask
    # from memory. When using a mask_function, then mask_next and
    # partial_mask_blocks are left undefined and not used in the kernel.
    if hasattr(unique_mask, 'q_sequence') and hasattr(
        unique_mask, 'mask_function'
    ):
      q_sequence = unique_mask.q_sequence
      mask_function = unique_mask.mask_function

  # Identify the partial mask blocks and the value of the block mask for each
  # block.
  # The partial mask blocks are "global", meaning it is a collection of such
  # blocks across all the heads. Partial mask blocks are uniquified.
  # When partitioning, all partial mask blocks are replicated across shards.
  # As a future extension we could assign to each head shard only the relevant
  # partial blocks. This would reduce the amount of data each shard needs
  # from: num_unique_partial_block_blocks to
  # max(unique_partial_blocks_per_shard)
  for mask_id, unique_mask in enumerate(unique_masks):
    for coords in np.ndindex((q_blocks_count, kv_blocks_count)):
      (q_index, kv_index) = coords
      chunk = unique_mask[(
          slice(q_index * q_block_size, (q_index + 1) * q_block_size),
          slice(kv_index * kv_block_size, (kv_index + 1) * kv_block_size),
      )]
      has_nonzero = chunk.any()
      if has_nonzero:
        all_nonzero = chunk.all()
        if not all_nonzero:
          set_block_mask(mask_id, q_index, kv_index, 1)
          partial_mask_block_id = partial_mask_block_ids[
              _HashableNDArray(chunk)
          ]
          for head_index in mask_id_to_heads[mask_id]:
            block_id_to_block_coords[partial_mask_block_id].append(
                (head_index,) + coords
            )
        else:
          set_block_mask(mask_id, q_index, kv_index, 2)

  unique_partial_mask_blocks = [
      pair[0]
      for pair in sorted(partial_mask_block_ids.items(), key=lambda x: x[1])
  ]

  # For each position in the in q, kv grid that contains a partial mask block
  # record the index of its partial mask block.
  coords_to_partial_mask_block_index = {}
  for partial_mask_block_id, coords in block_id_to_block_coords.items():
    for coo in coords:
      coords_to_partial_mask_block_index[coo] = partial_mask_block_id

  partial_mask_blocks = None
  has_mask_next = False
  if len(unique_partial_mask_blocks) >= 1:
    partial_mask_blocks = [x.array for x in unique_partial_mask_blocks]
    partial_mask_blocks = np.stack(partial_mask_blocks, axis=0).astype(np.bool_)
    has_mask_next = True
  if is_dkv and partial_mask_blocks is not None:
    partial_mask_blocks = np.swapaxes(partial_mask_blocks, -1, -2)

  all_head_shards_identical = all(
      head_shard_to_mask_ids[0] == x and len(x) == 1
      for x in head_shard_to_mask_ids
  )
  # When all the head shards are identical, we can process only one and
  # broadcast the result across all the head shards, this avoids redundant work.
  shards_to_process = 1 if all_head_shards_identical else head_shards

  # Iterate over the shards.
  # Work on a fraction of the mask at the time to compute the mask and the data
  # indices. This is needed to compute the correct data indices, which are
  # relative to the current slice of the mask.

  q_sequence_axis = 1
  head_axis = 0

  # Collect mask_info shards along the head dimension, concatentate (or
  # broadcast) them after the loop.
  data_next_per_head_list, mask_next_per_head_list = [], []
  for head_shard in range(shards_to_process):
    # Collect mask_info shards along the q_sequence dimension, concatenate them
    # after the loop.
    data_next_sequence_slices, mask_next_sequence_slices = [], []
    for q_seq_len_shard in range(q_seq_shards):
      head_start = head_shard * heads_per_shard
      q_seq_len_shard_size = q_blocks_per_shard * q_block_size
      q_seq_len_start = q_seq_len_shard * q_seq_len_shard_size

      blocked_q_seq_len_start = q_seq_len_shard * q_blocks_per_shard
      blocked_q_seq_len_slice = slice(
          blocked_q_seq_len_start,
          (q_seq_len_shard + 1) * q_blocks_per_shard,
      )

      if masks_per_head_shard == 1:
        unique_mask = unique_masks[head_to_mask_id[head_start]]
        unique_mask = mask_lib.MultiHeadMask((unique_mask,))
        current_mask = unique_mask
        mask_head_slice = slice(head_shard, head_shard + 1)
      else:
        current_mask = mask
        mask_head_slice = slice(head_start, (head_shard + 1) * heads_per_shard)

      # The current iteration of the loop processes a slice of the mask info
      # tensors of this shape:
      mask_info_slice_shape = (
          mask_head_slice.stop - mask_head_slice.start,
          blocked_q_seq_len_slice.stop - blocked_q_seq_len_slice.start,
          kv_blocks_count,
      )
      # Generate data_next and mask_next for the current slice.
      data_next_slice, mask_next_slice = _get_mask_info_for_shard(
          output_shape=mask_info_slice_shape,
          has_mask_next=has_mask_next,
          mask=current_mask,
          block_shape=block_shape,
          coords_to_partial_mask_block_index=coords_to_partial_mask_block_index,
          head_start=head_start,
          masks_per_head_shard=masks_per_head_shard,
          num_heads=1 if masks_per_head_shard == 1 else heads_per_shard,
          q_seq_start=q_seq_len_start,
          q_seq_shard_size=q_seq_len_shard_size,
          blocked_q_seq_start=blocked_q_seq_len_start,
          is_dkv=is_dkv,
      )
      data_next_sequence_slices.append(data_next_slice)
      mask_next_sequence_slices.append(mask_next_slice)

    # Concatenate the sequence shards.
    data_next_per_head = np.concatenate(
        data_next_sequence_slices, axis=q_sequence_axis
    )
    data_next_per_head_list.append(data_next_per_head)
    if has_mask_next:
      mask_next_per_head = np.concatenate(
          mask_next_sequence_slices, axis=q_sequence_axis
      )
      mask_next_per_head_list.append(mask_next_per_head)

  # Concatenate (or broadcast) the head shards.
  mask_next = None
  if all_head_shards_identical:
    assert len(data_next_per_head_list) == 1
    data_next_shard = data_next_per_head_list[0]
    assert data_next_shard.shape == (1, q_blocks_count, kv_blocks_count)
    data_next = np.broadcast_to(
        data_next_shard,
        (head_shards, q_blocks_count, kv_blocks_count),
    )
    if has_mask_next:
      assert len(mask_next_per_head_list) == 1
      mask_next_shard = mask_next_per_head_list[0]
      assert mask_next_shard.shape == (1, q_blocks_count, kv_blocks_count)
      mask_next = np.broadcast_to(
          mask_next_shard,
          (head_shards, q_blocks_count, kv_blocks_count),
      )
  else:
    data_next = np.concatenate(data_next_per_head_list, axis=head_axis)
    if has_mask_next:
      mask_next = np.concatenate(mask_next_per_head_list, axis=head_axis)

  # Shrink the width of the mask info when possible.
  if (
      shrink_grid
      and block_mask.shape[0] == head_shards
      and len(unique_masks) == 1
  ):
    rows_per_q_shard = block_mask.shape[1] // q_seq_shards
    block_mask_shards = []
    data_next_shards = []
    mask_next_shards = []

    # Slice the mask info arrays along the Q dimension. This simulates Q
    # sharding and enabled shrinking each Q shard of the MaskInfo arrays
    # independently.
    for q_seq_len_shard in range(q_seq_shards):
      rows = slice(
          q_seq_len_shard * rows_per_q_shard,
          (q_seq_len_shard + 1) * rows_per_q_shard,
      )
      current_block_mask = block_mask[:, rows, :]
      current_data_next = data_next[:, rows, :]
      current_mask_next = (
          mask_next[:, rows, :] if mask_next is not None else None
      )

      shrink_function = _shrink_mask_info_dkv if is_dkv else _shrink_mask_info

      current_block_mask, current_data_next, current_mask_next = (
          shrink_function(
              block_mask=current_block_mask,
              data_next=current_data_next,
              mask_next=current_mask_next,
              head_shards=head_shards,
          )
      )

      assert current_block_mask.size > 0
      assert current_data_next.size > 0
      assert current_mask_next is None or current_mask_next.size > 0
      assert current_block_mask.shape == current_data_next.shape
      assert (
          current_mask_next is None
          or current_block_mask.shape == current_mask_next.shape
      )

      block_mask_shards.append(current_block_mask)
      data_next_shards.append(current_data_next)
      mask_next_shards.append(current_mask_next)

    if q_seq_shards == 1:
      block_mask = block_mask_shards[0]
      data_next = data_next_shards[0]
      mask_next = mask_next_shards[0]
    else:
      # Since shrinking happens independently for each shard along Q we might
      # end up with uneven mask info shards. Insert padding where necessary to
      # maintain the SPMD paradigm.
      padding_axis = 1 if is_dkv else 2

      max_size = max(x.shape[padding_axis] for x in block_mask_shards)
      padded_block_mask_shards = []
      padded_data_next_shards = []
      padded_mask_next_shards = []
      assert (
          len(block_mask_shards)
          == len(data_next_shards)
          == len(mask_next_shards)
      )
      for (
          current_block_mask,
          current_data_next,
          current_mask_next,
      ) in zip(
          block_mask_shards, data_next_shards, mask_next_shards
      ):
        # For dKV shrinking happens along axis Q (the rows of MaskInfo), for
        # fwd and dQ shrinking happens along axis KV (the columns of MaskInfo).
        if is_dkv:
          pad_width = (
              (0, 0),
              (0, max_size - current_block_mask.shape[padding_axis]),
              (0, 0),
          )
        else:
          pad_width = (
              (0, 0),
              (0, 0),
              (0, max_size - current_block_mask.shape[padding_axis]),
          )

        padded_block_mask_shards.append(
            np.pad(current_block_mask, pad_width=pad_width, constant_values=0)
        )
        # By padding data_next and mask_next with 'edge' policy we avoid
        # fetching new blocks thanks to 'immediate revisiting'.
        padded_data_next_shards.append(
            np.pad(current_data_next, pad_width=pad_width, mode='edge')
        )
        if current_mask_next is not None:
          padded_mask_next_shards.append(
              np.pad(current_mask_next, pad_width=pad_width, mode='edge')
          )

      block_mask = np.concatenate(padded_block_mask_shards, axis=1)
      data_next = np.concatenate(padded_data_next_shards, axis=1)
      mask_next = np.concatenate(padded_mask_next_shards, axis=1)

  if downcast_smem_data:
    data_next = _downcast_to_small_type(data_next)
    block_mask = _downcast_to_small_type(block_mask)
    if mask_next is not None:
      mask_next = _downcast_to_small_type(mask_next)

  assert (mask_function is not None) == (q_sequence is not None)
  # When the mask can be computed inside the kernel with a mask_function,
  # there is no need to load it from memory. So mask_next and
  # partial_mask_blocks are unused.
  return (
      MaskInfo(
          data_next=data_next,
          mask_next=mask_next if mask_function is None else None,
          block_mask=block_mask,
          partial_mask_blocks=partial_mask_blocks
          if mask_function is None
          else None,
          q_sequence=q_sequence,
      ),
      mask_function,
  )


def _shrink_mask_info(
    *,
    block_mask: np.ndarray,
    data_next: np.ndarray,
    mask_next: np.ndarray,
    head_shards: int,
):
  assert block_mask.ndim == 3
  assert data_next.ndim == 3
  assert mask_next is None or mask_next.ndim == 3

  head_block_mask = block_mask[0]

  grouped_non_zero_cols = []
  # Group non-zero columns based on which row they belong to.
  for row_index in range(head_block_mask.shape[0]):
    head_block_mask_row = head_block_mask[row_index, :]
    non_zero_cols = np.nonzero(head_block_mask_row)[0]
    grouped_non_zero_cols.append(non_zero_cols)

  # Pad each row in the non-zero indices to match the width of the longest
  # row. This avoids having jagged rows.
  max_non_zero_cols = max(len(x) for x in grouped_non_zero_cols)
  padded_non_zero_cols = []
  padding = -1
  for row in grouped_non_zero_cols:
    padded_non_zero_cols.append(
        np.pad(
            row,
            pad_width=(0, max_non_zero_cols - row.shape[0]),
            constant_values=padding,
        )
    )

  padded_non_zero_cols = np.stack(padded_non_zero_cols, axis=0)

  assert padded_non_zero_cols.shape[0] == block_mask.shape[1], (
      padded_non_zero_cols.shape,
      block_mask.shape,
  )

  # For each row of array, select the columns indices in padded_non_zero_cols,
  # ignore padding.
  def select_cols(array):
    assert array.ndim == 2
    assert padded_non_zero_cols.ndim == 2
    assert array.shape[0] == padded_non_zero_cols.shape[0]
    assert array.shape[1] >= padded_non_zero_cols.shape[1]
    selected_rows = []
    for row in range(array.shape[0]):
      col = padded_non_zero_cols[row]
      selected = array[row][col]
      selected = np.where(col != padding, selected, 0)
      selected_rows.append(selected)

    return np.stack(selected_rows, axis=0)

  return _slice_mask_info(
      block_mask=block_mask,
      data_next=data_next,
      mask_next=mask_next,
      head_shards=head_shards,
      slice_function=select_cols,
  )


def _shrink_mask_info_dkv(
    *,
    block_mask: np.ndarray,
    data_next: np.ndarray,
    mask_next: np.ndarray,
    head_shards: int,
):
  assert block_mask.ndim == 3
  assert data_next.ndim == 3
  assert mask_next is None or mask_next.ndim == 3

  head_block_mask = block_mask[0]
  grouped_non_zero_rows = []
  # Group non-zero rows based on which column they belong to.
  for col_index in range(head_block_mask.shape[1]):
    col = head_block_mask[:, col_index]
    non_zero_rows = np.nonzero(col)[0]
    grouped_non_zero_rows.append(non_zero_rows)

  # Pad each col in the non-zero indices to match the height of the longest
  # col. This avoids having jagged cols.
  max_non_zero_rows = max(len(x) for x in grouped_non_zero_rows)
  padded_non_zero_rows = []
  padding = -1
  for col in grouped_non_zero_rows:
    padded_non_zero_rows.append(
        np.pad(
            col,
            pad_width=(max_non_zero_rows - col.shape[0], 0),
            constant_values=padding,
        )
    )

  padded_non_zero_rows = np.stack(padded_non_zero_rows, axis=1)

  assert padded_non_zero_rows.shape[1] == block_mask.shape[2], (
      padded_non_zero_rows.shape,
      block_mask.shape,
  )

  # For each col of array, select the rows indices in padded_non_zero_rows,
  # ignore padding.
  def select_rows(array):
    assert array.ndim == 2
    assert padded_non_zero_rows.ndim == 2
    assert array.shape[1] == padded_non_zero_rows.shape[1]
    assert array.shape[0] >= padded_non_zero_rows.shape[0]
    selected_cols = []
    for col in range(array.shape[1]):
      row = padded_non_zero_rows[:, col]
      selected = array[:, col][row]
      selected = np.where(row != padding, selected, 0)
      selected_cols.append(selected)

    return np.stack(selected_cols, axis=1)

  return _slice_mask_info(
      block_mask=block_mask,
      data_next=data_next,
      mask_next=mask_next,
      head_shards=head_shards,
      slice_function=select_rows,
  )


def _slice_mask_info(
    *,
    block_mask: np.ndarray,
    data_next: np.ndarray,
    mask_next: np.ndarray,
    head_shards: int,
    slice_function: Callable[[np.ndarray], np.ndarray],
):
  new_block_mask = []
  new_data_next = []
  new_mask_next = []
  for head_shard in range(head_shards):
    head_block_mask = block_mask[head_shard]
    head_block_mask = slice_function(head_block_mask)
    new_block_mask.append(head_block_mask)

    head_data_next = data_next[head_shard]
    head_data_next = slice_function(head_data_next)
    new_data_next.append(head_data_next)

    if mask_next is not None:
      head_mask_next = mask_next[head_shard]
      head_mask_next = slice_function(head_mask_next)
      new_mask_next.append(head_mask_next)

  block_mask = np.stack(new_block_mask, axis=0)
  data_next = np.stack(new_data_next, axis=0)
  if mask_next is not None:
    mask_next = np.stack(new_mask_next, axis=0)

  return block_mask, data_next, mask_next


process_mask = functools.partial(_process_mask, is_dkv=False)
process_mask_dkv = functools.partial(_process_mask, is_dkv=True)
