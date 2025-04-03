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

"""NDArray adapter interface.

This module defines an interface for adapters that support rendering of
multi-dimensional arrays (tensors) in treescope. This can be used to add support
for a variety of array types, including numpy arrays, JAX arrays, and others.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Generic, TypeVar

import numpy as np

from treescope import rendering_parts

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class PositionalAxisInfo:
  """Marks an axis as being an ordinary (positional) axis.

  Attributes:
    axis_logical_index: The logical index of the axis. This is the index of the
      axis in the underlying array data.
    size: The size of the axis.
  """

  axis_logical_index: int
  size: int

  def logical_key(self) -> int:
    return self.axis_logical_index


@dataclasses.dataclass(frozen=True)
class NamedPositionlessAxisInfo:
  """Marks an axis as being accessible by name only.

  Attributes:
    axis_name: The name of the axis.
    size: The size of the axis.
  """

  axis_name: Any
  size: int

  def logical_key(self) -> Any:
    return self.axis_name


@dataclasses.dataclass(frozen=True)
class NamedPositionalAxisInfo:
  """Marks an axis as being accessible by name or by position.

  Attributes:
    axis_logical_index: The logical index of the axis. This is the index of the
      axis in the underlying array data.
    axis_name: The name of the axis.
    size: The size of the axis.
  """

  axis_logical_index: int
  axis_name: int
  size: int

  def logical_key(self) -> int:
    return self.axis_logical_index


@dataclasses.dataclass(frozen=True)
class ShardingInfo:
  """Summary of the sharding of an array.

  Attributes:
    shard_shape: Shape of a single shard. Should be the same length as the
      number of slices in each value of `device_index_to_shard_slices`.
    device_index_to_shard_slices: A mapping from device index to the tuple of
      per-axis indices or slices of the original array that is assigned to that
      device. Each entry of this tuple should either be an int or a slice
      object. If an int, that axis should not appear in shard_shape (e.g. the
      full array is formed by stacking the shards along a new axis). If a slice,
      the corresponding axis should appear in shard_shape, and the slice should
      be the full slice ``slice(None)`` (if the array is not sharded over this
      axis) or a slice that matches the corresponding entry in `shard_shape` (if
      the full array is formed by concatenating the shards along this axis).
    device_type: The type of device that the array is sharded across, as a
      string (e.g. "CPU", "TPU", "GPU").
    fully_replicated: Whether the array is fully replicated across all devices.
  """

  shard_shape: tuple[int, ...]
  device_index_to_shard_slices: dict[int, tuple[slice | int, ...]]
  device_type: str
  fully_replicated: bool = False


AxisInfo = (
    PositionalAxisInfo | NamedPositionlessAxisInfo | NamedPositionalAxisInfo
)


class NDArrayAdapter(abc.ABC, Generic[T]):
  """An adapter to support rendering a multi-dimensional array (tensor) type."""

  @abc.abstractmethod
  def get_axis_info_for_array_data(self, array: T) -> tuple[AxisInfo, ...]:
    """Returns axis information for each axis in the given array.

    This method should return a tuple with an AxisInfo entry for each axis in
    the array. Array axes can be one of three types:

      * Positional axes have an index and a size, and can be accessed by
        position. This is common in ordinary NDArrays.
      * Named positionless axes have a name and a size, and can be accessed by
        name only. This is how `penzai.core.named_axes` treats named axes.
      * Named positional axes have an index, a name, and a size, and can be
        accessed by either position or name. This is how PyTorch treats named
        axes.

    Note that positional axes have an explicit "logical index", which may or may
    not match their position in the underlying array data; this makes it
    possible to support "views" of underlying array data that have a different
    axis ordering than the original data. (`penzai.core.named_axes` uses this.)

    Args:
      array: The array to get axis information for.

    Returns:
      A tuple with an AxisInfo entry for each axis in the array. The ordering
      must be consistent with the ordering expected by
      `get_array_data_with_truncation`.
    """
    raise NotImplementedError(
        "Subclasses must override `get_axis_info_for_array_data`."
    )

  @abc.abstractmethod
  def get_array_data_with_truncation(
      self,
      array: T,
      mask: T | None,
      edge_items_per_axis: tuple[int | None, ...],
  ) -> tuple[np.ndarray, np.ndarray]:
    """Returns a numpy array with truncated array (and mask) data.

    This method should construct a numpy array whose contents are a truncated
    version of the given array's data; this array will be used to construct the
    actual array visualization. It is also responsible for broadcasting the mask
    appropriately and returning a compatible truncation of it.

    This method may be called many times when rendering a large structure of
    arrays (once per array), so it should be as fast as possible. We suggest
    doing truncation on an accelerator device and then copying the result, if
    possible, to avoid unnecessary data transfer.

    Args:
      array: The array to get data for.
      mask: An optional mask array provided by the user, which should be
        broadcast-compatible with ``array``. (If it is not compatible, the user
        has provided an invalid mask, and this method should raise an
        informative exception.) Can be None if no mask is provided.
      edge_items_per_axis: A tuple with one entry for each axis in the array.
        Each entry is either the number of items to keep on each side of this
        axis, or None to keep all items. The ordering will be consistent with
        the axis order returned by `get_axis_info_for_array_data`, i.e. the
        entry ``k`` in ``edge_items`` corresponds to the entry ``k`` in the axis
        info tuple, regardless of the logical indices or axis names.

    Returns:
      A tuple ``(truncated_data, truncated_mask)``. ``truncated_data`` should be
      a numpy array with a truncated version of the given array's data. If
      entry ``k`` in ``edge_items`` is ``None``, axis ``k`` should have
      the same size as the ``size`` field of the entry ``k`` returned by
      ``get_axis_info_for_array_data``. If entry ``k`` in ``edge_items``
      is not ``None``, axis ``k`` should have a size of
      ``edge_items[k] * 2 + 1``, and the middle element can be arbitrary.
      ``truncated_mask`` should be a numpy array with the same shape as
      ``truncated_data`` containing a truncated, broadcasted version of the
      mask; the middle element of the mask must be ``False`` for each truncated
      axis.
    """
    raise NotImplementedError(
        "Subclasses must override `get_array_data_with_truncation`."
    )

  @abc.abstractmethod
  def get_array_summary(
      self, array: T, fast: bool
  ) -> str | rendering_parts.RenderableTreePart:
    """Summarizes the contents of the given array.

    The summary returned by this method will be used as a one-line summary of
    the array in treescope when automatically visualized.

    If the ``fast`` argument is True, the method should return a summary that
    can be computed quickly, ideally without any device computation. If it is
    False, the method can return a more detailed summary, but it should still
    be fast enough to be called many times when rendering a large structure of
    arrays.

    Args:
      array: The array to summarize.
      fast: Whether to return a fast summary that can be computed without
        expensive device computation.

    Returns:
      A summary of the given array's contents. The summary should be a single
      line of text. It will be wrapped between angle brackets (< and >) when
      rendered. It may either be a string or a custom part (useful if the
      summary should be abbreviated).
    """
    raise NotImplementedError("Subclasses must override `get_array_summary`.")

  def get_numpy_dtype(self, array: T) -> np.dtype | None:
    """Returns the numpy dtype of the given array.

    This should match the dtype of the array returned by
    `get_array_data_with_truncation`.

    Args:
      array: The array to summarize.

    Returns:
      The numpy dtype of the given array, or None if the array does not have a
      numpy dtype.
    """
    raise NotImplementedError("Subclasses must override `get_numpy_dtype`.")

  def get_sharding_info_for_array_data(self, array: T) -> ShardingInfo | None:
    """Summarizes the sharding of the given array's data.

    The summary returned by this method will be used to render a sharding for
    the array when automatic visualization is enabled.

    Args:
      array: The array to summarize.

    Returns:
      A summary of the given array's sharding, or None if it does not have a
      sharding.
    """
    # Default implementation: don't show any sharding information.
    del array

  def should_autovisualize(self, array: T) -> bool:
    """Returns True if the given array should be automatically visualized.

    If this method returns True, the array will be automatically visualized
    by the array visualizer if it is enabled.

    Args:
      array: The array to possibly visualize.

    Returns:
      True if the given array should be automatically visualized.
    """
    del array
    return True
