# Copyright 2021 The JAX Authors.
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

from collections.abc import Mapping, Sequence
import functools

from jax._src.util import safe_zip, use_cpp_class, cache
from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src.op_shardings import (
    are_op_shardings_equal, get_num_ways_dim_sharded, is_op_sharding_replicated,
    op_sharding_to_indices)

Shape = tuple[int, ...]
Device = xc.Device
Index = tuple[slice, ...]
XLADeviceAssignment = Sequence[Device]


@cache(max_size=4096, trace_context_in_key=False)
def _addressable_devices_indices_map(
    sharding: Sharding, global_shape: Shape) -> Mapping[Device, Index | None]:
  global_map = sharding.devices_indices_map(global_shape)
  if sharding.is_fully_addressable:
    return global_map
  if hasattr(sharding, '_internal_device_list'):
    return {d: global_map[d]
            for d in sharding._internal_device_list.addressable_device_list}
  return {d: ind for d, ind in global_map.items()
          if d.process_index == d.client.process_index()}

@cache(max_size=4096, trace_context_in_key=False)
def common_devices_indices_map(
    s: Sharding, global_shape: Shape) -> Mapping[Device, Index]:
  s.shard_shape(global_shape)  # raises a good error message
  hlo_sharding = s._to_xla_hlo_sharding(len(global_shape))
  indices = op_sharding_to_indices(hlo_sharding, global_shape,
                                   len(s._device_assignment))
  return dict(safe_zip(s._device_assignment, indices))


@cache(max_size=4096, trace_context_in_key=False)
def _common_shard_shape(self, global_shape: Shape) -> Shape:
  hlo_sharding = self._to_xla_hlo_sharding(len(global_shape))
  if is_op_sharding_replicated(hlo_sharding):
    return global_shape
  partitions, _ = get_num_ways_dim_sharded(hlo_sharding)
  assert len(partitions) == len(global_shape), (len(partitions), len(global_shape))
  out = []
  for dim, (s, p) in enumerate(safe_zip(global_shape, partitions)):
    try:
      quotient, remainder = divmod(s, p)
    except TypeError:
      # TODO Figure out how to partition dynamic shapes
      raise NotImplementedError
    if remainder != 0:
      raise ValueError(
          f"Sharding {self} implies that array axis {dim} is partitioned "
          f"{p} times, but the dimension size is {s} "
          f"(full shape: {global_shape}, "
          f"per-dimension tiling factors: {partitions} should evenly divide "
          "the shape)")
    out.append(quotient)
  return tuple(out)


@use_cpp_class(xc.Sharding)
class Sharding:
  """Describes how a :class:`jax.Array` is laid out across devices.
  """

  # Abstract methods below that subclasses should implement.
  @property
  def device_set(self) -> set[Device]:
    """The set of devices that this :class:`Sharding` spans.

    In multi-controller JAX, the set of devices is global, i.e., includes
    non-addressable devices from other processes.
    """
    raise NotImplementedError('Subclasses should implement this method.')

  @property
  def is_fully_replicated(self) -> bool:
    """Is this sharding fully replicated?

    A sharding is fully replicated if each device has a complete copy of the
    entire data.
    """
    raise NotImplementedError('Subclasses should implement this method.')

  @property
  def is_fully_addressable(self) -> bool:
    """Is this sharding fully addressable?

    A sharding is fully addressable if the current process can address all of
    the devices named in the :class:`Sharding`. ``is_fully_addressable`` is
    equivalent to "is_local" in multi-process JAX.
    """
    raise NotImplementedError('Subclasses should implement this method.')

  @property
  def num_devices(self) -> int:
    """Number of devices that the sharding contains."""
    raise NotImplementedError('Subclasses should implement this method.')

  @property
  def memory_kind(self) -> str | None:
    """Returns the memory kind of the sharding."""
    raise NotImplementedError('Subclasses should implement this method.')

  def with_memory_kind(self, kind: str) -> Sharding:
    """Returns a new Sharding instance with the specified memory kind."""
    raise NotImplementedError('Subclasses should implement this method')

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    raise NotImplementedError('Subclasses should implement this method.')

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    raise NotImplementedError('Subclasses should implement this method.')

  def _to_sdy_sharding(self, num_dimensions: int):
    raise NotImplementedError('Subclasses should implement this method.')

  #############################################################################
  # Default implementations below that all subclasses will inherit.

  @functools.cached_property
  def addressable_devices(self) -> set[Device]:
    """The set of devices in the :class:`Sharding` that are addressable by the
       current process.
    """
    # Add a fast path for single controller runtimes.
    if xb.process_count() == 1:
      return self.device_set
    return {d for d in self.device_set
            if d.process_index == d.client.process_index()}

  def addressable_devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Index | None]:
    """A mapping from addressable devices to the slice of array data each contains.

    ``addressable_devices_indices_map`` contains that part of
    ``device_indices_map`` that applies to the addressable devices.
    """
    return _addressable_devices_indices_map(self, global_shape)

  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:
    """Returns a mapping from devices to the array slices each contains.

    The mapping includes all global devices, i.e., including
    non-addressable devices from other processes.
    """
    return common_devices_indices_map(self, global_shape)

  @functools.cached_property
  def _addressable_device_assignment(self) -> XLADeviceAssignment:
    if self.is_fully_addressable:
      return self._device_assignment
    if hasattr(self, '_internal_device_list'):
      return tuple(self._internal_device_list.addressable_device_list)
    return tuple(d for d in self._device_assignment
                 if d.process_index == d.client.process_index())

  def shard_shape(self, global_shape: Shape) -> Shape:
    """Returns the shape of the data on each device.

    The shard shape returned by this function is calculated from
    ``global_shape`` and the properties of the sharding.
    """
    return _common_shard_shape(self, global_shape)

  def is_equivalent_to(self: Sharding, other: Sharding, ndim: int) -> bool:
    """Returns ``True`` if two shardings are equivalent.

    Two shardings are equivalent if they place the same logical array shards on
    the same devices.

    For example, a :class:`NamedSharding` may be equivalent
    to a :class:`PositionalSharding` if both place the same shards of the array
    on the same devices.
    """
    try:
      return (are_op_shardings_equal(self._to_xla_hlo_sharding(ndim),
                                     other._to_xla_hlo_sharding(ndim))
              and self._internal_device_list == other._internal_device_list and  # type: ignore
              self.memory_kind == other.memory_kind)
    # NotImplementedError is raised by PmapSharding because it can't lower
    # to OpSharding. So if `other` is a PmapSharding, default to a strict
    # equality check.
    except NotImplementedError:
      return self == other
