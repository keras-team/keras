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

"""ShardingMetadata representing Sharding property."""

from __future__ import annotations

import abc
import dataclasses
import enum
import json
import logging
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import jax
from jax.experimental import mesh_utils
import numpy as np

PartitionSpecElement = Union[None, str, Tuple[str, ...]]

_PARTITION_SPEC = 'partition_spec'
_SHARDING = '_sharding'
_SHARDING_TYPE = 'sharding_type'
_DEVICE_STR = 'device_str'
_MESH_AXES = 'axis_names'
_MESH_SHAPE = 'shape'
_DEVICES_SHAPE = 'shape'
_DEVICE_MESH = 'device_mesh'
_MEMORY_KIND = 'memory_kind'
_ID = 'id'


class ShardingTypes(enum.Enum):
  NAMED_SHARDING = 'NamedSharding'
  SINGLE_DEVICE_SHARDING = 'SingleDeviceSharding'
  POSITIONAL_SHARDING = 'PositionalSharding'
  GSPMD_SHARDING = 'GSPMDSharding'


@dataclasses.dataclass
class DeviceMetadata:
  """TPU Device metadata class."""

  id: int

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> DeviceMetadata:
    return DeviceMetadata(
        id=data[_ID],
    )

  @classmethod
  def from_jax_device(cls, device: jax.Device) -> DeviceMetadata:
    return DeviceMetadata(
        id=device.id,
    )

  def __eq__(self, other: DeviceMetadata):
    return self.id == other.id


@dataclasses.dataclass
class DeviceMetadataMesh:
  """Contain a mesh of DeviceMetadata class."""

  mesh: Sequence[Any]

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> DeviceMetadataMesh:
    mesh = data['mesh']
    device_mesh = jax.tree.map(
        DeviceMetadata.from_dict,
        mesh,
        is_leaf=lambda x: isinstance(x, Mapping) and _ID in x,
    )
    return DeviceMetadataMesh(mesh=device_mesh)

  @classmethod
  def from_jax_mesh(
      cls, mesh: jax.sharding.Mesh
  ) -> Optional[DeviceMetadataMesh]:
    """Take in of jax.sharding.Mesh and convert into DeviceMetadata while keeping the sequences.

    Support only TPU-device.  If there is any non-TPU device, return None.

    Args:
      mesh: jax.sharding.Mesh

    Returns:
      DeviceMetadataMesh if only TPU-devices are in the mesh.
    """

    if isinstance(devices := mesh.devices, np.ndarray):
      devices = devices.tolist()
    device_mesh = jax.tree.map(
        DeviceMetadata.from_jax_device,
        devices,
        is_leaf=lambda x: isinstance(x, jax.Device),
    )

    return DeviceMetadataMesh(mesh=device_mesh)

  def to_jax_device_mesh(self):
    """return a jax Device mesh.

    Returns:
      Nested sequence of jax.Device
    """

    # build device.id to device map
    device_map = {}
    for device in jax.devices():
      device_map[device.id] = device

    def build_device(m: DeviceMetadata) -> jax.Device:
      if ret := device_map.get(m.id):
        return ret
      else:
        raise ValueError(
            'The available devices are different from the devices used to'
            ' save the checkpoint.  Please restore checkpoint by passing'
            f' new shardings for target devices. Original={self.mesh},'
            f' current available={jax.devices()}'
        )

    return jax.tree.map(
        build_device,
        self.mesh,
        is_leaf=lambda x: isinstance(x, DeviceMetadata),
    )

  def __eq__(self, other):
    return self.mesh == other.mesh


@dataclasses.dataclass
class ShardingMetadata(abc.ABC):
  """ShardingMetadata representing Sharding property.

  This ShardingMetadata only represents the following `jax.sharding.Sharding`:
    jax.sharding.NamedSharding
    jax.sharding.SingleDeviceSharding
    jax.sharding.GSPMDSharding
    jax.sharding.PositionalSharding
  """

  @classmethod
  @abc.abstractmethod
  def from_jax_sharding(cls, jax_sharding) -> ShardingMetadata:
    """Converts `jax.sharding.Sharding` to `ShardingMetadata`."""

  @abc.abstractmethod
  def to_jax_sharding(self) -> jax.sharding.Sharding:
    """Converts `ShardingMetadata` to `jax.sharding.Sharding`."""

  @classmethod
  @abc.abstractmethod
  def from_deserialized_dict(
      cls, deserialized_dict: dict[str, str]
  ) -> ShardingMetadata:
    """Converts serialized_string in the form of `dict[str, str]` to `ShardingMetadata`."""

  @abc.abstractmethod
  def to_serialized_string(self) -> str:
    """Converts `ShardingMetadata` to `serialized_string`."""


@dataclasses.dataclass
class NamedShardingMetadata(ShardingMetadata):
  """NamedShardingMetadata representing `jax.sharding.NamedSharding`."""

  shape: np.ndarray
  axis_names: List[str]
  partition_spec: Tuple[
      PartitionSpecElement, ...
  ]  # Each element is either ``None``, a string, or a tuple of strings.

  # Optional device mesh.  If it's None, use jax.devices(),
  # otherwise, the stored device_mesh will be used to recreate NamedSharding.
  device_mesh: Optional[DeviceMetadataMesh] = None

  @classmethod
  def from_jax_sharding(
      cls, jax_sharding: jax.sharding.NamedSharding
  ) -> NamedShardingMetadata:
    return cls(
        shape=np.array(list(jax_sharding.mesh.shape.values())),
        axis_names=list(jax_sharding.mesh.axis_names),
        partition_spec=tuple(jax_sharding.spec),
        device_mesh=DeviceMetadataMesh.from_jax_mesh(jax_sharding.mesh),
    )

  def to_jax_sharding(self) -> jax.sharding.NamedSharding:
    if self.device_mesh:
      mesh_devices = self.device_mesh.to_jax_device_mesh()
    else:
      mesh_devices = jax.devices()

    return jax.sharding.NamedSharding(
        jax.sharding.Mesh(
            np.asarray(mesh_devices).reshape(self.shape),
            axis_names=self.axis_names,
        ),
        spec=jax.sharding.PartitionSpec(*self.partition_spec),
    )

  @classmethod
  def from_deserialized_dict(
      cls, deserialized_dict: dict[str, Any]
  ) -> NamedShardingMetadata:
    if (
        _MESH_SHAPE in deserialized_dict
        and _MESH_AXES in deserialized_dict
        and _PARTITION_SPEC in deserialized_dict
    ):
      shape = np.array(deserialized_dict[_MESH_SHAPE])
      axis_names = list(deserialized_dict[_MESH_AXES])
      partition_spec = tuple(deserialized_dict[_PARTITION_SPEC])
      if device_mesh_dic := deserialized_dict.get(_DEVICE_MESH):
        device_mesh = DeviceMetadataMesh.from_dict(device_mesh_dic)
      else:
        device_mesh = None

      return cls(
          shape=shape,
          axis_names=axis_names,
          partition_spec=partition_spec,
          device_mesh=device_mesh,
      )
    else:
      raise ValueError(
          f'Sharding data not found in deserialized_dict: {deserialized_dict}'
      )

  def to_serialized_string(self) -> str:
    sharding_data = {}
    sharding_data[_SHARDING_TYPE] = ShardingTypes.NAMED_SHARDING.value
    sharding_data[_MESH_SHAPE] = self.shape.tolist()
    sharding_data[_MESH_AXES] = self.axis_names
    sharding_data[_PARTITION_SPEC] = self.partition_spec
    if self.device_mesh:
      sharding_data[_DEVICE_MESH] = dataclasses.asdict(self.device_mesh)
    return json.dumps(sharding_data)

  def __repr__(self):
    return (
        f'NamedShardingMetadata(shape={self.shape},'
        f' axis_names={self.axis_names}, partition_spec={self.partition_spec})'
        f' device_mesh={self.device_mesh}'
    )

  def __eq__(self, other):
    return (
        np.array_equal(self.shape, other.shape)
        and self.axis_names == other.axis_names
        and self.partition_spec == other.partition_spec
        and self.device_mesh == other.device_mesh
    )


@dataclasses.dataclass
class SingleDeviceShardingMetadata(ShardingMetadata):
  """SingleDeviceShardingMetadata representing `jax.sharding.SingleDeviceSharding`."""

  device_str: str

  @classmethod
  def from_jax_sharding(
      cls, jax_sharding: jax.sharding.SingleDeviceSharding
  ) -> SingleDeviceShardingMetadata:
    return cls(device_str=str(next(iter(jax_sharding.device_set))))

  def to_jax_sharding(self) -> jax.sharding.SingleDeviceSharding:
    device_map = {str(device): device for device in jax.local_devices()}
    device_str = self.device_str
    if device := device_map.get(device_str, None):
      return jax.sharding.SingleDeviceSharding(device)
    else:
      raise ValueError(
          f'Device {device_str} was not found in jax.local_devices().'
      )

  @classmethod
  def from_deserialized_dict(
      cls, deserialized_dict: dict[str, str]
  ) -> SingleDeviceShardingMetadata:
    if (
        _DEVICE_STR in deserialized_dict
        and deserialized_dict[_DEVICE_STR] is not None
    ):
      return cls(device_str=deserialized_dict[_DEVICE_STR])
    raise ValueError(
        f'Device str not found in deserialized_dict: {deserialized_dict}'
    )

  def to_serialized_string(self) -> str:
    sharding_data = {}
    sharding_data[_SHARDING_TYPE] = ShardingTypes.SINGLE_DEVICE_SHARDING.value
    sharding_data[_DEVICE_STR] = self.device_str
    return json.dumps(sharding_data)

  def __repr__(self):
    return f'SingleDeviceShardingMetadata(device_str={self.device_str})'

  def __eq__(self, other):
    return self.device_str == other.device_str


@dataclasses.dataclass
class GSPMDShardingMetadata(ShardingMetadata):
  pass


@dataclasses.dataclass
class PositionalShardingMetadata(ShardingMetadata):
  """PositionalShardingMetadata representing `jax.sharding.PositionalSharding`."""

  shape: np.ndarray
  memory_kind: Optional[str] = None

  @classmethod
  def from_jax_sharding(
      cls, jax_sharding: jax.sharding.PositionalSharding
  ) -> PositionalShardingMetadata:
    return cls(
        shape=np.array(list(jax_sharding.shape)),
        memory_kind=jax_sharding.memory_kind,
    )

  def to_jax_sharding(self) -> jax.sharding.PositionalSharding:
    return jax.sharding.PositionalSharding(
        mesh_utils.create_device_mesh(self.shape),
        memory_kind=self.memory_kind,
    )

  @classmethod
  def from_deserialized_dict(
      cls, deserialized_dict: dict[str, str]
  ) -> PositionalShardingMetadata:
    if _DEVICES_SHAPE in deserialized_dict:
      shape = np.array(deserialized_dict[_DEVICES_SHAPE])
      memory_kind = deserialized_dict.get(_MEMORY_KIND, None)
      return cls(
          shape=shape,
          memory_kind=memory_kind,
      )
    else:
      raise ValueError(
          f'Sharding data not found in deserialized_dict: {deserialized_dict}'
      )

  def to_serialized_string(self) -> str:
    sharding_data = {}
    sharding_data[_SHARDING_TYPE] = ShardingTypes.POSITIONAL_SHARDING.value
    sharding_data[_DEVICES_SHAPE] = self.shape.tolist()
    if self.memory_kind is not None:
      sharding_data[_MEMORY_KIND] = self.memory_kind
    return json.dumps(sharding_data)

  def __repr__(self):
    return (
        f'PositionalShardingMetadata(shape={self.shape},'
        f' memory_kind={self.memory_kind})'
    )

  def __eq__(self, other):
    return (
        np.array_equal(self.shape, other.shape)
        and self.memory_kind == other.memory_kind
    )


def from_jax_sharding(jax_sharding) -> Optional[ShardingMetadata]:
  """Converts `jax.sharding.Sharding` to `ShardingMetadata`."""
  if isinstance(jax_sharding, jax.sharding.NamedSharding):
    return NamedShardingMetadata.from_jax_sharding(jax_sharding)
  elif isinstance(jax_sharding, jax.sharding.SingleDeviceSharding):
    return SingleDeviceShardingMetadata.from_jax_sharding(jax_sharding)
  elif isinstance(jax_sharding, jax.sharding.PositionalSharding):
    return PositionalShardingMetadata.from_jax_sharding(jax_sharding)
  else:
    logging.warning(
        'Conversion for %s has not been implemented.', type(jax_sharding)
    )


def from_serialized_string(serialized_str) -> ShardingMetadata:
  """Converts `serialized_string` to `ShardingMetadata`."""
  deserialized_dict = json.loads(serialized_str)
  if deserialized_dict[_SHARDING_TYPE] == ShardingTypes.NAMED_SHARDING.value:
    return NamedShardingMetadata.from_deserialized_dict(deserialized_dict)
  elif (
      deserialized_dict[_SHARDING_TYPE]
      == ShardingTypes.SINGLE_DEVICE_SHARDING.value
  ):
    return SingleDeviceShardingMetadata.from_deserialized_dict(
        deserialized_dict
    )
  elif (
      deserialized_dict[_SHARDING_TYPE]
      == ShardingTypes.POSITIONAL_SHARDING.value
  ):
    return PositionalShardingMetadata.from_deserialized_dict(deserialized_dict)
  else:
    raise NotImplementedError(
        f'Conversion for {deserialized_dict[_SHARDING_TYPE]} has not been'
        ' implemented.'
    )


def get_sharding_or_none(serialized_string):
  try:
    return from_serialized_string(serialized_string.item()).to_jax_sharding()
  except ValueError as e:
    logging.error(e)
