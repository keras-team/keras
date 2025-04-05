# Copyright 2024 The JAX Authors.
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
"""Colocated Python serialization utilities."""

# TODO(jmudigonda): Use a string-typed array for output structure when it
# becomes available. Using a fixed uint8 array is only for prototyping.

from __future__ import annotations

import collections
import io
from typing import Any, Callable, Sequence

try:
  import cloudpickle  # type: ignore[import-not-found]
except ImportError:
  cloudpickle = None

import jax
from jax._src import api
from jax._src import tree_util
from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc
import numpy as np

DeviceList = xc.DeviceList

# Hard-coded limit for serialized specs size.
# TODO(jmudigonda): Use a string-typed array for output structure when it
# becomes available. Using a fixed uint8 array is only for prototyping.
_MAX_SERIALIZED_SPECS_SIZE = 1048576


@jax.util.cache(max_size=None)
def _get_cpu_device_map() -> dict[int, jax.Device]:
  """Returns a map from a device id to a matching device."""
  cpu_device_map: dict[int, jax.Device] = {}
  # TODO(hyeontaek): We should look up CPU devices for a specific CPU backend.
  # When deserializing a device on the controller, the backend should be the one
  # associated with colocated_python. When deserializing on the colocated_python
  # executor, it should be the CPU backend visible to the user function running
  # under colocated_python.

  # Look for CPU devices in the default backend.
  for d in xb.local_devices()[0].client._get_all_devices():  # pylint: disable=protected-access
    if d.device_kind == "cpu":
      if d.id in cpu_device_map:
        raise ValueError(
            f"Multiple CPU devices with id {d.id} found:"
            f" {cpu_device_map[d.id]} and {d}"
        )
      cpu_device_map[d.id] = d
  if cpu_device_map:
    return cpu_device_map

  # Fall back to searching CPU devices in all backends.
  for backend in xb.backends().values():
    for d in backend._get_all_devices():  # pylint: disable=protected-access
      if d.device_kind == "cpu":
        if d.id in cpu_device_map:
          raise ValueError(
              f"Multiple CPU devices with id {d.id} found:"
              f" {cpu_device_map[d.id]} and {d}"
          )
        cpu_device_map[d.id] = d
  return cpu_device_map


def _reduce_mesh(
    mesh: jax.sharding.Mesh,
) -> tuple[Callable[..., jax.sharding.Mesh], Any]:
  def make_mesh(
      mesh_device_ids: np.ndarray, axis_names: Any
  ) -> jax.sharding.Mesh:
    cpu_device_map = _get_cpu_device_map()
    mesh_devices = np.vectorize(lambda device_id: cpu_device_map[device_id])(
        mesh_device_ids
    )
    return jax.sharding.Mesh(mesh_devices, axis_names)

  mesh_device_ids = np.vectorize(lambda d: d.id, otypes=[int])(mesh.devices)
  return make_mesh, (mesh_device_ids, mesh.axis_names)


def _reduce_device_list(
    device_list: DeviceList,
) -> tuple[Callable[..., DeviceList], Any]:
  def make_device_list(device_ids: Sequence[int]) -> DeviceList:
    cpu_device_map = _get_cpu_device_map()
    devices = np.vectorize(lambda device_id: cpu_device_map[device_id])(
        device_ids
    )
    return DeviceList(tuple(devices))

  device_ids = [d.id for d in device_list]
  return make_device_list, (device_ids,)


def _reduce_single_device_sharding(
    sharding: jax.sharding.SingleDeviceSharding,
) -> tuple[Callable[..., jax.sharding.SingleDeviceSharding], Any]:

  def make_single_device_sharding(device_id: int):
    cpu_device_map = _get_cpu_device_map()
    return jax.sharding.SingleDeviceSharding(cpu_device_map[device_id])

  return make_single_device_sharding, (sharding.device_set.pop().id,)


def _serialize(obj: Any) -> bytes:
  """Serializes callables and input/output spec objects.

  DO NOT USE THIS FUNCTION EXCEPT FOR THE INTERNAL IMPLEMENTATION OF
  colocated_python.

  This module contains utility functions used internally for implementiong
  `colocated_python` when it ships callables and input/output specs through
  IFRT. The pickled data is produced and consumed in an ephermeral fashion
  without any persistence, and it does not expect any version compatibility
  (which cloudpickle does not guarantee). Furthermore, serialization and
  deserialization is expected to be done on machine(s) that are controlled by a
  single tenant, which allows unpickling done during deserialization to be
  trusted.

  Raises:
    ModuleNotFoundError: If cloudpickle is not available.
  """
  if cloudpickle is None:
    raise ModuleNotFoundError('No module named "cloudpickle"')

  class _CustomPickler(cloudpickle.Pickler):
    dispatch_table = collections.ChainMap(
        {jax.sharding.Mesh: _reduce_mesh},
        {DeviceList: _reduce_device_list},
        {jax.sharding.SingleDeviceSharding: _reduce_single_device_sharding},
        cloudpickle.CloudPickler.dispatch_table,  # pylint: disable=attribute-error
    )
    dispatch = dispatch_table

  with io.BytesIO() as file:
    _CustomPickler(file).dump(obj)
    return file.getvalue()


def _deserialize(serialized: bytes) -> Any:
  """Deserializes callables and input/output spec objects.

  DO NOT USE THIS FUNCTION EXCEPT FOR THE INTERNAL IMPLEMENTATION OF
  colocated_python. See serialize() for details.

  Raises:
    ModuleNotFoundError: If cloudpickle is not available.
  """
  if cloudpickle is None:
    raise ModuleNotFoundError('No module named "cloudpickle"')

  return cloudpickle.loads(serialized)


def _make_specs_for_serialized_specs(
    devices: DeviceList,
) -> tuple[api.ShapeDtypeStruct, api.ShapeDtypeStruct]:
  """Makes output specs for serialized specs."""
  # TODO(jmudigonda): Use a string-typed array for output structure when it
  # becomes available. Using a fixed uint8 array is only for prototyping.
  mesh = jax.sharding.Mesh(tuple(devices), ("x",))
  replicated_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec()
  )
  return (
      api.ShapeDtypeStruct(
          shape=(), dtype=np.int32, sharding=replicated_sharding
      ),
      api.ShapeDtypeStruct(
          shape=(_MAX_SERIALIZED_SPECS_SIZE,),
          dtype=np.uint8,
          sharding=replicated_sharding,
      ),
  )


def _serialize_specs(
    specs_treedef: tree_util.PyTreeDef,
    specs_leaves: tuple[api.ShapeDtypeStruct, ...],
    devices: DeviceList,
) -> tuple[jax.Array, ...]:
  """Serializes the output specs into a tuple of arrays.

  DO NOT USE THIS FUNCTION EXCEPT FOR THE INTERNAL IMPLEMENTATION OF
  colocated_python. See serialize() for details.
  """
  s = _serialize((specs_treedef, specs_leaves))
  assert (
      len(s) <= _MAX_SERIALIZED_SPECS_SIZE
  ), f"Too large serialized spec size: {len(s)}"
  # TODO(jmudigonda): Use a string-typed array for output structure when it
  # becomes available. Using a fixed uint8 array is only for prototyping.
  mesh = jax.sharding.Mesh(tuple(devices), ("x",))
  replicated_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec()
  )
  len_array = jax.make_array_from_callback(
      shape=(),
      sharding=replicated_sharding,
      data_callback=lambda _: np.array(len(s), dtype=np.int32),
  )
  data_array = jax.make_array_from_callback(
      shape=(_MAX_SERIALIZED_SPECS_SIZE,),
      sharding=replicated_sharding,
      data_callback=lambda _: np.frombuffer(
          s + b"\0" * (_MAX_SERIALIZED_SPECS_SIZE - len(s)),
          dtype=np.uint8,
      ),
  )
  return len_array, data_array


def _deserialize_specs(
    serialized_specs: tuple[jax.Array, ...],
) -> tuple[tree_util.PyTreeDef, tuple[api.ShapeDtypeStruct, ...]]:
  """Deserializes the specs from the serialized specs.

  DO NOT USE THIS FUNCTION EXCEPT FOR THE INTERNAL IMPLEMENTATION OF
  colocated_python. See serialize() for details.
  """
  # TODO(jmudigonda): Use a string-typed array for output structure when it
  # becomes available. Using a fixed uint8 array is only for prototyping.
  len_array, data_array = serialized_specs
  length = int(len_array.addressable_shards[0].data)
  data = np.asarray(data_array.addressable_shards[0].data).tobytes()
  return _deserialize(data[:length])
