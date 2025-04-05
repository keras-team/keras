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

from __future__ import annotations

import math

from jax._src import basearray
from jax._src import core
from jax._src import tree_util
from jax._src import sharding_impls
from jax._src.interpreters import pxla
from jax._src.interpreters import xla
from jax._src.util import safe_zip, safe_map

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

# EArray is an Array that can contain extended dtypes.
class EArray(basearray.Array):
  __slots__ = ['aval', '_data']
  __hash__ = None  # type: ignore[assignment]
  __array_priority__ = 100

  def __init__(self, aval, data):
    self.aval = aval
    self._data = data

  def block_until_ready(self):
    _ = self._data.block_until_ready()
    return self

  def copy_to_host_async(self):
    self._data.copy_to_host_async()

  def copy(self):
    return EArray(self.aval, self._data.copy())

  def __repr__(self):
    return 'E' + repr(self._data)

  def __iter__(self):
    if self.ndim == 0: raise TypeError('iteration over a 0-d array')
    raise NotImplementedError

  # forward to aval
  shape = property(lambda self: self.aval.shape)  # type: ignore[assignment]
  dtype = property(lambda self: self.aval.dtype)  # type: ignore[assignment]

  # computed from shape and dtype
  ndim = property(lambda self: len(self.aval.shape))  # type: ignore[assignment]
  size = property(lambda self: math.prod(self.aval.shape))  # type: ignore[assignment]
  itemsize = property(lambda self: self.aval.dtype.itemsize)  # type: ignore[assignment]
  def __len__(self):
    if self.ndim == 0: raise TypeError('len() of unsized object')
    return self.shape[0]

  # forward to self._data
  devices = property(lambda self: self._data.devices)  # type: ignore[assignment]
  _committed = property(lambda self: self._data._committed)
  is_fully_addressable = property(lambda self: self._data.is_fully_addressable)  # type: ignore[assignment]
  is_fully_replicated = property(lambda self: self._data.is_fully_replicated)  # type: ignore[assignment]
  delete = property(lambda self: self._data.delete)  # type: ignore[assignment]
  is_deleted = property(lambda self: self._data.is_deleted)  # type: ignore[assignment]
  on_device_size_in_bytes = property(lambda self: self._data.on_device_size_in_bytes)  # type: ignore[assignment]
  unsafe_buffer_pointer = property(lambda self: self._data.unsafe_buffer_pointer)  # type: ignore[assignment]

  # defer to extended dtype rules
  @property
  def sharding(self):
    phys_sharding = self._data.sharding
    return sharding_impls.logical_sharding(self.aval, phys_sharding)

  @property
  def committed(self):
    return self._data.committed

  @property
  def device(self):
    if isinstance(self._data.sharding, sharding_impls.SingleDeviceSharding):
      return self._data.device
    return self.sharding

  # TODO(mattjj): not implemented below here, need more methods from ArrayImpl

  def addressable_data(self, index: int) -> EArray:
    raise NotImplementedError

  @property
  def addressable_shards(self):
    raise NotImplementedError

  @property
  def global_shards(self):
    raise NotImplementedError

# TODO(mattjj): _set_array_base_attributes

def _earray_shard_arg_handler(xs, shardings, layouts, copy_semantics):
  arrs = [x._data for x in xs]
  phys_shardings = [sharding_impls.physical_sharding(x.aval, sharding)
                    for x, sharding in zip(xs, shardings)]
  # TODO(yashkatariya): `layouts` should be converted to physical layouts.
  return pxla.shard_args(phys_shardings, layouts, copy_semantics, arrs)
pxla.shard_arg_handlers[EArray] = _earray_shard_arg_handler

core.pytype_aval_mappings[EArray] = lambda x: x.aval
xla.canonicalize_dtype_handlers[EArray] = lambda x: x
tree_util.dispatch_registry.register_node(
    EArray, lambda x: ((x._data,), x.aval), lambda a, xs: EArray(a, xs[0]))
