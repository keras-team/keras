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
"""Pickling support for precompiled binaries."""

from __future__ import annotations

import pickle
import io

import jax
from jax._src.lib import xla_client as xc


def serialize(compiled: jax.stages.Compiled):
  """Serializes a compiled binary.

  Because pytrees are not serializable, they are returned so that
  the user can handle them properly.
  """
  unloaded_executable = getattr(compiled._executable,
                                '_unloaded_executable', None)
  if unloaded_executable is None:
    raise ValueError("Compilation does not support serialization")
  args_info_flat, in_tree = jax.tree_util.tree_flatten(compiled.args_info)

  with io.BytesIO() as file:
    _JaxPjrtPickler(file).dump(
        (unloaded_executable, args_info_flat, compiled._no_kwargs))
    return file.getvalue(), in_tree, compiled.out_tree


def deserialize_and_load(serialized,
                         in_tree,
                         out_tree,
                         backend: str | xc.Client | None = None):
  """Constructs a jax.stages.Compiled from a serialized executable."""

  if backend is None or isinstance(backend, str):
    backend = jax.devices(backend)[0].client

  (unloaded_executable, args_info_flat,
   no_kwargs) = _JaxPjrtUnpickler(io.BytesIO(serialized), backend).load()

  args_info = in_tree.unflatten(args_info_flat)

  loaded_compiled_obj = unloaded_executable.load()

  return jax.stages.Compiled(
      loaded_compiled_obj, args_info, out_tree, no_kwargs=no_kwargs)


class _JaxPjrtPickler(pickle.Pickler):
  device_types = (xc.Device,)
  client_types = (xc.Client,)

  def persistent_id(self, obj):
    if isinstance(obj, xc.LoadedExecutable):
      return ('exec', obj.client.serialize_executable(obj))
    if isinstance(obj, xc._xla.Executable):
      return ('exec', obj.serialize())
    if isinstance(obj, self.device_types):
      return ('device', obj.id)
    if isinstance(obj, self.client_types):
      return ('client',)


class _JaxPjrtUnpickler(pickle.Unpickler):

  def __init__(self, file, backend):
    super().__init__(file)
    self.backend = backend
    self.devices_by_id = {d.id: d for d in backend.devices()}

  def persistent_load(self, pid):
    if pid[0] == 'exec':
      return self.backend.deserialize_executable(pid[1])
    if pid[0] == 'device':
      return self.devices_by_id[pid[1]]
    if pid[0] == 'client':
      return self.backend
    raise pickle.UnpicklingError
