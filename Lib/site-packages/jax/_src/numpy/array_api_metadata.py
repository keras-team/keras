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

"""
This module contains metadata related to the `Python array API`_.

.. _Python array API: https://data-apis.org/array-api/
"""
from __future__ import annotations

from types import ModuleType

import jax
from jax._src.sharding import Sharding
from jax._src.lib import xla_client as xc
from jax._src import dtypes as _dtypes, config


__array_api_version__ = '2023.12'


def __array_namespace__(self, *, api_version: None | str = None) -> ModuleType:
  """Return the `Python array API`_ namespace for JAX.

  .. _Python array API: https://data-apis.org/array-api/
  """
  if api_version is not None and api_version != __array_api_version__:
    raise ValueError(f"{api_version=!r} is not available; "
                     f"available versions are: {[__array_api_version__]}")
  return jax.numpy


def __array_namespace_info__() -> ArrayNamespaceInfo:
  return ArrayNamespaceInfo()


class ArrayNamespaceInfo:
  """Metadata for the `Python array API`_

  .. _Python array API: https://data-apis.org/array-api/
  """
  _capabilities = {
    "boolean indexing": True,
    "data-dependent shapes": False,
  }

  def _build_dtype_dict(self):
    array_api_types = {
      "bool", "int8", "int16",
      "int32", "uint8", "uint16",
      "uint32", "float32", "complex64"
    }
    if config.enable_x64.value:
      array_api_types |= {"int64", "uint64", "float64", "complex128"}
    return {category: {t.name: t for t in types if t.name in array_api_types}
            for category, types in _dtypes._dtype_kinds.items()}

  def default_device(self):
    # By default JAX arrays are uncommitted (device=None), meaning that
    # JAX is free to choose the most efficient device placement.
    return None

  def devices(self):
    return jax.devices()

  def capabilities(self):
    return self._capabilities

  def default_dtypes(self, *, device: xc.Device | Sharding | None = None):
    # Array API supported dtypes are device-independent in JAX
    del device
    default_dtypes = {
      "real floating": "f",
      "complex floating": "c",
      "integral": "i",
      "indexing": "i",
    }
    return {
      dtype_name: _dtypes.canonicalize_dtype(
        _dtypes._default_types.get(kind)
      ) for dtype_name, kind in default_dtypes.items()
    }

  def dtypes(
      self, *,
      device: xc.Device | Sharding | None = None,
      kind: str | tuple[str, ...] | None = None):
    # Array API supported dtypes are device-independent in JAX
    del device
    data_types = self._build_dtype_dict()
    if kind is None:
      out_dict = data_types["numeric"] | data_types["bool"]
    elif isinstance(kind, tuple):
      out_dict = {}
      for _kind in kind:
        out_dict |= data_types[_kind]
    else:
      out_dict = data_types[kind]
    return out_dict
