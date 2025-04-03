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

"""Utils for writing in msgpack format. Derived from flax.serialization."""
import enum
from typing import Any, Dict

import jax
import msgpack
import numpy as np


# On-the-wire / disk serialization format

# We encode state-dicts via msgpack, using its custom type extension.
# https://github.com/msgpack/msgpack/blob/master/spec.md
#
# - ndarrays and DeviceArrays are serialized to nested msgpack-encoded string
#   of (shape-tuple, dtype-name (e.g. 'float32'), row-major array-bytes).
#   Note: only simple ndarray types are supported, no objects or fields.
#
# - native complex scalars are converted to nested msgpack-encoded tuples
#   (real, imag).


def _ndarray_to_bytes(arr) -> bytes:
  """Save ndarray to simple msgpack encoding."""
  if isinstance(arr, jax.Array):
    arr = np.array(arr)
  if arr.dtype.hasobject or arr.dtype.isalignedstruct:
    raise ValueError('Object and structured dtypes not supported '
                     'for serialization of ndarrays.')
  tpl = (arr.shape, arr.dtype.name, arr.tobytes('C'))
  return msgpack.packb(tpl, use_bin_type=True)


def _dtype_from_name(name: str):
  """Handle JAX bfloat16 dtype correctly."""
  if name == b'bfloat16':
    return jax.numpy.bfloat16
  else:
    return np.dtype(name)


def _ndarray_from_bytes(data: bytes) -> np.ndarray:
  """Load ndarray from simple msgpack encoding."""
  shape, dtype_name, buffer = msgpack.unpackb(data, raw=True)
  return np.frombuffer(buffer,
                       dtype=_dtype_from_name(dtype_name),
                       count=-1,
                       offset=0).reshape(shape, order='C')


class _MsgpackExtType(enum.IntEnum):
  """Messagepack custom type ids."""
  NDARRAY = 1
  NATIVE_COMPLEX = 2
  NPSCALAR = 3
  TUPLE = 4


def _msgpack_ext_pack(x):
  """Messagepack encoders for custom types."""
  # TODO(flax-dev): Array here only work when they are fully addressable.
  # If they are not fully addressable, use the GDA path for checkpointing.
  if isinstance(x, (np.ndarray, jax.Array)):
    return msgpack.ExtType(_MsgpackExtType.NDARRAY, _ndarray_to_bytes(x))
  if issubclass(type(x), np.generic):
    # pack scalar as ndarray
    return msgpack.ExtType(
        _MsgpackExtType.NPSCALAR, _ndarray_to_bytes(np.asarray(x))
    )
  elif isinstance(x, complex):
    return msgpack.ExtType(
        _MsgpackExtType.NATIVE_COMPLEX, msgpack.packb((x.real, x.imag))
    )
  elif isinstance(x, tuple):
    return msgpack.ExtType(
        _MsgpackExtType.TUPLE,
        msgpack.packb(
            list(x),
            strict_types=True,
            use_bin_type=True,
            default=_msgpack_ext_pack,
        ),
    )
  else:
    raise ValueError(f'Unsupported msgpack object: {x}')
  return x


def _msgpack_ext_unpack(code, data):
  """Messagepack decoders for custom types."""
  if code == _MsgpackExtType.NDARRAY:
    return _ndarray_from_bytes(data)
  elif code == _MsgpackExtType.NATIVE_COMPLEX:
    complex_tuple = msgpack.unpackb(data)
    return complex(complex_tuple[0], complex_tuple[1])
  elif code == _MsgpackExtType.NPSCALAR:
    ar = _ndarray_from_bytes(data)
    return ar[()]  # unpack ndarray to scalar
  elif code == _MsgpackExtType.TUPLE:
    return tuple(
        msgpack.unpackb(data, raw=False, ext_hook=_msgpack_ext_unpack)
    )
  else:
    raise ValueError(f'Unsupported msgpack code: {code}')
  return msgpack.ExtType(code, data)


# Chunking array leaves

# msgpack has a hard limit of 2**31 - 1 bytes per object leaf.  To circumvent
# this limit for giant arrays (e.g. embedding tables), we traverse the tree
# and break up arrays near the limit into flattened array chunks.

# True limit is 2**31 - 1, but leave a margin for encoding padding.
MAX_CHUNK_SIZE = 2**30


def _np_convert_in_place(d):
  """Convert any jax devicearray leaves to numpy arrays in place."""
  if isinstance(d, dict):
    for k, v in d.items():
      if isinstance(v, jax.Array):
        d[k] = np.array(v)
      elif isinstance(v, dict):
        _np_convert_in_place(v)
  elif isinstance(d, jax.Array):
    return np.array(d)
  return d


_tuple_to_dict = lambda tpl: {str(x): y for x, y in enumerate(tpl)}
_dict_to_tuple = lambda dct: tuple(dct[str(i)] for i in range(len(dct)))


def _chunk(arr) -> Dict[str, Any]:
  """Convert array to a canonical dictionary of chunked arrays."""
  chunksize = max(1, int(MAX_CHUNK_SIZE / arr.dtype.itemsize))
  data = {'__msgpack_chunked_array__': True,
          'shape': _tuple_to_dict(arr.shape)}
  flatarr = arr.reshape(-1)
  chunks = [flatarr[i:i + chunksize] for i in range(0, flatarr.size, chunksize)]
  data['chunks'] = _tuple_to_dict(chunks)
  return data


def _unchunk(data: Dict[str, Any]):
  """Convert canonical dictionary of chunked arrays back into array."""
  assert '__msgpack_chunked_array__' in data
  shape = _dict_to_tuple(data['shape'])
  flatarr = np.concatenate(_dict_to_tuple(data['chunks']))
  return flatarr.reshape(shape)


def _chunk_array_leaves_in_place(d):
  """Convert oversized array leaves to safe chunked form in place."""
  if isinstance(d, dict):
    for k, v in d.items():
      if isinstance(v, np.ndarray):
        if v.size * v.dtype.itemsize > MAX_CHUNK_SIZE:
          d[k] = _chunk(v)
      elif isinstance(v, dict):
        _chunk_array_leaves_in_place(v)
  elif isinstance(d, np.ndarray):
    if d.size * d.dtype.itemsize > MAX_CHUNK_SIZE:
      return _chunk(d)
  return d


def _unchunk_array_leaves_in_place(d):
  """Convert chunked array leaves back into array leaves, in place."""
  if isinstance(d, dict):
    if '__msgpack_chunked_array__' in d:
      return _unchunk(d)
    else:
      for k, v in d.items():
        if isinstance(v, dict) and '__msgpack_chunked_array__' in v:
          d[k] = _unchunk(v)
        elif isinstance(v, dict):
          _unchunk_array_leaves_in_place(v)
  return d


def msgpack_serialize(pytree, in_place: bool = False) -> bytes:
  """Save data structure to bytes in msgpack format.

  Low-level function that only supports python trees with array leaves,
  for custom objects use `to_bytes`.  It splits arrays above MAX_CHUNK_SIZE into
  multiple chunks.

  Args:
    pytree: python tree of dict, list, tuple with python primitives
      and array leaves.
    in_place: boolean specifyng if pytree should be modified in place.

  Returns:
    msgpack-encoded bytes of pytree.
  """
  if not in_place:
    pytree = jax.tree.map(lambda x: x, pytree)
  pytree = _np_convert_in_place(pytree)
  pytree = _chunk_array_leaves_in_place(pytree)
  return msgpack.packb(pytree, default=_msgpack_ext_pack, strict_types=True)


def msgpack_restore(encoded_pytree: bytes):
  """Restore data structure from bytes in msgpack format.

  Low-level function that only supports python trees with array leaves,
  for custom objects use `from_bytes`.

  Args:
    encoded_pytree: msgpack-encoded bytes of python tree.

  Returns:
    Python tree of dict, list, tuple with python primitive
    and array leaves.
  """
  state_dict = msgpack.unpackb(
      encoded_pytree, ext_hook=_msgpack_ext_unpack, raw=False)
  return _unchunk_array_leaves_in_place(state_dict)
