# Copyright 2024 The etils Authors.
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

"""Compat utils between TF/Torch/Numpy/Jax.

Currently, each numpy API has slightly different behavior. Those functions
ensure compatibility so that the code works seamlessly between all APIs.

In the future, those functions could be deleted and replaced by the
official numpy API.
"""

from __future__ import annotations

import functools
import typing
from typing import Any, Optional

from etils.enp import numpy_utils
from etils.enp.typing import Array, FloatArray  # pylint: disable=g-multiple-import
import numpy as np

if typing.TYPE_CHECKING:
  import torch as torch_  # pytype: disable=import-error

_NpDType = Any

lazy = numpy_utils.lazy


# ======== Torch issues ========


@functools.lru_cache()
def _torch_to_np_dtypes() -> dict[torch_.dtype, _NpDType]:
  """Returns mapping torch -> numpy dtypes."""
  torch = lazy.torch
  return {
      torch.bool: np.bool_,
      torch.uint8: np.uint8,
      torch.int8: np.int8,
      torch.int16: np.int16,
      torch.int32: np.int32,
      torch.int64: np.int64,
      # TODO(epot): torch.bfloat:
      torch.float16: np.float16,
      torch.float32: np.float32,
      torch.float64: np.float64,
      torch.complex64: np.complex64,
      torch.complex128: np.complex128,
  }


@functools.lru_cache()
def _np_to_torch_dtypes() -> dict[np.dtype, torch_.dtype]:
  """Returns mapping numpy -> torch dtypes."""
  return dict((np.dtype(n), t) for t, n in _torch_to_np_dtypes().items())


def dtype_torch_to_np(dtype) -> np.dtype:
  """Returns the numpy dtype for the given torch dtype."""
  return _torch_to_np_dtypes()[dtype]


def dtype_np_to_torch(dtype):
  """Returns the torch dtype for the given numpy dtype."""
  return _np_to_torch_dtypes()[np.dtype(dtype)]


def is_array_xnp(x, xnp) -> bool:
  """`isinstance(x, xnp.Array)`."""
  if lazy.has_torch and xnp is lazy.torch:
    return isinstance(x, xnp.Tensor)
  else:
    return isinstance(x, xnp.ndarray)


def astype(x: Array['*d'], dtype) -> Array['*d']:
  """`x.astype(dtype)`."""
  if lazy.is_torch(x):
    return x.type(dtype)
  elif lazy.is_tf(x):
    return lazy.tf.cast(x, dtype)
  else:
    return x.astype(dtype)


def expand_dims(x: Array['*d'], *, axis) -> Array['*d']:
  """`xnp.expand_dims(x, axis=axis)`."""
  xnp = lazy.get_xnp(x)
  if lazy.is_torch(x):
    return xnp.unsqueeze(x, axis=axis)
  else:
    return xnp.expand_dims(x, axis=axis)


def concat(x: list[Array['*d']], *, axis) -> Array['*d']:
  """`xnp.concatenate(x, axis=axis)`."""
  xnp = lazy.get_xnp(x[0])
  if lazy.is_torch(x[0]):
    return xnp.concat(x, axis=axis)
  else:
    return xnp.concatenate(x, axis=axis)


# ======== TF issues ========


def round(x: FloatArray['*d']) -> FloatArray['*d']:  # pylint: disable=redefined-builtin
  """`x.round()` for jnp, tnp, np, otrch."""
  if lazy.is_tf(x):  # TODO(b/219427516): missing method
    return lazy.tnp.around(x)
  return x.round()


def norm(
    x: FloatArray['*d'],
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> FloatArray['*d']:
  """Like `np.linalg.norm` but auto-support jnp, tnp, np."""
  if lazy.is_tf(x):  # TODO(b/219427516): tnp.linalg.norm missing
    return lazy.tf.norm(x, axis=axis, keepdims=keepdims)
  xnp = lazy.get_xnp(x)
  return xnp.linalg.norm(x, axis=axis, keepdims=keepdims)


def inv(x: FloatArray['*d']) -> FloatArray['*d']:
  """Like `np.linalg.inv` but auto-support jnp, tnp, np."""
  return _tf_or_xnp(x).linalg.inv(x)


def det(x: FloatArray['*d m m']) -> FloatArray['*d']:
  """Like `np.linalg.det` but auto-support jnp, tnp, np."""
  return _tf_or_xnp(x).linalg.det(x)


def _tf_or_xnp(x: Array['*d']):
  xnp = lazy.get_xnp(x)
  if lazy.has_tf and xnp is lazy.tnp:
    return lazy.tf
  else:
    return xnp
