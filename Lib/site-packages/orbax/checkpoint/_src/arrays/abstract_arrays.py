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

"""Utilities for dealing with abstract arrays."""

from typing import Protocol
import jax
from jax import numpy as jnp
import numpy as np
from orbax.checkpoint._src.arrays import types
from orbax.checkpoint._src.metadata import sharding as sharding_metadata


class AbstractArrayLike(Protocol):
  """Abstract representation of an array.

  Can include objects like jax.Array, jax.ShapeDtypeStruct,
  ArrayRestoreArgs, and value_metadata.ArrayMetadata.
  """

  shape: types.Shape
  dtype: jnp.dtype | None
  sharding: jax.sharding.Sharding | sharding_metadata.ShardingMetadata | None


class AbstractArrayLikeGlobalShape(Protocol):
  """Same as above, but with `global_shape` property instead."""

  global_shape: types.Shape
  dtype: jnp.dtype | None
  sharding: jax.sharding.Sharding | sharding_metadata.ShardingMetadata | None


def _is_scalar(arr):
  return isinstance(arr, (ScalarType, np.number))


def _get_shape(
    arr: AbstractArrayLike | AbstractArrayLikeGlobalShape,
) -> types.Shape:
  if hasattr(arr, 'shape'):
    return arr.shape
  if hasattr(arr, 'global_shape'):
    return arr.global_shape
  raise ValueError(f'Object does not have a `shape` property: {arr}')


ArrayLike = AbstractArrayLike | AbstractArrayLikeGlobalShape | np.ndarray

ScalarType = int | float


def to_shape_dtype_struct(
    arr: ArrayLike,
    dtype: jnp.dtype | None = None,
    scalar_dtype: ScalarType | None = None,
) -> jax.ShapeDtypeStruct | ScalarType:
  """Get ShapeDtypeStruct from array-like object.

  Args:
    arr: Array-like object. This can include jax.Array, jax.ShapeDtypeStruct,
      ArrayRestoreArgs, value_metadata.ArrayMetadata - anything that has
      `shape`/`global_shape`, `dtype`, and `sharding` properties. It may also be
      a numpy array or a scalar value.
    dtype: Optional dtype that overrides the dtype of `arr` in the result.
    scalar_dtype: Optional dtype to use for scalars. Useful for converting to
      Python scalar types.

  Returns:
    jax.ShapeDtypeStruct or scalar value.
  """
  if _is_scalar(arr):
    if scalar_dtype is not None:
      return scalar_dtype(arr)
    return arr
  elif isinstance(arr, np.ndarray):
    dtype = dtype or arr.dtype
    return jax.ShapeDtypeStruct(_get_shape(arr), dtype)
  else:
    shape = _get_shape(arr)
    dtype = dtype or arr.dtype
    sharding = arr.sharding
    if isinstance(sharding, sharding_metadata.ShardingMetadata):
      sharding = sharding.to_jax_sharding()
    return jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
