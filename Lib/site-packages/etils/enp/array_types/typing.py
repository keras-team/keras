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

"""Typing utils."""

from typing import Any, List, Optional, Tuple, Type, TypeVar, Union

from etils.enp.array_types import dtypes
import numpy as np

_T = TypeVar('_T')

# Match both `np.dtype('int32')` and np.int32
_DType = Union[np.dtype, Type[np.generic], dtypes.DType]

# Shape definition spec (e.g. `h w c`, `batch ...`)
ShapeSpec = str

_EllipsisType = type(Ellipsis)  # TODO(py310): Use types.EllipsisType
_ShapeItem = Union[ShapeSpec, int, _EllipsisType, None]
_ShapeSpecInput = Union[_ShapeItem, Tuple[_ShapeItem, ...]]


class ArrayAliasMeta(type):
  """Metaclass to create array aliases.

  This allow to annotate the array shape/dtype with named axis.
  The dtype is defined by the class name (`f32` for `float32`, `ui8` for
  `uint8`, `Array` for any type).
  The shape is defined either as tuple of `str`, `int` or `...`. (e.g
  `f32['b h w c']`, `f32[32, 256, 256, 3]`, `f32[..., 'h w', 3]`).

  All tuple values are concatenated, so `f32[..., 'h', 'w', 'c']` is the
  same as `f32['... h w c']`.

  """

  shape: ShapeSpec
  dtype: dtypes.DType

  def __new__(  # pylint: disable=bad-mcs-classmethod-argument
      cls,
      shape: Optional[_ShapeSpecInput],
      dtype: Optional[_DType],
  ):
    dtype = dtypes.DType.from_value(dtype)
    # Normalize to str
    if shape is None:
      shape = '...'
    elif isinstance(shape, tuple):
      shape = ' '.join(_normalize_shape_item(x) for x in shape)
    else:
      shape = _normalize_shape_item(shape)
    return super().__new__(
        cls,
        dtype.array_cls_name,
        (cls,),
        {
            'shape': shape,
            'dtype': dtype,
        },
    )

  def __init__(cls, shape: Optional[ShapeSpec], dtype: Optional[_DType]):
    del shape, dtype
    super().__init__(cls, cls.__name__, (cls,), {})  # pytype: disable=wrong-arg-count

  def __getitem__(cls, shape: _ShapeSpecInput) -> 'ArrayAliasMeta':
    if shape is None:  # Normalize 'Array[None]'
      shape = (shape,)
    return ArrayAliasMeta(shape=shape, dtype=cls.dtype)

  def __eq__(cls, other: 'ArrayAliasMeta') -> bool:
    return (
        isinstance(other, ArrayAliasMeta)
        and cls.shape == other.shape
        and cls.dtype == other.dtype
    )

  def __hash__(cls) -> int:
    return hash((cls.shape, cls.dtype))

  def __repr__(cls) -> str:
    return f'{cls.__name__}[{cls.shape}]'

  def __instancecheck__(cls, instance: np.ndarray) -> bool:
    """`isinstance(array, f32['h w c'])`."""
    raise NotImplementedError


def _normalize_shape_item(item: _ShapeItem) -> ShapeSpec:
  """Returns the `str` representation associated with the shape element."""
  if isinstance(item, str):
    return item
  elif isinstance(item, int):
    return str(item)
  elif isinstance(item, _EllipsisType):
    return '...'
  elif item is None:
    return '_'
  else:
    raise TypeError(f'Invalid shape type {type(item)} of: {item}')


_ArrayT = TypeVar('_ArrayT', bound=ArrayAliasMeta)

# ArrayLike indicates that any `np.array` input is also supported.
# For example: `ArrayLike[i32[2]]` accept `(28, 28)`, `[x, y]`, `np.ones((2,))`
ArrayLike = Union[_ArrayT, Tuple[Any, ...], List[Any]]
