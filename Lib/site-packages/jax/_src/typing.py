# Copyright 2022 The JAX Authors.
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
`jax._src.typing`: JAX type annotations
---------------------------------------

This submodule is a work in progress; when we finalize the contents here, it will be
exported at `jax.typing`. Until then, the contents here should be considered unstable
and may change without notice.

To see the proposal that led to the development of these tools, see
https://github.com/jax-ml/jax/pull/11859/.
"""

from __future__ import annotations

from collections.abc import Sequence
import enum
import typing
from typing import Any, Protocol, Union

from jax._src.basearray import (
    ArrayLike as ArrayLike,
    Array as Array,
    StaticScalar as StaticScalar,
)
import numpy as np

DType = np.dtype

# TODO(jakevdp, froystig): make ExtendedDType a protocol
ExtendedDType = Any


@typing.runtime_checkable
class SupportsDType(Protocol):
  @property
  def dtype(self) -> DType: ...

# DTypeLike is meant to annotate inputs to np.dtype that return
# a valid JAX dtype. It's different than numpy.typing.DTypeLike
# because JAX doesn't support objects or structured dtypes.
# Unlike np.typing.DTypeLike, we exclude None, and instead require
# explicit annotations when None is acceptable.
# TODO(jakevdp): consider whether to add ExtendedDtype to the union.
DTypeLike = Union[
  str,            # like 'float32', 'int32'
  type[Any],      # like np.float32, np.int32, float, int
  np.dtype,       # like np.dtype('float32'), np.dtype('int32')
  SupportsDType,  # like jnp.float32, jnp.int32
]

# Shapes are tuples of dimension sizes, which are normally integers. We allow
# modules to extend the set of dimension sizes to contain other types, e.g.,
# symbolic dimensions in export.DimExpr.
DimSize = Union[int, Any]  # extensible
Shape = Sequence[DimSize]

class DuckTypedArray(Protocol):
  @property
  def dtype(self) -> DType: ...
  @property
  def shape(self) -> Shape: ...

# Array is a type annotation for standard JAX arrays and tracers produced by
# core functions in jax.lax and jax.numpy; it is not meant to include
# future non-standard array types like KeyArray and BInt. It is imported above.

# ArrayLike is a Union of all objects that can be implicitly converted to a standard
# JAX array (i.e. not including future non-standard array types like KeyArray and BInt).
# It's different than np.typing.ArrayLike in that it doesn't accept arbitrary sequences,
# nor does it accept string data.

# We use a class for deprecated args to avoid using Any/object types which can
# introduce complications and mistakes in static analysis
class DeprecatedArg:
  def __repr__(self):
    return "Deprecated"

# Mirror of dlpack.h enum
class DLDeviceType(enum.IntEnum):
  kDLCPU = 1
  kDLCUDA = 2
  kDLROCM = 10
