# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Type definitions to use for type annotations."""

from typing import Any, Iterable, Mapping, Sequence, Union

import jax
import numpy as np

# Special types of arrays.
ArrayNumpy = np.ndarray

# For instance checking, use `isinstance(x, jax.Array)`.
ArrayDevice = jax.Array

# Types for backward compatibility.
ArraySharded = jax.Array
ArrayBatched = jax.Array

# Generic array type.
# Similar to `jax.typing.ArrayLike` but does not accept python scalar types.
Array = Union[
    ArrayDevice,
    ArrayBatched,
    ArraySharded,  # JAX array type
    ArrayNumpy,  # NumPy array type
    np.bool_,
    np.number,  # NumPy scalar types
]

# A tree of generic arrays.
ArrayTree = Union[Array, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]
ArrayDeviceTree = Union[
    ArrayDevice, Iterable['ArrayDeviceTree'], Mapping[Any, 'ArrayDeviceTree']
]
ArrayNumpyTree = Union[
    ArrayNumpy, Iterable['ArrayNumpyTree'], Mapping[Any, 'ArrayNumpyTree']
]

# Other types.
Scalar = Union[float, int]
Numeric = Union[Array, Scalar]
Shape = Sequence[Union[int, Any]]
PRNGKey = jax.Array
PyTreeDef = jax.tree_util.PyTreeDef
Device = jax.Device

# TODO(iukemaev, jakevdp): upgrade minimum jax version & remove this condition.
if hasattr(jax.typing, 'DTypeLike'):
  # jax version 0.4.19 or newer
  ArrayDType = jax.typing.DTypeLike  # pylint:disable=invalid-name
else:
  ArrayDType = Any  # pylint:disable=invalid-name
