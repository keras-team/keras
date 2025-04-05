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

# pylint: disable=g-importing-member,unused-import

from etils.enp.array_types import dtypes
from etils.enp.array_types.typing import ArrayAliasMeta
from etils.enp.array_types.typing import ArrayLike
import numpy as np

# Automated documentation info
# See: https://github.com/conchylicultor/sphinx-apitree
__apitree__ = dict(
    is_package=True,
)

Array = ArrayAliasMeta(shape=None, dtype=dtypes.AnyDType())
FloatArray = ArrayAliasMeta(shape=None, dtype=dtypes.AnyFloat())
IntArray = ArrayAliasMeta(shape=None, dtype=dtypes.AnyInt())
BoolArray = ArrayAliasMeta(shape=None, dtype=np.bool_)
StrArray = ArrayAliasMeta(shape=None, dtype=np.dtype('O'))

ui8 = ArrayAliasMeta(shape=None, dtype=np.uint8)
ui16 = ArrayAliasMeta(shape=None, dtype=np.uint16)
ui32 = ArrayAliasMeta(shape=None, dtype=np.uint32)
ui64 = ArrayAliasMeta(shape=None, dtype=np.uint64)
i8 = ArrayAliasMeta(shape=None, dtype=np.int8)
i16 = ArrayAliasMeta(shape=None, dtype=np.int16)
i32 = ArrayAliasMeta(shape=None, dtype=np.int32)
i64 = ArrayAliasMeta(shape=None, dtype=np.int64)
f16 = ArrayAliasMeta(shape=None, dtype=np.float16)
f32 = ArrayAliasMeta(shape=None, dtype=np.float32)
f64 = ArrayAliasMeta(shape=None, dtype=np.float64)
complex64 = ArrayAliasMeta(shape=None, dtype=np.complex64)
complex128 = ArrayAliasMeta(shape=None, dtype=np.complex128)
bool_ = ArrayAliasMeta(shape=None, dtype=np.bool_)


# Random number generator jax key
PRNGKey = ui32[2]

# Keep API clean
del np, dtypes
