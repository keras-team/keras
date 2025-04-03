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

"""Typing utils for arrays.

See doc at: https://github.com/google/etils/blob/main/etils/array_types/README.md

"""

# This is an alias for `enp.typing`.

from etils import enp

ArrayAliasMeta = enp.typing.ArrayAliasMeta
ArrayLike = enp.typing.ArrayLike

Array = enp.typing.Array
FloatArray = enp.typing.FloatArray
IntArray = enp.typing.IntArray
BoolArray = enp.typing.BoolArray
StrArray = enp.typing.StrArray

ui8 = enp.typing.ui8
ui16 = enp.typing.ui16
ui32 = enp.typing.ui32
ui64 = enp.typing.ui64
i8 = enp.typing.i8
i16 = enp.typing.i16
i32 = enp.typing.i32
i64 = enp.typing.i64
f16 = enp.typing.f16
f32 = enp.typing.f32
f64 = enp.typing.f64
complex64 = enp.typing.complex64
complex128 = enp.typing.complex128
bool_ = enp.typing.bool_

# Random number generator jax key
PRNGKey = enp.typing.PRNGKey

# Keep API clean
del enp
