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

"""Einops utils."""

from typing import TypeVar

from etils import epy

with epy.lazy_imports():
  import einops  # pylint: disable=g-import-not-at-top

_ArrayT = TypeVar('_ArrayT')
_Shape = tuple[int, ...]


def flatten(array: _ArrayT, pattern: str) -> tuple[_ArrayT, _Shape]:
  """Flatten an array along custom dimensions.

  Uses `einops` syntax.

  ```python
  flat_x, batch_shape = enp.flatten(x, '... h w c')
  y = enp.unflatten(y, batch_shape, '... h w c')
  ```

  * `x.shape == (h, w, c)`       -> `flat_x.shape == (1, h, w, c)`
  * `x.shape == (b, h, w, c)`    -> `flat_x.shape == (b, h, w, c)`
  * `x.shape == (b, n, h, w, c)` -> `flat_x.shape == (b * n, h, w, c)`

  Args:
    array: Array to flatten.
    pattern: Einops pattern to flatten the array.

  Returns:
    Tuple of (flattened array, batch shape).
  """
  array, (batch_shape,) = einops.pack([array], pattern.replace('...', '*'))
  return array, tuple(batch_shape)


def unflatten(array: _ArrayT, batch_shape: _Shape, pattern: str) -> _ArrayT:
  (array,) = einops.unpack(array, [batch_shape], pattern.replace('...', '*'))
  return array
