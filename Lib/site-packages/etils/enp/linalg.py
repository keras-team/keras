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

"""`linalg` compat module for interoperability between tf, jax, numpy."""

from __future__ import annotations

from etils.enp import compat
from etils.enp import numpy_utils
from etils.enp.typing import FloatArray  # pylint: disable=g-multiple-import

lazy = numpy_utils.lazy


def normalize(x: FloatArray['*d'], axis: int = -1) -> FloatArray['*d']:
  """Normalize the vector to the unit norm."""
  return x / compat.norm(x, axis=axis, keepdims=True)
