# Copyright 2025 The JAX Authors.
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
"""Pallas helper functions."""

from typing import Any, Protocol

import jax
import jax.numpy as jnp
from jax._src.pallas import pallas_call
from jax._src.pallas import core as pl_core


@jax.named_call
def empty(
    shape: tuple[int, ...], dtype: jnp.dtype, *, memory_space: Any = None
):
  def _empty_kernel(_):
    # No-op to leave the out_ref uninitialized
    pass

  if memory_space is None:
    kernel_memory_space = pl_core.MemorySpace.ANY
    memory_space = jax.ShapeDtypeStruct
  else:
    kernel_memory_space = memory_space
  return pallas_call.pallas_call(
      _empty_kernel,
      in_specs=[],
      out_specs=pl_core.BlockSpec(memory_space=kernel_memory_space),
      out_shape=memory_space(shape, dtype),
  )()


class ArrayLike(Protocol):
  shape: tuple[int, ...]
  dtype: jnp.dtype


def empty_like(x: ArrayLike, *, memory_space: Any = None):
  return empty(x.shape, x.dtype, memory_space=memory_space)
