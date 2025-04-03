# Copyright 2024 The JAX Authors.
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

"""Common utilities for GMM kernels."""

import re

import jax
import jax.numpy as jnp


def is_tpu() -> bool:
  return "TPU" in jax.devices()[0].device_kind


def tpu_kind() -> str:
  """Query identification string for the currently attached TPU."""
  return jax.devices()[0].device_kind


_TPU_KIND_PATTERN = re.compile(r"TPU v(\d+)")


def tpu_generation() -> int:
  """Generation number of the currently attached TPU."""
  if version := _TPU_KIND_PATTERN.match(tpu_kind()):
    return int(version[1])
  raise NotImplementedError("only TPU devices are supported")


def supports_bfloat16_matmul() -> bool:
  """Does the currently attached CPU support bfloat16 inputs?"""
  return not is_tpu() or tpu_generation() >= 4


def assert_is_supported_dtype(dtype: jnp.dtype) -> None:
  if dtype != jnp.bfloat16 and dtype != jnp.float32:
    raise ValueError(f"Expected bfloat16 or float32 array but got {dtype}.")


def select_input_dtype(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.dtype:
  """A type to which both input should be adapted to before dot product."""
  # bf16xbf16 matmul is only supported since TPUv4 generation. In case of mixed
  # input precision, we need to convert bf16 argument to fp32 beforehand.
  if (
      supports_bfloat16_matmul()
      and lhs.dtype == jnp.bfloat16
      and rhs.dtype == jnp.bfloat16
  ):
    return jnp.bfloat16
  else:
    return jnp.float32
