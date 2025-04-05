# Copyright 2023 The JAX Authors.
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

"""Example matmul TPU kernel.

See discussion in https://jax.readthedocs.io/en/latest/pallas/tpu/matmul.html.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def matmul_kernel(x_tile_ref, y_tile_ref, o_tile_ref, acc_ref):
  @pl.when(pl.program_id(2) == 0)
  def init():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  acc_ref[...] = acc_ref[...] + jnp.dot(
      x_tile_ref[...],
      y_tile_ref[...],
      preferred_element_type=acc_ref.dtype,
  )
  # It is possible to make this conditional but in general this bundle packs
  # quite well for a simple matmul kernel
  o_tile_ref[...] = acc_ref[...].astype(o_tile_ref.dtype)


@functools.partial(
    jax.jit, static_argnames=["block_shape", "block_k", "debug", "out_dtype"]
)
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    block_shape,
    block_k: int = 256,
    out_dtype: jnp.dtype | None = None,
    debug: bool = False,
) -> jax.Array:
  if out_dtype is None:
    if x.dtype != y.dtype:
      # TODO(tlongeri): Maybe we could use a deduction similar to jnp.dot
      raise TypeError(
          f"Cannot deduce output dtype for different input dtypes: {x.dtype},"
          f" {y.dtype}"
      )
    out_dtype = x.dtype
  acc_dtype = jnp.float32
  if x.dtype in [jnp.int8, jnp.int4, jnp.uint8, jnp.uint4]:
    acc_dtype = jnp.int32

  l, r = block_shape
  return pl.pallas_call(
      matmul_kernel,
      out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), out_dtype),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((l, block_k), lambda i, _, k: (i, k)),
              pl.BlockSpec((block_k, r), lambda _, j, k: (k, j)),
          ],
          out_specs=pl.BlockSpec((l, r), lambda i, j, k: (i, j)),
          grid=(x.shape[0] // l, y.shape[1] // r, x.shape[1] // block_k),
          scratch_shapes=[pltpu.VMEM((l, r), acc_dtype)],
      ),
      compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
      debug=debug,
  )(x, y)
