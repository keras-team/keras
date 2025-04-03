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

"""Grouped matrix multiplication operations with custom VJPs."""

import jax
from jax.experimental.pallas.ops.tpu.megablox import gmm as backend
import jax.numpy as jnp


gmm = jax.custom_vjp(
    backend.gmm,
    nondiff_argnums=(3, 4, 7, 8),
)


def _gmm_fwd(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] = (128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> tuple[
    jnp.ndarray,
    tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray | None,
        int,
    ],
]:
  """Forward function for GMM VJP."""
  out = backend.gmm(
      lhs,
      rhs,
      group_sizes,
      preferred_element_type,
      tiling,
      group_offset,
      existing_out,
      transpose_rhs=transpose_rhs,
      interpret=interpret,
  )
  return out, (lhs, rhs, group_sizes, group_offset, rhs.shape[0])


def _gmm_bwd(
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int],
    transpose_rhs: bool,
    interpret: bool,
    residual: tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray | None,
        int,
    ],
    grad: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, None, None, jnp.ndarray]:
  """Backward function for throughput GMM VJP."""
  del preferred_element_type
  lhs, rhs, group_sizes, group_offset, num_actual_groups = residual
  grad_lhs = backend.gmm(
      grad,
      rhs,
      group_sizes,
      lhs[0].dtype,
      tiling,
      group_offset,
      transpose_rhs=not transpose_rhs,
      interpret=interpret,
  )
  grad_rhs = backend.tgmm(
      lhs.swapaxes(0, 1),
      grad,
      group_sizes,
      rhs.dtype,
      tiling,
      group_offset,
      num_actual_groups,
      interpret=interpret,
  )

  # NOTE: If the rhs transposition is fused into the forward pass we need to
  # return the transpose of the rhs gradient that we calculated above.
  #
  # TODO(tgale, enriqueps, apaske): Fuse this transposition into the tgmm.
  grad_rhs = grad_rhs.swapaxes(1, 2) if transpose_rhs else grad_rhs
  return grad_lhs, grad_rhs, None, None, grad


gmm.defvjp(_gmm_fwd, _gmm_bwd)
