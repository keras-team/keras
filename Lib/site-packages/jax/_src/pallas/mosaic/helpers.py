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

"""Helpers for Pallas TPU kernels."""

import functools
import jax
from jax._src.pallas import primitives as pl_primitives
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import primitives as plm_primitives


def sync_copy(src_ref, dst_ref):
  """Copies a PyTree of Refs to another PyTree of Refs.

  Args:
    src_ref: A Pytree of source Refs/TransformedRefs.
    dst_ref: A Pytree of destination Refs/TransformedRefs.
  """
  if not jax.tree.leaves(src_ref):
    # No buffers to copy so skip the function.
    return

  @functools.partial(
      pl_primitives.run_scoped, sem=tpu_core.SemaphoreType.DMA(())
  )
  def _(sem):
    def _copy_start_or_wait(action, src_ref, dst_ref):
      descriptor = plm_primitives.make_async_copy(src_ref, dst_ref, sem)
      if action == "start":
        descriptor.start()
      elif action == "wait":
        descriptor.wait()
      else:
        raise ValueError(f"Unknown action: {action}")

    jax.tree.map(
        functools.partial(_copy_start_or_wait, "start"),
        src_ref,
        dst_ref,
    )
    jax.tree.map(
        functools.partial(_copy_start_or_wait, "wait"),
        src_ref,
        dst_ref,
    )
