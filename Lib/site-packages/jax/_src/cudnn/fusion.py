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

import functools
import jax
from jax._src import core as jax_core
from jax.interpreters import mlir
from jax.interpreters.mlir import hlo
from jax.interpreters.mlir import ir



def _cudnn_fusion_impl(*args, jaxpr, **unused_kwargs):
  del unused_kwargs
  return jax_core.jaxpr_as_fun(jaxpr)(*args)


def _custom_abstract_eval(*args, jaxpr, **unused_kwargs):
  del unused_kwargs
  del args
  return jaxpr.out_avals


cudnn_fusion_p = jax_core.Primitive("cudnn_fusion")
cudnn_fusion_p.multiple_results = True
cudnn_fusion_p.def_abstract_eval(_custom_abstract_eval)
cudnn_fusion_p.def_impl(_cudnn_fusion_impl)


def call_cudnn_fusion(f, *args, **kwargs):
  """Creates a new cudnn_fusion corresponding to calling
  the given function f with args and kwargs."""
  jaxpr, out_shapes = jax.make_jaxpr(
    functools.partial(f, **kwargs), return_shape=True
  )(*args)
  flat_args = jax.tree.leaves(args)
  out_tree = jax.tree.structure(out_shapes)
  out_flat = cudnn_fusion_p.bind(*flat_args, name=f.__name__, jaxpr=jaxpr)
  return jax.tree.unflatten(out_tree, out_flat)


def _cudnn_fusion_stablehlo_lowering(
  ctx,
  *args,
  name,
  jaxpr,
):
  """Make cudnn_fusion which calls the implementation function.
  Currently this leaks a CallOp since we're using the `core_call_lowering`
  function, but this should get cleaned up by DCE easily.
  """
  impl = mlir.core_call_lowering(
    ctx, *args, name=name + ".impl", call_jaxpr=jaxpr
  )
  call_op = impl[0].owner
  called_fn = call_op.attributes["callee"]
  cudnn_fusion = hlo.CustomCallOp(
    [r.type for r in call_op.results],
    call_op.operands,
    call_target_name="__cudnn$fusion",
    called_computations=ir.ArrayAttr.get([called_fn]),
  )
  return cudnn_fusion.results


mlir.register_lowering(
    cudnn_fusion_p, _cudnn_fusion_stablehlo_lowering, platform="cuda"
  )


def cudnn_fusion(f):
  """Makes a function become a cuDNN kernel. Relies on XLA's handling of
  custom fusions with __cudnn$fusion backend. Currently limited to GEMM
  fusions. For example - batch matmul with mixed types and addition:

  @cudnn_fusion
  def fn(x, y, z):
      return jnp.float32(jax.lax.batch_matmul(jnp.bfloat16(x), y)) + z
  """
  return functools.partial(call_cudnn_fusion, f)
