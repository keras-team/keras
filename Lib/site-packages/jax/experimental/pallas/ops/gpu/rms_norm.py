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

"""Module containing rms forward and backward pass."""

from __future__ import annotations

import functools

import jax
from jax import lax
import jax.numpy as jnp
from jax._src.lax.control_flow.for_loop import for_loop

from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

def rms_norm_forward_kernel(
    x_ref, weight_ref, bias_ref, # Input arrays
    o_ref, rstd_ref=None, # Output arrays
    *, eps: float, block_size: int):
  n_col = x_ref.shape[0]

  def var_body(i, acc_ref):
    col_idx = i * block_size + jnp.arange(block_size)
    mask = col_idx < n_col
    a = pl.load(x_ref, (col_idx,), mask=mask, other=0.,
                eviction_policy="evict_last").astype(jnp.float32)
    a = jnp.where(mask, a, 0.)
    acc_ref[:] += a * a
  var = for_loop(pl.cdiv(n_col, block_size), var_body,
                 jnp.zeros(block_size)).sum() / n_col
  rstd = 1 / jnp.sqrt(var + eps)
  if rstd_ref is not None:
    rstd_ref[...] = rstd.astype(rstd_ref.dtype)

  def body(i, _):
    col_idx = i * block_size + jnp.arange(block_size)
    mask = col_idx < n_col
    weight = pl.load(weight_ref, (col_idx,), mask=mask)
    bias = pl.load(bias_ref, (col_idx,), mask=mask)
    x = pl.load(x_ref, (col_idx,), mask=mask, other=0.,
                eviction_policy="evict_first").astype(jnp.float32)
    out = x * rstd * weight + bias
    pl.store(o_ref, (col_idx,), out.astype(o_ref.dtype), mask=mask)
  for_loop(pl.cdiv(n_col, block_size), body, ())


def rms_norm_forward(
    x, weight, bias,
    num_warps: int | None = None,
    num_stages: int | None = 3,
    eps: float = 1e-5,
    backward_pass_impl: str = 'triton',
    interpret: bool = False):
  del num_stages
  del backward_pass_impl
  n = x.shape[-1]
  # Triton heuristics
  # Less than 64KB per feature: enqueue fused kernel
  max_fused_size = 65536 // x.dtype.itemsize
  block_size = min(max_fused_size, pl.next_power_of_2(n))
  block_size = min(max(block_size, 128), 4096)
  num_warps = min(max(block_size // 256, 1), 8)

  kernel = functools.partial(rms_norm_forward_kernel, eps=eps,
                             block_size=block_size)
  out_shape = [
          jax.ShapeDtypeStruct(shape=(n,), dtype=x.dtype),
          jax.ShapeDtypeStruct(shape=(), dtype=x.dtype)
  ]
  method = pl.pallas_call(
      kernel,
      compiler_params=plgpu.TritonCompilerParams(num_warps=num_warps),
      grid=(),
      out_shape=out_shape,
      debug=False,
      interpret=interpret,
      name="rms_forward",
  )

  method = jax.vmap(jax.vmap(method, in_axes=(0, None, None)), in_axes=(0, None, None))
  out, rstd = method(x, weight, bias)
  return out, (x, weight, bias, rstd)


def rms_norm_backward_kernel_dx(
    # Inputs
    x_ref, weight_ref, bias_ref, do_ref,
    rstd_ref,
    # Outputs
    dx_ref,
    *, eps: float, block_size: int):
  n_col = x_ref.shape[0]

  def mean_body(i, c1_acc_ref):
    col_idx = i * block_size + jnp.arange(block_size)
    mask = col_idx < n_col
    a = pl.load(x_ref, (col_idx,), mask=mask, other=0.,
                eviction_policy="evict_last").astype(jnp.float32)
    dout = pl.load(do_ref, (col_idx,), mask=mask, other=0.,
                   eviction_policy="evict_last").astype(jnp.float32)
    weight = pl.load(weight_ref, (col_idx,), mask=mask, other=0.,
                     eviction_policy="evict_last").astype(jnp.float32)
    a_hat = a * rstd_ref[...]
    wdout = weight * dout
    c1_acc_ref[:] += a_hat * wdout
  c1 = for_loop(pl.cdiv(n_col, block_size), mean_body, jnp.zeros(block_size))
  c1 = c1.sum() / n_col

  def dx_body(i, acc_ref):
    col_idx = i * block_size + jnp.arange(block_size)
    mask = col_idx < n_col
    a = pl.load(x_ref, (col_idx,), mask=mask, other=0.,
                eviction_policy="evict_last").astype(jnp.float32)
    dout = pl.load(do_ref, (col_idx,), mask=mask, other=0.,
                eviction_policy="evict_last").astype(jnp.float32)
    weight = pl.load(weight_ref, (col_idx,), mask=mask, other=0.,
                eviction_policy="evict_last").astype(jnp.float32)
    a_hat = a * rstd_ref[...]
    wdout = weight * dout
    da = (wdout - (a_hat * c1)) * rstd_ref[...]
    pl.store(dx_ref, (col_idx,), da.astype(dx_ref.dtype), mask=mask)
  for_loop(pl.cdiv(n_col, block_size), dx_body, ())


def rms_norm_backward_kernel_dw_db(
    # Inputs
    x_ref, weight_ref, bias_ref, do_ref,
    rstd_ref,
    # Outputs
    dw_ref, db_ref,
    *, eps: float, block_m: int, block_n: int):
  m, n_col = x_ref.shape
  j = pl.program_id(0)
  col_idx = j * block_n + jnp.arange(block_n)
  col_mask = col_idx < n_col

  def body(i, acc_ref):
    row_idx = i * block_m + jnp.arange(block_m)
    row_mask = row_idx < m
    mask = row_mask[:, None] & col_mask[None, :]
    a = pl.load(
        x_ref, (row_idx[:, None], col_idx[None]), mask=mask, other=0.0
    ).astype(jnp.float32)
    dout = pl.load(
        do_ref, (row_idx[:, None], col_idx[None]), mask=mask, other=0.0
    ).astype(jnp.float32)
    rstd = pl.load(rstd_ref, (row_idx,), mask=row_mask, other=0.).astype(jnp.float32)
    a_hat = a * rstd[:, None]
    dw_acc_ref, db_acc_ref = acc_ref
    dw_acc_ref[:] += (dout * a_hat).sum(axis=0)
    db_acc_ref[:] += dout.sum(axis=0)
  dw_acc, db_acc = for_loop(pl.cdiv(m, block_m), body, (jnp.zeros(block_n), jnp.zeros(block_n)))
  pl.store(dw_ref, (col_idx,), dw_acc.astype(dw_ref.dtype), mask=col_mask)
  pl.store(db_ref, (col_idx,), db_acc.astype(db_ref.dtype), mask=col_mask)


def rms_norm_backward(
    num_warps: int | None,
    num_stages: int | None,
    eps: float,
    backward_pass_impl: str,
    interpret: bool,
    res, do):
  del num_stages
  x, weight, bias, rstd = res
  if backward_pass_impl == 'xla':
    return jax.vjp(rms_norm_reference, x, weight, bias)[1](do)

  *shape_prefix, n = x.shape
  reshaped_x = x.reshape((-1, n))
  reshaped_rstd = rstd.reshape((-1,))
  reshaped_do = do.reshape((-1, n))
  # Triton heuristics
  # Less than 64KB per feature: enqueue fused kernel
  max_fused_size = 65536 // x.dtype.itemsize
  block_size = min(max_fused_size, pl.next_power_of_2(n))
  block_size = min(max(block_size, 128), 4096)
  num_warps = min(max(block_size // 256, 1), 8)

  # rms_norm_backward_kernel_dx parallel over batch dims
  kernel = functools.partial(rms_norm_backward_kernel_dx, eps=eps,
                             block_size=block_size)
  out_shape_dx = jax.ShapeDtypeStruct(shape=(n,), dtype=x.dtype)
  method = pl.pallas_call(
      kernel,
      compiler_params=plgpu.TritonCompilerParams(num_warps=num_warps),
      grid=(),
      out_shape=out_shape_dx,
      debug=False,
      interpret=interpret,
      name="ln_backward_dx",
  )

  method = jax.vmap(method, in_axes=(0, None, None, 0, 0))
  dx = method(reshaped_x, weight, bias, reshaped_do, reshaped_rstd)
  dx = dx.reshape((*shape_prefix, n))

  # rms_norm_backward_kernel_dw_db reduce over batch dims
  # Triton heuristics
  if n > 10240:
    block_n = 128
    block_m = 32
    num_warps = 4
  else:
    # maximize occupancy for small N
    block_n = 16
    block_m = 16
    num_warps = 8
  kernel = functools.partial(rms_norm_backward_kernel_dw_db, eps=eps,
                             block_m=block_m, block_n=block_n)
  out_shape_dwbias = [
          jax.ShapeDtypeStruct(shape=weight.shape, dtype=weight.dtype),
          jax.ShapeDtypeStruct(shape=bias.shape, dtype=bias.dtype)
  ]
  grid_ = (pl.cdiv(reshaped_x.shape[1], block_n),)
  method = pl.pallas_call(
      kernel,
      compiler_params=dict(triton=dict(num_warps=num_warps)),
      grid=grid_,
      out_shape=out_shape_dwbias,
      debug=False,
      interpret=interpret,
      name="ln_backward_dw_db",
  )
  dw, dbias = method(reshaped_x, weight, bias, reshaped_do, reshaped_rstd)
  return dx, dw, dbias


@functools.partial(jax.custom_vjp, nondiff_argnums=[3, 4, 5, 6, 7])
@functools.partial(jax.jit, static_argnames=["num_warps", "num_stages",
                                             "num_stages", "eps",
                                             "backward_pass_impl",
                                             "interpret"])
def rms_norm(
    x, weight, bias,
    num_warps: int | None = None,
    num_stages: int | None = 3,
    eps: float = 1e-5,
    backward_pass_impl: str = 'triton',
    interpret: bool = False):
  n = x.shape[-1]
  # Triton heuristics
  # Less than 64KB per feature: enqueue fused kernel
  max_fused_size = 65536 // x.dtype.itemsize
  block_size = min(max_fused_size, pl.next_power_of_2(n))
  block_size = min(max(block_size, 128), 4096)
  num_warps = min(max(block_size // 256, 1), 8)

  kernel = functools.partial(rms_norm_forward_kernel, eps=eps,
                             block_size=block_size)
  out_shape = jax.ShapeDtypeStruct(shape=(n,), dtype=x.dtype)
  method = pl.pallas_call(
      kernel,
      compiler_params=dict(
          triton=dict(num_warps=num_warps, num_stages=num_stages)
      ),
      grid=(),
      out_shape=out_shape,
      debug=False,
      interpret=interpret,
  )
  method = jax.vmap(jax.vmap(method, in_axes=(0, None, None)), in_axes=(0, None, None))
  return method(x, weight, bias)
rms_norm.defvjp(rms_norm_forward, rms_norm_backward)


@functools.partial(jax.jit, static_argnames=["eps"])
@functools.partial(jax.vmap, in_axes=(0, None, None), out_axes=0)
def rms_norm_reference(x, weight, bias, *, eps: float = 1e-5):
  var = jnp.mean(jnp.square(x), axis=1)
  mul = lax.rsqrt(var + eps)
  return x * mul[:, None] * weight[None] + bias[None]
