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
"""FlashAttention3 implementation (using Mosaic GPU as the backend)."""

import dataclasses
import functools
import itertools
import math
import jax
from jax import lax
from jax._src import test_util as jtu  # noqa: F401
from jax.experimental.mosaic.gpu import profiler
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class TuningConfig:
  block_q: int
  block_kv: int
  max_concurrent_steps: int

  def __post_init__(self):
    if self.block_q % 64:
      raise ValueError(f"{self.block_q=} must be a multiple of 64")
    if self.block_kv % 64:
      raise ValueError(f"{self.block_kv=} must be a multiple of 64")
    if self.max_concurrent_steps < 2:
      raise ValueError(f"{self.max_concurrent_steps=} must be at least 2")


@functools.partial(jax.jit, static_argnames=["config"])
def attention(q, k, v, config: TuningConfig):
  if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
    raise ValueError(f"q, k, and v should all be 4D, got: {q.ndim=}, {k.ndim=}, {v.ndim=}")
  batch_size, q_seq_len, num_q_heads, head_dim = q.shape
  _, kv_seq_len, num_kv_heads, _ = k.shape
  kv_shape = (batch_size, kv_seq_len, num_kv_heads, head_dim)
  if k.shape != kv_shape:
    raise ValueError(f"Expected {k.shape=} to be {kv_shape} (inferred from q)")
  if k.shape != kv_shape:
    raise ValueError(f"Expected {v.shape=} to be {kv_shape} (inferred from q)")
  if (dtype := q.dtype) != k.dtype or dtype != v.dtype:
    raise ValueError(f"q, k, and v should all have the same dtype, got: {q.dtype}, {k.dtype}, {v.dtype}")
  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_q_heads=} must be divisible by and {num_kv_heads=}")
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if head_dim % 64:
    raise ValueError(f"{head_dim=} must be divisible by 64")
  if jnp.dtype(dtype) not in map(jnp.dtype, [jnp.float16, jnp.bfloat16]):
    raise NotImplementedError(f"Only f16 and bf16 are supported, got dtype: {dtype}")

  max_concurrent_steps = min(
      config.max_concurrent_steps, kv_seq_len // config.block_kv
  )
  block_q, block_kv = config.block_q, config.block_kv

  def kernel(q_ref, k_ref, v_ref, out_ref, scoped):
    batch = lax.axis_index("batch")
    smem_buffers, buffer_barriers, consumed_barriers, schedule_barrier = scoped
    wg_idx = lax.axis_index("wg")
    qo_smem2, k_smem, v_smem = smem_buffers
    k_barriers, v_barriers, q_barriers = buffer_barriers
    k_consumed_barriers, v_consumed_barriers = consumed_barriers
    def perform_schedule_barrier():
      plgpu.barrier_arrive(schedule_barrier)
      plgpu.barrier_wait(schedule_barrier)

    @pl.when(wg_idx < 2)
    def _compute_wg():
      plgpu.set_max_registers(232, action="increase")
      qo_smem = qo_smem2.at[wg_idx]
      q_seq_base = lax.axis_index("q_seq") * (2 * block_q) + wg_idx * block_q
      q_head = lax.axis_index("heads")

      plgpu.copy_gmem_to_smem(
          q_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
          qo_smem,
          q_barriers.at[wg_idx],
      )
      plgpu.barrier_wait(q_barriers.at[wg_idx])

      m_i = plgpu.layout_cast(
          jnp.full((block_q,), -jnp.inf, dtype=jnp.float32), plgpu.Layout.WGMMA_ROW,
      )
      l_i = plgpu.layout_cast(
          jnp.full((block_q,), 0, dtype=jnp.float32), plgpu.Layout.WGMMA_ROW,
      )
      acc = plgpu.layout_cast(
          jnp.full((block_q, head_dim), 0, dtype=jnp.float32), plgpu.Layout.WGMMA,
      )

      plgpu.barrier_wait(k_barriers.at[0])

      pl.when(wg_idx == 1)(perform_schedule_barrier)
      def kv_loop(kv_step, carry):
        acc, m_i, l_i = carry
        slot = lax.rem(kv_step, max_concurrent_steps)

        # QK
        def compute_qk(acc_ref):
          plgpu.wgmma(acc_ref, qo_smem, plgpu.transpose_ref(k_smem.at[slot], (1, 0)))
          perform_schedule_barrier()
          return acc_ref[...]
        qk = pl.run_scoped(compute_qk, plgpu.ACC((block_q, block_kv), jnp.float32))
        plgpu.barrier_arrive(k_consumed_barriers.at[slot])

        # Softmax
        # We keep m scaled by log2e to use FMA instructions when computing p.
        log2e = math.log2(math.e)
        m_ij = jnp.maximum(m_i, qk.max(axis=1) * log2e)
        alpha = jnp.exp2(m_i - m_ij)
        m_i = m_ij
        p = jnp.exp2(qk * log2e - lax.broadcast_in_dim(m_ij, qk.shape, [0]))
        acc *= lax.broadcast_in_dim(alpha, acc.shape, [0])
        l_i *= alpha
        p16 = p.astype(dtype)

        def end_softmax_barriers():
          plgpu.barrier_arrive(schedule_barrier)  # Done with softmax!
          plgpu.barrier_wait(v_barriers.at[slot])
          plgpu.barrier_wait(schedule_barrier)  # Wait until TensorCore is free.
        # Can't fully explain why, but empirically the ordering here influences
        # the performance of the final kernel quite significantly.
        if head_dim <= 128:
          l_i += p.sum(axis=1)
          acc, l_i, m_i, p16 = lax.optimization_barrier((acc, l_i, m_i, p16))
          end_softmax_barriers()
        else:
          end_softmax_barriers()
          l_i += p.sum(axis=1)

        # PV
        def compute_pv(acc_ref):
          plgpu.wgmma(acc_ref, p16, v_smem.at[slot])

          wait_step = kv_step + 1
          wait_slot = lax.rem(wait_step, max_concurrent_steps)
          @pl.when(wait_step < kv_seq_len // block_kv)
          def _wait():
            plgpu.barrier_wait(k_barriers.at[wait_slot])
        acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
        plgpu.barrier_arrive(v_consumed_barriers.at[slot])
        return acc, m_i, l_i
      if kv_seq_len % block_kv:
        raise ValueError(f"{kv_seq_len=} must be a multiple of {block_kv=}")
      acc, m_i, l_i = lax.fori_loop(
          0, kv_seq_len // block_kv, kv_loop, (acc, m_i, l_i)
      )
      pl.when(wg_idx == 0)(perform_schedule_barrier)
      del m_i  # Not needed anymore

      # TODO(apaszke): Invert and multiply to avoid expensive divisions.
      acc /= lax.broadcast_in_dim(l_i, (block_q, head_dim), [0])
      qo_smem[...] = acc.astype(dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(
          qo_smem, out_ref.at[batch, pl.ds(q_seq_base, block_q), q_head],
      )
      plgpu.wait_smem_to_gmem(0)
    @pl.when(wg_idx == 2)
    def _memory_wg():
      plgpu.set_max_registers(40, action="decrease")
      kv_head = lax.div(lax.axis_index("heads"), q_heads_per_kv_head)
      for i in range(max_concurrent_steps):
        s = (batch, pl.ds(i * block_kv, block_kv), kv_head)
        plgpu.copy_gmem_to_smem(k_ref.at[s], k_smem.at[i], k_barriers.at[i])
        plgpu.copy_gmem_to_smem(v_ref.at[s], v_smem.at[i], v_barriers.at[i])

      def kv_loop(kv_step, _):
        tma_step = kv_step + max_concurrent_steps
        tma_slot = lax.rem(kv_step, max_concurrent_steps)
        s = (batch, pl.ds(tma_step * block_kv, block_kv), kv_head)
        plgpu.barrier_wait(k_consumed_barriers.at[tma_slot])
        plgpu.copy_gmem_to_smem(k_ref.at[s], k_smem.at[tma_slot], k_barriers.at[tma_slot])
        plgpu.barrier_wait(v_consumed_barriers.at[tma_slot])
        plgpu.copy_gmem_to_smem(v_ref.at[s], v_smem.at[tma_slot], v_barriers.at[tma_slot])
      lax.fori_loop(0, kv_seq_len // block_kv - max_concurrent_steps, kv_loop, None)

  def run(refs):
    q_ref, k_ref, v_ref, out_ref = refs

    num_q_tiles, rem = divmod(q_seq_len, block_q * 2)
    if rem:
      raise NotImplementedError(f"{q_seq_len=} must be a multiple of {block_q * 2=}")
    mesh = plgpu.GPUMesh(
        grid=(batch_size, num_q_tiles, num_q_heads),
        num_threads=3,
        axis_names=("batch", "q_seq", "heads", "wg"),
    )

    @pl.core_map(
        mesh, compiler_params=plgpu.GPUCompilerParams(approx_math=True)
    )
    def _kernel_entry():
      compute_wgs = 2
      tiling = plgpu.TilingTransform((64, 64))
      swizzle = plgpu.SwizzleTransform(128)
      qo_scratch = plgpu.SMEM(
          (compute_wgs, block_q, head_dim), jnp.float16,
          transforms=(tiling, swizzle),
      )
      k_scratch = plgpu.SMEM(
          (max_concurrent_steps, block_kv, head_dim), jnp.float16,
          transforms=(tiling, plgpu.TransposeTransform((0, 2, 1, 3, 4)), swizzle),
      )
      v_scratch = plgpu.SMEM(
          (max_concurrent_steps, block_kv, head_dim), jnp.float16,
          transforms=(tiling, swizzle),
      )
      pl.run_scoped(
          lambda *args: kernel(q_ref, k_ref, v_ref, out_ref, args),
          (qo_scratch, k_scratch, v_scratch),
          (
              plgpu.Barrier(1, num_barriers=max_concurrent_steps),
              plgpu.Barrier(1, num_barriers=max_concurrent_steps),
              plgpu.Barrier(1, num_barriers=compute_wgs),
          ),
          (plgpu.Barrier(num_arrivals=compute_wgs, num_barriers=max_concurrent_steps),) * 2,
          plgpu.Barrier(num_arrivals=compute_wgs),
      )

  _, _, _, out = pl.run_state(run)((q, k, v, jnp.full_like(q, jnp.inf)))
  return out


@jax.jit
def attention_reference(q, k, v):
  batch_size, q_seq_len, num_q_heads, head_dim = q.shape
  num_kv_heads = k.shape[2]
  q, k, v = map(lambda x: x.astype(jnp.float32), (q, k, v))
  q_reshaped = q.reshape(
      batch_size, q_seq_len, num_kv_heads, num_q_heads // num_kv_heads, head_dim
  )
  logits = jnp.einsum("bqHhc,bkHc->bqHhk", q_reshaped, k)
  m = logits.max(axis=-1, keepdims=True)
  unnormalized = jnp.exp(logits - m)
  l = unnormalized.sum(axis=-1, keepdims=True)
  weights = unnormalized / l
  return jnp.einsum("bqHhk,bkHc->bqHhc", weights, v).reshape(*q.shape)


def main(unused_argv):
  num_q_heads = 16
  num_kv_heads = 16
  problem_it = itertools.product((1,), (4096, 32768,), (64, 128, 256,))
  for batch_size, seq_len, head_dim in problem_it:
    q_seq_len = kv_seq_len = seq_len
    print(f"==== {batch_size=:<6} {kv_seq_len=:<6} {q_seq_len=:<6}"
          f"{num_q_heads=:<4} {head_dim=:<6} ====")
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    q = jax.random.normal(k1, (batch_size, q_seq_len, num_q_heads, head_dim), jnp.float16)
    k = jax.random.normal(k2, (batch_size, kv_seq_len, num_kv_heads, head_dim), jnp.float16)
    v = jax.random.normal(k3, (batch_size, kv_seq_len, num_kv_heads, head_dim), jnp.float16)
    block_q = 64
    best = None
    for block_kv in (256, 128, 64):
      config = TuningConfig(block_q=block_q, block_kv=block_kv, max_concurrent_steps=2)
      try:
        out, runtime_ms = profiler.measure(functools.partial(attention, config=config))(q, k, v)
        if seq_len < 32768:
          out_ref = attention_reference(q, k, v)
          np.testing.assert_allclose(out, out_ref, atol=2e-3, rtol=1e-3)
      except ValueError as e:
        if "exceeds available shared memory" in e.args[0]:
          continue
        raise
      runtime_us = runtime_ms * 1e3
      matmul_flops = (
          4 * q_seq_len * kv_seq_len * head_dim * num_q_heads * batch_size
      )
      peak_flops = 1e15  # f16 TensorCore peak = 1000TFLOPS
      optimal_time = matmul_flops / peak_flops * 1e6  # us
      achieved_tc_util = optimal_time / runtime_us * 100
      print(
          f"block_q={block_q:<4}block_kv={block_kv:<4}:  {runtime_us:<7.1f}us"
          f" = {achieved_tc_util:4.1f}% TC utilization"
      )
      if best is None or runtime_us < best[0]:
        best = (runtime_us, achieved_tc_util)
      break  # Remove this for full autotuning.
    if best is not None:
      print(f"Best: {best[0]:<7.1f}us = {best[1]:4.1f}% TC utilization")


if __name__ == "__main__":
  from absl import app
  import jax
  jax.config.config_with_absl()
  app.run(main)
