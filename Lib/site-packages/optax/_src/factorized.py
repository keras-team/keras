# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Factorized optimizers."""

from collections.abc import Callable
import dataclasses
from typing import NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import base
from optax._src import numerics


def _decay_rate_pow(i: int, exponent: float = 0.8) -> chex.Array:
  """Second-order moment decay schedule."""
  t = jnp.array(i + 1, jnp.float32)
  return 1.0 - t ** (-exponent)


def _factored_dims(
    shape: base.Shape, factored: bool, min_dim_size_to_factor: int
) -> Optional[tuple[int, int]]:
  """Whether to use a factored second moment estimator.

  This function returns a tuple with the two largest axes to reduce over.
  If no two dimensions have size >= min_dim_size_to_factor, return None.

  Args:
    shape: an input shape
    factored: whether to use factored second-moment estimator for 2d vars.
    min_dim_size_to_factor: only factor accumulator if two array dimensions have
      at least this size.

  Returns:
    None or a tuple of ints
  """
  if not factored or len(shape) < 2:
    return None
  sorted_dims = np.argsort(shape)
  if shape[sorted_dims[-2]] < min_dim_size_to_factor:
    return None
  return int(sorted_dims[-2]), int(sorted_dims[-1])


@dataclasses.dataclass
class _UpdateResult:
  """Opaque container that is not traversed by jax.tree.map."""

  update: chex.Array  # the update to apply to params
  v_row: chex.Array  # used for factored params.
  v_col: chex.Array  # used for factored params.
  v: chex.Array  # used for params where factoring is skipped.


class FactoredState(NamedTuple):
  """Overall state of the gradient transformation."""

  count: chex.Array  # number of update steps.
  v_row: chex.ArrayTree  # Tree of factored params.
  v_col: chex.ArrayTree  # Tree of factored params.
  v: chex.ArrayTree  # Tree for params where factoring is skipped.


def scale_by_factored_rms(
    factored: bool = True,
    decay_rate: float = 0.8,
    step_offset: int = 0,
    min_dim_size_to_factor: int = 128,
    epsilon: float = 1e-30,
    decay_rate_fn: Callable[[int, float], chex.Array] = _decay_rate_pow,
):
  """Scaling by a factored estimate of the gradient rms (as in Adafactor).

  This is a so-called "1+epsilon" scaling algorithms, that is extremely memory
  efficient compared to RMSProp/Adam, and has had wide success when applied to
  large-scale training of attention-based models.

  Args:
    factored: boolean: whether to use factored second-moment estimates..
    decay_rate: float: controls second-moment exponential decay schedule.
    step_offset: for finetuning, one may set this to the starting step-number of
      the fine tuning phase.
    min_dim_size_to_factor: only factor accumulator if two array dimensions are
      at least this size.
    epsilon: Regularization constant for squared gradient.
    decay_rate_fn: A function that accepts the current step, the decay rate
      parameter and controls the schedule for the second momentum. Defaults to
      the original adafactor's power decay schedule. One potential shortcoming
      of the original schedule is the fact that second momentum converges to 1,
      which effectively freezes the second momentum. To prevent this the user
      can opt for a custom schedule that sets an upper bound for the second
      momentum, like in Zhai et al., 2021.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  References:
    Shazeer et al, `Adafactor: Adaptive Learning Rates with Sublinear Memory
    Cost <https://arxiv.org/abs/1804.04235>`_, 2018

    Zhai et al, `Scaling Vision Transformers
    <https://arxiv.org/abs/2106.04560>`_, 2021
  """

  def _to_state(count: chex.Array, result_tree):
    """Maps from a tree of (factored) values to separate trees of values."""
    return FactoredState(
        count=count,
        v_row=jax.tree.map(lambda o: o.v_row, result_tree),
        v_col=jax.tree.map(lambda o: o.v_col, result_tree),
        v=jax.tree.map(lambda o: o.v, result_tree),
    )

  def init_fn(params):
    """Initialise the optimizer's state."""

    def _init(param):
      shape, dtype = param.shape, param.dtype
      factored_dims = _factored_dims(shape, factored, min_dim_size_to_factor)
      if factored_dims is not None:
        d1, d0 = factored_dims
        vr_shape = np.delete(shape, d0)
        vc_shape = np.delete(shape, d1)
        return _UpdateResult(
            update=jnp.zeros((1,), dtype=dtype),
            v_row=jnp.zeros(vr_shape, dtype=dtype),
            v_col=jnp.zeros(vc_shape, dtype=dtype),
            v=jnp.zeros((1,), dtype=dtype),
        )
      else:
        return _UpdateResult(
            update=jnp.zeros((1,), dtype=dtype),
            v_row=jnp.zeros((1,), dtype=dtype),
            v_col=jnp.zeros((1,), dtype=dtype),
            v=jnp.zeros(param.shape, dtype=dtype),
        )

    return _to_state(jnp.zeros([], jnp.int32), jax.tree.map(_init, params))

  def update_fn(grads, state, params):
    """Apply gradient transformation."""
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)

    def _update(grad, v_row, v_col, v, param, step):
      shape, dtype = param.shape, param.dtype
      decay_rate_t = decay_rate_fn(step - step_offset, decay_rate)

      # Scaled by factorized second moment statistics.
      new_v_row = jnp.zeros((1,), dtype=dtype)
      new_v_col = jnp.zeros((1,), dtype=dtype)
      new_v = jnp.zeros((1,), dtype=dtype)

      factored_dims = _factored_dims(shape, factored, min_dim_size_to_factor)
      if factored_dims is not None:
        d1, d0 = factored_dims
        grad_sqr = numerics.abs_sq(grad) + epsilon
        new_v_row = decay_rate_t * v_row + (1.0 - decay_rate_t) * jnp.mean(
            grad_sqr, axis=d0
        )
        new_v_col = decay_rate_t * v_col + (1.0 - decay_rate_t) * jnp.mean(
            grad_sqr, axis=d1
        )
        new_v_row = new_v_row.astype(dtype)
        new_v_col = new_v_col.astype(dtype)
        reduced_d1 = d1 - 1 if d1 > d0 else d1
        row_col_mean = jnp.mean(new_v_row, axis=reduced_d1, keepdims=True)
        row_factor = (new_v_row / row_col_mean) ** -0.5
        col_factor = (new_v_col) ** -0.5
        update = (
            grad
            * jnp.expand_dims(row_factor, axis=d0)
            * jnp.expand_dims(col_factor, axis=d1)
        )
      else:
        grad_sqr = numerics.abs_sq(grad) + epsilon
        new_v = decay_rate_t * v + (1.0 - decay_rate_t) * grad_sqr
        new_v = new_v.astype(dtype)
        update = grad * (new_v) ** -0.5

      return _UpdateResult(update, new_v_row, new_v_col, new_v)

    # Transform grad and compute new per-parameter stats.
    output = jax.tree.map(
        lambda *args: _update(*args, state.count),
        grads,
        state.v_row,
        state.v_col,
        state.v,
        params,
    )

    # Unpack updates / stats and return.
    updates = jax.tree.map(lambda o: o.update, output)
    return updates, _to_state(numerics.safe_increment(state.count), output)

  return base.GradientTransformation(init_fn, update_fn)
