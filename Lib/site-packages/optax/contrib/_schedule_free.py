# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Schedule-Free wrapper for faster training & removes the need for lr decay."""

from typing import NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax.schedules import _schedule
from optax.transforms import _adding
import optax.tree_utils as otu


class ScheduleFreeState(NamedTuple):
  """State for schedule_free."""

  b1: chex.Array
  weight_sum: chex.Array
  step_count: chex.Array
  max_lr: chex.Array
  base_optimizer_state: base.OptState
  z: base.Params


def schedule_free_eval_params(state: base.OptState, params: base.Params):
  """Params for evaluation of :func:`optax.contrib.schedule_free`."""
  # Using ScheduleFreeState as a type hint above results in pytype errors in
  # tests.
  b1 = getattr(state, 'b1')
  z = getattr(state, 'z')
  if b1 is None or z is None:
    raise ValueError(
        'schedule_free_eval_params requires a ScheduleFreeState as input.'
    )
  return jax.tree.map(lambda yi, zi: (yi - (1.0 - b1) * zi) / b1, params, z)


def schedule_free(
    base_optimizer: base.GradientTransformation,
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    weight_lr_power: float = 2.0,
    state_dtype: Optional[jax.typing.DTypeLike] = None,
) -> base.GradientTransformationExtraArgs:
  r"""Turn base_optimizer schedule_free.

  Accumulates updates returned by the base_optimizer w/o Momentum and
  replaces the momentum of an underlying optimizer with a combination of
  interpolation and averaging. In the case of gradient descent the update is

  .. math::

    \begin{align*}
      y_{t} & = (1-\beta_1)z_{t} + \beta_1 x_{t},\\
      z_{t+1} & =z_{t}-\gamma\nabla f(y_{t}),\\
      x_{t+1} & =\left(1-\frac{1}{t}\right)x_{t}+\frac{1}{t}z_{t+1},
    \end{align*}

  Here :math:`x` is the sequence that evaluations of test/val loss should occur
  at,  which differs from the primary iterates :math:`z` and the gradient
  evaluation locations :math:`y`. The updates to :math:`z` correspond to the
  underlying optimizer, in this case a simple gradient step. Note that,
  :math:`\beta_1` corresponds to `b1` in the code.

  As the name suggests, Schedule-Free learning does not require a decreasing
  learning rate schedule, yet typically out-performs, or at worst matches, SOTA
  schedules such as cosine-decay and linear decay. Only two sequences need to be
  stored at a time (the third can be computed from the other two on the fly) so
  this method has the same memory requirements as the base optimizer (parameter
  buffer + momentum).

  In practice, authors recommend tuning :math:`\beta_1`, `warmup_steps` and
  `peak_lr` for each problem seperately. Default for :math:`\beta_1` is 0.9 but
  `0.95` and `0.98` may also work well. Schedule-Free can be wrapped on top of
  any optax optimizer. At test time, the parameters should be evaluated using
  :func:`optax.contrib.schedule_free_eval_params` as presented below.

  For example, change this::

    learning_rate_fn = optax.warmup_cosine_decay_schedule(peak_value=tuned_lr)
    optimizer = optax.adam(learning_rate_fn, b1=b1)

  To::

    learning_rate_fn = optax.warmup_constant_schedule(peak_value=retuned_lr)
    optimizer = optax.adam(learning_rate_fn, b1=0.)
    optimizer = optax.contrib.schedule_free(optimizer, learning_rate_fn, b1=b1)
    ..
    params_for_eval = optax.contrib.schedule_free_eval_params(state, params)

  Especially note that is important to switch off Momentum of the base
  optimizer. As of Apr, 2024, schedule_free is tested with SGD and Adam.

  Args:
    base_optimizer: Base optimizer to compute updates from.
    learning_rate: learning_rate schedule w/o decay but with warmup.
    b1: beta_1 parameter in the y update.
    weight_lr_power: we downweight the weight of averaging using this. This is
      especially helpful in early iterations during warmup.
    state_dtype: dtype for z sequence in the schedule free method.

  Returns:
    A :class:`optax.GradientTransformationExtraArgs`.

  References:
    Defazio et al, `The Road Less Scheduled
    <https://arxiv.org/abs/2405.15682>`_, 2024

    Defazio et al, `Schedule-Free Learning - A New Way to Train
    <https://github.com/facebookresearch/schedule_free/tree/main>`_, 2024

  .. warning::
    The current implementation requires the parameter ``b1`` to be strictly
    positive.
  """
  base_optimizer = base.with_extra_args_support(base_optimizer)

  def init_fn(params: base.Params) -> ScheduleFreeState:
    # Define state parameters with the lowest dtype of the parameters to avoid
    # dtype promotion of parameters resulting in a dtype mismatch between
    # parameters and updates.
    params_dtype = otu.tree_dtype(params, 'lowest')
    if state_dtype is not None:
      z = otu.tree_cast(params, dtype=state_dtype)
    else:
      z = params
    # It's imporant to copy the params here so that z is a distinct array and
    # we can donate both z and the params to JITted functions.
    z = jax.tree_util.tree_map(lambda t: t.copy(), z)
    return ScheduleFreeState(
        b1=jnp.asarray(b1, dtype=params_dtype),
        weight_sum=jnp.zeros([], dtype=params_dtype),
        step_count=jnp.ones([], dtype=jnp.int32),
        max_lr=jnp.zeros([], dtype=params_dtype),
        base_optimizer_state=base_optimizer.init(params),
        z=z,
    )

  def update_fn(
      grads: base.Updates,
      state: ScheduleFreeState,
      params: Optional[base.Params] = None,
      **extra_args,
  ):
    lr = learning_rate
    if callable(learning_rate):
      lr = jnp.asarray(
          learning_rate(state.step_count), dtype=state.max_lr.dtype
      )
    max_lr = jnp.maximum(state.max_lr, lr)

    next_step_count = numerics.safe_increment(state.step_count)

    weight = max_lr**weight_lr_power
    next_total_weight = state.weight_sum + weight
    # We add this to avoid NaNs in the case of a small learning rate.
    ck = jnp.where(
        jnp.logical_or(jnp.isnan(weight), jnp.isnan(next_total_weight)),
        jnp.full(weight.shape, jnp.nan),
        jnp.nan_to_num(weight / next_total_weight, nan=0.0, posinf=jnp.inf),
    )

    base_updates, next_base_optimizer_state = base_optimizer.update(
        grads,
        state.base_optimizer_state,
        params,
        **extra_args,
    )
    z = jax.tree.map(
        lambda pi, ui: jnp.asarray(pi + ui).astype(jnp.asarray(pi).dtype),
        state.z,
        base_updates,
    )

    # Important: recompute x to both save memory and maintain accurate x seq
    # especially if y is modified by another transform wrapped on top.
    prev_x = jax.tree.map(
        lambda yi, zi: (yi - (1.0 - b1) * zi) / b1, params, state.z
    )

    x = jax.tree.map(
        lambda xi, zi: (1.0 - ck) * xi + ck * zi,
        prev_x,
        z,
    )
    new_params = jax.tree.map(
        lambda xi, zi: b1 * xi + (1.0 - b1) * zi,
        x,
        z,
    )
    updates = jax.tree.map(lambda npi, pi: npi - pi, new_params, params)

    next_state = ScheduleFreeState(
        b1=state.b1,
        weight_sum=next_total_weight,
        step_count=next_step_count,
        max_lr=max_lr,
        base_optimizer_state=next_base_optimizer_state,
        z=z,
    )

    return updates, next_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


def schedule_free_sgd(
    learning_rate: float = 1.0,
    warmup_steps: Optional[int] = None,
    b1: float = 0.9,
    weight_decay: Optional[float] = None,
    weight_lr_power: float = 2.0,
    state_dtype: Optional[jax.typing.DTypeLike] = None,
) -> base.GradientTransformationExtraArgs:
  """Schedule-Free wrapper for SGD.

  Shortcut example for using schedule_free with SGD, which is a common use case.
  Note that this is just an example, and other use cases are possible, e.g.
  using a weight decay mask. Note also that the EMA parameter of the
  schedule free method (b1) must be strictly positive.

  Args:
    learning_rate: SGD learning rate.
    warmup_steps: positive integer, the length of the linear warmup.
    b1: beta_1 parameter in the y update.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent with
      other frameworks such as PyTorch, but different from (Loshchilov et al,
      2019) where the weight decay is only multiplied with the "schedule
      multiplier", but not the base learning rate.
    weight_lr_power: we downweight the weight of averaging using this. This is
      especially helpful in early iterations during warmup.
    state_dtype: dtype for z sequence in the schedule free method.

  Returns:
    A :class:`optax.GradientTransformationExtraArgs`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.contrib.schedule_free_sgd()
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  eval_params = optax.contrib.schedule_free_eval_params(
    ...      opt_state, params)
    ...  print('Objective function: {:.2E}'.format(f(eval_params)))
    Objective function: 1.40E+01
    Objective function: 1.75E-14
    Objective function: 9.96E-01
    Objective function: 8.06E-01
    Objective function: 2.41E-01
  """
  if warmup_steps is not None:
    learning_rate = _schedule.warmup_constant_schedule(
        init_value=0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
    )
  optimizer = alias.sgd(learning_rate)
  if weight_decay is not None:
    optimizer = combine.chain(
        _adding.add_decayed_weights(weight_decay), optimizer
    )
  return schedule_free(
      optimizer,
      learning_rate=learning_rate,
      b1=b1,
      weight_lr_power=weight_lr_power,
      state_dtype=state_dtype,
  )


def schedule_free_adamw(
    learning_rate: float = 0.0025,
    warmup_steps: Optional[int] = None,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    weight_lr_power: float = 2.0,
    state_dtype: Optional[jax.typing.DTypeLike] = None,
) -> base.GradientTransformationExtraArgs:
  """Schedule-Free wrapper for AdamW.

  Shortcut example for using schedule_free with AdamW, which is a common use
  case. Note that this is just an example, and other usecases are possible, e.g.
  using a weight decay mask, nesterov, etc. Note also that the EMA parameter of
  the schedule free method (b1) must be strictly positive.

  Args:
    learning_rate: AdamW learning rate.
    warmup_steps: positive integer, the length of the linear warmup.
    b1: beta_1 parameter in the y update.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root (as
      in the Adam paper) to avoid dividing by zero when rescaling.
    weight_decay: Strength of the weight decay regularization.
    weight_lr_power: we downweight the weight of averaging using this. This is
      especially helpful in early iterations during warmup.
    state_dtype: dtype for z sequence in the schedule free method.

  Returns:
    A :class:`optax.GradientTransformationExtraArgs`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.contrib.schedule_free_adamw(1.0)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  eval_params = optax.contrib.schedule_free_eval_params(
    ...      opt_state, params)
    ...  print('Objective function: {:.2E}'.format(f(eval_params)))
    Objective function: 5.00E+00
    Objective function: 3.05E+00
    Objective function: 1.73E+00
    Objective function: 8.94E-01
    Objective function: 4.13E-01

  .. note::
    Note that :func:`optax.scale_by_adam` with ``b1=0`` stores in its state an
    unused first moment always equal to zero. To avoid this waste of memory,
    we replace
    :func:`optax.scale_by_adam` with ``b1=0`` by the equivalent
    :func:`optax.scale_by_rms` with ``eps_in_sqrt=False, bias_correction=True``.

  """
  if warmup_steps is not None:
    learning_rate = _schedule.warmup_constant_schedule(
        init_value=0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
    )
  # The following is the same as adamw, but with the momentum term removed.
  optimizer = combine.chain(
      transform.scale_by_rms(
          decay=b2, eps=eps, eps_in_sqrt=False, bias_correction=True
      ),
      _adding.add_decayed_weights(weight_decay),
      transform.scale_by_learning_rate(learning_rate),
  )
  return schedule_free(
      optimizer,
      learning_rate=learning_rate,
      b1=b1,
      weight_lr_power=weight_lr_power,
      state_dtype=state_dtype,
  )
