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
"""Gradient transformations for accumulating gradients across updates."""

from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Protocol, Union

import chex
import jax
from jax import lax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from optax._src import utils


class TraceState(NamedTuple):
  """Holds an aggregation of past updates."""

  trace: base.Params


def trace(
    decay: float,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """Compute a trace of past updates.

  Args:
    decay: Decay rate for the trace of past updates.
    nesterov: Whether to use Nesterov momentum.
    accumulator_dtype: Optional `dtype` to be used for the accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A :class:`optax.GradientTransformation` object.

  .. note::
    :func:`optax.trace` and :func:`optax.ema` have very similar but distinct
    updates; ``trace = decay * trace + t``, while
    ``ema = decay * ema + (1-decay) * t``.
    Both are frequently found in the optimization literature.
  """

  accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)

  def init_fn(params):
    return TraceState(
        trace=otu.tree_zeros_like(params, dtype=accumulator_dtype)
    )

  def update_fn(updates, state, params=None):
    del params
    f = lambda g, t: g + decay * t
    new_trace = jax.tree.map(f, updates, state.trace)
    updates = jax.tree.map(f, updates, new_trace) if nesterov else new_trace
    new_trace = otu.tree_cast(new_trace, accumulator_dtype)
    return updates, TraceState(trace=new_trace)

  return base.GradientTransformation(init_fn, update_fn)


class EmaState(NamedTuple):
  """Holds an exponential moving average of past updates."""

  count: chex.Array  # shape=(), dtype=jnp.int32.
  ema: base.Params


def ema(
    decay: float, debias: bool = True, accumulator_dtype: Optional[Any] = None
) -> base.GradientTransformation:
  """Compute an exponential moving average of past updates.

  Args:
    decay: Decay rate for the exponential moving average.
    debias: Whether to debias the transformed gradient.
    accumulator_dtype: Optional `dtype` to used for the accumulator; if `None`
      then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A :class:`optax.GradientTransformation` object.

  .. note::
    :func:`optax.trace` and :func:`optax.ema` have very similar but distinct
    updates; ``trace = decay * trace + t``, while
    ``ema = decay * ema + (1-decay) * t``.
    Both are frequently found in the optimization literature.
  """

  accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)

  def init_fn(params):
    return EmaState(
        count=jnp.zeros([], jnp.int32),
        ema=otu.tree_zeros_like(params, dtype=accumulator_dtype),
    )

  def update_fn(updates, state, params=None):
    del params
    updates = new_ema = otu.tree_update_moment(
        updates, state.ema, decay, order=1
    )
    count_inc = numerics.safe_increment(state.count)
    if debias:
      updates = otu.tree_bias_correction(new_ema, decay, count_inc)
    state_ema = otu.tree_cast(new_ema, accumulator_dtype)
    return updates, EmaState(count=count_inc, ema=state_ema)

  return base.GradientTransformation(init_fn, update_fn)


class ShouldSkipUpdateFunction(Protocol):

  def __call__(
      self,
      updates: base.Updates,
      gradient_step: chex.Array,
      params: Optional[base.Params],
  ) -> tuple[chex.Array, chex.ArrayTree]:
    """Returns true to indicate that updates should be skipped in a multi-step.

    Args:
      updates: The updates that the gradient transformation has proposed.
      gradient_step: The current gradient step (see
        `MultiStepsState.gradient_step`). This can be used for example to reject
        large gradients with an annealed maximum allowed gradient norm.
      params: If known, the current params of the function being transformed.

    Returns:
      A tuple:
      * First element is an array with a single bool indicating whether or not
        the updates should be applied.
      * Second element is an arbitrary py-tree that will be stored in
        `MultiStepsState.skip_state`. Debugging info can be put here.
    """


def skip_not_finite(
    updates: base.Updates,
    gradient_step: chex.Array,
    params: Optional[base.Params],
) -> tuple[chex.Array, chex.ArrayTree]:
  """Returns True iff any of the `updates` contains an inf or a NaN.

  Args:
    updates: see `ShouldSkipUpdateFunction`.
    gradient_step: see `ShouldSkipUpdateFunction`.
    params: see `ShouldSkipUpdateFunction`.

  Returns:
    A tuple:
    * First element is a scalar array of type bool.
    * Second element is a dictionary with keys:
      - `should_skip`: True iff `updates` contains an inf or a NaN.
      - `num_not_finite`: total number of inf and NaN found in `updates`.
  """
  del gradient_step, params
  all_is_finite = [
      jnp.sum(jnp.logical_not(jnp.isfinite(p)))
      for p in jax.tree.leaves(updates)
  ]
  num_not_finite = jnp.sum(jnp.array(all_is_finite))
  should_skip = num_not_finite > 0
  return should_skip, dict(
      should_skip=should_skip, num_not_finite=num_not_finite
  )


def skip_large_updates(
    updates: base.Updates,
    gradient_step: chex.Array,
    params: Optional[base.Params],
    max_squared_norm: float,
) -> tuple[chex.Array, chex.ArrayTree]:
  """Returns True if the global norm square of `updates` is small enough.

  Args:
    updates: see `ShouldSkipUpdateFunction`.
    gradient_step: see `ShouldSkipUpdateFunction`.
    params: see `ShouldSkipUpdateFunction`.
    max_squared_norm: max square norm that can be accepted in updates.

  Returns:
    A tuple:
    * First element is a scalar array of type bool.
    * Second element is a dictionary with keys:
      - `should_skip`: iff ||updates||^2 is greater than `max_squared_norm`.
      - `norm_squared`: overall norm square of the `updates`.
  """
  del gradient_step, params
  norm_sq = jnp.sum(
      jnp.array([jnp.sum(p**2) for p in jax.tree.leaves(updates)])
  )
  # This will also return True if `norm_sq` is NaN.
  should_skip = jnp.logical_not(norm_sq < max_squared_norm)
  return should_skip, dict(should_skip=should_skip, norm_squared=norm_sq)


class MultiStepsState(NamedTuple):
  """State of the `GradientTransformation` returned by `MultiSteps`.

  Attributes:
    mini_step: current mini-step counter. At an update, this either increases by
      1 or is reset to 0.
    gradient_step: gradient step counter. This only increases after enough
      mini-steps have been accumulated.
    inner_opt_state: the state of the wrapped optimizer.
    acc_grads: accumulated gradients over multiple mini-steps.
    skip_state: an arbitrarily py tree. This is only relevant when passing a
      `should_skip_update_fn` to `MultiSteps`.
  """

  mini_step: chex.Array
  gradient_step: chex.Array
  inner_opt_state: Any
  acc_grads: Any
  skip_state: chex.ArrayTree = ()


class MultiSteps:
  """An optimizer wrapper to accumulate gradients over multiple steps.

  This wrapper collects together the updates passed to its ``update`` function
  over consecutive steps until a given number of scheduled steps is reached.
  In each of these intermediate steps, the returned value from the optimizer is
  a tree of zeros of the same shape of the updates passed as input.

  Once the scheduled number of intermediate 'mini-steps' has been reached, the
  gradients accumulated to the current time will be passed to the wrapped
  optimizer's update function, (with the inner optimizer's state being updated
  appropriately) and then returned to the caller. The wrapper's accumulated
  gradients are then set back to zero and the process starts again.

  The number of mini-steps per gradient update is controlled by a function, and
  can vary over training, this also allows varying batch size over training.
  """

  def __init__(
      self,
      opt: base.GradientTransformation,
      every_k_schedule: Union[int, Callable[[chex.Array], chex.Array]],
      use_grad_mean: bool = True,
      should_skip_update_fn: Optional[ShouldSkipUpdateFunction] = None,
  ):
    """Initializer.

    Args:
      opt: the wrapped optimizer.
      every_k_schedule: an int or a function.  * As a function, it returns how
        many mini-steps should be accumulated in a single gradient step. Its
        only argument is the current gradient step count. By varying the
        returned value, users can vary the overall training batch size. * If an
        ``int``, this is the constant number of mini-steps per gradient update.
      use_grad_mean: if ``True`` (the default), gradients accumulated over
        multiple mini-steps are averaged. Otherwise, they are summed.
      should_skip_update_fn: if provided, this function is used to decide when
        to accept or reject the updates from a mini-step. When a mini-step is
        rejected, the inner state of `MultiSteps` is not updated. In other
        words, it is as if this mini-step never happened. For example:  * to
        ignore updates containing inf or NaN, do
        ``should_skip_update_fn=skip_not_finite``; * to ignore updates with a
        norm square larger then 42, do:
        ``should_skip_update_fn=functools.partial(skip_large_updates,
        max_norm_sq=42.)``  Note that the optimizer's state
        :class:`optax.MultiStepsState` contains a keyword argument
        ``skip_state`` in which debugging and monitoring information returned by
        ``should_skip_update_fn`` is written.
    """
    self._opt = base.with_extra_args_support(opt)

    if isinstance(every_k_schedule, int):
      self._every_k_schedule = lambda step: every_k_schedule
    else:
      self._every_k_schedule = every_k_schedule
    self._use_grad_mean = use_grad_mean

    if self._use_grad_mean:
      # Use Welford algorithm for numerically stable aggregation of mean.
      self._acc_update = lambda grad, acc, *, n_acc: acc + (grad - acc) / (
          n_acc + 1
      )
    else:
      self._acc_update = lambda grad, acc, *, n_acc: grad + acc

    if should_skip_update_fn is None:

      def should_skip_update_fn(*unused_args, **unused_kwargs):
        return jnp.array(False, dtype=jnp.bool_), ()

    self._should_skip_update_fn = should_skip_update_fn

  @property
  def inner_opt(self):
    return self._opt

  def init(self, params: Any) -> MultiStepsState:
    """Builds and returns initial `MultiStepsState`."""
    updates = otu.tree_zeros_like(params)
    gradient_step = jnp.zeros([], dtype=jnp.int32)
    _, skip_state = self._should_skip_update_fn(updates, gradient_step, params)
    init_state = MultiStepsState(
        mini_step=jnp.zeros([], dtype=jnp.int32),
        gradient_step=gradient_step,
        inner_opt_state=self._opt.init(params),
        acc_grads=updates,
        skip_state=skip_state,
    )
    return init_state

  def update(
      self,
      updates: base.Updates,
      state: MultiStepsState,
      params: Optional[base.Params] = None,
      **extra_args: Any,
  ) -> tuple[base.Updates, MultiStepsState]:
    """Accumulates gradients and proposes non-zero updates every `k_steps`."""
    k_steps = self._every_k_schedule(state.gradient_step)
    should_skip_update, skip_state = self._should_skip_update_fn(
        updates, state.gradient_step, params
    )
    if (should_skip_update.dtype, should_skip_update.shape) != (jnp.bool_, ()):
      raise ValueError(
          'The `should_skip_update_fn` function should return a boolean scalar '
          f'array, but it returned an array of dtype {should_skip_update.dtype}'
          f' and shape {should_skip_update.shape}'
      )

    # Note: we do not enclose variables to allow JAX to re-use memory buffers.
    def _do_update(updates, state, params):
      acc_grads = jax.tree.map(
          lambda upd, acc: self._acc_update(upd, acc, n_acc=state.mini_step),
          updates,
          state.acc_grads,
      )

      final_updates, new_inner_state = self._opt.update(
          acc_grads, state.inner_opt_state, params=params, **extra_args
      )

      emit = state.mini_step == (k_steps - 1)
      new_state = MultiStepsState(
          mini_step=numerics.safe_increment(state.mini_step) % k_steps,
          gradient_step=emit * numerics.safe_increment(state.gradient_step)
          + (1 - emit) * state.gradient_step,
          inner_opt_state=jax.tree.map(
              lambda st, nst: jnp.where(emit, nst, st),
              state.inner_opt_state,
              new_inner_state,
          ),
          acc_grads=jax.tree.map(
              lambda ga, upd: (1 - emit) * ga.astype(upd.dtype),
              acc_grads,
              final_updates,
          ),
          skip_state=skip_state,
      )

      final_updates = jax.tree.map(lambda ga: emit * ga, final_updates)
      return final_updates, new_state

    def _skip_update(updates, state, params):
      # Create new skip state with correct dtype
      zero_updates, new_inner_state = jax.eval_shape(
          self._opt.update,
          updates,
          state.inner_opt_state,
          params=params,
          **extra_args,
      )
      zero_updates = otu.tree_zeros_like(zero_updates)

      multi_state_when_skip = MultiStepsState(
          mini_step=state.mini_step,
          gradient_step=state.gradient_step,
          inner_opt_state=jax.tree.map(
              lambda x, y: (
                  x.astype(y.dtype) if isinstance(x, jax.Array) else x
              ),
              state.inner_opt_state,
              new_inner_state,
          ),
          acc_grads=jax.tree.map(
              lambda acc, upd: acc.astype(upd.dtype),
              state.acc_grads,
              zero_updates,
          ),
          skip_state=skip_state,
      )

      return zero_updates, multi_state_when_skip

    new_updates, new_state = lax.cond(
        should_skip_update, _skip_update, _do_update, *(updates, state, params)
    )
    return new_updates, new_state

  def has_updated(
      self, state: Union[MultiStepsState, chex.ArrayTree]
  ) -> chex.Array:
    # Use `getattr` to bypass pytype checks.
    return jnp.logical_and(
        getattr(state, 'mini_step') == 0, getattr(state, 'gradient_step') > 0
    )

  def gradient_transformation(self) -> base.GradientTransformation:
    return base.GradientTransformation(init=self.init, update=self.update)
