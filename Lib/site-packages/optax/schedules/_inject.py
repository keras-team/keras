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
"""Utilities to inject dynamically changing hyper-parameters."""

from collections.abc import Callable
import functools
import inspect
from typing import Iterable, NamedTuple, Optional, Union
import warnings

import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import base
from optax._src import numerics


def _convert_floats(x, dtype):
  """Convert float-like inputs to dtype, rest pass through."""
  if jax.dtypes.scalar_type_of(x) == float:
    return jnp.asarray(x, dtype=dtype)
  return x


class InjectHyperparamsState(NamedTuple):
  """Deprecated class kept for backwards compatibility.

  .. deprecated:: 0.1.9
    Use :class:`InjectStatefulHyperparamsState` instead.
  """

  count: jnp.ndarray  # shape=(), dtype=jnp.int32
  hyperparams: dict[str, chex.Numeric]
  inner_state: base.OptState


class InjectStatefulHyperparamsState(NamedTuple):
  """Maintains inner transform state, hyperparameters, and step count."""

  count: jnp.ndarray  # shape=(), dtype=jnp.int32
  hyperparams: dict[str, chex.Numeric]
  hyperparams_states: dict[str, base.ScheduleState]
  inner_state: base.OptState


def inject_hyperparams(
    inner_factory: Callable[..., base.GradientTransformation],
    static_args: Union[str, Iterable[str]] = (),
    hyperparam_dtype: Optional[jnp.dtype] = None,
) -> Callable[..., base.GradientTransformationExtraArgs]:
  """Wrapper to injects stateful hyperparameters into GradientTransformations.

  This wrapper allows you to pass schedules (i.e. a function that returns a
  numeric value given a step count) instead of constants for
  hyperparameters. You may only schedule numeric hyperparameters (i.e. boolean
  flags cannot be scheduled).

  This function supports both passing simple schedules
  that are function exclusively of the step count and also passing stateful
  schedules that rely on a complex internal state. The state updating can rely
  on additional information fed to gradient transformations via extra_args.

  For example, to use :func:`optax.scale_by_adam` with a piecewise linear
  schedule for beta_1 and constant for beta_2::

    >>> import optax
    >>> import jax.numpy as jnp
    >>> # create a learning rate that increases linearly from 0.1 to 1.0
    ... # over 100 iterations
    >>> linear_schedule = optax.piecewise_interpolate_schedule(
    ...    'linear', init_value=0.1, boundaries_and_scales={100: 1.})
    >>> scheduled_adam = optax.inject_hyperparams(optax.scale_by_adam)(
    ...     b1=linear_schedule, b2=0.99)

  You may manually change numeric hyperparameters that were not scheduled
  through the ``hyperparams`` dict in the ``InjectHyperparamState``::

    >>> params, grads = jnp.array(0.), jnp.array(0.)
    >>> state = scheduled_adam.init(params)
    >>> updates, state = scheduled_adam.update(grads, state)
    >>> state.hyperparams['b2'] = 0.95
    >>> updates, state = scheduled_adam.update(updates, state)  # uses b2 = 0.95

  Manually overriding scheduled hyperparameters will have no effect (e.g.
  in the code sample above, you cannot manually adjust ``b1``).

  Args:
    inner_factory: a function that returns the inner
      ``optax.GradientTransformation`` with dynamic hyperparameters.
    static_args: a string or iterable of strings specifying which callable
      parameters are not schedules. inject_hyperparams treats all callables as
      schedules by default, so if a hyperparameter is a non-schedule callable,
      you must specify that using this argument.
    hyperparam_dtype: Optional datatype override. If specified, all float
      hyperparameters will be cast to this type.

  Returns:
    A callable that returns a ``optax.GradientTransformationExtraArgs``. This
    callable accepts the same arguments as ``inner_factory``, except you may
    provide schedules in place of the constant arguments.

  .. versionchanged:: 0.1.9
    New parameter ``hyperparam_dtype``, the returned callable outputs a
    ``GradientTransformationExtraArgs`` instead of a ``GradientTransformation``.
  """
  static_args = (
      {static_args} if isinstance(static_args, str) else set(static_args)
  )
  inner_signature = inspect.signature(inner_factory)

  if not static_args.issubset(inner_signature.parameters):
    raise ValueError(
        "`static_args` must specify a subset of `inner_factory`'s parameters. "
        f'Given `static_args`: {static_args}. `inner_factory` parameters: '
        f'{set(inner_signature.parameters.keys())}'
    )

  @functools.wraps(inner_factory)
  def wrapped_transform(
      *args, **kwargs
  ) -> base.GradientTransformationExtraArgs:
    bound_arguments = inner_signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()

    sched_hps, numeric_hps, other_hps = {}, {}, {}
    for name, value in bound_arguments.arguments.items():
      if name in static_args or isinstance(value, bool):
        other_hps[name] = value
      elif isinstance(value, base.StatefulSchedule):
        sched_hps[name] = value
      elif callable(value):
        sched_hps[name] = WrappedSchedule(value)
      elif isinstance(value, (int, float, jax.Array, np.ndarray)):
        numeric_hps[name] = value
      else:
        other_hps[name] = value

    def init_fn(params):
      count = jnp.zeros([], jnp.int32)
      if hyperparam_dtype is None:
        dtype = getattr(
            next(iter(jax.tree.leaves(params)), None), 'dtype', None
        )
      else:
        dtype = hyperparam_dtype
      hparams = {
          k: jnp.asarray(_convert_floats(v, dtype))
          for k, v in numeric_hps.items()
      }
      hparams_states = {k: f.init() for k, f in sched_hps.items()}
      hparams.update({
          k: _convert_floats(f(hparams_states[k]), dtype)
          for k, f in sched_hps.items()
      })
      return InjectStatefulHyperparamsState(
          count=count,
          hyperparams=hparams,
          hyperparams_states=hparams_states,
          inner_state=inner_factory(**other_hps, **hparams).init(params),
      )

    def update_fn(updates, state, params=None, **extra_args):
      if hyperparam_dtype is None:
        dtype = getattr(
            next(iter(jax.tree.leaves(updates)), None), 'dtype', None
        )
      else:
        dtype = hyperparam_dtype
      hparams = {
          k: _convert_floats(v, dtype) for k, v in state.hyperparams.items()
      }
      hparams.update({
          k: _convert_floats(
              f(state.hyperparams_states[k], **extra_args), dtype
          )
          for k, f in sched_hps.items()
      })
      hyperparams_states = {
          k: f.update(state.hyperparams_states[k], **extra_args)
          for k, f in sched_hps.items()
      }

      updates, inner_state = base.with_extra_args_support(
          inner_factory(**other_hps, **hparams)
      ).update(updates, state.inner_state, params, **extra_args)

      return updates, InjectStatefulHyperparamsState(
          count=numerics.safe_increment(state.count),
          hyperparams=hparams,
          hyperparams_states=hyperparams_states,
          inner_state=inner_state,
      )

    return base.GradientTransformationExtraArgs(init_fn, update_fn)

  return wrapped_transform


def inject_stateful_hyperparams(
    inner_factory: Callable[..., base.GradientTransformation],
    static_args: Union[str, Iterable[str]] = (),
    hyperparam_dtype: Optional[jnp.dtype] = None,
) -> Callable[..., base.GradientTransformationExtraArgs]:
  """Wrapper to injects stateful hyperparameters into GradientTransformations.

  Similar to `inject_hyperparams`, but supports both passing simple schedules
  that are function exclusively of the step count and also passing stateful
  schedules that rely on a complex internal state. The state updating can rely
  on additional information fed to gradient transformations via extra_args.

  Args:
    inner_factory: a function that returns the inner
      ``optax.GradientTransformation`` with dynamic hyperparameters.
    static_args: a string or iterable of strings specifying which callable
      parameters are not schedules. inject_hyperparams treats all callables as
      schedules by default, so if a hyperparameter is a non-schedule callable,
      you must specify that using this argument.
    hyperparam_dtype: Optional datatype override. If specified, all float
      hyperparameters will be cast to this type.

  Returns:
    A callable that returns a ``optax.GradientTransformation``. This callable
    accepts the same arguments as ``inner_factory``, except you may provide
    schedules in place of the constant arguments.

  .. deprecated:: 0.1.9
    Use :func:`inject_hyperparams` instead.
  """
  # raise deprecationwarning
  warnings.warn(
      'inject_stateful_hyperparams is deprecated, use inject_hyperparams'
      ' instead',
      DeprecationWarning,
  )
  return inject_hyperparams(inner_factory, static_args, hyperparam_dtype)


class WrappedScheduleState(NamedTuple):
  """The state for a wrapped schedule."""

  count: chex.Numeric


class WrappedSchedule:
  """A stateful schedule that wraps a stateless schedule."""

  def __init__(self, schedule_fn: base.Schedule):
    self.schedule_fn = schedule_fn

  def init(
      self,
  ) -> WrappedScheduleState:
    return WrappedScheduleState(count=jnp.zeros([], jnp.int32))

  def update(
      self,
      state: WrappedScheduleState,
      **extra_args,
  ) -> WrappedScheduleState:
    del extra_args
    new_count = numerics.safe_increment(state.count)
    return WrappedScheduleState(count=new_count)

  def __call__(
      self,
      state: WrappedScheduleState,
      **extra_args,
  ) -> chex.Numeric:
    return self.schedule_fn(state.count)
