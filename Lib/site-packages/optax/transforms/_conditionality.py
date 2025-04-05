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
"""Wrappers that allow transformations to be applied conditionally."""

from typing import Any, NamedTuple, Protocol

import chex
import jax
from jax import lax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics


class ConditionFn(Protocol):
  """Condition function for conditional transformations."""

  def __call__(
      self,
      step: chex.Array,
      **extra_args: Any,
  ) -> chex.Array:
    """Update function with optional extra arguments.

    Args:
      step: a counter (array of shape [] and dtype ``int32``)
      **extra_args: Additional keyword arguments passed to this condition fn.

    Returns:
      a boolean array of shape [] and dtype ``bool`` indicating whether the
      inner transformation should be called.
    """


class ConditionallyTransformState(NamedTuple):
  """Maintains inner transform state and adds a step counter."""

  inner_state: Any
  step: chex.Array


def conditionally_transform(
    inner: base.GradientTransformation,
    should_transform_fn: ConditionFn,
    forward_extra_args: bool = False,
) -> base.GradientTransformationExtraArgs:
  """Calls the inner update function only at certain steps.

  Creates a transformation wrapper that conditionally applies the inner gradient
  transformation, and if the condition is not met, just passes the updates and
  inner state through unchanged. The behavior is controlled by a user specified
  function ``should_transform_fn`` that is called by ``conditionally_transform``
  passing as input a counter of the number of times that the ``update`` function
  has been previously called, the user specified function must returns a boolean
  controlling whether the inner transformation should be called.

  Args:
    inner: the inner transformation.
    should_transform_fn: function takes in a ``step`` counter (array of shape []
      and dtype ``int32``), and returns a boolean array of shape []. If
      ``forward_extra_args`` is set to True, any extra arguments are also
      forwarded to the ``should_transform_fn``.
    forward_extra_args: forward extra args to ``should_transform_fn``.

  Returns:
    A new :class:`optax.GradientTransformationExtraArgs`.

  .. warning::
    If instead you want to set the ``updates`` to zero when the condition
    is not met, you can use the ``conditionally_mask`` wrapper.

  .. versionadded:: 0.2.3
  """
  inner = base.with_extra_args_support(inner)

  def init_fn(params):
    return ConditionallyTransformState(
        inner_state=inner.init(params), step=jnp.zeros([], dtype=jnp.int32)
    )

  def update_fn(updates, state, params=None, **extra_args):

    def do_update(_):
      return inner.update(updates, state.inner_state, params, **extra_args)

    def reject_update(_):
      return updates, state.inner_state

    condition_kwargs = extra_args if forward_extra_args else {}
    updates, new_inner_state = lax.cond(
        should_transform_fn(state.step, **condition_kwargs),
        do_update,
        reject_update,
        operand=None,
    )
    return updates, ConditionallyTransformState(
        new_inner_state, numerics.safe_increment(state.step)
    )

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


class ConditionallyMaskState(NamedTuple):
  step: chex.Array
  inner_state: base.OptState


def conditionally_mask(
    inner: base.GradientTransformation,
    should_transform_fn: ConditionFn,
    forward_extra_args: bool = False,
) -> base.GradientTransformationExtraArgs:
  """Calls the inner update function only at certain steps.

  Creates a transformation wrapper that conditionally applies the inner gradient
  transformation, and if the condition is not met, the updates are set to 0,
  while the inner state is passed through unchanged. The behavior is controlled
  by a user specified function ``should_transform_fn`` that is called
  by ``conditionally_transform`` passing as input a counter of the number of
  times that the ``update`` function has been previously called, the user
  specified function must returns a boolean controlling whether the inner
  transformation should be called.

  Args:
    inner: the inner transformation.
    should_transform_fn: function takes in a step counter (array of shape [] and
      dtype ``int32``), and returns a boolean array of shape []. If
      ``forward_extra_args`` is set to True, any extra arguments are also
      forwarded to the ``should_transform_fn``.
    forward_extra_args: forward extra args to ``should_transform_fn``.

  Returns:
    A new :class:`optax.GradientTransformationExtraArgs`.

  .. warning::
    If instead you want to leave ``updates`` unchanged when the condition
    is not met, you can use the ``conditionally_transform`` wrapper.

  .. versionadded:: 0.2.3
  """
  inner = base.with_extra_args_support(inner)

  def init_fn(params):
    return ConditionallyMaskState(
        step=jnp.zeros([], jnp.int32), inner_state=inner.init(params)
    )

  def update_fn(updates, state, params=None, **extra_args):

    def do_update(_):
      return inner.update(updates, state.inner_state, params, **extra_args)

    def reject_update(_):
      return otu.tree_zeros_like(updates), state.inner_state

    condition_kwargs = extra_args if forward_extra_args else {}
    updates, new_inner_state = lax.cond(
        should_transform_fn(state.step, **condition_kwargs),
        do_update,
        reject_update,
        operand=None,
    )

    return updates, ConditionallyMaskState(
        step=numerics.safe_increment(state.step),
        inner_state=new_inner_state,
    )

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


class ApplyIfFiniteState(NamedTuple):
  """State of the `GradientTransformation` returned by `apply_if_finite`.

  Attributes:
    notfinite_count: Number of consecutive gradient updates containing an Inf or
      a NaN. This number is reset to 0 whenever a gradient update without an Inf
      or a NaN is done.
    last_finite: Whether or not the last gradient update contained an Inf or a
      NaN.
    total_notfinite: Total number of gradient updates containing an Inf or a NaN
      since this optimizer was initialised. This number is never reset.
      inner_state: The state of the inner `GradientTransformation`.
  """

  notfinite_count: Any
  last_finite: Any
  total_notfinite: Any
  inner_state: Any


def apply_if_finite(
    inner: base.GradientTransformation, max_consecutive_errors: int
) -> base.GradientTransformation:
  """A function that wraps an optimizer to make it robust to a few NaNs or Infs.

  The purpose of this function is to prevent any optimization to happen if the
  gradients contain NaNs or Infs. That is, when a NaN or Inf is detected in the
  gradients, the wrapped optimizer ignores that gradient update. If the NaNs or
  Infs persist after a given number of updates, the wrapped optimizer gives up
  and accepts the update.

  Args:
    inner: Inner transformation to be wrapped.
    max_consecutive_errors: Maximum number of consecutive gradient updates
      containing NaNs or Infs that the wrapped optimizer will ignore. After that
      many ignored updates, the optimizer will give up and accept.

  Returns:
    New :class:`optax.GradientTransformationExtraArgs`.
  """

  inner = base.with_extra_args_support(inner)

  def init(params):
    return ApplyIfFiniteState(
        notfinite_count=jnp.zeros([], jnp.int32),
        last_finite=jnp.array(True, jnp.bool_),
        total_notfinite=jnp.zeros([], jnp.int32),
        inner_state=inner.init(params),
    )

  def update(updates, state, params=None, **extra_args):
    inner_state = state.inner_state
    flat_updates = jax.tree.flatten(updates)[0]
    isfinite = jnp.all(
        jnp.array([jnp.all(jnp.isfinite(p)) for p in flat_updates])
    )
    notfinite_count = jnp.where(
        isfinite,
        jnp.zeros([], jnp.int32),
        numerics.safe_increment(state.notfinite_count),
    )

    def do_update(_):
      return inner.update(updates, inner_state, params, **extra_args)

    def reject_update(_):
      return otu.tree_zeros_like(updates), inner_state

    updates, new_inner_state = lax.cond(
        jnp.logical_or(isfinite, notfinite_count > max_consecutive_errors),
        do_update,
        reject_update,
        operand=None,
    )

    return updates, ApplyIfFiniteState(
        notfinite_count=notfinite_count,
        last_finite=isfinite,
        total_notfinite=jnp.where(
            isfinite,
            state.total_notfinite,
            numerics.safe_increment(state.total_notfinite),
        ),
        inner_state=new_inner_state,
    )

  return base.GradientTransformationExtraArgs(init=init, update=update)
