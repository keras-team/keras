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
"""Base interfaces and datatypes."""

from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Protocol, Sequence, Union, runtime_checkable

import chex
import jax
import jax.numpy as jnp

NO_PARAMS_MSG = (
    'You are using a transformation that requires the current value of '
    'parameters, but you are not passing `params` when calling `update`.'
)

PyTree = Any
Shape = Sequence[int]

OptState = chex.ArrayTree  # States are arbitrary nests of `jnp.ndarrays`.
Params = chex.ArrayTree  # Parameters are arbitrary nests of `jnp.ndarrays`.
Updates = Params  # Gradient updates are of the same type as parameters.

Schedule = Callable[[chex.Numeric], chex.Numeric]
ScheduleState = Any
ScalarOrSchedule = Union[float, jax.Array, Schedule]


@runtime_checkable
class StatefulSchedule(Protocol):
  """Base interface for stateful schedules."""

  def init(self) -> ScheduleState:
    """Initialize the state of the stateful schedule."""

  def update(
      self,
      state: ScheduleState,
      **extra_args,
  ) -> ScheduleState:
    """Updates the current schedule state."""

  def __call__(
      self,
      state: ScheduleState,
      **extra_args,
  ) -> chex.Numeric:
    """Computes the current schedule value."""


class TransformInitFn(Protocol):
  """A callable type for the `init` step of a `GradientTransformation`.

  The `init` step takes a tree of `params` and uses these to construct an
  arbitrary structured initial `state` for the gradient transformation. This
  may hold statistics of the past updates or any other non static information.
  """

  def __call__(self, params: Params) -> OptState:
    """The `init` function.

    Args:
      params: The initial value of the parameters.

    Returns:
      The initial state of the gradient transformation.
    """


class TransformUpdateFn(Protocol):
  """A callable type for the `update` step of a `GradientTransformation`.

  The `update` step takes a tree of candidate parameter `updates` (e.g. their
  gradient with respect to some loss), an arbitrary structured `state`, and the
  current `params` of the model being optimized. The `params` argument is
  optional, it must however be provided when using transformations that require
  access to the current values of the parameters.

  For the case where additional arguments are required, an alternative interface
  may be used, see ``TransformUpdateExtraArgsFn`` for details.
  """

  def __call__(
      self, updates: Updates, state: OptState, params: Optional[Params] = None
  ) -> tuple[Updates, OptState]:
    """The `update` function.

    Args:
      updates: A tree of candidate updates.
      state: The state of the gradient transformation.
      params: (Optionally) the current value of the parameters.

    Returns:
      The transformed updates, and the updated state.
    """


class TransformUpdateExtraArgsFn(Protocol):
  """An update function accepting additional keyword arguments."""

  def __call__(
      self,
      updates: Updates,
      state: OptState,
      params: Optional[Params] = None,
      **extra_args: Any,
  ) -> tuple[Updates, OptState]:
    """Update function with optional extra arguments.

    For example, an update function that requires an additional loss parameter
    (which might be useful for implementing learning rate schedules that depend
    on the current loss value) could be expressed as follows:

    >>> def update(updates, state, params=None, *, loss, **extra_args):
    ...   del extra_args
    ...   # use loss value

    Note that the loss value is keyword only, (it follows a ``*`` in the
    signature of the function). This implies users will get explicit errors if
    they try to use this gradient transformation without providing the required
    argument.

    Args:
      updates: The gradient updates passed to this transformation.
      state: The state associated with this transformation
      params: Optional params.
      **extra_args: Additional keyword arguments passed to this transform. All
        implementors of this interface should accept arbitrary keyword
        arguments, ignoring those that are not needed for the current
        transformation. Transformations that require specific extra args should
        specify these using keyword-only arguments.

    Returns:
      Transformed updates, and an updated value for the state.
    """


class GradientTransformation(NamedTuple):
  # pylint: disable=line-too-long
  """A pair of pure functions implementing a gradient transformation.

  Optax optimizers are all implemented as *gradient transformations*.
  A gradient transformation is defined to be a pair of pure functions, which
  are combined together in a `NamedTuple` so that they can be referred to by
  name.

  Note that an extended API is provided for users wishing to build optimizers
  that take additional arguments during the update step. For more details,
  see :func:`optax.GradientTransoformationExtraArgs`.

  Since gradient transformations do not contain any internal state, all stateful
  optimizer properties (such as the current step count when using optimizer
  schedules or  momentum values) are passed through optax gradient
  transformations by using the optimizer *state* pytree. Each time a gradient
  transformation is applied, a new state is computed and returned, ready to be
  passed to the next call to the gradient transformation.

  Since gradient transformations are pure, idempotent functions, the only way
  to change the behavior of a gradient transformation between steps, is to
  change the values in the optimizer state. To see an example of mutating the
  optimizer state in order to control the behavior of an optax gradient
  transformation see the `meta-learning example
  <https://optax.readthedocs.io/en/latest/_collections/examples/meta_learning.html>`_
  in the optax documentation.

  Attributes:
    init: A pure function which, when called with an example instance of the
      parameters whose gradients will be transformed, returns a pytree
      containing the initial value for the optimizer state.
    update: A pure function which takes as input a pytree of updates (with the
      same tree structure as the original params pytree passed to init), the
      previous optimizer state (which may have been initialized using the init
      function), and optionally the current params. The update function then
      returns the computed gradient updates, and a new optimizer state.
  """
  # pylint: disable=line-too-long
  init: TransformInitFn
  update: TransformUpdateFn


class GradientTransformationExtraArgs(GradientTransformation):
  """A specialization of GradientTransformation that supports extra args.

  Extends the existing GradientTransformation interface by adding support for
  passing extra arguments to the update function.

  Note that if no extra args are provided, then the API of this function is
  identical to the case of ``TransformUpdateFn``. This means that we can safely
  wrap any gradient transformation (that does not support extra args) as one
  that does. The new gradient transformation will accept (and ignore) any
  extra arguments that a user might pass to it. This is the behavior implemented
  by ``optax.with_extra_args_support()``.

  Attributes:
    update: Overrides the type signature of the update in the base type to
      accept extra arguments.
  """

  update: TransformUpdateExtraArgsFn


class EmptyState(NamedTuple):
  """An empty state for the simplest stateless transformations."""


def init_empty_state(params: Params) -> EmptyState:
  """Init function for a :class:`GradientTransformation` with empty state."""
  del params
  return EmptyState()


def identity() -> GradientTransformation:
  """Stateless identity transformation that leaves input gradients untouched.

  This function passes through the *gradient updates* unchanged.

  Note, this should not to be confused with `set_to_zero`, which maps the input
  updates to zero - which is the transform required for the *model parameters*
  to be left unchanged when the updates are applied to them.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def update_fn(updates, state, params=None):
    del params
    return updates, state

  return GradientTransformation(init_empty_state, update_fn)


def set_to_zero() -> GradientTransformation:
  """Stateless transformation that maps input gradients to zero.

  The resulting update function, when called, will return a tree of zeros
  matching the shape of the input gradients. This means that when the updates
  returned from this transformation are applied to the model parameters, the
  model parameters will remain unchanged.

  This can be used in combination with `multi_transform` or `masked` to freeze
  (i.e. keep fixed) some parts of the tree of model parameters while applying
  gradient updates to other parts of the tree.

  When updates are set to zero inside the same jit-compiled function as the
  calculation of gradients, optax transformations, and application of updates to
  parameters, unnecessary computations will in general be dropped.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def update_fn(updates, state, params=None):
    del params  # Unused by the zero transform.
    return jax.tree.map(jnp.zeros_like, updates), state

  return GradientTransformation(init_empty_state, update_fn)


def stateless(
    f: Callable[[Updates, Optional[Params]], Updates],
) -> GradientTransformation:
  """Creates a stateless transformation from an update-like function.

  This wrapper eliminates the boilerplate needed to create a transformation that
  does not require saved state between iterations.

  Args:
    f: Update function that takes in updates (e.g. gradients) and parameters and
      returns updates. The parameters may be `None`.

  Returns:
    A :class:`optax.GradientTransformation`.
  """

  def update_fn(updates, state, params=None):
    del state
    return f(updates, params), EmptyState()

  return GradientTransformation(init_empty_state, update_fn)


def stateless_with_tree_map(
    f: Callable[[chex.Array, Optional[chex.Array]], chex.Array],
) -> GradientTransformation:
  """Creates a stateless transformation from an update-like function for arrays.

  This wrapper eliminates the boilerplate needed to create a transformation that
  does not require saved state between iterations, just like optax.stateless.
  In addition, this function will apply the tree map over update/params for you.

  Args:
    f: Update function that takes in an update array (e.g. gradients) and
      parameter array and returns an update array. The parameter array may be
      `None`.

  Returns:
    A :class:`optax.GradientTransformation`.
  """

  def update_fn(updates, state, params=None):
    del state
    if params is not None:
      return jax.tree.map(f, updates, params), EmptyState()
    else:
      f_ = lambda u: f(u, None)
      return jax.tree.map(f_, updates), EmptyState()

  return GradientTransformation(init_empty_state, update_fn)


def with_extra_args_support(
    tx: GradientTransformation,
) -> GradientTransformationExtraArgs:
  """Wraps a gradient transformation, so that it ignores extra args."""

  if isinstance(tx, GradientTransformationExtraArgs):
    return tx

  def update(updates, state, params=None, **extra_args):
    del extra_args
    return tx.update(updates, state, params)

  return GradientTransformationExtraArgs(tx.init, update)
