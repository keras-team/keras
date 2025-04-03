# Copyright 2018 The JAX Authors.
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

"""Examples of how to write optimizers with JAX.

You likely do not mean to import this module! The optimizers in this library
are intended as examples only. If you are looking for a fully featured optimizer
library, consider Optax_.

This module contains some convenient optimizer definitions, specifically
initialization and update functions, which can be used with ndarrays or
arbitrarily-nested tuple/list/dicts of ndarrays.

An optimizer is modeled as an ``(init_fun, update_fun, get_params)`` triple of
functions, where the component functions have these signatures:

::

  init_fun(params)

  Args:
    params: pytree representing the initial parameters.

  Returns:
    A pytree representing the initial optimizer state, which includes the
    initial parameters and may also include auxiliary values like initial
    momentum. The optimizer state pytree structure generally differs from that
    of `params`.

::

  update_fun(step, grads, opt_state)

  Args:
    step: integer representing the step index.
    grads: a pytree with the same structure as `get_params(opt_state)`
      representing the gradients to be used in updating the optimizer state.
    opt_state: a pytree representing the optimizer state to be updated.

  Returns:
    A pytree with the same structure as the `opt_state` argument representing
    the updated optimizer state.

::

  get_params(opt_state)

  Args:
    opt_state: pytree representing an optimizer state.

  Returns:
    A pytree representing the parameters extracted from `opt_state`, such that
    the invariant `params == get_params(init_fun(params))` holds true.


Notice that an optimizer implementation has a lot of flexibility in the form of
opt_state: it just has to be a pytree of JaxTypes (so that it can be passed to
the JAX transforms defined in api.py) and it has to be consumable by update_fun
and get_params.

Example Usage:

.. code-block:: python

  opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
  opt_state = opt_init(params)

  def step(step, opt_state):
    value, grads = jax.value_and_grad(loss_fn)(get_params(opt_state))
    opt_state = opt_update(step, grads, opt_state)
    return value, opt_state

  for i in range(num_steps):
    value, opt_state = step(i, opt_state)


.. _Optax: https://github.com/google-deepmind/optax
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from collections import namedtuple
import functools
from functools import partial

import jax
import jax.numpy as jnp
from jax._src.util import safe_zip, safe_map, unzip2

map = safe_map
zip = safe_zip


# The implementation here basically works by flattening pytrees. There are two
# levels of pytrees to think about: the pytree of params, which we can think of
# as defining an "outer pytree", and a pytree produced by applying init_fun to
# each leaf of the params pytree, which we can think of as the "inner pytrees".
# Since pytrees can be flattened, that structure is isomorphic to a list of
# lists (with no further nesting).

OptimizerState = namedtuple("OptimizerState",
                            ["packed_state", "tree_def", "subtree_defs"])
jax.tree_util.register_pytree_node(
    OptimizerState,
    lambda xs: ((xs.packed_state,), (xs.tree_def, xs.subtree_defs)),
    lambda data, xs: OptimizerState(xs[0], data[0], data[1]))


Array = Any
Params = Any  # Parameters are arbitrary nests of `jnp.ndarrays`.
State = Any   # internal State
Updates = Params  # Gradient updates are of the same type as parameters.

InitFn = Callable[[Params], OptimizerState]
Step = int
UpdateFn = Callable[[Step, Updates, OptimizerState], OptimizerState]
ParamsFn = Callable[[OptimizerState], Params]

class Optimizer(NamedTuple):
  init_fn: InitFn
  update_fn: UpdateFn
  params_fn: ParamsFn

Schedule = Callable[[Step], float]

def optimizer(opt_maker: Callable[...,
  tuple[Callable[[Params], State],
        Callable[[Step, Updates, Params], Params],
        Callable[[State], Params]]]) -> Callable[..., Optimizer]:
  """Decorator to make an optimizer defined for arrays generalize to containers.

  With this decorator, you can write init, update, and get_params functions that
  each operate only on single arrays, and convert them to corresponding
  functions that operate on pytrees of parameters. See the optimizers defined in
  optimizers.py for examples.

  Args:
    opt_maker: a function that returns an ``(init_fun, update_fun, get_params)``
      triple of functions that might only work with ndarrays, as per

      .. code-block:: haskell

          init_fun :: ndarray -> OptStatePytree ndarray
          update_fun :: OptStatePytree ndarray -> OptStatePytree ndarray
          get_params :: OptStatePytree ndarray -> ndarray

  Returns:
    An ``(init_fun, update_fun, get_params)`` triple of functions that work on
    arbitrary pytrees, as per

    .. code-block:: haskell

          init_fun :: ParameterPytree ndarray -> OptimizerState
          update_fun :: OptimizerState -> OptimizerState
          get_params :: OptimizerState -> ParameterPytree ndarray

    The OptimizerState pytree type used by the returned functions is isomorphic
    to ``ParameterPytree (OptStatePytree ndarray)``, but may store the state
    instead as e.g. a partially-flattened data structure for performance.
  """

  @functools.wraps(opt_maker)
  def tree_opt_maker(*args, **kwargs):
    init, update, get_params = opt_maker(*args, **kwargs)

    @functools.wraps(init)
    def tree_init(x0_tree):
      x0_flat, tree = jax.tree.flatten(x0_tree)
      initial_states = [init(x0) for x0 in x0_flat]
      states_flat, subtrees = unzip2(map(jax.tree.flatten, initial_states))
      return OptimizerState(states_flat, tree, subtrees)

    @functools.wraps(update)
    def tree_update(i, grad_tree, opt_state):
      states_flat, tree, subtrees = opt_state
      grad_flat, tree2 = jax.tree.flatten(grad_tree)
      if tree2 != tree:
        msg = ("optimizer update function was passed a gradient tree that did "
               "not match the parameter tree structure with which it was "
               "initialized: parameter tree {} and grad tree {}.")
        raise TypeError(msg.format(tree, tree2))
      states = map(jax.tree.unflatten, subtrees, states_flat)
      new_states = map(partial(update, i), grad_flat, states)
      new_states_flat, subtrees2 = unzip2(map(jax.tree.flatten, new_states))
      for subtree, subtree2 in zip(subtrees, subtrees2):
        if subtree2 != subtree:
          msg = ("optimizer update function produced an output structure that "
                 "did not match its input structure: input {} and output {}.")
          raise TypeError(msg.format(subtree, subtree2))
      return OptimizerState(new_states_flat, tree, subtrees)

    @functools.wraps(get_params)
    def tree_get_params(opt_state):
      states_flat, tree, subtrees = opt_state
      states = map(jax.tree.unflatten, subtrees, states_flat)
      params = map(get_params, states)
      return jax.tree.unflatten(tree, params)

    return Optimizer(tree_init, tree_update, tree_get_params)
  return tree_opt_maker


### optimizers

@optimizer
def sgd(step_size):
  """Construct optimizer triple for stochastic gradient descent.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    return x0
  def update(i, g, x):
    return x - step_size(i) * g
  def get_params(x):
    return x
  return Optimizer(init, update, get_params)

@optimizer
def momentum(step_size: Schedule, mass: float):
  """Construct optimizer triple for SGD with momentum.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.
    mass: positive scalar representing the momentum coefficient.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    v0 = jnp.zeros_like(x0)
    return x0, v0
  def update(i, g, state):
    x, velocity = state
    velocity = mass * velocity + g
    x = x - step_size(i) * velocity
    return x, velocity
  def get_params(state):
    x, _ = state
    return x
  return init, update, get_params


@optimizer
def nesterov(step_size: Schedule, mass: float):
  """Construct optimizer triple for SGD with Nesterov momentum.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.
    mass: positive scalar representing the momentum coefficient.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    v0 = jnp.zeros_like(x0)
    return x0, v0
  def update(i, g, state):
    x, velocity = state
    velocity = mass * velocity + g
    x = x - step_size(i) * (mass * velocity + g)
    return x, velocity
  def get_params(state):
    x, _ = state
    return x
  return init, update, get_params


@optimizer
def adagrad(step_size, momentum=0.9):
  """Construct optimizer triple for Adagrad.

  Adaptive Subgradient Methods for Online Learning and Stochastic Optimization:
  http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.
    momentum: optional, a positive scalar value for momentum

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)

  def init(x0):
    g_sq = jnp.zeros_like(x0)
    m = jnp.zeros_like(x0)
    return x0, g_sq, m

  def update(i, g, state):
    x, g_sq, m = state
    g_sq += jnp.square(g)
    g_sq_inv_sqrt = jnp.where(g_sq > 0, 1. / jnp.sqrt(g_sq), 0.0)
    m = (1. - momentum) * (g * g_sq_inv_sqrt) + momentum * m
    x = x - step_size(i) * m
    return x, g_sq, m

  def get_params(state):
    x, _, _ = state
    return x

  return init, update, get_params


@optimizer
def rmsprop(step_size, gamma=0.9, eps=1e-8):
  """Construct optimizer triple for RMSProp.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.
      gamma: Decay parameter.
      eps: Epsilon parameter.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    avg_sq_grad = jnp.zeros_like(x0)
    return x0, avg_sq_grad
  def update(i, g, state):
    x, avg_sq_grad = state
    avg_sq_grad = avg_sq_grad * gamma + jnp.square(g) * (1. - gamma)
    x = x - step_size(i) * g / jnp.sqrt(avg_sq_grad + eps)
    return x, avg_sq_grad
  def get_params(state):
    x, _ = state
    return x
  return init, update, get_params


@optimizer
def rmsprop_momentum(step_size, gamma=0.9, eps=1e-8, momentum=0.9):
  """Construct optimizer triple for RMSProp with momentum.

  This optimizer is separate from the rmsprop optimizer because it needs to
  keep track of additional parameters.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.
    gamma: Decay parameter.
    eps: Epsilon parameter.
    momentum: Momentum parameter.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    avg_sq_grad = jnp.zeros_like(x0)
    mom = jnp.zeros_like(x0)
    return x0, avg_sq_grad, mom
  def update(i, g, state):
    x, avg_sq_grad, mom = state
    avg_sq_grad = avg_sq_grad * gamma + jnp.square(g) * (1. - gamma)
    mom = momentum * mom + step_size(i) * g / jnp.sqrt(avg_sq_grad + eps)
    x = x - mom
    return x, avg_sq_grad, mom
  def get_params(state):
    x, _, _ = state
    return x
  return init, update, get_params


@optimizer
def adam(step_size, b1=0.9, b2=0.999, eps=1e-8):
  """Construct optimizer triple for Adam.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    m0 = jnp.zeros_like(x0)
    v0 = jnp.zeros_like(x0)
    return x0, m0, v0
  def update(i, g, state):
    x, m, v = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * jnp.square(g) + b2 * v  # Second moment estimate.
    mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
    vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
    x = x - step_size(i) * mhat / (jnp.sqrt(vhat) + eps)
    return x, m, v
  def get_params(state):
    x, _, _ = state
    return x
  return init, update, get_params


@optimizer
def adamax(step_size, b1=0.9, b2=0.999, eps=1e-8):
  """Construct optimizer triple for AdaMax (a variant of Adam based on infinity norm).

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    m0 = jnp.zeros_like(x0)
    u0 = jnp.zeros_like(x0)
    return x0, m0, u0
  def update(i, g, state):
    x, m, u = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    u = jnp.maximum(b2 * u, jnp.abs(g))  # Update exponentially weighted infinity norm.
    x = (x - (step_size(i) / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))) * m
         / (u + eps))
    return x, m, u
  def get_params(state):
    x, _, _ = state
    return x
  return init, update, get_params


@optimizer
def sm3(step_size, momentum=0.9):
  """Construct optimizer triple for SM3.

  Memory-Efficient Adaptive Optimization for Large-Scale Learning.
  https://arxiv.org/abs/1901.11150

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.
    momentum: optional, a positive scalar value for momentum

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)

  def splice(seq, i, x):
    lst = list(seq)
    lst[i:i+1] = x
    return lst

  def broadcast_into(ndim, x, axis):
    idx = splice([None] * ndim, axis, [slice(None)])
    return x[tuple(idx)]

  def init(x0):
    x_shape = x0.shape
    x0 = jnp.atleast_1d(x0)
    vs = [jnp.zeros(sz, dtype=x0.dtype) for sz in x0.shape]
    return x0, jnp.zeros_like(x0), vs, x_shape

  def update(i, g, state):
    x, m, vs, x_shape = state
    vs = [broadcast_into(g.ndim, v, i) for i, v in enumerate(vs)]
    accum = functools.reduce(jnp.minimum, vs) + jnp.square(g)
    accum_inv_sqrt = jnp.where(accum > 0, 1. / jnp.sqrt(accum), 0)
    m = (1. - momentum) * (g * accum_inv_sqrt) + momentum * m
    x = x - step_size(i) * m
    vs = [accum.max(splice(range(x.ndim), j, [])) for j in range(x.ndim)]
    return x, m, vs, x_shape

  def get_params(state):
    x, _, _, x_shape = state
    return x.reshape(x_shape)

  return init, update, get_params


### learning rate schedules

def constant(step_size) -> Schedule:
  def schedule(i):
    return step_size
  return schedule

def exponential_decay(step_size, decay_steps, decay_rate):
  def schedule(i):
    return step_size * decay_rate ** (i / decay_steps)
  return schedule

def inverse_time_decay(step_size, decay_steps, decay_rate, staircase=False):
  if staircase:
    def schedule(i):
      return step_size / (1 + decay_rate * jnp.floor(i / decay_steps))
  else:
    def schedule(i):
      return step_size / (1 + decay_rate * i / decay_steps)
  return schedule

def polynomial_decay(step_size, decay_steps, final_step_size, power=1.0):
  def schedule(step_num):
    step_num = jnp.minimum(step_num, decay_steps)
    step_mult = (1 - step_num / decay_steps) ** power
    return step_mult * (step_size - final_step_size) + final_step_size

  return schedule

def piecewise_constant(boundaries: Any, values: Any):
  boundaries = jnp.array(boundaries)
  values = jnp.array(values)
  if not boundaries.ndim == values.ndim == 1:
    raise ValueError("boundaries and values must be sequences")
  if not boundaries.shape[0] == values.shape[0] - 1:
    raise ValueError("boundaries length must be one shorter than values length")

  def schedule(i):
    return values[jnp.sum(i > boundaries)]
  return schedule

def make_schedule(scalar_or_schedule: float | Schedule) -> Schedule:
  if callable(scalar_or_schedule):
    return scalar_or_schedule
  elif jnp.ndim(scalar_or_schedule) == 0:
    return constant(scalar_or_schedule)
  else:
    raise TypeError(type(scalar_or_schedule))


### utilities

def l2_norm(tree):
  """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
  leaves, _ = jax.tree.flatten(tree)
  return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))

def clip_grads(grad_tree, max_norm):
  """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
  norm = l2_norm(grad_tree)
  normalize = lambda g: jnp.where(norm < max_norm, g, g * (max_norm / norm))
  return jax.tree.map(normalize, grad_tree)


### serialization utilities

class JoinPoint:
  """Marks the boundary between two joined (nested) pytrees."""
  def __init__(self, subtree):
    self.subtree = subtree

  # Since pytrees are containers of numpy arrays, look iterable.
  def __iter__(self):
    yield self.subtree

def unpack_optimizer_state(opt_state):
  """Converts an OptimizerState to a marked pytree.

  Converts an OptimizerState to a marked pytree with the leaves of the outer
  pytree represented as JoinPoints to avoid losing information. This function is
  intended to be useful when serializing optimizer states.

  Args:
    opt_state: An OptimizerState
  Returns:
    A pytree with JoinPoint leaves that contain a second level of pytrees.
  """
  states_flat, tree_def, subtree_defs = opt_state
  subtrees = map(jax.tree.unflatten, subtree_defs, states_flat)
  sentinels = [JoinPoint(subtree) for subtree in subtrees]
  return jax.tree.unflatten(tree_def, sentinels)

def pack_optimizer_state(marked_pytree):
  """Converts a marked pytree to an OptimizerState.

  The inverse of unpack_optimizer_state. Converts a marked pytree with the
  leaves of the outer pytree represented as JoinPoints back into an
  OptimizerState. This function is intended to be useful when deserializing
  optimizer states.

  Args:
    marked_pytree: A pytree containing JoinPoint leaves that hold more pytrees.
  Returns:
    An equivalent OptimizerState to the input argument.
  """
  sentinels, tree_def = jax.tree.flatten(marked_pytree)
  assert all(isinstance(s, JoinPoint) for s in sentinels)
  subtrees = [s.subtree for s in sentinels]
  states_flat, subtree_defs = unzip2(map(jax.tree.flatten, subtrees))
  return OptimizerState(states_flat, tree_def, subtree_defs)
