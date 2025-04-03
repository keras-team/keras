# Copyright 2024 The Flax Authors.
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
from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import optax

from flax import nnx
from flax.nnx import filterlib
from flax.nnx import variablelib
from flax.nnx.object import Object
from flax.nnx.variablelib import Variable, VariableState

M = tp.TypeVar('M', bound=nnx.Module)

# TODO: add tests and docstrings


class OptState(Variable):
  """Holds any optimizer state."""

  pass


class OptArray(OptState):
  """Holds an array of optimizer state."""

  pass


class OptVariable(OptState):
  """Holds Variable state."""

  source_type: type[Variable]
  pass


def _wrap_optimizer_state(opt_state):
  def wrap_optimizer_state_fn(x):
    if isinstance(x, variablelib.VariableState):
      new_state = x.copy()
      new_state.source_type = x.type
      new_state.type = OptVariable
      return new_state.to_variable()
    else:
      return OptArray(x)

  return jax.tree.map(
    wrap_optimizer_state_fn,
    opt_state,
    is_leaf=lambda x: isinstance(x, variablelib.VariableState),
  )


def _opt_state_variables_to_state(opt_state):
  def optimizer_variable_to_state_fn(x):
    if isinstance(x, OptVariable):
      state = x.to_state()
      state.type = x.source_type
      del state.source_type
      return state
    elif isinstance(x, OptArray):
      return x.value
    else:
      raise TypeError(
        f'Unexpected type when converting optimizer state: {type(x)}'
      )

  return jax.tree.map(optimizer_variable_to_state_fn, opt_state)


def _update_opt_state(opt_state, updates):
  def optimizer_update_variables(x, update):
    if isinstance(x, OptVariable):
      if not isinstance(update, VariableState):
        raise TypeError(
          f'Expected update to be VariableState, got {type(update)}'
        )
      x.raw_value = update.value
    elif isinstance(x, OptArray):
      if isinstance(update, VariableState):
        raise TypeError(
          f'Expected update to not to be a VariableState, got {update}'
        )
      x.raw_value = update
    else:
      raise TypeError(
        f'Unexpected type when updating optimizer state: {type(x)}'
      )

  return jax.tree.map(optimizer_update_variables, opt_state, updates)


class Optimizer(Object, tp.Generic[M]):
  """Simple train state for the common case with a single Optax optimizer.

  Example usage::

    >>> import jax, jax.numpy as jnp
    >>> from flax import nnx
    >>> import optax
    ...
    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear1 = nnx.Linear(2, 3, rngs=rngs)
    ...     self.linear2 = nnx.Linear(3, 4, rngs=rngs)
    ...   def __call__(self, x):
    ...     return self.linear2(self.linear1(x))
    ...
    >>> x = jax.random.normal(jax.random.key(0), (1, 2))
    >>> y = jnp.ones((1, 4))
    ...
    >>> model = Model(nnx.Rngs(0))
    >>> tx = optax.adam(1e-3)
    >>> state = nnx.Optimizer(model, tx)
    ...
    >>> loss_fn = lambda model: ((model(x) - y) ** 2).mean()
    >>> loss_fn(model)
    Array(2.3359997, dtype=float32)
    >>> grads = nnx.grad(loss_fn)(state.model)
    >>> state.update(grads)
    >>> loss_fn(model)
    Array(2.310461, dtype=float32)

  Note that you can easily extend this class by subclassing it for storing
  additional data (e.g. adding metrics).

  Example usage::

    >>> class TrainState(nnx.Optimizer):
    ...   def __init__(self, model, tx, metrics):
    ...     self.metrics = metrics
    ...     super().__init__(model, tx)
    ...   def update(self, *, grads, **updates):
    ...     self.metrics.update(**updates)
    ...     super().update(grads)
    ...
    >>> metrics = nnx.metrics.Average()
    >>> state = TrainState(model, tx, metrics)
    ...
    >>> grads = nnx.grad(loss_fn)(state.model)
    >>> state.update(grads=grads, values=loss_fn(state.model))
    >>> state.metrics.compute()
    Array(2.310461, dtype=float32)
    >>> state.update(grads=grads, values=loss_fn(state.model))
    >>> state.metrics.compute()
    Array(2.2978127, dtype=float32)

  For more exotic usecases (e.g. multiple optimizers) it's probably best to
  fork the class and modify it.

  Args:
    step: An ``OptState`` :class:`Variable` that tracks the step count.
    model: The wrapped :class:`Module`.
    tx: An Optax gradient transformation.
    opt_state: The Optax optimizer state.
  """

  def __init__(
    self,
    model: M,
    tx: optax.GradientTransformation,
    wrt: filterlib.Filter = nnx.Param,
  ):
    """
    Instantiate the class and wrap the :class:`Module` and Optax gradient
    transformation. Instantiate the optimizer state to keep track of
    :class:`Variable` types specified in ``wrt``. Set the step count to 0.

    Args:
      model: An NNX Module.
      tx: An Optax gradient transformation.
      wrt: optional argument to filter for which :class:`Variable`'s to keep
        track of in the optimizer state. These should be the :class:`Variable`'s
        that you plan on updating; i.e. this argument value should match the
        ``wrt``  argument passed to the ``nnx.grad`` call that will generate the
        gradients that will be passed into the ``grads`` argument of the
        :func:`update` method.
    """
    self.step = OptState(jnp.array(0, dtype=jnp.uint32))
    self.model = model
    self.tx = tx
    self.opt_state = _wrap_optimizer_state(tx.init(nnx.state(model, wrt)))
    self.wrt = wrt

  def update(self, grads, **kwargs):
    """Updates ``step``, ``params``, ``opt_state`` and ``**kwargs`` in return value.
    The ``grads`` must be derived from ``nnx.grad(..., wrt=self.wrt)``, where the
    gradients are with respect to the same :class:`Variable` types as defined in
    ``self.wrt`` during instantiation of this ``Optimizer``. For example::

      >>> from flax import nnx
      >>> import jax, jax.numpy as jnp
      >>> import optax

      >>> class CustomVariable(nnx.Variable):
      ...   pass

      >>> class Model(nnx.Module):
      ...   def __init__(self, rngs):
      ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
      ...     self.custom_variable = CustomVariable(jnp.ones((1, 3)))
      ...   def __call__(self, x):
      ...     return self.linear(x) + self.custom_variable
      >>> model = Model(rngs=nnx.Rngs(0))
      >>> jax.tree.map(jnp.shape, nnx.state(model))
      State({
        'custom_variable': VariableState(
          type=CustomVariable,
          value=(1, 3)
        ),
        'linear': {
          'bias': VariableState(
            type=Param,
            value=(3,)
          ),
          'kernel': VariableState(
            type=Param,
            value=(2, 3)
          )
        }
      })

      >>> # update:
      >>> # - only Linear layer parameters
      >>> # - only CustomVariable parameters
      >>> # - both Linear layer and CustomVariable parameters
      >>> loss_fn = lambda model, x, y: ((model(x) - y) ** 2).mean()
      >>> for variable in (nnx.Param, CustomVariable, (nnx.Param, CustomVariable)):
      ...   # make sure `wrt` arguments match for `nnx.Optimizer` and `nnx.grad`
      ...   state = nnx.Optimizer(model, optax.adam(1e-3), wrt=variable)
      ...   grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, variable))(
      ...     state.model, jnp.ones((1, 2)), jnp.ones((1, 3))
      ...   )
      ...   state.update(grads=grads)

    Note that internally this function calls ``.tx.update()`` followed by a call
    to ``optax.apply_updates()`` to update ``params`` and ``opt_state``.

    Args:
      grads: the gradients derived from ``nnx.grad``.
      **kwargs: additional keyword arguments passed to the tx.update, to support
      ``GradientTransformationExtraArgs``, such as ``optax.scale_by_backtracking_linesearch``.
    """
    params = nnx.state(self.model, self.wrt)
    opt_state = _opt_state_variables_to_state(self.opt_state)

    updates, new_opt_state = self.tx.update(grads, opt_state, params, **kwargs)
    new_params = optax.apply_updates(params, updates)
    assert isinstance(new_params, nnx.State)

    self.step.value += 1
    nnx.update(self.model, new_params)
    _update_opt_state(self.opt_state, new_opt_state)