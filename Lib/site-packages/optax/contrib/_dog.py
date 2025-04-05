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
"""Distance over gradients algorithm and its variants.

References:
  Ivgi et al., `DoG is SGD's Best Friend: A Parameter-Free Dynamic Step
  Size Schedule<https://arxiv.org/abs/2302.12022>`_, 2023.

  Khaled et al., `DoWG Unleashed: An Efficient Universal Parameter-Free
  Gradient Descent Method<https://arxiv.org/pdf/2305.16284>`_, 2023.
"""

from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import transform
import optax.tree_utils as otu


class DoGState(NamedTuple):
  """State for DoG optimizer."""

  first_step: jax.Array  # bool
  init_params: chex.ArrayTree
  estim_dist: jax.Array
  sum_sq_norm_grads: jax.Array


def scale_by_dog(
    reps_rel: float = 1e-6,
    eps: float = 1e-8,
    init_learning_rate: Optional[float] = None,
) -> base.GradientTransformation:
  r"""Scale by Distance over Gradients (DoG).

  See :func:`optax.contrib.dog` for more details.

  Args:
    reps_rel: value to use to compute the  initial distance 
      (r_epsilon in the paper). Namely, the first step size is given by:
      (reps_rel * (1+\|x_0\|)) / (\|g_0\|^2 + eps)^{1/2}  where x_0 are the 
      initial  weights of  the model (or the parameter group), and g_0 is the
      gradient of the first step.
      As discussed in the paper, this value should be small enough to ensure
      that the first update step will be small enough to not cause the model to
      diverge.
      Suggested value is 1e-6, unless the model uses batch-normalization,
      in which case the suggested value is 1e-4.
    eps: epsilon used for numerical stability - added to the sum of squared
      norm of gradients.
    init_learning_rate: if specified, this value will be used the the initial
      learning rate (i.e. first step size) instead of the rule described above
      with reps_rel.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  .. versionadded:: 0.2.3
  """

  def init_fn(params: base.Params) -> DoGState:
    # Define state parameters with the lowest dtype of the parameters to avoid
    # dtype promotion of parameters resulting in a dtype mismatch between
    # parameters and updates.
    params_dtype = otu.tree_dtype(params, 'lowest')
    return DoGState(
        first_step=jnp.asarray(True),
        init_params=otu.tree_zeros_like(params),
        estim_dist=jnp.asarray(0.0, dtype=params_dtype),
        sum_sq_norm_grads=jnp.asarray(0.0, dtype=params_dtype),
    )

  def update_fn(
      updates: base.Updates, state: DoGState, params: base.Params
  ) -> tuple[base.Updates, DoGState]:

    # Reduces to norm of init_params for first step
    curr_distance = otu.tree_l2_norm(otu.tree_sub(state.init_params, params))
    curr_distance = jnp.where(
        state.first_step, reps_rel * (1 + curr_distance), curr_distance
    )
    init_params = jax.tree.map(
        lambda p, ip: jnp.where(state.first_step, p, ip),
        params,
        state.init_params,
    )

    estim_dist = jnp.maximum(state.estim_dist, curr_distance)
    sq_norm_grads = otu.tree_l2_norm(updates, squared=True)
    sum_sq_norm_grads = sq_norm_grads + state.sum_sq_norm_grads
    learning_rate = estim_dist / jnp.sqrt(sum_sq_norm_grads + eps)

    if init_learning_rate is not None:
      learning_rate = jnp.where(
          state.first_step, init_learning_rate, learning_rate
      )

    new_updates = otu.tree_scalar_mul(learning_rate, updates)
    return new_updates, DoGState(
        first_step=jnp.asarray(False),
        init_params=init_params,
        estim_dist=estim_dist,
        sum_sq_norm_grads=sum_sq_norm_grads,
    )

  return base.GradientTransformation(init_fn, update_fn)


def dog(
    learning_rate: base.ScalarOrSchedule = 1.0,
    reps_rel: float = 1e-6,
    eps: float = 1e-8,
    init_learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
):
  r"""Distance over Gradients optimizer.

  DoG updates parameters :math:`w_t` with stochastic gradients :math:`g_t` 
  according to the update rule:

  .. math::

    \begin{align*}
      \eta_t &= \frac{\max_{i\le t}{\|x_i-x_0\|}}{
        \sqrt{\sum_{i\le t}{\|g_i\|^2+eps}}}\\
      x_{t+1} & = x_{t} - \eta_t\, g_t,
    \end{align*}

  Args:
    learning_rate: optional learning rate (potentially varying according to 
      some predetermined scheduler).
    reps_rel: value to use to compute the  initial distance 
      (r_epsilon in the paper). Namely, the first step size is given by:
      (reps_rel * (1+\|x_0\|)) / (\|g_0\|^2 + eps)^{1/2}  where x_0 are the 
      initial  weights of  the model (or the parameter group), and g_0 is the
      gradient of the first step.
      As discussed in the paper, this value should be small enough to ensure
      that the first update step will be small enough to not cause the model to
      diverge.
      Suggested value is 1e-6, unless the model uses batch-normalization,
      in which case the suggested value is 1e-4.
    eps: epsilon used for numerical stability - added to the sum of squared
      norm of gradients.
    init_learning_rate: if specified, this value will be used the the initial
      learning rate (i.e. first step size) instead of the rule described above
      with reps_rel.
    weight_decay: Strength of the weight decay regularization.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the gradient transformations is applied to all parameters.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> from optax import contrib
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = contrib.dog()
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  value, grad = jax.value_and_grad(f)(params)
    ...  updates, opt_state = solver.update(
    ...    grad, opt_state, params, value=value)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: ', f(params))
    Objective function:  13.99...
    Objective function:  13.99...
    Objective function:  13.99...
    Objective function:  13.99...
    Objective function:  13.99...

  References:
    Ivgi et al., `DoG is SGD's Best Friend: A Parameter-Free Dynamic Step
    Size Schedule <https://arxiv.org/abs/2302.12022>`_, 2023.

  .. versionadded:: 0.2.3
  """
  return combine.chain(
      transform.add_decayed_weights(weight_decay, mask)
      if weight_decay is not None
      else base.identity(),
      scale_by_dog(reps_rel, eps, init_learning_rate),
      transform.scale_by_learning_rate(learning_rate),
  )


class DoWGState(NamedTuple):
  """State for DoWG optimizer."""

  init_params: chex.ArrayTree
  weighted_sq_norm_grads: jax.Array
  estim_sq_dist: jax.Array


def scale_by_dowg(
    init_estim_sq_dist: Optional[float] = None,
    eps: float = 1e-4,
) -> base.GradientTransformation:
  """Scale by Distance over Weighted Gradients (DoWG).

  See :func:`optax.contrib.dowg` for more details.

  Args:
    init_estim_sq_dist: initial guess of the squared distance to solution.
    eps: small value to prevent division by zero in the denominator definining,
      the learning rate, also used as initial guess for the distance to solution
      if ``init_estim_sq_dist`` is None.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  .. versionadded:: 0.2.3
  """

  def init_fn(params: base.Params) -> DoWGState:
    # Define state parameters with the lowest dtype of the parameters to avoid
    # dtype promotion of parameters resulting in a dtype mismatch between
    # parameters and updates.
    params_dtype = otu.tree_dtype(params, 'lowest')
    if init_estim_sq_dist is None:
      init_estim_sq_dist_ = eps
    else:
      init_estim_sq_dist_ = init_estim_sq_dist
    return DoWGState(
        init_params=params,
        estim_sq_dist=jnp.asarray(init_estim_sq_dist_, dtype=params_dtype),
        weighted_sq_norm_grads=jnp.asarray(0.0, dtype=params_dtype),
    )

  def update_fn(
      updates: base.Updates, state: DoWGState, params: base.Params
  ) -> tuple[base.Updates, DoWGState]:
    curr_sq_dist = otu.tree_l2_norm(
        otu.tree_sub(state.init_params, params), squared=True
    )
    estim_sq_dist = jnp.maximum(state.estim_sq_dist, curr_sq_dist)
    step_sq_norm_grads = otu.tree_l2_norm(updates, squared=True)
    weighted_sq_norm_grads = (
        estim_sq_dist * step_sq_norm_grads + state.weighted_sq_norm_grads
    )
    learning_rate = estim_sq_dist / (jnp.sqrt(weighted_sq_norm_grads) + eps)

    new_updates = otu.tree_scalar_mul(learning_rate, updates)
    return new_updates, state._replace(
        estim_sq_dist=estim_sq_dist,
        weighted_sq_norm_grads=weighted_sq_norm_grads,
    )

  return base.GradientTransformation(init_fn, update_fn)


def dowg(
    learning_rate: base.ScalarOrSchedule = 1.0,
    init_estim_sq_dist: Optional[float] = None,
    eps: float = 1e-4,
    weight_decay: Optional[float] = None,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
):
  r"""Distance over weighted Gradients optimizer.

  Examples:
    >>> import optax
    >>> from optax import contrib
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = contrib.dowg()
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  value, grad = jax.value_and_grad(f)(params)
    ...  updates, opt_state = solver.update(
    ...    grad, opt_state, params, value=value)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: ', f(params))
    Objective function:  13.925367
    Objective function:  13.872763
    Objective function:  13.775433
    Objective function:  13.596172
    Objective function:  13.268837

  References:
    Khaled et al., `DoWG Unleashed: An Efficient Universal Parameter-Free
    Gradient Descent Method <https://arxiv.org/pdf/2305.16284>`_, 2023.

  Args:
    learning_rate: optional learning rate (potentially varying according to some
      predetermined scheduler).
    init_estim_sq_dist: initial guess of the squared distance to solution.
    eps: small value to prevent division by zero in the denominator definining,
      the learning rate, also used as initial guess for the distance to solution
      if ``init_estim_sq_dist`` is None.
    weight_decay: Strength of the weight decay regularization.
    mask: A tree with same structure as (or a prefix of) the params PyTree, or a
      Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, `True` for leaves/subtrees you want to apply the
      weight decay to, and `False` for those you want to skip. Note that the
      gradient transformations is applied to all parameters.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  .. versionadded:: 0.2.3
  """
  return combine.chain(
      transform.add_decayed_weights(weight_decay, mask)
      if weight_decay is not None
      else base.identity(),
      scale_by_dowg(init_estim_sq_dist, eps),
      transform.scale_by_learning_rate(learning_rate),
  )
