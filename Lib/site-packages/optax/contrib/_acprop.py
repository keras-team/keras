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
"""ACProp Optimizer.

A contributed implementation of the method from "Momentum Centering and
Asynchronous Update for Adaptive Gradient Methods" by Zhuang et al.
(https://arxiv.org/abs/2110.05454).
"""
from collections.abc import Callable
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform


def scale_by_acprop(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16,
) -> base.GradientTransformation:
  """Rescale updates according to ACProp (asynchronous version of AdaBelief).

  See :func:`optax.contrib.acprop` for more details.

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of variance of grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the second moment of the prediction error to improve
      numerical stability. If backpropagating gradients through the gradient
      transformation (e.g. for meta-learning), this must be non-zero.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    mu = otu.tree_zeros_like(params)  # First moment
    s = otu.tree_zeros_like(params)  # Second Central moment
    return transform.ScaleByBeliefState(
        count=jnp.zeros([], jnp.int32), mu=mu, nu=s
    )

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    prediction_error = jax.tree.map(lambda g, m: g - m, updates, state.mu)
    nu = otu.tree_update_moment_per_elem_norm(prediction_error, state.nu, b2, 2)
    nu = jax.tree.map(lambda v: v + eps_root, nu)
    count_inc = numerics.safe_increment(state.count)

    # On initial step, avoid division by zero and force nu_hat to be 1.
    initial = state.count == 0
    t = jnp.where(initial, count_inc, state.count)
    nu_hat = otu.tree_bias_correction(state.nu, b2, t)
    nu_hat = jax.tree.map(lambda x: jnp.where(initial, 1, x), nu_hat)

    updates = jax.tree.map(
        lambda m, v: m / (jnp.sqrt(v) + eps), updates, nu_hat
    )
    return updates, transform.ScaleByBeliefState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def acprop(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  r"""The ACProp optimizer.

  Follows the implementation from the original repo in PyTorch:
  https://github.com/juntang-zhuang/ACProp-Optimizer.

  ACProp is an adaptive optimizer that combines centering of second momentum
  and asynchronous update. For the update at step t, the denominator uses
  information up to step t-1, while the numerator uses the gradient at step t.

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\varepsilon`, :math:`\bar{\varepsilon}` represent the arguments
  ``b1``, ``b2``, ``eps`` and ``eps_root`` respectively. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0, s_0) = (0, 0)`, representing initial estimates for the
  first and second moments. In practice these values are stored as pytrees
  containing all zeros, with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t` and optimizer state :math:`S_t`
  and computes updates :math:`u_t` and new state :math:`S_{t+1}`. Thus, for
  :math:`t > 0`, we have,

  .. math::

    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      s_t &\leftarrow \beta_2 \cdot s_{t-1} + (1-\beta_2) \cdot (g_t - m_t)^2
      + \bar{\varepsilon} \\
      \hat{s}_t &\leftarrow s_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow -\alpha_t \cdot g_t / \left(\sqrt{\hat{s}_{t-1}}
      + \varepsilon \right) \\
      S_t &\leftarrow (m_t, s_t).
    \end{align*}

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the second moment of the prediction error to
      improve numerical stability. If backpropagating gradients through the
      gradient transformation (e.g. for meta-learning), this must be non-zero.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Adam gradient transformations are applied to all parameters.

  Returns:
    The corresponding `GradientTransformation`.

  References:
    Zuhang et al, `Momentum Centering and Asynchronous Update for Adaptive
    Gradient Methods <https://arxiv.org/abs/2110.05454>`_, 2021
  """
  return combine.chain(
      scale_by_acprop(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )
