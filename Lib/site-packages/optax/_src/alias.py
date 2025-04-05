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
"""Aliases for popular optimizers."""

from collections.abc import Callable
import functools
from typing import Any, Optional, Union

import jax.numpy as jnp
from optax._src import base
from optax._src import clipping
from optax._src import combine
from optax._src import factorized
from optax._src import linesearch as _linesearch
from optax._src import transform
from optax._src import wrappers


MaskOrFn = Optional[Union[Any, Callable[[base.Params], Any]]]


def adabelief(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
  r"""The AdaBelief optimizer.

  AdaBelief is an adaptive learning rate optimizer that focuses on fast
  convergence, generalization, and stability. It adapts the step size depending
  on its "belief" in the gradient direction — the optimizer adaptively scales
  the step size by the difference between the predicted and observed gradients.
  AdaBelief is a modified version of :func:`optax.adam` and contains the same
  number of parameters.

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
      \hat{m}_t &\leftarrow m_t / {(1-\beta_1^t)} \\
      \hat{s}_t &\leftarrow s_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow -\alpha_t \cdot \hat{m}_t / \left(\sqrt{\hat{s}_t}
      + \varepsilon \right) \\
      S_t &\leftarrow (m_t, s_t).
    \end{align*}

  With the keyword argument `nesterov=True`, the optimizer uses Nesterov
  momentum, replacing the above :math:`\hat{m}_t` with

  .. math::
      \hat{m}_t \leftarrow
        \beta_1 m_t / {(1-\beta_1^{t+1})} + (1 - \beta_1) g_t / {(1-\beta_1^t)}.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the second moment of the prediction error to
      improve numerical stability. If backpropagating gradients through the
      gradient transformation (e.g. for meta-learning), this must be non-zero.
    nesterov: Whether to use Nesterov momentum.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adabelief(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.38E+01

  References:
    Zhuang, `AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed
    Gradients <https://arxiv.org/abs/2010.07468>`_, 2020
  """
  return combine.chain(
      transform.scale_by_belief(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          nesterov=nesterov,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )


def adadelta(
    learning_rate: Optional[base.ScalarOrSchedule] = None,
    rho: float = 0.9,
    eps: float = 1e-6,
    weight_decay: float = 0.0,
    weight_decay_mask: MaskOrFn = None,
) -> base.GradientTransformation:
  """The Adadelta optimizer.

  Adadelta is a stochastic gradient descent method that adapts learning rates
  based on a moving window of gradient updates. Adadelta is a modification of
  Adagrad.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    rho: A coefficient used for computing a running average of squared
      gradients.
    eps: Term added to the denominator to improve numerical stability.
    weight_decay: Optional rate at which to decay weights.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> f = lambda x: jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adadelta(learning_rate=10.)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.36E+01
    Objective function: 1.32E+01
    Objective function: 1.29E+01
    Objective function: 1.25E+01
    Objective function: 1.21E+01

  References:
    Zeiler, `Adadelta: An Adaptive Learning Rate Optimizer
    <https://arxiv.org/pdf/1212.5701.pdf>`_, 2012
  """
  return combine.chain(
      transform.add_decayed_weights(weight_decay, mask=weight_decay_mask),
      transform.scale_by_adadelta(rho=rho, eps=eps),
      transform.scale_by_learning_rate(learning_rate),
  )


def adafactor(
    learning_rate: Optional[base.ScalarOrSchedule] = None,
    min_dim_size_to_factor: int = 128,
    decay_rate: float = 0.8,
    decay_offset: int = 0,
    multiply_by_parameter_scale: float = True,
    clipping_threshold: Optional[float] = 1.0,
    momentum: Optional[float] = None,
    dtype_momentum: Any = jnp.float32,
    weight_decay_rate: Optional[float] = None,
    eps: float = 1e-30,
    factored: bool = True,
    weight_decay_mask: MaskOrFn = None,
) -> base.GradientTransformation:
  """The Adafactor optimizer.

  Adafactor is an adaptive learning rate optimizer that focuses on fast
  training of large scale neural networks. It saves memory by using a factored
  estimate of the second order moments used to scale gradients.

  Args:
      learning_rate: A global scaling factor, either fixed or evolving along
        iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
        Note that the natural scale for Adafactor's LR is markedly different
        from Adam, one doesn't use the 1/sqrt(hidden) correction for this optim
        with attention-based models.
      min_dim_size_to_factor: Only factor the statistics if two array dimensions
        have at least this size.
      decay_rate: Controls second-moment exponential decay schedule.
      decay_offset: For fine-tuning, one may set this to the starting step
        number of the fine-tuning phase.
      multiply_by_parameter_scale: If True, then scale learning_rate by
        parameter norm. If False, provided learning_rate is absolute step size.
      clipping_threshold: Optional clipping threshold. Must be >= 1. If None,
        clipping is disabled.
      momentum: Optional value between 0 and 1, enables momentum and uses extra
        memory if non-None! None by default.
      dtype_momentum: Data type of momentum buffers.
      weight_decay_rate: Optional rate at which to decay weights.
      eps: Regularization constant for root mean squared gradient.
      factored: Whether to use factored second-moment estimates.
      weight_decay_mask: A tree with same structure as (or a prefix of) the
        params PyTree, or a Callable that returns such a pytree given the
        params/updates. The leaves should be booleans, `True` for
        leaves/subtrees you want to apply the transformation to, and `False` for
        those you want to skip.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adafactor(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.36E+01

  References:
    Shazeer et al, `Adafactor: Adaptive Learning Rates with Sublinear Memory
    Cost <https://arxiv.org/abs/1804.04235>`_, 2018
  """
  # The core of the algorithm is a procedure for rescaling gradients
  # by a factored estimate of the root mean squared gradients.
  # This reduces memory compared to algorithms such as Adam or RmsProp,
  # by not having to hold a separate estimate for each weight.
  tx = [
      factorized.scale_by_factored_rms(
          factored, decay_rate, decay_offset, min_dim_size_to_factor, eps
      )
  ]
  # This basic rescaling is typically combined with one or more of the following
  # transformation (all can be disabled via adafactor's constructor args).
  if clipping_threshold is not None:
    tx.append(clipping.clip_by_block_rms(clipping_threshold))
  if learning_rate is not None:
    tx.append(transform.scale_by_learning_rate(learning_rate, flip_sign=False))
  if multiply_by_parameter_scale:
    tx.append(transform.scale_by_param_block_rms())
  if momentum is not None:
    tx.append(
        transform.ema(momentum, debias=False, accumulator_dtype=dtype_momentum)
    )
  if weight_decay_rate is not None:
    tx.append(
        transform.add_decayed_weights(weight_decay_rate, mask=weight_decay_mask)
    )
  # In gradient "descent" we follow the negative gradient.
  tx.append(transform.scale(-1))
  return combine.chain(*tx)


def adagrad(
    learning_rate: base.ScalarOrSchedule,
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-7,
) -> base.GradientTransformation:
  r"""The Adagrad optimizer.

  AdaGrad is a sub-gradient algorithm for stochastic optimization that adapts
  the learning rate individually for each feature based on its gradient history.

  The updated parameters adopt the form:

      .. math::

        w_{t+1}^{(i)} = w_{t}^{(i)} - \eta \frac{g_{t}^{(i)}}
                     {\sqrt{\sum_{\tau=1}^{t} (g_{\tau}^{(i)})^2 + \epsilon}}

  where:
    - :math:`w_t^{(i)}` is the parameter :math:`i` at time step :math:`t`,
    - :math:`\eta` is the learning rate,
    - :math:`g_t^{(i)}` is the gradient of parameter :math:`i` at time step
      :math:`t`,
    - :math:`\epsilon` is a small constant to ensure numerical stability.

  Defining :math:`G = \sum_{t=1}^\tau g_t g_t^\top`, the update can be
  written as

      .. math::

          w_{t+1} = w_{t} - \eta \cdot \text{diag}(G + \epsilon I)^{-1/2}
          \cdot g_t

  where :math:`\text{diag} (G) = (G_{ii})_{i=1}^p` is the vector of diagonal
  entries of :math:`G \in \mathbb{R}^p` and :math:`I` is the identity matrix
  in :math:`\mathbb{R}^p`.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    initial_accumulator_value: Initial value for the accumulator.
    eps: A small constant applied to denominator inside of the square root (as
      in RMSProp) to avoid dividing by zero when rescaling.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adagrad(learning_rate=1.0)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 5.01E+00
    Objective function: 2.40E+00
    Objective function: 1.25E+00
    Objective function: 6.86E-01
    Objective function: 3.85E-01

  References:
    Duchi et al, `Adaptive Subgradient Methods for Online Learning and
    Stochastic Optimization <https://jmlr.org/papers/v12/duchi11a.html>`_,
    2011

  .. warning::
    Adagrad's main limit is the monotonic accumulation of squared
    gradients in the denominator: since all terms are >0, the sum keeps growing
    during training and the learning rate eventually becomes vanishingly small.
  """
  return combine.chain(
      transform.scale_by_rss(
          initial_accumulator_value=initial_accumulator_value, eps=eps
      ),
      transform.scale_by_learning_rate(learning_rate),
  )


def adam(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
  r"""The Adam optimizer.

  Adam is an SGD variant with gradient scaling adaptation. The scaling
  used for each parameter is computed from estimates of first and second-order
  moments of the gradients (using suitable exponential moving averages).

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\varepsilon`, :math:`\bar{\varepsilon}` represent the arguments
  ``b1``, ``b2``, ``eps`` and ``eps_root`` respectively. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0, v_0) = (0, 0)`, representing initial estimates for the
  first and second moments. In practice these values are stored as pytrees
  containing all zeros, with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t` and optimizer state :math:`S_t`
  and computes updates :math:`u_t` and new state :math:`S_{t+1}`. Thus, for
  :math:`t > 0`, we have,

  .. math::

    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      v_t &\leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot {g_t}^2 \\
      \hat{m}_t &\leftarrow m_t / {(1-\beta_1^t)} \\
      \hat{v}_t &\leftarrow v_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow -\alpha_t \cdot \hat{m}_t / \left({\sqrt{\hat{v}_t +
      \bar{\varepsilon}} + \varepsilon} \right)\\
      S_t &\leftarrow (m_t, v_t).
    \end{align*}

  With the keyword argument `nesterov=True`, the optimizer uses Nesterov
  momentum, replacing the above :math:`\hat{m}_t` with

  .. math::
      \hat{m}_t \leftarrow
        \beta_1 m_t / {(1-\beta_1^{t+1})} + (1 - \beta_1) g_t / {(1-\beta_1^t)}.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      example when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    nesterov: Whether to use Nesterov momentum. The solver with
      nesterov=True is equivalent to the :func:`optax.nadam` optimizer, and
      described in [Dozat 2016].

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adam(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Kingma et al, `Adam: A Method for Stochastic Optimization
    <https://arxiv.org/abs/1412.6980>`_, 2014

    Dozat, `Incorporating Nesterov Momentum into Adam
    <https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ>`_, 2016

  .. warning::
    PyTorch and optax's implementation follow Algorithm 1 of [Kingma et al.
    2014]. Note that TensorFlow used instead the formulation just before Section
    2.1 of the paper. See https://github.com/deepmind/optax/issues/571 for more
    detail.

  .. seealso:: :func:`optax.nadam`, :func:`optax.adamw`.
  """
  return combine.chain(
      transform.scale_by_adam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )


nadam = functools.partial(adam, nesterov=True)
nadam.__doc__ = r"""The NAdam optimizer.

  Nadam is a variant of :func:`optax.adam` with Nesterov's momentum. The update
  rule of this solver is as follows:

  .. math::

    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      v_t &\leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot {g_t}^2 \\
      \hat{m}_t &\leftarrow
      \beta_1 m_t / {(1-\beta_1^{t+1})} + (1 - \beta_1) g_t / {(1-\beta_1^t)}\\
      \hat{v}_t &\leftarrow v_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow -\alpha_t \cdot \hat{m}_t / \left({\sqrt{\hat{v}_t +
      \bar{\varepsilon}} + \varepsilon} \right)\\
      S_t &\leftarrow (m_t, v_t).
    \end{align*}

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      example when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp
      >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
      >>> solver = optax.nadam(learning_rate=0.003)
      >>> params = jnp.array([1., 2., 3.])
      >>> print('Objective function: ', f(params))
      Objective function:  14.0
      >>> opt_state = solver.init(params)
      >>> for _ in range(5):
      ...  grad = jax.grad(f)(params)
      ...  updates, opt_state = solver.update(grad, opt_state, params)
      ...  params = optax.apply_updates(params, updates)
      ...  print('Objective function: {:.2E}'.format(f(params)))
      Objective function: 1.39E+01
      Objective function: 1.39E+01
      Objective function: 1.39E+01
      Objective function: 1.38E+01
      Objective function: 1.38E+01

  References:
    Dozat, `Incorporating Nesterov Momentum into Adam
    <https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ>`_, 2016

  .. seealso:: :func:`optax.adam`, :func:`optax.nadamw`.

  .. versionadded:: 0.1.9
"""


def adamw(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
  r"""Adam with weight decay regularization.

  AdamW uses weight decay to regularize learning towards small weights, as
  this leads to better generalization. In SGD you can also use L2 regularization
  to implement this as an additive loss term, however L2 regularization
  does not behave as intended for adaptive gradient algorithms such as Adam,
  see [Loshchilov et al, 2019].

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\varepsilon`, :math:`\bar{\varepsilon}` represent the arguments
  ``b1``, ``b2``, ``eps`` and ``eps_root`` respectively. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function. Let :math:`\lambda` be the weight decay and 
  :math:`\theta_t` the parameter vector at time :math:`t`.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0, v_0) = (0, 0)`, representing initial estimates for the
  first and second moments. In practice these values are stored as pytrees
  containing all zeros, with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t`, the optimizer state :math:`S_t` 
  and the parameters :math:`\theta_t` and computes updates :math:`u_t` and 
  new state :math:`S_{t+1}`. Thus, for :math:`t > 0`, we have,

  .. math::

    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      v_t &\leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot {g_t}^2 \\
      \hat{m}_t &\leftarrow m_t / {(1-\beta_1^t)} \\
      \hat{v}_t &\leftarrow v_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow -\alpha_t \cdot \left( \hat{m}_t / \left({\sqrt{\hat{v}_t 
      + \bar{\varepsilon}} + \varepsilon} \right) + \lambda \theta_{t} \right)\\
      S_t &\leftarrow (m_t, v_t).
    \end{align*}

  This implementation can incorporate a momentum a la Nesterov introduced by
  [Dozat 2016]. The resulting optimizer is then often referred as NAdamW.
  With the keyword argument `nesterov=True`, the optimizer uses Nesterov
  momentum, replacing the above :math:`\hat{m}_t` with

  .. math::
      \hat{m}_t \leftarrow
        \beta_1 m_t / {(1-\beta_1^{t+1})} + (1 - \beta_1) g_t / {(1-\beta_1^t)}.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
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
    nesterov: Whether to use Nesterov momentum. The solver with
      nesterov=True is equivalent to the :func:`optax.nadamw` optimizer. This
      modification is described in [Dozat 2016].

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adamw(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Loshchilov et al, `Decoupled Weight Decay
    Regularization <https://arxiv.org/abs/1711.05101>`_, 2019

    Dozat, `Incorporating Nesterov Momentum into Adam
    <https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ>`_, 2016

  .. seealso::
    See the related functions :func:`optax.adam`, :func:`optax.nadamw`, as well
    as the example :doc:`../_collections/examples/nanolm` for a use case.
  """
  return combine.chain(
      transform.scale_by_adam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
      ),
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )


nadamw = functools.partial(adamw, nesterov=True)
nadamw.__doc__ = (
    r"""NAdamW optimizer, implemented as part of the AdamW optimizer.

  NadamW is variant of :func:`optax.adamw` with Nesterov's momentum. Compared
  to AdamW, this optimizer replaces the assignment

  .. math::

      \hat{m}_t \leftarrow m_t / {(1-\beta_1^t)}

  with

  .. math::

      \hat{m}_t \leftarrow
        \beta_1 m_t / {(1-\beta_1^{t+1})} + (1 - \beta_1) g_t / {(1-\beta_1^t)}.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
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
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.nadamw(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.38E+01

  References:
    Loshchilov et al, `Decoupled Weight Decay 
    Regularization <https://arxiv.org/abs/1711.05101>`_, 2019

    Dozat, `Incorporating Nesterov Momentum into Adam
    <https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ>`_, 2016

  .. seealso:: :func:`optax.adam`, :func:`optax.adamw`.

  .. versionadded:: 0.1.9
"""
)


def adan(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.98,
    b2: float = 0.92,
    b3: float = 0.99,
    eps: float = 1e-8,
    eps_root: float = 1e-8,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  r"""The ADAptive Nesterov momentum algorithm (Adan).

  Adan first reformulates the vanilla Nesterov acceleration to develop a new
  Nesterov momentum estimation (NME) method, which avoids the extra overhead of
  computing gradient at the extrapolation point. Then Adan adopts NME to
  estimate the gradient's first- and second-order moments in adaptive gradient
  algorithms for convergence acceleration.

  The algorithm is as follows. First, we define the following parameters:

  - :math:`\eta > 0`: the step size.
  - :math:`\beta_1 \in [0, 1]`: the decay rate for the exponentially weighted
    average of gradients.
  - :math:`\beta_2 \in [0, 1]`: the decay rate for the exponentially weighted
    average of differences of gradients.
  - :math:`\beta_3 \in [0, 1]`: the decay rate for the exponentially weighted
    average of the squared term.
  - :math:`\varepsilon > 0`: a small constant for numerical stability.
  - :math:`\lambda > 0`: a weight decay.

  Second, we define the following variables:

  - :math:`\theta_t`: the parameters.
  - :math:`g_t`: the incoming stochastic gradient.
  - :math:`m_t`: the exponentially weighted average of gradients.
  - :math:`v_t`: the exponentially weighted average of differences of gradients.
  - :math:`n_t`: the exponentially weighted average of the squared term.
  - :math:`u_t`: the outgoing update vector.
  - :math:`S_t`: the saved state of the optimizer.

  Third, we initialize these variables as follows:

  - :math:`m_0 = g_0`
  - :math:`v_0 = 0`
  - :math:`v_1 = g_1 - g_0`
  - :math:`n_0 = g_0^2`

  Finally, on each iteration, we update the variables as follows:

  .. math::

    \begin{align*}
      m_t &\gets (1 - \beta_1) m_{t-1} + \beta_1 g_t \\
      v_t &\gets (1 - \beta_2) v_{t-1} + \beta_2 (g_t - g_{t-1}) \\
      n_t &\gets (1 - \beta_3) n_{t-1} + \beta_3 (g_t + (1 - \beta_2)
        (g_t - g_{t-1}))^2 \\
      \eta_t &\gets \eta / ({\sqrt{n_t + \bar{\varepsilon}} + \varepsilon}) \\
      u_t &\gets (\theta_t - \eta_t \circ (m_t + (1 - \beta_2) v_t))
        / (1 + \lambda \eta) \\
      S_t &\leftarrow (m_t, v_t, n_t).
    \end{align*}

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: Decay rate for the EWMA of gradients.
    b2: Decay rate for the EWMA of differences of gradients.
    b3: Decay rate for the EMWA of the algorithm's squared term.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    weight_decay: Strength of the weight decay regularization.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip.

  Returns:
    the corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> f = lambda x: x @ x  # simple quadratic function
    >>> solver = optax.adan(learning_rate=1e-1)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.28E+01
    Objective function: 1.17E+01
    Objective function: 1.07E+01
    Objective function: 9.68E+00
    Objective function: 8.76E+00

  References:
    Xie et al, `Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing
    Deep Models
    <https://arxiv.org/abs/2208.06677>`_, 2022
  """
  return combine.chain(
      transform.scale_by_adan(
          b1=b1,
          b2=b2,
          b3=b3,
          eps=eps,
          eps_root=eps_root,
      ),
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )


def lion(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-3,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  r"""The Lion optimizer.

  Lion is discovered by symbolic program search. Unlike most adaptive optimizers
  such as AdamW, Lion only tracks momentum, making it more memory-efficient.
  The update of Lion is produced through the sign operation, resulting in a
  larger norm compared to updates produced by other optimizers such as SGD and
  AdamW. A suitable learning rate for Lion is typically 3-10x smaller than that
  for AdamW, the weight decay for Lion should be in turn 3-10x larger than that
  for AdamW to maintain a similar strength (lr * wd).

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  represent the arguments ``b1`` and ``b2`` respectively. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function. Let :math:`\lambda` be the weight decay and
  :math:`\theta_t` the parameter vector at time :math:`t`.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0) = (0)`, representing the intial estimate for the
  first moment. In practice these values are stored as pytrees
  containing all zeros, with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t`, the optimizer state :math:`S_t`
  and the parameters :math:`\theta_t` and computes updates :math:`u_t` and
  new state :math:`S_{t+1}`. Thus, for :math:`t > 0`, we have,

  .. math::

    \begin{align*}
      c_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      u_t &\leftarrow -\alpha_t \cdot \left( sign \left( c_t \right) + 
      \lambda \theta_{t} \right)\\
      m_t &\leftarrow \beta_2 \cdot m_{t-1} + (1-\beta_2) \cdot g_t \\
      S_t &\leftarrow (m_t).
    \end{align*}

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Rate to combine the momentum and the current gradient.
    b2: Exponential decay rate to track the momentum of past gradients.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent with
      other frameworks such as PyTorch, but different from (Loshchilov et al,
      2019) where the weight decay is only multiplied with the "schedule
      multiplier", but not the base learning rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree, or a
      Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, `True` for leaves/subtrees you want to apply the
      weight decay to, and `False` for those you want to skip. Note that the
      Adam gradient transformations are applied to all parameters.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.lion(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Chen et al, `Symbolic Discovery of Optimization Algorithms
    <https://arxiv.org/abs/2302.06675>`_, 2023
  """
  return combine.chain(
      transform.scale_by_lion(b1=b1, b2=b2, mu_dtype=mu_dtype),
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )


def amsgrad(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """The AMSGrad optimizer.

  The original Adam can fail to converge to the optimal solution in some cases.
  AMSGrad guarantees convergence by using a long-term memory of past gradients.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root (as
      in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.amsgrad(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Reddi et al, `On the Convergence of Adam and Beyond
    <https://openreview.net/forum?id=ryQu7f-RZ>`_, 2023
  """
  return combine.chain(
      transform.scale_by_amsgrad(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype
      ),
      transform.scale_by_learning_rate(learning_rate),
  )


def fromage(
    learning_rate: float, min_norm: float = 1e-6
) -> base.GradientTransformation:
  """The Frobenius matched gradient descent (Fromage) optimizer.

  Fromage is a learning algorithm that does not require learning rate tuning.
  The optimizer is based on modeling neural network gradients via deep relative
  trust (a distance function on deep neural networks). Fromage is similar to the
  LARS optimizer and can work on a range of standard neural network benchmarks,
  such as natural language Transformers and generative adversarial networks.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    min_norm: A minimum value that the norm of the gradient updates and the norm
      of the layer parameters can be clipped to to avoid dividing by zero when
      computing the trust ratio (as in the LARS paper).

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.fromage(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.37E+01
    Objective function: 1.36E+01

  References:
    Bernstein et al, `On the distance between two neural networks and the
    stability of learning <https://arxiv.org/abs/2002.03432>`_, 2020
  """
  mult = 1 / jnp.sqrt(1 + learning_rate**2)
  return combine.chain(
      transform.scale_by_trust_ratio(min_norm),
      transform.scale_by_learning_rate(learning_rate * mult),
      transform.add_decayed_weights((mult - 1)),
  )


def lars(
    learning_rate: base.ScalarOrSchedule,
    weight_decay: float = 0.0,
    weight_decay_mask: MaskOrFn = True,
    trust_coefficient: float = 0.001,
    eps: float = 0.0,
    trust_ratio_mask: MaskOrFn = True,
    momentum: float = 0.9,
    nesterov: bool = False,
) -> base.GradientTransformation:
  """The LARS optimizer.

  LARS is a layer-wise adaptive optimizer introduced to help scale SGD to
  larger batch sizes. LARS later inspired the LAMB optimizer.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    weight_decay: Strength of the weight decay regularization.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.
    trust_coefficient: A multiplier for the trust ratio.
    eps: Optional additive constant in the trust ratio denominator.
    trust_ratio_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.
    momentum: Decay rate for momentum.
    nesterov: Whether to use Nesterov momentum.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.lars(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.40E+01
    Objective function: 1.40E+01
    Objective function: 1.40E+01
    Objective function: 1.40E+01

  References:
    You et al, `Large Batch Training of Convolutional Networks
    <https://arxiv.org/abs/1708.03888>`_, 2017
  """
  return combine.chain(
      transform.add_decayed_weights(weight_decay, mask=weight_decay_mask),
      wrappers.masked(
          inner=transform.scale_by_trust_ratio(
              trust_coefficient=trust_coefficient, eps=eps
          ),
          mask=trust_ratio_mask,
      ),
      transform.scale_by_learning_rate(learning_rate),
      transform.trace(decay=momentum, nesterov=nesterov),
  )


def lamb(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    mask: MaskOrFn = None,
) -> base.GradientTransformation:
  """The LAMB optimizer.

  LAMB is a general purpose layer-wise adaptive large batch optimizer designed
  to provide consistent training performance across a wide range of tasks,
  including those that use attention-based models (such as Transformers) and
  ResNet-50. The optimizer is able to work with small and large batch sizes.
  LAMB was inspired by the LARS learning algorithm.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root (as
      in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    weight_decay: Strength of the weight decay regularization.
    mask: A tree with same structure as (or a prefix of) the params PyTree, or a
      Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, `True` for leaves/subtrees you want to apply the
      transformation to, and `False` for those you want to skip.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.lamb(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.36E+01

  References:
    You et al, `Large Batch Optimization for Deep Learning: Training BERT in 76
    minutes <https://arxiv.org/abs/1904.00962>`_, 2020
  """
  return combine.chain(
      transform.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
      transform.add_decayed_weights(weight_decay=weight_decay, mask=mask),
      transform.scale_by_trust_ratio(),
      transform.scale_by_learning_rate(learning_rate),
  )


def noisy_sgd(
    learning_rate: base.ScalarOrSchedule,
    eta: float = 0.01,
    gamma: float = 0.55,
    seed: int = 0,
) -> base.GradientTransformation:
  r"""A variant of SGD with added noise.

  Noisy SGD is a variant of :func:`optax.sgd` that incorporates Gaussian noise
  into the updates. It has been found that adding noise to the gradients can
  improve both the training error and the generalization error in very deep
  networks.

  The update :math:`u_t` is modified to include this noise as follows:

  .. math::
    u_t \leftarrow -\alpha_t (g_t + N(0, \sigma_t^2)),

  where :math:`N(0, \sigma_t^2)` represents Gaussian noise with zero mean and a
  variance of :math:`\sigma_t^2`.

  The variance of this noise decays over time according to the formula

  .. math::
    \sigma_t^2 = \frac{\eta}{(1+t)^\gamma},

  where :math:`\gamma` is the decay rate parameter ``gamma`` and :math:`\eta`
  represents the initial variance ``eta``.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    eta: Initial variance for the Gaussian noise added to gradients.
    gamma: A parameter controlling the annealing of noise over time ``t``, the
      variance decays according to ``(1+t)**(-gamma)``.
    seed: Seed for the pseudo-random generation process.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.noisy_sgd(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.35E+01
    Objective function: 1.33E+01
    Objective function: 1.32E+01

  References:
    Neelakantan et al, `Adding Gradient Noise Improves Learning for Very Deep
    Networks <https://arxiv.org/abs/1511.06807>`_, 2015
  """
  return combine.chain(
      transform.add_noise(eta, gamma, seed),
      transform.scale_by_learning_rate(learning_rate),
  )


def sign_sgd(
    learning_rate: base.ScalarOrSchedule,
) -> base.GradientTransformation:
  r"""A variant of SGD using only the signs of the gradient components.

  SignSGD is a variant of SGD that uses the signs of the gradient components in
  the update, not their actual values. The update :math:`u_t` is modified as
  follows:

  .. math::
    u_t \leftarrow -\alpha_t\, \text{sign}\,(g_t),

  for :math:`\alpha_t` a given learning rate at iteration :math:`t`, and
  :math:`\text{sign}\,(g_t)` the sign of each component of the gradient
  :math:`g_t`.

  SGD variants that use only the signs of the gradient update have historically
  been used since RProp, with modern forms including RMSProp, Adam, and Lion.
  SignSGD uses only the signs of the gradient update. SignSGD enables
  significant gradient compression, substantially reducing the bottleneck
  imposed by communicating gradients when distributing learning across multiple
  workers.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.sign_sgd(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Bernstein et al., `signSGD: Compressed optimization for Non-Convex Problems
    <https://arxiv.org/abs/1802.04434>`_, 2018

    Balles et al., `The Geometry of Sign Gradient Descent
    <https://arxiv.org/abs/2002.08056>`_, 2020
  """
  return combine.chain(
      transform.scale_by_sign(),
      transform.scale_by_learning_rate(learning_rate),
  )


def novograd(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.25,
    eps: float = 1e-6,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
) -> base.GradientTransformation:
  """NovoGrad optimizer.

  NovoGrad is more robust to the initial learning rate and
  weight initialization than other methods. For example,
  NovoGrad works well without LR warm-up, while other methods require it.
  NovoGrad performs exceptionally well for large batch training, e.g. it
  outperforms other methods for ResNet-50 for all batches up to 32K.
  In addition, NovoGrad requires half the memory compared to Adam.
  It was introduced together with Jasper ASR model.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: An exponential decay rate to track the first moment of past gradients.
    b2: An exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root (as
      in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    weight_decay: Strength of the weight decay regularization.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.novograd(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.37E+01

  References:
    Ginsburg et al, `Stochastic Gradient Methods with Layer-wise Adaptive
    Moments for Training of Deep Networks <https://arxiv.org/abs/1905.11286>`_,
    2019

    Li et al, `Jasper: An End-to-End Convolutional Neural Acoustic Model
    <https://arxiv.org/abs/1904.03288>`_, 2019
  """
  return combine.chain(
      transform.scale_by_novograd(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, weight_decay=weight_decay
      ),
      transform.scale_by_learning_rate(learning_rate),
  )


def optimistic_gradient_descent(
    learning_rate: base.ScalarOrSchedule,
    alpha: base.ScalarOrSchedule = 1.0,
    beta: base.ScalarOrSchedule = 1.0,
) -> base.GradientTransformation:
  """An Optimistic Gradient Descent optimizer.

  Optimistic gradient descent is an approximation of extra-gradient methods
  which require multiple gradient calls to compute the next update. It has
  strong formal guarantees for last-iterate convergence in min-max games, for
  which standard gradient descent can oscillate or even diverge.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    alpha: Coefficient for generalized OGD.
    beta: Coefficient for generalized OGD negative momentum.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.optimistic_gradient_descent(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.35E+01
    Objective function: 1.33E+01
    Objective function: 1.32E+01

  References:
    Mokhtari et al, `A Unified Analysis of Extra-gradient and
    Optimistic Gradient Methods for Saddle Point Problems: Proximal
    Point Approach <https://arxiv.org/abs/1901.08511v2>`_, 2019

  .. seealso::
    :doc:`../_collections/examples/ogda_example`
  """
  return combine.chain(
      transform.scale_by_optimistic_gradient(alpha=alpha, beta=beta),
      transform.scale_by_learning_rate(learning_rate),
  )


def optimistic_adam(
    learning_rate: base.ScalarOrSchedule,
    optimism: Optional[float] = None,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-08,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = True,
) -> base.GradientTransformation:
  r"""The Optimistic Adam optimizer.

  This is an optimistic version of the Adam optimizer. It addresses the issue
  of limit cycling behavior in training Generative Adversarial Networks and
  other saddle-point min-max problems.

  The algorithm is as follows. First, we define the following parameters:

  - :math:`\alpha_t`: the learning rate or stepsize at iteration :math:`t`.
  - :math:`o_t` the optimism rate at iteration :math:`t`.
  - :math:`\beta_1` the exponential decay rate for the first moment estimate.
  - :math:`\beta_2` the exponential decay rate for the second moment estimate.

  Second, we define the following variables:

  - :math:`g_t`: the incoming gradient.
  - :math:`m_t`: the biased first moment estimate.
  - :math:`v_t`: the biased second raw moment estimate.
  - :math:`\hat{m}_t`: the bias-corrected first moment estimate.
  - :math:`\hat{v}_t`: the bias-corrected second raw moment estimate.
  - :math:`r_t`: the signal-to-noise ratio (SNR) vector.
  - :math:`u_t`: the outgoing update vector.
  - :math:`S_t`: the state of the optimizer.

  Finally, on each iteration, the variables are updated as follows:

  .. math::

    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t - 1} + (1 - \beta_1) \cdot g_t \\
      v_t &\leftarrow \beta_2 \cdot v_{t - 1} + (1 - \beta_2) \cdot g_t^2 \\
      \hat{m}_t &\leftarrow m_t / {(1 - \beta_1^t)} \\
      \hat{v}_t &\leftarrow v_t / {(1 - \beta_2^t)} \\
      r_t &\leftarrow \hat{m}_t / \left({\sqrt{\hat{v}_t +
        \bar{\varepsilon}} + \varepsilon} \right) \\
      u_t &\leftarrow -\alpha_t r_t - o_t (r_t - r_{t - 1}) \\
      S_t &\leftarrow (m_t, v_t, r_t).
    \end{align*}

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    optimism: The amount of optimism to be applied. If None, defaults to
      learning_rate, as in the paper.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the second moment of the prediction error to
      improve numerical stability. If backpropagating gradients through the
      gradient transformation (e.g. for meta-learning), this must be non-zero.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    nesterov: Whether to use Nesterov momentum.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> from jax import numpy as jnp, lax
    >>> def f(x, y):
    ...  return x * y  # simple bilinear function
    >>> opt = optax.optimistic_adam(1e-2, 1.0)
    >>> def step(state, _):
    ...  params, opt_state = state
    ...  distance = jnp.hypot(*params)
    ...  grads = jax.grad(f, argnums=(0, 1))(*params)
    ...  grads = grads[0], -grads[1]
    ...  updates, opt_state = opt.update(grads, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  return (params, opt_state), distance
    >>> params = 1.0, 2.0
    >>> opt_state = opt.init(params)
    >>> _, distances = lax.scan(step, (params, opt_state), length=1025)
    >>> for i in range(6):
    ...  print(f"{distances[4**i]:.3f}")
    2.243
    2.195
    2.161
    2.055
    0.796
    0.001

  References:
    Daskalakis et al, `Training GANs with Optimism
    <https://arxiv.org/abs/1711.00141>`_, 2017

  .. seealso::
    :doc:`../_collections/examples/ogda_example`
  """
  if optimism is None:
    optimism = learning_rate
  return combine.chain(
      transform.scale_by_adam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
      ),
      transform.scale_by_optimistic_gradient(
          alpha=learning_rate,
          beta=optimism,
      ),
      transform.scale_by_learning_rate(1.0),  # flips sign of updates
  )


def radam(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    threshold: float = 5.0,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
  """The Rectified Adam optimizer.

  The adaptive learning rate in Adam has undesirably large variance in early
  stages of training, due to the limited number of training samples used to
  estimate the optimizer's statistics. Rectified Adam addresses this issue
  by analytically reducing the large variance.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root (as
      in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    threshold: Threshold for variance tractability.
    nesterov: Whether to use Nesterov momentum.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.radam(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.35E+01
    Objective function: 1.33E+01
    Objective function: 1.32E+01

  References:
    Liu et al, 2020: `On the Variance of the Adaptive Learning Rate and Beyond
    <https://arxiv.org/abs/1908.03265>`_, 2020
  """
  return combine.chain(
      transform.scale_by_radam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          threshold=threshold,
          nesterov=nesterov,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )


def rmsprop(
    learning_rate: base.ScalarOrSchedule,
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    eps_in_sqrt: bool = True,
    centered: bool = False,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    bias_correction: bool = False,
) -> base.GradientTransformation:
  r"""A flexible RMSProp optimizer.

  RMSProp is an SGD variant with learning rate adaptation. The `learning_rate`
  used for each weight is scaled by a suitable estimate of the magnitude of the
  gradients on previous steps. Several variants of RMSProp can be found
  in the literature. This alias provides an easy to configure RMSProp
  optimizer that can be used to switch between several of these variants.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    decay: Decay used to track the magnitude of previous gradients.
    eps: A small numerical constant to avoid dividing by zero when rescaling.
    initial_scale: Initial value of accumulators tracking the magnitude of
      previous updates. PyTorch uses `0`, TF1 uses `1`. When reproducing results
      from a paper, verify the value used by the authors.
    eps_in_sqrt: Whether to add ``eps`` in the square root of the denominator or
      outside the square root.
    centered: Whether the second moment or the variance of the past gradients is
      used to rescale the latest gradients.
    momentum: Decay rate used by the momentum term, when it is set to `None`,
      then momentum is not used at all.
    nesterov: Whether Nesterov momentum is used.
    bias_correction: Whether to apply bias correction to the estimates of the
      second moments (and first moment if ``centered=True``).

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.rmsprop(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.39E+01
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.37E+01
    Objective function: 1.36E+01

  References:
    Hinton, `Overview of mini-batch gradient descent`
    <www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_, 2012

    Graves, `Generating Sequences With Recurrent Neural Networks
    <https://arxiv.org/pdf/1308.0850v5>`_, 2014

    Ziyin, `LaProp: Separating Momentum and Adaptivity in Adam`
    <https://arxiv.org/pdf/2002.04839>`_, 2021

  .. warning::
    Default behavior of optax's RMSprop (``eps_in_sqrt=True``) differs from
    Pytorch's implementation and could impact performance.
    If ``eps_in_sqrt=True``, in the denominator, optax uses
    :math:`\sqrt{v + \epsilon}` in the denominator whereas PyTorch uses
    :math:`\sqrt{v} + \epsilon`.
    Using ``eps_in_sqrt=False`` in optax will match PyTorch's behavior.
    See
    https://github.com/google-deepmind/optax/issues/532 for more detail.
  """
  if centered:
    return combine.chain(
        transform.scale_by_stddev(
            decay=decay,
            eps=eps,
            initial_scale=initial_scale,
            eps_in_sqrt=eps_in_sqrt,
            bias_correction=bias_correction,
        ),
        transform.scale_by_learning_rate(learning_rate),
        (
            transform.trace(decay=momentum, nesterov=nesterov)
            if momentum is not None
            else base.identity()
        ),
    )
  return combine.chain(
      transform.scale_by_rms(
          decay=decay,
          eps=eps,
          initial_scale=initial_scale,
          eps_in_sqrt=eps_in_sqrt,
          bias_correction=bias_correction,
      ),
      transform.scale_by_learning_rate(learning_rate),
      (
          transform.trace(decay=momentum, nesterov=nesterov)
          if momentum is not None
          else base.identity()
      ),
  )


def sgd(
    learning_rate: base.ScalarOrSchedule,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  r"""A canonical Stochastic Gradient Descent optimizer.

  This implements stochastic gradient descent. It also includes support for
  momentum, and Nesterov acceleration, as these are standard practice when
  using stochastic gradient descent to train deep neural networks.


  The canonical stochastic gradient descent returns an update
  :math:`u_t` of the form

  .. math::
    u_t \leftarrow -\alpha_t g_t,

  where :math:`g_t` is the gradient of the objective (potentially preprocessed
  by other transformations) and :math:`\alpha_t` is the ``learning_rate`` at
  time :math:`t` (constant or selected by an :class:`optax.Schedule`).

  Stochastic gradient descent with momentum takes two possible forms.

  .. math::

    \begin{align*}
      m_t &\leftarrow g_t + \mu m_{t-1} \\
      u_t &\leftarrow \begin{cases}
        -\alpha_t m_t & \text{ if } \texttt{nesterov = False} \\
        -\alpha_t (g_t + \mu m_t) & \text{ if } \texttt{nesterov = True}
        \end{cases} \\
      S_t &\leftarrow m_t,
    \end{align*}

  where :math:`\mu` is the ``momentum`` parameter and :math:`S_t` is the state
  of the optimizer.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    momentum: Decay rate used by the momentum term, when it is set to ``None``,
      then momentum is not used at all.
    nesterov: Whether Nesterov momentum is used.
    accumulator_dtype: Optional ``dtype`` to be used for the accumulator; if
      ``None`` then the ``dtype`` is inferred from ``params`` and ``updates``.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.sgd(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.38E+01
    Objective function: 1.37E+01
    Objective function: 1.35E+01
    Objective function: 1.33E+01
    Objective function: 1.32E+01

  References:
    Sutskever et al, `On the importance of initialization and momentum in deep
    learning <http://proceedings.mlr.press/v28/sutskever13.pdf>`_, 2013
  """
  if momentum is not None:
    opt = transform.trace(
        decay=momentum,
        nesterov=nesterov,
        accumulator_dtype=accumulator_dtype,
    )
  else:
    opt = base.identity()
  return combine.chain(
      opt,
      transform.scale_by_learning_rate(learning_rate),
  )


def sm3(
    learning_rate: float, momentum: float = 0.9
) -> base.GradientTransformation:
  r"""The SM3 optimizer.

  SM3 (Square-root of Minima of Sums of Maxima of Squared-gradients Method) is a
  memory-efficient adaptive optimizer designed to decrease memory overhead when
  training very large models, such as the Transformer for machine translation,
  BERT for language modeling, and AmoebaNet-D for image classification. SM3: 1)
  applies to tensors of arbitrary dimensions and any predefined cover of the
  parameters; 2) adapts the learning rates in an adaptive and data-driven manner
  (like Adagrad and unlike Adafactor); and 3) comes with rigorous convergence
  guarantees in stochastic convex optimization settings.

  The init function of this optimizer initializes an internal state 
  :math:`S_0 := \{\mu_0, w_1\} = \{0, 0\}`, representing initial estimates for 
  the cumulative squared gradients and the weights. These values are stored as 
  pytrees containing all zeros, with the same shape as the model updates. At 
  step :math:`t`, the update function of this optimizer takes as arguments 
  the incoming gradients :math:`g_t` and optimizer state :math:`S_t` and 
  computes updates :math:`u_t` and new state :math:`S_{t+1}`. Thus, for 
  :math:`t > 0`, we have:

  SM3-I Algorithm

  .. math::

      \begin{array}{l}
      \text{parameters: learning rate } \eta \\
      \text{initialize } w_1 = 0; \forall r \in [k]: \mu_0(r) = 0 \\
      \text{for } t = 1, \ldots, T \text{ do} \\
      \quad \text{receive gradient } g_t = \nabla \ell_t(w_t) \\
      \quad \text{for } r = 1, \ldots, k \text{ do} \\
      \quad \quad \mu_t(r) \leftarrow \mu_{t-1}(r) + 
      \max_{j \in S_r} g_t^2(j) \\
      \quad \text{for } i = 1, \ldots, d \text{ do} \\
      \quad \quad \nu_t(i) \leftarrow \min_{r:S_r \ni i} \mu_t(r) \\
      \quad \quad w_{t+1}(i) \leftarrow w_t(i) - 
      \eta \frac{g_t(i)}{\sqrt{\nu_t(i)}} \\
      \quad \quad \text{with the convention that } 0/0 = 0
      \end{array}

  SM3-II Algorithm

  The SM3-II optimizer initializes with parameters like the learning rate 
  :math:\eta and weight :math:w_1. It updates weights iteratively using 
  gradients :math:g_t, adjusting each component with minimum accumulated 
  values :math:\nu'_t(i) and maintaining cumulative maximums :math:\mu'_t(r) 
  for subsets :math:S_r. SM3-II starts with an initial state 
  :math:S_0 := (m_0, s_0) set to zero, storing estimates for first and second 
  moments as pytrees matching model updates' shape

  .. math::

      \begin{array}{l}
      \text{parameters: learning rate } \eta \\
      \text{initialize } w_1 = 0; \forall r \in [k]: \mu'_0(r) = 0 \\
      \text{for } t = 1, \ldots, T \text{ do} \\
      \quad \text{receive gradient } g_t = \nabla \ell_t(w_t) \\
      \quad \text{initialize } \mu'_t(r) = 0 \text{ for all } r \in [k] \\
      \quad \text{for } i = 1, \ldots, d \text{ do} \\
      \quad \quad \nu'_t(i) \leftarrow \min_{r:S_r \ni i} 
      \mu'_{t-1}(r) + g_t^2(i) \\
      \quad \quad w_{t+1}(i) \leftarrow w_t(i) - 
      \eta \frac{g_t(i)}{\sqrt{\nu'_t(i)}} \\
      \quad \quad \text{with the convention that } 0/0 = 0 \\
      \quad \text{for all } r : S_r \ni i \text{ do} \\
      \quad \quad \mu'_t(r) \leftarrow \max\{\mu'_t(r), \nu'_t(i)\}
      \end{array}

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    momentum: Decay rate used by the momentum term (when it is not set to
      `None`, then momentum is not used at all).

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.sm3(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.40E+01
    Objective function: 1.40E+01
    Objective function: 1.40E+01
    Objective function: 1.40E+01

  References:
    Anil et al, `Memory-Efficient Adaptive Optimization
    <https://arxiv.org/abs/1901.11150>`_, 2019
  """
  return combine.chain(
      transform.scale_by_sm3(momentum),
      transform.scale(-learning_rate),
  )


def yogi(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-3,
) -> base.GradientTransformation:
  # pylint: disable=line-too-long
  """The Yogi optimizer.

  Yogi is an adaptive optimizer, which provides control in tuning the effective
  learning rate to prevent it from increasing. By doing so, it focuses on
  addressing the issues of convergence and generalization in exponential moving
  average-based adaptive methods (such as Adam and RMSprop). Yogi is a
  modification of Adam and uses the same parameters.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root (as
      in the Adam paper) to avoid dividing by zero when rescaling.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.yogi(learning_rate=0.002)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.40E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01

  References:
    Zaheer et al, `Adaptive Methods for Nonconvex Optimization
    <https://proceedings.neurips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf>`_,
    2018
  """
  # pylint: enable=line-too-long
  return combine.chain(
      transform.scale_by_yogi(b1=b1, b2=b2, eps=eps),
      transform.scale_by_learning_rate(learning_rate),
  )


def adamax(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> base.GradientTransformation:
  r"""A variant of the Adam optimizer that uses the infinity norm.

  AdaMax is a variant of the :func:`optax.adam` optimizer. By generalizing 
  Adam's :math:`L^2` norm to an :math:`L^p` norm and taking the limit as
  :math:`p \rightarrow \infty`, we obtain a simple and stable update rule.

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\varepsilon` represent the arguments
  ``b1``, ``b2`` and ``eps`` respectively. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0, v_0) = (0, 0)`, representing initial estimates for the
  first and second moments. In practice these values are stored as pytrees
  containing all zeros, with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t` and optimizer state :math:`S_t`
  and computes updates :math:`u_t` and new state :math:`S_{t+1}`. Thus, for
  :math:`t > 0`, we have,

  .. math::

    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      v_t &\leftarrow \max(\left| g_t \right| + \varepsilon, \beta_2 \cdot
      v_{t-1}) \\
      \hat{m}_t &\leftarrow m_t / (1-\beta_1^t) \\
      u_t &\leftarrow -\alpha_t \cdot \hat{m}_t / v_t \\
      S_t &\leftarrow (m_t, v_t).
    \end{align*}

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the maximum of past gradients.
    eps: A small constant applied to denominator to avoid dividing by zero when
      rescaling.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adamax(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Kingma et al, 2014: https://arxiv.org/abs/1412.6980

  .. seealso:: :func:`optax.adam`, :func:`optax.adamaxw`.
  """
  return combine.chain(
      transform.scale_by_adamax(
          b1=b1,
          b2=b2,
          eps=eps,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )


def adamaxw(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  """Adamax with weight decay regularization.

  AdamaxW uses weight decay to regularize learning towards small weights, as
  this leads to better generalization. In SGD you can also use L2 regularization
  to implement this as an additive loss term, however L2 regularization
  does not behave as intended for adaptive gradient algorithms such as Adam.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the maximum of past gradients.
    eps: A small constant applied to denominator to avoid dividing by zero when
      rescaling.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent with
      other frameworks such as PyTorch, but different from (Loshchilov et al,
      2019) where the weight decay is only multiplied with the "schedule
      multiplier", but not the base learning rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree, or a
      Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, `True` for leaves/subtrees you want to apply the
      weight decay to, and `False` for those you want to skip. Note that the
      Adamax gradient transformations are applied to all parameters.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adamaxw(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101

  .. warning::
    Sometimes you may want to skip weight decay for BatchNorm scale
    or for the bias parameters. You can use `optax.masked` to make your own
    AdamaxW variant where `additive_weight_decay` is applied only to a subset of
    `params`.

  .. seealso:: :func:`optax.adam`, :func:`optax.adamax`.
  """
  return combine.chain(
      transform.scale_by_adamax(b1=b1, b2=b2, eps=eps),
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )


def rprop(
    learning_rate: float,
    eta_minus: float = 0.5,
    eta_plus: float = 1.2,
    min_step_size: float = 1e-6,
    max_step_size: float = 50.0,
) -> base.GradientTransformation:
  """The Rprop optimizer.

  Rprop, short for resillient backpropogation, is a first order variant of
  gradient descent. It responds only to the sign of the gradient by increasing
  or decreasing the step size selected per parameter exponentially to speed up
  convergence and avoid oscillations.

  Args:
    learning_rate: The initial step size.
    eta_minus: Multiplicative factor for decreasing step size. This is applied
      when the gradient changes sign from one step to the next.
    eta_plus: Multiplicative factor for increasing step size. This is applied
      when the gradient has the same sign from one step to the next.
    min_step_size: Minimum allowed step size. Smaller steps will be clipped to
      this value.
    max_step_size: Maximum allowed step size. Larger steps will be clipped to
      this value.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.rprop(learning_rate=0.003)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.40E+01
    Objective function: 1.39E+01
    Objective function: 1.39E+01
    Objective function: 1.38E+01

  References:
    Riedmiller et al. `A direct adaptive method for faster backpropagation
    learning: the RPROP algorithm
    <https://ieeexplore.ieee.org/document/298623>`_, 1993

    Igel et al. `Empirical evaluation of the improved Rprop learning
    algorithms
    <https://www.sciencedirect.com/science/article/abs/pii/S0925231201007007>`_,
    2003
  """
  return combine.chain(
      transform.scale_by_rprop(
          learning_rate=learning_rate,
          eta_minus=eta_minus,
          eta_plus=eta_plus,
          min_step_size=min_step_size,
          max_step_size=max_step_size,
      ),
      transform.scale(-1.0),
  )


def polyak_sgd(
    max_learning_rate: float = 1.0,
    scaling: base.ScalarOrSchedule = 1.0,
    f_min: float = 0.0,
    eps: float = 0.0,
) -> base.GradientTransformationExtraArgs:
  r"""SGD with Polyak step-size.

  This solver implements the SGD with Polyak step size of (Loizou et al. 2021).
  It sets the step-size as

  .. math::
    s \min\left\{\frac{f(x) - f^\star}{\|\nabla f(x)\|^2 + \epsilon},
      \gamma_{\max}\right\}\,,

  where :math:`f` is the function from which a gradient is computed,
  :math:`\gamma_{\max}` is a maximal acceptable learning rate set  by
  ``max_learning_rate``, :math:`\epsilon` is a constant preventing division by
  zero set with ``eps``, :math:`s` scales the formula by ``scaling``, and
  :math:`f^\star` is a guess of the minimum value of the function set with
  ``f_min``.

  Args:
    max_learning_rate: a maximum step size to use (defaults to 1).
    scaling: A global scaling factor, either fixed or evolving along iterations
      with a scheduler (defaults to 1).
    f_min: a lower bound on the objective function (defaults to 0). Corresponds
      to :math:`f^\star` in the formula above.
    eps: a value to add in the denominator of the update (defaults to 0).

  Returns:
    A :class:`optax.GradientTransformationExtraArgs`, where the ``update``
    functiontakes an additional keyword argument ``value`` containing the
    current value of the objective function.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.polyak_sgd()
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  value, grad = jax.value_and_grad(f)(params)
    ...  params, opt_state = solver.update(grad, opt_state, params, value=value)
    ...  print('Objective function: ', f(params))
    Objective function:  3.5
    Objective function:  0.875
    Objective function:  0.21875
    Objective function:  0.0546875
    Objective function:  0.013671875

  References:
    Loizou et al. `Stochastic polyak step-size for SGD: An adaptive learning
    rate for fast convergence <https://arxiv.org/abs/2002.10542>`_, 2021

    Berrada et al., `Training neural networks for and by interpolation
    <https://arxiv.org/pdf/1906.05661.pdf>`_, 2020

  .. warning::
    This method requires knowledge of an approximate value of the of the
    objective function minimum, passed through the ``f_min`` argument.
    For models that interpolate the data, this can be set to 0 (default
    value).
    Failing to set an appropriate value for ``f_min`` can lead to
    divergence or convergence to a suboptimal solution.
  """
  return combine.chain(
      sgd(learning_rate=scaling),
      transform.scale_by_polyak(
          max_learning_rate=max_learning_rate, f_min=f_min, eps=eps
      ),
  )


def lbfgs(
    learning_rate: Optional[base.ScalarOrSchedule] = None,
    memory_size: int = 10,
    scale_init_precond: bool = True,
    linesearch: Optional[
        base.GradientTransformationExtraArgs
    ] = _linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=20, initial_guess_strategy='one'
    ),
) -> base.GradientTransformationExtraArgs:
  r"""L-BFGS optimizer.

  L-BFGS is a quasi-Newton method that multiplies the update (gradient)
  with an approximation of the inverse Hessian. This algorithm does not need
  access to the Hessian, as this approximation is constructed from the gradient
  evaluations seen during optimization. L-BFGS is a limited-memory variant of
  the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm. The BFGS algorithm
  requires storing a matrix of size :math:`p \times p` with :math:`p` the
  dimension of the parameters.
  The limited variant circuments this issue by computing the approximation of
  the inverse using only :math:`m` (``memory_size``) past differences of
  parameters/gradients. Namely, the approximation of the Hessian inverse is
  denoted :math:`P_k = P_{k, k}`, where

  .. math::

    \begin{align*}
      P_{k, j+1} & = V_j^\top P_{k, j} V_j + \rho_j \delta w_j \delta w_j^\top
      \quad \text{for} \ j \in \{k-m, \ldots, k-1\}\\
      P_{k, k-m} & = \gamma_k I \\
      V_k & = I - \rho_k \delta u_k \delta w_k^\top \\
      \rho_k & = 1/(\delta u_k^\top \delta w_k) \\
      \delta w_k & = w_{k+1} - w_k \\
      \delta u_k & = u_{k+1} - u_k \\
      \gamma_k & =
        \begin{cases}
          (\delta w_{k-1}^\top \delta u_{k-1}) /
          (\delta u_{k-1}^\top \delta u_{k-1})
          & \text{if} \ \texttt{scale\_init\_hess} \\
          1 & \text{otherwise}
        \end{cases},
    \end{align*}

  for
  :math:`u_k` the gradients/updates at iteration :math:`k`,
  :math:`w_k` the parameters at iteration :math:`k`.

  The formula for updating :math:`P_k` is obtained by computing the optimal
  preconditioning matrix subject to some secant condition, see references
  for more details. Computing :math:`P_k u_k` can be done by a sequence of
  vector operations using past differences of parameters and gradients stored in
  a memory bufffer.

  The present function just outputs the LBFGS direction :math:`P_k u_k`.
  It can be chained with a linesearch ensuring sufficient decrease and low
  curvature, such as a zoom linesearch. The linesearch computes a stepsize
  :math:`\eta_k`, such that the updated parameters
  (using :func:`optax.apply_updates`) take the form
  :math:`w_{k+1} = w_k - \eta_k P_k u_k`.

  Args:
    learning_rate: optional global scaling factor, either fixed or evolving
      along iterations with a scheduler, see 
      :func:`optax.scale_by_learning_rate`. By default the learning rate is
      handled by a linesearch.
    memory_size: number of past updates to keep in memory to approximate the
      Hessian inverse.
    scale_init_precond: whether to use a scaled identity as the initial
      preconditioner, see formula of :math:`\gamma_k` above.
    linesearch: an instance of :class:`optax.GradientTransformationExtraArgs`
      such as :func:`optax.scale_by_zoom_linesearch` that computes a
      learning rate, a.k.a. stepsize, to satisfy some criterion such as a
      sufficient decrease of the objective by additional calls to the objective.

  Returns:
    A :class:`optax.GradientTransformationExtraArgs` object.

  Example:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)
    >>> solver = optax.lbfgs()
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> value_and_grad = optax.value_and_grad_from_state(f)
    >>> for _ in range(5):
    ...   value, grad = value_and_grad(params, state=opt_state)
    ...   updates, opt_state = solver.update(
    ...      grad, opt_state, params, value=value, grad=grad, value_fn=f
    ...   )
    ...   params = optax.apply_updates(params, updates)
    ...   print('Objective function: ', f(params))
    Objective function:  7.5166864
    Objective function:  7.460699e-14
    Objective function:  2.6505726e-28
    Objective function:  0.0
    Objective function:  0.0

  References:
    Algorithms 7.4, 7.5 (page 199) of Nocedal et al, `Numerical Optimization
    <https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf>`_
    , 1999

    Liu et al., `On the limited memory BFGS method for large scale optimization
    <https://users.iems.northwestern.edu/~nocedal/PDFfiles/limited-memory.pdf>`_
    , 1989.

  .. warning::
    This method is memory intensive optimizer, best used for small to medium
    scale problems.

  .. warning::
    This optimizer works best with a linesearch (current default is a
    zoom linesearch). See example above for best use in a non-stochastic
    setting, where we can recycle gradients computed by the linesearch using
    :func:`optax.value_and_grad_from_state`.

  .. note::
    We initialize the scaling of the identity as a capped reciprocal of the
    gradient norm. This avoids wasting linesearch iterations for the first step
    by taking into account the magnitude of the gradients. In other words, we
    constrain the trust-region of the first step to an Euclidean ball of radius
    1 at the first iteration. The choice of :math:`\gamma_0` is not detailed in
    the references above, so this is a heuristic choice.
  """
  if learning_rate is None:
    base_scaling = transform.scale(-1.0)
  else:
    base_scaling = transform.scale_by_learning_rate(learning_rate)
  if linesearch is None:
    linesearch = base.identity()
  return combine.chain(
      transform.scale_by_lbfgs(
          memory_size=memory_size, scale_init_precond=scale_init_precond
      ),
      base_scaling,
      linesearch,
  )
