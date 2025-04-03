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
r"""Implementation of control variates.

We are interested in computing the gradient using control variates:
\nabla_{\theta} E_{p(x; \theta)} f(x)
  = \nabla_{\theta} [E_{p(x; \theta)} f(x) - h(x; \theta) + E_{p(x; \theta)}]
  = \nabla_{\theta} [E_{p(x; \theta)} f(x) - h(x; \theta)]
      + \nabla_{\theta} E_{p(x; \theta)}]
 = \nabla_{\theta} [E_{p(x; \theta)} f(x) - h(x; \theta)]
      + \nabla_{\theta} E_{p(x; \theta)}]
= \nabla_{\theta} \int {p(x; \theta)} (f(x) - h(x; \theta)) dx
     + \nabla_{\theta} E_{p(x; \theta)}]
=  \int \nabla_{\theta} {p(x; \theta)} (f(x) - h(x; \theta)) dx
     + [E_{p(x; \theta)} \nabla_{\theta} (f(x) - h(x; \theta))
     + \nabla_{\theta} E_{p(x; \theta)}]
=  \int \nabla_{\theta} {p(x; \theta)} (f(x) - h(x; \theta)) dx
     - [E_{p(x; \theta)} \nabla_{\theta} h(x; \theta)
     + \nabla_{\theta} E_{p(x; \theta)}]

The above computation is performed in `control_variates_jacobians`.

When adding a new control variate, one does not need to implement the jacobian
computation, but instead has to implement the forward computation.

Each control variate implemented has to satisfy the following API:
  * control_variate(function)
      This returns a tuple of three functions:
         * The first element of the tuple is a function which returns the
            control variate value for a set of samples. It takes in as
            arguments the parameters used to construct the distribution,
            the distributional samples, and the state of the control variate
            (if any). The return value of this function will have shape
            `num_samples`, where `num_samples` is the number of samples
            provided as input.
         * The second is a function returns the expected value of the control
            variate. The input arguments of this function are the parameters
            of the distribution and the state of the control variate.
         * The third is a function which updates the state of the control
            variate, and returns the updated states.

For examples, see `control_delta_method` and `moving_avg_baseline`.
"""
from collections.abc import Callable
from typing import Any, Sequence

import chex
import jax
import jax.numpy as jnp
from optax._src import base


CvState = Any
ComputeCv = Callable[[base.Params, chex.Array, CvState], chex.Array]
CvExpectedValue = Callable[[base.Params, CvState], CvState]
UpdateCvState = Callable[[base.Params, chex.Array, CvState], CvState]
ControlVariate = tuple[ComputeCv, CvExpectedValue, UpdateCvState]


@chex.warn_deprecated_function
def control_delta_method(
    function: Callable[[chex.Array], float],
) -> ControlVariate:
  """The control delta covariant method.

  Control variate obtained by performing a second order Taylor expansion
    on the cost function f at the mean of the input distribution.

  Only implemented for Gaussian random variables.

  For details, see: https://icml.cc/2012/papers/687.pdf

  Args:
    function: The function for which to compute the control variate. The
      function takes in one argument (a sample from the distribution) and
      returns a floating point value.

  Returns:
    A tuple of three functions, to compute the control variate, the
    expected value of the control variate, and to update the control variate
    state.

  .. deprecated:: 0.2.4
    This function will be removed in 0.3.0
  """

  def delta(
      params: base.Params, sample: chex.Array, state: CvState = None
  ) -> chex.Array:
    """Second order expansion of `function` at the mean of the input dist."""
    del state
    mean_dist = params[0]
    centered_sample = sample - mean_dist
    # Function is a function of samples. Here, we use the mean as the input
    # since we do a Taylor expansion of function around the mean.
    grads = jax.grad(function)(mean_dist)
    hessians = jax.hessian(function)(mean_dist)
    assert hessians.ndim == 2
    control_variate = function(mean_dist)
    control_variate += jnp.dot(centered_sample, grads)
    control_variate += (
        jnp.dot(jnp.dot(centered_sample, hessians), centered_sample) / 2.0
    )
    return control_variate

  def expected_value_delta(params: base.Params, state: CvState) -> jax.Array:
    """Expected value of second order expansion of `function` at dist mean."""
    del state
    mean_dist = params[0]
    var_dist = jnp.square(jnp.exp(params[1]))
    hessians = jax.hessian(function)(mean_dist)

    assert hessians.ndim == 2
    hess_diags = jnp.diag(hessians)
    assert hess_diags.ndim == 1

    # Trace (Hessian * Sigma) and we use that Sigma is diagonal.
    expected_second_order_term = jnp.sum(var_dist * hess_diags) / 2.0

    expected_control_variate = function(mean_dist)
    expected_control_variate += expected_second_order_term
    return expected_control_variate

  def update_state(
      params: base.Params, samples: chex.Array, state: CvState = None
  ) -> CvState:
    """No state kept, so no operation is done."""
    del params, samples
    return state

  return delta, expected_value_delta, update_state


@chex.warn_deprecated_function
def moving_avg_baseline(
    function: Callable[[chex.Array], float],
    decay: float = 0.99,
    zero_debias: bool = True,
    use_decay_early_training_heuristic=True,
) -> ControlVariate:
  """A moving average baseline.

  It has no effect on the pathwise or measure valued estimator.

  Args:
    function: The function for which to compute the control variate. The
      function takes in one argument (a sample from the distribution) and
      returns a floating point value.
    decay: The decay rate for the moving average.
    zero_debias: Whether or not to use zero debiasing for the moving average.
    use_decay_early_training_heuristic: Whether or not to use a heuristic which
      overrides the decay value early in training based on min(decay, (1.0 + i)
      / (10.0 + i)). This stabilizes training and was adapted from the
      Tensorflow codebase.

  Returns:
    A tuple of three functions, to compute the control variate, the
    expected value of the control variate, and to update the control variate
    state.

  .. deprecated:: 0.2.4
    This function will be removed in 0.3.0
  """

  def moving_avg(
      params: base.Params, samples: chex.Array, state: CvState = None
  ) -> CvState:
    """Return the moving average."""
    del params, samples
    return state[0]

  def expected_value_moving_avg(
      params: base.Params, state: CvState
  ) -> chex.Array:
    """Return the moving average."""
    del params
    return state[0]

  def update_state(
      params: base.Params, samples: chex.Array, state: CvState = None
  ) -> CvState:
    """Update the moving average."""
    del params
    value, i = state

    if use_decay_early_training_heuristic:
      iteration_decay = jnp.minimum(decay, (1.0 + i) / (10.0 + i))
    else:
      iteration_decay = decay

    updated_value = iteration_decay * value
    updated_value += (1 - iteration_decay) * jnp.mean(
        jax.vmap(function)(samples)
    )

    if zero_debias:
      updated_value /= jnp.ones([]) - jnp.power(iteration_decay, i + 1)

    return (jax.lax.stop_gradient(updated_value), i + 1)

  return moving_avg, expected_value_moving_avg, update_state


def _map(cv, params, samples, state):
  return jax.vmap(lambda x: cv(params, x, state))(samples)


@chex.warn_deprecated_function
def control_variates_jacobians(
    function: Callable[[chex.Array], float],
    control_variate_from_function: Callable[
        [Callable[[chex.Array], float]], ControlVariate
    ],
    grad_estimator: Callable[..., jnp.ndarray],
    params: base.Params,
    dist_builder: Callable[..., Any],
    rng: chex.PRNGKey,
    num_samples: int,
    control_variate_state: CvState = None,
    estimate_cv_coeffs: bool = False,
    estimate_cv_coeffs_num_samples: int = 20,
) -> tuple[Sequence[chex.Array], CvState]:
  r"""Obtain jacobians using control variates.

  We will compute each term individually. The first term will use stochastic
  gradient estimation. The second term will be computes using Monte
  Carlo estimation and automatic differentiation to compute
  \nabla_{\theta} h(x; \theta). The the third term will be computed using
  automatic differentiation, as we restrict ourselves to control variates
  which compute this expectation in closed form.

  This function updates the state of the control variate (once), before
  computing the control variate coefficients.

  Args:
    function: Function f(x) for which to estimate grads_{params} E_dist f(x).
      The function takes in one argument (a sample from the distribution) and
      returns a floating point value.
    control_variate_from_function: The control variate to use to reduce
      variance. See `control_delta_method` and `moving_avg_baseline` examples.
    grad_estimator: The gradient estimator to be used to compute the gradients.
      Note that not all control variates will reduce variance for all
      estimators. For example, the `moving_avg_baseline` will make no difference
      to the measure valued or pathwise estimators.
    params: A tuple of jnp arrays. The parameters for which to construct the
      distribution and for which we want to compute the jacobians.
    dist_builder: a constructor which builds a distribution given the input
      parameters specified by params. `dist_builder(params)` should return a
      valid distribution.
    rng: a PRNGKey key.
    num_samples: Int, the number of samples used to compute the grads.
    control_variate_state: The control variate state. This is used for control
      variates which keep states (such as the moving average baselines).
    estimate_cv_coeffs: Boolean. Whether or not to estimate the optimal control
      variate coefficient via `estimate_control_variate_coefficients`.
    estimate_cv_coeffs_num_samples: The number of samples to use to estimate the
      optimal coefficient. These need to be new samples to ensure that the
      objective is unbiased.

  Returns:
    A tuple of size two:

    * A tuple of size `params`, each element is `num_samples x param.shape`
      jacobian vector containing the estimates of the gradients obtained
      for each sample.
      The mean of this vector is the gradient wrt to parameters that can be
      used for learning. The entire jacobian vector can be used to assess
      estimator variance.
    * The updated CV state.

  .. deprecated:: 0.2.4
    This function will be removed in 0.3.0
  """
  control_variate = control_variate_from_function(function)
  stochastic_cv, expected_value_cv, update_state_cv = control_variate
  data_dim = jax.tree.leaves(params)[0].shape[0]
  if estimate_cv_coeffs:
    cv_coeffs = estimate_control_variate_coefficients(
        function,
        control_variate_from_function,
        grad_estimator,
        params,
        dist_builder,
        rng,
        estimate_cv_coeffs_num_samples,
        control_variate_state,
    )
  else:
    cv_coeffs = [1.0] * len(params)

  # \int \nabla_{\theta} {p(x; \theta)} (f(x) - h(x; \theta)) dx
  function_jacobians = grad_estimator(
      function, params, dist_builder, rng, num_samples
  )

  # Chain rule since CVs can also depend on parameters - for example, for the
  # pathwise gradient estimator they have in order to have an effect on
  # gradient.
  # The rng has to be the same as passed to the grad_estimator above so that we
  # obtain the same samples.
  samples = dist_builder(*params).sample((num_samples,), seed=rng)
  # If the CV has state, update it.
  control_variate_state = update_state_cv(
      params, samples, control_variate_state
  )

  def samples_fn(x):
    return stochastic_cv(
        jax.lax.stop_gradient(params), x, control_variate_state
    )

  cv_jacobians = grad_estimator(
      samples_fn, params, dist_builder, rng, num_samples
  )

  # The gradients of the stochastic covariant with respect to the parameters.
  def param_fn(x):
    return jnp.mean(
        _map(
            stochastic_cv,
            x,
            jax.lax.stop_gradient(samples),
            control_variate_state,
        )
    )

  # [E_{p(x; \theta)} \nabla_{\theta} h(x; \theta)
  cv_param_grads = jax.grad(param_fn)(params)
  # The gradients of the closed form expectation of the control variate
  # with respect to the parameters: # \nabla_{\theta} E_{p(x; \theta)}].
  expected_value_grads = jax.grad(
      lambda x: expected_value_cv(x, control_variate_state)
  )(params)

  jacobians = []
  for param_index, param in enumerate(jax.tree.leaves(params)):
    chex.assert_shape(function_jacobians[param_index], (num_samples, data_dim))
    chex.assert_shape(cv_jacobians[param_index], (num_samples, data_dim))
    chex.assert_shape(cv_param_grads[param_index], (data_dim,))
    chex.assert_shape(expected_value_grads[param_index], (data_dim,))

    cv_coeff = cv_coeffs[param_index]
    # \int \nabla_{\theta} {p(x; \theta)} (f(x) - h(x; \theta)) dx
    param_jacobians = function_jacobians[param_index]
    param_jacobians -= cv_coeff * cv_jacobians[param_index]
    # - [E_{p(x; \theta)} \nabla_{\theta} h(x; \theta)
    param_jacobians -= cv_coeff * cv_param_grads[param_index]
    # \nabla_{\theta} E_{p(x; \theta)}]
    param_jacobians += cv_coeff * expected_value_grads[param_index]

    chex.assert_shape(param_jacobians, (num_samples,) + param.shape)
    jacobians.append(param_jacobians)

  return jacobians, control_variate_state


@chex.warn_deprecated_function
def estimate_control_variate_coefficients(
    function: Callable[[chex.Array], float],
    control_variate_from_function: Callable[
        [Callable[[chex.Array], float]], ControlVariate
    ],
    grad_estimator: Callable[..., jnp.ndarray],
    params: base.Params,
    dist_builder: Callable[..., Any],
    rng: chex.PRNGKey,
    num_samples: int,
    control_variate_state: CvState = None,
    eps: float = 1e-3,
) -> Sequence[float]:
  r"""Estimates the control variate coefficients for the given parameters.

  For each variable `var_k`, the coefficient is given by:
    \sum_k cov(df/d var_k, d cv/d var_k) / (\sum var(d cv/d var_k) + eps)

  Where var_k is the k'th element of the parameters in `params`.
  The covariance and variance calculations are done from samples obtained
  from the distribution obtained by calling `dist_builder` on the input
  `params`.

  This function does not update the state of the control variate.

  Args:
    function: Function f(x) for which to estimate grads_{params} E_dist f(x).
      The function takes in one argument (a sample from the distribution) and
      returns a floating point value.
    control_variate_from_function: The control variate to use to reduce
      variance. See `control_delta_method` and `moving_avg_baseline` examples.
    grad_estimator: The gradient estimator to be used to compute the gradients.
      Note that not all control variates will reduce variance for all
      estimators. For example, the `moving_avg_baseline` will make no difference
      to the measure valued or pathwise estimators.
    params: A tuple of jnp arrays. The parameters for which to construct the
      distribution and for which we want to compute the jacobians.
    dist_builder: a constructor which builds a distribution given the input
      parameters specified by params. `dist_builder(params)` should return a
      valid distribution.
    rng: a PRNGKey key.
    num_samples: Int, the number of samples used to compute the grads.
    control_variate_state: The control variate state. This is used for control
      variates which keep states (such as the moving average baselines).
    eps: A small constant used to avoid numerical issues. Float.

  Returns:
    A list of control variate coefficients (each a scalar), for each parameter
    in `params`.

  .. deprecated:: 0.2.4
    This function will be removed in 0.3.0
  """
  # Resample to avoid biased gradients.
  cv_rng, _ = jax.random.split(rng)
  del rng  # Avoid using rng in this function.
  stochastic_cv, _, _ = control_variate_from_function(function)

  # Samples have to be the same so we use the same rng.
  cv_jacobians = grad_estimator(
      lambda x: stochastic_cv(params, x, control_variate_state),
      params,
      dist_builder,
      cv_rng,
      num_samples,
  )
  function_jacobians = grad_estimator(
      function, params, dist_builder, cv_rng, num_samples
  )

  def compute_coeff(param_cv_jacs, param_f_jacs):
    assert param_f_jacs.ndim == 2
    assert param_cv_jacs.ndim == 2

    mean_f = jnp.mean(param_f_jacs, axis=0)
    mean_cv = jnp.mean(param_cv_jacs, axis=0)

    cov = jnp.mean((param_f_jacs - mean_f) * (param_cv_jacs - mean_cv), axis=0)

    assert cov.ndim == 1

    # Compute the coefficients which minimize variance.
    # Since we want to minimize the variances across parameter dimensions,
    # the optimal coefficients are given by the sum of covariances per
    # dimensions over the sum of variances per dimension.
    cv_coeff = jnp.sum(cov) / (jnp.sum(jnp.var(param_cv_jacs, axis=0)) + eps)
    return jax.lax.stop_gradient(cv_coeff)

  return [
      compute_coeff(cv_jacobians[i], function_jacobians[i])
      for i in range(len(params))
  ]
