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
r"""Stochastic Monte Carlo gradient estimators.

Utility functions to approximate gradients of the form using Monte Carlo
estimation:
  \nabla_{\theta} E_{p(x; \theta)} f(x)

Here f is assumed to have no dependence on the parameters theta - if f has
dependence on theta, the functions below need to be called with `stop_grad(f)`
and the chain rule needs to be applied outside these functions in order
to obtain unbiased gradient.

For more details, see:
S. Mohamed, M. Rosca, M. Figurnov, A Mnih.
  Monte Carlo Gradient Estimation in Machine Learning. JMLR, 2020.
"""

from collections.abc import Callable
import math
from typing import Any, Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import base
from optax._src import utils


@chex.warn_deprecated_function
def score_function_jacobians(
    function: Callable[[chex.Array], float],
    params: base.Params,
    dist_builder: Callable[..., Any],
    rng: chex.PRNGKey,
    num_samples: int,
) -> Sequence[chex.Array]:
  r"""Score function gradient estimation.

  Approximates:
     \nabla_{\theta} E_{p(x; \theta)} f(x)
  With:
    E_{p(x; \theta)} f(x) \nabla_{\theta} \log p(x; \theta)

  Requires: p to be differentiable wrt to theta. Applicable to both continuous
    and discrete random variables. No requirements on f.

  Args:
    function: Function f(x) for which to estimate grads_{params} E_dist f(x).
      The function takes in one argument (a sample from the distribution) and
      returns a floating point value.
    params: A tuple of jnp arrays. The parameters for which to construct the
      distribution.
    dist_builder: a constructor which builds a distribution given the input
      parameters specified by params. `dist_builder(params)` should return a
      valid distribution.
    rng: a PRNGKey key.
    num_samples: Int, the number of samples used to compute the grads.

  Returns:
    A tuple of size `params`, each element is `num_samples x param.shape`
      jacobian vector containing the estimates of the gradients obtained for
      each sample.
    The mean of this vector is the gradient wrt to parameters that can be used
      for learning. The entire jacobian vector can be used to assess estimator
      variance.

  .. deprecated:: 0.2.4
    This function will be removed in 0.3.0
  """

  def surrogate(params):
    dist = dist_builder(*params)
    one_sample_surrogate_fn = lambda x: function(x) * dist.log_prob(x)
    samples = jax.lax.stop_gradient(dist.sample((num_samples,), seed=rng))
    # We vmap the function application over samples - this ensures that the
    # function we use does not have to be vectorized itself.
    return jax.vmap(one_sample_surrogate_fn)(samples)

  return jax.jacfwd(surrogate)(params)


@chex.warn_deprecated_function
def pathwise_jacobians(
    function: Callable[[chex.Array], float],
    params: base.Params,
    dist_builder: Callable[..., Any],
    rng: chex.PRNGKey,
    num_samples: int,
) -> Sequence[chex.Array]:
  r"""Pathwise gradient estimation.

  Approximates:
     \nabla_{\theta} E_{p(x; \theta)} f(x)
  With:
    E_{p(\epsilon)} \nabla_{\theta} f(g(\epsilon, \theta))
      where x = g(\epsilon, \theta). g depends on the distribution p.

  Requires: p to be reparametrizable and the reparametrization to be implemented
    in tensorflow_probability. Applicable to continuous random variables.
    f needs to be differentiable.

  Args:
    function: Function f(x) for which to estimate grads_{params} E_dist f(x).
      The function takes in one argument (a sample from the distribution) and
      returns a floating point value.
    params: A tuple of jnp arrays. The parameters for which to construct the
      distribution.
    dist_builder: a constructor which builds a distribution given the input
      parameters specified by params. `dist_builder(params)` should return a
      valid distribution.
    rng: a PRNGKey key.
    num_samples: Int, the number of samples used to compute the grads.

  Returns:
    A tuple of size `params`, each element is `num_samples x param.shape`
      jacobian vector containing the estimates of the gradients obtained for
      each sample.
    The mean of this vector is the gradient wrt to parameters that can be used
      for learning. The entire jacobian vector can be used to assess estimator
      variance.

  .. deprecated:: 0.2.4
    This function will be removed in 0.3.0
  """

  def surrogate(params):
    # We vmap the function application over samples - this ensures that the
    # function we use does not have to be vectorized itself.
    dist = dist_builder(*params)
    return jax.vmap(function)(dist.sample((num_samples,), seed=rng))

  return jax.jacfwd(surrogate)(params)


@chex.warn_deprecated_function
def measure_valued_jacobians(
    function: Callable[[chex.Array], float],
    params: base.Params,
    dist_builder: Callable[..., Any],
    rng: chex.PRNGKey,
    num_samples: int,
    coupling: bool = True,
) -> Sequence[chex.Array]:
  r"""Measure valued gradient estimation.

  Approximates:
     \nabla_{\theta} E_{p(x; \theta)} f(x)
  With:
    1./ c (E_{p1(x; \theta)} f(x) - E_{p2(x; \theta)} f(x)) where p1 and p2 are
    measures which depend on p.

  Currently only supports computing gradients of expectations of Gaussian RVs.

  Args:
    function: Function f(x) for which to estimate grads_{params} E_dist f(x).
      The function takes in one argument (a sample from the distribution) and
      returns a floating point value.
    params: A tuple of jnp arrays. The parameters for which to construct the
      distribution.
    dist_builder: a constructor which builds a distribution given the input
      parameters specified by params. `dist_builder(params)` should return a
      valid distribution.
    rng: a PRNGKey key.
    num_samples: Int, the number of samples used to compute the grads.
    coupling: A boolean. Whether or not to use coupling for the positive and
      negative samples. Recommended: True, as this reduces variance.

  Returns:
    A tuple of size `params`, each element is `num_samples x param.shape`
      jacobian vector containing the estimates of the gradients obtained for
      each sample.
    The mean of this vector is the gradient wrt to parameters that can be used
      for learning. The entire jacobian vector can be used to assess estimator
      variance.

  .. deprecated:: 0.2.4
    This function will be removed in 0.3.0
  """
  if dist_builder is not utils.multi_normal:
    raise ValueError(
        'Unsupported distribution builder for measure_valued_jacobians!'
    )
  dist = dist_builder(*params)
  # Need to apply chain rule for log scale grad (instead of scale grad).
  return [
      measure_valued_estimation_mean(
          function, dist, rng, num_samples, coupling=coupling
      ),
      jnp.exp(dist.log_scale)
      * measure_valued_estimation_std(
          function, dist, rng, num_samples, coupling=coupling
      ),
  ]


@chex.warn_deprecated_function
def measure_valued_estimation_mean(
    function: Callable[[chex.Array], float],
    dist: Any,
    rng: chex.PRNGKey,
    num_samples: int,
    coupling: bool = True,
) -> chex.Array:
  """Measure valued grads of a Gaussian expectation of `function` wrt the mean.

  Args:
    function: Function f(x) for which to estimate grads_{mean} E_dist f(x). The
      function takes in one argument (a sample from the distribution) and
      returns a floating point value.
    dist: a distribution on which we can call `sample`.
    rng: a PRNGKey key.
    num_samples: Int, the number of samples used to compute the grads.
    coupling: A boolean. Whether or not to use coupling for the positive and
      negative samples. Recommended: True, as this reduces variance.

  Returns:
    A `num_samples x D` vector containing the estimates of the gradients
    obtained for each sample. The mean of this vector can be used to update
    the mean parameter. The entire vector can be used to assess estimator
    variance.

  .. deprecated:: 0.2.4
    This function will be removed in 0.3.0
  """
  mean, log_std = dist.params
  std = jnp.exp(log_std)

  dist_samples = dist.sample((num_samples,), seed=rng)

  pos_rng, neg_rng = jax.random.split(rng)
  pos_sample = jax.random.weibull_min(
      pos_rng, scale=math.sqrt(2.0), concentration=2.0, shape=dist_samples.shape
  )

  if coupling:
    neg_sample = pos_sample
  else:
    neg_sample = jax.random.weibull_min(
        neg_rng,
        scale=math.sqrt(2.0),
        concentration=2.0,
        shape=dist_samples.shape,
    )

  # N x D
  positive_diag = mean + std * pos_sample
  # N x D
  negative_diag = mean - std * neg_sample

  # NOTE: you can sample base samples here if you use the same rng
  # Duplicate the D dimension - N x D x D.
  base_dist_samples = utils.tile_second_to_last_dim(dist_samples)
  positive = utils.set_diags(base_dist_samples, positive_diag)
  negative = utils.set_diags(base_dist_samples, negative_diag)

  c = np.sqrt(2 * np.pi) * std  # D
  # Apply function. We apply the function to each element of N x D x D.
  # We apply a function that takes a sample and returns one number, so the
  # output will be N x D (which is what we want, batch by dimension).
  # We apply a function in parallel to the batch.
  # Broadcast the division.
  vmaped_function = jax.vmap(jax.vmap(function, 1, 0))
  grads = (vmaped_function(positive) - vmaped_function(negative)) / c

  chex.assert_shape(grads, (num_samples,) + std.shape)
  return grads


@chex.warn_deprecated_function
def measure_valued_estimation_std(
    function: Callable[[chex.Array], float],
    dist: Any,
    rng: chex.PRNGKey,
    num_samples: int,
    coupling: bool = True,
) -> chex.Array:
  """Measure valued grads of a Gaussian expectation of `function` wrt the std.

  Args:
    function: Function f(x) for which to estimate grads_{std} E_dist f(x). The
      function takes in one argument (a sample from the distribution) and
      returns a floating point value.
    dist: a distribution on which we can call `sample`.
    rng: a PRNGKey key.
    num_samples: Int, the number of samples used to compute the grads.
    coupling: A boolean. Whether or not to use coupling for the positive and
      negative samples. Recommended: True, as this reduces variance.

  Returns:
    A `num_samples x D` vector containing the estimates of the gradients
    obtained for each sample. The mean of this vector can be used to update
    the scale parameter. The entire vector can be used to assess estimator
    variance.

  .. deprecated:: 0.2.4
    This function will be removed in 0.3.0
  """
  mean, log_std = dist.params
  std = jnp.exp(log_std)

  dist_samples = dist.sample((num_samples,), seed=rng)

  pos_rng, neg_rng = jax.random.split(rng)

  # The only difference between mean and std gradients is what we sample.
  pos_sample = jax.random.double_sided_maxwell(
      pos_rng, loc=0.0, scale=1.0, shape=dist_samples.shape
  )
  if coupling:
    unif_rvs = jax.random.uniform(neg_rng, dist_samples.shape)
    neg_sample = unif_rvs * pos_sample
  else:
    neg_sample = jax.random.normal(neg_rng, dist_samples.shape)

  # Both need to be positive in the case of the scale.
  # N x D
  positive_diag = mean + std * pos_sample
  # N x D
  negative_diag = mean + std * neg_sample

  # NOTE: you can sample base samples here if you use the same rng
  # Duplicate the D dimension - N x D x D.
  base_dist_samples = utils.tile_second_to_last_dim(dist_samples)
  positive = utils.set_diags(base_dist_samples, positive_diag)
  negative = utils.set_diags(base_dist_samples, negative_diag)

  # Different C for the scale
  c = std  # D
  # Apply function. We apply the function to each element of N x D x D.
  # We apply a function that takes a sample and returns one number, so the
  # output will be N x D (which is what we want, batch by dimension).
  # We apply a function in parallel to the batch.
  # Broadcast the division.
  vmaped_function = jax.vmap(jax.vmap(function, 1, 0))
  grads = (vmaped_function(positive) - vmaped_function(negative)) / c

  chex.assert_shape(grads, (num_samples,) + std.shape)
  return grads
