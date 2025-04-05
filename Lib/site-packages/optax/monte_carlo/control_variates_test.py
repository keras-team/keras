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
"""Tests for methods in `control_variates.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import utils
from optax.monte_carlo import control_variates
from optax.monte_carlo import stochastic_gradient_estimators as sge


# Set seed for deterministic sampling.
np.random.seed(42)


def _assert_equal(actual, expected, rtol=1e-2, atol=1e-2):
  """Asserts that arrays are equal."""
  # Note: assert_allclose does not check shapes
  chex.assert_equal_shape((actual, expected))

  # Scalar.
  if not actual.shape:
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol, atol
    )
    return

  # We get around the bug https://github.com/numpy/numpy/issues/13801
  zero_indices = np.argwhere(expected == 0)
  if not np.all(np.abs(actual[zero_indices]) <= atol):
    raise AssertionError(f'Larger than {atol} diff in {actual[zero_indices]}')

  non_zero_indices = np.argwhere(expected != 0)
  np.testing.assert_allclose(
      np.asarray(actual)[non_zero_indices],
      expected[non_zero_indices],
      rtol,
      atol,
  )


def _map(cv, params, samples, state=None):
  return jax.vmap(lambda x: cv(params, x, state))(samples)


def _map_variant(variant):
  return variant(_map, static_argnums=0)


def _cv_jac_variant(variant):
  return variant(
      control_variates.control_variates_jacobians,
      static_argnums=(0, 1, 2, 4, 6, 7, 8),
  )


class DeltaControlVariateTest(chex.TestCase):

  @chex.all_variants
  @parameterized.parameters([(1.0, 0.5)])
  def testQuadraticFunction(self, effective_mean, effective_log_scale):
    data_dims = 20
    num_samples = 10**6
    rng = jax.random.PRNGKey(1)

    mean = effective_mean * jnp.ones(shape=(data_dims), dtype=jnp.float32)
    log_scale = effective_log_scale * jnp.ones(
        shape=(data_dims), dtype=jnp.float32
    )
    params = [mean, log_scale]

    dist = utils.multi_normal(*params)
    dist_samples = dist.sample((num_samples,), rng)
    function = lambda x: jnp.sum(x**2)

    cv, expected_cv, _ = control_variates.control_delta_method(function)
    avg_cv = jnp.mean(_map_variant(self.variant)(cv, params, dist_samples))
    expected_cv_value = jnp.sum(dist_samples**2) / num_samples

    # This should be an analytical computation, the result needs to be
    # accurate.
    _assert_equal(avg_cv, expected_cv_value, rtol=1e-1, atol=1e-3)
    _assert_equal(expected_cv(params, None), expected_cv_value, rtol=0.02)

  @chex.all_variants
  @parameterized.parameters([(1.0, 1.0)])
  def testPolynomialFunction(self, effective_mean, effective_log_scale):
    data_dims = 10
    num_samples = 10**3

    mean = effective_mean * jnp.ones(shape=(data_dims), dtype=jnp.float32)
    log_scale = effective_log_scale * jnp.ones(
        shape=(data_dims), dtype=jnp.float32
    )
    params = [mean, log_scale]

    dist = utils.multi_normal(*params)
    rng = jax.random.PRNGKey(1)
    dist_samples = dist.sample((num_samples,), rng)
    function = lambda x: jnp.sum(x**5)

    cv, expected_cv, _ = control_variates.control_delta_method(function)
    avg_cv = jnp.mean(_map_variant(self.variant)(cv, params, dist_samples))

    # Check that the average value of the control variate is close to the
    # expected value.
    _assert_equal(avg_cv, expected_cv(params, None), rtol=1e-1, atol=1e-3)

  @chex.all_variants
  def testNonPolynomialFunction(self):
    data_dims = 10
    num_samples = 10**3

    mean = jnp.ones(shape=(data_dims), dtype=jnp.float32)
    log_scale = jnp.ones(shape=(data_dims), dtype=jnp.float32)
    params = [mean, log_scale]

    rng = jax.random.PRNGKey(1)
    dist = utils.multi_normal(*params)
    dist_samples = dist.sample((num_samples,), rng)
    function = lambda x: jnp.sum(jnp.log(x**2))

    cv, expected_cv, _ = control_variates.control_delta_method(function)
    avg_cv = jnp.mean(_map_variant(self.variant)(cv, params, dist_samples))

    # Check that the average value of the control variate is close to the
    # expected value.
    _assert_equal(avg_cv, expected_cv(params, None), rtol=1e-1, atol=1e-3)

    # Second order expansion is log(\mu**2) + 1/2 * \sigma**2 (-2 / \mu**2)
    expected_cv_val = -np.exp(1.0) ** 2 * data_dims
    _assert_equal(
        expected_cv(params, None), expected_cv_val, rtol=1e-1, atol=1e-3
    )


class MovingAverageBaselineTest(chex.TestCase):

  @chex.all_variants
  @parameterized.parameters([(1.0, 0.5, 0.9), (1.0, 0.5, 0.99)])
  def testLinearFunction(self, effective_mean, effective_log_scale, decay):
    weights = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    num_samples = 10**4
    data_dims = len(weights)

    mean = effective_mean * jnp.ones(shape=(data_dims), dtype=jnp.float32)
    log_scale = effective_log_scale * jnp.ones(
        shape=(data_dims), dtype=jnp.float32
    )

    params = [mean, log_scale]
    function = lambda x: jnp.sum(weights * x)

    rng = jax.random.PRNGKey(1)
    dist = utils.multi_normal(*params)
    dist_samples = dist.sample((num_samples,), rng)

    cv, expected_cv, update_state = control_variates.moving_avg_baseline(
        function,
        decay=decay,
        zero_debias=False,
        use_decay_early_training_heuristic=False,
    )

    state_1 = jnp.array(1.0)
    avg_cv = jnp.mean(
        _map_variant(self.variant)(cv, params, dist_samples, (state_1, 0))
    )
    _assert_equal(avg_cv, state_1)
    _assert_equal(expected_cv(params, (state_1, 0)), state_1)

    state_2 = jnp.array(2.0)
    avg_cv = jnp.mean(
        _map_variant(self.variant)(cv, params, dist_samples, (state_2, 0))
    )
    _assert_equal(avg_cv, state_2)
    _assert_equal(expected_cv(params, (state_2, 0)), state_2)

    update_state_1 = update_state(params, dist_samples, (state_1, 0))[0]
    _assert_equal(
        update_state_1, decay * state_1 + (1 - decay) * function(mean)
    )

    update_state_2 = update_state(params, dist_samples, (state_2, 0))[0]
    _assert_equal(
        update_state_2, decay * state_2 + (1 - decay) * function(mean)
    )

  @chex.all_variants
  @parameterized.parameters([(1.0, 0.5, 0.9), (1.0, 0.5, 0.99)])
  def testLinearFunctionWithHeuristic(
      self, effective_mean, effective_log_scale, decay
  ):
    weights = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    num_samples = 10**5
    data_dims = len(weights)

    mean = effective_mean * jnp.ones(shape=(data_dims), dtype=jnp.float32)
    log_scale = effective_log_scale * jnp.ones(
        shape=(data_dims), dtype=jnp.float32
    )

    params = [mean, log_scale]
    function = lambda x: jnp.sum(weights * x)

    rng = jax.random.PRNGKey(1)
    dist = utils.multi_normal(*params)
    dist_samples = dist.sample((num_samples,), rng)

    cv, expected_cv, update_state = control_variates.moving_avg_baseline(
        function,
        decay=decay,
        zero_debias=False,
        use_decay_early_training_heuristic=True,
    )

    state_1 = jnp.array(1.0)
    avg_cv = jnp.mean(
        _map_variant(self.variant)(cv, params, dist_samples, (state_1, 0))
    )
    _assert_equal(avg_cv, state_1)
    _assert_equal(expected_cv(params, (state_1, 0)), state_1)

    state_2 = jnp.array(2.0)
    avg_cv = jnp.mean(
        _map_variant(self.variant)(cv, params, dist_samples, (state_2, 0))
    )
    _assert_equal(avg_cv, state_2)
    _assert_equal(expected_cv(params, (state_2, 0)), state_2)

    first_step_decay = 0.1
    update_state_1 = update_state(params, dist_samples, (state_1, 0))[0]
    _assert_equal(
        update_state_1,
        first_step_decay * state_1 + (1 - first_step_decay) * function(mean),
    )

    second_step_decay = 2.0 / 11
    update_state_2 = update_state(params, dist_samples, (state_2, 1))[0]
    _assert_equal(
        update_state_2,
        second_step_decay * state_2 + (1 - second_step_decay) * function(mean),
    )

  @parameterized.parameters([(1.0, 0.5, 0.9), (1.0, 0.5, 0.99)])
  def testLinearFunctionZeroDebias(
      self, effective_mean, effective_log_scale, decay
  ):
    weights = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    num_samples = 10**5
    data_dims = len(weights)

    mean = effective_mean * jnp.ones(shape=(data_dims), dtype=jnp.float32)
    log_scale = effective_log_scale * jnp.ones(
        shape=(data_dims), dtype=jnp.float32
    )

    params = [mean, log_scale]
    function = lambda x: jnp.sum(weights * x)

    rng = jax.random.PRNGKey(1)
    dist = utils.multi_normal(*params)
    dist_samples = dist.sample((num_samples,), rng)

    update_state = control_variates.moving_avg_baseline(
        function,
        decay=decay,
        zero_debias=False,
        use_decay_early_training_heuristic=False,
    )[-1]

    update_state_zero_debias = control_variates.moving_avg_baseline(
        function,
        decay=decay,
        zero_debias=True,
        use_decay_early_training_heuristic=False,
    )[-1]

    updated_state = update_state(params, dist_samples, (jnp.array(0.0), 0))[0]
    _assert_equal(updated_state, (1 - decay) * function(mean))

    updated_state_zero_debias = update_state_zero_debias(
        params, dist_samples, (jnp.array(0.0), 0)
    )[0]
    _assert_equal(updated_state_zero_debias, function(mean))


class DeltaMethodAnalyticalExpectedGrads(chex.TestCase):
  """Tests for grads approximations."""

  @chex.all_variants
  @parameterized.named_parameters(
      chex.params_product(
          [
              (
                  '_score_function_jacobians',
                  1.0,
                  1.0,
                  sge.score_function_jacobians,
              ),
              ('_pathwise_jacobians', 1.0, 1.0, sge.pathwise_jacobians),
              (
                  '_measure_valued_jacobians',
                  1.0,
                  1.0,
                  sge.measure_valued_jacobians,
              ),
          ],
          [
              ('estimate_cv_coeffs', True),
              ('no_estimate_cv_coeffs', False),
          ],
          named=True,
      )
  )
  def testQuadraticFunction(
      self,
      effective_mean,
      effective_log_scale,
      grad_estimator,
      estimate_cv_coeffs,
  ):
    data_dims = 3
    num_samples = 10**3

    mean = effective_mean * jnp.ones(shape=(data_dims), dtype=jnp.float32)
    log_scale = effective_log_scale * jnp.ones(
        shape=(data_dims), dtype=jnp.float32
    )

    params = [mean, log_scale]
    function = lambda x: jnp.sum(x**2)
    rng = jax.random.PRNGKey(1)

    jacobians = _cv_jac_variant(self.variant)(
        function,
        control_variates.control_delta_method,
        grad_estimator,
        params,
        utils.multi_normal,  # dist_builder
        rng,
        num_samples,
        None,  # No cv state.
        estimate_cv_coeffs,
    )[0]

    expected_mean_grads = (
        2 * effective_mean * np.ones(data_dims, dtype=np.float32)
    )
    expected_log_scale_grads = (
        2
        * np.exp(2 * effective_log_scale)
        * np.ones(data_dims, dtype=np.float32)
    )

    mean_jacobians = jacobians[0]
    chex.assert_shape(mean_jacobians, (num_samples, data_dims))
    mean_grads_from_jacobian = jnp.mean(mean_jacobians, axis=0)

    log_scale_jacobians = jacobians[1]
    chex.assert_shape(log_scale_jacobians, (num_samples, data_dims))
    log_scale_grads_from_jacobian = jnp.mean(log_scale_jacobians, axis=0)

    _assert_equal(
        mean_grads_from_jacobian, expected_mean_grads, rtol=1e-1, atol=1e-3
    )
    _assert_equal(
        log_scale_grads_from_jacobian,
        expected_log_scale_grads,
        rtol=1e-1,
        atol=1e-3,
    )

  @chex.all_variants
  @parameterized.named_parameters(
      chex.params_product(
          [
              (
                  '_score_function_jacobians',
                  1.0,
                  1.0,
                  sge.score_function_jacobians,
              ),
              ('_pathwise_jacobians', 1.0, 1.0, sge.pathwise_jacobians),
              (
                  '_measure_valued_jacobians',
                  1.0,
                  1.0,
                  sge.measure_valued_jacobians,
              ),
          ],
          [
              ('estimate_cv_coeffs', True),
              ('no_estimate_cv_coeffs', False),
          ],
          named=True,
      )
  )
  def testCubicFunction(
      self,
      effective_mean,
      effective_log_scale,
      grad_estimator,
      estimate_cv_coeffs,
  ):
    data_dims = 1
    num_samples = 10**5

    mean = effective_mean * jnp.ones(shape=(data_dims), dtype=jnp.float32)
    log_scale = effective_log_scale * jnp.ones(
        shape=(data_dims), dtype=jnp.float32
    )

    params = [mean, log_scale]
    function = lambda x: jnp.sum(x**3)
    rng = jax.random.PRNGKey(1)

    jacobians = _cv_jac_variant(self.variant)(
        function,
        control_variates.control_delta_method,
        grad_estimator,
        params,
        utils.multi_normal,
        rng,
        num_samples,
        None,  # No cv state.
        estimate_cv_coeffs,
    )[0]

    # The third order uncentered moment of the Gaussian distribution is
    # mu**3 + 2 mu * sigma **2. We use that to compute the expected value
    # of the gradients. Note: for the log scale we need use the chain rule.
    expected_mean_grads = (
        3 * effective_mean**2 + 3 * np.exp(effective_log_scale) ** 2
    )
    expected_mean_grads *= np.ones(data_dims, dtype=np.float32)
    expected_log_scale_grads = (
        6 * effective_mean * np.exp(effective_log_scale) ** 2
    )
    expected_log_scale_grads *= np.ones(data_dims, dtype=np.float32)

    mean_jacobians = jacobians[0]
    chex.assert_shape(mean_jacobians, (num_samples, data_dims))
    mean_grads_from_jacobian = jnp.mean(mean_jacobians, axis=0)

    log_scale_jacobians = jacobians[1]
    chex.assert_shape(log_scale_jacobians, (num_samples, data_dims))
    log_scale_grads_from_jacobian = jnp.mean(log_scale_jacobians, axis=0)

    _assert_equal(
        mean_grads_from_jacobian, expected_mean_grads, rtol=1e-1, atol=1e-3
    )

    _assert_equal(
        log_scale_grads_from_jacobian,
        expected_log_scale_grads,
        rtol=1e-1,
        atol=1e-3,
    )

  @chex.all_variants
  @parameterized.named_parameters(
      chex.params_product(
          [
              (
                  '_score_function_jacobians',
                  1.0,
                  1.0,
                  sge.score_function_jacobians,
              ),
              ('_pathwise_jacobians', 1.0, 1.0, sge.pathwise_jacobians),
              (
                  '_measure_valued_jacobians',
                  1.0,
                  1.0,
                  sge.measure_valued_jacobians,
              ),
          ],
          [
              ('estimate_cv_coeffs', True),
              ('no_estimate_cv_coeffs', False),
          ],
          named=True,
      )
  )
  def testForthPowerFunction(
      self,
      effective_mean,
      effective_log_scale,
      grad_estimator,
      estimate_cv_coeffs,
  ):
    data_dims = 1
    num_samples = 10**5

    mean = effective_mean * jnp.ones(shape=(data_dims), dtype=jnp.float32)
    log_scale = effective_log_scale * jnp.ones(
        shape=(data_dims), dtype=jnp.float32
    )

    params = [mean, log_scale]
    function = lambda x: jnp.sum(x**4)
    rng = jax.random.PRNGKey(1)

    jacobians = _cv_jac_variant(self.variant)(
        function,
        control_variates.control_delta_method,
        grad_estimator,
        params,
        utils.multi_normal,
        rng,
        num_samples,
        None,  # No cv state
        estimate_cv_coeffs,
    )[0]
    # The third order uncentered moment of the Gaussian distribution is
    # mu**4 + 6 mu **2 sigma **2 + 3 sigma**4. We use that to compute the
    # expected value of the gradients.
    # Note: for the log scale we need use the chain rule.
    expected_mean_grads = (
        3 * effective_mean**3
        + 12 * effective_mean * np.exp(effective_log_scale) ** 2
    )
    expected_mean_grads *= np.ones(data_dims, dtype=np.float32)
    expected_log_scale_grads = (
        12
        * (
            effective_mean**2 * np.exp(effective_log_scale)
            + np.exp(effective_log_scale) ** 3
        )
        * np.exp(effective_log_scale)
    )
    expected_log_scale_grads *= np.ones(data_dims, dtype=np.float32)

    mean_jacobians = jacobians[0]
    chex.assert_shape(mean_jacobians, (num_samples, data_dims))
    mean_grads_from_jacobian = jnp.mean(mean_jacobians, axis=0)

    log_scale_jacobians = jacobians[1]
    chex.assert_shape(log_scale_jacobians, (num_samples, data_dims))
    log_scale_grads_from_jacobian = jnp.mean(log_scale_jacobians, axis=0)

    _assert_equal(
        mean_grads_from_jacobian, expected_mean_grads, rtol=1e-1, atol=1e-3
    )

    _assert_equal(
        log_scale_grads_from_jacobian,
        expected_log_scale_grads,
        rtol=1e-1,
        atol=1e-3,
    )


class ConsistencyWithStandardEstimators(chex.TestCase):
  """Tests for consistency between estimators."""

  @chex.all_variants
  @parameterized.named_parameters(
      chex.params_product(
          [
              ('_score_function_jacobians', 1, 1, sge.score_function_jacobians),
              ('_pathwise_jacobians', 1, 1, sge.pathwise_jacobians),
              ('_measure_valued_jacobians', 1, 1, sge.measure_valued_jacobians),
          ],
          [
              (
                  'control_delta_method',
                  10**5,
                  control_variates.control_delta_method,
              ),
              (
                  'moving_avg_baseline',
                  10**6,
                  control_variates.moving_avg_baseline,
              ),
          ],
          named=True,
      )
  )
  def testWeightedLinearFunction(
      self,
      effective_mean,
      effective_log_scale,
      grad_estimator,
      num_samples,
      control_variate_from_function,
  ):
    """Check that the gradients are consistent between estimators."""
    weights = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    data_dims = len(weights)

    mean = effective_mean * jnp.ones(shape=(data_dims), dtype=jnp.float32)
    log_scale = effective_log_scale * jnp.ones(
        shape=(data_dims), dtype=jnp.float32
    )

    params = [mean, log_scale]
    function = lambda x: jnp.sum(weights * x)
    rng = jax.random.PRNGKey(1)
    cv_rng, ge_rng = jax.random.split(rng)

    jacobians = _cv_jac_variant(self.variant)(
        function,
        control_variate_from_function,
        grad_estimator,
        params,
        utils.multi_normal,  # dist_builder
        cv_rng,  # rng
        num_samples,
        (0.0, 0),  # control_variate_state
        False,
    )[0]

    mean_jacobians = jacobians[0]
    chex.assert_shape(mean_jacobians, (num_samples, data_dims))
    mean_grads = jnp.mean(mean_jacobians, axis=0)

    log_scale_jacobians = jacobians[1]
    chex.assert_shape(log_scale_jacobians, (num_samples, data_dims))
    log_scale_grads = jnp.mean(log_scale_jacobians, axis=0)

    # We use a different random number generator for the gradient estimator
    # without the control variate.
    no_cv_jacobians = grad_estimator(
        function,
        [mean, log_scale],
        utils.multi_normal,
        ge_rng,
        num_samples=num_samples,
    )

    no_cv_mean_jacobians = no_cv_jacobians[0]
    chex.assert_shape(no_cv_mean_jacobians, (num_samples, data_dims))
    no_cv_mean_grads = jnp.mean(no_cv_mean_jacobians, axis=0)

    no_cv_log_scale_jacobians = no_cv_jacobians[1]
    chex.assert_shape(no_cv_log_scale_jacobians, (num_samples, data_dims))
    no_cv_log_scale_grads = jnp.mean(no_cv_log_scale_jacobians, axis=0)

    _assert_equal(mean_grads, no_cv_mean_grads, rtol=1e-1, atol=5e-2)
    _assert_equal(log_scale_grads, no_cv_log_scale_grads, rtol=1, atol=5e-2)

  @chex.all_variants
  @parameterized.named_parameters(
      chex.params_product(
          [
              (
                  '_score_function_jacobians',
                  1,
                  1,
                  sge.score_function_jacobians,
                  10**5,
              ),
              ('_pathwise_jacobians', 1, 1, sge.pathwise_jacobians, 10**5),
              (
                  '_measure_valued_jacobians',
                  1,
                  1,
                  sge.measure_valued_jacobians,
                  10**5,
              ),
          ],
          [
              ('control_delta_method', control_variates.control_delta_method),
              ('moving_avg_baseline', control_variates.moving_avg_baseline),
          ],
          named=True,
      )
  )
  def testNonPolynomialFunction(
      self,
      effective_mean,
      effective_log_scale,
      grad_estimator,
      num_samples,
      control_variate_from_function,
  ):
    """Check that the gradients are consistent between estimators."""
    data_dims = 3

    mean = effective_mean * jnp.ones(shape=(data_dims), dtype=jnp.float32)
    log_scale = effective_log_scale * jnp.ones(
        shape=(data_dims), dtype=jnp.float32
    )

    params = [mean, log_scale]
    function = lambda x: jnp.log(jnp.sum(x**2))
    rng = jax.random.PRNGKey(1)
    cv_rng, ge_rng = jax.random.split(rng)

    jacobians = _cv_jac_variant(self.variant)(
        function,
        control_variate_from_function,
        grad_estimator,
        params,
        utils.multi_normal,
        cv_rng,
        num_samples,
        (0.0, 0),  # control_variate_state
        False,
    )[0]

    mean_jacobians = jacobians[0]
    chex.assert_shape(mean_jacobians, (num_samples, data_dims))
    mean_grads = jnp.mean(mean_jacobians, axis=0)

    log_scale_jacobians = jacobians[1]
    chex.assert_shape(log_scale_jacobians, (num_samples, data_dims))
    log_scale_grads = jnp.mean(log_scale_jacobians, axis=0)

    # We use a different random number generator for the gradient estimator
    # without the control variate.
    no_cv_jacobians = grad_estimator(
        function,
        [mean, log_scale],
        utils.multi_normal,
        ge_rng,
        num_samples=num_samples,
    )

    no_cv_mean_jacobians = no_cv_jacobians[0]
    chex.assert_shape(no_cv_mean_jacobians, (num_samples, data_dims))
    no_cv_mean_grads = jnp.mean(no_cv_mean_jacobians, axis=0)

    no_cv_log_scale_jacobians = no_cv_jacobians[1]
    chex.assert_shape(no_cv_log_scale_jacobians, (num_samples, data_dims))
    no_cv_log_scale_grads = jnp.mean(no_cv_log_scale_jacobians, axis=0)

    _assert_equal(mean_grads, no_cv_mean_grads, rtol=1e-1, atol=5e-2)
    _assert_equal(log_scale_grads, no_cv_log_scale_grads, rtol=1e-1, atol=5e-2)


if __name__ == '__main__':
  absltest.main()
