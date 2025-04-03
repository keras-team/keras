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

"""Tests for linear_algebraic methods in `linear_algebra.py`."""

from typing import Iterable

from absl.testing import absltest
from absl.testing import parameterized
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import linear_algebra
import optax.tree_utils as otu
import scipy.stats


class MLP(nn.Module):
  # Multi-layer perceptron (MLP).
  num_outputs: int
  hidden_sizes: Iterable[int]

  @nn.compact
  def __call__(self, x):
    for num_hidden in self.hidden_sizes:
      x = nn.Dense(num_hidden)(x)
      x = nn.gelu(x)
    return nn.Dense(self.num_outputs)(x)


class LinearAlgebraTest(chex.TestCase):

  def test_global_norm(self):
    flat_updates = jnp.array([2.0, 4.0, 3.0, 5.0], dtype=jnp.float32)
    nested_updates = dict(
        a=jnp.array([2.0, 4.0], dtype=jnp.float32),
        b=jnp.array([3.0, 5.0], dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        jnp.sqrt(jnp.sum(flat_updates**2)),
        linear_algebra.global_norm(nested_updates),
    )

  def test_power_iteration_cond_fun(self, dim=6):
    """Test the condition function for power iteration."""
    matrix = jax.random.normal(jax.random.PRNGKey(0), (dim, dim))
    matrix = matrix @ matrix.T
    all_eigenval, all_eigenvec = jax.numpy.linalg.eigh(matrix)
    dominant_eigenval = all_eigenval[-1]
    dominant_eigenvec = all_eigenvec[:, -1] * jnp.sign(all_eigenvec[:, -1][0])
    # loop variables for _power_iteration_cond_fun
    loop_vars = (
        dominant_eigenvec,
        dominant_eigenval * dominant_eigenvec,
        dominant_eigenval,
        10,
    )
    # when given the correct dominant eigenvector, the condition function
    # should stop and return False.
    cond_fun_result = linear_algebra._power_iteration_cond_fun(
        100, 1e-3, loop_vars
    )
    self.assertEqual(cond_fun_result, False)

  @chex.all_variants
  @parameterized.parameters(
      dict(implicit=True),
      dict(implicit=False),
  )
  def test_power_iteration(self, implicit, dim=6, tol=1e-3, num_iters=100):
    """Test power_iteration by comparing to numpy.linalg.eigh."""

    if implicit:
      # test the function when the matrix is given in implicit form by a
      # matrix-vector product.
      def power_iteration(matrix, *, v0):
        return linear_algebra.power_iteration(
            lambda x: matrix @ x,
            v0=v0,
            error_tolerance=tol,
            num_iters=num_iters,
        )

    else:
      power_iteration = linear_algebra.power_iteration

    # test this function with/without jax.jit and on different devices
    power_iteration = self.variant(power_iteration)

    # create a random PSD matrix
    matrix = jax.random.normal(jax.random.PRNGKey(0), (dim, dim))
    matrix = matrix @ matrix.T
    v0 = jnp.ones((dim,))

    eigval_power, eigvec_power = power_iteration(matrix, v0=v0)
    all_eigenval, all_eigenvec = jax.numpy.linalg.eigh(matrix)

    self.assertAlmostEqual(eigval_power, all_eigenval[-1], delta=10 * tol)
    np.testing.assert_array_almost_equal(
        all_eigenvec[:, -1] * jnp.sign(all_eigenvec[:, -1][0]),
        eigvec_power * jnp.sign(eigvec_power[0]),
        decimal=3,
    )

  @chex.all_variants
  def test_power_iteration_pytree(self, dim=6, tol=1e-3, num_iters=100):
    """Test power_iteration for matrix-vector products acting on pytrees."""

    def matrix_vector_product(x):
      # implements a block-diagonal matrix where each block is a scaled
      # identity matrix. The scaling factor is 2 and 1 for the first and second
      # block respectively.
      return {'a': 2 * x['a'], 'b': x['b']}

    @self.variant
    def power_iteration(*, v0):
      return linear_algebra.power_iteration(
          matrix_vector_product,
          v0=v0,
          error_tolerance=tol,
          num_iters=num_iters,
      )

    v0 = {'a': jnp.ones((dim,)), 'b': jnp.ones((dim,))}

    eigval_power, _ = power_iteration(v0=v0)

    # from the block-diagonal structure of matrix, largest eigenvalue is 2.
    self.assertAlmostEqual(eigval_power, 2.0, delta=10 * tol)

  @chex.all_variants
  def test_power_iteration_mlp_hessian(
      self, input_dim=16, output_dim=4, tol=1e-3
  ):
    """Test power_iteration on the Hessian of an MLP."""
    mlp = MLP(num_outputs=output_dim, hidden_sizes=[input_dim, 8, output_dim])
    key = jax.random.PRNGKey(0)
    key_params, key_input, key_output = jax.random.split(key, 3)
    # initialize the mlp
    params = mlp.init(key_params, jnp.ones(input_dim))
    x = jax.random.normal(key_input, (input_dim,))
    y = jax.random.normal(key_output, (output_dim,))

    @self.variant
    def train_obj(params_):
      z = mlp.apply(params_, x)
      return jnp.sum((z - y) ** 2)

    def hessian_vector_product(tangents_):
      return jax.jvp(jax.grad(train_obj), (params,), (tangents_,))[1]

    eigval_power, eigvec_power = linear_algebra.power_iteration(
        hessian_vector_product, v0=otu.tree_ones_like(params)
    )

    params_flat, unravel = jax.flatten_util.ravel_pytree(params)
    eigvec_power_flat, _ = jax.flatten_util.ravel_pytree(eigvec_power)

    def train_obj_flat(params_flat_):
      params_ = unravel(params_flat_)
      return train_obj(params_)

    hessian = jax.hessian(train_obj_flat)(params_flat)
    all_eigenval, all_eigenvec = jax.numpy.linalg.eigh(hessian)

    self.assertAlmostEqual(all_eigenval[-1], eigval_power, delta=10 * tol)
    np.testing.assert_array_almost_equal(
        all_eigenvec[:, -1] * jnp.sign(all_eigenvec[:, -1][0]),
        eigvec_power_flat * jnp.sign(eigvec_power_flat[0]),
        decimal=3,
    )

  def test_matrix_inverse_pth_root(self):
    """Test for matrix inverse pth root."""

    def _gen_symmetrix_matrix(dim, condition_number):
      u = scipy.stats.ortho_group.rvs(dim=dim).astype(np.float64)
      v = u.T
      diag = np.diag([condition_number ** (-i / (dim - 1)) for i in range(dim)])
      return u @ diag @ v

    # Fails after it reaches a particular condition number.
    for e in range(2, 12):
      condition_number = 10**e
      ms = _gen_symmetrix_matrix(16, condition_number)
      self.assertLess(
          np.abs(np.linalg.cond(ms) - condition_number), condition_number * 0.01
      )
      error = linear_algebra.matrix_inverse_pth_root(
          ms.astype(np.float32), 4, ridge_epsilon=1e-12
      )[1]
      if e < 7:
        self.assertLess(error, 0.1)
      else:
        # No guarantee of success after e >= 7
        pass


if __name__ == '__main__':
  absltest.main()
