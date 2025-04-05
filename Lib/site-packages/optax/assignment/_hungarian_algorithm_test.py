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
"""Tests for the Hungarian algorithm."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jax.random as jrd
from optax.assignment import _hungarian_algorithm
import scipy


class HungarianAlgorithmTest(parameterized.TestCase):

  @parameterized.product(
      n=[0, 1, 2, 4, 8, 16],
      m=[0, 1, 2, 4, 8, 16],
  )
  def test_hungarian_algorithm(self, n, m):
    key = jrd.key(0)
    costs = jrd.normal(key, (n, m))

    i, j = _hungarian_algorithm.hungarian_algorithm(costs)

    r = min(costs.shape)

    with self.subTest('i has correct shape'):
      assert i.shape == (r,)

    with self.subTest('j has correct shape'):
      assert j.shape == (r,)

    with self.subTest('i has correct dtype'):
      assert jnp.issubdtype(i.dtype, jnp.integer)

    with self.subTest('j has correct dtype'):
      assert jnp.issubdtype(j.dtype, jnp.integer)

    with self.subTest('each element of i is non-negative'):
      assert jnp.all(0 <= i)

    with self.subTest('each element of j is non-negative'):
      assert jnp.all(0 <= j)

    with self.subTest('each element of i is less than the number of rows'):
      assert (i < costs.shape[0]).all()

    with self.subTest('each element of j is less than the number of columns'):
      assert (j < costs.shape[1]).all()

    x = jnp.zeros(costs.shape[0], int).at[i].add(1)

    with self.subTest('all elements of i lie in the valid range'):
      assert x.sum() == r

    with self.subTest('no two elements of i are equal'):
      assert (x <= 1).all()

    y = jnp.zeros(costs.shape[1], int).at[j].add(1)

    with self.subTest('all elements of j lie in the valid range'):
      assert y.sum() == r

    with self.subTest('no two elements of j are equal'):
      assert (y <= 1).all()

    cost_optax = costs[i, j].sum()

    i_scipy, j_scipy = scipy.optimize.linear_sum_assignment(costs)
    cost_scipy = costs[i_scipy, j_scipy].sum()

    with self.subTest('cost matches that obtained by scipy'):
      assert jnp.isclose(cost_optax, cost_scipy)

  @parameterized.product(
      k=[0, 1, 2, 4],
      n=[0, 1, 2, 4],
      m=[0, 1, 2, 4],
  )
  def test_hungarian_algorithm_vmap(self, k, n, m):
    key = jrd.key(0)
    costs = jrd.normal(key, (k, n, m))

    with self.subTest('works under vmap'):
      i, j = jax.vmap(_hungarian_algorithm.hungarian_algorithm)(costs)

    r = min(costs.shape[1:])

    with self.subTest('batch i has correct shape'):
      assert i.shape == (k, r)

    with self.subTest('batch j has correct shape'):
      assert j.shape == (k, r)

  def test_hungarian_algorithm_jit(self):
    key = jrd.key(0)
    costs = jrd.normal(key, (20, 30))

    with self.subTest('works under jit'):
      i, j = jax.jit(_hungarian_algorithm.hungarian_algorithm)(costs)

    r = min(costs.shape)

    with self.subTest('i has correct shape'):
      assert i.shape == (r,)

    with self.subTest('j has correct shape'):
      assert j.shape == (r,)


if __name__ == '__main__':
  absltest.main()
