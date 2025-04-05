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
"""Tests for optax.losses._ranking."""

import doctest
import functools
import math
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax.losses import _ranking

# Export symbols from math for conciser test value definitions.
exp = math.exp
log = math.log
logloss = lambda x: log(1.0 + exp(-x))
sigmoid = lambda x: 1.0 / (1.0 + exp(-x))


class RankingLossesTest(parameterized.TestCase):

  @parameterized.parameters([
      {
          "loss_fn": _ranking.ranking_softmax_loss,
          "expected_value": -(
              log(exp(2.0) / (exp(0.0) + exp(3.0) + exp(1.0) + exp(2.0)))
              + log(exp(1.0) / (exp(0.0) + exp(3.0) + exp(1.0) + exp(2.0)))
          ),
      },
  ])
  def test_computes_loss_value(self, loss_fn, expected_value):
    scores = jnp.asarray([0.0, 3.0, 1.0, 2.0])
    labels = jnp.asarray([0.0, 0.0, 1.0, 1.0])

    loss = loss_fn(scores, labels, reduce_fn=jnp.sum)

    np.testing.assert_allclose(jnp.asarray(expected_value), loss, rtol=1e-3)

  @parameterized.parameters([
      {
          "loss_fn": _ranking.ranking_softmax_loss,
          "expected_value": -(
              (-2.1e26 - (0.0 + -2.1e26 + 3.4e37 + 42.0))
              + (3.4e37 - (0.0 + -2.1e26 + 3.4e37 + 42.0))
          ),
      },
  ])
  def test_computes_loss_with_extreme_inputs(self, loss_fn, expected_value):
    scores = jnp.asarray([0.0, -2.1e26, 3.4e37, 42.0])
    labels = jnp.asarray([0.0, 1.0, 1.0, 0.0])

    loss = loss_fn(scores, labels, reduce_fn=jnp.sum)

    np.testing.assert_allclose(jnp.asarray(expected_value), loss, rtol=1e-3)

  @parameterized.parameters([
      {"loss_fn": _ranking.ranking_softmax_loss, "expected_value": 0.0},
  ])
  def test_computes_loss_for_zero_labels(self, loss_fn, expected_value):
    scores = jnp.asarray([0.0, 3.0, 1.0, 2.0])
    labels = jnp.asarray([0.0, 0.0, 0.0, 0.0])

    loss = loss_fn(scores, labels, reduce_fn=jnp.sum)

    np.testing.assert_allclose(jnp.asarray(expected_value), loss, rtol=1e-3)

  @parameterized.parameters([
      {
          "loss_fn": _ranking.ranking_softmax_loss,
          "expected_value": -(
              2.0 * log(exp(2.0) / (exp(0.0) + exp(3.0) + exp(1.0) + exp(2.0)))
              + log(exp(1.0) / (exp(0.0) + exp(3.0) + exp(1.0) + exp(2.0)))
          ),
      },
  ])
  def test_computes_weighted_loss_value(self, loss_fn, expected_value):
    scores = jnp.asarray([0.0, 3.0, 2.0, 1.0])
    labels = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    weights = jnp.asarray([1.0, 1.0, 2.0, 1.0])

    loss = loss_fn(scores, labels, weights=weights, reduce_fn=jnp.sum)

    np.testing.assert_allclose(jnp.asarray(expected_value), loss, rtol=1e-3)

  @parameterized.parameters([
      {
          "loss_fn": _ranking.ranking_softmax_loss,
          "expected_value": [
              -(
                  log(exp(2.0) / (exp(0.0) + exp(3.0) + exp(1.0) + exp(2.0)))
                  + log(exp(1.0) / (exp(0.0) + exp(3.0) + exp(1.0) + exp(2.0)))
              ),
              -(
                  2.0
                  * log(exp(3.0) / (exp(3.0) + exp(1.0) + exp(4.0) + exp(2.0)))
                  + log(exp(4.0) / (exp(3.0) + exp(1.0) + exp(4.0) + exp(2.0)))
              ),
          ],
      },
  ])
  def test_computes_loss_value_with_vmap(self, loss_fn, expected_value):
    scores = jnp.asarray([[0.0, 3.0, 1.0, 2.0], [3.0, 1.0, 4.0, 2.0]])
    labels = jnp.asarray([[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 0.0]])

    loss_fn = functools.partial(loss_fn, reduce_fn=jnp.sum)
    vmap_loss_fn = jax.vmap(loss_fn, in_axes=(0, 0), out_axes=0)
    loss = vmap_loss_fn(scores, labels)

    np.testing.assert_allclose(jnp.asarray(expected_value), loss, rtol=1e-3)

  @parameterized.parameters([
      {
          "loss_fn": _ranking.ranking_softmax_loss,
          "expected_value": [
              -log(exp(2.0) / (exp(2.0) + exp(1.0) + exp(3.0))),
              -log(exp(1.5) / (exp(1.0) + exp(0.5) + exp(1.5))),
          ],
          "normalizer": 2.0,
      },
  ])
  def test_computes_reduced_loss(self, loss_fn, expected_value, normalizer):
    scores = jnp.array([[2.0, 1.0, 3.0], [1.0, 0.5, 1.5]])
    labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expected_value = jnp.asarray(expected_value)

    mean_loss = loss_fn(scores, labels, reduce_fn=jnp.mean)
    sum_loss = loss_fn(scores, labels, reduce_fn=jnp.sum)

    np.testing.assert_allclose(
        mean_loss, jnp.sum(expected_value) / normalizer, rtol=1e-3
    )
    np.testing.assert_allclose(sum_loss, jnp.sum(expected_value), rtol=1e-3)

  @parameterized.parameters([
      {"loss_fn": _ranking.ranking_softmax_loss, "expected_shape": (2,)},
  ])
  def test_computes_unreduced_loss(self, loss_fn, expected_shape):
    scores = jnp.array([[2.0, 1.0, 3.0], [1.0, 0.5, 1.5]])
    labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    none_loss = loss_fn(scores, labels, reduce_fn=None)
    sum_loss = loss_fn(scores, labels, reduce_fn=jnp.sum)

    self.assertEqual(none_loss.shape, expected_shape)
    self.assertEqual(jnp.sum(none_loss), sum_loss)

  @parameterized.parameters([
      _ranking.ranking_softmax_loss,
  ])
  def test_computes_loss_value_with_where(self, loss_fn):
    scores = jnp.asarray([0.0, 3.0, 1.0, 2.0])
    labels = jnp.asarray([0.0, 0.0, 2.0, 1.0])
    where = jnp.asarray([True, True, True, False])
    expected_scores = jnp.asarray([0.0, 3.0, 1.0])
    expected_labels = jnp.asarray([0.0, 0.0, 2.0])

    loss = loss_fn(scores, labels, where=where)
    expected_loss = loss_fn(expected_scores, expected_labels)

    np.testing.assert_allclose(expected_loss, loss, rtol=1e-3)

  @parameterized.parameters([
      _ranking.ranking_softmax_loss,
  ])
  def test_computes_loss_value_with_all_masked(self, loss_fn):
    scores = jnp.asarray([0.0, 3.0, 1.0, 2.0])
    labels = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    where = jnp.asarray([False, False, False, False])

    loss = loss_fn(scores, labels, where=where)

    np.testing.assert_allclose(jnp.asarray(0.0), loss, rtol=1e-3)

  @parameterized.parameters([
      _ranking.ranking_softmax_loss,
  ])
  def test_computes_loss_with_arbitrary_batch_dimensions(self, loss_fn):
    scores = jnp.asarray([2.0, 3.0, 1.0])
    labels = jnp.asarray([0.0, 0.0, 1.0])
    where = jnp.asarray([False, True, True])
    original_loss = loss_fn(scores, labels, where=where)

    scores = jnp.asarray([[[[2.0, 3.0, 1.0]]]])
    labels = jnp.asarray([[[[0.0, 0.0, 1.0]]]])
    where = jnp.asarray([[[[False, True, True]]]])
    batched_loss = loss_fn(scores, labels, where=where)

    np.testing.assert_allclose(original_loss, batched_loss, rtol=1e-3)

  @parameterized.parameters([
      _ranking.ranking_softmax_loss,
  ])
  def test_grad_does_not_return_nan_for_zero_labels(self, loss_fn):
    scores = jnp.asarray([0.0, 3.0, 1.0, 2.0])
    labels = jnp.asarray([0.0, 0.0, 0.0, 0.0])

    grads = jax.grad(loss_fn)(scores, labels, reduce_fn=jnp.mean)

    np.testing.assert_array_equal(
        jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads))
    )

  @parameterized.parameters([
      _ranking.ranking_softmax_loss,
  ])
  def test_grad_does_not_return_nan_with_all_masked(self, loss_fn):
    scores = jnp.asarray([0.0, 3.0, 1.0, 2.0])
    labels = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    where = jnp.asarray([False, False, False, False])

    grads = jax.grad(loss_fn)(scores, labels, where=where, reduce_fn=jnp.mean)

    np.testing.assert_array_equal(
        jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads))
    )

  @parameterized.parameters([
      _ranking.ranking_softmax_loss,
  ])
  def test_ignores_lists_containing_only_invalid_items(self, loss_fn):
    scores = jnp.asarray([[0.0, 3.0, 1.0, 2.0], [3.0, 1.0, 4.0, 2.0]])
    labels = jnp.asarray([[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 0.0]])
    mask = jnp.asarray([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=jnp.bool_)

    output = loss_fn(scores, labels, where=mask)
    expected = loss_fn(scores[0, :], labels[0, :])

    np.testing.assert_allclose(output, expected, rtol=1e-3)


def load_tests(loader, tests, ignore):
  del loader, ignore  # Unused.
  tests.addTests(
      doctest.DocTestSuite(
          _ranking, globs={"jax": jax, "jnp": jnp, "optax": optax}
      )
  )
  return tests


if __name__ == "__main__":
  absltest.main()
