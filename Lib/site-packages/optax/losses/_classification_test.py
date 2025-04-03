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
"""Tests for losses in `optax.losses._classification.py`."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import jax.test_util as jaxtest
import numpy as np
from optax import projections
from optax.losses import _classification


class SoftmaxCrossEntropyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = np.array(
        [
            [10.0, 1.0, -2.0],
            [1.0, 4.0, 0.2],
        ],
        dtype=np.float32,
    )
    self.ts = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    self.exp = np.array(
        [
            9.00013,
            3.0696733,
        ],
        dtype=np.float32,
    )

  @chex.all_variants
  def test_scalar(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(_classification.softmax_cross_entropy)(
            self.ys[0], self.ts[0]
        ),
        self.exp[0],
        atol=1e-4,
    )

  @chex.all_variants
  def test_batched(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(_classification.softmax_cross_entropy)(self.ys, self.ts),
        self.exp,
        atol=1e-4,
    )

  def test_gradient(self):
    """Tests gradient ok."""
    jaxtest.check_grads(
        _classification.softmax_cross_entropy,
        (self.ys[:2], self.ts[:2]),
        order=1,
    )

  @parameterized.parameters(dict(size=5), dict(size=10))
  def test_mask(self, size):
    preds = np.random.normal(size=size)
    targets = np.random.dirichlet(np.ones(size))
    mask = np.random.randint(2, size=size, dtype=bool)
    x = _classification.softmax_cross_entropy(preds[mask], targets[mask])
    y = _classification.softmax_cross_entropy(preds, targets, where=mask)
    np.testing.assert_allclose(x, y, atol=1e-4)

  @parameterized.parameters(
      dict(axis=0, shape=[4, 5, 6]),
      dict(axis=1, shape=[4, 5, 6]),
      dict(axis=2, shape=[4, 5, 6]),
  )
  def test_axis(self, shape, axis):
    preds = np.random.normal(size=shape)
    targets = np.random.dirichlet(np.ones(shape[-1]), size=shape[:-1])
    x = _classification.softmax_cross_entropy(preds, targets, axis=axis)
    y = _classification.softmax_cross_entropy(
        np.moveaxis(preds, axis, -1),
        np.moveaxis(targets, axis, -1),
    )
    np.testing.assert_allclose(x, y, atol=1e-4)


class SafeSoftmaxCrossEntropyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = np.array(
        [
            [10.0, 1.0, -2.0],
            [1.0, 4.0, 0.2],
            [-np.inf, 0.0, 0.0],
            [-np.inf, 0.0, 0.0],
            [-np.inf, 0.0, -np.inf],
        ],
        dtype=np.float32,
    )
    self.ts = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.4, 0.3, 0.3],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    self.exp = np.array(
        [
            9.00013,
            3.0696733,
            0.693147,
            np.inf,
            0.0,
        ],
        dtype=np.float32,
    )

  @chex.all_variants
  def test_scalar(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(_classification.safe_softmax_cross_entropy)(
            self.ys[0], self.ts[0]
        ),
        self.exp[0],
        atol=1e-4,
    )

  @chex.all_variants
  def test_batched(self):
    """Tests for a full batch."""
    np.testing.assert_allclose(
        self.variant(_classification.safe_softmax_cross_entropy)(
            self.ys, self.ts
        ),
        self.exp,
        atol=1e-4,
    )

  def test_gradient(self):
    """Tests gradient ok."""
    jaxtest.check_grads(
        _classification.safe_softmax_cross_entropy,
        (self.ys[:2], self.ts[:2]),
        order=1,
    )

  def test_against_plain_implementation(self):
    """Tests against plain implementation which does not handle -inf."""

    plain_val_and_grad = jax.value_and_grad(
        _classification.softmax_cross_entropy
    )(self.ys[0], self.ts[0])
    val_and_grad = jax.value_and_grad(
        _classification.safe_softmax_cross_entropy
    )(self.ys[0], self.ts[0])
    chex.assert_trees_all_close(plain_val_and_grad, val_and_grad, atol=1e-4)


class SoftmaxCrossEntropyWithIntegerLabelsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = np.array([[10.0, 1.0, -2.0], [1.0, 4.0, 0.2]], dtype=np.float32)
    self.ts = np.array([1, 0], dtype=np.int32)

  @chex.all_variants
  def test_consistent_with_softmax_cross_entropy_scalar(self):
    """Tests for a scalar."""
    exp = _classification.softmax_cross_entropy(
        self.ys[0], jax.nn.one_hot(self.ts[0], 3)
    )
    np.testing.assert_allclose(
        self.variant(_classification.softmax_cross_entropy_with_integer_labels)(
            self.ys[0], self.ts[0]
        ),
        exp,
        rtol=1e-6,
    )

  @chex.all_variants
  def test_consistent_with_softmax_cross_entropy_batched(self):
    """Tests for a full batch."""
    exp = _classification.softmax_cross_entropy(
        self.ys, jax.nn.one_hot(self.ts, 3)
    )
    np.testing.assert_allclose(
        self.variant(_classification.softmax_cross_entropy_with_integer_labels)(
            self.ys, self.ts
        ),
        exp,
        rtol=1e-6,
    )

  def test_gradient(self):
    """Tests gradient ok."""
    jaxtest.check_grads(
        functools.partial(
            _classification.softmax_cross_entropy_with_integer_labels,
            labels=self.ts,
        ),
        (self.ys,),
        order=1,
    )

  @parameterized.parameters(
      dict(axis=0, shape=[4, 5, 6]),
      dict(axis=1, shape=[4, 5, 6]),
      dict(axis=2, shape=[4, 5, 6]),
  )
  def test_axis(self, shape, axis):
    preds = np.random.normal(size=shape)
    targets = np.random.randint(
        shape[axis], size=shape[:axis] + shape[axis + 1 :]
    )
    f = _classification.softmax_cross_entropy_with_integer_labels
    x = f(preds, targets, axis=axis)
    y = f(
        np.moveaxis(preds, axis, -1),
        targets,
    )
    np.testing.assert_allclose(x, y, atol=1e-4)


class SigmoidCrossEntropyTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          preds=np.array([-1e09, -1e-09]),
          labels=np.array([1.0, 0.0]),
          expected=5e08,
      ),
      dict(
          preds=np.array([-1e09, -1e-09]),
          labels=np.array([0.0, 1.0]),
          expected=0.3465736,
      ),
      dict(
          preds=np.array([1e09, 1e-09]),
          labels=np.array([1.0, 0.0]),
          expected=0.3465736,
      ),
      dict(
          preds=np.array([1e09, 1e-09]),
          labels=np.array([0.0, 1.0]),
          expected=5e08,
      ),
      dict(
          preds=np.array([-1e09, 1e-09]),
          labels=np.array([1.0, 0.0]),
          expected=5e08,
      ),
      dict(
          preds=np.array([-1e09, 1e-09]),
          labels=np.array([0.0, 1.0]),
          expected=0.3465736,
      ),
      dict(
          preds=np.array([1e09, -1e-09]),
          labels=np.array([1.0, 0.0]),
          expected=0.3465736,
      ),
      dict(
          preds=np.array([1e09, -1e-09]),
          labels=np.array([0.0, 1.0]),
          expected=5e08,
      ),
      dict(
          preds=np.array([0.0, 0.0]),
          labels=np.array([1.0, 0.0]),
          expected=0.6931472,
      ),
      dict(
          preds=np.array([0.0, 0.0]),
          labels=np.array([0.0, 1.0]),
          expected=0.6931472,
      ),
  )
  def testSigmoidCrossEntropy(self, preds, labels, expected):
    tested = jnp.mean(
        _classification.sigmoid_binary_cross_entropy(preds, labels)
    )
    np.testing.assert_allclose(tested, expected, rtol=1e-6, atol=1e-6)


class PolyLossTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.logits = np.array([0.14, 1.456, 2.356, -0.124, -2.47])
    self.labels = np.array([0.1, 0.15, 0.2, 0.25, 0.3])

    self.batched_logits = np.array([[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]])
    self.batched_labels = np.array([[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]])
    # all expected values are computed using tf version of `poly1_cross_entropy`
    # see page 10 here https://arxiv.org/pdf/2204.12511.pdf for more

  @chex.all_variants
  @parameterized.parameters(
      dict(eps=2, expected=4.5317),
      dict(eps=1, expected=3.7153),
      dict(eps=-1, expected=2.0827),
      dict(eps=0, expected=2.8990),
      dict(eps=-0.5, expected=2.4908),
      dict(eps=1.15, expected=3.8378),
      dict(eps=1.214, expected=3.8900),
      dict(eps=5.45, expected=7.3480),
  )
  def test_scalar(self, eps, expected):
    np.testing.assert_allclose(
        self.variant(_classification.poly_loss_cross_entropy)(
            self.logits, self.labels, epsilon=eps
        ),
        expected,
        atol=1e-4,
    )

  @chex.all_variants
  @parameterized.parameters(
      dict(eps=2, expected=np.array([0.4823, 1.2567])),
      dict(eps=1, expected=np.array([0.3261, 1.0407])),
      dict(eps=0, expected=np.array([0.1698, 0.8247])),
      dict(eps=-0.5, expected=np.array([0.0917, 0.7168])),
      dict(eps=1.15, expected=np.array([0.3495, 1.0731])),
      dict(eps=1.214, expected=np.array([0.3595, 1.0870])),
      dict(eps=5.45, expected=np.array([1.0211, 2.0018])),
  )
  def test_batched(self, eps, expected):
    np.testing.assert_allclose(
        self.variant(_classification.poly_loss_cross_entropy)(
            self.batched_logits, self.batched_labels, epsilon=eps
        ),
        expected,
        atol=1e-4,
    )

  @chex.all_variants
  @parameterized.parameters(
      dict(
          logits=np.array(
              [[4.0, 2.0, 1.0], [0.0, 5.0, 1.0], [0.134, 1.234, 3.235]]
          ),
          labels=np.array(
              [[1.0, 0.0, 0.0], [0.0, 0.8, 0.2], [0.34, 0.33, 0.33]]
          ),
      ),
      dict(
          logits=np.array([[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]]),
          labels=np.array([[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]]),
      ),
      dict(
          logits=np.array(
              [[4.0, 2.0, 1.0, 0.134, 1.3515], [0.0, 5.0, 1.0, 0.5215, 5.616]]
          ),
          labels=np.array(
              [[0.5, 0.0, 0.0, 0.0, 0.5], [0.0, 0.12, 0.2, 0.56, 0.12]]
          ),
      ),
      dict(logits=np.array([1.89, 2.39]), labels=np.array([0.34, 0.66])),
      dict(logits=np.array([0.314]), labels=np.array([1.0])),
  )
  def test_equals_to_cross_entropy_when_eps0(self, logits, labels):
    np.testing.assert_allclose(
        self.variant(_classification.poly_loss_cross_entropy)(
            logits, labels, epsilon=0.0
        ),
        self.variant(_classification.softmax_cross_entropy)(logits, labels),
        atol=1e-4,
    )

  @parameterized.parameters(dict(size=5), dict(size=10))
  def test_mask(self, size):
    preds = np.random.normal(size=size)
    targets = np.random.dirichlet(np.ones(size))
    mask = np.random.randint(2, size=size, dtype=bool)
    x = _classification.poly_loss_cross_entropy(preds[mask], targets[mask])
    y = _classification.poly_loss_cross_entropy(preds, targets, where=mask)
    np.testing.assert_allclose(x, y, atol=1e-4)

  @parameterized.parameters(
      dict(axis=0, shape=[4, 5, 6]),
      dict(axis=1, shape=[4, 5, 6]),
      dict(axis=2, shape=[4, 5, 6]),
  )
  def test_axis(self, shape, axis):
    preds = np.random.normal(size=shape)
    targets = np.random.dirichlet(np.ones(shape[-1]), size=shape[:-1])
    x = _classification.poly_loss_cross_entropy(preds, targets, axis=axis)
    y = _classification.poly_loss_cross_entropy(
        np.moveaxis(preds, axis, -1),
        np.moveaxis(targets, axis, -1),
    )
    np.testing.assert_allclose(x, y, atol=1e-4)


class HingeTest(parameterized.TestCase):

  def test_binary(self):
    label = jnp.array(1)
    signed_label = jnp.array(2.0 * label - 1.0)
    score = jnp.array(10.0)

    def reference_impl(label, logit):
      return jax.nn.relu(1 - logit * (2.0 * label - 1.0))

    expected = reference_impl(label, score)
    result = _classification.hinge_loss(score, signed_label)
    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_batched_binary(self):
    labels = jnp.array([1, 0])
    signed_labels = jnp.array(2.0 * labels - 1.0)
    scores = jnp.array([10.0, 20.0])

    def reference_impl(label, logit):
      return jax.nn.relu(1 - logit * (2.0 * label - 1.0))

    expected = jax.vmap(reference_impl)(labels, scores)
    # no need to vmap the optax loss. leading dimensions automatically handled.
    result = _classification.hinge_loss(scores, signed_labels)
    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_multi_class(self):
    label = jnp.array(1)
    scores = jnp.array([10.0, 3.0])

    def reference_impl(label, scores):
      one_hot_label = jax.nn.one_hot(label, scores.shape[-1])
      return jnp.max(scores + 1.0 - one_hot_label) - scores[label]

    expected = reference_impl(label, scores)
    result = _classification.multiclass_hinge_loss(scores, label)
    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_batched_multi_class(self):
    label = jnp.array([1, 0])
    scores = jnp.array([[10.0, 3.0], [11.0, -2.0]])

    def reference_impl(label, scores):
      one_hot_label = jax.nn.one_hot(label, scores.shape[-1])
      return jnp.max(scores + 1.0 - one_hot_label) - scores[label]

    expected = jax.vmap(reference_impl)(label, scores)
    # no need to vmap the optax loss. leading dimensions automatically handled.
    result = _classification.multiclass_hinge_loss(scores, label)
    np.testing.assert_allclose(result, expected, atol=1e-4)


class SparsemaxTest(parameterized.TestCase):

  def test_binary(self):
    label = 1
    score = 10.0

    def reference_impl(label, logit):
      scores = -(2 * label - 1) * logit
      if scores <= -1.0:
        return 0.0
      elif scores >= 1.0:
        return scores
      else:
        return (scores + 1.0) ** 2 / 4

    expected = reference_impl(label, score)
    result = _classification.sparsemax_loss(
        jnp.asarray(score), jnp.asarray(label)
    )
    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_batched_binary(self):
    labels = jnp.array([1, 0])
    scores = jnp.array([10.0, 20.0])

    def reference_impl(label, logit):
      scores = -(2 * label - 1) * logit
      if scores <= -1.0:
        return 0.0
      elif scores >= 1.0:
        return scores
      else:
        return (scores + 1.0) ** 2 / 4

    expected = jnp.asarray([
        reference_impl(labels[0], scores[0]),
        reference_impl(labels[1], scores[1]),
    ])
    # in the optax loss the leading dimensions are automatically handled.
    result = _classification.sparsemax_loss(scores, labels)
    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_multi_class_zero_loss(self):
    # Check that putting large scores on the correct class gives a zero loss.
    labels = jnp.array([1, 0, 2])
    scores = jnp.array([[0.0, 1e5, 0.0], [1e5, 0.0, 0.0], [0.0, 0.0, 1e5]])
    losses = _classification.multiclass_sparsemax_loss(scores, labels)
    np.testing.assert_allclose(losses, np.array([0.0, 0.0, 0.0]), atol=1e-4)

  def test_multi_class_gradient(self):
    # Check that the gradient is correct.
    def loss_mean(scores, labels):
      return jnp.mean(_classification.multiclass_sparsemax_loss(scores, labels))

    labels = jnp.array([1, 0, 2])
    scores = jnp.array([[0.0, 1e5, 0.0], [1e5, 0.0, 0.0], [0.0, 0.0, 1e5]])
    grad = jax.grad(loss_mean)(scores, labels)
    projection_vmap = jax.vmap(projections.projection_simplex)
    grad_expected = projection_vmap(scores) - jax.nn.one_hot(labels, 3)
    np.testing.assert_allclose(grad, grad_expected, atol=1e-4)


class ConvexKLDivergenceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.log_ps = np.array([
        [-2.9957, -3.5066, -3.9120, -1.2040, -0.6931, -2.3026],
        [-1.6094, -1.6094, -1.6094, -2.3026, -1.8971, -1.8971],
    ])
    self.qs = np.array(
        [[0.2, 0.2, 0.2, 0.1, 0.15, 0.15], [0.05, 0.03, 0.02, 0.3, 0.5, 0.0]]
    )

    # Computed convex kullback-leibler divergence of P from Q.
    self.exp = np.array([0.88757247, 0.859308])

  @chex.all_variants
  def test_scalar(self):
    np.testing.assert_allclose(
        self.variant(_classification.convex_kl_divergence)(
            self.log_ps[0], self.qs[0]
        ),
        self.exp[0],
        atol=1e-4,
    )

  @chex.all_variants
  def test_batched(self):
    np.testing.assert_allclose(
        self.variant(_classification.convex_kl_divergence)(
            self.log_ps, self.qs
        ),
        self.exp,
        atol=1e-4,
    )

  @parameterized.parameters(dict(size=5), dict(size=10))
  def test_mask(self, size):
    preds = np.random.normal(size=size)
    targets = np.random.dirichlet(np.ones(size))
    mask = np.random.randint(2, size=size, dtype=bool)
    x = _classification.convex_kl_divergence(preds[mask], targets[mask])
    y = _classification.convex_kl_divergence(preds, targets, where=mask)
    np.testing.assert_allclose(x, y, atol=1e-4)

  @parameterized.parameters(
      dict(axis=0, shape=[4, 5, 6]),
      dict(axis=1, shape=[4, 5, 6]),
      dict(axis=2, shape=[4, 5, 6]),
  )
  def test_axis(self, shape, axis):
    preds = np.random.normal(size=shape)
    targets = np.random.dirichlet(np.ones(shape[-1]), size=shape[:-1])
    x = _classification.convex_kl_divergence(preds, targets, axis=axis)
    y = _classification.convex_kl_divergence(
        np.moveaxis(preds, axis, -1),
        np.moveaxis(targets, axis, -1),
    )
    np.testing.assert_allclose(x, y, atol=1e-4)


class PerceptronTest(parameterized.TestCase):

  def test_binary(self):
    label = jnp.array(1)
    signed_label = jnp.array(2.0 * label - 1.0)
    score = jnp.array(10.0)

    def reference_impl(label, logit) -> float:
      return jax.nn.relu(-logit * (2.0 * label - 1.0))

    expected = reference_impl(label, score)
    result = _classification.perceptron_loss(score, signed_label)
    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_batched_binary(self):
    labels = jnp.array([1, 0])
    signed_labels = jnp.array(2.0 * labels - 1.0)
    scores = jnp.array([10.0, 20.0])

    def reference_impl(label, logit) -> float:
      return jax.nn.relu(-logit * (2.0 * label - 1.0))

    expected = jax.vmap(reference_impl)(labels, scores)
    # no need to vmap the optax loss. leading dimensions automatically handled.
    result = _classification.perceptron_loss(scores, signed_labels)
    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_multi_class(self):
    label = jnp.array(1)
    scores = jnp.array([10.0, 3.0])

    def reference_impl(label, scores):
      return jnp.max(scores) - scores[label]

    expected = reference_impl(label, scores)
    result = _classification.multiclass_perceptron_loss(scores, label)
    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_batched_multi_class(self):
    label = jnp.array([1, 0])
    scores = jnp.array([[10.0, 3.0], [11.0, -2.0]])

    def reference_impl(label, scores):
      return jnp.max(scores) - scores[label]

    expected = jax.vmap(reference_impl)(label, scores)
    # no need to vmap the optax loss. leading dimensions automatically handled.
    result = _classification.multiclass_perceptron_loss(scores, label)
    np.testing.assert_allclose(result, expected, atol=1e-4)


class KLDivergenceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.log_ps = np.array([
        [-2.9957, -3.5066, -3.9120, -1.2040, -0.6931, -2.3026],
        [-1.6094, -1.6094, -1.6094, -2.3026, -1.8971, -1.8971],
    ])
    self.qs = np.array(
        [[0.2, 0.2, 0.2, 0.1, 0.15, 0.15], [0.05, 0.03, 0.02, 0.3, 0.5, 0.0]]
    )
    # Computed kullback-leibler divergence of P from Q.
    self.exp = np.array([0.8875577, 0.7592807])

  @chex.all_variants
  def test_scalar(self):
    np.testing.assert_allclose(
        self.variant(_classification.kl_divergence)(self.log_ps[0], self.qs[0]),
        self.exp[0],
        atol=1e-4,
    )

  @chex.all_variants
  def test_batched(self):
    np.testing.assert_allclose(
        self.variant(_classification.kl_divergence)(self.log_ps, self.qs),
        self.exp,
        atol=1e-4,
    )

  @parameterized.parameters(dict(size=5), dict(size=10))
  def test_mask(self, size):
    preds = np.random.normal(size=size)
    targets = np.random.dirichlet(np.ones(size))
    mask = np.random.randint(2, size=size, dtype=bool)
    x = _classification.kl_divergence(preds[mask], targets[mask])
    y = _classification.kl_divergence(preds, targets, where=mask)
    np.testing.assert_allclose(x, y, atol=1e-4)

  @parameterized.parameters(
      dict(axis=0, shape=[4, 5, 6]),
      dict(axis=1, shape=[4, 5, 6]),
      dict(axis=2, shape=[4, 5, 6]),
  )
  def test_axis(self, shape, axis):
    preds = np.random.normal(size=shape)
    targets = np.random.dirichlet(np.ones(shape[-1]), size=shape[:-1])
    x = _classification.kl_divergence(preds, targets, axis=axis)
    y = _classification.kl_divergence(
        np.moveaxis(preds, axis, -1),
        np.moveaxis(targets, axis, -1),
    )
    np.testing.assert_allclose(x, y, atol=1e-4)


class KLDivergenceWithLogTargetsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.log_ps = np.array([
        [-2.9957, -3.5066, -3.9120, -1.2040, -0.6931, -2.3026],
        [-1.6094, -1.6094, -1.6094, -2.3026, -1.8971, -1.8971],
    ])
    self.qs = np.array([
        [-1.6094, -1.6094, -1.6094, -2.3026, -1.8971, -1.8971],
        [-2.9957, -3.5066, -3.9120, -1.2040, -0.6931, -2.3026],
    ])
    # Computed kullback-leibler divergence of P from Q.
    self.exp = np.array([0.8875625, 0.7187435584901326])

  @chex.all_variants
  def test_scalar(self):
    np.testing.assert_allclose(
        self.variant(_classification.kl_divergence_with_log_targets)(
            self.log_ps[0], self.qs[0]
        ),
        self.exp[0],
        atol=1e-4,
    )

  @chex.all_variants
  def test_batched(self):
    np.testing.assert_allclose(
        self.variant(_classification.kl_divergence_with_log_targets)(
            self.log_ps, self.qs
        ),
        self.exp,
        atol=1e-4,
    )

  @parameterized.parameters(dict(size=5), dict(size=10))
  def test_mask(self, size):
    preds = np.random.normal(size=size)
    targets = np.log(np.random.dirichlet(np.ones(size)))
    mask = np.random.randint(2, size=size, dtype=bool)
    f = _classification.kl_divergence_with_log_targets
    x = f(preds[mask], targets[mask])
    y = f(preds, targets, where=mask)
    np.testing.assert_allclose(x, y, atol=1e-4)

  @parameterized.parameters(
      dict(axis=0, shape=[4, 5, 6]),
      dict(axis=1, shape=[4, 5, 6]),
      dict(axis=2, shape=[4, 5, 6]),
  )
  def test_axis(self, shape, axis):
    preds = np.random.normal(size=shape)
    targets = np.log(np.random.dirichlet(np.ones(shape[-1]), size=shape[:-1]))
    f = _classification.kl_divergence_with_log_targets
    x = f(preds, targets, axis=axis)
    y = f(
        np.moveaxis(preds, axis, -1),
        np.moveaxis(targets, axis, -1),
    )
    np.testing.assert_allclose(x, y, atol=1e-4)


def _lengths_to_paddings(lengths: chex.Array, maxlength: int) -> chex.Array:
  indices = jnp.arange(maxlength).reshape((1,) * lengths.ndim + (maxlength,))
  lengths = jnp.expand_dims(lengths, axis=-1)
  elem_valid = indices < lengths
  return np.logical_not(elem_valid).astype(np.float32)


def _average_ctc_loss(
    logprobs: chex.Array,
    logprob_paddings: chex.Array,
    labels: chex.Array,
    label_paddings: chex.Array,
) -> chex.Array:
  return jnp.average(
      _classification.ctc_loss(
          logprobs, logprob_paddings, labels, label_paddings
      )
  )


class CTCTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(1234)
    self._rtol = 5e-3 if jax.default_backend() != 'cpu' else 1e-6

  @chex.all_variants
  def test_with_one_to_one_alignment(self):
    # when inputsteps and outputsteps are equal, no blank will be allowed.
    batchsize = 8
    steps = 50
    nclasses = 40
    logits = np.random.randn(batchsize, steps, nclasses)
    labels = np.random.uniform(1, nclasses, size=(batchsize, steps)).astype(
        np.int32
    )

    # This function only covers the cases without same-label repetition.
    # `test_repeat_with_one_to_one_alignment` below complements those cases.
    # So, redraw the samples for satisfying the non-repetition constraint.
    for n in range(labels.shape[0]):
      for t in range(1, labels.shape[1]):
        while labels[n, t] == labels[n, t - 1]:
          labels[n, t] = np.random.uniform(1, nclasses)

    results = self.variant(_classification.ctc_loss_with_forward_probs)(
        logits, np.zeros(logits.shape[:2]), labels, np.zeros(labels.shape)
    )
    (per_seq_loss, logalpha_blank, logalpha_emit) = results

    logprobs = jax.nn.log_softmax(logits)
    for b in range(batchsize):
      p = 0.0
      for t in range(steps):
        p += logprobs[b, t, labels[b, t]]
      np.testing.assert_allclose(np.array(-p), per_seq_loss[b], rtol=self._rtol)

      # Check forward-probabilities.
      # 1. All-phi path: logalpha_blank[-1, b, 0] must be a probability of
      #   the path that outputs blank symbols for all the frames.
      np.testing.assert_allclose(
          logalpha_blank[-1, b, 0], np.sum(logprobs[b, :, 0]), rtol=self._rtol
      )

      # 2. After emitting all the labels
      #   the negated loss must be identical with the forward probability of
      #   paths after consuming all the labels (because one-to-one alignment
      #   doesn't allow extra blank symbols)
      np.testing.assert_allclose(
          logalpha_emit[-1, b, steps - 1], -per_seq_loss[b], rtol=self._rtol
      )
      #   and, this forward probability must be copied to the blank forward
      #   probability of the next step.
      np.testing.assert_allclose(
          logalpha_blank[-1, b, steps], -per_seq_loss[b], rtol=self._rtol
      )

  @chex.all_variants
  def test_with_one_to_one_alignment_and_paddings(self):
    batch_size = 5
    nclasses = 13
    steps = 7
    logits = np.random.normal(size=[batch_size, steps, nclasses])
    logprobs = jax.nn.log_softmax(logits)

    labels = []
    for _ in range(batch_size):
      row = list(range(1, nclasses))
      np.random.shuffle(row)
      labels.append(row[:steps])
    labels = np.array(labels)

    lengths = np.random.randint(3, 6, size=(batch_size,))
    paddings = _lengths_to_paddings(lengths, steps)

    actual_loss = self.variant(_classification.ctc_loss)(
        logits, paddings, labels, paddings
    )

    value_and_grad = self.variant(jax.value_and_grad(_average_ctc_loss))
    unused_avg_loss, actual_gradients = value_and_grad(
        logits, paddings, labels, paddings
    )

    for n in range(batch_size):
      expected_loss = -sum(
          logprobs[n, t, k] for t, k in enumerate(labels[n, : lengths[n]])
      )
      np.testing.assert_allclose(expected_loss, actual_loss[n], rtol=self._rtol)

      expected_gradients = np.array(jax.nn.softmax(logits[n]))
      expected_gradients[lengths[n] :] = 0.0
      for t, k in enumerate(labels[n, : lengths[n]]):
        expected_gradients[t, k] -= 1.0
      expected_gradients /= batch_size
      np.testing.assert_allclose(
          expected_gradients, actual_gradients[n], rtol=self._rtol
      )

  @chex.all_variants
  def test_repeat_with_one_to_one_alignment(self):
    # test if it can correctly handle the same-label repetition.
    nclasses = 5
    labels = np.array([
        [1, 2, 2, 3],
        [2, 3, 4, 4],
        [1, 1, 1, 1],
        [1, 1, 2, 3],
        [1, 1, 1, 2],
    ])
    expected_alignment = [  # expected minimal alignment
        [1, 2, 0, 2, 3],
        [2, 3, 4, 0, 4],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 2, 3],
        [1, 0, 1, 0, 1, 2],
    ]
    batch_size = len(labels)
    label_lens = np.array([4] * batch_size)
    label_steps = 6
    # Designed to have two padding elements on the right.
    labels = np.pad(labels, [(0, 0), (0, label_steps - labels.shape[1])])
    label_paddings = _lengths_to_paddings(label_lens, label_steps)

    logit_lengths = np.array([len(seq) for seq in expected_alignment])
    logit_steps = max(logit_lengths)
    logits = np.random.randn(batch_size, logit_steps, nclasses)
    logit_paddings = _lengths_to_paddings(logit_lengths, logit_steps)

    per_seq_loss = self.variant(_classification.ctc_loss)(
        logits, logit_paddings, labels, label_paddings
    )

    logprobs = jax.nn.log_softmax(logits)
    for n in range(batch_size):
      expected_loss = -sum(
          logprobs[n, t, k] for t, k in enumerate(expected_alignment[n])
      )
      np.testing.assert_allclose(
          jnp.array(expected_loss), per_seq_loss[n], rtol=self._rtol
      )


class SigmoidFocalLossTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = np.array([[2.0, 0.1, -2.0], [0.3, -0.1, 1.2]], dtype=np.float32)
    self.ts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    self._rtol = 5e-3 if jax.default_backend() != 'cpu' else 1e-6

    logit = lambda x: jnp.log(x / (1.0 - x))
    self.large_ys = logit(jnp.array([0.9, 0.98, 0.3, 0.99]))
    self.small_ys = logit(jnp.array([0.1, 0.02, 0.09, 0.15]))
    self.ones_ts = jnp.array([1.0, 1.0, 1.0, 1.0])

  @chex.all_variants
  def test_focal_equals_ce(self):
    """If gamma == 0 and alpha == 0 we expect a CE loss."""
    np.testing.assert_allclose(
        self.variant(_classification.sigmoid_focal_loss)(
            self.ys, self.ts, gamma=0.0
        ),
        _classification.sigmoid_binary_cross_entropy(self.ys, self.ts),
        rtol=self._rtol,
    )

  @chex.all_variants
  def test_scale(self):
    """This test should catch problems with p_t."""
    gamma = 2
    focal_loss = self.variant(_classification.sigmoid_focal_loss)(
        self.ys, self.ts, gamma=gamma
    )
    p = jax.nn.sigmoid(self.ys)
    ce_loss = _classification.sigmoid_binary_cross_entropy(self.ys, self.ts)
    p_t = p * self.ts + (1 - p) * (1 - self.ts)
    scale = (1 - p_t) ** gamma
    focal_scale = focal_loss / ce_loss
    np.testing.assert_allclose(focal_scale, scale, rtol=self._rtol)

  @chex.all_variants
  def test_large_logit_fl_less_than_ce(self):
    """If gamma == 2 and alpha == 0.5, the impact of large logits is reduced."""
    focal_loss = self.variant(_classification.sigmoid_focal_loss)(
        self.large_ys, self.ones_ts, gamma=2, alpha=0.5
    )
    ce_loss = _classification.sigmoid_binary_cross_entropy(
        self.large_ys, self.ones_ts
    )
    loss_ratio = ce_loss / focal_loss
    expected_ratio = 2.0 / ((1.0 - jax.nn.sigmoid(self.large_ys)) ** 2)
    np.testing.assert_allclose(loss_ratio, expected_ratio, rtol=self._rtol)

  @chex.all_variants
  def test_small_logit_fl_less_than_ce(self):
    """If gamma == 2, small logits retain their weight."""
    focal_loss = self.variant(_classification.sigmoid_focal_loss)(
        self.small_ys, self.ones_ts, gamma=2
    )
    ce_loss = _classification.sigmoid_binary_cross_entropy(
        self.small_ys, self.ones_ts
    )
    loss_ratio = ce_loss / focal_loss
    expected_ratio = 1.0 / ((1.0 - jax.nn.sigmoid(self.small_ys)) ** 2)
    np.testing.assert_allclose(loss_ratio, expected_ratio, rtol=self._rtol)

  @chex.all_variants
  def test_alpha_one(self):
    """Test if re-weighting with alpha=1 is ok."""
    np.testing.assert_allclose(
        self.variant(_classification.sigmoid_focal_loss)(
            self.ys, self.ts, gamma=0.0, alpha=1
        ),
        _classification.sigmoid_binary_cross_entropy(self.ys, self.ts)
        * self.ts,
        rtol=self._rtol,
    )

  @chex.all_variants
  def test_ignore_positive(self):
    """If alpha == 0 positive examples do not matter."""
    focal_loss = self.variant(_classification.sigmoid_focal_loss)(
        self.ys, self.ts, alpha=0
    )
    ce_loss = _classification.sigmoid_binary_cross_entropy(self.ys, self.ts)
    assert all(ce_loss[self.ts == 1] > 0)
    assert all(focal_loss[self.ts == 1] == 0)

  @chex.all_variants
  def test_ignore_negative(self):
    """If alpha == 1 negative examples do not matter."""
    focal_loss = self.variant(_classification.sigmoid_focal_loss)(
        self.ys, self.ts, alpha=1
    )
    ce_loss = _classification.sigmoid_binary_cross_entropy(self.ys, self.ts)
    assert all(ce_loss[self.ts == 0] > 0)
    assert all(focal_loss[self.ts == 0] == 0)


if __name__ == '__main__':
  absltest.main()
