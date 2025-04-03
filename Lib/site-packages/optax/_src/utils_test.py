# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for utilities in `utils.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import alias
from optax._src import combine
from optax._src import linesearch
from optax._src import transform
from optax._src import update
from optax._src import utils


def _shape_to_tuple(shape):
  if isinstance(shape, tuple):
    return shape
  return tuple([shape])


class ScaleGradientTest(parameterized.TestCase):

  @parameterized.product(scale=[-0.5, 0.0, 0.5, 1.0, 2.0])
  def test_scale_gradient_pytree(self, scale):
    def fn(inputs):
      outputs = utils.scale_gradient(inputs, scale)
      outputs = jax.tree.map(lambda x: x**2, outputs)
      return sum(jax.tree.leaves(outputs))

    inputs = dict(a=-1.0, b=dict(c=(2.0,), d=0.0))

    grad = jax.grad(fn)
    grads = grad(inputs)
    jax.tree.map(lambda i, g: self.assertEqual(g, 2 * i * scale), inputs, grads)
    self.assertEqual(
        fn(inputs),
        sum(jax.tree.leaves(jax.tree.map(lambda x: x**2, inputs))),
    )


class MultiNormalDiagFromLogScaleTest(parameterized.TestCase):

  def _get_loc_scale(self, loc_shape, scale_shape):
    loc = 1.5 * jnp.ones(shape=loc_shape, dtype=jnp.float32)
    scale = 0.5 * jnp.ones(shape=scale_shape, dtype=jnp.float32)
    return loc, scale

  @parameterized.parameters(
      (1, 1, 1),
      (5, 5, 5),
      ((2, 3), (2, 3), (2, 3)),
      ((1, 4), (3, 4), (3, 4)),
      ((1, 2, 1, 3), (2, 1, 4, 3), (2, 2, 4, 3)),
  )
  def test_init_successful_broadcast(
      self, loc_shape, scale_shape, broadcasted_shape
  ):
    loc, scale = self._get_loc_scale(loc_shape, scale_shape)
    dist = utils.multi_normal(loc, scale)
    self.assertIsInstance(dist, utils.MultiNormalDiagFromLogScale)
    mean, log_scale = dist.params
    self.assertEqual(tuple(mean.shape), _shape_to_tuple(loc_shape))
    self.assertEqual(tuple(log_scale.shape), _shape_to_tuple(scale_shape))
    self.assertEqual(
        tuple(dist._param_shape), _shape_to_tuple(broadcasted_shape)
    )

  @parameterized.parameters(
      (2, 3),
      ((2, 3), (3, 2)),
      ((2, 4), (3, 4)),
      ((1, 2, 1, 3), (2, 1, 4, 4)),
  )
  def test_init_unsuccessful_broadcast(self, loc_shape, scale_shape):
    loc, scale = self._get_loc_scale(loc_shape, scale_shape)
    with self.assertRaisesRegex(
        ValueError, 'Incompatible shapes for broadcasting'
    ):
      utils.multi_normal(loc, scale)

  @parameterized.parameters(list, tuple)
  def test_sample_input_sequence_types(self, sample_type):
    sample_shape = sample_type((4, 5))
    loc_shape = scale_shape = (2, 3)
    loc, scale = self._get_loc_scale(loc_shape, scale_shape)
    dist = utils.multi_normal(loc, scale)
    samples = dist.sample(sample_shape, jax.random.PRNGKey(239))
    self.assertEqual(samples.shape, tuple(sample_shape) + loc_shape)

  @parameterized.named_parameters([
      ('1d', 1),
      ('2d', (2, 3)),
      ('4d', (1, 2, 3, 4)),
  ])
  def test_log_prob(self, shape):
    loc, scale = self._get_loc_scale(shape, shape)
    dist = utils.multi_normal(loc, scale)
    probs = dist.log_prob(jnp.ones(shape=shape, dtype=jnp.float32))
    self.assertEqual(probs.shape, ())


class HelpersTest(chex.TestCase):

  @parameterized.parameters([
      (1, 1),
      (3, 3),
      (1, 3),
      (2, 3),
  ])
  def test_set_diags_valid(self, n, d):
    def _all_but_diag(matrix):
      return matrix - jnp.diag(jnp.diag(matrix))

    a = jnp.ones(shape=(n, d, d)) * 10
    new_diags = jnp.arange(n * d).reshape((n, d))
    res = utils.set_diags(a, new_diags)
    for i in range(n):
      np.testing.assert_array_equal(jnp.diag(res[i]), new_diags[i])
      np.testing.assert_array_equal(_all_but_diag(res[i]), _all_but_diag(a[i]))

  @parameterized.named_parameters([
      ('1d', 1),
      ('2d', (2, 3)),
      ('4d', (1, 2, 3, 4)),
  ])
  def test_set_diag_a_raises(self, a_shape):
    a = jnp.ones(shape=a_shape)
    new_diags = jnp.zeros(shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'Expected `a` to be a 3D tensor'):
      utils.set_diags(a, new_diags)

  @parameterized.named_parameters([
      ('1d', 1),
      ('3d', (2, 3, 4)),
      ('4d', (1, 2, 3, 4)),
  ])
  def test_set_diag_new_diags_raises(self, new_diags_shape):
    a = jnp.ones(shape=(3, 2, 2))
    new_diags = jnp.zeros(shape=new_diags_shape)
    with self.assertRaisesRegex(
        ValueError, 'Expected `new_diags` to be a 2D array'
    ):
      utils.set_diags(a, new_diags)

  @parameterized.parameters([
      (1, 1, 2),
      (3, 3, 4),
      (1, 3, 5),
      (2, 3, 2),
  ])
  def test_set_diag_a_shape_mismatch_raises(self, n, d, d1):
    a = jnp.ones(shape=(n, d, d1))
    new_diags = jnp.zeros(shape=(n, d))
    with self.assertRaisesRegex(
        ValueError, 'Shape mismatch: expected `a.shape`'
    ):
      utils.set_diags(a, new_diags)

  @parameterized.parameters([
      (1, 1, 1, 3),
      (3, 3, 4, 3),
      (1, 3, 1, 5),
      (2, 3, 6, 7),
  ])
  def test_set_diag_new_diags_shape_mismatch_raises(self, n, d, n1, d1):
    a = jnp.ones(shape=(n, d, d))
    new_diags = jnp.zeros(shape=(n1, d1))
    with self.assertRaisesRegex(
        ValueError, 'Shape mismatch: expected `new_diags.shape`'
    ):
      utils.set_diags(a, new_diags)

  @parameterized.named_parameters([
      ('1d-single', 1),
      ('1d', 10),
      ('2d', (1, 2)),
      ('3d', (10, 3, 2)),
      ('4d', (2, 3, 4, 5)),
      ('6d', (1, 2, 3, 4, 5, 6)),
      ('8d', (5, 4, 7, 6, 1, 2, 3, 1)),
  ])
  def test_tile_second_to_last_dim(self, shape):
    shape = _shape_to_tuple(shape)
    elems = jnp.prod(jnp.array(shape))
    matrix = jnp.arange(elems).reshape(shape)
    result = utils.tile_second_to_last_dim(matrix)
    self.assertEqual(result.shape, shape + (shape[-1],))
    np.testing.assert_array_equal(result[..., -1], matrix)
    np.testing.assert_array_equal(result[..., 0], matrix)

  @parameterized.parameters([
      (None, None),
      (jnp.float32, np.dtype('float32')),
      (jnp.int32, np.dtype('int32')),
      (jnp.bfloat16, np.dtype('bfloat16')),
  ])
  def test_canonicalize_dtype(self, dtype, expected_dtype):
    canonical = utils.canonicalize_dtype(dtype)
    self.assertIs(canonical, expected_dtype)

  @chex.variants(
      with_jit=True,
      without_jit=True,
      with_pmap=False,
      with_device=True,
      without_device=True,
  )
  def test_value_and_grad_from_state(self):
    def fn(x):
      return jnp.sum(x**2)

    value_and_grad_ = utils.value_and_grad_from_state(fn)

    value_and_grad = self.variant(value_and_grad_)

    params = jnp.array([1.0, 2.0, 3.0])

    # No value and grad in this transform so it should raise an error
    opt = transform.scale_by_adam()
    state = opt.init(params)
    self.assertRaises(ValueError, value_and_grad, params, state=state)

    # Multiple values and grads in this transform so it should raise an error
    opt = combine.chain(
        linesearch.scale_by_backtracking_linesearch(max_backtracking_steps=15),
        linesearch.scale_by_backtracking_linesearch(max_backtracking_steps=15),
    )
    state = opt.init(params)
    self.assertRaises(KeyError, value_and_grad, params, state=state)

    # It should work efficiently when the linesearch stores the gradient
    opt = combine.chain(
        alias.sgd(learning_rate=1.0),
        linesearch.scale_by_backtracking_linesearch(
            max_backtracking_steps=15, store_grad=True
        ),
    )
    state = opt.init(params)
    value, grad = value_and_grad(params, state=state)
    updates, state = opt.update(
        grad, state, params, value=value, grad=grad, value_fn=fn
    )
    params = update.apply_updates(params, updates)
    params = jax.block_until_ready(params)

    def false_fn(_):
      return 1.0

    false_value_and_grad_ = utils.value_and_grad_from_state(false_fn)
    false_value_and_grad = self.variant(false_value_and_grad_)

    # At the second step we should not evaluate the function
    # so in this case it should not return the output of false_fn
    value, _ = false_value_and_grad(params, state=state)
    self.assertNotEqual(value, 1.0)

  def test_extract_fns_kwargs(self):
    def fn1(a, b):
      return a + b

    def fn2(c, d):
      return c + d

    kwargs = {'b': 1.0, 'd': 2.0, 'e': 3.0}
    fns_kwargs, remaining_kwargs = utils._extract_fns_kwargs((fn1, fn2), kwargs)
    self.assertEqual(fns_kwargs, [{'b': 1.0}, {'d': 2.0}])
    self.assertEqual(remaining_kwargs, {'e': 3.0})


if __name__ == '__main__':
  absltest.main()
