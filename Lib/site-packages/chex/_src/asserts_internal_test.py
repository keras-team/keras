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
"""Tests for `asserts_internal.py`."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from chex._src import asserts_internal as ai
from chex._src import variants
import jax
import jax.numpy as jnp


class IsTraceableTest(variants.TestCase):

  @variants.variants(with_jit=True, with_pmap=True)
  def test_is_traceable(self):
    def dummy_wrapper(fn):

      @functools.wraps(fn)
      def fn_wrapped(fn, *args):
        return fn(args)

      return fn_wrapped

    fn = lambda x: x.sum()
    wrapped_fn = dummy_wrapper(fn)
    self.assertFalse(ai.is_traceable(fn))
    self.assertFalse(ai.is_traceable(wrapped_fn))

    var_fn = self.variant(fn)
    wrapped_var_f = dummy_wrapper(var_fn)
    var_wrapped_f = self.variant(wrapped_fn)
    self.assertTrue(ai.is_traceable(var_fn))
    self.assertTrue(ai.is_traceable(wrapped_var_f))
    self.assertTrue(ai.is_traceable(var_wrapped_f))


class ExceptionMessageFormatTest(variants.TestCase):

  @parameterized.product(
      include_default_msg=(False, True),
      include_custom_msg=(False, True),
      exc_type=(AssertionError, ValueError),
  )
  def test_format(self, include_default_msg, include_custom_msg, exc_type):

    exc_msg = lambda x: f'{x} is non-positive.'

    @functools.partial(ai.chex_assertion, jittable_assert_fn=None)
    def assert_positive(x):
      if x <= 0:
        raise AssertionError(exc_msg(x))

    @functools.partial(ai.chex_assertion, jittable_assert_fn=None)
    def assert_each_positive(*args):
      for x in args:
        assert_positive(x)

    # Pass.
    assert_positive(1)
    assert_each_positive(1, 2, 3)

    # Check the format of raised exceptions' messages.
    def expected_exc_msg(x, custom_msg):
      msg = exc_msg(x) if include_default_msg else ''
      msg = rf'{msg} \[{custom_msg}\]' if custom_msg else msg
      return msg

    # Run in a loop to generate different custom messages.
    for i in range(3):
      custom_msg = f'failed at iter {i}' if include_custom_msg else ''

      with self.assertRaisesRegex(
          exc_type, ai.get_err_regex(expected_exc_msg(-1, custom_msg))):
        assert_positive(  # pylint:disable=unexpected-keyword-arg
            -1,
            custom_message=custom_msg,
            include_default_message=include_default_msg,
            exception_type=exc_type)

      with self.assertRaisesRegex(
          exc_type, ai.get_err_regex(expected_exc_msg(-3, custom_msg))):
        assert_each_positive(  # pylint:disable=unexpected-keyword-arg
            1,
            -3,
            2,
            custom_message=custom_msg,
            include_default_message=include_default_msg,
            exception_type=exc_type)


class JitCompatibleTest(variants.TestCase):

  def test_api(self):

    def assert_fn(x):
      if x.shape != (2,):
        raise AssertionError(f'shape != (2,) {x.shape}!')

    for transform_fn in (jax.jit, jax.grad, jax.vmap):
      x_ok = jnp.ones((2,))
      x_wrong = jnp.ones((3,))
      is_vmap = transform_fn is jax.vmap
      if is_vmap:
        x_ok, x_wrong = (jnp.expand_dims(x, 0) for x in (x_ok, x_wrong))

      # Jax-compatible.
      assert_compat_fn = ai.chex_assertion(assert_fn, jittable_assert_fn=None)

      def compat_fn(x, assertion=assert_compat_fn):
        assertion(x)
        return x.sum()

      if not is_vmap:
        compat_fn(x_ok)
      transform_fn(compat_fn)(x_ok)
      with self.assertRaisesRegex(AssertionError, 'shape !='):
        transform_fn(compat_fn)(x_wrong)

      # JAX-incompatible.
      assert_incompat_fn = ai.chex_assertion(
          assert_fn, jittable_assert_fn=assert_fn)

      def incompat_fn(x, assertion=assert_incompat_fn):
        assertion(x)
        return x.sum()

      if not is_vmap:
        incompat_fn(x_ok)
      with self.assertRaisesRegex(RuntimeError,
                                  'Value assertions can only be called from'):
        transform_fn(incompat_fn)(x_wrong)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
