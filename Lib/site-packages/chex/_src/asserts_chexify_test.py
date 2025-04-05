# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for `asserts_chexify.py`."""

import functools
import re
import sys
import threading
import time
from typing import Any, Optional, Sequence, Type

from absl.testing import absltest
from absl.testing import parameterized
from chex._src import asserts
from chex._src import asserts_chexify
from chex._src import asserts_internal as _ai
from chex._src import variants
import jax
import jax.numpy as jnp
import numpy as np

EPS = 1e-6

chexify_async = functools.partial(asserts_chexify.chexify, async_check=True)
chexify_sync = functools.partial(asserts_chexify.chexify, async_check=False)


def get_chexify_err_regex(name, msg):
  return re.escape(_ai.get_chexify_err_message(name, 'ANY')).replace(
      'ANY', f'.*{msg}.*'
  )


# Follows `ai.TChexAssertion`'s API.
def _assert_noop(*args,
                 custom_message: Optional[str] = None,
                 custom_message_format_vars: Sequence[Any] = (),
                 include_default_message: bool = True,
                 exception_type: Type[Exception] = AssertionError,
                 **kwargs) -> None:
  """No-op."""
  del args, custom_message, custom_message_format_vars
  del include_default_message, exception_type, kwargs


# Define a simple Chex assertion for testing purposes.
def _assert_tree_positive(tree):
  # Use jnp instead of np for testing purposes.
  if not all((x > 0).all() for x in jax.tree_util.tree_leaves(tree)):
    raise AssertionError('Tree contains non-positive elems!')


def _jittable_assert_tree_positive(tree):
  # Jittable version of `_assert_tree_positive`.
  pred = jnp.all(
      jnp.array([(x > 0).all() for x in jax.tree_util.tree_leaves(tree)]))
  asserts_chexify.checkify.check(pred, 'Tree contains non-positive elems!')
  return pred


chex_static_assert_positive = _ai.chex_assertion(
    assert_fn=_assert_tree_positive,
    jittable_assert_fn=None,
    name='assert_tree_positive_test')

chex_value_assert_positive = _ai.chex_assertion(
    assert_fn=_assert_tree_positive,
    jittable_assert_fn=_jittable_assert_tree_positive,
    name='assert_tree_positive_test')


class AssertsChexifyTest(variants.TestCase):
  """Simple tests for chexify assertions."""

  @variants.variants(with_jit=True, without_jit=True)
  def test_static_assertion(self):
    # Tests that static assertions can be used w/ and w/o `chexify()`.
    shape = (2, 3)

    # Define a simple static assertion.
    @asserts._static_assertion
    def chex_assert_shape(array, expected):
      if array.shape != expected:
        raise AssertionError('Wrong shape!')

    # Define a simple function that uses the assertion.
    def _sum_fn(tree):
      jax.tree.map(lambda x: chex_assert_shape(x, shape), tree)
      return sum(x.sum() for x in jax.tree_util.tree_leaves(tree))

    chexified_sum_fn = chexify_sync(self.variant(_sum_fn))
    # Passes in all contexts.
    x = {'a': np.ones(shape), 'b': {'c': np.ones(shape)}}
    self.assertEqual(_sum_fn(x), 2 * np.prod(shape))
    self.assertEqual(self.variant(_sum_fn)(x), 2 * np.prod(shape))
    self.assertEqual(chexified_sum_fn(x), 2 * np.prod(shape))

    # Fails in all contexts.
    x_wrong_shape = {'a': np.ones(shape), 'b': {'c': np.ones(shape + shape)}}
    with self.assertRaisesRegex(AssertionError, 'Wrong shape!'):
      _sum_fn(x_wrong_shape)
    with self.assertRaisesRegex(AssertionError, 'Wrong shape!'):
      self.variant(_sum_fn)(x_wrong_shape)
    with self.assertRaisesRegex(AssertionError, 'Wrong shape!'):
      chexified_sum_fn(x_wrong_shape)

  @variants.variants(with_jit=True, without_jit=True)
  def test_nested_chexification(self):
    """Tests nested wrapping."""

    @chexify_sync
    @self.variant
    def _pos_sum(x_1, x_2):

      @chexify_sync
      def _chexified_assert_fn(x):
        chex_value_assert_positive(x, custom_message='err_label_1')

      _chexified_assert_fn(x_1)
      chex_value_assert_positive(x_2, custom_message='err_label_2')

      return x_1 + x_2

    with self.assertRaisesRegex(RuntimeError,
                                'Nested @chexify wrapping is disallowed'):
      _pos_sum(1, 1)

  def test_async_mode(self):

    @jax.jit
    def _pos_sq(x):
      chex_value_assert_positive(x, custom_message='err_label')
      return jnp.dot(x, x) + 3

    valid_x = jnp.ones((1000, 1000))
    invalid_x = -valid_x

    # Test sync.
    sync_check = chexify_sync(_pos_sq)
    sync_check(valid_x)
    with self.assertRaisesRegex(AssertionError, 'err_label'):
      sync_check(invalid_x)

    # Test async.
    async_check = chexify_async(_pos_sq)
    async_check(valid_x)

    # Implicit wait, through timer.
    async_check(invalid_x)  # enqueued and immediately returned
    time.sleep(5)
    with self.assertRaisesRegex(AssertionError, 'err_label'):
      async_check(valid_x)  # the error gets retrieved

    # Explicit wait, through object-local wait.
    async_check(invalid_x)  # enqueued
    with self.assertRaisesRegex(AssertionError, 'err_label'):
      # Retrieve the error.
      async_check.wait_checks()  # pytype: disable=attribute-error

    # Explicit wait, through module-level wait.
    async_check(invalid_x)  # enqueued
    with self.assertRaisesRegex(AssertionError, 'err_label'):
      asserts_chexify.block_until_chexify_assertions_complete()

  def test_uninspected_checks(self):

    @jax.jit
    def _pos_sum(x):
      chex_value_assert_positive(x, custom_message='err_label')
      return x.sum()

    invalid_x = -jnp.ones(3)
    chexify_async(_pos_sum)(invalid_x)  # async error

    raised_exception = None

    def unraisablehook(unraisable):
      nonlocal raised_exception
      raised_exception = unraisable.exc_value

    if sys.version_info >= (3, 10):
      # In Python 3.10+, _run_exitfuncs logs the exception using unraisablehook
      # without raising the last exception.
      old_unraisablehook = sys.unraisablehook
      sys.unraisablehook = unraisablehook
      try:
        asserts_chexify.atexit._run_exitfuncs()
      finally:
        sys.unraisablehook = old_unraisablehook
      self.assertIsInstance(raised_exception, AssertionError)
      self.assertIn('err_label', str(raised_exception))
    else:
      with self.assertRaisesRegex(AssertionError, 'err_label'):
        asserts_chexify.atexit._run_exitfuncs()

  def test_docstring_example(self):

    @chexify_async
    @jax.jit
    def logp1_abs_safe(x):
      asserts.assert_tree_all_finite(x)
      return jnp.log(jnp.abs(x) + 1)

    logp1_abs_safe(jnp.ones(2))  # OK
    asserts_chexify.block_until_chexify_assertions_complete()

    err_regex = re.escape(_ai.get_chexify_err_message('assert_tree_all_finite'))
    with self.assertRaisesRegex(AssertionError, f'{err_regex}.*chexify_test'):
      logp1_abs_safe(jnp.array([jnp.nan, 3]))  # FAILS
      logp1_abs_safe.wait_checks()  # pytype: disable=attribute-error

  def test_checkify_errors(self):
    @jax.jit
    def take_by_index_and_div(x, i, y):
      return x[i] / y

    # Checks only Div-by-0.
    take_by_index_and_div_safe_div = asserts_chexify.chexify(
        take_by_index_and_div,
        async_check=False,
        errors=asserts_chexify.ChexifyChecks.div,
    )
    take_by_index_and_div_safe_div(jnp.ones(2), 1, 2)  # OK
    take_by_index_and_div_safe_div(jnp.ones(2), 10, 2)  # OOB undetected

    with self.assertRaisesRegex(
        asserts_chexify.checkify.JaxRuntimeError, 'division by zero'
    ):
      take_by_index_and_div_safe_div(jnp.ones(2), 1, 0)  # Div-by-0

    # Checks both OOB and Div-by-0.
    take_by_index_and_div_safe = asserts_chexify.chexify(
        take_by_index_and_div,
        async_check=False,
        errors=(
            asserts_chexify.ChexifyChecks.index
            | asserts_chexify.ChexifyChecks.div
        ),
    )
    take_by_index_and_div_safe(jnp.ones(2), 1, 2)  # OK

    with self.assertRaisesRegex(
        asserts_chexify.checkify.JaxRuntimeError, 'out-of-bounds'
    ):
      take_by_index_and_div_safe(jnp.ones(2), 10, 2)  # OOB

    with self.assertRaisesRegex(
        asserts_chexify.checkify.JaxRuntimeError, 'division by zero'
    ):
      take_by_index_and_div_safe(jnp.ones(2), 1, 0)  # Div-by-0

    with self.assertRaisesRegex(
        asserts_chexify.checkify.JaxRuntimeError, 'out-of-bounds'
    ):
      take_by_index_and_div_safe(jnp.ones(2), 10, 0)  # OOB (first) & Div-by-0

  def test_partial_python_fn(self):
    def fn(x, y):
      asserts.assert_trees_all_equal(x, y)
      return jax.tree.map(jnp.add, x, y)

    partial_fn = functools.partial(fn, y=jnp.array([1]))
    chexified_fn = chexify_async(partial_fn)  # note: fn is not transformed

    chexified_fn(jnp.array([1]))
    chexified_fn.wait_checks()  # pytype: disable=attribute-error

    with self.assertRaisesRegex(AssertionError, '0 and 1 differ'):
      chexified_fn(jnp.array([2]))
      # Fail: not equal.
      chexified_fn.wait_checks()  # pytype: disable=attribute-error

  def test_wrong_order_of_wrapping(self):

    @chexify_async
    def inner_fn(x, y):
      asserts.assert_trees_all_equal(x, y)
      return jax.tree.map(jnp.add, x, y)

    def outer_fn(x, y):
      z = inner_fn(x, y)
      return jax.tree.map(jnp.square, z)

    x = jnp.array([1])
    y = jnp.array([1])
    with self.assertRaisesRegex(
        RuntimeError, '@chexify must be applied on top of all'
    ):
      jax.jit(inner_fn)(x, y)

    with self.assertRaisesRegex(
        RuntimeError, '@chexify must be applied on top of all'
    ):
      jax.jit(outer_fn)(x, y)

    with self.assertRaisesRegex(
        RuntimeError, '@chexify must be applied on top of all'
    ):
      jax.jit(chexify_async(outer_fn))(x, y)

    outer_fn(x, y)


class AssertsChexifyTestSuite(variants.TestCase):
  """Test suite for chexify assertions."""

  def run_test_suite(self, make_test_fn, all_valid_args, all_invalid_args,
                     failure_labels, jax_transform, run_pure):
    """Runs a set of tests for static & value assertions.

    See `run_test_suite_with_log_abs_fn` for example.

    Args:
      make_test_fn: A function that returns a pure function to transform.
      all_valid_args: A list of collections of args that pass assertions.
      all_invalid_args: A list of collections of args that fail assertions.
      failure_labels: A list of custom labels, one per every failed assertion.
      jax_transform: A function that accepts a pure function and returns its
        transformed version.
      run_pure: A bool suggesting whether pure_fn can be called without
        transforms (e.g. it isn't the case for f-ns that use JAX collectives).
    """
    assert len(all_invalid_args) == len(failure_labels)

    # Define 3 versions of the tested function.
    if run_pure:
      fn_no_assert = make_test_fn(_assert_noop)
      fn_static_assert = make_test_fn(chex_static_assert_positive)
    fn_value_assert = make_test_fn(chex_value_assert_positive)

    # Wrapped fn with value asserts.
    chexified_fn_with_value_asserts = chexify_sync(
        jax_transform(fn_value_assert))

    # Run tests with valid arguments.
    for valid_args in all_valid_args:
      if run_pure:
        # Test all versions return the same outputs.
        asserts.assert_trees_all_equal(
            fn_no_assert(*valid_args), fn_static_assert(*valid_args))
        asserts.assert_trees_all_equal(
            fn_no_assert(*valid_args), fn_value_assert(*valid_args))

        # `ConcretizationTypeError` if static assertion is used in value checks.
        with self.assertRaises(jax.errors.ConcretizationTypeError):
          jax_transform(fn_static_assert)(*valid_args)

      # Value assertions pass.
      chexified_fn_with_value_asserts(*valid_args)

      # Reports incorrect usage without `chexify()`.
      with self.assertRaisesRegex(
          RuntimeError, 'can only be called from functions wrapped .*chexify'):
        # Create a local object to avoid reusing jax internal cache.
        local_fn_value_assert = make_test_fn(chex_value_assert_positive)
        jax_transform(local_fn_value_assert)(*valid_args)

    # Run tests with invalid arguments.
    for invalid_args, label in zip(all_invalid_args, failure_labels):
      if run_pure:
        # Static assertion fails on incorrect inputs (without transformations).
        with self.assertRaisesRegex(AssertionError, re.escape(label)):
          fn_static_assert(*invalid_args)

      # Value assertion fails on incorrect inputs (with transformations).
      err_regex = get_chexify_err_regex('assert_tree_positive_test', label)
      with self.assertRaisesRegex(AssertionError, err_regex):
        chexified_fn_with_value_asserts(*invalid_args)

      # Reports incorrect usage without `chexify()`.
      with self.assertRaisesRegex(
          RuntimeError, 'can only be called from functions wrapped .*chexify'):
        # Create a local object to avoid reusing jax internal cache.
        local_fn_value_assert = make_test_fn(chex_value_assert_positive)
        jax_transform(local_fn_value_assert)(*invalid_args)

  def run_test_suite_with_log_abs_fn(self, make_log_fn, jax_transform, devices,
                                     run_pure, run_in_thread):
    """Generates valid and invalid inputs for log_abs_fn and runs the tests."""
    x_pos = {
        'a': np.ones((10, 2)),
        'b': {
            'c': np.array([[5, 2] for _ in range(10)])
        }
    }
    x_with_neg = {
        'a': np.ones((10, 2)),
        'b': {
            'c': np.array([[5, -1] for _ in range(10)])
        }
    }
    (x_pos, x_with_neg) = jax.device_put_replicated((x_pos, x_with_neg),
                                                    devices)

    all_valid_args = ((x_pos, x_pos),)
    all_invalid_args = (
        (x_with_neg, x_pos),
        (x_pos, x_with_neg),
        (x_with_neg, x_with_neg),
    )
    failure_labels = (
        'label_1',
        'label_2',
        'label_1',
    )

    run_tests_callback = functools.partial(self.run_test_suite, make_log_fn,
                                           all_valid_args, all_invalid_args,
                                           failure_labels, jax_transform,
                                           run_pure)
    if run_in_thread:
      failure = AssertionError('not executed')

      def _run_tests_in_thread():
        nonlocal failure
        failure = None
        try:
          run_tests_callback()
        except Exception as e:  # pylint:disable=broad-except
          failure = e

      thr = threading.Thread(target=_run_tests_in_thread, daemon=False)
      thr.start()
      thr.join(timeout=30)
      if thr.is_alive():
        raise TimeoutError('Thread is alive after 30 seconds.')
      if failure is not None:
        raise AssertionError(f'Thread failed with: {failure}.')
    else:
      run_tests_callback()

  @parameterized.named_parameters(('main', False), ('thread', True))
  def test_log_abs_fn_jitted(self, run_in_thread):
    """Tests simple jit transformation."""

    def _make_log_fn(assert_input_fn: _ai.TChexAssertion):

      def _pure_log_fn(tree_1, tree_2):
        # Call twice to make sure all deps are retained after XLA optimizations.
        assert_input_fn(tree_1, custom_message='label_1')
        assert_input_fn(tree_2, custom_message='label_2')

        return jax.tree.map(lambda x1, x2: jnp.log(jnp.abs(x1 + x2) + EPS),
                            tree_1, tree_2)

      return _pure_log_fn

    with jax.checking_leaks():
      self.run_test_suite_with_log_abs_fn(
          make_log_fn=_make_log_fn,
          jax_transform=jax.jit,
          devices=jax.local_devices()[:1],
          run_pure=True,
          run_in_thread=run_in_thread)

  @parameterized.named_parameters(('main', False), ('thread', True))
  def test_log_abs_fn_jitted_nested_wrap(self, run_in_thread):
    """Tests nested jit transforms (wrapping)."""

    def _make_log_fn(assert_input_fn: _ai.TChexAssertion):

      @jax.jit
      def _abs(tree):
        assert_input_fn(tree, custom_message='label_1')
        tree_p1 = jax.tree.map(lambda x: x + 1, tree)
        return jax.tree.map(jnp.abs, tree_p1)

      def _pure_log_fn(tree_1, tree_2):
        tree_1 = _abs(tree_1)
        assert_input_fn(tree_2, custom_message='label_2')

        return jax.tree.map(lambda x1, x2: jnp.log(jnp.abs(x1 + x2) + EPS),
                            tree_1, tree_2)

      return _pure_log_fn

    with jax.checking_leaks():
      self.run_test_suite_with_log_abs_fn(
          make_log_fn=_make_log_fn,
          jax_transform=jax.jit,
          devices=jax.local_devices()[:1],
          run_pure=False,  # do not run because internal jit is not checkified
          run_in_thread=run_in_thread)

  @parameterized.named_parameters(('main', False), ('thread', True))
  def test_log_abs_fn_jitted_nested_call(self, run_in_thread):
    """Tests nested jit transforms (calling)."""

    def _make_log_fn(assert_input_fn: _ai.TChexAssertion):

      def _abs(tree):
        assert_input_fn(tree, custom_message='label_1')
        tree_p1 = jax.tree.map(lambda x: x + 1, tree)
        return jax.tree.map(jnp.abs, tree_p1)

      def _pure_log_fn(tree_1, tree_2):
        tree_1 = jax.jit(_abs)(tree_1)
        assert_input_fn(tree_2, custom_message='label_2')

        return jax.tree.map(lambda x1, x2: jnp.log(jnp.abs(x1 + x2) + EPS),
                            tree_1, tree_2)

      return _pure_log_fn

    with jax.checking_leaks():
      self.run_test_suite_with_log_abs_fn(
          make_log_fn=_make_log_fn,
          jax_transform=jax.jit,
          devices=jax.local_devices()[:1],
          run_pure=False,  # do not run because internal jit is not checkified
          run_in_thread=run_in_thread)

  @parameterized.named_parameters(('main', False), ('thread', True))
  def test_log_abs_fn_pmapped(self, run_in_thread):
    """Tests pmap transform."""

    def _make_log_fn(assert_input_fn: _ai.TChexAssertion):

      def _pure_log_fn(tree_1, tree_2):
        # Call twice to make sure all deps are retained after XLA optimizations.
        assert_input_fn(tree_1, custom_message='label_1')
        assert_input_fn(tree_2, custom_message='label_2')

        tree_1 = jax.lax.pmean(tree_1, axis_name='i')
        return jax.tree.map(lambda x1, x2: jnp.log(jnp.abs(x1 + x2) + EPS),
                            tree_1, tree_2)

      return _pure_log_fn

    with jax.checking_leaks():
      self.run_test_suite_with_log_abs_fn(
          make_log_fn=_make_log_fn,
          jax_transform=lambda fn: jax.pmap(fn, axis_name='i'),
          devices=jax.local_devices(),
          run_pure=False,  # do not run because the f-n contains collective ops
          run_in_thread=run_in_thread)

  @parameterized.named_parameters(('main', False), ('thread', True))
  def test_log_abs_fn_jitted_vmapped(self, run_in_thread):
    """Tests vmap transform."""

    def _make_log_fn(assert_input_fn: _ai.TChexAssertion):

      def _pure_log_fn(tree_1, tree_2):
        # Call twice to make sure all deps are retained after XLA optimizations.
        assert_input_fn(tree_1, custom_message='label_1')
        assert_input_fn(tree_2, custom_message='label_2')

        return jax.tree.map(lambda x1, x2: jnp.log(jnp.abs(x1 + x2) + EPS),
                            tree_1, tree_2)

      return _pure_log_fn

    with jax.checking_leaks():
      self.run_test_suite_with_log_abs_fn(
          make_log_fn=_make_log_fn,
          jax_transform=lambda fn: jax.jit(jax.vmap(fn)),  # jax + vmap
          devices=jax.local_devices()[:1],
          run_pure=True,
          run_in_thread=run_in_thread,
      )


class AssertsLibraryTest(parameterized.TestCase):

  def test_assert_tree_all_finite(self):
    @jax.jit
    def fn(x):
      asserts.assert_tree_all_finite(x)
      return jax.tree.map(jnp.sum, x)

    chexified_fn = asserts_chexify.chexify(fn, async_check=False)
    chexified_fn({'a': 0, 'b': jnp.ones(3)})  # OK

    err_regex = get_chexify_err_regex(
        'assert_tree_all_finite', 'Tree contains non-finite value'
    )
    with self.assertRaisesRegex(AssertionError, err_regex):
      chexified_fn({'a': 1, 'b': jnp.array([1, jnp.nan, 3])})  # NaN
    with self.assertRaisesRegex(AssertionError, re.escape("'b': Array(nan")):
      chexified_fn({'a': 1, 'b': jnp.array([1, jnp.nan, 3])})  # NaN

  def test_assert_trees_all_equal(self):
    @jax.jit
    def fn(x, y):
      asserts.assert_trees_all_equal(x, y)
      return jax.tree.map(jnp.add, x, y)

    chexified_fn = asserts_chexify.chexify(fn, async_check=False)

    tree_1 = {'a': jnp.array([3]), 'b': jnp.array([10, 10])}
    tree_2 = {'a': jnp.array([3]), 'b': jnp.array([10, 20])}
    chexified_fn(tree_1, tree_1)  # OK

    err_regex = get_chexify_err_regex(
        'assert_trees_all_equal', 'Values not exactly equal:'
    )
    with self.assertRaisesRegex(AssertionError, err_regex):
      chexified_fn(tree_1, tree_2)  # Fail: not equal

    with self.assertRaisesRegex(
        AssertionError, re.escape("Trees 0 and 1 differ in leaves '('b',)'")
    ):
      chexified_fn(tree_1, tree_2)  # Fail: not equal

  def test_assert_trees_all_equal_with_prng_keys(self):
    @jax.jit
    def fn(x, y):
      asserts.assert_trees_all_equal(x, y)
      return x['a'] + y['a']

    chexified_fn = asserts_chexify.chexify(fn, async_check=False)
    tree1 = {'a': jnp.array([3]), 'key': jax.random.split(jax.random.key(1))}
    tree2 = {'a': jnp.array([3]), 'key': jax.random.split(jax.random.key(2))}
    chexified_fn(tree1, tree1)  # OK
    with self.assertRaisesRegex(
        AssertionError, re.escape("Trees 0 and 1 differ in leaves '('key',)'")
    ):
      chexified_fn(tree1, tree2)  # Fail: not equal

  def test_assert_trees_all_close(self):
    @jax.jit
    def fn(x, y, z):
      asserts.assert_trees_all_close(x, y, z, rtol=0.1, atol=0.1)
      return jax.tree.map(jnp.add, x, y)

    chexified_fn = asserts_chexify.chexify(fn, async_check=False)

    tree_1 = {1: {'a': jnp.array([3]), 'b': jnp.array([10, 10])}}
    tree_1_close = {1: {'a': jnp.array([3.1]), 'b': jnp.array([10.1, 10.1])}}
    tree_2 = {1: {'a': jnp.array([3]), 'b': jnp.array([10, 20])}}
    chexified_fn(tree_1, tree_1, tree_1)  # OK
    chexified_fn(tree_1, tree_1_close, tree_1)  # OK

    err_regex = get_chexify_err_regex(
        'assert_trees_all_close', 'Values not approximately equal'
    )
    with self.assertRaisesRegex(AssertionError, err_regex):
      chexified_fn(tree_1, tree_2, tree_1)  # Fail: not close

    with self.assertRaisesRegex(
        AssertionError, re.escape("Trees 0 and 2 differ in leaves '(1, 'b')':")
    ):
      chexified_fn(tree_1, tree_1_close, tree_2)  # Fail: not close

  def test_custom_message(self):
    @jax.jit
    def fn(x, y):
      asserts.assert_trees_all_equal(
          x,
          y,
          custom_message='sum(x)={}',
          custom_message_format_vars=[
              sum(l.sum() for l in jax.tree_util.tree_leaves(x))
          ],
      )
      return jax.tree.map(jnp.add, x, y)

    chexified_fn = asserts_chexify.chexify(fn, async_check=False)

    tree_1 = {'a': jnp.array([3]), 'b': jnp.array([10, 10])}
    tree_2 = {'a': jnp.array([3]), 'b': jnp.array([10, 320])}

    with self.assertRaisesRegex(
        AssertionError, re.escape('Custom message: sum(x)=') + '.*23'
    ):
      chexified_fn(tree_1, tree_2)  # Fail: not equal

    with self.assertRaisesRegex(
        AssertionError, re.escape('Custom message: sum(x)=') + '.*333'
    ):
      chexified_fn(tree_2, tree_1)  # Fail: not equal


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
