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
"""Tests for `variants.py`.

To run tests in multi-cpu regime, one need to set the flag `--n_cpu_devices=N`.
"""

import inspect
import itertools
import unittest

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from chex._src import asserts
from chex._src import fake
from chex._src import pytypes
from chex._src import variants

import jax
import jax.numpy as jnp
import numpy as np

FLAGS = flags.FLAGS

ArrayBatched = pytypes.ArrayBatched

DEFAULT_FN = lambda arg_0, arg_1: arg_1 - arg_0
DEFAULT_PARAMS = ((1, 2, 1), (4, 6, 2))
DEFAULT_NDARRAY_PARAMS_SHAPE = (5, 7)
DEFAULT_NAMED_PARAMS = (('case_0', 1, 2, 1), ('case_1', 4, 6, 2))

make_suite = unittest.defaultTestLoader.loadTestsFromTestCase


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule():
  fake.set_n_cpu_devices()
  asserts.assert_devices_available(
      FLAGS['chex_n_cpu_devices'].value, 'cpu', backend='cpu')


def _scalar_to_ndarray(x, shape=None):
  return np.broadcast_to(x, shape or DEFAULT_NDARRAY_PARAMS_SHAPE)


def _variant_default_tests_generator(fn, is_jit_context, which_variants,
                                     **var_kwargs):
  """Returns a generator with standard tests.

  For internal usage. Allows to dynamically generate common tests.
  See tests' names and comments for more information.

  Args:
    fn: a separate function to be tested (without `self` argument).
    is_jit_context: is a function is supposed to be JIT-ted.
    which_variants: chex variants to use in tests generation.
    **var_kwargs: kwargs for variants wrappers.

  Returns:
    A generator with tests.
  """

  # All generated tests use default arguments (defined at the top of this file).
  arg_0, arg_1, expected = DEFAULT_PARAMS[0]
  varg_0, varg_1, vexpected = (
      _scalar_to_ndarray(a) for a in (arg_0, arg_1, expected))

  # We test whether the function has been jitted by introducing a counter
  # variable as a side-effect. When the function is repeatedly called, jitted
  # code will only execute the side-effect once
  python_execution_count = 0

  def fn_with_counter(*args, **kwargs):
    nonlocal python_execution_count
    python_execution_count += 1
    return fn(*args, **kwargs)

  def exec_with_tracing_counter_checks(self, var_fn, arg_0, arg_1):
    self.assertEqual(python_execution_count, 0)
    _ = var_fn(arg_0, arg_1)
    # In jit context, JAX can omit retracing a function from the previous
    # test, hence `python_execution_count` will be equal to 0.
    # In non-jit context, `python_execution_count` must always increase.
    if not is_jit_context:
      self.assertEqual(python_execution_count, 1)
    actual = var_fn(arg_0, arg_1)
    if is_jit_context:
      # Either 1 (initial tracing) or 0 (function reuse).
      self.assertLess(python_execution_count, 2)
    else:
      self.assertEqual(python_execution_count, 2)
    return actual

  # Here, various tests follow. Tests' names intended to be self-descriptive.
  @variants.variants(**which_variants)
  def test_with_scalar_args(self):
    nonlocal python_execution_count
    python_execution_count = 0
    var_fn = self.variant(fn_with_counter, **var_kwargs)
    actual = exec_with_tracing_counter_checks(self, var_fn, arg_0, arg_1)
    self.assertEqual(actual, expected)

  @variants.variants(**which_variants)
  def test_called_variant(self):
    nonlocal python_execution_count
    python_execution_count = 0
    var_fn = self.variant(**var_kwargs)(fn_with_counter)
    actual = exec_with_tracing_counter_checks(self, var_fn, arg_0, arg_1)
    self.assertEqual(actual, expected)

  @variants.variants(**which_variants)
  def test_with_kwargs(self):
    nonlocal python_execution_count
    python_execution_count = 0
    var_fn = self.variant(fn_with_counter, **var_kwargs)
    actual = exec_with_tracing_counter_checks(
        self, var_fn, arg_1=arg_1, arg_0=arg_0)
    self.assertEqual(actual, expected)

  @variants.variants(**which_variants)
  @parameterized.parameters(*DEFAULT_PARAMS)
  def test_scalar_parameters(self, arg_0, arg_1, expected):
    nonlocal python_execution_count
    python_execution_count = 0
    var_fn = self.variant(fn_with_counter, **var_kwargs)
    actual = exec_with_tracing_counter_checks(self, var_fn, arg_0, arg_1)
    self.assertEqual(actual, expected)

  @variants.variants(**which_variants)
  @parameterized.named_parameters(*DEFAULT_NAMED_PARAMS)
  def test_named_scalar_parameters(self, arg_0, arg_1, expected):
    nonlocal python_execution_count
    python_execution_count = 0
    var_fn = self.variant(fn_with_counter, **var_kwargs)
    actual = exec_with_tracing_counter_checks(self, var_fn, arg_0, arg_1)
    self.assertEqual(actual, expected)

  @variants.variants(**which_variants)
  def test_with_ndarray_args(self):
    nonlocal python_execution_count
    python_execution_count = 0
    var_fn = self.variant(fn_with_counter, **var_kwargs)
    actual = exec_with_tracing_counter_checks(self, var_fn, varg_0, varg_1)
    vexpected_ = vexpected
    # pmap variant case.
    if len(actual.shape) == len(DEFAULT_NDARRAY_PARAMS_SHAPE) + 1:
      vexpected_ = jnp.broadcast_to(vexpected_, actual.shape)
    np.testing.assert_array_equal(actual, vexpected_)

  @variants.variants(**which_variants)
  @parameterized.parameters(*DEFAULT_PARAMS)
  def test_ndarray_parameters(self, arg_0, arg_1, expected):
    nonlocal python_execution_count
    python_execution_count = 0
    varg_0, varg_1, vexpected = (
        _scalar_to_ndarray(a) for a in (arg_0, arg_1, expected))
    var_fn = self.variant(fn_with_counter, **var_kwargs)
    actual = exec_with_tracing_counter_checks(self, var_fn, varg_0, varg_1)
    # pmap variant case.
    if len(actual.shape) == len(DEFAULT_NDARRAY_PARAMS_SHAPE) + 1:
      vexpected = jnp.broadcast_to(vexpected, actual.shape)
    np.testing.assert_array_equal(actual, vexpected)

  @variants.variants(**which_variants)
  @parameterized.named_parameters(*DEFAULT_NAMED_PARAMS)
  def test_ndarray_named_parameters(self, arg_0, arg_1, expected):
    nonlocal python_execution_count
    python_execution_count = 0
    varg_0, varg_1, vexpected = (
        _scalar_to_ndarray(a) for a in (arg_0, arg_1, expected))
    var_fn = self.variant(fn_with_counter, **var_kwargs)
    actual = exec_with_tracing_counter_checks(self, var_fn, varg_0, varg_1)
    # pmap variant case.
    if len(actual.shape) == len(DEFAULT_NDARRAY_PARAMS_SHAPE) + 1:
      vexpected = jnp.broadcast_to(vexpected, actual.shape)
    np.testing.assert_array_equal(actual, vexpected)

  all_tests = (test_with_scalar_args, test_called_variant, test_with_kwargs,
               test_scalar_parameters, test_named_scalar_parameters,
               test_with_ndarray_args, test_ndarray_parameters,
               test_ndarray_named_parameters)

  # Each test is a generator itself, hence we use chaining from itertools.
  return itertools.chain(*all_tests)


class ParamsProductTest(absltest.TestCase):

  def test_product(self):
    l1 = (
        ('x1', 1, 10),
        ('x2', 2, 20),
    )

    l2 = (
        ('y1', 3),
        ('y2', 4),
    )

    l3 = (
        ('z1', 5, 50),
        ('z2', 6, 60),
    )

    l4 = (('aux', 'AUX'),)

    expected = [('x1', 1, 10, 'y1', 3, 'z1', 5, 50, 'aux', 'AUX'),
                ('x1', 1, 10, 'y1', 3, 'z2', 6, 60, 'aux', 'AUX'),
                ('x1', 1, 10, 'y2', 4, 'z1', 5, 50, 'aux', 'AUX'),
                ('x1', 1, 10, 'y2', 4, 'z2', 6, 60, 'aux', 'AUX'),
                ('x2', 2, 20, 'y1', 3, 'z1', 5, 50, 'aux', 'AUX'),
                ('x2', 2, 20, 'y1', 3, 'z2', 6, 60, 'aux', 'AUX'),
                ('x2', 2, 20, 'y2', 4, 'z1', 5, 50, 'aux', 'AUX'),
                ('x2', 2, 20, 'y2', 4, 'z2', 6, 60, 'aux', 'AUX')]
    product = list(variants.params_product(l1, l2, l3, l4, named=False))
    self.assertEqual(product, expected)

    named_expected = [('x1_y1_z1_aux', 1, 10, 3, 5, 50, 'AUX'),
                      ('x1_y1_z2_aux', 1, 10, 3, 6, 60, 'AUX'),
                      ('x1_y2_z1_aux', 1, 10, 4, 5, 50, 'AUX'),
                      ('x1_y2_z2_aux', 1, 10, 4, 6, 60, 'AUX'),
                      ('x2_y1_z1_aux', 2, 20, 3, 5, 50, 'AUX'),
                      ('x2_y1_z2_aux', 2, 20, 3, 6, 60, 'AUX'),
                      ('x2_y2_z1_aux', 2, 20, 4, 5, 50, 'AUX'),
                      ('x2_y2_z2_aux', 2, 20, 4, 6, 60, 'AUX')]
    named_product = list(variants.params_product(l1, l2, l3, l4, named=True))
    self.assertEqual(named_product, named_expected)


class FailedTestsTest(absltest.TestCase):
  # Inner class prevents FailedTest being run by `absltest.main()`.

  class FailedTest(variants.TestCase):

    @variants.variants(without_jit=True)
    def test_failure(self):
      self.assertEqual('meaning of life', 1337)

    @variants.variants(without_jit=True)
    def test_error(self):
      raise ValueError('this message does not specify the Chex variant')

  def setUp(self):
    super().setUp()
    self.chex_info = str(variants.ChexVariantType.WITHOUT_JIT)
    self.res = unittest.TestResult()
    ts = make_suite(self.FailedTest)  # pytype: disable=module-attr
    ts.run(self.res)

  def test_useful_failures(self):
    self.assertIsNotNone(self.res.failures)
    for test_method, _ in self.res.failures:
      self.assertIn(self.chex_info, test_method._testMethodName)

  def test_useful_errors(self):
    self.assertIsNotNone(self.res.errors)

    for test_method, msg in self.res.errors:
      self.assertIn(self.chex_info, test_method._testMethodName)
      self.assertIn('this message does not specify the Chex variant', msg)


class OneFailedVariantTest(variants.TestCase):
  # Inner class prevents MaybeFailedTest being run by `absltest.main()`.

  class MaybeFailedTest(variants.TestCase):

    @variants.variants(with_device=True, without_device=True)
    def test_failure(self):

      @self.variant
      def fails_for_without_device_variant(x):
        self.assertIsInstance(x, jax.Array)

      fails_for_without_device_variant(42)

  def test_useful_failure(self):
    expected_info = str(variants.ChexVariantType.WITHOUT_DEVICE)
    unexpected_info = str(variants.ChexVariantType.WITH_DEVICE)

    res = unittest.TestResult()
    ts = make_suite(self.MaybeFailedTest)  # pytype: disable=module-attr
    ts.run(res)
    self.assertLen(res.failures, 1)

    for test_method, _ in res.failures:
      self.assertIn(expected_info, test_method._testMethodName)
      self.assertNotIn(unexpected_info, test_method._testMethodName)


class WrongBaseClassTest(variants.TestCase):
  # Inner class prevents InnerTest being run by `absltest.main()`.

  class InnerTest(absltest.TestCase):

    @variants.all_variants
    def test_failure(self):
      pass

  def test_wrong_base_class(self):
    res = unittest.TestResult()
    ts = make_suite(self.InnerTest)  # pytype: disable=module-attr
    ts.run(res)
    self.assertLen(res.errors, 1)

    for _, msg in res.errors:
      self.assertRegex(msg,
                       'RuntimeError.+make sure.+inherit from `chex.TestCase`')


class BaseClassesTest(parameterized.TestCase):
  """Tests different combinations of base classes for a variants test."""

  def generate_test_class(self, base_1, base_2):
    """Returns a test class derived from the specified bases."""

    class InnerBaseClassTest(base_1, base_2):

      @variants.all_variants(with_pmap=False)
      @parameterized.parameters(*DEFAULT_PARAMS)
      def test_should_pass(self, arg_0, arg_1, expected):
        actual = self.variant(DEFAULT_FN)(arg_0, arg_1)
        self.assertEqual(actual, expected)

    return InnerBaseClassTest

  @parameterized.named_parameters(
      ('parameterized', (parameterized.TestCase, object)),
      ('variants', (variants.TestCase, object)),
      ('variants_and_parameterized',
       (variants.TestCase, parameterized.TestCase)),
  )
  def test_inheritance(self, base_classes):
    res = unittest.TestResult()
    test_class = self.generate_test_class(*base_classes)
    for base_class in base_classes:
      self.assertTrue(issubclass(test_class, base_class))
    ts = make_suite(test_class)  # pytype: disable=module-attr
    ts.run(res)
    self.assertEqual(res.testsRun, 8)
    self.assertEmpty(res.errors or res.failures)


class VariantsTestCaseWithParameterizedTest(absltest.TestCase):
  # Inner class prevents InnerTest being run by `absltest.main()`.

  class InnerTest(variants.TestCase):

    @variants.all_variants(with_pmap=False)
    @parameterized.parameters(*DEFAULT_PARAMS)
    def test_should_pass(self, arg_0, arg_1, expected):
      actual = self.variant(DEFAULT_FN)(arg_0, arg_1)
      self.assertEqual(actual, expected)

  def test_should_pass(self):
    res = unittest.TestResult()
    ts = make_suite(self.InnerTest)  # pytype: disable=module-attr
    ts.run(res)
    self.assertEqual(res.testsRun, 8)
    self.assertEmpty(res.errors or res.failures)


class WrongWrappersOrderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._error_msg = ('A test wrapper attempts to access __name__ of '
                       'VariantsTestCaseGenerator')

  def test_incorrect_wrapping_order_named_all_variants(self):
    with self.assertRaisesRegex(RuntimeError, self._error_msg):

      @parameterized.named_parameters(*DEFAULT_NAMED_PARAMS)
      @variants.all_variants()
      def _(*unused_args):
        pass

  def test_incorrect_wrapping_order_named_some_variants(self):
    with self.assertRaisesRegex(RuntimeError, self._error_msg):

      @parameterized.named_parameters(*DEFAULT_NAMED_PARAMS)
      @variants.variants(with_jit=True, with_device=True)
      def _(*unused_args):
        pass

  def test_incorrect_wrapping_order_all_variants(self):
    with self.assertRaisesRegex(RuntimeError, self._error_msg):

      @parameterized.parameters(*DEFAULT_PARAMS)
      @variants.all_variants()
      def _(*unused_args):
        pass

  def test_incorrect_wrapping_order_some_variants(self):
    with self.assertRaisesRegex(RuntimeError, self._error_msg):

      @parameterized.parameters(*DEFAULT_PARAMS)
      @variants.variants(without_jit=True, without_device=True)
      def _(*unused_args):
        pass


class UnusedVariantTest(absltest.TestCase):
  # Inner class prevents InnerTest being run by `absltest.main()`.

  class InnerTest(variants.TestCase):

    @variants.all_variants(with_pmap=False)
    def test_noop(self):
      pass

  def test_unused_variant(self):
    res = unittest.TestResult()
    ts = make_suite(self.InnerTest)  # pytype: disable=module-attr
    ts.run(res)
    self.assertLen(res.errors, 4)
    for _, msg in res.errors:
      self.assertRegex(
          msg, 'RuntimeError: Test is wrapped .+ but never calls self.variant')


class NoVariantsTest(absltest.TestCase):
  """Checks that Chex raises ValueError when no variants are selected."""

  def test_no_variants(self):

    with self.assertRaisesRegex(ValueError, 'No variants selected'):

      class InnerTest(variants.TestCase):  # pylint:disable=unused-variable

        @variants.variants()
        def test_noop(self):
          pass


class UnknownVariantArgumentsTest(absltest.TestCase):
  # Inner class prevents InnerTest being run by `absltest.main()`.

  class InnerTest(variants.TestCase):

    @variants.all_variants(with_pmap=False)
    def test_arg(self):
      self.variant(lambda: None, some_unknown_arg=16)

  def test_unknown_argument(self):
    res = unittest.TestResult()
    ts = make_suite(self.InnerTest)  # pytype: disable=module-attr
    ts.run(res)
    self.assertLen(res.errors, 4)
    for _, msg in res.errors:
      self.assertRegex(msg, 'Unknown arguments in .+some_unknown_arg')


class VariantTypesTest(absltest.TestCase):
  # Inner class prevents InnerTest being run by `absltest.main()`.

  class InnerTest(variants.TestCase):
    var_types = set()

    @variants.all_variants()
    def test_var_type(self):
      self.variant(lambda: None)
      self.var_types.add(self.variant.type)

  def test_var_type_fetch(self):
    ts = make_suite(self.InnerTest)  # pytype: disable=module-attr
    ts.run(unittest.TestResult())
    expected_types = set(variants.ChexVariantType)
    if jax.device_count() == 1:
      expected_types.remove(variants.ChexVariantType.WITH_PMAP)
    self.assertSetEqual(self.InnerTest.var_types, expected_types)

  def test_consistency(self):
    self.assertLen(variants._variant_decorators, len(variants.ChexVariantType))

    for arg in inspect.getfullargspec(variants.variants).args:
      if arg == 'test_method':
        continue
      self.assertTrue(hasattr(variants.ChexVariantType, arg.upper()))


class CountVariantsTest(absltest.TestCase):
  # Inner class prevents InnerTest being run by `absltest.main()`.

  class InnerTest(variants.TestCase):
    test_1_count = 0
    test_2_count = 0
    test_3_count = 0
    test_4_count = 0

    @variants.all_variants
    def test_1(self):
      type(self).test_1_count += 1

    @variants.all_variants(with_pmap=False)
    def test_2(self):
      type(self).test_2_count += 1

    @variants.variants(with_jit=True)
    def test_3(self):
      type(self).test_3_count += 1

    @variants.variants(with_jit=True)
    @variants.variants(without_jit=False)
    @variants.variants(with_device=True)
    @variants.variants(without_device=False)
    def test_4(self):
      type(self).test_4_count += 1

  def test_counters(self):
    res = unittest.TestResult()
    ts = make_suite(self.InnerTest)  # pytype: disable=module-attr
    ts.run(res)

    active_pmap = int(jax.device_count() > 1)
    self.assertEqual(self.InnerTest.test_1_count, 4 + active_pmap)
    self.assertEqual(self.InnerTest.test_2_count, 4)
    self.assertEqual(self.InnerTest.test_3_count, 1)
    self.assertEqual(self.InnerTest.test_4_count, 2)

    # Test methods do not use `self.variant`.
    self.assertLen(res.errors, 1 + 2 + 4 + 4 + active_pmap)
    for _, msg in res.errors:
      self.assertRegex(
          msg, 'RuntimeError: Test is wrapped .+ but never calls self.variant')


class MultipleVariantsTest(parameterized.TestCase):

  @variants.all_variants()
  def test_all_variants(self):
    # self.variant must be used at least once.
    self.variant(lambda x: x)(0)
    self.assertNotEqual('meaning of life', 1337)

  @variants.all_variants
  def test_all_variants_no_parens(self):
    # self.variant must be used at least once.
    self.variant(lambda x: x)(0)
    self.assertNotEqual('meaning of life', 1337)

  @variants.variants(
      with_jit=True, without_jit=True, with_device=True, without_device=True)
  @parameterized.named_parameters(*DEFAULT_NAMED_PARAMS)
  def test_many_variants(self, arg_0, arg_1, expected):

    @self.variant
    def fn(arg_0, arg_1):
      return arg_1 - arg_0

    actual = fn(arg_0, arg_1)
    self.assertEqual(actual, expected)


class VmappedFunctionTest(parameterized.TestCase):

  @variants.all_variants(with_pmap=True)
  @parameterized.named_parameters(*DEFAULT_NAMED_PARAMS)
  def test_vmapped_fn_named_params(self, arg_0, arg_1, expected):
    varg_0, varg_1, vexpected = (
        _scalar_to_ndarray(x) for x in (arg_0, arg_1, expected))

    vmapped_fn = jax.vmap(DEFAULT_FN)
    actual = self.variant(vmapped_fn)(varg_0, varg_1)
    # pmap variant.
    if len(actual.shape) == len(DEFAULT_NDARRAY_PARAMS_SHAPE) + 1:
      vexpected = jnp.broadcast_to(vexpected, actual.shape)
    np.testing.assert_array_equal(actual, vexpected)


class WithoutJitTest(parameterized.TestCase):

  tests = _variant_default_tests_generator(
      fn=DEFAULT_FN,
      is_jit_context=False,
      which_variants=dict(without_jit=True))


class WithJitTest(parameterized.TestCase):

  tests = _variant_default_tests_generator(
      fn=DEFAULT_FN, is_jit_context=True, which_variants=dict(with_jit=True))

  @variants.variants(with_jit=True)
  @parameterized.parameters(*DEFAULT_PARAMS)
  def test_different_jit_kwargs(self, arg_0, arg_1, expected):
    kwarg_0 = arg_0
    kwarg_1 = arg_1

    arg_0_type = type(arg_0)
    arg_1_type = type(arg_1)
    kwarg_0_type = type(kwarg_0)
    kwarg_1_type = type(kwarg_1)

    @self.variant(static_argnums=(0,), static_argnames=('kwarg_1',))
    def fn_0(arg_0, arg_1, kwarg_0, kwarg_1):
      self.assertIsInstance(arg_0, arg_0_type)
      self.assertNotIsInstance(arg_1, arg_1_type)
      self.assertNotIsInstance(kwarg_0, kwarg_0_type)
      self.assertIsInstance(kwarg_1, kwarg_1_type)
      return DEFAULT_FN(arg_0 + kwarg_0, arg_1 + kwarg_1)

    actual_0 = fn_0(arg_0, arg_1, kwarg_0=kwarg_0, kwarg_1=kwarg_1)
    self.assertEqual(actual_0, 2 * expected)

    @self.variant(static_argnums=(1, 3), static_argnames=('kwarg_1',))
    def fn_1(arg_0, arg_1, kwarg_0, kwarg_1):
      self.assertNotIsInstance(arg_0, arg_0_type)
      self.assertIsInstance(arg_1, arg_1_type)
      self.assertNotIsInstance(kwarg_0, kwarg_0_type)
      self.assertIsInstance(kwarg_1, kwarg_1_type)
      return DEFAULT_FN(arg_0 + kwarg_0, arg_1 + kwarg_1)

    actual_1 = fn_1(arg_0, arg_1, kwarg_0=kwarg_0, kwarg_1=kwarg_1)
    self.assertEqual(actual_1, 2 * expected)

    @self.variant(static_argnums=(), static_argnames=('kwarg_0',))
    def fn_2(arg_0, arg_1, kwarg_0, kwarg_1):
      self.assertNotIsInstance(arg_0, arg_0_type)
      self.assertNotIsInstance(arg_1, arg_1_type)
      self.assertIsInstance(kwarg_0, kwarg_0_type)
      self.assertNotIsInstance(kwarg_1, kwarg_1_type)
      return DEFAULT_FN(arg_0 + kwarg_0, arg_1 + kwarg_1)

    actual_2 = fn_2(arg_0, arg_1, kwarg_0=kwarg_0, kwarg_1=kwarg_1)
    self.assertEqual(actual_2, 2 * expected)

    def fn_3(arg_0, arg_1):
      self.assertIsInstance(arg_0, arg_0_type)
      self.assertNotIsInstance(arg_1, arg_1_type)
      return DEFAULT_FN(arg_0, arg_1)

    fn_3_v0 = self.variant(static_argnums=0, static_argnames='arg_0')(fn_3)
    fn_3_v1 = self.variant(static_argnums=0)(fn_3)
    fn_3_v2 = self.variant(static_argnums=(), static_argnames='arg_0')(fn_3)
    self.assertEqual(fn_3_v0(arg_0, arg_1), expected)
    self.assertEqual(fn_3_v1(arg_0=arg_0, arg_1=arg_1), expected)
    self.assertEqual(fn_3_v1(arg_0, arg_1=arg_1), expected)
    self.assertEqual(fn_3_v2(arg_0=arg_0, arg_1=arg_1), expected)


def _test_fn_without_device(arg_0, arg_1):
  tc = unittest.TestCase()
  tc.assertNotIsInstance(arg_0, jax.Array)
  tc.assertNotIsInstance(arg_1, jax.Array)
  return DEFAULT_FN(arg_0, arg_1)


class WithoutDeviceTest(parameterized.TestCase):

  tests = _variant_default_tests_generator(
      fn=_test_fn_without_device,
      is_jit_context=False,
      which_variants=dict(without_device=True))

  @variants.variants(without_device=True)
  @parameterized.named_parameters(*DEFAULT_NAMED_PARAMS)
  def test_emplace(self, arg_0, arg_1, expected):
    (arg_0, arg_1) = self.variant(lambda x: x)((arg_0, arg_1))
    actual = _test_fn_without_device(arg_0, arg_1)
    self.assertEqual(actual, expected)


def _test_fn_with_device(arg_0, arg_1):
  tc = unittest.TestCase()
  tc.assertIsInstance(arg_0, jax.Array)
  tc.assertIsInstance(arg_1, jax.Array)
  return DEFAULT_FN(arg_0, arg_1)


class WithDeviceTest(parameterized.TestCase):

  tests = _variant_default_tests_generator(
      fn=_test_fn_with_device,
      is_jit_context=False,
      which_variants=dict(with_device=True))

  @variants.variants(with_device=True)
  @parameterized.named_parameters(*DEFAULT_NAMED_PARAMS)
  def test_emplace(self, arg_0, arg_1, expected):
    (arg_0, arg_1) = self.variant(lambda x: x)((arg_0, arg_1))
    actual = _test_fn_with_device(arg_0, arg_1)
    self.assertEqual(actual, expected)

  @variants.variants(with_device=True)
  @parameterized.named_parameters(*DEFAULT_NAMED_PARAMS)
  def test_ignore_argnums(self, arg_0, arg_1, expected):
    static_type = type(arg_0)

    @self.variant(ignore_argnums=(0, 2))
    def fn(arg_0, arg_1, float_arg):
      self.assertIsInstance(arg_0, static_type)
      self.assertIsInstance(arg_1, jax.Array)
      self.assertIsInstance(float_arg, float)
      return DEFAULT_FN(arg_0, arg_1)

    actual = fn(arg_0, arg_1, 5.3)
    self.assertEqual(actual, expected)


def _test_fn_single_device(arg_0, arg_1):
  tc = unittest.TestCase()
  tc.assertIn(np.shape(arg_0), {(), DEFAULT_NDARRAY_PARAMS_SHAPE})
  tc.assertIn(np.shape(arg_1), {(), DEFAULT_NDARRAY_PARAMS_SHAPE})
  res = DEFAULT_FN(arg_0, arg_1)
  psum_res = jax.lax.psum(res, axis_name='i')
  return psum_res


class WithPmapSingleDeviceTest(parameterized.TestCase):

  tests_single_device = _variant_default_tests_generator(
      fn=_test_fn_single_device,
      is_jit_context=True,
      which_variants=dict(with_pmap=True),
      n_devices=1)


class WithPmapAllAvailableDeviceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # Choose devices and a backend.
    n_tpu = asserts._ai.num_devices_available('tpu')
    n_gpu = asserts._ai.num_devices_available('gpu')
    if n_tpu > 1:
      self.n_devices, self.backend = n_tpu, 'tpu'
    elif n_gpu > 1:
      self.n_devices, self.backend = n_gpu, 'gpu'
    else:
      self.n_devices, self.backend = FLAGS['chex_n_cpu_devices'].value, 'cpu'

  @variants.variants(with_pmap=True)
  @parameterized.parameters(*DEFAULT_PARAMS)
  def test_pmap(self, arg_0, arg_1, expected):
    n_devices, backend = self.n_devices, self.backend
    n_copies = 3

    arg_0_type = type(arg_0)
    arg_1_type = type(arg_1)

    @self.variant(reduce_fn=None, n_devices=n_devices, backend=backend)
    def fn(arg_0, arg_1):
      self.assertNotIsInstance(arg_0, arg_0_type)
      self.assertNotIsInstance(arg_1, arg_1_type)
      asserts.assert_shape(arg_0, [n_copies])
      asserts.assert_shape(arg_1, [n_copies])

      res = arg_1 - arg_0
      psum_res = jax.lax.psum(res, axis_name='i')
      return psum_res

    arg_0 = jnp.zeros((n_copies,)) + arg_0
    arg_1 = jnp.zeros((n_copies,)) + arg_1
    actual = fn(arg_0, arg_1)
    self.assertEqual(actual.shape, (n_devices, n_copies))
    # Exponents of `n_devices`:
    # +1: psum() inside fn()
    # +1: jnp.sum() to aggregate results
    self.assertEqual(jnp.sum(actual), n_copies * n_devices**2 * expected)

  @variants.variants(with_pmap=True)
  @parameterized.parameters(*DEFAULT_PARAMS)
  def test_pmap_vmapped_fn(self, arg_0, arg_1, expected):
    n_devices, backend = self.n_devices, self.backend
    n_copies = 7
    actual_shape = (n_copies,) + DEFAULT_NDARRAY_PARAMS_SHAPE
    varg_0 = _scalar_to_ndarray(arg_0, actual_shape)
    varg_1 = _scalar_to_ndarray(arg_1, actual_shape)
    vexpected = _scalar_to_ndarray(expected)

    arg_0_type = type(varg_0)
    arg_1_type = type(varg_1)

    @self.variant(reduce_fn=None, n_devices=n_devices, backend=backend)
    def fn(arg_0, arg_1):
      self.assertNotIsInstance(arg_0, arg_0_type)
      self.assertNotIsInstance(arg_1, arg_1_type)

      @jax.vmap
      def vmapped_fn(arg_0, arg_1):
        self.assertIsInstance(arg_0, ArrayBatched)
        self.assertIsInstance(arg_1, ArrayBatched)
        asserts.assert_shape(arg_0, actual_shape[1:])
        asserts.assert_shape(arg_1, actual_shape[1:])
        return arg_1 - arg_0

      res = vmapped_fn(arg_0, arg_1)
      psum_res = jax.lax.psum(res, axis_name='i')
      return psum_res

    actual = fn(varg_0, varg_1)
    self.assertEqual(actual.shape, (n_devices,) + actual_shape)

    # Sum over `n_devices` and `n_copies` axes.
    actual = actual.sum(axis=0).sum(axis=0)

    # Exponents of `n_devices`:
    # +1: psum() inside fn()
    # +1: jnp.sum() to aggregate results
    np.testing.assert_array_equal(actual, n_copies * n_devices**2 * vexpected)

  @variants.variants(with_pmap=True)
  @parameterized.parameters(*DEFAULT_PARAMS)
  def test_pmap_static_argnums(self, arg_0, arg_1, expected):
    n_devices, backend = self.n_devices, self.backend
    n_copies = 5

    actual_shape = (n_copies,)
    varg_0 = _scalar_to_ndarray(arg_0, actual_shape)
    arg_0_type = type(varg_0)
    arg_1_type = type(arg_1)

    @self.variant(
        reduce_fn=None,
        n_devices=n_devices,
        backend=backend,
        static_argnums=(1,),
        axis_name='j',
    )
    def fn_static(arg_0, arg_1):
      self.assertNotIsInstance(arg_0, arg_0_type)
      self.assertIsInstance(arg_1, arg_1_type)
      asserts.assert_shape(arg_0, [n_copies])
      arg_1 = _scalar_to_ndarray(arg_1, actual_shape)
      asserts.assert_shape(arg_1, [n_copies])

      arg_1 = np.array(arg_1)  # don't stage out operations on arg_1
      psum_arg_1 = np.sum(jax.lax.psum(arg_1, axis_name='j'))
      self.assertEqual(psum_arg_1, arg_1[0] * (n_copies * n_devices))
      res = arg_1 - arg_0
      psum_res = jax.lax.psum(res, axis_name='j')
      return psum_res

    actual = fn_static(varg_0, arg_1)
    self.assertEqual(actual.shape, (n_devices, n_copies))
    # Exponents of `n_devices`:
    # +1: psum() inside fn()
    # +1: jnp.sum() to aggregate results
    self.assertEqual(jnp.sum(actual), n_copies * n_devices**2 * expected)

  @variants.variants(with_pmap=True)
  def test_pmap_static_argnums_zero(self):
    n_devices, backend = self.n_devices, self.backend
    n_copies = 5

    varg_0 = 10
    varg_1 = jnp.zeros(n_copies) + 20
    arg_0_type = type(varg_0)
    arg_1_type = type(varg_1)

    @self.variant(
        reduce_fn=None,
        n_devices=n_devices,
        backend=backend,
        static_argnums=0,
    )
    def fn_static(arg_0, arg_1):
      self.assertIsInstance(arg_0, arg_0_type)
      self.assertNotIsInstance(arg_1, arg_1_type)
      arg_0 = jnp.zeros(n_copies) + arg_0
      asserts.assert_shape(arg_0, [n_copies])
      asserts.assert_shape(arg_1, [n_copies])

      res = arg_1 - arg_0
      return jax.lax.psum(res, axis_name='i')

    actual = fn_static(varg_0, varg_1)
    self.assertEqual(actual.shape, (n_devices, n_copies))
    # Exponents of `n_devices`:
    # +1: psum() inside fn()
    # +1: jnp.sum() to aggregate results
    self.assertEqual(jnp.sum(actual), n_copies * n_devices**2 * 10)

  @variants.variants(with_pmap=True)
  def test_pmap_in_axes(self):
    n_devices, backend = self.n_devices, self.backend
    n_copies = 7

    varg_0 = jnp.zeros((n_devices, n_copies)) + 1
    varg_1 = jnp.zeros((n_devices, n_copies)) + 2
    arg_0_type = type(varg_0)
    arg_1_type = type(varg_1)

    @self.variant(
        broadcast_args_to_devices=False,
        reduce_fn=None,
        n_devices=n_devices,
        backend=backend,
        # Only 0 or None are supported (06/2020).
        in_axes=(0, None),
    )
    def fn(arg_0, arg_1):
      self.assertNotIsInstance(arg_0, arg_0_type)
      self.assertNotIsInstance(arg_1, arg_1_type)
      asserts.assert_shape(arg_0, [n_copies])
      asserts.assert_shape(arg_1, [n_devices, n_copies])

      res = arg_1 - arg_0
      psum_res = jax.lax.psum(res, axis_name='i')
      return psum_res

    actual = fn(varg_0, varg_1)
    self.assertEqual(actual.shape, (n_devices, n_devices, n_copies))
    self.assertEqual(jnp.sum(actual), n_copies * n_devices**3)

  @variants.variants(with_pmap=True)
  def test_pmap_wrong_axis_size(self):
    n_devices, backend = self.n_devices, self.backend

    @self.variant(
        broadcast_args_to_devices=False,
        n_devices=n_devices,
        backend=backend,
        # Only 0 or None are supported (06/2020).
        in_axes=(None, 0),
    )
    def fn(arg_0, arg_1):
      raise RuntimeError('This line should not be executed.')

    varg_0 = jnp.zeros(n_devices + 1)
    varg_1 = jnp.zeros(n_devices + 2)
    with self.assertRaisesRegex(
        ValueError, 'Pmappable.* axes size must be equal to number of devices.*'
        f'expected the first dim to be {n_devices}'):
      fn(varg_0, varg_1)


if __name__ == '__main__':
  absltest.main()
