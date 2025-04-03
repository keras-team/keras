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
"""Tests for `asserts.py`."""

import functools
import re

from absl.testing import absltest
from absl.testing import parameterized
from chex._src import asserts
from chex._src import asserts_internal
from chex._src import variants
import jax
import jax.numpy as jnp
import numpy as np

_get_err_regex = asserts_internal.get_err_regex
_num_devices_available = asserts_internal.num_devices_available


def as_arrays(arrays):
  return [np.asarray(a) for a in arrays]


def array_from_shape(*shape):
  return np.ones(shape=shape)


def emplace(arrays, dtype):
  return jnp.array(arrays, dtype=dtype)


class AssertsSwitchTest(parameterized.TestCase):
  """Tests for enable/disable_asserts."""

  def test_enable_disable_asserts(self):
    with self.assertRaisesRegex(AssertionError, _get_err_regex('scalar')):
      asserts.assert_scalar('test')

    asserts.disable_asserts()
    asserts.assert_scalar('test')

    asserts.enable_asserts()
    with self.assertRaisesRegex(AssertionError, _get_err_regex('scalar')):
      asserts.assert_scalar('test')

    asserts.disable_asserts()
    asserts.assert_is_divisible(13, 5)

    # To avoid side effects.
    asserts.enable_asserts()


class AssertMaxTracesTest(variants.TestCase):

  def setUp(self):
    super().setUp()
    asserts.clear_trace_counter()

  def _init(self, fn_, init_type, max_traces, kwargs, static_arg):
    """Initializes common test cases."""
    variant_kwargs = {}
    if static_arg:
      variant_kwargs['static_argnums'] = 1

    if kwargs:
      args, kwargs = [], {'n': max_traces}
    else:
      args, kwargs = [max_traces], {}

    if init_type == 't1':

      @asserts.assert_max_traces(*args, **kwargs)
      def fn(x, y):
        if static_arg:
          self.assertNotIsInstance(y, jax.core.Tracer)
        return fn_(x, y)

      fn_jitted = self.variant(fn, **variant_kwargs)
    elif init_type == 't2':

      def fn(x, y):
        if static_arg:
          self.assertNotIsInstance(y, jax.core.Tracer)
        return fn_(x, y)

      fn = asserts.assert_max_traces(fn, *args, **kwargs)
      fn_jitted = self.variant(fn, **variant_kwargs)
    elif init_type == 't3':

      def fn(x, y):
        if static_arg:
          self.assertNotIsInstance(y, jax.core.Tracer)
        return fn_(x, y)

      @self.variant(**variant_kwargs)
      @asserts.assert_max_traces(*args, **kwargs)
      def fn_jitted(x, y):
        self.assertIsInstance(x, jax.core.Tracer)
        return fn_(x, y)
    else:
      raise ValueError(f'Unknown type {init_type}.')

    return fn, fn_jitted

  @variants.variants(with_jit=True, with_pmap=True)
  @parameterized.named_parameters(
      variants.params_product((
          ('type1', 't1'),
          ('type2', 't2'),
          ('type3', 't3'),
      ), (
          ('args', False),
          ('kwargs', True),
      ), (
          ('no_static_arg', False),
          ('with_static_arg', True),
      ), (
          ('max_traces_0', 0),
          ('max_traces_1', 1),
          ('max_traces_2', 2),
          ('max_traces_10', 10),
      ),
                              named=True))
  def test_assert(self, init_type, kwargs, static_arg, max_traces):
    fn_ = lambda x, y: x + y
    fn, fn_jitted = self._init(fn_, init_type, max_traces, kwargs, static_arg)

    # Original function.
    for _ in range(max_traces + 3):
      self.assertEqual(fn(1, 2), 3)

    # Every call results in re-tracing because arguments' shapes are different.
    for i in range(max_traces):
      for k in range(5):
        arg = jnp.zeros(i + 1) + k
        np.testing.assert_array_equal(fn_jitted(arg, 2), arg + 2)

    # Original function.
    for _ in range(max_traces + 3):
      self.assertEqual(fn(1, 2), 3)
      self.assertEqual(fn([1], [2]), [1, 2])
      self.assertEqual(fn('a', 'b'), 'ab')

    # (max_traces + 1)-th re-tracing.
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('fn.* is traced > .* times!')):
      arg = jnp.zeros(max_traces + 1)
      fn_jitted(arg, 2)

  def test_incorrect_ordering(self):
    # pylint:disable=g-error-prone-assert-raises,unused-variable
    with self.assertRaisesRegex(ValueError, 'change wrappers ordering'):

      @asserts.assert_max_traces(1)
      @jax.jit
      def fn(_):
        pass

    def dummy_wrapper(fn):

      @functools.wraps(fn)
      def fn_wrapped():
        return fn()

      return fn_wrapped

    with self.assertRaisesRegex(ValueError, 'change wrappers ordering'):

      @asserts.assert_max_traces(1)
      @dummy_wrapper
      @jax.jit
      def fn_2():
        pass

    # pylint:enable=g-error-prone-assert-raises,unused-variable

  def test_redefined_traced_function(self):

    def outer_fn(x):

      @jax.jit
      @asserts.assert_max_traces(3)
      def inner_fn(y):
        return y.sum()

      return inner_fn(2 * x)

    self.assertEqual(outer_fn(1), 2)
    self.assertEqual(outer_fn(2), 4)
    self.assertEqual(outer_fn(3), 6)

    # Fails since the traced inner function is redefined at each call.
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('fn.* is traced > .* times!')):
      outer_fn(4)

    asserts.clear_trace_counter()
    for i in range(10):
      if i > 2:
        with self.assertRaisesRegex(
            AssertionError, _get_err_regex('fn.* is traced > .* times!')):
          outer_fn(1)
      else:
        outer_fn(1)

  def test_nested_functions(self):

    @jax.jit
    def jitted_outer_fn(x):

      @jax.jit
      @asserts.assert_max_traces(1)
      def inner_fn(y):
        return y.sum()

      return inner_fn(2 * x)

    # Inner assert_max_traces have no effect since the outer_fn is traced once.
    for i in range(10):
      self.assertEqual(jitted_outer_fn(i), 2 * i)


class ScalarAssertTest(parameterized.TestCase):

  def test_scalar(self):
    asserts.assert_scalar(1)
    asserts.assert_scalar(1.)
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('must be a scalar')):
      asserts.assert_scalar(np.array(1.))  # pytype: disable=wrong-arg-types

  def test_scalar_positive(self):
    asserts.assert_scalar_positive(0.5)
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('must be positive')):
      asserts.assert_scalar_positive(-0.5)

  def test_scalar_non_negative(self):
    asserts.assert_scalar_non_negative(0.5)
    asserts.assert_scalar_non_negative(0.)
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('must be non-negative')):
      asserts.assert_scalar_non_negative(-0.5)

  def test_scalar_negative(self):
    asserts.assert_scalar_negative(-0.5)
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('argument must be negative')):
      asserts.assert_scalar_negative(0.5)

  def test_scalar_in(self):
    asserts.assert_scalar_in(0.5, 0, 1)
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('argument must be in')):
      asserts.assert_scalar_in(-0.5, 0, 1)
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('argument must be in')):
      asserts.assert_scalar_in(1.5, 0, 1)

  def test_scalar_in_excluded(self):
    asserts.assert_scalar_in(0.5, 0, 1, included=False)
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('argument must be in')):
      asserts.assert_scalar_in(0, 0, 1, included=False)
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('argument must be in')):
      asserts.assert_scalar_in(1, 0, 1, included=False)


class EqualSizeAssertTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar_vector_matrix', [1, 2, [3], [[4, 5]]]),
      ('vector_matrix', [[1], [2], [[3, 5]]]),
      ('matrix', [[[1, 2]], [[3], [4], [5]]]),
  )
  def test_equal_size_should_fail(self, arrays):
    arrays = as_arrays(arrays)
    with self.assertRaisesRegex(
        AssertionError, _get_err_regex('Arrays have different sizes')
    ):
      asserts.assert_equal_size(arrays)

  @parameterized.named_parameters(
      ('scalar_vector_matrix', [1, 2, [3], [[4]]]),
      ('vector_matrix', [[1], [2], [[3]]]),
      ('matrix', [[[1, 2]], [[3], [4]]]),
  )
  def test_equal_size_should_pass(self, arrays):
    arrays = as_arrays(arrays)
    asserts.assert_equal_size(arrays)


class SizeAssertTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('wrong_size', [1, 2], 2),
      ('some_wrong_size', [[1, 2], [3, 4]], (2, 3)),
      ('wrong_common_shape', [[1, 2], [3, 4, 3]], 3),
      ('wrong_common_shape_2', [[1, 2, 3], [1, 2]], 2),
      ('some_wrong_size_set', [[1, 2], [3, 4]], (2, {3, 4})),
  )
  def test_size_should_fail(self, arrays, sizes):
    arrays = as_arrays(arrays)
    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex('input .+ has size .+ but expected .+')):
      asserts.assert_size(arrays, sizes)

  @parameterized.named_parameters(
      ('too_many_sizes', [[1]], (1, 1)),
      ('not_enough_sizes', [[1, 2], [3, 4], [5, 6]], (2, 2)),
  )
  def test_size_should_fail_wrong_length(self, arrays, sizes):
    arrays = as_arrays(arrays)
    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex('Length of `inputs` and `expected_sizes` must match')):
      asserts.assert_size(arrays, sizes)

  @parameterized.named_parameters(
      ('scalars', [1, 2], 1),
      ('vectors', [[1, 2], [3, 4, 5]], [2, 3]),
      ('matrices', [[[1, 2], [3, 4]]], 4),
      ('common_size_set', [[[1, 2], [3, 4]], [[1], [3]]], (4, {1, 2})),
  )
  def test_size_should_pass(self, arrays, sizes):
    arrays = as_arrays(arrays)
    asserts.assert_size(arrays, sizes)

  def test_pytypes_pass(self):
    arrays = as_arrays([[[1, 2], [3, 4]], [[1], [3]]])
    asserts.assert_size(arrays, (4, None))
    asserts.assert_size(arrays, (4, {1, 2}))
    asserts.assert_size(arrays, (4, ...))

  @parameterized.named_parameters(
      ('single_ellipsis', [[1, 2, 3, 4], [1, 2]], (..., 2)),
      ('multiple_ellipsis', [[1, 2, 3], [1, 2, 3]], (..., ...)),
  )
  def test_ellipsis_should_pass(self, arrays, expected_size):
    arrays = as_arrays(arrays)
    asserts.assert_size(arrays, expected_size)


class EqualShapeAssertTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('not_scalar', [1, 2, [3]]),
      ('wrong_rank', [[1], [2], 3]),
      ('wrong_length', [[1], [2], [3, 4]]),
  )
  def test_equal_shape_should_fail(self, arrays):
    arrays = as_arrays(arrays)
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('Arrays have different shapes')):
      asserts.assert_equal_shape(arrays)

  @parameterized.named_parameters(
      ('scalars', [1, 2, 3]),
      ('vectors', [[1], [2], [3]]),
      ('matrices', [[[1], [2]], [[3], [4]]]),
  )
  def test_equal_shape_should_pass(self, arrays):
    arrays = as_arrays(arrays)
    asserts.assert_equal_shape(arrays)

  @parameterized.named_parameters(
      ('scalars', [1, 2, 3]),
      ('vectors', [[1], [2], [[3, 4]]]),
  )
  def test_equal_shape_prefix_should_pass(self, arrays):
    arrays = as_arrays(arrays)
    asserts.assert_equal_shape_prefix(arrays, prefix_len=1)

  @parameterized.named_parameters(
      ('scalars', [1, 2, [3]]),
      ('vectors', [[1], [2], [[3], [4]]]),
  )
  def test_equal_shape_prefix_should_fail(self, arrays):
    arrays = as_arrays(arrays)
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('different shape prefixes')):
      asserts.assert_equal_shape_prefix(arrays, prefix_len=1)

  @parameterized.named_parameters(
      ('first_dim', [[2, 3], [2, 4], [2, 5]], 0),
      ('last_dim', [[3, 5, 7], [2, 7], [4, 7]], -1),  # Note different ranks.
      ('first_few_dims', [[1, 2, 3], [1, 2, 4], [1, 2, 5]], [0, 1]),
      ('first_and_last', [[1, 2, 1], [1, 3, 1], [1, 4, 1]], [0, 2]),
      ('first_and_last_neg', [[1, 2, 3, 4], [1, 5, 4], [1, 4]], [0, -1]),
  )
  def test_equal_shape_at_dims_should_pass(self, shapes, dims):
    arrays = [array_from_shape(*shape) for shape in shapes]
    asserts.assert_equal_shape(arrays, dims=dims)

  @parameterized.named_parameters(
      ('first_dim', [[1, 2], [2, 2]], 0),
      ('last_dim', [[1, 3], [1, 4]], 1),
      ('last_dim_neg', [[1, 3], [1, 4]], -1),
      ('multiple_dims', [[1, 2, 3], [1, 2, 4]], [0, 2]),
  )
  def test_equal_shape_at_dims_should_fail(self, shapes, dims):
    arrays = [array_from_shape(*shape) for shape in shapes]
    with self.assertRaisesRegex(
        AssertionError, _get_err_regex('have different shapes at dims')):
      asserts.assert_equal_shape(arrays, dims=dims)


class ShapeAssertTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('wrong_rank', [1], (1,)),
      ('wrong_shape', [1, 2], (1, 3)),
      ('some_wrong_shape', [[1, 2], [3, 4]], [(1, 2), (1, 3)]),
      ('wrong_common_shape', [[1, 2], [3, 4, 3]], (2,)),
      ('wrong_common_shape_2', [[1, 2, 3], [1, 2]], (2,)),
      ('some_wrong_shape_set', [[1, 2], [3, 4]], [(1, 2), (1, {3, 4})]),
  )
  def test_shape_should_fail(self, arrays, shapes):
    arrays = as_arrays(arrays)
    with self.subTest('list'):
      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex('input .+ has shape .+ but expected .+')):
        asserts.assert_shape(arrays, list(shapes))
    with self.subTest('tuple'):
      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex('input .+ has shape .+ but expected .+')):
        asserts.assert_shape(arrays, tuple(shapes))

  @parameterized.named_parameters(
      ('too_many_shapes', [[1]], [(1,), (2,)]),
      ('not_enough_shapes', [[1, 2], [3, 4]], [(3,)]),
  )
  def test_shape_should_fail_wrong_length(self, arrays, shapes):
    arrays = as_arrays(arrays)
    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex('Length of `inputs` and `expected_shapes` must match')):
      asserts.assert_shape(arrays, tuple(shapes))
    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex('Length of `inputs` and `expected_shapes` must match')):
      asserts.assert_shape(arrays, list(shapes))

  @parameterized.named_parameters(
      ('scalars', [1, 2], ()),
      ('vectors', [[1, 2], [3, 4, 5]], [(2,), (3,)]),
      ('matrices', [[[1, 2], [3, 4]]], (2, 2)),
      ('matrices_variable_shape', [[[1, 2], [3, 4]]], (None, 2)),
      ('vectors_common_shape', [[1, 2], [3, 4]], (2,)),
      ('variable_common_shape', [[[1, 2], [3, 4]], [[1], [3]]], (2, None)),
      ('common_shape_set', [[[1, 2], [3, 4]], [[1], [3]]], (2, {1, 2})),
  )
  def test_shape_should_pass(self, arrays, shapes):
    arrays = as_arrays(arrays)
    with self.subTest('tuple'):
      asserts.assert_shape(arrays, tuple(shapes))
    with self.subTest('list'):
      asserts.assert_shape(arrays, list(shapes))

  @parameterized.named_parameters(
      ('variable_shape', (2, None)),
      ('shape_set', (2, {1, 2})),
      ('suffix', (2, ...)),
  )
  def test_pytypes_pass(self, shape):
    arrays = as_arrays([[[1, 2], [3, 4]], [[1], [3]]])
    with self.subTest('tuple'):
      asserts.assert_shape(arrays, tuple(shape))
    with self.subTest('list'):
      asserts.assert_shape(arrays, list(shape))

  @parameterized.named_parameters(
      ('prefix_2', array_from_shape(2, 3, 4, 5, 6), (..., 4, 5, 6)),
      ('prefix_1', array_from_shape(3, 4, 5, 6), (..., 4, 5, 6)),
      ('prefix_0', array_from_shape(4, 5, 6), (..., 4, 5, 6)),
      ('inner_2', array_from_shape(2, 3, 4, 5, 6), (2, 3, ..., 6)),
      ('inner_1', array_from_shape(2, 3, 4, 6), (2, 3, ..., 6)),
      ('inner_0', array_from_shape(2, 3, 6), (2, 3, ..., 6)),
      ('suffix_2', array_from_shape(2, 3, 4, 5, 6), (2, 3, 4, ...)),
      ('suffix_1', array_from_shape(2, 3, 4, 5), (2, 3, 4, ...)),
      ('suffix_0', array_from_shape(2, 3, 4), (2, 3, 4, ...)),
  )
  def test_ellipsis_should_pass(self, array, expected_shape):
    with self.subTest('list'):
      asserts.assert_shape(array, list(expected_shape))
    with self.subTest('tuple'):
      asserts.assert_shape(array, tuple(expected_shape))

  @parameterized.named_parameters(
      ('prefix', array_from_shape(3, 1, 5), (..., 4, 5, 6)),
      ('inner_bad_prefix', array_from_shape(2, 1, 4, 6), (2, 3, ..., 6)),
      ('inner_bad_suffix', array_from_shape(2, 3, 1, 5), (2, 3, ..., 6)),
      ('inner_both_bad', array_from_shape(2, 1, 4, 5), (2, 3, ..., 6)),
      ('suffix', array_from_shape(2, 3, 1, 5), (2, 3, 4, ...)),
      ('short_rank_prefix', array_from_shape(2, 3), (..., 4, 5, 6)),
      ('short_rank_inner', array_from_shape(2, 3), (2, 3, ..., 6)),
      ('short_rank_suffix', array_from_shape(2, 3), (2, 3, 4, ...)),
  )
  def test_ellipsis_should_fail(self, array, expected_shape):
    with self.subTest('tuple'):
      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex('input .+ has shape .+ but expected .+')):
        asserts.assert_shape(array, tuple(expected_shape))
    with self.subTest('list'):
      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex('input .+ has shape .+ but expected .+')):
        asserts.assert_shape(array, list(expected_shape))

  @parameterized.named_parameters(
      ('prefix_and_suffix', array_from_shape(2, 3), (..., 2, 3, ...)),)
  def test_multiple_ellipses(self, array, expected_shape):
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        ValueError,
        '`expected_shape` may not contain more than one ellipsis, but got .+'):
      asserts.assert_shape(array, expected_shape)


def rank_array(n):
  return np.zeros(shape=[2] * n)


class BroadcastAssertTest(parameterized.TestCase):

  @parameterized.parameters(
      {'shape_a': (), 'shape_b': ()},
      {'shape_a': (), 'shape_b': (2, 3)},
      {'shape_a': (2, 3), 'shape_b': (2, 3)},
      {'shape_a': (1, 3), 'shape_b': (2, 3)},
      {'shape_a': (2, 1), 'shape_b': (2, 3)},
      {'shape_a': (4,), 'shape_b': (2, 3, 4)},
      {'shape_a': (3, 4), 'shape_b': (2, 3, 4)},
  )
  def test_shapes_are_broadcastable(self, shape_a, shape_b):
    asserts.assert_is_broadcastable(shape_a, shape_b)

  @parameterized.parameters(
      {'shape_a': (2,), 'shape_b': ()},
      {'shape_a': (2, 3, 4), 'shape_b': (3, 4)},
      {'shape_a': (3, 5), 'shape_b': (3, 4)},
      {'shape_a': (3, 4), 'shape_b': (3, 1)},
      {'shape_a': (3, 4), 'shape_b': (1, 4)},
  )
  def test_shapes_are_not_broadcastable(self, shape_a, shape_b):
    with self.assertRaises(AssertionError):
      asserts.assert_is_broadcastable(shape_a, shape_b)


class RankAssertTest(parameterized.TestCase):

  def test_rank_should_fail_array_expectations(self):
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        ValueError,
        'expected ranks should be .* but was an array'):
      asserts.assert_rank(rank_array(2), np.array([2]))

  def test_rank_should_fail_wrong_expectation_structure(self):
    # pytype: disable=wrong-arg-types
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        ValueError, 'Expected ranks should be integers or sets of integers'):
      asserts.assert_rank(rank_array(2), [[1, 2]])

    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        ValueError, 'Expected ranks should be integers or sets of integers'):
      asserts.assert_rank([rank_array(1), rank_array(2)], [[1], [2]])
    # pytype: enable=wrong-arg-types

  @parameterized.named_parameters(
      ('rank_1', rank_array(1), 2),
      ('rank_2', rank_array(2), 1),
      ('rank_3', rank_array(3), {2, 4}),
  )
  def test_rank_should_fail_single(self, array, rank):
    array = np.asarray(array)
    with self.assertRaisesRegex(
        AssertionError, _get_err_regex('input .+ has rank .+ but expected .+')):
      asserts.assert_rank(array, rank)

  @parameterized.named_parameters(
      ('wrong_1', [rank_array(1), rank_array(2)], [2, 2]),
      ('wrong_2', [rank_array(1), rank_array(2)], [1, 3]),
      ('wrong_3', [rank_array(1), rank_array(2)], [{2, 3}, 2]),
      ('wrong_4', [rank_array(1), rank_array(2)], [1, {1, 3}]),
  )
  def test_assert_rank_should_fail_sequence(self, arrays, ranks):
    arrays = as_arrays(arrays)
    with self.assertRaisesRegex(
        AssertionError, _get_err_regex('input .+ has rank .+ but expected .+')):
      asserts.assert_rank(arrays, ranks)

  @parameterized.named_parameters(
      ('not_enough_ranks', [1, 3, 4], [1, 1]),
      ('too_many_ranks', [1, 2], [1, 1, 1]),
  )
  def test_rank_should_fail_wrong_length(self, array, rank):
    array = np.asarray(array)
    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex('Length of inputs and expected_ranks must match.')):
      asserts.assert_rank(array, rank)

  @parameterized.named_parameters(
      ('rank_1', rank_array(1), 1),
      ('rank_2', rank_array(2), 2),
      ('rank_3', rank_array(3), {1, 2, 3}),
  )
  def test_rank_should_pass_single_input(self, array, rank):
    array = np.asarray(array)
    asserts.assert_rank(array, rank)

  @parameterized.named_parameters(
      ('rank_1', rank_array(1), 1),
      ('rank_2', rank_array(2), 2),
      ('rank_3', rank_array(3), {1, 2, 3}),
  )
  def test_rank_should_pass_repeated_input(self, array, rank):
    arrays = as_arrays([array] * 3)
    asserts.assert_rank(arrays, rank)

  @parameterized.named_parameters(
      ('single_option', [rank_array(1), rank_array(2)], {1, 2}),
      ('seq_options_1', [rank_array(1), rank_array(2)], [{1, 2}, 2]),
      ('seq_options_2', [rank_array(1), rank_array(2)], [1, {1, 2}]),
  )
  def test_rank_should_pass_multiple_options(self, arrays, ranks):
    arrays = as_arrays(arrays)
    asserts.assert_rank(arrays, ranks)


class TypeAssertTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('one_float', 3., int),
      ('one_int', 3, float),
      ('many_floats', [1., 2., 3.], int),
      ('many_floats_verbose', [1., 2., 3.], [float, float, int]),
      ('one_bool_as_float', True, float),
      ('one_bool_as_int', True, int),
      ('one_float_as_bool', 3., bool),
      ('one_int_as_bool', 3, bool),
  )
  def test_type_should_fail_scalar(self, scalars, wrong_type):
    with self.assertRaisesRegex(
        AssertionError, _get_err_regex('input .+ has type .+ but expected .+')):
      asserts.assert_type(scalars, wrong_type)

  @variants.variants(with_device=True, without_device=True)
  @parameterized.named_parameters(
      ('one_float_array', [1., 2.], float, int),
      ('one_int_array', [1, 2], int, float),
      ('bfloat16_array', [1, 2], jnp.bfloat16, jnp.float32),
      ('int8_array', [1, 2], jnp.int8, jnp.int32),
      ('float32_array', [1, 2], jnp.float32, np.integer),
  )
  def test_type_should_fail_array(self, array, dtype, wrong_type):
    array = self.variant(emplace)(array, dtype)
    with self.assertRaisesRegex(
        AssertionError, _get_err_regex('input .+ has type .+ but expected .+')):
      asserts.assert_type(array, wrong_type)

  @parameterized.named_parameters(
      ('one_float', 3., float),
      ('one_int', 3, int),
      ('one_bool', True, bool),
      ('many_floats', [1., 2., 3.], float),
      ('many_floats_verbose', [1., 2., 3.], [float, float, float]),
  )
  def test_type_should_pass_scalar(self, array, expected_type):
    asserts.assert_type(array, expected_type)

  @variants.variants(with_device=True, without_device=True)
  @parameterized.named_parameters(
      ('one_float_array', [1., 2.], float, float),
      ('one_int_array', [1, 2], int, int),
      ('one_integer_array', [1, 2], int, np.integer),
      ('one_bool_array', [True], bool, bool),
  )
  def test_type_should_pass_array(self, array, dtype, expected_type):
    array = self.variant(emplace)(array, dtype)
    asserts.assert_type(array, expected_type)

  def test_type_should_fail_mixed(self):
    a_float = 1.
    an_int = 2
    a_np_float = np.asarray([3., 4.])
    a_jax_int = jnp.asarray([5, 6])
    with self.assertRaisesRegex(
        AssertionError, _get_err_regex('input .+ has type .+ but expected .+')):
      asserts.assert_type([a_float, an_int, a_np_float, a_jax_int],
                          [float, int, float, float])

  def test_type_should_pass_mixed(self):
    a_float = 1.
    an_int = 2
    a_np_float = np.asarray([3., 4.])
    a_jax_int = jnp.asarray([5, 6])
    asserts.assert_type([a_float, an_int, a_np_float, a_jax_int],
                        [float, int, float, int])

  @parameterized.named_parameters(
      ('too_many_types', [1., 2], [float, int, float]),
      ('not_enough_types', [1., 2], [float]),
  )
  def test_type_should_fail_wrong_length(self, array, wrong_type):
    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex('Length of `inputs` and `expected_types` must match')):
      asserts.assert_type(array, wrong_type)


class AxisDimensionAssertionsTest(parameterized.TestCase):

  def test_assert_axis_dimension_pass(self):
    tensor = jnp.ones((3, 2, 7, 2))
    for i in range(-tensor.ndim, tensor.ndim):
      s = tensor.shape[i]
      asserts.assert_axis_dimension(tensor, axis=i, expected=s)

  def test_assert_axis_dimension_fail(self):
    tensor = jnp.ones((3, 2, 7, 2))
    for i in range(-tensor.ndim, tensor.ndim):
      s = tensor.shape[i]
      with self.assertRaisesRegex(
          AssertionError, _get_err_regex('Expected tensor to have dimension')):
        asserts.assert_axis_dimension(tensor, axis=i, expected=s + 1)

  def test_assert_axis_dimension_axis_invalid(self):
    tensor = jnp.ones((3, 2))
    for i in (2, -3):
      with self.assertRaisesRegex(AssertionError,
                                  _get_err_regex('not available')):
        asserts.assert_axis_dimension(tensor, axis=i, expected=1)

  def test_assert_axis_dimension_gt_pass(self):
    tensor = jnp.ones((3, 2, 7, 2))
    for i in range(-tensor.ndim, tensor.ndim):
      s = tensor.shape[i]
      asserts.assert_axis_dimension_gt(tensor, axis=i, val=s - 1)

  def test_assert_axis_dimension_gt_fail(self):
    tensor = jnp.ones((3, 2, 7, 2))
    for i in range(-tensor.ndim, tensor.ndim):
      s = tensor.shape[i]
      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex('Expected tensor to have dimension greater than')):
        asserts.assert_axis_dimension_gt(tensor, axis=i, val=s)

  def test_assert_axis_dimension_gt_axis_invalid(self):
    tensor = jnp.ones((3, 2))
    for i in (2, -3):
      with self.assertRaisesRegex(AssertionError,
                                  _get_err_regex('not available')):
        asserts.assert_axis_dimension_gt(tensor, axis=i, val=0)

  def test_assert_axis_dimension_gteq_pass(self):
    tensor = jnp.ones((3, 2, 7, 2))
    for i in range(-tensor.ndim, tensor.ndim):
      s = tensor.shape[i]
      asserts.assert_axis_dimension_gteq(tensor, axis=i, val=s)

  def test_assert_axis_dimension_gteq_fail(self):
    tensor = jnp.ones((3, 2, 7, 2))
    for i in range(-tensor.ndim, tensor.ndim):
      s = tensor.shape[i]
      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex('Expected tensor to have dimension greater than or')):
        asserts.assert_axis_dimension_gteq(tensor, axis=i, val=s + 1)

  def test_assert_axis_dimension_gteq_axis_invalid(self):
    tensor = jnp.ones((3, 2))
    for i in (2, -3):
      with self.assertRaisesRegex(AssertionError,
                                  _get_err_regex('not available')):
        asserts.assert_axis_dimension_gteq(tensor, axis=i, val=0)

  def test_assert_axis_dimension_lt_pass(self):
    tensor = jnp.ones((3, 2, 7, 2))
    for i in range(-tensor.ndim, tensor.ndim):
      s = tensor.shape[i]
      asserts.assert_axis_dimension_lt(tensor, axis=i, val=s + 1)

  def test_assert_axis_dimension_lt_fail(self):
    tensor = jnp.ones((3, 2, 7, 2))
    for i in range(-tensor.ndim, tensor.ndim):
      s = tensor.shape[i]
      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex('Expected tensor to have dimension less than')):
        asserts.assert_axis_dimension_lt(tensor, axis=i, val=s)

  def test_assert_axis_dimension_lt_axis_invalid(self):
    tensor = jnp.ones((3, 2))
    for i in (2, -3):
      with self.assertRaisesRegex(AssertionError,
                                  _get_err_regex('not available')):
        asserts.assert_axis_dimension_lt(tensor, axis=i, val=0)

  def test_assert_axis_dimension_lteq_pass(self):
    tensor = jnp.ones((3, 2, 7, 2))
    for i in range(-tensor.ndim, tensor.ndim):
      s = tensor.shape[i]
      asserts.assert_axis_dimension_lteq(tensor, axis=i, val=s)

  def test_assert_axis_dimension_lteq_fail(self):
    tensor = jnp.ones((3, 2, 7, 2))
    for i in range(-tensor.ndim, tensor.ndim):
      s = tensor.shape[i]
      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex('Expected tensor to have dimension less than or')):
        asserts.assert_axis_dimension_lteq(tensor, axis=i, val=s - 1)

  def test_assert_axis_dimension_lteq_axis_invalid(self):
    tensor = jnp.ones((3, 2))
    for i in (2, -3):
      with self.assertRaisesRegex(AssertionError,
                                  _get_err_regex('not available')):
        asserts.assert_axis_dimension_lteq(tensor, axis=i, val=0)

  def test_assert_axis_dimension_string_tensor(self):
    tensor = ['ab', 'cddd']
    asserts.assert_axis_dimension(tensor, axis=0, expected=2)
    asserts.assert_axis_dimension(np.array(tensor), axis=0, expected=2)


class TreeAssertionsTest(parameterized.TestCase):

  def _assert_tree_structs_validation(self, assert_fn):
    """Checks that assert_fn correctly processes invalid args' structs."""

    get_val = lambda: jnp.zeros([3])
    tree1 = [[get_val(), get_val()], get_val()]
    tree2 = [[get_val(), get_val()], get_val()]
    tree3 = [get_val(), [get_val(), get_val()]]
    tree4 = [get_val(), [get_val(), get_val()], get_val()]
    tree5 = dict(x=1, y=2, z=3)
    tree6 = dict(x=1, y=2, z=3, n=None)

    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex('Error in tree structs equality check.*trees 0 and 1')):
      assert_fn(tree1, tree5)

    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex('Error in tree structs equality check.*trees 0 and 1')):
      assert_fn(tree1, tree3)

    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex('Error in tree structs equality check.*trees 0 and 2')):
      assert_fn([], [], tree1)

    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex('Error in tree structs equality check.*trees 0 and 3')):
      assert_fn(tree2, tree1, tree2, tree3, tree1)

    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex('Error in tree structs equality check.*trees 0 and 2')):
      assert_fn(tree2, tree1, tree4)

    # Test `None`s.
    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex('Error in tree structs equality check.*trees 0 and 1')):
      assert_fn(tree5, tree6)

  def test_assert_tree_no_nones(self):
    with self.subTest('tree_no_nones'):
      tree_ok = {'a': [jnp.zeros((1,))], 'b': 1}
      asserts.assert_tree_no_nones(tree_ok)

    with self.subTest('tree_with_nones'):
      tree_with_none = {'a': [jnp.zeros((1,))], 'b': None}
      with self.assertRaisesRegex(
          AssertionError, _get_err_regex('Tree contains `None`')
      ):
        asserts.assert_tree_no_nones(tree_with_none)

    # Check `None`.
    with self.subTest('input_none'):
      with self.assertRaisesRegex(
          AssertionError, _get_err_regex('Tree contains `None`')
      ):
        asserts.assert_tree_no_nones(None)

  def test_tree_all_finite_passes_finite(self):
    finite_tree = {'a': jnp.ones((3,)), 'b': jnp.array([0.0, 0.0])}
    asserts.assert_tree_all_finite(finite_tree)
    self.assertTrue(asserts._assert_tree_all_finite_jittable(finite_tree))

  def test_tree_all_finite_should_fail_inf(self):
    inf_tree = {
        'finite_var': jnp.ones((3,)),
        'inf_var': jnp.array([0.0, jnp.inf]),
    }
    err_msg = 'Tree contains non-finite value'
    with self.assertRaisesRegex(AssertionError, _get_err_regex(err_msg)):
      asserts.assert_tree_all_finite(inf_tree)

    with self.assertRaisesRegex(ValueError, err_msg):
      asserts._assert_tree_all_finite_jittable(inf_tree)

  def test_assert_trees_all_equal_prng_keys(self):
    tree1 = {'a': jnp.array([3]), 'key': jax.random.split(jax.random.key(1))}
    tree2 = {'a': jnp.array([3]), 'key': jax.random.split(jax.random.key(2))}
    asserts.assert_trees_all_equal(tree1, tree1)  # OK

    err_regex = _get_err_regex(r'Trees 0 and 1 differ in leaves \'key\'')
    with self.assertRaisesRegex(AssertionError, err_regex):
      asserts.assert_trees_all_equal(tree1, tree2)  # Fail: not equal

  def test_assert_trees_all_equal_passes_same_tree(self):
    tree = {
        'a': [jnp.zeros((1,))],
        'b': ([0], (0,), 0),
    }
    asserts.assert_trees_all_equal(tree, tree)
    tree = jax.tree.map(jnp.asarray, tree)
    self.assertTrue(asserts._assert_trees_all_equal_jittable(tree, tree))

  def test_assert_trees_all_equal_passes_values_equal(self):
    tree1 = (jnp.array([0.0, 0.0]),)
    tree2 = (jnp.array([0.0, 0.0]),)
    asserts.assert_trees_all_equal(tree1, tree2)
    self.assertTrue(asserts._assert_trees_all_equal_jittable(tree1, tree2))

  def test_assert_trees_all_equal_fail_values_close_but_not_equal(self):
    tree1 = (jnp.array([1.0, 1.0]),)
    tree2 = (jnp.array([1.0, 1.0 + 5e-7]),)
    error_msg = 'Values not exactly equal'
    with self.assertRaisesRegex(AssertionError, _get_err_regex(error_msg)):
      asserts.assert_trees_all_equal(tree1, tree2)
    with self.assertRaisesRegex(ValueError, error_msg):
      asserts._assert_trees_all_equal_jittable(tree1, tree2)

  def test_assert_trees_all_equal_strict_mode(self):
    # See 'notes' section of
    # https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_array_equal.html
    # for details about the 'strict' mode of `numpy.testing.assert_array_equal`.
    # tldr; it has special handling for scalar values (by default).
    tree1 = {'a': jnp.array([1.0], dtype=jnp.float32), 'b': 0.0}
    tree2 = {'a': jnp.array(1.0, dtype=jnp.float32), 'b': 0.0}

    asserts.assert_trees_all_equal(tree1, tree2)
    asserts.assert_trees_all_equal(tree1, tree2, strict=False)
    err_regex = _get_err_regex(r'Trees 0 and 1 differ in leaves \'a\'')
    with self.assertRaisesRegex(AssertionError, err_regex):
      asserts.assert_trees_all_equal(tree1, tree2, strict=True)

    err_regex = r'Trees 0 and 1 differ in leaves'
    with self.assertRaisesRegex(ValueError, err_regex):
      asserts._assert_trees_all_equal_jittable(tree1, tree2, strict=True)

    # We do not implement this special scalar handling in the jittable
    # assertion (it's possible, but doesn't seem worth the effort).
    err_regex = r'`strict=False` is not implemented'
    with self.assertRaisesRegex(NotImplementedError, err_regex):
      asserts._assert_trees_all_equal_jittable(tree1, tree2, strict=False)

  def test_assert_trees_all_close_passes_same_tree(self):
    tree = {
        'a': [jnp.zeros((1,))],
        'b': ([0], (0,), 0),
    }
    asserts.assert_trees_all_close(tree, tree)
    tree = jax.tree.map(jnp.asarray, tree)
    self.assertTrue(asserts._assert_trees_all_close_jittable(tree, tree))

  def test_assert_trees_all_close_passes_values_equal(self):
    tree1 = (jnp.array([0.0, 0.0]),)
    tree2 = (jnp.array([0.0, 0.0]),)
    asserts.assert_trees_all_close(tree1, tree2)
    self.assertTrue(asserts._assert_trees_all_close_jittable(tree1, tree2))

  def test_assert_trees_all_close_passes_values_close_but_not_equal(self):
    tree1 = (jnp.array([1.0, 1.0]),)
    tree2 = (jnp.array([1.0, 1.0 + 5e-7]),)
    asserts.assert_trees_all_close(tree1, tree2, rtol=1e-6)
    self.assertTrue(
        asserts._assert_trees_all_close_jittable(tree1, tree2, rtol=1e-6))

  def test_assert_trees_all_close_bfloat16(self):
    tree1 = {'a': jnp.asarray([0.8, 1.6], dtype=jnp.bfloat16)}
    tree2 = {
        'a': jnp.asarray([0.8, 1.6], dtype=jnp.bfloat16).astype(jnp.float32)
    }
    tree3 = {'a': jnp.asarray([0.8, 1.7], dtype=jnp.bfloat16)}
    asserts.assert_trees_all_close(tree1, tree1)
    asserts.assert_trees_all_close(tree1, tree2)
    self.assertTrue(asserts._assert_trees_all_close_jittable(tree1, tree2))

    err_msg = 'Values not approximately equal'
    err_regex = _get_err_regex(err_msg)

    with self.assertRaisesRegex(AssertionError, err_regex):
      asserts.assert_trees_all_close(tree1, tree3)
    with self.assertRaisesRegex(ValueError, err_msg):
      asserts._assert_trees_all_close_jittable(tree1, tree3)

    with self.assertRaisesRegex(AssertionError, err_regex):
      asserts.assert_trees_all_close(tree2, tree3)
    with self.assertRaisesRegex(ValueError, err_msg):
      asserts._assert_trees_all_close_jittable(tree2, tree3)

  def test_assert_trees_all_close_ulp_jittable_raises_valueerror(self):
    tree = (jnp.array([1.0]),)
    err_msg = 'assert_trees_all_close_ulp is not supported within JIT contexts.'
    err_regex = _get_err_regex(err_msg)
    with self.assertRaisesRegex(RuntimeError, err_regex):
      asserts._assert_trees_all_close_ulp_jittable(tree, tree)

  def test_assert_trees_all_close_ulp_passes_same_tree(self):
    tree = {
        'a': [jnp.zeros((1,))],
        'b': ([0], (0,), 0),
    }
    asserts.assert_trees_all_close_ulp(tree, tree)

  def test_assert_trees_all_close_ulp_passes_values_equal(self):
    tree1 = (jnp.array([0.0, 0.0]),)
    tree2 = (jnp.array([0.0, 0.0]),)
    try:
      asserts.assert_trees_all_close_ulp(tree1, tree2)
    except AssertionError:
      self.fail('assert_trees_all_close_ulp raised AssertionError')

  def test_assert_trees_all_close_ulp_passes_values_within_maxulp(self):
    # np.spacing(np.float32(1 << 23)) == 1.0.
    value_where_ulp_is_1 = np.float32(1 << 23)
    tree1 = (jnp.array([value_where_ulp_is_1, value_where_ulp_is_1]),)
    tree2 = (jnp.array([value_where_ulp_is_1, value_where_ulp_is_1 + 1.0]),)
    assert tree2[0][0] != tree2[0][1]
    try:
      asserts.assert_trees_all_close_ulp(tree1, tree2, maxulp=2)
    except AssertionError:
      self.fail('assert_trees_all_close_ulp raised AssertionError')

  def test_assert_trees_all_close_ulp_passes_values_maxulp_apart(self):
    # np.spacing(np.float32(1 << 23)) == 1.0.
    value_where_ulp_is_1 = np.float32(1 << 23)
    tree1 = (jnp.array([value_where_ulp_is_1, value_where_ulp_is_1]),)
    tree2 = (jnp.array([value_where_ulp_is_1, value_where_ulp_is_1 + 1.0]),)
    assert tree2[0][0] != tree2[0][1]
    try:
      asserts.assert_trees_all_close_ulp(tree1, tree2, maxulp=1)
    except AssertionError:
      self.fail('assert_trees_all_close_ulp raised AssertionError')

  def test_assert_trees_all_close_ulp_fails_values_gt_maxulp_apart(self):
    # np.spacing(np.float32(1 << 23)) == 1.0.
    value_where_ulp_is_1 = np.float32(1 << 23)
    tree1 = (jnp.array([value_where_ulp_is_1, value_where_ulp_is_1]),)
    tree2 = (jnp.array([value_where_ulp_is_1, value_where_ulp_is_1 + 2.0]),)
    assert tree2[0][0] != tree2[0][1]
    err_msg = re.escape(
        'not almost equal up to 1 ULP (max difference is 2 ULP)'
    )
    err_regex = _get_err_regex(err_msg)
    with self.assertRaisesRegex(AssertionError, err_regex):
      asserts.assert_trees_all_close_ulp(tree1, tree2, maxulp=1)

  def test_assert_trees_all_close_ulp_fails_bfloat16(self):
    tree_f32 = (jnp.array([0.0]),)
    tree_bf16 = (jnp.array([0.0], dtype=jnp.bfloat16),)
    err_msg = 'ULP assertions are not currently supported for bfloat16.'
    err_regex = _get_err_regex(err_msg)
    with self.assertRaisesRegex(ValueError, err_regex):  # pylint: disable=g-error-prone-assert-raises
      asserts.assert_trees_all_close_ulp(tree_bf16, tree_bf16)
    with self.assertRaisesRegex(ValueError, err_regex):  # pylint: disable=g-error-prone-assert-raises
      asserts.assert_trees_all_close_ulp(tree_bf16, tree_f32)

  def test_assert_tree_has_only_ndarrays(self):
    # Check correct inputs.
    asserts.assert_tree_has_only_ndarrays({'a': jnp.zeros(1), 'b': np.ones(3)})
    asserts.assert_tree_has_only_ndarrays(np.zeros(4))
    asserts.assert_tree_has_only_ndarrays(())

    # Check incorrect inputs.
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('\'b\' is not an ndarray')):
      asserts.assert_tree_has_only_ndarrays({'a': jnp.zeros((1,)), 'b': 1})
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('\'b/1\' is not an ndarray')):
      asserts.assert_tree_has_only_ndarrays({'a': jnp.zeros(101), 'b': [1, 2]})

  def test_assert_tree_is_on_host(self):
    cpu = jax.local_devices(backend='cpu')[0]

    # Check Numpy arrays.
    for flag in (False, True):
      asserts.assert_tree_is_on_host({'a': np.zeros(1), 'b': np.ones(3)},
                                     allow_cpu_device=flag)
      asserts.assert_tree_is_on_host(np.zeros(4), allow_cpu_device=flag)
      asserts.assert_tree_is_on_host(
          jax.device_get(jax.device_put(np.ones(3))), allow_cpu_device=flag)
      asserts.assert_tree_is_on_host((), allow_cpu_device=flag)

    # Check DeviceArray (for platforms other than CPU).
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('\'a\' resides on')):
      asserts.assert_tree_is_on_host({'a': jnp.zeros(1)},
                                     allow_cpu_device=False)
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('\'a\' resides on')):
      asserts.assert_tree_is_on_host({'a': jax.device_put(np.zeros(1))},
                                     allow_cpu_device=False)

    # Check Jax arrays on CPU.
    cpu_arr = jax.device_put(np.ones(5), cpu)
    asserts.assert_tree_is_on_host({'a': cpu_arr})
    asserts.assert_tree_is_on_host({'a': np.zeros(1), 'b': cpu_arr})

    # Check sharded Jax arrays on CPUs.
    asserts.assert_tree_is_on_host(
        {'a': jax.device_put_replicated(np.zeros(1), (cpu,))},
        allow_cpu_device=True,
        allow_sharded_arrays=True,
    )

    # Disallow JAX arrays on CPU.
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('\'a\' resides on.*CPU')):
      asserts.assert_tree_is_on_host({'a': cpu_arr},
                                     allow_cpu_device=False)

    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('\'b\' resides on.*CPU')):
      asserts.assert_tree_is_on_host({'a': np.zeros(1), 'b': cpu_arr},
                                     allow_cpu_device=False)

    # Check incorrect inputs.
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('\'b\' is not an ndarray')):
      asserts.assert_tree_is_on_host({'a': np.zeros(1), 'b': 1})

    # ShardedArrays are disallowed.
    with self.assertRaisesRegex(
        AssertionError, _get_err_regex('sharded arrays are disallowed')
    ):
      asserts.assert_tree_is_on_host(
          {'a': jax.device_put_replicated(np.zeros(1), (cpu,))},
          allow_cpu_device=False,
      )

    # ShardedArrays on CPUs, CPUs disallowed.
    with self.assertRaisesRegex(
        AssertionError, _get_err_regex("'a' is sharded and resides on.*CPU")
    ):
      asserts.assert_tree_is_on_host(
          {'a': jax.device_put_replicated(np.zeros(1), (cpu,))},
          allow_cpu_device=False,
          allow_sharded_arrays=True,
      )

  def test_assert_tree_is_on_device(self):
    # Check CPU platform.
    cpu = jax.local_devices(backend='cpu')[0]
    to_cpu = lambda x: jax.device_put(x, cpu)

    cpu_tree = {'a': to_cpu(np.zeros(1)), 'b': to_cpu(np.ones(3))}
    asserts.assert_tree_is_on_device(cpu_tree, device=cpu)
    asserts.assert_tree_is_on_device(cpu_tree, platform='cpu')
    asserts.assert_tree_is_on_device(cpu_tree, platform=['cpu'])
    asserts.assert_tree_is_on_device(cpu_tree, device=cpu, platform='')

    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('\'a\' resides on \'cpu\'')):
      asserts.assert_tree_is_on_device(cpu_tree, platform='tpu')

    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('\'b\' resides on \'cpu\'')):
      asserts.assert_tree_is_on_device(cpu_tree, platform=('tpu', 'gpu'))

    # Check TPU platform (if available).
    if _num_devices_available('tpu') > 1:
      tpu_1, tpu_2 = jax.devices('tpu')[:2]
      to_tpu_1 = lambda x: jax.device_put(x, tpu_1)
      to_tpu_2 = lambda x: jax.device_put(x, tpu_2)

      tpu_1_tree = {'a': to_tpu_1(np.zeros(1)), 'b': to_tpu_1(np.ones(3))}
      tpu_2_tree = {'a': to_tpu_2(np.zeros(1)), 'b': to_tpu_2(np.ones(3))}
      tpu_1_2_tree = {'a': to_tpu_1(np.zeros(1)), 'b': to_tpu_2(np.ones(3))}

      # Device asserts.
      asserts.assert_tree_is_on_device(tpu_1_tree, device=tpu_1)
      asserts.assert_tree_is_on_device(tpu_2_tree, device=tpu_2)

      with self.assertRaisesRegex(
          AssertionError, _get_err_regex(r"'a' resides on.*TpuDevice\(id=0")
      ):
        asserts.assert_tree_is_on_device(tpu_1_tree, device=tpu_2)

      with self.assertRaisesRegex(
          AssertionError, _get_err_regex(r"'a' resides on.*TpuDevice\(id=1")
      ):
        asserts.assert_tree_is_on_device(tpu_2_tree, device=tpu_1)

      with self.assertRaisesRegex(
          AssertionError, _get_err_regex("'a' resides on .*Cpu")
      ):
        asserts.assert_tree_is_on_device(cpu_tree, device=tpu_2)

      # Platform asserts.
      asserts.assert_tree_is_on_device(tpu_1_tree, platform='tpu')
      asserts.assert_tree_is_on_device(tpu_2_tree, platform='tpu')
      with self.assertRaisesRegex(AssertionError,
                                  _get_err_regex('\'a\' resides on \'tpu\'')):
        asserts.assert_tree_is_on_device(tpu_1_tree, platform='cpu')

      with self.assertRaisesRegex(AssertionError,
                                  _get_err_regex('\'a\' resides on \'tpu\'')):
        asserts.assert_tree_is_on_device(tpu_2_tree, platform='gpu')

      # Mixed cases.
      asserts.assert_tree_is_on_device(tpu_1_2_tree, platform='tpu')
      asserts.assert_tree_is_on_device((tpu_1_2_tree, cpu_tree),
                                       platform=('cpu', 'tpu'))
      with self.assertRaisesRegex(AssertionError,
                                  _get_err_regex('\'1/a\' resides on \'cpu\'')):
        asserts.assert_tree_is_on_device((tpu_1_2_tree, cpu_tree),
                                         platform='tpu')
      with self.assertRaisesRegex(AssertionError,
                                  _get_err_regex('\'0/a\' resides on \'tpu\'')):
        asserts.assert_tree_is_on_device((tpu_1_2_tree, cpu_tree),
                                         platform=('cpu', 'gpu'))

    # Check incorrect inputs.
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('\'b\' is not an ndarray')):
      asserts.assert_tree_is_on_device({'a': np.zeros(1), 'b': 1})

    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('\'b\' has unexpected type')):
      asserts.assert_tree_is_on_device({'a': jnp.zeros(1), 'b': np.ones(3)})

    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('\'a\' is a ShardedDeviceArra')):
      # ShardedArrays are disallowed.
      asserts.assert_tree_is_on_device(
          {'a': jax.device_put_replicated(np.zeros(1), (cpu,))}, device=cpu)

  def test_assert_tree_is_sharded(self):
    np_tree = {'a': np.zeros(1), 'b': np.ones(3)}

    def _format(*devs):
      return re.escape(f'{devs}')

    # Check single-device case.
    cpu = jax.local_devices(backend='cpu')[0]
    cpu_tree = jax.device_put_replicated(np_tree, (cpu,))

    asserts.assert_tree_is_sharded(cpu_tree, devices=(cpu,))
    asserts.assert_tree_is_sharded((), devices=(cpu,))

    with self.assertRaisesRegex(
        AssertionError, _get_err_regex(r'\'a\' is sharded.*expected \(\)')):
      asserts.assert_tree_is_sharded(cpu_tree, devices=())

    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex(f'\'a\' is sharded across {_format(cpu)}.*'
                       f'expected {_format(cpu, cpu)}')):
      asserts.assert_tree_is_sharded(cpu_tree, devices=(cpu, cpu))

    # Check multiple-devices case (if available).
    if _num_devices_available('tpu') > 1:
      tpu_1, tpu_2 = jax.devices('tpu')[:2]

      tpu_1_tree = jax.device_put_replicated(np_tree, (tpu_1,))
      tpu_2_tree = jax.device_put_replicated(np_tree, (tpu_2,))
      tpu_1_2_tree = jax.device_put_replicated(np_tree, (tpu_1, tpu_2))
      tpu_2_1_tree = jax.device_put_replicated(np_tree, (tpu_2, tpu_1))

      asserts.assert_tree_is_sharded(tpu_1_2_tree, devices=(tpu_1, tpu_2))
      asserts.assert_tree_is_sharded(tpu_2_1_tree, devices=(tpu_2, tpu_1))

      # Wrong device.
      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex(f'\'a\' is sharded across {_format(tpu_1)}.*'
                         f'expected {_format(tpu_2)}')):
        asserts.assert_tree_is_sharded(tpu_1_tree, devices=(tpu_2,))

      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex(f'\'a\' is sharded across {_format(cpu)}.*'
                         f'expected {_format(tpu_2)}')):
        asserts.assert_tree_is_sharded(cpu_tree, devices=(tpu_2,))

      # Too many devices.
      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex(f'\'a\' is sharded across {_format(tpu_1)}.*'
                         f'expected {_format(tpu_1, tpu_2)}')):
        asserts.assert_tree_is_sharded(tpu_1_tree, devices=(tpu_1, tpu_2))

      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex(f'\'a\' is sharded across {_format(tpu_1, tpu_2)}.*'
                         f'expected {_format(tpu_1, tpu_2, cpu)}')):
        asserts.assert_tree_is_sharded(
            tpu_1_2_tree, devices=(tpu_1, tpu_2, cpu))

      # Wrong order.
      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex(f'\'a\' is sharded across {_format(tpu_2, tpu_1)}.*'
                         f'expected {_format(tpu_1, tpu_2)}')):
        asserts.assert_tree_is_sharded(tpu_2_1_tree, devices=(tpu_1, tpu_2))

      # Mixed cases.
      mixed_tree = (tpu_1_tree, tpu_2_tree)

      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex(f'\'0/a\' is sharded across {_format(tpu_1)}.*'
                         f'expected {_format(tpu_2)}')):
        asserts.assert_tree_is_sharded(mixed_tree, devices=(tpu_2,))
      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex(f'\'1/a\' is sharded across {_format(tpu_2)}.*'
                         f'expected {_format(tpu_1)}')):
        asserts.assert_tree_is_sharded(mixed_tree, devices=(tpu_1,))

      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex(f'\'0/a\' is sharded across {_format(tpu_1)}.*'
                         f'expected {_format(tpu_1, tpu_2)}')):
        asserts.assert_tree_is_sharded(mixed_tree, devices=(tpu_1, tpu_2))

    # Check incorrect inputs.
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('\'1\' is not an ndarray')):
      asserts.assert_tree_is_sharded((cpu_tree, 1123), devices=(cpu,))

    with self.assertRaisesRegex(
        AssertionError, _get_err_regex('\'a\' is not a jax.Array')):
      asserts.assert_tree_is_sharded({'a': np.zeros(1)}, devices=(cpu,))

    with self.assertRaisesRegex(
        AssertionError, _get_err_regex('\'a\' is not sharded')):
      asserts.assert_tree_is_sharded({'a': jnp.zeros(1)}, devices=(cpu,))

    with self.assertRaisesRegex(
        AssertionError, _get_err_regex("'a' is not sharded.*Cpu")
    ):
      asserts.assert_tree_is_sharded({'a': jax.device_put(np.zeros(1), cpu)},
                                     devices=(cpu,))

  def test_assert_trees_all_close_fails_different_structure(self):
    self._assert_tree_structs_validation(asserts.assert_trees_all_close)

  def test_assert_trees_all_close_fails_values_differ(self):
    tree1 = jnp.array([0.0, 2.0])
    tree2 = jnp.array([0.0, 2.1])
    asserts.assert_trees_all_close(tree1, tree2, atol=0.1)
    with self.assertRaisesRegex(
        AssertionError, _get_err_regex('Values not approximately equal')):
      asserts.assert_trees_all_close(tree1, tree2, atol=0.01)

    asserts.assert_trees_all_close(tree1, tree2, rtol=0.1)
    with self.assertRaisesRegex(
        AssertionError, _get_err_regex('Values not approximately equal')):
      asserts.assert_trees_all_close(tree1, tree2, rtol=0.01)

  def test_assert_trees_all_equal_sizes(self):
    get_val = lambda s1, s2: jnp.zeros([s1, s2])
    tree1 = dict(a1=get_val(3, 1), d=dict(a2=get_val(4, 1), a3=get_val(5, 3)))
    tree2 = dict(a1=get_val(3, 1), d=dict(a2=get_val(4, 1), a3=get_val(5, 3)))
    tree3 = dict(a1=get_val(3, 1), d=dict(a2=get_val(4, 2), a3=get_val(5, 3)))

    self._assert_tree_structs_validation(asserts.assert_trees_all_equal_sizes)
    asserts.assert_trees_all_equal_sizes(tree1, tree1)
    asserts.assert_trees_all_equal_sizes(tree2, tree1)

    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex(
            r'Trees 0 and 1 differ in leaves \'d/a2\': sizes: 4 != 8'
        )):
      asserts.assert_trees_all_equal_sizes(tree1, tree3)

    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex(
            r'Trees 0 and 3 differ in leaves \'d/a2\': sizes: 4 != 8'
        )):
      asserts.assert_trees_all_equal_sizes(tree1, tree2, tree2, tree3, tree1)

  def test_assert_trees_all_equal_shapes(self):
    get_val = lambda s: jnp.zeros([s])
    tree1 = dict(a1=get_val(3), d=dict(a2=get_val(4), a3=get_val(5)))
    tree2 = dict(a1=get_val(3), d=dict(a2=get_val(4), a3=get_val(5)))
    tree3 = dict(a1=get_val(3), d=dict(a2=get_val(7), a3=get_val(5)))

    self._assert_tree_structs_validation(asserts.assert_trees_all_equal_shapes)
    asserts.assert_trees_all_equal_shapes(tree1, tree1)
    asserts.assert_trees_all_equal_shapes(tree2, tree1)

    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex(
            r'Trees 0 and 1 differ in leaves \'d/a2\': shapes: \(4,\) != \(7,\)'
        )):
      asserts.assert_trees_all_equal_shapes(tree1, tree3)

    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex(
            r'Trees 0 and 3 differ in leaves \'d/a2\': shapes: \(4,\) != \(7,\)'
        )):
      asserts.assert_trees_all_equal_shapes(tree1, tree2, tree2, tree3, tree1)

  def test_assert_trees_all_equal_structs(self):
    get_val = lambda: jnp.zeros([3])
    tree1 = [[get_val(), get_val()], get_val()]
    tree2 = [[get_val(), get_val()], get_val()]
    tree3 = [get_val(), [get_val(), get_val()]]

    asserts.assert_trees_all_equal_structs(tree1, tree2, tree2, tree1)
    asserts.assert_trees_all_equal_structs(tree3, tree3)
    self._assert_tree_structs_validation(asserts.assert_trees_all_equal_structs)

  @parameterized.named_parameters(
      ('scalars', ()),
      ('vectors', (3,)),
      ('matrices', (3, 2)),
  )
  def test_assert_tree_shape_prefix(self, shape):
    tree = {'x': {'y': np.zeros([3, 2])}, 'z': np.zeros([3, 2, 1])}
    with self.subTest('tuple'):
      asserts.assert_tree_shape_prefix(tree, tuple(shape))
    with self.subTest('list'):
      asserts.assert_tree_shape_prefix(tree, list(shape))

  def test_leaf_shape_should_fail_wrong_length(self):
    tree = {'x': {'y': np.zeros([3, 2])}, 'z': np.zeros([3, 2, 1])}
    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex(r'leaf \'x/y\' has a shape of length 2')):
      asserts.assert_tree_shape_prefix(tree, (3, 2, 1))
    with self.assertRaisesRegex(
        AssertionError,
        _get_err_regex(r'leaf \'x/y\' has a shape of length 2')):
      asserts.assert_tree_shape_prefix(tree, [3, 2, 1])

  @parameterized.named_parameters(
      ('scalars', ()),
      ('vectors', (1,)),
      ('matrices', (2, 1)),
  )
  def test_assert_tree_shape_suffix_matching(self, shape):
    tree = {'x': {'y': np.zeros([4, 2, 1])}, 'z': np.zeros([2, 1])}
    with self.subTest('tuple'):
      asserts.assert_tree_shape_suffix(tree, tuple(shape))
    with self.subTest('list'):
      asserts.assert_tree_shape_suffix(tree, list(shape))

  @parameterized.named_parameters(
      ('bad_suffix_leaf_1', 'z', (1, 1), (2, 1)),
      ('bad_suffix_leaf_2', 'x/y', (2, 1), (1, 1)),
  )
  def test_assert_tree_shape_suffix_mismatch(self, leaf, shape_true, shape):
    tree = {'x': {'y': np.zeros([4, 2, 1])}, 'z': np.zeros([1, 1])}

    error_msg = (
        r'Tree leaf \'' + str(leaf) + '\'.*different from expected: '
        + re.escape(str(shape_true)) + ' != ' + re.escape(str(shape))
    )
    with self.subTest('tuple'):
      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex(
              error_msg)):
        asserts.assert_tree_shape_suffix(tree, tuple(shape))

    with self.subTest('list'):
      with self.assertRaisesRegex(
          AssertionError,
          _get_err_regex(
              error_msg)):
        asserts.assert_tree_shape_suffix(tree, list(shape))

  def test_assert_tree_shape_suffix_long_suffix(self):
    tree = {'x': {'y': np.zeros([4, 2, 1])}, 'z': np.zeros([4, 2, 1])}
    asserts.assert_tree_shape_suffix(tree, (4, 2, 1))
    asserts.assert_tree_shape_suffix(tree, [4, 2, 1])

    with self.assertRaisesRegex(
        AssertionError, _get_err_regex('which is smaller than the expected')):
      asserts.assert_tree_shape_suffix(tree, (3, 4, 2, 1))

    with self.assertRaisesRegex(
        AssertionError, _get_err_regex('which is smaller than the expected')):
      asserts.assert_tree_shape_suffix(tree, [3, 4, 2, 1])

  def test_assert_trees_all_equal_dtypes(self):
    t_0 = {'x': np.zeros(3, dtype=np.int16), 'y': np.ones(2, dtype=np.float32)}
    t_1 = {'x': np.zeros(5, dtype=np.uint16), 'y': np.ones(4, dtype=np.float32)}

    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('Trees 0 and 1 differ')):
      asserts.assert_trees_all_equal_dtypes(t_0, t_1)

    t_2 = {'x': np.zeros(6, dtype=jnp.int16), 'y': np.ones(6, dtype=np.float32)}
    asserts.assert_trees_all_equal_dtypes(t_0, t_2, t_0)

    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('Trees 0 and 4 differ')):
      asserts.assert_trees_all_equal_dtypes(t_0, t_0, t_2, t_0, t_1, t_2)

    # np vs jnp
    t_3 = {'x': np.zeros(1, dtype=np.int16), 'y': np.ones(2, dtype=jnp.float32)}
    t_4 = {
        'x': np.zeros(1, dtype=jnp.int16),
        'y': np.ones(2, dtype=jnp.float32)
    }
    asserts.assert_trees_all_equal_dtypes(t_0, t_2, t_3, t_4)

    # bfloat16
    t_5 = {'y': np.ones(2, dtype=np.float16)}
    t_6 = {'y': np.ones(2, dtype=jnp.bfloat16)}
    asserts.assert_trees_all_equal_dtypes(t_6, t_6)
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('Trees 0 and 1 differ')):
      asserts.assert_trees_all_equal_dtypes(t_5, t_6)

  def test_assert_trees_all_equal_shapes_and_dtypes(self):
    # Test dtypes
    t_0 = {'x': np.zeros(3, dtype=np.int16), 'y': np.ones(2, dtype=np.float32)}
    t_1 = {'x': np.zeros(3, dtype=np.uint16), 'y': np.ones(2, dtype=np.float32)}

    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('Trees 0 and 1 differ')):
      asserts.assert_trees_all_equal_shapes_and_dtypes(t_0, t_1)

    t_2 = {'x': np.zeros(3, dtype=np.int16), 'y': np.ones(2, dtype=np.float32)}
    asserts.assert_trees_all_equal_shapes_and_dtypes(t_0, t_2, t_0)

    # Test shapes
    t_0 = {'x': np.zeros(3, dtype=np.int16), 'y': np.ones(2, dtype=np.float32)}
    t_1 = {'x': np.zeros(4, dtype=np.int16), 'y': np.ones(2, dtype=np.float32)}

    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('Trees 0 and 1 differ')):
      asserts.assert_trees_all_equal_shapes_and_dtypes(t_0, t_1)

    t_2 = {'x': np.zeros(3, dtype=np.int16), 'y': np.ones(2, dtype=np.float32)}
    asserts.assert_trees_all_equal_shapes_and_dtypes(t_0, t_2, t_0)

  def test_assert_trees_all_equal_wrong_usage(self):
    # not an array
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex(r'is not a \(j-\)np array')):
      asserts.assert_trees_all_equal_dtypes({'x': 1.}, {'x': np.array(1.)})

    # 1 tree
    with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
        ValueError, 'Assertions over only one tree does not make sense'):
      asserts.assert_trees_all_equal_dtypes({'x': 1.})

  def test_assert_trees_all_equal_none(self):
    t_0 = {'x': None, 'y': np.array(2, dtype=np.int32)}
    t_1 = {'x': None, 'y': np.array([23], dtype=np.int32)}
    t_2 = {'x': None, 'y': np.array(3, dtype=np.float32)}
    t_3 = {'y': np.array([23], dtype=np.int32)}
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('Trees 0 and 2 differ')):
      asserts.assert_trees_all_equal_dtypes(t_0, t_1, t_2)
    asserts.assert_trees_all_equal_dtypes(t_0, t_1)
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('trees 0 and 1 do not match')):
      asserts.assert_trees_all_equal_dtypes(t_0, t_3)
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('trees 0 and 1 do not match')):
      asserts.assert_trees_all_equal_dtypes(t_0, t_3)


class DevicesAssertTest(parameterized.TestCase):

  def _device_count(self, backend):
    try:
      return jax.device_count(backend)
    except RuntimeError:
      return 0

  @parameterized.parameters('cpu', 'gpu', 'tpu')
  def test_not_less_than(self, devtype):
    n = self._device_count(devtype)
    if n > 0:
      asserts.assert_devices_available(
          n - 1, devtype, backend=devtype, not_less_than=True)
      with self.assertRaisesRegex(AssertionError,
                                  _get_err_regex(f'Only {n} < {n + 1}')):
        asserts.assert_devices_available(
            n + 1, devtype, backend=devtype, not_less_than=True)
    else:
      with self.assertRaisesRegex(RuntimeError,  # pylint: disable=g-error-prone-assert-raises
                                  '(failed to initialize)|(Unknown backend)'):
        asserts.assert_devices_available(
            n - 1, devtype, backend=devtype, not_less_than=True)

  def test_unsupported_device(self):
    with self.assertRaisesRegex(ValueError, 'Unknown device type'):  # pylint: disable=g-error-prone-assert-raises
      asserts.assert_devices_available(1, 'unsupported_devtype')

  def test_gpu_assert(self):
    n_gpu = self._device_count('gpu')
    asserts.assert_devices_available(n_gpu, 'gpu')
    if n_gpu:
      asserts.assert_gpu_available()
    else:
      with self.assertRaisesRegex(AssertionError,
                                  _get_err_regex('No 2 GPUs available')):
        asserts.assert_devices_available(2, 'gpu')
      with self.assertRaisesRegex(AssertionError,
                                  _get_err_regex('No GPU devices available')):
        asserts.assert_gpu_available()

    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('No 2 GPUs available')):
      asserts.assert_devices_available(2, 'gpu', backend='cpu')

  def test_cpu_assert(self):
    n_cpu = jax.device_count('cpu')
    asserts.assert_devices_available(n_cpu, 'cpu', backend='cpu')

  def test_tpu_assert(self):
    n_tpu = self._device_count('tpu')
    asserts.assert_devices_available(n_tpu, 'tpu')
    if n_tpu:
      asserts.assert_tpu_available()
    else:
      with self.assertRaisesRegex(AssertionError,
                                  _get_err_regex('No 3 TPUs available')):
        asserts.assert_devices_available(3, 'tpu')
      with self.assertRaisesRegex(AssertionError,
                                  _get_err_regex('No TPU devices available')):
        asserts.assert_tpu_available()
    with self.assertRaisesRegex(AssertionError,
                                _get_err_regex('No 3 TPUs available')):
      asserts.assert_devices_available(3, 'tpu', backend='cpu')


class NumericalGradsAssertTest(parameterized.TestCase):

  def _test_fn(self, fn, init_args, seed, n=10):
    rng_key = jax.random.PRNGKey(seed)
    for _ in range(n):
      rng_key, *tree_keys = jax.random.split(rng_key, len(init_args) + 1)
      x = jax.tree_util.tree_map(
          lambda k, x: jax.random.uniform(k, shape=x.shape),
          list(tree_keys), list(init_args))
      asserts.assert_numerical_grads(fn, x, order=1)

  @parameterized.parameters(([1], 24), ([5], 6), ([3, 5], 20))
  def test_easy(self, x_shape, seed):
    f_easy = lambda x: jnp.sum(x**2 - 2 * x + 10)
    init_args = (jnp.zeros(x_shape),)
    self._test_fn(f_easy, init_args, seed)

  @parameterized.parameters(([1], 24), ([5], 6), ([3, 5], 20))
  def test_easy_with_stop_gradient(self, x_shape, seed):
    f_easy_sg = lambda x: jnp.sum(jax.lax.stop_gradient(x**2) - 2 * x + 10)
    init_args = (jnp.zeros(x_shape),)
    self._test_fn(f_easy_sg, init_args, seed)

  @parameterized.parameters(([1], 24), ([5], 6), ([3, 5], 20))
  def test_hard(self, x_shape, seed):

    def f_hard_with_sg(lr, x):
      inner_loss = lambda y: jnp.sum((y - 1.0)**2)
      inner_loss_grad = jax.grad(inner_loss)

      def fu(lr, x):
        for _ in range(10):
          x1 = x - lr * inner_loss_grad(x) + 100 * lr**2
          x2 = x - lr * inner_loss_grad(x) - 100 * lr**2
          x = jax.lax.select((x > 3.).any(), x1, x2 + lr)
        return x

      y = fu(lr, x)
      return jnp.sum(inner_loss(y))

    lr = jnp.zeros([1] * len(x_shape))
    x = jnp.zeros(x_shape)

    self._test_fn(f_hard_with_sg, (lr, x), seed)

  @parameterized.parameters(([1], 24), ([5], 6), ([3, 5], 20))
  def test_hard_with_stop_gradient(self, x_shape, seed):

    def f_hard_with_sg(lr, x):
      inner_loss = lambda y: jnp.sum((y - 1.0)**2)
      inner_loss_grad = jax.grad(inner_loss)

      def fu(lr, x):
        for _ in range(10):
          x1 = x - lr * inner_loss_grad(x) + 100 * jax.lax.stop_gradient(lr)**2
          x2 = x - lr * inner_loss_grad(x) - 100 * lr**2
          x = jax.lax.select((x > 3.).any(), x1, x2 + jax.lax.stop_gradient(lr))
        return x

      y = fu(lr, x)
      return jnp.sum(inner_loss(y))

    lr = jnp.zeros([1] * len(x_shape))
    x = jnp.zeros(x_shape)

    self._test_fn(f_hard_with_sg, (lr, x), seed)


class EqualAssertionsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('dtypes', jnp.int32, jnp.int32),
      ('lists', [1, 2], [1, 2]),
      ('dicts', dict(a=[7, jnp.int32]), dict(a=[7, jnp.int32])),
  )
  def test_assert_equal_pass(self, first, second):
    asserts.assert_equal(first, second)

  def test_assert_equal_pass_on_arrays(self):
    # Not using named_parameters, becase JAX cannot be used before app.run().
    asserts.assert_equal(jnp.ones([]), np.ones([]))
    asserts.assert_equal(
        jnp.ones([], dtype=jnp.int32), np.ones([], dtype=np.float64))

  @parameterized.named_parameters(
      ('dtypes', jnp.int32, jnp.float32),
      ('lists', [1, 2], [1, 7]),
      ('lists2', [1, 2], [1]),
      ('dicts1', dict(a=[7, jnp.int32]), dict(b=[7, jnp.int32])),
      ('dicts2', dict(a=[7, jnp.int32]), dict(b=[1, jnp.int32])),
      ('dicts3', dict(a=[7, jnp.int32]), dict(a=[1, jnp.int32], b=2)),
      ('dicts4', dict(a=[7, jnp.int32]), dict(a=[1, jnp.float32])),
      ('arrays', np.zeros([]), np.ones([])),
  )
  def test_assert_equal_fail(self, first, second):
    with self.assertRaises(AssertionError):
      asserts.assert_equal(first, second)


class IsDivisibleTest(parameterized.TestCase):

  def test_assert_is_divisible(self):
    asserts.assert_is_divisible(6, 3)

  def test_assert_is_divisible_fail(self):
    with self.assertRaises(AssertionError):
      asserts.assert_is_divisible(7, 3)


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
