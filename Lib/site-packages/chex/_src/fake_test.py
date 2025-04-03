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
"""Tests for `fake.py`."""

import dataclasses
import functools

from absl.testing import absltest
from absl.testing import parameterized
from chex._src import asserts
from chex._src import fake
from chex._src import pytypes
import jax
import jax.numpy as jnp

ArrayBatched = pytypes.ArrayBatched
ArraySharded = pytypes.ArraySharded


# Set `FLAGS.chex_n_cpu_devices` CPU devices for all tests.
def setUpModule():
  fake.set_n_cpu_devices()


def _assert_jitted(fn, fn_input, is_jitted):
  """Asserts that a function can be jitted or not.

  Args:
    fn: The function to be tested
    fn_input: Input to pass to the function
    is_jitted: Assert that the function can be jitted with jax.jit (True) or
    cannot be jitted (False), i.e. the fake jit is working correctly.
  """
  asserts.clear_trace_counter()
  max_traces = 1 if is_jitted else 0
  wrapped_fn = jax.jit(asserts.assert_max_traces(fn, max_traces))
  wrapped_fn(fn_input)


def _assert_pmapped(fn, fn_input, is_pmapped, should_jit=False):
  """Asserts whether a function can be pmapped or not.

  Args:
    fn: The function to be tested
    fn_input: Input to pass to the function
    is_pmapped: Assert that the function can be pmapped with jax.pmap (True) or
    cannot be pmapped (False), i.e. the fake pmap is working correctly.
    should_jit: if True, asserts that the function is jitted, regardless of it
    being pmapped or not.
  """
  num_devices = len(jax.devices())
  if should_jit:
    asserts.clear_trace_counter()
    fn = asserts.assert_max_traces(fn, n=1)
  wrapped_fn = jax.pmap(fn, axis_size=num_devices)

  fn_input = jnp.broadcast_to(fn_input, (num_devices,) + fn_input.shape)
  output = wrapped_fn(fn_input)

  # We test whether the function has been pmapped by inspecting the type of
  # the function output, if it is a sharded array type then the function has
  # been pmapped
  if is_pmapped:
    expected_type = jax.Array
    assert_message = f'Output is type {type(output)}, expected {expected_type}'
    assert isinstance(output, expected_type), assert_message
  else:
    expected_type = 'DeviceArray'
    assert_message = f'Output is type {type(output)}, expected {expected_type}'
    # ShardedDeviceArray is a subclass of DeviceArray. So, to enforce we have
    # a DeviceArray, we also check it's not a sharded one.
    assert (isinstance(output, jax.Array) and
            len(output.sharding.device_set) == 1), assert_message


class PmapFakeTest(parameterized.TestCase):

  def test_assert_pmapped(self):
    def foo(x):
      return x * 2
    fn_input = jnp.ones((4,))

    _assert_pmapped(foo, fn_input, True)
    # Since this test runs only on 1 device, having a test to check if the
    # output is sharded or not is not correct. With jax.Array, you can check
    # the `len(output.sharding.device_set)` to see if its sharded or not, but
    # here because of a single device it fails.

  def test_assert_jitted(self):
    fn_input = jnp.ones((4,))
    def foo(x):
      return x * 2

    _assert_jitted(foo, fn_input, True)
    with self.assertRaises(AssertionError):
      _assert_jitted(foo, fn_input, False)

  @parameterized.named_parameters([
      ('plain_jit', {'enable_patching': True}, False),
      ('faked_jit', {'enable_patching': False}, True),
  ])
  def test_fake_jit(self, fake_kwargs, is_jitted):
    fn_input = jnp.ones((4,))
    def foo(x):
      return x * 2

    # Call with context manager
    with fake.fake_jit(**fake_kwargs):
      _assert_jitted(foo, fn_input, is_jitted)

    # Call with start/stop
    ctx = fake.fake_jit(**fake_kwargs)
    ctx.start()
    _assert_jitted(foo, fn_input, is_jitted)
    ctx.stop()

  @parameterized.named_parameters([
      ('plain_pmap_but_jit', True, True),
      ('plain_pmap', True, False),
      ('faked_pmap_but_jit', False, True),
      ('faked_pmap', False, False),
  ])
  def test_fake_pmap_(self, is_pmapped, jit_result):
    enable_patching = not is_pmapped

    fn_input = jnp.ones((4,))
    def foo(x):
      return x * 2

    # Call with context manager
    with fake.fake_pmap(enable_patching=enable_patching, jit_result=jit_result):
      _assert_pmapped(foo, fn_input, is_pmapped, jit_result)

    # Call with start/stop
    ctx = fake.fake_pmap(enable_patching=enable_patching, jit_result=jit_result)
    ctx.start()
    _assert_pmapped(foo, fn_input, is_pmapped, jit_result)
    ctx.stop()

  def test_fake_pmap_axis_name(self):

    with fake.fake_pmap():

      @functools.partial(jax.pmap, axis_name='i')
      @functools.partial(jax.pmap, axis_name='j')
      def f(_):
        return jax.lax.axis_index('i'), jax.lax.axis_index('j')
      x, y = f(jnp.zeros((4, 2)))

    self.assertEqual(x.tolist(), [[0, 0], [1, 1], [2, 2], [3, 3]])
    self.assertEqual(y.tolist(), [[0, 1], [0, 1], [0, 1], [0, 1]])

  @parameterized.named_parameters([
      ('fake_nothing', {
          'enable_pmap_patching': False,
          'enable_jit_patching': False
      }, True, True),
      ('fake_pmap', {
          'enable_pmap_patching': True,
          'enable_jit_patching': False
      }, False, True),
      # Default pmap will implicitly compile the function
      ('fake_jit', {
          'enable_pmap_patching': False,
          'enable_jit_patching': True
      }, True, False),
      ('fake_both', {
          'enable_pmap_patching': True,
          'enable_jit_patching': True
      }, False, False),
  ])
  def test_pmap_and_jit(self, fake_kwargs, is_pmapped, is_jitted):
    fn_input = jnp.ones((4,))
    def foo(x):
      return x * 2

    # Call with context manager
    with fake.fake_pmap_and_jit(**fake_kwargs):
      _assert_pmapped(foo, fn_input, is_pmapped)
      _assert_jitted(foo, fn_input, is_jitted)

    # Call with start/stop
    ctx = fake.fake_pmap_and_jit(**fake_kwargs)
    ctx.start()
    _assert_pmapped(foo, fn_input, is_pmapped)
    _assert_jitted(foo, fn_input, is_jitted)
    ctx.stop()

  @parameterized.named_parameters([
      ('fake_nothing', False, False),
      ('fake_pmap', True, False),
      ('fake_jit', False, True),
      ('fake_both', True, True),
  ])
  def test_with_kwargs(self, fake_pmap, fake_jit):
    with fake.fake_pmap_and_jit(fake_pmap, fake_jit):
      num_devices = len(jax.devices())

      @functools.partial(jax.pmap, axis_size=num_devices)
      @jax.jit
      def foo(x, y):
        return (x * 2) + y

      # pmap over all available devices
      inputs = jnp.array([1, 2])
      inputs = jnp.broadcast_to(inputs, (num_devices,) + inputs.shape)
      expected = jnp.broadcast_to(jnp.array([3, 6]), (num_devices, 2))

      asserts.assert_trees_all_close(foo(x=inputs, y=inputs), expected)

  @parameterized.named_parameters([
      ('fake_nothing', False, 1),
      ('fake_pmap', True, 1),
      ('fake_nothing_no_static_args', False, ()),
      ('fake_pmap_no_static_args', True, ()),
  ])
  def test_with_static_broadcasted_argnums(self, fake_pmap, static_argnums):
    with fake.fake_pmap_and_jit(fake_pmap, enable_jit_patching=False):
      num_devices = len(jax.devices())

      # Note: mode='bar' is intended to test that we correctly handle kwargs
      # with defaults for which we don't pass a value at call time.
      @functools.partial(
          jax.pmap,
          axis_size=num_devices,
          static_broadcasted_argnums=static_argnums,
      )
      @functools.partial(
          jax.jit,
          static_argnums=static_argnums,
      )
      def foo(x, multiplier, y, mode='bar'):
        if static_argnums == 1 or 1 in static_argnums:
          # Verify that the static arguments are not replaced with tracers.
          self.assertIsInstance(multiplier, int)

        if mode == 'bar':
          return (x * multiplier) + y
        else:
          return x

      # pmap over all available devices
      inputs = jnp.array([1, 2])
      inputs = jnp.broadcast_to(inputs, (num_devices,) + inputs.shape)
      func = lambda: foo(inputs, 100, inputs)  # Pass multiplier=100.

      if static_argnums == 1:  # Should work.
        expected = jnp.broadcast_to(jnp.array([101, 202]), (num_devices, 2))
        result = func()
        asserts.assert_trees_all_close(result, expected)
      else:  # Should error.
        with self.assertRaises(ValueError):
          result = func()

  @parameterized.parameters(1, [1])
  def test_pmap_with_complex_static_broadcasted_object(self, static_argnums):

    @dataclasses.dataclass
    class Multiplier:
      x: int
      y: int

    def foo(x, multiplier, y):
      if static_argnums == 1 or 1 in static_argnums:
        # Verify that the static arguments are not replaced with tracers.
        self.assertIsInstance(multiplier, Multiplier)

      return x * multiplier.x + y * multiplier.y

    with fake.fake_pmap_and_jit():
      num_devices = jax.device_count()

      # pmap over all available devices
      transformed_foo = jax.pmap(
          foo,
          axis_size=num_devices,
          static_broadcasted_argnums=static_argnums,
      )
      x, y = jax.random.randint(
          jax.random.PRNGKey(27), (2, num_devices, 3, 5), 0, 10
      )

      # Test 1.
      mult = Multiplier(x=2, y=7)
      asserts.assert_trees_all_equal(
          transformed_foo(x, mult, y),
          foo(x, mult, y),
          x * mult.x + y * mult.y,
      )

      # Test 2.
      mult = Multiplier(x=72, y=21)
      asserts.assert_trees_all_equal(
          transformed_foo(x, mult, y),
          foo(x, mult, y),
          x * mult.x + y * mult.y,
      )

  @parameterized.named_parameters([
      ('fake_nothing', False, False),
      ('fake_pmap', True, False),
      ('fake_jit', False, True),
      ('fake_both', True, True),
  ])
  def test_with_partial(self, fake_pmap, fake_jit):
    with fake.fake_pmap_and_jit(fake_pmap, fake_jit):
      num_devices = len(jax.devices())

      # Testing a common use-case where non-parallel arguments are partially
      # applied before pmapping
      def foo(x, y, flag):
        return (x * 2) + y if flag else (x + y)
      foo = functools.partial(foo, flag=True)

      foo = jax.pmap(foo, axis_size=num_devices)
      foo = jax.jit(foo)

      # pmap over all available devices
      inputs = jnp.array([1, 2])
      inputs = jnp.broadcast_to(inputs, (num_devices,) + inputs.shape)
      expected = jnp.broadcast_to(jnp.array([3, 6]), (num_devices, 2))

      asserts.assert_trees_all_close(foo(inputs, inputs), expected)
      asserts.assert_trees_all_close(foo(x=inputs, y=inputs), expected)

  @parameterized.named_parameters([
      ('fake_nothing', False, False),
      ('fake_pmap', True, False),
      ('fake_jit', False, True),
      ('fake_both', True, True),
  ])
  def test_with_default_params(self, fake_pmap, fake_jit):
    with fake.fake_pmap_and_jit(fake_pmap, fake_jit):
      num_devices = len(jax.devices())

      # Default flag specified at definition time
      def foo(x, y, flag=True):
        return (x * 2) + y if flag else (x + y)

      default_foo = jax.pmap(foo, axis_size=num_devices)
      default_foo = jax.jit(default_foo)

      inputs = jnp.array([1, 2])
      inputs = jnp.broadcast_to(inputs, (num_devices,) + inputs.shape)
      expected = jnp.broadcast_to(jnp.array([3, 6]), (num_devices, 2))
      asserts.assert_trees_all_close(default_foo(inputs, inputs), expected)
      asserts.assert_trees_all_close(default_foo(x=inputs, y=inputs), expected)

      # Default overriden by partial to execute other branch
      overidden_foo = functools.partial(foo, flag=False)
      overidden_foo = jax.pmap(overidden_foo, axis_size=num_devices)
      overidden_foo = jax.jit(overidden_foo)

      expected = jnp.broadcast_to(jnp.array([2, 4]), (num_devices, 2))
      asserts.assert_trees_all_close(overidden_foo(inputs, inputs), expected)
      asserts.assert_trees_all_close(
          overidden_foo(x=inputs, y=inputs), expected)

  def test_parallel_ops_equivalence(self):
    """Test equivalence between parallel operations using pmap and vmap."""
    num_devices = len(jax.devices())
    inputs = jax.random.uniform(shape=(num_devices, num_devices, 2),
                                key=jax.random.PRNGKey(1))

    def test_equivalence(fn):
      with fake.fake_pmap(enable_patching=False):
        outputs1 = jax.pmap(fn, axis_name='i', axis_size=num_devices)(inputs)
      with fake.fake_pmap(enable_patching=True):
        outputs2 = jax.pmap(fn, axis_name='i', axis_size=num_devices)(inputs)
      with fake.fake_pmap(enable_patching=True, jit_result=True):
        outputs3 = jax.pmap(fn, axis_name='i', axis_size=num_devices)(inputs)
      asserts.assert_trees_all_close(outputs1, outputs2, outputs3)

    parallel_ops_and_kwargs = [
        (jax.lax.psum, {}),
        (jax.lax.pmax, {}),
        (jax.lax.pmin, {}),
        (jax.lax.pmean, {}),
        (jax.lax.all_gather, {}),
        (jax.lax.all_to_all, {
            'split_axis': 0,
            'concat_axis': 1
        }),
        (jax.lax.ppermute, {
            'perm': [(x, (x + 1) % num_devices) for x in range(num_devices)]
        }),
    ]

    def fn(op, kwargs, x, y=2.0):
      return op(x * y, axis_name='i', **kwargs)
    partial_fn = functools.partial(fn, y=4.0)
    lambda_fn = lambda op, kwargs, x: fn(op, kwargs, x, y=5.0)

    for op, kwargs in parallel_ops_and_kwargs:
      test_equivalence(functools.partial(fn, op, kwargs))
      test_equivalence(functools.partial(fn, op, kwargs, y=3.0))
      test_equivalence(functools.partial(partial_fn, op, kwargs))
      test_equivalence(functools.partial(lambda_fn, op, kwargs))

  def test_fake_parallel_axis(self):
    inputs = jnp.ones(shape=(2, 2))
    with fake.fake_pmap(fake_parallel_axis=False):
      @jax.pmap
      def no_fake_parallel_axis_fn(x):
        asserts.assert_shape(x, (2,))
        return 2.0 * x

      outputs = no_fake_parallel_axis_fn(inputs)
      asserts.assert_trees_all_close(outputs, 2.0)

    with fake.fake_pmap(fake_parallel_axis=True):
      @jax.pmap
      def fake_parallel_axis_fn(x):
        asserts.assert_shape(x, (2, 2,))
        return 2.0 * x

      outputs = fake_parallel_axis_fn(inputs)
      asserts.assert_trees_all_close(outputs, 2.0)


class _Counter():
  """Counts how often an instance is called."""

  def __init__(self):
    self.count = 0

  def __call__(self, *unused_args, **unused_kwargs):
    self.count += 1


class OnCallOfTransformedFunctionTest(parameterized.TestCase):

  def test_on_call_of_transformed_function(self):
    counter = _Counter()
    with fake.OnCallOfTransformedFunction('jax.jit', counter):
      jax.jit(jnp.sum)(jnp.zeros((10,)))
      jax.jit(jnp.max)(jnp.zeros((10,)))
    self.assertEqual(counter.count, 2)


if __name__ == '__main__':
  absltest.main()
