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
"""Chex variants utilities."""

import enum
import functools
import inspect
import itertools
from typing import Any, Sequence
import unittest

from absl import flags
from absl.testing import parameterized
from chex._src import fake
from chex._src import pytypes
import jax
from jax import tree_util
import jax.numpy as jnp
import toolz

FLAGS = flags.FLAGS
flags.DEFINE_bool(
    "chex_skip_pmap_variant_if_single_device", True,
    "Whether to skip pmap variant if only one device is available.")


# We choose to subclass instead of a simple alias, as Python doesn't allow
# multiple inheritance from the same class, and users may want to subclass their
# tests from both `chex.TestCase` and `parameterized.TestCase`.
#
# User is free to use any base class that supports generators unrolling
# instead of `variants.TestCase` or `parameterized.TestCase`. If a base class
# doesn't support this feature variant test fails with a corresponding error.
class TestCase(parameterized.TestCase):
  """A class for Chex tests that use variants.

  See the docstring for ``chex.variants`` for more information.

  Note: ``chex.variants`` returns a generator producing one test per variant.
  Therefore, the used test class must support dynamic unrolling of these
  generators during module import. It is implemented (and battle-tested) in
  ``absl.parameterized.TestCase``, and here we subclass from it.
  """

  def variant(self, *args, **kwargs):
    """Raises a RuntimeError if not overriden or redefined."""
    raise RuntimeError(
        "self.variant is not defined: forgot to wrap a test in @chex.variants?")


class ChexVariantType(enum.Enum):
  """An enumeration of available Chex variants.

  Use ``self.variant.type`` to get type of the current test variant.
  See the docstring of ``chex.variants`` for more information.
  """

  WITH_JIT = 1
  WITHOUT_JIT = 2
  WITH_DEVICE = 3
  WITHOUT_DEVICE = 4
  WITH_PMAP = 5

  def __str__(self) -> str:
    return "_" + self.name.lower()


tree_map = tree_util.tree_map


def params_product(*params_lists: Sequence[Sequence[Any]],
                   named: bool = False) -> Sequence[Sequence[Any]]:
  """Generates a cartesian product of `params_lists`.

  See tests from ``variants_test.py`` for examples of usage.

  Args:
    *params_lists: A list of params combinations.
    named: Whether to generate test names (for
      `absl.parameterized.named_parameters(...)`).

  Returns:
    A cartesian product of `params_lists` combinations.
  """

  def generate():
    for combination in itertools.product(*params_lists):
      if named:
        name = "_".join(t[0] for t in combination)
        args_tuples = (t[1:] for t in combination)
        args = sum(args_tuples, ())
        yield (name, *args)
      else:
        yield sum(combination, ())

  return list(generate())


def count_num_calls(fn):
  """Counts the number of times the function was called."""
  num_calls = 0

  @functools.wraps(fn)
  def fn_wrapped(*args, **kwargs):
    nonlocal num_calls
    num_calls += 1
    return fn(*args, **kwargs)

  return fn_wrapped, lambda: num_calls


class VariantsTestCaseGenerator:
  """TestCase generator for chex variants. Supports sharding."""

  def __init__(self, test_object, which_variants):
    self._which_variants = which_variants
    self._generated_names_freq = {}
    if hasattr(test_object, "__iter__"):
      # `test_object` is a generator (e.g. parameterised test).
      self._test_methods = list(test_object)
    else:
      # `test_object` is a single test method.
      self._test_methods = [test_object]

  def add_variants(self, which_variants):
    """Merge variants."""
    for var, incl in which_variants.items():
      self._which_variants[var] = self._which_variants.get(var, False) or incl

  @property
  def __name__(self):
    msg = ("A test wrapper attempts to access __name__ of "
           "VariantsTestCaseGenerator. Usually, this happens when "
           "@parameterized wraps @variants.variants. Make sure that the "
           "@variants.variants wrapper is an outer one, i.e. nothing wraps it.")
    raise RuntimeError(msg)

  def __call__(self):
    msg = ("A test wrapper attempts to invoke __call__ of "
           "VariantsTestCaseGenerator: make sure that all `TestCase` instances "
           "that use variants inherit from `chex.TestCase`.")
    raise RuntimeError(msg)

  def _set_test_name(self, test_method, variant):
    """Set a name for the generated test."""
    name = getattr(test_method, "__name__", "")
    params_repr = getattr(test_method, "__x_params_repr__", "")
    chex_suffix = f"{variant}"

    candidate_name = "_".join(filter(None, [name, params_repr, chex_suffix]))
    name_freq = self._generated_names_freq.get(candidate_name, 0)
    if name_freq:
      # Ensure that test names are unique.
      new_name = name + "_" + str(name_freq)
      unique_name = "_".join(filter(None, [new_name, params_repr, chex_suffix]))
    else:
      unique_name = candidate_name
    self._generated_names_freq[candidate_name] = name_freq + 1

    # Always use name for compatibility with `absl.testing.parameterized`.
    setattr(test_method, "__name__", unique_name)
    setattr(test_method, "__x_params_repr__", "")
    setattr(test_method, "__x_use_name__", True)
    return test_method

  def _inner_iter(self, test_method):
    """Generate chex variants for a single test."""

    def make_test(variant: ChexVariantType):

      @functools.wraps(test_method)
      def test(self, *args, **kwargs):
        # Skip pmap variant if only one device is available.

        if (variant is ChexVariantType.WITH_PMAP and
            FLAGS["chex_skip_pmap_variant_if_single_device"].value and
            jax.device_count() < 2):
          raise unittest.SkipTest(
              f"Only 1 device is available ({jax.devices()}).")

        # n_cpu_devices assert.
        if FLAGS["chex_assert_multiple_cpu_devices"].value:
          required_n_cpus = fake.get_n_cpu_devices_from_xla_flags()
          if required_n_cpus < 2:
            raise RuntimeError(
                f"Required number of CPU devices is {required_n_cpus} < 2."
                "Consider setting up your test module to use multiple CPU "
                " devices (see README.md) or disabling "
                "`chex_assert_multiple_cpu_devices` flag.")
          available_n_cpus = jax.device_count("cpu")
          if required_n_cpus != available_n_cpus:
            raise RuntimeError(
                "Number of available CPU devices is not equal to the required: "
                f"{available_n_cpus} != {required_n_cpus}")

        # Set up the variant.
        self.variant, num_calls = count_num_calls(_variant_decorators[variant])
        self.variant.type = variant
        res = test_method(self, *args, **kwargs)
        if num_calls() == 0:
          raise RuntimeError(
              "Test is wrapped in @chex.variants, but never calls self.variant."
              " Consider debugging the test or removing @chex.variants wrapper."
              f" (variant: {variant})")
        return res

      self._set_test_name(test, variant)
      return test

    selected_variants = [
        var_name for var_name, is_included in self._which_variants.items()
        if is_included
    ]
    if not selected_variants:
      raise ValueError(f"No variants selected for test: {test_method}.")

    return (make_test(var_name) for var_name in selected_variants)

  def __iter__(self):
    """Generate chex variants for each test case."""
    return itertools.chain(*(self._inner_iter(m) for m in self._test_methods))


@toolz.curry
def _variants_fn(test_object, **which_variants) -> VariantsTestCaseGenerator:
  """Implements `variants` and `all_variants`."""

  # Convert keys to enum entries.
  which_variants = {
      ChexVariantType[name.upper()]: var
      for name, var in which_variants.items()
  }
  if isinstance(test_object, VariantsTestCaseGenerator):
    # Merge variants for nested wrappers.
    test_object.add_variants(which_variants)
  else:
    test_object = VariantsTestCaseGenerator(test_object, which_variants)

  return test_object


@toolz.curry
# pylint: disable=redefined-outer-name
def variants(test_method,
             with_jit: bool = False,
             without_jit: bool = False,
             with_device: bool = False,
             without_device: bool = False,
             with_pmap: bool = False) -> VariantsTestCaseGenerator:
  # pylint: enable=redefined-outer-name
  """Decorates a test to expose Chex variants.

  The decorated test has access to a decorator called ``self.variant``, which
  may be applied to functions to test different JAX behaviors. Consider:

  .. code-block:: python

    @chex.variants(with_jit=True, without_jit=True)
    def test(self):
      @self.variant
      def f(x, y):
        return x + y

      self.assertEqual(f(1, 2), 3)

  In this example, the function ``test`` will be called twice: once with `f`
  jitted (i.e. using `jax.jit`) and another where `f` is not jitted.

  Variants `with_jit=True` and `with_pmap=True` accept additional specific to
  them arguments. Example:

  .. code-block:: python

    @chex.variants(with_jit=True)
    def test(self):
      @self.variant(static_argnums=(1,))
      def f(x, y):
        # `y` is not traced.
        return x + y

      self.assertEqual(f(1, 2), 3)

  Variant `with_pmap=True` also accepts `broadcast_args_to_devices`
  (whether to broadcast each input argument to all participating devices),
  `reduce_fn` (a function to apply to results of pmapped `fn`), and
  `n_devices` (number of devices to use in the `pmap` computation).
  See the docstring of `_with_pmap` for more details (including default values).

  If used with ``absl.testing.parameterized``, `@chex.variants` must wrap it:

  .. code-block:: python

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters('test', *args)
    def test(self, *args):
      ...

  Tests that use this wrapper must be inherited from ``parameterized.TestCase``.
  For more examples see ``variants_test.py``.

  Args:
    test_method: A test method to decorate.
    with_jit: Whether to test with `jax.jit`.
    without_jit: Whether to test without `jax.jit`. Any jit compilation done
      within the test method will not be affected.
    with_device: Whether to test with args placed on device, using
      `jax.device_put`.
    without_device: Whether to test with args (explicitly) not placed on device,
      using `jax.device_get`.
    with_pmap: Whether to test with `jax.pmap`, with computation duplicated
      across devices.

  Returns:
    A decorated ``test_method``.
  """
  return _variants_fn(
      test_method,
      with_jit=with_jit,
      without_jit=without_jit,
      with_device=with_device,
      without_device=without_device,
      with_pmap=with_pmap)


@toolz.curry
# pylint: disable=redefined-outer-name
def all_variants(test_method,
                 with_jit: bool = True,
                 without_jit: bool = True,
                 with_device: bool = True,
                 without_device: bool = True,
                 with_pmap: bool = True) -> VariantsTestCaseGenerator:
  # pylint: enable=redefined-outer-name
  """Equivalent to ``chex.variants`` but with flipped defaults."""
  return _variants_fn(
      test_method,
      with_jit=with_jit,
      without_jit=without_jit,
      with_device=with_device,
      without_device=without_device,
      with_pmap=with_pmap)


def check_variant_arguments(variant_fn):
  """Raises `ValueError` if `variant_fn` got an unknown argument."""

  @functools.wraps(variant_fn)
  def wrapper(*args, **kwargs):
    unknown_args = set(kwargs.keys()) - _valid_kwargs_keys
    if unknown_args:
      raise ValueError(f"Unknown arguments in `self.variant`: {unknown_args}.")
    return variant_fn(*args, **kwargs)

  return wrapper


@toolz.curry
@check_variant_arguments
def _with_jit(fn,
              static_argnums=None,
              static_argnames=None,
              device=None,
              backend=None,
              **unused_kwargs):
  """Variant that applies `jax.jit` to fn."""

  return jax.jit(
      fn,
      static_argnums=static_argnums,
      static_argnames=static_argnames,
      device=device,
      backend=backend)


@toolz.curry
@check_variant_arguments
def _without_jit(fn, **unused_kwargs):
  """Variant that does not apply `jax.jit` to a fn (identity)."""

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    return fn(*args, **kwargs)

  return wrapper


@toolz.curry
@check_variant_arguments
def _with_device(fn, ignore_argnums=(), static_argnums=(), **unused_kwargs):
  """Variant that applies `jax.device_put` to the args of fn."""

  if isinstance(ignore_argnums, int):
    ignore_argnums = (ignore_argnums,)
  if isinstance(static_argnums, int):
    static_argnums = (static_argnums,)

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):

    def put(x):
      try:
        return jax.device_put(x)
      except TypeError:  # not a valid JAX type
        return x

    device_args = [
        arg if (idx in ignore_argnums or idx in static_argnums) else tree_map(
            put, arg) for idx, arg in enumerate(args)
    ]
    device_kwargs = tree_map(put, kwargs)
    return fn(*device_args, **device_kwargs)

  return wrapper


@toolz.curry
@check_variant_arguments
def _without_device(fn, **unused_kwargs):
  """Variant that applies `jax.device_get` to the args of fn."""

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):

    def get(x):
      if isinstance(x, jax.Array):
        return jax.device_get(x)
      return x

    no_device_args = tree_map(get, args)
    no_device_kwargs = tree_map(get, kwargs)
    return fn(*no_device_args, **no_device_kwargs)

  return wrapper


@toolz.curry
@check_variant_arguments
def _with_pmap(fn,
               broadcast_args_to_devices=True,
               reduce_fn="first_device_output",
               n_devices=None,
               axis_name="i",
               devices=None,
               in_axes=0,
               static_broadcasted_argnums=(),
               static_argnums=(),
               backend=None,
               **unused_kwargs):
  """Variant that applies `jax.pmap` to fn.

  Args:
    fn: A function to wrap.
    broadcast_args_to_devices: Whether to broadcast `fn` args to pmap format
      (i.e. pmapped axes' sizes == a number of devices).
    reduce_fn: A function to apply to outputs of `fn`.
    n_devices: A number of devices to use (can specify a `backend` if required).
    axis_name: An argument for `pmap`.
    devices: An argument for `pmap`.
    in_axes: An argument for `pmap`.
    static_broadcasted_argnums: An argument for `pmap`.
    static_argnums: An alias of ``static_broadcasted_argnums``.
    backend: An argument for `pmap`.
    **unused_kwargs: Unused kwargs (e.g. related to other variants).

  Returns:
    Wrapped `fn` that accepts `args` and `kwargs` and returns a superposition of
    `reduce_fn` and `fn` applied to them.

  Raises:
    ValueError: If `broadcast_args_to_devices` used with `in_axes` or
      `static_broadcasted_argnums`; if number of available devices is less than
      required; if pmappable arg axes' sizes are not equal to the number of
      devices.
    SkipTest: If the flag ``chex_skip_pmap_variant_if_single_device`` is set and
      there is only one device available.
  """
  if (FLAGS["chex_skip_pmap_variant_if_single_device"].value and
      jax.device_count() < 2):
    raise unittest.SkipTest(f"Only 1 device is available ({jax.devices()}).")

  if broadcast_args_to_devices and in_axes != 0:
    raise ValueError(
        "Do not use `broadcast_args_to_devices` when specifying `in_axes`.")

  # Set up a reduce function.
  if reduce_fn == "first_device_output":
    reduce_fn = lambda t: tree_map(lambda x: x[0], t)
  elif reduce_fn == "identity" or reduce_fn is None:  # Identity.
    reduce_fn = lambda t: t

  if not static_argnums and static_argnums != 0:
    static_argnums = static_broadcasted_argnums
  if isinstance(static_argnums, int):
    static_argnums = (static_argnums,)

  pmap_kwargs = dict(
      axis_name=axis_name,
      devices=devices,
      in_axes=in_axes,
      static_broadcasted_argnums=static_argnums,
      backend=backend)
  pmapped_fn = jax.pmap(fn, **pmap_kwargs)

  @functools.wraps(pmapped_fn)
  def wrapper(*args: pytypes.ArrayTree, **kwargs: pytypes.ArrayTree):
    if kwargs and (in_axes != 0 or static_argnums):
      raise ValueError("Do not use kwargs with `in_axes` or `static_argnums` "
                       "in pmapped function.")
    devices_ = list(devices or jax.devices(backend))
    n_devices_ = n_devices or len(devices_)
    devices_ = devices_[:n_devices_]
    if len(devices_) != n_devices_:
      raise ValueError("Number of available devices is less than required for "
                       f"test ({len(devices_)} < {n_devices_})")

    bcast_fn = lambda x: jnp.broadcast_to(x, (n_devices_,) + jnp.array(x).shape)
    if broadcast_args_to_devices:
      args = [
          tree_map(bcast_fn, arg) if idx not in static_argnums else arg
          for idx, arg in enumerate(args)
      ]
      kwargs = tree_map(bcast_fn, kwargs)
    else:
      # Pmappable axes size must be equal to number of devices.
      in_axes_ = in_axes if isinstance(in_axes,
                                       (tuple, list)) else [in_axes] * len(args)
      is_pmappable_arg = [
          idx not in static_argnums and in_axes_[idx] is not None
          for idx in range(len(args))
      ]
      for is_pmappable_arg, arg in zip(is_pmappable_arg, args):
        if not is_pmappable_arg:
          continue
        if not all(
            x.shape[0] == n_devices_ for x in jax.tree_util.tree_leaves(arg)):
          shapes = tree_map(jnp.shape, arg)
          raise ValueError(
              f"Pmappable arg axes size must be equal to number of devices, "
              f"got: {shapes} (expected the first dim to be {n_devices_}). "
              "Consider setting `broadcast_args_to_devices=True`.")

    new_kwargs = dict(
        axis_name=axis_name,
        devices=devices_,
        in_axes=in_axes,
        static_broadcasted_argnums=static_argnums,
        backend=backend)

    # Re-compile fn if kwargs changed.
    nonlocal pmap_kwargs
    nonlocal pmapped_fn
    if new_kwargs != pmap_kwargs:
      pmap_kwargs = new_kwargs
      pmapped_fn = jax.pmap(fn, **pmap_kwargs)

    res = pmapped_fn(*args, **kwargs)
    return reduce_fn(res)

  return wrapper


_variant_decorators = dict({
    ChexVariantType.WITH_JIT: _with_jit,
    ChexVariantType.WITHOUT_JIT: _without_jit,
    ChexVariantType.WITH_DEVICE: _with_device,
    ChexVariantType.WITHOUT_DEVICE: _without_device,
    ChexVariantType.WITH_PMAP: _with_pmap,
})


class Variant:
  """Variant class for typing and string representation."""

  def __init__(self, name, fn):
    self._fn = fn
    self._name = name

  def __repr__(self):
    return self._name

  def __call__(self, *args, **kwargs):
    # Could apply decorators (currying, arg-checking) here
    return self._fn(*args, **kwargs)


# Expose variant objects.
without_device = Variant("chex_without_device", _without_device)
without_jit = Variant("chex_without_jit", _without_jit)
with_device = Variant("chex_with_device", _with_device)
with_jit = Variant("chex_with_jit", _with_jit)
with_pmap = Variant("chex_with_pmap", _with_pmap)
ALL_VARIANTS = (without_device, without_jit, with_device, with_jit, with_pmap)

# Collect valid argument names from all variant decorators.
_valid_kwargs_keys = set()
for fn_ in _variant_decorators.values():
  original_fn = fn_.func.__wrapped__
  _valid_kwargs_keys.update(inspect.getfullargspec(original_fn).args)
