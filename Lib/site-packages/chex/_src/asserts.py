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
"""Chex assertion utilities."""

import collections
import collections.abc
import functools
import inspect
import traceback
from typing import Any, Callable, List, Optional, Sequence, Set, Union, cast
import unittest
from unittest import mock

from chex._src import asserts_internal as _ai
from chex._src import pytypes
import jax
from jax.experimental import checkify
import jax.numpy as jnp
import jax.test_util as jax_test
import numpy as np

Scalar = pytypes.Scalar
Array = pytypes.Array
ArrayDType = pytypes.ArrayDType  # pylint:disable=invalid-name
ArrayTree = pytypes.ArrayTree


_value_assertion = _ai.chex_assertion
_static_assertion = functools.partial(
    _ai.chex_assertion, jittable_assert_fn=None)


def disable_asserts() -> None:
  """Disables all Chex assertions.

  Use wisely.
  """
  _ai.DISABLE_ASSERTIONS = True


def enable_asserts() -> None:
  """Enables Chex assertions."""
  _ai.DISABLE_ASSERTIONS = False


def if_args_not_none(fn, *args, **kwargs):
  """Wrap chex assertion to only be evaluated if positional args not `None`."""
  found_none = False
  for x in args:
    found_none = found_none or (x is None)
  if not found_none:
    fn(*args, **kwargs)


def clear_trace_counter() -> None:
  """Clears Chex traces' counter for ``assert_max_traces`` checks.

  Use it to isolate unit tests that rely on ``assert_max_traces``,
  by calling it at the start of the test case.
  """
  _ai.TRACE_COUNTER.clear()


def assert_max_traces(fn: Optional[Union[Callable[..., Any], int]] = None,
                      n: Optional[Union[Callable[..., Any], int]] = None):
  """Checks that a function is traced at most `n` times (inclusively).

  JAX re-traces jitted functions every time the structure of passed arguments
  changes. Often this behaviour is inadvertent and leads to a significant
  performance drop which is hard to debug. This wrapper checks that
  the function is re-traced at most `n` times during program execution.

  Examples:

  .. code-block:: python

    @jax.jit
    @chex.assert_max_traces(n=1)
    def fn_sum_jitted(x, y):
      return x + y

    def fn_sub(x, y):
      return x - y

    fn_sub_pmapped = jax.pmap(chex.assert_max_retraces(fn_sub), n=10)

  More about tracing:
    https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html

  Args:
    fn: A pure python function to wrap (i.e. it must not be a jitted function).
    n: The maximum allowed number of retraces (non-negative).

  Returns:
    Decorated function that raises exception when it is re-traced `n+1`-st time.

  Raises:
    ValueError: If ``fn`` has already been jitted.
  """
  if not callable(fn) and n is None:
    # Passed n as a first argument.
    n, fn = fn, n

  # Currying.
  if fn is None:
    return lambda fn_: assert_max_traces(fn_, n)

  # Args are expected to be in the right order from here onwards.
  fn = cast(Callable[..., Any], fn)
  n = cast(int, n)
  assert_scalar_non_negative(n)

  # Check wrappers ordering.
  if _ai.is_traceable(fn):
    raise ValueError(
        "@assert_max_traces must not wrap JAX-transformed function "
        "(@jit, @vmap, @pmap etc.); change wrappers ordering.")

  # Footprint is defined as a stacktrace of modules' names at the function's
  # definition place + its name and source code. This allows to catch retracing
  # event both in loops and in sequential calls, and makes this wrapper
  # with Colab envs.
  fn_footprint = (
      tuple(frame.name for frame in traceback.extract_stack()[:-1]) +
      (inspect.getsource(fn), fn.__name__))
  fn_hash = hash(fn_footprint)

  @functools.wraps(fn)
  def fn_wrapped(*args, **kwargs):
    # We assume that a function without arguments is not being traced.
    # That is, case of n=0 for no-arguments function won't raise a error.
    has_tracers_in_args = _ai.has_tracers((args, kwargs))

    nonlocal fn_hash
    _ai.TRACE_COUNTER[fn_hash] += int(has_tracers_in_args)
    if not _ai.DISABLE_ASSERTIONS and _ai.TRACE_COUNTER[fn_hash] > n:
      raise AssertionError(
          f"{_ai.ERR_PREFIX}Function '{fn.__name__}' is traced > {n} times!\n"
          "It often happens when a jitted function is defined inside another "
          "function that is called multiple times (i.e. the jitted f-n is a "
          "new object every time). Make sure that your code does not exploit "
          "this pattern (move the nested functions to the top level to fix it)."
          " See `chex.clear_trace_counter()` if `@chex.assert_max_traces` is "
          "used in any unit tests (especially @parameterized tests).")

    return fn(*args, **kwargs)

  return fn_wrapped


@_static_assertion
def assert_devices_available(n: int,
                             devtype: str,
                             backend: Optional[str] = None,
                             not_less_than: bool = False) -> None:
  """Checks that `n` devices of a given type are available.

  Args:
    n: A required number of devices of the given type.
    devtype: A type of devices, one of ``{'cpu', 'gpu', 'tpu'}``.
    backend: A type of backend to use (uses Jax default if not provided).
    not_less_than: Whether to check if the number of devices is not less than
      `n`, instead of precise comparison.

  Raises:
    AssertionError: If number of available device of a given type is not equal
                    or less than `n`.
  """
  n_available = _ai.num_devices_available(devtype, backend=backend)
  devs = jax.devices(backend)
  if not_less_than and n_available < n:
    raise AssertionError(
        f"Only {n_available} < {n} {devtype.upper()}s available in {devs}.")
  elif not not_less_than and n_available != n:
    raise AssertionError(f"No {n} {devtype.upper()}s available in {devs}.")


@_static_assertion
def assert_tpu_available(backend: Optional[str] = None) -> None:
  """Checks that at least one TPU device is available.

  Args:
    backend: A type of backend to use (uses JAX default if not provided).

  Raises:
    AssertionError: If no TPU device available.
  """
  if not _ai.num_devices_available("tpu", backend=backend):
    raise AssertionError(f"No TPU devices available in {jax.devices(backend)}.")


@_static_assertion
def assert_gpu_available(backend: Optional[str] = None) -> None:
  """Checks that at least one GPU device is available.

  Args:
    backend: A type of backend to use (uses JAX default if not provided).

  Raises:
    AssertionError: If no GPU device available.
  """
  if not _ai.num_devices_available("gpu", backend=backend):
    raise AssertionError(f"No GPU devices available in {jax.devices(backend)}.")


@_static_assertion
def assert_equal(first: Any, second: Any) -> None:
  """Checks that the two objects are equal as determined by the `==` operator.

  Arrays with more than one element cannot be compared.
  Use ``assert_trees_all_close`` to compare arrays.

  Args:
    first: A first object.
    second: A second object.

  Raises:
    AssertionError: If not ``(first == second)``.
  """
  unittest.TestCase().assertEqual(first, second)


@_static_assertion
def assert_not_both_none(first: Any, second: Any) -> None:
  """Checks that at least one of the arguments is not `None`.

  Args:
    first: A first object.
    second: A second object.

  Raises:
    AssertionError: If ``(first is None) and (second is None)``.
  """
  if first is None and second is None:
    raise AssertionError(
        "At least one of the arguments must be different from `None`.")


@_static_assertion
def assert_exactly_one_is_none(first: Any, second: Any) -> None:
  """Checks that one and only one of the arguments is `None`.

  Args:
    first: A first object.
    second: A second object.

  Raises:
    AssertionError: If ``(first is None) xor (second is None)`` is `False`.
  """
  if (first is None) == (second is None):
    raise AssertionError(f"One and exactly one of inputs should be `None`, "
                         f"got {first} and {second}.")


@_static_assertion
def assert_is_divisible(numerator: int, denominator: int) -> None:
  """Checks that ``numerator`` is divisible by ``denominator``.

  Args:
    numerator: A numerator.
    denominator: A denominator.

  Raises:
    AssertionError: If ``numerator`` is not divisible by ``denominator``.
  """
  if numerator % denominator != 0:
    raise AssertionError(f"{numerator} is not divisible by {denominator}.")


@_static_assertion
def assert_scalar(x: Scalar) -> None:
  """Checks that ``x`` is a scalar, as defined in `pytypes.py` (int or float).

  Args:
    x: An object to check.

  Raises:
    AssertionError: If ``x`` is not a scalar as per definition in pytypes.py.
  """
  if not isinstance(x, (int, float)):
    raise AssertionError(f"The argument {x} must be a scalar, got {type(x)}.")


@_static_assertion
def assert_scalar_in(x: Any,
                     min_: Scalar,
                     max_: Scalar,
                     included: bool = True) -> None:
  """Checks that argument is a scalar within segment (by default).

  Args:
    x: An object to check.
    min_: A left border of the segment.
    max_: A right border of the segment.
    included: Whether to include the borders of the segment in the set of
      allowed values.

  Raises:
    AssertionError: If ``x`` is not a scalar; if ``x`` falls out of the segment.
  """
  assert_scalar(x)
  if included:
    if not min_ <= x <= max_:
      raise AssertionError(
          f"The argument must be in [{min_}, {max_}], got {x}.")
  else:
    if not min_ < x < max_:
      raise AssertionError(
          f"The argument must be in ({min_}, {max_}), got {x}.")


@_static_assertion
def assert_scalar_positive(x: Scalar) -> None:
  """Checks that a scalar is positive.

  Args:
    x: A value to check.

  Raises:
    AssertionError: If ``x`` is not a scalar or strictly positive.
  """
  assert_scalar(x)
  if x <= 0:
    raise AssertionError(f"The argument must be positive, got {x}.")


@_static_assertion
def assert_scalar_non_negative(x: Scalar) -> None:
  """Checks that a scalar is non-negative.

  Args:
    x: A value to check.

  Raises:
    AssertionError: If ``x`` is not a scalar or negative.
  """
  assert_scalar(x)
  if x < 0:
    raise AssertionError(f"The argument must be non-negative, was {x}.")


@_static_assertion
def assert_scalar_negative(x: Scalar) -> None:
  """Checks that a scalar is negative.

  Args:
    x: A value to check.

  Raises:
    AssertionError: If ``x`` is not a scalar or strictly negative.
  """
  assert_scalar(x)
  if x >= 0:
    raise AssertionError(f"The argument must be negative, was {x}.")


@_static_assertion
def assert_equal_size(inputs: Sequence[Array]) -> None:
  """Checks that all arrays have the same size.

  Args:
    inputs: A collection of arrays.

  Raises:
    AssertionError: If the size of all arrays do not match.
  """
  _ai.assert_collection_of_arrays(inputs)
  size = inputs[0].size
  expected_sizes = [size] * len(inputs)
  sizes = [x.size for x in inputs]
  if sizes != expected_sizes:
    raise AssertionError(f"Arrays have different sizes: {sizes}")


@_static_assertion
def assert_size(
    inputs: Union[Scalar, Union[Array, Sequence[Array]]],
    expected_sizes: Union[_ai.TShapeMatcher,
                          Sequence[_ai.TShapeMatcher]]) -> None:
  """Checks that the size of all inputs matches specified ``expected_sizes``.
  
  Valid usages include:

  .. code-block:: python

    assert_size(x, 1)                   # x is scalar (size 1)
    assert_size([x, y], (2, {1, 3}))    # x has size 2, y has size 1 or 3
    assert_size([x, y], (2, ...))       # x has size 2, y has any size
    assert_size([x, y], 1)              # x and y are scalar (size 1)
    assert_size((x, y), (5, 2))         # x has size 5, y has size 2

  Args:
    inputs: An array or a sequence of arrays.
    expected_sizes: A sqeuence of expected sizes associated with each input,
      where the expected size is a sequence of integer and `None` dimensions;
      if all inputs have same size, a single size may be passed as 
      ``expected_sizes``.

  Raises:
    AssertionError: If the lengths of ``inputs`` and ``expected_sizes`` do not
      match; if ``expected_sizes`` has wrong type; if size of ``input`` does
      not match ``expected_sizes``.
  """
  # Ensure inputs and expected sizes are sequences.
  if not isinstance(inputs, collections.abc.Sequence):
    inputs = [inputs]

  if isinstance(expected_sizes, int):
    expected_sizes = [expected_sizes] * len(inputs)

  if not isinstance(expected_sizes, (list, tuple)):
    raise AssertionError(
        "Error in size compatibility check: expected sizes should be an int, "
        f"list, or tuple of ints, got {expected_sizes}.")

  if len(inputs) != len(expected_sizes):
    raise AssertionError(
        "Length of `inputs` and `expected_sizes` must match: "
        f"{len(inputs)} is not equal to {len(expected_sizes)}.")

  errors = []
  for idx, (x, expected) in enumerate(zip(inputs, expected_sizes)):
    size = getattr(x, "size", 1)  # scalars have size 1 by definition.
    # Allow any size for the ellipsis case and allow handling of integer
    # expected sizes or collection of acceptable expected sizes.
    int_condition = expected in {Ellipsis, None} or size == expected
    set_condition = (isinstance(expected, collections.abc.Collection) and
                     size in expected)
    if not (int_condition or set_condition):
      errors.append((idx, size, expected))

  if errors:
    msg = "; ".join(
        f"input {e[0]} has size {e[1]} but expected {e[2]}" for e in errors)
    raise AssertionError(f"Error in size compatibility check: {msg}.")


@_static_assertion
def assert_equal_shape(
    inputs: Sequence[Array],
    *,
    dims: Optional[Union[int, Sequence[int]]] = None) -> None:
  """Checks that all arrays have the same shape.

  Args:
    inputs: A collection of arrays.
    dims: An optional integer or sequence of integers. If not provided, every
      dimension of every shape must match. If provided, equality of shape will
      only be asserted for the specified dim(s), i.e. to ensure all of a group
      of arrays have the same size in the first two dimensions, call
      ``assert_equal_shape(tensors_list, dims=(0, 1))``.

  Raises:
    AssertionError: If the shapes of all arrays at specified dims do not match.
    ValueError: If the provided ``dims`` are invalid indices into any of arrays;
      or if ``inputs`` is not a collection of arrays.
  """
  _ai.assert_collection_of_arrays(inputs)

  # NB: Need explicit dims argument, closing over it triggers linter bug.
  def extract_relevant_dims(shape, dims):
    try:
      if dims is None:
        return shape
      elif isinstance(dims, int):
        return shape[dims]
      else:
        return [shape[d] for d in dims]
    except IndexError as err:
      raise ValueError(
          f"Indexing error when trying to extra dim(s) {dims} from array shape "
          f"{shape}") from err

  shape = extract_relevant_dims(inputs[0].shape, dims)
  expected_shapes = [shape] * len(inputs)
  shapes = [extract_relevant_dims(x.shape, dims) for x in inputs]
  if shapes != expected_shapes:
    if dims is not None:
      msg = f"Arrays have different shapes at dims {dims}: {shapes}"
    else:
      msg = f"Arrays have different shapes: {shapes}."
    raise AssertionError(msg)


@_static_assertion
def assert_equal_shape_prefix(inputs: Sequence[Array], prefix_len: int) -> None:
  """Checks that the leading ``prefix_dims`` dims of all inputs have same shape.

  Args:
    inputs: A collection of input arrays.
    prefix_len: A number of leading dimensions to compare; each input's shape
      will be sliced to ``shape[:prefix_len]``. Negative values are accepted and
      have the conventional Python indexing semantics.

  Raises:
    AssertionError: If the shapes of all arrays do not match.
    ValuleError: If ``inputs`` is not a collection of arrays.
  """
  _ai.assert_collection_of_arrays(inputs)

  shapes = [array.shape[:prefix_len] for array in inputs]
  if shapes != [shapes[0]] * len(shapes):
    raise AssertionError(f"Arrays have different shape prefixes: {shapes}")


@_static_assertion
def assert_equal_shape_suffix(inputs: Sequence[Array], suffix_len: int) -> None:
  """Checks that the final ``suffix_len`` dims of all inputs have same shape.

  Args:
    inputs: A collection of input arrays.
    suffix_len: A number of trailing dimensions to compare; each input's shape
      will be sliced to ``shape[-suffix_len:]``. Negative values are accepted
      and have the conventional Python indexing semantics.

  Raises:
    AssertionError: If the shapes of all arrays do not match.
    ValuleError: If ``inputs`` is not a collection of arrays.
  """
  _ai.assert_collection_of_arrays(inputs)

  shapes = [array.shape[-suffix_len:] for array in inputs]
  if shapes != [shapes[0]] * len(shapes):
    raise AssertionError(f"Arrays have different shape suffixes: {shapes}")


def _unelided_shape_matches(
    actual_shape: Sequence[int],
    expected_shape: Sequence[Optional[Union[int, Set[int]]]]) -> bool:
  """Returns True if `actual_shape` is compatible with `expected_shape`."""
  if len(actual_shape) != len(expected_shape):
    return False
  for actual, expected in zip(actual_shape, expected_shape):
    if expected is None:
      continue
    if isinstance(expected, set):
      if actual not in expected:
        return False
    elif actual != expected:
      return False
  return True


def _shape_matches(actual_shape: Sequence[int],
                   expected_shape: _ai.TShapeMatcher) -> bool:
  """Returns True if `actual_shape` is compatible with `expected_shape`."""
  # Splits `expected_shape` based on the position of the ellipsis, if present.
  expected_prefix: List[_ai.TDimMatcher] = []
  expected_suffix: Optional[List[_ai.TDimMatcher]] = None
  for dim in expected_shape:
    if dim is Ellipsis:
      if expected_suffix is not None:
        raise ValueError(
            "`expected_shape` may not contain more than one ellipsis, "
            f"but got {_ai.format_shape_matcher(expected_shape)}")
      expected_suffix = []
    elif expected_suffix is None:
      expected_prefix.append(dim)
    else:
      expected_suffix.append(dim)

  # If there is no ellipsis, just compare to the full `actual_shape`.
  if expected_suffix is None:
    assert len(expected_prefix) == len(expected_shape)
    return _unelided_shape_matches(actual_shape, expected_prefix)

  # Checks that the actual rank is least the number of non-elided dimensions.
  if len(actual_shape) < len(expected_prefix) + len(expected_suffix):
    return False

  if expected_prefix:
    actual_prefix = actual_shape[:len(expected_prefix)]
    if not _unelided_shape_matches(actual_prefix, expected_prefix):
      return False

  if expected_suffix:
    actual_suffix = actual_shape[-len(expected_suffix):]
    if not _unelided_shape_matches(actual_suffix, expected_suffix):
      return False

  return True


@_static_assertion
def assert_shape(
    inputs: Union[Scalar, Union[Array, Sequence[Array]]],
    expected_shapes: Union[_ai.TShapeMatcher,
                           Sequence[_ai.TShapeMatcher]]) -> None:
  """Checks that the shape of all inputs matches specified ``expected_shapes``.

  Valid usages include:

  .. code-block:: python

    assert_shape(x, ())                  # x is scalar
    assert_shape(x, (2, 3))              # x has shape (2, 3)
    assert_shape(x, (2, {1, 3}))         # x has shape (2, 1) or (2, 3)
    assert_shape(x, (2, None))           # x has rank 2 and `x.shape[0] == 2`
    assert_shape(x, (2, ...))            # x has rank >= 1 and `x.shape[0] == 2`
    assert_shape([x, y], ())             # x and y are scalar
    assert_shape([x, y], [(), (2,3)])    # x is scalar and y has shape (2, 3)

  Args:
    inputs: An array or a sequence of arrays.
    expected_shapes: A sequence of expected shapes associated with each input,
      where the expected shape is a sequence of integer and `None` dimensions;
      if all inputs have same shape, a single shape may be passed as
      ``expected_shapes``.

  Raises:
    AssertionError: If the lengths of ``inputs`` and ``expected_shapes`` do not
      match; if ``expected_shapes`` has wrong type; if shape of ``input`` does
      not match ``expected_shapes``.
  """
  if not isinstance(expected_shapes, (list, tuple)):
    raise AssertionError(
        "Error in shape compatibility check: expected shapes should be a list "
        f"or tuple of ints, got {expected_shapes}.")

  # Ensure inputs and expected shapes are sequences.
  if not isinstance(inputs, collections.abc.Sequence):
    inputs = [inputs]

  # Shapes are always lists or tuples, not scalars.
  if (not expected_shapes or not isinstance(expected_shapes[0], (list, tuple))):
    expected_shapes = [expected_shapes] * len(inputs)
  if len(inputs) != len(expected_shapes):
    raise AssertionError(
        "Length of `inputs` and `expected_shapes` must match: "
        f"{len(inputs)} is not equal to {len(expected_shapes)}.")

  errors = []
  for idx, (x, expected) in enumerate(zip(inputs, expected_shapes)):
    shape = getattr(x, "shape", ())  # scalars have shape () by definition.
    if not _shape_matches(shape, expected):
      errors.append((idx, shape, _ai.format_shape_matcher(expected)))

  if errors:
    msg = "; ".join(
        f"input {e[0]} has shape {e[1]} but expected {e[2]}" for e in errors)
    raise AssertionError(f"Error in shape compatibility check: {msg}.")


@_static_assertion
def assert_is_broadcastable(shape_a: Sequence[int],
                            shape_b: Sequence[int]) -> None:
  """Checks that an array of ``shape_a`` is broadcastable to one of ``shape_b``.

  Args:
    shape_a: A shape of the array to check.
    shape_b: A target shape after broadcasting.

  Raises:
    AssertionError: If ``shape_a`` is not broadcastable to ``shape_b``.
  """
  error = AssertionError(
      f"Shape {shape_a} is not broadcastable to shape {shape_b}.")
  ndim_a = len(shape_a)
  ndim_b = len(shape_b)
  if ndim_a > ndim_b:
    raise error
  else:
    for i in range(1, ndim_a + 1):
      if shape_a[-i] != 1 and shape_a[-i] != shape_b[-i]:
        raise error


@_static_assertion
def assert_equal_rank(inputs: Sequence[Array]) -> None:
  """Checks that all arrays have the same rank.

  Args:
    inputs: A collection of arrays.

  Raises:
    AssertionError: If the ranks of all arrays do not match.
    ValueError: If ``inputs`` is not a collection of arrays.
  """
  _ai.assert_collection_of_arrays(inputs)

  rank = len(inputs[0].shape)
  expected_ranks = [rank] * len(inputs)
  ranks = [len(x.shape) for x in inputs]
  if ranks != expected_ranks:
    raise AssertionError(f"Arrays have different rank: {ranks}.")


@_static_assertion
def assert_rank(
    inputs: Union[Scalar, Union[Array, Sequence[Array]]],
    expected_ranks: Union[int, Set[int], Sequence[Union[int,
                                                        Set[int]]]]) -> None:
  """Checks that the rank of all inputs matches specified ``expected_ranks``.

  Valid usages include:

  .. code-block:: python

    assert_rank(x, 0)                      # x is scalar
    assert_rank(x, 2)                      # x is a rank-2 array
    assert_rank(x, {0, 2})                 # x is scalar or rank-2 array
    assert_rank([x, y], 2)                 # x and y are rank-2 arrays
    assert_rank([x, y], [0, 2])            # x is scalar and y is a rank-2 array
    assert_rank([x, y], {0, 2})            # x and y are scalar or rank-2 arrays

  Args:
    inputs: An array or a sequence of arrays.
    expected_ranks: A sequence of expected ranks associated with each input,
      where the expected rank is either an integer or set of integer options; if
      all inputs have same rank, a single scalar or set of scalars may be passed
      as ``expected_ranks``.

  Raises:
    AssertionError: If lengths of ``inputs`` and ``expected_ranks`` don't match;
      if ``expected_ranks`` has wrong type;
      if the ranks of ``inputs`` do not match ``expected_ranks``.
    ValueError: If ``expected_ranks`` is not an integer and not a sequence of
     integets.
  """
  if not isinstance(expected_ranks, (collections.abc.Collection, int)):
    raise ValueError(
        f"Error in rank compatibility check: expected ranks should be a single "
        f"integer or a collection of integers, got {expected_ranks}.")

  if isinstance(expected_ranks, np.ndarray):  # ndarray is abc.Collection
    raise ValueError(
        f"Error in rank compatibility check: expected ranks should be a single "
        f"integer or a collection of integers, but was an array: "
        f"{expected_ranks}.")

  # Ensure inputs and expected ranks are sequences.
  if not isinstance(inputs, collections.abc.Sequence):
    inputs = [inputs]
  if (not isinstance(expected_ranks, collections.abc.Sequence) or
      isinstance(expected_ranks, collections.abc.Set)):
    expected_ranks = [expected_ranks] * len(inputs)
  if len(inputs) != len(expected_ranks):
    raise AssertionError(
        "Length of inputs and expected_ranks must match: inputs has length "
        f"{len(inputs)}, expected_ranks has length {len(expected_ranks)}.")

  errors = []
  for idx, (x, expected) in enumerate(zip(inputs, expected_ranks)):
    if hasattr(x, "shape"):
      shape = x.shape
    else:
      shape = ()  # scalars have shape () by definition.
    rank = len(shape)

    # Multiple expected options can be specified.

    # Check against old usage where options could be any sequence
    if (isinstance(expected, collections.abc.Sequence) and
        not isinstance(expected, collections.abc.Set)):
      raise ValueError("Error in rank compatibility check: "
                       "Expected ranks should be integers or sets of integers.")

    options = (
        expected if isinstance(expected, collections.abc.Set) else {expected})

    if rank not in options:
      errors.append((idx, rank, shape, expected))

  if errors:
    msg = "; ".join(
        f"input {e[0]} has rank {e[1]} (shape {e[2]}) but expected {e[3]}"
        for e in errors)

    raise AssertionError(f"Error in rank compatibility check: {msg}.")


@_static_assertion
def assert_type(
    inputs: Union[Scalar, Union[Array, Sequence[Array]]],
    expected_types: Union[ArrayDType, Sequence[ArrayDType]]) -> None:
  """Checks that the type of all inputs matches specified ``expected_types``.

  If the expected type is a Python type or abstract dtype (e.g. `np.floating`),
  assert that the input has the same sub-type. If the expected type is a
  concrete dtype (e.g. np.float32), assert that the input's type is the same.

  Example usage:

  .. code-block:: python

    assert_type(7, int)
    assert_type(7.1, float)
    assert_type(False, bool)
    assert_type([7, 8], int)
    assert_type([7, 7.1], [int, float])
    assert_type(np.array(7), int)
    assert_type(np.array(7.1), float)
    assert_type([jnp.array([7, 8]), np.array(7.1)], [int, float])
    assert_type(jnp.array(1., dtype=jnp.bfloat16)), jnp.bfloat16)
    assert_type(jnp.ones(1, dtype=np.int8), np.int8)

  Args:
    inputs: An array or a sequence of arrays or scalars.
    expected_types: A sequence of expected types associated with each input; if
      all inputs have same type, a single type may be passed as
      ``expected_types``.

  Raises:
    AssertionError: If lengths of ``inputs`` and ``expected_types`` don't match;
      if ``expected_types`` contains unsupported pytype;
      if the types of inputs do not match the expected types.
  """
  if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]
  if not isinstance(expected_types, (list, tuple)):
    expected_types = [expected_types] * len(inputs)

  errors = []
  if len(inputs) != len(expected_types):
    raise AssertionError(
        "Length of `inputs` and `expected_types` must match, "
        f"got {len(inputs)} != {len(expected_types)}."
    )
  for idx, (x, expected) in enumerate(zip(inputs, expected_types)):
    dtype = np.result_type(x)
    if expected in {float, jnp.floating}:
      if not jnp.issubdtype(dtype, jnp.floating):
        errors.append((idx, dtype, expected))
    elif expected in {int, jnp.integer}:
      if not jnp.issubdtype(dtype, jnp.integer):
        errors.append((idx, dtype, expected))
    else:
      expected = np.dtype(expected)
      if dtype != expected:
        errors.append((idx, dtype, expected))

  if errors:
    msg = "; ".join(
        f"input {e[0]} has type {e[1]} but expected {e[2]}" for e in errors)

    raise AssertionError(f"Error in type compatibility check: {msg}.")


@_static_assertion
def assert_axis_dimension_comparator(tensor: Array, axis: int,
                                     pass_fn: Callable[[int], bool],
                                     error_string: str):
  """Asserts that `pass_fn(tensor.shape[axis])` passes.

  Used to implement ==, >, >=, <, <= checks.

  Args:
    tensor: A JAX array.
    axis: An integer specifying which axis to assert.
    pass_fn: A callable which takes the size of the give dimension and returns
      false when the assertion should fail.
    error_string: string which is inserted in assertion failure messages -
      'expected tensor to have dimension {error_string} on axis ...'.

  Raises:
    AssertionError: if `pass_fn(tensor.shape[axis], val)` does not return true.
  """
  if not isinstance(tensor, (jax.Array, np.ndarray)):
    tensor = np.asarray(tensor)  # np is broader than jnp (it supports strings)
  if axis >= len(tensor.shape) or axis < -len(tensor.shape):
    raise AssertionError(
        f"Expected tensor to have dim {error_string} on axis "
        f"'{axis}' but axis '{axis}' not available: tensor rank is "
        f"'{len(tensor.shape)}'.")
  if not pass_fn(tensor.shape[axis]):
    raise AssertionError(
        f"Expected tensor to have dimension {error_string} on axis"
        f" '{axis}' but got '{tensor.shape[axis]}' instead.")


@_static_assertion
def assert_axis_dimension(tensor: Array, axis: int, expected: int) -> None:
  """Checks that ``tensor.shape[axis] == expected``.

  Args:
    tensor: A JAX array.
    axis: An integer specifying which axis to assert.
    expected: An expected value of ``tensor.shape[axis]``.

  Raises:
    AssertionError:
      The dimension of the specified axis does not match the prescribed value.
  """
  assert_axis_dimension_comparator(
      tensor,
      axis,
      pass_fn=lambda tensor_dim: tensor_dim == expected,
      error_string=f"equal to '{expected}'")


@_static_assertion
def assert_axis_dimension_gt(tensor: Array, axis: int, val: int) -> None:
  """Checks that ``tensor.shape[axis] > val``.

  Args:
    tensor: A JAX array.
    axis: An integer specifying which axis to assert.
    val: A value ``tensor.shape[axis]`` must be greater than.

  Raises:
    AssertionError: if the dimension of ``axis`` is <= ``val``.
  """
  assert_axis_dimension_comparator(
      tensor,
      axis,
      pass_fn=lambda tensor_dim: tensor_dim > val,
      error_string=f"greater than '{val}'")


@_static_assertion
def assert_axis_dimension_gteq(tensor: Array, axis: int, val: int) -> None:
  """Checks that ``tensor.shape[axis] >= val``.

  Args:
    tensor: A JAX array.
    axis: An integer specifying which axis to assert.
    val: A value ``tensor.shape[axis]`` must be greater than or equal to.

  Raises:
    AssertionError: if the dimension of ``axis`` is < ``val``.
  """
  assert_axis_dimension_comparator(
      tensor,
      axis,
      pass_fn=lambda tensor_dim: tensor_dim >= val,
      error_string=f"greater than or equal to '{val}'")


@_static_assertion
def assert_axis_dimension_lt(tensor: Array, axis: int, val: int) -> None:
  """Checks that ``tensor.shape[axis] < val``.

  Args:
    tensor: A JAX Array.
    axis: An integer specifiying with axis to assert.
    val: A value ``tensor.shape[axis]`` must be less than.

  Raises:
    AssertionError: if the dimension of ``axis`` is >= ``val``.
  """
  assert_axis_dimension_comparator(
      tensor,
      axis,
      pass_fn=lambda tensor_dim: tensor_dim < val,
      error_string=f"less than '{val}'")


@_static_assertion
def assert_axis_dimension_lteq(tensor: Array, axis: int, val: int) -> None:
  """Checks that ``tensor.shape[axis] <= val``.

  Args:
    tensor: A JAX array.
    axis: An integer specifying which axis to assert.
    val: A value ``tensor.shape[axis]`` must be less than or equal to.

  Raises:
    AssertionError: if the dimension of ``axis`` is > ``val``.
  """
  assert_axis_dimension_comparator(
      tensor,
      axis,
      pass_fn=lambda tensor_dim: tensor_dim <= val,
      error_string=f"less than or equal to '{val}'")


@_static_assertion
def assert_numerical_grads(f: Callable[..., Array],
                           f_args: Sequence[Array],
                           order: int,
                           atol: float = 0.01,
                           **check_kwargs) -> None:
  """Checks that autodiff and numerical gradients of a function match.

  Args:
    f: A function to check.
    f_args: Arguments of the function.
    order: An order of gradients.
    atol: An absolute tolerance.
    **check_kwargs: Kwargs for ``jax_test.check_grads``.

  Raises:
    AssertionError: If automatic differentiation gradients deviate from finite
      difference gradients.
  """
  # Correct scaling.
  # Remove after https://github.com/google/jax/issues/3130 is fixed.
  atol *= f_args[0].size

  # Mock `jax.lax.stop_gradient` because finite diff. method does not honour it.
  mock_sg = lambda t: jax.tree_util.tree_map(jnp.ones_like, t)
  with mock.patch("jax.lax.stop_gradient", mock_sg):
    jax_test.check_grads(f, f_args, order=order, atol=atol, **check_kwargs)


# "static" because tracers can be compared with `None`.
@_static_assertion
def assert_tree_no_nones(tree: ArrayTree) -> None:
  """Checks that a tree does not contain `None`.

  Args:
    tree: A tree to assert.

  Raises:
    AssertionError: If the tree contains at least one `None`.
  """
  has_nones = False

  def _is_leaf(value):
    if value is None:
      nonlocal has_nones
      has_nones = True
    return False

  treedef = jax.tree_util.tree_structure(tree, is_leaf=_is_leaf)
  if has_nones:
    raise AssertionError(f"Tree contains `None`(s): {treedef}.")


@_static_assertion
def assert_tree_has_only_ndarrays(tree: ArrayTree) -> None:
  """Checks that all `tree`'s leaves are n-dimensional arrays (tensors).

  Args:
    tree: A tree to assert.

  Raises:
    AssertionError: If the tree contains an object which is not an ndarray.
  """
  errors = []

  def _assert_fn(path, leaf):
    if leaf is not None:
      if not isinstance(leaf, (np.ndarray, jnp.ndarray)):
        nonlocal errors
        errors.append((f"Tree leaf '{_ai.format_tree_path(path)}' is not an "
                       f"ndarray (type={type(leaf)})."))

  for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
    _assert_fn(_ai.convert_jax_path_to_dm_path(path), leaf)
  if errors:
    raise AssertionError("\n".join(errors))


# Only look the sharding attribute after jax version >= 0.3.22 i.e. remove this
# function and use `isinstance(x.sharding, jax.sharding.PmapSharding)` after
# jax version >= 0.3.22.
# This is for backwards compatibility.
def _check_sharding(x):
  if hasattr(jax, "Array") and isinstance(x, jax.Array):
    if isinstance(x.sharding, jax.sharding.PmapSharding):
      return True
    else:
      return len(x.sharding.device_set) > 1
  # pytype: disable=attribute-error
  return (
      hasattr(jax, "pxla")
      and hasattr(jax.pxla, "ShardedDeviceArray")
      and isinstance(x, jax.pxla.ShardedDeviceArray)
  )
  # pytype: enable=attribute-error


@_static_assertion
def assert_tree_is_on_host(
    tree: ArrayTree,
    *,
    allow_cpu_device: bool = True,
    allow_sharded_arrays: bool = False,
) -> None:
  """Checks that all leaves are ndarrays residing in the host memory (on CPU).

  This assertion only accepts trees consisting of ndarrays.

  Args:
    tree: A tree to assert.
    allow_cpu_device: Whether to allow JAX arrays that reside on a CPU device.
    allow_sharded_arrays: Whether to allow sharded JAX arrays. Sharded arrays
      are considered "on host" only if they are sharded across CPU devices and
      `allow_cpu_device` is `True`.

  Raises:
    AssertionError: If the tree contains a leaf that is not an ndarray or does
      not reside on host.
  """
  assert_tree_has_only_ndarrays(tree)
  errors = []

  def _assert_fn(path, leaf):
    if leaf is not None:
      if not isinstance(leaf, np.ndarray):
        nonlocal errors

        if isinstance(leaf, jax.Array):
          if _check_sharding(leaf):
            # Sharded array.
            if not allow_sharded_arrays:
              errors.append(
                  f"Tree leaf '{_ai.format_tree_path(path)}' is sharded and"
                  f" resides on {leaf.devices()} (sharded arrays are"
                  " disallowed)."
              )
            elif allow_cpu_device:
              if any(d.platform != "cpu" for d in leaf.devices()):
                errors.append(
                    f"Tree leaf '{_ai.format_tree_path(path)}' is sharded and"
                    f" resides on {leaf.devices()}."
                )
            else:
              errors.append(
                  f"Tree leaf '{_ai.format_tree_path(path)}' is sharded and"
                  f" resides on {leaf.devices()} (CPU devices are disallowed)."
              )
          elif allow_cpu_device:
            # Device array.
            leaf_device = list(leaf.devices())[0]
            if leaf_device.platform != "cpu":
              errors.append(
                  f"Tree leaf '{_ai.format_tree_path(path)}' resides"
                  f" on {leaf_device}."
              )
          else:
            errors.append((
                f"Tree leaf '{_ai.format_tree_path(path)}' resides "
                f"on {leaf.devices()} (CPU devices are disallowed)."
            ))
        else:
          # Not a jax.Array.
          errors.append((
              f"Tree leaf '{_ai.format_tree_path(path)}' has "
              f"unexpected type: {type(leaf)}."
          ))

  for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
    _assert_fn(_ai.convert_jax_path_to_dm_path(path), leaf)
  if errors:
    raise AssertionError("\n".join(errors))


@_static_assertion
def assert_tree_is_on_device(tree: ArrayTree,
                             *,
                             platform: Union[Sequence[str],
                                             str] = ("gpu", "tpu"),
                             device: Optional[pytypes.Device] = None) -> None:
  """Checks that all leaves are ndarrays residing in device memory (in HBM).

  Sharded DeviceArrays are disallowed.

  Args:
    tree: A tree to assert.
    platform: A platform or a list of platforms where the leaves are expected to
      reside. Ignored if `device` is specified.
    device: An optional device where the tree's arrays are expected to reside.
      Any device (except CPU) is accepted if not specified.

  Raises:
    AssertionError: If the tree contains a leaf that is not an ndarray or does
      not reside on the specified device or platform.
  """
  assert_tree_has_only_ndarrays(tree)

  # If device is specified, require its platform.
  if device is not None:
    platform = (device.platform,)
  elif not isinstance(platform, collections.abc.Sequence):
    platform = (platform,)

  errors = []

  def _assert_fn(path, leaf):
    if leaf is not None:
      nonlocal errors

      # Check that the leaf is a DeviceArray.
      if isinstance(leaf, jax.Array):
        if _check_sharding(leaf):
          errors.append((f"Tree leaf '{_ai.format_tree_path(path)}' is a "
                         f"ShardedDeviceArray which are disallowed. "
                         f" (type={type(leaf)})."))
        else:  # DeviceArray and not ShardedDeviceArray
          # Check the platform.
          leaf_device = list(leaf.devices())[0]
          if leaf_device.platform not in platform:
            errors.append((
                f"Tree leaf '{_ai.format_tree_path(path)}' resides on "
                f"'{leaf_device.platform}', expected '{platform}'."
            ))

          # Check the device.
          if device is not None and leaf.devices() != {device}:
            errors.append((
                f"Tree leaf '{_ai.format_tree_path(path)}' resides on "
                f"{leaf.devices()}, expected {device}."
            ))
      else:
        errors.append((f"Tree leaf '{_ai.format_tree_path(path)}' has "
                       f"unexpected type: {type(leaf)}."))

  for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
    _assert_fn(_ai.convert_jax_path_to_dm_path(path), leaf)
  if errors:
    raise AssertionError("\n".join(errors))


@_static_assertion
def assert_tree_is_sharded(tree: ArrayTree,
                           *,
                           devices: Sequence[pytypes.Device]) -> None:
  """Checks that all leaves are ndarrays sharded across the specified devices.

  Args:
    tree: A tree to assert.
    devices: A list of devices which the tree's leaves are expected to be
      sharded across. This list is order-sensitive.

  Raises:
    AssertionError: If the tree contains a leaf that is not a device array
      sharded across the specified devices.
  """
  assert_tree_has_only_ndarrays(tree)

  errors = []
  devices = tuple(devices)

  def _assert_fn(path, leaf):
    if leaf is not None:
      nonlocal errors

      # Check that the leaf is a ShardedArray.
      if isinstance(leaf, jax.Array):
        if _check_sharding(leaf):
          shards = tuple(shard.device for shard in leaf.addressable_shards)
          if shards != devices:
            errors.append(
                f"Tree leaf '{_ai.format_tree_path(path)}' is sharded "
                f"across {shards} devices, expected {devices}."
            )
        else:
          errors.append(
              f"Tree leaf '{_ai.format_tree_path(path)}' is not sharded"
              f" (devices={leaf.devices()})."
          )
      else:
        errors.append(
            f"Tree leaf '{_ai.format_tree_path(path)}' is not a "
            f"jax.Array (type={type(leaf)})."
        )

  for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
    _assert_fn(_ai.convert_jax_path_to_dm_path(path), leaf)
  if errors:
    raise AssertionError("\n".join(errors))


@_static_assertion
def assert_tree_shape_prefix(tree: ArrayTree,
                             shape_prefix: Sequence[int]) -> None:
  """Checks that all ``tree`` leaves' shapes have the same prefix.

  Args:
    tree: A tree to check.
    shape_prefix: An expected shape prefix.

  Raises:
    AssertionError: If some leaf's shape doesn't start with ``shape_prefix``.
  """
  # To compare with the leaf's `shape`, convert int sequence to tuple.
  shape_prefix = tuple(shape_prefix)

  if not shape_prefix:
    return  # No prefix, this is trivially true.

  errors = []

  def _assert_fn(path, leaf):
    nonlocal errors
    if len(shape_prefix) > len(leaf.shape):
      errors.append(
          (f"Tree leaf '{_ai.format_tree_path(path)}' has a shape "
           f"of length {leaf.ndim} (shape={leaf.shape}) which is smaller "
           f"than the expected prefix of length {len(shape_prefix)} "
           f"(prefix={shape_prefix})."))
      return

    suffix = leaf.shape[:len(shape_prefix)]
    if suffix != shape_prefix:
      errors.append(
          (f"Tree leaf '{_ai.format_tree_path(path)}' has a shape prefix "
           f"different from expected: {suffix} != {shape_prefix}."))

  for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
    _assert_fn(_ai.convert_jax_path_to_dm_path(path), leaf)
  if errors:
    raise AssertionError("\n".join(errors))


@_static_assertion
def assert_tree_shape_suffix(
    tree: ArrayTree, shape_suffix: Sequence[int]
) -> None:
  """Checks that all ``tree`` leaves' shapes have the same suffix.

  Args:
    tree: A tree to check.
    shape_suffix: An expected shape suffix.

  Raises:
    AssertionError: If some leaf's shape doesn't end with ``shape_suffix``.
  """
  # To compare with the leaf's `shape`, convert int sequence to tuple.
  shape_suffix = tuple(shape_suffix)

  if not shape_suffix:
    return  # No suffix, this is trivially true.

  errors = []

  def _assert_fn(path, leaf):
    nonlocal errors
    if len(shape_suffix) > len(leaf.shape):
      errors.append(
          (f"Tree leaf '{_ai.format_tree_path(path)}' has a shape "
           f"of length {len(leaf.shape)} (shape={leaf.shape}) which is smaller "
           f"than the expected suffix of length {len(shape_suffix)} "
           f"(suffix={shape_suffix})."))
      return

    suffix = leaf.shape[-len(shape_suffix):]
    if suffix != shape_suffix:
      errors.append(
          (f"Tree leaf '{_ai.format_tree_path(path)}' has a shape suffix "
           f"different from expected: {suffix} != {shape_suffix}."))

  for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
    _assert_fn(_ai.convert_jax_path_to_dm_path(path), leaf)
  if errors:
    raise AssertionError("\n".join(errors))


@_static_assertion
def assert_trees_all_equal_structs(*trees: ArrayTree) -> None:
  """Checks that trees have the same structure.

  Args:
    *trees: A sequence of (at least 2) trees to assert equal structure between.

  Raises:
    ValueError: If ``trees`` does not contain at least 2 elements.
    AssertionError: If structures of any two trees are different.
  """
  if len(trees) < 2:
    raise ValueError(
        "assert_trees_all_equal_structs on a single tree does not make sense. "
        "Maybe you wrote `assert_trees_all_equal_structs([a, b])` instead of "
        "`assert_trees_all_equal_structs(a, b)` ?")

  first_treedef = jax.tree_util.tree_structure(trees[0])
  other_treedefs = (jax.tree_util.tree_structure(t) for t in trees[1:])
  for i, treedef in enumerate(other_treedefs, start=1):
    if first_treedef != treedef:
      raise AssertionError(
          f"Error in tree structs equality check: trees 0 and {i} do not match,"
          f"\n tree 0: {first_treedef}"
          f"\n tree {i}: {treedef}")


@_static_assertion
def assert_trees_all_equal_comparator(equality_comparator: _ai.TLeavesEqCmpFn,
                                      error_msg_fn: _ai.TLeavesEqCmpErrorFn,
                                      *trees: ArrayTree) -> None:
  """Checks that all trees are equal as per the custom comparator for leaves.

  Args:
    equality_comparator: A custom function that accepts two leaves and checks
      whether they are equal. Expected to be transitive.
    error_msg_fn: A function accepting two unequal as per
      ``equality_comparator`` leaves and returning an error message.
    *trees: A sequence of (at least 2) trees to check on equality as per
      ``equality_comparator``.

  Raises:
    ValueError: If ``trees`` does not contain at least 2 elements.
    AssertionError: if ``equality_comparator`` returns `False` for any pair of
                    trees from ``trees``.
  """
  if len(trees) < 2:
    raise ValueError(
        "Assertions over only one tree does not make sense. Maybe you wrote "
        "`assert_trees_xxx([a, b])` instead of `assert_trees_xxx(a, b)`, or "
        "forgot the `error_msg_fn` arg to `assert_trees_all_equal_comparator`?")
  assert_trees_all_equal_structs(*trees)

  def tree_error_msg_fn(l_1: _ai.TLeaf, l_2: _ai.TLeaf, path: str, i_1: int,
                        i_2: int):
    msg = error_msg_fn(l_1, l_2)
    if path:
      return f"Trees {i_1} and {i_2} differ in leaves '{path}': {msg}."
    else:
      return f"Trees (arrays) {i_1} and {i_2} differ: {msg}."

  cmp_fn = functools.partial(_ai.assert_leaves_all_eq_comparator,
                             equality_comparator, tree_error_msg_fn)

  # Trees are guaranteed to have the same structure.
  paths = [
      _ai.convert_jax_path_to_dm_path(path)
      for path, _ in jax.tree_util.tree_flatten_with_path(trees[0])[0]]
  trees_leaves = [jax.tree_util.tree_leaves(tree) for tree in trees]
  for leaf_i, path in enumerate(paths):
    cmp_fn(path, *[leaves[leaf_i] for leaves in trees_leaves])


@_static_assertion
def assert_trees_all_equal_dtypes(*trees: ArrayTree) -> None:
  """Checks that trees' leaves have the same dtype.

  Args:
    *trees: A sequence of (at least 2) trees to check.

  Raises:
    AssertionError: If leaves' dtypes for any two trees differ.
  """
  def cmp_fn(arr_1, arr_2):
    return (hasattr(arr_1, "dtype") and hasattr(arr_2, "dtype") and
            arr_1.dtype == arr_2.dtype)

  def err_msg_fn(arr_1, arr_2):
    if not hasattr(arr_1, "dtype"):
      return f"{type(arr_1)} is not a (j-)np array (has no `dtype` property)"
    if not hasattr(arr_2, "dtype"):
      return f"{type(arr_2)} is not a (j-)np array (has no `dtype` property)"
    return f"types: {arr_1.dtype} != {arr_2.dtype}"

  assert_trees_all_equal_comparator(cmp_fn, err_msg_fn, *trees)


@_static_assertion
def assert_trees_all_equal_sizes(*trees: ArrayTree) -> None:
  """Checks that trees have the same structure and leaves' sizes.

  Args:
    *trees: A sequence of (at least 2) trees with array leaves.

  Raises:
    AssertionError: If trees' structures or leaves' sizes are different.
  """
  cmp_fn = lambda arr_1, arr_2: arr_1.size == arr_2.size
  err_msg_fn = lambda arr_1, arr_2: f"sizes: {arr_1.size} != {arr_2.size}"
  assert_trees_all_equal_comparator(cmp_fn, err_msg_fn, *trees)


@_static_assertion
def assert_trees_all_equal_shapes(*trees: ArrayTree) -> None:
  """Checks that trees have the same structure and leaves' shapes.

  Args:
    *trees: A sequence of (at least 2) trees with array leaves.

  Raises:
    AssertionError: If trees' structures or leaves' shapes are different.
  """
  cmp_fn = lambda arr_1, arr_2: arr_1.shape == arr_2.shape
  err_msg_fn = lambda arr_1, arr_2: f"shapes: {arr_1.shape} != {arr_2.shape}"
  assert_trees_all_equal_comparator(cmp_fn, err_msg_fn, *trees)


@_static_assertion
def assert_trees_all_equal_shapes_and_dtypes(*trees: ArrayTree) -> None:
  """Checks that trees' leaves have the same shape and dtype.

  Args:
    *trees: A sequence of (at least 2) trees to check.

  Raises:
    AssertionError: If leaves' shapes or dtypes for any two trees differ.
  """
  assert_trees_all_equal_shapes(*trees)
  assert_trees_all_equal_dtypes(*trees)


############# Value assertions. #############


def _assert_tree_all_finite_static(tree_like: ArrayTree) -> None:
  """Checks that all leaves in a tree are finite.

  Args:
    tree_like: A pytree with array leaves.

  Raises:
    AssertionError: If any leaf in ``tree_like`` is non-finite.
  """
  all_finite = jax.tree_util.tree_all(
      jax.tree_util.tree_map(lambda x: np.all(np.isfinite(x)), tree_like))
  if not all_finite:
    is_finite = lambda x: "Finite" if np.all(np.isfinite(x)) else "Nonfinite"
    error_msg = jax.tree.map(is_finite, tree_like)
    raise AssertionError(f"Tree contains non-finite value: {error_msg}.")


def _assert_tree_all_finite_jittable(tree_like: ArrayTree) -> Array:
  """A jittable version of `_assert_tree_all_finite_static`."""
  labeled_tree = jax.tree.map(
      lambda x: jax.lax.select(jnp.isfinite(x).all(), .0, jnp.nan), tree_like
  )
  predicate = jnp.all(
      jnp.isfinite(jnp.asarray(jax.tree_util.tree_leaves(labeled_tree)))
  )
  checkify.check(
      pred=predicate,
      msg="Tree contains non-finite value: {tree}.",
      tree=labeled_tree,
  )
  return predicate


assert_tree_all_finite = _value_assertion(
    assert_fn=_assert_tree_all_finite_static,
    jittable_assert_fn=_assert_tree_all_finite_jittable,
    name="assert_tree_all_finite")


@_static_assertion
def _assert_trees_all_equal_static(
    *trees: ArrayTree, strict: bool = False
) -> None:
  """Checks that all trees have leaves with *exactly* equal values.

  If you are comparing floating point numbers, an exact equality check may not
  be appropriate; consider using ``assert_trees_all_close``.

  Args:
    *trees: A sequence of (at least 2) trees with array leaves.
    strict: If True, disable special scalar handling as described in
      `np.testing.assert_array_equals` notes section.

  Raises:
    AssertionError: If the leaf values actual and desired are not exactly equal.
  """
  def assert_fn(arr_1, arr_2):
    if isinstance(arr_1, jax.Array) and jax.dtypes.issubdtype(
        arr_1.dtype, jax.dtypes.prng_key
    ) and isinstance(arr_2, jax.Array) and jax.dtypes.issubdtype(
        arr_2.dtype, jax.dtypes.prng_key
    ):
      assert jax.random.key_impl(arr_1) == jax.random.key_impl(arr_2)
      arr_1 = jax.random.key_data(arr_1)
      arr_2 = jax.random.key_data(arr_2)
    np.testing.assert_array_equal(
        _ai.jnp_to_np_array(arr_1),
        _ai.jnp_to_np_array(arr_2),
        err_msg="Error in value equality check: Values not exactly equal",
        strict=strict,
    )

  def cmp_fn(arr_1, arr_2) -> bool:
    try:
      # Raises an AssertionError if values are not equal.
      assert_fn(arr_1, arr_2)
    except AssertionError:
      return False
    return True

  def err_msg_fn(arr_1, arr_2) -> str:
    try:
      assert_fn(arr_1, arr_2)
    except AssertionError as e:
      dtype_1 = (
          arr_1.dtype
          if isinstance(arr_1, jax.Array)
          else np.asarray(arr_1).dtype
      )
      dtype_2 = (
          arr_2.dtype
          if isinstance(arr_2, jax.Array)
          else np.asarray(arr_1).dtype
      )
      return f"{str(e)} \nOriginal dtypes: {dtype_1}, {dtype_2}"
    return ""

  assert_trees_all_equal_comparator(cmp_fn, err_msg_fn, *trees)


def _assert_trees_all_equal_jittable(
    *trees: ArrayTree, strict: bool = True,
) -> Array:
  """A jittable version of `_assert_trees_all_equal_static`."""
  if not strict:
    raise NotImplementedError(
        "`strict=False` is not implemented by"
        " `_assert_trees_all_equal_jittable`. This is a feature of"
        " `np.testing.assert_array_equal` used in the static implementation of"
        " `assert_trees_all_equal` that we do not implement in the jittable"
        " version."
    )

  err_msg_template = "Values not exactly equal: {arr_1} != {arr_2}."
  cmp_fn = lambda x, y: jnp.array_equal(x, y, equal_nan=True)
  return _ai.assert_trees_all_eq_comparator_jittable(
      cmp_fn, err_msg_template, *trees
  )


assert_trees_all_equal = _value_assertion(
    assert_fn=_assert_trees_all_equal_static,
    jittable_assert_fn=_assert_trees_all_equal_jittable,
    name="assert_trees_all_equal",
)


def _assert_trees_all_close_static(*trees: ArrayTree,
                                   rtol: float = 1e-06,
                                   atol: float = .0) -> None:
  """Checks that all trees have leaves with approximately equal values.

  This compares the difference between values of actual and desired up to
   ``atol + rtol * abs(desired)``.

  Args:
    *trees: A sequence of (at least 2) trees with array leaves.
    rtol: A relative tolerance.
    atol: An absolute tolerance.

  Raises:
    AssertionError: If actual and desired values are not equal up to
      specified tolerance.
  """
  def assert_fn(arr_1, arr_2):
    np.testing.assert_allclose(
        _ai.jnp_to_np_array(arr_1),
        _ai.jnp_to_np_array(arr_2),
        rtol=rtol,
        atol=atol,
        err_msg="Error in value equality check: Values not approximately equal")

  def cmp_fn(arr_1, arr_2) -> bool:
    try:
      # Raises an AssertionError if values are not close.
      assert_fn(arr_1, arr_2)
    except AssertionError:
      return False
    return True

  def err_msg_fn(arr_1, arr_2) -> str:
    try:
      assert_fn(arr_1, arr_2)
    except AssertionError as e:
      return (f"{str(e)} \nOriginal dtypes: "
              f"{np.asarray(arr_1).dtype}, {np.asarray(arr_2).dtype}")
    return ""

  assert_trees_all_equal_comparator(cmp_fn, err_msg_fn, *trees)


def _assert_trees_all_close_jittable(*trees: ArrayTree,
                                     rtol: float = 1e-06,
                                     atol: float = .0) -> Array:
  """A jittable version of `_assert_trees_all_close_static`."""
  err_msg_template = (
      f"Values not approximately equal ({rtol=}, {atol=}): "
      + "{arr_1} != {arr_2}."
  )
  cmp_fn = lambda x, y: jnp.isclose(x, y, rtol=rtol, atol=atol).all()
  return _ai.assert_trees_all_eq_comparator_jittable(
      cmp_fn, err_msg_template, *trees
  )


assert_trees_all_close = _value_assertion(
    assert_fn=_assert_trees_all_close_static,
    jittable_assert_fn=_assert_trees_all_close_jittable,
    name="assert_trees_all_close")


def _assert_trees_all_close_ulp_static(
    *trees: ArrayTree,
    maxulp: int = 1,
) -> None:
  """Checks that tree leaves differ by at most `maxulp` Units in the Last Place.

  This is the Chex version of np.testing.assert_array_max_ulp.

  Assertions on floating point values are tricky because the precision varies
  depending on the value. For example, with float32, the precision at 1 is
  np.spacing(np.float32(1.0)) ≈ 1e-7, but the precision at 5,000,000 is only
  np.spacing(np.float32(5e6)) = 0.5. This makes it hard to predict ahead of time
  what tolerance to use when checking whether two numbers are equal: a
  difference of only a couple of bits can equate to an arbitrarily large
  absolute difference.

  Assertions based on _relative_ differences are one solution to this problem,
  but have the disadvantage that it's hard to choose the tolerance. If you want
  to verify that two calculations produce _exactly_ the same result
  modulo the inherent non-determinism of floating point operations, do you set
  the tolerance to...0.01? 0.001? It's hard to be sure you've set it low enough
  that you won't miss one of your computations being slightly wrong.

  Assertions based on 'units in the last place' (ULP) instead solve this
  problem by letting you specify tolerances in terms of the precision actually
  available at the current scale of your values. The ULP at some value x is
  essentially the spacing between the floating point numbers actually
  representable in the vicinity of x - equivalent to the 'precision' we
  discussed above. above. With a tolerance of, say, `maxulp=5`, you're saying
  that two values are within 5 actually-representable-numbers of each other -
  a strong guarantee that two computations are as close as possible to
  identical, while still allowing reasonable wiggle room for small differences
  due to e.g. different operator orderings.

  Note that this function is not currently supported within JIT contexts,
  and does not currently support bfloat16 dtypes.

  Args:
    *trees: A sequence of (at least 2) trees with array leaves.
    maxulp: The maximum number of ULPs by which leaves may differ.

  Raises:
    AssertionError: If actual and desired values are not equal up to
      specified tolerance.
  """
  def assert_fn(arr_1, arr_2):
    if (
        getattr(arr_1, "dtype", None) == jnp.bfloat16
        or getattr(arr_2, "dtype", None) == jnp.bfloat16
    ):
      # jnp_to_np_array currently converts bfloat16 to float32, which will cause
      # assert_array_max_ulp to give incorrect results -
      # and assert_array_max_ulp itself does not currently support bfloat16:
      # https://github.com/jax-ml/ml_dtypes/issues/56
      raise ValueError(
          f"{_ai.ERR_PREFIX}ULP assertions are not currently supported for "
          "bfloat16."
      )
    np.testing.assert_array_max_ulp(
        _ai.jnp_to_np_array(arr_1),
        _ai.jnp_to_np_array(arr_2),
        maxulp=maxulp,
    )

  def cmp_fn(arr_1, arr_2) -> bool:
    try:
      # Raises an AssertionError if values are not close.
      assert_fn(arr_1, arr_2)
    except AssertionError:
      return False
    return True

  def err_msg_fn(arr_1, arr_2) -> str:
    try:
      assert_fn(arr_1, arr_2)
    except AssertionError as e:
      return (
          f"{str(e)} \nOriginal dtypes: "
          f"{np.asarray(arr_1).dtype}, {np.asarray(arr_2).dtype}"
      )
    return ""

  assert_trees_all_equal_comparator(cmp_fn, err_msg_fn, *trees)


# The return should be typing.NoReturn, but that would significantly complicate
# the signature of _value_assertion, so we pretend the return is jax.Array.
def _assert_trees_all_close_ulp_jittable(
    *trees: ArrayTree,
    maxulp: int = 1,
) -> jax.Array:
  """A dummy jittable version of `_assert_trees_all_close_ulp_static`.

  JAX does not yet have a native version of assert_array_max_ulp, so at the
  moment making ULP assertions on tracer objects simply isn't supported.
  This function exists only to make sure a sensible error is given.

  Args:
    *trees: Ignored.
    maxulp: Ignored.

  Raises:
    NotImplementedError: unconditionally.

  Returns:
    Never returns. (We pretend jax.Array to satisfy the type checker.)
  """
  del trees, maxulp
  raise NotImplementedError(
      f"{_ai.ERR_PREFIX}assert_trees_all_close_ulp is not supported within JIT "
      "contexts."
  )


assert_trees_all_close_ulp = _value_assertion(
    assert_fn=_assert_trees_all_close_ulp_static,
    jittable_assert_fn=_assert_trees_all_close_ulp_jittable,
    name="assert_trees_all_close_ulp",
)
