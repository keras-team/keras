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
"""Chex assertion internal utilities and symbols.

[README!]

We reserve the right to change the code in this module at any time without
providing any guarantees of backward compatibility. For this reason,
we strongly recommend that you avoid using this module directly at all costs!
Instead, consider opening an issue on GitHub and describing your use case.
"""

import collections
import collections.abc
from collections.abc import Hashable
import functools
import re
import threading
import traceback
from typing import Any, Sequence, Union, Callable, List, Optional, Set, Tuple, Type

from absl import logging
from chex._src import pytypes
import jax
from jax.experimental import checkify
import jax.numpy as jnp
import numpy as np

# Custom pytypes.
TLeaf = Any
TLeavesEqCmpFn = Callable[[TLeaf, TLeaf], bool]
TLeavesEqCmpErrorFn = Callable[[TLeaf, TLeaf], str]

# TODO(iukemaev): define a typing protocol for TChexAssertion.
# Chex assertion signature:
# (*args,
#  custom_message: Optional[str] = None,
#  custom_message_format_vars: Sequence[Any] = (),
#  include_default_message: bool = True,
#  exception_type: Type[Exception] = AssertionError,
#  **kwargs)
TChexAssertion = Callable[..., None]
TAssertFn = Callable[..., None]
TJittableAssertFn = Callable[..., pytypes.Array]  # a predicate function

# Matchers.
TDimMatcher = Optional[Union[int, Set[int], type(Ellipsis)]]
TShapeMatcher = Sequence[TDimMatcher]


class _ChexifyStorage(threading.local):
  """Thread-safe storage for internal variables used in @chexify."""
  wait_fns = []
  level = 0


# Chex namespace variables.
ERR_PREFIX = "[Chex] "
TRACE_COUNTER = collections.Counter()
DISABLE_ASSERTIONS = False

# This variable is used for _chexify_ transformations, see `asserts_chexify.py`.
CHEXIFY_STORAGE = _ChexifyStorage()


def assert_collection_of_arrays(inputs: Sequence[pytypes.Array]):
  """Checks if ``inputs`` is a collection of arrays."""
  if not isinstance(inputs, collections.abc.Collection):
    raise ValueError(f"`inputs` is not a collection of arrays: {inputs}.")


def jnp_to_np_array(arr: pytypes.Array) -> np.ndarray:
  """Converts `jnp.ndarray` to `np.ndarray`."""
  if getattr(arr, "dtype", None) == jnp.bfloat16:
    # Numpy does not support `bfloat16`.
    arr = arr.astype(jnp.float32)
  return jax.device_get(arr)


def deprecation_wrapper(new_fn, old_name, new_name):
  """Allows deprecated functions to continue running, with a warning logged."""

  def inner_fn(*args, **kwargs):
    logging.warning(
        "chex.%s has been renamed to chex.%s, please update your code.",
        old_name, new_name)
    return new_fn(*args, **kwargs)

  return inner_fn


def get_stacktrace_without_chex_internals() -> List[traceback.FrameSummary]:
  """Returns the latest non-chex frame from the call stack."""
  stacktrace = list(traceback.extract_stack())
  for i in reversed(range(len(stacktrace))):
    fname = stacktrace[i].filename
    if fname.find("/chex/") == -1 or fname.endswith("_test.py"):
      return stacktrace[:i+1]

  debug_info = "\n-----\n".join(traceback.format_stack())
  raise RuntimeError(
      "get_stacktrace_without_chex_internals() failed. "
      "Please file a bug at https://github.com/deepmind/chex/issues and "
      "include the following debug info in it. "
      "Please make sure it does not include any private information! "
      f"Debug: '{debug_info}'.")


def get_err_regex(message: str) -> str:
  """Constructs a regexp for the exception message.

  Args:
    message: an exception message.

  Returns:
    Regexp that ensures the message follows the standard chex formatting.
  """
  # (ERR_PREFIX + any symbols (incl. \n) + message)
  return f"{re.escape(ERR_PREFIX)}[\\s\\S]*{message}"


def get_chexify_err_message(name: str, msg: str = "") -> str:
  """Constructs an error message for the chexify exception."""
  return f"{ERR_PREFIX}chexify assertion '{name}' failed: {msg}"


def _make_host_assertion(assert_fn: TAssertFn,
                         name: Optional[str] = None) -> TChexAssertion:
  """Constructs a host assertion given `assert_fn`.

  This wrapper should only be applied to the assertions that are either
    a) never used in jitted code, or
    b) when used in jitted code they do not check/access tensor values (i.e.
       they do not introduce value-dependent python control flow, see
       https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError).

  Args:
    assert_fn: A function implementing the check.
    name: A name for assertion.

  Returns:
    A chex assertion.
  """
  if name is None:
    name = assert_fn.__name__

  def _assert_on_host(*args,
                      custom_message: Optional[str] = None,
                      custom_message_format_vars: Sequence[Any] = (),
                      include_default_message: bool = True,
                      exception_type: Type[Exception] = AssertionError,
                      **kwargs) -> None:
    # Format error's stack trace to remove Chex' internal frames.
    assertion_exc = None
    value_exc = None
    try:
      assert_fn(*args, **kwargs)
    except AssertionError as e:
      assertion_exc = e
    except ValueError as e:
      value_exc = e
    finally:
      if value_exc is not None:
        raise ValueError(str(value_exc))

      if assertion_exc is not None:
        # Format the exception message.
        error_msg = str(assertion_exc)

        # Include only the name of the outermost chex assertion.
        if error_msg.startswith(ERR_PREFIX):
          error_msg = error_msg[error_msg.find("failed:") + len("failed:"):]

        # Whether to include the default error message.
        default_msg = (f"Assertion {name} failed: "
                       if include_default_message else "")
        error_msg = f"{ERR_PREFIX}{default_msg}{error_msg}"

        # Whether to include a custom error message.
        if custom_message:
          if custom_message_format_vars:
            custom_message = custom_message.format(*custom_message_format_vars)
          error_msg = f"{error_msg} [{custom_message}]"

        raise exception_type(error_msg)

  return _assert_on_host


def chex_assertion(
    assert_fn: TAssertFn,
    jittable_assert_fn: Optional[TJittableAssertFn],
    name: Optional[str] = None) -> TChexAssertion:
  """Wraps Chex assert functions to control their common behaviour.

  Extends the assertion to support the following optional auxiliary kwargs:
    custom_message: A string to include into the emitted exception messages.
    custom_message_format_vars: A list of variables to pass as arguments to
      `custom_message.format()`.
    include_default_message: Whether to include the default Chex message into
      the emitted exception messages.
    exception_type: An exception type to use. `AssertionError` by default.

  Args:
    assert_fn: A function implementing the check.
    jittable_assert_fn: An optional jittable version of `assert_fn` implementing
      a predicate (returning `True` only if assertion passes).
      Required for value assertions.
    name: A name for assertion. If not provided, use `assert_fn.__name__`.

  Returns:
    A Chex assertion (with auxiliary kwargs).
  """
  if name is None:
    name = assert_fn.__name__

  host_assertion_fn = _make_host_assertion(assert_fn, name)

  @functools.wraps(assert_fn)
  def _chex_assert_fn(*args,
                      custom_message: Optional[str] = None,
                      custom_message_format_vars: Sequence[Any] = (),
                      include_default_message: bool = True,
                      exception_type: Type[Exception] = AssertionError,
                      **kwargs) -> None:
    if DISABLE_ASSERTIONS:
      return
    if (jittable_assert_fn is not None and has_tracers((args, kwargs))):
      if not CHEXIFY_STORAGE.level:
        raise RuntimeError(
            "Value assertions can only be called from functions wrapped "
            "with `@chex.chexify`. See the docs.")

      # A wrapped to inject auxiliary debug info and `custom_message`.
      original_check = checkify.check

      def _check(pred, msg, *fmt_args, **fmt_kwargs):
        # Add chex info.
        msg = get_chexify_err_message(name, msg)

        # Add a custom message.
        if custom_message:
          msg += f" Custom message: {custom_message}."
          fmt_args = list(fmt_args) + list(custom_message_format_vars)

        # Add a traceback and a pointer to the callsite.
        stacktrace = get_stacktrace_without_chex_internals()
        msg += (
            f" [failed at: {stacktrace[-1].filename}:{stacktrace[-1].lineno}]"
        )

        # Call original `checkify.check()`.
        original_check(pred, msg, *fmt_args, **fmt_kwargs)

      # Mock during the assertion's execution time.
      checkify.check = _check
      pred = jittable_assert_fn(*args, **kwargs)  # execute the assertion
      checkify.check = original_check  # return the original implementation

      # A safeguard to ensure that the results of a check are not ignored.
      # In particular, this check fails when `pred` is False and no
      # `checkify.check` calls took place in `jittable_assert_fn`, which would
      # be a bug in the assertion's implementation.
      checkify.check(pred, "assertion failed!")
    else:
      try:
        host_assertion_fn(
            *args,
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
            **kwargs)
      except jax.errors.ConcretizationTypeError as exc:
        msg = ("Chex assertion detected `ConcretizationTypeError`: it is very "
               "likely that it tried to access tensors' values during tracing. "
               "Make sure that you defined a jittable version of this chex "
               "assertion; if that does not help, please file a bug.")
        raise exc from RuntimeError(msg)

  # Override name.
  setattr(_chex_assert_fn, "__name__", name)
  return _chex_assert_fn


def format_tree_path(path: Sequence[Any]) -> str:
  return "/".join(str(p) for p in path)


def format_shape_matcher(shape: TShapeMatcher) -> str:
  return f"({', '.join('...' if d is Ellipsis else str(d) for d in shape)})"  # pylint: disable=inconsistent-quotes


def num_devices_available(devtype: str, backend: Optional[str] = None) -> int:
  """Returns the number of available device of the given type."""
  devtype = devtype.lower()
  supported_types = ("cpu", "gpu", "tpu")
  if devtype not in supported_types:
    raise ValueError(
        f"Unknown device type '{devtype}' (expected one of {supported_types}).")

  return sum(d.platform == devtype for d in jax.devices(backend))


def get_tracers(tree: pytypes.ArrayTree) -> Tuple[jax.core.Tracer]:
  """Returns a tuple with tracers from a tree."""
  return tuple(
      x for x in jax.tree_util.tree_leaves(tree)
      if isinstance(x, jax.core.Tracer))


def has_tracers(tree: pytypes.ArrayTree) -> bool:
  """Checks whether a tree contains any tracers."""
  return any(
      isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(tree))


def is_traceable(fn) -> bool:
  """Checks if function is traceable.

  JAX traces a function when it is wrapped with @jit, @pmap, or @vmap.
  In other words, this function checks whether `fn` is wrapped with any of
  the aforementioned JAX transformations.

  Args:
    fn: function to assert.

  Returns:
    Bool indicating whether fn is traceable.
  """

  fn_string_tokens = (
      ".reraise_with_filtered_traceback",  # JIT    in Python ver. >= 3.7
      "CompiledFunction",  # C++ JIT in jaxlib 0.1.66 or newer.
      "pmap.",  # Python pmap
      "PmapFunction",  # C++ pmap in jaxlib 0.1.72 or newer.
      "vmap.",  # vmap
      "_python_pjit",
      "_cpp_pjit",
  )

  fn_type_tokens = (
      "PmapFunction",
      "PjitFunction",
  )

  # Un-wrap `fn` and check if any internal fn is jitted by pattern matching.
  fn_ = fn
  while True:
    if any(t in str(fn_) for t in fn_string_tokens):
      return True

    if any(t in str(type(fn_)) for t in fn_type_tokens):
      return True

    if hasattr(fn_, "__wrapped__"):
      # Wrapper.
      fn_globals = getattr(fn_, "__globals__", {})

      if fn_globals.get("__name__", None) == "jax.api":
        # Wrapper from `jax.api`.
        return True

      if "api_boundary" in fn_globals:
        # api_boundary is a JAX wrapper for traced functions.
        return True

      try:
        if isinstance(fn_, jax.lib.xla_extension.PjitFunction):
          return True
      except AttributeError:
        pass
    else:
      break

    fn_ = fn_.__wrapped__
  return False


def assert_leaves_all_eq_comparator(
    equality_comparator: TLeavesEqCmpFn,
    error_msg_fn: Callable[[TLeaf, TLeaf, str, int, int],
                           str], path: Sequence[Any], *leaves: Sequence[TLeaf]):
  """Asserts all leaves are equal using custom comparator. Not jittable."""
  path_str = format_tree_path(path)
  for i in range(1, len(leaves)):
    if not equality_comparator(leaves[0], leaves[i]):
      raise AssertionError(error_msg_fn(leaves[0], leaves[i], path_str, 0, i))


def assert_trees_all_eq_comparator_jittable(
    equality_comparator: TLeavesEqCmpFn,
    error_msg_template: str,
    *trees: Sequence[pytypes.ArrayTree]) -> pytypes.Array:
  """Asserts all trees are equal using custom comparator. JIT-friendly."""

  if len(trees) < 2:
    raise ValueError(
        "Assertions over only one tree does not make sense. Maybe you wrote "
        "`assert_trees_xxx([a, b])` instead of `assert_trees_xxx(a, b)`, or "
        "forgot the `error_msg_fn` arg to `assert_trees_xxx`?")

  def _tree_error_msg_fn(
      path: Tuple[Union[int, str, Hashable]], i_1: int, i_2: int):
    if path:
      return (
          f"Trees {i_1} and {i_2} differ in leaves '{path}':"
          f" {error_msg_template}"
      )
    else:
      return f"Trees (arrays) {i_1} and {i_2} differ: {error_msg_template}."

  def _cmp_leaves(path, *leaves):
    verdict = jnp.array(True)
    for i in range(1, len(leaves)):
      check_res = equality_comparator(leaves[0], leaves[i])
      checkify.check(
          pred=check_res,
          msg=_tree_error_msg_fn(path, 0, i),
          arr_1=leaves[0],
          arr_2=leaves[i],
      )
      verdict = jnp.logical_and(verdict, check_res)
    return verdict

  # Trees are guaranteed to have the same structure.
  paths = [
      convert_jax_path_to_dm_path(path)
      for path, _ in jax.tree_util.tree_flatten_with_path(trees[0])[0]]
  trees_leaves = [jax.tree_util.tree_leaves(tree) for tree in trees]

  verdict = jnp.array(True)
  for leaf_i, path in enumerate(paths):
    verdict = jnp.logical_and(
        verdict, _cmp_leaves(path, *[leaves[leaf_i] for leaves in trees_leaves])
    )

  return verdict


JaxKeyType = Union[
    int,
    str,
    Hashable,
    jax.tree_util.SequenceKey,
    jax.tree_util.DictKey,
    jax.tree_util.FlattenedIndexKey,
    jax.tree_util.GetAttrKey,
]


def convert_jax_path_to_dm_path(
    jax_tree_path: Sequence[JaxKeyType],
) -> Tuple[Union[int, str, Hashable]]:
  """Converts a path from jax.tree_util to one from dm-tree."""

  # pytype:disable=attribute-error
  def _convert_key_fn(key: JaxKeyType) -> Union[int, str, Hashable]:
    if isinstance(key, (str, int)):
      return key  # int | str.
    if isinstance(key, jax.tree_util.SequenceKey):
      return key.idx  # int.
    if isinstance(key, jax.tree_util.DictKey):
      return key.key  # Hashable
    if isinstance(key, jax.tree_util.FlattenedIndexKey):
      return key.key  # int.
    if isinstance(key, jax.tree_util.GetAttrKey):
      return key.name  # str.
    raise ValueError(f"Jax tree key '{key}' of type '{type(key)}' not valid.")
  # pytype:enable=attribute-error

  return tuple(_convert_key_fn(key) for key in jax_tree_path)
