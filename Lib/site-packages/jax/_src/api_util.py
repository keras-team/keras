# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import inspect
import operator
from functools import partial, lru_cache
from typing import Any

from jax._src import core
from jax._src import config
from jax._src import dtypes
from jax._src.state.types import AbstractRef
from jax._src.tree_util import (
    PyTreeDef, tree_flatten, tree_unflatten, tree_map,
    treedef_children, generate_key_paths, keystr, broadcast_prefix,
    prefix_errors)
from jax._src.tree_util import _replace_nones
from jax._src import linear_util as lu
from jax._src.linear_util import TracingDebugInfo
from jax._src.util import (safe_map, WrapKwArgs, Hashable, HashableFunction,
                           Unhashable, safe_zip)
from jax._src import traceback_util
traceback_util.register_exclusion(__file__)

map = safe_map

def _ensure_index(x: Any) -> int | tuple[int, ...]:
  """Ensure x is either an index or a tuple of indices."""
  x = core.concrete_or_error(None, x, "expected a static index or sequence of indices.")
  try:
    return operator.index(x)
  except TypeError:
    return tuple(map(operator.index, x))

def _ensure_index_tuple(x: Any) -> tuple[int, ...]:
  """Convert x to a tuple of indices."""
  x = core.concrete_or_error(None, x, "expected a static index or sequence of indices.")
  try:
    return (operator.index(x),)
  except TypeError:
    return tuple(map(operator.index, x))

def _ensure_str(x: str) -> str:
  if not isinstance(x, str):
    raise TypeError(f"argument is not a string: {x}")
  return x

def _ensure_str_tuple(x: str | Iterable[str]) -> tuple[str, ...]:
  """Convert x to a tuple of strings."""
  if isinstance(x, str):
    return (x,)
  else:
    return tuple(map(_ensure_str, x))

@lu.transformation_with_aux2
def flatten_fun(f, store, in_tree, *args_flat):
  py_args, py_kwargs = tree_unflatten(in_tree, args_flat)
  ans = f(*py_args, **py_kwargs)
  ans, out_tree = tree_flatten(ans)
  store.store(out_tree)
  return ans

def apply_flat_fun(fun, io_tree, *py_args):
  in_tree_expected, out_tree = io_tree
  args, in_tree = tree_flatten((py_args, {}))
  if in_tree != in_tree_expected:
    raise TypeError(f"Expected {in_tree_expected}, got {in_tree}")
  ans = fun(*args)
  return tree_unflatten(out_tree, ans)

@lu.transformation_with_aux2
def flatten_fun_nokwargs(f, store, in_tree, *args_flat):
  py_args = tree_unflatten(in_tree, args_flat)
  ans = f(*py_args)
  ans, out_tree = tree_flatten(ans)
  store.store(out_tree)
  return ans

def apply_flat_fun_nokwargs(fun, io_tree, py_args):
  in_tree_expected, out_tree = io_tree
  args, in_tree = tree_flatten(py_args)
  if in_tree != in_tree_expected:
    raise TypeError(f"Expected {in_tree_expected}, got {in_tree}")
  ans = fun(*args)
  return tree_unflatten(out_tree, ans)

def flattened_fun_in_tree(
    fn: lu.WrappedFun
  ) -> tuple[PyTreeDef, Callable[[], PyTreeDef], bool] | None:
  # This implementation relies on internal details of linear_util.py's
  # WrappedFun, but it's for the worthy cause of better user error messages.
  # It can fail (i.e. return None) if its WrappedFun argument is not transformed
  # with flatten_fun or flatten_fun_nokwargs, which could happen e.g. when
  # core.eval_jaxpr encounters a call primitive (though at that point we're just
  # round-tripping jaxprs and the user errors in question are impossible).
  assert isinstance(flatten_fun, partial) and len(flatten_fun.args) == 1
  assert (isinstance(flatten_fun_nokwargs, partial) and
          len(flatten_fun_nokwargs.args) == 1)
  flattens = {flatten_fun.args[0], flatten_fun_nokwargs.args[0]}
  try:
    ((in_tree,), out_tree_store, has_kwargs), = (
        (args, store, f is flatten_fun.args[0])
        for (f, args), store in zip(fn.transforms, fn.stores) if f in flattens)
  except ValueError:
    return None
  else:
    return in_tree, lambda: out_tree_store.val, has_kwargs  # type: ignore[union-attr]

@lu.transformation_with_aux2
def flatten_fun_nokwargs2(f, store, in_tree, *args_flat):
  py_args = tree_unflatten(in_tree, args_flat)
  pair = f(*py_args)
  if not isinstance(pair, (list, tuple)) or len(pair) != 2:
    raise TypeError("expected function with aux output to return a two-element "
                    f"tuple, but got type {type(pair)} with value {pair!r}")
  ans, aux = pair
  ans_flat, ans_tree = tree_flatten(ans)
  aux_flat, aux_tree = tree_flatten(aux)
  store.store((ans_tree, aux_tree))
  return ans_flat, aux_flat

class _HashableWithStrictTypeEquality:
  """Box object used when comparing static arguments as a jit key.

  Requires exact type equality using `is` and value equality."""
  __slots__ = ["val"]

  def __init__(self, val):
    self.val = val

  def __hash__(self):
    return hash(self.val)

  def __eq__(self, other):
    return type(self.val) is type(other.val) and self.val == other.val

_POSITIONAL_ARGUMENTS = (
  inspect.Parameter.POSITIONAL_ONLY,
  inspect.Parameter.POSITIONAL_OR_KEYWORD
)

def _validate_argnums(sig: inspect.Signature, argnums: tuple[int, ...], argnums_name: str) -> None:
  """
  Validate that the argnums are sensible for a given function.

  For functions that accept a variable number of positions arguments
  (`f(..., *args)`) all positive argnums are considered valid.
  """
  n_pos_args = 0
  for param in sig.parameters.values():
    if param.kind in _POSITIONAL_ARGUMENTS:
      n_pos_args += 1

    elif param.kind is inspect.Parameter.VAR_POSITIONAL:
      # We can have any number of positional arguments
      return

  if argnums and (-min(argnums) > n_pos_args or max(argnums) >= n_pos_args):
    raise ValueError(f"Jitted function has {argnums_name}={argnums}, "
                     f"but only accepts {n_pos_args} positional arguments.")

_INVALID_KEYWORD_ARGUMENTS = (
  inspect.Parameter.POSITIONAL_ONLY,
  inspect.Parameter.VAR_POSITIONAL
)


_KEYWORD_ARGUMENTS = (
  inspect.Parameter.POSITIONAL_OR_KEYWORD,
  inspect.Parameter.KEYWORD_ONLY,
)
def _validate_argnames(
    sig: inspect.Signature, argnames: tuple[str, ...], argnames_name: str
) -> None:
  """
  Validate that the argnames are sensible for a given function.

  For functions that accept a variable keyword arguments
  (`f(..., **kwargs)`) all argnames are considered valid except those
  marked as position-only (`f(pos_only, /, ...)`).
  """
  var_kwargs = False
  valid_kwargs: set[str] = set()
  invalid_kwargs: set[str] = set()
  for param_name, param in sig.parameters.items():
    if param.kind in _KEYWORD_ARGUMENTS:
      valid_kwargs.add(param_name)

    elif param.kind is inspect.Parameter.VAR_KEYWORD:
      var_kwargs = True

    elif param.kind in _INVALID_KEYWORD_ARGUMENTS:
      invalid_kwargs.add(param_name)

  # Check whether any kwargs are invalid due to position only
  if invalid_argnames := (invalid_kwargs & set(argnames)):
    raise ValueError(f"Jitted function has invalid argnames {invalid_argnames} "
                     f"in {argnames_name}. These are positional-only")

  # Takes any kwargs
  if var_kwargs:
    return

  # Check that all argnames exist on function
  if invalid_argnames := (set(argnames) - valid_kwargs):
    raise ValueError(f"Jitted function has invalid argnames {invalid_argnames} "
                     f"in {argnames_name}. Function does not take these args.")


def argnums_partial(f, dyn_argnums, args, require_static_args_hashable=True):
  dyn_argnums = _ensure_index_tuple(dyn_argnums)
  dyn_argnums = _ensure_inbounds(False, len(args), dyn_argnums)
  if require_static_args_hashable:
    fixed_args = []
    for i, arg in enumerate(args):
      if i in dyn_argnums: continue
      if not is_hashable(arg):
        raise ValueError(
            "Non-hashable static arguments are not supported, as this can lead "
            f"to unexpected cache-misses. Static argument (index {i}) of type "
            f"{type(arg)} for function {f.__name__} is non-hashable.")
      fixed_args.append(_HashableWithStrictTypeEquality(arg))
  else:
    fixed_args = [Unhashable(arg) for i, arg in enumerate(args)
                  if i not in dyn_argnums]
  dyn_args = tuple(args[i] for i in dyn_argnums)
  return _argnums_partial(f, dyn_argnums, tuple(fixed_args)), dyn_args

def _ensure_inbounds(allow_invalid: bool, num_args: int, argnums: Sequence[int]
                     ) -> tuple[int, ...]:
  """Ensure argnum is within bounds. Also resolves negative argnums."""
  result = []
  for i in argnums:
    if i >= num_args and allow_invalid: continue
    if not -num_args <= i < num_args:
      raise ValueError(
          "Positional argument indices, e.g. for `static_argnums`, must have "
          "value greater than or equal to -len(args) and less than len(args), "
          f"but got value {i} for len(args) == {num_args}.")
    result.append(i % num_args)  # Resolve negative
  return tuple(result)


def argnums_partial_except(f: lu.WrappedFun, static_argnums: tuple[int, ...],
                           args: tuple[Any, ...], *, allow_invalid: bool):
  "Version of ``argnums_partial`` that checks hashability of static_argnums."
  if not static_argnums:
    return f, args
  static_argnums = _ensure_inbounds(allow_invalid, len(args), static_argnums)
  dyn_argnums = tuple(i for i in range(len(args)) if i not in static_argnums)
  dyn_args = tuple(args[i] for i in dyn_argnums)

  fixed_args = []
  for i in static_argnums:
    # TODO(shoyer): set allow_invalid=True permanently after static_argnames.
    if allow_invalid and i >= len(args):
      continue
    static_arg = args[i]
    if not is_hashable(static_arg):
      raise ValueError(
          "Non-hashable static arguments are not supported, as this can lead "
          f"to unexpected cache-misses. Static argument (index {i}) of type "
          f"{type(static_arg)} for function {f.__name__} is non-hashable.")
    else:
      fixed_args.append(_HashableWithStrictTypeEquality(static_arg))

  return _argnums_partial(f, dyn_argnums, tuple(fixed_args)), dyn_args

@lu.transformation2
def _argnums_partial(_fun, _dyn_argnums, _fixed_args, *dyn_args, **kwargs):
  sentinel = object()
  args = [sentinel] * (len(_fixed_args) + len(dyn_args))
  for i, arg in zip(_dyn_argnums, dyn_args):
    args[i] = arg
  fixed_args_ = iter(_fixed_args)
  args = [next(fixed_args_).val if x is sentinel else x for x in args]
  assert next(fixed_args_, sentinel) is sentinel
  return _fun(*args, **kwargs)

def argnames_partial_except(f: lu.WrappedFun, static_argnames: tuple[str, ...],
                            kwargs: dict[str, Any]):
  if not static_argnames:
    return f, kwargs
  dyn_kwargs = {k: v for k, v in kwargs.items() if k not in static_argnames}

  fixed_kwargs: dict[str, Any] = {}
  for k, arg in kwargs.items():
    if k not in dyn_kwargs:
      try:
        hash(arg)
      except TypeError:
        raise ValueError(
            "Non-hashable static arguments are not supported, as this can lead "
            f"to unexpected cache-misses. Static argument (name {k}) of type "
            f"{type(arg)} for function {f.__name__} is non-hashable.")
      else:
        fixed_kwargs[k] = Hashable(arg)

  return _argnames_partial(f, WrapKwArgs(fixed_kwargs)), dyn_kwargs

@lu.transformation2
def _argnames_partial(_fun, _fixed_kwargs: WrapKwArgs, *args, **dyn_kwargs):
  kwargs = dict({k: v.val for k, v in _fixed_kwargs.val.items()}, **dyn_kwargs)
  return _fun(*args, **kwargs)


@lru_cache(maxsize=4096)
def donation_vector(donate_argnums, donate_argnames, in_tree,
                    kws: bool = True) -> tuple[bool, ...]:
  """Returns a tuple with a boolean value for each leaf in args and kwargs.

  What if a user specifies donate_argnums but calls the function with kwargs
  or vice-versa? In that case, in `resolve_argnums` using the signature of the
  function, the counterpart (donate_argnames or donate_argnums respectively) is
  calculated so when this function is called both donate_argnums and
  donate_argnames are available. This allows JAX to donate kwargs when only
  donate_argnums is specified and vice-versa.

  When both donate_argnums and donate_argnames are specified, only the args and
  kwargs specified are donated.
  """
  res: list[bool] = []
  if kws:
    args_tree, kwargs_tree = treedef_children(in_tree)
  else:
    args_tree, kwargs_tree = in_tree, None
  for i, arg in enumerate(args_tree.children()):
    donate = bool(i in donate_argnums)
    res.extend((donate,) * arg.num_leaves)
  if kwargs_tree is not None:
    for key, val in safe_zip(kwargs_tree.node_data()[1], kwargs_tree.children()):  # type: ignore
      donate = key in donate_argnames
      res.extend((donate,) * val.num_leaves)
  return tuple(res)

def rebase_donate_argnums(donate_argnums, static_argnums) -> tuple[int, ...]:
  """Shifts donate to account for static.

  >>> rebase_donate_argnums((3, 4), (0, 1))
  (1, 2)

  Args:
    donate_argnums: An iterable of ints.
    static_argnums: An iterable of ints.

  Returns:
    A tuple of unique, sorted integer values based on donate_argnums with each
    element offset to account for static_argnums.
  """
  if not (static_argnums or donate_argnums):
    return tuple(sorted(donate_argnums))

  static_argnums = sorted(set(static_argnums))
  donate_argnums = sorted(set(donate_argnums))
  i = j = o = 0
  out = []
  while j < len(donate_argnums):
    if i < len(static_argnums) and static_argnums[i] == donate_argnums[j]:
      raise ValueError(f"`static_argnums` {static_argnums} and "
                       f"`donate_argnums` {donate_argnums} cannot intersect.")

    if i < len(static_argnums) and static_argnums[i] < donate_argnums[j]:
      o += 1
      i += 1
    else:
      out.append(donate_argnums[j] - o)
      j += 1
  return tuple(out)


def is_hashable(arg):
  try:
    hash(arg)
    return True
  except TypeError:
    return False


def flatten_axes(name, treedef, axis_tree, *, kws=False, tupled_args=False):
  # given an axis spec tree axis_tree (a pytree with integers and Nones at the
  # leaves, i.e. the Nones are to be considered leaves) that is a tree prefix of
  # the given treedef, build a complete axis spec tree with the same structure
  # and return the flattened result
  # TODO(mattjj,phawkins): improve this implementation
  proxy = object()
  dummy = tree_unflatten(treedef, [object()] * treedef.num_leaves)
  axes = []
  add_leaves = lambda i, x: axes.extend([i] * len(tree_flatten(x)[0]))
  try:
    tree_map(add_leaves, _replace_nones(proxy, axis_tree), dummy)
  except ValueError:
    if kws:
      # if keyword arguments are included in the tree, we make adapt the error
      # message only to be about the positional arguments
      treedef, _ = treedef_children(treedef)
      axis_tree, _ = axis_tree
    hint = ""
    if tupled_args:
      hint += (f" Note that {name} that are non-trivial pytrees should always be "
               f"wrapped in a tuple representing the argument list.")
      if len(treedef.children()) == 1:
        try:
          flatten_axes(name, treedef, (axis_tree,))
        except ValueError:
          pass  # That's not the issue.
        else:
          hint += (f" In particular, you're passing in a single argument which "
                   f"means that {name} might need to be wrapped in "
                   f"a singleton tuple.")
    raise ValueError(f"{name} specification must be a tree prefix of the "
                     f"corresponding value, got specification {axis_tree} "
                     f"for value tree {treedef}.{hint}") from None
  axes = [None if a is proxy else a for a in axes]
  assert len(axes) == treedef.num_leaves
  return axes

def flat_out_axes(
    f: lu.WrappedFun, out_spec: Any
) -> tuple[lu.WrappedFun, Callable]:
  leaves, treedef = tree_flatten(out_spec)
  f, out_axes = _flat_out_axes(f, tuple(leaves), treedef)
  return f, HashableFunction(out_axes, closure=(tuple(leaves), treedef))

@lu.transformation_with_aux2
def _flat_out_axes(_fun, _store, _leaves, _treedef, *args, **kwargs):
  ans = _fun(*args, **kwargs)
  spec = tree_unflatten(_treedef, _leaves)
  try:
    spec_flat = tuple(broadcast_prefix(spec, ans, is_leaf=lambda x: x is None))
  except ValueError:
    e, *_ = prefix_errors(spec, ans)
    # TODO(mattjj): currently hardcoded for pmap; generalize to vmap in followup
    msg, = e('pmap out_axes').args
    msg += ("\n\nThe full pytree is the output of the pmapped function. Ensure "
            "that the `out_axes` argument to `pmap` is a pytree prefix of the "
            "pmapped function's output.")
    raise ValueError(msg) from None
  _store.store(spec_flat)
  return ans

def check_callable(fun):
  # In Python 3.10+, the only thing stopping us from supporting staticmethods
  # is that we can't take weak references to them, which the C++ JIT requires.
  if isinstance(fun, staticmethod):
    raise TypeError(f"staticmethod arguments are not supported, got {fun}")
  if not callable(fun):
    raise TypeError(f"Expected a callable value, got {fun}")
  if inspect.isgeneratorfunction(fun):
    raise TypeError(f"Expected a function, got a generator function: {fun}")

_POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD

def infer_argnums_and_argnames(
    sig: inspect.Signature,
    argnums: int | Iterable[int] | None,
    argnames: str | Iterable[str] | None,
  ) -> tuple[tuple[int, ...], tuple[str, ...]]:
  """Infer missing argnums and argnames for a function with inspect."""
  if argnums is None and argnames is None:
    return (), ()

  if argnums is not None and argnames is not None:
    argnums = _ensure_index_tuple(argnums)
    argnames = _ensure_str_tuple(argnames)
    return argnums, argnames

  parameters = sig.parameters
  if argnums is None:
    assert argnames is not None
    argnames = _ensure_str_tuple(argnames)
    argnums = tuple(
        i for i, (k, param) in enumerate(parameters.items())
        if param.kind == _POSITIONAL_OR_KEYWORD and k in argnames
    )
  else:
    argnums = _ensure_index_tuple(argnums)
    argnames = tuple(
        k for i, (k, param) in enumerate(parameters.items())
        if param.kind == _POSITIONAL_OR_KEYWORD and i in argnums
    )

  return argnums, argnames


def resolve_argnums(
    fun: Callable,
    signature: inspect.Signature | None,
    donate_argnums: int | Sequence[int] | None,
    donate_argnames: str | Iterable[str] | None,
    static_argnums: int | Sequence[int] | None,
    static_argnames: str | Iterable[str] | None,
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
  """Validates and completes the argnum/argname specification for a jit.

  * fills in any missing pieces (e.g., names given numbers, or vice versa),
  * validates the argument names/numbers against the function signature,
  * validates that donated and static arguments don't intersect.
  * rebases the donated arguments so they index into the dynamic arguments,
    (after static arguments have been removed), in the order that parameters
    are passed into the compiled function.
  """
  if signature is None:
    # Some built-in functions don't support signature.
    # See: https://github.com/python/cpython/issues/73485
    # In this case no validation is done
    static_argnums = () if static_argnums is None else _ensure_index_tuple(
        static_argnums)
    static_argnames = () if static_argnames is None else _ensure_str_tuple(
        static_argnames)
    donate_argnums = () if donate_argnums is None else _ensure_index_tuple(
        donate_argnums)
    if donate_argnames is not None:
      raise ValueError(f"Getting the signature of function {fun} failed. "
                       "Pass donate_argnums instead of donate_argnames.")
    assert donate_argnames is None
    donate_argnames = ()
  else:
    # Infer argnums and argnames according to docstring
    # If nums is None and names is not None, then nums are inferred from the
    # names and vice-versa.
    static_argnums, static_argnames = infer_argnums_and_argnames(
        signature, static_argnums, static_argnames)
    donate_argnums, donate_argnames = infer_argnums_and_argnames(
        signature, donate_argnums, donate_argnames)

    # Validation
    _validate_argnums(signature, static_argnums, "static_argnums")
    _validate_argnames(signature, static_argnames, "static_argnames")
    _validate_argnums(signature, donate_argnums, "donate_argnums")
    _validate_argnames(signature, donate_argnames, "donate_argnames")

  # Compensate for static argnums absorbing args
  _assert_no_intersection(static_argnames, donate_argnames)
  donate_argnums = rebase_donate_argnums(donate_argnums, static_argnums)
  return donate_argnums, donate_argnames, static_argnums, static_argnames


def _assert_no_intersection(static_argnames, donate_argnames):
  out = set(static_argnames).intersection(set(donate_argnames))
  if out:
    raise ValueError(
        "static_argnames and donate_argnames cannot intersect. Argument names "
        f"{out} appear in both static_argnames and donate_argnames")


def resolve_kwargs(fun: Callable, args, kwargs) -> tuple[Any, ...]:
  """Resolve input arguments to positional following a function's signature.

  This will raise a TypeError if any keyword-only arguments were passed by the
  caller.
  """
  if isinstance(fun, partial):
    # functools.partial should have an opaque signature.
    fun = lambda *args, **kwargs: None
  ba = inspect.signature(fun).bind(*args, **kwargs)
  ba.apply_defaults()
  if ba.kwargs:
    passed_kwargs = [k for k in ba.kwargs if k in kwargs]
    if passed_kwargs:
      raise TypeError(
          f"keyword arguments ({passed_kwargs}) could not be resolved to "
          "positions")
  return ba.args


def _dtype(x):
  try:
    return dtypes.result_type(x)
  except ValueError:
    return dtypes.result_type(getattr(x, 'dtype'))


# This decorator exists to make it easier to monkey-patch APIs in JAX.
# By default it does nothing, but it can be monkey-patched to do other things.
def api_hook(fun, tag: str):
  return fun


def debug_info(
    traced_for: str, fun_src_info: str | None,
    fun_signature: inspect.Signature | None,
    args: tuple[Any, ...], kwargs: dict[str, Any],
    static_argnums: tuple[int, ...],
    static_argnames: tuple[str, ...]
) -> TracingDebugInfo | None:
  """Try to build trace-time debug info for fun when applied to args/kwargs."""
  arg_names = _arg_names(fun_signature, args, kwargs, static_argnums,
                         static_argnames)
  if arg_names is None:
    return None
  return TracingDebugInfo(traced_for, fun_src_info, arg_names, None)

def fun_signature(fun: Callable) -> inspect.Signature | None:
  try:
    return inspect.signature(fun)
  except (ValueError, TypeError):
    return None

def save_wrapped_fun_sourceinfo(wrapper: Callable, wrapped: Callable):
  # Prefer this to functools.wraps because it does not create a reference to
  # the wrapped function.
  sourceinfo = fun_sourceinfo(wrapped)
  if sourceinfo is not None:
    setattr(wrapper, "__fun_sourceinfo__", fun_sourceinfo(wrapped))

# TODO(mattjj): make this function internal to this module
def fun_sourceinfo(fun: Callable) -> str | None:
  res = getattr(fun, "__fun_sourceinfo__", None)
  if res is not None: return res
  while isinstance(fun, partial):
    fun = fun.func
  fun = inspect.unwrap(fun)
  try:
    filename = fun.__code__.co_filename
    lineno = fun.__code__.co_firstlineno
    return f"{fun.__name__} at {filename}:{lineno}"
  except AttributeError:
    return None

def _arg_names(fn_signature, args, kwargs, static_argnums, static_argnames,
               ) -> tuple[str, ...] | None:
  if fn_signature is None: return None
  static = object()
  static_argnums_ = _ensure_inbounds(True, len(args), static_argnums)
  static_argnames_ = set(static_argnames)
  args_ = [static if i in static_argnums_ else x for i, x in enumerate(args)]
  kwargs = {k:static if k in static_argnames_ else x for k, x in kwargs.items()}
  try:
    ba = fn_signature.bind(*args_, **kwargs)
  except (ValueError, TypeError):
    return None
  return tuple(f'{name}{keystr(path)}' for name, x in ba.arguments.items()
               for path, l in generate_key_paths(x) if l is not static)

@lu.transformation_with_aux2
def result_paths(_fun, _store, *args, **kwargs):
  "linear_util transform to get output pytree paths of pre-flattened function."
  ans = _fun(*args, **kwargs)
  _store.store([keystr(path) for path, _ in generate_key_paths(ans)])
  return ans

def add_jaxpr_debug_info(jaxpr: core.Jaxpr,
                         trace_debug: TracingDebugInfo | None,
                         result_paths: tuple[str, ...] | None = None,
                         ) -> core.Jaxpr:
  """Add debug info to jaxpr, given trace-time debug info and result paths."""
  if trace_debug is None:
    return jaxpr
  assert (result_paths is not None) ^ (trace_debug.result_paths_thunk is not None)
  if result_paths is None:
    result_paths = trace_debug.result_paths_thunk()  # type: ignore
  debug_info = core.JaxprDebugInfo(
      trace_debug.traced_for, trace_debug.func_src_info,
      trace_debug.arg_names, tuple(result_paths))
  return jaxpr.replace(debug_info=debug_info)

def debug_info_final(f: lu.WrappedFun, dbg: TracingDebugInfo | None,
                     res_paths_thunk: Callable[[], tuple[str, ...]]
                     ) -> lu.WrappedFun:
  "Attach trace-time debug info and result paths lazy thunk to an lu.WrappedFun"
  if dbg is None: return f
  assert dbg.result_paths_thunk is None
  res_paths_thunk_ = HashableFunction(res_paths_thunk, closure=())
  return lu.add_debug_info(f, dbg._replace(result_paths_thunk=res_paths_thunk_))


def hoist_obj_attrs(f, flat_args):
  idxs, objs, flat_args_ = [], [], []
  for i, x in enumerate(flat_args):
    if type(x) in _class_with_attrs:
      objs.append(_HashableByObjectId(x))
    else:
      idxs.append(i)
      flat_args_.append(x)
  return _argnums_partial(f, tuple(idxs), tuple(objs)), flat_args_

class _HashableByObjectId:
  __slots__ = ['val']
  def __init__(self, val):
    self.val = val
  def __hash__(self):
    return id(self.val)
  def __eq__(self, other):
    return self.val is other.val

def register_class_with_attrs(t: type) -> None:
  _class_with_attrs.add(t)
_class_with_attrs: set[type] = set()

# TODO(mattjj): make this function faster
def _check_no_aliased_ref_args(dbg, avals, args):
  assert config.mutable_array_checks.value
  refs: dict[int, int] = {}
  for i, (a, x) in enumerate(zip(avals, args)):
    if (isinstance(a, AbstractRef) and
        (dup_idx := refs.setdefault(id(core.get_referent(x)), i)) != i):
      raise ValueError(
        "only one reference to a mutable array may be passed as an argument "
        f"to a function, but when tracing {dbg.func_src_info} for {dbg.traced_for} "
        f"the mutable array reference of type {a.str_short()} appeared at both "
        f"{dbg.arg_names[dup_idx]} and {dbg.arg_names[i]}."
        if dbg else
        f"at both flat index {dup_idx} and flat index {i}") from None

def _check_no_aliased_closed_over_refs(dbg, consts, args) -> None:
  assert config.mutable_array_checks.value
  refs: set[int] = {id(core.get_referent(c)) for c in consts
                    if isinstance(core.get_aval(c), AbstractRef)}
  for i, x in enumerate(args):
    if id(core.get_referent(x)) in refs:
      a = core.shaped_abstractify(x)
      raise ValueError(
          f"when tracing {dbg.func_src_info} for {dbg.traced_for}, a mutable "
          f"array reference of type {a.str_short()} was both closed over and "
          f"passed as the argument "
          f"{dbg.arg_names[i]}" if dbg else "at flat index {i}")
