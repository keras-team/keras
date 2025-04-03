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

"""
Utilities for defining functions composed with transformations.

For example,

   from jax._src import linear_util as lu

   wf = lu.wrap_init(f)  # Produce a WrappedFun for applying transformations on `f`

A `WrappedFun` object represents a function `f`, together with a sequence of
nested transformations that are to be applied to the positional and keyword
arguments at call time and function return values at return time.
A transformation can take some static positional arguments that are given
at the wrapping time, and may also return some auxiliary output:

    wf, aux_out_thunk = trans1(wf, static_arg)

We can call the transformed function. First, the transformation is applied
to the dynamic args and keyword args to produce new dynamic and keyword args.
Then the underlying function is called and the transformation is applied to
the results.
If there are multiple transformations, they form a stack. The arguments are
transformed first with the last applied transformation; the results are
transformed first with the first applied transformation.

    res = wf.call_wrapped(dynamic_args, kwargs)
    # Now `aux_out_thunk()` is the auxiliary output.

A transformation is written as a generator function that takes zero or more
static positional arguments (given when the transformation is instantiated),
along with positional and keyword arguments to be transformed.
The generator will yield twice:

    @lu.transformation_with_aux
    def trans1(static_arg, *dynamic_args, **kwargs):
      ...
      # First yield: pair of transformed (args, kwargs). Get back the results.
      results = yield (new_dynamic_args, new_kwargs)
      ...
      # Second yield: pair of (transformed results, and auxiliary output)
      yield new_results, auxiliary_output


`WrappedFun` objects explicitly represent the set of transformations so that
they can be used as dictionary keys for memoization. `WrappedFun` objects
compare as equal only if they compute the same function. The static and the
dynamic positional arguments for the generators, and also the auxiliary output
data must be immutable, because it will be stored in function memoization tables.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple
import weakref

from jax._src import config
from jax._src import core
from jax._src import traceback_util
from jax._src.util import curry, cache_clearing_funs


traceback_util.register_exclusion(__file__)


class StoreException(Exception): pass


class EmptyStoreValue: pass
_EMPTY_STORE_VALUE = EmptyStoreValue()

class Store:
  """Storage for a value, with checks for overwriting or reading empty store."""
  __slots__ = ("_val",)

  def __init__(self):
    self._val = _EMPTY_STORE_VALUE

  def store(self, val):
    if self._val is not _EMPTY_STORE_VALUE:
      raise StoreException("Store occupied")
    self._val = val

  def reset(self):
    # This should only be called in exceptional circumstances (e.g. debugging).
    self._val = _EMPTY_STORE_VALUE

  @property
  def val(self):
    if not self:
      raise StoreException("Store empty")
    return self._val

  def __nonzero__(self):
    return self._val is not _EMPTY_STORE_VALUE

  __bool__ = __nonzero__

class EqualStore:
  __slots__ = ('_store',)

  def __init__(self):
    self._store = Store()

  @property
  def val(self):
    return self._store.val

  def store(self, val):
    try:
      self._store.store(val)
    except StoreException as e:
      try:
        okay = bool(self._store._val == val)
      except:
        raise e from None
      else:
        if not okay:
          raise StoreException("Store occupied with not-equal value") from None

  def reset(self):
    self._store.reset()


class WrappedFun:
  """Represents a function `f` to which `transforms` are to be applied.

  Args:
    f: the function to be transformed.
    transforms: a list of `(gen, gen_static_args)` tuples representing
      transformations to apply to `f.` Here `gen` is a generator function and
      `gen_static_args` is a tuple of static arguments for the generator. See
      description at the start of this module for the expected behavior of the
      generator.
    stores: a list of out_store for the auxiliary output of the `transforms`.
    params: extra parameters to pass as keyword arguments to `f`, along with the
      transformed keyword arguments.
  """
  __slots__ = ("f", "f_transformed", "transforms", "stores", "params", "in_type", "debug_info")

  def __init__(self, f: Callable,
               f_transformed: Callable,
               transforms,
               stores: tuple[Store | EqualStore | None, ...], params, in_type,
               debug_info: TracingDebugInfo | None):
    self.f = f
    self.f_transformed = f_transformed
    self.transforms = transforms
    self.stores = stores
    self.params = params
    self.in_type = in_type
    self.debug_info = debug_info

  @property
  def __name__(self):
    return getattr(self.f, '__name__', '<unnamed wrapped function>')

  def wrap(self, gen, gen_static_args,
           out_store: Store | EqualStore | None) -> WrappedFun:
    """Add another transform and its store."""
    if out_store is None:
      return WrappedFun(self.f, partial(gen, self.f_transformed, *gen_static_args),
                        ((gen, gen_static_args),) + self.transforms,
                        (out_store,) + self.stores, self.params, None, None)
    else:
      return WrappedFun(self.f, partial(gen, self.f_transformed, out_store, *gen_static_args),
                        ((gen, gen_static_args),) + self.transforms,
                        (out_store,) + self.stores, self.params, None, None)

  def populate_stores(self, stores):
    """Copy the values from the `stores` into `self.stores`."""
    for self_store, other_store in zip(self.stores, stores):
      if self_store is not None:
        self_store.store(other_store.val)

  def call_wrapped(self, *args, **kwargs):
    """Calls the transformed function"""
    return self.f_transformed(*args, **kwargs)

  def __repr__(self):
    def transform_to_str(x):
      i, (gen, args) = x
      return f"{i}   : {fun_name(gen)}   {fun_name(args)}"
    transformation_stack = map(transform_to_str, enumerate(self.transforms))
    return "Wrapped function:\n" + '\n'.join(transformation_stack) + '\nCore: ' + fun_name(self.f) + '\n'

  def __hash__(self):
    return hash((self.f, self.transforms, self.params, self.in_type,
                 self.debug_info))

  def __eq__(self, other):
    return (self.f == other.f and self.transforms == other.transforms and
            self.params == other.params and self.in_type == other.in_type and
            self.debug_info == other.debug_info)

@curry
def transformation2(gen, fun: WrappedFun, *gen_static_args) -> WrappedFun:
  """Adds one more transformation to a WrappedFun.

  Args:
    gen: the transformation generator function
    fun: a WrappedFun on which to apply the transformation
    gen_static_args: static args for the generator function
  """
  return fun.wrap(gen, gen_static_args, None)

# Backwards compat only. TODO: deprecate
@curry
def transformation(gen, fun: WrappedFun, *gen_static_args) -> WrappedFun:
  def gen2(f, *args, **kwargs):
    gen_inst = gen(*args, **kwargs)
    args_, kwargs_ = next(gen_inst)
    return gen_inst.send(f(*args_, **kwargs_))
  return transformation2(gen2, fun, *gen_static_args)()

# Backwards compat only. TODO: deprecate
@curry
def transformation_with_aux(gen, fun: WrappedFun, *gen_static_args) -> WrappedFun:
  def gen2(f, store, *args, **kwargs):
    gen_inst = gen(*args, **kwargs)
    args_, kwargs_ = next(gen_inst)
    ans, aux = gen_inst.send(f(*args_, **kwargs_))
    store.store(aux)
    return ans
  return transformation_with_aux2(gen2, fun, *gen_static_args)()

@curry
def transformation_with_aux2(
    gen, fun: WrappedFun, *gen_static_args, use_eq_store: bool = False
) -> tuple[WrappedFun, Callable[[], Any]]:
  """Adds one more transformation with auxiliary output to a WrappedFun."""
  out_store = Store() if not use_eq_store else EqualStore()
  out_thunk = lambda: out_store.val
  return fun.wrap(gen, gen_static_args, out_store), out_thunk

def fun_name(f):
  try:
    return f.__name__
  except:
    return str(f)

class TracingDebugInfo(NamedTuple):
  # Packages up trace/staging-time debug info about a func and its parameters,
  # formed just before staging to a jaxpr and read in trace-time error messages.
  traced_for: str             # e.g. 'jit', 'scan', etc
  func_src_info: str | None   # e.g. f'{fun.__name__} at {filename}:{lineno}'
  arg_names: tuple[str | None, ...]  # e.g. ('args[0]', ... )
  # e.g. ('[0]', '[1]', ...)
  result_paths_thunk: Callable[[], tuple[str, ...]] | None

  @classmethod
  def from_jaxpr(cls, jaxpr: core.ClosedJaxpr) -> TracingDebugInfo | None:
    jaxpr_dbg = jaxpr.jaxpr._debug_info
    if jaxpr_dbg is None: return None
    return TracingDebugInfo(jaxpr_dbg.traced_for,
                            jaxpr_dbg.func_src_info,
                            jaxpr_dbg.arg_names,
                            lambda: jaxpr_dbg.result_paths)

def wrap_init(f: Callable, params=None, *,
              debug_info: TracingDebugInfo | None = None) -> WrappedFun:
  """Wraps function `f` as a `WrappedFun`, suitable for transformation."""
  params_dict = {} if params is None else params
  params = () if params is None else tuple(sorted(params.items()))
  return WrappedFun(f, partial(f, **params_dict), (), (), params, None, debug_info)


def annotate(f: WrappedFun, in_type: core.InputType | None) -> WrappedFun:
  assert f.in_type is None
  if in_type is None:
    return f
  _check_input_type(in_type)
  return WrappedFun(f.f, f.f_transformed, f.transforms, f.stores, f.params, in_type, f.debug_info)

def _check_input_type(in_type: core.InputType) -> None:
  # Check that in_type is syntactically well-formed
  assert type(in_type) is tuple and all(type(e) is tuple for e in in_type)
  assert all(isinstance(a, core.AbstractValue) and type(b) is bool
             for a, b in in_type)

  def valid_size(d) -> bool:
    if isinstance(d, core.DBIdx) and type(d.val) is int and d.val >= 0:
      return True
    return (isinstance(d, (int, core.DBIdx, core.DArray)) and
            (not isinstance(d, core.DArray) or type(d) is core.bint and not d.shape))
  assert all(valid_size(d) for a, _ in in_type if type(a) is core.DShapedArray
             for d in a.shape)

  # Check that all DBIdx point to positions to the left of the input on which
  # they appear.
  assert all(d.val < i for i, (aval, _) in enumerate(in_type)
             if isinstance(aval, core.DShapedArray) for d in aval.shape
             if isinstance(d, core.DBIdx))

  # Check that all implicit arguments have at least one DBIdx pointing to them.
  provided = [e for _, e in in_type]
  for aval, _ in in_type:
    if type(aval) is core.DShapedArray:
      for d in aval.shape:
        if isinstance(d, core.DBIdx):
          provided[d.val] = True
  assert all(provided)

def add_debug_info(f: WrappedFun, debug_info: TracingDebugInfo | None
                   ) -> WrappedFun:
  """Produce a new WrappedFun with debug_info attached."""
  assert f.debug_info is None
  if debug_info is None:
    return f
  return WrappedFun(f.f, f.f_transformed, f.transforms, f.stores, f.params, f.in_type, debug_info)


def cache(call: Callable, *, explain: Callable | None = None):
  """Memoization decorator for functions taking a WrappedFun as first argument.

  Args:
    call: a Python callable that takes a WrappedFun as its first argument. The
      underlying transforms and params on the WrappedFun are used as part of the
      memoization cache key.

  Returns:
     A memoized version of ``call``.
  """
  fun_caches: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

  def memoized_fun(fun: WrappedFun, *args):
    cache = fun_caches.setdefault(fun.f, new_cache := {})  # type: ignore
    key = (fun.transforms, fun.params, fun.in_type, args, config.enable_x64.value,
           config.default_device.value, config.trace_context())
    result = cache.get(key, None)
    if result is not None:
      ans, stores = result
      fun.populate_stores(stores)
    else:
      ans = call(fun, *args)
      if explain and config.explain_cache_misses.value:
        explain(fun.f, cache is new_cache, cache, key)
      cache[key] = (ans, fun.stores)

    return ans

  def _evict_function(f):
    fun_caches.pop(f, None)

  memoized_fun.cache_clear = fun_caches.clear  # type: ignore
  memoized_fun.evict_function = _evict_function  # type: ignore
  cache_clearing_funs.add(memoized_fun.cache_clear)
  return memoized_fun

@transformation2
def hashable_partial(f, *args):
  return f(*args)


def merge_linear_aux(aux1, aux2):
  try:
    out1 = aux1()
  except StoreException:
    # store 1 was not occupied, so store 2 better be
    try:
      out2 = aux2()
    except StoreException:
      raise StoreException("neither store occupied") from None
    else:
      return False, out2
  else:
    # store 1 was occupied, so let's check store 2 is not occupied
    try:
      out2 = aux2()
    except StoreException:
      return True, out1
    else:
      raise StoreException("both stores occupied")
