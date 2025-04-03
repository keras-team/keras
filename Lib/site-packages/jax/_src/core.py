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

from collections import Counter, defaultdict, deque, namedtuple
from collections.abc import (Callable, Collection, Hashable, Iterable, Iterator,
                             Sequence, MutableSet, MutableMapping)
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import functools
from functools import partial, total_ordering
import gc
import inspect
import itertools as it
import math
import operator
import threading
import types
from typing import (Any, ClassVar, Generic, NamedTuple, TypeVar,
                    overload, Union)
import warnings
from weakref import ref

import numpy as np

from jax._src import dtypes
from jax._src import config
from jax._src import effects
from jax._src import compute_on
from jax._src import mesh as mesh_lib
from jax._src.partition_spec import PartitionSpec as P
from jax._src.errors import (
    ConcretizationTypeError, TracerArrayConversionError, TracerBoolConversionError,
    TracerIntegerConversionError, UnexpectedTracerError)
from jax._src import linear_util as lu

from jax._src import source_info_util
from jax._src.util import (safe_zip, safe_map, curry, tuple_insert,
                           tuple_delete,
                           HashableFunction, HashableWrapper, weakref_lru_cache,
                           partition_list, StrictABCMeta)
import jax._src.pretty_printer as pp
from jax._src.lib import jax_jit
from jax._src import traceback_util
from jax._src.typing import Array, DimSize, Shape
from jax._src import typing
from jax._src import xla_metadata as xla_metadata_lib

traceback_util.register_exclusion(__file__)

zip, unsafe_zip = safe_zip, zip
map, unsafe_map = safe_map, map


_TRACER_ERROR_NUM_TRACEBACK_FRAMES = config.int_flag(
    'jax_tracer_error_num_traceback_frames',
    config.int_env('JAX_TRACER_ERROR_NUM_TRACEBACK_FRAMES', 5),
    help='Set the number of stack frames in JAX tracer error messages.'
)


# -------------------- jaxprs --------------------

Effect = effects.Effect
Effects = effects.Effects
EffectTypeSet = effects.EffectTypeSet
no_effects: Effects = effects.no_effects

class JaxprDebugInfo(NamedTuple):
  traced_for: str     # e.g. 'jit', 'scan', etc
  func_src_info: str | None  # e.g. f'{fun.__name__} at {filename}:{lineno}'
  arg_names: tuple[str | None, ...]     # e.g. ('args[0]', ... )
  result_paths: tuple[str, ...]  # e.g. ('[0]', '[1]', ...)

class Jaxpr:
  __slots__ = ['__weakref__', '_constvars', '_invars', '_outvars', '_eqns',
               '_effects', '_debug_info']

  _constvars: list[Var]
  _invars: list[Var]
  _outvars: list[Atom]
  _eqns: list[JaxprEqn]
  _effects: Effects
  _debug_info: JaxprDebugInfo | None

  @property
  def constvars(self) -> list[Var]:
    return self._constvars

  @property
  def invars(self) -> list[Var]:
    return self._invars

  @property
  def outvars(self) -> list[Atom]:
    return self._outvars

  @property
  def eqns(self) -> list[JaxprEqn]:
    return self._eqns

  @property
  def effects(self) -> Effects:
    return self._effects

  @property
  def debug_info(self) -> JaxprDebugInfo | None:
    return self._debug_info

  def __init__(self, constvars: Sequence[Var], invars: Sequence[Var],
               outvars: Sequence[Atom], eqns: Sequence[JaxprEqn],
               effects: Effects = no_effects,
               debug_info: JaxprDebugInfo | None = None):
    """
    Args:
      constvars: list of variables introduced for constants. Array constants are
        replaced with such variables while scalar constants are kept inline.
      invars: list of input variables. Together, `constvars` and `invars` are
        the inputs to the Jaxpr.
      outvars: list of output atoms.
      eqns: list of equations.
      effects: set of effects. The effects on a jaxpr are a superset of the
        union of the effects for each equation.
      debug_info: optional JaxprDebugInfo.
    """
    self._constvars = list(constvars)
    self._invars = list(invars)
    self._outvars = list(outvars)
    self._eqns = list(eqns)
    self._effects = effects
    self._debug_info = debug_info
    assert (not debug_info or len(debug_info.arg_names) == len(invars) and
            len(debug_info.result_paths) == len(outvars))

  def __str__(self):
    return str(self.pretty_print())

  __repr__ = __str__

  def pretty_print(self, *, source_info=False, print_shapes=True,
                   custom_pp_eqn_rules=True, name_stack=False,
                   print_effects: bool = False, **kwargs):
    doc = pp_toplevel_jaxpr(
      self, source_info=source_info, print_shapes=print_shapes,
      custom_pp_eqn_rules=custom_pp_eqn_rules, name_stack=name_stack,
      print_effects=print_effects)
    return doc.format(**kwargs)

  def _repr_pretty_(self, p, cycle):
    return p.text(self.pretty_print(use_color=True))

  def replace(self, **kwargs):
    jaxpr = Jaxpr(
        constvars=kwargs.pop("constvars", self.constvars),
        invars=kwargs.pop("invars", self.invars),
        outvars=kwargs.pop("outvars", self.outvars),
        eqns=kwargs.pop("eqns", self.eqns),
        effects=kwargs.pop("effects", self.effects),
        debug_info=kwargs.pop("debug_info", self.debug_info),
    )
    if kwargs:
      raise ValueError(f"Unknown keyword arguments: {kwargs}")
    return jaxpr


def join_effects(*effects: Effects) -> Effects:
  return set().union(*effects) if effects else no_effects

def jaxprs_in_params(params) -> Iterator[Jaxpr]:
  for val in params.values():
    vals = val if isinstance(val, tuple) else (val,)
    for v in vals:
      if isinstance(v, Jaxpr):
        yield v
      elif isinstance(v, ClosedJaxpr):
        yield v.jaxpr


def subjaxprs(jaxpr: Jaxpr) -> Iterator[Jaxpr]:
  """Generator for all subjaxprs found in the params of jaxpr.eqns.
  Does not descend recursively into the found subjaxprs.
  """
  for eqn in jaxpr.eqns:
    yield from jaxprs_in_params(eqn.params)


class ClosedJaxpr:
  __slots__ = ['__weakref__', '_jaxpr', '_consts']

  _jaxpr: Jaxpr
  _consts: list[Any]

  jaxpr = property(lambda self: self._jaxpr)
  consts = property(lambda self: self._consts)

  def __init__(self, jaxpr: Jaxpr, consts: Sequence):
    assert len(consts) == len(jaxpr.constvars)
    # assert not any(isinstance(c, Tracer) for c in consts)  # TODO(mattjj): enable
    self._jaxpr = jaxpr
    self._consts = list(consts)

  @property
  def in_avals(self):
    return [v.aval for v in self.jaxpr.invars]

  @property
  def out_avals(self):
    return [v.aval for v in self.jaxpr.outvars]

  @property
  def literals(self):
    return self.consts  # backwards compatible alias

  @property
  def eqns(self):
    return self.jaxpr.eqns

  @property
  def effects(self) -> Effects:
    return self.jaxpr.effects

  def map_jaxpr(self, f):
    return ClosedJaxpr(f(self.jaxpr), self.consts)

  def replace(self, *, jaxpr=None, consts=None):
    jaxpr = self.jaxpr if jaxpr is None else jaxpr
    consts = self.consts if consts is None else consts
    return ClosedJaxpr(jaxpr, consts)

  def __str__(self): return str(self.jaxpr)
  def __repr__(self): return repr(self.jaxpr)

  def pretty_print(self, *, source_info=False, print_shapes=True,
                   name_stack=False, custom_pp_eqn_rules=True,
                   print_effects=False, **kwargs):
    return self.jaxpr.pretty_print(
        source_info=source_info,
        print_shapes=print_shapes,
        name_stack=name_stack,
        custom_pp_eqn_rules=custom_pp_eqn_rules,
        print_effects=print_effects,
        **kwargs)

  def _repr_pretty_(self, p, cycle):
    return p.text(self.pretty_print(use_color=True))

@curry
def jaxpr_as_fun(closed_jaxpr: ClosedJaxpr, *args):
  # TODO(dougalm): remove this hack when we add contexts to jaxpr.
  # debug_nans is sometimes disabled locally at the traceable level by ops that
  # work with nans internally, like jnp.var. The right thing to do is to add
  # contexts to our jaxpr representation so that we can capture these local
  # context modifications. In the meantime, disabling the checks when we
  # round-trip prevents those ops producing spurious errors.
  with config.debug_nans(False):
    return eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args)


class JaxprEqnContext:

  def __init__(self, compute_type: str | None, threefry_partitionable: bool,
               xla_metadata=None):
    self.compute_type = compute_type
    self.threefry_partitionable = threefry_partitionable
    self.cur_abstract_mesh = mesh_lib.get_abstract_mesh()
    self.xla_metadata = xla_metadata
    self._managers = [
        (compute_on.extend_compute_type, self.compute_type),
        (config.threefry_partitionable.__call__, self.threefry_partitionable),
        (mesh_lib.set_abstract_mesh, self.cur_abstract_mesh),
        (xla_metadata_lib.set_xla_metadata, self.xla_metadata),
    ]

  @property
  @contextmanager
  def manager(self):
    with ExitStack() as stack:
      for manager, val in self._managers:
        stack.enter_context(manager(val))
      yield

  def __repr__(self):
    return (
        f"JaxprEqnContext(compute_type={self.compute_type}, "
        f"threefry_partitionable={self.threefry_partitionable}, "
        f"cur_abstract_mesh={self.cur_abstract_mesh}, "
        f"xla_metadata={self.xla_metadata})"
    )


class JaxprEqn:
  invars: list[Atom]
  outvars: list[Var]
  primitive: Primitive
  params: dict[str, Any]
  effects: Effects
  source_info: source_info_util.SourceInfo
  ctx: JaxprEqnContext

  # It's slightly faster to use a class with __slots__ than a NamedTuple.
  __slots__ = ['invars', 'outvars', 'primitive', 'params', 'effects',
               'source_info', 'ctx']

  def __init__(self, invars, outvars, primitive, params, effects, source_info,
               ctx):
    self.invars = invars
    self.outvars = outvars
    self.primitive = primitive
    self.params = params
    self.effects = effects
    self.source_info = source_info
    self.ctx = ctx

  def __repr__(self):
    return str(pp_eqn(self, JaxprPpContext(), JaxprPpSettings())).rstrip()

  def replace(
      self,
      invars: list[Atom] | None = None,
      outvars: list[Var] | None = None,
      primitive: Primitive | None = None,
      params: dict[str, Any] | None = None,
      effects: Effects | None = None,
      source_info: source_info_util.SourceInfo | None = None,
      ctx: JaxprEqnContext | None = None
  ):
    return JaxprEqn(
      self.invars if invars is None else invars,
      self.outvars if outvars is None else outvars,
      self.primitive if primitive is None else primitive,
      self.params if params is None else params,
      self.effects if effects is None else effects,
      self.source_info if source_info is None else source_info,
      self.ctx if ctx is None else ctx,
    )


# TODO(mattjj): call typecheck rules here, so we don't form bad eqns
def new_jaxpr_eqn(invars, outvars, primitive, params, effects, source_info=None,
                  ctx=None):
  source_info = source_info or source_info_util.new_source_info()
  ctx = ctx or JaxprEqnContext(
      compute_on.current_compute_type(),
      config.threefry_partitionable.value,
      xla_metadata_lib.current_xla_metadata())
  if config.enable_checks.value:
    assert all(isinstance(x, (Var, Literal)) for x in  invars)
    assert all(isinstance(v,  Var)           for v in outvars)
  return JaxprEqn(invars, outvars, primitive, params, effects, source_info, ctx)

_var_counter = it.count()

@total_ordering
class Var:
  __slots__ = ["count", "suffix", "aval"]

  count: int
  suffix: str
  aval: AbstractValue

  def __init__(self, suffix: str, aval: AbstractValue):
    self.count = next(_var_counter)
    self.suffix = suffix
    self.aval = aval

  # TODO(phawkins, mattjj): remove ordering of variables. JAX itself does not
  # care about variable ordering, but the downstream package kfac_jax does.
  def __lt__(self, other):
    return self.count < other.count

  def __repr__(self):
    return f'Var(id={id(self)}){self.suffix}:{self.aval.str_short()}'


def gensym(suffix: str = '') -> Callable[[AbstractValue], Var]:
  """Produce distinct variables, printed with the optional suffix."""
  return partial(Var, suffix)

# In a jaxpr, `dropvar` can appear in place of a bound variable to indicate that
# the assignment is dropped, i.e. that an expression's output value will never
# be read. In that sense, `dropvar` is not a variable, but it is convenient to
# treat it as a special case of one. Its `aval` is similarly inexact.
class DropVar(Var):
  def __init__(self, aval: AbstractValue):
    super().__init__('', aval)
  def __repr__(self): return '_'

class Literal:
  __slots__ = ["val", "aval", "hash"]

  val: Any
  aval: AbstractValue
  hash: int | None

  def __init__(self, val, aval):
    self.val = val
    self.aval = aval
    try:
      self.hash = hash(val)
    except TypeError:
      if type(val) in literalable_types:
        try:
          self.hash = hash((val.item(), val.dtype))
        except (TypeError, AttributeError, ValueError):
          self.hash = None

  __hash__ = None  # type: ignore

  def __repr__(self):
    if hasattr(self, 'hash'):
      return f'{self.val}'
    else:
      return f'Literal(val={self.val})'

literalable_types: set[type] = set()

Atom = Union[Var, Literal]

class Primitive:
  name: str
  # set for multi-output primitives.
  multiple_results: bool = False
  # set for call primitives processed in final style.
  call_primitive: bool = False
  # set for map primitives processed in final style.
  map_primitive: bool = False
  # set for ref primitives
  ref_primitive: bool = False

  def __init__(self, name: str):
    self.name = name

  def __repr__(self):
    return f'{self.name}'

  def bind(self, *args, **params):
    for arg in args:
      if (isinstance(arg, Tracer)
          and not arg._trace.is_valid()
          and not config.data_dependent_tracing_fallback.value):
        raise escaped_tracer_error(arg)
    # TODO: figure out how to handle function arguments
    # assert (not config.enable_checks.value or
    #         all(isinstance(arg, Tracer) or valid_jaxtype(arg) for arg in args)), args

    # This is equivalent to "with take_current_trace()", but the bind() code
    # is called frequently and it's slightly faster to avoid using a context
    # manager object.
    prev_trace = trace_ctx.trace
    trace_ctx.set_trace(eval_trace)
    try:
      return self.bind_with_trace(prev_trace, args, params)
    finally:
      trace_ctx.set_trace(prev_trace)

  def bind_with_trace(self, trace, args, params):
    return trace.process_primitive(self, args, params)

  def def_impl(self, impl):
    self.impl = impl
    return impl

  def def_abstract_eval(self, abstract_eval):
    self.abstract_eval = _effect_free_abstract_eval(abstract_eval)
    return abstract_eval

  def def_effectful_abstract_eval(self, effectful_abstract_eval):
    self.abstract_eval = effectful_abstract_eval
    return effectful_abstract_eval

  def def_bind_with_trace(self, bind_with_trace):
    self.bind_with_trace = bind_with_trace
    return bind_with_trace

  def impl(self, *args, **params):
    raise NotImplementedError("Evaluation rule for '{}' not implemented"
                              .format(self.name))

  def abstract_eval(self, *args, **params):
    raise NotImplementedError("Abstract evaluation for '{}' not implemented"
                              .format(self.name))

  def get_bind_params(self, params):
    return [], params


def _effect_free_abstract_eval(abstract_eval):
  def abstract_eval_(*args, **kwargs):
    return abstract_eval(*args, **kwargs), no_effects
  return abstract_eval_

# -------------------- lifting --------------------

# TODO(mattjj): replace this approach with a primitive-keyed table of rules
def traverse_jaxpr_params(f, params):
  """Applies f to each jaxpr parameter and returns a tuple of returned values."""
  return {name: f(p)
          for name, param in params.items()
          for p in (param if isinstance(param, (tuple, list)) else [param])
          if type(p) in (Jaxpr, ClosedJaxpr)}


def eval_jaxpr(jaxpr: Jaxpr, consts, *args, propagate_source_info=True) -> list[Any]:
  def read(v: Atom) -> Any:
    return v.val if isinstance(v, Literal) else env[v]

  def write(v: Var, val: Any) -> None:
    if config.enable_checks.value and not config.dynamic_shapes.value:
      assert typecheck(v.aval, val), (v.aval, val)
    env[v] = val

  env: dict[Var, Any] = {}
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  lu = last_used(jaxpr)
  for eqn in jaxpr.eqns:
    subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
    name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
    traceback = eqn.source_info.traceback if propagate_source_info else None
    with source_info_util.user_context(
        traceback, name_stack=name_stack), eqn.ctx.manager:
      ans = eqn.primitive.bind(*subfuns, *map(read, eqn.invars), **bind_params)
    if eqn.primitive.multiple_results:
      map(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
    clean_up_dead_vars(eqn, env, lu)
  return map(read, jaxpr.outvars)

def check_avals_context_mesh(avals, prim_name):
  if config.sharding_in_types.value:
    for a in avals:
      cur_mesh = mesh_lib.get_abstract_mesh()
      if a.sharding.mesh != cur_mesh:
        raise ValueError(
            f"For primitive {prim_name}, context mesh {cur_mesh} should match"
            f" the aval mesh {a.sharding.mesh} for shape {a.str_short()}. This"
            " error occurs at source: "
            f" {source_info_util.summarize(source_info_util.current())}")


# -------------------- tracing --------------------

TracerType = TypeVar('TracerType', bound='Tracer')

class Trace(Generic[TracerType]):

  def process_primitive(self, primitive, tracers, params):
    raise NotImplementedError("must override")

  def invalidate(self):
    self._invalidated = True

  def is_valid(self):
    return not hasattr(self, "_invalidated")

  def __repr__(self):
    return '{}'.format(self.__class__.__name__)

  def process_call(self, call_primitive, f, tracers, params):
    msg = (f"{type(self)} must override process_call to handle call-like "
           "primitives")
    raise NotImplementedError(msg)

  def process_map(self, map_primitive, f, tracers, params):
    msg = (f"{type(self)} must override process_map to handle map-like "
           "primitives")
    raise NotImplementedError(msg)

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *,
                              symbolic_zeros):
    msg = (f"{type(self)} must override process_custom_jvp_call "
           "to handle custom_jvp primitives")
    raise NotImplementedError(msg)

  def process_custom_transpose(self, prim, call, tracers, **params):
    msg = (f"{type(self)} must override process_custom_transpose "
           "to handle custom_transpose_call primitives")
    raise NotImplementedError(msg)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers,
                              out_trees, symbolic_zeros):
    msg = (f"{type(self)} must override process_custom_vjp_call "
           "to handle custom_vjp primitives")
    raise NotImplementedError(msg)

  # TODO(dougalm): deprecate/delete
  def full_raise(self, x):
    return x

  # TODO(dougalm): deprecate/delete
  @property
  def main(self):
    return getattr(self, "tag", None)

def escaped_tracer_error(tracer, detail=None):
  num_frames = _TRACER_ERROR_NUM_TRACEBACK_FRAMES.value
  msg = ('Encountered an unexpected tracer. A function transformed by JAX '
         'had a side effect, allowing for a reference to an intermediate value '
         f'with type {tracer.aval.str_short()} wrapped in a '
         f'{type(tracer).__name__} to escape the scope of the transformation.\n'
         'JAX transformations require that functions explicitly return their '
         'outputs, and disallow saving intermediate values to global state.')
  dbg = getattr(tracer, '_debug_info', None)
  if dbg is not None:
    msg += ('\nThe function being traced when the value leaked was '
            f'{dbg.func_src_info} traced for {dbg.traced_for}.')
  line_info = getattr(tracer, '_line_info', None)
  if line_info is not None:
    divider = '\n' + '-'*30 + '\n'
    msg += divider
    msg += ('The leaked intermediate value was created on line '
            f'{source_info_util.summarize(line_info)}. ')
    msg += divider
    if num_frames > 0:
      msg += (f'When the value was created, the final {num_frames} stack '
              'frames (most recent last) excluding JAX-internal frames were:')
      msg += divider + source_info_util.summarize(
          line_info, num_frames=num_frames) + divider
  msg += ('\nTo catch the leak earlier, try setting the environment variable '
          'JAX_CHECK_TRACER_LEAKS or using the `jax.checking_leaks` context '
          'manager.')
  if detail:
    msg += f'Detail: {detail}'
  return UnexpectedTracerError(msg)


def check_scalar_conversion(arr: Array):
  if arr.ndim > 0:
    raise TypeError("Only scalar arrays can be converted to Python scalars; "
                    f"got {arr.ndim=}")


def check_integer_conversion(arr: Array):
  if not (arr.shape == () and dtypes.issubdtype(arr.dtype, np.integer)):
    raise TypeError("Only integer scalar arrays can be converted to a scalar index.")


def check_bool_conversion(arr: Array):
  if arr.size == 0:
    raise ValueError("The truth value of an empty array is ambiguous. Use"
                     " `array.size > 0` to check that an array is not empty.")
  if arr.size > 1:
    raise ValueError("The truth value of an array with more than one element"
                     " is ambiguous. Use a.any() or a.all()")


pytype_aval_mappings: dict[type, Callable[[Any], AbstractValue]] = {}

def _str_abstractify(x):
  raise TypeError(f"Argument '{x}' of type {type(x)} is not a valid JAX type")
pytype_aval_mappings[str] = _str_abstractify


def _aval_property(name):
  return property(lambda self: getattr(self.aval, name))


class Tracer(typing.Array, metaclass=StrictABCMeta):
  __array_priority__ = 1000
  __slots__ = ['_trace', '_line_info']
  __hash__ = None  # type: ignore

  _trace: Trace
  _line_info: source_info_util.SourceInfo | None

  dtype = _aval_property('dtype')
  ndim = _aval_property('ndim')
  size = _aval_property('size')
  shape = _aval_property('shape')

  def __init__(self, trace: Trace):
    self._trace = trace

  def _error_repr(self):
    if self.aval is None:
      return f"traced array with aval {self.aval}"
    return f"traced array with shape {self.aval.str_short()}"

  def __array__(self, *args, **kw):
    raise TracerArrayConversionError(self)

  def __dlpack__(self, *args, **kw):
    raise ConcretizationTypeError(self,
      f"The __dlpack__() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def tolist(self):
    raise ConcretizationTypeError(self,
      f"The tolist() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def tobytes(self, order="C"):
    del order
    raise ConcretizationTypeError(self,
      f"The tobytes() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  # TODO(dougalm): deprecate/delete
  def full_lower(self):
    raise NotImplementedError("must override: ", type(self))

  def __iter__(self):
    return iter(self.aval._iter(self))

  def __reversed__(self):
    return iter(self[::-1])

  def __len__(self):
    return self.aval._len(self)

  def to_concrete_value(self):
    # Should return the concrete value if there is one, or else None.
    return None

  @property
  def sharding(self):
    # This attribute is part of the jax.Array API, but only defined on concrete arrays.
    # Raising a ConcretizationTypeError would make sense, but for backward compatibility
    # we raise an AttributeError so that hasattr() and getattr() work as expected.
    raise AttributeError(
        self,
        f"The 'sharding' attribute is not available on {self._error_repr()}."
        f"{self._origin_msg()}")

  @property
  def committed(self):
    raise ConcretizationTypeError(
        self,
        f"The 'committed' attribute is not available on {self._error_repr()}."
        f"{self._origin_msg()}")

  @property
  def device(self):
    # This attribute is part of the jax.Array API, but only defined on concrete arrays.
    # Raising a ConcretizationTypeError would make sense, but for backward compatibility
    # we raise an AttributeError so that hasattr() and getattr() work as expected.
    raise AttributeError(self,
      f"The 'device' attribute is not available on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def addressable_shards(self):
    raise ConcretizationTypeError(self,
      f"The 'addressable_shards' attribute is not available on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def at(self):
    return self.aval.at.fget(self)

  @property
  def aval(self):
    raise NotImplementedError("must override")

  def get_referent(self) -> Any:
    return self  # Override for object equivalence checking

  def __bool__(self):
    if is_concrete(self): return bool(self.to_concrete_value())  # pytype: disable=wrong-arg-types
    check_bool_conversion(self)
    return self.aval._bool(self)

  def __int__(self):
    if is_concrete(self): return int(self.to_concrete_value())  # pytype: disable=wrong-arg-types
    check_scalar_conversion(self)
    return self.aval._int(self)

  def __float__(self):
    check_scalar_conversion(self)
    return self.aval._float(self)

  def __complex__(self):
    check_scalar_conversion(self)
    return self.aval._complex(self)

  def __hex__(self):
    if is_concrete(self): return hex(self.to_concrete_value())  # pytype: disable=wrong-arg-types
    check_integer_conversion(self)
    return self.aval._hex(self)

  def __oct__(self):
    if is_concrete(self): return oct(self.to_concrete_value())  # pytype: disable=wrong-arg-types
    check_integer_conversion(self)
    return self.aval._oct(self)

  def __index__(self):
    if is_concrete(self): return operator.index(self.to_concrete_value())  # pytype: disable=wrong-arg-types
    check_integer_conversion(self)
    return self.aval._index(self)

  # raises a useful error on attempts to pickle a Tracer.
  def __reduce__(self):
    raise ConcretizationTypeError(
      self, ("The error occurred in the __reduce__ method, which may "
             "indicate an attempt to serialize/pickle a traced value."))

  # raises the better error message from ShapedArray
  def __setitem__(self, idx, val): return self.aval._setitem(self, idx, val)

  # NumPy also only looks up special methods on classes.
  def __array_module__(self, types): return self.aval._array_module(self, types)

  def __getattr__(self, name):
    # if the aval property raises an AttributeError, gets caught here
    assert not config.enable_checks.value or name != "aval"

    try:
      attr = getattr(self.aval, name)
    except AttributeError as err:
      raise AttributeError(
          f"{self.__class__.__name__} has no attribute {name}"
      ) from err
    else:
      t = type(attr)
      if t is aval_property:
        return attr.fget(self)
      elif t is aval_method:
        return types.MethodType(attr.fun, self)
      else:
        return attr

  def _pretty_print(self):
    base = pp.text(f'Traced<{self.aval}>with<{self._trace}>')
    contents = [(name, attr._pretty_print() if isinstance(attr, Tracer)
                 else pp.text(repr(attr))) for name, attr in self._contents()]
    if contents:
      base = pp.group(pp.nest(2, pp.concat([
        base, pp.text(' with'), pp.brk(), pp.join(pp.brk(), [
          pp.text(f'{name} = ') + pp_payload
          for name, pp_payload in contents])
      ])))
    return base

  def __repr__(self):
    return self._pretty_print().format()

  def _contents(self):
    try:
      return [(name, getattr(self, name)) for name in self.__slots__]
    except AttributeError:
      return ()

  def _origin_msg(self) -> str:
    return ""

  # Methods that are only valid for materialized arrays
  def addressable_data(self, index):
    raise ConcretizationTypeError(self,
      f"The addressable_data() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def block_until_ready(self):
    # Raise AttributeError for backward compatibility with hasattr() and getattr() checks.
    raise AttributeError(self,
      f"The 'block_until_ready' method is not available on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def copy_to_host_async(self):
    # Raise AttributeError for backward compatibility with hasattr() and getattr() checks.
    raise AttributeError(self,
      f"The 'copy_to_host_async' method is not available on {self._error_repr()}."
      f"{self._origin_msg()}")

  def delete(self):
    raise ConcretizationTypeError(self,
      f"The delete() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def devices(self):
    raise ConcretizationTypeError(self,
      f"The devices() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def global_shards(self):
    raise ConcretizationTypeError(self,
      f"The global_shards property was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def is_deleted(self):
    raise ConcretizationTypeError(self,
      f"The is_deleted() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def is_fully_addressable(self):
    raise ConcretizationTypeError(self,
      f"The is_fully_addressable property was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def is_fully_replicated(self):
    raise ConcretizationTypeError(self,
      f"The is_fully_replicated property was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def on_device_size_in_bytes(self):
    raise ConcretizationTypeError(self,
      f"The on_device_size_in_bytes() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def traceback(self):
    raise ConcretizationTypeError(self,
      f"The traceback property was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def unsafe_buffer_pointer(self):
    raise ConcretizationTypeError(self,
      f"The unsafe_buffer_pointer() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

# these can be used to set up forwarding of properties and instance methods from
# Tracer instances to the underlying avals
aval_property = namedtuple("aval_property", ["fget"])
aval_method = namedtuple("aval_method", ["fun"])

pytype_aval_mappings[Tracer] = lambda x: x.aval

def check_eval_args(args):
  for arg in args:
    if isinstance(arg, Tracer):
      raise escaped_tracer_error(arg)

class EvalTrace(Trace):

  def process_primitive(self, primitive, args, params):
    if config.debug_key_reuse.value:
      # Import here to avoid circular imports
      from jax.experimental.key_reuse._core import call_impl_with_key_reuse_checks  # pytype: disable=import-error
      return call_impl_with_key_reuse_checks(primitive, primitive.impl, *args, **params)
    else:
      # TODO(dougalm): delete. this shouldn't be necessary
      args = map(full_lower, args)
      if config.data_dependent_tracing_fallback.value:
        for arg in args:
          if isinstance(arg, Tracer):
            return primitive.bind_with_trace(arg._trace, args, params)
      check_eval_args(args)
      return primitive.impl(*args, **params)

  def process_call(self, primitive, f, tracers, params):
    if config.debug_key_reuse.value:
      # Import here to avoid circular imports
      from jax.experimental.key_reuse._core import call_impl_with_key_reuse_checks  # pytype: disable=import-error
      return call_impl_with_key_reuse_checks(primitive, primitive.impl, f, *tracers, **params)
    else:
      return primitive.impl(f, *tracers, **params)
  process_map = process_call

  def process_custom_transpose(self, primitive, call, tracers, **_):
    del primitive, _
    return call.call_wrapped(*tracers)

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers, **_):
    del primitive, jvp, _  # Unused.
    return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, **_):  # pytype: disable=signature-mismatch
    del primitive, fwd, bwd, _  # Unused.
    return fun.call_wrapped(*tracers)


class TraceTag:
  # TODO: this works for surprisingly subtle reasons. Function transformations
  # like `jvp_subtrace` are parameterized by a tag that identifies the set of
  # pre-existing tracers we want to unpack during the transformation. A function
  # defined in an outer scope can't have any closed-over traces, so the tag is
  # irrelevant. A function defined in the current scope may have closed-over
  # traces, but the tag will never change so we'll never get a spurious cache
  # hit. The plan is to do away with `lu.cache` altogether, and use a simpler
  # caching scheme that only caches top-level functions. Then we can remove this
  # hack.
  def __hash__(self):
    return hash(TraceTag)
  def __eq__(self, other):
    return isinstance(other, TraceTag)

ParamDict = dict[str, Any]
AxisName = Hashable

no_axis_name = object()

@dataclass(frozen=True)
class AxisEnv:
  axis_sizes : dict[AxisName, int]
  spmd_axis_names : set[AxisName]

  def axis_size(self, axis_name):
    if axis_name not in self.axis_sizes:
      raise NameError(f"unbound axis name: {axis_name}")
    else:
      return self.axis_sizes[axis_name]

  def axis_exists(self, axis_name):
    return axis_name in self.axis_sizes

  def axis_names(self):
    return tuple(k for k in self.axis_sizes)

  def pop_pure(self, axis_name):
    new_sizes = self.axis_sizes.copy()
    new_sizes.pop(axis_name)
    return AxisEnv(new_sizes, self.spmd_axis_names)

  def extend_pure(self, name_size_pairs):
    new_sizes = self.axis_sizes.copy()
    new_sizes.update((name, size) for name, size in name_size_pairs
                    if name is not no_axis_name)
    return AxisEnv(new_sizes, self.spmd_axis_names)

  def add_spmd_axis_names(self, axis_names):
    new_spmd_axis_names = self.spmd_axis_names | set(axis_names)
    return AxisEnv(self.axis_sizes, new_spmd_axis_names)

  def as_hashable_key(self):
    return tuple((name, size) for (name, size) in self.axis_sizes.items()
                 if name is not no_axis_name)

eval_trace = EvalTrace()
top_axis_env = AxisEnv({}, set())

class TracingContext(threading.local):
  trace: Trace | None
  axis_env : AxisEnv

  def __init__(self):
    self.reset()

  def reset(self):
    self.trace = eval_trace
    self.axis_env = top_axis_env

  def is_top_level(self) -> bool:
    return (self.trace is eval_trace and
            self.axis_env is top_axis_env)

  def set_trace(self, trace):
    self.trace = trace
    ts = ref(trace) if trace is not None else None
    config.trace_state.set_local(ts)

  def set_axis_env(self, axis_env):
    self.axis_env = axis_env
    config.axis_env_state.set_local(axis_env.as_hashable_key())

  def update_thread_local_jit_state(self):
    ts = ref(self.trace) if self.trace is not None else None
    config.trace_state.set_local(ts)
    config.axis_env_state.set_local(self.axis_env.as_hashable_key())

trace_ctx = TracingContext()


@contextmanager
def take_current_trace():
  prev = trace_ctx.trace
  try:
    trace_ctx.set_trace(eval_trace)
    yield prev
  finally:
    trace_ctx.set_trace(prev)

@contextmanager
def set_current_trace(trace, check_leaks=False):
  prev = trace_ctx.trace
  try:
    trace_ctx.set_trace(trace)
    yield
  finally:
    trace_ctx.set_trace(prev)
    if check_leaks and config.check_tracer_leaks.value:
      trace.invalidate()
      trace_ref = ref(trace)
      del trace
      live_trace = trace_ref()
      if live_trace is not None:
        leaked_tracers = maybe_find_leaked_tracers(live_trace)
        if leaked_tracers:
          raise leaked_tracer_error("trace", live_trace, leaked_tracers)

@contextmanager
def extend_axis_env_nd(name_size_pairs : Iterable[tuple[AxisName, int]]):
  prev = trace_ctx.axis_env
  try:
    trace_ctx.set_axis_env(prev.extend_pure(name_size_pairs))
    yield
  finally:
    trace_ctx.set_axis_env(prev)

@contextmanager
def add_spmd_axis_names(axis_names: AxisName | None):
  prev = trace_ctx.axis_env
  try:
    if axis_names is not None:
      trace_ctx.set_axis_env(prev.add_spmd_axis_names(axis_names))
    yield
  finally:
    trace_ctx.set_axis_env(prev)

def get_axis_env():
  return trace_ctx.axis_env

def _initialize_jax_jit_thread_local_state():
  """Initializes the C++ thread-local context.

  When the user spawns threads, the C++ `jax_jit.thread_local_state` is None.
  The C++ accessor calls this function if it realizes the thread_local_state
  is None (which means it's not yet initialized for this thread).

  This function does not live in `config.py`, to prevent circular imports.
  """
  trace_ctx.update_thread_local_jit_state()

jax_jit.set_thread_local_state_initialization_callback(
    _initialize_jax_jit_thread_local_state)

def trace_state_clean() -> bool:
  return trace_ctx.is_top_level()

def reset_trace_state() -> bool:
  """Resets the global trace state and returns True if it was already clean."""
  if not trace_ctx.is_top_level():
    trace_ctx.reset()
    trace_ctx.update_thread_local_jit_state()
    return False
  else:
    return True

TRACER_LEAK_DEBUGGER_WARNING = """\
JAX check_tracer_leaks behavior can trigger false positives when used with a debugger.
To avoid false positives and silence this warning, you can disable thread tracing using
the following:

  import threading
  threading.current_thread().pydev_do_not_trace = True
"""

@contextmanager
def ensure_no_leaks(trace:Trace):
  yield
  trace.invalidate()
  if config.check_tracer_leaks.value:
    trace_ref = ref(trace)
    del trace
    live_trace = trace_ref()
    if live_trace is not None:
      leaked_tracers = maybe_find_leaked_tracers(live_trace)
      if leaked_tracers:
        raise leaked_tracer_error("trace", live_trace, leaked_tracers)

def maybe_find_leaked_tracers(trace: Trace) -> list[Tracer]:
  """Find the leaked tracers holding a reference to the Trace
  """
  if not getattr(threading.current_thread(), 'pydev_do_not_trace', True):
    warnings.warn(TRACER_LEAK_DEBUGGER_WARNING)
  # Trigger garbage collection to filter out unreachable objects that are alive
  # only due to cyclical dependencies. (We don't care about unreachable leaked
  # tracers since they can't interact with user code and cause a problem.)
  gc.collect()
  tracers = list(filter(lambda x: isinstance(x, Tracer), gc.get_referrers(trace)))
  return tracers

def leaked_tracer_error(name: str, t, tracers: list[Tracer]) -> Exception:
  assert tracers
  why = partial(_why_alive, {id(tracers)})
  msgs = '\n\n'.join(f'{tracers[i]}{tracers[i]._origin_msg()}{why(tracers[i])}'
                     for i in range(len(tracers)))
  return Exception(f'Leaked {name} {t}. Leaked tracer(s):\n\n{msgs}\n')

def _why_alive(ignore_ids: set[int], x: Any) -> str:
  parents = lambda x: [r for r in gc.get_referrers(x) if id(r) not in ignore_ids]
  child, lines, seen = x, [], set()
  while (id(child) not in seen and type(child) is not types.ModuleType
         and parents(child)):
    parent = parents(child)[0]  # just pick one parent

    # For namespaces (like modules and class instances) and closures, the
    # references may form a simple chain: e.g. instance refers to its own
    # __dict__ which refers to child, or function refers to its __closure__
    # which refers to cells which refer to child. In these cases, we can provide
    # a more intuitive description by collapsing the chain into a single
    # parent->child jump. We do that by setting `parent` here to be a
    # grandparent (or great-grandparent) of `child`, and then handling that case
    # in _why_alive_container_info. See example:
    #  https://github.com/jax-ml/jax/pull/13022#discussion_r1008456599
    # To prevent this collapsing behavior, just comment out this code block.
    if (isinstance(parent, dict) and
        getattr(parents(parent)[0], '__dict__', None) is parents(child)[0]):
      parent = parents(parent)[0]
    elif type(parent) is types.CellType:
      parent = parents(parents(parent)[0])[0]

    line = f'<{type(child).__name__} {id(child)}> is referred to by '
    lines.append(line + _why_alive_container_info(parent, id(child)))
    seen.add(id(child))
    child = parent
  return '\n' + '\n'.join(lines) if lines else ''

def _why_alive_container_info(container, obj_id) -> str:
  name = f'<{type(container).__name__} {id(container)}>'
  if type(container) is types.ModuleType:
    name = getattr(container, '__name__', name)
  if type(container) is types.FunctionType:
    name_ = getattr(container, '__name__', '<no-name>')
    closure = inspect.getclosurevars(container)
    keys = [k for k, v in dict(closure.nonlocals, **closure.globals).items()
            if id(v) == obj_id]
    if len(keys) == 1: return f'{name} ({name_}) closed-over variable {keys[0]}'
    elif len(keys) > 1: return (f'{name} in closed-over variables ' +
                                ', '.join(map(repr, keys)))
  if hasattr(container, '__dict__'):
    keys = [k for k in vars(container) if id(vars(container)[k]) == obj_id]
    if len(keys) == 1: return f'{name}.{keys[0]}'
    elif len(keys) > 1: return f'{name} in vars ' + ', '.join(map(repr, keys))
  if isinstance(container, (list, tuple)):
    idxs = [i for i, x in enumerate(container) if id(x) == obj_id]
    if len(idxs) == 1: return f'{name}[{idxs[0]}]'
    else: return f'{name} at indices ' + ', '.join(map(str, idxs))
  if isinstance(container, dict):
    keys = [k for k in container if id(container[k]) == obj_id]
    if len(keys) == 1: return f'{name}[{keys[0]!r}]'
    else: return f'{name} at keys ' + ', '.join(map(repr, keys))
  if isinstance(container, types.ModuleType):
    return f' named {container.__name__}'
  return name

@contextmanager
def ensure_compile_time_eval():
  """Context manager to ensure evaluation at trace/compile time (or error).

  Some JAX APIs like :func:`jax.jit` and :func:`jax.lax.scan` involve staging,
  i.e., delaying the evaluation of numerical expressions (like :mod:`jax.numpy`
  function applications) so that instead of performing those computations
  eagerly while evaluating the corresponding Python expressions, their
  computation is carried out separately, e.g. after optimized compilation. But
  this delay can be undesirable. For example, numerical values might be needed
  to evaluate Python control flow and so their evaluation cannot be delayed. As
  another example, it may be beneficial to ensure compile time evaluation (or
  "constant folding") for performance reasons.

  This context manager ensures that JAX computations are evaluated eagerly. If
  eager evaluation is not possible, a ``ConcretizationTypeError`` is raised.

  Here's a contrived example::

    import jax
    import jax.numpy as jnp

    @jax.jit
    def f(x):
      with jax.ensure_compile_time_eval():
        y = jnp.sin(3.0)
        z = jnp.sin(y)
        z_positive = z > 0
      if z_positive:  # z_positive is usable in Python control flow
        return jnp.sin(x)
      else:
        return jnp.cos(x)

  Here's a real-world example from https://github.com/jax-ml/jax/issues/3974::

    import jax
    import jax.numpy as jnp
    from jax import random

    @jax.jit
    def jax_fn(x):
      with jax.ensure_compile_time_eval():
        y = random.randint(random.key(0), (1000,1000), 0, 100)
      y2 = y @ y
      x2 = jnp.sum(y2) * x
      return x2

  A similar behavior can often be achieved simply by 'hoisting' the constant
  expression out of the corresponding staging API::

    y = random.randint(random.key(0), (1000,1000), 0, 100)

    @jax.jit
    def jax_fn(x):
      y2 = y @ y
      x2 = jnp.sum(y2)*x
      return x2

  But in some cases it can be more convenient to use this context manager.
  """
  with config.eager_constant_folding(True):
    yield

@contextmanager
def eval_context():
  with set_current_trace(eval_trace):
    yield

# TODO(dougalm): deprecate/delete
def full_lower(val):
  if isinstance(val, Tracer):
    return val.full_lower()
  else:
    return val

def get_referent(x: Any) -> Any:
  return x.get_referent() if isinstance(x, Tracer) else x

def same_referent(x: Any, y: Any) -> bool:
  return get_referent(x) is get_referent(y)

def dedup_referents(itr: Iterable[Any]) -> list[Any]:
  return list({HashableWrapper(get_referent(x)):x for x in itr}.values())

def definitely_equal(x, y):
  if isinstance(x, Tracer) or isinstance(y, Tracer):
    return same_referent(x, y)
  elif x is y:
    return True
  try:
    return x == y
  except InconclusiveDimensionOperation:
    return False

# -------------------- abstract values --------------------

class AbstractValue:
  __slots__: list[str] = []

  def to_tangent_aval(self):
    raise NotImplementedError("must override")

  # TODO(dougalm): deprecate this alias
  def at_least_vspace(self):
    return self.to_tangent_aval()

  def __repr__(self):
    try:
      kv_pairs = (f'{k}={v}' for k, v in self.__dict__.items())
      return '{}({})'.format(self.__class__.__name__, ','.join(kv_pairs))
    except AttributeError:
      return self.__class__.__name__

  def update_weak_type(self, weak_type):
    return self

  def strip_weak_type(self) -> AbstractValue:
    return self.update_weak_type(False)

  def normalize(self) -> AbstractValue:
    return self.strip_weak_type()

  def update(self, **kwargs):
    raise NotImplementedError("must override")

  def str_short(self, short_dtypes=False):
    return str(self)

# For type signatures involving dynamic shapes, we use lists of abstract values
# which may contain (reverse) de Bruijn indices in their shapes.
class DBIdx(NamedTuple):
  val: int

@dataclass(frozen=True)
class InDBIdx:
  val: int

@dataclass(frozen=True)
class OutDBIdx:
  val: int

# For annotating input types of callables (i.e. linear_util.WrappedFuns), we use
# a sequence of pairs where the first element of each pair is an AbstractValue
# (possibly containing DBIdx instances in its shape) and the second is a boolean
# indicating whether that argument is explicit (i.e. passed to the callable).
InputType = tuple[tuple[AbstractValue, bool], ...]  # DBIdx in shapes

# For annotating jaxpr output types, we use a sequence of pairs where the first
# element of each pair is an AbstractValue (possibly containing InDBIdx and/or
# OutDBIdx instances in its shape) and the second is a boolean indicating
# whether that argument is explicit (i.e. returned by the callable).
OutputType = tuple[tuple[AbstractValue, bool], ...]  # InDBIdx / OutDBIdx shapes


def _jaxpr_type_to_callable_annotation(jaxpr: Jaxpr) -> InputType:
  idxs = {v: DBIdx(i) for i, v in enumerate((*jaxpr.constvars, *jaxpr.invars))}
  out = [(v.aval.update(shape=tuple(idxs.get(d, d) for d in v.aval.shape))  # type: ignore
          if type(v.aval) is DShapedArray else v.aval, True)
         for v in jaxpr.invars]
  return tuple(out)

# TODO(dougalm): Deprecate. This is here for backwards compat.
def lattice_join(x, y):
  assert typematch(x, y)
  return x

# For use in typing annotations to denote either a Tracer or a `valid_jaxtype`.
Value = Any

def valid_jaxtype(x) -> bool:
  try:
    abstractify(x)
  except TypeError:
    return False
  else:
    return True

def check_valid_jaxtype(x):
  if not valid_jaxtype(x):
    raise TypeError(
      f"Value {x!r} of type {type(x)} is not a valid JAX type")


# We have three flavors of abstractification APIs here which each used to have
# their own separate implementation. Now they're effectively the same, with the
# following differences:
#
# - abstractify returns avals for non-traced array-like objects.
# - get_aval is like abstractify, but also accepts tracers.
# - shaped_abstractify is like get_aval, but also accepts duck-typed arrays.
#
# TODO(jakevdp): can these be unified further?

def shaped_abstractify(x):
  from jax._src.sharding_impls import NamedSharding  # type: ignore

  typ = type(x)
  if (aval_fn := pytype_aval_mappings.get(typ)):  # fast path
    return aval_fn(x)
  for t in typ.__mro__[1:]:
    if (aval_fn := pytype_aval_mappings.get(t)):
      return aval_fn(x)
  if isinstance(x, AbstractValue):
    return x
  if hasattr(x, '__jax_array__'):
    return shaped_abstractify(x.__jax_array__())
  if hasattr(x, 'dtype'):
    aval = ShapedArray(np.shape(x), x.dtype,
                       weak_type=getattr(x, 'weak_type', False))
    if (config.sharding_in_types.value and hasattr(x, 'sharding') and
        isinstance(x.sharding, NamedSharding)):
      return aval.update(sharding=NamedSharding(
          x.sharding.mesh.abstract_mesh,
          x.sharding.spec._normalized_spec(aval.ndim)))
    return aval
  raise TypeError(
      f"Cannot interpret value of type {typ} as an abstract array; it "
      "does not have a dtype attribute")


def abstractify(x):
  if isinstance(x, Tracer):
    raise TypeError(f"Argument '{x}' of type '{type(x)}' is not a valid JAX type")
  return get_aval(x)


def get_aval(x):
  typ = type(x)
  if (aval_fn := pytype_aval_mappings.get(typ)):  # fast path
    return aval_fn(x)
  for t in typ.__mro__[1:]:
    if (aval_fn := pytype_aval_mappings.get(t)):
      return aval_fn(x)
  if hasattr(x, '__jax_array__'):
    return get_aval(x.__jax_array__())
  raise TypeError(f"Argument '{x}' of type '{typ}' is not a valid JAX type")


def is_concrete(x):
  return to_concrete_value(x) is not None

def to_concrete_value(x):
  if isinstance(x, Tracer):
    return x.to_concrete_value()
  else:
    return x

def concretization_function_error(fun, suggest_astype=False):
  fname = getattr(fun, "__name__", fun)
  fname_context = f"The problem arose with the `{fname}` function. "
  if suggest_astype:
    fname_context += ("If trying to convert the data type of a value, "
                      f"try using `x.astype({fun.__name__})` "
                      f"or `jnp.array(x, {fun.__name__})` instead.")
  if fun is bool:
    def error(self, arg):
      raise TracerBoolConversionError(arg)
  elif fun in (hex, oct, operator.index):
    def error(self, arg):
      raise TracerIntegerConversionError(arg)
  else:
    def error(self, arg):
      raise ConcretizationTypeError(arg, fname_context)
  return error

def concrete_or_error(force: Any, val: Any, context=""):
  """Like force(val), but gives the context in the error message."""
  if force is None:
    force = lambda x: x
  if isinstance(val, Tracer):
    maybe_concrete = val.to_concrete_value()
    if maybe_concrete is None:
      raise ConcretizationTypeError(val, context)
    else:
      return force(maybe_concrete)
  else:
    return force(val)

def concrete_dim_or_error(val: Any, context=""):
  """Like concrete_or_error(operator.index), allowing symbolic dimensions."""
  if is_symbolic_dim(val):
    return val
  else:
    return concrete_or_error(operator.index, val, context=context)

### Extended dtypes
#
# Extended dtypes are JAX-specific dtypes that allow us to represent logical
# arrays of element types that do not have an obvious direct correspondence
# to ("physical") arrays of basic types in a compiler. In particular, their
# element types differ from those of XLA and NumPy (e.g. int32). These dtypes
# are only known to JAX. Their implementation is determined by:
# a) an object representing the extended dtype, accessible via the `dtype`
#    attribute on corresponding JAX arrays and, internally, on avals such
#    as ShapedArrays that correspond to such JAX arrays;
# b) a set of rules, available via a private attribute on the extended dtype
#    object in (a).
# The rules in (b) tell JAX internals how to ground out the element
# type for interaction with the compiler and runtime, e.g. when lowering
# to the compiler's language.

@overload
def physical_aval(aval: ShapedArray) -> ShapedArray: ...
@overload
def physical_aval(aval: DShapedArray) -> DShapedArray: ...
@overload                       # TODO(frostig): remove this case
def physical_aval(aval: AbstractValue) -> AbstractValue: ...

def physical_aval(aval):
  if (isinstance(aval, (ShapedArray, DShapedArray)) and
      isinstance(aval.dtype, dtypes.ExtendedDType)):
    elt_aval = physical_element_aval(aval.dtype)
    if isinstance(aval, ShapedArray):
      return ShapedArray((*aval.shape, *elt_aval.shape), elt_aval.dtype)
    return DShapedArray((*aval.shape, *elt_aval.shape), elt_aval.dtype)
  return aval

def physical_element_aval(edtype: dtypes.ExtendedDType) -> ShapedArray:
  duck = edtype._rules.physical_element_aval(edtype)  # type: ignore
  return ShapedArray(duck.shape, dtypes.dtype(duck.dtype))


def _dtype_object(dtype):
  return dtype if isinstance(dtype, dtypes.ExtendedDType) else np.dtype(dtype)

class UnshapedArray(AbstractValue):
  __slots__ = ['dtype', 'weak_type']
  array_abstraction_level = 4

  def __init__(self, dtype, weak_type=False):
    # Is it silly to initialize this object and then complain that we should
    # never create one? Yes. But otherwise pytype complains.
    self.dtype = _dtype_object(dtype)
    self.weak_type = weak_type
    raise Exception("We should never create an UnshapedArray object")

  def __eq__(self, other):
    return (type(self) is type(other) and self.dtype == other.dtype and
            self.weak_type == other.weak_type)

  def __ne__(self, other):
    return not self == other

  def __hash__(self):
    # can use hash(self.dtype) and rely on the fact that numpy reuses base dtype
    # objects, e.g. `np.zeros(3).dtype is np.zeros(4).dtype`, or we can use
    # the unique character code via hash(self.dtype.char)
    return hash((self.dtype, self.weak_type))

  def __repr__(self):
    return '{}({}{})'.format(self.__class__.__name__, self.str_short(),
                             ", weak_type=True" if self.weak_type else "")

  _bool    = concretization_function_error(bool)
  _int     = concretization_function_error(int, True)
  _float   = concretization_function_error(float, True)
  _complex = concretization_function_error(complex, True)
  _hex     = concretization_function_error(hex)
  _oct     = concretization_function_error(oct)
  _index   = concretization_function_error(operator.index)

  def str_short(self, short_dtypes=False) -> str:
    return dtypes.short_dtype_name(self.dtype) if short_dtypes else self.dtype.name

  def update_weak_type(self, weak_type):
    return self.update(weak_type=weak_type)

def _canonicalize_dimension(dim: DimSize) -> DimSize:
  # Dimensions are most commonly integral (by far), so we check that first.
  try:
    return operator.index(dim)
  except TypeError as e:
    type_error = e
  if isinstance(dim, Tracer) and config.dynamic_shapes.value:
    if not (dim.ndim == 0 and (dtypes.issubdtype(dim.dtype, np.integer)
                               or isinstance(dim.dtype, bint))):
      raise TypeError(f"Dimensions must be integer scalars; got {dim.ndim=} {dim.dtype=}")
    return dim
  elif (config.dynamic_shapes.value and isinstance(dim, DArray) and
        type(dim._aval.dtype) is bint and not dim._aval.shape):
    return dim
  elif is_dim(dim):
    return dim
  else:
    raise type_error

def canonicalize_shape(shape: Shape, context: str="") -> tuple[Any, ...]:
  """Canonicalizes and checks for errors in a user-provided shape value.

  Args:
    shape: a Python value that represents a shape.

  Returns:
    A tuple of canonical dimension values.
  """
  if isinstance(shape, int):
    shape = shape,
  try:
    return tuple(unsafe_map(_canonicalize_dimension, shape))
  except TypeError:
    pass
  raise _invalid_shape_error(shape, context)

def canonicalize_dim(d: DimSize, context: str="") -> DimSize:
  """Canonicalizes and checks for errors in a user-provided shape dimension value.

  Args:
    d: a Python value that represents a dimension.

  Returns:
    A canonical dimension value.
  """
  return canonicalize_shape((d,), context)[0]

def _invalid_shape_error(shape: Shape, context: str=""):
  if config.dynamic_shapes.value:
    msg = ("Shapes must be 1D sequences of integer scalars, "
           f"got {shape}")
  else:
    msg = ("Shapes must be 1D sequences of concrete values of integer type, "
           f"got {shape}.")
  if context:
    msg += f" {context}."
  if not config.dynamic_shapes.value and any(
         isinstance(x, Tracer) and isinstance(get_aval(x), ShapedArray)
         and not is_concrete(x) for x in shape):
    msg += ("\nIf using `jit`, try using `static_argnums` or applying `jit` to "
            "smaller subfunctions.")
    for x in shape:
      if isinstance(x, Tracer) and hasattr(x, "_origin_msg"):
        msg += x._origin_msg()

  return TypeError(msg)

# TODO(yashkatariya): Only works with User/Auto. Generalize it to work with
# Collective too.
def modify_spec_for_hidden(spec, mesh) -> P:
  new_spec = []  # type: ignore
  for s in spec:
    if s is None:
      new_spec.append(s)
    else:
      temp_s = s[0] if isinstance(s, tuple) else s
      new_spec.append(
          None if mesh._name_to_type[temp_s] == mesh_lib.AxisTypes.Hidden else s)
  return P(*new_spec)

def _maybe_modify_sharding(sharding):
  if mesh_lib.AxisTypes.Hidden not in sharding.mesh.axis_types:
    return sharding
  new_spec = modify_spec_for_hidden(sharding.spec, sharding.mesh)
  return sharding.with_spec(new_spec)


def get_sharding(sharding, ndim):
  from jax._src.sharding_impls import NamedSharding  # type: ignore

  if sharding is not None:
    if len(sharding.spec) != ndim:
      raise ValueError(
          "Length of sharding.spec must be equal to aval's ndim. Got"
          f" sharding.spec {sharding.spec} and aval.ndim {ndim}")
    return _maybe_modify_sharding(sharding)

  context_mesh = mesh_lib.get_abstract_mesh()
  if not context_mesh:
    raise RuntimeError("Please set the mesh via `jax.set_mesh` API.")
  assert sharding is None
  return NamedSharding(context_mesh, P(*[None] * ndim))


class ShapedArray(UnshapedArray):
  __slots__ = ['shape', 'sharding']  # inherits slots from parent
  array_abstraction_level = 2

  def __init__(self, shape, dtype, weak_type=False, sharding=None):
    self.shape = canonicalize_shape(shape)
    self.dtype = _dtype_object(dtype)
    self.weak_type = weak_type
    if config.sharding_in_types.value:
      self.sharding = get_sharding(sharding, len(self.shape))
      if not isinstance(self.sharding.mesh, mesh_lib.AbstractMesh):
        raise ValueError(
            f"Mesh of an aval must be an AbstractMesh. Got {self.sharding.mesh}")

  def update(self, shape=None, dtype=None, weak_type=None, **kwargs):
    if shape is None:
      shape = self.shape
    if dtype is None:
      dtype = self.dtype
    if weak_type is None:
      weak_type = self.weak_type
    if 'sharding' not in kwargs:
      kwargs['sharding'] = getattr(self, 'sharding', None)
    return ShapedArray(shape, dtype, weak_type, **kwargs)

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self:
                  0 if any(type(d) is int and d == 0 for d in self.shape)
                  else math.prod(self.shape))

  broadcast: ClassVar[aval_method | None] = None
  transpose: ClassVar[aval_method | None] = None
  reshape: ClassVar[aval_method | None] = None
  _iter: ClassVar[staticmethod | None] = None

  def __eq__(self, other):
    return (type(self) is type(other)
            and self.dtype == other.dtype and self.shape == other.shape
            and self.weak_type == other.weak_type
            and getattr(self, 'sharding', None) == getattr(other, 'sharding', None))

  def __hash__(self):
    # can use hash(self.dtype) and rely on the fact that numpy reuses base dtype
    # objects, e.g. `np.zeros(3).dtype is np.zeros(4).dtype`, or we can use
    # the unique character code via hash(self.dtype.char)
    return hash((self.shape, self.dtype, self.weak_type,
                 getattr(self, 'sharding', None)))

  def to_tangent_aval(self):
    if config.sharding_in_types.value:
      return ShapedArray(self.shape, primal_dtype_to_tangent_dtype(self.dtype),
                         self.weak_type, self.sharding)
    else:
      return ShapedArray(self.shape, primal_dtype_to_tangent_dtype(self.dtype),
                         self.weak_type)

  def str_short(self, short_dtypes=False):
    dt_str = (dtypes.short_dtype_name(self.dtype) if short_dtypes else
              self.dtype.name)
    dt_str = dt_str.replace('void', 'float0')
    if hasattr(self, 'sharding') and self.sharding is not None:
      shapestr = _get_shape_sharding_str(self.shape, self.sharding.spec)  # type: ignore
      return f'{dt_str}[{shapestr}]'
    else:
      shapestr = ','.join(map(str, self.shape))
      return f'{dt_str}[{shapestr}]'

  def _len(self, ignored_tracer):
    try:
      return self.shape[0]
    except IndexError as err:
      raise TypeError("len() of unsized object") from err  # same as numpy error


def _get_shape_sharding_str(shape, spec):
  out = []
  for s1, s2 in zip(shape, spec):
    if s2 is None:
      out.append(f"{s1}")
    elif isinstance(s2, tuple):
      ss = ','.join(s for s in s2)
      out.append(f"{s1}@({ss})")
    else:
      out.append(f"{s1}@{s2}")
  return ','.join(out)

def _get_abstract_sharding(val):
  from jax._src.sharding_impls import NamedSharding  # pytype: disable=import-error

  if (config.sharding_in_types.value and hasattr(val, 'sharding') and
      isinstance(val.sharding, NamedSharding)):
    return NamedSharding(val.sharding.mesh.abstract_mesh,
                         val.sharding.spec._normalized_spec(val.ndim))
  return None

def primal_dtype_to_tangent_dtype(primal_dtype):
  if isinstance(primal_dtype, dtypes.ExtendedDType):
    return primal_dtype._rules.tangent_dtype(primal_dtype)
  elif not dtypes.issubdtype(primal_dtype, np.inexact):
    return dtypes.float0
  else:
    return primal_dtype


# Dynamic shape stuff below here! We keep the abstract values distinct just so
# as not to interfere with any static shape machinery.

# We have a convention of reusing AbsractValues as types, even though we could
# make a distinction and use abstract values during tracing only. This reuse
# becomes a bit more extreme with DShapedArrays. A DShapedArray's shape
# attribute is a tuple which can contain several different types: int, DArray
# (scalar and with dtype of bint type), Tracer (while tracing), Var (when used
# as jaxpr type annotations), or DBIdx/InDBIdx/OutDBIdx (when used in InputType
# or OutputType). We could reduce this polymorphism if it seems cleaner, though
# it's kind of convenient!
class DShapedArray(UnshapedArray):
  __slots__ = ['shape']
  shape: tuple[AxisSize, ...]  # noqa: F821
  array_abstraction_level: int = 3

  def __init__(self, shape, dtype, weak_type=False):
    self.shape = shape
    self.dtype = dtype
    self.weak_type = weak_type

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self:
                  0 if any(type(d) is int and d == 0 for d in self.shape)
                  else math.prod(self.shape))

  def str_short(self, short_dtypes=False) -> str:
    del short_dtypes  # ignored
    shape = f'{",".join(str(d) for d in self.shape)}' if self.shape else ''
    dtype = dtypes.short_dtype_name(self.dtype)
    return f'{dtype}[{shape}]'
  __str__ = __repr__ = str_short

  def update(self, shape=None, dtype=None, weak_type=None):
    if shape is None:
      shape = self.shape
    if dtype is None:
      dtype = self.dtype
    if weak_type is None:
      weak_type = self.weak_type
    return DShapedArray(shape, dtype, weak_type)

  def _len(self, tracer):
    return self.shape[0]

  def __eq__(self, other):
    return (type(self) is type(other)
            and self.dtype == other.dtype and self.shape == other.shape
            and self.weak_type == other.weak_type)

  def __hash__(self):
    return hash((self.shape, self.dtype, self.weak_type))

  def to_tangent_aval(self):
    return DShapedArray(self.shape, primal_dtype_to_tangent_dtype(self.dtype),
                        self.weak_type)


class DArray:
  _aval: DShapedArray
  _data: Any  # standard array type
  def __init__(self, aval, data):
    pad_shape = tuple(d.dtype.bound if type(d) is DArray and
                      type(d.dtype) is bint else d for d in aval.shape)
    assert data.shape == pad_shape
    self._aval = aval
    self._data = data

  shape = property(lambda self: self._aval.shape)
  dtype = property(lambda self: self._aval.dtype)
  aval = property(lambda self: self._aval)
  def __repr__(self) -> str:
    if not self.shape and type(self.dtype) is bint:
      # special-case scalar bints
      return f'{int(self._data)}{{≤{self.dtype.bound}}}'

    dtypestr = dtypes.short_dtype_name(self._aval.dtype)
    shapestr = ','.join(map(str, self.shape))
    data = self.data
    return f'{dtypestr}[{shapestr}] with value: {data}'

  def __hash__(self) -> int:
    if not self.shape:
      return hash((self._aval, int(self._data)))
    raise TypeError("unhashable type: DArray")

  def __eq__(self, other):
    if isinstance(other, DArray) and self._aval == other._aval:
      return self._data == other._data
    return False

  def __len__(self):
    return self.shape[0]

  @property
  def data(self):
    if not self.shape and type(self.dtype) is bint:
      # special-case scalar bints
      return self._data

    slices = tuple(
        slice(int(d._data))
        if type(d) is DArray and type(d.dtype) is bint
        else slice(None)
        for d in self.shape
    )
    data = self._data[slices]
    return data

def _darray_aval(x):
  return DShapedArray(x._aval.shape, x._aval.dtype, x._aval.weak_type)

pytype_aval_mappings[DArray] = _darray_aval


@dataclass(frozen=True)
class bint(dtypes.ExtendedDType):
  bound: int

  @property
  def type(self) -> type:
    return dtypes.extended

  @property
  def name(self) -> str:
    return f'bint{{≤{self.bound}}}'

  def __str__(self) -> str:
    return self.name

AxisSize = Union[int, DArray, Tracer, Var, DBIdx, InDBIdx, OutDBIdx]


class MutableArray:
  _aval: ShapedArray
  _buf: Array
  def __init__(self, aval, buf):
    self._aval = aval
    self._buf = buf
  aval = property(lambda self: self._aval)
  shape = property(lambda self: self._aval.shape)
  dtype = property(lambda self: self._aval.dtype)
  def __getitem__(self, idx): return self._aval._getitem(self, idx)
  def __setitem__(self, idx, x): return self._aval._setitem(self, idx, x)
  def __repr__(self) -> str: return 'Mutable' + repr(self[...])
pytype_aval_mappings[MutableArray] = lambda x: x._aval

def mutable_array(init_val):
  return mutable_array_p.bind(init_val)
mutable_array_p = Primitive('mutable_array')
mutable_array_p.ref_primitive = True

class InternalMutableArrayEffect(effects.Effect):
  pass
internal_mutable_array_effect = InternalMutableArrayEffect()
effects.control_flow_allowed_effects.add_type(InternalMutableArrayEffect)

@mutable_array_p.def_effectful_abstract_eval
def mutable_array_abstract_eval(init_aval):
  from jax._src.state.types import AbstractRef  # pytype: disable=import-error
  return AbstractRef(init_aval), {internal_mutable_array_effect}

@mutable_array_p.def_impl
def _mutable_array_impl(init_val):
  from jax._src.state.types import AbstractRef  # pytype: disable=import-error
  aval = get_aval(init_val)
  # TODO(mattjj): improve spelling of 'defensive copy' here, avoid circular dep
  init_val = init_val.copy() if hasattr(init_val, 'copy') else init_val
  return MutableArray(AbstractRef(aval), init_val)

def freeze(ref):
  return freeze_p.bind(ref)
freeze_p = Primitive('freeze')
freeze_p.ref_primitive = True

@freeze_p.def_effectful_abstract_eval
def freeze_abstract_eval(ref_aval):
  return ref_aval.inner_aval, {internal_mutable_array_effect}

@freeze_p.def_impl
def _freeze_impl(ref):
  return ref[()]

class AbstractToken(AbstractValue):
  def str_short(self, short_dtypes=False): return 'Tok'
  def to_tangent_aval(self): return self
abstract_token: AbstractToken = AbstractToken()

# Singleton shaped array used by all abstract tokens when shape/dtype is needed.
token_shaped_array: ShapedArray = ShapedArray((0,), np.dtype(np.bool_))

# Concrete token object
class Token:
  # The underlying data wrapped by the token, could be used to threaded in and
  # out of computations to build up data dependency.
  _buf: Array
  def __init__(self, buf):
    self._buf = buf
  def block_until_ready(self):
    self._buf.block_until_ready()
pytype_aval_mappings[Token] = lambda _: abstract_token


# TODO(dougalm): Deprecate these. They're just here for backwards compat.
def raise_to_shaped(aval):
  return aval
raise_to_shaped_mappings: dict[type, Callable] = {}

### Operations on shapes and dimension sizes.

class InconclusiveDimensionOperation(Exception):
  """Raised when we cannot conclusively compute with symbolic dimensions."""
  pass

def is_symbolic_dim(v: Any) -> bool:
  """Checks if a value is a symbolic dimension used for shape polymorphism.

  This should be used very rarely, because symbolic dimensions overload all
  operators, and should just work.
  """
  return hasattr(v, "dimension_as_value")

def is_constant_dim(d: DimSize) -> bool:
  # Whether the dimension is a static integer constant.
  try:
    operator.index(d)
    return True
  except:
    return False

def is_dim(v: Any) -> bool:
  return is_symbolic_dim(v) or is_constant_dim(v)

def is_constant_shape(s: Shape) -> bool:
  # Whether the shape is a static constant.
  return all(is_constant_dim(d) for d in s)

def definitely_equal_one_of_dim(d1: DimSize, dlist: Sequence[DimSize]) -> bool:
  return any(definitely_equal(d1, d) for d in dlist)

def definitely_equal_shape(s1: Shape, s2: Shape) -> bool:
  """Check that two shapes are guaranteed to be element-wise equal.

  In presence of dynamic shapes may return False even when the shapes may
  be equal at runtime.
  """
  return (len(s1) == len(s2) and
          all(unsafe_map(definitely_equal, s1, s2)))

def divide_shape_sizes(s1: Shape, s2: Shape) -> DimSize:
  """Returns an integer "i" s.t., i * size(s2) == size(s1).
  Raises InconclusiveDimensionOperation if there is no such integer."""
  sz1 = math.prod(s1)
  sz2 = math.prod(s2)
  if definitely_equal(sz1, sz2):  # Takes care of sz1 and sz2 being 0
    return 1
  q, r = divmod(sz1, sz2)
  if isinstance(r, Tracer) or r != 0:
    raise InconclusiveDimensionOperation(
        f"Cannot divide evenly the sizes of shapes {tuple(s1)} and {tuple(s2)}. "
        f"The remainder {r} should be 0.")
  return q

def cancel_divide_tracers(num, denom):
  partition = lambda l: partition_list([isinstance(d, Tracer) for d in l], l)
  num, num_tracers = partition(num)
  denom, denom_tracers = partition(denom)
  if num_tracers or denom_tracers:
    factor = _cancel_divide(num_tracers, denom_tracers)
    if factor is not None:
      size1 = math.prod(num)
      size2 = math.prod(denom)
      if size1 == size2 or size2 != 0:
        return factor * (size1 // size2 if size1 != size2 else 1)

def _cancel_divide(num, denom):
  num = list(num)
  for a in denom:
    i = next((i for i, b in enumerate(num) if definitely_equal(a, b)), None)
    if i is None:
      break  # couldn't cancel
    del num[i]
  else:
    return math.prod(num)

def is_empty_shape(s: Shape) -> bool:
  return any(definitely_equal(d, 0) for d in s)

def dilate_dim(d: DimSize, dilation: DimSize) -> DimSize:
  """max(0, 1 + dilation * (d - 1)).

  Assumes dilation >= 1.
  """
  if definitely_equal(dilation, 1):  # fast path
    return d
  return max_dim(1 + dilation * (d - 1), 0)

def stride_dim(d: DimSize, window_size: DimSize, window_stride: DimSize) -> DimSize:
  """max(0, (d - window_size) // window_stride + 1)

  If d < window_size, returns 0.
  We assume window_size >= 1 and window_stride >= 1.
  """
  # If d < window_size then (d - window_size) // window_stride < 0
  return max_dim((d - window_size) // window_stride + 1, 0)

def min_dim(d1: DimSize, d2: DimSize) -> DimSize:
  """Like min(d1, d2) but for both constant and symbolic dimensions."""
  d1_is_constant = is_constant_dim(d1)
  if d1_is_constant and is_constant_dim(d2):
    return min(d1, d2)
  d1 = concrete_dim_or_error(d1, "argument `d1` of `core.min_dim`")
  d2 = concrete_dim_or_error(d2, "argument `d2` of `core.min_dim`")
  if d1_is_constant:
    return d2.rmin(d1)
  else:
    return d1.min(d2)

def max_dim(d1: DimSize, d2: DimSize) -> DimSize:
  """Like max(d1, d2) but for both constant and symbolic dimensions."""
  d1_is_constant = is_constant_dim(d1)
  if d1_is_constant and is_constant_dim(d2):
      return max(d1, d2)
  d1 = concrete_dim_or_error(d1, "argument `d1` of `core.max_dim`")
  d2 = concrete_dim_or_error(d2, "argument `d2` of `core.max_dim`")
  if d1_is_constant:
    return d2.rmax(d1)
  else:
    return d1.max(d2)

def dimension_as_value(d: DimSize):
  """Turns a dimension size into a JAX array.
     This is the identity function for constant dimensions.

     Has the same abstract value as Python constants.
     """
  if isinstance(d, (int, Tracer, np.int32, np.int64)): return d
  # For shape_poly._DimPolynomial
  if hasattr(d, "dimension_as_value"): return d.dimension_as_value()
  return operator.index(d)

def canonicalize_slice(
    s: slice,
    axis_size: DimSize
  ) -> tuple[DimSize, DimSize, DimSize]:
  """Computes the start index, step, and size of the slice `x[s]`.

  This is similar to `s.indices(axis_size)`, except that it returns
  `(start, step, size)`, and it works when the slice and/or the
  `axis_size` are symbolic.

  See https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding
  """
  def convert_to_index(d: DimSize) -> DimSize:
    # Convert np.array and jax.Array to int, leave symbolic dimensions alone
    try:
      return operator.index(d)
    except:
      return d

  # Must resolve statically if step is {<0, ==0, >0}
  step = convert_to_index(s.step) if s.step is not None else 1
  try:
    if step == 0:
      raise ValueError("slice step cannot be zero")
    step_gt_0 = (step > 0)
  except InconclusiveDimensionOperation as e:
    raise InconclusiveDimensionOperation(
        f"In slice with non-constant elements the step ({step}) must " +
        f"be resolved statically if it is > 0 or < 0.\nDetails: {e}")

  def clamp_index(i: DimSize, which: str):
    try:
      i_ge_0 = (i >= 0)
    except InconclusiveDimensionOperation as e:
      raise InconclusiveDimensionOperation(
          f"In slice with non-constant elements the {which} ({i}) must " +
          f"be resolved statically if it is >= 0.\nDetails: {e}")
    if i_ge_0:
      if step_gt_0:
        return min_dim(axis_size, i)
      else:
        return min_dim(axis_size - 1, i)
    else:
      if step_gt_0:
        return max_dim(0, axis_size + i)
      else:
        return max_dim(-1, axis_size + i)

  if s.start is None:
    start = 0 if step_gt_0 else axis_size - 1
  else:
    start = clamp_index(convert_to_index(s.start), "start")

  if s.stop is None:
    stop = axis_size if step_gt_0 else -1
  else:
    stop = clamp_index(convert_to_index(s.stop), "stop")

  gap = step if step_gt_0 else - step
  distance = (stop - start) if step_gt_0 else (start - stop)
  slice_size = max_dim(0, distance + gap - 1) // gap
  return start, step, slice_size


class SomeTracer:
  __slots__ = ()
  def __repr__(self): return "[dynamic]"

def replace_tracer_for_error_message(obj):
  # TODO(mattjj): Many ideas for improving this.  Crawl the stack and see if
  # there are user variables whose value is == to this object?  Or search
  # parameters of functions being transformed, at least?  Or at least assign
  # short unique ids to them?
  if isinstance(obj, Tracer):
    return SomeTracer()
  else:
    return obj

def evaluate_shape(shape: Shape, dim_vars: Sequence[str],
                   *dim_values: Array) -> Sequence[Array]:
  """Evaluates a shape possibly containing non-constants.

  Args:
    shape: the shape to evaluate.
    dim_vars: the dimension variables names that may appear in `shape`.
    dim_values: the dimension values corresponding to `dim_vars`.

  Returns:
     a tuple of JAX values corresponding to `shape`, of type
     `dim_value_dtype`.
  """
  env = dict(zip(dim_vars, dim_values))
  def eval_one_dim(d: DimSize):
    try:
      return operator.index(d)
    except:
      # Is a _DimExpr
      return d._evaluate(env)  # type: ignore
  return tuple(eval_one_dim(d) for d in shape)

def dim_value_dtype():
  """The dtype to be used for dimension values."""
  return dtypes.canonicalize_dtype(np.int64)

def dim_constant(ct: int):
  dtype = dim_value_dtype()
  assert dtype in (np.int32, np.int64)
  if dtype == np.int32:
    return np.int32(ct)
  elif dtype == np.int64:
    return np.int64(ct)

def dim_value_aval() -> AbstractValue:
  return ShapedArray((), dim_value_dtype(), weak_type=True)

# ------------------- Call -------------------

class CallPrimitive(Primitive):
  multiple_results = True
  call_primitive = True

  def bind_with_trace(self, trace, fun_and_args, params):
    fun = fun_and_args[0]
    args = fun_and_args[1:]
    return trace.process_call(self, fun, args, params)

  def get_bind_params(self, params):
    new_params = dict(params)
    jaxpr = new_params.pop('call_jaxpr')
    subfun = lu.hashable_partial(lu.wrap_init(eval_jaxpr), jaxpr, ())
    if config.dynamic_shapes.value:
      subfun = lu.annotate(subfun, _jaxpr_type_to_callable_annotation(jaxpr))
    return [subfun], new_params

def call_impl(f: lu.WrappedFun, *args, **params):
  del params  # params parameterize the call primitive, not the function
  return f.call_wrapped(*args)

call_p: CallPrimitive = CallPrimitive('call')
call = call_p.bind
call_p.def_impl(call_impl)


class ClosedCallPrimitive(CallPrimitive):
  def get_bind_params(self, params):
    new_params = dict(params)
    jaxpr = new_params.pop('call_jaxpr')
    subfun = lu.wrap_init(partial(eval_jaxpr, jaxpr.jaxpr, jaxpr.consts))
    return [subfun], new_params

closed_call_p: ClosedCallPrimitive = ClosedCallPrimitive('closed_call')
closed_call_p.def_impl(call_impl)
closed_call_p.def_effectful_abstract_eval(
    lambda *_, call_jaxpr: (call_jaxpr.out_avals, call_jaxpr.effects))

# ------------------- Map -------------------

class MapPrimitive(Primitive):
  multiple_results = True
  map_primitive = True

  def bind_with_trace(self, trace, fun_and_args, params):
    fun = fun_and_args[0]
    args = fun_and_args[1:]
    assert len(params['in_axes']) == len(args)
    return trace.process_map(self, fun, args, params)

  def process(self, trace, fun, tracers, params):
    return trace.process_map(self, fun, tracers, params)

  def get_bind_params(self, params):
    new_params = dict(params)
    jaxpr = new_params.pop('call_jaxpr')
    subfun = lu.hashable_partial(lu.wrap_init(eval_jaxpr), jaxpr, ())
    axes = new_params.pop('out_axes')
    new_params['out_axes_thunk'] = HashableFunction(lambda: axes, closure=axes)
    return [subfun], new_params

def mapped_aval(size: AxisSize, axis: int | None,
                aval: AbstractValue) -> AbstractValue:
  handler, _ = aval_mapping_handlers.get(type(aval), (None, None))
  if handler is not None:
    return handler(size, axis, aval)
  else:
    raise TypeError(f"no mapping handler for {aval} of type {type(aval)}")

def unmapped_aval(size: AxisSize, axis_name, axis: int | None,
                  aval: AbstractValue) -> AbstractValue:
  _, handler = aval_mapping_handlers.get(type(aval), (None, None))
  if handler is not None:
    return handler(size, axis_name, axis, aval)
  else:
    raise TypeError(f"no unmapping handler for {aval} of type {type(aval)}")


def _map_shaped_array(
    size: int, axis: int | None, aval: ShapedArray) -> ShapedArray:
  assert axis is None or aval.shape[axis] == size
  # TODO: Extend the named shape
  if axis is None: return aval
  sharding = (aval.sharding.with_spec(tuple_delete(aval.sharding.spec, axis))
              if config.sharding_in_types.value else None)
  return ShapedArray(tuple_delete(aval.shape, axis), aval.dtype,
                     weak_type=aval.weak_type, sharding=sharding)

def _unmap_shaped_array(
    size: int, axis_name: AxisName, axis: int | None, aval: ShapedArray
  ) -> ShapedArray:
  if axis is None: return aval
  elif type(axis) is int:
    sharding = (aval.sharding.with_spec(tuple_insert(aval.sharding.spec, axis, axis_name))
                if config.sharding_in_types.value else None)
    return ShapedArray(tuple_insert(aval.shape, axis, size), aval.dtype,
                       weak_type=aval.weak_type, sharding=sharding)
  else: raise TypeError(axis)

def _map_dshaped_array(
    size: AxisSize, axis: int | None, aval: DShapedArray) -> DShapedArray:
  if axis is None: return aval
  return DShapedArray(tuple_delete(aval.shape, axis), aval.dtype,
                      aval.weak_type)

def _unmap_dshaped_array(
    size: AxisSize, axis_name: AxisName, axis: int | None, aval: DShapedArray
  ) -> DShapedArray:
  if axis is None: return aval
  elif type(axis) is int:
    return DShapedArray(tuple_insert(aval.shape, axis, size), aval.dtype,
                        weak_type=aval.weak_type)
  else:
    raise TypeError(axis)

AvalMapHandlerPair = tuple[Callable, Callable]
aval_mapping_handlers: dict[type, AvalMapHandlerPair] = {
    DShapedArray:   (_map_dshaped_array, _unmap_dshaped_array),
    ShapedArray:   (_map_shaped_array, _unmap_shaped_array),
    AbstractToken: (lambda _, __, a: a, lambda _, __, ___, a: a)
}

# When a mapped function is given no axis name, we generate a name object based
# on the id of the function object. Collisions aren't important because this
# name can't be used in collectives, as user code never gets a ref to this
# object. We don't want to use the function object itself because that might
# persist references to the function object.
# TODO(mattjj): revisit this unique axis name strategy
@total_ordering
class _TempAxisName:

  def __init__(self, obj):
    self.id = id(obj)

  def __repr__(self):
    return f'<axis {hex(self.id)}>'

  def __hash__(self):
    return hash(self.id)

  def __eq__(self, other):
    return type(other) is _TempAxisName and self.id == other.id

  def __lt__(self, other):
    return type(other) is _TempAxisName and self.id < other.id


@dataclass(frozen=True)
class NamedAxisEffect(effects.Effect):
  """A side-effect introducing a new named axis into the current scope."""

  name: AxisName


effects.control_flow_allowed_effects.add_type(NamedAxisEffect)
effects.custom_derivatives_allowed_effects.add_type(NamedAxisEffect)
effects.lowerable_effects.add_type(NamedAxisEffect)
effects.remat_allowed_effects.add_type(NamedAxisEffect)


def filter_named_axis_effects(
    effects: Effects, names: Collection[AxisName]
) -> Effects:
  return {e for e in effects
          if not isinstance(e, NamedAxisEffect) or e.name not in names}


def remove_named_axis_effects(
    jaxpr: Jaxpr, names: Collection[AxisName]
) -> Jaxpr:
  if not names or not jaxpr.effects:
    return jaxpr
  return jaxpr.replace(effects=filter_named_axis_effects(jaxpr.effects, names))

def used_axis_names_jaxpr(jaxpr: Jaxpr | ClosedJaxpr):
  return {e.name for e in jaxpr.effects if isinstance(e, NamedAxisEffect)}

def replace_jaxpr_effects(jaxpr: ClosedJaxpr, effects: Effects):
  return _replace_jaxpr_effects(jaxpr, frozenset(effects))

@weakref_lru_cache
def _replace_jaxpr_effects(jaxpr: ClosedJaxpr, effects: frozenset[Effect]):
  return jaxpr.replace(jaxpr=jaxpr.jaxpr.replace(effects=set(effects)))

# ------------------- Jaxpr checking -------------------

def typecheck(aval: AbstractValue, x) -> bool:
  return typecompat(aval, get_aval(x))

def typecompat(aval_ref: AbstractValue, aval: AbstractValue) -> bool:
  """Determine whether `aval` conforms to `aval_ref`. Ignores weak_type."""
  try:
    return typematch(aval_ref, aval)
  except TypeError:
    return False

def typematch(t1: AbstractValue, t2: AbstractValue) -> bool:
  """Determine whether `t1` and `t2` are equivalent. Ignores weak_type."""
  t1 = t1.normalize()
  t2 = t2.normalize()
  if t1 == t2:
    return True
  elif (isinstance(t1, (ShapedArray, DShapedArray)) and
        isinstance(t2, (ShapedArray, DShapedArray))):
    # This case handles DShapedArray and shape polynomials. Alternatively we
    # could try normalizing first and then doing simple equality.
    return t1.dtype == t2.dtype and definitely_equal_shape(t1.shape, t2.shape)
  else:
    return False

class JaxprTypeError(TypeError): pass

custom_typechecks: dict[Primitive, Callable] = {}

def _check_closed_call(_, *in_atoms, call_jaxpr):
  in_avals = [x.aval for x in in_atoms]
  if not all(map(typecompat, call_jaxpr.in_avals, in_avals)):
    raise JaxprTypeError("Closed call in_avals mismatch")
  return call_jaxpr.out_avals, call_jaxpr.effects
custom_typechecks[closed_call_p] = _check_closed_call

def check_jaxpr(jaxpr: Jaxpr):
  """Checks well-formedness of a jaxpr.

  Specifically, check that:
  - variables that are read are bound beforehand
  - variables are typed equally throughout a jaxpr
  - variable type annotations are compatible with their binding expression

  Raises `JaxprTypeError` if `jaxpr` is determined invalid. Returns `None`
  otherwise.
  """
  @functools.cache
  def ctx_factory():
    ctx = JaxprPpContext()
    pp_settings = JaxprPpSettings()
    try: pp_jaxpr(jaxpr, ctx, pp_settings)  # side-effect on ctx, build variable names
    except: pass
    return ctx, pp_settings

  try:
    _check_jaxpr(ctx_factory, jaxpr)
  except JaxprTypeError as e:
    ctx, pp_settings = ctx_factory()
    if len(e.args) == 2:
      msg, eqnidx = e.args
      jaxpr_str = str(pp_jaxpr_eqn_range(jaxpr, eqnidx - 10, eqnidx + 10, ctx,
                                         pp_settings))
    else:
      msg, = e.args
      jaxpr_str = str(pp_jaxpr_eqn_range(jaxpr, 0, 20, ctx, pp_settings))
    msg = "\n\n".join([msg, "while checking jaxpr:", jaxpr_str])
    raise JaxprTypeError(msg) from None

  # Run key reuse checker after validating jaxpr:
  if config.debug_key_reuse.value:
    # Import here to avoid circular imports
    from jax.experimental.key_reuse._core import check_key_reuse_jaxpr  # pytype: disable=import-error
    check_key_reuse_jaxpr(jaxpr)


def _check_jaxpr(
    ctx_factory: Callable[[], tuple[JaxprPpContext, JaxprPpSettings]],
    jaxpr: Jaxpr
  ) -> None:
  # Use set of variables to types to check that variables are in scope.
  env: set[Var] = set()

  def read(x: Atom) -> Atom:
    # Check the type annotation is itself well-typed.
    check_type(ctx_factory, env, x.aval)
    if isinstance(x, Var):
      # Check the variable is in-scope and consistently typed.
      if x not in env:
        ctx, _ = ctx_factory()
        raise JaxprTypeError(f"Variable '{pp_var(x, ctx)}' not defined")
      return x
    elif isinstance(x, Literal):
      # Check that the literal matches its type annotation.
      if not typecheck(x.aval, x.val):
        ctx, _ = ctx_factory()
        raise JaxprTypeError(
            f"Literal value {x.val} does not match its type annotation "
            f"{pp_aval(x.aval, ctx)}")
      return x
    else:
      assert False, "syntactically invalid jaxpr"

  def write(v: Var, a: AbstractValue) -> None:
    assert isinstance(v, Var), "syntactically invalid jaxpr"
    # Check the type annotation of the binder is itself well-typed.
    check_type(ctx_factory, env, v.aval)
    # Check that the variable is not already bound.
    if v in env:
      ctx, _ = ctx_factory()
      raise JaxprTypeError(f"Variable '{pp_var(v, ctx)}' already bound")
    # Check that the computed type is consistent with the binder annotation.
    if not typematch(v.aval, a):
      ctx, _ = ctx_factory()
      raise JaxprTypeError(
          f"Value for variable '{pp_var(v, ctx)}' inconsistently typed "
          f"as {pp_aval(a, ctx)} for let-binder of type {pp_aval(v.aval, ctx)}")
    # If the variable is not a DropVar, add it to the environment.
    if not isinstance(v, DropVar):
      env.add(v)

  # Check type annotations on lambda binders.
  for v in it.chain(jaxpr.constvars, jaxpr.invars):
    check_type(ctx_factory, env, v.aval)
    write(v, v.aval)

  # Check each eqn.
  sentinel = object()
  in_idx = {v: i for i, v in enumerate(it.chain(jaxpr.constvars, jaxpr.invars))}
  mut_arrays = set()
  for eqn_idx, eqn in enumerate(jaxpr.eqns):
    prim = eqn.primitive
    try:
      in_atoms = map(read, eqn.invars)
      in_avals = [x.aval for x in in_atoms]  # use in_atoms for dyn shapes

      # Compute the type of the primitive application.
      with eqn.ctx.manager:
        if prim in custom_typechecks:
          out_type, eqn_effects = custom_typechecks[prim](
            ctx_factory, *in_atoms, **eqn.params)
        elif prim.call_primitive:
          out_type, eqn_effects = _check_call(ctx_factory, prim, in_atoms,
                                              eqn.params)
        elif prim.map_primitive:
          out_type, eqn_effects = _check_map(ctx_factory, prim, in_avals,
                                            eqn.params)
        else:
          out_type, eqn_effects = check_eqn(prim, in_avals, eqn.params)

      # Check the computed effect type matches the eqn's annotation, and is
      # included in the jaxpr's annotation.
      if prim.ref_primitive:
        if prim is mutable_array_p:
          outvar, = eqn.outvars
          in_idx[outvar] = None  # type: ignore
          mut_arrays.add(outvar)
      if eqn.effects != eqn_effects:
        raise JaxprTypeError("Inferred effects do not match equation effects. "
                             f"Equation effects: {eqn.effects}. "
                             f"Inferred effects: {eqn_effects}")
      for eff in eqn.effects:
        if isinstance(eff, effects.JaxprInputEffect):
          eqn_invar = eqn.invars[eff.input_index]
          if eqn_invar in mut_arrays:
            continue
          if (jaxpr_index := in_idx.get(eqn_invar, sentinel)) is sentinel:
            raise JaxprTypeError(
                "Invalid `JaxprInputEffect`: must correspond to a jaxpr invar")
          jaxpr_effect = eff.replace(input_index=jaxpr_index)
          if jaxpr_effect not in jaxpr.effects:
            raise JaxprTypeError(
                "Invalid `JaxprInputEffect`: must be present in jaxpr. "
                f"{jaxpr_effect} is not in {jaxpr.effects}.")
        elif isinstance(eff, NamedAxisEffect):
          # It is valid for a primitive to discharge the named axis effect.
          continue
        elif eff not in jaxpr.effects:
          raise JaxprTypeError("Equation effect not present in jaxpr effects. "
                               f"Equation effect: {eff}. "
                               f"Jaxpr effects: {jaxpr.effects}")

      # Check out_type matches the let-binders' annotation (after substitution).
      out_type = substitute_vars_in_output_ty(out_type, eqn.invars, eqn.outvars)
      map(write, eqn.outvars, out_type)

    except JaxprTypeError as e:
      ctx, settings = ctx_factory()
      msg, = e.args
      src = source_info_util.summarize(eqn.source_info)
      msg = "\n\n".join([msg, "in equation:", str(pp.nest(2, pp_eqn(eqn, ctx, settings))),
                         f"from source: {src}"])
      raise JaxprTypeError(msg, eqn_idx) from None

  # TODO(mattjj): include output type annotation on jaxpr and check it here
  map(read, jaxpr.outvars)

def check_type(
    ctx_factory: Callable[[], tuple[JaxprPpContext, JaxprPpSettings]],
    env: set[Var],
    ty: AbstractValue,
  ) -> None:
  if isinstance(ty, DShapedArray):
    # Check all elements in the shape tuple are well-typed.
    for d in ty.shape:
      if (isinstance(d, int) or
          isinstance(d, DArray) and not d.shape and type(d.dtype) == bint):
        continue
      elif isinstance(d, Var):
        if d not in env:
          ctx, _ = ctx_factory()
          raise JaxprTypeError(f"unbound axis size: '{pp_var(d, ctx)}'")
        if not isinstance(d.aval, (ShapedArray, DShapedArray)):
          raise JaxprTypeError(f"axis size with unexpected type annotation: "
                               f"{d.aval} of type {type(d.aval)}")
        if isinstance(d.aval, ShapedArray):
          shape, dtype = d.aval.shape, d.aval.dtype
          if shape: raise JaxprTypeError(f"axis size nonscalar: {d.aval}")
          if not dtypes.issubdtype(dtype, np.integer):
            raise JaxprTypeError(f"axis size with non-integer dtype: {d.aval}")
        else:
          assert isinstance(d.aval, DShapedArray)
          shape, dtype = d.aval.shape, d.aval.dtype
          if shape: raise JaxprTypeError(f"axis size nonscalar: {d.aval}")
          if type(dtype) is not bint:
            raise JaxprTypeError(
                f"DArray axis size with non-bint dtype: {d.aval}")
      else:
        raise JaxprTypeError(f"unexpected type in shape: {type(d)}")
  else:
    return  # Except in above case(s), all syntactic forms are valid

def substitute_vars_in_output_ty(
    out_type: Sequence[AbstractValue],  # shapes may contain InDBIdx / OutDBIdx
    in_atoms: Sequence[Atom],
    out_binders: Sequence[Var],
  ) -> list[AbstractValue]:  # shapes may contain Vars
  in_atoms = [x.val if type(x) is Literal else x for x in in_atoms]
  result = []
  for aval in out_type:
    if type(aval) is DShapedArray:
      shape = [in_atoms[d.val] if type(d) is InDBIdx else
               out_binders[d.val] if type(d) is OutDBIdx else
               d for d in aval.shape]
      aval = aval.update(shape=tuple(shape))
    result.append(aval)
  return result

def check_eqn(prim, in_avals, params):
  for jaxpr in jaxprs_in_params(params):
    check_jaxpr(jaxpr)

  out_avals, effects = prim.abstract_eval(*in_avals, **params)
  if not prim.multiple_results:
    out_avals = [out_avals]
  return out_avals, effects

def _check_call(ctx_factory, prim, in_atoms, params):
  if "call_jaxpr" not in params:
    raise JaxprTypeError(
        f"Call primitive {prim} missing 'call_jaxpr' parameter")
  call_jaxpr = params["call_jaxpr"]

  if len(in_atoms) != len(call_jaxpr.invars):
    raise JaxprTypeError(f"Call primitive {prim} with {len(in_atoms)} "
                         f"operands cannot call jaxpr with "
                         f"{len(call_jaxpr.invars)} inputs")

  # Check `call_jaxpr` can be applied to in_atoms.
  env: dict[Var, Atom] = {}
  def substitute(aval: AbstractValue):
    if isinstance(aval, DShapedArray):
      aval = aval.update(shape=tuple(env.get(d, d) for d in aval.shape))  # type: ignore
    return aval
  for v, x in zip(call_jaxpr.invars, in_atoms):
    if not typecompat(substitute(v.aval), x.aval):
      # TODO(mattjj): vars in error message are confusing b/c of Var.__repr__
      raise JaxprTypeError(f"Call primitive {prim} passes operand {x} of type "
                           f"{x.aval} to jaxpr expecting type "
                           f"{substitute(v.aval)}")
    env[v] = x if type(x) is Var else x.val

  _check_jaxpr(ctx_factory, call_jaxpr)

  invars, outvars = call_jaxpr.invars, call_jaxpr.outvars
  in_map : dict[Var,  InDBIdx] = {v:  InDBIdx(i) for i, v in enumerate( invars)}
  out_map: dict[Var, OutDBIdx] = {x: OutDBIdx(i) for i, x in enumerate(outvars)
                                  if type(x) is Var}
  out_avals = [x.aval for x in call_jaxpr.outvars]
  out_type = [a.update(shape=tuple(in_map.get(d, out_map.get(d))
                                   if type(d) is Var else d for d in a.shape))
              if type(a) is DShapedArray else a for a in out_avals]
  return out_type, call_jaxpr.effects

def _check_map(ctx_factory, prim, in_avals, params):
  if "call_jaxpr" not in params:
    raise JaxprTypeError(f"Map primitive {prim} missing 'call_jaxpr' parameter")
  call_jaxpr = params["call_jaxpr"]
  ordered_effects_ = effects.ordered_effects.filter_in(call_jaxpr.effects)
  if ordered_effects_:
    raise JaxprTypeError(
        f"Map primitive {prim} mapping ordered effects: {ordered_effects_}")
  if "axis_size" not in params:
    raise JaxprTypeError(f"Map primitive {prim} missing 'axis_size' parameter")
  axis_size = params["axis_size"]
  if "axis_name" not in params:
    raise JaxprTypeError(f"Map primitive {prim} missing 'axis_name' parameter")
  axis_name = params["axis_name"]
  if "in_axes" not in params:
    raise JaxprTypeError(f"Map primitive {prim} missing 'in_axes' parameter")
  in_axes = params["in_axes"]
  if "out_axes" not in params:
    raise JaxprTypeError(f"Map primitive {prim} missing 'out_axes' parameter")
  out_axes = params["out_axes"]

  binder_avals = [unmapped_aval(axis_size, axis_name, in_axis, v.aval)
                  if in_axis is not None else v.aval
                  for v, in_axis in zip(call_jaxpr.invars, in_axes)]
  for binder_aval, in_aval in zip(binder_avals, in_avals):
    if not typecompat(binder_aval, in_aval):
      raise JaxprTypeError(f"Call primitive {prim} passes operand {in_aval} "
                           f"to jaxpr expecting {binder_aval}")

  with extend_axis_env_nd([(params['axis_name'], axis_size)]):
    _check_jaxpr(ctx_factory, call_jaxpr)

  mapped_out_avals = [v.aval for v in call_jaxpr.outvars]
  out_avals = [unmapped_aval(axis_size, axis_name, out_axis, aval)
               if out_axis is not None else aval
               for aval, out_axis in zip(mapped_out_avals, out_axes)]
  return out_avals, filter_named_axis_effects(call_jaxpr.effects, {axis_name})


# ------------------- Jaxpr printed representation -------------------

def pp_toplevel_jaxpr(jaxpr_to_print, *, source_info=False, print_shapes=True,
                      custom_pp_eqn_rules=True, name_stack=False,
                      print_effects: bool = False) -> pp.Doc:
    context = JaxprPpContext()
    settings = JaxprPpSettings(
        source_info=source_info,
        print_shapes=print_shapes,
        custom_pp_eqn_rules=custom_pp_eqn_rules,
        name_stack=name_stack,
        print_effects=print_effects)

    # Compute how many times each jaxpr is used.
    names = defaultdict[Jaxpr, str](lambda: "jaxpr")
    jaxpr_counts = Counter[Jaxpr]()
    s = deque([jaxpr_to_print])
    while s:
      jaxpr = s.popleft()
      jaxpr_counts[jaxpr] += 1
      for eqn in jaxpr.eqns:
        # TODO(slebedev): Come up with a more elaborate heuristic for name=.
        name = eqn.params.get("name")
        if name is None:
          s.extend(jaxprs_in_params(eqn.params))
          continue
        name = name.strip("<>")  # <lambda> -> lambda
        for subjaxpr in jaxprs_in_params(eqn.params):
          s.append(subjaxpr)
          names.setdefault(subjaxpr, name)

    # Pull jaxprs occurring more than once to the top-level, making sure
    # that their names are unique.
    docs = []
    name_counts = Counter[str]()
    for jaxpr, c in jaxpr_counts.items():
      if c == 1:
        continue
      name = names[jaxpr]
      if (count := name_counts[name]) > 0:
        name_counts[name] += 1
        name += str(count)
        name_counts[name] += 1
      else:
        name_counts[name] += 1
      docs.append(pp_top_level_jaxpr(name, jaxpr, context, settings))
      context.used_names.add(name)
      context.top_level_jaxprs[jaxpr] = name
    docs.append(pp_jaxpr(jaxpr_to_print, context, settings))
    return pp.concat(docs)


class JaxprPpSettings(NamedTuple):
  print_shapes: bool = True
  source_info: bool = False
  name_stack: bool = False
  custom_pp_eqn_rules: bool = True
  print_effects: bool = False

def _encode_digits_alphabetic(n: int) -> str:
  if n == -1:
    return '*'
  s = ''
  while len(s) == 0 or n:
    n, i = n // 26, n % 26
    s = chr(97 + i % 26) + s
  return s

# A JaxprPpContext allows us to globally uniquify variable names within nested
# Jaxprs.
class JaxprPpContext:
  var_names: defaultdict[Var, str]
  used_names: MutableSet[str]
  top_level_jaxprs: MutableMapping[Jaxpr, str]

  def __init__(self) -> None:
    self.top_level_jaxprs = {}
    self.used_names = set()
    fresh_names: Iterator[str] = (
        name
        for i in it.count()
        if (name := _encode_digits_alphabetic(i)) not in self.used_names
    )
    self.var_names = defaultdict(fresh_names.__next__)


def pp_var(v: Var | Literal, context: JaxprPpContext) -> str:
  if isinstance(v, (Literal, DropVar)): return str(v)
  return f"{context.var_names[v]}{v.suffix}"

def pp_aval(a: AbstractValue, context: JaxprPpContext) -> str:
  if isinstance(a, DShapedArray):
    shape = [pp_var(d, context) if type(d) is Var else str(d) for d in a.shape]
    dtype = dtypes.short_dtype_name(a.dtype)
    return f'{dtype}[{",".join(shape)}]'
  else:
    return a.str_short(short_dtypes=True)

def pp_vars(vs: Sequence[Any], context: JaxprPpContext,
            *, separator="", print_shapes: bool = False) -> pp.Doc:
  if print_shapes:
    return pp.nest(2, pp.group(
      pp.join(pp.text(separator) + pp.group(pp.brk()), [
        pp.text(pp_var(v, context)) +
        pp.type_annotation(pp.text(":" + pp_aval(v.aval, context)))
        for v in vs
      ])
    ))
  else:
    return pp.nest(2, pp.group(
      pp.join(pp.text(separator) + pp.group(pp.brk()),
              [pp.text(pp_var(v, context)) for v in vs])
    ))

def pp_kv_pair(k:str, v: Any, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  if type(v) is tuple and all(isinstance(j, (Jaxpr, ClosedJaxpr)) for j in v):
    pp_v = pp_jaxprs(v, context, settings)
  elif isinstance(v, Jaxpr):
    pp_v = pp_jaxpr(v, context, settings)
  elif isinstance(v, ClosedJaxpr):
    pp_v = pp_jaxpr(v.jaxpr, context, settings)
  else:
    pp_v = pp.text(str(v))
  return pp.text(f'{k}=') + pp_v

def pp_kv_pairs(kv_pairs, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  if not kv_pairs:
    return pp.nil()
  return pp.group(
    pp.nest(2, pp.concat([
      pp.text("["),  pp.brk(""),
      pp.join(pp.brk(), [pp_kv_pair(k, v, context, settings) for k, v in kv_pairs])
    ]))
    + pp.brk("") + pp.text("]")
  )

def pp_eqn(eqn: JaxprEqn, context: JaxprPpContext, settings: JaxprPpSettings
           ) -> pp.Doc:
  rule = (_pp_eqn if not settings.custom_pp_eqn_rules else
          pp_eqn_rules.get(eqn.primitive, _pp_eqn))
  doc = rule(eqn, context, settings)
  user_frame = source_info_util.user_frame(eqn.source_info)
  return doc if user_frame is None else pp.source_map(doc, user_frame)

def _pp_eqn(eqn, context, settings, params=None) -> pp.Doc:
  annotation = (source_info_util.summarize(eqn.source_info)
                if settings.source_info else None)
  if params is None:
    params = sorted(eqn.params)
  name_stack_annotation = f'[{eqn.source_info.name_stack}]' if settings.name_stack else None
  lhs = pp_vars(eqn.outvars, context, print_shapes=settings.print_shapes)
  rhs = [pp.text(eqn.primitive.name, annotation=name_stack_annotation),
         pp_kv_pairs([(p, eqn.params[p]) for p in params], context, settings),
         pp.text(" ") + pp_vars(eqn.invars, context)]
  if lhs.format():
    return pp.concat([lhs, pp.text(" = ", annotation=annotation), *rhs])
  else:
    return pp.concat(rhs)
CustomPpEqnRule = Callable[[JaxprEqn, JaxprPpContext, JaxprPpSettings], pp.Doc]
pp_eqn_rules: dict[Primitive, CustomPpEqnRule]  = {}

def pp_eqns(eqns, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  return pp.join(
    pp.brk("; "),
    [pp_eqn(e, context, settings) for e in eqns])

def _compact_eqn_should_include(k: str, v: Any) -> bool:
  if k == 'branches': return False
  if isinstance(v, (Jaxpr, ClosedJaxpr)): return False
  if (isinstance(v, tuple) and
      any(isinstance(e, (Jaxpr, ClosedJaxpr)) for e in v)):
    return False
  return True

def str_eqn_compact(primitive: Primitive, params: dict[Any, Any]) -> str:
  "Compact equation to string conversion used in HLO metadata."
  if primitive in custom_str_eqn_compact_rules:
    return custom_str_eqn_compact_rules[primitive](primitive, params)
  primitive_name = primitive.name
  kvs = " ".join(f"{k}={v}" for k, v in params.items()
                 if _compact_eqn_should_include(k, v))
  return f"{primitive_name}[{kvs}]" if len(kvs) > 0 else primitive_name
custom_str_eqn_compact_rules: dict[
    Primitive, Callable[[Primitive, dict[Any, Any]], str]
] = {}

def pp_jaxpr_skeleton(jaxpr, eqns_fn, context: JaxprPpContext,
                      settings: JaxprPpSettings) -> pp.Doc:
  constvars = pp_vars(jaxpr.constvars, context, print_shapes=settings.print_shapes)
  invars = pp_vars(jaxpr.invars, context, print_shapes=settings.print_shapes)
  eqns = eqns_fn()
  outvars = pp.concat([
    pp.text("("), pp_vars(jaxpr.outvars, context, separator=","),
    pp.text(")" if len(jaxpr.outvars) != 1 else ",)")])
  if settings.print_effects:
    # TODO(sharadmv): render an entire signature here
    eff_text = [pp.text(" : { ")]
    for i, eff in enumerate(jaxpr.effects):
      if i > 0:
        eff_text.append(pp.text(", "))
      if isinstance(eff, effects.JaxprInputEffect):
        index = eff.input_index
        all_vars = [*jaxpr.constvars, *jaxpr.invars]
        eff_text.append(pp_effect(eff.replace(input_index=all_vars[index]),
                                  context))
      else:
        eff_text.append(pp_effect(eff, context))
    eff_text.append(pp.text(" }"))
  else:
    eff_text = []
  return pp.group(pp.nest(2, pp.concat([
    pp.text("{ "), pp.keyword(pp.text("lambda ")),
    constvars, pp.text("; "), invars,
    pp.text(". "), pp.keyword(pp.text("let")),
    pp.nest(2, pp.brk() + eqns), pp.brk(),
    pp.keyword(pp.text("in ")), outvars,
    pp.concat(eff_text)
  ])) + pp.text(" }"))


def pp_top_level_jaxpr(
    name: str,
    jaxpr: Jaxpr,
    context: JaxprPpContext,
    settings: JaxprPpSettings,
) -> pp.Doc:
  return pp.concat([
      pp.text("let " + name + " = "),
      pp_jaxpr(jaxpr, context, settings),
      pp.text(" in"),
      pp.brk(),
  ])


def pp_jaxpr(
    jaxpr: Jaxpr,
    context: JaxprPpContext,
    settings: JaxprPpSettings,
) -> pp.Doc:
  if name := context.top_level_jaxprs.get(jaxpr):
    return pp.text(name)
  eqns_fn = lambda: pp_eqns(jaxpr.eqns, context, settings)
  return pp_jaxpr_skeleton(jaxpr, eqns_fn, context, settings)


def pp_jaxprs(jaxprs, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  jaxprs = [j.jaxpr if isinstance(j, ClosedJaxpr) else j for j in jaxprs]
  return pp.group(pp.nest(2, pp.concat([
      pp.text('('), pp.brk(""),
      pp.join(pp.brk(), map(lambda x: pp_jaxpr(x, context, settings), jaxprs))]
    )) + pp.brk("") + pp.text(')')
  )


def pp_jaxpr_eqn_range(jaxpr: Jaxpr, lo: int, hi: int, context: JaxprPpContext,
                       settings: JaxprPpSettings) -> pp.Doc:
  lo = max(lo, 0)
  hi = max(lo, min(hi, len(jaxpr.eqns)))
  eqns = jaxpr.eqns[lo:hi]
  def eqns_fn():
    pps = []
    if len(eqns) == 0 and len(jaxpr.eqns) != 0:
      pps.append(pp.text('...'))
    else:
      if lo != 0:
        pps.append(pp.text('...'))
      pps.extend(map((lambda e: pp_eqn(e, context, settings)), eqns))
      if hi != len(jaxpr.eqns):
        pps.append(pp.text('...'))
    return pp.join(pp.brk("; "), pps)
  return pp_jaxpr_skeleton(jaxpr, eqns_fn, context, settings)

def pp_effect(effect: Effect, context: JaxprPpContext) -> pp.Doc:
  if hasattr(effect, "_pretty_print"):
    return effect._pretty_print(context)
  return pp.text(str(effect))

# ------------------- Jaxpr util -------------------

def last_used(jaxpr: Jaxpr) -> dict[Var, JaxprEqn | None]:
  """Returns a mapping from every var in jaxpr to what equation uses it last."""
  last_used: dict[Var, JaxprEqn | None] = {
      v: None for v in jaxpr.outvars if not isinstance(v, Literal)}
  for eqn in reversed(jaxpr.eqns):
    for v in eqn.invars:
      if not isinstance(v, Literal) and v not in last_used:
        last_used[v] = eqn
  return last_used

def clean_up_dead_vars(eqn: JaxprEqn, env: dict[Var, Any],
                       last_used: dict[Var, JaxprEqn | None]):
  """Remove all eqn.invars from env if eqn is the last time they were used."""
  for v in {v for v in eqn.invars if not isinstance(v, Literal)}:
    if last_used[v] is eqn:
      # Delete ref to variable when it is no longer needed by next equations.
      del env[v]

# Used in shard_map for converting avals
shard_aval_handlers = {}  # type: ignore
unshard_aval_handlers = {}  # type: ignore

# ----------------- external APIs for querying tracing context -----------------

# TODO(dougalm, jakevdp): expose these via jax.extend

# Comparable object for checking whether JAX's trace state has changed.
class OpaqueTraceState:
  def __init__(self, trace_ref):
    self._trace_ref = trace_ref

  def __eq__(self, other):
    if isinstance(other, OpaqueTraceState):
      return self._trace_ref == other._trace_ref
    else:
      return False

def get_opaque_trace_state(convention):
  del convention
  return OpaqueTraceState(ref(trace_ctx.trace))

def nonempty_axis_env() -> bool:
  return bool(trace_ctx.axis_env.axis_sizes)

def unsafe_am_i_under_a_jit() -> bool:
  return 'DynamicJaxprTrace' in str(unsafe_get_trace_stack(trace_ctx.trace))

def unsafe_am_i_under_a_vmap() -> bool:
  return 'BatchTrace' in str(unsafe_get_trace_stack(trace_ctx.trace))

# TODO(douglam): deprecate/delete
def find_top_trace(_):
  return unsafe_get_current_trace()


def unsafe_get_current_trace():
  return trace_ctx.trace

def unsafe_get_trace_stack(trace):
  if hasattr(trace, "parent_trace"):
    return unsafe_get_trace_stack(trace.parent_trace) + [trace]
  else:
    return [trace]

def unsafe_get_axis_names() -> list[Any]:
  return list(trace_ctx.axis_env.axis_sizes)

# TODO(douglam): deprecate/delete
def axis_frame(axis_name):
  return trace_ctx.axis_env.axis_size(axis_name)
