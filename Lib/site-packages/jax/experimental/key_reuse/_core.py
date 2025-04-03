# Copyright 2023 The JAX Authors.
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

from collections import defaultdict
from collections.abc import Callable, Iterator
from functools import partial, reduce, total_ordering
from typing import Any, NamedTuple

import jax
from jax import lax
from jax import tree_util
from jax.errors import KeyReuseError
from jax.interpreters import batching, mlir
from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src import pjit
from jax._src import prng
from jax._src import random
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import util
from jax._src.ad_checkpoint import remat_p
from jax._src.debugging import debug_callback_p
from jax._src.interpreters import partial_eval as pe
from jax._src.util import weakref_lru_cache

from jax.experimental.shard_map import shard_map_p
import numpy as np


traceback_util.register_exclusion(__file__)

_source_context_message = (
    'PRNG key first used at the above location was subsequently reused'
    ' at the following location:')

def key_reuse_error_with_source_traceback(
    message: str, traceback: source_info_util.Traceback | None) -> KeyReuseError:
  err = KeyReuseError(message)
  if traceback is not None:
    filtered_tb = traceback_util.filter_traceback(traceback.as_python_traceback())
    if filtered_tb:
      context_err = KeyReuseError(_source_context_message).with_traceback(filtered_tb)
      context_err.__context__ = err.__context__
      context_err.__cause__ = err.__cause__
      context_err.__suppress_context__ = err.__suppress_context__
      err.__context__ = None
      err.__cause__ = context_err
  return err


# Create Source() and Sink() objects which validate inputs, have
# correct equality semantics, and are hashable & immutable.
@total_ordering
class _SourceSinkBase:
  idx: int
  mask: bool | np.ndarray

  def __init__(self, idx: int, mask: bool | np.bool_ | np.ndarray = True):
    assert isinstance(idx, int)
    if isinstance(mask, np.ndarray):
      assert mask.dtype == np.dtype('bool')
      if np.all(mask):
        mask = True
      elif not np.any(mask):
        mask = False
      elif mask.flags.writeable:
          mask = np.array(mask, copy=True)
          mask.flags.writeable = False
    elif isinstance(mask, np.bool_):
      mask = bool(mask)
    else:
      assert isinstance(mask, bool)
    super().__setattr__("idx", idx)
    super().__setattr__("mask", mask)

  def __setattr__(self, *args, **kwargs):
    raise ValueError(f"{self.__class__.__name__} is immutable")

  def __eq__(self, other):
    return (self.__class__ == other.__class__
            and self.idx == other.idx
            and np.shape(self.mask) == np.shape(other.mask)
            and np.all(self.mask == other.mask))

  def __lt__(self, other):
    if isinstance(other, Forward):
      return True
    elif isinstance(other, _SourceSinkBase):
      return ((self.__class__.__name__, self.idx)
              < (other.__class__.__name__, other.idx))
    else:
      return NotImplemented

  def __hash__(self):
    if isinstance(self.mask, bool):
      return hash((self.__class__, self.idx, self.mask))
    else:
      mask = np.asarray(self.mask)
      return hash((self.__class__, self.idx, mask.shape,
                   tuple(mask.flatten().tolist())))

  def __repr__(self):
    if self.mask is True:
      return f"{self.__class__.__name__}({self.idx})"
    return f"{self.__class__.__name__}({self.idx}, {self.mask})"


class Sink(_SourceSinkBase):
  pass


class Source(_SourceSinkBase):
  pass


class Forward(NamedTuple):
  in_idx: int
  out_idx: int

  def __repr__(self):
    return f"Forward({self.in_idx}, {self.out_idx})"


# KeyReuseSignature is essentially a frozen set of Source/Sink/Forward
# objects, with a few convenience methods related to key reuse checking.
class KeyReuseSignature:
  _args: frozenset[Source | Sink | Forward]

  def __init__(self, *args):
    self._args = frozenset(args)

  def __repr__(self):
    return f"KeyReuseSignature{tuple(sorted(self._args))}"

  def __eq__(self, other):
    return isinstance(other, KeyReuseSignature) and self._args == other._args

  def __hash__(self):
    return hash(self._args)

  @property
  def sinks(self) -> Iterator[Sink]:
    yield from (s for s in self._args if isinstance(s, Sink))

  @property
  def sources(self) -> Iterator[Source]:
    yield from (s for s in self._args if isinstance(s, Source))

  @property
  def forwards(self) -> Iterator[Forward]:
    yield from (s for s in self._args if isinstance(s, Forward))

  def check_signature(self, *args, funcname="function", context=None):
    for sink in self.sinks:
      key = args[sink.idx]
      if not isinstance(key, prng.PRNGKeyArray):
        continue
      if np.any(key._consumed & sink.mask):
        msg = f"Previously-consumed key passed to {funcname} at index {sink.idx}"
        if context:
          msg += " {context}"
        raise key_reuse_error_with_source_traceback(
            msg, key._source_info and key._source_info.traceback)

  def update_consumption(self, args_in, args_out):
    for sink in self.sinks:
      arg = args_in[sink.idx]
      if isinstance(arg, prng.PRNGKeyArray):
        arg._consumed = arg._consumed | sink.mask
        if np.any(sink.mask):
          arg._source_info = source_info_util.current()
    for arg in args_out:
      if isinstance(arg, prng.PRNGKeyArray):
        arg._consumed = True
    for source in self.sources:
      if isinstance(args_out[source.idx], prng.PRNGKeyArray):
        args_out[source.idx]._consumed = ~np.asarray(source.mask)
    for forward in self.forwards:
      arg_in = args_in[forward.in_idx]
      arg_out = args_out[forward.out_idx]
      if isinstance(arg_in, prng.PRNGKeyArray) and isinstance(arg_out, prng.PRNGKeyArray):
        arg_out._consumed = arg_in._consumed


class DynamicKeyReuseSignature(NamedTuple):
  signature: Callable[[core.JaxprEqn], KeyReuseSignature]

def dynamic_key_reuse_signature(f: Callable[[core.JaxprEqn], KeyReuseSignature]) -> DynamicKeyReuseSignature:
  return DynamicKeyReuseSignature(f)

def key_reuse_signature_from_eqn(eqn: core.JaxprEqn) -> KeyReuseSignature:
  if eqn.primitive in key_reuse_signatures:
    sig = key_reuse_signatures[eqn.primitive]
    if isinstance(sig, KeyReuseSignature):
      return sig
    elif isinstance(sig, DynamicKeyReuseSignature):
      return sig.signature(eqn)
    else:
      raise TypeError(
        f"Unrecognized key reuse sigature of type {type(sig)}: {sig}")
  else:
    return unknown_signature(eqn)


def key_reuse_signature_from_primitive(prim, *args, **params):
  if prim == pjit.pjit_p:
    return jaxpr_type_signature(params['jaxpr'].jaxpr)
  if prim not in key_reuse_signatures:
    # TODO(jakevdp) should we generate an unknown signature here?
    raise RuntimeError(f"Internal: no key reuse rule for primitive {prim}")
  sig = key_reuse_signatures[prim]
  if isinstance(sig, KeyReuseSignature):
    return sig
  elif isinstance(sig, DynamicKeyReuseSignature):
    jaxpr = jax.make_jaxpr(partial(prim.bind, **params))(*args).jaxpr
    return jaxpr_type_signature(jaxpr)
  else:
    raise TypeError(
      f"Unrecognized key reuse sigature of type {type(sig)}: {sig}")


consume_p = core.Primitive("consume")
consume_p.def_impl(lambda x: x)
consume_p.def_abstract_eval(lambda x: x)
batching.defvectorized(consume_p)
mlir.register_lowering(
    consume_p,
    mlir.lower_fun(lambda x: x, multiple_results=False))

def consume(key):
  """Consume the key and return a consumed copy."""
  return consume_p.bind(key)


assert_consumed_value_p = core.Primitive("assert_consumed_value")
assert_consumed_value_p.def_impl(lambda x, *, value: x)
assert_consumed_value_p.def_abstract_eval(lambda x, *, value: x)
batching.defvectorized(assert_consumed_value_p)
mlir.register_lowering(
    assert_consumed_value_p,
    mlir.lower_fun(lambda x, *, value: x, multiple_results=False))

def assert_unconsumed(key):
  """Assert that a key is unconsumed"""
  assert_consumed_value_p.bind(key, value=False)

def assert_consumed(key, value=True):
  """Assert that a key is consumed"""
  assert_consumed_value_p.bind(key, value=value)


def _check_consumed_value(eqn, consumed):
  """Extra check for use with assert_consumed_value_p"""
  expected =  eqn.params['value']
  if not np.all(consumed == expected):
    if np.all(expected):
      raise AssertionError(f"Expected key to be consumed in {eqn}")
    elif not np.any(expected):
      raise AssertionError(f"Expected key to not be consumed in {eqn}")
    else:
      raise AssertionError(f"Expected {expected}, got {consumed} in {eqn}")


# The behavior of most primitives can be described via simple signatures.
key_reuse_signatures: dict[core.Primitive, KeyReuseSignature | DynamicKeyReuseSignature] = {}

key_reuse_signatures[consume_p] = KeyReuseSignature(Sink(0), Forward(0, 0))
key_reuse_signatures[assert_consumed_value_p] = KeyReuseSignature(Forward(0, 0))
key_reuse_signatures[random.random_clone_p] = KeyReuseSignature(Source(0))
key_reuse_signatures[prng.random_bits_p] = KeyReuseSignature(Sink(0))
# TODO(jakevdp): should fold_in sink its input key?
key_reuse_signatures[prng.random_fold_in_p] = KeyReuseSignature(Source(0))
key_reuse_signatures[prng.random_seed_p] = KeyReuseSignature(Source(0))
key_reuse_signatures[prng.random_split_p] = KeyReuseSignature(Sink(0), Source(0))
key_reuse_signatures[random.random_gamma_p] = KeyReuseSignature(Sink(0))
# TODO(jakevdp): broadcast should probably consume the input to avoid implicit duplication
key_reuse_signatures[lax.broadcast_in_dim_p] = KeyReuseSignature(Forward(0, 0))
key_reuse_signatures[lax.copy_p] = KeyReuseSignature(Forward(0, 0))
key_reuse_signatures[lax.convert_element_type_p] = KeyReuseSignature(Forward(0, 0))
key_reuse_signatures[lax.device_put_p] = KeyReuseSignature(Forward(0, 0))
key_reuse_signatures[lax.reshape_p] = KeyReuseSignature(Forward(0, 0))
key_reuse_signatures[lax.squeeze_p] = KeyReuseSignature(Forward(0, 0))
key_reuse_signatures[prng.random_wrap_p] = KeyReuseSignature(Source(0))
# TODO(jakevdp): should unwrap sink its input key?
key_reuse_signatures[prng.random_unwrap_p] = KeyReuseSignature()
key_reuse_signatures[debug_callback_p] = KeyReuseSignature()
key_reuse_signatures[lax.dynamic_slice_p] = KeyReuseSignature(Forward(0, 0))
key_reuse_signatures[lax.dynamic_update_slice_p] = KeyReuseSignature(Sink(1), Forward(0, 0))
key_reuse_signatures[lax.gather_p] = KeyReuseSignature(Forward(0, 0))
key_reuse_signatures[lax.scatter_p] = KeyReuseSignature(Sink(2), Forward(0, 0))
# Equality checks don't consume
key_reuse_signatures[lax.eq_p] = KeyReuseSignature()
key_reuse_signatures[lax.ne_p] = KeyReuseSignature()

# The default signature will Sink all key inputs, and not Source any.
def unknown_signature(eqn):
  def is_key(var: core.Atom):
    return hasattr(var.aval, "dtype") and jax.dtypes.issubdtype(var.aval.dtype, jax.dtypes.prng_key)
  return KeyReuseSignature(
    *(Sink(idx) for idx, var in enumerate(eqn.invars) if is_key(var))
  )

@weakref_lru_cache
def jaxpr_type_signature(jaxpr: core.Jaxpr) -> KeyReuseSignature:
  """Parse the jaxpr to determine key reuse signature"""
  consumed: dict[core.Atom, bool | np.ndarray] = {}
  forwards: dict[core.Atom, core.Atom] = {}  # map forwarded outputs to inputs.

  def resolve_forwards(var: core.Atom) -> core.Atom:
    if not forwards:
      return var
    for _ in range(len(forwards) + 1):
      if isinstance(var, core.Literal):
        return var
      if var in forwards:
        var = forwards[var]
      else:
        return var
    raise ValueError("forwarding cycle detected")

  def is_key(var: core.Atom):
    return hasattr(var.aval, "dtype") and jax.dtypes.issubdtype(var.aval.dtype, jax.dtypes.prng_key)

  def sink(var: core.Atom, mask=True):
    if not is_key(var):
      return
    var = resolve_forwards(var)
    assert not isinstance(var, core.Literal)
    if np.any(np.logical_and(consumed.get(var, False), mask)):
      return True
    consumed[var] = np.logical_or(consumed.get(var, False), mask)

  def source(var: core.Atom, mask=False):
    if not is_key(var):
      return
    var = resolve_forwards(var)
    assert not isinstance(var, core.Literal)
    consumed[var] = mask

  def is_consumed(var: core.Atom):
    var = resolve_forwards(var)
    if isinstance(var, core.Literal):
      return False
    return consumed.get(var, False)

  for eqn in jaxpr.eqns:
    traceback = eqn.source_info.traceback
    name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
    with source_info_util.user_context(traceback, name_stack=name_stack):
      signature = key_reuse_signature_from_eqn(eqn)
      if eqn.primitive == assert_consumed_value_p:
        # This is a special case that goes beyond normal key reuse logic.
        _check_consumed_value(eqn, is_consumed(eqn.invars[0]))

      for in_idx, out_idx in signature.forwards:
        forwards[eqn.outvars[out_idx]] = eqn.invars[in_idx]

      for snk in signature.sinks:
        if not 0 <= snk.idx < len(eqn.invars):
          raise KeyReuseError(f"In {eqn.primitive}, sink {snk.idx} out of range [0, {len(eqn.invars)}]")
        if sink(eqn.invars[snk.idx], snk.mask):
          raise KeyReuseError(f"In {eqn.primitive}, argument {snk.idx} is already consumed.")
      for var in eqn.outvars:
        if not isinstance(var, core.Literal) and var not in forwards:
          source(var, True)  # consumed unless in a Source.
      for src in signature.sources:
        if not 0 <= src.idx < len(eqn.outvars):
          raise KeyReuseError(f"In {eqn.primitive}, source {src.idx} out of range [0, {len(eqn.outvars)}]")
        source(eqn.outvars[src.idx])

  all_inputs = [*jaxpr.invars, *jaxpr.constvars]
  return KeyReuseSignature(
    *(Sink(i, consumed[v]) for i, v in enumerate(all_inputs)
      if is_key(v) and np.any(consumed.get(v, False))),
    *(Source(i) for i, v in enumerate(jaxpr.outvars)
      if is_key(v) and resolve_forwards(v) not in all_inputs and not consumed.get(v, False)),
    *(Forward(all_inputs.index(resolve_forwards(outvar)), idx_out)  # type: ignore[arg-type]
      for idx_out, outvar in enumerate(jaxpr.outvars)
      if is_key(outvar) and resolve_forwards(outvar) in all_inputs)
  )


def function_type_signature(fun: Callable[..., Any], *args: Any) -> KeyReuseSignature:
  args_flat, in_tree = tree_util.tree_flatten(args)
  in_avals_flat = [core.get_aval(arg) for arg in args_flat]
  wrapped_fun, _ = api_util.flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals_flat)
  return jaxpr_type_signature(jaxpr)


def check_key_reuse_jaxpr(jaxpr: core.Jaxpr) -> None:
  """Check the jaxpr for key reuse."""
  jaxpr_type_signature(jaxpr)


def check_key_reuse(fun: Callable[..., Any], /, *args: Any) -> None:
  """Function to statically check key reuse."""
  function_type_signature(fun, *args)


#----------------------------------------------------------------------------------
# key reuse rules for particular primitives:

@dynamic_key_reuse_signature
def _slice_signature(eqn):
  in_aval = eqn.invars[0].aval
  if not jax.dtypes.issubdtype(in_aval.dtype, jax.dtypes.prng_key):
    return KeyReuseSignature(Forward(0, 0))
  if any(core.is_symbolic_dim(s) for s in in_aval.shape):
    return KeyReuseSignature(Forward(0, 0))
  start_indices = eqn.params['start_indices']
  limit_indices = eqn.params['limit_indices']
  strides = eqn.params['strides'] or (1,) * len(start_indices)
  idx = tuple(slice(*tup) for tup in util.safe_zip(start_indices, limit_indices, strides))
  sink = np.zeros(in_aval.shape, dtype=bool)
  sink[idx] = True
  return KeyReuseSignature(Sink(0, sink), Source(0))

key_reuse_signatures[lax.slice_p] = _slice_signature

@dynamic_key_reuse_signature
def _concatenate_signature(eqn):
  num_vals = len(eqn.invars)
  # TODO(jakevdp): should this signature be more granular?
  if num_vals == 1:
    return KeyReuseSignature(Forward(0, 0))
  else:
    return KeyReuseSignature(*(Sink(i) for i in range(num_vals)), Source(0))

key_reuse_signatures[lax.concatenate_p] = _concatenate_signature

@dynamic_key_reuse_signature
def _pjit_key_type_signature(eqn):
  return jaxpr_type_signature(eqn.params['jaxpr'].jaxpr)

key_reuse_signatures[pjit.pjit_p] = _pjit_key_type_signature

@dynamic_key_reuse_signature
def _shard_map_type_signature(eqn):
  return jaxpr_type_signature(eqn.params['jaxpr'])

key_reuse_signatures[shard_map_p] = _shard_map_type_signature

@dynamic_key_reuse_signature
def _cond_key_type_signature(eqn):
  signatures = [jaxpr_type_signature(branch.jaxpr) for branch in eqn.params['branches']]
  sinks = defaultdict(list)
  sources = defaultdict(list)
  for sig in signatures:
    for sink in sig.sinks:
      sinks[sink.idx].append(sink.mask)
    for source in sig.sources:
      sources[source.idx].append(source.mask)

  combined_sinks = [Sink(i + 1, reduce(np.logical_or, m)) for i, m in sinks.items()]
  combined_sources = [Source(i, reduce(np.logical_and, m)) for i, m in sources.items()]
  combined_forwards = [Forward(f.in_idx + 1, f.out_idx) for f in
                       set.intersection(*(set(sig.forwards) for sig in signatures))]
  return KeyReuseSignature(*combined_sinks, *combined_sources, *combined_forwards)

key_reuse_signatures[lax.cond_p] = _cond_key_type_signature

@dynamic_key_reuse_signature
def _scan_key_type_signature(eqn):
  jaxpr = eqn.params['jaxpr'].jaxpr
  num_consts = eqn.params['num_consts']
  num_carry = eqn.params['num_carry']
  signature = jaxpr_type_signature(jaxpr)

  # scan body should not consume key in constants
  if any(np.any(s.mask) for s in signature.sinks if s.idx < num_consts):
    raise KeyReuseError("scan body function leads to key reuse when repeatedly executed, "
                        "because key constants are repeatedly consumed:\n"
                        f"  {signature=}\n"
                        f"  {eqn=}\n"
                        f"  {jaxpr=}")

  # scan carry should only consume keys that are sourced on output.
  carry_sinks = {s.idx - num_consts: s.mask for s in signature.sinks
                 if 0 <= s.idx - num_consts < num_carry and np.any(s.mask)}
  carry_sources = {s.idx: s.mask for s in signature.sources
                   if 0 <= s.idx < num_carry and np.any(s.mask)}
  if not set(carry_sinks).issubset(set(carry_sources)):  # TODO(jakevdp): check that masks match
    raise KeyReuseError("scan body function leads to key reuse when repeatedly executed, "
                        "because consumed inputs don't match sourced outputs:\n"
                        f"  {signature=}\n"
                        f"  {eqn=}\n"
                        f"  {jaxpr=}")
  return signature

key_reuse_signatures[jax.lax.scan_p] = _scan_key_type_signature

@dynamic_key_reuse_signature
def _while_key_type_signature(eqn):
  cond_jaxpr = eqn.params['cond_jaxpr'].jaxpr
  cond_nconsts = eqn.params['cond_nconsts']
  body_jaxpr = eqn.params['body_jaxpr'].jaxpr
  body_nconsts = eqn.params['body_nconsts']

  cond_signature = jaxpr_type_signature(cond_jaxpr)
  body_signature = jaxpr_type_signature(body_jaxpr)

  # Error if there are sinks among consts.
  if any(np.any(s.mask) for s in cond_signature.sinks if s.idx < cond_nconsts):
    raise KeyReuseError("while_loop cond function leads to key reuse when repeatedly executed: "
                        f"  {cond_signature=}\n"
                        f"  {eqn=}")
  if any(np.any(s.mask) for s in body_signature.sinks if s.idx < body_nconsts):
    raise KeyReuseError("while_loop body function leads to key reuse when repeatedly executed: "
                        f"  {body_signature=}\n"
                        f"  {eqn=}")

  # carry should only consume keys that are sourced on output.
  body_carry_sinks = {s.idx - body_nconsts: s.mask for s in body_signature.sinks if s.idx >= body_nconsts}
  cond_carry_sinks = {s.idx - cond_nconsts: s.mask for s in cond_signature.sinks if s.idx >= cond_nconsts}
  carry_sources = {s.idx: s.mask for s in body_signature.sources}
  # TODO(jakevdp): check masks at each index?
  if not (cond_carry_sinks.keys() <= carry_sources.keys()):
    raise KeyReuseError("while_loop cond function leads to key reuse when repeatedly executed: "
                        f"  {cond_signature=}\n"
                        f"  {eqn=}")
  if not (body_carry_sinks.keys() <= carry_sources.keys()):
    raise KeyReuseError("while_loop body function leads to key reuse when repeatedly executed: "
                        f"  {body_signature=}\n"
                        f"  {eqn=}")
  if body_carry_sinks.keys() & cond_carry_sinks.keys():
    raise KeyReuseError("while_loop cond and body functions both use the same key: "
                        f"  {cond_signature=}\n"
                        f"  {body_signature=}\n"
                        f"  {eqn=}")
  return body_signature

key_reuse_signatures[jax.lax.while_p] = _while_key_type_signature

@dynamic_key_reuse_signature
def _remat_key_type_signature(eqn):
  # The assumption here is that the non-differentiated pass contains all relevant
  # key usage, and the differentiated pass
  #  1) will only consume keys that are already consumed in the non-differentiated pass
  #  2) will never create keys
  # Therefore, the differentiated pass is a no-op.
  if eqn.params['differentiated']:
    return KeyReuseSignature()
  return jaxpr_type_signature(eqn.params['jaxpr'])

key_reuse_signatures[remat_p] = _remat_key_type_signature


def call_impl_with_key_reuse_checks(prim: core.Primitive, raw_impl: Callable[..., Any], *args, **kwargs) -> Any:
  if prim not in key_reuse_signatures:
    # TODO(jakevdp): should we use an unknown signature here?
    return raw_impl(*args, **kwargs)
  signature = key_reuse_signature_from_primitive(prim, *args, **kwargs)
  funcname = "jit-compiled function" if prim == pjit.pjit_p else str(prim)
  consts = kwargs['jaxpr'].consts if prim == pjit.pjit_p else []
  signature.check_signature(*args, *consts, funcname=funcname)
  result = raw_impl(*args, **kwargs)
  signature.update_consumption([*args, *consts], result if prim.multiple_results else [result])
  return result
