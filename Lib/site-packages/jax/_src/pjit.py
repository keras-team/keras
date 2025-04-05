# Copyright 2021 The JAX Authors.
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
from collections.abc import Callable, Sequence, Iterable
import contextlib
import dataclasses
from functools import partial
import inspect
import logging
import operator as op
import weakref
from typing import NamedTuple, Any, Union, cast
import warnings

import numpy as np

from jax._src import api
from jax._src import ad_util
from jax._src import api_util
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import mesh as mesh_lib
from jax._src import op_shardings
from jax._src import profiler
from jax._src import sharding_impls
from jax._src import source_info_util
from jax._src import stages
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.api_util import (
  argnums_partial_except, flatten_axes, flatten_fun, flatten_fun_nokwargs,
  donation_vector, check_callable, resolve_argnums,
  argnames_partial_except, debug_info, result_paths, add_jaxpr_debug_info,
  hoist_obj_attrs, _check_no_aliased_ref_args,
  _check_no_aliased_closed_over_refs)
from jax._src.interpreters import partial_eval as pe
from jax._src.partition_spec import PartitionSpec
from jax._src.interpreters import xla
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib import jax_jit
from jax._src.lib import xla_client as xc
from jax._src.mesh import AbstractMesh
from jax._src.sharding import Sharding
from jax._src.sharding_impls import (
    NamedSharding, GSPMDSharding,
    SingleDeviceSharding, PmapSharding, AUTO, UNSPECIFIED, UnspecifiedValue,
    ParsedPartitionSpec, get_single_pspec, prepare_axis_resources,
    parse_flatten_op_sharding, canonicalize_sharding)
from jax._src.layout import Layout, DeviceLocalLayout, AutoLayout
from jax._src.state import discharge as state_discharge, RefEffect, AbstractRef
from jax._src.traceback_util import api_boundary
from jax._src.tree_util import (
    tree_flatten, tree_unflatten, treedef_is_leaf, tree_structure, tree_leaves,
    treedef_children, broadcast_prefix, all_leaves, prefix_errors, keystr,
    PyTreeDef, none_leaf_registry as none_lr, tree_map)
from jax._src.util import (
    HashableFunction, safe_map, safe_zip, wraps,
    distributed_debug_log, split_list, weakref_lru_cache,
    merge_lists, subs_list, fun_name, fun_qual_name)

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

traceback_util.register_exclusion(__file__)

PjitSharding = Union[GSPMDSharding, UnspecifiedValue, AUTO]
PjitShardingMinusUnspecified = Union[GSPMDSharding, AUTO]
MeshSharding = Union[NamedSharding, UnspecifiedValue, AUTO]
MeshShardingMinusUnspecified = Union[NamedSharding, AUTO]

logger = logging.getLogger(__name__)


def _find_arg_mismatch(arg_list, fails, fun_name):
  mismatched_args_msg = []
  def mismatch(err):
    for name, inp_da, aval in arg_list:
      if err.m_type == pxla.MismatchType.ARG_SHARDING and err.da == inp_da:
        mismatched_args_msg.append(
            f"argument {name} of {fun_name} with shape {aval.str_short()} and "
            f"{err._dev_ids_plat_str}")
        break
  first_err, second_err = fails
  mismatch(first_err)
  mismatch(second_err)
  return mismatched_args_msg


def _device_assignment_mismatch_error(fun_name, fails, args_flat, api_name,
                                      arg_names):
  arg_list = []
  if arg_names is None:
    arg_names = [''] * len(args_flat)
  for a, n in zip(args_flat, arg_names):
    da = (a.sharding._device_assignment
          if getattr(a, 'sharding', None) is not None else None)
    arg_list.append((n, da, core.shaped_abstractify(a)))

  mismatched_args_msg = _find_arg_mismatch(arg_list, fails, fun_name)

  if len(mismatched_args_msg) == 2:
    first, second = mismatched_args_msg  # pytype: disable=bad-unpacking
    extra_msg = f" Got {first} and {second}"
  elif len(mismatched_args_msg) == 1:
    first, second  = fails
    # Choose the failure left which is not already covered by ARG_SHARDING.
    left = second if first.m_type == pxla.MismatchType.ARG_SHARDING else first
    extra_msg = f" Got {mismatched_args_msg[0]} and{left._str(api_name)}"
  else:
    first, second = fails
    extra_msg = f" Got{first._str(api_name)} and{second._str(api_name)}"
  msg = (f"Received incompatible devices for {api_name}ted computation.{extra_msg}")
  return msg


class PjitInfo(NamedTuple):
  """Things that we know about a jit instance before it is called.

  In other words, this structure contains arguments to jit()/pjit(),
  preprocessed and validated.
  """
  fun_sourceinfo: str | None
  fun_signature: inspect.Signature | None
  # Shardings, as specified by the user. These can either be UNSPECIFIED or they
  # can be a tree (prefix) of shardings or None.
  user_specified_in_shardings: bool
  in_shardings_treedef: PyTreeDef
  in_shardings_leaves: tuple[Any, ...]
  out_shardings_treedef: PyTreeDef
  out_shardings_leaves: tuple[Any, ...]
  in_layouts_treedef: PyTreeDef
  in_layouts_leaves: tuple[Any, ...]
  out_layouts_treedef: PyTreeDef
  out_layouts_leaves: tuple[Any, ...]
  static_argnums: tuple[int, ...]
  static_argnames: tuple[str, ...]
  donate_argnums: tuple[int, ...]
  donate_argnames: tuple[str, ...]
  device: xc.Device | None
  backend: str | None
  keep_unused: bool
  inline: bool
  abstracted_axes: Any | None
  use_resource_env: bool  # False for jit, True for pjit
  compiler_options_kvs: tuple[tuple[str, Any], ...]

  # Hash and compare PjitInfo by identity when used as a cache key.
  def __hash__(self):
    return id(self)

  def __eq__(self, other):
    return self is other


def _python_pjit_helper(fun, jit_info, *args, **kwargs):
  p, args_flat = _infer_params(fun, jit_info, args, kwargs)

  for arg in args_flat:
    dispatch.check_arg(arg)

  if p.attrs_tracked:
    init_states = _get_states(p.attrs_tracked)
    args_flat = [*init_states, *args_flat]

  try:
    # TODO(yashkatariya): Maybe thread this into pjit params like resource_env
    # and set the context manager down the stack?
    with mesh_lib.set_abstract_mesh(p.abstract_mesh):
      if (core.trace_state_clean() and
          not config.debug_key_reuse.value and
          not config.data_dependent_tracing_fallback.value):
        args_flat = map(core.full_lower, args_flat)
        core.check_eval_args(args_flat)
        out_flat, compiled, profiler = _pjit_call_impl_python(*args_flat, **p.params)
      else:
        out_flat = pjit_p.bind(*args_flat, **p.params)
        compiled = None
        profiler = None
  except pxla.DeviceAssignmentMismatchError as e:
    fails, = e.args
    api_name = 'jit' if p.params['resource_env'] is None else 'pjit'
    fun_name = getattr(fun, '__qualname__', getattr(fun, '__name__', str(fun)))
    msg = _device_assignment_mismatch_error(
        fun_name, fails, args_flat, api_name, p.arg_names)
    raise ValueError(msg) from None
  except xla.InvalidInputException as e:
    arg_names = [''] * len(args_flat) if p.arg_names is None else p.arg_names
    # Run canonicalization again to figure out which arg failed.
    if p.params['jaxpr'].consts:
      raise TypeError(e.args[0]) from e
    else:
      for arg, name, aval in zip(args_flat, arg_names, p.in_avals):
        try:
          xla.canonicalize_dtype(arg)
        except xla.InvalidInputException as _:
          # Reraise as TypeError with the new message.
          raise TypeError(
              f"Argument '{name}' of shape {aval.str_short()} of type"
              f' {type(arg)} is not a valid JAX type.') from e
      raise AssertionError("Unreachable") from e

  if p.attrs_tracked:
    num_states_out = sum(end_tree.num_leaves for _, end_tree, _ in p.attrs_tracked)
    final_states, out_flat = split_list(out_flat, [num_states_out])
    _set_states(p.attrs_tracked, final_states)

  outs = tree_unflatten(p.out_tree, out_flat)
  return (outs, out_flat, p.out_tree, args_flat, p.params['jaxpr'],
          p.attrs_tracked, compiled, profiler)


def _set_states(attrs_tracked, vals):
  from jax.experimental.attrs import jax_setattr
  valss = split_list(vals, [td.num_leaves for _, td, _ in attrs_tracked[:-1]])
  for ((_, treedef, (obj, attr)), leaves) in zip(attrs_tracked, valss):
    val = tree_unflatten(treedef, leaves)
    jax_setattr(obj, attr, val)

def _get_states(attrs_tracked):
  from jax.experimental.attrs import jax_getattr
  vals = []
  for treedef, _, (obj, attr) in attrs_tracked:
    tree = jax_getattr(obj, attr)
    leaves, treedef_ = tree_flatten(tree)
    assert treedef == treedef_
    vals.extend(leaves)
  return vals

def _need_to_rebuild_with_fdo(pgle_profiler):
  return (pgle_profiler is not None and pgle_profiler.is_enabled()
          and not pgle_profiler.is_fdo_consumed())

def _get_fastpath_data(
    executable, out_tree, args_flat, out_flat, attrs_tracked, effects,
    consts, abstracted_axes, pgle_profiler
) -> pxla.MeshExecutableFastpathData | None:
  out_reflattened, out_tree = pxla.reflatten_outputs_for_dispatch(out_tree, out_flat)

  use_fastpath = (
      executable is not None
      and isinstance(executable, pxla.MeshExecutable)
      and isinstance(executable.unsafe_call, pxla.ExecuteReplicated)
      # No effects in computation
      and not executable.unsafe_call.ordered_effects
      and not executable.unsafe_call.has_unordered_effects
      and not executable.unsafe_call.has_host_callbacks
      and all(isinstance(x, xc.ArrayImpl) for x in out_reflattened)
      and abstracted_axes is None
      # no attr state effects
      and not attrs_tracked
      # no ref state effects
      and not any(isinstance(e, RefEffect) for e in effects)
      # no prng reuse checking
      and not (config.debug_key_reuse.value and any(
        hasattr(arg, 'dtype') and dtypes.issubdtype(arg.dtype, dtypes.prng_key)
        for arg in (*args_flat, *out_flat, *consts)))
      and not _need_to_rebuild_with_fdo(pgle_profiler)
      )

  if use_fastpath:
    out_avals = [o.aval for o in out_reflattened]
    out_committed = [o._committed for o in out_reflattened]
    kept_var_bitvec = [i in executable._kept_var_idx
                       for i in range(len(args_flat))]
    in_shardings = [
        sharding_impls.physical_sharding(a, s)
        if a is not core.abstract_token and dtypes.issubdtype(a.dtype, dtypes.extended)
        else s
        for s, a in zip(executable._in_shardings, executable.in_avals)
    ]
    fastpath_data = pxla.MeshExecutableFastpathData(
        executable.xla_executable, out_tree, in_shardings,
        executable._out_shardings, out_avals, out_committed, kept_var_bitvec,
        executable._dispatch_in_layouts)
  else:
    fastpath_data = None
  return fastpath_data


def _cpp_pjit_evict_fn(self):
  self._clear_cache()
  _create_pjit_jaxpr.evict_function(self._fun)  # pytype: disable=attribute-error
  _infer_params_cached.cache_clear()


# The entries are doubled here from the default 4096 because _pjit_call_impl
# also has a cpp dispatch path and that would double the number of entries in
# the global shared cache.
# This cache is only used for jit's with only fun. For example: jax.jit(f)
_cpp_pjit_cache_fun_only = xc._xla.PjitFunctionCache(capacity=8192)

# This cache is used for jit where extra arguments are defined other than the
# fun. For example: jax.jit(f, donate_argnums=...) OR
# jax.jit(f, out_shardings=...), etc. We don't use the same cache because the
# capacity might get full very fast because of all the jitted function in JAX
# which might evict train_step for example.
_cpp_pjit_cache_explicit_attributes = xc._xla.PjitFunctionCache(capacity=8192)


def _get_cpp_global_cache(contains_explicit_attributes: bool):
  if contains_explicit_attributes:
    return _cpp_pjit_cache_explicit_attributes
  else:
    return _cpp_pjit_cache_fun_only


def _cpp_pjit(fun: Callable, jit_info: PjitInfo):

  @api_boundary
  def cache_miss(*args, **kwargs):
    if config.no_tracing.value:
      raise RuntimeError(f"re-tracing function {jit_info.fun_sourceinfo} for "
                         "`jit`, but 'no_tracing' is set")

    (outs, out_flat, out_tree, args_flat, jaxpr, attrs_tracked, executable,
     pgle_profiler) = _python_pjit_helper(fun, jit_info, *args, **kwargs)

    maybe_fastpath_data = _get_fastpath_data(
        executable, out_tree, args_flat, out_flat, attrs_tracked, jaxpr.effects,
        jaxpr.consts, jit_info.abstracted_axes,
        pgle_profiler)

    return outs, maybe_fastpath_data, _need_to_rebuild_with_fdo(pgle_profiler)

  cache_key = pxla.JitGlobalCppCacheKeys(
      donate_argnums=jit_info.donate_argnums,
      donate_argnames=jit_info.donate_argnames,
      device=jit_info.device, backend=jit_info.backend,
      in_shardings_treedef=jit_info.in_shardings_treedef,
      in_shardings_leaves=jit_info.in_shardings_leaves,
      out_shardings_treedef=jit_info.out_shardings_treedef,
      out_shardings_leaves=jit_info.out_shardings_leaves,
      in_layouts_treedef=jit_info.in_layouts_treedef,
      in_layouts_leaves=jit_info.in_layouts_leaves,
      out_layouts_treedef=jit_info.out_layouts_treedef,
      out_layouts_leaves=jit_info.out_layouts_leaves,
      use_resource_env=jit_info.use_resource_env,
      compiler_options_kvs=jit_info.compiler_options_kvs)
  cpp_pjit_f = xc._xla.pjit(
      fun_name(fun), fun, cache_miss, jit_info.static_argnums,
      jit_info.static_argnames, cache_key, tree_util.dispatch_registry,
      pxla.cc_shard_arg,
      _get_cpp_global_cache(cache_key.contains_explicit_attributes))

  cpp_pjitted_f = wraps(fun)(cpp_pjit_f)
  cpp_pjitted_f._fun = fun
  type(cpp_pjitted_f).clear_cache = _cpp_pjit_evict_fn
  return cpp_pjitted_f


def _split_layout_and_sharding(entries):
  entries_flat, treedef = tree_flatten(entries, is_leaf=lambda x: x is None)
  layouts, shardings = [], []

  for e in entries_flat:
    if isinstance(e, Layout):
      layouts.append(e.device_local_layout)
      shardings.append(e.sharding)
    elif isinstance(e, (DeviceLocalLayout, AutoLayout)):
      raise ValueError(
          '`jax.jit` does not accept device-local layouts directly. Create '
          'a `Layout` instance wrapping this device-local layout and pass '
          f'that to `jit` instead. Got {e}')
    else:
      layouts.append(None)
      shardings.append(e)

  assert len(layouts) == len(shardings)
  return tree_unflatten(treedef, layouts), tree_unflatten(treedef, shardings)


def _parse_jit_arguments(fun: Callable, in_shardings: Any, out_shardings: Any,
                         donate_argnums: int | Sequence[int] | None,
                         donate_argnames: str | Iterable[str] | None,
                         static_argnums: int | Sequence[int] | None,
                         static_argnames: str | Iterable[str] | None,
                         device: xc.Device | None, backend: str | None,
                         abstracted_axes: Any | None, keep_unused: bool,
                         inline: bool, compiler_options: dict[str, Any] | None,
                         use_resource_env: bool) -> PjitInfo:
  """Parses the arguments to jit/pjit.

  Performs any preprocessing and validation of the arguments that we can do
  ahead of time before the jit()-ed function is invoked.
  """
  if abstracted_axes and not config.dynamic_shapes.value:
    raise ValueError("abstracted_axes must be used with --jax_dynamic_shapes")

  check_callable(fun)

  if backend is not None or device is not None:
    warnings.warn(
        'backend and device argument on jit is deprecated. You can use'
        ' `jax.device_put(..., jax.local_devices("cpu")[0])` on the inputs to'
        ' the jitted function to get the same behavior.', DeprecationWarning)
    if device is not None and backend is not None:
      raise ValueError("can't specify both a device and a backend for jit, "
                       f"got {device=} and {backend=}")
    if in_shardings is not None and not isinstance(in_shardings, UnspecifiedValue):
      raise ValueError('If backend or device is specified on jit, then '
                       'in_shardings should not be specified.')
    if out_shardings is not None and not isinstance(out_shardings, UnspecifiedValue):
      raise ValueError('If backend or device is specified on jit, then '
                       'out_shardings should not be specified.')

  if isinstance(in_shardings, list):
    # To be a tree prefix of the positional args tuple, in_axes can never be a
    # list: if in_axes is not a leaf, it must be a tuple of trees. However,
    # in cases like these users expect tuples and lists to be treated
    # essentially interchangeably, so we canonicalize lists to tuples here
    # rather than raising an error. https://github.com/jax-ml/jax/issues/2367
    in_shardings = tuple(in_shardings)

  in_layouts, in_shardings = _split_layout_and_sharding(in_shardings)
  out_layouts, out_shardings = _split_layout_and_sharding(out_shardings)

  in_shardings = prepare_axis_resources(in_shardings, 'in_shardings')
  out_shardings = prepare_axis_resources(out_shardings, 'out_shardings')

  user_specified_in_shardings = (in_shardings is not None and
                                 not isinstance(in_shardings, UnspecifiedValue))

  in_shardings_leaves, in_shardings_treedef = none_lr.flatten(in_shardings)
  out_shardings_leaves, out_shardings_treedef = none_lr.flatten(out_shardings)
  in_layouts_leaves, in_layouts_treedef = none_lr.flatten(in_layouts)
  out_layouts_leaves, out_layouts_treedef = none_lr.flatten(out_layouts)

  fun_sourceinfo = api_util.fun_sourceinfo(fun)
  fun_signature = api_util.fun_signature(fun)

  donate_argnums, donate_argnames, static_argnums, static_argnames = resolve_argnums(
      fun, fun_signature, donate_argnums, donate_argnames, static_argnums,
      static_argnames)

  compiler_options_kvs = (() if compiler_options is None else
                          tuple(compiler_options.items()))
  return PjitInfo(
        fun_sourceinfo=fun_sourceinfo,
        fun_signature=fun_signature,
        user_specified_in_shardings=user_specified_in_shardings,
        in_shardings_treedef=in_shardings_treedef,
        in_shardings_leaves=tuple(in_shardings_leaves),
        out_shardings_treedef=out_shardings_treedef,
        out_shardings_leaves=tuple(out_shardings_leaves),
        in_layouts_treedef=in_layouts_treedef,
        in_layouts_leaves=tuple(in_layouts_leaves),
        out_layouts_treedef=out_layouts_treedef,
        out_layouts_leaves=tuple(out_layouts_leaves),
        static_argnums=static_argnums,
        static_argnames=static_argnames, donate_argnums=donate_argnums,
        donate_argnames=donate_argnames, device=device, backend=backend,
        keep_unused=keep_unused, inline=inline,
        abstracted_axes=abstracted_axes,
        use_resource_env=use_resource_env,
        compiler_options_kvs=compiler_options_kvs)


def _make_jit_wrapper(fun: Callable, jit_info: PjitInfo):

  @api_boundary
  def lower(*args, **kwargs):
    return trace(*args, **kwargs).lower()

  @api_boundary
  def eval_shape(*args, **kwargs):
    p, _ = _infer_params(fun, jit_info, args, kwargs)
    out_s = [None if isinstance(s, UnspecifiedValue) else s for s in p.params['out_shardings']]
    # TODO(yashkatariya): Add `Layout` to SDS.
    out = [api.ShapeDtypeStruct(x.shape, x.dtype, sharding=s,
                                weak_type=x.weak_type)
           for x, s in zip(p.params['jaxpr'].out_avals, out_s)]
    return tree_unflatten(p.out_tree, out)

  @api_boundary
  def trace(*args, **kwargs) -> stages.Traced:
    p, args_flat = _infer_params(fun, jit_info, args, kwargs)
    donate_argnums = tuple(i for i, d in enumerate(p.donated_invars) if d)
    args_info = stages.make_args_info(p.in_tree, p.in_avals, donate_argnums)
    lower_callable = partial(_resolve_and_lower, args_flat, **p.params,
                             pgle_profiler=None)
    return stages.Traced(
        p.params['jaxpr'], args_info, p.params["name"], p.out_tree,
        lower_callable, p.abstract_mesh, args_flat, p.arg_names, p.num_consts)

  wrapped = _cpp_pjit(fun, jit_info)
  wrapped.lower = lower
  wrapped.eval_shape = eval_shape
  wrapped.trace = trace
  return wrapped


def make_jit(fun: Callable, in_shardings: Any, out_shardings: Any,
             donate_argnums: int | Sequence[int] | None,
             donate_argnames: str | Iterable[str] | None,
             static_argnums: int | Sequence[int] | None,
             static_argnames: str | Iterable[str] | None,
             device: xc.Device | None, backend: str | None,
             abstracted_axes: Any | None, keep_unused: bool,
             inline: bool, compiler_options: dict[str, Any] | None,
             use_resource_env: bool) -> Any:
  """jit() and pjit() are thin wrappers around this function."""
  jit_info = _parse_jit_arguments(
        fun, in_shardings, out_shardings, donate_argnums, donate_argnames,
        static_argnums, static_argnames, device, backend, abstracted_axes,
        keep_unused, inline, compiler_options, use_resource_env)
  return _make_jit_wrapper(fun, jit_info)


class PjitParams(NamedTuple):
  consts: list[Any]  # Only jaxpr constants, we can't keep other arguments alive
  params: dict[str, Any]
  in_avals: tuple[core.AbstractValue, ...]
  in_tree: PyTreeDef
  out_tree: PyTreeDef
  donated_invars: tuple[bool, ...]
  arg_names: tuple[str | None, ...] | None
  num_consts: int
  attrs_tracked: list[tuple[PyTreeDef, PyTreeDef, tuple[Any, str]]]
  abstract_mesh: AbstractMesh


def _infer_params_impl(
    fun: Callable,
    ji: PjitInfo,
    pjit_mesh: mesh_lib.Mesh | None,
    resource_env: mesh_lib.ResourceEnv | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    in_avals: tuple[core.AbstractValue, ...] | None,
) -> tuple[PjitParams, list[Any]]:
  util.test_event("pjit._infer_params_impl", fun)
  have_kwargs = bool(kwargs)
  if have_kwargs and ji.user_specified_in_shardings:
    raise ValueError(
        "pjit does not support kwargs when in_shardings is specified.")

  if pjit_mesh is not None:
    if (ji.backend or ji.device) and not pjit_mesh.empty:
      raise ValueError(
          "Mesh context manager should not be used with jit when backend or "
          "device is also specified as an argument to jit.")

  axes_specs = _flat_axes_specs(ji.abstracted_axes, *args, **kwargs)

  dbg = debug_info('jit', ji.fun_sourceinfo, ji.fun_signature, args, kwargs,
                   ji.static_argnums, ji.static_argnames)
  f = lu.wrap_init(fun)
  f, res_paths = result_paths(f)
  f, dyn_args = argnums_partial_except(f, ji.static_argnums, args, allow_invalid=True)
  del args

  f, dyn_kwargs = argnames_partial_except(f, ji.static_argnames, kwargs)
  explicit_args, in_tree = tree_flatten((dyn_args, dyn_kwargs))
  flat_fun, out_tree = flatten_fun(f, in_tree)
  flat_fun, explicit_args = hoist_obj_attrs(flat_fun, explicit_args)

  if (ji.donate_argnums or ji.donate_argnames) and not config.debug_nans.value:
    donated_invars = donation_vector(ji.donate_argnums, ji.donate_argnames, in_tree)
  else:
    donated_invars = (False,) * len(explicit_args)

  # If backend or device is set as an arg on jit, then resolve them to
  # in_shardings and out_shardings as if user passed in in_shardings
  # and out_shardings.
  device_or_backend_set = bool(ji.backend or ji.device)
  if device_or_backend_set:
    sharding = _create_sharding_with_device_backend(ji.device, ji.backend)
    leaves, treedef = tree_flatten(sharding)
    in_shardings_leaves = out_shardings_leaves = tuple(leaves)
    in_shardings_treedef = out_shardings_treedef = treedef
  else:
    jit_name = 'pjit' if pjit_mesh is not None else 'jit'
    in_shardings_leaves = tuple(
        _create_sharding_for_array(pjit_mesh, x, 'in_shardings', jit_name)
        for x in ji.in_shardings_leaves)
    in_shardings_treedef = ji.in_shardings_treedef
    out_shardings_leaves = tuple(
        _create_sharding_for_array(pjit_mesh, x, 'out_shardings', jit_name)
        for x in ji.out_shardings_leaves)
    out_shardings_treedef = ji.out_shardings_treedef

  assert None not in in_shardings_leaves
  assert None not in out_shardings_leaves

  in_type: core.InputType | tuple[core.AbstractValue, ...]
  if config.dynamic_shapes.value:
    assert in_avals is None
    in_type = pe.infer_lambda_input_type(axes_specs, explicit_args)
    in_avals = tuple(a for a, e in in_type if e)
  else:
    in_type = in_avals  # type: ignore
  assert in_avals is not None

  in_shardings_flat, in_layouts_flat = _process_in_axis_resources(
      in_shardings_treedef, in_shardings_leaves,
      ji.in_layouts_treedef, ji.in_layouts_leaves,
      in_avals, in_tree, dbg, device_or_backend_set, have_kwargs)

  attr_token = _attr_token(flat_fun, in_type)

  abstract_mesh = (
      get_abstract_mesh_from_avals(in_type)
      if not mesh_lib.get_abstract_mesh() else mesh_lib.get_abstract_mesh())
  with mesh_lib.set_abstract_mesh(abstract_mesh):
    jaxpr, consts, out_avals, attrs_tracked = _create_pjit_jaxpr(
        flat_fun, in_type, attr_token, dbg,
        HashableFunction(res_paths, closure=()),
        IgnoreKey(ji.inline))
    if config.mutable_array_checks.value:
      _check_no_aliased_closed_over_refs(dbg, (*jaxpr.consts, *consts), explicit_args)
  _attr_update(flat_fun, in_type, attr_token, attrs_tracked)

  out_shardings_flat, out_layouts_flat = _check_and_canonicalize_out_shardings(
      out_shardings_treedef, out_shardings_leaves, ji.out_layouts_treedef,
      ji.out_layouts_leaves, HashableFunction(out_tree, closure=()),
      tuple(out_avals), jaxpr.jaxpr._debug_info, device_or_backend_set)

  assert len(explicit_args) == len(in_shardings_flat) == len(in_layouts_flat)

  if config.dynamic_shapes.value:
    implicit_args = _extract_implicit_args(
        cast(core.InputType, in_type), explicit_args)
  else:
    implicit_args = []
  args_flat = [*implicit_args, *explicit_args]

  num_states_in = sum(init_tree.num_leaves for init_tree, _, _ in attrs_tracked)
  num_extra_args = len(implicit_args) + num_states_in + len(consts)
  in_shardings_flat = (UNSPECIFIED,) * num_extra_args + in_shardings_flat
  in_layouts_flat = (None,) * num_extra_args + in_layouts_flat
  donated_invars = (False,) * num_extra_args + donated_invars
  assert (len(in_shardings_flat) == len(in_layouts_flat) ==
          len(donated_invars) == num_states_in + len(consts) + len(args_flat))

  params = dict(
      jaxpr=jaxpr,
      in_shardings=in_shardings_flat,
      out_shardings=out_shardings_flat,
      in_layouts=in_layouts_flat,
      out_layouts=out_layouts_flat,
      resource_env=resource_env,
      donated_invars=donated_invars,
      name=fun_qual_name(flat_fun),
      keep_unused=ji.keep_unused,
      inline=ji.inline,
      compiler_options_kvs=ji.compiler_options_kvs,
  )
  return PjitParams(consts, params, in_avals, in_tree, out_tree(),
                    donated_invars, dbg.arg_names if dbg else None, len(consts),
                    attrs_tracked, abstract_mesh), args_flat

def get_abstract_mesh_from_avals(in_avals):
  if not config.sharding_in_types.value:
    return None
  m = None
  for a in in_avals:
    if m is not None and m != a.sharding.mesh:
      raise ValueError(
          f'Mesh for all inputs should be equal. Got one mesh: {m} and'
          f' another mesh: {a.sharding.mesh}')
    m = a.sharding.mesh  # type: ignore
  assert isinstance(m, AbstractMesh)
  return m


class InferParamsCacheEntry:
  """Mutable value object for _infer_params_cached."""
  __slots__ = ['pjit_params']
  pjit_params: PjitParams | None
  def __init__(self):
    self.pjit_params = None


# We use an outer cache that is keyed on the signature of the arguments, but
# when populating a cache entry using _infer_params_impl, we need to provide
# actual arguments. In principle we could refactor _infer_params_impl to look
# only at an argument signature instead of args/kwargs in those cases that we
# cache, but this was a more minimal change.
@util.weakref_lru_cache
def _infer_params_cached(
    fun: Callable,
    jit_info: PjitInfo,
    signature: jax_jit.ArgumentSignature,
    in_avals: tuple[core.AbstractValue, ...],
    pjit_mesh: mesh_lib.Mesh | None,
    resource_env: mesh_lib.ResourceEnv | None,
) -> InferParamsCacheEntry:
  return InferParamsCacheEntry()


def _infer_params(
    fun: Callable, ji: PjitInfo, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[PjitParams, list[Any]]:
  if ji.use_resource_env:
    # We need to fetch the mesh from inside the wrapped function, because
    # meshes are dynamically scoped (i.e., with a context manager).
    resource_env = mesh_lib.thread_resources.env
    pjit_mesh = resource_env.physical_mesh
  else:
    resource_env = None
    pjit_mesh = None

  if config.dynamic_shapes.value:  # if dynamic shapes, don't use the cache
    p, args_flat = _infer_params_impl(fun, ji, pjit_mesh, resource_env, args,
                                      kwargs, in_avals=None)
    return p, p.consts + args_flat

  signature, dynargs = jax_jit.parse_arguments(
      args, tuple(kwargs.values()), tuple(kwargs.keys()), ji.static_argnums,
      ji.static_argnames, tree_util.default_registry)
  dbg = debug_info('jit', ji.fun_sourceinfo, ji.fun_signature, args, kwargs,
                   ji.static_argnums, ji.static_argnames)
  avals = _infer_input_type(fun, dbg, dynargs)
  entry = _infer_params_cached(fun, ji, signature, avals, pjit_mesh, resource_env)
  if entry.pjit_params is None:
    p, args_flat = _infer_params_impl(
        fun, ji, pjit_mesh, resource_env, args, kwargs, in_avals=avals)
    if p.attrs_tracked:  # if attrs, don't popoulate the cache
      return p, p.consts + args_flat
    entry.pjit_params = p
  return entry.pjit_params, entry.pjit_params.consts + dynargs

def _infer_input_type(fun, dbg, explicit_args) -> tuple[core.AbstractValue, ...]:
  avals = []
  try:
    for i, x in enumerate(explicit_args):
      avals.append(core.shaped_abstractify(x))
  except OverflowError:
    arg_path = (f"argument path is {dbg.arg_names[i]}" if dbg  # type: ignore
                else f"flattened argument number is {i}")  # type: ignore
    raise OverflowError(
      "An overflow was encountered while parsing an argument to a jitted "
      f"computation, whose {arg_path}."
    ) from None
  except TypeError:
    arg_description = (f"path {dbg.arg_names[i]}" if dbg  # type: ignore
                       else f"flattened argument number {i}")  # type: ignore
    raise TypeError(
      f"Error interpreting argument to {fun} as an abstract array."
      f" The problematic value is of type {type(x)} and was passed to"  # type: ignore
      f" the function at {arg_description}.\n"
      "This typically means that a jit-wrapped function was called with a non-array"
      " argument, and this argument was not marked as static using the"
      " static_argnums or static_argnames parameters of jax.jit."
    ) from None
  if config.mutable_array_checks.value:
    _check_no_aliased_ref_args(dbg, avals, explicit_args)
  return tuple(avals)

def _extract_implicit_args(
  in_type: Sequence[tuple[core.AbstractValue, bool]],
  explicit_args: Sequence[Any]
) -> Sequence[core.Tracer]:
  """
  Given an input type and explicitly-passed arguments (per the user-facing API
  calling convention), extract implicit axis size arguments from shapes of
  explicit arguments (for the trace-time / jaxpr-level calling convention).
  """
  # First, using `in_type` construct a list to represent the full argument list,
  # leaving the implicit arguments as None placeholders for now.
  explicit_args_ = iter(explicit_args)
  args = [next(explicit_args_) if expl else None for _, expl in in_type]
  assert next(explicit_args_, None) is None
  del explicit_args, explicit_args_

  # Next, populate the implicit arguments using the DBIdxs in `in_type`.
  for i, (aval, explicit) in enumerate(in_type):
    if not explicit or not isinstance(aval, core.DShapedArray):
      continue  # can't populate an implicit argument
    arg = args[i]
    assert arg is not None
    for d1, d2 in zip(aval.shape, arg.aval.shape):
      if isinstance(d1, core.DBIdx):
        if args[d1.val] is None:
          args[d1.val] = d2
        assert core.same_referent(args[d1.val], d2)
  assert all(x is not None for x in args)
  return [x for x, (_, e) in zip(args, in_type) if not e]  # type: ignore

def _flat_axes_specs(abstracted_axes, *args, **kwargs
                     ) -> list[pe.AbstractedAxesSpec] | None:
  if abstracted_axes is None: return None
  if kwargs: raise NotImplementedError
  def ax_leaf(l):
    return (isinstance(l, dict) and all_leaves(l.values()) or
            isinstance(l, tuple) and all_leaves(l, lambda x: x is None))
  return broadcast_prefix(abstracted_axes, args, ax_leaf)


class JitWrapped(stages.Wrapped):

  def eval_shape(self, *args, **kwargs):
    """See ``jax.eval_shape``."""
    raise NotImplementedError

  def trace(self, *args, **kwargs) -> stages.Traced:
    raise NotImplementedError


# in_shardings and out_shardings can't be None as the default value
# because `None` means that the input is fully replicated.
def pjit(
    fun: Callable,
    in_shardings=UNSPECIFIED,
    out_shardings=UNSPECIFIED,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    keep_unused: bool = False,
    device: xc.Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    abstracted_axes: Any | None = None,
    compiler_options: dict[str, Any] | None = None,
) -> JitWrapped:
  """Makes ``fun`` compiled and automatically partitioned across multiple devices.

  NOTE: This function is now equivalent to jax.jit please use that instead.
  The returned function has semantics equivalent to those of ``fun``, but is
  compiled to an XLA computation that runs across multiple devices
  (e.g. multiple GPUs or multiple TPU cores). This can be useful if the jitted
  version of ``fun`` would not fit in a single device's memory, or to speed up
  ``fun`` by running each operation in parallel across multiple devices.

  The partitioning over devices happens automatically based on the
  propagation of the input partitioning specified in ``in_shardings`` and
  the output partitioning specified in ``out_shardings``. The resources
  specified in those two arguments must refer to mesh axes, as defined by
  the :py:func:`jax.sharding.Mesh` context manager. Note that the mesh
  definition at :func:`~pjit` application time is ignored, and the returned function
  will use the mesh definition available at each call site.

  Inputs to a :func:`~pjit`'d function will be automatically partitioned across devices
  if they're not already correctly partitioned based on ``in_shardings``.
  In some scenarios, ensuring that the inputs are already correctly pre-partitioned
  can increase performance. For example, if passing the output of one
  :func:`~pjit`'d function to another :func:`~pjit`’d function (or the same
  :func:`~pjit`’d function in a loop), make sure the relevant
  ``out_shardings`` match the corresponding ``in_shardings``.

  .. note::
    **Multi-process platforms:** On multi-process platforms such as TPU pods,
    :func:`~pjit` can be used to run computations across all available devices across
    processes. To achieve this, :func:`~pjit` is designed to be used in SPMD Python
    programs, where every process is running the same Python code such that all
    processes run the same :func:`~pjit`'d function in the same order.

    When running in this configuration, the mesh should contain devices across
    all processes. All inputs arguments must be globally shaped.
    ``fun`` will still be executed across *all* devices in the mesh,
    including those from other processes, and will be given a global view of the
    data spread across multiple processes as a single array.

    The SPMD model also requires that the same multi-process :func:`~pjit`'d
    functions must be run in the same order on all processes, but they can be
    interspersed with arbitrary operations running in a single process.

  Args:
    fun: Function to be compiled. Should be a pure function, as side-effects may
      only be executed once. Its arguments and return value should be arrays,
      scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
      Positional arguments indicated by ``static_argnums`` can be anything at
      all, provided they are hashable and have an equality operation defined.
      Static arguments are included as part of a compilation cache key, which is
      why hash and equality operators must be defined.
    in_shardings: Pytree of structure matching that of arguments to ``fun``,
      with all actual arguments replaced by resource assignment specifications.
      It is also valid to specify a pytree prefix (e.g. one value in place of a
      whole subtree), in which case the leaves get broadcast to all values in
      that subtree.

      The ``in_shardings`` argument is optional. JAX will infer the shardings
      from the input :py:class:`jax.Array`'s, and defaults to replicating the input
      if the sharding cannot be inferred.

      The valid resource assignment specifications are:

      - :py:class:`Sharding`, which will decide how the value
        will be partitioned. With this, using a mesh context manager is not
        required.
      - :py:obj:`None` is a special case whose semantics are:
          - if the mesh context manager is *not* provided, JAX has the freedom to
            choose whatever sharding it wants.
            For in_shardings, JAX will mark is as replicated but this behavior
            can change in the future.
            For out_shardings, we will rely on the XLA GSPMD partitioner to
            determine the output shardings.
          - If the mesh context manager is provided, None will imply that the
            value will be replicated on all devices of the mesh.
      - For backwards compatibility, in_shardings still supports ingesting
        :py:class:`PartitionSpec`. This option can *only* be used with the
        mesh context manager.

        - :py:class:`PartitionSpec`, a tuple of length at most equal to the rank
          of the partitioned value. Each element can be a :py:obj:`None`, a mesh
          axis or a tuple of mesh axes, and specifies the set of resources assigned
          to partition the value's dimension matching its position in the spec.

      The size of every dimension has to be a multiple of the total number of
      resources assigned to it.
    out_shardings: Like ``in_shardings``, but specifies resource
      assignment for function outputs.
      The ``out_shardings`` argument is optional. If not specified, :py:func:`jax.jit`
      will use GSPMD's sharding propagation to determine how to shard the outputs.
    static_argnums: An optional int or collection of ints that specify which
      positional arguments to treat as static (compile-time constant).
      Operations that only depend on static arguments will be constant-folded in
      Python (during tracing), and so the corresponding argument values can be
      any Python object.

      Static arguments should be hashable, meaning both ``__hash__`` and
      ``__eq__`` are implemented, and immutable. Calling the jitted function
      with different values for these constants will trigger recompilation.
      Arguments that are not arrays or containers thereof must be marked as
      static.

      If ``static_argnums`` is not provided, no arguments are treated as static.
    static_argnames: An optional string or collection of strings specifying
      which named arguments to treat as static (compile-time constant). See the
      comment on ``static_argnums`` for details. If not
      provided but ``static_argnums`` is set, the default is based on calling
      ``inspect.signature(fun)`` to find corresponding named arguments.
    donate_argnums: Specify which positional argument buffers are "donated" to
      the computation. It is safe to donate argument buffers if you no longer
      need them once the computation has finished. In some cases XLA can make
      use of donated buffers to reduce the amount of memory needed to perform a
      computation, for example recycling one of your input buffers to store a
      result. You should not reuse buffers that you donate to a computation, JAX
      will raise an error if you try to. By default, no argument buffers are
      donated.

      If neither ``donate_argnums`` nor ``donate_argnames`` is provided, no
      arguments are donated. If ``donate_argnums`` is not provided but
      ``donate_argnames`` is, or vice versa, JAX uses
      :code:`inspect.signature(fun)` to find any positional arguments that
      correspond to ``donate_argnames``
      (or vice versa). If both ``donate_argnums`` and ``donate_argnames`` are
      provided, ``inspect.signature`` is not used, and only actual
      parameters listed in either ``donate_argnums`` or ``donate_argnames`` will
      be donated.

      For more details on buffer donation see the
      `FAQ <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>`_.
    donate_argnames: An optional string or collection of strings specifying
      which named arguments are donated to the computation. See the
      comment on ``donate_argnums`` for details. If not
      provided but ``donate_argnums`` is set, the default is based on calling
      ``inspect.signature(fun)`` to find corresponding named arguments.
    keep_unused: If `False` (the default), arguments that JAX determines to be
      unused by `fun` *may* be dropped from resulting compiled XLA executables.
      Such arguments will not be transferred to the device nor provided to the
      underlying executable. If `True`, unused arguments will not be pruned.
    device: This argument is deprecated. Please put your arguments on the
      device you want before passing them to jit.
      Optional, the Device the jitted function will run on. (Available devices
      can be retrieved via :py:func:`jax.devices`.) The default is inherited
      from XLA's DeviceAssignment logic and is usually to use
      ``jax.devices()[0]``.
    backend: This argument is deprecated. Please put your arguments on the
      backend you want before passing them to jit.
      Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    A wrapped version of ``fun``, set up for just-in-time compilation and
    automatically partitioned by the mesh available at each call site.

  For example, a convolution operator can be automatically partitioned over
  an arbitrary set of devices by a single :func:`~pjit` application:

  >>> import jax
  >>> import jax.numpy as jnp
  >>> import numpy as np
  >>> from jax.sharding import Mesh, PartitionSpec
  >>> from jax.experimental.pjit import pjit
  >>>
  >>> x = jnp.arange(8, dtype=jnp.float32)
  >>> f = pjit(lambda x: jax.numpy.convolve(x, jnp.asarray([0.5, 1.0, 0.5]), 'same'),
  ...         in_shardings=None, out_shardings=PartitionSpec('devices'))
  >>> with Mesh(np.array(jax.devices()), ('devices',)):
  ...   print(f(x))  # doctest: +SKIP
  [ 0.5  2.   4.   6.   8.  10.  12.  10. ]
  """
  return make_jit(
       fun, in_shardings, out_shardings, donate_argnums, donate_argnames,
       static_argnums, static_argnames, device, backend, abstracted_axes,
       keep_unused, inline, compiler_options, use_resource_env=True)


def hashable_pytree(pytree):
  vals, treedef = tree_flatten(pytree)
  vals = tuple(vals)
  return HashableFunction(lambda: tree_unflatten(treedef, vals),
                          closure=(treedef, vals))


def _create_sharding_for_array(mesh, x, name, api_name):
  if x is None and (mesh is None or mesh.empty):
    return UNSPECIFIED
  if isinstance(x, (AUTO, UnspecifiedValue, Sharding)):
    return x
  if mesh is None:
    msg = ('jax.jit only supports `Sharding`s being passed to'
           f' {name}. Looks like you are passing either `PartitionSpec` or `None`'
           f' which is not allowed in jax.jit.\n')
    if name == 'in_shardings':
      msg += (f'Note that {name} argument is optional. JAX will infer the shardings'
              " from the input jax.Array's and will default to replicating the"
              ' input if the sharding cannot be inferred.')
    elif name == 'out_shardings':
      msg += (f'Note that {name} is optional. If not specified, jax.jit will'
              " use GSPMD's sharding propagation to figure out what the sharding"
              ' of the output(s) should be.')
    raise RuntimeError(msg)
  if mesh.empty:
    raise RuntimeError(
        f'{api_name} requires a non-empty mesh if you are passing'
        f' `PartitionSpec`s or `None` to {name}! Is a mesh defined at the call'
        f' site? Alternatively, provide `Sharding`s to {name} and'
        ' then the mesh context manager is not required.')
  # A nice user error is raised in prepare_axis_resources.
  assert x is None or isinstance(x, ParsedPartitionSpec), x
  return (pxla.create_mesh_pspec_sharding(mesh, x) if x is None else
          pxla.create_mesh_pspec_sharding(mesh, x.get_partition_spec(), x))


def _create_sharding_with_device_backend(device, backend):
  if device is not None:
    assert backend is None
    out = SingleDeviceSharding(device)
  elif backend is not None:
    assert device is None
    out = SingleDeviceSharding(xb.get_backend(backend).local_devices()[0])
  else:
    raise AssertionError('Unreachable!')
  out._device_backend = True
  return out


def flatten_axis_resources(what, tree, shardings, tupled_args):
  try:
    return tuple(flatten_axes(what, tree, shardings, tupled_args=tupled_args))
  except ValueError:
    pass  # Raise a tree prefix error below

  # Tree leaves are always valid prefixes, so if there was a prefix error as
  # assumed here, axis_resources must not be a leaf.
  assert not treedef_is_leaf(tree_structure(shardings))

  # Check the type directly rather than using isinstance because of namedtuples.
  if tupled_args and (type(shardings) is not tuple or
                      len(shardings) != len(tree.children())):
    # We know axis_resources is meant to be a tuple corresponding to the args
    # tuple, but while it is a non-leaf pytree, either it wasn't a tuple or it
    # wasn't the right length.
    msg = (f"{what} specification must be a tree prefix of the positional "
           f"arguments tuple passed to the `pjit`-decorated function. In "
           f"particular, {what} must either be a None, a PartitionSpec, or "
           f"a tuple of length equal to the number of positional arguments.")
    # If `tree` represents an args tuple, then `axis_resources` must be a tuple.
    # TODO(mattjj,apaszke): disable implicit list casts, remove 'or list' below
    if type(shardings) is not tuple:
      msg += f" But {what} is not a tuple: got {type(shardings)} instead."
    elif len(shardings) != len(tree.children()):
      msg += (f" But {what} is the wrong length: got a tuple or list of length "
              f"{len(shardings)} for an args tuple of length "
              f"{len(tree.children())}.")

    # As an extra hint, let's check if the user just forgot to wrap
    # shardings in a singleton tuple.
    if len(tree.children()) == 1:
      try: flatten_axes(what, tree, (shardings,))
      except ValueError: pass  # That's not the issue.
      else:
        msg += (f" Given the corresponding argument being "
                f"passed, it looks like {what} might need to be wrapped in "
                f"a singleton tuple.")

    raise ValueError(msg)

  axis_tree = shardings

  # Because we only have the `tree` treedef and not the full pytree here,
  # we construct a dummy tree to compare against. Revise this in callers?
  dummy_tree = tree_unflatten(tree, [PytreeLeaf()] * tree.num_leaves)
  errors = prefix_errors(axis_tree, dummy_tree)
  if errors:
    e = errors[0]  # Only show information about the first disagreement found.
    raise e(what)

  # At this point we've failed to find a tree prefix error.
  assert False, "Please open a bug report!"  # This should be unreachable.

class PytreeLeaf:
  def __repr__(self): return "pytree leaf"


@util.cache(max_size=4096, trace_context_in_key=False)
def _process_in_axis_resources(in_shardings_treedef, in_shardings_leaves,
                               in_layouts_treedef, in_layouts_leaves,
                               in_avals, in_tree, debug_info,
                               device_or_backend_set, kws):
  if not kws:
    in_tree, _ = treedef_children(in_tree)

  orig_in_shardings = tree_unflatten(in_shardings_treedef, in_shardings_leaves)
  # Only do this if original in_shardings are unspecified. If it is AUTO, go
  # via flatten_axis_resources.
  if isinstance(orig_in_shardings, UnspecifiedValue):
    in_shardings_flat = (orig_in_shardings,) * len(in_avals)
  else:
    in_shardings_flat = flatten_axis_resources(
        "pjit in_shardings", in_tree, orig_in_shardings, tupled_args=True)

  in_layouts = tree_unflatten(in_layouts_treedef, in_layouts_leaves)
  if in_layouts is None:
    in_layouts_flat = (in_layouts,) * len(in_avals)
  else:
    in_layouts_flat = flatten_axis_resources(
        "pjit in_layouts", in_tree, in_layouts, tupled_args=True)

  # TODO(dougalm,mattjj): enable debug info with attrs_tracked
  attrs_tracked = debug_info and len(debug_info.arg_names) != len(in_avals)
  if not config.dynamic_shapes.value and not attrs_tracked:
    pjit_check_aval_sharding(in_shardings_flat, in_avals,
                             None if debug_info is None else debug_info.arg_names,
                             "pjit arguments", allow_uneven_sharding=False)
    check_aval_layout_compatibility(
        in_layouts_flat, in_avals,
        None if debug_info is None else debug_info.arg_names, "jit arguments")
  return in_shardings_flat, in_layouts_flat

callsites: set[str] = set()

def explain_tracing_cache_miss(
    f: Callable, unseen_f: bool, cache: dict, key: tuple):
  if config.check_tracer_leaks.value: return

  def unpack(key):
    transforms, (), _, (in_type, _, debug_info, _, inline), *_, ctx = key
    # TODO(dougalm,mattjj): enable cache miss explanation with attrs
    _, (_, (in_tree,)), *_ = transforms
    return in_tree, in_type, debug_info, inline.val, ctx
  in_tree, in_type, debug_info, inline, ctx = unpack(key)
  if inline: return

  msg: list[str] = []
  p = msg.append
  done = lambda: logger.log(logging.WARNING, '\n'.join(msg))

  callsite = source_info_util.summarize(source_info_util.current())
  p(f"TRACING CACHE MISS at {callsite} because:")

  # have we seen this function before at all?
  fun_name = getattr(f, '__qualname__', f)
  if debug_info is not None and debug_info.func_src_info:
    _, _, *rest = debug_info.func_src_info.split(' ')
    src_info = " defined at "  + ' '.join(rest)
  else:
    src_info = ''
  if unseen_f:
    p(f"  never seen function:\n    {fun_name} id={id(f)}{src_info}")
    if callsite in callsites:
      p("  but seen another function defined on the same line; maybe the function is\n"
        "  being re-defined repeatedly, preventing caching?")
    callsites.add(callsite)
    return done()
  else:
    p(f"  for {fun_name}{src_info}")

  seen_keys = map(unpack, cache.keys())

  # have we maybe switched some args to be kwargs or visa-versa?
  args_tree, kwargs_tree = treedef_children(in_tree)
  args_kwargs_trees = [treedef_children(k) for k, *_ in seen_keys]
  args_kwargs_match = [t for t in args_kwargs_trees
                       if t == [args_tree, kwargs_tree]]
  if not args_kwargs_match:
    num_args = len(treedef_children(args_tree))
    _, kwarg_keys = kwargs_tree.node_data()  # type: ignore
    p(f"  never seen passing {num_args} positional args and {len(kwarg_keys)} "
      "keyword args with keys:\n"
      f"    {', '.join(map(repr, kwarg_keys))}")
    dont_match = [set(t[1].node_data()[1]) for t in args_kwargs_trees  # type: ignore
                  if t != [args_tree, kwargs_tree]]
    close_kwargs = min(
        dont_match, key=set(kwarg_keys).symmetric_difference, default=None
    )
    if not close_kwargs:
      p("  closest seen is passing no keyword args")
    else:
      p(f"  closest seen passes {len(close_kwargs)} keyword args with keys:\n"
        f"    {', '.join(map(repr, close_kwargs))}")
    return done()

  # have we never seen this tracing context before?
  ctxs_match = [c for *_, c in seen_keys if c == ctx]
  if not ctxs_match:
    p("  tracing context doesn't match, e.g. due to config or context manager")
    dont_match = [c for *_, c in seen_keys if c != ctx]
    closest_ctx = min(dont_match, key=lambda c: sum(map(op.ne, c, ctx)))
    idxs = [i for i, (c1, c2) in enumerate(zip(ctx, closest_ctx)) if c1 != c2]
    p("  closest seen context tuple differs at positions:\n"
      f"    {', '.join(map(str, idxs))}\n"
      "  compare to tuple returned by config._trace_context() in jax/_src/config.py.")
    return done()

  # have we never seen this input pytree before?
  trees_match = [k for k in seen_keys if k[0] == in_tree]
  if not trees_match:
    in_tree_str = f':\n    {in_tree}' if len(str(in_tree)) < 76 else ''
    p(f"  never seen input pytree{in_tree_str}")
    dont_match = [t for t, *_ in seen_keys if t != in_tree]
    closest_tree = min(dont_match, key=lambda t: abs(t.num_leaves - in_tree.num_leaves))
    errs = list(tree_util.equality_errors_pytreedef(in_tree, closest_tree))  # type: ignore[arg-type]
    p(f"  closest seen input pytree has {len(errs)} mismatches, including:")
    for path, thing1, thing2, explanation in errs:
      fst, *path = path  # type: ignore
      base = ['args', 'kwargs'][fst.idx]
      p(f"    * at {base}{keystr(tuple(path))}, seen {thing2} but now given {thing1},"
        f"      so {explanation}")
    return done()

  # have we never seen these input types (eg shapes, dtypes) before?
  types_match = [k for k in trees_match if k[1] == in_type]
  if not types_match:
    if len(in_type) < 5:
      in_type_str = ':\n    {}'.format(',  '.join(
          f'{n}: {ty.str_short(short_dtypes=True)}'
          for n, ty in zip(debug_info.arg_names, in_type)))
    else:
      in_type_str = ''
    p(f"  never seen input type signature{in_type_str}")
    dont_match = [t for _, t, *_ in trees_match if t != in_type]
    closest_ty = min(dont_match, key=lambda t: sum(map(op.ne, t, in_type)))
    num_mismatch = sum(map(op.ne, closest_ty, in_type))
    p(f"  closest seen input type signature has {num_mismatch} mismatches, including:")
    add_weak_type_hint = False
    for name, ty1, ty2 in zip(debug_info.arg_names, closest_ty, in_type):
      if ty1 != ty2:
        if type(ty1) == type(ty2) == core.ShapedArray:
          s1, s2 = ty1.str_short(True), ty2.str_short(True)
          if s1 == s2:  # weak types don't show up in str_short()
            assert ty1.weak_type ^ ty2.weak_type
            s1 += f'{{weak_type={ty1.weak_type}}}'
            s2 += f'{{weak_type={ty2.weak_type}}}'
            add_weak_type_hint = True
        else:
          s1, s2 = str(ty1), str(ty2)
        p(f"    * at {name}, seen {s1}, but now given {s2}")
    if add_weak_type_hint:
      p('where weak_type=True often means a Python builtin numeric value, and ')
      p('weak_type=False means a jax.Array.')
      p('See https://jax.readthedocs.io/en/latest/type_promotion.html#weak-types')
    return done()

  # we think this is unreachable...
  p("explanation unavailable! please open an issue at https://github.com/jax-ml/jax")
  return done()

@partial(lu.cache, explain=explain_tracing_cache_miss)
def _create_pjit_jaxpr(
    fun: lu.WrappedFun,
    in_type: core.InputType | Sequence[core.AbstractValue],
    attr_data: int,
    debug_info: lu.TracingDebugInfo,
    out_paths: Callable,
    ignored_inline: IgnoreKey
) -> tuple[core.ClosedJaxpr, list[Any], list[core.AbstractValue],
           list[tuple[PyTreeDef, PyTreeDef, tuple[Any, str]]]]:
  util.test_event("create_pjit_jaxpr")
  del ignored_inline  # just for explain_cache_miss
  if config.no_tracing.value:
    raise RuntimeError(f"re-tracing function {fun.f} for `jit`, but "
                       "'no_tracing' is set")
  with dispatch.log_elapsed_time(
      "Finished tracing + transforming {fun_name} for pjit in {elapsed_time:.9f} sec",
      fun_name=fun.__name__, event=dispatch.JAXPR_TRACE_EVENT):
    pe_debug = debug_info and pe.tracing_debug_info_final(fun, debug_info.traced_for)
    if config.dynamic_shapes.value:
      jaxpr, global_out_avals, consts = pe.trace_to_jaxpr_dynamic2(
          lu.annotate(fun, cast(core.InputType, in_type)), debug_info=pe_debug)
      attrs_tracked = []
    else:
      jaxpr, global_out_avals, consts, attrs_tracked = pe.trace_to_jaxpr_dynamic(
          fun, in_type, debug_info=pe_debug)
      # assert attr_data is sentinel or attr_data matches attrs_tracked

  # TODO(dougalm,mattjj): enable debug info with attrs_tracked
  if not config.dynamic_shapes.value and not attrs_tracked:
    jaxpr = add_jaxpr_debug_info(jaxpr, debug_info, out_paths())

  if config.debug_key_reuse.value:
    # Import here to avoid circular imports
    from jax.experimental.key_reuse._core import check_key_reuse_jaxpr
    check_key_reuse_jaxpr(jaxpr)

  if any(isinstance(c, core.Tracer) for c in consts):
    closed_jaxpr = pe.close_jaxpr(pe.convert_constvars_jaxpr(jaxpr))
    final_consts = consts
  else:
    closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
    final_consts = []
  return closed_jaxpr, final_consts, global_out_avals, attrs_tracked


@util.cache(max_size=4096, trace_context_in_key=False)
def _check_and_canonicalize_out_shardings(
    out_shardings_treedef, out_shardings_leaves, out_layouts_treedef,
    out_layouts_leaves, out_tree, out_avals,
    debug_info: core.JaxprDebugInfo | None,
    device_or_backend_set):
  orig_out_shardings = tree_unflatten(out_shardings_treedef, out_shardings_leaves)
  if isinstance(orig_out_shardings, (UnspecifiedValue, Sharding)):
    out_shardings_flat = (orig_out_shardings,) * len(out_avals)
  else:
    out_shardings_flat = flatten_axis_resources(
        "pjit out_shardings", out_tree(), orig_out_shardings,
        tupled_args=False)

  out_layouts = tree_unflatten(out_layouts_treedef, out_layouts_leaves)
  if out_layouts is None:
    out_layouts_flat = (out_layouts,) * len(out_avals)
  else:
    out_layouts_flat = flatten_axis_resources(
        "pjit out_layouts", out_tree(), out_layouts, tupled_args=False)

  if not config.dynamic_shapes.value:
    pjit_check_aval_sharding(
        out_shardings_flat, out_avals,
        None if debug_info is None else debug_info.result_paths,
        "pjit outputs", allow_uneven_sharding=False)
    check_aval_layout_compatibility(
        out_layouts_flat, out_avals,
        None if debug_info is None else debug_info.result_paths, "jit outputs")
  return out_shardings_flat, out_layouts_flat


AttrRecord = tuple[object, str, PyTreeDef, list[core.AbstractValue]]
_seen_attrs = weakref.WeakKeyDictionary()  # type: ignore

def seen_attrs_get(
    fun: lu.WrappedFun,
    in_type: core.InputType | tuple[core.AbstractValue, ...]
) -> list:
  cache = _seen_attrs.setdefault(fun.f, defaultdict(list))
  assert fun.in_type is None or fun.in_type == in_type
  return cache[(fun.transforms, fun.params, in_type)]

def _attr_token(
    fun: lu.WrappedFun,
    in_type: core.InputType | tuple[core.AbstractValue, ...]
) -> int:
  from jax.experimental.attrs import jax_getattr
  cases = seen_attrs_get(fun, in_type)
  for i, records in enumerate(cases):
    for obj, attr, treedef, avals in records:
      val = jax_getattr(obj, attr)
      vals, treedef_ = tree_flatten(val)
      avals_ = map(core.shaped_abstractify, vals)
      if treedef != treedef_ or avals != avals_: break
    else:
      return i
  return len(cases)

def _attr_update(fun, in_type, i, attrs_tracked):
  from jax.experimental.attrs import jax_getattr
  leaves = lambda obj, attr: tree_leaves(jax_getattr(obj, attr))
  records = [(obj, attr, init_tree, map(core.shaped_abstractify, leaves(obj, attr)))
             for init_tree, _, (obj, attr) in attrs_tracked]
  cases = seen_attrs_get(fun, in_type)
  if i == len(cases):
    cases.append(records)
  else:
    assert i < len(cases) and cases[i] == records


@dataclasses.dataclass(frozen=True)
class IgnoreKey:
  val: Any
  def __hash__(self):
    return hash(self.__class__)
  def __eq__(self, other):
    return isinstance(other, IgnoreKey)  # ignore self.val!


def pjit_check_aval_sharding(
    shardings, flat_avals, names: tuple[str, ...] | None,
    what_aval: str, allow_uneven_sharding: bool):
  new_names = [''] * len(shardings) if names is None else names
  for aval, s, name in zip(flat_avals, shardings, new_names):
    if isinstance(s, (UnspecifiedValue, AUTO)):
      continue
    name_str = f' with pytree key path {name}' if name else ''
    shape = aval.shape
    try:
      # Sharding interfaces can implement `check_compatible_aval` as an optional
      # method to raise a more meaningful error.
      if hasattr(s, 'check_compatible_aval'):
        s.check_compatible_aval(shape)
      else:
        s._to_xla_hlo_sharding(len(shape))
    except ValueError as e:
      raise ValueError(
          f'One of {what_aval}{name_str} is incompatible with its sharding '
          f'annotation {s}: {e}')
    # Use the `OpSharding` proto to find out how many ways each dimension of
    # the aval is sharded. This approach will work across all
    # Sharding.
    hlo_sharding = s._to_xla_hlo_sharding(len(shape))
    assert hlo_sharding is not None
    num_ways_dim_sharded, _ = op_shardings.get_num_ways_dim_sharded(hlo_sharding)
    for i, size in enumerate(num_ways_dim_sharded):
      if not allow_uneven_sharding and shape[i] % size != 0:
        raise ValueError(f"One of {what_aval}{name_str} was given the sharding "
                         f"of {s}, which implies that "
                         f"the global size of its dimension {i} should be "
                         f"divisible by {size}, but it is equal to {shape[i]} "
                         f"(full shape: {shape})")


def check_aval_layout_compatibility(
    layouts, flat_avals, names: tuple[str, ...] | None, what_aval: str):
  new_names = [''] * len(layouts) if names is None else names
  for aval, l, name in zip(flat_avals, layouts, new_names):
    if l is None or isinstance(l, AutoLayout):
      continue
    name_str = f' with pytree key path {name}' if name else ''
    shape = aval.shape
    try:
      l.check_compatible_aval(shape)
    except ValueError as e:
      raise ValueError(
          f'One of {what_aval}{name_str} is incompatible with its layout '
          f'annotation {l}: {e}')


# -------------------- pjit rules --------------------

pjit_p = core.Primitive("pjit")
pjit_p.multiple_results = True


def _resolve_in_layouts(args, jit_in_layouts, resolved_in_shardings, in_avals):
  # If device or backend is set, return the default layout. This is because you
  # can pass arrays on cpu (with untiled layouts) to jit with backend='tpu'
  # which causes error checks to fail. Returning the default layout allows
  # this to exist. It's the same for handling shardings.
  if pxla.check_device_backend_on_shardings(resolved_in_shardings):
    return (None,) * len(jit_in_layouts)

  resolved_in_layouts = []
  for arg, jit_in_l, rs, aval in safe_zip(
      args, jit_in_layouts, resolved_in_shardings, in_avals):
    committed = getattr(arg, '_committed', True)
    # `arg_layout` is only used for checking purposes in the `else` branch
    # below. We cannot replace default layout with None to raise nicer errors.
    # `dispatch_arg_layout` replaces default layouts with `None` to simplify
    # dispatch and lowering logic downstream.
    if hasattr(arg, 'layout'):
      arg_layout = arg.layout.device_local_layout
      dispatch_arg_layout = (None if pxla.is_default_layout(arg_layout, rs, aval)
                             else arg_layout)
    else:
      arg_layout, dispatch_arg_layout = None, None
    # Sharding can be unspecified when array is committed if it's a PmapSharding.
    is_pmap_sharding = (isinstance(rs, UnspecifiedValue) or
                        isinstance(getattr(arg, 'sharding', None), PmapSharding))
    if jit_in_l is None:
      if committed:
        if is_pmap_sharding:
          resolved_in_layouts.append(None)
        else:
          resolved_in_layouts.append(dispatch_arg_layout)
      else:
        resolved_in_layouts.append(None)
    else:
      # arg_layout can be None because some backends don't implement the
      # required layout methods. Hence `arr.layout` can return
      # `Layout(None, sharding)`
      if (committed
          and not is_pmap_sharding
          and arg_layout is not None
          and not pxla.is_user_xla_layout_equal(jit_in_l, arg_layout)):
        extra_msg = ''
        if isinstance(jit_in_l, AutoLayout):
          extra_msg = (
              ' The layout given to `jax.jit` is `DeviceLocalLayout.AUTO` but'
              ' the corresponding argument passed is a `jax.Array` with a'
              ' concrete layout. Consider passing a `jax.ShapeDtypeStruct`'
              ' instead of `jax.Array` as an argument to the jitted function '
              ' when using `DeviceLocalLayout.AUTO`.'
          )
        raise ValueError('Layout passed to jit does not match the layout '
                          'on the respective arg. '
                          f'Got pjit layout: {jit_in_l},\n'
                          f'arg layout: {arg_layout} for '
                          f'arg shape: {core.shaped_abstractify(arg).str_short()}.'
                          f'{extra_msg}')
      resolved_in_layouts.append(jit_in_l)
  return tuple(resolved_in_layouts)


def _resolve_in_shardings(args, pjit_in_shardings: Sequence[PjitSharding]
                          ) -> Sequence[PjitSharding]:
  # If True, means that device or backend is set by the user on pjit and it
  # has the same semantics as device_put i.e. doesn't matter which device the
  # arg is on, reshard it to the device mentioned. So don't do any of the
  # checks and just return the pjit_in_shardings directly. `shard_args` will
  # handle the resharding.
  if pxla.check_device_backend_on_shardings(pjit_in_shardings):
    return pjit_in_shardings

  committed_arg_shardings = []
  for a in args:
    arg_s = getattr(a, 'sharding', None)
    # arg sharding can be None in case of ShapeDtypeStruct. jax.Array does
    # not allow None as the sharding.
    if arg_s is None:
      continue
    # Don't consider PmapSharding inputs as committed. They will get resharded
    # unconditionally.
    if isinstance(arg_s, PmapSharding):
      continue
    if getattr(a, '_committed', True):
      committed_arg_shardings.append((arg_s, pxla.MismatchType.ARG_SHARDING, None))

  resolved_in_shardings: list[PjitSharding] = []
  for arg, pjit_in_s in zip(args, pjit_in_shardings):
    # arg sharding can be None in case of ShapeDtypeStruct. jax.Array does
    # not allow None as the sharding.
    arg_s, committed = ((arg.sharding, getattr(arg, '_committed', True))
                        if hasattr(arg, 'sharding') and arg.sharding is not None
                        else (UNSPECIFIED, False))
    if isinstance(pjit_in_s, UnspecifiedValue):
      if isinstance(arg_s, UnspecifiedValue):
        resolved_in_shardings.append(arg_s)
      else:
        if committed:
          # If the arg has a PmapSharding, then reshard it unconditionally.
          if isinstance(arg_s, PmapSharding):
            resolved_in_shardings.append(UNSPECIFIED)
          else:
            resolved_in_shardings.append(arg_s)
        else:
          assert isinstance(arg_s, Sharding)
          if dispatch.is_single_device_sharding(arg_s):
            resolved_in_shardings.append(UNSPECIFIED)
          else:
            raise NotImplementedError('Having uncommitted Array sharded on '
                                      'multiple devices is not supported.')
    else:
      if (isinstance(arg, np.ndarray) and
          not pjit_in_s.is_fully_replicated and  # type: ignore[union-attr]
          xb.process_count() > 1):
        raise ValueError(
            'Passing non-trivial shardings for numpy '
            'inputs is not allowed. To fix this error, either specify a '
            'replicated sharding explicitly or use '
            '`jax.experimental.multihost_utils.host_local_array_to_global_array(...)` '
            'to convert your host local numpy inputs to a jax.Array which you '
            'can pass to pjit. '
            'If the numpy input is the same on each process, then you can use '
            '`jax.make_array_from_callback(...) to create a `jax.Array` which '
            'you can pass to pjit. '
            'Please see the jax.Array migration guide for more information '
            'https://jax.readthedocs.io/en/latest/jax_array_migration.html#handling-of-host-local-inputs-to-pjit-like-batch-etc. '
            f'Got arg shape: {arg.shape}, arg value: {arg}')
      if not isinstance(arg_s, UnspecifiedValue):
        # jax.jit does not allow resharding across different memory kinds even
        # if the argument is uncommitted. Use jax.device_put for those cases,
        # either outside or inside jax.jit.
        if pjit_in_s.memory_kind != arg_s.memory_kind:  # type: ignore[union-attr]
          raise ValueError(
              'Memory kinds passed to jax.jit does not match memory kind on the'
              f' respective arg. Got pjit memory kind: {pjit_in_s.memory_kind}, '  # type: ignore[union-attr]
              f'arg memory kind: {arg_s.memory_kind} for '
              f'arg shape: {core.shaped_abstractify(arg).str_short()}')
        if (committed and
            not isinstance(arg_s, PmapSharding) and
            not op_shardings.are_op_shardings_equal(
                pjit_in_s._to_xla_hlo_sharding(arg.ndim),  # type: ignore[union-attr]
                arg_s._to_xla_hlo_sharding(arg.ndim))):
          raise ValueError('Sharding passed to pjit does not match the sharding '
                           'on the respective arg. '
                           f'Got pjit sharding: {pjit_in_s},\n'
                           f'arg sharding: {arg_s} for '
                           f'arg shape: {core.shaped_abstractify(arg).str_short()}')
      resolved_in_shardings.append(pjit_in_s)

  return tuple(resolved_in_shardings)


def _resolve_and_lower(
    args, jaxpr, in_shardings, out_shardings, in_layouts,
    out_layouts, resource_env, donated_invars, name, keep_unused, inline,
    lowering_platforms, lowering_parameters, pgle_profiler,
    compiler_options_kvs):
  in_shardings = _resolve_in_shardings(args, in_shardings)
  in_layouts = _resolve_in_layouts(args, in_layouts, in_shardings,
                                   jaxpr.in_avals)
  return _pjit_lower(
      jaxpr, in_shardings, out_shardings, in_layouts, out_layouts, resource_env,
      donated_invars, name, keep_unused, inline, compiler_options_kvs,
      lowering_platforms=lowering_platforms,
      lowering_parameters=lowering_parameters,
      pgle_profiler=pgle_profiler)

_pgle_profiler_dict = weakref.WeakKeyDictionary()  # type: ignore

def _pjit_call_impl_python(
    *args, jaxpr, in_shardings, out_shardings, in_layouts, out_layouts,
    resource_env, donated_invars, name, keep_unused, inline,
    compiler_options_kvs):
  pgle_compile_options, pgle_profiler = {}, None
  if config.enable_pgle.value and config.pgle_profiling_runs.value > 0:
    compilation_target_key = jaxpr
    pgle_profiler = _pgle_profiler_dict.get(compilation_target_key)
    if pgle_profiler is None:
      pgle_profiler = profiler.PGLEProfiler(
          config.pgle_profiling_runs.value,
          config.pgle_aggregation_percentile.value)
      _pgle_profiler_dict[compilation_target_key] = pgle_profiler

    # The method below will return FDO profile when module was profiled
    # config.jax_pgle_profiling_runs amount of times, otherwise the result will
    # be None.
    fdo_profile = pgle_profiler.consume_fdo_profile()
    if fdo_profile is not None:
      pgle_compile_options['fdo_profile'] = fdo_profile

  compiler_options_kvs = compiler_options_kvs + tuple(pgle_compile_options.items())
  # Passing mutable PGLE profile here since it should be extracted by JAXPR to
  # initialize the fdo_profile compile option.
  compiled = _resolve_and_lower(
      args, jaxpr=jaxpr, in_shardings=in_shardings,
      out_shardings=out_shardings, in_layouts=in_layouts,
      out_layouts=out_layouts, resource_env=resource_env,
      donated_invars=donated_invars, name=name, keep_unused=keep_unused,
      inline=inline, lowering_platforms=None,
      lowering_parameters=mlir.LoweringParameters(),
      pgle_profiler=pgle_profiler,
      compiler_options_kvs=compiler_options_kvs,
  ).compile()

  # This check is expensive so only do it if enable_checks is on.
  if compiled._auto_spmd_lowering and config.enable_checks.value:
    pxla.check_array_xla_sharding_layout_match(
        args, compiled._in_shardings, compiled._in_layouts,
        jaxpr.jaxpr.tracing_debug_info, compiled._kept_var_idx)
  if config.distributed_debug.value:
    # Defensively only perform fingerprint logic if debug logging is enabled
    # NOTE(skyewm): I didn't benchmark this
    fingerprint = None
    if hasattr(compiled.runtime_executable(), "fingerprint"):
      fingerprint = compiled.runtime_executable().fingerprint
    if fingerprint is not None:
      fingerprint = fingerprint.hex()
    distributed_debug_log(("Running pjit'd function", name),
                          ("in_shardings", in_shardings),
                          ("out_shardings", out_shardings),
                          ("in_layouts", in_layouts),
                          ("out_layouts", out_layouts),
                          ("abstract args", map(core.abstractify, args)),
                          ("fingerprint", fingerprint))
  try:
    return compiled.unsafe_call(*args), compiled, pgle_profiler
  except FloatingPointError as e:
    assert config.debug_nans.value or config.debug_infs.value  # compiled_fun can only raise in this case

    if len(jaxpr.eqns) > 1:
      _ = core.jaxpr_as_fun(jaxpr)(*args)  # may raise, not return

    # If control reaches this line, we got a NaN on the output of `compiled`
    # but not `fun.call_wrapped` on the same arguments. Let's tell the user.
    msg = (f"{str(e)}. Because "
           "jax_config.debug_nans.value and/or config.jax_debug_infs is set, the "
           "de-optimized function (i.e., the function as if the `jit` "
           "decorator were removed) was called in an attempt to get a more "
           "precise error message. However, the de-optimized function did not "
           "produce invalid values during its execution. This behavior can "
           "result from `jit` optimizations causing the invalid value to be "
           "produced. It may also arise from having nan/inf constants as "
           "outputs, like `jax.jit(lambda ...: jax.numpy.nan)(...)`. "
           "\n\n"
           "It may be possible to avoid the invalid value by removing the "
           "`jit` decorator, at the cost of losing optimizations. "
           "\n\n"
           "If you see this error, consider opening a bug report at "
           "https://github.com/jax-ml/jax.")
    raise FloatingPointError(msg)


@weakref_lru_cache
def _get_jaxpr_as_fun(jaxpr, in_shardings, out_shardings, in_layouts,
                      out_layouts, resource_env, donated_invars, name,
                      keep_unused, inline, compiler_options_kvs):
  # The input jaxpr to `_get_jaxpr_as_fun` is under a weakref_lru_cache so
  # returning `core.jaxpr_as_fun(jaxpr)` directly creates a strong reference to
  # the jaxpr defeating the purpose of weakref_lru_cache. So return a function
  # that closes over a weakrefed jaxpr and gets called inside that function.
  # This way there won't be a strong reference to the jaxpr from the output
  # function.
  jaxpr = weakref.ref(jaxpr)
  return lambda *args: core.jaxpr_as_fun(jaxpr())(*args)  # pylint: disable=unnecessary-lambda


def _pjit_call_impl(*args, jaxpr,
                    in_shardings, out_shardings, in_layouts, out_layouts,
                    resource_env, donated_invars, name, keep_unused, inline,
                    compiler_options_kvs):
  def call_impl_cache_miss(*args_, **kwargs_):
    out_flat, compiled, pgle_profiler = _pjit_call_impl_python(
        *args, jaxpr=jaxpr, in_shardings=in_shardings,
        out_shardings=out_shardings, in_layouts=in_layouts,
        out_layouts=out_layouts, resource_env=resource_env,
        donated_invars=donated_invars, name=name, keep_unused=keep_unused,
        inline=inline, compiler_options_kvs=compiler_options_kvs)
    fastpath_data = _get_fastpath_data(
        compiled, tree_structure(out_flat), args, out_flat, [], jaxpr.effects,
        jaxpr.consts, None, pgle_profiler)
    return out_flat, fastpath_data, _need_to_rebuild_with_fdo(pgle_profiler)

  f = _get_jaxpr_as_fun(
      jaxpr, in_shardings, out_shardings, in_layouts, out_layouts,
      resource_env, donated_invars, name, keep_unused, inline,
      compiler_options_kvs)
  donated_argnums = tuple(i for i, d in enumerate(donated_invars) if d)
  cache_key = pxla.JitGlobalCppCacheKeys(
      donate_argnums=donated_argnums, donate_argnames=None,
      device=None, backend=None,
      in_shardings_treedef=None, in_shardings_leaves=in_shardings,
      out_shardings_treedef=None, out_shardings_leaves=out_shardings,
      in_layouts_treedef=None, in_layouts_leaves=in_layouts,
      out_layouts_treedef=None, out_layouts_leaves=out_layouts,
      use_resource_env=resource_env is not None)
  return xc._xla.pjit(
      name, f, call_impl_cache_miss, [], [], cache_key,
      tree_util.dispatch_registry, pxla.cc_shard_arg,
      _get_cpp_global_cache(cache_key.contains_explicit_attributes))(*args)

pjit_p.def_impl(_pjit_call_impl)


def _pjit_lower(
    jaxpr: core.ClosedJaxpr,
    in_shardings,
    out_shardings,
    in_layouts: pxla.MaybeLayout,
    out_layouts: pxla.MaybeLayout,
    resource_env,
    donated_invars,
    name: str,
    keep_unused: bool,
    inline: bool,
    compiler_options_kvs: tuple[tuple[str, Any], ...],
    *,
    lowering_platforms: tuple[str, ...] | None,
    lowering_parameters: mlir.LoweringParameters,
    pgle_profiler: profiler.PGLEProfiler | None):
  util.test_event("pjit_lower")
  if config.sharding_in_types.value:
    mesh, api_name = mesh_lib.get_concrete_mesh(), 'jit'
  else:
    mesh, api_name = ((resource_env.physical_mesh, 'pjit')
                      if resource_env is not None else (None, 'jit'))
  return pxla.lower_sharding_computation(
      jaxpr, api_name, name, in_shardings, out_shardings,
      in_layouts, out_layouts, tuple(donated_invars),
      keep_unused=keep_unused, context_mesh=mesh,
      compiler_options_kvs=compiler_options_kvs,
      lowering_platforms=lowering_platforms,
      lowering_parameters=lowering_parameters,
      pgle_profiler=pgle_profiler)


def pjit_staging_rule(trace, *args, **params):
  jaxpr, in_fwd, out_shardings, out_layouts = _pjit_forwarding(
      params['jaxpr'], params['out_shardings'], params['out_layouts'])
  params = dict(params, jaxpr=jaxpr, out_shardings=out_shardings,
                out_layouts=out_layouts)
  if (params["inline"] and
      all(isinstance(i, UnspecifiedValue) for i in params["in_shardings"]) and
      all(isinstance(o, UnspecifiedValue) for o in params["out_shardings"]) and
      all(i is None for i in params["in_layouts"]) and
      all(o is None for o in params["out_layouts"])):
    if config.dynamic_shapes.value:
      # Inline jaxpr doesn't handle dynamic shapes when inlining. If dynamic
      # shapes are enabled, use eval_jaxpr, which uses the tracing machinery,
      # but redundantly performs abstract evaluation again.
      with core.set_current_trace(trace):
        out_tracers = core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args,
                                      propagate_source_info=False)
    else:
      out_tracers = pe.inline_jaxpr_into_trace(
          trace, jaxpr.jaxpr, jaxpr.consts, *args)
  elif config.dynamic_shapes.value:
    source_info = source_info_util.current()
    out_tracers = []
    for aval in _out_type(jaxpr):
      if type(aval) is core.DShapedArray:
        shape = [args[d.val] if type(d) is core.InDBIdx else
                 out_tracers[d.val] if type(d) is core.OutDBIdx else
                 d for d in aval.shape]
        aval = aval.update(shape=tuple(core.get_referent(d) for d in shape))
      out_tracers.append(pe.DynamicJaxprTracer(trace, aval, source_info))
    eqn = core.new_jaxpr_eqn(
      map(trace.getvar, args), map(trace.makevar, out_tracers), pjit_p, params,
      jaxpr.effects, source_info)
    trace.frame.add_eqn(eqn)
  elif any(isinstance(c, core.MutableArray) for c in jaxpr.consts):
    jaxpr, consts = pxla._move_mutable_consts(jaxpr)
    consts = map(trace.new_const, consts)
    in_shardings = (*params['in_shardings'],) + (UNSPECIFIED,) * len(consts)
    in_layouts = (*params['in_layouts'],) + (None,) * len(consts)
    donated_invars = (*params['donated_invars'],) + (False,) * len(consts)
    new_params = dict(params, jaxpr=jaxpr, in_shardings=in_shardings,
                      in_layouts=in_layouts, donated_invars=donated_invars)
    out_tracers = trace.default_process_primitive(
        pjit_p, (*args, *consts), new_params)
  else:
    out_tracers = trace.default_process_primitive(pjit_p, args, params)

  out_tracers_ = iter(out_tracers)
  out_tracers = [args[f] if type(f) is int else next(out_tracers_)
                 for f in in_fwd]
  assert next(out_tracers_, None) is None
  return out_tracers
pe.custom_staging_rules[pjit_p] = pjit_staging_rule


def _pjit_forwarding(jaxpr, out_shardings, out_layouts):
  in_fwd: list[int | None] = pe._jaxpr_forwarding(jaxpr.jaxpr)
  in_fwd = [fwd if isinstance(os, UnspecifiedValue) and ol is None else None for fwd, os, ol
            in zip(in_fwd, out_shardings, out_layouts)]
  keep = [f is None for f in in_fwd]
  jaxpr = pe.prune_closed_jaxpr_outputs(jaxpr, keep)
  out_shardings = [o for o, k in zip(out_shardings, keep) if k]
  out_layouts   = [o for o, k in zip(out_layouts  , keep) if k]
  return jaxpr, in_fwd, out_shardings, out_layouts

def pjit_forwarding_rule(eqn):
  jaxpr, in_fwd, out_shardings, out_layouts = _pjit_forwarding(
      eqn.params['jaxpr'], eqn.params['out_shardings'], eqn.params['out_layouts'])
  new_outvars = [v for v, f in zip(eqn.outvars, in_fwd) if f is None]
  new_params = dict(eqn.params, jaxpr=jaxpr, out_shardings=(*out_shardings,),
                    out_layouts=(*out_layouts,))
  new_eqn = eqn.replace(params=new_params, outvars=new_outvars)
  fwd_vars = [eqn.invars[f] if f is not None else None for f in in_fwd]
  return fwd_vars, new_eqn
pe.forwarding_rules[pjit_p] = pjit_forwarding_rule


# TODO(mattjj): remove/trivialize this when jaxprs have type annotation on them,
# since it's actually not possible in general to infer the type from the term
def _out_type(jaxpr: core.ClosedJaxpr) -> list[core.AbstractValue]:
  out = []
  in_idx = {v: i for i, v in enumerate(jaxpr.jaxpr.invars)}
  out_idx = {x: i for i, x in enumerate(jaxpr.jaxpr.invars)
             if type(x) is core.Var}
  for x in jaxpr.jaxpr.outvars:
    aval = x.aval
    if type(aval) is core.DShapedArray:
      shape = [core.InDBIdx(in_idx[d]) if d in in_idx else
               core.OutDBIdx(out_idx[d]) if d in out_idx else
               d for d in x.aval.shape]
      aval = aval.update(shape=tuple(shape))
    out.append(aval)
  return out


def _pjit_typecheck(ctx_factory, *in_atoms, jaxpr, **params):
  return core._check_call(ctx_factory, pjit_p, in_atoms,
                          dict(params, call_jaxpr=jaxpr.jaxpr))
core.custom_typechecks[pjit_p] = _pjit_typecheck


def _pjit_abstract_eval(*args, jaxpr, out_shardings, **_):
  return jaxpr.out_avals, jaxpr.effects
pjit_p.def_effectful_abstract_eval(_pjit_abstract_eval)


def _pjit_cached_lower_jaxpr_to_fun(ctx, name, jaxpr, effects, in_shardings,
                                    out_shardings, in_layouts, out_layouts,
                                    api_name):
  mod_ctx = ctx.module_context
  axis_ctx = ctx.module_context.axis_context
  num_devices = None
  if isinstance(axis_ctx, sharding_impls.ShardingContext):
    num_devices = axis_ctx.num_devices
  elif isinstance(axis_ctx, sharding_impls.SPMDAxisContext):
    num_devices = axis_ctx.mesh.size
  key = (pjit_p, name, jaxpr, effects, num_devices,
         pxla.SemanticallyEqualShardings(in_shardings, jaxpr.in_avals),
         pxla.SemanticallyEqualShardings(out_shardings, jaxpr.out_avals),
         in_layouts, out_layouts, api_name)

  func = mod_ctx.cached_primitive_lowerings.get(key, None)
  if func is None:
    arg_shardings = [None if isinstance(i, UnspecifiedValue) else i for i in in_shardings]
    result_shardings = [None if isinstance(o, UnspecifiedValue) else o for o in out_shardings]
    # TODO(b/228598865): inlined calls cannot have shardings set directly on the
    # inputs or outputs because they are lost during MLIR->HLO conversion.
    # using_sharding_annotation=False means we add an identity operation instead.
    func = mlir.lower_jaxpr_to_fun(
        mod_ctx, name, jaxpr, effects, ctx.name_stack,
        arg_shardings=arg_shardings, result_shardings=result_shardings,
        use_sharding_annotations=False, api_name=api_name,
        arg_layouts=in_layouts, result_layouts=out_layouts)
    mod_ctx.cached_primitive_lowerings[key] = func
  return func


def _pjit_lowering(ctx, *args, name, jaxpr, in_shardings,
                   out_shardings, in_layouts, out_layouts, resource_env,
                   donated_invars, keep_unused, inline, compiler_options_kvs):
  effects = list(ctx.tokens_in.effects())
  output_types = map(mlir.aval_to_ir_type, ctx.avals_out)
  output_types = [mlir.token_type()] * len(effects) + output_types
  flat_output_types = mlir.flatten_ir_types(output_types)

  func = _pjit_cached_lower_jaxpr_to_fun(
      ctx, name, jaxpr, tuple(effects), in_shardings,
      out_shardings, in_layouts, out_layouts,
      api_name=('jit' if resource_env is None else 'pjit'))

  tokens_in = [ctx.tokens_in.get(eff) for eff in effects]
  args = (*ctx.dim_var_values, *tokens_in, *args)
  call = func_dialect.CallOp(flat_output_types,
                             ir.FlatSymbolRefAttr.get(func.name.value),
                             mlir.flatten_ir_values(args))
  mlir.wrap_compute_type_in_place(ctx, call)
  out_nodes = mlir.unflatten_ir_values_like_types(call.results, output_types)
  tokens, out_nodes = split_list(out_nodes, [len(effects)])
  tokens_out = ctx.tokens_in.update_tokens(mlir.TokenSet(zip(effects, tokens)))
  ctx.set_tokens_out(tokens_out)
  return out_nodes

mlir.register_lowering(pjit_p, _pjit_lowering)


def _pjit_batcher(axis_data, vals_in, dims_in,
                  jaxpr, in_shardings, out_shardings, in_layouts, out_layouts,
                  resource_env, donated_invars, name, keep_unused, inline,
                  compiler_options_kvs):
  segment_lens, dims_in = batching.indirectify_ragged_axes(dims_in)
  new_jaxpr, axes_out = batching.batch_jaxpr2(jaxpr, axis_data, dims_in)

  if resource_env is not None:
    mesh = resource_env.physical_mesh
  else:
    mesh = None

  # TODO(axch): prepend with Nones (?) to account for new segment_lens inputs
  in_shardings = tuple(
      _pjit_batcher_for_sharding(i, axis_in, axis_data.spmd_name, mesh, aval.ndim)
      if axis_in is not None else i
      for axis_in, i, aval in zip(dims_in, in_shardings, new_jaxpr.in_avals))
  out_shardings = tuple(
      _pjit_batcher_for_sharding(o, axis_out, axis_data.spmd_name, mesh, aval.ndim)
      if axis_out is not None else o
      for axis_out, o, aval in zip(axes_out, out_shardings, new_jaxpr.out_avals))
  # TODO(yashkatariya): Figure out layouts should change under vmap.
  if not (all(l is None for l in in_layouts) and
          all(l is None for l in out_layouts)):
    raise NotImplementedError(
        'Concrete layouts are not supported for vmap(jit).')

  vals_out = pjit_p.bind(
    *vals_in,
    jaxpr=new_jaxpr,
    in_shardings=in_shardings,
    out_shardings=out_shardings,
    in_layouts=in_layouts,
    out_layouts=out_layouts,
    resource_env=resource_env,
    donated_invars=donated_invars,
    name=name,
    keep_unused=keep_unused,
    inline=inline,
    compiler_options_kvs=compiler_options_kvs)

  resolved_axes_out = batching.resolve_ragged_axes_against_inputs_outputs(
      vals_in, vals_out, axes_out)
  return vals_out, resolved_axes_out

batching.fancy_primitive_batchers[pjit_p] = _pjit_batcher
batching.ragged_prop_rules[pjit_p] = batching.ragged_mask_no_op_rule

def _pjit_batcher_for_sharding(
    s: Sharding | UnspecifiedValue,
    dim: int, spmd_axis_name: tuple[str, ...] | None, mesh, ndim: int):
  if isinstance(s, UnspecifiedValue):
    return s
  hlo_s = s._to_xla_hlo_sharding(ndim)
  if spmd_axis_name is None:
    if sharding_impls.is_op_sharding_replicated(hlo_s):
      return s
    if isinstance(s, NamedSharding) and isinstance(s.mesh, AbstractMesh):
      parsed_pspec = s._parsed_pspec.insert_axis_partitions(dim, None)
      return NamedSharding._from_parsed_pspec(s.mesh, parsed_pspec)
    new_op = hlo_s.to_proto().clone()
    tad = list(new_op.tile_assignment_dimensions)
    tad.insert(dim, 1)
    new_op.tile_assignment_dimensions = tad
    new_gs = GSPMDSharding(
        s._device_assignment, new_op,
        _device_list=getattr(s, '_internal_device_list', None))
    return pxla._get_out_sharding_from_orig_sharding([new_gs], [None], s, None)[0]
  else:
    if isinstance(s, NamedSharding) and isinstance(s.mesh, AbstractMesh):
      parsed_pspec = s._parsed_pspec.insert_axis_partitions(dim, spmd_axis_name)
      return NamedSharding._from_parsed_pspec(s.mesh, parsed_pspec)
    if isinstance(s, NamedSharding):
      mesh = s.mesh
    if mesh is None or mesh.empty:
      raise ValueError(
          'If you are using spmd_axis_name parameter of jax.vmap,'
          ' please make sure to run your jitted function inside the mesh'
          ' context manager. Only `jax.lax.with_sharding_constraint` with'
          ' `jax.sharding.NamedSharding` as an input can be transformed with'
          ' spmd_axis_name batching rules outside of an explicit mesh context'
          f' manager scope{s!r}')
    parsed_pspec = parse_flatten_op_sharding(hlo_s, mesh)[0]
    parsed_pspec = parsed_pspec.insert_axis_partitions(dim, spmd_axis_name)
    return NamedSharding._from_parsed_pspec(mesh, parsed_pspec)


def _pjit_jvp(primals_in, tangents_in,
              jaxpr, in_shardings, out_shardings, in_layouts, out_layouts,
              resource_env, donated_invars, name, keep_unused, inline,
              compiler_options_kvs):
  if any(isinstance(c, core.MutableArray) for c in jaxpr.consts):
    jaxpr, mut_primals = pxla._move_mutable_consts(jaxpr)
    mut_tangents = map(ad_util.zeros_like_jaxval, mut_primals)
    primals_in = [*primals_in, *mut_primals]
    tangents_in = [*tangents_in, *mut_tangents]
    in_shardings = (*in_shardings,) + (UNSPECIFIED,) * len(mut_primals)
    in_layouts = (*in_layouts,) + (None,) * len(mut_primals)
    donated_invars = (*donated_invars,) + (False,) * len(mut_primals)

  tangents_in = [ad_util.zeros_like_aval(a) if isinstance(a, AbstractRef) else x
                 for x, a in zip(tangents_in, jaxpr.in_avals)]

  is_nz_tangents_in = [type(t) is not ad.Zero for t in tangents_in]
  jaxpr_jvp, is_nz_tangents_out = ad.jvp_jaxpr(
      jaxpr, is_nz_tangents_in, instantiate=False)

  def _filter_zeros(is_nz_l, l):
    return (x for nz, x in zip(is_nz_l, l) if nz)
  _filter_zeros_in = partial(_filter_zeros, is_nz_tangents_in)
  _filter_zeros_out = partial(_filter_zeros, is_nz_tangents_out)
  outputs = pjit_p.bind(
      *primals_in, *_filter_zeros_in(tangents_in),
      jaxpr=jaxpr_jvp,
      in_shardings=(*in_shardings, *_filter_zeros_in(in_shardings)),
      out_shardings=(*out_shardings, *_filter_zeros_out(out_shardings)),
      in_layouts=(*in_layouts, *_filter_zeros_in(in_layouts)),
      out_layouts=(*out_layouts, *_filter_zeros_out(out_layouts)),
      resource_env=resource_env,
      donated_invars=(*donated_invars, *_filter_zeros_in(donated_invars)),
      name=name,
      keep_unused=keep_unused,
      inline=inline,
      compiler_options_kvs=compiler_options_kvs)

  primals_out, tangents_out = split_list(outputs, [len(jaxpr.jaxpr.outvars)])
  assert len(primals_out) == len(jaxpr.jaxpr.outvars)
  tangents_out_it = iter(tangents_out)
  return primals_out, [next(tangents_out_it) if nz else ad.Zero(aval)
                       for nz, aval in zip(is_nz_tangents_out, jaxpr.out_avals)]
ad.primitive_jvps[pjit_p] = _pjit_jvp


def _pjit_linearization(nzs, *primals_in, jaxpr,
                        in_shardings, out_shardings, in_layouts, out_layouts,
                        resource_env, donated_invars, name, keep_unused, inline,
                        compiler_options_kvs):
  primal_jaxpr, num_residuals, nzs_out, tangent_jaxpr = ad.linearize_jaxpr(jaxpr, nzs)
  # constvars will become residuals. Move them to the end of the ordinary args.
  res_shardings = (UNSPECIFIED,) * num_residuals
  res_layouts = (None,) * num_residuals
  res_donated = (False,) * num_residuals
  def tangent_fun(consts_, *tangents):
    tangents_nz = _filter_zeros(nzs, tangents)
    assert len(consts_) == num_residuals
    nz_tangents_out = pjit_p.bind(*(*tangents_nz, *consts_),
        jaxpr=tangent_jaxpr,
        in_shardings=_filter_zeros(nzs, in_shardings) + res_shardings,
        out_shardings=_filter_zeros(nzs_out, out_shardings),
        in_layouts=_filter_zeros(nzs, in_layouts) + res_layouts,
        out_layouts=_filter_zeros(nzs_out, out_layouts),
        resource_env=resource_env,
        donated_invars=_filter_zeros(nzs, donated_invars) + res_donated,
        name=name,
        keep_unused=keep_unused,
        inline=inline,
        compiler_options_kvs=compiler_options_kvs)
    tangent_avals_out = [v.aval.to_tangent_aval() for v in jaxpr.jaxpr.outvars]
    nz_tangents_out_ = iter(nz_tangents_out)
    tangents_out = [next(nz_tangents_out_) if nz else ad.Zero(aval)
                   for (aval, nz) in zip(tangent_avals_out, nzs_out)]
    return tangents_out

  def _filter_zeros(is_nz_l, l):
    return tuple(x for nz, x in zip(is_nz_l, l) if nz)

  ans = pjit_p.bind(*primals_in, jaxpr=primal_jaxpr,
                    in_shardings=in_shardings,
                    out_shardings=(*res_shardings, *out_shardings),
                    in_layouts=in_layouts,
                    out_layouts=(*res_layouts, *out_layouts),
                    resource_env=resource_env,
                    donated_invars=donated_invars,
                    name=name,
                    keep_unused=keep_unused,
                    inline=inline,
                    compiler_options_kvs=compiler_options_kvs)
  residuals_ans, primal_ans = split_list(ans, [num_residuals])

  return primal_ans, nzs_out, residuals_ans, tangent_fun

ad.primitive_linearizations[pjit_p] = _pjit_linearization


def _pjit_partial_eval(trace, *in_tracers,
                       jaxpr, in_shardings, out_shardings,
                       in_layouts, out_layouts, resource_env, donated_invars,
                       name, keep_unused, inline, compiler_options_kvs):
  in_pvals = [t.pval for t in in_tracers]

  known_ins = tuple(pv.is_known() for pv in in_pvals)
  unknown_ins = tuple(not k for k in known_ins)
  if any(isinstance(e, (RefEffect, core.InternalMutableArrayEffect))
         for e in jaxpr.effects):
    known_jaxpr_, unknown_jaxpr_, unknown_outs, _, num_res_val, num_res_ref = \
        pe.partial_eval_jaxpr_stateful(jaxpr.jaxpr, unknown_ins, unknown_ins,
                                       False, False, None)
    if num_res_ref: raise NotImplementedError
    known_jaxpr = pe.ClosedJaxpr(known_jaxpr_, jaxpr.consts)
    unknown_jaxpr = pe.ClosedJaxpr(unknown_jaxpr_, jaxpr.consts)
    res_avals = unknown_jaxpr.in_avals[:num_res_val]
  else:
    known_jaxpr, unknown_jaxpr, unknown_outs, res_avals = \
        pe.partial_eval_jaxpr_nounits(jaxpr, unknown_ins, instantiate=False)
  unknown_outs = tuple(unknown_outs)
  known_outs = tuple(not uk for uk in unknown_outs)
  num_residuals = len(res_avals)
  res_shardings = (UNSPECIFIED,) * num_residuals
  res_layouts = (None,) * num_residuals

  def keep_where(l, should_keep):
    return tuple(x for x, keep in zip(l, should_keep) if keep)

  known_out_shardings = keep_where(out_shardings, known_outs) + res_shardings
  known_out_layouts = keep_where(out_layouts, known_outs) + res_layouts

  # Input-to-output forwarding: compute which outputs are just forwarded inputs.
  num_out_primals = len(known_jaxpr.out_avals) - num_residuals
  in_fwd: list[int | None] = pe._jaxpr_forwarding(known_jaxpr.jaxpr)
  # Only forward primal outputs when corresponding out_sharding is UNSPECIFIED.
  in_fwd_primal, in_fwd_res = split_list(in_fwd, [num_out_primals])
  in_fwd = [
      fwd if isinstance(os, UnspecifiedValue) and ol is None else None
      for os, ol, fwd in zip(
          keep_where(out_shardings, known_outs),
          keep_where(out_layouts, known_outs), in_fwd_primal)
  ] + in_fwd_res
  del in_fwd_primal, in_fwd_res
  # Prune jaxpr outputs and out_shardings by removing the input-forwards.
  keep = [f is None for f in in_fwd]
  known_jaxpr = pe.prune_closed_jaxpr_outputs(known_jaxpr, keep)
  known_out_shardings = keep_where(known_out_shardings, keep)
  known_out_layouts = keep_where(known_out_layouts, keep)
  # Update num_out_primals to reflect pruning.
  kept_primals, kept_res = split_list(keep, [num_out_primals])
  num_out_primals = sum(kept_primals)
  del keep, kept_primals, kept_res

  # Output-to-output forwarding: compute which residuals are just primal outputs
  out_vars, res_vars = split_list(known_jaxpr.jaxpr.outvars, [num_out_primals])
  idx_map = {id(v): i for i, v in enumerate(out_vars)}
  out_fwd = [None] * num_out_primals + [idx_map.get(id(v)) for v in res_vars]
  # Prune jaxpr outputs and out_shardings by removing forwarded residuals.
  keep = [f is None for f in out_fwd]
  known_jaxpr = pe.prune_closed_jaxpr_outputs(known_jaxpr, keep)
  known_out_shardings = keep_where(known_out_shardings, keep)
  known_out_layouts = keep_where(known_out_layouts, keep)
  del keep

  known_params = dict(
      jaxpr=known_jaxpr, in_shardings=keep_where(in_shardings, known_ins),
      out_shardings=known_out_shardings,
      in_layouts=keep_where(in_layouts, known_ins),
      out_layouts=known_out_layouts, resource_env=resource_env,
      donated_invars=keep_where(donated_invars, known_ins),
      name=name, keep_unused=keep_unused, inline=inline,
      compiler_options_kvs=compiler_options_kvs)
  assert len(known_params['out_shardings']) == len(known_params['jaxpr'].out_avals)
  assert len(known_params['out_layouts']) == len(known_params['jaxpr'].out_avals)

  # Bind known things to pjit_p.
  known_inputs = [pv.get_known() for pv in in_pvals if pv.is_known()]
  all_known_outs = pjit_p.bind(*known_inputs, **known_params)
  # Add back in the output fwds.
  all_known_outs = subs_list(out_fwd, all_known_outs, all_known_outs)
  # Add back in the input fwds.
  all_known_outs = subs_list(in_fwd, known_inputs, all_known_outs)

  known_out_vals, residual_vals = \
      split_list(all_known_outs, [len(all_known_outs) - num_residuals])
  residual_tracers = map(trace.new_instantiated_const, residual_vals)

  # The convention of partial_eval_jaxpr_nounits is to place residual binders at
  # the front of the jaxpr produced, so we move them to the back since both the
  # jaxpr equation built below and the pjit transpose rule assume a
  # residual-inputs-last convention.
  unknown_jaxpr = pe.move_binders_to_back(
      unknown_jaxpr, [True] * num_residuals + [False] * sum(unknown_ins))
  # Prepare unknown tracers
  unknown_params = dict(
      jaxpr=unknown_jaxpr,
      in_shardings=(keep_where(in_shardings, unknown_ins) + res_shardings),
      out_shardings=keep_where(out_shardings, unknown_outs),
      in_layouts=(keep_where(in_layouts, unknown_ins) + res_layouts),
      out_layouts=keep_where(out_layouts, unknown_outs),
      resource_env=resource_env,
      donated_invars=(keep_where(donated_invars, unknown_ins) +
                      (False,) * num_residuals),
      name=name,
      keep_unused=keep_unused,
      inline=inline,
      compiler_options_kvs=compiler_options_kvs)
  unknown_tracers_in = [t for t in in_tracers if not t.pval.is_known()]
  unknown_out_avals = unknown_jaxpr.out_avals
  unknown_tracers_out = [
      pe.JaxprTracer(trace, pe.PartialVal.unknown(aval), None)
      for aval in unknown_out_avals
  ]
  eqn = pe.new_eqn_recipe((*unknown_tracers_in, *residual_tracers),
                          unknown_tracers_out,
                          pjit_p,
                          unknown_params,
                          unknown_jaxpr.effects,
                          source_info_util.current())
  for t in unknown_tracers_out: t.recipe = eqn
  return merge_lists(unknown_outs, known_out_vals, unknown_tracers_out)

pe.custom_partial_eval_rules[pjit_p] = _pjit_partial_eval


def _pjit_partial_eval_custom_params_updater(
    unks_in: Sequence[bool], inst_in: Sequence[bool],
    kept_outs_known: Sequence[bool], kept_outs_staged: Sequence[bool],
    num_res_out: int, num_res_in: int, params_known: dict, params_staged: dict
  ) -> tuple[dict, dict]:
  # prune inputs to jaxpr_known according to unks_in
  donated_invars_known, _ = pe.partition_list(unks_in, params_known['donated_invars'])
  in_shardings_known, _ = pe.partition_list(unks_in, params_known['in_shardings'])
  _, out_shardings_known = pe.partition_list(kept_outs_known, params_known['out_shardings'])
  in_layouts_known, _ = pe.partition_list(unks_in, params_known['in_layouts'])
  _, out_layouts_known = pe.partition_list(kept_outs_known, params_known['out_layouts'])

  new_params_known = dict(params_known,
                          in_shardings=tuple(in_shardings_known),
                          out_shardings=(*out_shardings_known,
                                         *[UNSPECIFIED] * num_res_out),
                          in_layouts=tuple(in_layouts_known),
                          out_layouts=(*out_layouts_known, *[None] * num_res_out),
                          donated_invars=tuple(donated_invars_known))
  assert len(new_params_known['in_shardings']) == len(params_known['jaxpr'].in_avals)
  assert len(new_params_known['out_shardings']) == len(params_known['jaxpr'].out_avals)
  assert len(new_params_known['in_layouts']) == len(params_known['jaxpr'].in_avals)
  assert len(new_params_known['out_layouts']) == len(params_known['jaxpr'].out_avals)

  # added num_res new inputs to jaxpr_staged, and pruning according to inst_in
  _, donated_invars_staged = pe.partition_list(inst_in, params_staged['donated_invars'])
  donated_invars_staged = [False] * num_res_in + donated_invars_staged
  _, in_shardings_staged = pe.partition_list(inst_in, params_staged['in_shardings'])
  in_shardings_staged = [*[UNSPECIFIED] * num_res_in, *in_shardings_staged]
  _, out_shardings_staged = pe.partition_list(kept_outs_staged, params_staged['out_shardings'])
  _, in_layouts_staged = pe.partition_list(inst_in, params_staged['in_layouts'])
  in_layouts_staged = [*[None] * num_res_in, *in_layouts_staged]
  _, out_layouts_staged = pe.partition_list(kept_outs_staged, params_staged['out_layouts'])

  new_params_staged = dict(params_staged,
                           in_shardings=tuple(in_shardings_staged),
                           out_shardings=tuple(out_shardings_staged),
                           in_layouts=tuple(in_layouts_staged),
                           out_layouts=tuple(out_layouts_staged),
                           donated_invars=tuple(donated_invars_staged))
  assert len(new_params_staged['in_shardings']) == len(params_staged['jaxpr'].in_avals)
  assert len(new_params_staged['out_shardings']) == len(params_staged['jaxpr'].out_avals)
  assert len(new_params_staged['in_layouts']) == len(params_staged['jaxpr'].in_avals)
  assert len(new_params_staged['out_layouts']) == len(params_staged['jaxpr'].out_avals)
  return new_params_known, new_params_staged

pe.partial_eval_jaxpr_custom_rules[pjit_p] = \
    partial(pe.closed_call_partial_eval_custom_rule, 'jaxpr',
            _pjit_partial_eval_custom_params_updater)


@lu.cache
def _pjit_transpose_trace(fun, in_avals):
  transpose_jaxpr, _, consts, attrs_tracked = pe.trace_to_jaxpr_dynamic(
      fun, in_avals)
  transpose_jaxpr = core.ClosedJaxpr(transpose_jaxpr, consts)
  return transpose_jaxpr, attrs_tracked


def _pjit_transpose(cts_in, *primals_in,
                    jaxpr, in_shardings, out_shardings, in_layouts, out_layouts,
                    resource_env, donated_invars, name, keep_unused, inline,
                    compiler_options_kvs):
  def prune_type(ty, xs, maybe_zeros):
    return tuple(x for x, mz in zip(xs, maybe_zeros) if type(mz) is not ty)

  body = lu.wrap_init(ad.closed_backward_pass)
  body = lu.hashable_partial(body, jaxpr, False)
  primals_and_nz_cts_in, in_treedef = tree_flatten((primals_in, cts_in))
  body, cts_out_treedef_thunk = flatten_fun_nokwargs(body, in_treedef)

  transpose_in_shardings = (
    *prune_type(ad.UndefinedPrimal, in_shardings, primals_in),
    *prune_type(ad.Zero, out_shardings, cts_in)
  )
  transpose_in_layouts = (
    *prune_type(ad.UndefinedPrimal, in_layouts, primals_in),
    *prune_type(ad.Zero, out_layouts, cts_in)
  )
  global_cts_in_avals = tuple(core.get_aval(ct) for ct in primals_and_nz_cts_in)

  transpose_jaxpr, attrs_tracked = _pjit_transpose_trace(
      body, global_cts_in_avals)
  cts_out_treedef = cts_out_treedef_thunk()
  transpose_out_shardings = prune_type(
      ad.Zero,
      in_shardings,
      tree_unflatten(cts_out_treedef, [object()] * cts_out_treedef.num_leaves))
  transpose_out_layouts = prune_type(
      ad.Zero,
      in_layouts,
      tree_unflatten(cts_out_treedef, [object()] * cts_out_treedef.num_leaves))

  if attrs_tracked:
    init_states =  _get_states(attrs_tracked)
    primals_and_nz_cts_in = [*init_states, *primals_and_nz_cts_in]
    transpose_in_shardings = (UNSPECIFIED,) * len(attrs_tracked) + transpose_in_shardings
    transpose_out_shardings = (UNSPECIFIED,) * len(attrs_tracked) + transpose_out_shardings
    transpose_in_layouts = (None,) * len(attrs_tracked) + transpose_in_layouts
    transpose_out_layouts = (None,) * len(attrs_tracked) + transpose_out_layouts

  nz_cts_out = pjit_p.bind(
      *primals_and_nz_cts_in,
      jaxpr=transpose_jaxpr,
      in_shardings=transpose_in_shardings,
      out_shardings=transpose_out_shardings,
      in_layouts=transpose_in_layouts,
      out_layouts=transpose_out_layouts,
      resource_env=resource_env,
      donated_invars=(False,) * len(primals_and_nz_cts_in),
      name=name,
      keep_unused=keep_unused,
      inline=inline,
      compiler_options_kvs=compiler_options_kvs)

  if attrs_tracked:
    final_states, nz_cts_out = split_list(nz_cts_out, [len(init_states)])
    _set_states(attrs_tracked, final_states)

  return tree_unflatten(cts_out_treedef, nz_cts_out)
ad.primitive_transposes[pjit_p] = _pjit_transpose


@weakref_lru_cache
def _dce_jaxpr_pjit(
    jaxpr: core.ClosedJaxpr, used_outputs: tuple[bool, ...]
) -> tuple[core.ClosedJaxpr, list[bool]]:
  new_jaxpr, used_inputs = pe.dce_jaxpr(jaxpr.jaxpr, used_outputs)
  return core.ClosedJaxpr(new_jaxpr, jaxpr.consts), used_inputs


def dce_jaxpr_pjit_rule(used_outputs: list[bool], eqn: core.JaxprEqn
                        ) -> tuple[list[bool], core.JaxprEqn | None]:

  if not any(used_outputs) and not pe.has_effects(eqn):
    return [False] * len(eqn.invars), None

  dced_jaxpr, used_inputs = _dce_jaxpr_pjit(
      eqn.params['jaxpr'], tuple(used_outputs))

  def keep_where(xs, keeps):
    return tuple(x for x, keep in zip(xs, keeps) if keep)

  eqn_params = eqn.params
  new_params = dict(
      eqn_params,
      jaxpr=dced_jaxpr,
      in_shardings=keep_where(eqn_params["in_shardings"], used_inputs),
      out_shardings=keep_where(eqn_params["out_shardings"], used_outputs),
      in_layouts=keep_where(eqn_params["in_layouts"], used_inputs),
      out_layouts=keep_where(eqn_params["out_layouts"], used_outputs),
      donated_invars=keep_where(eqn_params["donated_invars"], used_inputs),
  )
  if not any(used_inputs) and not any(used_outputs) and not dced_jaxpr.effects:
    return used_inputs, None
  else:
    new_eqn = core.new_jaxpr_eqn(
        [v for v, used in zip(eqn.invars, used_inputs) if used],
        [v for v, used in zip(eqn.outvars, used_outputs) if used],
        eqn.primitive, new_params, dced_jaxpr.effects, eqn.source_info, eqn.ctx)
    return used_inputs, new_eqn

pe.dce_rules[pjit_p] = dce_jaxpr_pjit_rule


def _pjit_pp_rule(eqn, context, settings):
  params = dict(eqn.params)
  del params['inline']
  if not any(params['donated_invars']):
    del params['donated_invars']
  if all(isinstance(s, UnspecifiedValue) for s in params['in_shardings']):
    del params['in_shardings']
  if all(isinstance(s, UnspecifiedValue) for s in params['out_shardings']):
    del params['out_shardings']
  if all(l is None for l in params['in_layouts']):
    del params['in_layouts']
  if all(l is None for l in params['out_layouts']):
    del params['out_layouts']
  if not params['keep_unused']:
    del params['keep_unused']
  if (params['resource_env'] is None or
      params['resource_env'].physical_mesh.empty):
    del params['resource_env']
  if not params['compiler_options_kvs']:
    del params['compiler_options_kvs']

  # Move name= to the front to make the resulting equation easier to scan.
  del params["name"]
  return core._pp_eqn(eqn, context, settings, params=["name"] + sorted(params))

core.pp_eqn_rules[pjit_p] = _pjit_pp_rule


def _pjit_state_discharge_rule(
    in_avals, out_avals, *args, jaxpr, in_shardings, out_shardings,
    in_layouts, out_layouts, **params):
  if not all(isinstance(s, UnspecifiedValue) for s in (*in_shardings, *out_shardings)):
    raise NotImplementedError

  if not (all(l is None for l in in_layouts) and
          all(l is None for l in out_layouts)):
    raise NotImplementedError

  jaxpr, consts = jaxpr.jaxpr, jaxpr.consts
  num_outs = len(jaxpr.outvars)
  discharged_jaxpr, discharged_consts = state_discharge.discharge_state(jaxpr, consts)
  discharged_closed_jaxpr = core.ClosedJaxpr(discharged_jaxpr, discharged_consts)
  new_in_shardings = (UnspecifiedValue(),) * len(discharged_jaxpr.invars)
  new_out_shardings = (UnspecifiedValue(),) * len(discharged_jaxpr.outvars)
  new_in_layouts = (None,) * len(discharged_jaxpr.invars)
  new_out_layouts = (None,) * len(discharged_jaxpr.outvars)
  out_and_ref_vals = pjit_p.bind(
      *args, jaxpr=discharged_closed_jaxpr, in_shardings=new_in_shardings,
      out_shardings=new_out_shardings, in_layouts=new_in_layouts,
      out_layouts=new_out_layouts, **params)
  out_vals, ref_vals = split_list(out_and_ref_vals, [num_outs])
  ref_vals_iter = iter(ref_vals)
  new_invals = tuple(next(ref_vals_iter) if isinstance(aval, AbstractRef)
                     else None for aval in in_avals)
  sentinel = object()
  assert next(ref_vals_iter, sentinel) is sentinel
  return new_invals, out_vals
state_discharge.register_discharge_rule(pjit_p)(_pjit_state_discharge_rule)


# -------------------- with_sharding_constraint --------------------

def with_sharding_constraint(x, shardings):
  """Mechanism to constrain the sharding of an Array inside a jitted computation

  This is a strict constraint for the GSPMD partitioner and not a hint. For examples
  of how to use this function, see `Distributed arrays and automatic parallelization`_.

  Args:
    x: PyTree of jax.Arrays which will have their shardings constrained
    shardings: PyTree of sharding specifications. Valid values are the same as for
      the ``in_shardings`` argument of :func:`jax.experimental.pjit`.
  Returns:
    x_with_shardings: PyTree of jax.Arrays with specified sharding constraints.

  .. _Distributed arrays and automatic parallelization: https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
  """
  x_flat, tree = tree_flatten(x)

  layouts, shardings = _split_layout_and_sharding(shardings)

  user_shardings = prepare_axis_resources(
      shardings, "shardings", allow_unconstrained_dims=True)
  del shardings

  user_shardings_flat = tuple(
      flatten_axes("with_sharding_constraint shardings", tree, user_shardings))
  del user_shardings

  user_layouts_flat = tuple(
      flatten_axes("with_sharding_constraint layouts", tree, layouts))
  del layouts

  resource_env = mesh_lib.thread_resources.env
  mesh = resource_env.physical_mesh

  shardings_flat = [_create_sharding_for_array(mesh, a, 'shardings',
                                               'with_sharding_constraint')
                    for a in user_shardings_flat]
  for s, u in zip(shardings_flat, user_shardings_flat):
    if isinstance(s, (UnspecifiedValue, AUTO)):
      raise ValueError(
          f'One of with_sharding_constraint arguments got sharding {u} which is'
          ' not allowed. Please only pass `jax.sharding.Sharding` instances.')
  del user_shardings_flat

  # TODO(bartchr): remove `unconstrained_dims` after migrating to Shardy. It's
  # already part of the shardings.
  unconstrained_dims = [get_unconstrained_dims(s)
                        if isinstance(s, NamedSharding) else {}
                        for s in shardings_flat]

  pjit_check_aval_sharding(
      shardings_flat, x_flat, None, "with_sharding_constraint arguments",
      allow_uneven_sharding=True)

  check_aval_layout_compatibility(user_layouts_flat, x_flat, None,
                                  "with_sharding_constraint arguments")

  outs = [sharding_constraint_p.bind(xf, sharding=s, layout=l,
                                     resource_env=resource_env,
                                     unconstrained_dims=ud)
          for xf, s, l, ud in zip(x_flat, shardings_flat, user_layouts_flat,
                                  unconstrained_dims)]
  return tree_unflatten(tree, outs)

def _identity_fn(x): return x

def _sharding_constraint_impl(x, sharding, layout, resource_env,
                              unconstrained_dims):
  if (isinstance(sharding, NamedSharding) and
      isinstance(sharding.mesh, AbstractMesh)):
    aval = core.shaped_abstractify(x)
    if not hasattr(x, 'sharding'):
      raise ValueError(
          'Target sharding contains a `jax.sharding.AbstractMesh` which'
          ' requires the input passed should be a `jax.Array`. Got'
          f' {type(x)} with shape {aval.str_short()}')
    if not isinstance(x.sharding, NamedSharding):
      raise TypeError(
          'The sharding on the input must be a `NamedSharding` since the target'
          ' sharding has an `AbstractMesh` in it. Got sharding type'
          f' {type(x.sharding)} for shape {aval.str_short()}')
    if x.sharding.mesh.shape_tuple != sharding.mesh.shape_tuple:
      raise ValueError(
          f'Mesh shape of the input {x.sharding.mesh.shape_tuple} does not'
          ' match the mesh shape of the target sharding'
          f' {sharding.mesh.shape_tuple} for shape {aval.str_short()}')
    sharding = NamedSharding._from_parsed_pspec(
        x.sharding.mesh, sharding._parsed_pspec)

  if layout is None:
    if hasattr(x, 'sharding') and x.sharding.is_equivalent_to(sharding, x.ndim):
      return x
    # Run a jit here to raise good errors when device assignment don't match.
    return api.jit(_identity_fn, out_shardings=sharding)(x)
  else:
    if (hasattr(x, 'layout') and x.layout.device_local_layout == layout and
        x.sharding.is_equivalent_to(sharding, x.ndim)):
      return x
    return api.jit(_identity_fn, out_shardings=Layout(layout, sharding))(x)


sharding_constraint_p = core.Primitive("sharding_constraint")
sharding_constraint_p.def_impl(_sharding_constraint_impl)
sharding_constraint_p.def_abstract_eval(lambda x, **_: x)
ad.deflinear2(sharding_constraint_p,
              lambda ct, _, **params: (sharding_constraint_p.bind(ct, **params),))

def _sharding_constraint_hlo_lowering(ctx, x_node, *, sharding, layout,
                                      resource_env, unconstrained_dims):
  aval, = ctx.avals_in
  out_aval, = ctx.avals_out
  axis_ctx = ctx.module_context.axis_context
  if (isinstance(axis_ctx, sharding_impls.SPMDAxisContext) and
      axis_ctx.manual_axes):
    sharding = mlir.add_manual_axes(axis_ctx, sharding, aval.ndim)
  if config.use_shardy_partitioner.value:
    sharding = sharding._to_sdy_sharding(aval.ndim)
  else:
    sharding = sharding._to_xla_hlo_sharding(aval.ndim).to_proto()
  out = mlir.wrap_with_sharding_op(
      ctx, x_node, out_aval, sharding, unspecified_dims=unconstrained_dims)
  if layout is not None:
    out = mlir.wrap_with_layout_op(ctx, out, out_aval, layout, aval)
  return [out]
mlir.register_lowering(sharding_constraint_p,
                       _sharding_constraint_hlo_lowering)


def _sharding_constraint_batcher(
    axis_data, vals_in, dims_in, sharding, layout, resource_env, unconstrained_dims):
  if axis_data.spmd_name is not None and isinstance(sharding, NamedSharding):
    used = {n for ns in sharding.spec
            for n in (ns if isinstance(ns, tuple) else (ns,))}
    if set(axis_data.spmd_name) & used:
      raise ValueError(f"vmap spmd_axis_name {axis_data.spmd_name} cannot appear in "
                       "with_sharding_constraint spec, but got spec "
                       f"{sharding.spec}")
  x, = vals_in
  d, = dims_in
  # None means unconstrained in ParsedPartitionSpec
  unconstrained_dims = {ud + (d <= ud) for ud in unconstrained_dims}
  if axis_data.spmd_name is None:
    unconstrained_dims.add(d)

  vmapped_sharding = _pjit_batcher_for_sharding(
      sharding, d, axis_data.spmd_name, resource_env.physical_mesh, x.ndim)
  if unconstrained_dims and isinstance(vmapped_sharding, NamedSharding):
    new_spec = list(vmapped_sharding.spec) + [None] * (x.ndim - len(vmapped_sharding.spec))
    for u in unconstrained_dims:
      new_spec[u] = PartitionSpec.UNCONSTRAINED
    vmapped_sharding = NamedSharding(
        vmapped_sharding.mesh, PartitionSpec(*new_spec))

  # TODO(yashkatariya): Figure out layouts should change under vmap.
  if layout is not None:
    raise NotImplementedError(
        'Concrete layout is not supported for vmap(with_sharding_constraint). '
        f'Got layout {layout}')

  y = sharding_constraint_p.bind(
      x,
      sharding=vmapped_sharding,
      layout=layout,
      resource_env=resource_env,
      unconstrained_dims=unconstrained_dims)
  return y, d
batching.fancy_primitive_batchers[sharding_constraint_p] = _sharding_constraint_batcher
batching.skippable_batchers[sharding_constraint_p] = lambda _: ()

# -------------------- sharding_cast ---------------------------

def sharding_cast(xs, shardings):
  if isinstance(shardings, NamedSharding):
    return tree_map(
        lambda x: sharding_cast_p.bind(
            x, src_sharding=x.sharding, dst_sharding=canonicalize_sharding(
                shardings, check_mesh_consistency=False)),
        xs)

  x_flat, treedef = tree_flatten(xs)
  shardings_flat = flatten_axes("sharding_cast shardings", treedef, shardings)
  out_flat = [
      sharding_cast_p.bind(
          x, src_sharding=x.sharding,
          dst_sharding=canonicalize_sharding(s, check_mesh_consistency=False))
      for x, s in safe_zip(x_flat, shardings_flat)
  ]
  return tree_unflatten(treedef, out_flat)

sharding_cast_p = core.Primitive('sharding_cast')
def _sharding_cast_abstract_eval(aval, src_sharding, dst_sharding):
  if src_sharding.mesh.shape_tuple != dst_sharding.mesh.shape_tuple:
    raise ValueError(
        f'Mesh shape of the input {src_sharding.mesh.shape_tuple} does not'
        ' match the mesh shape of the target sharding'
        f' {dst_sharding.mesh.shape_tuple} for shape {aval.str_short()}')
  return aval.update(sharding=dst_sharding)
sharding_cast_p.def_abstract_eval(_sharding_cast_abstract_eval)

def _sharding_cast_impl(x, src_sharding, dst_sharding):
  return dispatch.apply_primitive(sharding_cast_p, x, src_sharding=src_sharding,
                                  dst_sharding=dst_sharding)
sharding_cast_p.def_impl(_sharding_cast_impl)

def _sharding_cast_transpose_rule(ct, _, src_sharding, dst_sharding):
  return [sharding_cast_p.bind(ct, src_sharding=dst_sharding,
                               dst_sharding=src_sharding)]
ad.deflinear2(sharding_cast_p, _sharding_cast_transpose_rule)

def _sharding_cast_hlo_lowering(ctx, x_node, *, src_sharding, dst_sharding):
  aval, = ctx.avals_in
  aval_out, = ctx.avals_out
  proto = (dst_sharding._to_sdy_sharding(aval.ndim)
           if config.use_shardy_partitioner.value else
           dst_sharding._to_xla_hlo_sharding(aval.ndim).to_proto())
  return [mlir.lower_sharding_under_shit(ctx, x_node, aval_out, proto)]
mlir.register_lowering(sharding_cast_p, _sharding_cast_hlo_lowering)

# TODO(yashkatariya): Comment this in after vmap ShiT tests are added.
# def _sharding_cast_batcher(axis_data, vals_in, dims_in, src_sharding,
#                            dst_sharding):
#   if axis_data.spmd_name is not None:
#     used = {n for ns in dst_sharding.spec
#             for n in (ns if isinstance(ns, tuple) else (ns,))}
#     if set(axis_data.spmd_name) & used:
#       raise ValueError(
#           f'vmap spmd_axis_name {axis_data.spmd_name} cannot '
#           f'appear in sharding_cast spec, but got spec {dst_sharding.spec}')
#   x, = vals_in
#   d, = dims_in

#   val = None if axis_data.spmd_name is None else axis_data.spmd_name
#   new_spec = PartitionSpec(*util.tuple_insert(dst_sharding.spec, d, val))
#   vmapped_dst_sharding = NamedSharding(dst_sharding.mesh, new_spec)
#   y = sharding_cast_p.bind(x, src_sharding=src_sharding,
#                            dst_sharding=vmapped_dst_sharding)
#   return y, d
# batching.fancy_primitive_batchers[sharding_cast_p] = _sharding_cast_batcher
# batching.skippable_batchers[sharding_cast_p] = lambda _: ()

# -------------------- auto and user mode -------------------------

def _get_new_mesh(axes: str | tuple[str, ...], axis_type: mesh_lib.AxisTypes):
  if not isinstance(axes, tuple):
    axes = (axes,)
  cur_mesh = mesh_lib.get_abstract_mesh()
  for a in axes:
    if cur_mesh._name_to_type[a] == axis_type:  # type: ignore
      raise ValueError(f'Axes {a} cannot be casted to type {axis_type} since '
                       f'it already is of type {axis_type}.')
  new_mesh = cur_mesh.update_axis_types({axis_type: axes})  # type: ignore
  return new_mesh

def hidden_mode(fun, *, axes: str | tuple[str, ...], out_specs):
  new_mesh = _get_new_mesh(axes, mesh_lib.AxisTypes.Hidden)
  def decorator(*args, **kwargs):
    with mesh_lib.set_abstract_mesh(new_mesh):
      in_specs = tree_map(lambda a: core.modify_spec_for_hidden(
          a.sharding.spec, new_mesh), args)
      args = sharding_cast(args, in_specs)
      out = fun(*args, **kwargs)
    return sharding_cast(out, out_specs)
  return decorator


@contextlib.contextmanager
def hidden_axes(axes: str | tuple[str, ...]):
  new_mesh = _get_new_mesh(axes, mesh_lib.AxisTypes.Hidden)
  with mesh_lib.set_abstract_mesh(new_mesh):
    yield


def visible_mode(fun, *, axes: str | tuple[str, ...], in_specs):
  new_mesh = _get_new_mesh(axes, mesh_lib.AxisTypes.Visible)
  def decorator(*args, **kwargs):
    with mesh_lib.set_abstract_mesh(new_mesh):
      args = sharding_cast(args, in_specs)
      out = fun(*args, **kwargs)
    out_specs = tree_map(lambda o: core.modify_spec_for_hidden(
        o.sharding.spec, mesh_lib.get_abstract_mesh()), out)
    return sharding_cast(out, out_specs)
  return decorator

@contextlib.contextmanager
def visible_axes(axes: str | tuple[str, ...]):
  new_mesh = _get_new_mesh(axes, mesh_lib.AxisTypes.Visible)
  with mesh_lib.set_abstract_mesh(new_mesh):
    yield

# -------------------- helpers --------------------

def get_unconstrained_dims(sharding: NamedSharding):
  assert sharding._parsed_pspec is not None
  return {i for i, axes in enumerate(sharding._parsed_pspec)
          if axes is None}


def _get_partition_spec(
    ppspec: Sequence[ParsedPartitionSpec]) -> Sequence[PartitionSpec]:
  return [get_single_pspec(p) for p in ppspec]


def get_op_sharding_from_executable(
    executable) -> tuple[Sequence[xc.OpSharding], Sequence[xc.OpSharding]]:
  in_op_shardings: list[xc.OpSharding] = []
  parameter_shardings_from_xla = executable.get_parameter_shardings()
  if parameter_shardings_from_xla is not None:
    in_op_shardings = parameter_shardings_from_xla

  out_op_shardings: list[xc.OpSharding] = []
  output_shardings_from_xla = executable.get_output_shardings()
  if output_shardings_from_xla is not None:
    out_op_shardings = output_shardings_from_xla

  return in_op_shardings, out_op_shardings


def _get_ppspec_from_executable(
    executable, mesh
  ) -> tuple[Sequence[ParsedPartitionSpec], Sequence[ParsedPartitionSpec]]:
  input_op_shardings, output_op_sharding = get_op_sharding_from_executable(
      executable
  )
  in_ppspec: list[ParsedPartitionSpec] = []
  for s in input_op_shardings:
    in_ppspec.extend(parse_flatten_op_sharding(s, mesh))

  out_ppspec: list[ParsedPartitionSpec] = []
  for s in output_op_sharding:
    out_ppspec.extend(parse_flatten_op_sharding(s, mesh))
  return in_ppspec, out_ppspec


def get_pspec_from_executable(
    executable, mesh: pxla.Mesh
) -> tuple[tuple[PartitionSpec, ...], tuple[PartitionSpec, ...]]:
  in_ppspec, out_ppspec = _get_ppspec_from_executable(executable, mesh)
  out_partition_spec = _get_partition_spec(out_ppspec)
  in_partition_spec = _get_partition_spec(in_ppspec)
  return tuple(in_partition_spec), tuple(out_partition_spec)
