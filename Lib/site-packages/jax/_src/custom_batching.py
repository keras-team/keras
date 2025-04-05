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

from collections.abc import Callable
from typing import Any
import functools
import operator

from jax import lax
from jax._src import api
from jax._src import core
from jax._src import custom_api_util
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import util
from jax._src.api_util import flatten_fun_nokwargs, resolve_kwargs
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters.batching import not_mapped
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import xla
from jax._src.tree_util import (tree_flatten, tree_map, tree_structure,
                                tree_unflatten, treedef_tuple)


source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


@custom_api_util.register_custom_decorator_type
class custom_vmap:
  """Customize the vmap behavior of a JAX-transformable function.

  This decorator is used to customize the behavior of a JAX function under the
  :func:`jax.vmap` transformation. A ``custom_vmap``-decorated function will
  mostly (see below for caveats) have the same behavior as the underlying
  function, except when batched using :py:func:`jax.vmap`. When batched, the
  rule defined using :py:func:`~jax.custom_batching.custom_vmap.def_vmap` will
  be used.

  For example:

    >>> @jax.custom_batching.custom_vmap
    ... def f(x, y):
    ...   return x + y
    ...
    >>> @f.def_vmap
    ... def f_vmap_rule(axis_size, in_batched, xs, ys):
    ...   assert all(in_batched)
    ...   assert xs.shape[0] == axis_size
    ...   assert ys.shape[0] == axis_size
    ...   out_batched = True
    ...   return xs * ys, out_batched
    ...
    >>> xs = jnp.arange(3)
    >>> ys = jnp.arange(1, 4)
    >>> jax.vmap(f)(xs, ys)  # prints xs * ys instead of xs + ys
    Array([0, 2, 6], dtype=int32)

  Of note, ``custom_vmap`` functions do not support reverse-mode autodiff. To
  customize both vmap and reverse-mode autodiff, combine ``custom_vmap`` with
  :py:class:`jax.custom_vjp`. For example:

    >>> @jax.custom_vjp
    ... @jax.custom_batching.custom_vmap
    ... def f(x, y):
    ...   return jnp.sin(x) * y
    ...
    >>> @f.def_vmap
    ... def f_vmap_rule(axis_size, in_batched, xs, ys):
    ...   return jnp.cos(xs) * ys, True
    ...
    >>> def f_fwd(x, y):
    ...   return f(x, y), (jnp.cos(x), jnp.sin(x), y)
    ...
    >>> def f_bwd(res, g):
    ...   cos_x, sin_x, y = res
    ...   return (cos_x * g * y, sin_x * g)
    ...
    >>> f.defvjp(f_fwd, f_bwd)
    >>> jax.vmap(f)(jnp.zeros(3), jnp.ones(3))
    Array([1., 1., 1.], dtype=float32)
    >>> jax.grad(f)(jnp.zeros(()), jnp.ones(()))
    Array(1., dtype=float32)

  Note that the :py:class:`jax.custom_vjp` must be on the ouside, wrapping the
  ``custom_vmap``-decorated function.
  """

  fun: Callable[..., Any]
  vmap_rule: Callable[..., tuple[Any, Any]] | None

  def __init__(self, fun: Callable[..., Any]):
    functools.update_wrapper(self, fun)
    self.fun = fun
    self.vmap_rule = None

  __getattr__ = custom_api_util.forward_attr

  def def_vmap(
      self,
      vmap_rule: Callable[..., tuple[Any, Any]],
  ) -> Callable[..., tuple[Any, Any]]:
    """Define the vmap rule for this custom_vmap function.

    Args:
      vmap_rule: A function that implements the vmap rule. This function should
        accept the following arguments: (1) an integer ``axis_size`` as its
        first argument, (2) a pytree of booleans with the same structure as the
        inputs to the function, specifying whether each argument is batched,
        and (3) the batched arguments. It should return a tuple of the batched
        output and a pytree of booleans with the same structure as the output,
        specifying whether each output element is batched. See the documentation
        for :py:func:`jax.custom_batching.custom_vmap` for some examples.

    Returns:
      This method passes the rule through, returning ``vmap_rule`` unchanged.
    """
    self.vmap_rule = vmap_rule
    return vmap_rule

  @traceback_util.api_boundary
  def __call__(self, *args, **kwargs):
    args = resolve_kwargs(self.fun, args, kwargs)
    fun_name = getattr(self.fun, "__name__", str(self.fun))
    if not self.vmap_rule:
      raise AttributeError(
          f"No batching rule defined for custom_vmap function {fun_name} "
          "using def_vmap.")
    args_flat, in_tree = tree_flatten(args)
    flat_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(self.fun), in_tree)
    in_avals = [core.get_aval(x) for x in args_flat]
    debug = pe.tracing_debug_info(self.fun, in_tree, out_tree, False, "custom_vmap")
    jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals, debug)
    closed_call = core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())
    in_tree = treedef_tuple((tree_structure(consts), in_tree))
    assert self.vmap_rule is not None
    out_flat = custom_vmap_p.bind(*consts, *args_flat,
                                  call=closed_call,
                                  rule=ClosedRule(self.vmap_rule),
                                  in_tree=in_tree,
                                  out_tree=out_tree())
    return tree_unflatten(out_tree(), out_flat)


### utils

# Define a class, instead of making a function closing over `rule`, so
# that we can override __str__
class ClosedRule:
  def __init__(self, rule):
    functools.update_wrapper(self, rule)
    self.rule = rule

  def __call__(self, axis_size, all_in_batched, *all_args):
    _, args = all_args
    consts_batched, in_batched = all_in_batched
    assert not any(tree_util.tree_leaves(consts_batched)), consts_batched
    return call_rule(self.rule, axis_size, in_batched, args)

  def __str__(self):
    return str(self.rule)

def ensure_list(xs):
  return xs if type(xs) is list else list(xs)

def rule_name(rule):
  return getattr(rule, '__name__', '<unnamed rule>')

def call_rule(rule, axis_size, in_batched, args):
  return rule(axis_size, ensure_list(in_batched), *args)

def check_vmap_rule_trees(rule, original_out_tree, out_tree, out_batched_tree):
  if out_tree != out_batched_tree:
    raise ValueError(
        'structure of output value and output batching specification returned '
        f'by custom vmap rule ({rule_name(rule)}) do not match.\n'
        f'Output values: {out_tree}\n'
        f'Batching spec: {out_batched_tree}')
  if out_tree != original_out_tree:
    raise ValueError(
        f'structure of output returned by custom vmap rule ({rule_name(rule)}) '
        'does not match that of original custom-vmapped function.\n'
        f'Original output: {original_out_tree}\n'
        f'Rule output: {out_tree}')

# Like batching.bdim_at_front, but doesn't broadcast if not mapped
def maybe_bdim_at_front(x, bdim):
  if bdim is not_mapped:
    return x
  else:
    return util.moveaxis(x, bdim, 0)

# Like batching.batch except (a) not curried and (b) returns inferred output
# axes instead of accepting and matching a given spec of output axes. Assumes
# `f` is pytree-flattened
def vmap_unrestricted(f: lu.WrappedFun, *args, in_axes, axis_name, axis_size):
  axis_data = batching.AxisData(axis_name, axis_size, None)
  tag = core.TraceTag()
  f, out_axes = batching.batch_subtrace(f, tag, axis_data, in_axes)
  outs = f.call_wrapped(*args)
  return outs, out_axes()


### custom_vmap_p rules


def custom_vmap_impl(*args, call, rule, in_tree, out_tree):
  del rule, in_tree, out_tree
  return core.jaxpr_as_fun(call)(*args)


def custom_vmap_batching(args_flat, dims, *, call, rule, in_tree, out_tree):
  del call
  axis_size, = {x.shape[d] for x, d in zip(args_flat, dims) if d is not None}
  args_flat = map(maybe_bdim_at_front, args_flat, dims)
  flat_in_batched = [d is not not_mapped for d in dims]

  args = tree_unflatten(in_tree, args_flat)
  in_batched = tree_unflatten(in_tree, flat_in_batched)
  out, out_batched = call_rule(rule, axis_size, in_batched, args)
  flat_outs, tree1 = tree_flatten(out)
  flat_out_batched, tree2 = tree_flatten(out_batched)
  check_vmap_rule_trees(rule, out_tree, tree1, tree2)
  flat_out_dims = [0 if b else not_mapped for b in flat_out_batched]
  return flat_outs, flat_out_dims


def custom_vmap_abstract_eval(*in_avals, call, **_):
  return call.out_avals


def custom_vmap_jvp(primals, tangents, *, call, rule, in_tree, out_tree):
  def jvp_of_rule_rule(axis_size, in_batched, primals, tangents):
    in_batched_ps, in_batched_ts = in_batched

    mutually_batched = tree_map(operator.and_, in_batched_ps, in_batched_ts)
    extra_batched_ps = tree_map(lambda pb, tb: 0 if pb and not tb else None,
                                in_batched_ps, in_batched_ts)
    extra_batched_ts = tree_map(lambda pb, tb: 0 if tb and not pb else None,
                                in_batched_ps, in_batched_ts)

    out_mutually_batched = lu.Store()
    flat_ps_ts, tree_ps_ts = tree_flatten((primals, tangents))
    flat_extra_batched_ps_ts, tree_ps_ts2 = tree_flatten(
        (extra_batched_ps, extra_batched_ts),
        is_leaf=lambda x: x is None)

    # TODO(frostig): assert these also equal:
    #   treedef_tuple((in_tree, in_tree))
    # once https://github.com/jax-ml/jax/issues/9066 is fixed
    assert tree_ps_ts == tree_ps_ts2
    del tree_ps_ts2

    def to_jvp(*primals):
      out, out_batched = call_rule(rule, axis_size, mutually_batched, primals)
      check_vmap_rule_trees(
          rule, out_tree, tree_structure(out), tree_structure(out_batched))
      out_mutually_batched.store(out_batched)
      return out

    def to_vmap_over_extra_batched_dims(primals, tangents):
      return api.jvp(to_jvp, primals, tangents)

    to_vmap_over_extra_batched_dims_flat, out_tree2 = flatten_fun_nokwargs(
        lu.wrap_init(to_vmap_over_extra_batched_dims),
        tree_ps_ts)

    flat_out_ps_ts, flat_out_axes = vmap_unrestricted(
        to_vmap_over_extra_batched_dims_flat, *flat_ps_ts,
        in_axes=flat_extra_batched_ps_ts,
        axis_name=core.no_axis_name, axis_size=axis_size)

    n, ragged = divmod(len(flat_out_ps_ts), 2)
    assert not ragged
    flat_out_ps, flat_out_ts = flat_out_ps_ts[:n], flat_out_ps_ts[n:]
    flat_out_axes_p, flat_out_axes_t = flat_out_axes[:n], flat_out_axes[n:]
    flat_out_ps = map(maybe_bdim_at_front, flat_out_ps, flat_out_axes_p)
    flat_out_extra_batched_ps = [d is not not_mapped for d in flat_out_axes_p]
    flat_out_ts = map(maybe_bdim_at_front, flat_out_ts, flat_out_axes_t)
    flat_out_extra_batched_ts = [d is not not_mapped for d in flat_out_axes_t]

    out_ps, out_ts = tree_unflatten(
        out_tree2(), [*flat_out_ps, *flat_out_ts])
    out_extra_batched_ps, out_extra_batched_ts = tree_unflatten(
        out_tree2(), [*flat_out_extra_batched_ps, *flat_out_extra_batched_ts])

    out_batched_ps = tree_map(
        operator.or_, out_mutually_batched.val, out_extra_batched_ps)
    out_batched_ts = tree_map(
        operator.or_, out_mutually_batched.val, out_extra_batched_ts)

    return (out_ps, out_ts), (out_batched_ps, out_batched_ts)

  tangents = map(ad.instantiate_zeros, tangents)
  jvp_call, _ = ad.jvp_jaxpr(call, [True] * len(primals), True)
  jvp_in_tree = treedef_tuple((in_tree, in_tree))
  jvp_out_tree = treedef_tuple((out_tree, out_tree))
  outs = custom_vmap_p.bind(
      *primals, *tangents,
      call=jvp_call, rule=jvp_of_rule_rule,
      in_tree=jvp_in_tree, out_tree=jvp_out_tree)
  assert len(outs) % 2 == 0, len(outs)
  out_primals, out_tangents = util.split_list(outs, [len(outs) // 2])
  return out_primals, out_tangents


custom_vmap_p = core.Primitive('custom_vmap_call')
custom_vmap_p.multiple_results = True
custom_vmap_p.def_impl(custom_vmap_impl)
custom_vmap_p.def_abstract_eval(custom_vmap_abstract_eval)
batching.primitive_batchers[custom_vmap_p] = custom_vmap_batching
ad.primitive_jvps[custom_vmap_p] = custom_vmap_jvp
xla.register_initial_style_primitive(custom_vmap_p)
mlir.register_lowering(custom_vmap_p, mlir.lower_fun(
    custom_vmap_impl, multiple_results=True))


# -- custom vmap applications


def tree_split(mask, tree):
  lhs = tree_map(lambda l, x: x if l else None, mask, tree)
  rhs = tree_map(lambda l, x: None if l else x, mask, tree)
  return lhs, rhs

def tree_merge(mask, lhs_tree, rhs_tree):
  return tree_map(lambda l, x_l, x_r: x_l if l else x_r,
                  mask, lhs_tree, rhs_tree)

def sequential_vmap(f):
  """A special case of ``custom_vmap`` that uses a loop.

  A function decorated with ``sequential_vmap`` will be called sequentially
  within a loop when batched. This is useful for functions that don't natively
  support batch dimensions.

  For example:

    >>> @jax.custom_batching.sequential_vmap
    ... def f(x):
    ...   jax.debug.print("{}", x)
    ...   return x + 1
    ...
    >>> jax.vmap(f)(jnp.arange(3))
    0
    1
    2
    Array([1, 2, 3], dtype=int32)

  Where the print statements demonstrate that this :py:func:`~jax.vmap` is being
  generated using a loop.

  See the documentation for :py:class:`~jax.custom_batching.custom_vmap` for
  more details.
  """
  f = custom_vmap(f)

  @f.def_vmap
  def rule(axis_size, in_batched, *args):
    del axis_size

    def to_map(mapped_args):
      args = tree_merge(in_batched, mapped_args, bcast_args)
      return f(*args)

    mapped_args, bcast_args = tree_split(in_batched, list(args))
    out = lax.map(to_map, mapped_args)
    out_batched = tree_map(lambda _: True, out)
    return out, out_batched

  return f
