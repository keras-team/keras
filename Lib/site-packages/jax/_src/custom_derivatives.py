# Copyright 2020 The JAX Authors.
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

from collections.abc import Callable, Sequence
import dataclasses
from functools import update_wrapper, reduce, partial, wraps
from typing import Any, Generic, TypeVar

from jax._src import config
from jax._src import core
from jax._src import custom_api_util
from jax._src.custom_transpose import custom_transpose
from jax._src import dtypes
from jax._src import effects
from jax._src import linear_util as lu
from jax._src import traceback_util
from jax._src.ad_util import (
    stop_gradient_p, SymbolicZero, Zero, zeros_like_aval)
from jax._src.api_util import (
    argnums_partial, flatten_fun_nokwargs, resolve_kwargs, fun_signature,
    _arg_names)
from jax._src.errors import UnexpectedTracerError
from jax._src.state.types import AbstractRef
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import xla
from jax._src.interpreters.batching import not_mapped
from jax._src.lax import lax
from jax._src.tree_util import (
    tree_flatten, tree_unflatten, tree_map, treedef_is_leaf, treedef_tuple,
    register_pytree_node_class, tree_leaves, tree_flatten_with_path,
    tree_leaves_with_path, keystr, treedef_children)
from jax._src.util import (cache, safe_zip, safe_map, split_list, Unhashable,
                           unzip2)


traceback_util.register_exclusion(__file__)

map = safe_map
zip = safe_zip


### util

def _initial_style_jaxpr(fun, in_avals):
  jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(fun, in_avals)
  return jaxpr, consts

def _close_jaxpr(jaxpr):
  return pe.close_jaxpr(pe.convert_constvars_jaxpr(jaxpr))

def _sum_tangents(_, x, *xs):
  return reduce(ad.add_tangents, xs, x)

def _zeros_like_pytree(x):
  return tree_map(Zero.from_primal_value, x)

_stop_gradient = partial(
    tree_map,
    lambda x: stop_gradient_p.bind(x) if isinstance(x, core.Tracer) else x,
)


# like the api_util.py function, but also grabs output avals for error checking
@lu.transformation_with_aux2
def _flatten_fun_nokwargs(f, store, in_tree, *args_flat):
  py_args = tree_unflatten(in_tree, args_flat)
  ans = f(*py_args)
  ans_flat, ans_tree = tree_flatten(ans)
  ans_avals = [core.get_aval(x) for x in ans_flat]
  store.store((ans_tree, ans_avals))
  return ans_flat


### JVPs
ReturnValue = TypeVar('ReturnValue')

@custom_api_util.register_custom_decorator_type
class custom_jvp(Generic[ReturnValue]):
  """Set up a JAX-transformable function for a custom JVP rule definition.

  This class is meant to be used as a function decorator. Instances are
  callables that behave similarly to the underlying function to which the
  decorator was applied, except when a differentiation transformation (like
  :py:func:`jax.jvp` or :py:func:`jax.grad`) is applied, in which case a custom
  user-supplied JVP rule function is used instead of tracing into and
  performing automatic differentiation of the underlying function's
  implementation.

  There are two instance methods available for defining the custom JVP rule:
  :py:func:`~jax.custom_jvp.defjvp` for defining a *single* custom JVP rule for
  all the function's inputs, and for convenience
  :py:func:`~jax.custom_jvp.defjvps`, which wraps
  :py:func:`~jax.custom_jvp.defjvp`, and allows you to provide separate
  definitions for the partial derivatives of the function w.r.t. each of its
  arguments.

  For example::

    @jax.custom_jvp
    def f(x, y):
      return jnp.sin(x) * y

    @f.defjvp
    def f_jvp(primals, tangents):
      x, y = primals
      x_dot, y_dot = tangents
      primal_out = f(x, y)
      tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
      return primal_out, tangent_out

  For a more detailed introduction, see the tutorial_.

  .. _tutorial: https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
  """
  fun: Callable[..., ReturnValue]
  nondiff_argnums: Sequence[int]
  jvp: Callable[..., tuple[ReturnValue, ReturnValue]] | None = None
  symbolic_zeros: bool = False

  def __init__(self,
               fun: Callable[..., ReturnValue],
               nondiff_argnums: Sequence[int] = (),
               ):
    update_wrapper(self, fun)
    self.fun = fun
    self.nondiff_argnums = nondiff_argnums

  __getattr__ = custom_api_util.forward_attr

  def defjvp(self,
             jvp: Callable[..., tuple[ReturnValue, ReturnValue]],
             symbolic_zeros: bool = False,
             ) -> Callable[..., tuple[ReturnValue, ReturnValue]]:
    """Define a custom JVP rule for the function represented by this instance.

    Args:
      jvp: a Python callable representing the custom JVP rule. When there are no
        ``nondiff_argnums``, the ``jvp`` function should accept two arguments,
        where the first is a tuple of primal inputs and the second is a tuple of
        tangent inputs. The lengths of both tuples are equal to the number of
        parameters of the :class:`~jax.custom_jvp` function. The ``jvp`` function
        should produce as output a pair where the first element is the primal
        output and the second element is the tangent output. Elements of the
        input and output tuples may be arrays or any nested tuples/lists/dicts
        thereof.
      symbolic_zeros: boolean, indicating whether the rule should be passed
        objects representing static symbolic zeros in its tangent argument in
        correspondence with unperturbed values; otherwise, only standard JAX
        types (e.g. array-likes) are passed. Setting this option to ``True``
        allows a JVP rule to detect whether certain inputs are not involved in
        differentiation, but at the cost of needing special handling for these
        objects (which e.g. can't be passed into jax.numpy functions). Default
        ``False``.

    Returns:
      Returns ``jvp`` so that ``defjvp`` can be used as a decorator.

    Examples:

      >>> @jax.custom_jvp
      ... def f(x, y):
      ...   return jnp.sin(x) * y
      ...
      >>> @f.defjvp
      ... def f_jvp(primals, tangents):
      ...   x, y = primals
      ...   x_dot, y_dot = tangents
      ...   primal_out = f(x, y)
      ...   tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
      ...   return primal_out, tangent_out

      >>> x = jnp.float32(1.0)
      >>> y = jnp.float32(2.0)
      >>> with jnp.printoptions(precision=2):
      ...   print(jax.value_and_grad(f)(x, y))
      (Array(1.68, dtype=float32), Array(1.08, dtype=float32))
    """
    self.jvp = jvp
    self.symbolic_zeros = symbolic_zeros
    return jvp

  def defjvps(self, *jvps: Callable[..., ReturnValue] | None) -> None:
    """Convenience wrapper for defining JVPs for each argument separately.

    This convenience wrapper cannot be used together with ``nondiff_argnums``.

    Args:
      *jvps: a sequence of functions, one for each positional argument of the
        :class:`~jax.custom_jvp` function. Each function takes as arguments
        the tangent value for the corresponding primal input, the primal
        output, and the ßprimal inputs. See the example below.

    Returns:
      None.

    Examples:

      >>> @jax.custom_jvp
      ... def f(x, y):
      ...   return jnp.sin(x) * y
      ...
      >>> f.defjvps(lambda x_dot, primal_out, x, y: jnp.cos(x) * x_dot * y,
      ...           lambda y_dot, primal_out, x, y: jnp.sin(x) * y_dot)

      >>> x = jnp.float32(1.0)
      >>> y = jnp.float32(2.0)
      >>> with jnp.printoptions(precision=2):
      ...   print(jax.value_and_grad(f)(x, y))
      (Array(1.68, dtype=float32), Array(1.08, dtype=float32))
    """
    if self.nondiff_argnums:
      raise TypeError("Can't use ``defjvps`` with ``nondiff_argnums``.")

    def jvp(primals, tangents):
      primal_out = self(*primals)
      zeros = _zeros_like_pytree(primal_out)
      all_tangents_out = [jvp(t, primal_out, *primals) if jvp else zeros
                          for t, jvp in zip(tangents, jvps)]
      tangent_out = tree_map(_sum_tangents, primal_out, *all_tangents_out)
      return primal_out, tangent_out

    self.defjvp(jvp)

  @traceback_util.api_boundary
  def __call__(self, *args: Any, **kwargs: Any) -> ReturnValue:  # pytype: disable=invalid-annotation
    primal_name = getattr(self.fun, '__name__', str(self.fun))
    if not self.jvp:
      msg = f"No JVP defined for custom_jvp function {primal_name} using defjvp."
      raise AttributeError(msg)
    jvp_name    = getattr(self.jvp, '__name__', str(self.jvp))
    args = resolve_kwargs(self.fun, args, kwargs)
    if self.nondiff_argnums:
      nondiff_argnums = set(self.nondiff_argnums)
      args = tuple(_stop_gradient(x) if i in nondiff_argnums else x
                   for i, x in enumerate(args))
      diff_argnums = [i for i in range(len(args)) if i not in nondiff_argnums]
      f_, dyn_args = argnums_partial(lu.wrap_init(self.fun), diff_argnums, args,
                                     require_static_args_hashable=False)
      static_args = [args[i] for i in self.nondiff_argnums]
      jvp = _add_args(lu.wrap_init(self.jvp), static_args)
    else:
      f_, dyn_args = lu.wrap_init(self.fun), args
      jvp = lu.wrap_init(self.jvp)
    args_flat, in_tree = tree_flatten(dyn_args)
    flat_fun, out_type1 = _flatten_fun_nokwargs(f_, in_tree)
    flat_jvp, out_type2 = _flatten_jvp(jvp, primal_name, jvp_name, in_tree,
                                       out_type1)
    out_flat = custom_jvp_call_p.bind(flat_fun, flat_jvp, *args_flat,
                                      symbolic_zeros=self.symbolic_zeros)
    _, (out_tree, _) = lu.merge_linear_aux(out_type1, out_type2)
    return tree_unflatten(out_tree, out_flat)

def _add_args(f, extra_args):
  return _add_args_(f, tuple(Unhashable(arg) for arg in extra_args))

@lu.transformation2
def _add_args_(f, extra_args, *args, **kwargs):
  extra_args = tuple(arg.val for arg in extra_args)
  all_args = (extra_args + args)
  return f(*all_args, **kwargs)

@partial(lu.transformation_with_aux2, use_eq_store=True)
def _flatten_jvp(f, store, primal_name, jvp_name, in_tree, maybe_out_type, *args):
  primals_in, tangents_in = split_list(args, [len(args) // 2])
  py_primals = tree_unflatten(in_tree, primals_in)
  py_tangents = tree_unflatten(in_tree, tangents_in)
  pair_out = f(py_primals, py_tangents)
  if not isinstance(pair_out, (list, tuple)) or len(pair_out) != 2:
    msg = (f"Custom JVP rule {jvp_name} for function {primal_name} "
           "must produce a pair (list or tuple of length two) representing "
           f"primal and tangent outputs, but got {pair_out}.")
    raise TypeError(msg)
  py_primals_out, py_tangents_out = pair_out
  primals_out, out_tree = tree_flatten(py_primals_out)
  tangents_out, out_tree2 = tree_flatten(py_tangents_out)
  primal_avals = [core.get_aval(x) for x in primals_out]
  if out_tree != out_tree2:
    msg = (f"Custom JVP rule {jvp_name} for function {primal_name} must "
           "produce primal and tangent outputs with equal container (pytree) "
           f"structures, but got {out_tree} and {out_tree2} respectively.")
    raise TypeError(msg)
  # If the primal function already ran, check out_tree agreement.
  try: out_type_ = maybe_out_type()
  except lu.StoreException: out_type_ = None
  if out_type_ is not None:
    out_tree_, primal_avals_ = out_type_
    ty_tree  = tree_unflatten(out_tree , [a.str_short() for a in primal_avals])
    ty_tree_ = tree_unflatten(out_tree_, [a.str_short() for a in primal_avals_])
    if out_tree_ != out_tree:
      m = (f"Custom JVP rule {jvp_name} for function {primal_name} must "
           "produce a pair (list or tuple of length two) "
           "where the first element represents the primal output "
           "(equal in value to the output of the custom_jvp-decorated function "
           f"{primal_name}, "
           "and in particular of the same container/pytree structure), but "
           "instead the JVP rule output's first element had container/pytree "
           "structure:\n"
           f"""    {str(ty_tree ).replace("'", "")}\n"""
           f"while the custom_jvp-decorated function {primal_name} had output "
           "container/pytree structure:\n"
           f"""    {str(ty_tree_).replace("'", "")}.""")
      raise TypeError(m)
    if not all(map(core.typematch, primal_avals, primal_avals_)):
      m = (f"Custom JVP rule {jvp_name} for function {primal_name} must "
           "produce a pair (list or tuple of length two) "
           "where the first element represents the primal output "
           "(equal in value to the output of the custom_jvp-decorated function "
           f"{primal_name}, "
           "and in particular with leaves of the same shape/dtype), but "
           "instead the JVP rule output's first element had shapes/dtypes of:\n"
           f"""    {str(ty_tree ).replace("'", "")}\n"""
           f"while the custom_jvp-decorated function {primal_name} had output "
           "shapes/dtypes of:\n"
           f"""    {str(ty_tree_).replace("'", "")}""")
      raise TypeError(m)
  primal_avals_out = [core.get_aval(x).strip_weak_type() for x in primals_out]
  expected_tangent_avals_out = [
    core.get_aval(x).strip_weak_type().to_tangent_aval()
    for x in primals_out]
  tangent_avals_out = [core.get_aval(t).strip_weak_type()
                       if type(t) is not SymbolicZero else t.aval.strip_weak_type()
                       for t in tangents_out]
  if expected_tangent_avals_out != tangent_avals_out:
    if len(expected_tangent_avals_out) == 1:
      (av_p,), (av_et,), (av_t,) = primal_avals_out, expected_tangent_avals_out, tangent_avals_out
      msg = ("Custom JVP rule must produce primal and tangent outputs with "
             "corresponding shapes and dtypes. Expected {} (tangent type of {}) but got {}.")
      raise TypeError(msg.format(av_et.str_short(), av_p.str_short(), av_t.str_short()))
    else:
      msg = ("Custom JVP rule must produce primal and tangent outputs with "
             "corresponding shapes and dtypes, but got:\n{}")
      disagreements = (
          f"  primal {av_p.str_short()} with tangent {av_t.str_short()}, expecting tangent {av_et}"
          for av_p, av_et, av_t in zip(primal_avals_out, expected_tangent_avals_out, tangent_avals_out)
          if av_et != av_t)

      raise TypeError(msg.format('\n'.join(disagreements)))
  store.store((out_tree, primal_avals))
  return primals_out + tangents_out

class CustomJVPCallPrimitive(core.Primitive):
  multiple_results = True

  def bind_with_trace(self, trace, args, params):
    fun, jvp, tracers = args[0], args[1], args[2:]
    return trace.process_custom_jvp_call(self, fun, jvp, tracers, **params)

  def impl(self, fun, _, *args):
    raise NotImplementedError

  def get_bind_params(self, params):
    new_params = dict(params)
    call_jaxpr = new_params.pop('call_jaxpr')
    num_consts = new_params.pop('num_consts')
    jvp_jaxpr_thunk = new_params.pop('jvp_jaxpr_thunk')
    fun = lu.wrap_init(core.jaxpr_as_fun(call_jaxpr))
    jvp = lift_jvp(num_consts, jvp_jaxpr_thunk)
    return [fun, jvp], new_params

def lift_jvp(num_consts: int, jvp_jaxpr_thunk: Callable) -> lu.WrappedFun:
  @lu.wrap_init
  def jvp(*xs):
    n, ragged = divmod(len(xs), 2)
    assert not ragged
    primals, tangents = xs[num_consts:n], xs[n+num_consts:]
    zeros = [type(t) is SymbolicZero for t in tangents]
    jvp_jaxpr, jvp_consts, out_zeros = jvp_jaxpr_thunk(*zeros)
    nonzero_tangents = [t for t in tangents if type(t) is not SymbolicZero]
    out = core.eval_jaxpr(jvp_jaxpr, jvp_consts, *primals, *nonzero_tangents)
    out_primals, nz_out_tangents = split_list(out, [len(out_zeros)])
    nz_out_tangents_ = iter(nz_out_tangents)
    out_tangents = [SymbolicZero(core.get_aval(p).to_tangent_aval())
                    if z else next(nz_out_tangents_)
                    for p, z in zip(out_primals, out_zeros)]
    assert next(nz_out_tangents_, None) is None
    return [*out_primals, *out_tangents]
  return jvp

effects.custom_derivatives_allowed_effects.add_type(lax.InOutFeedEffect)

custom_jvp_call_p = CustomJVPCallPrimitive('custom_jvp_call')

def _custom_jvp_call_typecheck(_, *in_avals, call_jaxpr, jvp_jaxpr_thunk,
                               num_consts, symbolic_zeros):
  # TODO(mattjj): could do more checking here...
  del in_avals, jvp_jaxpr_thunk, num_consts
  disallowed_effects = effects.custom_derivatives_allowed_effects.filter_not_in(call_jaxpr.effects)
  if disallowed_effects:
    raise NotImplementedError(
        f'Effects not supported in `custom_jvp`: {disallowed_effects}')
  return call_jaxpr.out_avals, call_jaxpr.effects
core.custom_typechecks[custom_jvp_call_p] = _custom_jvp_call_typecheck

def _custom_jvp_call_mlir_translation(ctx, *args, call_jaxpr, jvp_jaxpr_thunk,
                                      num_consts, symbolic_zeros):
  del jvp_jaxpr_thunk, num_consts, symbolic_zeros
  consts = mlir._ir_consts(call_jaxpr.consts)
  out, tokens = mlir.jaxpr_subcomp(ctx.module_context, call_jaxpr.jaxpr,
                                   ctx.name_stack, ctx.tokens_in, consts,
                                   *args, dim_var_values=ctx.dim_var_values)
  ctx.set_tokens_out(tokens)
  return out
mlir.register_lowering(custom_jvp_call_p, _custom_jvp_call_mlir_translation)

# If a (multi)linear function is defined with a custom jvp, then
# custom_jvp_call_ can appear in jaxprs to be transposed. Since it's already
# been linearized, we can drop the jvp rule.
def _custom_jvp_call_transpose(params, jaxpr, args, ct, _):
  del params
  return ad.backward_pass(jaxpr.jaxpr, None, jaxpr.consts, args, ct)
ad.primitive_transposes[custom_jvp_call_p] = _custom_jvp_call_transpose


### VJPs

@custom_api_util.register_custom_decorator_type
class custom_vjp(Generic[ReturnValue]):
  """Set up a JAX-transformable function for a custom VJP rule definition.

  This class is meant to be used as a function decorator. Instances are
  callables that behave similarly to the underlying function to which the
  decorator was applied, except when a reverse-mode differentiation
  transformation (like :py:func:`jax.grad`) is applied, in which case a custom
  user-supplied VJP rule function is used instead of tracing into and performing
  automatic differentiation of the underlying function's implementation. There
  is a single instance method, :py:func:`~jax.custom_vjp.defvjp`, which may be
  used to define the custom VJP rule.

  This decorator precludes the use of forward-mode automatic differentiation.

  For example::

    @jax.custom_vjp
    def f(x, y):
      return jnp.sin(x) * y

    def f_fwd(x, y):
      return f(x, y), (jnp.cos(x), jnp.sin(x), y)

    def f_bwd(res, g):
      cos_x, sin_x, y = res
      return (cos_x * g * y, sin_x * g)

    f.defvjp(f_fwd, f_bwd)

  For a more detailed introduction, see the tutorial_.

  .. _tutorial: https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
  """

  def __init__(self,
               fun: Callable[..., ReturnValue],
               nondiff_argnums: Sequence[int] = ()):
    update_wrapper(self, fun)
    self.fun = fun
    self.nondiff_argnums = nondiff_argnums
    self.fwd: Callable[..., tuple[ReturnValue, Any]] | None = None
    self.bwd: Callable[..., tuple[Any, ...]] | None = None
    self.symbolic_zeros = False
    self.optimize_remat = False

  __getattr__ = custom_api_util.forward_attr

  def defvjp(self,
             fwd: Callable[..., tuple[ReturnValue, Any]],
             bwd: Callable[..., tuple[Any, ...]],
             symbolic_zeros: bool = False,
             optimize_remat: bool = False,
             ) -> None:
    """Define a custom VJP rule for the function represented by this instance.

    Args:
      fwd: a Python callable representing the forward pass of the custom VJP
        rule. When there are no ``nondiff_argnums``, the ``fwd`` function has
        the same input signature as the underlying primal function. It should
        return as output a pair, where the first element represents the primal
        output and the second element represents any "residual" values to store
        from the forward pass for use on the backward pass by the function
        ``bwd``. Input arguments and elements of the output pair may be arrays
        or nested tuples/lists/dicts thereof.
      bwd: a Python callable representing the backward pass of the custom VJP
        rule. When there are no ``nondiff_argnums``, the ``bwd`` function takes
        two arguments, where the first is the "residual" values produced on the
        forward pass by ``fwd``, and the second is the output cotangent with the
        same structure as the primal function output. The output of ``bwd`` must
        be a tuple of length equal to the number of arguments of the primal
        function, and the tuple elements may be arrays or nested
        tuples/lists/dicts thereof so as to match the structure of the primal
        input arguments.
      symbolic_zeros: boolean, determining whether to indicate symbolic zeros
        to the ``fwd`` and ``bwd`` rules. Enabling this option allows custom
        derivative rules to detect when certain inputs, and when certain
        output cotangents, are not involved in differentiation. If ``True``:

        * ``fwd`` must accept, in place of each leaf value ``x`` in
          the pytree comprising an argument to the original function,
          an object (of type
          ``jax.custom_derivatives.CustomVJPPrimal``) with two
          attributes instead: ``value`` and ``perturbed``. The
          ``value`` field is the original primal argument, and
          ``perturbed`` is a boolean.  The ``perturbed`` bit indicates
          whether the argument is involved in differentiation (i.e.,
          if it is ``False``, then the corresponding Jacobian "column"
          is zero).

        * ``bwd`` will be passed objects representing static symbolic zeros in
          its cotangent argument in correspondence with unperturbed values;
          otherwise, only standard JAX types (e.g. array-likes) are passed.

        Setting this option to ``True`` allows these rules to detect whether
        certain inputs and outputs are not involved in differentiation, but at
        the cost of special handling. For instance:

        * The signature of ``fwd`` changes, and the objects it is passed cannot
          be output from the rule directly.

        * The ``bwd`` rule is passed objects that are not entirely array-like,
          and that cannot be passed to most ``jax.numpy`` functions.

        * Any custom pytree nodes involved in the primal function's arguments
          must accept, in their unflattening functions, the two-field record
          objects that are given as input leaves to the ``fwd`` rule.

        Default ``False``.
      optimize_remat: boolean, an experimental flag to enable an automatic
        optimization when this function is used under :func:`jax.remat`. This
        will be most useful when the ``fwd`` rule is an opaque call such as a
        Pallas kernel or a custom call. Default ``False``.

    Returns:
      None.

    Examples:

      >>> @jax.custom_vjp
      ... def f(x, y):
      ...   return jnp.sin(x) * y
      ...
      >>> def f_fwd(x, y):
      ...   return f(x, y), (jnp.cos(x), jnp.sin(x), y)
      ...
      >>> def f_bwd(res, g):
      ...   cos_x, sin_x, y = res
      ...   return (cos_x * g * y, sin_x * g)
      ...
      >>> f.defvjp(f_fwd, f_bwd)

      >>> x = jnp.float32(1.0)
      >>> y = jnp.float32(2.0)
      >>> with jnp.printoptions(precision=2):
      ...   print(jax.value_and_grad(f)(x, y))
      (Array(1.68, dtype=float32), Array(1.08, dtype=float32))
    """
    self.fwd = fwd
    self.bwd = bwd
    self.symbolic_zeros = symbolic_zeros
    self.optimize_remat = optimize_remat
    if self.symbolic_zeros and self.optimize_remat:
      raise NotImplementedError(
          "remat optimization for custom_vjp does not support symbolic zeros")

  @traceback_util.api_boundary
  def __call__(self, *args: Any, **kwargs: Any) -> ReturnValue:  # pytype: disable=invalid-annotation
    primal_name = getattr(self.fun, '__name__', str(self.fun))
    if not self.fwd or not self.bwd:
      msg = f"No VJP defined for custom_vjp function {primal_name} using defvjp."
      raise AttributeError(msg)
    fwd_name = getattr(self.fwd, '__name__', str(self.fwd))
    args = resolve_kwargs(self.fun, args, kwargs)
    if self.optimize_remat:
      fwd = optimize_remat_of_custom_vjp_fwd(
          self.fun, self.fwd, nondiff_argnums=self.nondiff_argnums,
          symbolic_zeros=self.symbolic_zeros)
    else:
      fwd = self.fwd
    if config.enable_custom_vjp_by_custom_transpose.value:
      if self.nondiff_argnums:
        raise NotImplementedError(
            'nondiff_argnums not implemented for new custom_vjp')
      return custom_vjp_by_custom_transpose(self.fun, self.fwd, self.bwd)(*args)
    else:
      if self.nondiff_argnums:
        for i in self.nondiff_argnums: _check_for_tracers(args[i])
        nondiff_argnums = set(self.nondiff_argnums)
        dyn_argnums = [i for i in range(len(args)) if i not in nondiff_argnums]
        f_, dyn_args = argnums_partial(lu.wrap_init(self.fun), dyn_argnums,
                                       args, require_static_args_hashable=False)
        static_args = [args[i] for i in self.nondiff_argnums]
        fwd_, _ = argnums_partial(lu.wrap_init(fwd), dyn_argnums, args,
                                 require_static_args_hashable=False)
        bwd = _add_args(lu.wrap_init(self.bwd), static_args)
      else:
        f_, dyn_args = lu.wrap_init(self.fun), args
        fwd_, bwd = lu.wrap_init(fwd), lu.wrap_init(self.bwd)
      args_flat, in_tree = tree_flatten(dyn_args)
      in_avals = [core.get_aval(x) for x in args_flat]
      if config.mutable_array_checks.value:
        f_ = _check_primal_refs(f_, self.nondiff_argnums)
      flat_fun, out_type = _flatten_fun_nokwargs(f_, in_tree)
      flat_fwd, out_trees = _flatten_fwd(
          fwd_, self.nondiff_argnums, self.symbolic_zeros, primal_name,
          fwd_name, in_tree, out_type)
      flat_bwd = _flatten_bwd(bwd, in_tree, in_avals, out_trees).call_wrapped
      out_flat = custom_vjp_call_p.bind(flat_fun, flat_fwd, flat_bwd,
                                        *args_flat, out_trees=out_trees,
                                        symbolic_zeros=self.symbolic_zeros)
      _, (out_tree, _) = lu.merge_linear_aux(out_type, out_trees)
      return tree_unflatten(out_tree, out_flat)

@lu.transformation2
def _check_primal_refs(f, nondiff_argnums, *args):
  _check_for_aliased_refs(f, nondiff_argnums, args)
  out = f(*args)
  _check_for_returned_refs(f, out, 'primal')
  return out

def _check_for_aliased_refs(f, nondiff_argnums, args):
  leaves = tree_leaves(args)
  refs: dict[int, int] = {}
  for i, x in enumerate(leaves):
    if (isinstance((a := core.get_aval(x)), AbstractRef) and
        (dup_idx := refs.setdefault(id(core.get_referent(x)), i)) != i):
      arg_names = _arg_names(fun_signature(f), args, {}, nondiff_argnums, ())
      if arg_names is None:
        arg_names = [f'flat index {j}' for j in range(len(leaves))]
      raise ValueError(
          "only one reference to a mutable array may be passed as an argument "
          f"to a function, but custom_vjp function {f} got the same mutable "
          f"array reference of type {a.str_short()} at {arg_names[dup_idx]} and"
          f" {arg_names[i]}.")

def _check_for_returned_refs(f, out, kind):
  leaves = tree_leaves_with_path(out)
  for path, leaf in leaves:
    if isinstance((a := core.get_aval(leaf)), AbstractRef):
      loc = f' at output tree path {keystr(path)}' if path else ''
      raise ValueError(f"custom_vjp {kind} function {f} returned a mutable "
                       f"a array reference of type {a.str_short()}{loc}, "
                       "but mutable array references cannot be returned.")

@dataclasses.dataclass
class CustomVJPPrimal:
  """Primal to a ``custom_vjp``'s forward rule when ``symbolic_zeros`` is set"""
  value: Any
  perturbed: bool

def custom_vjp_primal_tree_values(tree):
  """Strips away perturbation information from forward rule arguments.

  This is a helper function for user with the ``symbolic_zeros`` option to
  the ``defvjp`` method of a ``custom_vjp``-decorated function.

  In ``symbolic_zeros`` mode, the custom forward rule receives arguments
  whose pytree leaves are records with a ``value`` attribute that carries
  the primal argument. This is a way to convert such argument trees back to
  their original form, replacing each such record with its carried value at
  each leaf.
  """
  def value(leaf):
    if type(leaf) is not CustomVJPPrimal:
      raise TypeError(f"unexpected leaf type {type(leaf)}")
    return leaf.value
  return tree_map(value, tree)

def _check_for_tracers(x):
  for leaf in tree_leaves(x):
    if isinstance(leaf, core.Tracer):
      msg = ("Found a JAX Tracer object passed as an argument to a custom_vjp "
            "function in a position indicated by nondiff_argnums as "
            "non-differentiable. Tracers cannot be passed as non-differentiable "
            "arguments to custom_vjp functions; instead, nondiff_argnums should "
            "only be used for arguments that can't be or contain JAX tracers, "
            "e.g. function-valued arguments. In particular, array-valued "
            "arguments should typically not be indicated as nondiff_argnums.")
      raise UnexpectedTracerError(msg)

@partial(lu.transformation_with_aux2, use_eq_store=True)
def _flatten_fwd(f, store, nondiff_argnums, symbolic_zeros, primal_name,
                 fwd_name, in_tree, maybe_out_type, *args):
  if symbolic_zeros:
    args = [CustomVJPPrimal(x, z) for x, z in zip(args[::2], args[1::2])]
  else:
    args = args[::2]
  py_args = tree_unflatten(in_tree, args)
  if config.mutable_array_checks.value:
    _check_for_aliased_refs(f, nondiff_argnums, py_args)
  pair_out = f(*py_args)
  if config.mutable_array_checks.value:
    _check_for_returned_refs(f, pair_out, 'fwd')
  if not isinstance(pair_out, (list, tuple)) or len(pair_out) != 2:
    msg = (f"Custom VJP fwd rule {fwd_name} for function {primal_name} "
           "must produce a pair (list or tuple of length two) where the first "
           "element represents the primal output (equal to those of the "
           f"custom_vjp-decorated function {primal_name}) and the "
           "second element represents residuals (i.e. values stored from the "
           "forward pass for use on the backward pass), but "
           f"instead of a pair the fwd rule {fwd_name} produced {pair_out}.")
    raise TypeError(msg)
  py_primals_out, res = pair_out
  primals_out, out_tree = tree_flatten(py_primals_out)
  res, res_tree = tree_flatten(res)
  primal_avals = [core.get_aval(x) for x in primals_out]
  # If the primal function already ran, check out_tree agreement.
  try: out_type_ = maybe_out_type()
  except lu.StoreException: out_type_ = None
  if out_type_ is not None:
    out_tree_, primal_avals_ = out_type_
    ty_tree  = tree_unflatten(out_tree , [a.str_short() for a in primal_avals])
    ty_tree_ = tree_unflatten(out_tree_, [a.str_short() for a in primal_avals_])
    if out_tree_ != out_tree:
      m = (f"Custom VJP fwd rule {fwd_name} for function {primal_name} "
           "must produce a pair (list or tuple of length two) where the first "
           "element represents the primal output "
           "(equal to the output of the custom_vjp-decorated function "
           f"{primal_name}) and the "
           "second element represents residuals (i.e. values stored from the "
           "forward pass for use on the backward pass), but "
           "instead the fwd rule output's first element had container/pytree "
           "structure:\n"
           f"""    {str(ty_tree ).replace("'", "")}\n"""
           f"while the custom_vjp-decorated function {primal_name} had output "
           "container/pytree structure:\n"
           f"""    {str(ty_tree_).replace("'", "")}.""")
      raise TypeError(m)
    if not all(map(core.typematch, primal_avals, primal_avals_)):
      m = (f"Custom VJP fwd rule {fwd_name} for function {primal_name} must "
           "produce a pair (list or tuple of length two) "
           "where the first element represents the primal output "
           "(equal to the output of the custom_vjp-decorated function "
           f"{primal_name}) and the second element represents residuals "
           "(i.e. values stored from the forward pass for use on the "
           "backward pass), but "
           "instead the fwd rule output's first element had shapes/dtypes of:\n"
           f"""    {str(ty_tree ).replace("'", "")}\n"""
           f"while the custom_vjp-decorated function {primal_name} had output "
           "shapes/dtypes of:\n"
           f"""    {str(ty_tree_).replace("'", "")}""")
      raise TypeError(m)
  store.store((out_tree, res_tree))
  return (*res, *primals_out)

@lu.transformation2
def _flatten_bwd(f, in_tree, in_avals, out_trees, *args):
  out_tree, res_tree = out_trees()
  assert len(args) == res_tree.num_leaves + out_tree.num_leaves
  res, cts_out = split_list(args, [res_tree.num_leaves])
  py_res = tree_unflatten(res_tree, res)
  py_cts_out = tree_unflatten(out_tree, cts_out)
  py_cts_in = f(py_res, py_cts_out)
  if isinstance(py_cts_in, list) and len(py_cts_in) == len(treedef_children(in_tree)):
    py_cts_in = tuple(py_cts_in)
  # For each None in py_cts_in, indicating an argument for which the rule
  # produces no cotangent, we replace it with a pytree with the structure of the
  # corresponding subtree of in_tree and with leaves of a non-pytree sentinel
  # object, to be replaced with Nones in the final returned result.
  zero = object()  # non-pytree sentinel to replace Nones in py_cts_in
  dummy = tree_unflatten(in_tree, [object()] * in_tree.num_leaves)
  keypaths, _ = unzip2(tree_flatten_with_path(dummy)[0])
  cts_in_flat = []
  def append(x, d):
    num_leaves = len(tree_flatten(d)[0])
    if x is None and d is not None:
      cts_in_flat.extend([zero] * num_leaves)
    elif x is not None:
      cts_in_flat.extend([x] * num_leaves)
    return x
  try:
    if not isinstance(py_cts_in, tuple):
      raise ValueError
    tree_map(append, py_cts_in, dummy, is_leaf=lambda x: x is None)
  except ValueError:
    _, in_tree2 = tree_flatten(py_cts_in)
    msg = ("Custom VJP bwd rule must produce an output with the same container "
           "(pytree) structure as the args tuple of the primal function, "
           "and in particular must produce a tuple of length equal to the "
           "number of arguments to the primal function, but got bwd output "
           "structure {} for primal input structure {}.")
    raise TypeError(msg.format(in_tree2, in_tree)) from None
  results = []
  for kp, a, ct in zip(keypaths, in_avals, cts_in_flat):
    if ct is zero or getattr(a.to_tangent_aval(), 'dtype') == dtypes.float0:
      results.append(Zero(a.to_tangent_aval()))
    elif type(ct) is SymbolicZero:
      if not core.typecompat(a.to_tangent_aval(), a_ := ct.aval):
        msg = ("Custom VJP bwd rule produced a SymbolicZero with a shape/dtype "
               "that does not match the corresponding input tangent shape/dtype: "
               f"at output{keystr(kp)} the SymbolicZero had shape/dtype "
               f"{a_.str_short()} while the "
               f"corresponding input had shape/dtype {a.str_short()}. "
               "Consider just returning a None here instead of a SymbolicZero "
               "object.")
        raise ValueError(msg)
      results.append(Zero(ct.aval))
    else:
      if (not core.typecompat(a.to_tangent_aval(), a_ := core.get_aval(ct))
          and not (_temporary_dtype_exception(a, a_) or
                   _temporary_shape_exception(a, a_))):
        msg = ("Custom VJP bwd rule must produce an output with the same "
               "shape/dtypes as the args tuple of the primal function, but at "
               f"output{keystr(kp)} the bwd rule produced an output of "
               f"shape/dtype {a_.str_short()} corresponding "
               f"to an input of shape/dtype {a.str_short()}.")
        raise ValueError(msg)
      results.append(ct)
  return results

# TODO(mattjj): remove both these exceptions to cotangent compatibility check
def _temporary_dtype_exception(a, a_) -> bool:
  if isinstance(a, core.ShapedArray) and isinstance(a_, core.ShapedArray):
    return (a.shape == a_.shape and
            (dtypes.issubdtype(a_.dtype, dtypes.extended) or
             dtypes.issubdtype(a.dtype, dtypes.np.inexact)))
  return False

# TODO(mattjj): remove both these exceptions to cotangent compatibility check
def _temporary_shape_exception(a, a_) -> bool:
  return config.custom_vjp_disable_shape_check.value

class CustomVJPCallPrimitive(core.CallPrimitive):
  initial_style: core.Primitive

  def bind_with_trace(self, trace, args, params):
    fun, fwd, bwd, tracers = args[0], args[1], args[2], args[3:]
    return trace.process_custom_vjp_call(self, fun, fwd, bwd, tracers, **params)

custom_vjp_call_p = CustomVJPCallPrimitive('custom_vjp_call')

def _custom_vjp_call_jaxpr_impl(*args, fun_jaxpr, **_):
  return core.jaxpr_as_fun(fun_jaxpr)(*args)

def _custom_vjp_call_jaxpr_abstract_eval(*_, fun_jaxpr, **__):
  disallowed_effects = effects.custom_derivatives_allowed_effects.filter_not_in(fun_jaxpr.effects)
  if disallowed_effects:
    raise NotImplementedError(
        f'Effects not supported in `custom_vjp`: {disallowed_effects}')
  return fun_jaxpr.out_avals, fun_jaxpr.effects

custom_vjp_call_jaxpr_p = core.Primitive('custom_vjp_call_jaxpr')
custom_vjp_call_jaxpr_p.multiple_results = True
custom_vjp_call_jaxpr_p.def_impl(_custom_vjp_call_jaxpr_impl)
custom_vjp_call_jaxpr_p.def_effectful_abstract_eval(_custom_vjp_call_jaxpr_abstract_eval)
CustomVJPCallPrimitive.initial_style = custom_vjp_call_jaxpr_p

mlir.register_lowering(custom_vjp_call_jaxpr_p, mlir.lower_fun(
    _custom_vjp_call_jaxpr_impl, multiple_results=True))

def _custom_vjp_call_jaxpr_jvp(
    primals, tangents, *, fun_jaxpr: core.ClosedJaxpr,
    fwd_jaxpr_thunk: Callable[..., tuple[core.Jaxpr, Sequence[Any]]],
    num_consts: int, bwd: Callable, out_trees: Callable, symbolic_zeros: bool):
  _, args = split_list(primals, [num_consts])
  consts_dot, args_dot = split_list(tangents, [num_consts])
  if any(type(t) is not Zero for t in consts_dot):
    raise ad.CustomVJPException()
  zeros = [type(t) is not Zero for t in args_dot]
  fwd_jaxpr, fwd_consts = fwd_jaxpr_thunk(*zeros)  # consts can be tracers!
  _, res_tree = out_trees()
  res_and_primals_out = core.eval_jaxpr(fwd_jaxpr, fwd_consts, *args)
  res, primals_out = split_list(res_and_primals_out, [res_tree.num_leaves])
  avals_out = [core.get_aval(x).to_tangent_aval() for x in primals_out]
  args_dot = map(ad.instantiate_zeros, args_dot)
  tangents_out = ad.custom_lin_p.bind(
      *res, *args_dot, num_res=res_tree.num_leaves, bwd=bwd,
      out_avals=avals_out, symbolic_zeros=symbolic_zeros)
  tangents_out = map(lax.tie_p.bind, primals_out, tangents_out)
  return primals_out, tangents_out
ad.primitive_jvps[custom_vjp_call_jaxpr_p] = _custom_vjp_call_jaxpr_jvp

def _custom_vjp_call_jaxpr_vmap(
    axis_data, args, in_dims, *,
    fun_jaxpr: core.ClosedJaxpr,
    fwd_jaxpr_thunk: Callable[..., tuple[core.Jaxpr, Sequence[Any]]],
    num_consts: int, bwd: Callable, out_trees: Callable, symbolic_zeros: bool):
  args = [batching.moveaxis(x, d, 0) if d is not not_mapped and d != 0
          else x for x, d in zip(args, in_dims)]
  in_batched = [d is not not_mapped for d in in_dims]
  _, args_batched = split_list(in_batched, [num_consts])
  batched_fun_jaxpr, out_batched = batching.batch_jaxpr(
      fun_jaxpr, axis_data, in_batched, False)
  out_dims1 = [0 if b else not_mapped for b in out_batched]
  out_dims2 = []

  @pe._memoize
  def batched_fwd_jaxpr_thunk(*zeros):
    fwd_jaxpr = core.ClosedJaxpr(*fwd_jaxpr_thunk(*zeros))  # consts can be tracers
    batched_fwd_jaxpr, out_batched = batching.batch_jaxpr(
        fwd_jaxpr, axis_data, args_batched, False)
    out_dims2.append([0 if b else not_mapped for b in out_batched])
    return batched_fwd_jaxpr.jaxpr, batched_fwd_jaxpr.consts

  fwd_args_batched = [0 if b else not_mapped for b in args_batched]
  fwd_out_dims = lambda: out_dims2[0]
  tag = core.TraceTag()
  batched_bwd = batching.batch_custom_vjp_bwd(
    bwd, tag, axis_data, fwd_out_dims, fwd_args_batched)

  batched_outs = custom_vjp_call_jaxpr_p.bind(
      *args, fun_jaxpr=batched_fun_jaxpr,
      fwd_jaxpr_thunk=batched_fwd_jaxpr_thunk, bwd=batched_bwd,
      num_consts=num_consts, out_trees=out_trees, symbolic_zeros=symbolic_zeros)
  out_dims = out_dims2[0] if out_dims2 else out_dims1
  return batched_outs, out_dims
batching.fancy_primitive_batchers[custom_vjp_call_jaxpr_p] = _custom_vjp_call_jaxpr_vmap

xla.register_initial_style_primitive(custom_vjp_call_jaxpr_p)

batching.primitive_batchers[ad.custom_lin_p] = ad.raise_custom_vjp_error_on_jvp
mlir.register_lowering(ad.custom_lin_p, ad.raise_custom_vjp_error_on_jvp)


def custom_gradient(fun):
  """Convenience function for defining custom VJP rules (aka custom gradients).

  While the canonical way to define custom VJP rules is via ``jax.custom_vjp``,
  the ``custom_gradient`` convenience wrapper follows TensorFlow's
  ``tf.custom_gradient`` API. The difference here is that ``custom_gradient``
  can be used as a decorator on one function that returns both the primal value
  (representing the output of the mathematical function to be differentiated)
  and the VJP (gradient) function. See
  https://www.tensorflow.org/api_docs/python/tf/custom_gradient.

  If the mathematical function to be differentiated has Haskell-like signature
  ``a -> b``, then the Python callable ``fun`` should have the signature
  ``a -> (b, CT b --o CT a)`` where we use ``CT x`` to denote a cotangent type
  for ``x`` and the ``--o`` arrow to denote a linear function. See the example
  below. That is, ``fun`` should return a pair where the first element
  represents the value of the mathematical function to be differentiated and the
  second element is a function to be called on the backward pass of reverse-mode
  automatic differentiation (i.e. the "custom gradient" function).

  The function returned as the second element of the output of ``fun`` can close
  over intermediate values computed when evaluating the function to be
  differentiated. That is, use lexical closure to share work between the forward
  pass and the backward pass of reverse-mode automatic differentiation. However,
  it cannot perform Python control flow which depends on the values of the
  closed-over intermediate values or its cotangent arguments; if the function
  includes such control flow, an error is raised.

  Args:
    fun: a Python callable specifying both the mathematical function to be
      differentiated and its reverse-mode differentiation rule. It should return
      a pair consisting of an output value and a Python callable that represents
      the custom gradient function.

  Returns:
    A Python callable that accepts the same arguments as ``fun`` and returns the
    output value specified by the first element of ``fun``'s output pair.

  For example:

  >>> @jax.custom_gradient
  ... def f(x):
  ...   return x ** 2, lambda g: (g * x,)
  ...
  >>> print(f(3.))
  9.0
  >>> print(jax.grad(f)(3.))
  3.0

  An example with a function on two arguments, so that the VJP function must
  return a tuple of length two:

  >>> @jax.custom_gradient
  ... def f(x, y):
  ...   return x * y, lambda g: (g * y, g * x)
  ...
  >>> print(f(3., 4.))
  12.0
  >>> print(jax.grad(f, argnums=(0, 1))(3., 4.))
  (Array(4., dtype=float32, weak_type=True), Array(3., dtype=float32, weak_type=True))
  """
  @custom_vjp
  def wrapped_fun(*args, **kwargs):
    ans, _ = fun(*args, **kwargs)
    return ans

  def fwd(*args, **kwargs):
    ans, rule = fun(*args, **kwargs)
    ans_flat, out_tree = tree_flatten((ans,))
    rule, in_tree = flatten_fun_nokwargs(lu.wrap_init(rule), out_tree)
    ans_avals = [core.get_aval(x).to_tangent_aval() for x in ans_flat]
    jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(rule, ans_avals)
    return ans, Residuals(jaxpr, in_tree(), out_tree, consts)

  def bwd(res, cts):
    jaxpr, in_tree, out_tree, consts = res
    cts_flat, out_tree_ = tree_flatten((cts,))
    if out_tree != out_tree_: raise TypeError(f'{out_tree}\n!=\n{out_tree_}')
    cts_out = core.eval_jaxpr(jaxpr, consts, *cts_flat)
    cts_out = tree_unflatten(in_tree, cts_out)
    if treedef_is_leaf(in_tree):
      cts_out = (cts_out,)
    return cts_out

  wrapped_fun.defvjp(fwd, bwd)
  return wrapped_fun

@register_pytree_node_class
class Residuals:
  def __init__(self, jaxpr, in_tree, out_tree, consts):
    self.jaxpr = jaxpr
    self.in_tree = in_tree
    self.out_tree = out_tree
    self.consts = consts
  def __iter__(self):
    return iter((self.jaxpr, self.in_tree, self.out_tree, self.consts))
  def tree_flatten(self):
    return self.consts, (self.jaxpr, self.in_tree, self.out_tree)
  @classmethod
  def tree_unflatten(cls, aux, consts):
    jaxpr, in_tree, out_tree = aux
    return cls(jaxpr, in_tree, out_tree, consts)


def closure_convert(fun: Callable, *example_args) -> tuple[Callable, list[Any]]:
  """Closure conversion utility, for use with higher-order custom derivatives.

  To define custom derivatives such as with ``jax.custom_vjp(f)``, the target
  function ``f`` must take, as formal arguments, all values involved in
  differentiation. If ``f`` is a higher-order function, in that it accepts as an
  argument a Python function ``g``, then values stored away in ``g``'s closure
  will not be visible to the custom derivative rules, and attempts at AD
  involving these values will fail. One way around this is to convert the
  closure by extracting these values, and to pass them as explicit formal
  arguments across the custom derivative boundary. This utility carries out that
  conversion. More precisely, it closure-converts the function ``fun``
  specialized to the types of the arguments given in ``example_args``.

  When we refer here to "values in the closure" of ``fun``, we do not mean the
  values that are captured by Python directly when ``fun`` is defined (e.g. the
  Python objects in ``fun.__closure__``, if the attribute exists). Rather, we
  mean values encountered during the execution of ``fun`` on ``example_args``
  that determine its output. This may include, for instance, arrays captured
  transitively in Python closures, i.e. in the Python closure of functions
  called by ``fun``, the closures of the functions that they call, and so forth.

  The function ``fun`` must be a pure function.

  Example usage::

    def minimize(objective_fn, x0):
      converted_fn, aux_args = closure_convert(objective_fn, x0)
      return _minimize(converted_fn, x0, *aux_args)

    @partial(custom_vjp, nondiff_argnums=(0,))
    def _minimize(objective_fn, x0, *args):
      z = objective_fn(x0, *args)
      # ... find minimizer x_opt ...
      return x_opt

    def fwd(objective_fn, x0, *args):
      y = _minimize(objective_fn, x0, *args)
      return y, (y, args)

    def rev(objective_fn, res, g):
      y, args = res
      y_bar = g
      # ... custom reverse-mode AD ...
      return x0_bar, *args_bars

    _minimize.defvjp(fwd, rev)

  Args:
    fun: Python callable to be converted. Must be a pure function.
    example_args: Arrays, scalars, or (nested) standard Python
      containers (tuples, lists, dicts, namedtuples, i.e., pytrees)
      thereof, used to determine the types of the formal arguments to
      ``fun``. This type-specialized form of ``fun`` is the function
      that will be closure converted.

  Returns:
    A pair comprising (i) a Python callable, accepting the same
    arguments as ``fun`` followed by arguments corresponding to the
    values hoisted from its closure, and (ii) a list of values hoisted
    from the closure.
  """
  flat_args, in_tree = tree_flatten(example_args)
  in_avals = tuple(map(core.get_aval, flat_args))
  if config.check_tracer_leaks.value:
    return _closure_convert_for_avals.__wrapped__(fun, in_tree, in_avals)
  else:
    return _closure_convert_for_avals(fun, in_tree, in_avals)

def _maybe_perturbed(x: Any) -> bool:
  # False if x can't represent an AD-perturbed value (i.e. a value
  # with a nontrivial tangent attached), up to heuristics, and True otherwise.
  # See https://github.com/google/jax/issues/6415 for motivation.
  if not isinstance(x, core.Tracer):
    # If x is not a Tracer, it can't be perturbed.
    return False
  elif isinstance(x, ad.JVPTracer) and isinstance(x.tangent, ad.Zero):
    return _maybe_perturbed(x.primal)
  elif isinstance(x, pe.DynamicJaxprTracer):
    # If x is a DynamicJaxprTracer then we're staging out; differentiation could
    # happen later, but some types always have trivial tangents.
    vspace = x.aval.to_tangent_aval()
    return not (vspace is core.abstract_token or
                getattr(vspace, 'dtype', None) == dtypes.float0)
  elif not isinstance(x, ad.JVPTracer):
    # If x is not a JVPTracer, recursively check its contents.
    return any(_maybe_perturbed(attr) for name, attr in x._contents())
  else:
    return True  # We can't be sure!

@cache()
def _closure_convert_for_avals(fun, in_tree, in_avals):
  wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  jaxpr, out_pvals, consts, () = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals)
  out_tree = out_tree()

  (closure_consts, hoisted_consts), merge = partition_list(_maybe_perturbed, consts)
  num_consts = len(hoisted_consts)

  def converted_fun(*args_hconsts):
    num_args = len(args_hconsts) - num_consts
    args, hoisted_consts = split_list(args_hconsts, [num_args])
    consts = merge(closure_consts, hoisted_consts)
    all_args, in_tree2 = tree_flatten(tuple(args))
    if in_tree != in_tree2:
      msg = ("The inputs to the closure produced by closure_convert must have "
             "the same Pytree structure as the example arguments passed when "
             f"closure_convert was called. Expected {in_tree}, but got "
             f"{in_tree2}")
      raise TypeError(msg)
    out_flat = core.eval_jaxpr(jaxpr, consts, *all_args)
    return tree_unflatten(out_tree, out_flat)

  return converted_fun, hoisted_consts

def partition_list(choice, lst):
  out = [], []
  which = [out[choice(elt)].append(elt) or choice(elt) for elt in lst]
  def merge(l1, l2):
    i1, i2 = iter(l1), iter(l2)
    return [next(i2 if snd else i1) for snd in which]
  return out, merge


### Custom transposition

def linear_call(fun: Callable, fun_transpose: Callable, residual_args,
                linear_args):
  """Call a linear function, with a custom implementation for its transpose.

  The `Haskell-like type signatures`_ of ``fun`` and ``fun_transpose`` are:

  .. code-block:: haskell

    fun           :: r -> a -o b
    fun_transpose :: r -> b -o a

  where the ``-o`` arrow indicates a linear function, ``r`` is the
  residual input type and ``a`` is the linear input type.

  The functions ``fun`` and ``fun_transpose`` are coupled as
  transposes of one another. Specifically, the transpose of a
  ``linear_call`` primitive is another ``linear_call`` to
  ``fun_transpose``, with ``fun`` as its custom transposition.

  For example:

  >>> def f(r, x):
  ...   return x / r

  >>> def t(r, t):
  ...   return t / r

  >>> def div_add(x, denom):
  ...   return x + linear_call(f, t, denom, x)

  >>> def transpose(f, x_example):
  ...   def transposed(y):
  ...     x, = jax.linear_transpose(f, x_example)(y)
  ...     return x
  ...   return transposed

  >>> div_add(9., 3.)
  Array(12., dtype=float32, weak_type=True)

  >>> transpose(partial(div_add, denom=3.), 1.)(18.)  # custom
  Array(24., dtype=float32, weak_type=True)

  >>> transpose(lambda x: x + x / 3., 1.)(18.)  # reference
  Array(24., dtype=float32, weak_type=True)

  The above definition of ``f`` illustrates the purpose of a residual
  argument: division is linear in one of its inputs (the dividend
  ``x``) but not the other (the divisor ``r``).

  As another example:

  >>> def custom_id(x):
  ...   def f(_, x): return x
  ...   def t(_, t): return 7.
  ...   return linear_call(f, t, (), x)
  >>> custom_id(1.)
  1.0
  >>> transpose(custom_id, 1.)(1.)
  7.0
  >>> transpose(transpose(custom_id, 1.), 1.)(1.)
  1.0
  >>> transpose(transpose(transpose(custom_id, 1.), 1.), 1.)(1.)
  7.0

  Args:
    fun: a Python callable specifying a linear function. It should
      take two arguments: one of "residual" inputs (type ``r``),
      i.e. inputs in which the function is not necessarily linear, and
      one of "linear" inputs (type ``a``).  It should return output
      whose components are linear in the linear input (type ``b``).
    fun_transpose: a Python callable specifying a structurally linear
      function that is the transpose of ``fun`` with respect to its
      linear inputs. Its first argument is the same residual inputs
      (``r``) as ``fun``. Its second argument is of type
      ``b``. Finally, its output is of type ``a`` and each of its
      component are linear in its second argument (the ``b`` inputs).
    residual_args: Argument in which ``fun`` and ``fun_transpose`` are
      not necessarily linear. Not involved in transposition.
    linear_args: Argument in which ``fun`` and ``fun_transpose`` are
      linear and with respect to which the two are transposes.

  Returns:
    The call result, i.e. ``fun(residual_args, linear_args)``.

  .. _Haskell-like type signatures: https://wiki.haskell.org/Type_signature
  """
  operands_res, res_tree = tree_flatten(residual_args)
  operands_lin, lin_tree = tree_flatten(linear_args)

  f_in_tree = treedef_tuple((res_tree, lin_tree))
  f, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), f_in_tree)

  res_avals = map(core.get_aval, operands_res)
  lin_avals = map(core.get_aval, operands_lin)
  f_jaxpr, f_consts = _initial_style_jaxpr(f, (*res_avals, *lin_avals))
  f_jaxpr = _close_jaxpr(f_jaxpr)
  out_avals = f_jaxpr.out_avals

  t_in_tree = treedef_tuple((res_tree, out_tree()))
  t, t_out_tree = flatten_fun_nokwargs(lu.wrap_init(fun_transpose), t_in_tree)

  t_jaxpr, t_consts = _initial_style_jaxpr(t, (*res_avals, *out_avals))
  t_jaxpr = _close_jaxpr(t_jaxpr)

  if t_out_tree() != lin_tree:
    raise TypeError(
        'transpose output pytree structure must match that of linear inputs, '
        f'got output structure {t_out_tree()} '
        f'and input structure {lin_tree}.')

  out = linear_call_p.bind(*f_consts, *t_consts, *operands_res, *operands_lin,
                           callee=f_jaxpr,
                           transpose=t_jaxpr,
                           num_callee_consts=len(f_consts),
                           num_transpose_consts=len(t_consts),
                           num_res=len(operands_res))

  return tree_unflatten(out_tree(), out)

def _linear_call_impl(*args, callee, transpose, num_callee_consts,
                      num_transpose_consts, num_res):
  del transpose
  consts, _, operands_res, operands_lin = split_list(
      args, [num_callee_consts, num_transpose_consts, num_res])
  return core.eval_jaxpr(callee.jaxpr, (), *consts, *operands_res, *operands_lin)

def _linear_call_transpose_rule(cts, *args, callee, transpose,
                                num_callee_consts,
                                num_transpose_consts, num_res):
  f_consts, t_consts, operands_res, operands_lin = split_list(
      args, [num_callee_consts, num_transpose_consts, num_res])
  _, _, cts_avals = split_list(
      transpose.in_avals, [num_transpose_consts, num_res])

  assert all(ad.is_undefined_primal(x)     for x in operands_lin)
  assert all(not ad.is_undefined_primal(x) for x in operands_res)

  cts = [zeros_like_aval(a) if type(ct) is Zero else ct
         for ct, a in zip(cts, cts_avals)]

  cts_out = linear_call_p.bind(*t_consts, *f_consts, *operands_res, *cts,
                               callee=transpose,
                               transpose=callee,
                               num_callee_consts=len(t_consts),
                               num_transpose_consts=len(f_consts),
                               num_res=len(operands_res))

  return [None] * (num_callee_consts + num_transpose_consts + num_res) + cts_out

def _linear_call_abstract_eval(*args, **kwargs):
  return kwargs['callee'].out_avals

linear_call_p = core.Primitive('linear_call')
linear_call_p.multiple_results = True
linear_call_p.def_impl(_linear_call_impl)
linear_call_p.def_abstract_eval(_linear_call_abstract_eval)
ad.primitive_transposes[linear_call_p] = _linear_call_transpose_rule
xla.register_initial_style_primitive(linear_call_p)
mlir.register_lowering(linear_call_p, mlir.lower_fun(
    _linear_call_impl, multiple_results=True))


# A stageable primitive that fails when evaluated
unreachable_p: core.Primitive = core.Primitive('unreachable')
unreachable_p.multiple_results = True

def unreachable_impl(*_, out_avals, exc_type, message):
  del out_avals
  raise exc_type(message)

# Evaluation raises an exception
unreachable_p.def_impl(unreachable_impl)

# Translation raises an exception
# TODO(frostig,mattjj): We have no good way to translate a function
# that errs. Since MLIR lowering over-approximates concrete evaluation,
# we err on MLIR lowering for the time being.
mlir.register_lowering(unreachable_p, unreachable_impl)

# Abstract evaluation proceeds without issue, to allow for staging
unreachable_p.def_abstract_eval(lambda *_, out_avals, **__: out_avals)

def unreachable(*args, out_avals=None, exc_type=TypeError,
                message='unreachable'):
  """Fail when evaluated concretely (but allow for staging).

  This function allows one to assert an impossibility of
  evaluation. It can be used to guarantee that evaluation does not
  "reach" a certain point in the sense that it does not execute, but
  it can nonetheless be staged out by JAX without error.

  Args:
    *args: The arbitrary pytree of arguments to the function.
    out_avals: Optional specification of the output types of this
     function invocation from the point of view of staging. If
     ``None``, these are chosen as equal to types of input arguments.
    exc_type: Optional constructor for the Python exception raised if
      evaluated.
    message: Optional string message for the Python exception raised
      if evaluated.

  """
  if out_avals is None:
    out_avals = tree_map(core.get_aval, args)

  args_flat, in_tree = tree_flatten(args)
  out_avals_flat, out_tree = tree_flatten(out_avals)
  out = unreachable_p.bind(*args_flat, out_avals=out_avals_flat,
                           exc_type=exc_type, message=message)
  return tree_unflatten(out_tree, out)


disallow_jvp = partial(
    unreachable,
    exc_type=TypeError,
    message="can't apply forward-mode autodiff (jvp) to a custom_vjp function.")


def custom_vjp_by_custom_transpose(fun, fwd, bwd):
  fun = custom_jvp(fun)

  @fun.defjvp
  def jvp(primals, tangents):
    outs, residuals = fwd(*primals)
    tan_out_types = tree_map(lambda o: core.get_aval(o).to_tangent_aval(), outs)
    tan_fn = custom_transpose(partial(disallow_jvp, out_avals=tan_out_types))
    tan_fn.def_transpose(bwd)
    return outs, tan_fn(tan_out_types, residuals, tangents)

  return fun


# TODO(mattjj): remove these stubs, which exist to avoid breaking internal users
custom_jvp_call_jaxpr_p = core.Primitive("custom_jvp_call_jaxpr")


# The following is a helper for optimizing the behavior of custom_vjp when used
# under remat. This is really only useful when the `fwd` function to custom_vjp
# executes a black box kernel. Otherwise, DCE will perform this optimization
# automatically.
#
# TODO(dfm): Eventually this should probably be the default behavior for
# custom_vjp, if we can make it so that it is a no-op for most cases. Right now,
# it is written in "initial-style" so it doesn't support eager mode. This was
# a reasonable compromise when written because it made the implementation
# simpler, but it would be worth revisiting this.
def optimize_remat_of_custom_vjp_fwd(
    fun: Callable[..., ReturnValue],
    fwd: Callable[..., tuple[ReturnValue, Any]],
    nondiff_argnums: Sequence[int] = (),
    symbolic_zeros: bool = False,
) -> Callable[..., tuple[ReturnValue, Any]]:
  if symbolic_zeros:
    # TODO(dfm): This probably shouldn't be too hard to support.
    raise NotImplementedError(
        "remat optimization for custom_vjp does not support symbolic zeros")

  @wraps(fwd)
  def wrapped_fwd(*args, **kwargs) -> tuple[ReturnValue, Any]:
    # TODO(dfm): This initial logic is duplicated from custom_vjp.__call__
    # above and it would be good to consolidate it.
    primal_name = getattr(fun, "__name__", str(fun))
    fwd_name = getattr(fwd, "__name__", str(fwd))
    # Note: we use `fun` instead of `fwd` here for consistency with
    # custom_vjp.__call__ above.
    args = resolve_kwargs(fun, args, kwargs)
    if nondiff_argnums:
      for i in nondiff_argnums: _check_for_tracers(args[i])
      nondiff_argnums_ = set(nondiff_argnums)
      dyn_argnums = [i for i in range(len(args)) if i not in nondiff_argnums_]
      f_, dyn_args = argnums_partial(lu.wrap_init(fun), dyn_argnums,
                                      args, require_static_args_hashable=False)
      fwd_, _ = argnums_partial(lu.wrap_init(fwd), dyn_argnums, args,
                                require_static_args_hashable=False)
    else:
      f_, dyn_args = lu.wrap_init(fun), args
      fwd_ = lu.wrap_init(fwd)
    args_flat, in_tree = tree_flatten(dyn_args)
    flat_fun, out_type = _flatten_fun_nokwargs(f_, in_tree)
    flat_fwd, out_trees = _flatten_fwd(fwd_, nondiff_argnums, False,
                                       primal_name, fwd_name, in_tree, out_type)
    flat_fwd = _fix_fwd_args(flat_fwd)

    in_avals = [core.get_aval(x) for x in args_flat]
    fwd_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(flat_fwd, in_avals)
    fwd_jaxpr = pe.close_jaxpr(pe.convert_constvars_jaxpr(fwd_jaxpr))
    prim_tree, res_tree = out_trees()
    num_res = res_tree.num_leaves

    if fwd_jaxpr.effects:
      raise NotImplementedError(
          "remat optimization for custom_vjp does not support forward "
          f"functions with side effects, but {fwd_name} has the following "
          f"effects: {fwd_jaxpr.effects}")

    @pe._memoize
    def fun_jaxpr_thunk():
      jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
      return jaxpr, consts

    out_flat = remat_opt_p.bind(*consts, *args_flat,
                                num_consts=len(consts),
                                num_res=num_res,
                                fwd_jaxpr=fwd_jaxpr,
                                fun_jaxpr_thunk=fun_jaxpr_thunk)
    res, out_flat = split_list(out_flat, [num_res])
    out_tree = treedef_tuple((prim_tree, res_tree))
    return tree_unflatten(out_tree, (*out_flat, *res))

  return wrapped_fwd

@lu.transformation2
def _fix_fwd_args(f, *args):
  args = [(x, True) for x in args]
  args = [x for pair in args for x in pair]
  return f(*args)

def _remat_opt_impl(
    *args,
    num_consts: int,
    num_res: int,
    fwd_jaxpr: core.ClosedJaxpr,
    fun_jaxpr_thunk: Callable[[], core.ClosedJaxpr],
):
  del num_consts, num_res, fun_jaxpr_thunk  # unused
  return core.jaxpr_as_fun(fwd_jaxpr)(*args)

def _remat_opt_abstract_eval(*args, fwd_jaxpr: core.ClosedJaxpr, **_):
  del args
  return fwd_jaxpr.out_avals, fwd_jaxpr.effects

def _remat_opt_vmap(
    axis_data, args, in_dims,
    *,
    num_consts: int,
    num_res: int,
    fwd_jaxpr: core.ClosedJaxpr,
    fun_jaxpr_thunk: Callable[[], core.ClosedJaxpr],
):
  args = [batching.moveaxis(x, d, 0) if d is not not_mapped and d != 0
          else x for x, d in zip(args, in_dims)]
  in_batched = [d is not not_mapped for d in in_dims]
  batched_fwd_jaxpr, out_batched = batching.batch_jaxpr(
      fwd_jaxpr, axis_data, in_batched, False)
  extra_consts = batched_fwd_jaxpr.consts
  batched_fwd_jaxpr = pe.close_jaxpr(
      pe.convert_constvars_jaxpr(batched_fwd_jaxpr.jaxpr))
  out_dims = [0 if b else not_mapped for b in out_batched]

  _, prim_batched = split_list(in_batched, [num_consts])

  @pe._memoize
  def batched_fun_jaxpr_thunk():
    fun_jaxpr = core.ClosedJaxpr(*fun_jaxpr_thunk())
    batched_fun_jaxpr, out_batched = batching.batch_jaxpr(
        fun_jaxpr, axis_data, prim_batched, False)
    return batched_fun_jaxpr.jaxpr, batched_fun_jaxpr.consts

  batched_outs = remat_opt_p.bind(*extra_consts, *args,
                                  num_consts=num_consts + len(extra_consts),
                                  num_res=num_res,
                                  fwd_jaxpr=batched_fwd_jaxpr,
                                  fun_jaxpr_thunk=batched_fun_jaxpr_thunk)

  return batched_outs, out_dims

def _remat_opt_jvp(
    primals,
    tangents,
    *,
    num_consts: int,
    num_res: int,
    fwd_jaxpr: core.ClosedJaxpr,
    fun_jaxpr_thunk: Callable[[], core.ClosedJaxpr],
):
  consts, primals = split_list(primals, [num_consts])
  consts_dot, tangents = split_list(tangents, [num_consts])
  # Tangents must be instantated in case we end up DCEing later.
  tangents = map(ad.instantiate_zeros, tangents)
  consts_nz = [not isinstance(t, Zero) for t in consts_dot]
  consts_dot = [c for nz, c in zip(consts_nz, consts_dot) if nz]
  in_nz = consts_nz + [True] * len(tangents)
  fwd_jaxpr_jvp_, out_nz = ad.jvp_jaxpr(fwd_jaxpr, in_nz, True)
  num_out = len(out_nz) - num_res
  fwd_jaxpr_jvp_ = ad.rearrange_binders(
      fwd_jaxpr_jvp_, [num_consts, len(primals)],
      [len(consts_dot), len(tangents)], [num_res, num_out], [num_res, num_out])
  fwd_jaxpr_jvp = pe.close_jaxpr(pe.convert_constvars_jaxpr(fwd_jaxpr_jvp_.jaxpr))

  # @pe._memoize
  def fun_jvp_jaxpr_thunk():
    fun_jaxpr = core.ClosedJaxpr(*fun_jaxpr_thunk())
    in_nz = [True] * len(primals)
    fun_jvp_jaxpr, _ = ad.jvp_jaxpr(fun_jaxpr, in_nz, True)
    return fun_jvp_jaxpr.jaxpr, fun_jvp_jaxpr.consts

  new_num_consts = len(fwd_jaxpr_jvp_.consts) + num_consts + len(consts_dot)
  outs = remat_opt_p.bind(*fwd_jaxpr_jvp_.consts, *consts, *consts_dot,
                          *primals, *tangents, num_consts=new_num_consts,
                          num_res=2 * num_res, fwd_jaxpr=fwd_jaxpr_jvp,
                          fun_jaxpr_thunk=fun_jvp_jaxpr_thunk)
  res, res_dot, outs, outs_dot = split_list(outs, [num_res, num_res, num_out])
  return (*res, *outs), (*res_dot, *outs_dot)

def _remat_opt_transpose(
    cts, *args,
    num_consts: int,
    num_res: int,
    fwd_jaxpr: core.ClosedJaxpr,
    fun_jaxpr_thunk: Callable[[], core.ClosedJaxpr],
):
  # TODO(dfm): It shouldn't be too hard to implement this as needed in the
  # future.
  raise NotImplementedError(
      "remat optimization for custom_vjp does not support higher-order AD")

def _remat_opt_dce(used_outs: list[bool], eqn: core.JaxprEqn):
  if not any(used_outs) and not pe.has_effects(eqn):
    return [False] * len(eqn.invars), None
  used_res, used_prims = split_list(used_outs, [eqn.params["num_res"]])
  outvars = [v for used, v in zip(used_outs, eqn.outvars) if used]
  if any(used_res):
    # If any of the residuals are used, we still need to run fwd at this point,
    # but we may end up DCEing again in the future, so we must instantiate all
    # the input primals.
    instantiate = [False] * eqn.params["num_consts"]
    instantiate += [True] * (len(eqn.invars) - eqn.params["num_consts"])
    new_jaxpr, used_ins = pe.dce_jaxpr(eqn.params["fwd_jaxpr"].jaxpr, used_outs,
                                       instantiate=instantiate)
    assert not new_jaxpr.constvars
    closed_jaxpr = pe.close_jaxpr(new_jaxpr)
    invars = [v for used, v in zip(used_ins, eqn.invars) if used]
    new_params = dict(eqn.params)
    new_num_consts = sum(split_list(used_ins, [eqn.params["num_consts"]])[0])
    new_params["num_consts"] = new_num_consts
    new_params["fwd_jaxpr"] = closed_jaxpr
    new_params["num_res"] = sum(used_res)
    new_eqn = pe.new_jaxpr_eqn(
        invars, outvars, remat_opt_p, new_params, closed_jaxpr.effects,
        eqn.source_info, eqn.ctx)
    return used_ins, new_eqn
  else:
    # If none of the residuals are used, we run the primal computation instead.
    # At this point we drop this custom DCE behavior, but since the primal might
    # have different consts than fwd, we build a new JaxprEqn with a closed_call
    # primitive.
    fun_jaxpr, consts = eqn.params["fun_jaxpr_thunk"]()
    new_jaxpr, used_consts, used_ins = pe.dce_jaxpr_consts(fun_jaxpr, used_prims)
    consts = [c for used, c in zip(used_consts, consts) if used]
    closed_jaxpr = core.ClosedJaxpr(new_jaxpr, consts)
    _, invars = split_list(eqn.invars, [eqn.params["num_consts"]])
    invars = [v for used, v in zip(used_ins, invars) if used]
    new_eqn = pe.new_jaxpr_eqn(
        invars, outvars, core.closed_call_p, dict(call_jaxpr=closed_jaxpr),
        closed_jaxpr.effects, eqn.source_info, eqn.ctx)
    used_ins = [False] * eqn.params["num_consts"] + used_ins
    return used_ins, new_eqn

remat_opt_p = core.Primitive("remat_opt")
remat_opt_p.multiple_results = True
remat_opt_p.def_impl(_remat_opt_impl)
remat_opt_p.def_effectful_abstract_eval(_remat_opt_abstract_eval)
xla.register_initial_style_primitive(remat_opt_p)
mlir.register_lowering(remat_opt_p, mlir.lower_fun(
    _remat_opt_impl, multiple_results=True))


batching.fancy_primitive_batchers[remat_opt_p] = _remat_opt_vmap
ad.primitive_jvps[remat_opt_p] = _remat_opt_jvp
ad.primitive_transposes[remat_opt_p] = _remat_opt_transpose
pe.dce_rules[remat_opt_p] = _remat_opt_dce
