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

r"""Jet is an experimental module for higher-order automatic differentiation
  that does not rely on repeated first-order automatic differentiation.

  How? Through the propagation of truncated Taylor polynomials.
  Consider a function :math:`f = g \circ h`, some point :math:`x`
  and some offset :math:`v`.
  First-order automatic differentiation (such as :func:`jax.jvp`)
  computes the pair :math:`(f(x), \partial f(x)[v])` from the pair
  :math:`(h(x), \partial h(x)[v])`.

  :func:`jet` implements the higher-order analogue:
  Given the tuple

  .. math::
    (h_0, ... h_K) :=
    (h(x), \partial h(x)[v], \partial^2 h(x)[v, v], ..., \partial^K h(x)[v,...,v]),

  which represents a :math:`K`-th order Taylor approximation
  of :math:`h` at :math:`x`, :func:`jet` returns a :math:`K`-th order
  Taylor approximation of :math:`f` at :math:`x`,

  .. math::
    (f_0, ..., f_K) :=
    (f(x), \partial f(x)[v], \partial^2 f(x)[v, v], ..., \partial^K f(x)[v,...,v]).

  More specifically, :func:`jet` computes

  .. math::
    f_0, (f_1, . . . , f_K) = \texttt{jet} (f, h_0, (h_1, . . . , h_K))

  and can thus be used for high-order
  automatic differentiation of :math:`f`.
  Details are explained in
  `these notes <https://github.com/jax-ml/jax/files/6717197/jet.pdf>`__.

  Note:
    Help improve :func:`jet` by contributing
    `outstanding primitive rules <https://github.com/jax-ml/jax/issues/2431>`__.
"""

from collections.abc import Callable
from typing import Any

from functools import partial

import numpy as np

from jax import lax
import jax.numpy as jnp
from jax.experimental import pjit
from jax.tree_util import (register_pytree_node, tree_structure,
                           treedef_is_leaf, tree_flatten, tree_unflatten,)

from jax._src import ad_util
from jax._src import core
from jax._src import dispatch
from jax._src import linear_util as lu
from jax._src import sharding_impls
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax as lax_internal
from jax._src.util import unzip2, weakref_lru_cache, safe_zip


def jet(fun, primals, series):
  r"""Taylor-mode higher-order automatic differentiation.

  Args:
    fun: Function to be differentiated. Its arguments should be arrays, scalars,
      or standard Python containers of arrays or scalars. It should return an
      array, scalar, or standard Python container of arrays or scalars.
    primals: The primal values at which the Taylor approximation of ``fun`` should be
      evaluated. Should be either a tuple or a list of arguments,
      and its length should be equal to the number of positional parameters of
      ``fun``.
    series: Higher order Taylor-series-coefficients.
      Together, `primals` and `series` make up a truncated Taylor polynomial.
      Should be either a tuple or a list of tuples or lists,
      and its length dictates the degree of the truncated Taylor polynomial.

  Returns:
    A ``(primals_out, series_out)`` pair, where ``primals_out`` is ``fun(*primals)``,
    and together, ``primals_out`` and ``series_out`` are a
    truncated Taylor polynomial of :math:`f(h(\cdot))`.
    The ``primals_out`` value has the same Python tree structure as ``primals``,
    and the ``series_out`` value the same Python tree structure as ``series``.

  For example:

  >>> import jax
  >>> import jax.numpy as np

  Consider the function :math:`h(z) = z^3`, :math:`x = 0.5`,
  and the first few Taylor coefficients
  :math:`h_0=x^3`, :math:`h_1=3x^2`, and :math:`h_2=6x`.
  Let :math:`f(y) = \sin(y)`.

  >>> h0, h1, h2 = 0.5**3., 3.*0.5**2., 6.*0.5
  >>> f, df, ddf = np.sin, np.cos, lambda *args: -np.sin(*args)

  :func:`jet` returns the Taylor coefficients of :math:`f(h(z)) = \sin(z^3)`
  according to Faà di Bruno's formula:

  >>> f0, (f1, f2) =  jet(f, (h0,), ((h1, h2),))
  >>> print(f0,  f(h0))
  0.12467473 0.12467473

  >>> print(f1, df(h0) * h1)
  0.7441479 0.74414825

  >>> print(f2, ddf(h0) * h1 ** 2 + df(h0) * h2)
  2.9064622 2.9064634
  """
  try:
    order, = set(map(len, series))
  except ValueError:
    msg = "jet terms have inconsistent lengths for different arguments"
    raise ValueError(msg) from None

  # TODO(mattjj): consider supporting pytree inputs
  for i, (x, terms) in enumerate(zip(primals, series)):
    treedef = tree_structure(x)
    if not treedef_is_leaf(treedef):
      raise ValueError(f"primal value at position {i} is not an array")
    for j, t in enumerate(terms):
      treedef = tree_structure(t)
      if not treedef_is_leaf(treedef):
        raise ValueError(f"term {j} for argument {i} is not an array")

  @lu.transformation_with_aux2
  def flatten_fun_output(f, store, *args):
    ans = f(*args)
    ans, tree = tree_flatten(ans)
    store.store(tree)
    return ans

  f, out_tree = flatten_fun_output(lu.wrap_init(fun))
  out_primals, out_terms = jet_fun(jet_subtrace(f), order).call_wrapped(primals, series)
  return tree_unflatten(out_tree(), out_primals), tree_unflatten(out_tree(), out_terms)

@lu.transformation2
def jet_fun(f, order, primals, series):
  tag = core.TraceTag()
  out_primals, out_terms = f(tag, order, primals, series)
  out_terms = [[jnp.zeros_like(p)] * order if s is zero_series else s
               for p, s in zip(out_primals, out_terms)]
  return out_primals, out_terms

@lu.transformation2
def jet_subtrace(f, tag, order, primals, series):
  with core.take_current_trace() as parent_trace:
    trace = JetTrace(tag, parent_trace, order)
    in_tracers = map(partial(JetTracer, trace), primals, series)
    with core.set_current_trace(trace):
       ans = f(*in_tracers)

    out_primals, out_terms = unzip2(map(trace.to_primal_terms_pair, ans))
    return out_primals, out_terms

@lu.transformation_with_aux2
def traceable(f, store, in_tree_def, *primals_and_series):
  primals_in, series_in = tree_unflatten(in_tree_def, primals_and_series)
  primals_out, series_out = f(primals_in, series_in)
  out_flat, out_tree_def = tree_flatten((primals_out, series_out))
  store.store(out_tree_def)
  return out_flat


class JetTracer(core.Tracer):
  __slots__ = ["primal", "terms"]

  def __init__(self, trace, primal, terms):
    assert type(terms) in (ZeroSeries, list, tuple)
    self._trace = trace
    self.primal = primal
    self.terms = terms

  @property
  def aval(self):
    return core.get_aval(self.primal)

  def full_lower(self):
    if self.terms is zero_series or all(t is zero_term for t in self.terms):
      return core.full_lower(self.primal)
    else:
      return self

class JetTrace(core.Trace):

  def __init__(self, tag, parent_trace, order):
    self.tag = tag
    self.parent_trace = parent_trace
    self.order = order

  def to_primal_terms_pair(self, val):
    if isinstance(val, JetTracer) and val._trace.tag is self.tag:
      return val.primal, val.terms
    else:
      return val, zero_series

  def process_primitive(self, primitive, tracers, params):
    order = self.order              # pytype: disable=attribute-error
    primals_in, series_in = unzip2(map(self.to_primal_terms_pair, tracers))

    if all(t is zero_series for t in series_in):
      primal_out = primitive.bind_with_trace(self.parent_trace, primals_in, params)
      if primitive.multiple_results:
        return [JetTracer(self, p, zero_series) for p in primal_out]
      else:
        return JetTracer(self, primal_out, zero_series)

    series_in = [[zero_term] * order if s is zero_series else s
                 for s in series_in]
    with core.set_current_trace(self.parent_trace):
      # TODO(mattjj): avoid always instantiating zeros
      series_in = [[jnp.zeros(np.shape(x), dtype=jnp.result_type(x))
                    if t is zero_term else t for t in series]
                   for x, series in zip(primals_in, series_in)]
      rule = jet_rules[primitive]
      primal_out, terms_out = rule(primals_in, series_in, **params)
    if not primitive.multiple_results:
      return JetTracer(self, primal_out, terms_out)
    else:
      return [JetTracer(self, p, ts) for p, ts in zip(primal_out, terms_out)]

  def process_call(self, call_primitive, f, tracers, params):
    primals_in, series_in = unzip2(map(self.to_primal_terms_pair, tracers))
    primals_and_series, in_tree_def = tree_flatten((primals_in, series_in))
    f_jet, out_tree_def = traceable(jet_subtrace(f, self.main), in_tree_def)
    update_params = call_param_updaters.get(call_primitive)
    new_params = (update_params(params, len(primals_and_series))
                  if update_params else params)
    result = call_primitive.bind(f_jet, *primals_and_series, **new_params)
    primals_out, series_out = tree_unflatten(out_tree_def(), result)
    return [JetTracer(self, p, ts) for p, ts in zip(primals_out, series_out)]

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *,
                              symbolic_zeros):
    # TODO(mattjj): don't just ignore custom jvp rules?
    del primitive, jvp  # Unused.
    return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
    del primitive, fwd, bwd, out_trees  # Unused.
    return fun.call_wrapped(*tracers)


class ZeroTerm: pass
zero_term = ZeroTerm()
register_pytree_node(ZeroTerm, lambda z: ((), None), lambda _, xs: zero_term)

class ZeroSeries: pass
zero_series = ZeroSeries()
register_pytree_node(ZeroSeries, lambda z: ((), None), lambda _, xs: zero_series)


call_param_updaters: dict[core.Primitive, Callable[..., Any]] = {}


### rule definitions

jet_rules = {}

def defzero(prim):
  jet_rules[prim] = partial(zero_prop, prim)

def zero_prop(prim, primals_in, series_in, **params):
  primal_out = prim.bind(*primals_in, **params)
  return primal_out, zero_series

defzero(lax.le_p)
defzero(lax.lt_p)
defzero(lax.gt_p)
defzero(lax.ge_p)
defzero(lax.eq_p)
defzero(lax.ne_p)
defzero(lax.not_p)
defzero(lax.and_p)
defzero(lax.or_p)
defzero(lax.xor_p)
defzero(lax.floor_p)
defzero(lax.ceil_p)
defzero(lax.round_p)
defzero(lax.sign_p)
defzero(ad_util.stop_gradient_p)
defzero(lax.is_finite_p)
defzero(lax.shift_left_p)
defzero(lax.shift_right_arithmetic_p)
defzero(lax.shift_right_logical_p)
defzero(lax.bitcast_convert_type_p)


def deflinear(prim):
  jet_rules[prim] = partial(linear_prop, prim)

def linear_prop(prim, primals_in, series_in, **params):
  primal_out = prim.bind(*primals_in, **params)
  series_out = [prim.bind(*terms_in, **params) for terms_in in zip(*series_in)]
  if prim.multiple_results:
    series_out = safe_zip(*series_out)
  return primal_out, series_out

deflinear(lax.neg_p)
deflinear(lax.real_p)
deflinear(lax.complex_p)
deflinear(lax.conj_p)
deflinear(lax.imag_p)
deflinear(lax.add_p)
deflinear(ad_util.add_jaxvals_p)
deflinear(lax.sub_p)
deflinear(lax.convert_element_type_p)
deflinear(lax.broadcast_in_dim_p)
deflinear(lax.concatenate_p)
deflinear(lax.split_p)
deflinear(lax.pad_p)
deflinear(lax.reshape_p)
deflinear(lax.squeeze_p)
deflinear(lax.rev_p)
deflinear(lax.transpose_p)
deflinear(lax.slice_p)
deflinear(lax.reduce_sum_p)
deflinear(lax.reduce_window_sum_p)
deflinear(lax.fft_p)
deflinear(lax.copy_p)
deflinear(dispatch.device_put_p)

def _dynamic_slice_jet_rule(primals_in, series_in, **params):
  operand, *start_indices = primals_in
  primal_out = lax.dynamic_slice_p.bind(operand, *start_indices, **params)
  series_out = [lax.dynamic_slice_p.bind(terms_in[0], *start_indices, **params)
                for terms_in in zip(*series_in)]
  return primal_out, series_out

jet_rules[lax.dynamic_slice_p] = _dynamic_slice_jet_rule

def _dynamic_update_slice_jet_rule(primals_in, series_in, **params):
  operand, update, *start_indices = primals_in
  primal_out = lax.dynamic_update_slice_p.bind(operand, update, *start_indices)
  series_out = [lax.dynamic_update_slice_p.bind(*terms_in[:2], *start_indices, **params)
                for terms_in in zip(*series_in)]
  return primal_out, series_out

jet_rules[lax.dynamic_update_slice_p] = _dynamic_update_slice_jet_rule

def _cumulative_jet_rule(primals_in, series_in, *, axis: int, reverse: bool,
                         combine_fn: Callable):
  # Irrespective of backend, we always use the parallel prefix scan
  # implementation when differentiating because reduce_window is not
  # arbitrarily differentiable.
  return jet(partial(lax.associative_scan, combine_fn, axis=axis,
                     reverse=reverse),
             primals_in, series_in)

deflinear(lax.cumsum_p)
jet_rules[lax.cumprod_p] = partial(_cumulative_jet_rule,
                                                combine_fn=lax.mul)
jet_rules[lax.cummax_p] = partial(_cumulative_jet_rule,
                                               combine_fn=lax.max)
jet_rules[lax.cummin_p] = partial(_cumulative_jet_rule,
                                               combine_fn=lax.min)


def def_deriv(prim, deriv):
  """
  Define the jet rule for a primitive in terms of its first derivative.
  """
  jet_rules[prim] = partial(deriv_prop, prim, deriv)


def deriv_prop(prim, deriv, primals_in, series_in):
  x, = primals_in
  series, = series_in
  primal_out = prim.bind(x)
  c0, cs = jet(deriv, primals_in, series_in)
  c = [c0] + cs
  u = [x] + series
  v = [primal_out] + [None] * len(series)
  for k in range(1, len(v)):
    v[k] = fact(k-1) * sum(_scale(k, j) * c[k-j] * u[j] for j in range(1, k + 1))
  primal_out, *series_out = v
  return primal_out, series_out


def_deriv(lax.erf_p,
          lambda x: lax.mul(
              lax_internal._const(x, 2. / np.sqrt(np.pi)),
              lax.exp(lax.neg(lax.square(x)))))


def def_comp(prim, comp):
  """
  Define the jet rule for a primitive in terms of a composition of simpler primitives.
  """
  jet_rules[prim] = partial(jet, comp)


def_comp(lax.expm1_p, lambda x: lax.exp(x) - 1)
def_comp(lax.log1p_p, lambda x: lax.log(1 + x))
def_comp(lax.sqrt_p, lambda x: x ** 0.5)
def_comp(lax.square_p, lambda x: x * x)
def_comp(lax.rsqrt_p, lambda x: x ** -0.5)
def_comp(lax.asinh_p, lambda x: lax.log(x + lax.sqrt(lax.square(x) + 1)))
def_comp(lax.acosh_p, lambda x: lax.log(x + lax.sqrt(lax.square(x) - 1)))
def_comp(lax.atanh_p, lambda x: 0.5 * lax.log(lax.div(1 + x, 1 - x)))
def_comp(lax.erfc_p, lambda x: 1 - lax.erf(x))
def_comp(lax.rem_p, lambda x, y: x - y * lax.floor(x / y))
def_comp(lax.clamp_p, lambda a, x, b: lax.min(lax.max(a, x), b))


def _erf_inv_rule(primals_in, series_in):
  x, = primals_in
  series, = series_in

  u = [x] + series
  primal_out = lax.erf_inv(x)
  v = [primal_out] + [None] * len(series)

  # derivative on co-domain for caching purposes
  deriv_const = np.sqrt(np.pi) / 2.
  deriv_y = lambda y: lax.mul(deriv_const, lax.exp(lax.square(y)))

  # manually propagate through deriv_y since we don't have lazy evaluation of sensitivities

  c = [deriv_y(primal_out)] + [None] * (len(series) - 1)
  tmp_sq = [lax.square(v[0])] + [None] * (len(series) - 1)
  tmp_exp = [lax.exp(tmp_sq[0])] + [None] * (len(series) - 1)
  for k in range(1, len(series)):
    # we know c[:k], we compute c[k]

    # propagate c to get v
    v[k] = fact(k-1) * sum(_scale(k, j) * c[k-j] * u[j] for j in range(1, k + 1))

    # propagate v to get next c

    # square
    tmp_sq[k] = fact(k) * sum(_scale2(k, j) * v[k-j] * v[j] for j in range(k + 1))

    # exp
    tmp_exp[k] = fact(k-1) * sum(_scale(k, j) * tmp_exp[k-j] * tmp_sq[j] for j in range(1, k + 1))

    # const
    c[k] = deriv_const * tmp_exp[k]

  # we can't, and don't need, to compute c[k+1], just need to get the last v[k]
  k = len(series)
  v[k] = fact(k-1) * sum(_scale(k, j) * c[k-j] * u[j] for j in range(1, k + 1))

  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.erf_inv_p] = _erf_inv_rule

### More complicated rules

def fact(n):
  return lax.exp(lax.lgamma(n+1.))

def _scale(k, j):
  return 1. / (fact(k - j) * fact(j - 1))

def _scale2(k, j):
  return 1. / (fact(k - j) * fact(j))

def _exp_taylor(primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [x] + series
  v = [lax.exp(x)] + [None] * len(series)
  for k in range(1,len(v)):
    v[k] = fact(k-1) * sum(_scale(k, j) * v[k-j] * u[j] for j in range(1, k+1))
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.exp_p] = _exp_taylor

def _pow_taylor(primals_in, series_in):
  u_, r_ = primals_in

  x, series = jet(lambda x, y: lax.mul(y, lax.log(x)), primals_in, series_in)

  u = [x] + series
  v = [u_ ** r_] + [None] * len(series)
  for k in range(1, len(v)):
    v[k] = fact(k-1) * sum(_scale(k, j) * v[k-j] * u[j] for j in range(1, k+1))
  primal_out, *series_out = v

  return primal_out, series_out
jet_rules[lax.pow_p] = _pow_taylor

def _pow_by_squaring(x, n):
  if n < 0:
    return _pow_by_squaring(1 / x, -n)
  elif n == 0:
    return 1
  elif n % 2 == 0:
    return _pow_by_squaring(x * x, n / 2)
  elif n % 2 == 1:
    return x * _pow_by_squaring(x * x, (n - 1) / 2)

def _integer_pow_taylor(primals_in, series_in, *, y):
  if y == 0:
    return jet(jnp.ones_like, primals_in, series_in)
  else:
    return jet(lambda x: _pow_by_squaring(x, y), primals_in, series_in)

jet_rules[lax.integer_pow_p] = _integer_pow_taylor


def _logistic_taylor(primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [x] + series
  v = [lax.logistic(x)] + [None] * len(series)
  e = [v[0] * (1 - v[0])] + [None] * len(series)  # terms for sigmoid' = sigmoid * (1 - sigmoid)
  for k in range(1, len(v)):
    v[k] = fact(k-1) * sum(_scale(k, j) * e[k-j] * u[j] for j in range(1, k+1))
    e[k] = (1 - v[0]) * v[k] - fact(k) * sum(_scale2(k, j) * v[j] * v[k-j] for j in range(1, k+1))

  primal_out, *series_out = v
  return primal_out, series_out

jet_rules[lax.logistic_p] = _logistic_taylor


def _tanh_taylor(primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [2*x] + [2 * series_ for series_ in series]
  primals_in, *series_in = u
  primal_out, series_out = _logistic_taylor((primals_in, ), (series_in, ))
  series_out = [2 * series_ for series_ in series_out]
  return 2 * primal_out - 1, series_out
jet_rules[lax.tanh_p] = _tanh_taylor

def _log_taylor(primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [x] + series
  v = [lax.log(x)] + [None] * len(series)
  for k in range(1, len(v)):
    conv = sum(_scale(k, j) * v[j] * u[k-j] for j in range(1, k))
    v[k] = (u[k] - fact(k - 1) * conv) / u[0]
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.log_p] = _log_taylor

def _atan2_taylor(primals_in, series_in):
  x, y = primals_in
  primal_out = lax.atan2(x, y)

  x, series = jet(lax.div, primals_in, series_in)
  one = lax_internal._const(x, 1)
  c0, cs = jet(lambda x: lax.div(one, 1 + lax.square(x)), (x, ), (series, ))
  c = [c0] + cs
  u = [x] + series
  v = [primal_out] + [None] * len(series)
  for k in range(1, len(v)):
    v[k] = fact(k-1) * sum(_scale(k, j) * c[k-j] * u[j] for j in range(1, k + 1))
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.atan2_p] = _atan2_taylor

def _div_taylor_rule(primals_in, series_in):
  x, y = primals_in
  x_terms, y_terms = series_in
  u = [x] + x_terms
  w = [y] + y_terms
  v = [None] * len(u)
  def scale(k, j): return 1. / (fact(k - j) * fact(j))
  for k in range(0, len(v)):
    conv = sum(scale(k, j) * v[j] * w[k-j] for j in range(0, k))
    v[k] = (u[k] - fact(k) * conv) / w[0]
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.div_p] = _div_taylor_rule

def _sinusoidal_rule(sign, prims, primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [x] + series
  s, c = prims
  s = [s(x)] + [None] * len(series)
  c = [c(x)] + [None] * len(series)
  for k in range(1, len(s)):
    s[k] = fact(k-1) * sum(_scale(k, j) * u[j] * c[k-j] for j in range(1, k + 1))
    c[k] = fact(k-1) * sum(_scale(k, j) * u[j] * s[k-j] for j in range(1, k + 1)) * sign
  return (s[0], s[1:]), (c[0], c[1:])

def _get_ind(f, ind):
  return lambda *args: f(*args)[ind]

jet_rules[lax.sin_p] = _get_ind(partial(_sinusoidal_rule, -1, (lax.sin, lax.cos)), 0)
jet_rules[lax.cos_p] = _get_ind(partial(_sinusoidal_rule, -1, (lax.sin, lax.cos)), 1)
jet_rules[lax.sinh_p] = _get_ind(partial(_sinusoidal_rule, 1, (lax.sinh, lax.cosh)), 0)
jet_rules[lax.cosh_p] = _get_ind(partial(_sinusoidal_rule, 1, (lax.sinh, lax.cosh)), 1)

def _bilinear_taylor_rule(prim, primals_in, series_in, **params):
  x, y = primals_in
  x_terms, y_terms = series_in
  u = [x] + x_terms
  w = [y] + y_terms
  v = [None] * len(u)
  op = partial(prim.bind, **params)
  def scale(k, j): return 1. / (fact(k - j) * fact(j))
  for k in range(0, len(v)):
    v[k] = fact(k) * sum(scale(k, j) * op(u[j], w[k-j]) for j in range(0, k+1))
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.dot_general_p] = partial(_bilinear_taylor_rule, lax.dot_general_p)
jet_rules[lax.mul_p] = partial(_bilinear_taylor_rule, lax.mul_p)
jet_rules[lax.conv_general_dilated_p] = partial(_bilinear_taylor_rule, lax.conv_general_dilated_p)

def _gather_taylor_rule(primals_in, series_in, **params):
  operand, start_indices = primals_in
  gs, _ = series_in
  primal_out = lax.gather_p.bind(operand, start_indices, **params)
  series_out = [lax.gather_p.bind(g, start_indices, **params) for g in gs]
  return primal_out, series_out
jet_rules[lax.gather_p] = _gather_taylor_rule

def _gen_reduce_choose_taylor_rule(chooser_fun):
  def chooser_taylor_rule(primals_in, series_in, **params):
    operand, = primals_in
    gs, = series_in
    primal_out = chooser_fun(operand, **params)
    axes = params.pop("axes", None)
    primal_dtype = gs[0].dtype
    shape = [1 if i in axes else d for i, d in enumerate(operand.shape)]
    location_indicators = lax.convert_element_type(
        lax_internal._eq_meet(operand, lax.reshape(primal_out, shape)),
        primal_dtype)
    counts = lax_internal._reduce_sum(location_indicators, axes)
    def _reduce_chooser_taylor_rule(g):
      return lax.div(
          lax_internal._reduce_sum(lax.mul(g, location_indicators), axes),
          counts)
    series_out = [_reduce_chooser_taylor_rule(g) for g in gs]
    return primal_out, series_out
  return chooser_taylor_rule
jet_rules[lax.reduce_max_p] = _gen_reduce_choose_taylor_rule(
    lax_internal._reduce_max)
jet_rules[lax.reduce_min_p] = _gen_reduce_choose_taylor_rule(
    lax_internal._reduce_min)

def _abs_taylor_rule(x, series_in, **params):
  x, = x
  zero = lax.full_like(x, 0, shape=())
  primal_out = lax.abs_p.bind(x, **params)
  negs = lax.select(lax.lt(x, zero), lax.full_like(x, -1), lax.full_like(x, 1.0))
  fix_sign = lambda y: negs * y
  series_out = [fix_sign(*terms_in, **params) for terms_in in zip(*series_in)]
  return primal_out, series_out
jet_rules[lax.abs_p] = _abs_taylor_rule

def _select_n_taylor_rule(primal_in, series_in, **params):
  b, *cases = primal_in
  primal_out = lax.select_n(b, *cases)
  sel = lambda _, *xs: lax.select_n(b, *xs)
  series_out = [sel(*terms_in) for terms_in in zip(*series_in)]
  return primal_out, series_out
jet_rules[lax.select_n_p] = _select_n_taylor_rule

def _lax_max_taylor_rule(primal_in, series_in):
    x, y = jnp.broadcast_arrays(*primal_in)

    xgy = x > y   # greater than mask
    xey = x == y  # equal to mask
    primal_out = lax.select(xgy, x, y)

    def select_max_and_avg_eq(x_i, y_i):
        """Select x where x>y or average when x==y"""
        max_i = lax.select(xgy, x_i, y_i)
        max_i = lax.select(xey, (x_i + y_i)/2, max_i)
        return max_i

    series_out = [select_max_and_avg_eq(*jnp.broadcast_arrays(*terms_in)) for terms_in in zip(*series_in)]
    return primal_out, series_out
jet_rules[lax.max_p] = _lax_max_taylor_rule

def _lax_min_taylor_rule(primal_in, series_in):
    x, y = primal_in
    xgy = x < y   # less than mask
    xey = x == y  # equal to mask
    primal_out = lax.select(xgy, x, y)

    def select_min_and_avg_eq(x_i, y_i):
        """Select x where x>y or average when x==y"""
        min_i = lax.select(xgy, x_i, y_i)
        min_i = lax.select(xey, (x_i + y_i)/2, min_i)
        return min_i

    series_out = [select_min_and_avg_eq(*terms_in) for terms_in in zip(*series_in)]
    return primal_out, series_out
jet_rules[lax.min_p] = _lax_min_taylor_rule

def _scatter_add_rule(primals_in, series_in, *, update_jaxpr, update_consts,
                      dimension_numbers, indices_are_sorted, unique_indices,
                      mode):
  bind = partial(lax.scatter_add_p.bind, update_jaxpr=update_jaxpr,
                 update_consts=update_consts, dimension_numbers=dimension_numbers,
                 indices_are_sorted=indices_are_sorted,
                 unique_indices=unique_indices, mode=mode)
  operand, scatter_indices, updates = primals_in
  primal_out = bind(operand, scatter_indices, updates)
  series_out = [bind(d1, scatter_indices, d2) for d1, _, d2 in zip(*series_in)]
  return primal_out, series_out
jet_rules[lax.scatter_add_p] = _scatter_add_rule


@weakref_lru_cache
def _jet_jaxpr(
    jaxpr: core.ClosedJaxpr, order: int, primals_and_series_avals, in_tree_def
) -> tuple[core.ClosedJaxpr, Any]:
  f = lu.wrap_init(core.jaxpr_as_fun(jaxpr))
  f_jet, out_tree_def = traceable(jet_fun(jet_subtrace(f), order), in_tree_def)
  jaxpr_jet, _, consts, () = pe.trace_to_jaxpr_dynamic(
      f_jet, primals_and_series_avals)
  return core.ClosedJaxpr(jaxpr_jet, consts), out_tree_def


def _pjit_jet_rule(primals_in, series_in, **params):
  primals_and_series, in_tree_def = tree_flatten((primals_in, series_in))
  order = len(series_in[0])
  primals_and_series_avals = tuple(core.shaped_abstractify(x) for x in primals_and_series)
  jaxpr_jet, out_tree_def = _jet_jaxpr(params['jaxpr'], order,
                                       primals_and_series_avals, in_tree_def)
  num_series_in = len(primals_in) * order
  num_series_out = len(params['out_shardings']) * order
  new_params = {
      **params,
      'jaxpr': jaxpr_jet,
      'in_shardings': (
          params['in_shardings'] + (sharding_impls.UNSPECIFIED,) * num_series_in
      ),
      'out_shardings': (
          params['out_shardings']
          + (sharding_impls.UNSPECIFIED,) * num_series_out
      ),
      'in_layouts': params['in_layouts'] + (None,) * num_series_in,
      'out_layouts': params['out_layouts'] + (None,) * num_series_out,
      'donated_invars': params['donated_invars'] + (False,) * num_series_in,
  }
  result = pjit.pjit_p.bind(*primals_and_series, **new_params)
  return tree_unflatten(out_tree_def(), result)

jet_rules[pjit.pjit_p] = _pjit_jet_rule
