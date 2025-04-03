# Copyright 2022 The JAX Authors.
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
"""Shape polymorphism support for deciding inequalities of symbolic dimensions.

"""

from __future__ import annotations

from collections.abc import Sequence
import itertools
import math

import numpy as np

from jax._src import core
from jax._src.export import shape_poly
from jax._src.export.shape_poly import (
  _DimExpr, _DimTerm, _DimFactor,
  SymbolicScope,
  DimSize,
  InconclusiveDimensionOperation,
  Comparator,
  BoundsPrecision,
)

def sgn(x): return 1 if x >= 0 else -1


def bounds_decision(e: DimSize,
                    prec: BoundsPrecision) -> tuple[float, float]:
  if not isinstance(e, _DimExpr):
    return (int(e), int(e))
  decision = _DecisionByElimination.build(e.scope)
  return decision.bounds(e, prec, add_implicit_constraints=True)

shape_poly._bounds_decision = bounds_decision


class _DecisionByElimination:
  """A decision procedure based on elimination of terms.

  Given an expression `e = t*t_k + rest_e` for which we want to compute bounds,
  and a constraint `c = t*t_c_k + rest_c >= 0`,

  Let `e0 = e*abs(t_c_k) - c*sgn(t_c_k)*t_k`. (Note that we eliminated `t` from
  `e0`, since `abs(t_c_k)*t_k = sgn(t_c_k)*t_k*t_c_k`.)

  Since `c >= 0`,
  if `sgn(t_c_k)*t_k > 0`:
     then `abs(t_c_k)*e >= e0`, hence, `LB(e) >= ceil(LB(e0) / abs(t_c_k))`,

  if `sgn(t_c_k)*t_k < 0`
    then `abs(t_c_k)*e <= e0`, hence, `UB(e) <= floor(UB(e0) / abs(t_c_k))`,

  See the implementation in self.combine_term_with_existing.
  Do not use the constructor directly, use the `build` static method.
  """
  def __init__(self, scope: SymbolicScope):
    self.scope = scope
    self._processed_for_internal_constraints: set[_DimTerm] = set()
    # The other fields are for keeping an efficient representation of
    # the explicit constraints.
    self._term_bounds: dict[_DimTerm, tuple[float, float]] = {}
    # The _expr_constraints represents a set of constraints that are not
    # just simple terms. The set is represented as a mapping from a
    # term "t" to tuples (cmp, k, c) where "c >= 0" (if cmp is GEQ else "c == 0")
    # represents a constraint that has "t" as the leading term with coefficient "k".
    self._expr_constraints: dict[_DimTerm, set[tuple[Comparator, int, _DimExpr]]] = {}

  def initialize(self) -> _DecisionByElimination:
    # Process the explicit constraints in the order in which the user specifies
    # them. This is because the heuristics depend on the order in which the
    # constraints are processed, and this way we give the user a way to control
    # the result (albeit, for now, without a good feedback loop to understand
    # how the order matters for inequalities).
    for constr in self.scope._explicit_constraints:
      if not core.is_constant_dim(constr.diff):
        self.add_implicit_constraints_expr(constr.diff)  # type: ignore

      self.combine_and_add_constraint(constr.cmp, constr.diff, 0,
                                      constr.debug_str)


      # Clear the cache, since we have added constraints.
      self.scope._bounds_cache.clear()
    return self

  @staticmethod
  def build(scope: SymbolicScope) -> _DecisionByElimination:
    """Builds an initialized DecisionByElimination for a scope.

    Caches the initial state of the decision procedure in the scope.
    """
    if not scope._initialized or not scope._explicit_constraints:
      # We do not cache until the scope is fully initialized.
      return _DecisionByElimination(scope).initialize()

    if not scope._decision_initial_state:
      scope._decision_initial_state = _DecisionByElimination(scope).initialize()
    d = scope._decision_initial_state
    # Return a copy, because the decision procedure state is mutable
    c = _DecisionByElimination(scope)
    c._processed_for_internal_constraints = d._processed_for_internal_constraints.copy()
    c._term_bounds = d._term_bounds.copy()
    c._expr_constraints = {
        lead_t: lead_t_constraints.copy()
        for lead_t, lead_t_constraints in d._expr_constraints.items()}
    return c

  def combine_and_add_constraint(self,
                                 cmp: Comparator,
                                 e1: _DimExpr | int | float,
                                 e2: _DimExpr | int | float,
                                 debug_str: str | None = None):
    """Adds a constraint "e1 >= e2" to the internal state."""
    if isinstance(e1, float):
      if np.isinf(e1) and e1 >= 0 and cmp == Comparator.GEQ: return
      assert e1 == np.floor(e1)
      e1 = int(e1)
    if isinstance(e2, float):
      if np.isinf(e2) and e2 <= 0 and cmp == Comparator.GEQ: return
      assert e2 == np.floor(e2)
      e2 = int(e2)
    e = e1 - e2
    if (const := _DimExpr._to_constant(e)) is not None:
      if const < 0:
        raise ValueError(f"Unsatisfiable constraint: {debug_str or str(e1) + ' >= ' + str(e2)}")
      return
    assert isinstance(e, _DimExpr)
    self.add_to_state(cmp, e, debug_str)
    geq_combinations = self.combine_constraint_with_existing(cmp, e, debug_str)
    for cmp, a in geq_combinations:
      self.add_to_state(cmp, a, None)

  def add_to_state(self,
                   cmp: Comparator,
                   e: _DimExpr,
                   debug_str: str | None):
    """Updates the internal state to reflect "e >= 0". """
    assert _DimExpr._to_constant(e) is None

    if (term_factors := e._to_single_term()) is not None:
      n, t_k, t = term_factors  # n + t * t_k [== | >=] 0
      lb, ub = self._term_bounds.get(t, (- np.inf, np.inf))
      if cmp == Comparator.EQ:
        # n + t_k * t == 0  ->  t == - n // t_k
        if n % t_k:
          raise ValueError(f"Unsatisfiable constraint: {debug_str}")
        t_val = - (n // t_k)
        lb = max(lb, t_val)
        ub = min(ub, t_val)
      else:  # GEQ
        if t_k > 0:
          lb = max(lb, int(np.ceil(- n / t_k)))
        else:
          ub = min(ub, int(np.floor(- n / t_k)))
      if lb > ub:
        raise ValueError(f"Unsatisfiable constraint: {debug_str}")

      self._term_bounds[t] = (lb, ub)
      return

    lead_t, lead_t_k = e._leading_term
    lead_t_constraints = self._expr_constraints.get(lead_t)
    if lead_t_constraints is None:
      lead_t_constraints = set()
      self._expr_constraints[lead_t] = lead_t_constraints
    lead_t_constraints.add((cmp, lead_t_k, e))

  def combine_term_with_existing(self, t: _DimTerm, t_k: int, *,
                                 scope: shape_poly.SymbolicScope,
                                 only_smaller_than_t=True,
                                 ) -> Sequence[tuple[Comparator,
                                                     _DimExpr,
                                                     int,
                                                     int]]:
    """
    Combine a term with existing constraints.
    For input (t, t_k) the tuple (c_eq, c, c_s, t_s) is among the returned
    tuples if there exists a constraint `c =[c_eq] 0` that can be combined
    with `t*t_k` to eliminate `t`, and:

      * `c =[c_eq] 0`
      * The term `comb = t*t_k*t_s + c*c_s` does not contain `t`, and if
        `only_smaller_than_t` then `comb` contains only terms structurally
         smaller than `t`.
      * `c_s > 0`
    """
    # TODO: maybe a generator is useful here instead of materializing the list
    acc: list[tuple[Comparator, _DimExpr, int, int]] = []
    # First combine with the existing term bounds
    t_lb, t_ub = self._term_bounds.get(t, (-np.inf, np.inf))
    if t_lb == t_ub:
      acc.append((Comparator.EQ, _DimExpr(((t, 1),), scope) - int(t_lb),
                  abs(t_k), - sgn(t_k)))
    else:
      if t_lb > -np.inf:
        acc.append((Comparator.GEQ, _DimExpr(((t, 1),), scope) - int(t_lb),
                    abs(t_k), - sgn(t_k)))
      if t_ub < np.inf:
        acc.append((Comparator.GEQ, _DimExpr(((t, -1),), scope) + int(t_ub),
                    abs(t_k), sgn(t_k)))

    prev_constraint: set[tuple[Comparator, int, _DimExpr]]
    for prev_constraint in ([self._expr_constraints.get(t, set())] if only_smaller_than_t
                            else self._expr_constraints.values()):
      for c_eq, _, c in prev_constraint:
        # TODO: optimize this dict()
        tc_k = dict(c._sorted_terms).get(t)
        if tc_k is not None:
          # c =[c_eq] 0 AND t*tc_k appears in c.
          c_s = abs(t_k)
          c_t = - tc_k * sgn(t_k)
          acc.append((c_eq, c, c_s, c_t))
    return acc

  def combine_constraint_with_existing(self,
                                       eq: Comparator,
                                       e: _DimExpr,
                                       debug_str: str | None) -> set[tuple[Comparator, _DimExpr]]:
    combinations: set[tuple[Comparator, _DimExpr]] = set()
    for t, t_k in e._sorted_terms:
      if t.is_constant: continue
      for (c_eq, c, c_s, t_s) in self.combine_term_with_existing(t, t_k,
                                                                 only_smaller_than_t=False,
                                                                 scope=e.scope):
        # c =[c_eq] 0 AND c_s > 0 AND t*t_k*t_s + c*c_s does not contain t
        if t_s > 0 or eq == Comparator.EQ:
          new_eq = Comparator.EQ if (eq == c_eq == Comparator.EQ) else Comparator.GEQ
          new_e = _DimExpr._linear_combination(e, t_s, c, c_s, e.scope)
          if (const := _DimExpr._to_constant(new_e)) is not None:
            if ((new_eq == Comparator.GEQ and const < 0) or
                (new_eq == Comparator.EQ and const != 0)):
              raise ValueError(f"Unsatisfiable constraints: {debug_str or str(e) + ' >= 0'}")
          else:
            combinations.add((new_eq, new_e))  # type: ignore
    return combinations

  def bounds(self, e: DimSize,
             prec: BoundsPrecision,
             add_implicit_constraints: bool = False
             ) -> tuple[float, float]:
    """Returns the lower and upper bounds, or -+inf.

    Args:
      e: the expression for which to compute the bounds.
      prec: the desired precision. See comments in `BoundsPrecision`.
      add_implicit_constraints: if True, then before computing the bounds
        add the implicit constraints for the terms inside `e`.
    """
    if (const := _DimExpr._to_constant(e)) is not None:
      return (const, const)
    assert isinstance(e, _DimExpr)
    # Caching bounds is tricky. Since the underlying _bounds_for_sorted_terms
    # is incomplete, and it may produce better results in the context of
    # specific queries (due to the implicit constraints), if we cache the
    # bounds computation we may stick to sub-optimal results. Also, we should
    # not use the precision as part of the cache key, because a certain result
    # may work for multiple precisions.
    if (res := self.scope._bounds_cache.get(e)) is not None:
      lb, ub, prev_prec = res
      if prec._bounds_are_sufficient(lb, ub): return (lb, ub)
      if prev_prec.value >= prec.value: return (lb, ub)

    if add_implicit_constraints:
      self.add_implicit_constraints_expr(e)
    lb, ub = self._bounds_for_sorted_terms(e.scope, e._sorted_terms, 0, prec)
    lb, ub = (int(lb) if lb > -np.inf else lb,
              int(ub) if ub < np.inf else ub)
    self.scope._bounds_cache[e] = (lb, ub, prec)
    return (lb, ub)

  def _bounds_for_sorted_terms(self,
                               scope: SymbolicScope,
                               e: Sequence[tuple[_DimTerm, int]],
                               i: int,
                               prec: BoundsPrecision) -> tuple[float, float]:
    """The lower and upper bounds of e[i:].

    See comments about soundness and `cmp_with` in the `shape_poly.bounds_decision`` method.
    Returns (lower-bound, upper-bound)
    """
    if i >= len(e): return (0, 0)

    t, t_k = e[i]
    if t.is_constant:
      assert i == len(e) - 1  # Must be last
      return (t_k, t_k)

    lb = -np.inf
    ub = np.inf

    for (c_eq, c, c_s, t_s) in self.combine_term_with_existing(t, t_k,
                                                               only_smaller_than_t=True,
                                                               scope=scope):
      # `c =[eq] 0` AND `t*t_k*t_s + c*c_s` contains only terms smaller than t
      # AND c_s > 0.
      # `rest = e[i:]*t_s + c*c_s` AND `rest_ub >= rest >= rest_lb`
      # `rest` contains only terms smaller than `t`.
      rest = _DimExpr._linear_combination_sorted_pairs(e, i, t_s,
                                                       c._sorted_terms, 0, c_s)
      rest_lb, rest_ub = self._bounds_for_sorted_terms(scope, rest, 0,
                                                       BoundsPrecision.BEST)
      if rest_ub < np.inf:
        # We have: e[i:]*t_s = rest - c*c_s <= rest_ub
        if t_s > 0:
          ub = min(ub, int(np.floor(rest_ub / t_s)))
        else:
          lb = max(lb, int(np.ceil(rest_ub / t_s)))

      if rest_lb > - np.inf and c_eq == Comparator.EQ:
        # We have: e[i:]*t_s = rest - c*c_s = rest >= rest_lb
        if t_s > 0:
          lb = max(lb, int(np.ceil(rest_lb / t_s)))
        else:
          ub = min(ub, int(np.floor(rest_lb / t_s)))

      if prec._bounds_are_sufficient(lb, ub): return (lb, ub)

    # Now look for special rules for factors
    if (t_f := t.to_factor()) is not None:
      if t_f.operation in [_DimFactor.MAX, _DimFactor.MIN]:
        # m_c*MAX(op1, op2) + rest_e >= max(m_c * op1 + rest_e, m_c * op2 + rest_e)
        #   if m_c > 0. Similar rules for when m_c < 0 and for MIN.
        op1, op2 = t_f.operands
        rest1 = _DimExpr._linear_combination_sorted_pairs(e, i + 1, 1,
                                                          op1._sorted_terms, 0, t_k)
        rest2 = _DimExpr._linear_combination_sorted_pairs(e, i + 1, 1,
                                                          op2._sorted_terms, 0, t_k)
        rest1_lb, rest1_ub = self._bounds_for_sorted_terms(scope, rest1, 0,
                                                           BoundsPrecision.BEST)
        rest2_lb, rest2_ub = self._bounds_for_sorted_terms(scope, rest2, 0,
                                                           BoundsPrecision.BEST)
        like_max = (t_k > 0 if t_f.operation == _DimFactor.MAX else t_k < 0)
        if like_max:
          lb = max(lb, max(rest1_lb, rest2_lb))
          ub = min(ub, max(rest1_ub, rest2_ub))
        else:
          lb = max(lb, min(rest1_lb, rest2_lb))
          ub = min(ub, min(rest1_ub, rest2_ub))
        if prec._bounds_are_sufficient(lb, ub, ): return (lb, ub)

    return lb, ub

  def add_implicit_constraints_expr(self, e: _DimExpr):
    """Adds the implicit constraints for the expression `e`"""
    for t, _ in e._sorted_terms:
      if t.is_constant: continue
      self.add_implicit_constraints_term(t)

  def add_implicit_constraints_term(self, t: _DimTerm):
    if t in self._processed_for_internal_constraints: return
    self._processed_for_internal_constraints.add(t)
    t_e = _DimExpr._from_term(t, 1, self.scope)  # m as a _DimExpr
    f = t.to_factor()
    if f is None:
      # This is a multiplication of factors. Try to compute bounds based on
      # the bounds of the factors.
      bounds = []
      for f1, f1_exp in t._factors:
        f1_t = _DimTerm.from_factor(f1, 1)
        f1_e = _DimExpr._from_term(f1_t, 1, self.scope)
        self.add_implicit_constraints_term(f1_t)
        a1_l, a1_u = self.bounds(f1_e, BoundsPrecision.BEST)
        assert a1_l <= a1_u
        bounds.append((a1_l ** f1_exp, a1_u ** f1_exp))

      candidate_bounds = [math.prod(factor_bounds)
                          for factor_bounds in itertools.product(*bounds)]
      m_l = min(*candidate_bounds)
      m_u = max(*candidate_bounds)
      self.combine_and_add_constraint(Comparator.GEQ, t_e, m_l)
      self.combine_and_add_constraint(Comparator.GEQ, m_u, t_e)
      return

    # It is a factor, is it a variable?
    if (v := f.to_var()) is not None:
      self.combine_and_add_constraint(Comparator.GEQ, t_e, 1)  # v >= 1
      return

    for oper in f.operands:
      self.add_implicit_constraints_expr(oper)

    if f.operation == _DimFactor.MOD:
      op1, op2 = f.operands
      op2_b_l, op2_b_u = self.bounds(op2, BoundsPrecision.FOR_GEQ0_OR_LT0)
      if op2_b_l > 0:  # positive divisor
        self.combine_and_add_constraint(Comparator.GEQ, t_e, 0)  # m >= 0
        self.combine_and_add_constraint(Comparator.GEQ, op2 - 1, t_e)  # m <= op2 - 1
        self.combine_and_add_constraint(Comparator.GEQ, op2_b_u - 1, t_e)
      elif op2_b_u < 0:  # negative divisor
        self.combine_and_add_constraint(Comparator.GEQ, t_e, op2 + 1)  # m >= op2 + 1
        self.combine_and_add_constraint(Comparator.GEQ, t_e, op2_b_l + 1)
        self.combine_and_add_constraint(Comparator.GEQ, 0, t_e)  # m <= 0
      return

    if f.operation == _DimFactor.FLOORDIV:
      op1, op2 = f.operands
      (op1_l, op1_u) = self.bounds(op1, BoundsPrecision.BEST)
      (op2_l, op2_u) = self.bounds(op2, BoundsPrecision.BEST)

      def math_floor_with_inf(a: float, b: float):
        # math.floor(a / b), but aware of inf.
        # When either a or b are infinite, the result represents the limit
        # of "a // b".
        assert b != 0  # we caught division by 0 earlier
        if not np.isinf(b):  # divisor b is finite
          if not np.isinf(a):  # both dividend a and divisor b are finite
            return math.floor(a / b)
          # a is infinite, b is finite
          return -np.inf if (a >= 0) != (b >= 0) else np.inf
        elif not np.isinf(a):  # dividend a is finite and divisor b is infinite
          return -1 if (a >= 0) != (b >= 0) else 0
        else:  # both dividend and divisor are infinite
          return -np.inf if (a >= 0) != (b >= 0) else np.inf

      # Same reasoning as for multiplication: the bounds are among the cross-product
      # of the bounds.
      if op2_l <= 0 <= op2_u:
        raise InconclusiveDimensionOperation(
            f"Possible division by 0 in division by {op2}")
      candidate_bounds = [math_floor_with_inf(op1_l, op2_l),
                          math_floor_with_inf(op1_l, op2_u),
                          math_floor_with_inf(op1_u, op2_l),
                          math_floor_with_inf(op1_u, op2_u)]
      m_l = min(*candidate_bounds)
      m_u = max(*candidate_bounds)
      self.combine_and_add_constraint(Comparator.GEQ, t_e, m_l)
      self.combine_and_add_constraint(Comparator.GEQ, m_u, t_e)
      if op2_l >= 0:
        if op1_l >= 0:
          self.combine_and_add_constraint(Comparator.GEQ, t_e, 0)
        mod_e = _DimExpr._from_operation(_DimFactor.MOD, op1, op2,
                                         scope=self.scope)
        if isinstance(mod_e, _DimExpr):
          self.add_implicit_constraints_expr(mod_e)
        combined = op2 * t_e + mod_e
        self.combine_and_add_constraint(Comparator.EQ, op1, combined)
      return

    if f.operation == _DimFactor.MAX:
      op1, op2 = f.operands
      op1_b_l, op1_b_u = self.bounds(op1, BoundsPrecision.BEST)
      op2_b_l, op2_b_u = self.bounds(op2, BoundsPrecision.BEST)
      self.combine_and_add_constraint(Comparator.GEQ, t_e, max(op1_b_l, op2_b_l))
      self.combine_and_add_constraint(Comparator.GEQ, max(op1_b_u, op2_b_u), t_e)
      self.combine_and_add_constraint(Comparator.GEQ, t_e, op1)
      self.combine_and_add_constraint(Comparator.GEQ, t_e, op2)
      return

    if f.operation == _DimFactor.MIN:
      op1, op2 = f.operands
      op1_b_l, op1_b_u = self.bounds(op1, BoundsPrecision.BEST)
      op2_b_l, op2_b_u = self.bounds(op2, BoundsPrecision.BEST)
      self.combine_and_add_constraint(Comparator.GEQ, t_e, min(op1_b_l, op2_b_l))
      self.combine_and_add_constraint(Comparator.GEQ, min(op1_b_u, op2_b_u), t_e)
      self.combine_and_add_constraint(Comparator.GEQ, op1, t_e)
      self.combine_and_add_constraint(Comparator.GEQ, op2, t_e)
      return
