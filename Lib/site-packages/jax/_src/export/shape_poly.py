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
"""Shape polymorphism support.

See documentation at https://jax.readthedocs.io/en/latest/export/shape_poly.html.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
import dataclasses
from enum import Enum
import functools
import itertools
import io
import copy
import operator as op
import tokenize
from typing import Any, Union, overload
import warnings

import numpy as np
import opt_einsum

import jax

from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import effects
from jax._src.lax import lax
from jax._src.interpreters import mlir
from jax._src.numpy import lax_numpy
from jax._src import source_info_util
from jax._src import tree_util
from jax._src import util


DimSize = Union["_DimExpr", int]
TfVal = Any
DimVarEnv = dict[str, jax.Array]
DType = Any

# Tuples of terms and their coefficients, sorted with the largest term first.
SortedTerms = Sequence[tuple["_DimTerm", int]]
SortedFactors = Sequence[tuple["_DimFactor", int]]

# Normalization rules represent the explicit constraint `t*tk == e` as
# a mapping of `t` to `(e, tk)`.
NormalizationRules = dict["_DimTerm", tuple["_DimExpr", int]]


class InconclusiveDimensionOperation(core.InconclusiveDimensionOperation):
  """Raised when we cannot conclusively compute with symbolic dimensions."""

  _help_msg = """
This error arises for comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a boolean value for all values of the symbolic dimensions involved.

Please see https://jax.readthedocs.io/en/latest/export/shape_poly.html#comparison-of-symbolic-dimensions-is-partially-supported
for more details.
"""

  def __init__(self, message: str):
    error_msg = f"{message}{InconclusiveDimensionOperation._help_msg}"
    # https://github.com/python/mypy/issues/5887
    super().__init__(error_msg)

class UnexpectedDimVar(Exception):
  pass

class Comparator(Enum):
  EQ = 1
  GEQ = 2

@dataclasses.dataclass(frozen=True)
class _SymbolicConstraint:
  # Either e1 == e2 if cmp == Comparator.EQ else e1 >= e2
  cmp: Comparator
  debug_str: str  # The form in which the user expressed it, for error messages
  # e1, e2, and diff == e1 - e2, are normalized w.r.t. previous constraints only
  e1: DimSize
  e2: DimSize
  # we pre-compute diff to avoid having the normalization rule kick in later.
  diff: DimSize

  def __repr__(self):
    return f"Constraint({self.debug_str})"


class _DimFactor:
  """Represents a factor in a symbolic dimension expression.

  Factors are either variables, or expressions of the form floordiv(E1, E2) or
  mod(E1, E2), or max(E1, E2), or min(E1, E2).
  Factors are multiplied to form terms (see _DimTerm), and
  terms are added to form symbolic expressions (see _DimExpr).

  Args:
    * var: if specified then the factor is a dimension variable. `operation`
      must be `None`.
    * operation: if specified then the factor is an operation applied to
      `operands`. One of `FLOORDIR` or `MOD` or `MAX` or `MIN`.
      `var` must be `None`
    * operands: the operands to which the operation is applied.
  """
  # The supported operations
  # FLOORDIV(e1, e2) and MOD(e1, e2) have the same semantics as in Python:
  # FLOORDIV(e1, e2) = e1 // e2 = floor(e1 / e2)
  # if e2 > 0 then 0 <= MOD(e1, e2) < e2
  # if e2 < 0 then e2 < MOD(e1, e2) <= 0
  # e1 = e2 * FLOORDIV(e1, e2) + MOD(e1, e2)
  #
  FLOORDIV = "floordiv"
  MOD = "mod"
  MAX = "max"
  MIN = "min"

  __slots__ = ["var", "operation", "operands", "_hash", "_size"]

  def __init__(self, *operands: _DimExpr,
               var: str | None = None,
               operation: str | None = None):
    if var is not None:
      assert operation is None
      assert not operands
    else:
      assert operation is not None
    self.var = var
    self.operation = operation
    self.operands = operands
    self._hash = None
    self._size: int = 1 if var is not None else 1 + sum(o._size for o in operands)

  @staticmethod
  def from_var(v: str) -> _DimFactor:
    return _DimFactor(var=v)

  @staticmethod
  def from_operation(operation: str, *operands: DimSize,
                     scope: SymbolicScope) -> _DimFactor:
    return _DimFactor(*(_ensure_poly(o, operation, scope) for o in operands),
                      operation=operation)

  def to_var(self) -> str | None:
    return self.var

  def get_vars(self) -> set[str]:
    # All the vars that appear
    if self.var is not None:
      return {self.var}
    else:
      acc = set()
      for opnd in self.operands:
        acc.update(opnd._get_vars())
      return acc

  def __str__(self):
    if self.var is not None:
      return self.var
    opnd_str = ", ".join([str(opnd) for opnd in self.operands])
    return f"{self.operation}({opnd_str})"
  __repr__ = __str__

  def __hash__(self):
    if self._hash is None:
      self._hash = hash((self.var, self.operation, *self.operands))
    return self._hash

  def _syntactic_cmp(self, other: _DimFactor) -> int:
    """Returns -1 if self < other, 0 if self == other, 1 if self > other.
    The comparison is done lexicographically (syntactic), to be used for sorting.
    The result is not related to the semantic value.
    """
    if c := cmp_comparable(self._size, other._size): return c
    if self.var is not None:
      return cmp_comparable(self.var, other.var)
    if c := cmp_comparable(self.operation, other.operation): return c
    return cmp_sequence(self.operands, other.operands,
                        lambda s_o, o_o: s_o._syntactic_cmp(o_o))

  def __eq__(self, other: Any):
    """Lexicographic comparison."""
    if not isinstance(other, _DimFactor): return False
    return self._syntactic_cmp(other) == 0

  def __lt__(self, other: _DimFactor):
    """Lexicographic comparison."""
    return self._syntactic_cmp(other) < 0

  def __le__(self, other: _DimFactor):
    """Lexicographic comparison."""
    return self._syntactic_cmp(other) <= 0

  def __gt__(self, other: _DimFactor):
    """Lexicographic comparison."""
    return self._syntactic_cmp(other) > 0

  def __ge__(self, other: _DimFactor):
    """Lexicographic comparison"""
    return self._syntactic_cmp(other) >= 0

  def evaluate(self, env: DimVarEnv, scope: SymbolicScope):
    if self.var is not None:
      try:
        return env[self.var]
      except KeyError:
        # Perhaps there is a normalization rule for this variable
        normalized_var = _DimExpr._from_var(self.var, scope)
        if core.is_constant_dim(normalized_var):
          return normalized_var
        non_trivial_normalization = (v1 := normalized_var._to_var()) is None or v1 != self.var  # type: ignore
        if non_trivial_normalization:
          return normalized_var._evaluate(env)  # type: ignore
        err_msg = (
            f"Encountered dimension variable '{self.var}' that is not appearing in the shapes of the function arguments.\n"
            "Please see https://jax.readthedocs.io/en/latest/export/shape_poly.html#dimension-variables-must-be-solvable-from-the-input-shapes for more details.")
        raise UnexpectedDimVar(err_msg)
    else:
      operand_values = [opnd._evaluate(env) for opnd in self.operands]
      if self.operation == _DimFactor.FLOORDIV:
        return divmod(*operand_values)[0]  # type: ignore
      elif self.operation == _DimFactor.MOD:
        return divmod(*operand_values)[1]  # type: ignore
      elif self.operation == _DimFactor.MAX:
        op1, op2 = operand_values
        if core.is_constant_dim(op1) and core.is_constant_dim(op2):
          return max(op1, op2)
        if core.is_symbolic_dim(op1) or core.is_symbolic_dim(op2):
          return core.max_dim(op1, op2)
        # In the context of `evaluate` dimension variables may be mapped to
        # JAX Tracers.
        return lax.max(op1, op2)
      elif self.operation == _DimFactor.MIN:
        op1, op2 = operand_values
        if core.is_constant_dim(op1) and core.is_constant_dim(op2):
          return min(op1, op2)
        if core.is_symbolic_dim(op1) or core.is_symbolic_dim(op2):
          return core.min_dim(op1, op2)
        # In the context of `evaluate` dimension variables may be mapped to
        # JAX Tracers.
        return lax.min(op1, op2)
      else:
        assert False, self.operation

  def __deepcopy__(self, memo):
    return _DimFactor(*copy.deepcopy(self.operands, memo),
                      var=copy.deepcopy(self.var, memo),
                      operation=copy.deepcopy(self.operation, memo))


class _DimTerm:
  """Represents a multiplication of factors.

  The representation is a sequence of _DimFactor factors along with their
  integer exponents (>= 1). The empty sequence represents the constant 1.
  """
  __slots__ = ["_factors", "_hash", "_size"]
  def __init__(self, sorted_factors: SortedFactors):
    self._factors = sorted_factors
    self._hash = None
    self._size = sum((1 + f_exp * f._size) for f, f_exp in self._factors)

  def __hash__(self):
    if self._hash is None:
      self._hash = hash(tuple(self._factors))
    return self._hash

  def __str__(self):
    return "*".join(f"{fact}^{exponent}" if exponent != 1 else str(fact)
                    for fact, exponent in sorted(self._factors))

  __repr__ = __str__

  @staticmethod
  def from_var(v: str) -> _DimTerm:
    return _DimTerm(((_DimFactor.from_var(v), 1),))

  @staticmethod
  def from_factor(f: _DimFactor, f_exp: int):
    return _DimTerm(((f, f_exp),))

  @staticmethod
  def from_operation(operation: str, *operands: DimSize,
                     scope: SymbolicScope) -> _DimTerm:
    return _DimTerm(((_DimFactor.from_operation(operation, *operands,
                                                scope=scope), 1),))

  def to_var(self) -> str | None:
    """Extract the variable name from a term.
     Return None if the term is not a single variable."""
    a = self.to_factor()
    return a.to_var() if a is not None else None

  def to_factor(self) -> _DimFactor | None:
    """Extract the single factor from a term.
     Return None if the term is not a single factor."""
    if len(self._factors) > 1: return None
    (f, f_exp), = self._factors
    if f_exp != 1: return None
    return f

  def get_vars(self) -> set[str]:
    # All the vars that appear in the term.
    acc = set()
    for (f, _) in self._factors:
      acc.update(f.get_vars())
    return acc

  @property
  def is_constant(self):
    return not self._factors

  def _syntactic_cmp(self, other: _DimTerm) -> int:
    """Returns -1 if self < other, 0 if self == other, 1 if self > other.
    The comparison is done lexicographically (syntactic), to be used for sorting.
    The result is not related to the semantic value.
    """
    if c := cmp_comparable(self._size, other._size): return c
    def cmp_factor(s_f: tuple[_DimFactor, int], o_f: tuple[_DimFactor, int]) -> int:
      if c := s_f[0]._syntactic_cmp(o_f[0]): return c
      # Consider the terms with exponents to be expanded as multiplications.
      # Then a higher exponent for a "large" factor should lead to a "larger" term.
      return cmp_comparable(s_f[1], o_f[1])

    return cmp_sequence(self._factors, other._factors, cmp_factor)

  def __lt__(self, other: _DimTerm):
    """Lexicographic comparison"""
    return self._syntactic_cmp(other) < 0

  def __le__(self, other: _DimTerm):
    """Lexicographic comparison"""
    return self._syntactic_cmp(other) <= 0

  def __gt__(self, other: _DimTerm):
    """Lexicographic comparison"""
    return self._syntactic_cmp(other) > 0

  def __ge__(self, other: _DimTerm):
    """Lexicographic comparison"""
    return self._syntactic_cmp(other) >= 0

  def __eq__(self, other) -> bool:
    if not isinstance(other, _DimTerm): return False
    return self._syntactic_cmp(other) == 0

  def __ne__(self, other) -> bool:
    return not (self == other)

  def mul(self, other: _DimTerm) -> _DimTerm:
    """
    Returns the product with another term. Example: (n^2*m) * n == n^3 * m.
    """
    return _DimTerm(_DimExpr._linear_combination_sorted_pairs(self._factors, 0, 1,
                                                              other._factors, 0, 1))

  def divide(self, divisor: _DimTerm) -> _DimTerm:
    """
    Divides by another term. Raises a InconclusiveDimensionOperation
    if the result is not a term.
    For example, (n^3 * m) // n == n^2*m, but n // m fails.
    """
    new_factors = _DimExpr._linear_combination_sorted_pairs(self._factors, 0, 1,
                                                            divisor._factors, 0, -1)
    for _, f_exp in new_factors:
      if f_exp <= 0:
        raise InconclusiveDimensionOperation(f"Cannot divide {self} by {divisor}.")
    return _DimTerm(new_factors)

  def evaluate(self, env: DimVarEnv, scope: SymbolicScope):
    prod = lambda xs: functools.reduce(_evaluate_multiply, xs) if xs else core.dim_constant(1)
    def pow_opt(v, p: int):
      return v if p == 1 else prod([v] * p)
    return prod([pow_opt(f.evaluate(env, scope), exp) for f, exp in self._factors])

  def __deepcopy__(self, memo):
    return _DimTerm(copy.deepcopy(self._factors, memo))

# The constant 1, as a term.
_DimTerm_one = _DimTerm(())


class _DimExpr:
  """Symbolic expressions using dimension variables.

  A dimension expression is an addition of terms (_DimTerm), which themselves
  are products of factors (_DimFactor).

  The representation of a _DimExpr is as sequence of pairs `(term, coeff)`,
  representing the linear combination of terms with the given coefficients.
  The sequence is sorted by lexicographic (syntactic) ordering of `_DimTerm`,
  with the largest terms first. The special term `_DimTerm_one` is mapped
  to the free integer coefficient of the expression.

  We overload integer operations, but we do that soundly, raising
  :class:`InconclusiveDimensionOperation` when the result is not
  representable as a _DimExpr.
  """
  __array_priority__ = 1000   # Same as tracer, for __radd__ and others on ndarray
  __slots__ = ("_sorted_terms", "_scope", "_hash", "_size")
  def __init__(self, sorted_terms: SortedTerms,
               scope: SymbolicScope):
    # Do not construct _DimExpr directly, unless you are sure that `terms` is
    # normalized; Use _DimExpr._normalize_sorted_terms.
    self._sorted_terms = tuple(sorted_terms) or ((_DimTerm_one, 0),)
    self._scope = scope
    self._hash = None
    # _size speeds up _syntactic_cmp, which is used a lot for hashing.
    self._size = sum((1 + abs(m_count) * m._size)
                     for m, m_count in self._sorted_terms)

  @property
  def scope(self):
    # We make the expression scope visible, but read-only.
    return self._scope

  @staticmethod
  def _coeff_to_sorted_terms(coeffs: dict[_DimTerm, int]) -> SortedTerms:
    return sorted((p for p in coeffs.items() if p[1] != 0), reverse=True)

  @staticmethod
  def _from_term(t: _DimTerm, t_k: int, scope: SymbolicScope) -> DimSize:
    return _DimExpr._normalize_sorted_terms(((t, t_k),), scope)

  @staticmethod
  def _from_var(v: str, scope: SymbolicScope) -> DimSize:
    return _DimExpr._normalize_sorted_terms(((_DimTerm.from_var(v), 1),), scope)

  @staticmethod
  def _from_operation(operation: str, *operands: DimSize,
                      scope: SymbolicScope) -> DimSize:
    return _DimExpr._from_term(
        _DimTerm.from_operation(operation, *operands, scope=scope), 1,
        scope=scope)

  @property
  def _leading_term(self) -> tuple[_DimTerm, int]:
    """Returns the highest degree term that comes last lexicographically."""
    return self._sorted_terms[0]

  def _to_single_term(self) -> tuple[int, int, _DimTerm] | None:
    """Extracts the single term: k + c * term.
    Returns None if the expression is not a single term, or (k, c, term)
    """
    n1 = 0
    n2 = 0
    term = None
    for t, t_k in self._sorted_terms:
      if t.is_constant:
        n1 = t_k
        continue
      if term is None:
        term = t
        n2 = t_k
        continue
      return None
    assert term is not None
    return (n1, n2, term)

  @staticmethod
  def _add_coeff(coeffs: dict[_DimTerm, int], t: _DimTerm, coeff: int):
    """coeffs[t] += coeff, with squashing 0 coefficients."""
    if coeff == 0: return
    coeffs[t] = coeffs.get(t, 0) + coeff

  @staticmethod
  def _normalize_term(t: _DimTerm, t_k: int,
                      scope: SymbolicScope) -> Sequence[tuple[_DimTerm, int]]:
    # If (t, t_k) is among the scope normalization rules, then return
    # a list of `term * coefficient` to add to the expression containing (t, t_k).
    # Returns the empty sequence if no normalizations are necessary.
    if not scope._normalization_rules: return []
    updates = []
    after, t_k_after = scope._normalization_rules.get(t, (None, 0))
    if after is not None and t_k % t_k_after == 0:
      # We have t*t_k_after -> after.
      # We subtract `t*t_k` and add `after * (t_k // t_k_after)`.
      updates.append((t, - t_k))
      updates.extend((t2, tc2 * (t_k // t_k_after))
                     for t2, tc2 in after._sorted_terms)
      return updates

    if len(t._factors) <= 1:
      return updates

    # A product of factors; look up individually
    for f, fexp in t._factors:
      f_after, f_k_after = scope._normalization_rules.get(_DimTerm(((f, fexp),)), (None, 0))
      if f_after is not None and t_k % f_k_after == 0:
        # We subtract `t*t_k`.
        updates.append((t, - t_k))
        # And add `(t // f**fexp) * f_after * (t_k // f_k_after)`
        t_without_f = t.divide(_DimTerm(((f, fexp),)))
        updates.extend((t2.mul(t_without_f), tc2 * (t_k // f_k_after))
                       for t2, tc2 in f_after._sorted_terms)
        return updates
    return updates

  @staticmethod
  def _normalize_sorted_terms(terms: SortedTerms,
                              scope: SymbolicScope) -> DimSize:
    """Constructs a _DimExpr in normal form from sorted terms.

    Ensures that the symbolic dimension is normalized, e.g., does not
    have terms with coefficient 0, it reflects all the scope
    normalization_rules, and it is represented as a Python integer if it is
    known to be a constant.

    Does not attempt to normalize the keys (terms) inside `terms`.
    """
    for t, t_k in terms:
      assert t_k != 0
      if updates := _DimExpr._normalize_term(t, t_k, scope):
        coeffs = dict(terms)
        for t1, t1_k in updates:
          _DimExpr._add_coeff(coeffs, t1, t1_k)
        terms = _DimExpr._coeff_to_sorted_terms(coeffs)
        # TODO: check the case when we need to apply multiple normalizations
        break

    if not terms: return 0
    if terms[0][0].is_constant: return terms[0][1]
    return _DimExpr(terms, scope)

  def _to_term(self) -> _DimTerm | None:
    """Extract the single term from a symbolic expression.
    Returns None if the expression is not a single term."""
    if len(self._sorted_terms) > 1: return None
    (t, t_k), = self._sorted_terms
    return t if t_k == 1 else None

  def _to_factor(self) -> _DimFactor | None:
    """Extract the factor from a symbolic expression.
    Returns None if the expression is not a single factor."""
    t = self._to_term()
    return t.to_factor() if t is not None else None

  def _to_var(self) -> str | None:
    """Extract the variable name from a symbolic expression.
    Returns None if the expression is not a single variable."""
    mon = self._to_factor()
    return mon.to_var() if mon is not None else None

  @staticmethod
  def _to_constant(e: DimSize) -> int | None:
    """Extract the constant from a symbolic expression.
    Returns None if the expression is not a single constant."""
    if not isinstance(e, _DimExpr):
      return int(e)
    m, m_c = e._leading_term
    return m_c if m.is_constant else None

  @property
  def _is_constant(self):
    return _DimExpr._to_constant(self) is not None

  def _get_vars(self) -> set[str]:
    """The variables that appear in a symbolic dimension."""
    acc = set()
    for mon, _ in self._sorted_terms:
      acc.update(mon.get_vars())
    return acc

  # There are some uses already of `get_vars`, we keep it a while longer
  # for backwards compatibility.
  get_vars = _get_vars

  @overload
  @staticmethod
  def _linear_combination_sorted_pairs(
      e1: SortedTerms, i1: int, f1: int,
      e2: SortedTerms, i2: int, f2: int) -> SortedTerms:  ...  # type: ignore[bad-return-type,unused-ignore]

  @overload
  @staticmethod
  def _linear_combination_sorted_pairs(
      e1: SortedFactors, i1: int, f1: int,
      e2: SortedFactors, i2: int, f2: int) -> SortedFactors:  ...  # type: ignore[bad-return-type,unused-ignore]

  @staticmethod
  def _linear_combination_sorted_pairs(
      pairs1, i1, f1,
      pairs2, i2, f2):
    """Computes e1[i1:] * f1 + e2[i2:] * f2.

    e1, e2, and the result are sorted with largest term first.
    This is an optimization for a common operation. The unoptimized code would
    compute each subexpression in turn. This works for both SortedTerms and SortedFactors.
    """
    len1 = len(pairs1)
    len2 = len(pairs2)
    acc = []
    while i1 < len1 and i2 < len2:
      m1, m1_c = pairs1[i1]
      m2, m2_c = pairs2[i2]
      cmp = m1._syntactic_cmp(m2)  # Pick the largest term
      if cmp < 0:
        acc.append((m2, m2_c * f2))
        i2 += 1
      elif cmp > 0:
        acc.append((m1, m1_c * f1))
        i1 += 1
      else:  # They are equal, combine them
        i1 += 1
        i2 += 1
        m1_c = m1_c * f1 + m2_c * f2
        if m1_c == 0: continue
        acc.append((m1, m1_c))

    if i1 < len1:
      acc.extend((m1, m1_c * f1) for m1, m1_c in itertools.islice(pairs1, i1, len1) if m1_c != 0)
    if i2 < len2:
      acc.extend((m2, m2_c * f2) for m2, m2_c in itertools.islice(pairs2, i2, len2) if m2_c != 0)
    return acc

  def _syntactic_cmp(self, other: _DimExpr) -> int:
    """Returns -1 if self < other, 0 if self == other, 1 if self > other.
    The comparison is done lexicographically (syntactic), to be used for sorting.
    The result is not related to the semantic value.
    """
    s_terms = self._sorted_terms
    o_terms = other._sorted_terms
    if c := cmp_comparable(self._size, other._size): return c
    def cmp_factor(s_f: tuple[_DimTerm, int], o_f: tuple[_DimTerm, int]) -> int:
      if c := s_f[0]._syntactic_cmp(o_f[0]): return c
      return cmp_comparable(s_f[1], o_f[1])
    return cmp_sequence(s_terms, o_terms, cmp_factor)

  def _eq(self, other: _DimExpr) -> bool:
    # Equality is used very frequently because expressions are cached. We could
    # implement a more precise version based on `(self - other).bounds() = (0, 0)`
    # but that would be too expensive. It would also have the unfortunate drawback
    # that we cannot then cache `e.bounds()` because hashing invokes equality
    # which would lead to infinite recursion.
    diff = self - other

    # We look for `self - other == k`, and we rely on the fact that when we
    # normalize _DimExpr that represent integers as ints.
    if is_symbolic_dim(diff):
      # Here we really ought to raise InconclusiveDimensionOperation, but __eq__
      # cannot raise exceptions, because it is used indirectly when hashing.
      # So, we say that the expressions are disequal, which is really unsound.
      # See https://jax.readthedocs.io/en/latest/export/shape_poly.html#comparison-of-symbolic-dimensions-is-partially-supported
      return False

    return diff == 0

  def __hash__(self):
    if self._hash is None:
      self._hash = hash((self._sorted_terms, self.scope))
    return self._hash

  def __str__(self):
    def _one_term(t, t_k):
      abs_t_k = abs(t_k)
      sgn_t_k = "+" if t_k > 0 else "-"
      if t.is_constant:
        return f"{sgn_t_k} {abs_t_k}" if abs_t_k != 0 else "0"
      if abs_t_k == 1:
        return f"{sgn_t_k} {t}"
      return f"{sgn_t_k} {abs_t_k}*{t}"
    # We print first the "larger" terms, so that the constant is last.
    res = " ".join(_one_term(t, t_k)
                   for t, t_k in self._sorted_terms)
    if res.startswith("+ "):
      res = res[2:]
    return res

  def __repr__(self):
    return str(self)

  # A special case for linear combinations because they are common
  @staticmethod
  def _linear_combination(e1: DimSize, k1: int,
                          e2: DimSize, k2: int,
                          scope: SymbolicScope) -> DimSize:
    """Computes and normalizes `e1 * k1 + e2 * k2`"""
    if isinstance(e1, _DimExpr):
      e1_terms = e1._sorted_terms
      if isinstance(e2, _DimExpr):
        e1.scope._check_same_scope(e2, when="for linear combination")
    else:
      if not isinstance(e2, _DimExpr):
        return e1 * k1 + e2 * k2  # Constants
      e1_terms = ((_DimTerm_one, op.index(e1)),)
    if isinstance(e2, _DimExpr):
      e2_terms = e2._sorted_terms
    elif e2 == 0:
      e2_terms = ()
    else:
      e2_terms = ((_DimTerm_one, op.index(e2)),)
    new_terms = _DimExpr._linear_combination_sorted_pairs(e1_terms, 0, k1,
                                                          e2_terms, 0, k2)
    return _DimExpr._normalize_sorted_terms(new_terms, scope)

  # We overload +, -, *, because they are fully defined for _DimExpr.
  def __add__(self, other):
    if isinstance(other, core.Tracer) or not _convertible_to_poly(other):
      return self.__jax_array__().__add__(other)
    if isinstance(other, int) and other == 0: return self
    return _DimExpr._linear_combination(self, 1, other, 1, self.scope)

  def __radd__(self, other):
    if isinstance(other, core.Tracer) or not _convertible_to_poly(other):
      return self.__jax_array__().__radd__(other)
    if isinstance(other, int) and other == 0: return self
    return _DimExpr._linear_combination(self, 1, other, 1, self.scope)

  def __sub__(self, other):
    if isinstance(other, core.Tracer) or not _convertible_to_poly(other):
      return self.__jax_array__().__sub__(other)
    if isinstance(other, int) and other == 0: return self
    return _DimExpr._linear_combination(self, 1, other, -1, self.scope)

  def __rsub__(self, other):
    if isinstance(other, core.Tracer) or not _convertible_to_poly(other):
      return self.__jax_array__().__rsub__(other)
    return _DimExpr._linear_combination(self, -1, other, 1, self.scope)

  def __neg__(self) -> DimSize:
    return _DimExpr._linear_combination(self, -1, 0, 0, self.scope)

  def __mul__(self, other):
    if isinstance(other, core.Tracer) or not _convertible_to_poly(other):
      return self.__jax_array__().__mul__(other)
    if isinstance(other, int):
      if other == 1: return self
      if other == 0: return 0
      return _DimExpr._linear_combination(self, other, 0, 0, self.scope)
    other = _ensure_poly(other, "mul", self.scope)
    coeffs: dict[_DimTerm, int] = {}
    for mon1, coeff1 in self._sorted_terms:
      for mon2, coeff2 in other._sorted_terms:
        mon = mon1.mul(mon2)
        _DimExpr._add_coeff(coeffs, mon, coeff1 * coeff2)
    return _DimExpr._normalize_sorted_terms(_DimExpr._coeff_to_sorted_terms(coeffs),
                                            self.scope)

  def __rmul__(self, other):
    if isinstance(other, core.Tracer) or not _convertible_to_poly(other):
      return self.__jax_array__().__rmul__(other)
    if isinstance(other, int):
      if other == 1: return self
      if other == 0: return 0
      return _DimExpr._linear_combination(self, other, 0, 0, self.scope)
    return _ensure_poly(other, "mul", self.scope).__mul__(self)

  def __pow__(self, power: core.DimSize, modulo=None):
    if modulo is not None:
      raise NotImplementedError("__pow__ modulo not implemented")
    if is_symbolic_dim(power):
      return power.__rpow__(self)  # type: ignore
    if power != int(power):
      raise ValueError(f"Symbolic dimension cannot be raised to non-integer powers: '{self}' ** '{power}'")
    if power >= 0:
      return functools.reduce(op.mul, [self] * power, 1)
    # We don't support negative powers, because JAX does not allow negative
    # powers for integers
    raise ValueError(f"Symbolic dimension cannot be raised to negative powers: '{self}' ** '{power}'")

  def __rpow__(self, other, modulo=None):
    if modulo is not None:
      raise NotImplementedError("__rpow__ modulo not implemented")
    return self.__jax_array__().__rpow__(other)

  def __floordiv__(self, divisor):
    if isinstance(divisor, core.Tracer) or not _convertible_to_poly(divisor):
      return self.__jax_array__().__floordiv__(divisor)
    return self._divmod(divisor)[0]

  def __rfloordiv__(self, other):
    if isinstance(other, core.Tracer) or not _convertible_to_poly(other):
      return self.__jax_array__().__rfloordiv__(other)
    return _ensure_poly(other, "floordiv", self.scope).__floordiv__(self)

  def __truediv__(self, divisor):
    # Used for "/", which always returns a float
    return self.__jax_array__().__truediv__(divisor)

  def __rtruediv__(self, dividend):
    # Used for "/", when dividend is not a _DimExpr
    return self.__jax_array__().__rtruediv__(dividend)

  def __mod__(self, divisor):
    if isinstance(divisor, core.Tracer) or not _convertible_to_poly(divisor):
      return self.__jax_array__().__mod__(divisor)
    return self._divmod(divisor)[1]

  def __rmod__(self, dividend):
    if isinstance(dividend, core.Tracer) or not _convertible_to_poly(dividend):
      return self.__jax_array__().__rmod__(dividend)
    return _ensure_poly(dividend, "mod", self.scope).__mod__(self)

  def __divmod__(self, divisor):
    if isinstance(divisor, core.Tracer) or not _convertible_to_poly(divisor):
      return self.__jax_array__().__divmod__(divisor)
    return self._divmod(divisor)

  def __rdivmod__(self, dividend):
    if isinstance(dividend, core.Tracer) or not _convertible_to_poly(dividend):
      return self.__jax_array__().__rdivmod__(dividend)
    return _ensure_poly(dividend, "divmod", self.scope).__divmod__(self)

  def __int__(self):
    if (c := _DimExpr._to_constant(self)) is not None:
      return c
    raise InconclusiveDimensionOperation(f"Symbolic dimension '{self}' used in a context that requires a constant")

  # We must overload __eq__ and __ne__, or else we get unsound defaults.
  def __eq__(self, other: Any) -> bool:
    if isinstance(other, _DimExpr):
      if self.scope is not other.scope:
        return False
    elif not core.is_constant_dim(other):
      return False

    # Equality is used very frequently because expressions are cached. We could
    # implement a more precise version based on `(self - other).bounds() = (0, 0)`
    # but that would be too expensive. It would also have the unfortunate drawback
    # that we cannot then cache `e.bounds()` because hashing invokes equality
    # which would lead to infinite recursion.
    diff = self - other

    # We look for `self - other == k`, and we rely on the fact that when we
    # normalize _DimExpr that represent integers as ints.
    if is_symbolic_dim(diff):
      # Here we really ought to raise InconclusiveDimensionOperation, but __eq__
      # cannot raise exceptions, because it is used indirectly when hashing.
      # So, we say that the expressions are disequal, which is really unsound.
      # See https://jax.readthedocs.io/en/latest/export/shape_poly.html#comparison-of-symbolic-dimensions-is-partially-supported
      return False

    return diff == 0

  def __ne__(self, other: Any) -> bool:
    return not self.__eq__(other)

  def __ge__(self, other: DimSize) -> bool:
    return _geq_decision(self, other, lambda: f"'{self}' >= '{other}'")

  def __le__(self, other: DimSize):
    return _geq_decision(other, self, lambda: f"'{self}' <= '{other}'")

  def __gt__(self, other: DimSize):
    return not _geq_decision(other, self, lambda: f"'{self}' > '{other}'")

  def __lt__(self, other: DimSize):
    return not _geq_decision(self, other, lambda: f"'{self}' < '{other}'")

  def _divmod(self, divisor: DimSize) -> tuple[DimSize, int]:
    """
    Floor division with remainder (divmod) generalized to expressions.
    If the `divisor` is not a constant, the remainder must be 0.
    If the `divisor` is a constant, the remainder may be non 0, for consistency
    with integer divmod.

    :return: Quotient resulting from polynomial division and integer remainder.
    """
    try:
      dividend, quotient = self, 0
      # invariant: self = dividend + divisor * quotient
      # quotient and dividend are changed in the loop; the leading term of
      # dividend decreases at each iteration.
      while is_symbolic_dim(dividend) and not dividend._is_constant:  # type: ignore[attribute-error,unused-ignore]
        mon, count = dividend._leading_term
        if isinstance(divisor, _DimExpr):
          dterm, dcount = divisor._leading_term
          qterm = mon.divide(dterm)
        else:
          qterm, dcount = mon, int(divisor)
        qcount, rcount = divmod(count, dcount)
        if rcount != 0:
          raise InconclusiveDimensionOperation("")

        q = _DimExpr._from_term(qterm, qcount, self.scope)
        quotient += q
        dividend -= q * divisor

      dividend = int(dividend)  # type: ignore[assignment]
      if isinstance(divisor, _DimExpr):
        if dividend != 0:
          raise InconclusiveDimensionOperation("")
        remainder = 0
      else:
        q, r = divmod(dividend, int(divisor))
        quotient += q
        remainder = r

      if config.enable_checks.value:
        v1 = divisor * quotient
        v2 = v1 + remainder
        assert self == _ensure_poly(v2, "check", self.scope), (
            self, v2, type(self), type(v2))
        assert self == _ensure_poly(divisor * quotient + remainder, "test", self.scope), (
            self, divisor, quotient, remainder)
      return quotient, remainder
    except InconclusiveDimensionOperation:
      return (_DimExpr._from_operation(_DimFactor.FLOORDIV, self, divisor,
                                       scope=self.scope),  # type: ignore
              _DimExpr._from_operation(_DimFactor.MOD, self, divisor,
                                       scope=self.scope))

  def _evaluate(self, env: DimVarEnv):
    # Evaluates as a value of dtype=core.dim_value_dtype()
    terms = [_evaluate_multiply(t.evaluate(env, self.scope), core.dim_constant(t_k))
             for t, t_k in self._sorted_terms]
    return functools.reduce(_evaluate_add, terms) if len(terms) > 1 else terms[0]

  def max(self, other: DimSize) -> DimSize:
    lb, ub = _bounds_decision(self - other, BoundsPrecision.FOR_GEQ0_OR_LEQ0)
    if 0 <= lb: return self
    if ub <= 0: return other
    return _DimExpr._from_operation(_DimFactor.MAX, self, other, scope=self.scope)

  def rmax(self, other: DimSize) -> DimSize:
    lb, ub = _bounds_decision(self - other, BoundsPrecision.FOR_GEQ0_OR_LEQ0)
    if 0 <= lb: return self
    if ub <= 0: return other
    return _DimExpr._from_operation(_DimFactor.MAX, other, self, scope=self.scope)

  def min(self, other: DimSize) -> DimSize:
    lb, ub = _bounds_decision(self - other, BoundsPrecision.FOR_GEQ0_OR_LEQ0)
    if 0 <= lb: return other
    if ub <= 0: return self
    return _DimExpr._from_operation(_DimFactor.MIN, self, other, scope=self.scope)

  def rmin(self, other: DimSize) -> DimSize:
    lb, ub = _bounds_decision(self - other, BoundsPrecision.FOR_GEQ0_OR_LEQ0)
    if 0 <= lb: return other
    if ub <= 0: return self
    return _DimExpr._from_operation(_DimFactor.MIN, other, self, scope=self.scope)

  @staticmethod
  def _get_aval(dim: _DimExpr):
    return core.dim_value_aval()

  def dimension_as_value(self):
    """Turns a dimension size into a Jax value that we can compute with."""
    return _dim_as_value(self)

  def __jax_array__(self):
    # Used for implicit coercions of polynomials as JAX arrays
    return _dim_as_value(self)

  def __deepcopy__(self, memo):
    return _DimExpr(
        copy.deepcopy(self._sorted_terms, memo),
        copy.deepcopy(self._scope, memo))


def cmp_comparable(i1, i2) -> int:
  if i1 < i2: return -1
  if i1 > i2: return 1
  return 0

def cmp_sequence(s1, s2, elem_cmp) -> int:
  """Compares two sequences using `elem_cmp`."""
  l2 = len(s2)
  for i, e1 in enumerate(s1):
    if i >= l2: return 1
    if c := elem_cmp(e1, s2[i]): return c
  if len(s1) < l2: return -1
  return 0


class SymbolicScope:
  """Indentifies a scope for symbolic expressions.

  All symbolic expressions that interact (e.g., appear in the argument shapes
  for one JAX function invocation, or are involved in arithmetic operations)
  must be from the same scope and must share the same SymbolicScope object.

  Holds the constraints on symbolic expressions.

  See [the README](https://jax.readthedocs.io/en/latest/export/shape_poly.html#user-specified-symbolic-constraints)
  for more details.

  Args:
    constraints_str: A sequence of constraints on symbolic dimension expressions,
      of the form `e1 >= e2` or `e1 <= e2` or `e1 == e2`.
  """

  def __init__(self,
               constraints_str: Sequence[str] = ()):
    if isinstance(constraints_str, str):
      raise ValueError(
          "The symbolic constraints should be a sequence of strings. "
          f"Got {repr(constraints_str)}")
    self._initialized = False
    self._location_frame = source_info_util.user_frame(source_info_util.current())
    # Keep the explicit constraints in the order in which they were added
    self._explicit_constraints: list[_SymbolicConstraint] = []

    # We cache the _DimExpr.bounds calls. The result depends only on the
    # explicit and implicit constraints, so it is safe to keep it in the
    # scope. Set the cache before we parse constraints. We also keep the
    # bounds precision with which we computed the cached result.
    self._bounds_cache: dict[_DimExpr,
                             tuple[float, float, BoundsPrecision]] = {}

    # We store here a decision procedure state initialized with all the
    # _explicit_constraints.
    self._decision_initial_state: Any | None = None

    # We turn the equality constraints into normalization rules.
    # For an explicit constraint `t*tk == e`, we keep
    # `_normalization_rules[t] = (e, tk)`.
    # During building of expressions, if we encounter the term
    # `t*tk1` and `tk1 % tk == 0`, we replace it with `e*(tk1 // tk)`.
    self._normalization_rules: NormalizationRules = {}

    for c_str in constraints_str:
      self._parse_and_process_explicit_constraint(c_str)
      self._bounds_cache.clear()
    self._initialized = True

  def __str__(self) -> str:
    extras = []
    if self._explicit_constraints:
      extras.append(" with constraints:")
      for constr in self._explicit_constraints:
        extras.append(f"  {constr.debug_str}")
    loc = source_info_util._summarize_frame(self._location_frame) if self._location_frame else "unknown"
    return f"{id(self)} created at {loc}" + "\n".join(extras)
  __repr__ = __str__

  def _parse_and_process_explicit_constraint(self, c_str: str):
    if not isinstance(c_str, str):
      raise ValueError(
          f"SymbolicScope constraint must be a string: got {repr(c_str)}")
    cmp_pos, cmp, is_geq = c_str.find("=="), Comparator.EQ, True
    if cmp_pos < 0:
      cmp_pos, cmp, is_geq = c_str.find(">="), Comparator.GEQ, True
      if cmp_pos < 0:
        cmp_pos, cmp, is_geq = c_str.find("<="), Comparator.GEQ, False
      if cmp_pos < 0:
        raise ValueError("Constraint parsing error: must contain one of '==' or '>=' or '<='")
    e1_str = c_str[:cmp_pos]
    e1, = _Parser(e1_str, None, repr(e1_str), self).parse()  # type: ignore[name-error,unused-ignore]
    e2_str = c_str[cmp_pos + 2:]
    e2, = _Parser(e2_str, None, repr(e2_str), self).parse()  # type: ignore[name-error,unused-ignore]
    if cmp == Comparator.GEQ and not is_geq:
      e1, e2 = e2, e1

    # Compute e1 - e2 before we add to normalization rules
    constr = _SymbolicConstraint(debug_str=c_str, cmp=cmp, e1=e1, e2=e2,
                                 diff=e1 - e2)
    self._process_explicit_constraint(constr)

  def _process_explicit_constraint(self, constr: _SymbolicConstraint):
    if (diff_const := _DimExpr._to_constant(constr.diff)) is not None:
      if ((constr.cmp == Comparator.EQ and diff_const != 0) or
          (constr.cmp == Comparator.GEQ and diff_const < 0)):
        raise ValueError(f"Unsatisfiable explicit constraint: {constr.debug_str}")
      return

    if constr.cmp == Comparator.EQ:
      if not isinstance(constr.e1, _DimExpr):
        raise ValueError("Invalid equality constraint: {e1} == {e2}. "
                         "The left-hand-side must be of the form `term * coefficient`.")
      (before, before_k), *rest = constr.e1._sorted_terms
      if rest:
        raise ValueError("Invalid equality constraint: {e1} == {e2}. "
                         "The left-hand-side must be of the form `term * coefficient`.")

      after = _ensure_poly(constr.e2, "parse_constraint", constr.e1.scope)  # type: ignore[name-error,unused-ignore]
      if before in self._normalization_rules:
        raise NotImplementedError(
            f"Found multiple equality constraints with the same left-hand-side: {before}")
      self._normalization_rules[before] = (after, before_k)
      # Look for constraints of the form mod(before_e1, before_k2) * 1 == 0
      if (before_k == 1 and
          isinstance(constr.e2, int) and constr.e2 == 0 and
          (before_f := before.to_factor()) and
          before_f.operation == _DimFactor.MOD and
          (before_k2 := _DimExpr._to_constant(before_f.operands[1])) is not None):
        # Add before_k2*floordiv(before_e1, before_k2) == before_e1
        k_times_floordiv = _DimExpr._from_term(
            _DimTerm.from_operation(
                _DimFactor.FLOORDIV, *before_f.operands, scope=constr.e1.scope),
            before_k2, scope=constr.e1.scope)
        before_e1 = before_f.operands[0]
        self._process_explicit_constraint(
            _SymbolicConstraint(cmp=Comparator.EQ,
                                e1=k_times_floordiv, e2=before_e1,
                                diff=k_times_floordiv - before_e1,
                                debug_str=f"{k_times_floordiv} == {before_e1}")
        )

    self._explicit_constraints.append(constr)

  def _check_same_scope(self, other: _DimExpr,
                        when: str = "",
                        self_descr: str = " ",
                        other_descr: str = "unknown"):
    if self is not other.scope:
      raise ValueError(
          f"Invalid mixing of symbolic scopes {when}.\n"
          f"Expected {self_descr}scope {self}\n"
          f"and found for '{other}' ({other_descr}) scope {other.scope}\n"
          f"See https://jax.readthedocs.io/en/latest/export/shape_poly.html#user-specified-symbolic-constraints.")

  def _clear_caches(self):
    self._bounds_cache.clear()


class BoundsPrecision(enum.Enum):
  """Specifies desired precision for the bounds calculation.

  Since the bounds calculations are expensive, we allow the caller to specify
  a sufficient condition for a result. As the bounds calculations progresses
  the lower bounds is progressively increased and the upper bounds is
  progressively decreased. Depending on the precision, we may stop the
  computation early, if the results are sufficient for the use case.

  The enumeration values are chosen such that, if "(lb, ub)" are sufficient
  for a precision value then they are also sufficient for any smaller
  precision.
  """

  # For static evaluation of "max(e1, e2)", can stop if "lb >= 0 OR ub <= 0"
  FOR_GEQ0_OR_LEQ0 = 0

  # For deciding inequalities, such as "e1 >= e2", can stop if "lb >= 0 OR ub < 0"
  FOR_GEQ0_OR_LT0 = 1

  # Do not stop early, refine bounds as much as possible.
  BEST = 2

  def _bounds_are_sufficient(self, lb: float, ub: float) -> bool:
    if self == BoundsPrecision.FOR_GEQ0_OR_LEQ0:
      return lb >= 0 or ub <= 0
    if self == BoundsPrecision.FOR_GEQ0_OR_LT0:
      return lb >= 0 or ub < 0
    return False

#
# Calling convention:
#  _bounds_decision(e, stop_early)
#    returns a tuple with the lower and upper bound of e.
#    `stop_early(lb, ub)` can be called in an iterative process to decide if the
#    current bounds are tight enough.
# TODO: remove this trampoline when we refactor the sources
def _bounds_decision_unimplemented(
    d: DimSize,
    prec: BoundsPrecision) -> tuple[float, float]:
  del d, prec
  raise NotImplementedError("_bounds_decision is uninitialized")

_bounds_decision: Callable[[DimSize, BoundsPrecision],
                           tuple[float, float]] = _bounds_decision_unimplemented

def _geq_decision(e1: DimSize, e2: DimSize, cmp_str: Callable[[], str]) -> bool:
  """Implements `e1 >= e2`.

  Args:
    e1, e2: the expressions to compare for greater-equal
    cmp_str: a callable such that `cmp_str()` describes the comparison
      for error messages, e.g., "a <= b". Without this all comparisons would
      be reported as ">=".

  Raises InconclusiveDimensionOperation if the result is not conclusive.
  """
  if isinstance(e1, _DimExpr):
    scope = e1.scope
    if isinstance(e2, _DimExpr):
      scope._check_same_scope(e2, f"when comparing {cmp_str()}")
  elif isinstance(e2, _DimExpr):
    scope = e2.scope
  else:
    return int(e1) >= int(e2)
  lb, ub = _bounds_decision(e1 - e2, BoundsPrecision.FOR_GEQ0_OR_LT0)
  if lb >= 0:
    return True
  if ub < 0:
    return False

  if scope._explicit_constraints:
    describe_scope = f"\nUsing symbolic scope {scope}"
  else:
    describe_scope = ""
  raise InconclusiveDimensionOperation(
      f"Symbolic dimension comparison {cmp_str()} is inconclusive.{describe_scope}")

core.pytype_aval_mappings[_DimExpr] = _DimExpr._get_aval
dtypes._weak_types.append(_DimExpr)

def _convertible_to_int(p: DimSize) -> bool:
  try:
    op.index(p)  # type: ignore
    return True
  except:
    return False

def _ensure_poly(p: DimSize,
                 operation_name: str,
                 scope: SymbolicScope) -> _DimExpr:
  if isinstance(p, _DimExpr):
    scope._check_same_scope(p, when=f"for operation {operation_name}")
    return p
  if _convertible_to_int(p):
    return _DimExpr(((_DimTerm_one, op.index(p)),), scope)
  raise TypeError(f"Symbolic dimension {operation_name} not supported for {p}.")

def _convertible_to_poly(p: DimSize) -> bool:
  return isinstance(p, _DimExpr) or _convertible_to_int(p)

def is_symbolic_dim(p: DimSize) -> bool:
  """Checks if a dimension is symbolic.
  """
  return isinstance(p, _DimExpr)

dtypes.python_scalar_dtypes[_DimExpr] = dtypes.python_scalar_dtypes[int]

def _einsum_contract_path(*operands, **kwargs):
  """Like opt_einsum.contract_path, with support for DimExpr shapes.

  We use opt_einsum.contract_path to compute the schedule, using a fixed
  constant for all dimension variables. This is safe because we throw an
  error if there are more than 1 contractions. Essentially, we just use
  opt_einsum.contract_path to parse the specification.
  """

  # Replace the polymorphic shapes with some concrete shapes for calling
  # into opt_einsum.contract_path, because the latter wants to compute the
  # sizes of operands and intermediate results.
  fake_ops = []
  for operand in operands:
    # We replace only array operands
    if not hasattr(operand, "dtype"):
      fake_ops.append(operand)
    else:
      shape = np.shape(operand)
      def fake_dim(d):
        if core.is_constant_dim(d):
          return d
        else:
          if not isinstance(d, _DimExpr):
            raise TypeError(f"Encountered unexpected shape dimension {d}")
          # It is Ok to replace all polynomials with the same value. We may miss
          # here some errors due to non-equal dimensions, but we catch them
          # later.
          return 8
      fake_ops.append(jax.ShapeDtypeStruct(tuple(map(fake_dim, shape)),
                                           operand.dtype))

  contract_fake_ops, contractions = opt_einsum.contract_path(*fake_ops,
                                                             **kwargs)
  contract_operands = []
  for operand in contract_fake_ops:
    idx = tuple(i for i, fake_op in enumerate(fake_ops) if operand is fake_op)
    assert len(idx) == 1
    contract_operands.append(operands[idx[0]])
  return contract_operands, contractions

lax_numpy._poly_einsum_handlers[_DimExpr] = _einsum_contract_path

# To implement shape-constraint checking we use a shape assertion primitive.
#    shape_assertion_p.bind(assert_what: bool, *error_message_inputs,
#                           error_message="...{0}...{1}")
# where "{0}" refers to error_message_inputs[0], etc.
shape_assertion_p = core.Primitive("shape_assertion")
shape_assertion_p.multiple_results = True
shape_assertion_p.def_effectful_abstract_eval(
  lambda *_, **__: ((), {shape_assertion_effect}))  # type: ignore

def _shape_assertion_lowering_rule(ctx: mlir.LoweringRuleContext,
                                   assert_what: mlir.ir.Value,
                                   *error_message_inputs: mlir.ir.Value,
                                   error_message: str):
  op = mlir.custom_call(
    "shape_assertion",
    result_types=[],  # No results
    operands=[assert_what, *error_message_inputs],
    has_side_effect=True,
    extra_attributes=dict(error_message=mlir.ir.StringAttr.get(error_message))
  )
  return op.results

mlir.register_lowering(shape_assertion_p, _shape_assertion_lowering_rule)

class ShapeAssertionEffect(effects.Effect):
  __str__ = lambda _: "ShapeAssertionEffect"

shape_assertion_effect = ShapeAssertionEffect()

effects.lowerable_effects.add_type(ShapeAssertionEffect)
effects.control_flow_allowed_effects.add_type(ShapeAssertionEffect)
effects.remat_allowed_effects.add_type(ShapeAssertionEffect)
effects.custom_derivatives_allowed_effects.add_type(ShapeAssertionEffect)

def shape_assertion(assert_what: jax.Array,
                    *error_message_inputs: jax.Array,
                    error_message: str) -> None:
  """Adds a shape assertion in the code.

  Args:
    assert_what: a boolean asserted to be true. Must be computed based only
      on dimension expressions, so that it can be evaluated after shape
      refinement.
    error_message_inputs: integers expressions whose values can be referenced
      in the `error_message`. Must be computed based only
      on dimension expressions, so that they can be evaluated after shape
      refinement.
    error_message: an error message, possibly containing format specifiers
      {0}, {1}, ..., referencing the values of the `error_message_inputs`.
      The format specifiers are sometimes processed with Python's
      `string::format` method, and sometimes with `llvm::formatv`.
  """
  shape_assertion_p.bind(assert_what, *error_message_inputs,
                         error_message=error_message)

# A JAX primitive with no array arguments but with a dimension parameter
# that is a DimExpr. The value of the primitive is the value of the dimension,
# using int64 in x64 mode or int32 otherwise (core.dim_value_dtype())
dim_as_value_p = core.Primitive("dim_as_value")
dim_as_value_p.def_abstract_eval(lambda dim: core.dim_value_aval())

def dim_as_value_impl(dim: DimSize):
  raise NotImplementedError(
      "Evaluation rule for 'dim_as_value' is not implemented. "
      "It seems that you are using shape polymorphism outside jax.export.")

dim_as_value_p.def_impl(dim_as_value_impl)
def _dim_as_value(dim: DimSize):
  return dim_as_value_p.bind(dim=dim)

def _dim_as_value_lowering(ctx: mlir.LoweringRuleContext, *,
                           dim):
  res, = mlir.eval_dynamic_shape(ctx, (dim,))
  out_type = mlir.aval_to_ir_type(ctx.avals_out[0])
  if out_type != res.type:  # type: ignore
    return [mlir.hlo.convert(out_type, res)]
  else:
    return [res]

mlir.register_lowering(dim_as_value_p, _dim_as_value_lowering)


class PolyShape(tuple):
  """Tuple of polymorphic dimension specifications.

  See docstring of :func:`jax2tf.convert`.
  """

  def __init__(self, *dim_specs):
    warnings.warn("PolyShape is deprecated, use string specifications for symbolic shapes",
                  DeprecationWarning, stacklevel=2)
    tuple.__init__(dim_specs)

  def __new__(cls, *dim_specs):
    warnings.warn("PolyShape is deprecated, use string specifications for symbolic shapes",
                  DeprecationWarning, stacklevel=2)
    for ds in dim_specs:
      if not isinstance(ds, (int, str)) and ds != ...:
        msg = (f"Invalid polymorphic shape element: {ds!r}; must be a string "
               "representing a dimension variable, or an integer, or ...")
        raise ValueError(msg)
    return tuple.__new__(PolyShape, dim_specs)

  def __str__(self):
    return "(" + ", ".join(["..." if d is ... else str(d) for d in self]) + ")"


def symbolic_shape(shape_spec: str | None,
                   *,
                   constraints: Sequence[str] = (),
                   scope: SymbolicScope | None = None,
                   like: Sequence[int | None] | None = None
                   ) -> Sequence[DimSize]:
  """Constructs a symbolic shape from a string representation.

  See https://jax.readthedocs.io/en/latest/export/shape_poly.html for examples.

  Args:
    shape_spec: a symbolic shape specification. None stands for "...".
      A shape specification is the string representation of a tuple (the
      parentheses are optional) with comma-separated dimension expressions.
      A dimension expression can be either: an integer constant,
      a dimension variable (alphanumeric
      starting with a letter), e1 + e2, e1 - e2, e1 * e2, floordiv(e1, e2),
      mod(e1, e2), max(e1, e2), or min(e1, e2).
    constraints: a sequence of constraints on symbolic dimension expressions, of
      the form `e1 >= e2` or `e1 <= e2`, or `e1 == e2`.
      See [the documentation](https://jax.readthedocs.io/en/latest/export/shape_poly.html#user-specified-symbolic-constraints)
      for usage.
    scope: optionally, you can specify that the parsed symbolic expressions
      be created in the given scope. If this is missing, then a new
      `SymbolicScope` is created with the given `constraints`.
      You cannot specify both a `scope` and `constraints`.
      See [the documentation](https://jax.readthedocs.io/en/latest/export/shape_poly.html#user-specified-symbolic-constraints)
      for usage.
    like: when `shape_spec` contains placeholders ("_", "..."), use this
      shape to fill in the placeholders.
      The dimensions of `like` that are used for filling
      must be not `None`. If a dimension in `like` is not `None` and
      the corresponding dimension in `shape_spec` is a constant then they
      must be equal.

  Returns: a tuple with integers or symbolic expressions involving dimension variables.
  """
  shape_spec_repr = repr(shape_spec)
  if shape_spec is None:
    shape_spec = "..."
  elif isinstance(shape_spec, PolyShape):  # TODO: deprecate
    shape_spec = str(shape_spec)
  elif not isinstance(shape_spec, str):
    raise ValueError("polymorphic shape spec should be None or a string. "
                     f"Found {shape_spec_repr}.")
  if scope is None:
    scope = SymbolicScope(constraints)
  elif constraints:
    raise ValueError("Cannot specify both a `scope` and `constraints`.")
  dimensions = _Parser(shape_spec, like, shape_spec_repr, scope).parse()
  return dimensions

def symbolic_args_specs(
    args,  # pytree of arguments
    shapes_specs,  # prefix pytree of strings
    constraints: Sequence[str] = (),
    scope: SymbolicScope | None = None,
):
  """Constructs a pytree of jax.ShapeDtypeSpec arguments specs for `export`.

  See the documentation of :func:`jax.export.symbolic_shape` and
  the [shape polymorphism documentation](https://jax.readthedocs.io/en/latest/export/shape_poly.html) for details.

  Args:
    args: a pytree of arguments. These can be jax.Array, or jax.ShapeDTypeSpec.
      They are used to learn the pytree structure of the arguments, their dtypes,
      and to fill-in the actual shapes where the `shapes_specs` contains
      placeholders. Note that only the shape dimensions for which
      `shapes_specs` is a placeholder are used from `args`.
    shapes_specs: should be `None` (all arguments have static shapes),
      a single string (see `shape_spec` for :func:`jax.export.symbolic_shape`;
      applies to all arguments), or a pytree matching a prefix
      of the `args`.
      See [how optional parameters are matched to
      arguments](https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees).
    constraints: as for :func:`jax.export.symbolic_shape`.
    scope: as for :func:`jax.export.symbolic_shape`.

  Returns: a pytree of jax.ShapeDTypeStruct matching the `args` with the shapes
    replaced with symbolic dimensions as specified by `shapes_specs`.
  """
  polymorphic_shapes = shapes_specs
  args_flat, args_tree = tree_util.tree_flatten(args)

  shapes_and_dtypes = tuple(map(shape_and_dtype_jax_array, args_flat))
  shapes, dtypes = util.unzip2(shapes_and_dtypes)

  if isinstance(args, tuple) and isinstance(polymorphic_shapes, list):
    # TODO: Remove backward-compatibility workaround
    polymorphic_shapes_ = tuple(polymorphic_shapes)
  else:
    polymorphic_shapes_ = polymorphic_shapes

  try:
    polymorphic_shapes_flat = tree_util.broadcast_prefix(
        polymorphic_shapes_, args,
        is_leaf=lambda x: x is None)
  except ValueError:
    e, *_ = tree_util.prefix_errors(
        polymorphic_shapes_, args,
        is_leaf=lambda x: x is None)
    raise e("export.symbolic_args_specs shapes_specs") from None

  # Now add in the polymorphic shapes
  if scope is None:
    scope = SymbolicScope(constraints)
  elif constraints:
    raise ValueError("Cannot use both `scope` and `constraints`")
  args_specs_flat = (
      jax.ShapeDtypeStruct(symbolic_shape(spec, like=s, scope=scope), t)
      for s, t, spec in zip(shapes, dtypes, polymorphic_shapes_flat))

  return args_tree.unflatten(args_specs_flat)

def shape_and_dtype_jax_array(a) -> tuple[Sequence[int | None], DType]:
  """Returns the shape and dtype of a jax.Array or a j"""
  if isinstance(a, jax.ShapeDtypeStruct):
    return a.shape, a.dtype
  aval = core.get_aval(a)
  return aval.shape, aval.dtype


class _Parser:
  def __init__(self,
               shape_spec: str,
               like_shape: Sequence[int | None] | None,
               shape_spec_repr: str,
               scope: SymbolicScope):
    self.shape_spec = shape_spec
    self.shape_spec_repr = shape_spec_repr  # For error messages
    self.like_shape = like_shape
    self.dimensions: list[DimSize] = []  # dimensions we have parsed
    self.scope = scope

  def parse(self) -> Sequence[DimSize]:
    self.tokstream = tokenize.tokenize(
        io.BytesIO(self.shape_spec.encode("utf-8")).readline)
    tok = self.consume_token(self.next_tok(), tokenize.ENCODING)  # Always 1st
    sh, tok = self.shape(tok)
    self.expect_token(tok, [tokenize.ENDMARKER])
    return sh

  def add_dim(self, expr: DimSize | None, tok: tokenize.TokenInfo):
    if expr is None:
      raise self.parse_err(tok,
                           ("unexpected placeholder for unknown dimension; "
                            f"like={self.like_shape}"))

    if core.is_constant_dim(expr) and self.like_shape is not None:
      like_shape_dim = self.like_shape[len(self.dimensions)]
      if expr != like_shape_dim:
        raise self.parse_err(tok,
                             (f"different size {expr} for known dimension; "
                              f"like={self.like_shape}"))
    self.dimensions.append(expr)

  def parse_err(self, tok: tokenize.TokenInfo | None, detail: str) -> Exception:
    msg = (
        f"syntax error in symbolic shape {self.shape_spec_repr} "
        f"in dimension {len(self.dimensions)}: {detail}. ")
    if tok is not None:
      msg += f"Parsed '{tok.line[:tok.start[1]]}', remaining '{tok.line[tok.start[1]:]}'."
    return ValueError(msg)

  def next_tok(self) -> tokenize.TokenInfo:
    while True:
      try:
        t = next(self.tokstream)  # type: ignore[attribute-error,unused-ignore]
      except StopIteration:
        raise self.parse_err(None, "unexpected end of string")
      if t.exact_type not in [tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT]:
        return t

  def expect_token(self, tok: tokenize.TokenInfo, expected: Sequence[int]) -> None:
    if tok.exact_type not in expected:
      msg = ("expecting one of {" +
             ", ".join(tokenize.tok_name[t] for t in expected) + "} but found " +
             tokenize.tok_name[tok.exact_type])
      raise self.parse_err(tok, msg)

  def consume_token(self, tok: tokenize.TokenInfo, expected: int) -> tokenize.TokenInfo:
    self.expect_token(tok, [expected])
    return self.next_tok()

  def integer(self, tok: tokenize.TokenInfo) -> tuple[int, tokenize.TokenInfo]:
    self.expect_token(tok, [tokenize.NUMBER])
    try:
      val = int(tok.string)
    except Exception:
      raise self.parse_err(tok, f"expecting integer, found {tok.string}")
    return val, self.next_tok()

  # What can follow a shape?
  FOLLOW_SHAPE = [tokenize.ENDMARKER, tokenize.RPAR]
  def shape(self, tok: tokenize.TokenInfo) -> tuple[Sequence[DimSize], tokenize.TokenInfo]:
    # A comma-separated list of _DimExpr, or "_", possibly ended with ...
    if tok.exact_type == tokenize.LPAR:
      res, tok = self.shape(self.next_tok())
      tok = self.consume_token(tok, tokenize.RPAR)
      return res, tok

    while True:
      if tok.exact_type in self.FOLLOW_SHAPE:
        break
      # Error checking in presence of placeholders
      if (tok.exact_type == tokenize.ELLIPSIS or
          tok.exact_type == tokenize.NAME and tok.string == "_"):
        if self.like_shape is None:
          raise self.parse_err(tok,
                               "spec contains ... but no 'like' shape was given")
        if tok.exact_type == tokenize.ELLIPSIS:
          min_len_like_shape = len(self.dimensions)
        else:
          min_len_like_shape = len(self.dimensions) + 1
        if len(self.like_shape) < min_len_like_shape:
          raise self.parse_err(
            tok,
            f"cannot resolve placeholder '{tok.string}' because we parsed "
            f"{len(self.dimensions)} already and 'like' shape has "
            f"only {len(self.like_shape)} dimensions")
      if tok.exact_type == tokenize.ELLIPSIS:
        to_add = self.like_shape[len(self.dimensions):]  # type: ignore[index]
        for ad in to_add:
          self.add_dim(ad, tok)
        tok = self.next_tok()
        break

      if tok.exact_type == tokenize.NAME and tok.string == "_":
        e = self.like_shape[len(self.dimensions)]  # type: ignore[index]
        tok = self.next_tok()
      else:
        e, tok = self.expr(tok)
      self.add_dim(e, tok)
      if tok.exact_type in self.FOLLOW_SHAPE:
        break
      tok = self.consume_token(tok, tokenize.COMMA)

    return tuple(self.dimensions), tok

  # What token can follow a _DimExpr
  FOLLOW_EXPR = FOLLOW_SHAPE + [tokenize.COMMA]

  def expr(self, tok: tokenize.TokenInfo) -> tuple[DimSize, tokenize.TokenInfo]:
    # A sum of terms
    next_t_negated = (tok.exact_type == tokenize.MINUS)
    if next_t_negated:
      tok = self.next_tok()
    elif tok.exact_type == tokenize.PLUS:
      tok = self.next_tok()
    acc = None
    while True:
      t, tok = self.term(tok)
      t_sign = - t if next_t_negated else t
      acc = acc + t_sign if acc is not None else t_sign  # type: ignore[operator]
      if tok.exact_type in self.FOLLOW_EXPR:
        return acc, tok
      next_t_negated = (tok.exact_type == tokenize.MINUS)
      self.expect_token(tok, [tokenize.PLUS, tokenize.MINUS])
      tok = self.next_tok()

  FOLLOW_TERM = FOLLOW_EXPR + [tokenize.PLUS, tokenize.MINUS]
  def term(self, tok: tokenize.TokenInfo) -> tuple[DimSize, tokenize.TokenInfo]:
    # A term is product of factors. Each factor may be raised to an integer power.
    acc = None
    while True:
      f, tok = self.factor(tok)
      if tok.exact_type == tokenize.CIRCUMFLEX:
        tok = self.next_tok()
        self.expect_token(tok, [tokenize.NUMBER])
        power, tok = self.integer(tok)
        f = f ** power

      acc = acc * f if acc is not None else f  # type: ignore[operator]
      if tok.exact_type in self.FOLLOW_TERM:
        return acc, tok  # type: ignore[bad-return-type,unused-ignore]
      tok = self.consume_token(tok, tokenize.STAR)

  def factor(self, tok: tokenize.TokenInfo) -> tuple[DimSize, tokenize.TokenInfo]:
    if tok.exact_type == tokenize.NAME:
      if tok.string in (_DimFactor.MOD, _DimFactor.FLOORDIV, _DimFactor.MAX, _DimFactor.MIN):
        return self.factor_binary_op(tok.string, self.next_tok())
      return _DimExpr._from_var(tok.string, self.scope), self.next_tok()
    number_sign = 1
    if tok.exact_type == tokenize.MINUS:  # -k are negative constants
      number_sign = -1
      tok = self.next_tok()
      self.expect_token(tok, [tokenize.NUMBER])
    if tok.exact_type == tokenize.NUMBER:
      v, tok = self.integer(tok)
      return v * number_sign, tok
    self.expect_token(tok, [tokenize.NAME, tokenize.MINUS, tokenize.NUMBER])
    assert False

  def factor_unary_op(self, op: str, tok: tokenize.TokenInfo) -> tuple[DimSize, tokenize.TokenInfo]:
    tok = self.consume_token(tok, tokenize.LPAR)
    e1, tok = self.expr(tok)
    tok = self.consume_token(tok, tokenize.RPAR)
    return _DimExpr._from_operation(op, e1,
                                    scope=self.scope), tok

  def factor_binary_op(self, op: str, tok) -> tuple[DimSize, tokenize.TokenInfo]:
    tok = self.consume_token(tok, tokenize.LPAR)
    e1, tok = self.expr(tok)
    tok = self.consume_token(tok, tokenize.COMMA)
    e2, tok = self.expr(tok)
    tok = self.consume_token(tok, tokenize.RPAR)
    if op == _DimFactor.MAX:
      return core.max_dim(e1, e2), tok
    if op == _DimFactor.MIN:
      return core.min_dim(e1, e2), tok
    return _DimExpr._from_operation(op, e1, e2,
                                    scope=self.scope), tok


def _evaluate_add(v1, v2):
  try:
    if op.index(v1) == 0:
      return v2
  except:
    pass
  try:
    if op.index(v2) == 0:
      return v1
  except:
    pass
  return v1 + v2

def _evaluate_multiply(v1, v2):
  try:
    if op.index(v1) == 1:
      return v2
  except:
    pass
  try:
    if op.index(v2) == 1:
      return v1
  except:
    pass
  return v1 * v2

# dimension_size(operand, dimension=i) get the operand.shape[i] as a
# value of type shape_poly.dim_as_value_dtype().
dimension_size_p = core.Primitive("dimension_size")
def _dimension_size_abstract_eval(aval: core.AbstractValue, **_) -> core.AbstractValue:
  return core.dim_value_aval()

dimension_size_p.def_abstract_eval(_dimension_size_abstract_eval)

def _dimension_size_impl(arg, *, dimension):
  return core.dim_constant(arg.shape[dimension])
dimension_size_p.def_impl(_dimension_size_impl)

def _dimension_size_lowering_rule(ctx, arg, *, dimension):
  dim_size = mlir.hlo.get_dimension_size(arg, dimension)
  dim_type = mlir.aval_to_ir_type(core.dim_value_aval())
  if dim_size.type != dim_type:
    dim_size = mlir.hlo.convert(dim_type, dim_size)
  return [dim_size]

mlir.register_lowering(dimension_size_p, _dimension_size_lowering_rule)


def all_dim_vars(args_avals: Sequence[core.ShapedArray]) -> Sequence[str]:
  dim_vars: set[str] = set()
  for a in args_avals:
    for d in a.shape:
      if is_symbolic_dim(d):
        dim_vars = dim_vars.union(d._get_vars())
  return sorted(dim_vars)


class ShapeEvaluator:
  def __init__(self, env: DimVarEnv):
    self.env = env

  def evaluate(self, e: DimSize):
    if core.is_constant_dim(e):
      res = op.index(e)  # type: ignore
    else:
      res = e._evaluate(self.env)  # type: ignore
    return res


@dataclasses.dataclass(frozen=True)
class ShapeConstraint:

  comp: Comparator
  left: DimSize
  right: DimSize
  # `error_message_pieces` is a list of strings and DimSize. The error message
  # is formed by evaluating the DimSize and concatenating the sequence.
  error_message_pieces: Sequence[str | DimSize]

  def check_statically(self, eval: ShapeEvaluator) -> None:
    """Evaluates a constraint statically."""
    left, right = eval.evaluate(self.left), eval.evaluate(self.right)
    try:
      if self.comp == Comparator.EQ:
        ok = (left == right)
      elif self.comp == Comparator.GEQ:
        ok = (left >= right)
      else:
        assert False  # We are in a context where we know we can evaluate
                      # all symbolic expressions to constants.
    except InconclusiveDimensionOperation as e:
      raise self.make_error(eval) from e
    if not ok:
      raise self.make_error(eval)

  def compute(self, eval: ShapeEvaluator) -> jax.Array | None:
    """Computes if the constraint is satisfied.

    If the constraint can be resolved statically returns None
    or raises ValueError otherwise. If the constraint cannot be
    resolved statically, returns a value representing if the
    constraint is satisfied.
    """
    left, right = eval.evaluate(self.left), eval.evaluate(self.right)
    # Try to evaluate the constraint statically.
    if core.is_constant_shape((left, right)):
      left_int, right_int = op.index(left), op.index(right)
      if self.comp == Comparator.EQ:
        if not (left_int == right_int):
          raise self.make_error(eval)
      elif self.comp == Comparator.GEQ:
        if not (left_int >= right_int):
          raise self.make_error(eval)
      else: assert False
      return None

    if self.comp == Comparator.EQ:
      is_ok = lax.eq(left, right)
    elif self.comp == Comparator.GEQ:
      is_ok = lax.ge(left, right)
    else: assert False
    return is_ok

  def __str__(self):
    return (f"{self.left} {'==' if self.comp == Comparator.EQ else '>='} {self.right}"
            f" ({self.error_message_pieces})")
  __repr__ = __str__

  def error_message_and_inputs(
    self,
    eval: ShapeEvaluator) -> tuple[str, Sequence[Any]]:
    """Forms the error_message and error message_inputs.
    See shape_assertion.
    """
    # There is currently a limitation in the shape assertion checker that
    # it supports at most 32 error_message_inputs. We try to stay within the
    # limit, reusing a format specifier if possible.
    max_error_message_inputs = 32
    format_specifiers: dict[DimSize, str] = {}
    error_message_inputs: list[Any] = []
    error_message_strings: list[str] = []
    for e in self.error_message_pieces:
      if isinstance(e, str):
        error_message_strings.append(e)
        continue
      cached_spec = format_specifiers.get(e)
      if cached_spec is not None:
        error_message_strings.append(cached_spec)
        continue
      if len(error_message_inputs) >= max_error_message_inputs:
        error_message_strings.append("N/A")
        continue
      spec = "{" + str(len(error_message_inputs)) + "}"
      format_specifiers[e] = spec
      error_message_strings.append(spec)
      error_message_inputs.append(eval.evaluate(e))
    return ("".join(error_message_strings),
            error_message_inputs)

  def make_error(self, eval: ShapeEvaluator) -> Exception:
    error_message, error_message_inputs = self.error_message_and_inputs(eval)
    return ValueError(error_message.format(*error_message_inputs))


class ShapeConstraints:
  def __init__(self):
    self.constraints: list[ShapeConstraint] = []

  def add_constraint(self,
                     comp: Comparator,
                     left: DimSize, right: DimSize,
                     error_message_pieces: Sequence[str | DimSize]):
    c = ShapeConstraint(comp, left, right, error_message_pieces)
    self.constraints.append(c)

  def check_statically(self, eval: ShapeEvaluator) -> None:
    """Evaluates all the constraints statically.

    If the static checking of any constraint fails, raises ValueError.
    """
    for constraint in self.constraints:
      constraint.check_statically(eval)

  def shape_assertions(self, eval: ShapeEvaluator) -> None:
    """Computes the shape assertions for the set of constraints.

    See jax_export.Exported docstring.
    """
    # We want to report the errors in the same order as `check_statically`.
    # So, we process them in order, in case some fail statically, and we
    # generate the shape assertions in the same order.
    for constraint in self.constraints:
      is_ok = constraint.compute(eval)
      if is_ok is None: continue   # Was resolved statically
      error_message, error_message_inputs = constraint.error_message_and_inputs(eval)
      shape_assertion(
          is_ok, *error_message_inputs,
          error_message=error_message)

@dataclasses.dataclass
class _DimEquation:
  # Encodes that `aval_dim_expr`, which is a symbolic expressions containing
  # unknown dimension variables from the abstract values, is the specification
  # for dimension named `dim_name` (e.g., "args[0].field.shape[2]").
  aval_dim_expr: _DimExpr
  dim_name: str

  def __str__(self):
    return f"Dimension size of {self.dim_name} with specification '{self.aval_dim_expr}'"
  __repr__ = __str__


def args_kwargs_path_to_str(path: tree_util.KeyPath) -> str:
  # String description of `args` or `kwargs`, assuming the path for a tree for
  # the tuple `(args, kwargs)`.
  if path[0] == tree_util.SequenceKey(0):
    return f"args{tree_util.keystr(path[1:])}"
  elif path[0] == tree_util.SequenceKey(1):
    return f"kwargs{tree_util.keystr(path[1:])}"
  else:
    assert False

@functools.lru_cache(128)
def _cached_pretty_print_dimension_descriptor(
    args_kwargs_tree: tree_util.PyTreeDef,
    flat_arg_idx: int) -> str:
  args_kwargs_with_paths, _ = tree_util.tree_flatten_with_path(
      args_kwargs_tree.unflatten((0,) * args_kwargs_tree.num_leaves))
  arg_str = args_kwargs_path_to_str(args_kwargs_with_paths[flat_arg_idx][0])
  return arg_str

def pretty_print_dimension_descriptor(
    args_kwargs_tree: tree_util.PyTreeDef,
    flat_arg_idx: int, dim_idx: int | None) -> str:
  arg_str = _cached_pretty_print_dimension_descriptor(args_kwargs_tree, flat_arg_idx)
  if dim_idx is not None:
    arg_str += f".shape[{dim_idx}]"
  return arg_str

@util.cache()
def solve_dim_vars(
    args_avals: Sequence[core.ShapedArray],
    args_kwargs_tree: tree_util.PyTreeDef,
    ) -> tuple[DimVarEnv, ShapeConstraints, Sequence[tuple[str, int, int]]]:
  """Solves dimension variables in a called function's avals in terms of actual argument shapes.

  For example, given:

     args_avals = [ShapedArray((3, a, a + b), f32)]

  we introduce fresh "synthetic" dimension variables to represent the actual
  dimension size of actual arguments for each non-constant dimension.
  Each synthetic variable has a name, an arg_idx, and a dim_idx, e.g.:

    synthetic_vars = [("args[0].shape[1]", 0, 1), ("args[0].shape[2]", 0, 2)]

  and then we express the solution for the unknown dimension variables {a, b}
  as symbolic expressions in terms of the synthetic variables:

    dict(a=args[0].shape[1], b=args[0].shape[2] - args[0].shape[1])

  Not all equations are solvable. For now, we solve first the linear
  uni-variate equations, then the solved variables are used to simplify the
  remaining equations to linear uni-variate equations, and the process
  continues until all dimension variables are solved.

  Args:
    args_avals: the abstract values of the `args`, with shapes that may
      include unknown dimension variables.
    args_kwargs_tree: a PyTreeDef that describes the tuple `(args, kwargs)`
      from which the flat sequence `args_avals` is extracted. Used for
      describing args and kwargs in synthetic variable names and in
      error messages.

  Returns: a 3-tuple with: (a) the solution for the unknown dimension variables
   (b) a list of constraints that must be satisfied for the solution to be a
   valid one, and (c) and the list of synthetic variables that may appear in
   the solution and the constraints.

  Raises ValueError if it cannot solve some dimension variable.
  """
  dim_equations: list[_DimEquation] = []
  synth_dimension_vars: list[tuple[str, int, int]] = []
  # tuples with argument name and its polymorphic shape ('args[0]', '(a, a + b'))
  polymorphic_shape_specs: list[tuple[str, str]] = []
  for arg_idx, aval in enumerate(args_avals):
    if all(not is_symbolic_dim(d) for d in aval.shape):
      continue
    polymorphic_shape_specs.append(
      (pretty_print_dimension_descriptor(args_kwargs_tree, arg_idx, None),
       str(aval.shape)))
    for dim_idx, aval_d in enumerate(aval.shape):
      if is_symbolic_dim(aval_d):
        synth_dim_var = pretty_print_dimension_descriptor(args_kwargs_tree,
                                                          arg_idx, dim_idx)
        synth_dimension_vars.append((synth_dim_var, arg_idx, dim_idx))
        dim_equations.append(
            _DimEquation(aval_dim_expr=aval_d,
                         dim_name=synth_dim_var))

  solution, shape_constraints = _solve_dim_equations(dim_equations,
                                                     polymorphic_shape_specs)
  return solution, shape_constraints, synth_dimension_vars


def compute_dim_vars_from_arg_shapes(
    args_avals: Sequence[core.ShapedArray],
    *actual_args: jax.Array,
    args_kwargs_tree: tree_util.PyTreeDef) -> Sequence[jax.Array]:
  """Computes values of dimension variables to unify args_avals with actual arguments.

  Like `solve_dim_vars` except that here we express the solution as
  JAX arrays that reference the `actual_args`. This function can be used to
  generate the code for computing the dimension variables. It also generates
  the shape assertions.

  Returns:
    The values of the dimension variables, in the order determined by
    `all_dim_vars(args_avals)`.
  """
  dim_vars = all_dim_vars(args_avals)
  solution, shape_constraints, synth_dim_vars = solve_dim_vars(
      tuple(args_avals), args_kwargs_tree=args_kwargs_tree)

  # Replace the synthetic vars with the dynamic shape of the actual arg
  synthetic_env: DimVarEnv = {
      vname: dimension_size_p.bind(actual_args[arg_idx], dimension=dim_idx)
      for (vname, arg_idx, dim_idx) in synth_dim_vars
  }
  synthetic_eval = ShapeEvaluator(synthetic_env)
  shape_constraints.shape_assertions(synthetic_eval)
  return tuple(synthetic_eval.evaluate(solution[var]) for var in dim_vars)

def _solve_dim_equations(
    eqns: list[_DimEquation],
    polymorphic_shape_specs: Sequence[tuple[str, str]]
) -> tuple[DimVarEnv, ShapeConstraints]:
  # Returns a shape environment and the shape constraints if it can solve all
  # dimension variables. Raises an exception if it cannot.
  shape_env: DimVarEnv = {}
  solution_error_message_pieces: list[str | DimSize] = [
    " Obtained dimension variables: "
  ]  # Error message describing the solution
  # Prepare error message piece describing the polymorphic shape specs
  poly_specs_err_msg = (
    " Using the following polymorphic shapes specifications: " +
    ",".join(f"{arg_name}.shape = {arg_spec}"
             for arg_name, arg_spec in polymorphic_shape_specs)) + "."
  solution_err_msg_trailer_errors = ". Please see https://jax.readthedocs.io/en/latest/export/shape_poly.html#shape-assertion-errors for more details."

  shape_constraints = ShapeConstraints()  # accumulate shape constraints
  scope: SymbolicScope | None = None

  def process_one_eqn(eqn: _DimEquation) -> bool:
    # We start with a DimEquation of the form `dim_expr = dim_value`
    # Try to rewrite the equation as `var * factor_var = dim_value_2` (a linear
    # uni-variate equation). Returns `False` if this rewrite fails.
    # Otherwise, compute the `var` value as `dim_value_2 // factor`, add it to
    # `shape_env` and return `True`.
    #
    # Invariant:
    #     var * factor_var + remaining_terms_from_dim_expr = dim_value
    var, var_k = None, None
    nonlocal scope
    if scope is None:
      scope = eqn.aval_dim_expr.scope
    elif config.enable_checks.value:
      scope._check_same_scope(eqn.aval_dim_expr, when=f"solving equation {eqn}")

    dim_value = _DimExpr._from_var(eqn.dim_name, scope)

    for term, term_k in eqn.aval_dim_expr._sorted_terms:
      # Perhaps we can already evaluate this term (all vars solved)
      try:
        term_value = term.evaluate(shape_env, scope)
      except UnexpectedDimVar:
        # `mon` still uses some variables not yet solved. We handle only the
        # case when `mon` is a single variable.
        v = term.to_var()
        if v is not None and var is None:
          var, var_k = v, term_k
          continue
      else:
        dim_value = dim_value + core.dim_constant(-1) * _evaluate_multiply(term_value, core.dim_constant(term_k))
        continue
      return False  # This equation cannot yet be used to solve a variable

    if var is not None:
      if var_k == 1:
        var_value = dim_value
      else:
        var_value, var_remainder = divmod(dim_value, core.dim_constant(var_k))  # type: ignore
        shape_constraints.add_constraint(
            Comparator.EQ, var_remainder, 0,
            error_message_pieces=([
                "Input shapes do not match the polymorphic shapes specification. "
                "Division had remainder ", var_remainder,
                f" when computing the value of '{var}'." + poly_specs_err_msg
              ] + solution_error_message_pieces + [
                solution_err_msg_trailer_errors]))

      if not isinstance(var_value, _DimExpr):
        assert var_value.dtype == core.dim_value_dtype()  # type: ignore[attribute-error,unused-ignore]
      shape_env[var] = var_value  # type: ignore
      solution_error_message_pieces.extend([  # type: ignore[container-type-mismatch,unused-ignore]
        f"'{var}' = ", var_value,
        f" from specification '{eqn.aval_dim_expr}' "
        f"for dimension {eqn.dim_name} (= ",
        _DimExpr._from_var(eqn.dim_name, eqn.aval_dim_expr.scope),
        "), "])

      shape_constraints.add_constraint(
          Comparator.GEQ, var_value, 1,
          error_message_pieces=[
                "Input shapes do not match the polymorphic shapes specification. "
                f"Expected value >= 1 for dimension variable '{var}'." +
                poly_specs_err_msg
              ] + solution_error_message_pieces + [
              solution_err_msg_trailer_errors])

      return True
    else:
      # All variables are resolved for this equation, we emit an assertion
      shape_constraints.add_constraint(
          Comparator.EQ,
          _DimExpr._from_var(eqn.dim_name, eqn.aval_dim_expr.scope),
          eqn.aval_dim_expr._evaluate(shape_env),
          error_message_pieces=([
            "Input shapes do not match the polymorphic shapes specification. "
            f"Found inconsistency between dimension size {eqn.dim_name} (= ",
            _DimExpr._from_var(eqn.dim_name, eqn.aval_dim_expr.scope),
            f") and the specification '{eqn.aval_dim_expr}' (= ",
            eqn.aval_dim_expr._evaluate(shape_env),
            ")." + poly_specs_err_msg] + solution_error_message_pieces +
            [solution_err_msg_trailer_errors])
      )
      return True

  def add_explicit_symbolic_constraints(shape_env: DimVarEnv):
    if not shape_env: return
    assert scope is not None
    for constr in scope._explicit_constraints:
      # We can't just construct constr.e1 - constr.e2 because for an equality
      # constraint it would be reduced to 0.
      c_diff = constr.diff._evaluate(shape_env) if not core.is_constant_dim(constr.diff) else constr.diff  # type: ignore
      shape_constraints.add_constraint(
          constr.cmp, c_diff, 0,
          error_message_pieces=[
                f"Input shapes do not match the symbolic shape constraint {constr.debug_str}. "
                f"Expected '{constr.diff}' to be "
                f"{'greater or equal' if constr.cmp == Comparator.GEQ else 'equal'} to 0, "
                "but found ", c_diff,

                ". " + poly_specs_err_msg
              ] + solution_error_message_pieces + [
              solution_err_msg_trailer_errors])


  while True:
    nr_eqns = len(eqns)
    eqns = [eqn for eqn in eqns if not process_one_eqn(eqn)]
    if not eqns:
      add_explicit_symbolic_constraints(shape_env)
      # SUCCESS
      return shape_env, shape_constraints  # pytype: disable=bad-return-type
    elif len(eqns) >= nr_eqns:
      break

  # We have some equations that we cannot solve further
  unsolved_vars: set[str] = set()
  unsolved_polys: list[_DimExpr] = []
  for eqn in eqns:
    unsolved_vars = unsolved_vars.union(eqn.aval_dim_expr._get_vars())
    unsolved_polys.append(eqn.aval_dim_expr)
  unsolved_vars = unsolved_vars.difference(shape_env.keys())
  err_msg = (
      f"Cannot solve for values of dimension variables {unsolved_vars}. "
      "We can only solve linear uni-variate constraints." + poly_specs_err_msg +
      " Unprocessed specifications: " +
      ", ".join(f"'{eqn.aval_dim_expr}' for dimension size {eqn.dim_name}"
                for eqn in eqns) +
      ". Please see https://jax.readthedocs.io/en/latest/export/shape_poly.html#dimension-variables-must-be-solvable-from-the-input-shapes for more details."
  )
  raise ValueError(err_msg)
