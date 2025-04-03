//===- Utils.h - General utilities for Presburger library ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions required by the Presburger Library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_UTILS_H
#define MLIR_ANALYSIS_PRESBURGER_UTILS_H

#include "mlir/Analysis/Presburger/Matrix.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>

namespace mlir {
namespace presburger {
class IntegerRelation;

/// This class represents the result of operations optimizing something subject
/// to some constraints. If the constraints were not satisfiable the, kind will
/// be Empty. If the optimum is unbounded, the kind is Unbounded, and if the
/// optimum is bounded, the kind will be Bounded and `optimum` holds the optimal
/// value.
enum class OptimumKind { Empty, Unbounded, Bounded };
template <typename T>
class MaybeOptimum {
public:
private:
  OptimumKind kind = OptimumKind::Empty;
  T optimum;

public:
  MaybeOptimum() = default;
  MaybeOptimum(OptimumKind kind) : kind(kind) {
    assert(kind != OptimumKind::Bounded &&
           "Bounded optima should be constructed by specifying the optimum!");
  }
  MaybeOptimum(const T &optimum)
      : kind(OptimumKind::Bounded), optimum(optimum) {}

  OptimumKind getKind() const { return kind; }
  bool isBounded() const { return kind == OptimumKind::Bounded; }
  bool isUnbounded() const { return kind == OptimumKind::Unbounded; }
  bool isEmpty() const { return kind == OptimumKind::Empty; }

  std::optional<T> getOptimumIfBounded() const { return optimum; }
  const T &getBoundedOptimum() const {
    assert(kind == OptimumKind::Bounded &&
           "This should be called only for bounded optima");
    return optimum;
  }
  T &getBoundedOptimum() {
    assert(kind == OptimumKind::Bounded &&
           "This should be called only for bounded optima");
    return optimum;
  }
  const T &operator*() const { return getBoundedOptimum(); }
  T &operator*() { return getBoundedOptimum(); }
  const T *operator->() const { return &getBoundedOptimum(); }
  T *operator->() { return &getBoundedOptimum(); }
  bool operator==(const MaybeOptimum<T> &other) const {
    if (kind != other.kind)
      return false;
    if (kind != OptimumKind::Bounded)
      return true;
    return optimum == other.optimum;
  }

  // Given f that takes a T and returns a U, convert this `MaybeOptimum<T>` to
  // a `MaybeOptimum<U>` by applying `f` to the bounded optimum if it exists, or
  // returning a MaybeOptimum of the same kind otherwise.
  template <class Function>
  auto map(const Function &f) const & -> MaybeOptimum<decltype(f(optimum))> {
    if (kind == OptimumKind::Bounded)
      return f(optimum);
    return kind;
  }
};

/// `ReprKind` enum is used to set the constraint type in `MaybeLocalRepr`.
enum class ReprKind { Inequality, Equality, None };

/// `MaybeLocalRepr` contains the indices of the constraints that can be
/// expressed as a floordiv of an affine function. If it's an `equality`
/// constraint, `equalityIdx` is set, in case of `inequality` the
/// `lowerBoundIdx` and `upperBoundIdx` is set. By default the kind attribute is
/// set to None.
struct MaybeLocalRepr {
  ReprKind kind = ReprKind::None;
  explicit operator bool() const { return kind != ReprKind::None; }
  union {
    unsigned equalityIdx;
    struct {
      unsigned lowerBoundIdx, upperBoundIdx;
    } inequalityPair;
  } repr;
};

/// Class storing division representation of local variables of a constraint
/// system. The coefficients of the dividends are stored in order:
/// [nonLocalVars, localVars, constant]. Each local variable may or may not have
/// a representation. If the local does not have a representation, the dividend
/// of the division has no meaning and the denominator is zero. If it has a
/// representation, the denominator will be positive.
///
/// The i^th division here, represents the division representation of the
/// variable at position `divOffset + i` in the constraint system.
class DivisionRepr {
public:
  DivisionRepr(unsigned numVars, unsigned numDivs)
      : dividends(numDivs, numVars + 1), denoms(numDivs, DynamicAPInt(0)) {}

  DivisionRepr(unsigned numVars) : dividends(0, numVars + 1) {}

  unsigned getNumVars() const { return dividends.getNumColumns() - 1; }
  unsigned getNumDivs() const { return dividends.getNumRows(); }
  unsigned getNumNonDivs() const { return getNumVars() - getNumDivs(); }
  // Get the offset from where division variables start.
  unsigned getDivOffset() const { return getNumVars() - getNumDivs(); }

  // Check whether the `i^th` division has a division representation or not.
  bool hasRepr(unsigned i) const { return denoms[i] != 0; }
  // Check whether all the divisions have a division representation or not.
  bool hasAllReprs() const { return !llvm::is_contained(denoms, 0); }

  // Clear the division representation of the i^th local variable.
  void clearRepr(unsigned i) { denoms[i] = 0; }

  // Get the dividend of the `i^th` division.
  MutableArrayRef<DynamicAPInt> getDividend(unsigned i) {
    return dividends.getRow(i);
  }
  ArrayRef<DynamicAPInt> getDividend(unsigned i) const {
    return dividends.getRow(i);
  }

  // For a given point containing values for each variable other than the
  // division variables, try to find the values for each division variable from
  // their division representation.
  SmallVector<std::optional<DynamicAPInt>, 4>
  divValuesAt(ArrayRef<DynamicAPInt> point) const;

  // Get the `i^th` denominator.
  DynamicAPInt &getDenom(unsigned i) { return denoms[i]; }
  DynamicAPInt getDenom(unsigned i) const { return denoms[i]; }

  ArrayRef<DynamicAPInt> getDenoms() const { return denoms; }

  void setDiv(unsigned i, ArrayRef<DynamicAPInt> dividend,
              const DynamicAPInt &divisor) {
    dividends.setRow(i, dividend);
    denoms[i] = divisor;
  }

  // Find the greatest common divisor (GCD) of the dividends and divisor for
  // each valid division. Divide the dividends and divisor by the GCD to
  // simplify the expression.
  void normalizeDivs();

  void insertDiv(unsigned pos, ArrayRef<DynamicAPInt> dividend,
                 const DynamicAPInt &divisor);
  void insertDiv(unsigned pos, unsigned num = 1);

  /// Removes duplicate divisions. On every possible duplicate division found,
  /// `merge(i, j)`, where `i`, `j` are current index of the duplicate
  /// divisions, is called and division at index `j` is merged into division at
  /// index `i`. If `merge(i, j)` returns `true`, the divisions are merged i.e.
  /// `j^th` division gets eliminated and it's each instance is replaced by
  /// `i^th` division. If it returns `false`, the divisions are not merged.
  /// `merge` can also do side effects, For example it can merge the local
  /// variables in IntegerRelation.
  void
  removeDuplicateDivs(llvm::function_ref<bool(unsigned i, unsigned j)> merge);

  void print(raw_ostream &os) const;
  void dump() const;

private:
  /// Each row of the Matrix represents a single division dividend. The
  /// `i^th` row represents the dividend of the variable at `divOffset + i`
  /// in the constraint system (and the `i^th` local variable).
  IntMatrix dividends;

  /// Denominators of each division. If a denominator of a division is `0`, the
  /// division variable is considered to not have a division representation.
  /// Otherwise, the denominator is positive.
  SmallVector<DynamicAPInt, 4> denoms;
};

/// If `q` is defined to be equal to `expr floordiv d`, this equivalent to
/// saying that `q` is an integer and `q` is subject to the inequalities
/// `0 <= expr - d*q <= c - 1` (quotient remainder theorem).
///
/// Rearranging, we get the bounds on `q`: d*q <= expr <= d*q + d - 1.
///
/// `getDivUpperBound` returns `d*q <= expr`, and
/// `getDivLowerBound` returns `expr <= d*q + d - 1`.
///
/// The parameter `dividend` corresponds to `expr` above, `divisor` to `d`, and
/// `localVarIdx` to the position of `q` in the coefficient list.
///
/// The coefficient of `q` in `dividend` must be zero, as it is not allowed for
/// local variable to be a floor division of an expression involving itself.
/// The divisor must be positive.
SmallVector<DynamicAPInt, 8> getDivUpperBound(ArrayRef<DynamicAPInt> dividend,
                                              const DynamicAPInt &divisor,
                                              unsigned localVarIdx);
SmallVector<DynamicAPInt, 8> getDivLowerBound(ArrayRef<DynamicAPInt> dividend,
                                              const DynamicAPInt &divisor,
                                              unsigned localVarIdx);

llvm::SmallBitVector getSubrangeBitVector(unsigned len, unsigned setOffset,
                                          unsigned numSet);

/// Check if the pos^th variable can be expressed as a floordiv of an affine
/// function of other variables (where the divisor is a positive constant).
/// `foundRepr` contains a boolean for each variable indicating if the
/// explicit representation for that variable has already been computed.
/// Return the given array as an array of DynamicAPInts.
SmallVector<DynamicAPInt, 8> getDynamicAPIntVec(ArrayRef<int64_t> range);
/// Return the given array as an array of int64_t.
SmallVector<int64_t, 8> getInt64Vec(ArrayRef<DynamicAPInt> range);

/// Returns the `MaybeLocalRepr` struct which contains the indices of the
/// constraints that can be expressed as a floordiv of an affine function. If
/// the representation could be computed, `dividend` and `divisor` are set,
/// in which case, denominator will be positive. If the representation could
/// not be computed, the kind attribute in `MaybeLocalRepr` is set to None.
MaybeLocalRepr computeSingleVarRepr(const IntegerRelation &cst,
                                    ArrayRef<bool> foundRepr, unsigned pos,
                                    MutableArrayRef<DynamicAPInt> dividend,
                                    DynamicAPInt &divisor);

/// The following overload using int64_t is required for a callsite in
/// AffineStructures.h.
MaybeLocalRepr computeSingleVarRepr(const IntegerRelation &cst,
                                    ArrayRef<bool> foundRepr, unsigned pos,
                                    SmallVector<int64_t, 8> &dividend,
                                    unsigned &divisor);

/// Given two relations, A and B, add additional local vars to the sets such
/// that both have the union of the local vars in each set, without changing
/// the set of points that lie in A and B.
///
/// While taking union, if a local var in any set has a division representation
/// which is a duplicate of division representation, of another local var in any
/// set, it is not added to the final union of local vars and is instead merged.
///
/// On every possible merge, `merge(i, j)` is called. `i`, `j` are position
/// of local variables in both sets which are being merged. If `merge(i, j)`
/// returns true, the divisions are merged, otherwise the divisions are not
/// merged.
void mergeLocalVars(IntegerRelation &relA, IntegerRelation &relB,
                    llvm::function_ref<bool(unsigned i, unsigned j)> merge);

/// Compute the gcd of the range.
DynamicAPInt gcdRange(ArrayRef<DynamicAPInt> range);

/// Divide the range by its gcd and return the gcd.
DynamicAPInt normalizeRange(MutableArrayRef<DynamicAPInt> range);

/// Normalize the given (numerator, denominator) pair by dividing out the
/// common factors between them. The numerator here is an affine expression
/// with integer coefficients. The denominator must be positive.
void normalizeDiv(MutableArrayRef<DynamicAPInt> num, DynamicAPInt &denom);

/// Return `coeffs` with all the elements negated.
SmallVector<DynamicAPInt, 8> getNegatedCoeffs(ArrayRef<DynamicAPInt> coeffs);

/// Return the complement of the given inequality.
///
/// The complement of a_1 x_1 + ... + a_n x_ + c >= 0 is
/// a_1 x_1 + ... + a_n x_ + c < 0, i.e., -a_1 x_1 - ... - a_n x_ - c - 1 >= 0,
/// since all the variables are constrained to be integers.
SmallVector<DynamicAPInt, 8> getComplementIneq(ArrayRef<DynamicAPInt> ineq);

/// Compute the dot product of two vectors.
/// The vectors must have the same sizes.
Fraction dotProduct(ArrayRef<Fraction> a, ArrayRef<Fraction> b);

/// Find the product of two polynomials, each given by an array of
/// coefficients.
std::vector<Fraction> multiplyPolynomials(ArrayRef<Fraction> a,
                                          ArrayRef<Fraction> b);

bool isRangeZero(ArrayRef<Fraction> arr);

/// Example usage:
/// Print .12, 3.4, 56.7
/// preAlign = ".", minSpacing = 1,
///    .12   .12
///   3.4   3.4
///  56.7  56.7
struct PrintTableMetrics {
  // If unknown, set to 0 and pass the struct into updatePrintMetrics.
  unsigned maxPreIndent;
  unsigned maxPostIndent;
  std::string preAlign;
};

/// Iterate over each val in the table and update 'm' where
/// .maxPreIndent and .maxPostIndent are initialized to 0.
/// class T is any type that can be handled by llvm::raw_string_ostream.
template <class T>
void updatePrintMetrics(T val, PrintTableMetrics &m) {
  std::string str;
  llvm::raw_string_ostream(str) << val;
  if (str.empty())
    return;
  unsigned preIndent = str.find(m.preAlign);
  preIndent = (preIndent != (unsigned)std::string::npos) ? preIndent + 1 : 0;
  m.maxPreIndent = std::max(m.maxPreIndent, preIndent);
  m.maxPostIndent =
      std::max(m.maxPostIndent, (unsigned int)(str.length() - preIndent));
}

/// Print val in the table with metrics specified in 'm'.
template <class T>
void printWithPrintMetrics(raw_ostream &os, T val, unsigned minSpacing,
                           const PrintTableMetrics &m) {
  std::string str;
  llvm::raw_string_ostream(str) << val;
  unsigned preIndent;
  if (!str.empty()) {
    preIndent = str.find(m.preAlign);
    preIndent = (preIndent != (unsigned)std::string::npos) ? preIndent + 1 : 0;
  } else {
    preIndent = 0;
  }
  for (unsigned i = 0; i < (minSpacing + m.maxPreIndent - preIndent); ++i)
    os << " ";
  os << str;
  for (unsigned i = 0; i < m.maxPostIndent - (str.length() - preIndent); ++i)
    os << " ";
}
} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_UTILS_H
