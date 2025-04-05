//===- PWMAFunction.h - MLIR PWMAFunction Class------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support for piece-wise multi-affine functions. These are functions that are
// defined on a domain that is a union of IntegerPolyhedrons, and on each domain
// the value of the function is a tuple of integers, with each value in the
// tuple being an affine expression in the vars of the IntegerPolyhedron.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PWMAFUNCTION_H
#define MLIR_ANALYSIS_PRESBURGER_PWMAFUNCTION_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include <optional>

namespace mlir {
namespace presburger {

/// Enum representing a binary comparison operator: equal, not equal, less than,
/// less than or equal, greater than, greater than or equal.
enum class OrderingKind { EQ, NE, LT, LE, GT, GE };

/// This class represents a multi-affine function with the domain as Z^d, where
/// `d` is the number of domain variables of the function. For example:
///
/// (x, y) -> (x + 2, 2*x - 3y + 5, 2*x + y).
///
/// The output expressions are represented as a matrix with one row for every
/// output, one column for each var including division variables, and an extra
/// column at the end for the constant term.
///
/// Checking equality of two such functions is supported, as well as finding the
/// value of the function at a specified point.
class MultiAffineFunction {
public:
  MultiAffineFunction(const PresburgerSpace &space, const IntMatrix &output)
      : space(space), output(output),
        divs(space.getNumVars() - space.getNumRangeVars()) {
    assertIsConsistent();
  }

  MultiAffineFunction(const PresburgerSpace &space, const IntMatrix &output,
                      const DivisionRepr &divs)
      : space(space), output(output), divs(divs) {
    assertIsConsistent();
  }

  unsigned getNumDomainVars() const { return space.getNumDomainVars(); }
  unsigned getNumSymbolVars() const { return space.getNumSymbolVars(); }
  unsigned getNumOutputs() const { return space.getNumRangeVars(); }
  unsigned getNumDivs() const { return space.getNumLocalVars(); }

  /// Get the space of this function.
  const PresburgerSpace &getSpace() const { return space; }
  /// Get the domain/output space of the function. The returned space is a set
  /// space.
  PresburgerSpace getDomainSpace() const { return space.getDomainSpace(); }
  PresburgerSpace getOutputSpace() const { return space.getRangeSpace(); }

  /// Get a matrix with each row representing row^th output expression.
  const IntMatrix &getOutputMatrix() const { return output; }
  /// Get the `i^th` output expression.
  ArrayRef<DynamicAPInt> getOutputExpr(unsigned i) const {
    return output.getRow(i);
  }

  /// Get the divisions used in this function.
  const DivisionRepr &getDivs() const { return divs; }

  /// Remove the specified range of outputs.
  void removeOutputs(unsigned start, unsigned end);

  /// Given a MAF `other`, merges division variables such that both functions
  /// have the union of the division vars that exist in the functions.
  void mergeDivs(MultiAffineFunction &other);

  //// Return the output of the function at the given point.
  SmallVector<DynamicAPInt, 8> valueAt(ArrayRef<DynamicAPInt> point) const;
  SmallVector<DynamicAPInt, 8> valueAt(ArrayRef<int64_t> point) const {
    return valueAt(getDynamicAPIntVec(point));
  }

  /// Return whether the `this` and `other` are equal when the domain is
  /// restricted to `domain`. This is the case if they lie in the same space,
  /// and their outputs are equal for every point in `domain`.
  bool isEqual(const MultiAffineFunction &other) const;
  bool isEqual(const MultiAffineFunction &other,
               const IntegerPolyhedron &domain) const;
  bool isEqual(const MultiAffineFunction &other,
               const PresburgerSet &domain) const;

  void subtract(const MultiAffineFunction &other);

  /// Return the set of domain points where the output of `this` and `other`
  /// are ordered lexicographically according to the given ordering.
  /// For example, if the given comparison is `LT`, then the returned set
  /// contains all points where the first output of `this` is lexicographically
  /// less than `other`.
  PresburgerSet getLexSet(OrderingKind comp,
                          const MultiAffineFunction &other) const;

  /// Get this function as a relation.
  IntegerRelation getAsRelation() const;

  void print(raw_ostream &os) const;
  void dump() const;

private:
  /// Assert that the MAF is consistent.
  void assertIsConsistent() const;

  /// The space of this function. The domain variables are considered as the
  /// input variables of the function. The range variables are considered as
  /// the outputs. The symbols parametrize the function and locals are used to
  /// represent divisions. Each local variable has a corressponding division
  /// representation stored in `divs`.
  PresburgerSpace space;

  /// The function's output is a tuple of integers, with the ith element of the
  /// tuple defined by the affine expression given by the ith row of this output
  /// matrix.
  IntMatrix output;

  /// Storage for division representation for each local variable in space.
  DivisionRepr divs;
};

/// This class represents a piece-wise MultiAffineFunction. This can be thought
/// of as a list of MultiAffineFunction with disjoint domains, with each having
/// their own affine expressions for their output tuples. For example, we could
/// have a function with two input variables (x, y), defined as
///
/// f(x, y) = (2*x + y, y - 4)  if x >= 0, y >= 0
///         = (-2*x + y, y + 4) if x < 0,  y < 0
///         = (4, 1)            if x < 0,  y >= 0
///
/// Note that the domains all have to be *disjoint*. Otherwise, the behaviour of
/// this class is undefined. The domains need not cover all possible points;
/// this represents a partial function and so could be undefined at some points.
///
/// As in PresburgerSets, the input vars are partitioned into dimension vars and
/// symbolic vars.
///
/// Support is provided to compare equality of two such functions as well as
/// finding the value of the function at a point.
class PWMAFunction {
public:
  struct Piece {
    PresburgerSet domain;
    MultiAffineFunction output;

    bool isConsistent() const {
      return domain.getSpace().isCompatible(output.getDomainSpace());
    }
  };

  PWMAFunction(const PresburgerSpace &space) : space(space) {
    assert(space.getNumLocalVars() == 0 &&
           "PWMAFunction cannot have local vars.");
  }

  // Get the space of this function.
  const PresburgerSpace &getSpace() const { return space; }

  // Add a piece ([domain, output] pair) to this function.
  void addPiece(const Piece &piece);

  unsigned getNumPieces() const { return pieces.size(); }
  unsigned getNumVarKind(VarKind kind) const {
    return space.getNumVarKind(kind);
  }
  unsigned getNumDomainVars() const { return space.getNumDomainVars(); }
  unsigned getNumOutputs() const { return space.getNumRangeVars(); }
  unsigned getNumSymbolVars() const { return space.getNumSymbolVars(); }

  /// Remove the specified range of outputs.
  void removeOutputs(unsigned start, unsigned end);

  /// Get the domain/output space of the function. The returned space is a set
  /// space.
  PresburgerSpace getDomainSpace() const { return space.getDomainSpace(); }
  PresburgerSpace getOutputSpace() const { return space.getDomainSpace(); }

  /// Return the domain of this piece-wise MultiAffineFunction. This is the
  /// union of the domains of all the pieces.
  PresburgerSet getDomain() const;

  /// Return the output of the function at the given point.
  std::optional<SmallVector<DynamicAPInt, 8>>
  valueAt(ArrayRef<DynamicAPInt> point) const;
  std::optional<SmallVector<DynamicAPInt, 8>>
  valueAt(ArrayRef<int64_t> point) const {
    return valueAt(getDynamicAPIntVec(point));
  }

  /// Return all the pieces of this piece-wise function.
  ArrayRef<Piece> getAllPieces() const { return pieces; }

  /// Return whether `this` and `other` are equal as PWMAFunctions, i.e. whether
  /// they have the same dimensions, the same domain and they take the same
  /// value at every point in the domain.
  bool isEqual(const PWMAFunction &other) const;

  /// Return a function defined on the union of the domains of this and func,
  /// such that when only one of the functions is defined, it outputs the same
  /// as that function, and if both are defined, it outputs the lexmax/lexmin of
  /// the two outputs. On points where neither function is defined, the returned
  /// function is not defined either.
  ///
  /// Currently this does not support PWMAFunctions which have pieces containing
  /// divisions.
  /// TODO: Support division in pieces.
  PWMAFunction unionLexMin(const PWMAFunction &func);
  PWMAFunction unionLexMax(const PWMAFunction &func);

  void print(raw_ostream &os) const;
  void dump() const;

private:
  /// Return a function defined on the union of the domains of `this` and
  /// `func`, such that when only one of the functions is defined, it outputs
  /// the same as that function, and if neither is defined, the returned
  /// function is not defined either.
  ///
  /// The provided `tiebreak` function determines which of the two functions'
  /// output should be used on inputs where both the functions are defined. More
  /// precisely, given two `MultiAffineFunction`s `mafA` and `mafB`, `tiebreak`
  /// returns the subset of the intersection of the two functions' domains where
  /// the output of `mafA` should be used.
  ///
  /// The PresburgerSet returned by `tiebreak` should be disjoint.
  /// TODO: Remove this constraint of returning disjoint set.
  PWMAFunction unionFunction(
      const PWMAFunction &func,
      llvm::function_ref<PresburgerSet(Piece mafA, Piece mafB)> tiebreak) const;

  /// The space of this function. The domain variables are considered as the
  /// input variables of the function. The range variables are considered as
  /// the outputs. The symbols paramterize the function.
  PresburgerSpace space;

  // The pieces of the PWMAFunction.
  SmallVector<Piece, 4> pieces;
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PWMAFUNCTION_H
