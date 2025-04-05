//===- PresburgerRelation.h - MLIR PresburgerRelation Class -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class to represent unions of IntegerRelations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERRELATION_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERRELATION_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include <optional>

namespace mlir {
namespace presburger {

/// The SetCoalescer class contains all functionality concerning the coalesce
/// heuristic. It is built from a `PresburgerRelation` and has the `coalesce()`
/// function as its main API.
class SetCoalescer;

/// A PresburgerRelation represents a union of IntegerRelations that live in
/// the same PresburgerSpace with support for union, intersection, subtraction,
/// and complement operations, as well as sampling.
///
/// The IntegerRelations (disjuncts) are stored in a vector, and the set
/// represents the union of these relations. An empty list corresponds to
/// the empty set.
///
/// Note that there are no invariants guaranteed on the list of disjuncts
/// other than that they are all in the same PresburgerSpace. For example, the
/// relations may overlap with each other.
class PresburgerRelation {
public:
  /// Return a universe set of the specified type that contains all points.
  static PresburgerRelation getUniverse(const PresburgerSpace &space);

  /// Return an empty set of the specified type that contains no points.
  static PresburgerRelation getEmpty(const PresburgerSpace &space);

  explicit PresburgerRelation(const IntegerRelation &disjunct);

  unsigned getNumDomainVars() const { return space.getNumDomainVars(); }
  unsigned getNumRangeVars() const { return space.getNumRangeVars(); }
  unsigned getNumSymbolVars() const { return space.getNumSymbolVars(); }
  unsigned getNumLocalVars() const { return space.getNumLocalVars(); }
  unsigned getNumVars() const { return space.getNumVars(); }

  /// Return the number of disjuncts in the union.
  unsigned getNumDisjuncts() const;

  const PresburgerSpace &getSpace() const { return space; }

  /// Set the space to `oSpace`. `oSpace` should not contain any local ids.
  /// `oSpace` need not have the same number of ids as the current space;
  /// it could have more or less. If it has less, the extra ids become
  /// locals of the disjuncts. It can also have more, in which case the
  /// disjuncts will have fewer locals. If its total number of ids
  /// exceeds that of some disjunct, an assert failure will occur.
  void setSpace(const PresburgerSpace &oSpace);

  void insertVarInPlace(VarKind kind, unsigned pos, unsigned num = 1);

  /// Converts variables of the specified kind in the column range [srcPos,
  /// srcPos + num) to variables of the specified kind at position dstPos. The
  /// ranges are relative to the kind of variable.
  ///
  /// srcKind and dstKind must be different.
  void convertVarKind(VarKind srcKind, unsigned srcPos, unsigned num,
                      VarKind dstKind, unsigned dstPos);

  /// Return a reference to the list of disjuncts.
  ArrayRef<IntegerRelation> getAllDisjuncts() const;

  /// Return the disjunct at the specified index.
  const IntegerRelation &getDisjunct(unsigned index) const;

  /// Mutate this set, turning it into the union of this set and the given
  /// disjunct.
  void unionInPlace(const IntegerRelation &disjunct);

  /// Mutate this set, turning it into the union of this set and the given set.
  void unionInPlace(const PresburgerRelation &set);

  /// Return the union of this set and the given set.
  PresburgerRelation unionSet(const PresburgerRelation &set) const;

  /// Return the intersection of this set and the given set.
  PresburgerRelation intersect(const PresburgerRelation &set) const;

  /// Return the range intersection of the given `set` with `this` relation.
  ///
  /// Formally, let the relation `this` be R: A -> B and `set` is C, then this
  /// operation returns A -> (B intersection C).
  PresburgerRelation intersectRange(const PresburgerSet &set) const;

  /// Return the domain intersection of the given `set` with `this` relation.
  ///
  /// Formally, let the relation `this` be R: A -> B and `set` is C, then this
  /// operation returns (A intersection C) -> B.
  PresburgerRelation intersectDomain(const PresburgerSet &set) const;

  /// Return a set corresponding to the domain of the relation.
  PresburgerSet getDomainSet() const;
  /// Return a set corresponding to the range of the relation.
  PresburgerSet getRangeSet() const;

  /// Invert the relation, i.e. swap its domain and range.
  ///
  /// Formally, if `this`: A -> B then `inverse` updates `this` in-place to
  /// `this`: B -> A.
  void inverse();

  /// Compose `this` relation with the given relation `rel` in-place.
  ///
  /// Formally, if `this`: A -> B, and `rel`: B -> C, then this function updates
  /// `this` to `result`: A -> C where a point (a, c) belongs to `result`
  /// iff there exists b such that (a, b) is in `this` and, (b, c) is in rel.
  void compose(const PresburgerRelation &rel);

  /// Apply the domain of given relation `rel` to `this` relation.
  ///
  /// Formally, R1.applyDomain(R2) = R2.inverse().compose(R1).
  void applyDomain(const PresburgerRelation &rel);

  /// Same as compose, provided for uniformity with applyDomain.
  void applyRange(const PresburgerRelation &rel);

  /// Compute the symbolic integer lexmin of the relation, i.e. for every
  /// assignment of the symbols and domain the lexicographically minimum value
  /// attained by the range.
  SymbolicLexOpt findSymbolicIntegerLexMin() const;

  /// Compute the symbolic integer lexmax of the relation, i.e. for every
  /// assignment of the symbols and domain the lexicographically maximum value
  /// attained by the range.
  SymbolicLexOpt findSymbolicIntegerLexMax() const;

  /// Return true if the set contains the given point, and false otherwise.
  bool containsPoint(ArrayRef<DynamicAPInt> point) const;
  bool containsPoint(ArrayRef<int64_t> point) const {
    return containsPoint(getDynamicAPIntVec(point));
  }

  /// Return the complement of this set. All local variables in the set must
  /// correspond to floor divisions.
  PresburgerRelation complement() const;

  /// Return the set difference of this set and the given set, i.e.,
  /// return `this \ set`. All local variables in `set` must correspond
  /// to floor divisions, but local variables in `this` need not correspond to
  /// divisions.
  PresburgerRelation subtract(const PresburgerRelation &set) const;

  /// Return true if this set is a subset of the given set, and false otherwise.
  bool isSubsetOf(const PresburgerRelation &set) const;

  /// Return true if this set is equal to the given set, and false otherwise.
  /// All local variables in both sets must correspond to floor divisions.
  bool isEqual(const PresburgerRelation &set) const;

  /// Return true if all the sets in the union are known to be integer empty
  /// false otherwise.
  bool isIntegerEmpty() const;

  /// Return true if there is no disjunct, false otherwise.
  bool isObviouslyEmpty() const;

  /// Return true if the set is known to have one unconstrained disjunct, false
  /// otherwise.
  bool isObviouslyUniverse() const;

  /// Perform a quick equality check on `this` and `other`. The relations are
  /// equal if the check return true, but may or may not be equal if the check
  /// returns false. This is doing by directly comparing whether each internal
  /// disjunct is the same.
  bool isObviouslyEqual(const PresburgerRelation &set) const;

  /// Return true if the set is consist of a single disjunct, without any local
  /// variables, false otherwise.
  bool isConvexNoLocals() const;

  /// Find an integer sample from the given set. This should not be called if
  /// any of the disjuncts in the union are unbounded.
  bool findIntegerSample(SmallVectorImpl<DynamicAPInt> &sample);

  /// Compute an overapproximation of the number of integer points in the
  /// disjunct. Symbol vars are currently not supported. If the computed
  /// overapproximation is infinite, an empty optional is returned.
  ///
  /// This currently just sums up the overapproximations of the volumes of the
  /// disjuncts, so the approximation might be far from the true volume in the
  /// case when there is a lot of overlap between disjuncts.
  std::optional<DynamicAPInt> computeVolume() const;

  /// Simplifies the representation of a PresburgerRelation.
  ///
  /// In particular, removes all disjuncts which are subsets of other
  /// disjuncts in the union.
  PresburgerRelation coalesce() const;

  /// Check whether all local ids in all disjuncts have a div representation.
  bool hasOnlyDivLocals() const;

  /// Compute an equivalent representation of the same relation, such that all
  /// local ids in all disjuncts have division representations. This
  /// representation may involve local ids that correspond to divisions, and may
  /// also be a union of convex disjuncts.
  PresburgerRelation computeReprWithOnlyDivLocals() const;

  /// Simplify each disjunct, canonicalizing each disjunct and removing
  /// redundencies.
  PresburgerRelation simplify() const;

  /// Return whether the given PresburgerRelation is full-dimensional. By full-
  /// dimensional we mean that it is not flat along any dimension.
  bool isFullDim() const;

  /// Print the set's internal state.
  void print(raw_ostream &os) const;
  void dump() const;

protected:
  /// Construct an empty PresburgerRelation with the specified number of
  /// dimension and symbols.
  explicit PresburgerRelation(const PresburgerSpace &space) : space(space) {
    assert(space.getNumLocalVars() == 0 &&
           "PresburgerRelation cannot have local vars.");
  }

  PresburgerSpace space;

  /// The list of disjuncts that this set is the union of.
  SmallVector<IntegerRelation, 2> disjuncts;

  friend class SetCoalescer;
};

class PresburgerSet : public PresburgerRelation {
public:
  /// Return a universe set of the specified type that contains all points.
  static PresburgerSet getUniverse(const PresburgerSpace &space);

  /// Return an empty set of the specified type that contains no points.
  static PresburgerSet getEmpty(const PresburgerSpace &space);

  /// Create a set from a relation.
  explicit PresburgerSet(const IntegerPolyhedron &disjunct);
  explicit PresburgerSet(const PresburgerRelation &set);

  /// These operations are the same as the ones in PresburgeRelation, they just
  /// forward the arguement and return the result as a set instead of a
  /// relation.
  PresburgerSet unionSet(const PresburgerRelation &set) const;
  PresburgerSet intersect(const PresburgerRelation &set) const;
  PresburgerSet complement() const;
  PresburgerSet subtract(const PresburgerRelation &set) const;
  PresburgerSet coalesce() const;

protected:
  /// Construct an empty PresburgerRelation with the specified number of
  /// dimension and symbols.
  explicit PresburgerSet(const PresburgerSpace &space)
      : PresburgerRelation(space) {
    assert(space.getNumDomainVars() == 0 &&
           "Set type cannot have domain vars.");
    assert(space.getNumLocalVars() == 0 &&
           "PresburgerRelation cannot have local vars.");
  }
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERRELATION_H
