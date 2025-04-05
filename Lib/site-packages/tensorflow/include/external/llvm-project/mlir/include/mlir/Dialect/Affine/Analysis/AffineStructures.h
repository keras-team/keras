//===- AffineStructures.h - MLIR Affine Structures Class --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Structures for affine/polyhedral analysis of ML functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_ANALYSIS_AFFINESTRUCTURES_H
#define MLIR_DIALECT_AFFINE_ANALYSIS_AFFINESTRUCTURES_H

#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/OpDefinition.h"
#include <optional>

namespace mlir {
class AffineMap;
class IntegerSet;
class MemRefType;
class MLIRContext;
struct MutableAffineMap;
class Value;

namespace presburger {
class MultiAffineFunction;
} // namespace presburger

namespace affine {
class AffineCondition;
class AffineForOp;
class AffineIfOp;
class AffineParallelOp;
class AffineValueMap;

/// FlatAffineValueConstraints is an extension of FlatLinearValueConstraints
/// with helper functions for Affine dialect ops.
class FlatAffineValueConstraints : public FlatLinearValueConstraints {
public:
  using FlatLinearValueConstraints::FlatLinearValueConstraints;

  /// Return the kind of this object.
  Kind getKind() const override { return Kind::FlatAffineValueConstraints; }

  static bool classof(const IntegerRelation *cst) {
    return cst->getKind() >= Kind::FlatAffineValueConstraints &&
           cst->getKind() <= Kind::FlatAffineRelation;
  }

  /// Adds constraints (lower and upper bounds) for the specified 'affine.for'
  /// operation's Value using IR information stored in its bound maps. The
  /// right variable is first looked up using `forOp`'s Value. Asserts if the
  /// Value corresponding to the 'affine.for' operation isn't found in the
  /// constraint system. Returns failure for the yet unimplemented/unsupported
  /// cases.  Any new variables that are found in the bound operands of the
  /// 'affine.for' operation are added as trailing variables (either
  /// dimensional or symbolic depending on whether the operand is a valid
  /// symbol).
  LogicalResult addAffineForOpDomain(AffineForOp forOp);

  /// Add constraints (lower and upper bounds) for the specified
  /// 'affine.parallel' operation's Value using IR information stored in its
  /// bound maps. Returns failure for the yet unimplemented/unsupported cases.
  /// Asserts if the Value corresponding to the 'affine.parallel' operation
  /// isn't found in the constraint system.
  LogicalResult addAffineParallelOpDomain(AffineParallelOp parallelOp);

  /// Adds constraints (lower and upper bounds) for each loop in the loop nest
  /// described by the bound maps `lbMaps` and `ubMaps` of a computation slice.
  /// Every pair (`lbMaps[i]`, `ubMaps[i]`) describes the bounds of a loop in
  /// the nest, sorted outer-to-inner. `operands` contains the bound operands
  /// for a single bound map. All the bound maps will use the same bound
  /// operands. Note that some loops described by a computation slice might not
  /// exist yet in the IR so the Value attached to those dimension variables
  /// might be empty. For that reason, this method doesn't perform Value
  /// look-ups to retrieve the dimension variable positions. Instead, it
  /// assumes the position of the dim variables in the constraint system is
  /// the same as the position of the loop in the loop nest.
  LogicalResult addDomainFromSliceMaps(ArrayRef<AffineMap> lbMaps,
                                       ArrayRef<AffineMap> ubMaps,
                                       ArrayRef<Value> operands);

  /// Adds constraints imposed by the `affine.if` operation. These constraints
  /// are collected from the IntegerSet attached to the given `affine.if`
  /// instance argument (`ifOp`). It is asserted that:
  /// 1) The IntegerSet of the given `affine.if` instance should not contain
  /// semi-affine expressions,
  /// 2) The columns of the constraint system created from `ifOp` should match
  /// the columns in the current one regarding numbers and values.
  void addAffineIfOpDomain(AffineIfOp ifOp);

  /// Adds a bound for the variable at the specified position with constraints
  /// being drawn from the specified bound map and operands. In case of an
  /// EQ bound, the  bound map is expected to have exactly one result. In case
  /// of a LB/UB, the bound map may have more than one result, for each of which
  /// an inequality is added.
  LogicalResult addBound(presburger::BoundType type, unsigned pos,
                         AffineMap boundMap, ValueRange operands);
  using FlatLinearValueConstraints::addBound;

  /// Add the specified values as a dim or symbol var depending on its nature,
  /// if it already doesn't exist in the system. `val` has to be either a
  /// terminal symbol or a loop IV, i.e., it cannot be the result affine.apply
  /// of any symbols or loop IVs. The variable is added to the end of the
  /// existing dims or symbols. Additional information on the variable is
  /// extracted from the IR and added to the constraint system.
  void addInductionVarOrTerminalSymbol(Value val);

  /// Adds slice lower bounds represented by lower bounds in `lbMaps` and upper
  /// bounds in `ubMaps` to each variable in the constraint system which has
  /// a value in `values`. Note that both lower/upper bounds share the same
  /// operand list `operands`.
  /// This function assumes `values.size` == `lbMaps.size` == `ubMaps.size`.
  /// Note that both lower/upper bounds use operands from `operands`.
  LogicalResult addSliceBounds(ArrayRef<Value> values,
                               ArrayRef<AffineMap> lbMaps,
                               ArrayRef<AffineMap> ubMaps,
                               ArrayRef<Value> operands);

  /// Changes all symbol variables which are loop IVs to dim variables.
  void convertLoopIVSymbolsToDims();

  /// Returns the bound for the variable at `pos` from the inequality at
  /// `ineqPos` as a 1-d affine value map (affine map + operands). The returned
  /// affine value map can either be a lower bound or an upper bound depending
  /// on the sign of atIneq(ineqPos, pos). Asserts if the row at `ineqPos` does
  /// not involve the `pos`th variable.
  void getIneqAsAffineValueMap(unsigned pos, unsigned ineqPos,
                               AffineValueMap &vmap,
                               MLIRContext *context) const;

  /// Composes the affine value map with this FlatAffineValueConstrains, adding
  /// the results of the map as dimensions at the front
  /// [0, vMap->getNumResults()) and with the dimensions set to the equalities
  /// specified by the value map.
  ///
  /// Returns failure if the composition fails (when vMap is a semi-affine map).
  /// The vMap's operand Value's are used to look up the right positions in
  /// the FlatAffineValueConstraints with which to associate. Every operand of
  /// vMap should have a matching dim/symbol column in this constraint system
  /// (with the same associated Value).
  LogicalResult composeMap(const AffineValueMap *vMap);
};

/// A FlatAffineRelation represents a set of ordered pairs (domain -> range)
/// where "domain" and "range" are tuples of variables. The relation is
/// represented as a FlatAffineValueConstraints with separation of dimension
/// variables into domain and  range. The variables are stored as:
/// [domainVars, rangeVars, symbolVars, localVars, constant].
///
/// Deprecated: use IntegerRelation and store SSA Values in the PresburgerSpace
/// of the relation using PresburgerSpace::identifiers. Note that
/// FlatAffineRelation::numDomainDims and FlatAffineRelation::numRangeDims are
/// independent of numDomain and numRange of the relation's space. In
/// particular, operations such as FlatAffineRelation::compose do not ensure
/// consistency between numDomainDims/numRangeDims and numDomain/numRange which
/// may lead to unexpected behaviour.
class FlatAffineRelation : public FlatAffineValueConstraints {
public:
  FlatAffineRelation(unsigned numReservedInequalities,
                     unsigned numReservedEqualities, unsigned numReservedCols,
                     unsigned numDomainDims, unsigned numRangeDims,
                     unsigned numSymbols, unsigned numLocals,
                     ArrayRef<std::optional<Value>> valArgs = {})
      : FlatAffineValueConstraints(
            numReservedInequalities, numReservedEqualities, numReservedCols,
            numDomainDims + numRangeDims, numSymbols, numLocals, valArgs),
        numDomainDims(numDomainDims), numRangeDims(numRangeDims) {}

  FlatAffineRelation(unsigned numDomainDims = 0, unsigned numRangeDims = 0,
                     unsigned numSymbols = 0, unsigned numLocals = 0)
      : FlatAffineValueConstraints(numDomainDims + numRangeDims, numSymbols,
                                   numLocals),
        numDomainDims(numDomainDims), numRangeDims(numRangeDims) {}

  FlatAffineRelation(unsigned numDomainDims, unsigned numRangeDims,
                     FlatAffineValueConstraints &fac)
      : FlatAffineValueConstraints(fac), numDomainDims(numDomainDims),
        numRangeDims(numRangeDims) {}

  FlatAffineRelation(unsigned numDomainDims, unsigned numRangeDims,
                     IntegerPolyhedron &fac)
      : FlatAffineValueConstraints(fac), numDomainDims(numDomainDims),
        numRangeDims(numRangeDims) {}

  /// Return the kind of this object.
  Kind getKind() const override { return Kind::FlatAffineRelation; }

  static bool classof(const IntegerRelation *cst) {
    return cst->getKind() == Kind::FlatAffineRelation;
  }

  /// Returns a set corresponding to the domain/range of the affine relation.
  FlatAffineValueConstraints getDomainSet() const;
  FlatAffineValueConstraints getRangeSet() const;

  /// Returns the number of variables corresponding to domain/range of
  /// relation.
  inline unsigned getNumDomainDims() const { return numDomainDims; }
  inline unsigned getNumRangeDims() const { return numRangeDims; }

  /// Given affine relation `other: (domainOther -> rangeOther)`, this operation
  /// takes the composition of `other` on `this: (domainThis -> rangeThis)`.
  /// The resulting relation represents tuples of the form: `domainOther ->
  /// rangeThis`.
  void compose(const FlatAffineRelation &other);

  /// Swap domain and range of the relation.
  /// `(domain -> range)` is converted to `(range -> domain)`.
  void inverse();

  /// Insert `num` variables of the specified kind after the `pos` variable
  /// of that kind. The coefficient columns corresponding to the added
  /// variables are initialized to zero.
  void insertDomainVar(unsigned pos, unsigned num = 1);
  void insertRangeVar(unsigned pos, unsigned num = 1);

  /// Append `num` variables of the specified kind after the last variable
  /// of that kind. The coefficient columns corresponding to the added
  /// variables are initialized to zero.
  void appendDomainVar(unsigned num = 1);
  void appendRangeVar(unsigned num = 1);

  /// Removes variables in the column range [varStart, varLimit), and copies any
  /// remaining valid data into place, updates member variables, and resizes
  /// arrays as needed.
  void removeVarRange(VarKind kind, unsigned varStart,
                      unsigned varLimit) override;
  using IntegerRelation::removeVarRange;

protected:
  // Number of dimension variables corresponding to domain variables.
  unsigned numDomainDims;

  // Number of dimension variables corresponding to range variables.
  unsigned numRangeDims;
};

/// Builds a relation from the given AffineMap/AffineValueMap `map`, containing
/// all pairs of the form `operands -> result` that satisfy `map`. `rel` is set
/// to the relation built. For example, give the AffineMap:
///
///   (d0, d1)[s0] -> (d0 + s0, d0 - s0)
///
/// the resulting relation formed is:
///
///   (d0, d1) -> (r1, r2)
///   [d0  d1  r1  r2  s0  const]
///    1   0   -1   0  1     0     = 0
///    0   1    0  -1  -1    0     = 0
///
/// For AffineValueMap, the domain and symbols have Value set corresponding to
/// the Value in `map`. Returns failure if the AffineMap could not be flattened
/// (i.e., semi-affine is not yet handled).
LogicalResult getRelationFromMap(AffineMap &map,
                                 presburger::IntegerRelation &rel);
LogicalResult getRelationFromMap(const AffineValueMap &map,
                                 presburger::IntegerRelation &rel);

} // namespace affine
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_ANALYSIS_AFFINESTRUCTURES_H
