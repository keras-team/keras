//===- AffineValueMap.h - MLIR Affine Value Map Class -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An AffineValueMap is an affine map plus its ML value operands and results for
// analysis purposes.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_IR_AFFINEVALUEMAP_H
#define MLIR_DIALECT_AFFINE_IR_AFFINEVALUEMAP_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace affine {

/// An AffineValueMap is an affine map plus its ML value operands and
/// results for analysis purposes. The structure is still a tree form that is
/// same as that of an affine map or an AffineApplyOp. However, its operands,
/// results, and its map can themselves change  as a result of
/// substitutions, simplifications, and other analysis.
// An affine value map can readily be constructed from an AffineApplyOp, or an
// AffineBound of a AffineForOp. It can be further transformed, substituted
// into, or simplified. Unlike AffineMap's, AffineValueMap's are created and
// destroyed during analysis. Only the AffineMap expressions that are pointed by
// them are unique'd. An affine value map, and the operations on it, maintain
// the invariant that operands are always positionally aligned with the
// AffineDimExpr and AffineSymbolExpr in the underlying AffineMap.
class AffineValueMap {
public:
  // Creates an empty AffineValueMap (users should call 'reset' to reset map
  // and operands).
  AffineValueMap() = default;
  AffineValueMap(AffineMap map, ValueRange operands, ValueRange results = {});

  ~AffineValueMap();

  // Resets this AffineValueMap with 'map', 'operands', and 'results'.
  void reset(AffineMap map, ValueRange operands, ValueRange results = {});

  /// Composes all incoming affine.apply ops and then simplifies and
  /// canonicalizes the map and operands. This can change the number of
  /// operands, but the result count remains the same.
  void composeSimplifyAndCanonicalize();

  /// Return the value map that is the difference of value maps 'a' and 'b',
  /// represented as an affine map and its operands. The output map + operands
  /// are canonicalized and simplified.
  static void difference(const AffineValueMap &a, const AffineValueMap &b,
                         AffineValueMap *res);

  /// Return true if the idx^th result can be proved to be a multiple of
  /// 'factor', false otherwise.
  inline bool isMultipleOf(unsigned idx, int64_t factor) const;

  /// Return true if the idx^th result depends on 'value', false otherwise.
  bool isFunctionOf(unsigned idx, Value value) const;

  /// Return true if the result at 'idx' is a constant, false
  /// otherwise.
  bool isConstant(unsigned idx) const;

  /// Return true if this is an identity map.
  bool isIdentity() const;

  void setResult(unsigned i, AffineExpr e) { map.setResult(i, e); }
  AffineExpr getResult(unsigned i) { return map.getResult(i); }
  inline unsigned getNumOperands() const { return operands.size(); }
  inline unsigned getNumDims() const { return map.getNumDims(); }
  inline unsigned getNumSymbols() const { return map.getNumSymbols(); }
  inline unsigned getNumResults() const { return map.getNumResults(); }

  Value getOperand(unsigned i) const;
  ArrayRef<Value> getOperands() const;
  AffineMap getAffineMap() const;

  /// Attempts to canonicalize the map and operands. Return success if the map
  /// and/or operands have been modified.
  LogicalResult canonicalize();

private:
  // A mutable affine map.
  MutableAffineMap map;

  // TODO: make these trailing objects?
  /// The SSA operands binding to the dim's and symbols of 'map'.
  SmallVector<Value, 4> operands;
  /// The SSA results binding to the results of 'map'.
  SmallVector<Value, 4> results;
};

} // namespace affine
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_IR_AFFINEVALUEMAP_H
