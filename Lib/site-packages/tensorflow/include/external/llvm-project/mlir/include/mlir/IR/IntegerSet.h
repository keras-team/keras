//===- IntegerSet.h - MLIR Integer Set Class --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Integer sets are sets of points from the integer lattice constrained by
// affine equality/inequality constraints. This class is meant to represent
// integer sets in the IR - for 'affine.if' operations and as attributes of
// other operations. It is typically expected to contain only a handful of
// affine constraints, and is immutable like an affine map. Integer sets are not
// unique'd unless the number of constraints they contain are below a certain
// threshold - although affine expressions that make up its equalities and
// inequalities are themselves unique.

// This class is not meant for affine analysis and operations like set
// operations, emptiness checks, or other math operations for analysis and
// transformation. For the latter, use FlatAffineValueConstraints.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_INTEGERSET_H
#define MLIR_IR_INTEGERSET_H

#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {

namespace detail {
struct IntegerSetStorage;
} // namespace detail

class MLIRContext;

/// An integer set representing a conjunction of one or more affine equalities
/// and inequalities. An integer set in the IR is immutable like the affine map,
/// but integer sets are not unique'd unless the number of constraints in them
/// is below `kUniquingThreshold`. The affine expressions that make up the
/// equalities and inequalities of an integer set are themselves unique and are
/// allocated by the bump pointer allocator.
class IntegerSet {
public:
  using ImplType = detail::IntegerSetStorage;

  constexpr IntegerSet() = default;
  explicit IntegerSet(ImplType *set) : set(set) {}

  static IntegerSet get(unsigned dimCount, unsigned symbolCount,
                        ArrayRef<AffineExpr> constraints,
                        ArrayRef<bool> eqFlags);

  // Returns the canonical empty IntegerSet (i.e. a set with no integer points).
  static IntegerSet getEmptySet(unsigned numDims, unsigned numSymbols,
                                MLIRContext *context) {
    auto one = getAffineConstantExpr(1, context);
    // 1 == 0.
    return get(numDims, numSymbols, one, true);
  }

  /// Returns true if this is the canonical integer set.
  bool isEmptyIntegerSet() const;

  /// This method substitutes any uses of dimensions and symbols (e.g.
  /// dim#0 with dimReplacements[0]) in subexpressions and returns the modified
  /// integer set.  Because this can be used to eliminate dims and
  /// symbols, the client needs to specify the number of dims and symbols in
  /// the result.  The returned map always has the same number of results.
  IntegerSet replaceDimsAndSymbols(ArrayRef<AffineExpr> dimReplacements,
                                   ArrayRef<AffineExpr> symReplacements,
                                   unsigned numResultDims,
                                   unsigned numResultSyms);

  explicit operator bool() { return set; }
  bool operator==(IntegerSet other) const { return set == other.set; }
  bool operator!=(IntegerSet other) const { return set != other.set; }

  unsigned getNumDims() const;
  unsigned getNumSymbols() const;
  unsigned getNumInputs() const;
  unsigned getNumConstraints() const;
  unsigned getNumEqualities() const;
  unsigned getNumInequalities() const;

  ArrayRef<AffineExpr> getConstraints() const;

  AffineExpr getConstraint(unsigned idx) const;

  /// Returns the equality bits, which specify whether each of the constraints
  /// is an equality or inequality.
  ArrayRef<bool> getEqFlags() const;

  /// Returns true if the idx^th constraint is an equality, false if it is an
  /// inequality.
  bool isEq(unsigned idx) const;

  MLIRContext *getContext() const;

  /// Walk all of the AffineExpr's in this set's constraints. Each node in an
  /// expression tree is visited in postorder.
  void walkExprs(function_ref<void(AffineExpr)> callback) const;

  void print(raw_ostream &os) const;
  void dump() const;

  friend ::llvm::hash_code hash_value(IntegerSet arg);

  /// Methods supporting C API.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(set);
  }
  static IntegerSet getFromOpaquePointer(const void *pointer) {
    return IntegerSet(
        reinterpret_cast<ImplType *>(const_cast<void *>(pointer)));
  }

private:
  ImplType *set{nullptr};
};

// Make AffineExpr hashable.
inline ::llvm::hash_code hash_value(IntegerSet arg) {
  return ::llvm::hash_value(arg.set);
}

} // namespace mlir
namespace llvm {

// IntegerSet hash just like pointers.
template <>
struct DenseMapInfo<mlir::IntegerSet> {
  static mlir::IntegerSet getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::IntegerSet(static_cast<mlir::IntegerSet::ImplType *>(pointer));
  }
  static mlir::IntegerSet getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::IntegerSet(static_cast<mlir::IntegerSet::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::IntegerSet val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::IntegerSet LHS, mlir::IntegerSet RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm
#endif // MLIR_IR_INTEGERSET_H
