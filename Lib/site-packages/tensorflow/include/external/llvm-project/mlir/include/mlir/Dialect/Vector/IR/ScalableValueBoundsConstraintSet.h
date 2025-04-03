//===- ScalableValueBoundsConstraintSet.h - Scalable Value Bounds ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_IR_SCALABLEVALUEBOUNDSCONSTRAINTSET_H
#define MLIR_DIALECT_VECTOR_IR_SCALABLEVALUEBOUNDSCONSTRAINTSET_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace mlir::vector {

namespace detail {

/// Parent class for the value bounds RTTIExtends. Uses protected inheritance to
/// hide all ValueBoundsConstraintSet methods by default (as some do not use the
/// ScalableValueBoundsConstraintSet, so may produce unexpected results).
struct ValueBoundsConstraintSet : protected ::mlir::ValueBoundsConstraintSet {
  using ::mlir::ValueBoundsConstraintSet::ValueBoundsConstraintSet;
};
} // namespace detail

/// A version of `ValueBoundsConstraintSet` that can solve for scalable bounds.
struct ScalableValueBoundsConstraintSet
    : public llvm::RTTIExtends<ScalableValueBoundsConstraintSet,
                               detail::ValueBoundsConstraintSet> {
  ScalableValueBoundsConstraintSet(
      MLIRContext *context,
      ValueBoundsConstraintSet::StopConditionFn stopCondition,
      unsigned vscaleMin, unsigned vscaleMax)
      : RTTIExtends(context, stopCondition,
                    /*addConservativeSemiAffineBounds=*/true),
        vscaleMin(vscaleMin), vscaleMax(vscaleMax) {};

  using RTTIExtends::bound;
  using RTTIExtends::StopConditionFn;

  /// A thin wrapper over an `AffineMap` which can represent a constant bound,
  /// or a scalable bound (in terms of vscale). The `AffineMap` will always
  /// take at most one parameter, vscale, and returns a single result, which is
  /// the bound of value.
  struct ConstantOrScalableBound {
    AffineMap map;

    struct BoundSize {
      int64_t baseSize{0};
      bool scalable{false};
    };

    /// Get the (possibly) scalable size of the bound, returns failure if
    /// the bound cannot be represented as a single quantity.
    FailureOr<BoundSize> getSize() const;
  };

  /// Computes a (possibly) scalable bound for a given value. This is
  /// similar to `ValueBoundsConstraintSet::computeConstantBound()`, but
  /// uses knowledge of the range of vscale to compute either a constant
  /// bound, an expression in terms of vscale, or failure if no bound can
  /// be computed.
  ///
  /// The resulting `AffineMap` will always take at most one parameter,
  /// vscale, and return a single result, which is the bound of `value`.
  ///
  /// Note: `vscaleMin` must be `<=` to `vscaleMax`. If `vscaleMin` ==
  /// `vscaleMax`, the resulting bound (if found), will be constant.
  static FailureOr<ConstantOrScalableBound>
  computeScalableBound(Value value, std::optional<int64_t> dim,
                       unsigned vscaleMin, unsigned vscaleMax,
                       presburger::BoundType boundType, bool closedUB = true,
                       StopConditionFn stopCondition = nullptr);

  /// Get the value of vscale. Returns `nullptr` vscale as not been encountered.
  Value getVscaleValue() const { return vscale; }

  /// Sets the value of vscale. Asserts if vscale has already been set.
  void setVscale(vector::VectorScaleOp vscaleOp) {
    assert(!vscale && "expected vscale to be unset");
    vscale = vscaleOp.getResult();
  }

  /// The minimum possible value of vscale.
  unsigned getVscaleMin() const { return vscaleMin; }

  /// The maximum possible value of vscale.
  unsigned getVscaleMax() const { return vscaleMax; }

  static char ID;

private:
  const unsigned vscaleMin;
  const unsigned vscaleMax;

  // This will be set when the first `vector.vscale` operation is found within
  // the `ValueBoundsOpInterface` implementation then reused from there on.
  Value vscale = nullptr;
};

using ConstantOrScalableBound =
    ScalableValueBoundsConstraintSet::ConstantOrScalableBound;

} // namespace mlir::vector

#endif // MLIR_DIALECT_VECTOR_IR_SCALABLEVALUEBOUNDSCONSTRAINTSET_H
