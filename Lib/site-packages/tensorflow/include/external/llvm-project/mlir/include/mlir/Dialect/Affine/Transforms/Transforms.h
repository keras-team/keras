//===- Transforms.h - Transforms Entrypoints --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines a set of transforms specific for the AffineOps
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_AFFINE_TRANSFORMS_TRANSFORMS_H

#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class AffineMap;
class Location;
class OpBuilder;
class OpFoldResult;
class RewritePatternSet;
class RewriterBase;
class Value;

namespace presburger {
enum class BoundType;
} // namespace presburger

namespace affine {
class AffineApplyOp;

/// Populate patterns that expand affine index operations into more fundamental
/// operations (not necessarily restricted to Affine dialect).
void populateAffineExpandIndexOpsPatterns(RewritePatternSet &patterns);

/// Helper function to rewrite `op`'s affine map and reorder its operands such
/// that they are in increasing order of hoistability (i.e. the least hoistable)
/// operands come first in the operand list.
void reorderOperandsByHoistability(RewriterBase &rewriter, AffineApplyOp op);

/// Split an "affine.apply" operation into smaller ops.
/// This reassociates a large AffineApplyOp into an ordered list of smaller
/// AffineApplyOps. This can be used right before lowering affine ops to arith
/// to exhibit more opportunities for CSE and LICM.
/// Return the sink AffineApplyOp on success or failure if `op` does not
/// decompose into smaller AffineApplyOps.
/// Note that this can be undone by canonicalization which tries to
/// maximally compose chains of AffineApplyOps.
FailureOr<AffineApplyOp> decompose(RewriterBase &rewriter, AffineApplyOp op);

/// Reify a bound for the given variable in terms of SSA values for which
/// `stopCondition` is met.
///
/// By default, lower/equal bounds are closed and upper bounds are open. If
/// `closedUB` is set to "true", upper bounds are also closed.
FailureOr<OpFoldResult>
reifyValueBound(OpBuilder &b, Location loc, presburger::BoundType type,
                const ValueBoundsConstraintSet::Variable &var,
                ValueBoundsConstraintSet::StopConditionFn stopCondition,
                bool closedUB = false);

/// Reify a bound for the given index-typed value in terms of SSA values for
/// which `stopCondition` is met. If no stop condition is specified, reify in
/// terms of the operands of the owner op.
///
/// By default, lower/equal bounds are closed and upper bounds are open. If
/// `closedUB` is set to "true", upper bounds are also closed.
///
/// Example:
/// %0 = arith.addi %a, %b : index
/// %1 = arith.addi %0, %c : index
///
/// * If `stopCondition` evaluates to "true" for %0 and %c, "%0 + %c" is an EQ
///   bound for %1.
/// * If `stopCondition` evaluates to "true" for %a, %b and %c, "%a + %b + %c"
///   is an EQ bound for %1.
/// * Otherwise, if the owners of %a, %b or %c do not implement the
///   ValueBoundsOpInterface, no bound can be computed.
FailureOr<OpFoldResult> reifyIndexValueBound(
    OpBuilder &b, Location loc, presburger::BoundType type, Value value,
    ValueBoundsConstraintSet::StopConditionFn stopCondition = nullptr,
    bool closedUB = false);

/// Reify a bound for the specified dimension of the given shaped value in terms
/// of SSA values for which `stopCondition` is met. If no stop condition is
/// specified, reify in terms of the operands of the owner op.
///
/// By default, lower/equal bounds are closed and upper bounds are open. If
/// `closedUB` is set to "true", upper bounds are also closed.
FailureOr<OpFoldResult> reifyShapedValueDimBound(
    OpBuilder &b, Location loc, presburger::BoundType type, Value value,
    int64_t dim,
    ValueBoundsConstraintSet::StopConditionFn stopCondition = nullptr,
    bool closedUB = false);

/// Materialize an already computed bound with Affine dialect ops.
///
/// * `ValueBoundsOpInterface::computeBound` computes bounds but does not
///   create IR. It is dialect independent.
/// * `materializeComputedBound` materializes computed bounds with Affine
///   dialect ops.
/// * `reifyIndexValueBound`/`reifyShapedValueDimBound` are a combination of
///   the two functions mentioned above.
OpFoldResult materializeComputedBound(
    OpBuilder &b, Location loc, AffineMap boundMap,
    ArrayRef<std::pair<Value, std::optional<int64_t>>> mapOperands);

} // namespace affine
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_TRANSFORMS_TRANSFORMS_H
