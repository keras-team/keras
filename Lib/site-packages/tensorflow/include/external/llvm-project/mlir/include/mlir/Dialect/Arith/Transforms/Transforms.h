//===- Transforms.h - Arith Transforms --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITH_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_ARITH_TRANSFORMS_TRANSFORMS_H

#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace mlir {
class Location;
class OpBuilder;
class OpFoldResult;
class Value;

namespace presburger {
enum class BoundType;
} // namespace presburger

namespace arith {

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

} // namespace arith
} // namespace mlir

#endif // MLIR_DIALECT_ARITH_TRANSFORMS_TRANSFORMS_H
