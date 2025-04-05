//===- ValueBoundsOpInterfaceImpl.h - Impl. of ValueBoundsOpInterface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_IR_VALUEBOUNDSOPINTERFACEIMPL_H
#define MLIR_DIALECT_AFFINE_IR_VALUEBOUNDSOPINTERFACEIMPL_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class DialectRegistry;
class Value;

namespace affine {
void registerValueBoundsOpInterfaceExternalModels(DialectRegistry &registry);

/// Compute a constant delta of the given two values. Return "failure" if we
/// cannot determine a constant delta. `value1`/`value2` must be index-typed.
///
/// This function is similar to
/// `ValueBoundsConstraintSet::computeConstantDistance`. To work around
/// limitations in `FlatLinearConstraints`, this function fully composes
/// `value1` and `value2` (if they are the result of affine.apply ops) before
/// populating the constraint set. The folding/composing logic can see
/// opportunities for simplifications that the constraint set implementation
/// cannot see.
FailureOr<int64_t> fullyComposeAndComputeConstantDelta(Value value1,
                                                       Value value2);
} // namespace affine
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_IR_VALUEBOUNDSOPINTERFACEIMPL_H
