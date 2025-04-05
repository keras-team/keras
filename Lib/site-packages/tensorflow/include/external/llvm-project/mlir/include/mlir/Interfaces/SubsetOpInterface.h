//===- SubsetOpInterface.h - Tensor Subsets ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_SUBSETOPINTERFACE_H_
#define MLIR_INTERFACES_SUBSETOPINTERFACE_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace mlir {
class SubsetOpInterface;
class SubsetExtractionOpInterface;
class SubsetInsertionOpInterface;

namespace detail {

/// Return the destination/"init" operand of the op if it implements the
/// `DestinationStyleOpInterface` and has exactly one "init" operand. Asserts
/// otherwise.
OpOperand &defaultGetDestinationOperand(Operation *op);

/// Return the updated destination result of the op if it implements the
/// `DestinationStyleOpInterface`.
OpResult defaultGetUpdatedDestination(Operation *op);

/// Default implementation of `SubsetInsertionOpInterface::isEquivalentSubset`.
bool defaultIsEquivalentSubset(Operation *op, Value candidate,
                               function_ref<bool(Value, Value)> equivalenceFn);

/// Default implementation of `SubsetOpInterface::operatesOnEquivalentSubset`.
bool defaultOperatesOnEquivalentSubset(
    Operation *op, SubsetOpInterface candidate,
    function_ref<bool(Value, Value)> equivalenceFn);

/// Default implementation of `SubsetOpInterface::operatesOnDisjointSubset`.
bool defaultOperatesOnDisjointSubset(
    Operation *op, SubsetOpInterface candidate,
    function_ref<bool(Value, Value)> equivalenceFn);

/// Return the container that the given subset op is operating on.
Value getTensorContainer(Operation *op);

/// Verify `SubsetOpInterface`.
LogicalResult verifySubsetOpInterface(SubsetOpInterface op);

/// Verify `SubsetExtractionOpInterface`.
LogicalResult verifySubsetExtractionOpInterface(SubsetExtractionOpInterface op);

} // namespace detail
} // namespace mlir

#include "mlir/Interfaces/SubsetOpInterface.h.inc"

#endif // MLIR_INTERFACES_SUBSETOPINTERFACE_H_
