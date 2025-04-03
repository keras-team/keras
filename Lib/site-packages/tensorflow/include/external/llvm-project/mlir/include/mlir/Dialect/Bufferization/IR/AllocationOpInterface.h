//===- AllocationOpInterface.h - Allocation op interface ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interface for allocation ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_IR_ALLOCATIONOPINTERFACE_H_
#define MLIR_DIALECT_BUFFERIZATION_IR_ALLOCATIONOPINTERFACE_H_

#include "mlir/IR/Builders.h"

namespace mlir {
// Enum class representing different hoisting kinds for the allocation
// operation
enum class HoistingKind : uint8_t {
  None = 0,       // No hoisting kind selected
  Loop = 1 << 0,  // Indicates loop hoisting kind
  Block = 1 << 1, // Indicates dominated block hoisting kind
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ Block)
};
} // namespace mlir

#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h.inc"

#endif // MLIR_DIALECT_BUFFERIZATION_IR_ALLOCATIONOPINTERFACE_H_
